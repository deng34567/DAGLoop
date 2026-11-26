"""MOSS Step 2 Task 4 smoke: MOSS's ``prompt_patch`` actually STEERS a
downstream agent -- end to end, PROVEN, not merely unit-tested in isolation.

Single shared ``DeliveringBus`` + one tmp ``SpecStore`` + ONE tmp
``config_root`` shared by MOSS (the writer) and CoderAgent (the reader) --
this is the crux of the smoke: PatchWriter and ConfigLoader must agree on
the same flat path (``config_root/agent_config/{agent}/prompt_patch.md``)
for the patch to actually be "live". Mocked LLMs, no Redis, no real
execution, no concurrency, single linear session.

Step 1 -- MOSS writes the patch: a ``moss.intervention_requested{reason:
"protocol_violation"}`` is delivered directly to MOSS (mirrors
``test_moss_smoke.py``'s "publish the request event directly" style, minus
the ``OrchestratorEngine``/state-machine -- that cascade is already proven
by MOSS Step 1's smoke; this smoke's job is the patch, not routing). MOSS's
FakeLLM returns a canned decision with a ``prompt_patch`` for
``CoderAgent``. Assert ``moss.intervention_completed`` fires and the patch
file exists on disk with the expected content.

Step 2 -- the patch is LIVE on CoderAgent's next call: the store is seeded
with an ExperimentPlan + VerifierSpec (as ExperimentPlannerAgent/
EvaluatorAgent would), then ``code.implementation_requested`` is delivered
to CoderAgent. Assert CoderAgent's FakeLLM's CAPTURED system message
contains the patch content AND the ``[MOSS-injected guidance]`` marker
(``autoresearch/agents/_reuse/prompt_patch.py``'s ``apply_prompt_patch``),
and that ``code.implemented`` still fires (no ``agent.failed`` -- the patch
didn't break CoderAgent).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest

from autoresearch.agents.coder_agent.service import build_coder_service
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM as CoderFakeLLM
from autoresearch.agents.moss_agent.service import build_moss_service
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM as MossFakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

STEER_CONTENT = "STEER: prefer minimal, focused diffs"

EXPERIMENT_PLAN = {
    "hypothesis_id": "hyp-1",
    "change_summary": "Add a learned gating layer before retrieval fusion.",
    "changes": [
        {"file": "src/model.py", "description": "Add a gating module class."},
    ],
    "ablations": ["no_gating_baseline"],
    "expected_outcome": "Improved exact_match on the long-tail subset.",
    "resource_budget": "1x A100 GPU, 4h wall-clock",
    "success_criteria": "exact_match improves by >=2 points over baseline on the held-out split.",
}

VERIFIER_SPEC = {
    "evaluator_hash": "sha256:abc123",
    "metric": "exact_match",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py"],
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
    "result_parse_schema": {"type": "object"},
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _valid_experiment_patch() -> dict:
    return {
        "hypothesis_id": "hyp-1",
        "summary": "Add a learned gating layer before retrieval fusion.",
        "changes": [
            {"file": "src/model.py", "patch": "+class Gate(nn.Module):\n+    ..."},
        ],
    }


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "moss_patch_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


async def test_moss_prompt_patch_is_live_on_coder_agents_next_call(bus, store, tmp_path):
    # Shared configuration must not carry session-specific advice.
    config_root = tmp_path / "config_root"

    moss_llm = MossFakeLLM(
        [
            _json(
                {
                    "action": "rollback_patch",
                    "prompt_patch": {"agent": "CoderAgent", "content": STEER_CONTENT},
                    "rationale": "protocol violation detected",
                }
            )
        ]
    )
    build_moss_service(
        client=bus.view("MOSS"),
        spec_store=store,
        config_root=config_root,
        llm_client=moss_llm,
    )

    coder_llm = CoderFakeLLM([_json(_valid_experiment_patch())])
    build_coder_service(
        client=bus.view("CoderAgent"),
        spec_store=store,
        config_root=config_root,
        llm_client=coder_llm,
    )

    session_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Step 1: MOSS persists immutable scoped advice.
    # ------------------------------------------------------------------
    await bus.view("ResearchOrchestrator").publish(
        event_type="moss.intervention_requested",
        session_id=session_id,
        payload={"reason": "protocol_violation"},
    )

    completed = [e for e in bus.published if e.event_type == "moss.intervention_completed"]
    assert len(completed) == 1
    assert completed[0].session_id == session_id
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    patch_path = config_root / "agent_config" / "CoderAgent" / "prompt_patch.md"
    assert not patch_path.exists()

    # ------------------------------------------------------------------
    # Step 2: the patch is LIVE on CoderAgent's next call.
    # ------------------------------------------------------------------
    experiment_plan_ref = store.put(
        kind="experiment_plan",
        session_id=session_id,
        key="v1",
        obj=EXPERIMENT_PLAN,
        emitter="ExperimentPlannerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )

    await bus.view("ExperimentPlannerAgent").publish(
        event_type="code.implementation_requested",
        session_id=session_id,
        payload={
            "experiment_plan_ref": experiment_plan_ref,
            "verifier_spec_ref": verifier_spec_ref,
            "moss_intervention_ref": completed[0].payload["moss_intervention_ref"],
        },
    )

    assert len(coder_llm.calls) == 1
    [messages] = coder_llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert STEER_CONTENT in system_message
    assert "[MOSS-injected guidance]" in system_message

    implemented = [e for e in bus.published if e.event_type == "code.implemented"]
    assert len(implemented) == 1
    assert implemented[0].session_id == session_id
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
