"""
Functional tests for CoderAgent: drive the REAL ``build_coder_service`` on a
shared ``DeliveringBus`` (real subscribe/dispatch routing, not a hand-called
handler method), with only the LLM mocked (``FakeLLM``) and a real tmp
``SpecStore``.

One flow, one happy-path test:
  ImplementingExperiment: code.implementation_requested -> code.implemented.

针对 CoderAgent 的功能测试：在共享 ``DeliveringBus``（真实 subscribe/dispatch
路由，而非直接调用 handler 方法）上跑真实的 ``build_coder_service``，只 mock
LLM（``FakeLLM``），SpecStore 为真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.coder_agent.service import build_coder_service
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXPERIMENT_PATCH_VALIDATOR = _validator("experiment_patch.schema.yaml")


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "functional.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Canned shapes (mirrors tests/test_handler.py -- do not invent new shapes).
# ---------------------------------------------------------------------------

VALID_EXPERIMENT_PLAN = {
    "hypothesis_id": "h1",
    "change_summary": "Add a learned gating layer before retrieval fusion.",
    "changes": [
        {"file": "src/model.py", "description": "Add a gating module class."},
        {"file": "src/train.py", "description": "Wire the gating module into the training loop."},
    ],
    "ablations": ["no_gating_baseline"],
    "expected_outcome": "Improved exact_match on the long-tail subset.",
    "resource_budget": "1x A100 GPU, 4h wall-clock",
    "success_criteria": "exact_match improves by >=2 points over baseline on the held-out split.",
}

VALID_VERIFIER_SPEC = {
    "evaluator_hash": "sha256:abc123",
    "metric": "exact_match",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py", "src/train.py"],
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
    "result_parse_schema": {"type": "object"},
}


def _valid_patch(**overrides: object) -> dict:
    patch = {
        "hypothesis_id": "h1",
        "summary": "Add a learned gating layer before retrieval fusion.",
        "changes": [
            {"file": "src/model.py", "patch": "+class Gate(nn.Module):\n+    ..."},
            {"file": "src/train.py", "patch": "+model.gate = Gate()\n"},
        ],
    }
    patch.update(overrides)
    return patch


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# code.implementation_requested -> code.implemented
# ---------------------------------------------------------------------------


async def test_implementation_requested_produces_implemented(bus, store):
    llm = FakeLLM([_json(_valid_patch())])
    build_coder_service(
        client=bus.view("CoderAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-implementing-1"
    experiment_plan_ref = store.put(
        kind="experiment_plan",
        session_id=session_id,
        key="v1",
        obj=VALID_EXPERIMENT_PLAN,
        emitter="ExperimentPlannerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="code.implementation_requested",
        session_id=session_id,
        payload={
            "experiment_plan_ref": experiment_plan_ref,
            "verifier_spec_ref": verifier_spec_ref,
        },
    )

    implemented = [e for e in bus.published if e.event_type == "code.implemented"]
    assert len(implemented) == 1

    patch = store.get(implemented[0].payload["experiment_patch_ref"])
    errors = list(EXPERIMENT_PATCH_VALIDATOR.iter_errors(patch))
    assert errors == [], [e.message for e in errors]
    assert set(patch["touched_files"]) == {c["file"] for c in patch["changes"]}

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
