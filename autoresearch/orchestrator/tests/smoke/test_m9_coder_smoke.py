"""ImplementingExperiment -> RunningExperiment isolated integration smoke
(M9 Task 5): single session, no Redis, no real LLM.

``ImplementingExperiment`` is reached from ``ReviewingExperimentPlan`` (M8),
which is NOT wired to this smoke -- mirrors the isolation-construction
pattern in ``test_m8_planner_smoke.py``. This smoke seeds a VerifierSpec +
ExperimentPlan directly into the Spec Store, constructs + saves a
``SessionState`` directly in ``ImplementingExperiment``, then drives it
through a REAL ``DeliveringBus`` (genuine subscribe/dispatch routing, not a
manual handler call) so the full cascade -- entry action -> CoderAgent ->
orchestrator's INDEPENDENT protocol gate -> transition -- is exercised for
real across two independently-built components: ``OrchestratorEngine``
(constructed WITH ``spec_store=store``, required by the gate) and the
CoderAgent service.

Drives, in one linear session:
  1. Seed a VerifierSpec (``editable_region=["src/model.py"]``, disjoint
     ``forbidden_files=["eval/scorer.py"]``) and an ExperimentPlan (whose
     sole ``changes[].file`` is ``"src/model.py"``) directly into the Spec
     Store, as upstream agents would have.
  2. Construct + save a ``SessionState`` directly in
     ``ImplementingExperiment`` with ``current_experiment_id="exp1"``,
     ``current_hypothesis_id="h1"``, ``experiment_plan_refs=[<plan ref>]``,
     and ``verifier_spec_ref=<verifier ref>`` (isolated construction -- no
     prior states are driven).
  3. ``eng.resume_to_current_state(session_id)`` re-emits the
     ``ImplementingExperiment`` entry action, ``code.implementation_requested``
     -> (delivered) -> CoderAgent resolves the ExperimentPlan + VerifierSpec
     from the store, runs its (canned) ``implement_patch`` (changes[].file
     is "src/model.py", the SAME file as the seeded VerifierSpec's
     editable_region -- the coordination the CoderAgent does NOT itself
     police), persists the ExperimentPatch, and publishes
     ``code.implemented`` -> (delivered) -> orchestrator's INDEPENDENT
     protocol gate reads the FROZEN VerifierSpec from the store (not the
     CoderAgent's word), recomputes ``touched_files`` vs ``editable_region``
     / ``forbidden_files``, finds no violation, resets MAP counter F
     (``implementation_fix_round["exp1"]``, delete-key), and transitions
     ImplementingExperiment -> RunningExperiment, which has no entry action
     (M10 RunnerAgent's run mode not built), so the loop pauses there -- the
     mainline has now reached RunningExperiment with a frozen, in-scope
     ExperimentPatch on the books.

No concurrency, no Redis, no real LLM call anywhere in this file (one FakeLLM
canned queue). ``resume_to_current_state`` is called exactly once -- a second
call would re-emit the entry action and break the "exactly one" event
counts.

The ``DeliveringBus``/``BusClientView`` harness is imported from the shared
``autoresearch/testing/delivering_bus.py`` module (extracted from the copies
duplicated across earlier smokes, e.g. ``test_m8_planner_smoke.py``).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.coder_agent.service import build_coder_service
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
CODER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "coder_agent" / "data-contracts"
EXPERIMENT_PATCH_SCHEMA_PATH = CODER_CONTRACTS_DIR / "experiment_patch.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m9_coder_smoke.db"))
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
# Fixture data. The SAME fixed file ("src/model.py") is used in the
# VerifierSpec's editable_region and the CoderAgent's canned ExperimentPatch
# -- a mismatch would make the orchestrator's INDEPENDENT protocol gate
# find a violation, diverting the session back to ImplementingExperiment
# (or MossIntervening) instead of RunningExperiment.
# ---------------------------------------------------------------------------

VALID_EXPERIMENT_PLAN = {
    "hypothesis_id": "h1",
    "change_summary": "Add a learned gating layer before retrieval fusion.",
    "changes": [
        {"file": "src/model.py", "description": "Add a gating module class."},
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
    "editable_region": ["src/model.py"],
    "forbidden_files": ["eval/scorer.py"],
    "result_parse_schema": {"type": "object"},
}

# CoderAgent's canned response: a schema-valid ExperimentPatch whose sole
# changes[].file / touched_files entry ("src/model.py") is within
# VALID_VERIFIER_SPEC's editable_region and NOT in its forbidden_files.
VALID_EXPERIMENT_PATCH = {
    "hypothesis_id": "h1",
    "summary": "Add a learned gating layer before retrieval fusion.",
    "changes": [
        {"file": "src/model.py", "patch": "+class Gate(nn.Module):\n+    ..."},
    ],
    "touched_files": ["src/model.py"],
}


def _patch_json() -> str:
    return json.dumps(VALID_EXPERIMENT_PATCH)


# ===========================================================================
# Smoke: isolated ImplementingExperiment -> CoderAgent ->
# code.implemented -> orchestrator's INDEPENDENT protocol gate ->
# RunningExperiment (pauses; M10 RunnerAgent's run mode not built).
# ===========================================================================


async def test_implementing_experiment_reaches_running_experiment_single_session(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- the gate reads the
    # frozen VerifierSpec independently of the CoderAgent's payload) +
    # CoderAgent -- both on the SAME bus, each with its own emitter identity.
    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()

    coder_llm = FakeLLM([_patch_json()])
    build_coder_service(
        client=bus.view("CoderAgent"),
        spec_store=store,
        llm_client=coder_llm,
    )

    # --- Isolated construction: ImplementingExperiment is NOT reachable from
    # start_session yet (ReviewingExperimentPlan -> ImplementingExperiment,
    # M8, is not driven here), so seed its two upstream inputs directly into
    # the store, as ExperimentPlannerAgent / EvaluatorAgent would have, then
    # construct + save a SessionState directly in ImplementingExperiment with
    # both refs set.
    sid = str(uuid.uuid4())
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=sid,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
    experiment_plan_ref = store.put(
        kind="experiment_plan",
        session_id=sid,
        key="v1",
        obj=VALID_EXPERIMENT_PLAN,
        emitter="ExperimentPlannerAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ImplementingExperiment",
        current_experiment_id="exp1",
        current_hypothesis_id="h1",
        experiment_plan_refs=[experiment_plan_ref],
        verifier_spec_ref=verifier_spec_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits the
    # ImplementingExperiment entry action (code.implementation_requested) ->
    # (delivered) -> CoderAgent resolves the ExperimentPlan + VerifierSpec
    # from the store, runs its (canned) implement_patch, persists the
    # ExperimentPatch, and publishes code.implemented -> (delivered) ->
    # orchestrator's independent protocol gate appends the ref to
    # experiment_patch_refs, reads the FROZEN VerifierSpec from the store,
    # recomputes touched_files vs editable_region/forbidden_files, finds no
    # violation, resets MAP counter F (implementation_fix_round["exp1"],
    # delete-key), and transitions ImplementingExperiment -> RunningExperiment
    # (no entry action -- M10 RunnerAgent's run mode not built -- so the loop
    # pauses there). Called exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "RunningExperiment"

    # experiment_patch_refs has exactly the one ref appended by the gate, and
    # it resolves to a schema-valid ExperimentPatch whose touched_files are a
    # subset of the VerifierSpec's editable_region.
    assert len(final_state.experiment_patch_refs) == 1
    experiment_patch = store.get(final_state.experiment_patch_refs[-1])
    patch_errors = list(_validator(EXPERIMENT_PATCH_SCHEMA_PATH).iter_errors(experiment_patch))
    assert patch_errors == [], [e.message for e in patch_errors]

    editable_region = set(VALID_VERIFIER_SPEC["editable_region"])
    touched_files = set(experiment_patch["touched_files"])
    assert touched_files
    assert touched_files <= editable_region

    # MAP counter F (implementation_fix_round) reset = delete-key.
    assert "exp1" not in final_state.implementation_fix_round

    # Exactly one code.implemented.
    implemented_events = [e for e in bus.published if e.event_type == "code.implemented"]
    assert len(implemented_events) == 1
    assert implemented_events[0].session_id == sid
    assert implemented_events[0].payload["experiment_patch_ref"] == final_state.experiment_patch_refs[-1]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
