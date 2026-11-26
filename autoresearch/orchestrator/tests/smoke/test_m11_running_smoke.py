"""ImplementingExperiment -> RunningExperiment -> EvaluatingExperiment
isolated integration smoke (M11 Task 5): single session, no Redis, no real
LLM, no real execution.

Extends ``test_m9_coder_smoke.py`` one hop further along the M9 plan/code
chain: ImplementingExperiment -> RunningExperiment -> EvaluatingExperiment.
The M9 smoke stopped at RunningExperiment because M10's RunnerAgent run mode
was not yet built; this smoke adds RunnerAgent (``build_runner_service``,
wired with a ``FakeRunnerBackend``) on the SAME ``DeliveringBus`` so the
cascade continues past RunningExperiment's entry action
(``experiment.run_requested``) all the way to EvaluatingExperiment (which
pauses -- M12's EvaluatorAgent evaluate mode is not built).

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
     editable_region), persists the ExperimentPatch, and publishes
     ``code.implemented`` -> (delivered) -> orchestrator's INDEPENDENT
     protocol gate reads the FROZEN VerifierSpec from the store, recomputes
     ``touched_files`` vs ``editable_region``/``forbidden_files``, finds no
     violation, and transitions ImplementingExperiment -> RunningExperiment
     (entry action emits ``experiment.run_requested``) -> (delivered) ->
     RunnerAgent resolves the ExperimentPatch + VerifierSpec from the store,
     runs its (canned, ``exit_code=0``) ``run_experiment`` against
     ``FakeRunnerBackend``, persists the RunRecord, and publishes
     ``experiment.run_completed`` (verdict ``succeeded``) -> (delivered) ->
     orchestrator's RunningExperiment handler appends the ref, resets MAP
     counter F (``implementation_fix_round["exp1"]``, delete-key -- the ONLY
     place F is reset on the implement/run loop), and transitions
     RunningExperiment -> EvaluatingExperiment, which has an entry action
     (``evaluator.evaluate_requested``) but no subscriber wired in this
     smoke (M12 EvaluatorAgent's evaluate mode not built), so the loop pauses
     there.

No concurrency, no Redis, no real LLM/execution call anywhere in this file
(one FakeLLM canned queue + one FakeRunnerBackend canned result).
``resume_to_current_state`` is called exactly once -- a second call would
re-emit the entry action and break the "exactly one" event counts.

CRITICAL coordination (mirrors M9's smoke):
  - CoderAgent's canned patch's ``touched_files``/``changes[].file`` MUST be
    a subset of the seeded VerifierSpec's ``editable_region`` -- else the M9
    gate finds a violation and routes back to ImplementingExperiment (or
    MossIntervening) instead of RunningExperiment.
  - ``FakeRunnerBackend(exit_code=0, ...)`` MUST be used -- a nonzero exit
    code would make RunnerAgent publish ``verdict="failed"``, routing the
    session back to ImplementingExperiment instead of EvaluatingExperiment.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.coder_agent.service import build_coder_service
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
CODER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "coder_agent" / "data-contracts"
RUNNER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "runner_agent" / "data-contracts"
EXPERIMENT_PATCH_SCHEMA_PATH = CODER_CONTRACTS_DIR / "experiment_patch.schema.yaml"
RUN_RECORD_SCHEMA_PATH = RUNNER_CONTRACTS_DIR / "run_record.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m11_running_smoke.db"))
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
# Fixture data (mirrors test_m9_coder_smoke.py). The SAME fixed file
# ("src/model.py") is used in the VerifierSpec's editable_region and the
# CoderAgent's canned ExperimentPatch -- a mismatch would make the
# orchestrator's INDEPENDENT protocol gate find a violation, diverting the
# session away from RunningExperiment.
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
# RunningExperiment -> RunnerAgent -> experiment.run_completed(succeeded) ->
# EvaluatingExperiment (pauses; M12 EvaluatorAgent's evaluate mode not built).
# ===========================================================================


async def test_implementing_experiment_reaches_evaluating_experiment_single_session(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- the M9 gate reads the
    # frozen VerifierSpec independently of the CoderAgent's payload) +
    # CoderAgent + RunnerAgent -- all three on the SAME bus, each with its
    # own emitter identity.
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

    runner_backend = FakeRunnerBackend(exit_code=0, metric_value=0.91, stdout="ok")
    build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=runner_backend,
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
    # (delivered) -> CoderAgent -> code.implemented -> (delivered) ->
    # orchestrator's M9 gate -> RunningExperiment (entry action emits
    # experiment.run_requested) -> (delivered) -> RunnerAgent ->
    # experiment.run_completed(succeeded) -> (delivered) -> orchestrator's
    # RunningExperiment handler resets F and transitions to
    # EvaluatingExperiment (entry action emits evaluator.evaluate_requested,
    # no subscriber wired here -- M12 not built -- so the loop pauses there).
    # Called exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "EvaluatingExperiment"

    # experiment_patch_refs resolves to a schema-valid ExperimentPatch whose
    # touched_files are a subset of the VerifierSpec's editable_region.
    assert len(final_state.experiment_patch_refs) == 1
    experiment_patch = store.get(final_state.experiment_patch_refs[-1])
    patch_errors = list(_validator(EXPERIMENT_PATCH_SCHEMA_PATH).iter_errors(experiment_patch))
    assert patch_errors == [], [e.message for e in patch_errors]

    editable_region = set(VALID_VERIFIER_SPEC["editable_region"])
    touched_files = set(experiment_patch["touched_files"])
    assert touched_files
    assert touched_files <= editable_region

    # The last run_record_refs entry resolves to a schema-valid RunRecord
    # with succeeded==True.
    assert len(final_state.run_record_refs) == 1
    run_record = store.get(final_state.run_record_refs[-1])
    run_record_errors = list(_validator(RUN_RECORD_SCHEMA_PATH).iter_errors(run_record))
    assert run_record_errors == [], [e.message for e in run_record_errors]
    assert run_record["succeeded"] is True

    # MAP counter F (implementation_fix_round) reset = delete-key (F reset on
    # run-success, NOT on the M9 gate's compliant branch).
    assert "exp1" not in final_state.implementation_fix_round

    # Exactly one code.implemented + one experiment.run_completed(succeeded).
    implemented_events = [e for e in bus.published if e.event_type == "code.implemented"]
    assert len(implemented_events) == 1
    assert implemented_events[0].session_id == sid
    assert implemented_events[0].payload["experiment_patch_ref"] == final_state.experiment_patch_refs[-1]

    run_completed_events = [e for e in bus.published if e.event_type == "experiment.run_completed"]
    assert len(run_completed_events) == 1
    assert run_completed_events[0].session_id == sid
    assert run_completed_events[0].payload["verdict"] == "succeeded"
    assert run_completed_events[0].payload["run_record_ref"] == final_state.run_record_refs[-1]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
