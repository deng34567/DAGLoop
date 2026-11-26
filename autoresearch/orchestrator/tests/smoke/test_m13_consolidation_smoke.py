"""ImplementingExperiment -> RunningExperiment -> EvaluatingExperiment ->
VerifyingBestResult -> ConsolidatingMemory -> GeneratingHypotheses isolated
integration smoke (M13 Task 6): single session, no Redis, no real LLM, no
real execution -- PROVES THE EXPLORATION LOOP CLOSES.

Extends ``test_m12_evaluation_smoke.py`` one hop further along the M9-M13
plan/code/run/evaluate/consolidate chain: ImplementingExperiment ->
RunningExperiment -> EvaluatingExperiment -> VerifyingBestResult ->
ConsolidatingMemory -> GeneratingHypotheses. The M12 smoke stopped at
ConsolidatingMemory because M13's MemoryAgent was not yet built; this smoke
adds MemoryAgent (``build_memory_service``, wired with a ``FakeLLM`` canned
queue: one ``consolidate`` response with ``routing_verdict=continue_explore``)
on the SAME ``DeliveringBus`` so the cascade continues past ConsolidatingMemory
all the way back to GeneratingHypotheses -- THE CONNECTOR that closes the
exploration loop (state machine V1.10 SS1).

Drives, in one linear session:
  1. Seed a VerifierSpec, an ExperimentPlan, and a BenchmarkSpec (whose
     ``baseline.reported_metric`` (0.85) is LOWER than the RunRecord's
     ``metric_value`` (0.90) the FakeRunnerBackend will produce) directly
     into the Spec Store, as upstream agents would have.
  2. Construct + save a ``SessionState`` directly in
     ``ImplementingExperiment`` with ``current_experiment_id="exp1"``,
     ``experiment_plan_refs``, ``verifier_spec_ref``, AND
     ``benchmark_spec_ref`` set (isolated construction -- no prior states are
     driven).
  3. ``eng.resume_to_current_state(session_id)`` re-emits the
     ImplementingExperiment entry action, ``code.implementation_requested``
     -> (delivered) -> CoderAgent -> ``code.implemented`` -> (delivered) ->
     orchestrator's INDEPENDENT protocol gate (no violation) ->
     RunningExperiment (entry action emits ``experiment.run_requested``) ->
     (delivered) -> RunnerAgent (``FakeRunnerBackend(exit_code=0,
     metric_value=0.90, stdout="ok")``) -> ``experiment.run_completed``
     (verdict ``succeeded``) -> (delivered) -> orchestrator resets MAP
     counter F and transitions to EvaluatingExperiment (entry action emits
     ``experiment.evaluation_requested``) -> (delivered) -> EvaluatorAgent
     resolves the refs, runs ``core.evaluate_experiment`` (deterministic
     guard computes ``is_improvement=True`` since 0.90 > 0.85, consistent
     with the canned ``new_best`` verdict), persists the EvaluationReport,
     publishes ``experiment.evaluated`` (verdict ``new_best``) -> (delivered)
     -> orchestrator appends the ref and transitions to VerifyingBestResult
     (entry action emits ``result.verification_requested``) -> (delivered)
     -> EvaluatorAgent resolves the refs, runs ``core.verify_best_result``,
     persists the VerificationReport, publishes ``result.verified`` (verdict
     ``verified``) -> (delivered) -> orchestrator resets MAP counter G and
     transitions to ConsolidatingMemory (entry action emits
     ``memory.consolidation_requested`` with ``evaluation_report_ref``,
     ``run_record_ref``, ``research_memory_ref`` (None -- first cycle),
     ``verification_report_ref``) -> (delivered) -> MemoryAgent resolves the
     EvaluationReport + RunRecord + VerificationReport (no prior
     ResearchMemory -- first cycle), runs ``core.consolidate`` (the
     deterministic best-tracking guard computes ``best_metric=0.90`` since
     verdict==new_best AND verification verdict==verified), persists the
     ResearchMemory, publishes ``memory.consolidated`` (routing_verdict
     ``continue_explore``) -> (delivered) -> orchestrator stores the
     ResearchMemory ref onto SessionState and transitions to
     GeneratingHypotheses -- its EXISTING M5 entry action fires
     ``hypothesis.generation_requested`` (no HypothesisAgent wired in this
     smoke), so the loop pauses there -- HAVING CLOSED THE CYCLE.

No concurrency, no Redis, no real LLM/execution call anywhere in this file
(one CoderAgent FakeLLM canned queue, one FakeRunnerBackend canned result,
one EvaluatorAgent FakeLLM canned queue of exactly TWO responses, one
MemoryAgent FakeLLM canned queue of exactly ONE response).
``resume_to_current_state`` is called exactly once -- a second call would
re-emit the entry action and break the "exactly one" event counts.

CRITICAL coordination:
  - CoderAgent's canned patch's ``touched_files``/``changes[].file`` MUST be
    a subset of the seeded VerifierSpec's ``editable_region`` -- else the
    gate finds a violation and routes back to ImplementingExperiment (or
    MossIntervening) instead of RunningExperiment.
  - ``FakeRunnerBackend(exit_code=0, ...)`` MUST be used -- a nonzero exit
    code would make RunnerAgent publish ``verdict="failed"``, routing the
    session back to ImplementingExperiment instead of EvaluatingExperiment.
  - The seeded BenchmarkSpec's ``baseline.reported_metric`` (0.85) MUST be
    LOWER than the RunRecord's ``metric_value`` (0.90) the FakeRunnerBackend
    produces -- so ``core.evaluate_experiment``'s deterministic
    anti-reward-hacking guard COMPUTES ``is_improvement=True``, making the
    EvaluatorAgent's canned ``new_best`` verdict CONSISTENT. This ALSO
    matters for MemoryAgent's own deterministic best-tracking guard
    (``core.consolidate``): only ``verdict==new_best`` AND a present,
    ``verdict==verified`` VerificationReport make it compute
    ``best_metric=run_record.metric_value`` (0.90) instead of carrying
    forward the (absent) prior.
  - The EvaluatorAgent FakeLLM's response queue order matters: the FIRST
    canned response is consumed by ``evaluate_experiment``
    (EvaluatingExperiment), the SECOND by ``verify_best_result``
    (VerifyingBestResult) -- these are two separate handler invocations
    sharing one FakeLLM instance.
  - The MemoryAgent FakeLLM's single canned response's ``routing_verdict``
    MUST be ``"continue_explore"`` -- any other value routes ConsolidatingMemory
    to Reporting or PlanningExperiment instead of GeneratingHypotheses, and the
    loop-close assertion (``hypothesis.generation_requested`` re-emitted)
    would fail.
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
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM as CoderFakeLLM
from autoresearch.agents.evaluator_agent.service import build_evaluator_service
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM as EvaluatorFakeLLM
from autoresearch.agents.memory_agent.service import build_memory_service
from autoresearch.agents.memory_agent.tests.conftest import FakeLLM as MemoryFakeLLM
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
CODER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "coder_agent" / "data-contracts"
RUNNER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "runner_agent" / "data-contracts"
EVALUATOR_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "evaluator_agent" / "data-contracts"
MEMORY_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "memory_agent" / "data-contracts"
EXPERIMENT_PATCH_SCHEMA_PATH = CODER_CONTRACTS_DIR / "experiment_patch.schema.yaml"
RUN_RECORD_SCHEMA_PATH = RUNNER_CONTRACTS_DIR / "run_record.schema.yaml"
EVALUATION_REPORT_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "evaluation_report.schema.yaml"
VERIFICATION_REPORT_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "verification_report.schema.yaml"
RESEARCH_MEMORY_SCHEMA_PATH = MEMORY_CONTRACTS_DIR / "research_memory.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m13_consolidation_smoke.db"))
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
# Fixture data (mirrors test_m12_evaluation_smoke.py). The SAME fixed file
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

# BenchmarkSpec.baseline.reported_metric (0.85) is LOWER than the RunRecord's
# metric_value (0.90, set on FakeRunnerBackend below) -- so
# core.evaluate_experiment's deterministic guard computes
# is_improvement=True, making the canned "new_best" verdict consistent. This
# also feeds MemoryAgent's own best-tracking guard downstream.
VALID_BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.85,
    },
    "resource_requirements": "1x A100 GPU",
    "forbidden_files": ["eval/scorer.py"],
}

# EvaluatorAgent's canned LLM responses, consumed IN ORDER: the first by
# evaluate_experiment (EvaluatingExperiment), the second by
# verify_best_result (VerifyingBestResult).
EVALUATE_EXPERIMENT_LLM_RESPONSE = {
    "verdict": "new_best",
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": "0.90 beats the baseline 0.85.",
}

VERIFY_BEST_RESULT_LLM_RESPONSE = {
    "verdict": "verified",
    "checks": ["cross-seed: fixed seed=42", "held-out: metric on held-out split"],
    "anti_cheat_flags": [],
    "reasoning": "Result reproduces under the frozen protocol.",
}

# MemoryAgent's single canned LLM response: routing_verdict MUST be
# "continue_explore" for ConsolidatingMemory to route to GeneratingHypotheses
# (the loop-close transition). best_metric/best_result_ref are overwritten by
# core.consolidate's deterministic guard regardless of what is put here.
CONSOLIDATE_LLM_RESPONSE = {
    "research_memory": {
        "verified_findings": ["Gating layer improves exact_match by 0.05 over baseline."],
        "negative_results": [],
        "excluded_hypotheses": [],
        "consolidated_insights": ["Learned gating before fusion helps long-tail retrieval."],
        "divergence_seeds": ["Try gating conditioned on retrieval confidence."],
        "next_steps": ["Generate a follow-up hypothesis exploring confidence-conditioned gating."],
    },
    "routing_verdict": "continue_explore",
}


def _patch_json() -> str:
    return json.dumps(VALID_EXPERIMENT_PATCH)


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ===========================================================================
# Smoke: isolated ImplementingExperiment -> CoderAgent ->
# code.implemented -> orchestrator's INDEPENDENT protocol gate ->
# RunningExperiment -> RunnerAgent -> experiment.run_completed(succeeded) ->
# EvaluatingExperiment -> EvaluatorAgent -> experiment.evaluated(new_best) ->
# VerifyingBestResult -> EvaluatorAgent -> result.verified(verified) ->
# ConsolidatingMemory -> MemoryAgent -> memory.consolidated(continue_explore)
# -> GeneratingHypotheses (hypothesis.generation_requested re-emitted -- THE
# EXPLORATION LOOP CLOSES).
# ===========================================================================


async def test_implementing_experiment_reaches_generating_hypotheses_loop_closed(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- the M9 gate + M12/M13
    # entry actions read frozen specs independently of the agents' payloads)
    # + CoderAgent + RunnerAgent + EvaluatorAgent + MemoryAgent -- all five on
    # the SAME bus, each with its own emitter identity.
    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()

    coder_llm = CoderFakeLLM([_patch_json()])
    build_coder_service(
        client=bus.view("CoderAgent"),
        spec_store=store,
        llm_client=coder_llm,
    )

    runner_backend = FakeRunnerBackend(exit_code=0, metric_value=0.90, stdout="ok")
    build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=runner_backend,
    )

    evaluator_llm = EvaluatorFakeLLM(
        [
            _json(EVALUATE_EXPERIMENT_LLM_RESPONSE),
            _json(VERIFY_BEST_RESULT_LLM_RESPONSE),
        ]
    )
    build_evaluator_service(
        client=bus.view("EvaluatorAgent"),
        spec_store=store,
        llm_client=evaluator_llm,
    )

    memory_llm = MemoryFakeLLM([_json(CONSOLIDATE_LLM_RESPONSE)])
    build_memory_service(
        client=bus.view("MemoryAgent"),
        spec_store=store,
        llm_client=memory_llm,
    )

    # --- Isolated construction: ImplementingExperiment is NOT reachable from
    # start_session yet, so seed its upstream inputs directly into the store,
    # as ExperimentPlannerAgent / EvaluatorAgent would have, then construct +
    # save a SessionState directly in ImplementingExperiment with
    # experiment_plan_refs, verifier_spec_ref, AND benchmark_spec_ref set --
    # the last is read directly off SessionState by EvaluatingExperiment's
    # entry action (never derived along the way).
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
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=sid,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ImplementingExperiment",
        current_experiment_id="exp1",
        current_hypothesis_id="h1",
        experiment_plan_refs=[experiment_plan_ref],
        verifier_spec_ref=verifier_spec_ref,
        benchmark_spec_ref=benchmark_spec_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits the
    # ImplementingExperiment entry action and cascades all the way to
    # GeneratingHypotheses -- CLOSING THE LOOP. Called exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None

    # THE LOOP-CLOSE PROOF: the session is back at GeneratingHypotheses, the
    # meeting 甲: the exploration loop-back now re-plans another implementation round of the
    # confirmed method (ConsolidatingMemory -> PlanningExperiment), NOT a new hypothesis portfolio.
    assert final_state.current_state == "PlanningExperiment"

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
    # with succeeded==True and metric_value==0.90.
    assert len(final_state.run_record_refs) == 1
    run_record = store.get(final_state.run_record_refs[-1])
    run_record_errors = list(_validator(RUN_RECORD_SCHEMA_PATH).iter_errors(run_record))
    assert run_record_errors == [], [e.message for e in run_record_errors]
    assert run_record["succeeded"] is True
    assert run_record["metric_value"] == 0.90

    # The last evaluation_report_refs entry resolves to a schema-valid
    # EvaluationReport with verdict=="new_best" and is_improvement==True (the
    # deterministic anti-reward-hacking guard computed this from
    # metric=0.90 > baseline=0.85, not from the LLM).
    assert len(final_state.evaluation_report_refs) == 1
    evaluation_report = store.get(final_state.evaluation_report_refs[-1])
    evaluation_report_errors = list(
        _validator(EVALUATION_REPORT_SCHEMA_PATH).iter_errors(evaluation_report)
    )
    assert evaluation_report_errors == [], [e.message for e in evaluation_report_errors]
    assert evaluation_report["verdict"] == "new_best"
    assert evaluation_report["is_improvement"] is True

    # The last verification_report_refs entry resolves to a schema-valid
    # VerificationReport with verdict=="verified".
    assert len(final_state.verification_report_refs) == 1
    verification_report = store.get(final_state.verification_report_refs[-1])
    verification_report_errors = list(
        _validator(VERIFICATION_REPORT_SCHEMA_PATH).iter_errors(verification_report)
    )
    assert verification_report_errors == [], [e.message for e in verification_report_errors]
    assert verification_report["verdict"] == "verified"

    # SessionState.research_memory_ref (stored by the ConsolidatingMemory ->
    # GeneratingHypotheses handler) resolves to a schema-valid ResearchMemory
    # whose best_metric is DETERMINISTICALLY 0.90 (run_record.metric_value),
    # NOT whatever the canned LLM response happened to omit -- proof of the
    # deterministic best-tracking guard (verdict==new_best AND a present,
    # verified VerificationReport).
    assert final_state.research_memory_ref is not None
    research_memory = store.get(final_state.research_memory_ref)
    research_memory_errors = list(
        _validator(RESEARCH_MEMORY_SCHEMA_PATH).iter_errors(research_memory)
    )
    assert research_memory_errors == [], [e.message for e in research_memory_errors]
    assert research_memory["best_metric"] == 0.90

    # Exactly one code.implemented + one experiment.run_completed(succeeded)
    # + one experiment.evaluated(new_best) + one result.verified(verified)
    # + one memory.consolidated(continue_explore).
    implemented_events = [e for e in bus.published if e.event_type == "code.implemented"]
    assert len(implemented_events) == 1
    assert implemented_events[0].session_id == sid

    run_completed_events = [e for e in bus.published if e.event_type == "experiment.run_completed"]
    assert len(run_completed_events) == 1
    assert run_completed_events[0].payload["verdict"] == "succeeded"

    evaluated_events = [e for e in bus.published if e.event_type == "experiment.evaluated"]
    assert len(evaluated_events) == 1
    assert evaluated_events[0].payload["verdict"] == "new_best"

    verified_events = [e for e in bus.published if e.event_type == "result.verified"]
    assert len(verified_events) == 1
    assert verified_events[0].payload["verdict"] == "verified"

    consolidated_events = [e for e in bus.published if e.event_type == "memory.consolidated"]
    assert len(consolidated_events) == 1
    assert consolidated_events[0].session_id == sid
    assert consolidated_events[0].payload["routing_verdict"] == "continue_explore"
    assert consolidated_events[0].payload["research_memory_ref"] == final_state.research_memory_ref

    # THE LOOP-CLOSE PROOF (event side, meeting 甲): experiment.planning_requested WAS
    # re-emitted -- PlanningExperiment's entry action fired again as the cycle returned to it,
    # closing the exploration loop as another implementation round (no hypothesis portfolio).
    planning_requested_events = [
        e for e in bus.published if e.event_type == "experiment.planning_requested"
    ]
    assert len(planning_requested_events) >= 1
    assert planning_requested_events[-1].session_id == sid

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
