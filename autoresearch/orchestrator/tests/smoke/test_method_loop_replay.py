"""Deterministic replay cases for the confirmed-method experiment loop.

These tests replay frozen events and artifacts; they never call an LLM,
network service, sandbox, or GPU.  Their purpose is to catch cross-agent
routing, restart, and duplicate-delivery regressions before a live run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
import pytest

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler, _map_repo_result
from autoresearch.eventbus.broker.idempotency import (
    has_processed,
    init_idempotency,
    record_processed,
)
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient
from autoresearch.spec_store import SpecStore


SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Event:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)
    event_id: str = "event-1"


@pytest.fixture
async def replay_env(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "replay.db"))
    store = SpecStore(tmp_path / "store")
    try:
        yield FakeBusClient(conn), store
    finally:
        store.close()
        await conn.close()


async def _seed_running(bus, *, session_id: str, method_ref: str) -> None:
    state = SessionState(
        session_id=session_id,
        user_id="replay-user",
        current_state="RunningExperiment",
        method_ref=method_ref,
        current_experiment_id="experiment-1",
        research_spec_ref=f"research_spec://{session_id}/reinterpreted",
        spec_interpretation_ref=f"spec_interpretation://{session_id}/v1",
        benchmark_spec_ref=f"benchmark_spec://{session_id}/final",
        baseline_result_ref=f"baseline_result://{session_id}/final",
        experiment_plan_refs=[f"experiment_plan://{session_id}/final"],
    )
    await save_session_state(bus.db, state, auto_commit=True)


async def _tick(engine, event_type, session_id, emitter, payload, event_id):
    await engine.tick(_Event(
        event_type=event_type,
        session_id=session_id,
        emitter=emitter,
        payload=payload,
        event_id=event_id,
    ))


def _component_method() -> dict:
    return {
        "approach": "create a usable representation, then optimize with a new loss",
        "components": [
            {
                "component_id": "representation",
                "name": "Representation split",
                "type": "functional",
                "depends_on": [],
                "purpose": "provide separated inputs required by the later loss",
                "inputs": ["coupled feature"],
                "outputs": ["two separated features"],
                "acceptance_requires": [],
                "mechanism": "split the coupled feature",
                "expected_effect": "expose two usable outputs",
                "functional_check": "separation_score >= 0.9",
                "implementation_check": "diff and runtime output show two separated features",
                "optimization_space": "improve the separation while preserving its interface",
                "downstream_contract": "provide both separated features to the gain loss",
            },
            {
                "component_id": "loss",
                "name": "Gain loss",
                "type": "gain",
                "depends_on": ["representation"],
                "purpose": "turn the separated representation into final metric gain",
                "inputs": ["two separated features"],
                "outputs": ["optimized prediction"],
                "acceptance_requires": ["representation"],
                "mechanism": "optimize the separated outputs",
                "expected_effect": "improve the repository metric",
                "ablation_test": "signed_metric_improvement > 0",
                "implementation_check": "diff shows the confirmed loss consumes both features",
                "optimization_space": "tune the loss without changing its declared mechanism",
                "downstream_contract": "provide the final candidate to integration",
            },
        ],
    }


def _functional_result(*, valid: bool = True) -> dict:
    return {
        "verdict": "valid_not_best" if valid else "invalid",
        "accepted": False,
        "baseline_metric": 1.0,
        "method_metric": 1.1 if valid else None,
        "lower_is_better": True,
        "repo_path": "/checkpoints/representation" if valid else "",
        "diff": "representation diff" if valid else "",
        "method_implementation_review": {
            "verdict": "faithful" if valid else "insufficient_evidence"},
        "ablation_report": ([{
            "_host_bound": True, "component": "representation", "actual": 0.95,
            "evidence": {"source": "runtime", "ref": "attempt:functional",
                         "json_pointer": "/separation_score"},
        }] if valid else []),
    }


def _gain_result() -> dict:
    return {
        "verdict": "new_best", "accepted": True,
        "baseline_metric": 1.0, "method_metric": 0.9,
        "lower_is_better": True, "repo_path": "/checkpoints/loss",
        "diff": "loss diff",
        "method_implementation_review": {"verdict": "faithful"},
    }


async def _replay_planned_round(
        engine, store, sid: str, *, suffix: str, evaluation_verdict: str,
        memory_diagnosis: str, memory_routing: str) -> None:
    """Replay the production plan→implement→run→evaluate→verify→memory event chain."""
    await _tick(engine, "experiment.plan_ready", sid, "ExperimentPlannerAgent", {
        "experiment_plan_ref": f"experiment_plan://{sid}/{suffix}",
    }, f"plan-{suffix}")
    await _tick(engine, "experiment.plan_reviewed", sid, "ExperimentReviewAgent", {
        "review_report_ref": f"review_report://{sid}/{suffix}", "verdict": "pass",
    }, f"review-{suffix}")
    patch_ref = store.put(
        kind="experiment_patch", session_id=sid, key=suffix,
        obj={"touched_files": ["method.py"]}, emitter="CoderAgent")
    await _tick(engine, "code.implemented", sid, "CoderAgent", {
        "experiment_patch_ref": patch_ref,
    }, f"code-{suffix}")
    await _tick(engine, "experiment.run_completed", sid, "RunnerAgent", {
        "run_record_ref": f"run_record://{sid}/{suffix}", "verdict": "succeeded",
    }, f"run-{suffix}")
    await _tick(engine, "experiment.evaluated", sid, "EvaluatorAgent", {
        "evaluation_report_ref": f"evaluation_report://{sid}/{suffix}",
        "verdict": evaluation_verdict,
    }, f"evaluate-{suffix}")
    await _tick(engine, "result.verified", sid, "EvaluatorAgent", {
        "verification_report_ref": f"verification_report://{sid}/{suffix}",
        "verdict": "verified",
    }, f"verify-{suffix}")
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/{suffix}",
        "routing_verdict": memory_routing,
        "diagnosis": memory_diagnosis,
        "diagnosis_evidence": [f"fixed replay evidence: {suffix}"],
    }, f"memory-{suffix}")


async def test_replay_failed_attempt_is_verified_corrected_and_reaches_done(replay_env):
    """A measured failure is verified, fed back, corrected, and reported once."""
    bus, store = replay_env
    engine = OrchestratorEngine(
        client=bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    sid = "replay-failure-correction"
    method_ref = store.put(
        kind="method", session_id=sid, key="current",
        obj={"approach": "Use gradient clipping with warmup."},
        emitter="MethodAgent",
    )
    await _seed_running(bus, session_id=sid, method_ref=method_ref)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "verdict": "invalid", "accepted": False, "method_metric": None,
            "diagnosis": "execution_failure",
            "diagnosis_evidence": ["CUDA out of memory during backward"],
        },
        emitter="RunnerAgent",
    )

    # Attempt one: the wrapper returned, then Verification confirms the failure
    # before Memory routes the concrete cause back to implementation.
    await _tick(engine, "experiment.run_completed", sid, "RunnerAgent", {
        "run_record_ref": f"run_record://{sid}/attempt-1", "verdict": "succeeded",
    }, "run-1")
    await _tick(engine, "experiment.evaluated", sid, "EvaluatorAgent", {
        "evaluation_report_ref": f"evaluation_report://{sid}/attempt-1",
        "verdict": "invalid",
    }, "evaluate-1")
    await _tick(engine, "result.verified", sid, "EvaluatorAgent", {
        "verification_report_ref": f"verification_report://{sid}/attempt-1",
        "verdict": "verified",
    }, "verify-1")
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/attempt-1",
        "routing_verdict": "continue_explore",
        "diagnosis": "execution_failure",
        "diagnosis_evidence": ["CUDA out of memory during backward"],
    }, "memory-1")

    after_failure = await load_session_state(bus.db, sid)
    assert after_failure.current_state == "ImplementingExperiment"
    assert after_failure.implementation_fix_round == {"experiment-1": 1}
    assert after_failure.baseline_result_ref == f"baseline_result://{sid}/final"
    assert after_failure.method_ref == method_ref

    # Attempt two: corrected implementation succeeds, is numerically verified,
    # consolidated, reported, and reaches the terminal state.
    patch_ref = store.put(
        kind="experiment_patch", session_id=sid, key="attempt-2",
        obj={"touched_files": ["train.py"]}, emitter="CoderAgent",
    )
    await _tick(engine, "code.implemented", sid, "CoderAgent", {
        "experiment_patch_ref": patch_ref,
    }, "code-2")
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "verdict": "new_best", "accepted": True,
            "baseline_metric": 1.047, "method_metric": 1.044,
        },
        emitter="RunnerAgent",
    )
    await _tick(engine, "experiment.run_completed", sid, "RunnerAgent", {
        "run_record_ref": f"run_record://{sid}/attempt-2", "verdict": "succeeded",
    }, "run-2")
    await _tick(engine, "experiment.evaluated", sid, "EvaluatorAgent", {
        "evaluation_report_ref": f"evaluation_report://{sid}/attempt-2",
        "verdict": "new_best",
    }, "evaluate-2")
    await _tick(engine, "result.verified", sid, "EvaluatorAgent", {
        "verification_report_ref": f"verification_report://{sid}/attempt-2",
        "verdict": "verified",
    }, "verify-2")
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/final",
        "routing_verdict": "report",
        "diagnosis": "valid_improvement",
        "diagnosis_evidence": ["1.044 improves on 1.047"],
    }, "memory-2")
    await _tick(engine, "evidence.citation_audited", sid, "EvidenceAuditAgent", {
        "citation_audit_report_ref": f"citation_audit_report://{sid}/final",
    }, "citation-1")
    await _tick(engine, "report.ready", sid, "ReportAgent", {
        "report_ref": f"report://{sid}/final",
    }, "report-1")

    final = await load_session_state(bus.db, sid)
    assert final.current_state == "Done"
    assert final.final_report_ref == f"report://{sid}/final"
    assert final.baseline_result_ref == f"baseline_result://{sid}/final"
    assert final.implementation_fix_round == {}


async def test_replay_functional_then_gain_then_integration_reaches_done(
        replay_env, tmp_path, monkeypatch):
    """A no-gain prerequisite advances, then a gain node, then final integration reports."""
    bus, store = replay_env
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    sid = "replay-component-success"
    method_ref = store.put(
        kind="method", session_id=sid, key="current",
        obj=_component_method(), emitter="MethodAgent")
    runner = RunnerAgentHandler(
        spec_store=store, backend=FakeRunnerBackend(), p1_llm=object())
    engine = OrchestratorEngine(client=bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    await _seed_running(bus, session_id=sid, method_ref=method_ref)

    assert "CURRENT COMPONENT [representation]" in runner._with_operator(sid, "run")
    functional = runner._finalize_component_attempt(sid, _functional_result())
    assert functional["component_execution"]["status"] == "qualified"
    # A first pass keeps the same component active and schedules an explicit IMPROVE round.
    state = await load_session_state(bus.db, sid)
    state.current_state = "ConsolidatingMemory"
    await save_session_state(bus.db, state, auto_commit=True)
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/functional",
        "routing_verdict": "need_experiment", "diagnosis": "component_progress",
        "diagnosis_evidence": ["representation functional threshold passed"],
    }, "memory-functional")
    assert (await load_session_state(bus.db, sid)).current_state == "PlanningExperiment"
    assert "operator=IMPROVE" in runner._with_operator(sid, "improve representation")
    accepted_functional = runner._finalize_component_attempt(sid, _functional_result())
    assert accepted_functional["component_execution"]["status"] == "accepted"
    await _replay_planned_round(
        engine, store, sid, suffix="functional-improve",
        evaluation_verdict="valid_not_best",
        memory_diagnosis="component_accepted", memory_routing="need_experiment")

    assert "CURRENT COMPONENT [loss]" in runner._with_operator(sid, "run")
    gain = runner._finalize_component_attempt(sid, _gain_result())
    assert gain["component_execution"]["status"] == "qualified"
    await _replay_planned_round(
        engine, store, sid, suffix="gain", evaluation_verdict="new_best",
        memory_diagnosis="component_progress", memory_routing="need_experiment")
    assert "operator=IMPROVE" in runner._with_operator(sid, "improve gain")
    accepted_gain = runner._finalize_component_attempt(sid, _gain_result())
    assert accepted_gain["component_execution"]["status"] == "accepted"
    await _replay_planned_round(
        engine, store, sid, suffix="gain-improve", evaluation_verdict="new_best",
        memory_diagnosis="component_accepted", memory_routing="need_experiment")
    assert (await load_session_state(bus.db, sid)).current_state == "PlanningExperiment"

    assert "COMPONENT INTEGRATION" in runner._with_operator(sid, "integrate")
    integrated = runner._finalize_component_attempt(sid, {
        "verdict": "new_best", "accepted": True,
        "baseline_metric": 1.0, "method_metric": 0.85,
        "lower_is_better": True, "repo_path": "/checkpoints/integrated",
        "method_implementation_review": {"verdict": "faithful"},
    })
    assert integrated["component_execution"]["mode"] == "integration"
    await _replay_planned_round(
        engine, store, sid, suffix="integration", evaluation_verdict="new_best",
        memory_diagnosis="valid_improvement", memory_routing="report")
    await _tick(engine, "evidence.citation_audited", sid, "EvidenceAuditAgent", {
        "citation_audit_report_ref": f"citation_audit_report://{sid}/final",
    }, "citation-final")
    await _tick(engine, "report.ready", sid, "ReportAgent", {
        "report_ref": f"report://{sid}/final",
    }, "report-final")

    final = await load_session_state(bus.db, sid)
    assert final.current_state == "Done"
    assert final.final_report_ref == f"report://{sid}/final"


async def test_replay_component_failure_debug_then_final_no_gain_replans_composition(
        replay_env, tmp_path, monkeypatch):
    """A failed node stays local; after all nodes pass, a failed integration swaps Method."""
    bus, store = replay_env
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    sid = "replay-component-debug-swap"
    method_ref = store.put(
        kind="method", session_id=sid, key="current",
        obj=_component_method(), emitter="MethodAgent")
    runner = RunnerAgentHandler(
        spec_store=store, backend=FakeRunnerBackend(), p1_llm=object())
    engine = OrchestratorEngine(client=bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    await _seed_running(bus, session_id=sid, method_ref=method_ref)

    runner._with_operator(sid, "run")
    failed = runner._finalize_component_attempt(sid, _functional_result(valid=False))
    assert failed["component_execution"]["status"] == "execution_failed"
    assert "operator=DEBUG" in runner._with_operator(sid, "retry")

    # The fixed failure replay goes through Evaluation and Verification before Memory sends it
    # back to implementation; it neither advances the DAG nor changes the frozen Method.
    await _tick(engine, "experiment.run_completed", sid, "RunnerAgent", {
        "run_record_ref": f"run_record://{sid}/failed", "verdict": "succeeded",
    }, "run-failed")
    await _tick(engine, "experiment.evaluated", sid, "EvaluatorAgent", {
        "evaluation_report_ref": f"evaluation_report://{sid}/failed", "verdict": "invalid",
    }, "eval-failed")
    await _tick(engine, "result.verified", sid, "EvaluatorAgent", {
        "verification_report_ref": f"verification_report://{sid}/failed",
        "verdict": "verified",
    }, "verify-failed")
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/failed",
        "routing_verdict": "need_experiment", "diagnosis": "component_contract_retry",
        "diagnosis_evidence": ["representation candidate crashed"],
    }, "memory-failed")
    state = await load_session_state(bus.db, sid)
    assert state.current_state == "ImplementingExperiment"
    assert state.method_ref == method_ref

    # Deterministically accept the repaired node and its dependent gain node, then replay the
    # integrated candidate as a valid but non-improving result.
    runner._finalize_component_attempt(sid, _functional_result())
    runner._with_operator(sid, "improve functional")
    runner._finalize_component_attempt(sid, _functional_result())
    runner._with_operator(sid, "gain")
    runner._finalize_component_attempt(sid, _gain_result())
    runner._with_operator(sid, "improve gain")
    runner._finalize_component_attempt(sid, _gain_result())
    assert "COMPONENT INTEGRATION" in runner._with_operator(sid, "integrate")
    integrated = runner._finalize_component_attempt(sid, {
        "verdict": "valid_not_best", "accepted": False,
        "baseline_metric": 1.0, "method_metric": 1.02,
        "lower_is_better": True, "repo_path": "/checkpoints/integrated",
        "method_implementation_review": {"verdict": "faithful"},
    })
    assert integrated["component_execution"]["all_components_accepted"] is True

    patch_ref = store.put(
        kind="experiment_patch", session_id=sid, key="integration",
        obj={"touched_files": ["method.py"]}, emitter="CoderAgent")
    await _tick(engine, "code.implemented", sid, "CoderAgent", {
        "experiment_patch_ref": patch_ref,
    }, "code-integration")
    await _tick(engine, "experiment.run_completed", sid, "RunnerAgent", {
        "run_record_ref": f"run_record://{sid}/integration", "verdict": "succeeded",
    }, "run-integration")
    await _tick(engine, "experiment.evaluated", sid, "EvaluatorAgent", {
        "evaluation_report_ref": f"evaluation_report://{sid}/integration",
        "verdict": "valid_not_best",
    }, "eval-integration")
    await _tick(engine, "result.verified", sid, "EvaluatorAgent", {
        "verification_report_ref": f"verification_report://{sid}/integration",
        "verdict": "verified",
    }, "verify-integration")
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/integration",
        "routing_verdict": "need_experiment",
        "diagnosis": "component_combination_retry",
        "diagnosis_evidence": ["all components passed but integration did not improve"],
    }, "memory-integration")
    replanned = await load_session_state(bus.db, sid)
    assert replanned.current_state == "PlanningExperiment"
    assert replanned.method_ref == method_ref
    assert replanned.method_not_working is False


async def test_replay_partial_method_review_accepts_real_measured_win(
        replay_env, monkeypatch):
    """A real partial implementation win is accepted without crediting missing components."""
    bus, store = replay_env
    sid = "replay-partial-method"
    method_ref = store.put(
        kind="method", session_id=sid, key="current",
        obj={
            "approach": "Combine gradient clipping with warmup.",
            "components": [
                {"name": "gradient clipping", "type": "gain"},
                {"name": "warmup", "type": "gain"},
            ],
        },
        emitter="MethodAgent",
    )
    review = {
        "verdict": "partial",
        "approach_evidence": ["gradient_clip = 1.0"],
        "component_reviews": [
            {"name": "gradient clipping", "status": "implemented",
             "evidence": ["gradient_clip = 1.0"]},
            {"name": "warmup", "status": "missing", "evidence": []},
        ],
        "issues": ["warmup is missing"],
        "remediation": "Implement the confirmed warmup component.",
        "reasoning": "Only one of two gain components is present.",
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: review,
    )
    runner = RunnerAgentHandler(
        spec_store=store, backend=FakeRunnerBackend(), p1_llm=object())
    reviewed = runner._review_method_implementation(
        sid,
        {
            "baseline": {"metric": 1.047},
            "candidate": {"valid": True, "metric": 1.044},
            "verdict": {"improved": True},
            "diff": "gradient_clip = 1.0",
        },
        "method",
    )
    _, gen_result = _map_repo_result("replay", reviewed)
    assert gen_result["method_metric"] == 1.044
    assert gen_result["accepted"] is True
    assert gen_result["diagnosis"] is None
    assert "only the implemented frozen components may be credited" \
        in gen_result["reasons"][0]

    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj=gen_result, emitter="RunnerAgent",
    )
    engine = OrchestratorEngine(
        client=bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    await _seed_running(bus, session_id=sid, method_ref=method_ref)
    await _tick(engine, "experiment.run_completed", sid, "RunnerAgent", {
        "run_record_ref": f"run_record://{sid}/partial", "verdict": "succeeded",
    }, "partial-run")
    await _tick(engine, "experiment.evaluated", sid, "EvaluatorAgent", {
        "evaluation_report_ref": f"evaluation_report://{sid}/partial",
        "verdict": "new_best",
    }, "partial-evaluate")
    await _tick(engine, "result.verified", sid, "EvaluatorAgent", {
        "verification_report_ref": f"verification_report://{sid}/partial",
        "verdict": "verified",
    }, "partial-verify")
    await _tick(engine, "memory.consolidated", sid, "MemoryAgent", {
        "research_memory_ref": f"research_memory://{sid}/partial",
        "routing_verdict": "report",
        "diagnosis": "valid_improvement",
        "diagnosis_evidence": ["verified partial implementation improvement"],
    }, "partial-memory")

    state = await load_session_state(bus.db, sid)
    assert state.current_state == "Reporting"
    assert state.method_ref == method_ref
    assert state.baseline_result_ref == f"baseline_result://{sid}/final"
    assert state.exploration_loop_round == 0
    assert state.implementation_fix_round == {}


async def test_replay_restart_and_duplicate_delivery_preserve_exactly_once_state(replay_env):
    """Restart re-drives one state; broker replay does not duplicate its effect."""
    bus, store = replay_env
    sid = "replay-restart-duplicate"
    bootstrap = OrchestratorEngine(
        client=bus, skill_root=SM_ROOT, spec_store=store)
    await bootstrap.start()
    state = SessionState(
        session_id=sid,
        current_state="ImplementingExperiment",
        method_ref=f"method://{sid}/current",
        current_experiment_id="experiment-1",
        research_spec_ref=f"research_spec://{sid}/reinterpreted",
        benchmark_spec_ref=f"benchmark_spec://{sid}/final",
        baseline_result_ref=f"baseline_result://{sid}/final",
        experiment_plan_refs=[f"experiment_plan://{sid}/final"],
        implementation_fix_round={"experiment-1": 1},
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # A fresh engine instance reads the durable state and re-emits only the
    # current entry action; it does not reset refs or counters.
    restarted = OrchestratorEngine(
        client=bus, skill_root=SM_ROOT, spec_store=store)
    await restarted.start()
    result = await restarted.resume_to_current_state(sid)
    resumed = await load_session_state(bus.db, sid)
    assert result["resumed"] is True
    assert resumed.current_state == "ImplementingExperiment"
    assert resumed.implementation_fix_round == {"experiment-1": 1}
    assert resumed.baseline_result_ref == f"baseline_result://{sid}/final"
    assert bus.published[-1].event_type == "code.implementation_requested"

    # Replay the same provider event through the production idempotency
    # boundary.  Reporting stays in-state, so an unguarded duplicate would
    # visibly publish report.requested twice.
    resumed.current_state = "Reporting"
    resumed.research_memory_ref = f"research_memory://{sid}/final"
    await save_session_state(bus.db, resumed, auto_commit=True)
    await init_idempotency(bus.db)
    duplicate = _Event(
        event_type="evidence.citation_audited",
        session_id=sid,
        emitter="EvidenceAuditAgent",
        payload={
            "citation_audit_report_ref": f"citation_audit_report://{sid}/final",
        },
        event_id="provider-event-1",
    )

    async def deliver_once():
        if await has_processed(bus.db, "ResearchOrchestrator", duplicate.event_id):
            return
        await restarted.tick(duplicate)
        await record_processed(
            bus.db, "ResearchOrchestrator", duplicate.event_id,
            "autoresearch:events:replay", "1-0")
        await bus.db.commit()

    before = len([event for event in bus.published
                  if event.event_type == "report.requested"])
    await deliver_once()
    await deliver_once()
    after = len([event for event in bus.published
                 if event.event_type == "report.requested"])
    assert after - before == 1
