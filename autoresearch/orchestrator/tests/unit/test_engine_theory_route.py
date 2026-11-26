"""Regression coverage for the executable idea_theory route."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import fake_bus  # noqa: F401
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store.identity import artifact_digest


SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)
    event_id: str = "evt-input"


async def test_idea_theory_never_enters_benchmark_or_baseline(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        sid = "sess-theory-route"
        research_spec_ref = store.put(
            kind="research_spec", session_id=sid, key="reinterpreted",
            obj={"task_mode": "idea_theory", "research_goal": "derive propositions"},
            emitter="ProblemModelingAgent",
        )
        interpretation_ref = store.put(
            kind="spec_interpretation", session_id=sid, key="v1",
            obj={"disposition": "out_of_scope", "scope_reason": "theory has no benchmark"},
            emitter="ProblemModelingAgent",
        )
        method_ref = store.put(
            kind="method", session_id=sid, key="v1",
            obj={"approach": "Axioms to propositions", "changes_outline": ["Theorem 1"]},
            emitter="MethodAgent",
        )
        engine = OrchestratorEngine(
            client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await engine.start()
        state = SessionState(
            session_id=sid,
            user_id="u1",
            current_state="ReinterpretingSpec",
            research_spec_ref=research_spec_ref,
        )
        await save_session_state(fake_bus.db, state, auto_commit=True)

        await engine.tick(_Evt(
            "spec.reinterpreted", sid, "ProblemModelingAgent",
            payload={
                "disposition": "out_of_scope",
                "spec_interpretation_ref": interpretation_ref,
                "revised_research_spec_ref": research_spec_ref,
            },
        ))
        modeled = await load_session_state(fake_bus.db, sid)
        assert modeled.mode == "idea_theory"
        assert modeled.current_state == "ModelingProblem"

        modeled.current_state = "ConfirmingMethod"
        modeled.method_ref = method_ref
        modeled.problem_model_ref = "spec://problem_model/theory/v1"
        modeled.spec_interpretation_ref = interpretation_ref
        await save_session_state(fake_bus.db, modeled, auto_commit=True)
        fake_bus.published.clear()

        await engine.tick(_Evt(
            "method.confirmed", sid, "HumanView",
            payload={"inject": False},
        ))

        final = await load_session_state(fake_bus.db, sid)
        assert final.current_state == "ConsolidatingMemory"
        assert not any(p.event_type in {
            "benchmark.selection_requested", "baseline.reproduction_requested"
        } for p in fake_bus.published)
        request = next(
            p for p in fake_bus.published
            if p.event_type == "memory.consolidation_requested")
        assert request.payload["theory_method_ref"] == method_ref
        assert request.payload["run_record_ref"] is None
    finally:
        store.close()


async def test_experiment_validation_enters_planning_without_benchmark(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        sid = "sess-validation-route"
        research_spec_ref = store.put(
            kind="research_spec", session_id=sid, key="reinterpreted",
            obj={"task_mode": "experiment_validation", "research_goal": "test a theory"},
            emitter="ProblemModelingAgent",
        )
        interpretation_ref = store.put(
            kind="spec_interpretation", session_id=sid, key="v1",
            obj={
                "disposition": "in_scope",
                "scope_reason": "a lightweight public-data study is runnable",
                "validation_study": {
                    "kind": "survey_validation",
                    "question": "Does the observable trend support the claim?",
                    "sampling_frame": "A dated public-metadata sample.",
                    "data_sources": ["OpenAlex"],
                    "variables": ["year", "citation_count"],
                    "analysis": "Descriptive trend and permutation test.",
                    "falsification_condition": "No trend or opposite trend.",
                    "budget": "100 records and 10 minutes.",
                },
            },
            emitter="ProblemModelingAgent",
        )
        method_ref = store.put(
            kind="method", session_id=sid, key="v1",
            obj={"approach": "derive a falsifiable claim", "changes_outline": ["claim"]},
            emitter="MethodAgent",
        )
        engine = OrchestratorEngine(
            client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await engine.start()
        await save_session_state(
            fake_bus.db,
            SessionState(
                session_id=sid,
                user_id="u1",
                current_state="ConfirmingMethod",
                mode="experiment_validation",
                research_spec_ref=research_spec_ref,
                spec_interpretation_ref=interpretation_ref,
                method_ref=method_ref,
                problem_model_ref="spec://problem_model/validation/v1",
            ),
            auto_commit=True,
        )

        await engine.tick(_Evt(
            "method.confirmed", sid, "HumanView", payload={"inject": False}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "PlanningExperiment"
        assert not any(p.event_type in {
            "benchmark.selection_requested", "baseline.reproduction_requested"
        } for p in fake_bus.published)
        request = next(
            p for p in fake_bus.published
            if p.event_type == "experiment.planning_requested")
        assert request.payload["benchmark_spec_ref"] is None
        assert request.payload["baseline_result_ref"] is None
        assert request.payload["spec_interpretation_ref"] == interpretation_ref

        plan_ref = store.put(
            kind="experiment_plan", session_id=sid, key="v1",
            obj={"task_class": "survey_validation"}, emitter="ExperimentPlannerAgent")
        await engine.tick(_Evt(
            "experiment.plan_ready", sid, "ExperimentPlannerAgent",
            payload={"experiment_plan_ref": plan_ref}))
        await engine.tick(_Evt(
            "experiment.plan_reviewed", sid, "CriticAgent",
            payload={"verdict": "pass", "review_report_ref": store.put(
                kind="critique_report", session_id=sid, key="experiment_plan-v1",
                obj={"verdict": "pass"}, emitter="CriticAgent")}))
        patch_ref = store.put(
            kind="experiment_patch", session_id=sid, key="v1",
            obj={"touched_files": ["validation_study.py"]}, emitter="CoderAgent")
        await engine.tick(_Evt(
            "code.implemented", sid, "CoderAgent",
            payload={"experiment_patch_ref": patch_ref}))
        run = {
            "succeeded": True, "exit_code": 0, "metric_value": 0.03,
            "stdout_excerpt": "No stable association.",
            "environment": "survey_validation|host_rerun",
        }
        run_ref = store.put(
            kind="run_record", session_id=sid, key="v1", obj=run,
            emitter="RunnerAgent")
        await engine.tick(_Evt(
            "experiment.run_completed", sid, "RunnerAgent",
            payload={"verdict": "succeeded", "run_record_ref": run_ref}))
        evaluation_ref = store.put(
            kind="evaluation_report", session_id=sid, key="v1",
            obj={"verdict": "valid_not_best", "run_record_digest": artifact_digest(run)},
            emitter="EvaluatorAgent")
        await engine.tick(_Evt(
            "experiment.evaluated", sid, "EvaluatorAgent",
            payload={"verdict": "valid_not_best",
                     "evaluation_report_ref": evaluation_ref,
                     "run_record_digest": artifact_digest(run)}))
        verification_ref = store.put(
            kind="verification_report", session_id=sid, key="v1",
            obj={"verdict": "verified"}, emitter="EvaluatorAgent")
        await engine.tick(_Evt(
            "result.verified", sid, "EvaluatorAgent",
            payload={"verdict": "verified",
                     "verification_report_ref": verification_ref}))
        memory_ref = store.put(
            kind="research_memory", session_id=sid, key="v1",
            obj={"consolidated_insights": ["survey_validation:inconclusive"]},
            emitter="MemoryAgent")
        await engine.tick(_Evt(
            "memory.consolidated", sid, "MemoryAgent",
            payload={"routing_verdict": "report", "research_memory_ref": memory_ref}))
        citation_ref = store.put(
            kind="citation_audit_report", session_id=sid, key="v1",
            obj={"verdict": "pass"}, emitter="EvidenceAuditAgent")
        await engine.tick(_Evt(
            "evidence.citation_audited", sid, "EvidenceAuditAgent",
            payload={"citation_audit_report_ref": citation_ref}))
        report_ref = store.put(
            kind="report", session_id=sid, key="v1",
            obj={"title": "Validation report"}, emitter="ReportAgent")
        await engine.tick(_Evt(
            "report.ready", sid, "ReportAgent", payload={"report_ref": report_ref}))

        final = await load_session_state(fake_bus.db, sid)
        assert final.current_state == "Done"
        assert final.final_report_ref == report_ref
        run_request = next(
            p for p in fake_bus.published if p.event_type == "experiment.run_requested")
        eval_request = next(
            p for p in fake_bus.published if p.event_type == "experiment.evaluation_requested")
        verify_request = next(
            p for p in fake_bus.published if p.event_type == "result.verification_requested")
        assert run_request.payload["spec_interpretation_ref"] == interpretation_ref
        assert eval_request.payload["spec_interpretation_ref"] == interpretation_ref
        assert verify_request.payload["spec_interpretation_ref"] == interpretation_ref
    finally:
        store.close()


async def test_resume_recovers_legacy_failed_theory_without_new_benchmark(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        sid = "sess-legacy-theory"
        research_spec_ref = store.put(
            kind="research_spec", session_id=sid, key="reinterpreted",
            obj={"task_mode": "idea_theory", "research_goal": "derive propositions"},
            emitter="ProblemModelingAgent",
        )
        method_ref = store.put(
            kind="method", session_id=sid, key="v1",
            obj={"approach": "Axioms to propositions", "changes_outline": ["Theorem 1"]},
            emitter="MethodAgent",
        )
        engine = OrchestratorEngine(
            client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await engine.start()
        await save_session_state(
            fake_bus.db,
            SessionState(
                session_id=sid,
                user_id="u1",
                current_state="Failed",
                previous_state="ReproducingBaseline",
                mode="benchmark",
                research_spec_ref=research_spec_ref,
                method_ref=method_ref,
                problem_model_ref="spec://problem_model/theory/v1",
                spec_interpretation_ref="spec://spec_interpretation/theory/v1",
                awaiting_user_decision_reason="agent_failure",
                agent_failure_round=3,
            ),
            auto_commit=True,
        )

        result = await engine.resume_to_current_state(sid)

        state = await load_session_state(fake_bus.db, sid)
        assert result["resumed"] is True
        assert state.current_state == "ConsolidatingMemory"
        assert state.mode == "idea_theory"
        assert state.agent_failure_round == 0
        assert not any(p.event_type in {
            "benchmark.selection_requested", "baseline.reproduction_requested"
        } for p in fake_bus.published)
    finally:
        store.close()
