"""New-session handoff regressions; real SQLite and SpecStore, no model calls."""
from pathlib import Path
from types import SimpleNamespace
import aiosqlite
import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.spec_store.store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.mark.parametrize("kind,owner", [
    ("method_improvement_review", "s"), ("method", "s"),
    ("method_improvement_review", "other"),
])
async def test_applied_feedback_is_frozen_with_same_session_review_identity(tmp_path, kind, owner):
    store = SpecStore(tmp_path / "store")
    emitter = "MethodAgent" if kind == "method" else "MemoryAgent"
    ref = store.put(kind=kind, session_id=owner, key="v1",
                    obj={"decision": {"route": "revise_method"}}, emitter=emitter)
    async with aiosqlite.connect(":memory:") as db:
        engine = OrchestratorEngine(DeliveringBus(db).view("ResearchOrchestrator"),
                                    ROOT, spec_store=store)
        await engine.start()
        if kind != "method_improvement_review" or owner != "s":
            with pytest.raises(ValueError, match="invalid applied_method_feedback_ref identity"):
                engine._freeze_refs({"applied_method_feedback_ref": ref}, "s")
        else:
            frozen = engine._freeze_refs({"applied_method_feedback_ref": ref}, "s")
            assert frozen["applied_method_feedback_ref"] == store.snapshot(ref)
            store.put(kind=kind, session_id=owner, key="v1",
                      obj={"decision": {"route": "report"}}, emitter="MemoryAgent")
            assert store.get(frozen["applied_method_feedback_ref"])["decision"]["route"] == "revise_method"
    store.close()


@pytest.mark.parametrize("phase,event_type,payload", [
    ("PlanningExperiment", "experiment.plan_ready", {"experiment_plan_ref": "old"}),
    ("ReviewingExperimentPlan", "experiment.plan_reviewed", {"verdict": "pass"}),
    ("ReviewingExperimentPlan", "experiment.plan_reviewed", {"verdict": "fail"}),
    ("ImplementingExperiment", "code.implemented", {"experiment_patch_ref": "old"}),
    ("GeneratingMethod", "method.generated", {"method_ref": "old"}),
    ("RevisingMethod", "method.generated", {"method_ref": "old"}),
    ("DeliberatingMethod", "method.deliberated", {"method_ref": "old"}),
    ("RunningExperiment", "experiment.run_completed", {"run_record_ref": "old", "verdict": "succeeded"}),
    ("ReproducingBaseline", "baseline.reproduced", {"benchmark_id": "same", "verdict": "failed"}),
    ("ConsolidatingMemory", "memory.consolidated", {"research_memory_ref": "old", "diagnosis": "baseline_drift"}),
    ("Reporting", "report.ready", {"report_ref": "old"}),
    ("RunningExperiment", "agent.failed", {"agent_name": "ReportAgent", "reason_classification": "transient"}),
])
async def test_stale_reply_has_no_state_or_counter_effect(tmp_path, phase, event_type, payload):
    store = SpecStore(tmp_path / "store")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id="s", current_state=phase, request_binding_required=True,
                             active_request={"event_id": "current", "state": phase, "consumed": False})
        await save_session_state(db, state, auto_commit=True)
        before = (await load_session_state(db, "s")).to_json()
        await engine.tick(SimpleNamespace(session_id="s", event_type=event_type, payload=payload,
                                           emitter="Agent", event_id="late", causation_id="old"))
        assert (await load_session_state(db, "s")).to_json() == before
        assert not bus.published
    store.close()


async def test_request_input_and_identity_exist_before_immediate_delivery(tmp_path):
    store = SpecStore(tmp_path / "store")
    method = store.put(kind="method", session_id="s", key="v1", obj={"value": 1}, emitter="MethodAgent")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        seen = []
        async def receive(env):
            saved = await load_session_state(db, "s")
            assert saved.active_request["event_id"] == env.event_id
            assert saved.method_ref == env.payload["method_ref"]
            seen.append(env)
        bus.subscribe("method.deliberation_requested", receive)
        state = SessionState(session_id="s", current_state="DeliberatingMethod", method_ref=method,
                             request_binding_required=True)
        await engine._publish_request(state, "method.deliberation_requested", {"method_ref": method})
        await db.commit()
        store.put(kind="method", session_id="s", key="v1", obj={"value": 2}, emitter="MethodAgent")
        assert store.get(seen[0].payload["method_ref"]) == {"value": 1}
        await engine._publish_request(state, "method.deliberation_requested", {"method_ref": method}, replay=True)
        assert seen[-1].event_id == seen[0].event_id
        assert store.get(seen[-1].payload["method_ref"]) == {"value": 1}
    store.close()


async def test_running_experiment_restart_replay_marks_remote_recovery(tmp_path):
    store = SpecStore(tmp_path / "store")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(
            bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        payload = {
            "experiment_patch_ref": None,
            "verifier_spec_ref": None,
            "benchmark_spec_ref": None,
            "baseline_result_ref": None,
            "method_ref": None,
            "confirmed_repo_url": None,
            "confirmed_boundary": {},
            "spec_interpretation_ref": None,
        }
        state = SessionState(
            session_id="s", current_state="RunningExperiment",
            request_binding_required=True, experiment_status="running",
            active_request={
                "event_id": "original-run", "event_type": "experiment.run_requested",
                "state": "RunningExperiment", "payload": payload, "consumed": True,
            },
        )
        await save_session_state(db, state, auto_commit=True)

        result = await engine.resume_to_current_state("s")

        assert result["resumed"] is True
        resumed = await load_session_state(db, "s")
        assert resumed.active_request["event_id"] == "original-run"
        assert resumed.active_request["payload"]["recover_completed_result"] is True
        event = [item for item in bus.published
                 if item.event_type == "experiment.run_requested"][-1]
        assert event.event_id == "original-run"
        assert event.payload["recover_completed_result"] is True
    store.close()


async def test_unconfirmed_restart_failure_reattaches_running_experiment(tmp_path):
    store = SpecStore(tmp_path / "store")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(
            bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(
            session_id="s", current_state="AwaitingUserDecision",
            previous_state="MossIntervening", request_binding_required=True,
            experiment_status="running", awaiting_user_decision_reason="agent_failure",
            agent_failure_round=1,
            last_agent_failure={
                "underlying_reason_code": "execution_result_unconfirmed",
                "failed_state": "RunningExperiment",
            },
        )
        await save_session_state(db, state, auto_commit=True)

        result = await engine.resume_to_current_state("s")

        resumed = await load_session_state(db, "s")
        assert result["reason"] == (
            "reattached interrupted experiment to durable remote recovery")
        assert resumed.current_state == "RunningExperiment"
        assert resumed.awaiting_user_decision_reason is None
        assert resumed.agent_failure_round == 0
        event = [item for item in bus.published
                 if item.event_type == "experiment.run_requested"][-1]
        assert event.payload["recover_completed_result"] is True
    store.close()


async def test_bound_moss_discard_preserves_frozen_failure_artifact_and_advances_once(tmp_path):
    store = SpecStore(tmp_path / "store")
    method = store.put(kind="method", session_id="s", key="v1", obj={"approach": "current"},
                       emitter="MethodAgent")
    decision = store.put(kind="moss_intervention", session_id="s", key="discard",
                         obj={"method_ref": method, "action": "discard_hypothesis"}, emitter="MOSS")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id="s", current_state="MossIntervening", method_ref=method,
                             current_moss_reason="implementation_thrashing", request_binding_required=True,
                             active_request={"event_id": "current", "state": "MossIntervening",
                                             "event_type": "moss.intervention_requested", "consumed": False})
        await save_session_state(db, state, auto_commit=True)
        event = SimpleNamespace(session_id="s", event_type="moss.intervention_completed",
                                emitter="MOSS", event_id="completion", causation_id="current",
                                payload={"reason": "implementation_thrashing", "action": "discard_hypothesis",
                                         "moss_intervention_ref": decision, "degraded": False})
        await engine.tick(event)
        current = await load_session_state(db, "s")
        assert current.current_state == "ConsolidatingMemory"
        requests = [item for item in bus.published if item.event_type == "memory.consolidation_requested"]
        assert len(requests) == 1
        artifact = requests[0].payload["failure_context"]["artifact_ref"]
        assert "/snapshot-" in artifact
        store.put(kind="moss_intervention", session_id="s", key="discard",
                  obj={"action": "different"}, emitter="MOSS")
        assert store.get(artifact)["action"] == "discard_hypothesis"
        await engine.tick(event)
        assert (await load_session_state(db, "s")).to_json() == current.to_json()
        assert len([item for item in bus.published
                    if item.event_type == "memory.consolidation_requested"]) == 1
    store.close()


@pytest.mark.parametrize("kind,sid", [("method", "s"), ("moss_intervention", "other")])
async def test_failure_artifact_still_requires_moss_kind_and_same_session(tmp_path, kind, sid):
    store = SpecStore(tmp_path / "store")
    ref = store.put(kind=kind, session_id=sid, key="v1", obj={},
                    emitter="MethodAgent" if kind == "method" else "MOSS")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        with pytest.raises(ValueError, match="artifact_ref identity"):
            engine._freeze_refs({"failure_context": {"artifact_ref": ref}}, "s")
    store.close()


async def test_bound_baseline_failure_report_completes_without_a_successful_run(tmp_path):
    """An honest failure report cannot have the successful run required by the success path."""
    store = SpecStore(tmp_path / "store")
    sid = "s"
    observation = {
        "ref": f"spec://baseline_result/{sid}/snapshot-failed",
        "metric_value": None,
        "reproduced": False,
        "exit_code": 1,
    }
    memory = {
        "diagnosis": "baseline_reproduction_failed",
        "diagnosis_evidence": ["the repository baseline could not be reproduced"],
        "verified_findings": [],
        "negative_results": ["baseline reproduction failed"],
        "baseline_observations": [observation],
    }
    memory_ref = store.put(kind="research_memory", session_id=sid, key="v1",
                           obj=memory, emitter="MemoryAgent")
    request_payload = {
        "research_memory_ref": memory_ref,
        "research_spec_ref": f"spec://research_spec/{sid}/final",
    }
    report = {
        "request_digest": artifact_digest(request_payload),
        "summary": "No baseline metric or qualified candidate was produced.",
        "verified_findings": [],
        "negative_results": memory["negative_results"],
        "analysis_caveats": ["The failed baseline attempt is diagnostic only."],
    }
    report_ref = store.put(kind="report", session_id=sid, key="v1",
                           obj=report, emitter="ReportAgent")
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(
            session_id=sid,
            current_state="Reporting",
            mode="benchmark",
            research_memory_ref=memory_ref,
            request_binding_required=True,
            active_request={"payload": request_payload},
        )
        await save_session_state(db, state, auto_commit=True)

        await engine._on_Reporting_report_ready(
            state,
            SimpleNamespace(event_id="ready", payload={"report_ref": report_ref}),
        )

        completed = await load_session_state(db, sid)
        assert completed.current_state == "Done"
        assert completed.final_report_ref == report_ref
        assert completed.completion_verified is True
        assert completed.experiment_status == "failed"
    store.close()


@pytest.mark.parametrize("baseline_matches,expected_state", [
    (True, "Done"),
    (False, "AwaitingUserDecision"),
])
@pytest.mark.parametrize("has_attempt_history", [False, True])
async def test_bound_no_candidate_report_completes_without_a_successful_run(
    tmp_path, baseline_matches, expected_state, has_attempt_history,
):
    """Attempt history must not strand an honest no-qualified-candidate report."""
    store = SpecStore(tmp_path / "store")
    sid = "s-no-candidate"
    baseline = {"reproduced": True, "metric_value": 128.0, "exit_code": 0}
    baseline_ref = store.put(kind="baseline_result", session_id=sid, key="frozen",
                             obj=baseline, emitter="RunnerAgent")
    method_ref = store.put(kind="method", session_id=sid, key="frozen",
                           obj={"name": "confirmed"}, emitter="MethodAgent")
    memory = {
        "diagnosis": "inconclusive",
        "diagnosis_evidence": ["planning exhausted before a qualified candidate existed"],
        "verified_findings": [],
        "negative_results": ["no candidate qualified for complete evaluation"],
        "best_result_ref": None,
        "baseline_observations": [{
            "ref": (baseline_ref if baseline_matches
                    else f"spec://baseline_result/{sid}/snapshot-other"),
            "metric_value": 128.0,
            "reproduced": True,
            "exit_code": 0,
        }],
    }
    memory_ref = store.put(kind="research_memory", session_id=sid, key="v1",
                           obj=memory, emitter="MemoryAgent")
    request_payload = {
        "research_memory_ref": memory_ref,
        "research_spec_ref": f"spec://research_spec/{sid}/final",
    }
    report_ref = store.put(kind="report", session_id=sid, key="v1", obj={
        "request_digest": artifact_digest(request_payload),
        "summary": "The baseline reproduced, but no candidate qualified.",
        "verified_findings": [],
        "negative_results": memory["negative_results"],
        "analysis_caveats": ["No candidate run is claimed."],
    }, emitter="ReportAgent")
    async with aiosqlite.connect(":memory:") as db:
        engine = OrchestratorEngine(
            DeliveringBus(db).view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(
            session_id=sid,
            current_state="Reporting",
            mode="benchmark",
            method_ref=method_ref,
            baseline_result_ref=baseline_ref,
            research_memory_ref=memory_ref,
            request_binding_required=True,
            active_request={"payload": request_payload},
            run_record_refs=(
                [f"spec://run_record/{sid}/snapshot-rejected"]
                if has_attempt_history else []),
        )
        await save_session_state(db, state, auto_commit=True)

        await engine._on_Reporting_report_ready(
            state,
            SimpleNamespace(event_id="ready", payload={"report_ref": report_ref}),
        )

        completed = await load_session_state(db, sid)
        assert completed.current_state == expected_state
        if baseline_matches:
            assert completed.final_report_ref == report_ref
            assert completed.completion_verified is True
            assert completed.experiment_status == "failed"
        else:
            assert completed.final_report_ref is None
            assert completed.completion_verified is False
    store.close()


async def test_report_request_does_not_bind_partial_integration_as_complete_result(tmp_path):
    """A measured but partial integration remains evidence, not the report's formal result."""
    store = SpecStore(tmp_path / "store")
    sid = "s-partial-integration"
    memory_ref = store.put(kind="research_memory", session_id=sid, key="v1", obj={
        "diagnosis": "valid_improvement",
        "diagnosis_evidence": ["measured improvement with incomplete implementation review"],
        "verified_findings": ["the task evaluator measured an improvement"],
        "negative_results": ["the implementation review remained partial"],
        "best_result_ref": None,
    }, emitter="MemoryAgent")
    spec_ref = store.put(kind="research_spec", session_id=sid, key="final",
                         obj={"question": "test"}, emitter="ProblemModelingAgent")
    run = {"succeeded": True, "metric_value": 88.0}
    run_current_ref = store.put(kind="run_record", session_id=sid, key="v1",
                                obj=run, emitter="RunnerAgent")
    run_ref = store.snapshot(run_current_ref)
    candidate_current_ref = store.put(kind="gen_experiment_result", session_id=sid,
        key="current", obj={
            "run_record_snapshot_ref": run_ref,
            "method_metric": 88.0,
            "verdict": "new_best",
            "measurement_evidence": {"valid": True},
            "component_execution": {"mode": "integration", "all_components_accepted": True},
            "candidate_admissibility": {"verdict": "pass"},
            "method_implementation_review": {
                "verdict": "partial", "task_contract_review": {"verdict": "compliant"}},
        }, emitter="RunnerAgent")
    candidate_ref = store.snapshot(candidate_current_ref)
    async with aiosqlite.connect(":memory:") as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view("ResearchOrchestrator"), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id=sid, current_state="Reporting",
                             request_binding_required=True, run_record_refs=[run_ref])
        await engine._publish_request(state, "report.requested", {
            "research_memory_ref": memory_ref,
            "research_spec_ref": spec_ref,
        })
        request = [item for item in bus.published if item.event_type == "report.requested"][-1]
        assert request.payload["result_ref"] is None
        assert candidate_ref != request.payload["result_ref"]
    store.close()
