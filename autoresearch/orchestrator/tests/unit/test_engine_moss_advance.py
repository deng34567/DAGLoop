"""Unit tests for MOSS Step 1 Task 5: orchestrator MossIntervening routing.

Covers:
  - the MossIntervening entry action (moss.intervention_requested), reading
    ``reason`` off ``state.current_moss_reason``.
  - ``_on_MossIntervening_moss_intervention_completed``:
      - routes via the MOSS action matrix (``self.moss_guard.next_state``);
      - applies ``counter_updates`` (scalar "reset to 0" and map "delete key
        <field>"), resolved LETTER -> SessionState attr via
        ``self.skill.counters``;
      - appends a MossInterventionRecord to ``state.moss_interventions``;
      - clears ``state.current_moss_reason``;
      - an illegal action in the event is defensively routed via
        ``self.moss_guard.fallback`` instead of crashing.

Also regression-checks (f): M12 (evaluation advance) and M13 (consolidation
advance) still green (run alongside this file).

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_consolidation_advance.py's patterns (does NOT auto-deliver;
drives the engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autoresearch.eventbus.broker.registry import Registry
from autoresearch.orchestrator.engine import (
    OrchestratorEngine, _METHOD_REVISION_CAP, _METHOD_SWAP_CAP)
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
EVENTBUS_ROOT = Path(__file__).resolve().parents[3] / "eventbus"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


def _artifact_digest(obj: dict) -> str:
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_moss_intervening_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    current_moss_reason: str | None,
    direction_critique_round: int = 0,
    current_result_id: str | None = None,
    result_verification_fail_round: dict[str, int] | None = None,
) -> None:
    """Persist a session in MossIntervening with the given upstream context.

    ``evaluation_report_refs`` / ``run_record_refs`` are pre-populated because
    a couple of actions route into ConsolidatingMemory, whose EXISTING entry
    action reads ``[-1]`` off both.
    """
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason=current_moss_reason,
        direction_critique_round=direction_critique_round,
        current_result_id=current_result_id,
        result_verification_fail_round=result_verification_fail_round or {},
        evaluation_report_refs=[f"evaluation_report://{sid}/final"],
        run_record_refs=[f"run_record://{sid}/final"],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): entry action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_ref", [None, "method://sess-a/current"])
async def test_moss_intervening_entry_action_emits_intervention_requested(fake_bus, method_ref):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    await _seed_moss_intervening_session(eng, sid, current_moss_reason="direction_thrashing")
    state = await load_session_state(fake_bus.db, sid)
    state.method_ref = method_ref
    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "moss.intervention_requested"]
    assert len(emitted) == 1
    # 3.7: the payload also carries the consolidated ResearchMemory ref (None until one is frozen)
    # so MOSS's advice can be informed by the run's accumulated findings, plus the current
    # run_record_ref (the frozen run whose evidence MOSS is reacting to).
    assert emitted[0].payload == {"reason": "direction_thrashing", "method_ref": method_ref,
                                  "baseline_result_ref": None,
                                  "spec_interpretation_ref": None,
                                  "counter_snapshot": None,
                                  "research_memory_ref": None,
                                      "run_record_ref": "run_record://sess-a/final",
                                      "baseline_attempt_ref": None,
                                  "agent_failure": None,
                                  "critique_report_ref": None}


# ---------------------------------------------------------------------------
# (b): direction_thrashing / rewrite_framing -> ModelingProblem; A reset to 0
# ---------------------------------------------------------------------------


async def test_same_reason_moss_completion_cannot_discard_a_revised_method(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "revised-method"
        state = SessionState(
            session_id=sid, user_id="u", current_state="MossIntervening",
            current_moss_reason="implementation_thrashing", method_ref="spec://method/revised-method/new")
        await save_session_state(fake_bus.db, state, auto_commit=True)
        decision = {"reason": "implementation_thrashing", "action": "discard_hypothesis",
                    "method_ref": "spec://method/revised-method/old", "degraded": False}
        ref = store.put(kind="moss_intervention", session_id=sid, key="old-decision",
                        obj=decision, emitter="MOSS")
        before = len(fake_bus.published)
        await eng.tick(_Evt("moss.intervention_completed", sid, "MOSS", {
            "reason": "implementation_thrashing", "action": "discard_hypothesis",
            "moss_intervention_ref": ref, "degraded": False}))
        saved = await load_session_state(fake_bus.db, sid)
        assert saved.current_state == "MossIntervening"
        assert saved.method_ref == state.method_ref
        assert saved.moss_interventions == []
        assert len(fake_bus.published) == before
    finally:
        store.close()


async def test_intervention_completed_rewrite_framing_resets_counter_a(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="direction_thrashing", direction_critique_round=2
    )

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MossAgent",
            payload={
                "reason": "direction_thrashing",
                "action": "rewrite_framing",
                "moss_intervention_ref": "x",
                "degraded": False,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ModelingProblem"
    assert st.current_moss_reason is None
    assert st.direction_critique_round == 0
    assert len(st.moss_interventions) == 1
    assert st.moss_interventions[0] == {
        "reason": "direction_thrashing",
        "action": "rewrite_framing",
        "moss_intervention_ref": "x",
        "degraded": False,
    }


async def test_moss_simplify_resets_method_scoped_plan_review_budget(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-moss-plan-retry"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="experiment_design_thrashing")
    st = await load_session_state(fake_bus.db, sid)
    st.method_ref = "method://current"
    st.experiment_plan_review_round = {st.method_ref: 3, "method://prior": 2}
    await save_session_state(fake_bus.db, st, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "experiment_design_thrashing", "action": "simplify_experiment",
        "moss_intervention_ref": "moss://simplify", "degraded": False}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"
    assert st.experiment_plan_review_round == {"method://prior": 2}
    assert st.method_ref == "method://current"
    assert st.method_not_working is False


@pytest.mark.parametrize("baseline,rounds,has_memory,expected", [
    ("baseline://frozen", 0, True, "PlanningExperiment"),
    ("baseline://frozen", 9, True, "ConsolidatingMemory"),
    ("baseline://frozen", 10, True, "ConsolidatingMemory"),
    ("baseline://frozen", 10, False, "ConsolidatingMemory"),
    (None, 10, True, "PlanningExperiment"),
])
@pytest.mark.parametrize("action", ["simplify_experiment", "revise_component"])
async def test_plan_repairs_consume_existing_revision_budget_and_consolidate_before_closeout(
    fake_bus, baseline, rounds, has_memory, expected, action,
):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-bounded-plan-simplification"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="experiment_design_thrashing")
    state = await load_session_state(fake_bus.db, sid)
    state.method_ref = "method://frozen"
    state.baseline_result_ref = baseline
    state.research_memory_ref = "memory://existing" if has_memory else None
    state.component_revision_round = rounds
    state.experiment_plan_review_round = {state.method_ref: 3, "method://other": 1}
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "experiment_design_thrashing", "action": action,
        "moss_intervention_ref": "moss://simplify", "degraded": False}))
    state = await load_session_state(fake_bus.db, sid)
    if baseline and rounds < 9 and action == "revise_component":
        expected = "DeliberatingMethod"
    assert state.current_state == expected
    assert state.method_ref == "method://frozen"
    assert state.baseline_result_ref == baseline
    assert state.component_revision_round == rounds + int(bool(baseline))
    assert state.experiment_plan_review_round == {"method://other": 1}
    assert state.method_feedback_round == state.method_revision_round == 0
    if expected == "ConsolidatingMemory":
        request = next(p for p in fake_bus.published if p.event_type == "memory.consolidation_requested")
        assert request.payload["qualified_component_closeout_requested"] is True
        assert request.payload["research_memory_ref"] == state.research_memory_ref


@pytest.mark.parametrize("origin,problem", [
    *[("moss", problem) for problem in (None, "not_degraded", "component_only", "unaccepted",
        "wrong_baseline", "unverified_component", "no_candidate", "paused", "paused_wrong_method")],
    *[("planner", problem) for problem in (None, "component_only", "unaccepted",
        "wrong_baseline", "unverified_component", "no_candidate")],
])
async def test_plan_escalation_closes_with_candidates_or_negative_result_without_waiting(
    fake_bus, tmp_path, monkeypatch, origin, problem,
):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-plan-escalation-closeout"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="experiment_design_thrashing")
    baseline = {"reproduced": True, "metric_value": 3.0}
    baseline_ref = store.put(kind="baseline_result", session_id=sid, key="frozen",
                             obj=baseline, emitter="RunnerAgent")
    method_ref = store.put(kind="method", session_id=sid, key="current",
                           obj={"name": "current"}, emitter="MethodAgent")
    run = {"succeeded": True, "exit_code": 0, "metric_value": 2.0,
           "experiment_id": method_ref}
    run_ref = store.put(kind="run_record", session_id=sid,
        key="snapshot-" + _artifact_digest(run).split(":")[1], obj=run, emitter="RunnerAgent")
    result = {"accepted": problem not in {"unaccepted", "no_candidate"},
        "method_metric": None if problem == "no_candidate" else 2.0,
        "verdict": "invalid" if problem == "no_candidate" else "new_best",
        "checkpoint": None if problem == "no_candidate" else "/saved/candidate",
        "method_ref": method_ref,
        "run_record_snapshot_ref": run_ref,
        "baseline_digest": "another-baseline" if problem == "wrong_baseline" else _artifact_digest(baseline),
        "method_implementation_review": {"verdict": "faithful"}}
    if problem in {"component_only", "unverified_component"}:
        result["component_execution"] = {"mode": "component", "role": "direct_gain"}
    result_ref = store.put(kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + _artifact_digest(result).split(":")[1], obj=result, emitter="RunnerAgent")
    history = []
    if problem in {"component_only", "unverified_component"}:
        history = [{"component_id": "feature", "mode": "component", "status": "accepted",
            "role": "direct_gain", "metric": 2.0,
            "became_best": True, "execution_succeeded": True, "contract_passed": True,
            "verification_verdict": "failed" if problem == "unverified_component" else "verified",
            "mir_verdict": "faithful", "baseline_digest": _artifact_digest(baseline),
            "checkpoint": "/saved/candidate", "run_ref": run_ref}]
    memory = {"best_metric": 2.0, "best_result_ref": result_ref, "component_history": history}
    if history or problem == "no_candidate":
        memory.update(best_metric=None, best_result_ref=None)
    memory_ref = store.put(kind="research_memory", session_id=sid, key="current",
                           obj=memory, emitter="MemoryAgent")
    state = await load_session_state(fake_bus.db, sid)
    state.method_ref = method_ref
    state.baseline_result_ref = baseline_ref
    state.research_memory_ref = memory_ref
    state.run_record_refs = [run_ref]
    state.method_feedback_round = 3
    moss_ref = store.put(kind="moss_intervention", session_id=sid, key="plan-failure", obj={
        "reason": "experiment_design_thrashing", "action": "escalate_to_user",
        "method_ref": "method://obsolete" if problem == "paused_wrong_method" else method_ref,
        "degraded": problem != "not_degraded"}, emitter="MOSS")
    if problem in {"paused", "paused_wrong_method"}:
        state.current_state = "AwaitingUserDecision"
        state.awaiting_user_decision_reason = "experiment_design_thrashing"
        state.moss_interventions = [{"reason": "experiment_design_thrashing",
            "action": "escalate_to_user", "moss_intervention_ref": moss_ref, "degraded": True}]
    if origin == "planner":
        state.current_state = "PlanningExperiment"
        state.confirmed_repo_url = "https://example.org/research/repo"
        threshold, _ = eng._counter_threshold("agent_retry_count")
        state.agent_retry_count = {"PlanningExperiment": threshold - 1}
    await save_session_state(fake_bus.db, state, auto_commit=True)
    if origin == "planner":
        await eng.tick(_Evt("agent.failed", sid, "ExperimentPlannerAgent", payload={
            "agent_name": "ExperimentPlannerAgent", "reason_classification": "permanent",
            "underlying_reason_code": "planner_parse_error", "underlying_reason": "invalid output"}))
    elif problem in {"paused", "paused_wrong_method"}:
        await eng.resume_to_current_state(sid)
    else:
        await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
            "reason": "experiment_design_thrashing", "action": "escalate_to_user",
            "moss_intervention_ref": moss_ref, "degraded": problem != "not_degraded"}))
    state = await load_session_state(fake_bus.db, sid)
    # Once a benchmark has a frozen baseline and Method, exhausted planning is a
    # reportable outcome even when no candidate qualifies.  Retain a verified
    # candidate when one exists; otherwise close with the recorded negative result.
    # A paused intervention bound to another Method remains stale and must not move.
    closes = problem != "paused_wrong_method"
    assert state.current_state == ("ConsolidatingMemory" if closes else "AwaitingUserDecision")
    assert state.method_ref == method_ref and state.method_feedback_round == 3
    assert store.get(memory_ref) == memory and store.get(result_ref) == result
    if closes:
        request = next(p for p in fake_bus.published if p.event_type == "memory.consolidation_requested")
        assert request.payload["qualified_component_closeout_requested"] is True
        assert request.payload["failure_context"]["action"] == (
            "report_now" if origin == "planner" else "escalate_to_user")
        if origin == "moss":
            assert request.payload["failure_context"]["artifact_ref"] == moss_ref
        else:
            assert state.last_agent_failure["underlying_reason_code"] == "planner_parse_error"
        from autoresearch.eventbus.broker.registry import Registry
        registry = Registry(Path(__file__).resolve().parents[3] / "eventbus/eventbus-registry.yaml")
        assert not list(registry.get_payload_validator(
            request.event_type, 1).iter_errors(request.payload))
        assert state.awaiting_user_decision_reason is None
        if problem in {None, "no_candidate"}:
            from types import SimpleNamespace
            from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
            monkeypatch.setenv("AUTORESEARCH_METHOD_FEEDBACK", "0")
            monkeypatch.setenv("AUTORESEARCH_EXECUTION_STRATEGY", "efficient")
            monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "0")
            store.put(kind="gen_experiment_result", session_id=sid, key="current",
                      obj=result, emitter="RunnerAgent")
            memory_handler = MemoryAgentHandler(spec_store=store, llm_client=None)
            await memory_handler.handle_consolidation_requested(fake_bus, SimpleNamespace(
                session_id=sid, event_id="closing-memory", payload=request.payload))
            consolidated = next(p for p in reversed(fake_bus.published)
                                if p.event_type == "memory.consolidated")
            await eng.tick(_Evt("memory.consolidated", sid, "MemoryAgent",
                                payload=consolidated.payload))
            state = await load_session_state(fake_bus.db, sid)
            assert state.current_state == "Reporting"
            closed_memory = store.get(state.research_memory_ref)
            if problem is None:
                assert closed_memory["best_result_ref"] == result_ref
            else:
                assert consolidated.payload["diagnosis"] == "inconclusive"
                assert consolidated.payload["routing_verdict"] == "report"
                assert closed_memory["best_result_ref"] is None
                assert any("experiment_design_thrashing" in item
                           for item in closed_memory["negative_results"])
    store.close()


@pytest.mark.parametrize("baseline,rounds,expected", [
    ("baseline://frozen", 6, "DeliberatingMethod"),
    ("baseline://frozen", 10, "ConsolidatingMemory"),
    (None, 0, "PlanningExperiment"),
])
async def test_plan_review_can_reaudit_component_without_replacing_method(
    fake_bus, baseline, rounds, expected,
):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-review-component-revision"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="experiment_design_thrashing")
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = "method://frozen"
    state.baseline_result_ref = baseline
    state.component_revision_round = rounds
    state.experiment_plan_review_round = {state.method_ref: 3, "method://other": 1}
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "experiment_design_thrashing", "action": "revise_component",
        "moss_intervention_ref": "moss://component-contract", "degraded": False}))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == expected
    assert state.method_ref == "method://frozen"
    assert state.baseline_result_ref == baseline
    assert state.method_swap_round == 0
    assert state.method_not_working is False
    assert state.component_revision_round == rounds + int(bool(baseline))
    assert state.component_plan_audit_pending == (expected == "DeliberatingMethod")
    assert state.experiment_plan_review_round == {"method://other": 1}


async def test_repeated_implementation_thrashing_escalates_plan_then_method(fake_bus):
    """Repeated shrink-patch interventions must not reset F forever."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-implementation-escalation"
    method_ref = f"method://{sid}/v1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="implementation_thrashing",
        method_ref=method_ref,
        current_experiment_id=method_ref,
        implementation_fix_round={method_ref: 3},
        moss_interventions=[{
            "reason": "implementation_thrashing",
            "action": "shrink_patch",
            "moss_intervention_ref": "moss://first",
            "degraded": False,
            "method_ref": method_ref,
        }],
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing",
            "action": "shrink_patch",
            "moss_intervention_ref": "moss://second",
            "degraded": False,
        },
    ))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"
    assert state.moss_interventions[-1]["action"] == "change_plan"

    state.current_state = "MossIntervening"
    state.current_moss_reason = "implementation_thrashing"
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing",
            "action": "shrink_patch",
            "moss_intervention_ref": "moss://third",
            "degraded": False,
        },
    ))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "GeneratingMethod"
    assert state.method_swap_round == 1
    assert state.method_not_working is True
    assert state.moss_interventions[-1]["escalated_to"] == "GeneratingMethod"


async def test_repeated_implementation_thrashing_reports_after_method_swap_cap(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-implementation-swap-exhausted"
    method_ref = f"method://{sid}/v4"
    prior = [{
        "reason": "implementation_thrashing",
        "action": "shrink_patch",
        "moss_intervention_ref": f"moss://{index}",
        "degraded": False,
        "method_ref": method_ref,
    } for index in range(2)]
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="implementation_thrashing",
        method_ref=method_ref,
        current_experiment_id=method_ref,
        method_swap_round=_METHOD_SWAP_CAP,
        moss_interventions=prior,
        research_memory_ref=f"research_memory://{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing",
            "action": "shrink_patch",
            "moss_intervention_ref": "moss://final",
            "degraded": False,
        },
    ))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Reporting"
    assert state.method_swap_round == _METHOD_SWAP_CAP
    assert state.moss_interventions[-1]["escalated_to"] == "Reporting"


async def test_repeated_component_implementation_thrashing_reaudits_plan(fake_bus):
    """Post-baseline implementation failures must not replace the scientific Method."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-implementation-escalation"
    method_ref = f"method://{sid}/v1"
    prior = [{
        "reason": "implementation_thrashing",
        "action": "shrink_patch",
        "moss_intervention_ref": f"moss://{index}",
        "degraded": False,
        "method_ref": method_ref,
    } for index in range(2)]
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="implementation_thrashing",
        method_ref=method_ref,
        current_experiment_id=method_ref,
        baseline_result_ref=f"baseline_result://{sid}/v1",
        moss_interventions=prior,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing",
            "action": "shrink_patch",
            "moss_intervention_ref": "moss://third",
            "degraded": False,
        },
    ))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "DeliberatingMethod"
    assert state.method_ref == method_ref
    assert state.method_swap_round == 0
    assert state.method_not_working is False
    assert state.component_revision_round == 1
    assert state.component_ablation_failed is True
    assert state.component_plan_audit_pending is True
    assert state.moss_interventions[-1]["escalated_to"] == "DeliberatingMethod"


@pytest.mark.parametrize("prior_count,action,attempts,host_execution,expected", [
    (1, "shrink_patch", 0, {}, "ImplementingExperiment"),
    (2, "shrink_patch", 0, {}, "ImplementingExperiment"),
    (5, "shrink_patch", 0, {}, "ImplementingExperiment"),
    (2, "shrink_patch", 1, {}, "ImplementingExperiment"),
    (5, "shrink_patch", 4, {}, "ImplementingExperiment"),
    (2, "shrink_patch", 1, {"component_execution_started": True}, "DeliberatingMethod"),
    (2, "shrink_patch", 1, {"component_failed_attempt_id": "attempt-failed"}, "DeliberatingMethod"),
    (2, "shrink_patch", 1, {"component_attempt_id": "attempt-real"}, "DeliberatingMethod"),
    (2, "change_plan", 0, {}, "DeliberatingMethod"),
])
async def test_unexecuted_component_recovery_does_not_invalidate_plan(
        fake_bus, monkeypatch, prior_count, action, attempts, host_execution, expected):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-unexecuted-component-recovery"
    method_ref = f"method://{sid}/v1"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="implementation_thrashing", method_ref=method_ref,
        current_experiment_id=method_ref,
        baseline_result_ref=f"baseline_result://{sid}/v1",
        experiment_plan_refs=[f"experiment_plan://{sid}/v1"],
        moss_interventions=[{
            "reason": "implementation_thrashing", "action": "shrink_patch",
            "method_ref": method_ref,
        } for _ in range(prior_count)],
    )
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: {
        "error": "no result",
        **host_execution,
        "component_execution": {
            "mode": "component", "component_id": "current-component",
            "status": "execution_failed", "execution_succeeded": False,
            "attempts": attempts, "infrastructure_attempts": 3,
            "qualified_attempts": int(attempts > 0),
            "checkpoint": "checkpoint://prior-qualified" if attempts else "",
        },
    })
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "implementation_thrashing", "action": action,
        "moss_intervention_ref": "moss://recovery", "degraded": False,
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == expected
    assert state.method_ref == method_ref
    assert state.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert state.method_swap_round == 0
    assert state.component_revision_round == int(expected == "DeliberatingMethod")
    assert state.component_plan_audit_pending == (expected == "DeliberatingMethod")


@pytest.mark.parametrize("mode,status,succeeded,passed,review,expected", [
    ("component", "contract_failed", True, False, "faithful", "DeliberatingMethod"),
    ("component", "execution_failed", False, False, None, "DeliberatingMethod"),
    ("component", "drifted", True, True, "drifted", "DeliberatingMethod"),
    ("integration", "execution_failed", False, False, None, "PlanningExperiment"),
    ("integration", "contract_failed", True, False, "insufficient_evidence", "DeliberatingMethod"),
    (None, "execution_failed", False, False, None, "PlanningExperiment"),
])
async def test_explicit_change_plan_reaudits_component_without_requiring_success(
        fake_bus, monkeypatch, mode, status, succeeded, passed, review, expected):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-contract-plan-change"
    method_ref = f"method://{sid}/v1"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="implementation_thrashing", method_ref=method_ref,
        current_experiment_id=method_ref,
        baseline_result_ref=f"baseline_result://{sid}/v1",
    )
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: {
        "component_execution": {
            "mode": mode, "component_id": "current-component",
            "execution_succeeded": succeeded, "status": status,
            "attempt_contract_result": {"passed": passed},
        },
        "method_implementation_review": {
            "verdict": review,
            **({"review_error": "gain-bearing component cannot be scoped out of candidate implementation"}
               if review == "insufficient_evidence" else {}),
        },
    })
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "implementation_thrashing", "action": "change_plan",
        "moss_intervention_ref": "moss://first", "degraded": False,
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == expected
    assert state.method_ref == method_ref
    assert state.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert state.method_swap_round == 0
    assert state.component_plan_audit_pending == (expected == "DeliberatingMethod")
    assert state.component_revision_round == int(expected == "DeliberatingMethod")
    if expected == "DeliberatingMethod":
        assert state.ablation_failed_component == "current-component"


@pytest.mark.parametrize("bound,current_method,action,degraded", [
    (bound, current_method, action, degraded)
    for bound, current_method in ((True, True), (False, True), (True, False))
    for action, degraded in (("change_plan", False), ("escalate_to_user", True))
])
async def test_host_bound_no_change_component_reflects_on_method_before_reporting(
        fake_bus, monkeypatch, bound, current_method, action, degraded):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-no-change-component"
    method_ref = f"method://{sid}/current"
    run_ref = f"run_record://{sid}/current"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="implementation_thrashing", mode="benchmark",
        method_ref=method_ref, baseline_result_ref=f"baseline_result://{sid}/v1",
        component_revision_round=10, run_record_refs=[run_ref],
        best_result_id=f"result://{sid}/accepted-incumbent",
    )
    attempt_id = "attempt-verified-local-check"
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: {
        "method_ref": method_ref if current_method else f"method://{sid}/obsolete",
        "run_record_snapshot_ref": run_ref,
        "component_attempt_id": attempt_id,
        "component_execution_started": True,
        "diff": "", "parent_relative_diff": "",
        "ablation_report": [{
            "verdict": "met", "_host_bound": bound,
            "evidence": {"ref": attempt_id},
        }],
        "component_execution": {
            "mode": "component", "component_id": "already-satisfied-property",
            "role": "indirect_gain", "status": "implementation_failed",
            "execution_succeeded": True, "attempts": 3,
        },
        "method_implementation_review": {
            "method_ref": method_ref, "verdict": "insufficient_evidence",
            "issues": ["actual repository diff is empty"],
        },
    })
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "implementation_thrashing", "action": action,
        "moss_intervention_ref": "moss://no-change", "degraded": degraded,
    }))
    state = await load_session_state(fake_bus.db, sid)
    if bound and current_method:
        assert state.current_state == "RevisingMethod"
        assert state.component_revision_round == 10
        assert state.component_plan_audit_pending is False
        assert state.last_agent_failure["underlying_reason_code"] == "component_no_change_design_block"
        feedback = eng._method_revision_feedback(state)
        assert run_ref in feedback and attempt_id in feedback
    else:
        assert state.current_state == "ConsolidatingMemory"
        assert state.last_agent_failure is None


# ---------------------------------------------------------------------------
# (c): result_suspicious / discard_result -> ConsolidatingMemory; G delete key
# ---------------------------------------------------------------------------


async def test_intervention_completed_discard_result_deletes_map_counter_key(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_moss_intervening_session(
        eng,
        sid,
        current_moss_reason="result_suspicious",
        current_result_id="r1",
        result_verification_fail_round={"r1": 2},
    )

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MossAgent",
            payload={
                "reason": "result_suspicious",
                "action": "discard_result",
                "moss_intervention_ref": "y",
                "degraded": False,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConsolidatingMemory"
    assert "r1" not in st.result_verification_fail_round
    assert st.current_moss_reason is None


# ---------------------------------------------------------------------------
# (d): escalate_to_user -> AwaitingUserDecision (pauses)
# ---------------------------------------------------------------------------


async def test_intervention_completed_escalate_to_user_pauses_at_awaiting_user_decision(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_moss_intervening_session(eng, sid, current_moss_reason="direction_thrashing")

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MossAgent",
            payload={
                "reason": "direction_thrashing",
                "action": "escalate_to_user",
                "moss_intervention_ref": "z",
                "degraded": False,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "AwaitingUserDecision"
    assert st.current_moss_reason is None


# ---------------------------------------------------------------------------
# (e): illegal action -> defensive fallback (escalate_to_user), no crash
# ---------------------------------------------------------------------------


async def test_intervention_completed_illegal_action_falls_back(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_moss_intervening_session(eng, sid, current_moss_reason="direction_thrashing")

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MossAgent",
            payload={
                "reason": "direction_thrashing",
                "action": "bogus",
                "moss_intervention_ref": "w",
                "degraded": False,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    # fallback("direction_thrashing") == "escalate_to_user" -> AwaitingUserDecision
    assert st.current_state == "AwaitingUserDecision"
    assert st.current_moss_reason is None
    assert st.moss_interventions[0]["action"] == "escalate_to_user"


async def test_benchmark_back_to_hypothesis_replaces_confirmed_method(fake_bus):
    """The retired hypothesis loop must not bypass the benchmark method contract."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-benchmark-back-to-hypothesis"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        mode="benchmark",
        current_state="MossIntervening",
        current_moss_reason="experiment_design_thrashing",
        method_ref=f"method://{sid}/old",
        current_hypothesis_id="stale-hypothesis",
        problem_model_ref=f"problem_model://{sid}/v1",
        literature_brief_ref=f"literature_brief://{sid}/v1",
        research_memory_ref=f"research_memory://{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "experiment_design_thrashing",
            "action": "back_to_hypothesis",
            "moss_intervention_ref": f"moss_intervention://{sid}/v1",
            "degraded": False,
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "GeneratingMethod"
    assert state.method_not_working is True
    assert state.current_hypothesis_id is None
    requested = [
        event for event in fake_bus.published
        if event.event_type == "method.generation_requested"
    ]
    assert len(requested) == 1
    assert requested[0].payload["prior_method_ref"] == f"method://{sid}/old"
    assert requested[0].payload["moss_intervention_ref"] == f"moss_intervention://{sid}/v1"


async def test_resume_repairs_stale_benchmark_hypothesis_execution(fake_bus):
    """Sessions persisted by the old route must not re-run the frozen prior method."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-resume-stale-benchmark-hypothesis"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        mode="benchmark",
        current_state="ImplementingExperiment",
        method_ref=f"method://{sid}/old",
        current_hypothesis_id="h-new",
        current_experiment_id="h-new",
        problem_model_ref=f"problem_model://{sid}/v1",
        literature_brief_ref=f"literature_brief://{sid}/v1",
        research_memory_ref=f"research_memory://{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    result = await eng.resume_to_current_state(sid)

    state = await load_session_state(fake_bus.db, sid)
    assert result["state"] == "GeneratingMethod"
    assert state.current_state == "GeneratingMethod"
    assert state.method_not_working is True
    assert state.current_hypothesis_id is None


async def test_intervention_completed_stale_reason_is_ignored(fake_bus):
    """A completion for a superseded MOSS request cannot route the current intervention."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-stale-moss"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="implementation_thrashing")

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "direction_thrashing",
            "action": "rewrite_framing",
            "moss_intervention_ref": "stale",
            "degraded": False,
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "MossIntervening"
    assert state.current_moss_reason == "implementation_thrashing"
    assert state.moss_interventions == []


async def test_component_failure_reaudits_plan_without_replacing_method(fake_bus):
    """A component DAG failure preserves the Method and accepted experimental budgets."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-plan-revision"
    old_method = f"method://{sid}/old"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="component_ablation_failed",
        method_ref=old_method, current_experiment_id=old_method,
        component_ablation_failed=True, ablation_failed_component="router",
        exploration_loop_round=3, setting_rework_round=3,
        method_revision_round=3, component_revision_round=3,
        novelty_targeted_search_round=2,
        novelty_targeted_search_pending=True,
        novelty_targeted_search_id="old-method-search",
        implementation_fix_round={old_method: 3, "exp-old-secondary": 2},
        method_swap_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            # This old action is illegal and must fall back to the conservative plan audit.
            "reason": "component_ablation_failed", "action": "rethink_method",
            "moss_intervention_ref": f"moss_intervention://{sid}/rethink",
            "degraded": False,
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "DeliberatingMethod"
    assert state.component_plan_audit_pending is True
    assert state.method_ref == old_method
    assert state.current_experiment_id == old_method
    assert state.exploration_loop_round == 3
    assert state.setting_rework_round == 3
    assert state.method_revision_round == 3
    assert state.component_revision_round == 3
    assert state.implementation_fix_round == {old_method: 3, "exp-old-secondary": 2}
    assert state.novelty_targeted_search_round == 2
    assert state.novelty_targeted_search_pending is True
    assert state.novelty_targeted_search_id == "old-method-search"
    assert state.method_swap_round == 2


@pytest.mark.parametrize("revision_round", [
    _METHOD_REVISION_CAP - 2,
    _METHOD_REVISION_CAP - 1,
])
async def test_falsified_method_component_can_route_to_method_revision(
        fake_bus, revision_round):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-falsified-method-component"
    method_ref = f"method://{sid}/current"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="component_ablation_failed", method_ref=method_ref,
        component_ablation_failed=True, ablation_failed_component="scorer",
        component_plan_audit_pending=True, method_revision_round=revision_round,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "component_ablation_failed", "action": "revise_method",
            "moss_intervention_ref": f"moss_intervention://{sid}/falsified",
            "degraded": False,
        },
    ))

    revised = await load_session_state(fake_bus.db, sid)
    assert revised.current_state == "RevisingMethod"
    assert revised.method_ref == method_ref
    assert revised.component_plan_audit_pending is False
    assert revised.component_ablation_failed is True
    assert revised.ablation_failed_component == "scorer"
    assert revised.method_revision_round == revision_round


async def test_checkpoint_conflict_is_passed_to_component_plan_audit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-conflicting-component-parents"
    failure = {
        "underlying_reason_code": "component_plan_invalid",
        "underlying_reason": (
            "dependency checkpoint conflict for solve.py: "
            "checkpoint-a, checkpoint-b produced different states"),
    }
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="component_ablation_failed", mode="benchmark",
        method_ref=f"method://{sid}/current",
        baseline_result_ref=f"baseline_result://{sid}/v1",
        component_ablation_failed=True,
        ablation_failed_component="left",
        last_agent_failure=failure,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={"reason": "component_ablation_failed", "action": "revise_component",
                 "moss_intervention_ref": f"moss_intervention://{sid}/v1",
                 "degraded": False},
    ))

    requests = [event for event in fake_bus.published
                if event.event_type == "method.deliberation_requested"]
    assert requests[-1].payload["component_plan_failure"] == failure
    assert requests[-1].payload["component_plan_failed_component"] == "left"
    assert "checkpoint-a, checkpoint-b" in (
        requests[-1].payload["component_plan_revision_feedback"])

    state = await load_session_state(fake_bus.db, sid)
    state.last_agent_failure = {
        "underlying_reason_code": "component_plan_audit_failed",
        "underlying_reason": "conflicting parents unchanged",
        "component_plan_failure": failure,
    }
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng._transition(state, "DeliberatingMethod",
                          event_id="retry-conflicting-component-parents")
    requests = [event for event in fake_bus.published
                if event.event_type == "method.deliberation_requested"]
    assert requests[-1].payload["component_plan_failure"] == failure


async def test_invalid_root_checkpoint_keeps_latest_moss_repair_feedback(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "spec.db")
    sid = "sess-root-checkpoint-repair"
    method_ref = f"method://{sid}/current"
    guidance = "Remove excluded checkpoint references from inputs; preserve the baseline."
    moss_ref = store.put(kind="moss_intervention", session_id=sid, key="v1", obj={
        "action": "revise_component", "method_ref": method_ref,
        "rationale": "Repair the root input binding before running an experiment.",
        "prompt_patch": {"agent": "ExperimentPlannerAgent", "content": guidance},
    }, emitter="MOSS")
    memory_ref = store.put(kind="research_memory", session_id=sid, key="v1", obj={
        "diagnosis": "method_description_reconciliation",
        "diagnosis_evidence": ["obsolete failure from an earlier experiment"],
    }, emitter="MemoryAgent")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="component_ablation_failed", mode="benchmark",
        method_ref=method_ref, baseline_result_ref=f"baseline_result://{sid}/v1",
        research_memory_ref=memory_ref, method_revision_round=2,
        component_plan_audit_pending=True,
        component_ablation_failed=True, ablation_failed_component="follow-on",
        moss_interventions=[{"moss_intervention_ref": moss_ref}],
        last_agent_failure={
            "underlying_reason_code": "component_plan_invalid",
            "underlying_reason": "follow-on root checkpoint checkpoint-a11111 has no unique verified observation",
        },
    )
    try:
        _, build_payload = eng._entry_actions["DeliberatingMethod"]
        payload = build_payload(state, None)
        feedback = payload["component_plan_revision_feedback"]
        assert "checkpoint-a11111" in feedback
        assert guidance in feedback
        assert "Repair the root input binding" in feedback
        assert "obsolete failure" not in feedback
        assert payload["method_ref"] == method_ref
        assert payload["baseline_result_ref"] == state.baseline_result_ref
        assert payload["component_plan_failure"] == state.last_agent_failure
        assert state.method_revision_round == 2
        assert state.component_revision_round == 0
    finally:
        store.close()


async def test_baseline_fix_environment_pauses_instead_of_retrying_past_cap(fake_bus):
    """MOSS cannot claim an environment repair by blindly re-running the same command."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-cap"
    await _seed_moss_intervening_session(
        eng, sid, current_moss_reason="baseline_reproduction_thrashing")
    st = await load_session_state(fake_bus.db, sid)
    threshold, _ = eng._counter_threshold("baseline_reproduce_round")
    st.current_benchmark_id = "bench-1"
    st.baseline_reproduce_round = {"bench-1": threshold}
    await save_session_state(fake_bus.db, st, auto_commit=True)

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MossAgent",
            payload={
                "reason": "baseline_reproduction_thrashing",
                "action": "fix_environment",
                "moss_intervention_ref": "baseline-fix",
                "degraded": False,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "AwaitingUserDecision"
    assert st.awaiting_user_decision_reason == "baseline_unreproducible"
    assert st.baseline_reproduce_round == {"bench-1": threshold}


async def test_implementation_thrashing_discard_consolidates_without_evaluation(fake_bus):
    """The legal negative-result path emits a schema-valid failure consolidation request."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-implementation-discard"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="implementation_thrashing",
        current_experiment_id="exp-1", current_hypothesis_id="hyp-1",
        run_record_refs=[f"run_record://{sid}/failed"],
        evaluation_report_refs=[],
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing",
            "action": "discard_hypothesis",
            "moss_intervention_ref": f"moss_intervention://{sid}/discard",
            "degraded": False,
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ConsolidatingMemory"
    request = [p for p in fake_bus.published
               if p.event_type == "memory.consolidation_requested"][-1]
    validator = Registry(
        EVENTBUS_ROOT / "eventbus-registry.yaml").get_payload_validator(
            "memory.consolidation_requested", 1)
    assert list(validator.iter_errors(request.payload)) == []
    assert request.payload["evaluation_report_ref"] is None
    assert request.payload["run_record_ref"] == f"run_record://{sid}/failed"
    assert request.payload["failure_context"]["reason"] == "implementation_thrashing"


async def test_current_failed_round_does_not_reuse_historical_evaluation(fake_bus):
    """A prior round's report must not evaluate the newly failed RunRecord by position."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-stale-evaluation-history"
    historical_run = f"run_record://{sid}/historical"
    current_failed_run = f"run_record://{sid}/current-failed"
    historical_eval = f"evaluation_report://{sid}/historical"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="implementation_thrashing",
        current_experiment_id="exp-current", current_hypothesis_id="hyp-current",
        run_record_refs=[historical_run, current_failed_run],
        evaluation_report_refs=[historical_eval],
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing", "action": "discard_hypothesis",
            "moss_intervention_ref": f"moss_intervention://{sid}/discard",
            "degraded": False,
        },
    ))

    request = [p for p in fake_bus.published
               if p.event_type == "memory.consolidation_requested"][-1]
    assert request.payload["run_record_ref"] == current_failed_run
    assert request.payload["evaluation_report_ref"] is None
    assert request.payload["failure_context"]["reason"] == "implementation_thrashing"


@pytest.mark.parametrize("keep_searching", ["0", "1"])
async def test_discarded_execution_failure_reports_without_replacing_method(
        fake_bus, monkeypatch, keep_searching):
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", keep_searching)
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-discarded-execution"
    state = SessionState(
        session_id=sid, user_id="u", current_state="ConsolidatingMemory",
        method_ref="method-current", baseline_result_ref="baseline-frozen",
        method_swap_round=1, component_revision_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryAgent", payload={
            "research_memory_ref": "memory-execution-failure",
            "routing_verdict": "report", "diagnosis": "component_execution_blocked",
            "diagnosis_evidence": ["current attempt failed to export a result"],
        },
    ))
    current = await load_session_state(fake_bus.db, sid)
    assert current.current_state == "Reporting"
    assert current.method_ref == "method-current"
    assert current.baseline_result_ref == "baseline-frozen"
    assert current.method_swap_round == 1
    assert current.component_revision_round == 2
    assert current.method_not_working is False


async def test_discard_hypothesis_keeps_moss_context_even_with_current_evaluation(fake_bus):
    """MOSS discard must not be erased by re-consolidating the evaluated attempt."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-discard-evaluated-method"
    run_ref = f"run_record://{sid}/v1"
    evaluation_ref = f"evaluation_report://{sid}/v1"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        current_moss_reason="implementation_thrashing",
        current_experiment_id="method-1", method_ref="method-1",
        run_record_refs=[run_ref], evaluation_report_refs=[evaluation_ref],
        last_evaluated_run_record_ref=run_ref,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "moss.intervention_completed", sid, "MossAgent",
        payload={
            "reason": "implementation_thrashing", "action": "discard_hypothesis",
            "moss_intervention_ref": f"moss_intervention://{sid}/discard",
            "degraded": False,
        },
    ))

    request = [p for p in fake_bus.published
               if p.event_type == "memory.consolidation_requested"][-1]
    assert request.payload["evaluation_report_ref"] == evaluation_ref
    assert request.payload["failure_context"] == {
        "reason": "implementation_thrashing",
        "action": "discard_hypothesis",
        "artifact_ref": f"moss_intervention://{sid}/discard",
        "method_ref": "method-1",
    }


@pytest.mark.parametrize("already_requested", [False, True])
async def test_moss_revalidates_baseline_without_replacing_method(fake_bus, already_requested):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "baseline-environment-recovery"
    method = "spec://method/recovery/confirmed"
    prior = [{"reason": "implementation_thrashing", "action": "change_plan",
              "method_ref": method} for _ in range(3)]
    if already_requested:
        prior.append({"reason": "implementation_thrashing", "action": "revalidate_baseline",
                      "method_ref": method})
    state = SessionState(
        session_id=sid, user_id="u", current_state="MossIntervening", mode="benchmark",
        current_moss_reason="implementation_thrashing", method_ref=method,
        benchmark_spec_ref="spec://benchmark_spec/recovery/v1", current_benchmark_id="bench",
        confirmed_repo_url="https://example.org/research/repo",
        baseline_result_ref="spec://baseline_result/recovery/old", moss_interventions=prior,
        baseline_reproduce_round={"bench": 2}, component_revision_round=2,
        research_memory_ref="spec://research_memory/recovery/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "implementation_thrashing", "action": "revalidate_baseline",
        "moss_intervention_ref": "spec://moss_intervention/recovery/revalidate", "degraded": False,
    }))
    result = await load_session_state(fake_bus.db, sid)
    assert result.method_ref == method
    assert result.component_revision_round == 2
    assert result.baseline_reproduce_round == {"bench": 2}
    assert result.research_memory_ref == state.research_memory_ref
    if already_requested:
        assert result.current_state == "AwaitingUserDecision"
        assert result.baseline_result_ref == state.baseline_result_ref
    else:
        assert result.current_state == "ReproducingBaseline"
        assert result.baseline_result_ref is None
        assert result.last_baseline_attempt_ref == state.baseline_result_ref
        request = [p for p in fake_bus.published if p.event_type == "baseline.reproduction_requested"][-1]
        assert request.payload["moss_intervention_ref"] == "spec://moss_intervention/recovery/revalidate"
        from jsonschema import Draft202012Validator
        schema = json.loads((EVENTBUS_ROOT / "schemas/baseline.reproduction_requested.v1.schema.json").read_text())
        assert list(Draft202012Validator(schema).iter_errors(request.payload)) == []


async def test_overwritten_v1_run_rejects_stale_evaluation_by_content_identity(
        fake_bus, tmp_path):
    """Production reuses `run_record/v1`; the old report must bind to old bytes, not the ref."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(
            client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-overwritten-v1-run"
        prior_run = {
            "succeeded": True, "exit_code": 0, "metric_value": 0.9,
            "stdout_excerpt": "prior candidate", "stderr_excerpt": None,
            "environment": "sandbox", "artifacts": [],
        }
        same_run_ref = store.put(
            kind="run_record", session_id=sid, key="v1",
            obj=prior_run, emitter="RunnerAgent")
        prior_digest = _artifact_digest(prior_run)
        evaluation_ref = store.put(
            kind="evaluation_report", session_id=sid, key="v1",
            obj={
                "verdict": "valid_not_best", "is_improvement": False,
                "suspicious": False, "anti_cheat_flags": [],
                "run_record_digest": prior_digest,
            }, emitter="EvaluatorAgent")
        current_failed_run = {
            "succeeded": False, "exit_code": 1, "metric_value": None,
            "stdout_excerpt": "", "stderr_excerpt": "current candidate crashed",
            "environment": "fresh sandbox", "artifacts": [],
        }
        assert store.put(
            kind="run_record", session_id=sid, key="v1",
            obj=current_failed_run, emitter="RunnerAgent") == same_run_ref
        state = SessionState(
            session_id=sid, user_id="u1", current_state="MossIntervening",
            current_moss_reason="implementation_thrashing",
            current_experiment_id="exp-current", current_hypothesis_id="hyp-current",
            run_record_refs=[same_run_ref, same_run_ref],
            evaluation_report_refs=[evaluation_ref],
            last_evaluated_run_record_ref=same_run_ref,
        )
        await save_session_state(fake_bus.db, state, auto_commit=True)

        await eng.tick(_Evt(
            "moss.intervention_completed", sid, "MossAgent",
            payload={
                "reason": "implementation_thrashing", "action": "discard_hypothesis",
                "moss_intervention_ref": f"moss_intervention://{sid}/discard",
                "degraded": False,
            },
        ))

        request = [p for p in fake_bus.published
                   if p.event_type == "memory.consolidation_requested"][-1]
        assert request.payload["run_record_ref"] == same_run_ref
        assert request.payload["evaluation_report_ref"] is None
        assert request.payload["failure_context"]["reason"] == "implementation_thrashing"
    finally:
        store.close()
