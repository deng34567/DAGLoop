"""Unit tests for wiring AwaitingUserDecision's resume via the user-decision
reason-matrix (mirrors MOSS Step 1 Task 5's MossIntervening routing, see
test_engine_moss_advance.py).

Covers ``_on_AwaitingUserDecision_user_decision_provided``:
  - routes via the user-decision reason-matrix
    (``self.user_decision_guard.next_state``);
  - applies ``side_effects`` (scalar "reset to 0" counter resets, and prose
    annotations appended to ``state.decision_caveats``);
  - reason=literature_uncertain, decision=continue_with_caveat ->
    ModelingProblem -- THE loop-breaker: this is what lets a human get an
    always-failing literature audit out of its stuck loop.
  - reason=literature_uncertain, decision=deepen_literature -> DeepResearching;
    counter K (literature_audit_round) reset to 0.
  - reason=out_of_scope, decision=proceed_out_of_scope -> RoutingResearchMode;
    state.mode set to "lab_in_the_loop" (see also
    test_engine_lab_track_advance.py for the out_of_scope-specific entry
    coverage).
  - decision=abandon -> Failed (terminal), for any reason.
  - an illegal decision for the recorded reason -> defensive no-op (logged),
    stays paused at AwaitingUserDecision, no crash.
  - ``awaiting_user_decision_reason`` is cleared on every non-illegal,
    non-templated routing.

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_moss_advance.py's patterns (does NOT auto-deliver; drives the
engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.mark.parametrize("closeout,has_run,expected", [
    (False, True, "Reporting"),
    (True, False, "Reporting"),
    (True, True, "ConsolidatingMemory"),
])
async def test_requested_closeout_reconciles_existing_results_before_reporting(
        fake_bus, tmp_path, closeout, has_run, expected):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / "store")
    sid = "failure-closeout"
    store.put(kind="human_idea_seed", session_id=sid, key="closeout-v1",
              obj={"closeout": closeout, "idea": "Finalize retained measured evidence."},
              emitter="Human")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    state = SessionState(session_id=sid, user_id="u", mode="benchmark",
        current_state="AwaitingUserDecision", awaiting_user_decision_reason="agent_failure",
        method_ref="method://current", baseline_result_ref="baseline://frozen",
        research_memory_ref="memory://existing",
        method_feedback_ref="feedback://applied",
        run_record_refs=["run://existing"] if has_run else [],
        method_revision_round=1, component_revision_round=1)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("user.decision_provided", sid, "Human", {"decision": "report_now"}))
    result = await load_session_state(fake_bus.db, sid)
    assert result.current_state == expected
    assert result.method_ref == state.method_ref
    assert result.baseline_result_ref == state.baseline_result_ref
    assert result.method_revision_round == result.component_revision_round == 1
    if expected == "ConsolidatingMemory":
        request = [e for e in fake_bus.published if e.event_type == "memory.consolidation_requested"][-1]
        assert request.payload["qualified_component_closeout_requested"] is True
        assert request.payload["method_ref"] == state.method_ref
        assert request.payload["applied_method_feedback_ref"] == state.method_feedback_ref


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


@pytest.mark.parametrize("closeout,review,measured,bound,passed,violations,expected", [
    (True, "insufficient_evidence", True, True, True, [], "PlanningExperiment"),
    (False, "insufficient_evidence", True, True, True, [], "Reporting"),
    (True, "partial", True, True, True, [], "Reporting"),
    (True, "insufficient_evidence", False, True, True, [], "Reporting"),
    (True, "insufficient_evidence", True, False, True, [], "Reporting"),
    (True, "insufficient_evidence", True, True, False, [], "Reporting"),
    (True, "insufficient_evidence", True, True, True, ["protected edit"], "Reporting"),
])
async def test_closeout_retry_repairs_missing_implementation_evidence(
        fake_bus, tmp_path, monkeypatch, closeout, review, measured, bound, passed, violations, expected):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / "store")
    sid = "closeout-evidence-retry"
    store.put(kind="human_idea_seed", session_id=sid, key="closeout-v1",
              obj={"closeout": closeout, "idea": "Finalize retained measured evidence."},
              emitter="Human")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    result = {
        "method_ref": "method://current" if bound else "method://old",
        "baseline_result_ref": "baseline://frozen",
        "run_record_snapshot_ref": "run://existing",
        "component_execution": {"mode": "component", "execution_succeeded": True,
                                "contract_result": {"passed": passed}},
        "measurement_evidence": {"valid": measured, "value": 0.6},
        "method_implementation_review": {"verdict": review,
                                         "task_contract_review": {"violations": violations}},
    }
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: result)
    state = SessionState(session_id=sid, user_id="u", mode="benchmark",
        current_state="AwaitingUserDecision", previous_state="Reporting",
        awaiting_user_decision_reason="agent_failure",
        last_agent_failure={"underlying_reason_code": "completion_incomplete"},
        method_ref="method://current", baseline_result_ref="baseline://frozen",
        research_memory_ref="memory://existing", run_record_refs=["run://existing"],
        method_revision_round=7, component_revision_round=3)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("user.decision_provided", sid, "Human", {"decision": "retry_session"}))
    saved = await load_session_state(fake_bus.db, sid)
    assert saved.current_state == expected
    assert saved.method_ref == state.method_ref
    assert saved.baseline_result_ref == state.baseline_result_ref
    assert saved.run_record_refs == state.run_record_refs
    assert saved.method_revision_round == 7
    assert saved.component_revision_round == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_awaiting_user_decision_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    awaiting_user_decision_reason: str,
    literature_audit_round: int = 0,
) -> None:
    """Persist a session in AwaitingUserDecision with the given reason.

    ``research_spec_ref`` / ``literature_brief_ref`` are pre-populated because
    a couple of routes (ModelingProblem / DeepResearching) have entry actions
    that read them off ``state``.
    """
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AwaitingUserDecision",
        awaiting_user_decision_reason=awaiting_user_decision_reason,
        literature_audit_round=literature_audit_round,
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): literature_uncertain / continue_with_caveat -> ConfirmingMethod
#      (THE loop-breaker)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline,rounds,expected", [
    ("baseline://frozen", 6, "DeliberatingMethod"),
    ("baseline://frozen", 10, "ConsolidatingMemory"),
    (None, 0, "PlanningExperiment"),
])
async def test_escalated_plan_review_retains_local_component_repair(fake_bus, baseline, rounds, expected):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "escalated-component-repair"
    state = SessionState(session_id=sid, user_id="u1", mode="benchmark",
        current_state="AwaitingUserDecision", awaiting_user_decision_reason="experiment_design_thrashing",
        method_ref="method://frozen", baseline_result_ref=baseline, component_revision_round=rounds,
        experiment_plan_review_round={"method://frozen": 3, "method://other": 1})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("user.decision_provided", sid, "Human", payload={"decision": "revise_component"}))
    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed.current_state == expected
    assert resumed.method_ref == "method://frozen"
    assert resumed.baseline_result_ref == baseline
    assert resumed.method_not_working is False
    assert resumed.method_swap_round == 0
    assert resumed.component_revision_round == rounds + int(bool(baseline))
    assert resumed.component_plan_audit_pending is (expected == "DeliberatingMethod")
    assert resumed.experiment_plan_review_round == {"method://other": 1}


async def test_continue_with_caveat_breaks_the_literature_audit_loop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    await _seed_awaiting_user_decision_session(
        eng, sid, awaiting_user_decision_reason="literature_uncertain"
    )

    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "continue_with_caveat"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConfirmingMethod"
    assert st.awaiting_user_decision_reason is None
    assert st.decision_caveats == ["报告标注 novelty caveat"]


# ---------------------------------------------------------------------------
# (b): literature_uncertain / deepen_literature -> DeepResearching; K reset
# ---------------------------------------------------------------------------


async def test_deepen_literature_resets_counter_k_and_returns_to_deep_research(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_awaiting_user_decision_session(
        eng,
        sid,
        awaiting_user_decision_reason="literature_uncertain",
        literature_audit_round=2,
    )

    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "deepen_literature"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "DeepResearching"
    assert st.literature_audit_round == 0
    assert st.awaiting_user_decision_reason is None


# ---------------------------------------------------------------------------
# (c): out_of_scope / proceed_out_of_scope -> RoutingResearchMode; lab mode
# ---------------------------------------------------------------------------


async def test_proceed_out_of_scope_sets_lab_mode_and_routes_to_selecting_track(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_awaiting_user_decision_session(
        eng, sid, awaiting_user_decision_reason="out_of_scope"
    )

    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "proceed_out_of_scope"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "RoutingResearchMode"
    assert st.mode == "lab_in_the_loop"
    assert st.awaiting_user_decision_reason is None


# ---------------------------------------------------------------------------
# (d): abandon -> Failed, regardless of reason
# ---------------------------------------------------------------------------


async def test_abandon_transitions_to_failed(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_awaiting_user_decision_session(
        eng, sid, awaiting_user_decision_reason="literature_uncertain"
    )

    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "abandon"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "Failed"
    assert st.awaiting_user_decision_reason is None


@pytest.mark.parametrize("hypothesis,method,key", [
    ("h-current", "method-current", "h-current"),
    (None, "method-current", "method-current"),
    (None, None, "__unscoped_plan__"),
])
async def test_simplify_experiment_resets_the_actual_review_counter(fake_bus, hypothesis, method, key):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-simplify-review"
    await _seed_awaiting_user_decision_session(
        eng, sid, awaiting_user_decision_reason="experiment_design_thrashing")
    st = await load_session_state(fake_bus.db, sid)
    st.current_hypothesis_id = hypothesis
    st.method_ref = method
    st.experiment_plan_review_round = {key: 3, "unrelated-old-plan": 2}
    await save_session_state(fake_bus.db, st, auto_commit=True)
    await eng.tick(_Evt("user.decision_provided", sid, "Human",
                        payload={"decision": "simplify_experiment"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"
    assert st.experiment_plan_review_round == {"unrelated-old-plan": 2}
    assert st.method_ref == method
    assert st.method_not_working is False


async def test_experiment_design_thrashing_can_replace_frozen_method(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-revise-method"
    await _seed_awaiting_user_decision_session(
        eng, sid, awaiting_user_decision_reason="experiment_design_thrashing"
    )
    st = await load_session_state(fake_bus.db, sid)
    st.mode = "benchmark"
    st.method_ref = f"method://{sid}/old"
    st.current_hypothesis_id = "stale-plan"
    st.research_memory_ref = f"research_memory://{sid}/v1"
    await save_session_state(fake_bus.db, st, auto_commit=True)

    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "revise_method"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "GeneratingMethod"
    assert st.method_not_working is True
    assert st.current_hypothesis_id is None
    requested = [
        event for event in fake_bus.published
        if event.event_type == "method.generation_requested"
    ]
    assert requested[-1].payload["prior_method_ref"] == f"method://{sid}/old"
    assert requested[-1].payload["research_memory_ref"] == f"research_memory://{sid}/v1"


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,has_baseline,revision,expected", [
    ("component", True, 0, "DeliberatingMethod"),
    ("component", True, 2, "ConsolidatingMemory"),
    ("integration", True, 0, "PlanningExperiment"),
    ("component", False, 0, "PlanningExperiment"),
])
async def test_change_plan_reaudits_frozen_component_from_awaiting(
        fake_bus, monkeypatch, mode, has_baseline, revision, expected):
    monkeypatch.setattr("autoresearch.orchestrator.engine._COMPONENT_REVISION_CAP", 2)
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-plan-recovery"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="AwaitingUserDecision",
        awaiting_user_decision_reason="implementation_thrashing",
        method_ref=f"method://{sid}/v1",
        baseline_result_ref=f"baseline_result://{sid}/v1" if has_baseline else None,
        component_revision_round=revision,
    )
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: {
        "component_execution": {"mode": mode, "component_id": "current-component",
                                "status": "contract_failed"},
    })
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("user.decision_provided", sid, "Human", {"decision": "change_plan"}))
    result = await load_session_state(fake_bus.db, sid)
    assert result.current_state == expected
    assert result.method_ref == state.method_ref
    assert result.baseline_result_ref == state.baseline_result_ref
    assert result.method_swap_round == 0
    assert result.component_revision_round == revision + int(mode == "component" and has_baseline)
    assert result.component_plan_audit_pending == (expected == "DeliberatingMethod")
    assert result.awaiting_user_decision_reason is None
    if expected == "DeliberatingMethod":
        assert result.ablation_failed_component == "current-component"


async def test_discard_hypothesis_from_awaiting_reaches_memory_as_failure(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-human-discard"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="AwaitingUserDecision",
        awaiting_user_decision_reason="implementation_thrashing",
        mode="benchmark", method_ref="method-current",
        baseline_result_ref="baseline-current",
        research_memory_ref="memory-with-best-result",
        run_record_refs=["run-failed"],
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt("user.decision_provided", sid, "Human",
                        {"decision": "discard_hypothesis"}))

    current = await load_session_state(fake_bus.db, sid)
    assert current.current_state == "ConsolidatingMemory"
    assert current.research_memory_ref == "memory-with-best-result"
    request = [p for p in fake_bus.published
               if p.event_type == "memory.consolidation_requested"][-1]
    assert request.payload["failure_context"] == {
        "reason": "implementation_thrashing",
        "action": "discard_hypothesis",
        "method_ref": "method-current",
    }


# (e): illegal decision for the reason -> stays paused, no crash
# ---------------------------------------------------------------------------


async def test_illegal_decision_for_reason_stays_paused(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_awaiting_user_decision_session(
        eng, sid, awaiting_user_decision_reason="literature_uncertain"
    )

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "proceed_out_of_scope"},  # not legal for literature_uncertain
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "AwaitingUserDecision"
    assert st.awaiting_user_decision_reason == "literature_uncertain"
    assert fake_bus.published[n_before:] == []


async def test_retry_unreproducible_baseline_preserves_repository_and_history(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-retry"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="AwaitingUserDecision",
        awaiting_user_decision_reason="baseline_unreproducible",
        current_benchmark_id="bench-1", baseline_reproduce_round={"bench-1": 6},
        confirmed_repo_url="https://example.org/repository",
        last_baseline_attempt_ref=f"spec://baseline_result/{sid}/v1",
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("user.decision_provided", sid, "Human", {"decision": "retry_session"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "ReproducingBaseline"
    assert st.confirmed_repo_url == state.confirmed_repo_url
    assert st.baseline_reproduce_round == {"bench-1": 6}
    assert st.last_baseline_attempt_ref == state.last_baseline_attempt_ref
    assert st.baseline_result_ref is None
    assert st.awaiting_user_decision_reason is None


async def test_changing_unreproducible_benchmark_resets_its_retry_budget(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-change"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AwaitingUserDecision",
        awaiting_user_decision_reason="baseline_unreproducible",
        current_benchmark_id="bench-1",
        baseline_reproduce_round={"bench-1": 6},
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "user.decision_provided", sid, "Human",
        payload={"decision": "change_benchmark"},
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "SelectingBenchmark"
    # The old id remains available to the selector as context until a new selection arrives.
    assert st.current_benchmark_id == "bench-1"
    assert st.baseline_reproduce_round == {}


@pytest.mark.parametrize("via_moss", [False, True])
@pytest.mark.parametrize("execution_contract", [None, {"evaluation": {"required_values": {"/split": "dev"}}}])
async def test_failed_repository_reselection_preserves_method_and_returns_to_selector(fake_bus, via_moss, execution_contract):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "repo-reselection"
    state = SessionState(
        session_id=sid, user_id="u1",
        current_state="MossIntervening" if via_moss else "AwaitingUserDecision",
        awaiting_user_decision_reason=None if via_moss else "baseline_unreproducible",
        current_moss_reason="baseline_reproduction_thrashing" if via_moss else None,
        current_benchmark_id="old-benchmark", baseline_reproduce_round={"old-benchmark": 6},
        confirmed_repo_url="https://github.com/example/missing",
        confirmed_boundary={"run_cmd": "old-repository-command",
                            "download_authorization": {"approved": True}, **(
            {"execution_contract": execution_contract} if execution_contract else {})},
        method_ref="spec://method/reselection/confirmed",
        research_spec_ref="spec://research_spec/reselection/final",
        literature_brief_ref="spec://literature_brief/reselection/final",
        last_baseline_attempt_ref="spec://baseline_result/reselection/failed",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt(
        "moss.intervention_completed" if via_moss else "user.decision_provided", sid,
        "MOSS" if via_moss else "Human",
        payload=({"reason": "baseline_reproduction_thrashing", "action": "change_benchmark",
                  "moss_intervention_ref": "spec://moss_intervention/reselection/v1", "degraded": False}
                 if via_moss else {"decision": "change_benchmark"}),
    ))
    updated = await load_session_state(fake_bus.db, sid)
    assert updated.current_state == "HumanIdeaCheckpoint"
    assert updated.confirmed_repo_url is None
    assert updated.confirmed_boundary == {
        "download_authorization": {"approved": True}, **(
            {"execution_contract": execution_contract} if execution_contract else {})}
    assert updated.method_ref == state.method_ref
    assert updated.last_baseline_attempt_ref == state.last_baseline_attempt_ref
    assert updated.baseline_result_ref is None
    assert updated.baseline_reproduce_round == {}
    await eng.tick(_Evt("human.idea_provided", sid, "Human", payload={
        "confirmed_repo_url": "https://github.com/example/reachable"}))
    updated = await load_session_state(fake_bus.db, sid)
    assert updated.current_state == "SelectingBenchmark"
    assert updated.confirmed_repo_url == "https://github.com/example/reachable"
    assert updated.method_ref == state.method_ref
    assert updated.confirmed_boundary == {
        "download_authorization": {"approved": True}, **(
            {"execution_contract": execution_contract} if execution_contract else {})}


async def test_replication_failure_report_consolidates_failure_before_reporting(fake_bus):
    """A baseline failure has no RunRecord, but Reporting still needs durable memory."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-report"
    intervention_ref = f"spec://moss_intervention/{sid}/baseline-v1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AwaitingUserDecision",
        awaiting_user_decision_reason="baseline_unreproducible",
        current_benchmark_id="bench-1",
        baseline_reproduce_round={"bench-1": 6},
        moss_interventions=[{
            "reason": "baseline_reproduction_thrashing",
            "action": "fix_environment",
            "moss_intervention_ref": intervention_ref,
            "degraded": False,
        }],
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "user.decision_provided", sid, "Human",
        payload={"decision": "write_replication_failure_report"},
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConsolidatingMemory"
    emitted = [
        item for item in fake_bus.published
        if item.event_type == "memory.consolidation_requested"
    ]
    assert len(emitted) == 1
    assert emitted[0].payload["evaluation_report_ref"] is None
    assert emitted[0].payload["run_record_ref"] is None
    assert emitted[0].payload["failure_context"] == {
        "reason": "baseline_reproduction_thrashing",
        "action": "fix_environment",
        "artifact_ref": intervention_ref,
    }
