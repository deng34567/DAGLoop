"""Unit tests for the generic ``agent.failed`` engine handler (H1).

Before this, ``agent.failed`` had NO engine handler, so a permanent agent failure
stalled the session forever. The engine now routes a generic ``agent.failed`` in
any non-terminal state through the real MossAgent before a possible human
escalation. Terminal / Suspended states and normal flows are unaffected.

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; drives the
engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine, _COMPONENT_REVISION_CAP
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.mark.parametrize("bound", [True, False])
async def test_resume_retained_closeout_after_wrong_method_replacement_check(fake_bus, tmp_path, monkeypatch, bound):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / "store")
    sid = "closeout-repeat-repair"
    method = store.put(kind="method", session_id=sid, key="v1", obj={}, emitter="MethodAgent")
    baseline = store.put(kind="baseline_result", session_id=sid, key="v1", obj={}, emitter="RunnerAgent")
    review = store.put(kind="method_improvement_review", session_id=sid, key="slot-1", obj={
        "status": "completed", "trigger": "qualified_component_closeout", "method_ref": method,
        "baseline_ref": baseline if bound else "another-baseline",
        "qualified_closeout_components": [{"checkpoint": "measured-source"}],
        "decision": {"route": "revise_method"}}, emitter="MemoryAgent")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    state = SessionState(session_id=sid, user_id="u", current_state="Failed", previous_state="RevisingMethod",
        method_ref=method, baseline_result_ref=baseline, method_feedback_in_progress=True,
        method_feedback_ref=review, agent_failure_round=3,
        last_agent_failure={"underlying_reason_code": "method_not_replaced"},
        active_request={"state": "RevisingMethod", "payload": {
            "method_ref": method, "baseline_result_ref": baseline}})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    entries = []
    async def dispatch(current, **kwargs): entries.append(current.current_state)
    monkeypatch.setattr(eng, "_execute_entry_action", dispatch)
    await eng.resume_to_current_state(sid)
    restored = await load_session_state(fake_bus.db, sid)
    assert restored.current_state == ("RevisingMethod" if bound else "Failed")
    assert ("RevisingMethod" in entries) is bound
    assert restored.method_ref == method and restored.baseline_result_ref == baseline
    assert restored.last_agent_failure == state.last_agent_failure
    assert not restored.completion_verified
    store.close()


@pytest.mark.parametrize('bound', [True, False])
async def test_resume_reconciles_retained_component_after_premature_report(fake_bus, tmp_path, bound):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / 'store')
    sid = 'retained-before-report'
    method = store.put(kind='method', session_id=sid, key='v1', obj={}, emitter='MethodAgent')
    baseline = store.put(kind='baseline_result', session_id=sid, key='v1',
                         obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    review = store.put(kind='method_improvement_review', session_id=sid, key='slot-1', obj={
        'status': 'completed', 'trigger': 'component_method_reconciliation', 'method_ref': method,
        'baseline_ref': baseline if bound else 'another-baseline',
        'decision': {'route': 'report', 'reason': 'Official evaluation still needs to run.'}},
        emitter='MemoryAgent')
    memory = store.put(kind='research_memory', session_id=sid, key='v1',
                       obj={'method_feedback_refs': [review]}, emitter='MemoryAgent')
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    state = SessionState(session_id=sid, user_id='u', current_state='AwaitingUserDecision',
        previous_state='Reporting', awaiting_user_decision_reason='agent_failure',
        method_ref=method, baseline_result_ref=baseline, research_memory_ref=memory,
        last_agent_failure={'underlying_reason_code': 'completion_incomplete'})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.resume_to_current_state(sid)
    restored = await load_session_state(fake_bus.db, sid)
    assert restored.current_state == ('ConsolidatingMemory' if bound else 'AwaitingUserDecision')
    assert restored.baseline_result_ref == baseline and not restored.completion_verified
    assert store.get(review)['decision']['route'] == 'report'
    if bound:
        assert fake_bus.published[-1].event_type == 'memory.consolidation_requested'
    store.close()


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    event_id: str = "evt-x"
    payload: dict = field(default_factory=dict)


async def _seed(eng, sid, current_state):
    state = SessionState(session_id=sid, user_id="u1", current_state=current_state)
    await save_session_state(eng.client.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a) agent.failed in a mid-flow state -> MossIntervening(agent_failure).
# ---------------------------------------------------------------------------


async def test_agent_failed_midflow_routes_to_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-fail"
    await _seed(eng, sid, "RunningExperiment")

    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent",
                        payload={"agent_name": "RunnerAgent",
                                 "reason_classification": "permanent"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "agent_failure"
    assert st.awaiting_user_decision_reason is None
    assert st.previous_state == "RunningExperiment"
    assert st.last_agent_failure["failed_state"] == "RunningExperiment"
    assert fake_bus.published[-1].event_type == "moss.intervention_requested"
    assert fake_bus.published[-1].payload["agent_failure"] == st.last_agent_failure


@pytest.mark.parametrize("message", [
    "RuntimeError: cannot schedule new futures after shutdown",
    "RuntimeError: cannot schedule new futures after interpreter shutdown",
])
async def test_executor_shutdown_failure_preserves_unconsumed_restart_request(fake_bus, message):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-process-shutdown"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="MossIntervening",
        request_binding_required=True, agent_failure_round=2,
        active_request={"event_id": "request-1", "event_type": "moss.intervention_requested",
                        "state": "MossIntervening", "payload": {}, "consumed": False},
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    event = _Evt("agent.failed", sid, "MOSS", payload={
        "agent_name": "MOSS", "reason_classification": "transient",
        "underlying_reason": message,
    })
    event.causation_id = "request-1"

    await eng.tick(event)

    stored = await load_session_state(fake_bus.db, sid)
    assert stored.current_state == "MossIntervening"
    assert stored.agent_failure_round == 2
    assert stored.last_agent_failure is None
    assert stored.active_request["consumed"] is False
    assert not fake_bus.published


async def test_method_not_replaced_reuses_method_and_continues_planning(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-method-noop"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RevisingMethod",
        method_ref="method://frozen",
        baseline_result_ref="baseline://frozen",
        method_feedback_in_progress=True,
        method_feedback_ref="review://frozen",
        method_revision_round=1,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    failure = {
        "agent_name": "MethodAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "method_not_replaced",
        "underlying_reason": "bounded replacement retry repeated the prior method",
    }

    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload=failure))

    restored = await load_session_state(fake_bus.db, sid)
    assert restored.current_state == "PlanningExperiment"
    assert restored.previous_state == "RevisingMethod"
    assert restored.method_ref == state.method_ref
    assert restored.baseline_result_ref == state.baseline_result_ref
    assert restored.last_agent_failure == failure
    assert restored.agent_failure_round == 0
    assert restored.awaiting_user_decision_reason is None
    assert not restored.method_feedback_in_progress


@pytest.mark.parametrize('bound,has_baseline,expected_state', [
    (True, True, 'ConsolidatingMemory'),
    (True, False, 'AwaitingUserDecision'),
    (False, True, 'PlanningExperiment'),
])
async def test_method_not_replaced_does_not_execute_a_rejected_contract(
        fake_bus, tmp_path, bound, has_baseline, expected_state):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / 'store')
    sid = 'unchanged-rejected-method'
    method = store.put(kind='method', session_id=sid, key='v1', obj={}, emitter='MethodAgent')
    baseline = (store.put(kind='baseline_result', session_id=sid, key='v1',
                         obj={'reproduced': True}, emitter='RunnerAgent')
                if has_baseline else None)
    report = store.put(kind='method_debate_report', session_id=sid, key='contract', obj={
        'scope': 'component_plan',
        'confirmed_method_ref': method if bound else 'another-method',
        'baseline_result_ref': baseline,
        'method_revision_required': True,
    }, emitter='MethodAgent')
    rejected = {'underlying_reason_code': 'confirmed_method_contract_conflict',
                'underlying_reason': 'The frozen requirements are contradictory.',
                'component_plan_checkpoint_ref': report}
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    state = SessionState(session_id=sid, user_id='u1', mode='benchmark',
        current_state='RevisingMethod', method_ref=method, baseline_result_ref=baseline,
        current_experiment_id=method, last_agent_failure=rejected,
        method_revision_round=1, component_revision_round=2)
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt('agent.failed', sid, 'MethodAgent', payload={
        'agent_name': 'MethodAgent', 'reason_classification': 'permanent',
        'underlying_reason_code': 'method_not_replaced',
        'underlying_reason': 'bounded replacement retry repeated the prior method'}))

    restored = await load_session_state(fake_bus.db, sid)
    assert restored.current_state == expected_state
    assert restored.method_ref == method and restored.baseline_result_ref == baseline
    assert restored.method_revision_round == 1 and restored.component_revision_round == 2
    if bound:
        assert restored.current_experiment_id is None
        assert restored.last_agent_failure['unresolved_method_contract'] == rejected
        assert not restored.completion_verified
        if has_baseline:
            event = fake_bus.published[-1]
            assert event.event_type == 'memory.consolidation_requested'
            assert event.payload['failure_context']['action'] == 'report_now'
            assert event.payload['failure_context']['reason'] == 'method_contract_repair_exhausted'
            assert event.payload['qualified_component_closeout_requested'] is True
    store.close()


async def test_deep_research_parse_failure_continues_without_human_pause(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-deep-research-parse"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="DeepResearching",
        research_spec_ref="research_spec://frozen",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    failure = {
        "agent_name": "DeepResearchAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "deep_research_parse_error",
        "underlying_reason": "literature_brief failed schema validation",
    }

    await eng.tick(_Evt("agent.failed", sid, "DeepResearchAgent", payload=failure))

    restored = await load_session_state(fake_bus.db, sid)
    assert restored.current_state == "ModelingProblem"
    assert restored.previous_state == "DeepResearching"
    assert restored.last_agent_failure == failure
    assert restored.agent_failure_round == 0
    assert restored.awaiting_user_decision_reason is None


@pytest.mark.parametrize("literature_ref", [None, "spec://literature_brief/s/final"])
async def test_resume_missing_literature_after_legacy_fail_open(fake_bus, literature_ref):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    state = SessionState(
        session_id="legacy-missing-literature", current_state="AwaitingUserDecision",
        previous_state="ModelingProblem", awaiting_user_decision_reason="agent_failure",
        research_spec_ref="spec://research_spec/s/final", literature_brief_ref=literature_ref,
        agent_failure_round=1, last_agent_failure={
            "agent_name": "ProblemModelingAgent", "underlying_reason_code": "unexpected_error",
            "underlying_reason": "KeyError: 'no artifact for ref: None'"})
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.resume_to_current_state(state.session_id)

    restored = await load_session_state(fake_bus.db, state.session_id)
    assert restored.current_state == (
        "DeepResearching" if literature_ref is None else "AwaitingUserDecision")
    assert restored.last_agent_failure == state.last_agent_failure
    assert restored.agent_failure_round == 1
    searches = [p for p in fake_bus.published if p.event_type == "deep_research.requested"]
    assert bool(searches) is (literature_ref is None)
    if searches:
        assert searches[0].payload["research_spec_ref"] == state.research_spec_ref


@pytest.mark.parametrize("failed_state", ["RunningExperiment", "DeepResearching", "PlanningExperiment"])
@pytest.mark.parametrize("prior_rounds", [0, 2])
async def test_generic_failure_persists_current_diagnosis(fake_bus, failed_state, prior_rounds):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-failure-diagnosis"
    state = SessionState(
        session_id=sid, user_id="u1", current_state=failed_state,
        agent_failure_round=prior_rounds,
        last_agent_failure={"underlying_reason_code": "previous_failure"},
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    failure = {
        "agent_name": "TestAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "invalid_grounded_output",
        "underlying_reason": "A required evidence reference was not grounded.",
    }
    await eng.tick(_Evt("agent.failed", sid, "TestAgent", payload=failure))
    stored = await load_session_state(fake_bus.db, sid)
    assert stored.last_agent_failure == {**failure, "failed_state": failed_state}
    assert stored.agent_failure_round == prior_rounds + 1
    assert stored.current_state == ("MossIntervening" if prior_rounds == 0 else "Failed")


async def test_novelty_parse_failure_retries_without_human_pause(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-novelty-parse-retry"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AuditingLiteratureNovelty",
        research_spec_ref=f"spec://research_spec/{sid}/v1",
        literature_brief_ref=f"spec://literature_brief/{sid}/v1",
        spec_interpretation_ref=f"spec://spec_interpretation/{sid}/v1",
        method_ref=f"spec://method/{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "agent.failed", sid, "NoveltyAuditAgent", event_id="parse-1",
        payload={
            "agent_name": "NoveltyAuditAgent",
            "reason_classification": "permanent",
            "underlying_reason_code": "novelty_audit_parse_error",
            "underlying_reason": "structured report was not grounded",
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "AuditingLiteratureNovelty"
    assert st.awaiting_user_decision_reason is None
    assert st.agent_retry_count == {"AuditingLiteratureNovelty": 1}
    assert fake_bus.published[-1].event_type == "evidence.audit_requested"


async def test_novelty_parse_failure_escalates_to_moss_after_bound(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-novelty-parse-cap"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AuditingLiteratureNovelty",
        research_spec_ref=f"spec://research_spec/{sid}/v1",
        literature_brief_ref=f"spec://literature_brief/{sid}/v1",
        spec_interpretation_ref=f"spec://spec_interpretation/{sid}/v1",
        method_ref=f"spec://method/{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    failure = {
        "agent_name": "NoveltyAuditAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "novelty_audit_parse_error",
        "underlying_reason": "structured report was not grounded",
    }

    for attempt in range(1, 4):
        await eng.tick(_Evt(
            "agent.failed", sid, "NoveltyAuditAgent",
            event_id=f"parse-{attempt}", payload=failure,
        ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "literature_novelty_audit_failed"
    assert "AuditingLiteratureNovelty" not in st.agent_retry_count
    assert fake_bus.published[-1].event_type == "moss.intervention_requested"


async def test_llm_quota_exhaustion_suspends_without_spending_failure_budget(fake_bus):
    """A depleted external LLM account is resumable, not a retryable run failure."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-quota"
    await _seed(eng, sid, "ReinterpretingSpec")

    await eng.tick(_Evt(
        "agent.failed",
        sid,
        "ProblemModelingAgent",
        payload={
            "agent_name": "ProblemModelingAgent",
            # Some cores currently classify every client exception as transient;
            # the engine must still recognize the provider's explicit quota code.
            "reason_classification": "transient",
            "underlying_reason_code": "reinterpret_llm_error",
            "underlying_reason": (
                "PermissionDeniedError: Error code: 403 - "
                "{'code': 'insufficient_user_quota', 'message': '用户额度不足'}"
            ),
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Suspended"
    assert st.previous_state == "ReinterpretingSpec"
    assert st.agent_failure_round == 0
    assert st.awaiting_user_decision_reason is None
    assert st.last_agent_failure["underlying_reason_code"] == "reinterpret_llm_error"


@pytest.mark.parametrize("error", ["Conversation error code: LLMRateLimitError",
                                  "RateLimitError: Allocated quota exceeded",
                                  "PermissionDeniedError: 403 account_overdue"])
async def test_sdk_model_rate_limit_preserves_method_baseline_and_failure_counters(fake_bus, error):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-sdk-rate-limit"
    state = SessionState(session_id=sid, user_id="u1", current_state="RunningExperiment",
                         method_ref="method://frozen", baseline_result_ref="baseline://frozen",
                         implementation_fix_round={"method://frozen": 2})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload={
        "agent_name": "RunnerAgent", "reason_classification": "transient",
        "underlying_reason_code": "llm_provider_rate_limit",
        "underlying_reason": error,
    }))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Suspended"
    assert st.previous_state == "RunningExperiment"
    assert st.method_ref == "method://frozen"
    assert st.baseline_result_ref == "baseline://frozen"
    assert st.implementation_fix_round == {"method://frozen": 2}
    assert st.agent_failure_round == 0
    assert not st.agent_retry_count


@pytest.mark.parametrize("error", ["Conversation error code: LLMServiceUnavailableError",
                                  "Conversation error code: LLMError; llm_provider_timeout"])
async def test_sdk_service_unavailable_preserves_baseline_reproduction_budget(fake_bus, error):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-sdk-service-unavailable"
    state = SessionState(session_id=sid, user_id="u1", current_state="ReproducingBaseline",
                         current_benchmark_id="benchmark-1",
                         baseline_reproduce_round={"benchmark-1": 4})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload={
        "agent_name": "RunnerAgent", "reason_classification": "transient",
        "underlying_reason_code": "runner_parse_error",
        "underlying_reason": error,
    }))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Suspended"
    assert st.previous_state == "ReproducingBaseline"
    assert st.baseline_reproduce_round == {"benchmark-1": 4}
    assert st.agent_failure_round == 0
    assert not st.agent_retry_count


async def test_memory_provider_timeout_does_not_reach_failure_cap(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "memory-provider-timeout"
    state = SessionState(session_id=sid, user_id="u1", current_state="ConsolidatingMemory",
                         agent_failure_round=2, method_revision_round=4)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("agent.failed", sid, "MemoryAgent", payload={
        "agent_name": "MemoryAgent", "reason_classification": "transient",
        "underlying_reason_code": "unexpected_error",
        "underlying_reason": "RuntimeError: APITimeoutError: Request timed out.",
    }))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Suspended"
    assert st.previous_state == "ConsolidatingMemory"
    assert st.agent_failure_round == 2
    assert st.method_revision_round == 4


async def test_invalid_llm_token_is_a_resumable_external_outage(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-invalid-token"
    await _seed(eng, sid, "PlanningExperiment")

    await eng.tick(_Evt(
        "agent.failed", sid, "ExperimentPlannerAgent",
        payload={
            "agent_name": "ExperimentPlannerAgent",
            "reason_classification": "transient",
            "underlying_reason": "AuthenticationError: token:invalid - Invalid token",
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Suspended"
    assert st.previous_state == "PlanningExperiment"
    assert st.agent_failure_round == 0


@pytest.mark.parametrize("prior_failures", [0, 1])
async def test_method_format_failure_retries_frozen_revision_without_human_pause(
        fake_bus, prior_failures):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "method-format-cap"
    state = SessionState(session_id=sid, user_id="u1", current_state="RevisingMethod",
                         method_ref="method://frozen", baseline_result_ref="baseline://frozen",
                         agent_failure_round=prior_failures,
                         active_request={"state": "RevisingMethod", "payload": {
                             "method_ref": "method://frozen",
                             "baseline_result_ref": "baseline://frozen",
                             "experiment_feedback": "Preserve the measured mechanism."}})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload={
        "agent_name": "MethodAgent", "reason_classification": "permanent",
        "underlying_reason_code": "method_parse_error", "underlying_reason": "invalid component threshold"}))
    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed.current_state == "RevisingMethod"
    assert resumed.method_ref == state.method_ref
    assert resumed.baseline_result_ref == state.baseline_result_ref
    assert resumed.agent_failure_round == prior_failures + 1
    assert resumed.awaiting_user_decision_reason is None
    request = fake_bus.published[-1]
    assert request.event_type == "method.revise_requested"
    assert "Preserve the measured mechanism." in request.payload["experiment_feedback"]
    assert "invalid component threshold" in request.payload["experiment_feedback"]


async def test_method_format_failure_cap_reports_frozen_revision_without_human_pause(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "method-format-cap"
    state = SessionState(session_id=sid, user_id="u1", current_state="RevisingMethod",
                         method_ref="method://frozen", baseline_result_ref="baseline://frozen",
                         agent_failure_round=2)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload={
        "agent_name": "MethodAgent", "reason_classification": "permanent",
        "underlying_reason_code": "method_parse_error", "underlying_reason": "invalid component threshold"}))
    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed.current_state == "ConsolidatingMemory"
    assert resumed.previous_state == "RevisingMethod"
    assert resumed.method_ref == state.method_ref
    assert resumed.baseline_result_ref == state.baseline_result_ref
    assert resumed.agent_failure_round == 3
    assert resumed.awaiting_user_decision_reason is None
    request = fake_bus.published[-1]
    assert request.event_type == "memory.consolidation_requested"
    assert request.payload["failure_context"] == {
        "reason": "method_revision_format_exhausted",
        "action": "report_now",
        "method_ref": state.method_ref,
        "detail": "invalid component threshold",
    }


async def test_preconsume_token_quota_failure_is_resumable(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-preconsume-quota"
    await _seed(eng, sid, "ImplementingExperiment")

    await eng.tick(_Evt(
        "agent.failed", sid, "RunnerAgent",
        payload={
            "agent_name": "RunnerAgent",
            "underlying_reason": (
                "PermissionDeniedError: token quota is not enough; "
                "code=pre_consume_token_quota_failed"
            ),
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Suspended"
    assert st.previous_state == "ImplementingExperiment"
    assert st.agent_failure_round == 0


async def test_unrelated_permission_error_keeps_normal_failure_routing(fake_bus):
    """Only an explicit quota marker gets the resumable external-resource path."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-permission"
    await _seed(eng, sid, "ReinterpretingSpec")

    await eng.tick(_Evt(
        "agent.failed",
        sid,
        "ProblemModelingAgent",
        payload={
            "agent_name": "ProblemModelingAgent",
            "reason_classification": "permanent",
            "underlying_reason": "PermissionDeniedError: invalid project access policy",
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "agent_failure"


async def test_component_plan_exhaustion_routes_to_component_revision_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-exhausted"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RunningExperiment",
        confirmed_repo_url="https://example.invalid/repo",
        ablation_failed_component="join-component",
        agent_failure_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    failure = {
        "agent_name": "RunnerAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "component_plan_exhausted",
        "underlying_reason": "component plan has no ready node; exhausted: join-component",
    }
    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload=failure))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "component_ablation_failed"
    assert st.component_ablation_failed is True
    assert st.ablation_failed_component == "join-component"
    assert st.agent_failure_round == 0
    assert st.last_agent_failure == failure
    assert st.awaiting_user_decision_reason is None
    requested = [
        item for item in fake_bus.published
        if item.event_type == "moss.intervention_requested"
    ]
    assert len(requested) == 1


async def test_invalid_component_plan_routes_to_component_revision_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-invalid"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RunningExperiment",
        confirmed_repo_url="https://example.invalid/repo",
        ablation_failed_component="zero-warmup",
        agent_failure_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    failure = {
        "agent_name": "RunnerAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "component_plan_invalid",
        "underlying_reason": "acceptance check assumes an undeclared sibling component",
    }
    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload=failure))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "component_ablation_failed"
    assert st.component_ablation_failed is True
    assert st.ablation_failed_component == "zero-warmup"
    assert st.agent_failure_round == 0
    assert st.last_agent_failure == failure


async def test_integration_checkpoint_conflict_routes_to_dependency_plan_revision(fake_bus, monkeypatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-integration-conflict"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RunningExperiment",
        confirmed_repo_url="https://example.invalid/repo",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "component_execution": {"component_id": "accepted-predecessor", "status": "accepted"},
    })
    failure = {
        "agent_name": "RunnerAgent",
        "reason_classification": "permanent",
        "underlying_reason_code": "component_plan_invalid",
        "underlying_reason": (
            "dependency checkpoint conflict for environment/solve.cu: "
            "repo-17, repo-19 produced different states"
        ),
    }

    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload=failure))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.component_ablation_failed is True
    assert st.ablation_failed_component == "component dependency graph"
    intervention = [p for p in fake_bus.published
                    if p.event_type == "moss.intervention_requested"][-1]
    assert intervention.payload["agent_failure"] == failure
    feedback = eng._method_revision_feedback(st)
    assert "branches marked independent" in feedback
    assert "repo-17, repo-19 produced different states" in feedback


@pytest.mark.parametrize("revision_round", [0, 3, 9])
async def test_exhausted_component_plan_audit_retries_only_component_plan(fake_bus, revision_round):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-plan-revise"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="DeliberatingMethod",
        component_plan_audit_pending=True,
        component_revision_round=revision_round,
        method_ref=f"method://{sid}/v1",
        baseline_result_ref=f"baseline_result://{sid}/v1",
        method_revision_round=0,
        agent_failure_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    failure = {
        "agent_name": "MethodAgent",
        "reason_classification": "transient",
        "underlying_reason_code": "component_plan_audit_failed",
        "underlying_reason": "gate: acceptance check cannot establish the claimed property",
    }
    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload=failure))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == (
        "ConsolidatingMemory" if revision_round == _COMPONENT_REVISION_CAP - 1
        else "DeliberatingMethod")
    assert st.component_plan_audit_pending is (revision_round < _COMPONENT_REVISION_CAP - 1)
    assert st.component_revision_round == revision_round + 1
    assert st.method_revision_round == 0
    assert st.agent_failure_round == 0
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert st.last_agent_failure == failure
    requested = [item for item in fake_bus.published
                 if item.event_type == "method.deliberation_requested"]
    if revision_round == _COMPONENT_REVISION_CAP - 1:
        assert not requested
        closeout = [item for item in fake_bus.published
                    if item.event_type == "memory.consolidation_requested"]
        assert closeout[-1].payload["qualified_component_closeout_requested"] is True
        return
    assert requested
    assert requested[-1].payload["component_plan_audit"] is True
    assert "gate: acceptance check cannot establish" in (
        requested[-1].payload["component_plan_revision_feedback"]
    )


@pytest.mark.parametrize("code", ["component_plan_review_unavailable", "component_plan_format_invalid"])
async def test_unavailable_cap_review_retries_then_consolidates_without_design_budget(
        fake_bus, tmp_path, code):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / "store")
    sid = "cap-review-service-failure"
    method_ref = store.put(kind="method", session_id=sid, key="frozen",
                           obj={"name": "frozen"}, emitter="MethodAgent")
    baseline_ref = store.put(kind="baseline_result", session_id=sid, key="v1",
                             obj={"reproduced": True, "metric_value": 3.0,
                                  "benchmark_id": "bench"},
                             emitter="RunnerAgent")
    benchmark_ref = store.put(kind="benchmark_spec", session_id=sid, key="bench",
                              obj={"id": "bench"}, emitter="EvaluatorAgent")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    state = SessionState(
        session_id=sid, user_id="u1", current_state="DeliberatingMethod",
        mode="benchmark",
        component_plan_audit_pending=True, component_revision_round=3,
        method_ref=method_ref, baseline_result_ref=baseline_ref,
        benchmark_spec_ref=benchmark_ref, current_benchmark_id="bench")
    await save_session_state(fake_bus.db, state, auto_commit=True)
    failure = {
        "agent_name": "MethodAgent", "reason_classification": "transient",
        "underlying_reason_code": code,
        "underlying_reason": "Critic JSON unavailable",
        "component_plan_checkpoint_ref": f"method_debate_report://{sid}/retry-draft",
    }
    for attempt in range(1, 4):
        await eng.tick(_Evt("agent.failed", sid, "MethodAgent",
                            event_id=f"fail-{attempt}", payload=failure))
        st = await load_session_state(fake_bus.db, sid)
        assert st.component_revision_round == 3
        assert st.method_revision_round == 0
        assert st.agent_failure_round == attempt
        assert st.method_ref == state.method_ref
        assert st.current_state == (
            "DeliberatingMethod" if attempt < 3 else "ConsolidatingMemory")
    requests = [p for p in fake_bus.published if p.event_type == "method.deliberation_requested"]
    assert requests[-1].payload["component_plan_checkpoint_ref"] == failure["component_plan_checkpoint_ref"]
    consolidation = [p for p in fake_bus.published
                     if p.event_type == "memory.consolidation_requested"][-1]
    assert consolidation.payload["failure_context"] == {
        "reason": "component_plan_review_format_exhausted",
        "action": "report_now",
        "method_ref": method_ref,
        "detail": "Critic JSON unavailable",
    }
    assert consolidation.payload["qualified_component_closeout_requested"] is True
    assert consolidation.payload["method_ref"] == method_ref
    assert consolidation.payload["baseline_result_ref"] == baseline_ref
    store.close()


async def test_unavailable_cap_review_still_awaits_when_failure_cannot_close(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "cap-review-service-failure-no-baseline"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="DeliberatingMethod",
        component_plan_audit_pending=True, component_revision_round=3,
        method_ref=f"method://{sid}/frozen", agent_failure_round=2)
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload={
        "agent_name": "MethodAgent",
        "underlying_reason_code": "component_plan_review_unavailable",
        "underlying_reason": "Critic JSON unavailable",
    }))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "AwaitingUserDecision"
    assert st.awaiting_user_decision_reason == "agent_failure"


async def test_confirmed_method_contract_conflict_revises_method_and_keeps_baseline(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-method-contract-conflict"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="DeliberatingMethod",
        component_plan_audit_pending=True,
        component_revision_round=2,
        method_ref=f"method://{sid}/v1",
        baseline_result_ref=f"baseline_result://{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    failure = {
        "agent_name": "MethodAgent",
        "underlying_reason_code": "confirmed_method_contract_conflict",
        "underlying_reason": "official task contract prohibits the confirmed algorithm",
        "component_plan_checkpoint_ref": f"method_debate_report://{sid}/contract-repair",
    }

    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload=failure))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "RevisingMethod"
    assert st.component_plan_audit_pending is False
    assert st.method_revision_round == 0
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"
    requested = [item for item in fake_bus.published
                 if item.event_type == "method.revise_requested"]
    assert requested
    assert "official task contract prohibits" in requested[-1].payload["experiment_feedback"]
    assert requested[-1].payload["component_plan_checkpoint_ref"] == failure["component_plan_checkpoint_ref"]


async def test_exhausted_component_plan_revisions_consolidate_before_reporting(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "component-plan-revisions-exhausted"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="DeliberatingMethod",
        mode="benchmark",
        method_ref=f"method://{sid}/v4",
        baseline_result_ref=f"baseline_result://{sid}/v1",
        research_memory_ref=f"research_memory://{sid}/v1",
        component_plan_audit_pending=True,
        component_revision_round=_COMPONENT_REVISION_CAP,
        method_revision_round=0,
        method_swap_round=0,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "agent.failed", sid, "MethodAgent",
        payload={
            "agent_name": "MethodAgent",
            "underlying_reason_code": "component_plan_audit_failed",
            "underlying_reason": "terminal join is not executable",
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "ConsolidatingMemory"
    assert st.method_not_working is False
    assert st.method_swap_round == 0
    assert st.method_revision_round == 0
    assert st.component_plan_audit_pending is False
    requested = [item for item in fake_bus.published
                 if item.event_type == "memory.consolidation_requested"]
    assert len(requested) == 1
    assert requested[0].payload["research_memory_ref"] == state.research_memory_ref
    assert requested[0].payload["qualified_component_closeout_requested"] is True
    assert "terminal join is not executable" in requested[0].payload["failure_context"]["reason"]
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryAgent", event_id="memory-done",
        payload={"research_memory_ref": state.research_memory_ref,
                 "routing_verdict": "report", "diagnosis": "execution_failure"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Reporting"
    assert st.method_swap_round == st.method_revision_round == 0


async def test_exhausted_component_plan_without_memory_consolidates_before_report(
        fake_bus, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "1")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "component-plan-exhausted-no-memory"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="DeliberatingMethod",
        mode="benchmark", method_ref=f"method://{sid}/v1",
        baseline_result_ref=f"baseline_result://{sid}/v1",
        component_plan_audit_pending=True, component_revision_round=_COMPONENT_REVISION_CAP,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt("agent.failed", sid, "MethodAgent", payload={
        "agent_name": "MethodAgent",
        "underlying_reason_code": "component_plan_audit_failed",
        "underlying_reason": "dependency references an undeclared component",
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ConsolidatingMemory"
    requested = fake_bus.published[-1]
    assert requested.event_type == "memory.consolidation_requested"
    assert "undeclared component" in requested.payload["failure_context"]["reason"]
    assert requested.payload["run_record_ref"] is None
    assert requested.payload["baseline_result_ref"] == f"baseline_result://{sid}/v1"
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryAgent", event_id="memory-done",
        payload={"research_memory_ref": f"research_memory://{sid}/v1",
                 "routing_verdict": "report", "diagnosis": "execution_failure"}))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Reporting"
    assert state.method_revision_round == state.method_swap_round == 0
    assert fake_bus.published[-1].event_type == "evidence.citation_audit_requested"


async def test_redrive_stalled_cap_report_builds_missing_memory(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "legacy-cap-report-without-memory"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="Reporting",
        mode="benchmark", component_revision_round=_COMPONENT_REVISION_CAP + 1,
        baseline_result_ref=f"baseline_result://{sid}/v1",
        last_agent_failure={"underlying_reason_code": "component_plan_audit_failed"},
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.resume_to_current_state(sid)
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ConsolidatingMemory"
    assert fake_bus.published[-1].event_type == "memory.consolidation_requested"


async def test_agent_failed_caps_at_three_then_fails_terminal(fake_bus):
    # SAFETY CAP: a report-phase agent that keeps crashing (EvidenceAuditAgent on empty results)
    # must not bounce MossIntervening<->Reporting forever. After 3 failures the engine ends the
    # session as Failed (terminal) instead of looping.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-cap"
    await save_session_state(fake_bus.db, SessionState(
        session_id=sid, user_id="u1", current_state="Reporting",
        research_memory_ref=f"research_memory://{sid}/v1"), auto_commit=True)
    for i in range(2):
        await eng.tick(_Evt("agent.failed", sid, "EvidenceAuditAgent",
                            payload={"agent_name": "EvidenceAuditAgent", "error": "empty evidence"}))
        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "MossIntervening"
        await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
            "reason": "agent_failure", "action": "retry_failed_state",
            "moss_intervention_ref": f"moss://retry-{i}", "degraded": False,
        }))
        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "Reporting"
    # 3rd failure hits the cap -> Failed (terminal), no more bouncing
    await eng.tick(_Evt("agent.failed", sid, "EvidenceAuditAgent",
                        payload={"agent_name": "EvidenceAuditAgent", "error": "empty evidence"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Failed"
    assert st.agent_failure_round >= 3


async def test_runner_failure_cap_reports_prior_evaluated_result(fake_bus):
    """A later infrastructure failure must not erase an already evaluated experiment."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-runner-cap-with-result"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RunningExperiment",
        research_spec_ref=f"research_spec://{sid}/v1",
        research_memory_ref=f"research_memory://{sid}/v1",
        run_record_refs=[f"run_record://{sid}/v1"],
        evaluation_report_refs=[f"evaluation_report://{sid}/v1"],
        last_evaluated_run_record_ref=f"run_record://{sid}/v1",
        agent_failure_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    await eng.tick(_Evt(
        "agent.failed", sid, "RunnerAgent",
        payload={"agent_name": "RunnerAgent", "error": "ssh handshake failed"},
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Reporting"
    assert st.awaiting_user_decision_reason is None
    assert st.agent_failure_round == 0
    assert [item.event_type for item in fake_bus.published[n_before:]] == [
        "evidence.citation_audit_requested"
    ]


# ---------------------------------------------------------------------------
# (b) agent.failed in a DIFFERENT mid-flow state also routes (generic).
# ---------------------------------------------------------------------------


async def test_agent_failed_generic_across_states(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-fail2"
    await _seed(eng, sid, "ImplementingExperiment")

    await eng.tick(_Evt("agent.failed", sid, "CoderAgent",
                        payload={"agent_name": "CoderAgent"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "agent_failure"


# ---------------------------------------------------------------------------
# (b2) On a confirmed-repo run the component plan is load-bearing, while the
#      general Coder patch remains unused by the repo-improve runner.
# ---------------------------------------------------------------------------


async def test_planner_failure_on_confirmed_repo_retries_planning(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-repo-skip"
    # The runner needs the planner to bind work to the active CAP component, so a
    # failed planner call must be retried rather than silently skipped.
    state = SessionState(session_id=sid, user_id="u1", current_state="PlanningExperiment",
                         confirmed_repo_url="https://github.com/owner/baseline",
                         verifier_spec_ref="vspec-ref")
    await save_session_state(eng.client.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    await eng.tick(_Evt("agent.failed", sid, "ExperimentPlannerAgent",
                        payload={"agent_name": "ExperimentPlannerAgent",
                                 "reason_classification": "permanent"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"
    assert st.awaiting_user_decision_reason is None
    assert st.agent_retry_count == {"PlanningExperiment": 1}
    plan_reqs = [p for p in fake_bus.published[n_before:]
                 if p.event_type == "experiment.planning_requested"]
    assert len(plan_reqs) == 1


async def test_coder_failure_on_confirmed_repo_still_skips_to_running(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-repo-coder-skip"
    state = SessionState(session_id=sid, user_id="u1", current_state="ImplementingExperiment",
                         confirmed_repo_url="https://github.com/owner/baseline",
                         verifier_spec_ref="vspec-ref")
    await save_session_state(eng.client.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    await eng.tick(_Evt("agent.failed", sid, "CoderAgent",
                        payload={"agent_name": "CoderAgent",
                                 "reason_classification": "permanent"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "RunningExperiment"
    run_reqs = [p for p in fake_bus.published[n_before:]
                if p.event_type == "experiment.run_requested"]
    assert len(run_reqs) == 1
    assert run_reqs[0].payload["confirmed_repo_url"] == "https://github.com/owner/baseline"
    assert run_reqs[0].payload["experiment_patch_ref"] is None


async def test_planner_failure_without_confirmed_repo_still_pauses(fake_bus):
    # No confirmed repo -> a general self-write run; the planner IS load-bearing, so its failure
    # must still be diagnosed by MOSS before any human pause.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-noconfirm"
    await _seed(eng, sid, "PlanningExperiment")

    await eng.tick(_Evt("agent.failed", sid, "ExperimentPlannerAgent",
                        payload={"agent_name": "ExperimentPlannerAgent"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "agent_failure"


def _replacement_with_baseline(sid: str, current_state: str) -> SessionState:
    return SessionState(
        session_id=sid,
        user_id="u1",
        current_state=current_state,
        method_ref=f"method://{sid}/v2",
        method_swap_round=1,
        research_spec_ref=f"research_spec://{sid}/v1",
        benchmark_spec_ref=f"benchmark_spec://{sid}/v1",
        baseline_result_ref=f"baseline_result://{sid}/v1",
        current_benchmark_id="repo-benchmark",
        confirmed_repo_url="https://example.invalid/owner/repo",
    )


async def test_duplicate_baseline_failure_during_method_swap_reuses_prior_baseline(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-replacement-baseline-failure"
    await save_session_state(
        fake_bus.db, _replacement_with_baseline(sid, "ReproducingBaseline"),
        auto_commit=True,
    )

    n_before = len(fake_bus.published)
    await eng.tick(_Evt(
        "agent.failed", sid, "RunnerAgent",
        payload={"agent_name": "RunnerAgent", "error": "transport handshake failed"},
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "DeliberatingMethod"
    assert st.component_plan_audit_pending is True
    assert st.agent_failure_round == 0
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert [item.event_type for item in fake_bus.published[n_before:]] == [
        "method.deliberation_requested"
    ]


async def test_resume_recovers_method_swap_failed_only_by_duplicate_baseline(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-replacement-baseline-terminal"
    state = _replacement_with_baseline(sid, "Failed")
    state.previous_state = "ReproducingBaseline"
    state.agent_failure_round = 3
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    result = await eng.resume_to_current_state(sid)

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert result["resumed"] is True
    assert result["state"] == "DeliberatingMethod"
    assert st.current_state == "DeliberatingMethod"
    assert st.component_plan_audit_pending is True
    assert st.agent_failure_round == 0
    assert [item.event_type for item in fake_bus.published[n_before:]] == [
        "method.deliberation_requested"
    ]


async def test_resume_recovers_failed_runner_when_prior_result_is_reportable(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-runner-terminal-with-result"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="Failed",
        previous_state="RunningExperiment",
        research_spec_ref=f"research_spec://{sid}/v1",
        research_memory_ref=f"research_memory://{sid}/v1",
        run_record_refs=[f"run_record://{sid}/v1"],
        evaluation_report_refs=[f"evaluation_report://{sid}/v1"],
        last_evaluated_run_record_ref=f"run_record://{sid}/v1",
        agent_failure_round=3,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    result = await eng.resume_to_current_state(sid)

    st = await load_session_state(fake_bus.db, sid)
    assert result["resumed"] is True
    assert st.current_state == "Reporting"
    assert st.agent_failure_round == 0
    assert [item.event_type for item in fake_bus.published[n_before:]] == [
        "evidence.citation_audit_requested"
    ]


@pytest.mark.parametrize("suspended", [False, True])
@pytest.mark.parametrize("result_method,result_baseline,allowed", [
    ("method://previous", "baseline://frozen", True),
    ("method://current", "baseline://frozen", False),
    (None, "baseline://frozen", False),
    ("method://previous", "baseline://other", False),
])
async def test_plan_exhaustion_does_not_transfer_across_method_versions(
        fake_bus, monkeypatch, tmp_path, request, result_method, result_baseline, allowed, suspended):
    from autoresearch.spec_store import SpecStore

    sid = "sess-new-plan-old-failure"
    store = SpecStore(tmp_path / "store")
    request.addfinalizer(store.close)
    method_ref = store.snapshot(store.put(kind="method", session_id=sid, key="current",
        obj={"approach": "current mechanism"}, emitter="MethodAgent"))
    baseline_ref = store.snapshot(store.put(kind="baseline_result", session_id=sid, key="frozen",
        obj={"reproduced": True, "metric_value": 1.0, "benchmark_id": "bench"}, emitter="RunnerAgent"))
    benchmark_ref = store.put(kind="benchmark_spec", session_id=sid, key="bench",
        obj={"id": "bench"}, emitter="EvaluatorAgent")
    if result_method == "method://current":
        result_method = method_ref
    if result_baseline == "baseline://frozen":
        result_baseline = baseline_ref
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    original_publish = fake_bus.publish

    async def publish_bound_request(*, event_id=None, **kwargs):
        return await original_publish(**kwargs)

    monkeypatch.setattr(fake_bus, "publish", publish_bound_request)
    monkeypatch.setattr(eng, "_current_gen_result", lambda sid: {
        "method_ref": result_method, "baseline_result_ref": result_baseline,
        "component_execution": {"status": "exhausted"},
    })
    state = SessionState(
        session_id=sid, user_id="u1", mode="benchmark",
        current_state="DeliberatingMethod", method_ref=method_ref,
        baseline_result_ref=baseline_ref, component_plan_audit_pending=True,
        benchmark_spec_ref=benchmark_ref, current_benchmark_id="bench",
        component_ablation_failed=True,
        ablation_failed_component="component contract rejected during experiment-plan review",
    )
    saved_payload = None
    if suspended:
        saved_payload = eng._entry_actions["DeliberatingMethod"][1](state, None)
        saved_payload["component_plan_allow_unchanged"] = False
        saved_payload["saved_scientific_input"] = "preserve this exact input"
        state.current_state = "Suspended"
        state.previous_state = "DeliberatingMethod"
        state.request_binding_required = True
        state.active_request = {
            "state": "DeliberatingMethod", "event_type": "method.deliberation_requested",
            "event_id": "old-cap-request", "payload": saved_payload, "consumed": True,
        }
    await save_session_state(fake_bus.db, state, auto_commit=True)
    n_before = len(fake_bus.published)
    await eng.resume_to_current_state(state.session_id)
    requests = fake_bus.published[n_before:]
    assert [item.event_type for item in requests] == ["method.deliberation_requested"]
    assert requests[0].payload["component_plan_allow_unchanged"] is allowed
    assert requests[0].payload["method_ref"] == state.method_ref
    assert requests[0].payload["baseline_result_ref"] == state.baseline_result_ref
    assert requests[0].payload["component_plan_audit"] is True
    if suspended:
        assert {k: v for k, v in requests[0].payload.items()
                if k != "component_plan_allow_unchanged"} == {
            k: v for k, v in saved_payload.items() if k != "component_plan_allow_unchanged"}


@pytest.mark.parametrize("mismatch", [None, "run_record_ref", "evaluation_report_ref", "baseline_result_ref", "unconsumed"])
async def test_resume_failed_consolidation_requires_saved_evaluation_binding(fake_bus, mismatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-memory-terminal-with-evaluation"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="Failed",
        previous_state="ConsolidatingMemory", agent_failure_round=3,
        baseline_result_ref=f"baseline_result://{sid}/v1",
        research_memory_ref=f"research_memory://{sid}/v1",
        run_record_refs=[f"run_record://{sid}/v1"],
        evaluation_report_refs=[f"evaluation_report://{sid}/v1"],
        last_evaluated_run_record_ref=f"run_record://{sid}/v1",
    )
    payload = eng._memory_consolidation_payload(state)
    if mismatch and mismatch != "unconsumed":
        payload[mismatch] = "stale-reference"
    state.active_request = {
        "state": "ConsolidatingMemory", "event_type": "memory.consolidation_requested",
        "payload": payload, "consumed": mismatch != "unconsumed",
    }
    await save_session_state(fake_bus.db, state, auto_commit=True)
    n_before = len(fake_bus.published)

    result = await eng.resume_to_current_state(sid)
    restored = await load_session_state(fake_bus.db, sid)
    assert restored.baseline_result_ref == state.baseline_result_ref
    assert restored.run_record_refs == state.run_record_refs
    assert restored.evaluation_report_refs == state.evaluation_report_refs
    if mismatch:
        assert result["resumed"] is False
        assert restored.current_state == "Failed"
        assert len(fake_bus.published) == n_before
    else:
        assert result["resumed"] is True
        assert restored.current_state == "ConsolidatingMemory"
        assert restored.agent_failure_round == 0
        events = fake_bus.published[n_before:]
        assert [item.event_type for item in events] == ["memory.consolidation_requested"]
        assert events[0].payload == payload


async def test_resume_retries_failed_report_generation_from_existing_memory(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-report-terminal-with-memory"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="Failed",
        previous_state="Reporting",
        research_memory_ref=f"research_memory://{sid}/v1",
        agent_failure_round=3,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    result = await eng.resume_to_current_state(sid)

    st = await load_session_state(fake_bus.db, sid)
    assert result["resumed"] is True
    assert st.current_state == "Reporting"
    assert st.agent_failure_round == 0
    assert [item.event_type for item in fake_bus.published[n_before:]] == [
        "evidence.citation_audit_requested"
    ]


# ---------------------------------------------------------------------------
# (c) agent.failed in a TERMINAL state -> no-op (short-circuited).
# ---------------------------------------------------------------------------


async def test_agent_failed_in_terminal_state_is_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-done"
    await _seed(eng, sid, "Done")

    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent",
                        payload={"agent_name": "RunnerAgent"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Done"
    assert st.awaiting_user_decision_reason is None


# ---------------------------------------------------------------------------
# (d) a malformed agent.failed (no payload) never crashes; still routes.
# ---------------------------------------------------------------------------


async def test_agent_failed_missing_payload_is_defensive(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-fail3"
    await _seed(eng, sid, "EvaluatingExperiment")

    # payload=None mirrors a broker envelope with no payload
    await eng.tick(_Evt("agent.failed", sid, "EvaluatorAgent", payload=None))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "agent_failure"


# ---------------------------------------------------------------------------
# (e) a normal (non-agent.failed) event does NOT set agent_failure.
# ---------------------------------------------------------------------------


async def test_normal_event_unaffected(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-normal"
    await _seed(eng, sid, "RunningExperiment")

    await eng.tick(_Evt("some.unrelated_event", sid, "SomeAgent"))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "RunningExperiment"
    assert st.awaiting_user_decision_reason is None


# ---------------------------------------------------------------------------
# (f) MOSS retry resolves {failed_state} -> the original failed state.
# ---------------------------------------------------------------------------


async def test_agent_failure_moss_retry_reenters_failed_state(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-retry"
    # Seed the refs RunningExperiment's entry action re-emits on retry.
    state = SessionState(session_id=sid, user_id="u1", current_state="RunningExperiment",
                         experiment_patch_refs=["patch-ref"], verifier_spec_ref="vspec-ref")
    await save_session_state(eng.client.db, state, auto_commit=True)

    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload={"agent_name": "RunnerAgent"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "MossIntervening"
    assert st.previous_state == "RunningExperiment"

    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "agent_failure", "action": "retry_failed_state",
        "moss_intervention_ref": "moss://agent-retry", "degraded": False,
    }))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "RunningExperiment"      # {failed_state} resolved
    assert st.current_moss_reason is None
    assert st.awaiting_user_decision_reason is None


async def test_agent_failure_moss_escalation_preserves_original_retry_target(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-moss-escalation"
    state = SessionState(session_id=sid, user_id="u1", current_state="RunningExperiment",
                         experiment_patch_refs=["patch-ref"], verifier_spec_ref="vspec-ref")
    await save_session_state(eng.client.db, state, auto_commit=True)

    await eng.tick(_Evt("agent.failed", sid, "RunnerAgent", payload={"agent_name": "RunnerAgent"}))
    await eng.tick(_Evt("moss.intervention_completed", sid, "MossAgent", payload={
        "reason": "agent_failure", "action": "escalate_to_user",
        "moss_intervention_ref": "moss://agent-escalate", "degraded": False,
    }))
    paused = await load_session_state(fake_bus.db, sid)
    assert paused.current_state == "AwaitingUserDecision"
    assert paused.awaiting_user_decision_reason == "agent_failure"
    assert paused.previous_state == "MossIntervening"
    assert paused.last_agent_failure["failed_state"] == "RunningExperiment"

    await eng.tick(_Evt("user.decision_provided", sid, "Human",
                        payload={"decision": "retry_session"}))
    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed.current_state == "RunningExperiment"



async def test_agent_failure_transition_error_propagates(fake_bus, monkeypatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    await _seed(eng, "transition-error", "RunningExperiment")
    async def fail(*args):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(eng, "_handle_agent_failed", fail)
    with pytest.raises(RuntimeError, match="locked"):
        await eng.tick(_Evt("agent.failed", "transition-error", "RunnerAgent"))
    state = await load_session_state(fake_bus.db, "transition-error")
    assert state.current_state == "RunningExperiment"


async def test_review_failure_suspends_at_original_state(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    await _seed(eng, "review-recovery", "RunningExperiment")
    await eng.tick(_Evt("agent.failed", "review-recovery", "RunnerAgent", payload={
        "agent_name": "RunnerAgent", "reason_classification": "transient",
        "underlying_reason_code": "implementation_review_unavailable", "underlying_reason": "invalid JSON"}))
    state = await load_session_state(fake_bus.db, "review-recovery")
    assert state.current_state == "Suspended"
    assert state.previous_state == "RunningExperiment"


async def test_bound_review_retry_exhaustion_pauses_without_failing_candidate(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "bound-review-recovery"
    threshold, _ = eng._counter_threshold("agent_retry_count")
    state = SessionState(session_id=sid, user_id="u1", current_state="RunningExperiment",
        request_binding_required=True, agent_failure_round=3,
        method_ref="method://frozen", baseline_result_ref="baseline://frozen",
        agent_retry_count={"RunningExperiment": threshold - 1})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng._handle_agent_failed(state, _Evt("agent.failed", sid, "RunnerAgent", payload={
        "agent_name": "RunnerAgent", "reason_classification": "transient",
        "underlying_reason_code": "implementation_review_unavailable",
        "underlying_reason": "component names do not match",
    }))
    stored = await load_session_state(fake_bus.db, sid)
    assert stored.current_state == "Suspended"
    assert stored.previous_state == "RunningExperiment"
    assert stored.agent_failure_round == 3
    assert stored.method_ref == "method://frozen"
    assert stored.baseline_result_ref == "baseline://frozen"
