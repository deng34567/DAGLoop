"""Unit tests for the minimal M0 OrchestratorEngine skeleton.

Scope (Task 7 M0): subscribe, atomic start_session (create SessionState +
emit the ClarifyingResearchGoal entry action problem_modeling.requested in ONE
commit), atomic _transition, suspend/resume via _transition + resume_to_current_state,
and the pure classify_resumability. Counter / MOSS / full tick routing are DEFERRED.

The engine reads its sqlite handle from ``client.db`` and publishes via
``client.publish(...)``. Here we drive it with a FakeBusClient whose ``.db`` is a
REAL aiosqlite connection (so state persistence + atomic commit are exercised for
real) but whose ``.publish`` merely captures the outbound event (no broker /
schema validation / Redis).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import pytest

from autoresearch.orchestrator.engine import (
    OrchestratorEngine,
    classify_resumability,
)
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


async def test_explicit_campaign_contract_is_persisted_without_aliasing_input(fake_bus):
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await engine.start()
    boundary = {"execution_contract": {"evaluation": {"required_values": {"/split": "dev"}}}}
    sid = await engine.start_session("Open implementation under a fixed evaluator", confirmed_boundary=boundary)
    boundary["execution_contract"]["evaluation"]["required_values"]["/split"] = "test"
    state = await load_session_state(fake_bus.db, sid)
    assert state.confirmed_boundary["execution_contract"]["evaluation"]["required_values"]["/split"] == "dev"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]
    emitter: str = "ResearchOrchestrator"


class _CountingConn:
    """Thin proxy over a real aiosqlite connection that counts commits.

    Lets a test assert that start_session / _transition issue exactly ONE
    commit (the save + emit atomicity invariant, INV-4) while still exercising
    real SQLite persistence for everything else.
    """

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn
        self.commit_count = 0

    async def commit(self) -> None:
        self.commit_count += 1
        await self._conn.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class FakeBusClient:
    """Minimal EventBusClient stand-in mirroring the real publish/subscribe/.db surface."""

    def __init__(self, conn: aiosqlite.Connection):
        self.db = _CountingConn(conn)
        self.emitter_name = "ResearchOrchestrator"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        self.published.append(
            _Published(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                causation_id=causation_id,
            )
        )
        if auto_commit:
            await self.db.commit()
        return f"evt-{len(self.published)}"


@pytest.fixture
async def fake_bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "engine.db"))
    try:
        yield FakeBusClient(conn)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_start_subscribes_wildcard(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    assert any(pattern == "*" and handler == eng.tick for pattern, handler in fake_bus.subscriptions)


async def test_start_session_emits_first_event_atomically(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()

    commits_before = fake_bus.db.commit_count
    sid = await eng.start_session("优化 modded-nanogpt speedrun", user_id="u1")

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ClarifyingResearchGoal"
    assert st.user_id == "u1"
    assert st.research_goal_text == "优化 modded-nanogpt speedrun"

    emitted = [p for p in fake_bus.published if p.event_type == "problem_modeling.requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {"research_goal_text": "优化 modded-nanogpt speedrun"}

    # Atomicity: exactly ONE commit bundles the SessionState save + the emit.
    assert fake_bus.db.commit_count - commits_before == 1


def test_classify_resumability():
    assert classify_resumability("Done") == "terminal"
    assert classify_resumability("Failed") == "terminal"
    assert classify_resumability("AwaitingUserDecision") == "awaiting_user"
    assert classify_resumability("Suspended") == "awaiting_user"
    assert classify_resumability("ImplementingExperiment") == "resumable"
    assert classify_resumability("ClarifyingResearchGoal") == "resumable"


async def test_transition_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    state = await load_session_state(fake_bus.db, sid)
    commits_before = fake_bus.db.commit_count
    await eng._transition(state, "PlanningExperiment")

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"
    assert st.previous_state == "ClarifyingResearchGoal"
    # No entry action wired for PlanningExperiment (M0) => still exactly one commit.
    assert fake_bus.db.commit_count - commits_before == 1


async def test_suspend_then_resume_restores_pre_suspend_state(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    # Enter Suspended via _transition (there is no literal suspend() method).
    state = await load_session_state(fake_bus.db, sid)
    await eng._transition(state, "Suspended")

    suspended = await load_session_state(fake_bus.db, sid)
    assert suspended.current_state == "Suspended"
    assert suspended.previous_state == "ClarifyingResearchGoal"

    n_before = len(fake_bus.published)
    res = await eng.resume_to_current_state(sid)

    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed.current_state == "ClarifyingResearchGoal"
    assert res["resumed"] is True
    # Entry action re-emitted (idempotent re-drive of the pre-suspend state).
    new_events = fake_bus.published[n_before:]
    resumed_requests = [
        p for p in new_events if p.event_type == "problem_modeling.requested"
    ]
    assert len(resumed_requests) == 1
    assert resumed_requests[0].payload == {"research_goal_text": "goal"}

@pytest.mark.parametrize("dispatch_fails", [False, True])
async def test_resume_is_visible_before_entry_dispatch(fake_bus, tmp_path, monkeypatch, dispatch_fails):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = await load_session_state(fake_bus.db, sid)
    await eng._transition(state, "Suspended")

    async def dispatch(restored, **kwargs):
        # The Web reads through another connection while synchronous handlers
        # may spend minutes in an LLM call.
        async with aiosqlite.connect(str(tmp_path / "engine.db")) as reader:
            visible = await load_session_state(reader, sid)
        assert visible.current_state == "ClarifyingResearchGoal"
        assert visible.previous_state is None
        if dispatch_fails:
            raise RuntimeError("dispatch interrupted")

    monkeypatch.setattr(eng, "_execute_entry_action", dispatch)
    if dispatch_fails:
        with pytest.raises(RuntimeError, match="dispatch interrupted"):
            await eng.resume_to_current_state(sid)
    else:
        await eng.resume_to_current_state(sid)
    # A failed dispatch remains an idle active state, recoverable via redrive.
    assert (await load_session_state(fake_bus.db, sid)).current_state == "ClarifyingResearchGoal"


@pytest.mark.parametrize("pending_matches", [True, False])
@pytest.mark.parametrize("blocked_at_reporting", [False, True])
async def test_failed_review_resume_requires_pending_evidence_bound_to_frozen_inputs(
        fake_bus, tmp_path, monkeypatch, pending_matches, blocked_at_reporting):
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    store = SpecStore(tmp_path / "store")
    try:
        sid = "failed-review-recovery"
        baseline = {"metric_value": 1.0}
        baseline_ref = store.put(kind="baseline_result", session_id=sid, key="v1",
            obj=baseline, emitter="RunnerAgent")
        input_ref = store.put(kind="implementation_review_input", session_id=sid, key="v1",
            obj={"result": {"checkpoint": "retained"}}, emitter="RunnerAgent")
        store.put(kind="pending_implementation_review", session_id=sid, key="v1", obj={
            "status": "pending", "method_ref": "method://frozen" if pending_matches else "method://other",
            "baseline_digest": artifact_digest(baseline), "input_ref": input_ref}, emitter="RunnerAgent")
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        failure = {"underlying_reason_code": (
            "completion_incomplete" if blocked_at_reporting else "implementation_review_unavailable")}
        original_state = "AwaitingUserDecision" if blocked_at_reporting else "Failed"
        state = SessionState(session_id=sid, user_id="u1", current_state=original_state,
            previous_state="Reporting" if blocked_at_reporting else "RunningExperiment", method_ref="method://frozen",
            baseline_result_ref=baseline_ref, last_agent_failure=failure,
            agent_failure_round=4, agent_retry_count={"RunningExperiment": 3})
        await save_session_state(fake_bus.db, state, auto_commit=True)
        entries = []
        async def dispatch(current, **kwargs): entries.append(current.current_state)
        monkeypatch.setattr(eng, "_execute_entry_action", dispatch)

        result = await eng.resume_to_current_state(sid)
        resumed = await load_session_state(fake_bus.db, sid)
        assert result["resumed"] is (pending_matches or blocked_at_reporting)
        assert resumed.current_state == ("RunningExperiment" if pending_matches else original_state)
        assert entries == ([resumed.current_state] if pending_matches or blocked_at_reporting else [])
        assert resumed.method_ref == state.method_ref
        assert resumed.baseline_result_ref == baseline_ref
        assert resumed.last_agent_failure == failure
    finally:
        store.close()


@pytest.mark.parametrize("mismatch", [None, "method", "baseline", "candidate_error", "missing_snapshot"])
@pytest.mark.parametrize("previous_state,error", [
    ("Reporting", "Allocated quota exceeded"),
    ("ConsolidatingMemory", "Conversation error code: BadGatewayError"),
])
async def test_legacy_provider_failure_resume_preserves_frozen_execution(
        fake_bus, tmp_path, monkeypatch, mismatch, previous_state, error):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / "store")
    try:
        sid = "legacy-provider-recovery"
        run_ref = store.snapshot(store.put(kind="run_record", session_id=sid, key="v1",
            obj={"succeeded": False, "metric_value": None}, emitter="RunnerAgent"))
        failed_result = {"verdict": "invalid", "error": error,
                         "method_ref": "method://frozen", "baseline_result_ref": "baseline://frozen",
                         "run_record_snapshot_ref": run_ref}
        if mismatch == "method": failed_result["method_ref"] = "method://other"
        if mismatch == "baseline": failed_result["baseline_result_ref"] = "baseline://other"
        if mismatch == "candidate_error": failed_result["error"] = "candidate compile failed"
        if mismatch != "missing_snapshot":
            store.snapshot(store.put(kind="gen_experiment_result", session_id=sid, key="current",
                obj=failed_result, emitter="RunnerAgent"))
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        state = SessionState(session_id=sid, user_id="u1", current_state="AwaitingUserDecision",
            previous_state=previous_state, method_ref="method://frozen",
            baseline_result_ref="baseline://frozen", run_record_refs=[run_ref],
            awaiting_user_decision_reason="agent_failure")
        await save_session_state(fake_bus.db, state, auto_commit=True)
        entries = []
        async def dispatch(current, **kwargs): entries.append(current.current_state)
        monkeypatch.setattr(eng, "_execute_entry_action", dispatch)
        await eng.resume_to_current_state(sid)
        resumed = await load_session_state(fake_bus.db, sid)
        assert resumed.current_state == ("RunningExperiment" if mismatch is None else "AwaitingUserDecision")
        assert resumed.method_ref == state.method_ref
        assert resumed.baseline_result_ref == state.baseline_result_ref
        assert resumed.run_record_refs == [run_ref]
        assert ("RunningExperiment" in entries) is (mismatch is None)
    finally:
        store.close()


@pytest.mark.parametrize("current_state,reason", [
    ("Failed", None), ("AwaitingUserDecision", "agent_failure"),
    ("AwaitingUserDecision", "user_paused"),
])
@pytest.mark.parametrize("mismatch", [None, "method", "baseline"])
async def test_failed_method_format_resume_requires_original_frozen_request(
        fake_bus, monkeypatch, current_state, reason, mismatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    state = SessionState(session_id="method-format-resume", user_id="u1", current_state=current_state,
        previous_state="RevisingMethod", method_ref="method://frozen", baseline_result_ref="baseline://frozen",
        awaiting_user_decision_reason=reason,
        agent_failure_round=4, last_agent_failure={"underlying_reason_code": "method_parse_error"},
        active_request={"state": "RevisingMethod", "payload": {
            "method_ref": "method://other" if mismatch == "method" else "method://frozen",
            "baseline_result_ref": "baseline://other" if mismatch == "baseline" else "baseline://frozen"}})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    entries = []
    async def dispatch(current, **kwargs): entries.append(current.current_state)
    monkeypatch.setattr(eng, "_execute_entry_action", dispatch)
    result = await eng.resume_to_current_state(state.session_id)
    resumed = await load_session_state(fake_bus.db, state.session_id)
    matches = mismatch is None and reason != "user_paused"
    if matches:
        assert result["resumed"] is True
    assert resumed.current_state == ("RevisingMethod" if matches else current_state)
    assert ("RevisingMethod" in entries) is matches
    assert resumed.method_ref == state.method_ref
    assert resumed.baseline_result_ref == state.baseline_result_ref
    assert resumed.last_agent_failure == state.last_agent_failure
    if matches:
        assert resumed.awaiting_user_decision_reason is None


async def test_method_format_repair_preserves_original_scientific_feedback(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    science = "The measured feature set failed the frozen 0.95 accuracy gate. Revise its mechanism."
    error = "ablation.functional_check must expose one aggregate scalar threshold"
    state = SessionState(session_id="method-format-feedback", user_id="u1",
        last_agent_failure={"underlying_reason_code": "method_parse_error", "underlying_reason": error},
        active_request={"payload": {"experiment_feedback": science}})
    feedback = eng._method_revision_feedback(state)
    assert feedback.startswith(science)
    assert error in feedback
    assert "Do not discard checks or weaken thresholds" in feedback


async def test_tick_skips_self_emitted_and_unknown_pairs(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    @dataclass
    class _Evt:
        event_type: str
        session_id: str
        emitter: str

    # Self-emitted event: ignored (no crash, no state change).
    await eng.tick(_Evt("problem_modeling.requested", sid, "ResearchOrchestrator"))
    # Unknown (state, event) pair from another emitter: no-op (no handler).
    await eng.tick(_Evt("some.random_event", sid, "ProblemModelingAgent"))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "ClarifyingResearchGoal"
