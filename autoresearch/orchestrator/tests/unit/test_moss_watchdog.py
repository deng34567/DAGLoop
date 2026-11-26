"""Unit tests for the MOSS watchdog decision logic + the check_once wiring.

Pure decision helpers (decide_recovery / conservative_action / timeout_for) are
tested directly; the thread runner is exercised via check_once with fake
read/inject callables (no threads, no sqlite, no docker, no bus).
"""
from __future__ import annotations

from autoresearch.orchestrator.moss_watchdog import (
    DEFAULT_TIMEOUT,
    DEFAULT_TIMEOUTS,
    KIND_AGENT_FAILED,
    KIND_DECISION,
    KIND_FINALIZE_CLARIFY,
    KIND_IDEA_NO_INJECT,
    KIND_METHOD_NO_INJECT,
    KIND_RESUME_SUSPENDED,
    MossWatchdog,
    conservative_action,
    decide_recovery,
    timeout_for,
)


# ---- timeout_for ----

def test_timeout_for_uses_override_then_default():
    assert timeout_for("AwaitingUserDecision", DEFAULT_TIMEOUTS, DEFAULT_TIMEOUT) == 300.0
    assert timeout_for("Suspended", DEFAULT_TIMEOUTS, DEFAULT_TIMEOUT) == 300.0
    assert timeout_for("SomeOtherState", DEFAULT_TIMEOUTS, DEFAULT_TIMEOUT) == DEFAULT_TIMEOUT


def test_baseline_watchdog_allows_full_agentic_hard_cap():
    assert timeout_for("ReproducingBaseline", DEFAULT_TIMEOUTS, DEFAULT_TIMEOUT) == 14400.0


# ---- conservative_action ----

def test_conservative_prefers_report_now():
    assert conservative_action(["retry_session", "report_now", "abandon"]) == "report_now"


def test_conservative_prefers_accept_over_retry():
    assert conservative_action(["strict_rerun", "accept_with_caveat"]) == "accept_with_caveat"


def test_conservative_avoids_abandon_when_alternative_exists():
    # deepen_literature is not in the priority list, but it beats abandon.
    assert conservative_action(["deepen_literature", "abandon"]) == "deepen_literature"


def test_conservative_uses_abandon_only_as_last_resort():
    assert conservative_action(["abandon"]) == "abandon"


def test_conservative_none_when_no_actions():
    assert conservative_action([]) is None


# ---- decide_recovery ----

def _decide(state, ago, busy=False):
    return decide_recovery(
        state, ago, busy=busy, timeouts=DEFAULT_TIMEOUTS, default_timeout=DEFAULT_TIMEOUT
    )


def test_no_recovery_before_timeout():
    assert _decide("AwaitingUserDecision", 10.0) is None


def test_no_recovery_for_terminal():
    assert _decide("Done", 10_000.0) is None
    assert _decide("Failed", 10_000.0) is None


def test_no_recovery_when_busy_even_if_long():
    # A live experiment container -> progressing, never reclaimed.
    assert _decide("ImplementingExperiment", 10_000.0, busy=True) is None


def test_agent_work_stall_yields_agent_failed():
    # Not busy + past a generic state's default timeout.
    assert _decide("DesigningExperiment", DEFAULT_TIMEOUT + 1) == KIND_AGENT_FAILED


def test_running_experiment_stall_when_not_busy():
    # Past the generous ceiling AND no live container -> reclaimed as agent-work.
    assert _decide("RunningExperiment", 3600.0 + 1) == KIND_AGENT_FAILED
    assert _decide("RunningExperiment", 3600.0 - 1) is None


def test_awaiting_user_decision_stall_yields_decision():
    assert _decide("AwaitingUserDecision", 301.0) == KIND_DECISION


def test_suspended_provider_pause_resumes_after_existing_timeout():
    suspended_timeout = DEFAULT_TIMEOUTS["Suspended"]
    assert _decide("Suspended", suspended_timeout - 1) is None
    assert _decide("Suspended", suspended_timeout + 1) == KIND_RESUME_SUSPENDED


# ---- whole-session budget (livelock cover) ----

def _decide_budget(state, entered_ago, session_ago, session_timeout=1800.0, busy=False):
    return decide_recovery(
        state, entered_ago, busy=busy, timeouts=DEFAULT_TIMEOUTS,
        default_timeout=DEFAULT_TIMEOUT, session_ago=session_ago, session_timeout=session_timeout,
    )


def test_session_budget_forces_wrapup_despite_fresh_state():
    # A livelock: state just changed (entered_ago tiny, per-state timer would NOT
    # fire) but the whole session is past budget -> wrap up.
    assert _decide_budget("MossIntervening", entered_ago=5.0, session_ago=1801.0) == KIND_AGENT_FAILED
    assert _decide_budget("PlanningExperiment", entered_ago=5.0, session_ago=1801.0) == KIND_AGENT_FAILED


def test_session_budget_not_fired_before_budget():
    assert _decide_budget("PlanningExperiment", entered_ago=5.0, session_ago=100.0) is None


def test_session_budget_suppressed_while_busy():
    # A long experiment past the budget must NOT be cut off.
    assert _decide_budget("ImplementingExperiment", entered_ago=5.0, session_ago=9999.0, busy=True) is None


def test_session_budget_decision_state_still_reports():
    assert _decide_budget("AwaitingUserDecision", entered_ago=5.0, session_ago=1801.0) == KIND_DECISION


def test_session_budget_does_not_immediately_resume_fresh_suspension():
    # The failure event that persisted Suspended may still be finishing its
    # delivery cascade.  An expired whole-session budget must not race it.
    assert _decide_budget("Suspended", entered_ago=5.0, session_ago=9999.0) is None
    assert (
        _decide_budget(
            "Suspended", entered_ago=DEFAULT_TIMEOUTS["Suspended"] + 1,
            session_ago=9999.0,
        )
        == KIND_RESUME_SUSPENDED
    )


def test_session_budget_never_cuts_finishing_states():
    # Past budget but in the reporting home-stretch -> let it finish (per-state
    # timer is fresh, so no recovery).
    for s in ("EvaluatingExperiment", "ConsolidatingMemory", "Reporting", "VerifyingBestResult"):
        assert _decide_budget(s, entered_ago=5.0, session_ago=9999.0) is None


def test_finishing_state_genuine_stall_still_recovered():
    # A finishing state that truly hangs past its per-state timeout is still
    # reclaimed via the stall path (not the budget path).
    assert _decide("Reporting", DEFAULT_TIMEOUT + 1) == KIND_AGENT_FAILED


def test_clarify_and_checkpoint_stalls():
    assert _decide("ClarifyingResearchGoal", 481.0) == KIND_FINALIZE_CLARIFY
    assert _decide("HumanIdeaCheckpoint", 481.0) == KIND_IDEA_NO_INJECT
    assert _decide("ConfirmingMethod", DEFAULT_TIMEOUT + 1) == KIND_METHOD_NO_INJECT


# ---- MossWatchdog.check_once wiring ----

class _Recorder:
    def __init__(self, state, *, entered_at=0.0, reason=None, busy=False,
                 legal=(), qid=None):
        self._st = {
            "session_id": "sess-1",
            "current_state": state,
            "state_entered_at": entered_at,
            "awaiting_user_decision_reason": reason,
        }
        self._busy = busy
        self._legal = list(legal)
        self._qid = qid
        self.injected: list[dict] = []

    def read_state(self):
        return dict(self._st)

    def is_busy(self, state):
        return self._busy

    def legal_actions_for(self, reason):
        return self._legal

    def latest_qid(self):
        return self._qid

    def inject(self, **kw):
        self.injected.append(kw)


def _watchdog(rec, now=10_000.0):
    return MossWatchdog(
        read_state=rec.read_state,
        legal_actions_for=rec.legal_actions_for,
        latest_qid=rec.latest_qid,
        is_busy=rec.is_busy,
        inject=rec.inject,
        clock=lambda: now,
        timeouts=DEFAULT_TIMEOUTS,
        default_timeout=DEFAULT_TIMEOUT,
    )


def test_check_once_injects_agent_failed_for_dead_state():
    rec = _Recorder("DesigningExperiment", entered_at=0.0, busy=False)
    wd = _watchdog(rec)
    assert wd.check_once() == KIND_AGENT_FAILED
    assert rec.injected == [
        {"kind": KIND_AGENT_FAILED, "session_id": "sess-1", "reason": None, "action": None, "qid": None}
    ]


def test_check_once_busy_experiment_not_touched():
    rec = _Recorder("ImplementingExperiment", entered_at=0.0, busy=True)
    wd = _watchdog(rec)
    assert wd.check_once() is None
    assert rec.injected == []


def test_check_once_resumes_suspended_session():
    rec = _Recorder("Suspended", entered_at=0.0, busy=False)
    wd = _watchdog(rec)
    assert wd.check_once() == KIND_RESUME_SUSPENDED
    assert rec.injected == [{
        "kind": KIND_RESUME_SUSPENDED, "session_id": "sess-1", "reason": None,
        "action": None, "qid": None,
    }]


def test_agent_failure_retries_once_then_reports():
    # transient agent_failure: retry_session ONCE, then conservative report_now.
    rec = _Recorder("AwaitingUserDecision", entered_at=0.0, reason="agent_failure",
                    legal=["retry_session", "report_now", "abandon"])
    wd = _watchdog(rec)
    assert wd.check_once() == KIND_DECISION
    assert rec.injected[-1]["action"] == "retry_session"   # first: retry
    # the retry re-failed -> agent_failure recurs (new entered_at) -> report_now
    rec._st["state_entered_at"] = 5000.0
    assert wd.check_once() == KIND_DECISION
    assert rec.injected[-1]["action"] == "report_now"       # then: salvage a report


def test_non_agent_failure_decision_is_conservative_directly():
    # a non-agent_failure reason (no transient retry) -> conservative action now.
    rec = _Recorder("AwaitingUserDecision", entered_at=0.0, reason="literature_uncertain",
                    legal=["deepen_literature", "continue_with_caveat", "abandon"])
    wd = _watchdog(rec)
    assert wd.check_once() == KIND_DECISION
    assert rec.injected[-1]["action"] == "continue_with_caveat"


def test_check_once_debounces_same_state_entered_at():
    rec = _Recorder("DesigningExperiment", entered_at=0.0, busy=False)
    wd = _watchdog(rec)
    assert wd.check_once() == KIND_AGENT_FAILED
    # Second poll on the SAME (state, entered_at) must not re-inject.
    assert wd.check_once() is None
    assert len(rec.injected) == 1


def test_check_once_clarify_skips_without_qid_then_fires_with_one():
    rec = _Recorder("ClarifyingResearchGoal", entered_at=0.0, qid=None)
    wd = _watchdog(rec)
    assert wd.check_once() is None  # no open question yet
    assert rec.injected == []
    rec._qid = "q-007"
    assert wd.check_once() == KIND_FINALIZE_CLARIFY
    assert rec.injected[0]["qid"] == "q-007"


def test_check_once_none_when_no_session():
    rec = _Recorder("DesigningExperiment")
    rec._st = None  # no active session
    wd = _watchdog(rec)
    assert wd.check_once() is None
