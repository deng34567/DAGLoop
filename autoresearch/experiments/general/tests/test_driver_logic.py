"""Unit tests for the headless driver's pure decision logic (``pick_action``).

No LLM, no container, no engine -- just the reason-matrix SSOT + the opinionated
PREFERRED_ACTION table. Verifies:
  - every reason yields its preferred, most-progressing action given the full
    legal set;
  - a templated / abandon / Failed action is NEVER returned;
  - the fallback picks the first legal action with a concrete, non-Failed
    next_state when the preferred action is not legal;
  - None is returned when nothing safe is legal.
"""

from __future__ import annotations

import pytest

from autoresearch.experiments.general.driver_logic import PREFERRED_ACTION, pick_action
from autoresearch.orchestrator.user_decision_matrix import (
    UserDecisionGuard,
    load_user_decision_matrix,
)

_GUARD = UserDecisionGuard(load_user_decision_matrix())


# ---------------------------------------------------------------------------
# 1. Each reason -> its preferred action, given the reason's FULL legal set.
# ---------------------------------------------------------------------------


# agent_failure is special: it retries the failed step ONCE (retry_session) before
# its PREFERRED_ACTION (report_now) salvage, so it is covered by its own tests below.
@pytest.mark.parametrize(
    "reason,expected",
    sorted((r, a) for r, a in PREFERRED_ACTION.items() if r != "agent_failure"),
)
def test_pick_action_returns_preferred_for_full_legal_set(reason, expected):
    legal = _GUARD.legal_actions(reason)
    assert expected in legal, f"table says {expected!r} but matrix legal={legal}"
    assert pick_action(reason, legal) == expected


def test_agent_failure_preferred_after_retry_is_report_now():
    legal = _GUARD.legal_actions("agent_failure")
    # once the single retry is spent, fall back to the conservative report_now salvage
    assert pick_action("agent_failure", legal, agent_failure_retries=1) == "report_now"


# ---------------------------------------------------------------------------
# 2. The preferred action for every reason is itself concrete / non-Failed /
#    non-abandon (a self-consistency guard on the table).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason,action", sorted(PREFERRED_ACTION.items()))
def test_preferred_action_next_state_is_concrete(reason, action):
    next_state = _GUARD.next_state(reason, action)
    assert action != "abandon"
    assert next_state != "Failed"
    assert "{" not in next_state


# ---------------------------------------------------------------------------
# 3. abandon is NEVER returned, even when it is the ONLY legal action.
# ---------------------------------------------------------------------------


def test_abandon_never_returned_even_if_only_legal():
    assert pick_action("out_of_scope", ["abandon"]) is None


def test_abandon_never_returned_when_legal_alongside_others():
    # protocol_violation legal set includes abandon; it must never win.
    legal = _GUARD.legal_actions("protocol_violation")
    assert "abandon" in legal
    assert pick_action("protocol_violation", legal) != "abandon"


# ---------------------------------------------------------------------------
# 4. A templated next_state action is never returned.
# ---------------------------------------------------------------------------


def test_agent_failure_retries_session_once_then_report_now():
    # {failed_state} IS engine-resolvable (engine substitutes previous_state), so on a
    # FIRST agent_failure retry the failed step (retry_session) -- mirroring the
    # MossWatchdog -- so a transient hiccup doesn't discard a successful experiment's
    # finding. Only after the retry is spent do we take the report_now salvage.
    legal = ["retry_session", "switch_model_or_tool", "report_now"]
    assert pick_action("agent_failure", legal) == "retry_session"                      # 1st time
    assert pick_action("agent_failure", legal, agent_failure_retries=1) == "report_now"  # retry spent


def test_agent_failure_retry_bounded_by_max():
    # With max_agent_retries=0 (retry disabled) it goes straight to report_now.
    legal = ["retry_session", "report_now"]
    assert pick_action("agent_failure", legal, max_agent_retries=0) == "report_now"


def test_templated_action_never_returned_for_non_retry_reasons():
    # For NON-agent_failure reasons, a templated next_state is still never returned:
    # result_suspicious's request_reproduction is "{previous_state}" (not resolvable).
    legal = _GUARD.legal_actions("result_suspicious")
    action = pick_action("result_suspicious", legal)
    assert action == PREFERRED_ACTION["result_suspicious"]      # accept_with_caveat (concrete)
    assert "{" not in _GUARD.next_state("result_suspicious", action)


def test_agent_failure_only_templated_after_retry_returns_none():
    # After the one retry is spent, if only templated actions remain (no concrete
    # report_now), nothing safe is legal -> None (switch_model_or_tool is {failed_state}
    # but is not the bounded retry, and the fallback rejects all templates).
    legal = ["retry_session", "switch_model_or_tool"]
    assert pick_action("agent_failure", legal) == "retry_session"                     # retry first
    assert pick_action("agent_failure", legal, agent_failure_retries=1) is None       # then nothing safe


# ---------------------------------------------------------------------------
# 5. Fallback: preferred not legal -> first legal action with a concrete,
#    non-Failed next_state.
# ---------------------------------------------------------------------------


def test_fallback_picks_first_concrete_when_preferred_absent():
    # baseline_unreproducible preferred = write_replication_failure_report;
    # drop it.  The remaining concrete action is change_benchmark.
    action = pick_action(
        "baseline_unreproducible",
        ["change_benchmark", "abandon"],
    )
    assert action == "change_benchmark"
    assert _GUARD.next_state("baseline_unreproducible", action) == "SelectingBenchmark"


def test_unreproducible_baseline_reports_instead_of_reusing_invalid_attempt():
    action = pick_action(
        "baseline_unreproducible",
        ["retry_session", "use_available_artifact",
         "write_replication_failure_report", "abandon"],
    )
    assert action == "write_replication_failure_report"
    assert _GUARD.next_state("baseline_unreproducible", action) == "Reporting"


def test_fallback_skips_templated_to_reach_concrete():
    # protocol_violation: rollback_and_retry is templated ("{violation_state}");
    # abandon_result is concrete (-> ConsolidatingMemory). Fallback skips the
    # templated one.
    action = pick_action(
        "protocol_violation", ["rollback_and_retry", "abandon_result", "abandon"]
    )
    assert action == "abandon_result"


# ---------------------------------------------------------------------------
# 6. Unknown reason / empty legal set -> None (defensive).
# ---------------------------------------------------------------------------


def test_unknown_reason_returns_none():
    assert pick_action("no_such_reason", ["abandon"]) is None


def test_empty_legal_returns_none():
    assert pick_action("out_of_scope", []) is None


def test_result_suspicious_accept_routes_through_consolidation():
    """P4 fix: result_suspicious/accept_with_caveat must go via ConsolidatingMemory so the
    finding is written to ResearchMemory before Reporting -- routing straight to Reporting
    produced an EMPTY report (method won, nothing consolidated)."""
    assert _GUARD.next_state("result_suspicious", "accept_with_caveat") == "ConsolidatingMemory"
    # discard still consolidates too; the two converging actions both pass through memory
    assert _GUARD.next_state("result_suspicious", "discard_result") == "ConsolidatingMemory"
