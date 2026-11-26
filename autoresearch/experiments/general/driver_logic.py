"""Pure decision logic for the headless AutoResearch driver.

The 34-state machine pauses at ``AwaitingUserDecision`` with a
``awaiting_user_decision_reason`` (V1.10 §3, ``reason-matrix.yaml``). A headless
driver must auto-answer that pause with the action that makes the MOST forward
progress toward a terminal ``Done`` -- without ever ``abandon`` (that routes to
``Failed``) and without an action whose ``next_state`` is a runtime template
(``"{previous_state}"`` / ``"{failed_state}"`` / ``"{violation_state}"``) that
the engine's ``_on_AwaitingUserDecision_user_decision_provided`` cannot resolve
in v1 (it logs + stays paused).

``pick_action`` is PURE (no LLM, no container, no I/O beyond the one-time
reason-matrix load) and therefore fully unit-testable. The matrix is loaded via
``UserDecisionGuard``/``load_user_decision_matrix`` so ``next_state`` templating
is checked against the SSOT YAML, not a hardcoded copy.
"""

from __future__ import annotations

from typing import Optional

from autoresearch.orchestrator.user_decision_matrix import (
    UserDecisionGuard,
    load_user_decision_matrix,
)

# The MOST-PROGRESSING action per AwaitingUserDecision reason (the driver's
# opinionated table). Every value is a concrete (non-templated), non-Failed,
# non-abandon next_state in reason-matrix.yaml -- i.e. it drives the session
# FORWARD toward Done rather than looping, retrying, or failing.
PREFERRED_ACTION: dict[str, str] = {
    "out_of_scope": "narrow_to_verifiable",              # -> ClarifyingResearchGoal
    "literature_uncertain": "continue_with_caveat",      # -> ModelingProblem
    "direction_thrashing": "continue_with_risk",         # -> RoutingResearchMode
    # Reaching this pause means no baseline passed the live-evidence gate.  An
    # unconditional "use_available_artifact" can therefore recycle the same
    # null/invalid attempt and return to baseline reproduction.  Preserve the
    # diagnosis and finish honestly; a valid frozen artifact never reaches this
    # reason in the first place.
    "baseline_unreproducible": "write_replication_failure_report",  # -> Reporting
    "hypothesis_stagnation": "narrow_scope",             # -> ModelingProblem
    "experiment_design_thrashing": "simplify_experiment",  # -> PlanningExperiment
    "implementation_thrashing": "discard_hypothesis",    # -> ConsolidatingMemory
    "result_suspicious": "accept_with_caveat",           # -> ConsolidatingMemory -> Reporting
    "research_thrashing": "accept_negative_result",      # -> Reporting
    "cost_exceeded": "report_now",                       # -> Reporting
    "agent_failure": "report_now",                       # -> Reporting
    "protocol_violation": "abandon_result",              # -> ConsolidatingMemory
    "search_space_collapse": "accept_local_search",      # -> PlanningExperiment
    "verifier_missing": "report_replication_limits",     # -> Reporting
}

# One-time load of the reason-matrix SSOT; used to check whether an action's
# next_state is concrete (not templated) and not Failed.
_GUARD = UserDecisionGuard(load_user_decision_matrix())


def _is_safe(reason: str, action: str) -> bool:
    """True iff ``action`` under ``reason`` has a CONCRETE, non-Failed
    next_state and is not ``abandon``.

    "Safe" = the engine can actually resolve + transition into it in v1: never
    ``abandon`` (-> Failed), never a templated next_state (contains ``"{"``),
    never ``Failed`` itself, and never an action absent from the matrix.
    """
    if action == "abandon":
        return False
    try:
        next_state = _GUARD.next_state(reason, action)
    except KeyError:
        return False
    if not next_state or "{" in next_state or next_state == "Failed":
        return False
    return True


def pick_action(reason: str, legal_actions: list[str], *,
                agent_failure_retries: int = 0, max_agent_retries: int = 1) -> Optional[str]:
    """Return the most-progressing legal action for an AwaitingUserDecision
    ``reason``, or None if nothing safe is legal.

    0. agent_failure ONLY: retry the failed step ONCE (``retry_session``) before the
       conservative ``report_now`` salvage. Its ``{failed_state}`` next_state IS
       engine-resolvable (the engine substitutes ``previous_state`` -- see
       ``_on_AwaitingUserDecision_user_decision_provided``), so unlike other templates
       it drives real forward progress. This mirrors ``MossWatchdog._decision_action``
       so a TRANSIENT agent hiccup after a successful experiment doesn't discard the
       finding by jumping straight to an empty report. Bounded by ``max_agent_retries``
       (the caller tracks ``agent_failure_retries`` per session to avoid a retry loop).
    1. If the reason's PREFERRED_ACTION is legal AND safe -> return it.
    2. Otherwise fall back to the FIRST legal action that is safe (concrete,
       non-Failed next_state; never abandon; never templated).
    3. If no legal action is safe -> None (the driver stops answering).
    """
    if (reason == "agent_failure" and agent_failure_retries < max_agent_retries
            and "retry_session" in legal_actions):
        return "retry_session"
    preferred = PREFERRED_ACTION.get(reason)
    if preferred is not None and preferred in legal_actions and _is_safe(reason, preferred):
        return preferred
    for action in legal_actions:
        if _is_safe(reason, action):
            return action
    return None


__all__ = ["pick_action", "PREFERRED_ACTION"]
