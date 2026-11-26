"""MOSS watchdog: a stall-recovery safety net so a driven session NEVER hangs.

The 34-state engine is purely event-driven: if an agent never publishes its
response event (a *genuine* silent stall -- an agent handler that dies without
emitting ``agent.failed``, or a human-wait state with no human present), nothing
ticks and the session sits forever. This watchdog runs OFF the engine loop (on
its own thread, so it survives an engine-loop block), reads the session state
straight from sqlite, detects a session that has made no state progress for
longer than a per-state timeout, and injects the most CONSERVATIVE progressing
event -- MOSS stepping in for the absent human:

  - ClarifyingResearchGoal -> finalize the spec now (answer the open question
                              with "use best judgement, stop asking").
  - HumanIdeaCheckpoint     -> idea_provided {inject: False} (add nothing).
  - ConfirmingMethod        -> method.confirmed {inject: False} (accept the
                              audited method without changing it).
  - AwaitingUserDecision    -> the most conservative legal decision (prefer
                              ``report_now``: salvage a report -> Done; never
                              ``abandon`` unless it is the only legal option).
  - Suspended               -> resume the exact saved pre-suspend request after
                              the normal stall timeout (provider outages remain
                              bounded by the existing retry and timeout policy).
  - any other (agent-work) state -> ``agent.failed``, which the engine routes
                              (H1) to AwaitingUserDecision(agent_failure),
                              resolved to ``report_now`` on the next poll.

So any *true* stall converges to a report + Done within a couple of poll cycles.

CRUCIAL: the P1 general-experiment runs the whole ~20-min Docker experiment
INLINE inside RunningExperiment's cascade, and ``DeliveringBus`` commits that
transition only AFTER the cascade returns -- so during the experiment the DB
still shows the PRE-transition state (ImplementingExperiment). A naive
timeout would wrongly kill a healthy experiment. The watchdog therefore takes a
``is_busy(state)`` liveness probe (the web server passes one that reports True
while an OpenHands experiment container is live); a busy state is treated as
progressing and is never intervened on, regardless of timeout.

The pure decision helpers (``decide_recovery`` / ``conservative_action`` /
``timeout_for``) carry no I/O and are unit-tested; the thread runner
(``MossWatchdog``) wires them to injected read/inject callables so the web
server owns all transport.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

TERMINAL_STATES = frozenset({"Done", "Failed"})

# Post-experiment "home stretch": once here the session is about to reach Done
# naturally (evaluate the result, consolidate it into memory, write the report).
# The whole-session BUDGET must NOT force-wrap these -- cutting the tail early
# (e.g. skipping ConsolidatingMemory) would drop the experiment's findings from
# the report. The per-state STALL timer still guards them (a genuine hang here is
# still recovered), but a merely-slow finish is left to complete.
FINISHING_STATES = frozenset({
    "EvaluatingExperiment", "VerifyingBestResult", "ConsolidatingMemory", "Reporting",
})

# ``Suspended`` is entered deliberately after a bounded provider failure.  The
# event that persists the suspension may still be finishing its delivery
# cascade when the watchdog observes it.  Let the ordinary per-state stall
# timer govern recovery here; otherwise an already-over-budget session can be
# resumed immediately and race with the tail of that same failure event.
SESSION_BUDGET_EXEMPT_STATES = FINISHING_STATES | {"Suspended"}

# Per-state "no progress in this state" timeout, seconds. Human-wait states get a
# few minutes (a present human answers sooner; an absent one is covered); the P1
# experiment window (Implementing/Running -- the state shows Implementing while
# the experiment runs inline) gets a generous ceiling ON TOP of the is_busy
# liveness guard, so only a truly dead experiment is ever reclaimed.
DEFAULT_TIMEOUTS: dict[str, float] = {
    # Provider/API outages are persisted as Suspended without spending scientific
    # retry budgets.  Give the endpoint five minutes to recover, then re-drive the
    # exact saved request.
    "Suspended": 300.0,
    "ClarifyingResearchGoal": 480.0,
    "HumanIdeaCheckpoint": 480.0,
    "AwaitingUserDecision": 300.0,
    "AwaitingExternalExperimentResult": 480.0,
    "ImplementingExperiment": 3600.0,
    "RunningExperiment": 3600.0,
    # Training-heavy states run REAL training/compilation (baseline reproduction, candidate
    # scoring, and best-result verification which re-trains N times to check reproducibility).
    # A repo's training runs as a LOCAL subprocess, so the docker-based is_busy probe often
    # can't see it -- with the 900s default these legitimately-long states were falsely
    # reclaimed as "no progress" (observed: repeated agent_failure loops in VerifyingBestResult /
    # ReproducingBaseline while train.py was actively running on the GPU). Give them generous
    # ceilings so only a truly dead experiment is ever cut off. VerifyingBestResult gets the
    # most (it does several full reproduction runs, each with a torch.compile warmup).
    "ReproducingBaseline": 14400.0,
    "EvaluatingExperiment": 1800.0,
    "VerifyingBestResult": 5400.0,
    # LLM-heavy inference/design states make SEVERAL model calls each. On a slow model (glm
    # trickling reasoning, each call bounded by the 600s stream deadline) a couple of calls can
    # exceed the 900s default, falsely reclaiming the state as "no progress" -> agent_failure ->
    # AwaitingUserDecision -> retry churn (observed live in SelectingBenchmark on glm). Give it a
    # generous ceiling; the per-call 600s stream deadline still bounds a genuinely hung call.
    "SelectingBenchmark": 2400.0,
}
# Raised 900 -> 1800 for the same reason: a state not listed above may still make a few slow LLM
# calls on glm; 1800 covers ~3 calls at the 600s stream ceiling. The whole-session budget + the
# per-call stream deadline remain the real backstops against a true hang.
DEFAULT_TIMEOUT = 1800.0

# Whole-session budget: even if the state keeps CHANGING (a livelock -- e.g. the
# plan-review / experiment_design_thrashing loop where MOSS keeps re-routing so
# no single state is "stuck"), once the session has run this long without
# terminating, MOSS wraps it up to a report. This is the "时间上限 -> 兜底" the per-
# state no-progress timer cannot catch. Suppressed while a P1 experiment is busy
# (is_busy) so a legitimately long experiment is never cut off. Raised to 3h: a full
# repo-improve run trains MANY times (baseline reproduction + several candidate rounds +
# best-result verification across N seeds), and since local training subprocesses are often
# invisible to the docker-based is_busy probe, a 60-min budget falsely wrapped up legitimately
# long runs. 3h is a livelock backstop, not a normal-path limit.
DEFAULT_SESSION_TIMEOUT = 10800.0

# Conservative decision priority: salvage a report / accept-with-caveat / proceed
# with what we have. NEVER pick a compute-burning retry (retry_session,
# deepen_literature, strict_rerun, broaden_search, switch_model_or_tool,
# change_benchmark, modify_goal, inject_idea, human_literature_seed) and never
# ``abandon`` (-> Failed) unless it is the ONLY legal action.
CONSERVATIVE_PRIORITY: tuple[str, ...] = (
    "report_now",
    "accept_with_caveat",
    "write_replication_failure_report",
    "use_available_artifact",
    "continue_with_caveat",
    "narrow_to_verifiable",
    "narrow_scope",
    "proceed_out_of_scope",
)

# Recovery KINDs returned by decide_recovery.
KIND_FINALIZE_CLARIFY = "finalize_clarify"
KIND_IDEA_NO_INJECT = "idea_no_inject"
KIND_METHOD_NO_INJECT = "method_no_inject"
KIND_DECISION = "decision"
KIND_RESUME_SUSPENDED = "resume_suspended"
KIND_AGENT_FAILED = "agent_failed"

# A firm canned answer to any open clarification question: stop asking, finalize.
CANNED_FINALIZE = (
    "Use your best judgment. Do NOT ask any more questions. Finalize the "
    "research spec NOW with reasonable defaults and proceed."
)


def timeout_for(state_name: str, timeouts: dict[str, float], default_timeout: float) -> float:
    """The no-progress timeout for ``state_name`` (per-state override or default)."""
    return timeouts.get(state_name, default_timeout)


def conservative_action(legal_actions) -> Optional[str]:
    """Pick the most conservative legal AwaitingUserDecision action.

    Prefer (in order) a report-salvaging / accept-with-caveat / proceed action;
    else any non-``abandon`` legal action; else ``abandon``; else None.
    """
    legal = list(legal_actions or [])
    for action in CONSERVATIVE_PRIORITY:
        if action in legal:
            return action
    non_abandon = [a for a in legal if a != "abandon"]
    if non_abandon:
        return non_abandon[0]
    if "abandon" in legal:
        return "abandon"
    return None


def decide_recovery(
    state_name: Optional[str],
    entered_ago: float,
    *,
    busy: bool,
    timeouts: dict[str, float],
    default_timeout: float,
    session_ago: float = 0.0,
    session_timeout: Optional[float] = None,
) -> Optional[str]:
    """Return the recovery KIND for a possibly-stalled/runaway session, or None.

    None when: terminal, or a liveness probe says the state is busy (a live P1
    experiment container). Otherwise MOSS intervenes when EITHER
      - the session has been in this state longer than its per-state timeout
        (a stall / absent human), OR
      - the whole session has run past ``session_timeout`` without terminating
        (a livelock the per-state timer can't see -- state keeps changing).
    The stalled/runaway state then maps to a recovery KIND.
    """
    if state_name is None or state_name in TERMINAL_STATES:
        return None
    if busy:
        return None
    # The whole-session budget forces a wrap-up on a livelock -- but never on the
    # post-experiment home stretch or a freshly persisted provider suspension.
    # Those remain protected by their ordinary per-state stall timeout.
    over_budget = (
        session_timeout is not None
        and session_ago >= session_timeout
        and state_name not in SESSION_BUDGET_EXEMPT_STATES
    )
    if not over_budget and entered_ago < timeout_for(state_name, timeouts, default_timeout):
        return None
    if state_name == "ClarifyingResearchGoal":
        return KIND_FINALIZE_CLARIFY
    if state_name == "HumanIdeaCheckpoint":
        return KIND_IDEA_NO_INJECT
    if state_name == "ConfirmingMethod":
        return KIND_METHOD_NO_INJECT
    if state_name == "AwaitingUserDecision":
        return KIND_DECISION
    if state_name == "Suspended":
        return KIND_RESUME_SUSPENDED
    return KIND_AGENT_FAILED


class MossWatchdog:
    """Thread runner: poll session state, decide, inject conservative recovery.

    All transport is injected so the runner stays testable:
      - ``read_state()`` -> dict|None with keys ``session_id``, ``current_state``,
        ``state_entered_at`` (wall clock), ``awaiting_user_decision_reason``.
      - ``legal_actions_for(reason)`` -> list of legal AwaitingUserDecision actions.
      - ``latest_qid()`` -> the newest open clarification ``question_id`` (or None).
      - ``is_busy(state)`` -> True while the state is legitimately working (a live
        P1 experiment container); such a state is never intervened on.
      - ``inject(kind, session_id, reason, action, qid)`` -> perform the actual
        publish onto the engine loop (the server wires this to Engine.submit).
    """

    def __init__(
        self,
        *,
        read_state: Callable[[], Optional[dict]],
        legal_actions_for: Callable[[Optional[str]], Any],
        latest_qid: Callable[[], Optional[str]],
        is_busy: Callable[[str], bool],
        inject: Callable[..., None],
        clock: Callable[[], float] = time.time,
        poll_interval: float = 30.0,
        timeouts: Optional[dict[str, float]] = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        session_timeout: Optional[float] = DEFAULT_SESSION_TIMEOUT,
        max_agent_retries: int = 1,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.read_state = read_state
        self.legal_actions_for = legal_actions_for
        self.latest_qid = latest_qid
        self.is_busy = is_busy
        self.inject = inject
        self.clock = clock
        self.poll_interval = poll_interval
        self.timeouts = dict(DEFAULT_TIMEOUTS if timeouts is None else timeouts)
        self.default_timeout = default_timeout
        self.session_timeout = session_timeout
        # A TRANSIENT agent_failure is retried this many times (retry_session) before
        # falling back to the conservative report_now -- so a one-off hiccup doesn't
        # discard the whole run.
        self.max_agent_retries = max_agent_retries
        self.log = log or logger
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Debounce: at most one intervention per unique (session, state, entered_at)
        # and (for clarify) per unique question_id, so a queued inject on a blocked
        # engine loop is not re-issued every poll.
        self._done: set = set()
        self._agent_retries: dict = {}  # per-session count of retry_session injected

    def _decision_action(self, sid, reason, legal) -> Optional[str]:
        """Pick the AwaitingUserDecision action. A transient agent_failure gets ONE
        ``retry_session`` (re-run the failed step) before the conservative
        ``report_now`` salvage."""
        if (reason == "agent_failure" and "retry_session" in (legal or [])
                and self._agent_retries.get(sid, 0) < self.max_agent_retries):
            self._agent_retries[sid] = self._agent_retries.get(sid, 0) + 1
            return "retry_session"
        return conservative_action(legal)

    # ---- one poll (pure-ish: only read_state / is_busy / inject I/O) ----

    def check_once(self) -> Optional[str]:
        """Read state once; if stalled, inject recovery. Returns the KIND injected
        (for logging/tests) or None. Never raises out (defensive)."""
        try:
            st = self.read_state()
        except Exception:  # noqa: BLE001 -- a read hiccup must not kill the loop
            self.log.exception("watchdog: read_state raised")
            return None
        if not st:
            return None
        state = st.get("current_state")
        entered_at = st.get("state_entered_at")
        created_at = st.get("created_at")
        sid = st.get("session_id")
        reason = st.get("awaiting_user_decision_reason")
        if state is None or entered_at is None or sid is None:
            return None

        try:
            busy = bool(self.is_busy(state))
        except Exception:  # noqa: BLE001 -- probe failure => assume NOT busy (safe: timeout still guards)
            self.log.exception("watchdog: is_busy raised; treating as not busy")
            busy = False

        now = self.clock()
        session_ago = (now - float(created_at)) if created_at is not None else 0.0
        kind = decide_recovery(
            state, now - float(entered_at),
            busy=busy, timeouts=self.timeouts, default_timeout=self.default_timeout,
            session_ago=session_ago, session_timeout=self.session_timeout,
        )
        if kind is None:
            return None

        # Resolve the clarify question_id up-front (it is part of the dedup key), then
        # dedup BEFORE computing a decision action (so the retry counter is only bumped
        # on an intervention that actually fires, never on a deduped poll).
        qid = None
        if kind == KIND_FINALIZE_CLARIFY:
            qid = self.latest_qid()
            if qid is None:
                return None  # nothing to answer yet; try again next poll

        dedup_key = (sid, state, round(float(entered_at), 3), qid)
        if dedup_key in self._done:
            return None
        self._done.add(dedup_key)

        action = None
        if kind == KIND_DECISION:
            action = self._decision_action(sid, reason, self.legal_actions_for(reason))
            if action is None:
                self.log.warning(
                    "watchdog: AwaitingUserDecision(reason=%r) has no legal action; skipping", reason
                )
                return None

        self.log.warning(
            "MOSS watchdog: no progress in %s for >%.0fs -> intervening (kind=%s, action=%s)",
            state, timeout_for(state, self.timeouts, self.default_timeout), kind, action,
        )
        try:
            self.inject(kind=kind, session_id=sid, reason=reason, action=action, qid=qid)
        except Exception:  # noqa: BLE001
            self.log.exception("watchdog: inject raised for kind=%s", kind)
        return kind

    # ---- thread lifecycle ----

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(self.poll_interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="moss-watchdog", daemon=True)
        self._thread.start()
        self.log.info("MOSS watchdog started (poll=%.0fs)", self.poll_interval)

    def stop(self) -> None:
        self._stop.set()


def watchdog_enabled() -> bool:
    """The watchdog is ON by default; set AUTORESEARCH_WATCHDOG=0 to disable."""
    return os.environ.get("AUTORESEARCH_WATCHDOG", "1") != "0"


__all__ = [
    "MossWatchdog",
    "decide_recovery",
    "conservative_action",
    "timeout_for",
    "watchdog_enabled",
    "DEFAULT_TIMEOUTS",
    "DEFAULT_TIMEOUT",
    "CANNED_FINALIZE",
    "TERMINAL_STATES",
    "KIND_FINALIZE_CLARIFY",
    "KIND_IDEA_NO_INJECT",
    "KIND_METHOD_NO_INJECT",
    "KIND_DECISION",
    "KIND_RESUME_SUSPENDED",
    "KIND_AGENT_FAILED",
]
