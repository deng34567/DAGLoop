"""
MossObserver — MOSS Mode B, cross-session pattern detection (non-blocking).

Subscribed to the ALL-events pattern. For each event it extracts a compact
structured signal (agent, reason, detail) and records it in
MossObservationStore. Detection is aggregation + a DISTINCT-session threshold
(no per-event LLM): a pattern only counts once it recurs across at least
``threshold`` distinct sessions, at which point it raises an L3 alert (a
flagged record + a WARNING log — MOSS does not publish any event for this).

The L1 auto-patch path is gated OFF by default (env
AUTORESEARCH_MOSS_MODEB_AUTOPATCH=1 to "enable" it) and is intentionally a
no-op in v1: it never writes a patch, calls an LLM, or touches PatchWriter.

on_event() must never raise — an observer crash must not break the bus.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MossObserver:
    def __init__(
        self,
        store: Any,
        *,
        threshold_sessions: Optional[int] = None,
    ) -> None:
        self.store = store
        if threshold_sessions is not None:
            self.threshold = threshold_sessions
        else:
            self.threshold = int(
                os.environ.get("AUTORESEARCH_MOSS_MODEB_THRESHOLD", "5")
            )
        self.autopatch = os.environ.get("AUTORESEARCH_MOSS_MODEB_AUTOPATCH") == "1"

    # ---- event ingestion ----

    async def on_event(self, env: Any) -> None:
        """EventBus handler. NEVER raises — an observer crash must not affect
        the bus. Emits nothing (no publish)."""
        try:
            self._ingest(env)
        except Exception:
            logger.debug(
                "MOSS Mode B: ingest failed for %s",
                getattr(env, "event_type", "?"),
                exc_info=True,
            )

    def _ingest(self, env: Any) -> None:
        c = self._classify(env.event_type, env.payload)
        if c is None:
            return
        agent, reason, detail = c
        self.store.record_signal(
            env.session_id, agent, reason, detail=detail, task_mode=None
        )
        self._detect(agent, reason, env.session_id)

    @staticmethod
    def _classify(event_type: str, payload: dict) -> Optional[tuple[str, str, dict]]:
        """Map an event to (agent, reason, detail), or None to ignore."""
        if event_type == "agent.failed":
            return (
                payload["agent_name"],
                payload.get("underlying_reason")
                or payload.get("reason_classification")
                or "unknown",
                {"classification": payload.get("reason_classification")},
            )
        if event_type == "moss.intervention_completed":
            return (
                "ResearchOrchestrator",
                f"moss_intervention:{payload['reason']}",
                {"action": payload.get("action"), "degraded": payload.get("degraded")},
            )
        return None

    # ---- detection / action ----

    def _detect(self, agent: str, reason: str, session_id: str) -> None:
        n = self.store.distinct_sessions(agent, reason)
        if n < self.threshold:
            return
        self.store.flag_l3(
            session_id,
            "recurring_pattern",
            {"agent": agent, "reason": reason, "distinct_sessions": n},
        )
        logger.warning(
            "MOSS Mode B: recurring pattern (%s, %s) across %d sessions — flagged L3",
            agent,
            reason,
            n,
        )
        if self.autopatch:
            logger.info(
                "MOSS Mode B L1 auto-patch hook enabled but intentionally a "
                "no-op in v1 — not writing a patch."
            )


__all__ = ["MossObserver"]
