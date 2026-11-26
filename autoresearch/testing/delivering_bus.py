"""Shared in-memory ``DeliveringBus`` test harness.

Extracted verbatim (semantics-wise) from the ``DeliveringBus``/``BusClientView``
copies duplicated across the orchestrator smoke tests -- see
``autoresearch/orchestrator/tests/smoke/test_m7_verifier_smoke.py`` for the
original. The smoke files could later import this module instead of keeping
their own copies, but they are left untouched for now.
"""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
import fnmatch
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

from autoresearch.eventbus.broker.envelope import Envelope


class DeliveringBus:
    """Shared broker state: subscriptions, the real sqlite handle, and every
    published ``Envelope`` -- delivered synchronously to matching subscribers
    (mirrors ``Dispatcher.dispatch``'s sequential fnmatch routing)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        self.subscriptions: list[tuple[str, Any]] = []
        self.published: list[Envelope] = []
        self._pending: ContextVar[deque[Envelope] | None] = ContextVar(
            "delivering_bus_pending", default=None)

    async def _delivery_table(self) -> None:
        await self.db.execute("""CREATE TABLE IF NOT EXISTS web_pending_delivery (
            event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            envelope TEXT NOT NULL, next_handler INTEGER NOT NULL DEFAULT 0)""")

    async def _sync_restart_recovery_request(
        self, session_id: str, event_id: str, payload: dict
    ) -> None:
        """Keep the persisted request binding identical to its replayed envelope."""
        cursor = await self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_state'"
        )
        if await cursor.fetchone() is None:
            return
        cursor = await self.db.execute(
            "SELECT state_data, current_state FROM session_state WHERE session_id=?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return
        state_data = json.loads(row[0])
        request = state_data.get("active_request") or {}
        if not (
            row[1] == "RunningExperiment"
            and request.get("state") == "RunningExperiment"
            and request.get("event_type") == "experiment.run_requested"
            and request.get("event_id") == event_id
        ):
            return
        now = time.time()
        request["payload"] = dict(payload)
        state_data["active_request"] = request
        if "last_updated_at" in state_data:
            state_data["last_updated_at"] = now
        await self.db.execute(
            "UPDATE session_state SET state_data=?, updated_at=? WHERE session_id=?",
            (json.dumps(state_data), now, session_id),
        )

    async def replay_pending(
        self, session_id: str, *, restart_recovery: bool = False
    ) -> bool:
        """Resume saved handoffs, instead of repeating the state's experiment."""
        await self._delivery_table()
        cursor = await self.db.execute(
            "SELECT event_id, envelope FROM web_pending_delivery "
            "WHERE session_id=? ORDER BY rowid",
            (session_id,))
        rows = await cursor.fetchall()
        if not rows:
            return False
        pending = deque()
        for event_id, raw in rows:
            saved = json.loads(raw)
            if (restart_recovery
                    and saved.get("event_type") == "experiment.run_requested"):
                saved["payload"] = {
                    **(saved.get("payload") or {}),
                    "recover_completed_result": True,
                }
                raw = json.dumps(saved)
                await self.db.execute(
                    "UPDATE web_pending_delivery SET envelope=? WHERE event_id=?",
                    (raw, event_id),
                )
                await self._sync_restart_recovery_request(
                    session_id, event_id, saved["payload"]
                )
            pending.append(Envelope(**saved))
        await self.db.commit()
        await self._drain(pending)
        return True

    async def _drain(self, pending: deque[Envelope]) -> None:
        token = self._pending.set(pending)
        try:
            while pending:
                current = pending.popleft()
                cursor = await self.db.execute(
                    "SELECT next_handler FROM web_pending_delivery WHERE event_id=?",
                    (current.event_id,))
                row = await cursor.fetchone()
                if row is None:
                    continue
                for index, (pattern, handler) in enumerate(list(self.subscriptions)):
                    if index < row[0]:
                        continue
                    if fnmatch.fnmatchcase(current.event_type, pattern):
                        started = time.monotonic()
                        try:
                            await handler(current)
                        finally:
                            logging.getLogger(__name__).info(
                                "research.delivery session=%s request=%s event=%s handler=%s elapsed=%.3f",
                                current.session_id, current.causation_id, current.event_type,
                                index, time.monotonic() - started)
                    await self.db.execute(
                        "UPDATE web_pending_delivery SET next_handler=? WHERE event_id=?",
                        (index + 1, current.event_id))
                    await self.db.commit()
                await self.db.execute("DELETE FROM web_pending_delivery WHERE event_id=?",
                                      (current.event_id,))
                await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise
        finally:
            self._pending.reset(token)

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        emitter: str,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
        event_id: Optional[str] = None,
    ) -> str:
        env = Envelope(
            event_id=event_id or str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            emitter=emitter,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            schema_version=1,
            cost_delta=cost_delta,
            payload=payload,
            causation_id=causation_id,
        )
        self.published.append(env)
        await self._delivery_table()
        await self.db.execute(
            "INSERT OR IGNORE INTO web_pending_delivery (event_id, session_id, envelope) VALUES (?, ?, ?)",
            (env.event_id, session_id, env.to_json()))
        if auto_commit:
            await self.db.commit()
        # A handler commonly publishes the next state-machine event.  Recursively
        # dispatching that event keeps every prior handler frame alive and long
        # sessions eventually hit Python's recursion limit.  The outermost publish
        # drains the same FIFO cascade iteratively; nested publishes only enqueue.
        pending = self._pending.get()
        if pending is not None:
            pending.append(env)
            return env.event_id
        await self._drain(deque([env]))
        return env.event_id

    def view(self, emitter: str) -> "BusClientView":
        return BusClientView(self, emitter)


class BusClientView:
    """One emitter's handle onto a shared ``DeliveringBus`` -- mirrors how
    each real process holds its own ``EventBusClient`` (its own
    ``emitter_name``) while every client shares one broker/outbox."""

    def __init__(self, bus: DeliveringBus, emitter_name: str) -> None:
        self._bus = bus
        self.emitter_name = emitter_name

    @property
    def db(self) -> aiosqlite.Connection:
        return self._bus.db

    def subscribe(self, pattern: str, handler: Any) -> None:
        self._bus.subscribe(pattern, handler)

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
        event_id: Optional[str] = None,
    ) -> str:
        return await self._bus.publish(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            emitter=self.emitter_name,
            cost_delta=cost_delta,
            causation_id=causation_id,
            auto_commit=auto_commit,
            event_id=event_id,
        )
