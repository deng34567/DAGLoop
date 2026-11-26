"""
Outbox pattern for transactional event emission.

The producer side of publish() writes the envelope to a SQLite outbox table
within the same transaction as any business state update. A background worker
then drains the outbox to Redis Streams via XADD.

This decouples business-state durability from Redis availability:
- if Redis is briefly down, events accumulate in outbox; worker drains when Redis returns
- if the process crashes after committing the business state but before XADD,
  the outbox row survives and is delivered on next start
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Awaitable

import aiosqlite

logger = logging.getLogger(__name__)


_OUTBOX_DDL = """
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_key TEXT NOT NULL,
    envelope_json TEXT NOT NULL,
    status TEXT NOT NULL,                   -- 'pending' | 'sent'
    attempts INTEGER NOT NULL DEFAULT 0,
    next_retry_at REAL NOT NULL,            -- unix timestamp seconds
    last_error TEXT,
    created_at REAL NOT NULL,
    sent_at REAL
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON outbox(status, next_retry_at)
    WHERE status='pending';
"""


async def init_outbox(db: aiosqlite.Connection) -> None:
    """Create outbox table if not exists."""
    for stmt in _OUTBOX_DDL.split(";"):
        s = stmt.strip()
        if s:
            await db.execute(s)
    await db.commit()


async def write_outbox(
    db: aiosqlite.Connection,
    stream_key: str,
    envelope_json: str,
) -> int:
    """Insert a pending row. Returns the row id. Caller must commit (or be in transaction)."""
    now = time.time()
    cursor = await db.execute(
        """
        INSERT INTO outbox (stream_key, envelope_json, status, attempts,
                            next_retry_at, created_at)
        VALUES (?, ?, 'pending', 0, ?, ?)
        """,
        (stream_key, envelope_json, now, now),
    )
    return cursor.lastrowid


async def claim_due_pending(
    db: aiosqlite.Connection,
    limit: int = 100,
) -> list[tuple]:
    """Return rows of (id, stream_key, envelope_json, attempts) that are due."""
    cursor = await db.execute(
        """
        SELECT id, stream_key, envelope_json, attempts
        FROM outbox
        WHERE status='pending' AND next_retry_at <= ?
        ORDER BY id LIMIT ?
        """,
        (time.time(), limit),
    )
    return list(await cursor.fetchall())


async def mark_sent(db: aiosqlite.Connection, row_id: int) -> None:
    await db.execute(
        "UPDATE outbox SET status='sent', sent_at=? WHERE id=?",
        (time.time(), row_id),
    )
    await db.commit()


async def bump_failure(
    db: aiosqlite.Connection,
    row_id: int,
    attempts: int,
    error: str,
) -> float:
    """Reschedule with capped exponential backoff. Returns the next_retry_at."""
    # 1s, 2s, 4s, ..., capped at 300s (5 min)
    backoff = min(2 ** attempts, 300)
    next_retry = time.time() + backoff
    await db.execute(
        """
        UPDATE outbox
        SET attempts = attempts + 1,
            next_retry_at = ?,
            last_error = ?
        WHERE id = ?
        """,
        (next_retry, error[:500], row_id),
    )
    await db.commit()
    return next_retry


class OutboxWorker:
    """Drains the outbox into Redis. Runs as an asyncio task in the background."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        xadd_callback: Callable[[str, str], Awaitable[None]],
        poll_interval_sec: float = 0.1,
    ):
        """
        Args:
            db: open aiosqlite connection (shared with publish-side)
            xadd_callback: async (stream_key, envelope_json) -> None
                that performs the actual XADD. Raises on Redis failure.
            poll_interval_sec: sleep between empty polls
        """
        self.db = db
        self.xadd_callback = xadd_callback
        self.poll_interval_sec = poll_interval_sec
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rows = await claim_due_pending(self.db)
            except Exception as e:
                logger.error("Outbox claim error: %s", e)
                await asyncio.sleep(self.poll_interval_sec)
                continue

            if not rows:
                # Sleep but wake up early if stop is requested
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.poll_interval_sec
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            for row_id, stream_key, envelope_json, attempts in rows:
                try:
                    await self.xadd_callback(stream_key, envelope_json)
                    await mark_sent(self.db, row_id)
                    if attempts > 0:
                        logger.info(
                            "Outbox row %d delivered after %d retries", row_id, attempts
                        )
                except Exception as e:
                    next_retry = await bump_failure(self.db, row_id, attempts, str(e))
                    if attempts + 1 >= 10:
                        logger.warning(
                            "Outbox row %d failed >=10 times, next_retry=%s, last_error=%s",
                            row_id,
                            next_retry,
                            e,
                        )
