"""
Idempotency check: tracks (subscriber_name, stream_key) -> last_processed_event_id.

Used by the subscribe loop:
1. After reading an event from XREADGROUP, query has_processed(event_id, stream)
2. If yes, XACK without invoking handler
3. If no, invoke handler, then record (event_id, stream_id) within the same SQLite transaction
4. After commit, XACK Redis

This guarantees at-most-once handler invocation under Redis at-least-once delivery.
"""

from __future__ import annotations

import time

import aiosqlite


_IDEM_DDL = """
CREATE TABLE IF NOT EXISTS last_processed_event (
    subscriber_name TEXT NOT NULL,
    stream_key TEXT NOT NULL,
    last_event_id TEXT NOT NULL,             -- Redis Stream entry ID
    last_business_event_id TEXT,             -- envelope.event_id (UUID)
    updated_at REAL NOT NULL,
    PRIMARY KEY (subscriber_name, stream_key)
);
CREATE TABLE IF NOT EXISTS processed_event_ids (
    subscriber_name TEXT NOT NULL,
    business_event_id TEXT NOT NULL,         -- envelope.event_id
    stream_key TEXT NOT NULL,
    processed_at REAL NOT NULL,
    PRIMARY KEY (subscriber_name, business_event_id)
);
"""


async def init_idempotency(db: aiosqlite.Connection) -> None:
    for stmt in _IDEM_DDL.split(";"):
        s = stmt.strip()
        if s:
            await db.execute(s)
    await db.commit()


async def has_processed(
    db: aiosqlite.Connection,
    subscriber_name: str,
    business_event_id: str,
) -> bool:
    cursor = await db.execute(
        """
        SELECT 1 FROM processed_event_ids
        WHERE subscriber_name=? AND business_event_id=?
        """,
        (subscriber_name, business_event_id),
    )
    row = await cursor.fetchone()
    return row is not None


async def record_processed(
    db: aiosqlite.Connection,
    subscriber_name: str,
    business_event_id: str,
    stream_key: str,
    redis_stream_id: str,
) -> None:
    """Records that this subscriber processed this business_event_id. Caller must commit."""
    now = time.time()
    # processed_event_ids row (unique by subscriber + business_event_id)
    await db.execute(
        """
        INSERT OR IGNORE INTO processed_event_ids
        (subscriber_name, business_event_id, stream_key, processed_at)
        VALUES (?, ?, ?, ?)
        """,
        (subscriber_name, business_event_id, stream_key, now),
    )
    # last_processed_event row (one per stream per subscriber, tracks position)
    await db.execute(
        """
        INSERT INTO last_processed_event
        (subscriber_name, stream_key, last_event_id, last_business_event_id, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(subscriber_name, stream_key) DO UPDATE SET
            last_event_id = excluded.last_event_id,
            last_business_event_id = excluded.last_business_event_id,
            updated_at = excluded.updated_at
        """,
        (subscriber_name, stream_key, redis_stream_id, business_event_id, now),
    )


async def get_last_redis_id(
    db: aiosqlite.Connection,
    subscriber_name: str,
    stream_key: str,
) -> str:
    """Returns the Redis Stream entry ID where this subscriber left off; '0' if no record."""
    cursor = await db.execute(
        """
        SELECT last_event_id FROM last_processed_event
        WHERE subscriber_name=? AND stream_key=?
        """,
        (subscriber_name, stream_key),
    )
    row = await cursor.fetchone()
    return row[0] if row else "0"
