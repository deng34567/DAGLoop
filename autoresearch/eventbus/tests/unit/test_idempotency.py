"""Unit tests for autoresearch.eventbus.broker.idempotency."""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.eventbus.broker.idempotency import (
    get_last_redis_id,
    has_processed,
    init_idempotency,
    record_processed,
)


@pytest.fixture
async def sqlite_db(tmp_path: Path):
    """An open aiosqlite.Connection backed by a real file under tmp_path."""
    import aiosqlite
    db_path = tmp_path / "broker_test.db"
    conn = await aiosqlite.connect(str(db_path))
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
async def idem_db(sqlite_db):
    """sqlite_db with idempotency schema initialized."""
    await init_idempotency(sqlite_db)
    return sqlite_db


async def _tables(db) -> set[str]:
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    rows = await cursor.fetchall()
    return {r[0] for r in rows}


async def test_init_idempotency_creates_both_tables(sqlite_db):
    await init_idempotency(sqlite_db)
    tables = await _tables(sqlite_db)
    assert "last_processed_event" in tables
    assert "processed_event_ids" in tables


async def test_init_idempotency_is_idempotent(sqlite_db):
    await init_idempotency(sqlite_db)
    await init_idempotency(sqlite_db)  # should not raise


async def test_has_processed_false_initially(idem_db):
    assert await has_processed(idem_db, "CoderAgent", "event-abc") is False


async def test_record_then_has_processed_returns_true(idem_db):
    await record_processed(
        idem_db,
        subscriber_name="CoderAgent",
        business_event_id="event-abc",
        stream_key="events:s1",
        redis_stream_id="1700000000000-0",
    )
    await idem_db.commit()
    assert await has_processed(idem_db, "CoderAgent", "event-abc") is True


async def test_record_processed_is_idempotent_on_same_event(idem_db):
    """Second record_processed with same (subscriber, business_event_id)
    must not raise (INSERT OR IGNORE)."""
    await record_processed(
        idem_db, "CoderAgent", "event-abc", "events:s1", "1700000000000-0"
    )
    await idem_db.commit()
    # second call -- must NOT raise IntegrityError
    await record_processed(
        idem_db, "CoderAgent", "event-abc", "events:s1", "1700000000001-0"
    )
    await idem_db.commit()
    # still processed
    assert await has_processed(idem_db, "CoderAgent", "event-abc") is True


async def test_get_last_redis_id_returns_latest(idem_db):
    await record_processed(
        idem_db, "CoderAgent", "event-1", "events:s1", "1700000000000-0"
    )
    await idem_db.commit()
    await record_processed(
        idem_db, "CoderAgent", "event-2", "events:s1", "1700000000005-0"
    )
    await idem_db.commit()
    last = await get_last_redis_id(idem_db, "CoderAgent", "events:s1")
    assert last == "1700000000005-0"


async def test_get_last_redis_id_default_zero_when_no_record(idem_db):
    last = await get_last_redis_id(idem_db, "CoderAgent", "events:nonexistent")
    assert last == "0"


async def test_subscriber_isolation(idem_db):
    """Subscriber A processing an event should not mark it processed for
    subscriber B."""
    await record_processed(
        idem_db, "CoderAgent", "shared-event-id", "events:s1", "1700000000000-0"
    )
    await idem_db.commit()
    assert await has_processed(idem_db, "CoderAgent", "shared-event-id") is True
    assert await has_processed(idem_db, "CriticAgent", "shared-event-id") is False


async def test_subscriber_isolation_on_last_redis_id(idem_db):
    await record_processed(
        idem_db, "CoderAgent", "ev-1", "events:s1", "1700000000000-0"
    )
    await idem_db.commit()
    # CriticAgent has no record for the same stream
    assert (
        await get_last_redis_id(idem_db, "CriticAgent", "events:s1")
        == "0"
    )
    assert (
        await get_last_redis_id(idem_db, "CoderAgent", "events:s1")
        == "1700000000000-0"
    )
