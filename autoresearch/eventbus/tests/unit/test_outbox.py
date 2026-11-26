"""Unit tests for autoresearch.eventbus.broker.outbox.

Uses real aiosqlite with a tmp_path-backed DB file (not mocked).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from autoresearch.eventbus.broker.outbox import (
    OutboxWorker,
    bump_failure,
    claim_due_pending,
    init_outbox,
    mark_sent,
    write_outbox,
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
async def outbox_db(sqlite_db):
    """sqlite_db with outbox schema initialized."""
    await init_outbox(sqlite_db)
    return sqlite_db


async def _outbox_columns(db) -> set[str]:
    cursor = await db.execute("PRAGMA table_info(outbox)")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def _row_by_id(db, row_id: int) -> dict:
    cursor = await db.execute(
        "SELECT id, stream_key, envelope_json, status, attempts, "
        "next_retry_at, last_error, created_at, sent_at "
        "FROM outbox WHERE id=?",
        (row_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    keys = [
        "id",
        "stream_key",
        "envelope_json",
        "status",
        "attempts",
        "next_retry_at",
        "last_error",
        "created_at",
        "sent_at",
    ]
    return dict(zip(keys, row))


async def test_init_outbox_creates_table(sqlite_db):
    await init_outbox(sqlite_db)
    cols = await _outbox_columns(sqlite_db)
    expected = {
        "id",
        "stream_key",
        "envelope_json",
        "status",
        "attempts",
        "next_retry_at",
        "last_error",
        "created_at",
        "sent_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


async def test_init_outbox_is_idempotent(sqlite_db):
    """Calling twice must not fail."""
    await init_outbox(sqlite_db)
    await init_outbox(sqlite_db)


async def test_write_outbox_inserts_pending_row(outbox_db):
    row_id = await write_outbox(outbox_db, "events:sess-1", '{"data":"x"}')
    await outbox_db.commit()
    assert isinstance(row_id, int) and row_id >= 1
    row = await _row_by_id(outbox_db, row_id)
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["stream_key"] == "events:sess-1"
    assert row["envelope_json"] == '{"data":"x"}'
    assert row["sent_at"] is None
    assert row["last_error"] is None
    # next_retry_at should be set to ~now (so the row is immediately due)
    assert abs(row["next_retry_at"] - time.time()) < 5


async def test_claim_due_pending_returns_due_rows(outbox_db):
    id1 = await write_outbox(outbox_db, "events:s1", '{"a":1}')
    id2 = await write_outbox(outbox_db, "events:s2", '{"a":2}')
    await outbox_db.commit()
    rows = await claim_due_pending(outbox_db)
    ids = [r[0] for r in rows]
    assert id1 in ids and id2 in ids


async def test_claim_due_pending_skips_future_retries(outbox_db):
    id1 = await write_outbox(outbox_db, "events:s1", '{"a":1}')
    id2 = await write_outbox(outbox_db, "events:s2", '{"a":2}')
    # Push id2's next_retry far into the future
    await outbox_db.execute(
        "UPDATE outbox SET next_retry_at=? WHERE id=?",
        (time.time() + 10_000, id2),
    )
    await outbox_db.commit()
    rows = await claim_due_pending(outbox_db)
    ids = [r[0] for r in rows]
    assert id1 in ids
    assert id2 not in ids


async def test_mark_sent_updates_status_and_sent_at(outbox_db):
    row_id = await write_outbox(outbox_db, "events:s1", '{"x":1}')
    await outbox_db.commit()
    await mark_sent(outbox_db, row_id)
    row = await _row_by_id(outbox_db, row_id)
    assert row["status"] == "sent"
    assert row["sent_at"] is not None
    assert abs(row["sent_at"] - time.time()) < 5


async def test_bump_failure_exponential_backoff_with_cap(outbox_db):
    row_id = await write_outbox(outbox_db, "events:s1", '{"x":1}')
    await outbox_db.commit()

    # attempts param represents current attempts (the *previous* attempt count
    # before the increment). bump_failure uses 2**attempts capped at 300.
    t0 = time.time()
    next_retry = await bump_failure(outbox_db, row_id, attempts=0, error="boom")
    # backoff = min(2**0, 300) = 1s
    assert 0.5 <= (next_retry - t0) <= 5.0, f"expected ~1s backoff, got {next_retry - t0}"

    # row state should now show attempts=1
    row = await _row_by_id(outbox_db, row_id)
    assert row["attempts"] == 1
    assert row["last_error"] == "boom"
    assert row["status"] == "pending"  # NOT 'sent'

    # call again with attempts=1 -> backoff = 2s
    t1 = time.time()
    next_retry = await bump_failure(outbox_db, row_id, attempts=1, error="boom2")
    assert 1.0 <= (next_retry - t1) <= 5.0  # ~2s
    row = await _row_by_id(outbox_db, row_id)
    assert row["attempts"] == 2

    # huge attempts -> capped at 300s
    t2 = time.time()
    next_retry = await bump_failure(outbox_db, row_id, attempts=20, error="cap")
    backoff = next_retry - t2
    assert backoff <= 305.0, f"backoff must be capped at ~300s, got {backoff}"
    assert backoff >= 295.0, f"backoff should be near 300s when capped, got {backoff}"


async def test_bump_failure_truncates_error_to_500_chars(outbox_db):
    row_id = await write_outbox(outbox_db, "events:s1", '{"x":1}')
    await outbox_db.commit()
    long_err = "E" * 1000
    await bump_failure(outbox_db, row_id, attempts=0, error=long_err)
    row = await _row_by_id(outbox_db, row_id)
    assert row["last_error"] is not None
    assert len(row["last_error"]) <= 500


# ----------------------------------------------------------------------------
# OutboxWorker
# ----------------------------------------------------------------------------


async def test_outbox_worker_drains_pending_to_sent(outbox_db):
    # Pre-load 3 pending rows
    ids = []
    for i in range(3):
        ids.append(
            await write_outbox(outbox_db, f"events:s{i}", '{"v":%d}' % i)
        )
    await outbox_db.commit()

    delivered: list[tuple[str, str]] = []

    async def xadd_ok(stream_key: str, envelope_json: str) -> None:
        delivered.append((stream_key, envelope_json))

    worker = OutboxWorker(outbox_db, xadd_ok, poll_interval_sec=0.01)
    worker.start()
    # Give the worker a moment to drain
    for _ in range(50):
        await asyncio.sleep(0.05)
        cursor = await outbox_db.execute(
            "SELECT COUNT(*) FROM outbox WHERE status='pending'"
        )
        (remaining,) = await cursor.fetchone()
        if remaining == 0:
            break
    await worker.stop()

    cursor = await outbox_db.execute(
        "SELECT COUNT(*) FROM outbox WHERE status='sent'"
    )
    (sent_count,) = await cursor.fetchone()
    assert sent_count == 3
    assert len(delivered) == 3


async def test_outbox_worker_failure_increments_attempts(outbox_db):
    row_id = await write_outbox(outbox_db, "events:s1", '{"x":1}')
    await outbox_db.commit()

    call_count = {"n": 0}

    async def xadd_fail(stream_key: str, envelope_json: str) -> None:
        call_count["n"] += 1
        raise RuntimeError("redis dead")

    worker = OutboxWorker(outbox_db, xadd_fail, poll_interval_sec=0.01)
    worker.start()
    # Wait briefly so worker picks the row at least once.
    # After one failure, next_retry_at is pushed ~1s into the future,
    # so further attempts within this window will be skipped.
    await asyncio.sleep(0.3)
    await worker.stop()

    assert call_count["n"] >= 1
    row = await _row_by_id(outbox_db, row_id)
    assert row["status"] == "pending"
    assert row["attempts"] >= 1
    assert row["last_error"] == "redis dead"
    # next_retry pushed into the future
    assert row["next_retry_at"] > time.time() - 0.1


async def test_outbox_worker_one_failing_row_does_not_block_batch(outbox_db):
    """A failing row must not prevent subsequent rows in the same batch from
    being delivered (each row is independent)."""
    # 5 rows, alternating success/fail by stream_key
    ids = []
    for i in range(5):
        ids.append(
            await write_outbox(outbox_db, f"events:s{i}", '{"v":%d}' % i)
        )
    await outbox_db.commit()

    delivered: list[str] = []

    async def xadd_partial(stream_key: str, envelope_json: str) -> None:
        if stream_key == "events:s2":
            raise RuntimeError("s2 broken")
        delivered.append(stream_key)

    worker = OutboxWorker(outbox_db, xadd_partial, poll_interval_sec=0.01)
    worker.start()
    # wait until 4 of 5 are sent (s2 will keep retrying)
    for _ in range(50):
        await asyncio.sleep(0.05)
        cursor = await outbox_db.execute(
            "SELECT COUNT(*) FROM outbox WHERE status='sent'"
        )
        (n_sent,) = await cursor.fetchone()
        if n_sent >= 4:
            break
    await worker.stop()

    cursor = await outbox_db.execute(
        "SELECT COUNT(*) FROM outbox WHERE status='sent'"
    )
    (n_sent,) = await cursor.fetchone()
    assert n_sent == 4, (
        f"expected the 4 non-broken rows delivered, got {n_sent}; "
        f"delivered={delivered}"
    )

    # s2 is still pending with attempts>=1
    cursor = await outbox_db.execute(
        "SELECT status, attempts FROM outbox WHERE stream_key='events:s2'"
    )
    row = await cursor.fetchone()
    assert row[0] == "pending"
    assert row[1] >= 1


async def test_outbox_worker_handles_100_pending_rows(outbox_db):
    """Worker should drain 100 rows in a reasonable time."""
    for i in range(100):
        await write_outbox(outbox_db, f"events:s{i % 10}", '{"v":%d}' % i)
    await outbox_db.commit()

    async def xadd_ok(stream_key: str, envelope_json: str) -> None:
        return None

    worker = OutboxWorker(outbox_db, xadd_ok, poll_interval_sec=0.01)
    worker.start()
    for _ in range(100):  # up to 5s
        await asyncio.sleep(0.05)
        cursor = await outbox_db.execute(
            "SELECT COUNT(*) FROM outbox WHERE status='pending'"
        )
        (remaining,) = await cursor.fetchone()
        if remaining == 0:
            break
    await worker.stop()

    cursor = await outbox_db.execute(
        "SELECT COUNT(*) FROM outbox WHERE status='sent'"
    )
    (n_sent,) = await cursor.fetchone()
    assert n_sent == 100
