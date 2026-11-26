"""Failed delivery must preserve Redis pending state and completed siblings."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from autoresearch.eventbus.broker.client import EventBusClient
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.eventbus.broker.idempotency import init_idempotency, has_processed


async def test_failed_subscriber_is_not_acked_and_successful_sibling_is_not_repeated(tmp_path):
    client = EventBusClient('redis://unused', str(tmp_path / 'events.db'), 'offline', Path('.'))
    client.db = await aiosqlite.connect(str(tmp_path / 'events.db'))
    await init_idempotency(client.db)
    client.redis = SimpleNamespace(xack=AsyncMock())
    client.registry = SimpleNamespace(envelope_validator=SimpleNamespace(iter_errors=lambda _: []))
    calls = []
    failed = True
    async def completed(env):
        calls.append('measured')
    async def handoff(env):
        nonlocal failed
        calls.append('handoff')
        if failed:
            failed = False
            raise RuntimeError('database is locked')
    client.subscribe('*', completed)
    client.subscribe('*', handoff)
    env = Envelope('event-1', 'session-1', 'offline.result', 'runner', '2026-10-04T00:00:00Z', 1, '0', {})
    try:
        with pytest.raises(RuntimeError, match='locked'):
            await client._handle_one('stream', '1-0', env.to_redis_fields())
        assert not await has_processed(client.db, 'offline', env.event_id)
        assert client.redis.xack.await_count == 0
        await client._handle_one('stream', '1-0', env.to_redis_fields())
        assert calls == ['measured', 'handoff', 'handoff']
        assert await has_processed(client.db, 'offline', env.event_id)
        assert client.redis.xack.await_count == 1
    finally:
        await client.db.close()
