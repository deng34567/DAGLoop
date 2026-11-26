"""
Discovery: periodically SCAN for `events:*` streams so the subscriber can pick up
new sessions started after this process began.

V1 simple approach: every SESSION_DISCOVERY_INTERVAL_SEC, SCAN for matching keys
and emit a callback for any newly-seen stream. V1.5+ can upgrade to Redis
keyspace notifications for millisecond-level latency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SessionDiscovery:
    def __init__(
        self,
        redis: Redis,
        on_new_stream: Callable[[str], Awaitable[None]],
        pattern: str = "events:*",
        interval_sec: float = 5.0,
    ):
        self.redis = redis
        self.on_new_stream = on_new_stream
        self.pattern = pattern
        self.interval_sec = interval_sec
        self._known: set[str] = set()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def scan_now(self) -> set[str]:
        """One-shot SCAN. Returns the set of newly discovered streams."""
        current: set[str] = set()
        async for key in self.redis.scan_iter(match=self.pattern):
            # Skip Redis Streams that store internal metadata (none currently)
            if isinstance(key, bytes):
                key = key.decode()
            current.add(key)
        new = current - self._known
        self._known = current
        for stream in sorted(new):
            try:
                await self.on_new_stream(stream)
            except Exception as e:
                logger.error("on_new_stream(%s) failed: %s", stream, e)
        return new

    async def _run(self) -> None:
        # Initial scan
        await self.scan_now()
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)
            except asyncio.TimeoutError:
                await self.scan_now()
