"""Concurrency regression: a handler's synchronous LLM call must NOT block the
shared web event loop, or one session's slow intervention starves every other
session (the measured 3-way-concurrency starvation, item ④).

MOSS is the representative offender: ``handle_intervention_requested`` runs
``core.decide(..., self.llm_client, ...)`` -- a synchronous, blocking LLM call.
On the web ``DeliveringBus`` (which dispatches ``await handler(env)`` inline on
one loop) that call, if left on the loop, serializes every session. Wrapping it
in ``asyncio.to_thread`` (the same offload ``_adapter/broker_bus.py`` already
does for the multi-process path) lets sibling sessions proceed in parallel.

This test drives the REAL ``build_moss_service`` on a shared ``DeliveringBus``
with a deliberately slow (synchronous) LLM and fires two interventions for two
sessions concurrently. If the blocking call runs on the loop, wall-time is
~2*BLOCK (serialized); if it is offloaded, ~BLOCK (interleaved).
"""
from __future__ import annotations

import asyncio
import json
import time

import aiosqlite
import pytest

from autoresearch.agents.moss_agent.service import build_moss_service
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

BLOCK = 0.4  # seconds each synchronous "LLM call" blocks


class SlowLLM(FakeLLM):
    """A FakeLLM whose ``chat`` blocks synchronously, standing in for a real
    (blocking) LLM round-trip."""

    def chat(self, **kwargs):
        time.sleep(BLOCK)
        return super().chat(**kwargs)


async def test_concurrent_moss_interventions_do_not_starve(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "concurrency.db"))
    try:
        bus = DeliveringBus(conn)
        store = SpecStore(tmp_path / "store")
        llm = SlowLLM([
            json.dumps({"action": "rewrite_framing", "rationale": "a"}),
            json.dumps({"action": "rewrite_framing", "rationale": "b"}),
        ])
        build_moss_service(client=bus.view("MOSS"), spec_store=store, llm_client=llm)
        orch = bus.view("ResearchOrchestrator")

        async def fire(session_id: str) -> None:
            await orch.publish(
                event_type="moss.intervention_requested",
                session_id=session_id,
                payload={"reason": "direction_thrashing"},
            )

        start = time.monotonic()
        await asyncio.gather(fire("sess-a"), fire("sess-b"))
        elapsed = time.monotonic() - start

        completed = [e for e in bus.published if e.event_type == "moss.intervention_completed"]
        assert len(completed) == 2
        assert not [e for e in bus.published if e.event_type == "agent.failed"]
        # Interleaved (~BLOCK), not serialized (~2*BLOCK). The 1.7x threshold leaves
        # generous headroom for thread-scheduling jitter while still failing loudly
        # if the blocking call runs on the loop.
        assert elapsed < BLOCK * 1.7, (
            f"MOSS interventions serialized ({elapsed:.2f}s ~= 2*{BLOCK}s): the "
            f"synchronous LLM call is blocking the shared event loop."
        )
        store.close()
    finally:
        await conn.close()
