"""Tests for the shared in-memory ``DeliveringBus`` test harness.

Mirrors the semantics of the ``DeliveringBus``/``BusClientView`` copies found
in ``autoresearch/orchestrator/tests/smoke/test_m7_verifier_smoke.py``.
"""

from __future__ import annotations

import json

import aiosqlite
import pytest

from autoresearch.testing.delivering_bus import BusClientView, DeliveringBus


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "delivering_bus_test.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


async def test_subscriber_receives_matching_event(bus):
    received = []

    async def handler(env):
        received.append(env)

    bus.view("A").subscribe("x.y", handler)

    event_id = await bus.view("B").publish(
        event_type="x.y", session_id="s1", payload={"k": "v"}
    )

    assert len(received) == 1
    assert received[0].event_id == event_id
    assert len(bus.published) == 1
    published = bus.published[0]
    assert published.event_type == "x.y"
    assert published.emitter == "B"
    assert published.payload == {"k": "v"}
    assert received[0] is published


async def test_subscriber_does_not_receive_non_matching_event(bus):
    received = []

    async def handler(env):
        received.append(env)

    bus.view("A").subscribe("z.*", handler)

    await bus.view("B").publish(event_type="x.y", session_id="s1", payload={})

    assert received == []
    assert len(bus.published) == 1


async def test_fnmatch_wildcard_subscription(bus):
    received = []

    async def handler(env):
        received.append(env)

    bus.view("A").subscribe("x.*", handler)

    await bus.view("B").publish(event_type="x.y", session_id="s1", payload={})

    assert len(received) == 1
    assert received[0].event_type == "x.y"


async def test_view_returns_bus_client_view(bus):
    view = bus.view("A")
    assert isinstance(view, BusClientView)
    assert view.emitter_name == "A"


async def test_reentrant_publish_from_handler(bus):
    received = []

    async def handler_x(env):
        received.append(env.event_type)
        if env.event_type == "x.y":
            await bus.view("A").publish(event_type="x.z", session_id="s1", payload={})

    async def handler_z(env):
        received.append(env.event_type)

    bus.view("A").subscribe("x.*", handler_x)
    bus.view("A").subscribe("x.z", handler_z)

    await bus.view("B").publish(event_type="x.y", session_id="s1", payload={})

    assert received.count("x.y") == 1
    assert received.count("x.z") == 2
    assert len(bus.published) == 2


async def test_long_reentrant_chain_is_drained_without_recursion(bus):
    seen = []
    limit = 1200

    async def handler(env):
        step = env.payload["step"]
        seen.append(step)
        if step < limit:
            await bus.view("A").publish(
                event_type="chain.step",
                session_id="s1",
                payload={"step": step + 1},
                auto_commit=False,
            )

    bus.view("A").subscribe("chain.step", handler)

    await bus.view("B").publish(
        event_type="chain.step", session_id="s1", payload={"step": 0}
    )

    assert seen == list(range(limit + 1))
    assert len(bus.published) == limit + 1


async def test_failed_handoff_survives_new_bus_without_replaying_success(bus):
    calls = []
    failed = True
    async def producer(env):
        calls.append("experiment")
        await bus.publish(event_type="result.ready", session_id="s1", payload={"ref": "saved"}, emitter="runner")
    async def handoff(env):
        nonlocal failed
        calls.append("handoff")
        assert env.payload["ref"] == "saved"
        if failed:
            failed = False
            raise RuntimeError("database is locked")
    bus.subscribe("experiment.run", producer)
    bus.subscribe("result.ready", handoff)
    with pytest.raises(RuntimeError, match="locked"):
        await bus.publish(event_type="experiment.run", session_id="s1", payload={}, emitter="engine")
    recovered = DeliveringBus(bus.db)
    recovered.subscribe("experiment.run", producer)
    recovered.subscribe("result.ready", handoff)
    assert await recovered.replay_pending("s1") is True
    assert calls == ["experiment", "handoff", "handoff"]
    assert await recovered.replay_pending("s1") is False


async def test_restart_replay_marks_pending_experiment_for_remote_recovery(bus):
    await bus.db.execute("""CREATE TABLE session_state (
        session_id TEXT PRIMARY KEY,
        state_data TEXT NOT NULL,
        current_state TEXT NOT NULL,
        updated_at REAL NOT NULL,
        created_at REAL NOT NULL
    )""")
    await bus.db.execute(
        "INSERT INTO session_state VALUES (?, ?, ?, ?, ?)",
        (
            "s1",
            json.dumps({
                "active_request": {
                    "event_id": "pending-run",
                    "event_type": "experiment.run_requested",
                    "state": "RunningExperiment",
                    "payload": {
                        "experiment_patch_ref": "patch",
                        "recover_completed_result": False,
                    },
                    "consumed": False,
                },
            }),
            "RunningExperiment",
            1.0,
            1.0,
        ),
    )
    await bus.db.commit()

    async def interrupted(_env):
        raise RuntimeError("controller restarted")

    bus.subscribe("experiment.run_requested", interrupted)
    with pytest.raises(RuntimeError, match="controller restarted"):
        await bus.publish(
            event_type="experiment.run_requested", session_id="s1",
            payload={"experiment_patch_ref": "patch"}, emitter="engine",
            event_id="pending-run")

    recovered = DeliveringBus(bus.db)
    seen = []

    async def runner(env):
        seen.append(env)

    recovered.subscribe("experiment.run_requested", runner)
    assert await recovered.replay_pending("s1", restart_recovery=True) is True
    assert len(seen) == 1
    assert seen[0].payload == {
        "experiment_patch_ref": "patch", "recover_completed_result": True}
    cursor = await bus.db.execute(
        "SELECT state_data FROM session_state WHERE session_id='s1'")
    persisted = json.loads((await cursor.fetchone())[0])
    assert persisted["active_request"]["payload"] == seen[0].payload
    assert await recovered.replay_pending("s1") is False
