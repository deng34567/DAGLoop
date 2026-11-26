"""
Tests for MossObserver — MOSS Mode B cross-session pattern detection.

No Redis / network / EventBus / real LLM — a tmp MossObservationStore plus
Envelopes constructed directly via EnvelopeBuilder. The load-bearing
properties: (1) on_event NEVER raises (non-blocking for the bus), (2) the
detection gate counts DISTINCT sessions (not raw signal count), (3) the L1
auto-patch hook is a no-op even when AUTORESEARCH_MOSS_MODEB_AUTOPATCH=1.
"""

from __future__ import annotations

import pytest

from autoresearch.agents.moss_agent.observation_store import MossObservationStore
from autoresearch.agents.moss_agent.observer import MossObserver
from autoresearch.eventbus.broker.envelope import EnvelopeBuilder


def _agent_failed_env(session_id: str, **payload_overrides):
    payload = {
        "agent_name": "CoderAgent",
        "reason_classification": "permanent",
        "underlying_reason": "permanent",
    }
    payload.update(payload_overrides)
    return EnvelopeBuilder(emitter="CoderAgent").build(
        event_type="agent.failed",
        session_id=session_id,
        payload=payload,
    )


def _intervention_completed_env(session_id: str, **payload_overrides):
    payload = {
        "reason": "protocol_violation",
        "action": "rollback_patch",
        "degraded": False,
    }
    payload.update(payload_overrides)
    return EnvelopeBuilder(emitter="ResearchOrchestrator").build(
        event_type="moss.intervention_completed",
        session_id=session_id,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_below_threshold_no_alert(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store)

    for i in range(4):
        await observer.on_event(_agent_failed_env(f"s{i}"))

    assert store.list_alerts() == []
    store.close()


@pytest.mark.asyncio
async def test_fifth_distinct_session_triggers_one_l3_alert(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store)

    for i in range(5):
        await observer.on_event(_agent_failed_env(f"s{i}"))

    alerts = store.list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["alert_kind"] == "recurring_pattern"
    assert alerts[0]["detail"]["distinct_sessions"] == 5
    store.close()


@pytest.mark.asyncio
async def test_same_session_repeated_no_alert(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store)

    for _ in range(5):
        await observer.on_event(_agent_failed_env("same-session"))

    assert store.list_alerts() == []
    assert store.distinct_sessions("CoderAgent", "permanent") == 1
    store.close()


@pytest.mark.asyncio
async def test_intervention_completed_recorded_and_alerts(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store)

    for i in range(5):
        await observer.on_event(_intervention_completed_env(f"s{i}"))

    assert (
        store.signal_count("ResearchOrchestrator", "moss_intervention:protocol_violation")
        == 5
    )
    alerts = store.list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["alert_kind"] == "recurring_pattern"
    store.close()


@pytest.mark.asyncio
async def test_missing_agent_name_swallowed_no_signal(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store)

    env = EnvelopeBuilder(emitter="CoderAgent").build(
        event_type="agent.failed",
        session_id="s-missing",
        payload={"reason_classification": "permanent"},  # no agent_name
    )

    await observer.on_event(env)  # must not raise

    assert store.signal_count("CoderAgent", "permanent") == 0
    store.close()


@pytest.mark.asyncio
async def test_unclassified_event_ignored(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store)

    env = EnvelopeBuilder(emitter="HypothesisAgent").build(
        event_type="hypothesis.generated",
        session_id="s-x",
        payload={"anything": 1},
    )

    await observer.on_event(env)  # must not raise

    assert store.list_alerts() == []
    store.close()


@pytest.mark.asyncio
async def test_parameterized_threshold(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store, threshold_sessions=3)

    for i in range(2):
        await observer.on_event(_agent_failed_env(f"t{i}"))
    assert store.list_alerts() == []

    await observer.on_event(_agent_failed_env("t2"))
    alerts = store.list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["detail"]["distinct_sessions"] == 3
    store.close()


@pytest.mark.asyncio
async def test_autopatch_env_is_still_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_MOSS_MODEB_AUTOPATCH", "1")
    store = MossObservationStore(tmp_path / "moss.db")
    observer = MossObserver(store, threshold_sessions=3)

    for i in range(3):
        await observer.on_event(_agent_failed_env(f"ap{i}"))

    alerts = store.list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["alert_kind"] == "recurring_pattern"
    store.close()
