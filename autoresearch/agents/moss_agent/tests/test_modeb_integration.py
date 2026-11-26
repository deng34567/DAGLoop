"""
Integration test for MOSS Mode B (Observe-Evolve), wired through the REAL
``build_moss_service`` on a shared ``DeliveringBus`` -- real subscribe/
dispatch routing (not a hand-called observer method), only the LLM mocked
(``FakeLLM``), a real tmp ``SpecStore`` + a real tmp ``MossObservationStore``.

针对 MOSS Mode B 的集成测试：在共享 ``DeliveringBus``（真实 subscribe/
dispatch 路由）上跑真实的 ``build_moss_service``，只 mock LLM，SpecStore /
MossObservationStore 均为真实的临时实例。

Proves, in one wiring:
1. Cross-session recurrence: 3 ``agent.failed`` events for
   ``(CoderAgent, permanent)`` from 3 DISTINCT sessions raises an L3
   ``recurring_pattern`` alert (``modeb_threshold=3``).
2. Mode A is unbroken by the additive Mode-B subscriptions: a
   ``moss.intervention_requested`` still produces exactly one
   ``moss.intervention_completed`` -- and that completed event is ITSELF
   observed (Mode B records the MOSS-internal signal via the same bus).
3. Mode B emits nothing -- the only events on ``bus.published`` are the
   Mode-A ones the handler publishes; the observer never calls
   ``client.publish``.

NOTE on the DeliveringBus's non-isolation: ``DeliveringBus.publish`` calls
each subscribed handler directly with no per-handler ``try/except`` (see
``autoresearch/testing/delivering_bus.py``) -- unlike a production
dispatcher, it does NOT shield one subscriber's exception from breaking
delivery to the rest, or from bubbling up to the publisher. That means
``MossObserver.on_event``'s own internal swallow (a bare ``try/except`` in
``observer.py``) is the ONLY thing standing between a malformed
``agent.failed`` payload and a broken bus in this harness. Test 4 below
exercises exactly that: a malformed ``agent.failed`` (missing
``agent_name``) delivered through the real wired service must not raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.moss_agent.observation_store import MossObservationStore
from autoresearch.agents.moss_agent.service import build_moss_service
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


MOSS_DECISION_VALIDATOR = _validator("moss_decision.schema.yaml")


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "modeb_integration.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


async def test_cross_session_alert_mode_a_unbroken_and_observed_emits_nothing(
    bus, store, tmp_path
):
    observation_db_path = tmp_path / "moss.db"
    llm = FakeLLM([json.dumps({"action": "rewrite_framing", "rationale": "thrashed twice"})])

    handler = build_moss_service(
        client=bus.view("MOSS"),
        spec_store=store,
        llm_client=llm,
        observation_db_path=observation_db_path,
        modeb_threshold=3,
    )
    assert handler is not None  # return shape unchanged: still the Mode-A handler

    # --- (1) cross-session L3 alert: 3 agent.failed from 3 DISTINCT sessions.
    # underlying_reason is included -- it's schema-required, and the real bus
    # (EnvelopeBuilder/validated publish path this harness mirrors) validates
    # against it.
    for i in range(3):
        await bus.view("CoderAgent").publish(
            event_type="agent.failed",
            session_id=f"sess-modeb-{i}",
            payload={
                "agent_name": "CoderAgent",
                "reason_classification": "permanent",
                "underlying_reason": "permanent",
            },
        )

    # Read back via a fresh handle onto the SAME sqlite file (build_moss_service
    # doesn't hand back the store/observer -- the return shape is left as-is
    # per Mode A's existing contract).
    read_store = MossObservationStore(observation_db_path)
    try:
        assert read_store.distinct_sessions("CoderAgent", "permanent") == 3
        alerts = read_store.list_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_kind"] == "recurring_pattern"
        assert alerts[0]["detail"]["agent"] == "CoderAgent"
        assert alerts[0]["detail"]["reason"] == "permanent"
        assert alerts[0]["detail"]["distinct_sessions"] == 3
    finally:
        read_store.close()

    # --- (2) Mode A still works, and is itself observed by Mode B.
    session_id = "sess-modeb-modea"
    await bus.view("ResearchOrchestrator").publish(
        event_type="moss.intervention_requested",
        session_id=session_id,
        payload={"reason": "direction_thrashing"},
    )

    completed = [e for e in bus.published if e.event_type == "moss.intervention_completed"]
    assert len(completed) == 1
    payload = completed[0].payload
    assert payload["reason"] == "direction_thrashing"
    assert payload["action"] == "rewrite_framing"
    assert payload["degraded"] is False

    decision = store.get(payload["moss_intervention_ref"])
    errors = list(MOSS_DECISION_VALIDATOR.iter_errors(decision))
    assert errors == [], [e.message for e in errors]

    # No agent.failed FROM MOSS ITSELF (i.e. Mode A didn't fail) -- the 3
    # agent.failed events already on the bus are the ones WE published in
    # step (1) above, from the CoderAgent emitter, as Mode B's raw material.
    assert not [
        e for e in bus.published if e.event_type == "agent.failed" and e.emitter == "MOSS"
    ]

    # The moss.intervention_completed above was delivered to the observer
    # (subscribed on the same client) and recorded as a
    # moss_intervention:direction_thrashing signal.
    read_store = MossObservationStore(observation_db_path)
    try:
        assert (
            read_store.signal_count("ResearchOrchestrator", "moss_intervention:direction_thrashing")
            == 1
        )
    finally:
        read_store.close()

    # --- (3) Mode B emits nothing: the observer is never given a client
    # handle (its on_event(env) signature takes no client -- see
    # observer.py), so it structurally cannot call publish. Confirm the
    # bus's published list contains ONLY what WE (the test) and Mode A
    # published: the 3 agent.failed + 1 moss.intervention_requested from the
    # test, plus the 1 moss.intervention_completed from Mode A -- nothing
    # observer-originated.
    assert len(bus.published) == 5
    assert {e.emitter for e in bus.published} == {"CoderAgent", "ResearchOrchestrator", "MOSS"}
    assert {e.event_type for e in bus.published} == {
        "agent.failed",
        "moss.intervention_requested",
        "moss.intervention_completed",
    }


async def test_malformed_agent_failed_does_not_break_the_bus(bus, store, tmp_path):
    """Regression guard for the DeliveringBus's non-isolation (see module
    docstring): a malformed agent.failed (missing agent_name) delivered
    through the REAL wired service must not raise -- MossObserver.on_event's
    own swallow is the only thing protecting this harness's bus."""
    observation_db_path = tmp_path / "moss.db"
    llm = FakeLLM([])

    build_moss_service(
        client=bus.view("MOSS"),
        spec_store=store,
        llm_client=llm,
        observation_db_path=observation_db_path,
        modeb_threshold=3,
    )

    # Must not raise, despite missing agent_name (the classifier does
    # payload["agent_name"] -- a KeyError swallowed inside on_event).
    await bus.view("CoderAgent").publish(
        event_type="agent.failed",
        session_id="sess-malformed",
        payload={"reason_classification": "permanent"},
    )

    read_store = MossObservationStore(observation_db_path)
    try:
        assert read_store.list_alerts() == []
    finally:
        read_store.close()
