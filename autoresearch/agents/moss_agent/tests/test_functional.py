"""
Functional test for MOSS: drive the REAL ``build_moss_service`` on a shared
``DeliveringBus`` (real subscribe/dispatch routing, not a hand-called handler
method), with only the LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

针对 MOSS 的功能测试：在共享 ``DeliveringBus``（真实 subscribe/dispatch 路由，
而非直接调用 handler 方法）上跑真实的 ``build_moss_service``，只 mock LLM
（``FakeLLM``），SpecStore 为真实的临时实例。

One flow: ``moss.intervention_requested{reason:"direction_thrashing"}`` ->
``moss.intervention_completed`` with a ``moss_intervention_ref`` that resolves
to a schema-valid MossDecision.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

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
    conn = await aiosqlite.connect(str(tmp_path / "functional.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


async def test_intervention_requested_produces_intervention_completed(bus, store):
    llm = FakeLLM([json.dumps({"action": "rewrite_framing", "rationale": "thrashed twice"})])
    build_moss_service(client=bus.view("MOSS"), spec_store=store, llm_client=llm)

    session_id = "sess-moss-functional-1"
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
    assert decision["action"] == "rewrite_framing"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
