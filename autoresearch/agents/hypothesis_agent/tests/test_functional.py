"""
Functional test for HypothesisAgent: drive the REAL
``build_hypothesis_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

One flow, one happy-path test: hypothesis.generation_requested ->
hypothesis.generated.

针对 HypothesisAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_hypothesis_service``，只 mock LLM（``FakeLLM``），SpecStore 为
真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.hypothesis_agent.service import build_hypothesis_service
from autoresearch.agents.hypothesis_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


HYPOTHESIS_BANK_VALIDATOR = _validator("hypothesis_bank.schema.yaml")


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


# ---------------------------------------------------------------------------
# Canned shapes (mirrors tests/test_handler.py -- do not invent new shapes).
# ---------------------------------------------------------------------------

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

VALID_LITERATURE_BRIEF = {
    "summary": "The field combines dense retrieval with generative models to ground answers.",
    "related_work": [],
    "key_papers": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "id_or_url": "arxiv:2005.11401",
            "year": 2020,
        },
    ],
    "existing_benchmarks": [],
}


def _hypothesis(hid: str, kind: str = "idea") -> dict:
    return {
        "id": hid,
        "statement": f"Statement for {hid}: doing X improves Y by Z%.",
        "rationale": f"Rationale for {hid}, grounded in the literature brief.",
        "kind": kind,
        "verifiable": True,
        "expected_effect": "Metric improves by an estimated 5-10%.",
    }


VALID_HYPOTHESES = [
    _hypothesis("h1", "idea"),
    _hypothesis("h2", "training"),
]


def _bank_json(hypotheses: list[dict]) -> str:
    return json.dumps({"hypotheses": hypotheses})


# ---------------------------------------------------------------------------
# hypothesis.generation_requested -> hypothesis.generated
# ---------------------------------------------------------------------------


async def test_generation_requested_produces_hypothesis_generated(bus, store):
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])
    build_hypothesis_service(
        client=bus.view("HypothesisAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-hypothesis-1"
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    literature_brief_ref = store.put(
        kind="literature_brief",
        session_id=session_id,
        key="final",
        obj=VALID_LITERATURE_BRIEF,
        emitter="DeepResearchAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="hypothesis.generation_requested",
        session_id=session_id,
        payload={
            "research_spec_ref": research_spec_ref,
            "literature_brief_ref": literature_brief_ref,
        },
    )

    generated = [e for e in bus.published if e.event_type == "hypothesis.generated"]
    assert len(generated) == 1
    assert generated[0].payload["count"] == len(VALID_HYPOTHESES)

    bank = store.get(generated[0].payload["hypothesis_bank_ref"])
    errors = list(HYPOTHESIS_BANK_VALIDATOR.iter_errors(bank))
    assert errors == [], [e.message for e in errors]
    assert bank == {"hypotheses": VALID_HYPOTHESES}

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
