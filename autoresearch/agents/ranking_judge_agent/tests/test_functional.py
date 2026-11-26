"""
Functional test for RankingJudgeAgent: drive the REAL
``build_ranking_judge_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

One flow, happy-path: ranking.requested -> ranking.ready.

针对 RankingJudgeAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_ranking_judge_service``，只 mock LLM（``FakeLLM``），SpecStore 为
真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.ranking_judge_agent.service import build_ranking_judge_service
from autoresearch.agents.ranking_judge_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


RANKING_REPORT_VALIDATOR = _validator("ranking_report.schema.yaml")


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


VALID_HYPOTHESIS_BANK = {
    "hypotheses": [
        {
            "id": "H1",
            "statement": "Dense retrieval with a reranker improves long-tail QA accuracy.",
            "kind": "idea",
            "verifiable": True,
        },
        {
            "id": "H2",
            "statement": "Sparse retrieval alone is sufficient for long-tail QA.",
            "kind": "idea",
            "verifiable": True,
        },
        {
            "id": "H3",
            "statement": "Hybrid retrieval outperforms both dense-only and sparse-only.",
            "kind": "idea",
            "verifiable": True,
        },
    ]
}

VALID_RANKING_LLM_RESPONSE = {
    "pairwise_results": [
        {"a": "H1", "b": "H2", "winner": "H1"},
        {"a": "H1", "b": "H3", "winner": "H3"},
        {"a": "H2", "b": "H3", "winner": "H3"},
    ],
    "diversity_status": "diverse",
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


async def test_ranking_requested_produces_ready(bus, store):
    llm = FakeLLM([_json(VALID_RANKING_LLM_RESPONSE)])
    build_ranking_judge_service(
        client=bus.view("RankingJudgeAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-ranking-judge-1"
    bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=session_id,
        key="final",
        obj=VALID_HYPOTHESIS_BANK,
        emitter="HypothesisAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="ranking.requested",
        session_id=session_id,
        payload={"candidate_kind": "hypothesis", "source_ref": bank_ref},
    )

    ready = [e for e in bus.published if e.event_type == "ranking.ready"]
    assert len(ready) == 1
    assert ready[0].payload["candidate_kind"] == "hypothesis"

    report = store.get(ready[0].payload["ranking_report_ref"])
    errors = list(RANKING_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert set(report["ranked_shortlist"]) == {"H1", "H2", "H3"}

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
