"""
Functional test for DeepResearchAgent: drive the REAL
``build_deep_research_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
search backend (``FakeBackend``) and LLM (``FakeLLM``) mocked, and a real
tmp ``SpecStore``.

Flow: deep_research.requested -> literature.brief_ready, with BOTH a
literature_brief and a field_trajectory_brief persisted and schema-valid.

针对 DeepResearchAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由）上跑真实的 ``build_deep_research_service``，只 mock
搜索后端与 LLM，SpecStore 为真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.deep_research_backend import FakeBackend
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


LITERATURE_BRIEF_VALIDATOR = _validator("literature_brief.schema.yaml")
FIELD_TRAJECTORY_BRIEF_VALIDATOR = _validator("field_trajectory_brief.schema.yaml")


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


VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Survey RAG methods for long-tail scientific question answering.",
    "task_mode": "survey",
    "budget": "one week",
    "deliverable_form": "literature review",
    "editable_scope": [],
    "forbidden_items": [],
    "open_questions": [],
}

# Matches FakeBackend's default seminal + frontier papers, so key_papers is
# grounded (see core.py's anti-hallucination check).
SEMINAL_PAPER = {
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "id_or_url": "arxiv:2005.11401",
    "year": 2020,
    "note": "Seminal RAG paper.",
}
FRONTIER_PAPER = {
    "title": "Long-Tail Scientific QA via Retrieval-Augmented Fine-Tuning",
    "id_or_url": "arxiv:2401.00000",
    "year": 2024,
    "note": "Recent frontier work on long-tail scientific QA.",
}


def _valid_synthesis_json() -> str:
    return json.dumps(
        {
            "literature_brief": {
                "summary": "RAG combines dense retrieval with generation.",
                "related_work": [],
                "key_papers": [SEMINAL_PAPER, FRONTIER_PAPER],
                "existing_benchmarks": [],
            },
            "field_trajectory_brief": {
                "origin": "RAG originated from combining retrieval with seq2seq generation.",
                "current_state": "Frontier work extends RAG to long-tail scientific QA.",
                "main_method_families": ["dense retrieval", "hybrid retrieval"],
                "open_problems": ["long-tail factual accuracy"],
            },
        }
    )


async def test_deep_research_requested_produces_both_briefs(bus, store):
    backend = FakeBackend()
    # keyword-planning response (now always runs before retrieval) + synthesis response
    llm = FakeLLM([json.dumps({"query": "retrieval augmented generation scientific QA"}),
                  _valid_synthesis_json()])
    build_deep_research_service(
        client=bus.view("DeepResearchAgent"),
        spec_store=store,
        backend=backend,
        llm_client=llm,
    )

    session_id = "sess-deep-research-1"
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="deep_research.requested",
        session_id=session_id,
        payload={"research_spec_ref": research_spec_ref},
    )

    ready = [e for e in bus.published if e.event_type == "literature.brief_ready"]
    assert len(ready) == 1
    payload = ready[0].payload
    assert set(payload.keys()) == {"literature_brief_ref", "field_trajectory_brief_ref"}

    literature_brief = store.get(payload["literature_brief_ref"])
    lb_errors = list(LITERATURE_BRIEF_VALIDATOR.iter_errors(literature_brief))
    assert lb_errors == [], [e.message for e in lb_errors]

    field_trajectory_brief = store.get(payload["field_trajectory_brief_ref"])
    ftb_errors = list(FIELD_TRAJECTORY_BRIEF_VALIDATOR.iter_errors(field_trajectory_brief))
    assert ftb_errors == [], [e.message for e in ftb_errors]

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
