"""
Functional test for NoveltyAuditAgent: drive the REAL
``build_novelty_audit_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

One flow, one happy-path test: novelty.audit_requested -> novelty.audit_ready.

针对 NoveltyAuditAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_novelty_audit_service``，只 mock LLM（``FakeLLM``），SpecStore 为
真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


NOVELTY_AUDIT_REPORT_VALIDATOR = _validator("novelty_audit_report.schema.yaml")


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

VALID_SPEC_INTERPRETATION = {
    "disposition": "in_scope",
    "real_open_problem": "Long-tail scientific QA remains under-addressed by existing RAG methods.",
    "scope_reason": "The proposed direction remains within the original research scope.",
}

VALID_REPORT_PASS = {
    "verdict": "pass",
    "citation_sufficiency": "Citations are sufficient and current for the proposed direction.",
    "stale_citations": [],
    "unsupported_claims": [],
    "missing_work": [],
    "novelty_status": "novel",
    "notes": "No critical gaps found.",
}


def _report_json(report: dict) -> str:
    return json.dumps(report)


# ---------------------------------------------------------------------------
# novelty.audit_requested -> novelty.audit_ready
# ---------------------------------------------------------------------------


async def test_audit_requested_produces_audit_ready(bus, store):
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])
    build_novelty_audit_service(
        client=bus.view("NoveltyAuditAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-novelty-audit-1"
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
    spec_interpretation_ref = store.put(
        kind="spec_interpretation",
        session_id=session_id,
        key="final",
        obj=VALID_SPEC_INTERPRETATION,
        emitter="ProblemModelingAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="novelty.audit_requested",
        session_id=session_id,
        payload={
            "research_spec_ref": research_spec_ref,
            "literature_brief_ref": literature_brief_ref,
            "spec_interpretation_ref": spec_interpretation_ref,
        },
    )

    ready = [e for e in bus.published if e.event_type == "novelty.audit_ready"]
    assert len(ready) == 1
    assert ready[0].payload["verdict"] == "pass"

    report = store.get(ready[0].payload["novelty_audit_report_ref"])
    errors = list(NOVELTY_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report == VALID_REPORT_PASS

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
