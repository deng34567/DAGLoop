"""
Functional test for EvidenceAuditAgent: drive the REAL
``build_evidence_audit_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

Two flows, one happy-path test each:
  - evidence.audit_requested -> evidence.audited (Mode A)
  - evidence.citation_audit_requested -> evidence.citation_audited (Mode C)

针对 EvidenceAuditAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_evidence_audit_service``，只 mock LLM（``FakeLLM``），SpecStore 为
真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EVIDENCE_AUDIT_REPORT_VALIDATOR = _validator("evidence_audit_report.schema.yaml")
CITATION_AUDIT_REPORT_VALIDATOR = _validator("citation_audit_report.schema.yaml")


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
    "rationale": "The proposed direction remains within the original research scope.",
}

VALID_EVIDENCE_AUDIT_REPORT = {
    "claim_citation_matrix": [
        {
            "claim": "RAG improves long-tail QA accuracy over closed-book baselines.",
            "citations": ["arxiv:2005.11401"],
            "support": "strong",
        }
    ],
    "missing_papers": [],
    "expired_citations": [],
    "evidence_strength": "strong",
    "notes": "Well grounded in the core RAG paper.",
}

VALID_RESEARCH_MEMORY = {
    "verified_findings": ["RAG with dense retrieval beats closed-book QA on the long-tail set."],
    "negative_results": [],
    "excluded_hypotheses": [],
    "consolidated_insights": ["Retrieval quality dominates end-task accuracy."],
    "divergence_seeds": [],
    "next_steps": [],
    "best_metric": None,
    "best_result_ref": None,
}

VALID_CITATION_AUDIT_REPORT = {
    "unsupported_claims": [],
    "checked_count": 2,
    "notes": None,
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Mode A: evidence.audit_requested -> evidence.audited
# ---------------------------------------------------------------------------


async def test_audit_requested_produces_audited(bus, store):
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    build_evidence_audit_service(
        client=bus.view("EvidenceAuditAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-evidence-audit-1"
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
        event_type="evidence.audit_requested",
        session_id=session_id,
        payload={
            "research_spec_ref": research_spec_ref,
            "literature_brief_ref": literature_brief_ref,
            "spec_interpretation_ref": spec_interpretation_ref,
        },
    )

    audited = [e for e in bus.published if e.event_type == "evidence.audited"]
    assert len(audited) == 1

    report = store.get(audited[0].payload["evidence_audit_report_ref"])
    errors = list(EVIDENCE_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report == VALID_EVIDENCE_AUDIT_REPORT

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# Mode C: evidence.citation_audit_requested -> evidence.citation_audited
# ---------------------------------------------------------------------------


async def test_citation_audit_requested_produces_citation_audited(bus, store):
    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT)])
    build_evidence_audit_service(
        client=bus.view("EvidenceAuditAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-evidence-audit-2"
    research_memory_ref = store.put(
        kind="research_memory",
        session_id=session_id,
        key="v1",
        obj=VALID_RESEARCH_MEMORY,
        emitter="MemoryAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="evidence.citation_audit_requested",
        session_id=session_id,
        payload={
            "research_memory_ref": research_memory_ref,
            "research_spec_ref": research_spec_ref,
        },
    )

    audited = [e for e in bus.published if e.event_type == "evidence.citation_audited"]
    assert len(audited) == 1

    report = store.get(audited[0].payload["citation_audit_report_ref"])
    errors = list(CITATION_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report == VALID_CITATION_AUDIT_REPORT

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
