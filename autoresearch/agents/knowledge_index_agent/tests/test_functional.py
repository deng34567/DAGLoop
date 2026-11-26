"""
Functional tests for KnowledgeIndexAgent: drive the REAL
``build_knowledge_index_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method). No LLM
anywhere in this agent; the default real ``KeywordBackend`` is used, and
SpecStore is a real tmp instance.

Three flows, one happy-path test each:
  1. literature.brief_ready -> auto-ingest (store write only, NO output event).
  2. knowledge.ingest_requested -> ingest (store write only, NO output event).
  3. knowledge.retrieve_requested -> knowledge.retrieve_result.

针对 KnowledgeIndexAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由）上跑真实的 ``build_knowledge_index_service``，本
agent 全程无 LLM，使用真实默认的 ``KeywordBackend``，SpecStore 为真实的
临时实例。
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.knowledge_index_agent.service import build_knowledge_index_service
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store import refs as spec_refs
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


INDEX_ENTRY_VALIDATOR = _validator("index_entry.schema.yaml")


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


def _knowledge_index_blob(store: SpecStore, session_id: str) -> dict:
    ref = spec_refs.encode("knowledge_index", session_id, "v1")
    return store.get(ref)


SEMINAL_PAPER = {
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "id_or_url": "arxiv:2005.11401",
    "year": 2020,
    "note": "Seminal RAG paper.",
}
FRONTIER_PAPER = {
    "title": "Long-Tail Scientific QA via Retrieval-Augmented Fine-Tuning",
    "id_or_url": "arxiv:2401.00000",
}

VALID_LITERATURE_BRIEF = {
    "summary": "RAG combines dense retrieval with generation.",
    "related_work": [],
    "key_papers": [SEMINAL_PAPER, FRONTIER_PAPER],
    "existing_benchmarks": [],
}


# ---------------------------------------------------------------------------
# (1) literature.brief_ready -> auto-ingest, NO output event
# ---------------------------------------------------------------------------


async def test_brief_ready_ingests_key_papers_with_no_output_event(bus, store):
    build_knowledge_index_service(client=bus.view("KnowledgeIndexAgent"), spec_store=store)

    session_id = "sess-ki-brief-1"
    literature_brief_ref = store.put(
        kind="literature_brief",
        session_id=session_id,
        key="final",
        obj=VALID_LITERATURE_BRIEF,
        emitter="DeepResearchAgent",
    )

    await bus.view("DeepResearchAgent").publish(
        event_type="literature.brief_ready",
        session_id=session_id,
        payload={
            "literature_brief_ref": literature_brief_ref,
            "field_trajectory_brief_ref": "spec://field_trajectory_brief/x/final",
        },
    )

    blob = _knowledge_index_blob(store, session_id)
    entries = blob["entries"]
    assert len(entries) == 2
    for entry in entries:
        errors = list(INDEX_ENTRY_VALIDATOR.iter_errors(entry))
        assert errors == [], [e.message for e in errors]

    # No result event is published for ingest -- the only bus.published entry
    # is the literature.brief_ready trigger itself.
    assert len(bus.published) == 1
    assert bus.published[0].event_type == "literature.brief_ready"
    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (2) knowledge.ingest_requested -> ingest, NO output event
# ---------------------------------------------------------------------------


async def test_ingest_requested_ingests_items_with_no_output_event(bus, store):
    build_knowledge_index_service(client=bus.view("KnowledgeIndexAgent"), spec_store=store)

    session_id = "sess-ki-ingest-1"
    items = [
        {
            "path_or_ref": "evidence://sess/e1",
            "summary": "Docker build failed due to missing dependency.",
        },
    ]

    await bus.view("ResearchOrchestrator").publish(
        event_type="knowledge.ingest_requested",
        session_id=session_id,
        payload={"items": items},
    )

    blob = _knowledge_index_blob(store, session_id)
    entries = blob["entries"]
    assert len(entries) == 1
    assert entries[0]["path_or_ref"] == "evidence://sess/e1"
    errors = list(INDEX_ENTRY_VALIDATOR.iter_errors(entries[0]))
    assert errors == [], [e.message for e in errors]

    assert len(bus.published) == 1
    assert bus.published[0].event_type == "knowledge.ingest_requested"
    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (3) knowledge.retrieve_requested -> knowledge.retrieve_result
# ---------------------------------------------------------------------------


async def test_retrieve_requested_returns_matching_entries(bus, store):
    build_knowledge_index_service(client=bus.view("KnowledgeIndexAgent"), spec_store=store)

    session_id = "sess-ki-retrieve-1"
    seeded_entries = [
        {
            "path_or_ref": SEMINAL_PAPER["id_or_url"],
            "type": "paper",
            "summary": f"{SEMINAL_PAPER['title']} — {SEMINAL_PAPER['note']}",
            "tags": [],
            "source": "literature.brief_ready",
            "produced_by_state": "DeepResearching",
        },
        {
            "path_or_ref": FRONTIER_PAPER["id_or_url"],
            "type": "paper",
            "summary": FRONTIER_PAPER["title"],
            "tags": [],
            "source": "literature.brief_ready",
            "produced_by_state": "DeepResearching",
        },
    ]
    store.put(
        kind="knowledge_index",
        session_id=session_id,
        key="v1",
        obj={"entries": seeded_entries},
        emitter="KnowledgeIndexAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="knowledge.retrieve_requested",
        session_id=session_id,
        payload={"query": "Retrieval-Augmented"},
    )

    results = [e for e in bus.published if e.event_type == "knowledge.retrieve_result"]
    assert len(results) == 1
    payload = results[0].payload
    assert payload["query"] == "Retrieval-Augmented"
    assert payload["hit_count"] == len(payload["entries"])
    assert payload["hit_count"] >= 1

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
