"""
Tests for KnowledgeIndexAgent's EventBus handler.
KnowledgeIndexAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature — same shape as the M0 FakeBusClient in
autoresearch/orchestrator/tests/unit/test_engine_min.py, and the local copies
in deep_research_agent/problem_modeling_agent's test_handler.py) + a real tmp
SpecStore + the real KeywordBackend (no LLM anywhere in this agent). The
FakeBusClient does NOT auto-deliver events to subscribed handlers — each test
calls the handler's event method directly with a constructed Envelope.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from autoresearch.agents._reuse.knowledge_index_backend import KeywordBackend
from autoresearch.agents.knowledge_index_agent.handler import KnowledgeIndexAgentHandler
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store import refs as spec_refs

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    """Captures publishes; mirrors the real EventBusClient.publish/subscribe
    signature. Does NOT auto-deliver events to subscribed handlers."""

    def __init__(self) -> None:
        self.emitter_name = "KnowledgeIndexAgent"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        self.published.append(
            _Published(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                causation_id=causation_id,
            )
        )
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    """The subset of the real broker Envelope the handler reads."""

    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


def _make_envelope(event_type: str, session_id: str, payload: dict, emitter: str) -> _Envelope:
    return _Envelope(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type=event_type,
        emitter=emitter,
        payload=payload,
    )


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


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


def _seed_literature_brief(store: SpecStore, session_id: str) -> str:
    return store.put(
        kind="literature_brief",
        session_id=session_id,
        key="final",
        obj=VALID_LITERATURE_BRIEF,
        emitter="DeepResearchAgent",
    )


def _handler(store: SpecStore) -> tuple[KnowledgeIndexAgentHandler, FakeBusClient]:
    handler = KnowledgeIndexAgentHandler(spec_store=store, backend=KeywordBackend())
    client = FakeBusClient()
    handler.register(client)
    return handler, client


def _knowledge_index_blob(store: SpecStore, session_id: str) -> dict:
    ref = spec_refs.encode("knowledge_index", session_id, "v1")
    return store.get(ref)


# ---------------------------------------------------------------------------
# (a) literature.brief_ready -> auto-ingest key_papers, title mapped to summary
# ---------------------------------------------------------------------------


async def test_brief_ready_ingests_key_papers_title_into_summary(store):
    handler, client = _handler(store)
    session_id = str(uuid.uuid4())
    literature_brief_ref = _seed_literature_brief(store, session_id)

    env = _make_envelope(
        "literature.brief_ready",
        session_id,
        {
            "literature_brief_ref": literature_brief_ref,
            "field_trajectory_brief_ref": "spec://field_trajectory_brief/x/final",
        },
        "DeepResearchAgent",
    )

    await handler.handle_brief_ready(client, env)

    blob = _knowledge_index_blob(store, session_id)
    entries = blob["entries"]
    assert len(entries) == 2

    seminal = next(e for e in entries if e["path_or_ref"] == SEMINAL_PAPER["id_or_url"])
    assert SEMINAL_PAPER["title"] in seminal["summary"]
    assert SEMINAL_PAPER["note"] in seminal["summary"]
    assert seminal["type"] == "paper"
    assert seminal["source"] == "literature.brief_ready"
    assert seminal["produced_by_state"] == "DeepResearching"

    frontier = next(e for e in entries if e["path_or_ref"] == FRONTIER_PAPER["id_or_url"])
    assert frontier["summary"] == FRONTIER_PAPER["title"]

    # No result event is published for ingest.
    assert client.published == []


# ---------------------------------------------------------------------------
# (b) knowledge.ingest_requested -> items appear in the index
# ---------------------------------------------------------------------------


async def test_ingest_requested_adds_items_to_index(store):
    handler, client = _handler(store)
    session_id = str(uuid.uuid4())

    items = [
        {
            "path_or_ref": "evidence://sess/e1",
            "summary": "Docker build failed due to missing dependency.",
        },
    ]
    env = _make_envelope(
        "knowledge.ingest_requested",
        session_id,
        {"items": items},
        "AcceptanceAgent",
    )

    await handler.handle_ingest_requested(client, env)

    blob = _knowledge_index_blob(store, session_id)
    entries = blob["entries"]
    assert len(entries) == 1
    assert entries[0]["path_or_ref"] == "evidence://sess/e1"
    assert entries[0]["summary"] == items[0]["summary"]
    assert entries[0]["type"] == "artifact"
    assert entries[0]["source"] == "knowledge.ingest_requested"

    assert client.published == []


async def test_ingest_requested_merges_with_existing_index(store):
    handler, client = _handler(store)
    session_id = str(uuid.uuid4())
    literature_brief_ref = _seed_literature_brief(store, session_id)

    await handler.handle_brief_ready(
        client,
        _make_envelope(
            "literature.brief_ready",
            session_id,
            {
                "literature_brief_ref": literature_brief_ref,
                "field_trajectory_brief_ref": "spec://field_trajectory_brief/x/final",
            },
            "DeepResearchAgent",
        ),
    )

    items = [{"path_or_ref": "evidence://sess/e1", "summary": "Something happened."}]
    await handler.handle_ingest_requested(
        client,
        _make_envelope("knowledge.ingest_requested", session_id, {"items": items}, "AcceptanceAgent"),
    )

    blob = _knowledge_index_blob(store, session_id)
    assert len(blob["entries"]) == 3


# ---------------------------------------------------------------------------
# (c) knowledge.retrieve_requested -> retrieve_result with matches, hit_count,
#     causation_id
# ---------------------------------------------------------------------------


async def test_retrieve_requested_publishes_result_with_matches(store):
    handler, client = _handler(store)
    session_id = str(uuid.uuid4())
    literature_brief_ref = _seed_literature_brief(store, session_id)

    await handler.handle_brief_ready(
        client,
        _make_envelope(
            "literature.brief_ready",
            session_id,
            {
                "literature_brief_ref": literature_brief_ref,
                "field_trajectory_brief_ref": "spec://field_trajectory_brief/x/final",
            },
            "DeepResearchAgent",
        ),
    )

    env = _make_envelope(
        "knowledge.retrieve_requested",
        session_id,
        {"query": "Retrieval-Augmented"},
        "ResearchOrchestrator",
    )

    await handler.handle_retrieve_requested(client, env)

    results = [p for p in client.published if p.event_type == "knowledge.retrieve_result"]
    assert len(results) == 1
    payload = results[0].payload
    assert payload["query"] == "Retrieval-Augmented"
    assert payload["hit_count"] == len(payload["entries"])
    assert payload["hit_count"] >= 1
    refs_found = {e["path_or_ref"] for e in payload["entries"]}
    assert SEMINAL_PAPER["id_or_url"] in refs_found or FRONTIER_PAPER["id_or_url"] in refs_found
    assert results[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (d) retrieve before any ingest -> empty entries, hit_count 0, no crash
# ---------------------------------------------------------------------------


async def test_retrieve_requested_before_any_ingest_returns_empty(store):
    handler, client = _handler(store)
    session_id = str(uuid.uuid4())

    env = _make_envelope(
        "knowledge.retrieve_requested",
        session_id,
        {"query": "anything"},
        "ResearchOrchestrator",
    )

    await handler.handle_retrieve_requested(client, env)

    results = [p for p in client.published if p.event_type == "knowledge.retrieve_result"]
    assert len(results) == 1
    assert results[0].payload == {"query": "anything", "entries": [], "hit_count": 0}
    assert results[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (e) malformed ingest item -> agent.failed with correct classification
# ---------------------------------------------------------------------------


async def test_ingest_requested_malformed_item_publishes_agent_failed(store):
    handler, client = _handler(store)
    session_id = str(uuid.uuid4())

    items = [{"summary": "Missing path_or_ref."}]
    env = _make_envelope(
        "knowledge.ingest_requested",
        session_id,
        {"items": items},
        "AcceptanceAgent",
    )

    await handler.handle_ingest_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "KnowledgeIndexAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    # No index mutation, no retrieve_result on failure.
    with pytest.raises(KeyError):
        _knowledge_index_blob(store, session_id)
