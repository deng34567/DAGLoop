"""M3 end-to-end integration smoke (Task 4): single session, no Redis, no LLM
in the KnowledgeIndex path.

Extends the M2 smoke's ``DeliveringBus`` pattern (copied here so this file
stays self-contained) to genuinely exercise subscribe/dispatch routing
between TWO independently-built agent services -- ``DeepResearchAgent`` and
``KnowledgeIndexAgent`` -- wired on one shared in-memory bus, showing the
ingest -> retrieve round-trip:

  1. A valid ``ResearchSpec`` is seeded directly in the Spec Store
     (kind="research_spec", emitter="ProblemModelingAgent" -- the only
     allowed emitter for that kind) -- standing in for the
     ProblemModelingAgent output that M1/M2 already cover end-to-end.
  2. ``deep_research.requested {research_spec_ref}`` is published (emitter
     "ResearchOrchestrator") -> (delivered) -> ``DeepResearchAgent`` resolves
     the spec, runs its two-pass ``FakeBackend`` search + ``FakeLLM``
     synthesis, persists a ``LiteratureBrief``, and publishes
     ``literature.brief_ready`` -> (delivered) -> ``KnowledgeIndexAgent``
     auto-ingests the brief's ``key_papers`` into its per-session
     ``knowledge_index`` blob (real ``KeywordBackend``, no LLM).
  3. ``knowledge.retrieve_requested {query}`` is published (emitter
     "CriticAgent") with a distinctive word from one ``FakeBackend`` paper's
     title -> (delivered) -> ``KnowledgeIndexAgent`` ranks the index and
     publishes ``knowledge.retrieve_result`` whose entries include that
     paper.

No concurrency, no Redis, no real LLM anywhere in the KnowledgeIndex path
(one ``FakeLLM`` canned queue for ``DeepResearchAgent`` only; ``FakeBackend``
canned papers; ``PaperQA2Backend``/``SemanticBackend`` are never
instantiated).
"""

from __future__ import annotations

import fnmatch
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import pytest

from autoresearch.agents._reuse.deep_research_backend import FakeBackend
from autoresearch.agents._reuse.knowledge_index_backend import KeywordBackend
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM as DeepResearchFakeLLM
from autoresearch.agents.knowledge_index_agent.service import build_knowledge_index_service
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store import refs as spec_refs

RESEARCH_SPEC = {
    "topic": "modded-nanogpt speedrun optimization",
    "research_goal": (
        "Reduce modded-nanogpt training wall-clock time to a new speedrun "
        "record while preserving the target validation loss."
    ),
    "task_mode": "experiment_validation",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["training script", "optimizer config"],
    "forbidden_items": ["changing the model architecture"],
    "open_questions": [],
}

# Grounded in the FakeBackend's default seminal + frontier papers (see
# autoresearch/agents/_reuse/deep_research_backend.py) so core's
# anti-hallucination check (key_papers subset of retrieved papers) passes.
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

# Distinctive word from SEMINAL_PAPER's title -- rare enough not to collide
# with the frontier paper's title, so a retrieve on it is unambiguous.
RETRIEVE_QUERY = "Knowledge-Intensive"


def _synthesis_json() -> str:
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


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes.
# Copied from test_m2_deep_research_smoke.py to keep this file self-contained.
# ---------------------------------------------------------------------------


class DeliveringBus:
    """Shared broker state: subscriptions, the real sqlite handle, and every
    published ``Envelope`` -- delivered synchronously to matching subscribers
    (mirrors ``Dispatcher.dispatch``'s sequential fnmatch routing)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        self.subscriptions: list[tuple[str, Any]] = []
        self.published: list[Envelope] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        emitter: str,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        env = Envelope(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            emitter=emitter,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            schema_version=1,
            cost_delta=cost_delta,
            payload=payload,
            causation_id=causation_id,
        )
        self.published.append(env)
        if auto_commit:
            await self.db.commit()
        for pattern, handler in list(self.subscriptions):
            if fnmatch.fnmatchcase(event_type, pattern):
                await handler(env)
        return env.event_id


class BusClientView:
    """One emitter's handle onto a shared ``DeliveringBus`` -- mirrors how
    each real process holds its own ``EventBusClient`` (its own
    ``emitter_name``) while every client shares one broker/outbox."""

    def __init__(self, bus: DeliveringBus, emitter_name: str) -> None:
        self._bus = bus
        self.emitter_name = emitter_name

    @property
    def db(self) -> aiosqlite.Connection:
        return self._bus.db

    def subscribe(self, pattern: str, handler: Any) -> None:
        self._bus.subscribe(pattern, handler)

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
        return await self._bus.publish(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            emitter=self.emitter_name,
            cost_delta=cost_delta,
            causation_id=causation_id,
            auto_commit=auto_commit,
        )


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m3_smoke.db"))
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
# The smoke
# ---------------------------------------------------------------------------


async def test_m3_knowledge_index_smoke_single_session(bus, store):
    # --- wire DeepResearchAgent + KnowledgeIndexAgent on the SAME bus, each -
    # with its own emitter identity. No OrchestratorEngine needed here -- the
    # DeepResearching -> literature.brief_ready hop is already covered
    # end-to-end (with the orchestrator) by the M2 smoke; this smoke drives
    # deep_research.requested directly to focus on the ingest -> retrieve hop.
    backend = FakeBackend()
    deep_research_llm = DeepResearchFakeLLM([json.dumps({"query": "literature search keywords"}), _synthesis_json()])
    build_deep_research_service(
        client=BusClientView(bus, "DeepResearchAgent"),
        spec_store=store,
        backend=backend,
        llm_client=deep_research_llm,
    )

    build_knowledge_index_service(
        client=BusClientView(bus, "KnowledgeIndexAgent"),
        spec_store=store,
        backend=KeywordBackend(),
    )

    sid = str(uuid.uuid4())

    # --- 1. seed a valid ResearchSpec directly in the store (standing in for
    # ProblemModelingAgent's frozen output; only ProblemModelingAgent may
    # write kind="research_spec").
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=sid,
        key="final",
        obj=RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )

    # --- 2. deep_research.requested -> (delivered) -> DeepResearchAgent runs
    # its two-pass FakeBackend search + FakeLLM synthesis, persists the
    # LiteratureBrief, and publishes literature.brief_ready -> (delivered) ->
    # KnowledgeIndexAgent auto-ingests key_papers into the session's index.
    # All synchronous inside this one publish call (genuine subscribe/
    # dispatch routing, not manual).
    orchestrator = BusClientView(bus, "ResearchOrchestrator")
    await orchestrator.publish(
        event_type="deep_research.requested",
        session_id=sid,
        payload={"research_spec_ref": research_spec_ref},
    )

    brief_ready_events = [e for e in bus.published if e.event_type == "literature.brief_ready"]
    assert len(brief_ready_events) == 1
    literature_brief_ref = brief_ready_events[0].payload["literature_brief_ref"]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # The knowledge_index blob exists, session-keyed, stored as {"entries": [...]}.
    index_ref = spec_refs.encode("knowledge_index", sid, "v1")
    index_blob = store.get(index_ref)
    entries = index_blob["entries"]
    assert len(entries) == 2
    assert {e["path_or_ref"] for e in entries} == {
        SEMINAL_PAPER["id_or_url"],
        FRONTIER_PAPER["id_or_url"],
    }

    # --- 3. knowledge.retrieve_requested -> (delivered) -> KnowledgeIndexAgent
    # ranks the index against the query and publishes knowledge.retrieve_result.
    critic = BusClientView(bus, "CriticAgent")
    retrieve_event_id = await critic.publish(
        event_type="knowledge.retrieve_requested",
        session_id=sid,
        payload={"query": RETRIEVE_QUERY},
    )

    retrieve_results = [e for e in bus.published if e.event_type == "knowledge.retrieve_result"]
    assert len(retrieve_results) == 1
    result = retrieve_results[0]
    assert result.causation_id == retrieve_event_id

    payload = result.payload
    assert payload["hit_count"] > 0
    assert payload["hit_count"] == len(payload["entries"])
    matched_summaries = " ".join(e["summary"] for e in payload["entries"])
    assert SEMINAL_PAPER["title"] in matched_summaries

    # Grounding sanity: the retrieved entry really is the seminal paper.
    matched_refs = {e["path_or_ref"] for e in payload["entries"]}
    assert SEMINAL_PAPER["id_or_url"] in matched_refs

    # Provenance sanity: the LiteratureBrief that was auto-ingested resolves
    # in the store and its key_papers match what was indexed.
    literature_brief = store.get(literature_brief_ref)
    assert {p["id_or_url"] for p in literature_brief["key_papers"]} == {
        SEMINAL_PAPER["id_or_url"],
        FRONTIER_PAPER["id_or_url"],
    }
