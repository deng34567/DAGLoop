"""
Tests for DeepResearchAgent's EventBus handler.
DeepResearchAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature — same shape as the M1 FakeBusClient in
problem_modeling_agent/tests/test_handler.py) + FakeBackend (from
autoresearch/agents/_reuse/deep_research_backend.py) + FakeLLM (from this
package's tests/conftest.py) + a real tmp SpecStore. The FakeBusClient does
NOT auto-deliver events to subscribed handlers — each test calls the
handler's event method directly with a constructed Envelope (mirrors M1's
test_handler.py).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.deep_research_backend import FakeBackend
from autoresearch.agents.deep_research_agent import core
from autoresearch.agents.deep_research_agent.handler import DeepResearchAgentHandler
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[1] / "data-contracts"
)


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


LITERATURE_BRIEF_VALIDATOR = _validator("literature_brief.schema.yaml")
FIELD_TRAJECTORY_BRIEF_VALIDATOR = _validator("field_trajectory_brief.schema.yaml")

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


# Keyword-planning response (now always runs before retrieval) queued ahead of synthesis.
_PLAN_JSON = json.dumps({"query": "retrieval augmented generation scientific QA"})


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
        self.emitter_name = "DeepResearchAgent"
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


def _seed_research_spec(store: SpecStore, session_id: str) -> str:
    """Seed the store with a valid frozen ResearchSpec, as ProblemModelingAgent
    would (kind='research_spec'); return its ref."""
    return store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )


# ---------------------------------------------------------------------------
# (a) + (b): deep_research.requested -> literature.brief_ready, refs resolve
# to schema-valid briefs; ResearchSpec is read from the STORE (provenance),
# never from the event payload (which carries only the ref).
# ---------------------------------------------------------------------------


async def test_requested_publishes_brief_ready_with_resolvable_refs(store):
    backend = FakeBackend()
    llm = FakeLLM([_PLAN_JSON, _valid_synthesis_json()])
    handler = DeepResearchAgentHandler(spec_store=store, backend=backend, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)

    env = _make_envelope(
        "deep_research.requested",
        session_id,
        {"research_spec_ref": spec_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "literature.brief_ready"]
    assert len(ready) == 1
    payload = ready[0].payload
    assert set(payload.keys()) == {"literature_brief_ref", "field_trajectory_brief_ref"}

    literature_brief = store.get(payload["literature_brief_ref"])
    errors = list(LITERATURE_BRIEF_VALIDATOR.iter_errors(literature_brief))
    assert errors == [], [e.message for e in errors]

    field_trajectory_brief = store.get(payload["field_trajectory_brief_ref"])
    errors = list(FIELD_TRAJECTORY_BRIEF_VALIDATOR.iter_errors(field_trajectory_brief))
    assert errors == [], [e.message for e in errors]

    # (b) provenance: the payload passed to the LLM must have been built from
    # the ResearchSpec resolved via the STORE ref, not (e.g.) an empty/garbage
    # spec — i.e. the query derivation actually saw VALID_RESEARCH_SPEC.
    assert backend.seminal_calls == 1
    assert backend.frontier_calls == 1


async def test_requested_reads_research_spec_from_store_not_payload(store):
    """The event payload carries ONLY {research_spec_ref} (per the
    deep_research.requested schema) — the handler must resolve the full
    ResearchSpec via the store, never expect it inline in the payload."""
    backend = FakeBackend()
    llm = FakeLLM([_PLAN_JSON, _valid_synthesis_json()])
    handler = DeepResearchAgentHandler(spec_store=store, backend=backend, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)

    env = _make_envelope(
        "deep_research.requested",
        session_id,
        {"research_spec_ref": spec_ref},
        "ResearchOrchestrator",
    )
    assert set(env.payload.keys()) == {"research_spec_ref"}

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "literature.brief_ready"]
    assert len(ready) == 1

    # The user message given to the LLM must reflect the spec's topic, which
    # only exists in the STORE (not in the event payload).
    messages = llm.calls[-1]  # synthesis call (calls[0] is keyword-planning)
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message


async def test_targeted_request_grounds_query_and_synthesis_in_method_and_audit(store):
    """A novelty retry must search the frozen method family and the audit's missing work."""
    backend = FakeBackend()
    llm = FakeLLM([_PLAN_JSON, _valid_synthesis_json()])
    handler = DeepResearchAgentHandler(spec_store=store, backend=backend, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)
    method_ref = store.put(
        kind="method", session_id=session_id, key="v1",
        obj={"approach": "causal-token-scheduler-X"}, emitter="MethodAgent")
    audit_ref = store.put(
        kind="novelty_audit_report", session_id=session_id, key="v1",
        obj={"verdict": "evidence_insufficient", "citation_sufficiency": "insufficient",
             "novelty_status": "uncertain",
             "missing_work": ["closest prior curriculum work Y"]},
        emitter="NoveltyAuditAgent")

    await handler.handle_requested(
        client,
        _make_envelope(
            "deep_research.requested", session_id,
            {"research_spec_ref": spec_ref, "method_ref": method_ref,
             "novelty_audit_report_ref": audit_ref,
             "targeted_search_id": "target-abc123"},
            "ResearchOrchestrator",
        ),
    )

    keyword_prompt = llm.calls[0][-1]["content"]
    synthesis_prompt = llm.calls[1][-1]["content"]
    for prompt in (keyword_prompt, synthesis_prompt):
        assert "causal-token-scheduler-X" in prompt
        assert "closest prior curriculum work Y" in prompt
    [ready] = [p for p in client.published if p.event_type == "literature.brief_ready"]
    assert ready.payload["targeted_search_id"] == "target-abc123"
    assert ready.payload["method_ref"] == method_ref
    assert ready.payload["novelty_audit_report_ref"] == audit_ref
    assert "target-abc123" in ready.payload["literature_brief_ref"]


async def test_targeted_redelivery_cannot_overwrite_frozen_briefs(store):
    first = json.loads(_valid_synthesis_json())
    second = json.loads(_valid_synthesis_json())
    second["literature_brief"]["summary"] = "A different nondeterministic synthesis."
    llm = FakeLLM([_PLAN_JSON, json.dumps(first), _PLAN_JSON, json.dumps(second)])
    handler = DeepResearchAgentHandler(
        spec_store=store, backend=FakeBackend(), llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)
    method_ref = store.put(
        kind="method", session_id=session_id, key="v1",
        obj={"approach": "frozen method"}, emitter="MethodAgent")
    audit_ref = store.put(
        kind="novelty_audit_report", session_id=session_id, key="v1",
        obj={"verdict": "evidence_insufficient"}, emitter="NoveltyAuditAgent")
    payload = {
        "research_spec_ref": spec_ref,
        "method_ref": method_ref,
        "novelty_audit_report_ref": audit_ref,
        "targeted_search_id": "target-redelivery",
    }

    for _ in range(2):
        await handler.handle_requested(
            client,
            _make_envelope("deep_research.requested", session_id, payload,
                           "ResearchOrchestrator"),
        )

    first_ready, second_ready = [
        item for item in client.published if item.event_type == "literature.brief_ready"]
    assert first_ready.payload["literature_brief_ref"] != second_ready.payload["literature_brief_ref"]
    assert store.get(first_ready.payload["literature_brief_ref"])["summary"] == (
        first["literature_brief"]["summary"])
    assert store.get(second_ready.payload["literature_brief_ref"])["summary"] == (
        second["literature_brief"]["summary"])


# ---------------------------------------------------------------------------
# (c) core error (malformed LLM JSON) -> explicit evidence-gap briefs
# ---------------------------------------------------------------------------


async def test_core_parse_error_publishes_evidence_gap_briefs(store):
    backend = FakeBackend()
    llm = FakeLLM([_PLAN_JSON] + ["not json at all"] * (core.MAX_RETRIES + 1))
    handler = DeepResearchAgentHandler(spec_store=store, backend=backend, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)

    env = _make_envelope(
        "deep_research.requested",
        session_id,
        {"research_spec_ref": spec_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "literature.brief_ready"]
    assert len(ready) == 1
    assert not [p for p in client.published if p.event_type == "agent.failed"]
    literature = store.get(ready[0].payload["literature_brief_ref"])
    trajectory = store.get(ready[0].payload["field_trajectory_brief_ref"])
    assert list(LITERATURE_BRIEF_VALIDATOR.iter_errors(literature)) == []
    assert list(FIELD_TRAJECTORY_BRIEF_VALIDATOR.iter_errors(trajectory)) == []
    assert literature["key_papers"] == []
    assert literature["evidence_gaps"][0]["attempted"] is True


# ---------------------------------------------------------------------------
# (d) causation_id is set on published events
# ---------------------------------------------------------------------------


async def test_causation_id_set_on_brief_ready(store):
    backend = FakeBackend()
    llm = FakeLLM([_PLAN_JSON, _valid_synthesis_json()])
    handler = DeepResearchAgentHandler(spec_store=store, backend=backend, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)

    env = _make_envelope(
        "deep_research.requested",
        session_id,
        {"research_spec_ref": spec_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "literature.brief_ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (e) MOSS prompt_patch: config_root/agent_config/DeepResearchAgent/prompt_patch.md
# gets read by self.config.prompt_patch() and injected into the system message.
# ---------------------------------------------------------------------------


async def test_prompt_patch_from_config_root_injected_into_system_message(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "DeepResearchAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    backend = FakeBackend()
    llm = FakeLLM([_PLAN_JSON, _valid_synthesis_json()])
    handler = DeepResearchAgentHandler(
        spec_store=store, backend=backend, llm_client=llm, config_root=tmp_path
    )
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    spec_ref = _seed_research_spec(store, session_id)

    env = _make_envelope(
        "deep_research.requested",
        session_id,
        {"research_spec_ref": spec_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_requested(client, env)

    messages = llm.calls[-1]  # synthesis call (calls[0] is keyword-planning)
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
