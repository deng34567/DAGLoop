"""
Tests for EvidenceAuditAgent's EventBus handler.
EvidenceAuditAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as novelty_audit_agent/tests/
test_handler.py) + FakeLLM (from this package's tests/conftest.py) + a real
tmp SpecStore. The FakeBusClient does NOT auto-deliver events to subscribed
handlers -- each test calls the handler's event method directly with a
constructed Envelope.

Two subscriptions are covered:
  - evidence.audit_requested -> evidence.audited (Mode A)
  - evidence.citation_audit_requested -> evidence.citation_audited (Mode C)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evidence_audit_agent import core
from autoresearch.agents.evidence_audit_agent.handler import EvidenceAuditAgentHandler
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EVIDENCE_AUDIT_REPORT_VALIDATOR = _validator("evidence_audit_report.schema.yaml")
CITATION_AUDIT_REPORT_VALIDATOR = _validator("citation_audit_report.schema.yaml")

# ---------------------------------------------------------------------------
# Mode A fixtures
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

# ---------------------------------------------------------------------------
# Mode C fixtures
# ---------------------------------------------------------------------------

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
        self.emitter_name = "EvidenceAuditAgent"
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


def _seed_mode_a_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
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
    return {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "spec_interpretation_ref": spec_interpretation_ref,
    }


def _seed_mode_c_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
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
    return {
        "research_memory_ref": research_memory_ref,
        "research_spec_ref": research_spec_ref,
    }


# ---------------------------------------------------------------------------
# Mode A: evidence.audit_requested -> evidence.audited
# ---------------------------------------------------------------------------


async def test_audit_requested_publishes_audited_with_resolvable_ref(store):
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_a_inputs(store, session_id)

    env = _make_envelope("evidence.audit_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_audit_requested(client, env)

    audited = [p for p in client.published if p.event_type == "evidence.audited"]
    assert len(audited) == 1
    payload = audited[0].payload
    assert set(payload.keys()) == {"evidence_audit_report_ref"}

    report = store.get(payload["evidence_audit_report_ref"])
    errors = list(EVIDENCE_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["evidence_strength"] == "strong"


async def test_targeted_audit_echoes_search_identity(store):
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_mode_a_inputs(store, session_id)
    payload["targeted_search_id"] = "target-evidence-1"

    await handler.handle_audit_requested(
        client,
        _make_envelope("evidence.audit_requested", session_id, payload,
                       "ResearchOrchestrator"),
    )

    [audited] = [p for p in client.published if p.event_type == "evidence.audited"]
    assert audited.payload["targeted_search_id"] == "target-evidence-1"


async def test_targeted_redelivery_cannot_overwrite_frozen_evidence(store):
    first = dict(VALID_EVIDENCE_AUDIT_REPORT)
    second = dict(VALID_EVIDENCE_AUDIT_REPORT, notes="Different second audit.")
    llm = FakeLLM([_json(first), _json(second)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_mode_a_inputs(store, session_id)
    payload["targeted_search_id"] = "target-evidence-redelivery"

    for _ in range(2):
        await handler.handle_audit_requested(
            client,
            _make_envelope("evidence.audit_requested", session_id, payload,
                           "ResearchOrchestrator"),
        )

    first_ready, second_ready = [
        item for item in client.published if item.event_type == "evidence.audited"]
    first_ref = first_ready.payload["evidence_audit_report_ref"]
    second_ref = second_ready.payload["evidence_audit_report_ref"]
    assert first_ref != second_ref
    assert store.get(first_ref)["notes"] == first["notes"]
    assert store.get(second_ref)["notes"] == second["notes"]


async def test_audit_requested_reads_inputs_from_store_not_payload(store):
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_a_inputs(store, session_id)

    env = _make_envelope("evidence.audit_requested", session_id, refs, "ResearchOrchestrator")
    assert set(env.payload.keys()) == {
        "research_spec_ref",
        "literature_brief_ref",
        "spec_interpretation_ref",
    }

    await handler.handle_audit_requested(client, env)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_LITERATURE_BRIEF["summary"] in user_message
    assert VALID_SPEC_INTERPRETATION["rationale"] in user_message


async def test_audit_requested_causation_id_set(store):
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_a_inputs(store, session_id)

    env = _make_envelope("evidence.audit_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_audit_requested(client, env)

    audited = [p for p in client.published if p.event_type == "evidence.audited"]
    assert len(audited) == 1
    assert audited[0].causation_id == env.event_id


async def test_audit_requested_infra_error_publishes_agent_failed(store):
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=RaisingLLM())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_a_inputs(store, session_id)

    env = _make_envelope("evidence.audit_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_audit_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "EvidenceAuditAgent"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "evidence.audited"]


async def test_audit_requested_prompt_patch_from_config_root_injected(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "EvidenceAuditAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer recent citations", encoding="utf-8")

    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_a_inputs(store, session_id)

    env = _make_envelope("evidence.audit_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_audit_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer recent citations" in system_message
    assert "[MOSS-injected guidance]" in system_message


# ---------------------------------------------------------------------------
# Mode C: evidence.citation_audit_requested -> evidence.citation_audited
# ---------------------------------------------------------------------------


async def test_citation_audit_requested_publishes_citation_audited_with_resolvable_ref(store):
    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_c_inputs(store, session_id)

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_citation_audit_requested(client, env)

    audited = [p for p in client.published if p.event_type == "evidence.citation_audited"]
    assert len(audited) == 1
    payload = audited[0].payload
    assert set(payload.keys()) == {"citation_audit_report_ref"}

    report = store.get(payload["citation_audit_report_ref"])
    errors = list(CITATION_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["checked_count"] == 2


async def test_citation_audit_requested_reads_inputs_from_store_not_payload(store):
    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_c_inputs(store, session_id)

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"research_memory_ref", "research_spec_ref"}

    await handler.handle_citation_audit_requested(client, env)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_MEMORY["consolidated_insights"][0] in user_message
    assert VALID_RESEARCH_SPEC["topic"] in user_message


async def test_citation_audit_requested_causation_id_set(store):
    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_c_inputs(store, session_id)

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_citation_audit_requested(client, env)

    audited = [p for p in client.published if p.event_type == "evidence.citation_audited"]
    assert len(audited) == 1
    assert audited[0].causation_id == env.event_id


async def test_citation_audit_requested_infra_error_publishes_agent_failed(store):
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=RaisingLLM())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_c_inputs(store, session_id)

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_citation_audit_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "EvidenceAuditAgent"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "evidence.citation_audited"]


async def test_citation_audit_parse_exhaustion_is_non_terminal_and_disclosed(store):
    """Malformed citation-audit JSON must not fail completed research.

    The handler publishes a schema-valid, explicitly unavailable audit while
    leaving infrastructure failures on the normal retry/failure route.
    """
    llm = FakeLLM(["not-json", "still-not-json", "also-not-json"])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    refs = _seed_mode_c_inputs(store, session_id)

    await handler.handle_citation_audit_requested(
        client,
        _make_envelope(
            "evidence.citation_audit_requested",
            session_id,
            refs,
            "ResearchOrchestrator",
        ),
    )

    assert not [p for p in client.published if p.event_type == "agent.failed"]
    [audited] = [
        p for p in client.published if p.event_type == "evidence.citation_audited"
    ]
    report = store.get(audited.payload["citation_audit_report_ref"])
    assert list(CITATION_AUDIT_REPORT_VALIDATOR.iter_errors(report)) == []
    assert report["unsupported_claims"] == []
    assert report["checked_count"] == 0
    assert "unavailable" in report["notes"].lower()
    assert len(llm.calls) == core.MAX_RETRIES + 1


async def test_citation_audit_requested_with_none_research_memory_ref_short_circuits(store):
    """A session that reached Reporting via a decision/MOSS path that skipped
    ConsolidatingMemory has research_memory_ref=None. The handler must NOT
    call spec_store.get(None) or the LLM -- it short-circuits to an empty,
    schema-valid CitationAuditReport and still publishes
    evidence.citation_audited (never agent.failed)."""
    llm = FakeLLM([])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    payload = {"research_memory_ref": None, "research_spec_ref": research_spec_ref}

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_citation_audit_requested(client, env)

    assert llm.calls == []

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert failed == []

    audited = [p for p in client.published if p.event_type == "evidence.citation_audited"]
    assert len(audited) == 1
    payload_out = audited[0].payload
    assert set(payload_out.keys()) == {"citation_audit_report_ref"}
    assert audited[0].causation_id == env.event_id

    report = store.get(payload_out["citation_audit_report_ref"])
    errors = list(CITATION_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["unsupported_claims"] == []
    assert report["checked_count"] == 0


async def test_citation_audit_requested_with_missing_research_memory_ref_short_circuits(store):
    """Same as above, but research_memory_ref is absent from the payload
    entirely (not just None)."""
    llm = FakeLLM([])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"research_spec_ref": "some-ref-not-used"}

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_citation_audit_requested(client, env)

    assert llm.calls == []
    assert not [p for p in client.published if p.event_type == "agent.failed"]
    audited = [p for p in client.published if p.event_type == "evidence.citation_audited"]
    assert len(audited) == 1


async def test_citation_audit_requested_prompt_patch_from_config_root_injected(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "EvidenceAuditAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer recent citations", encoding="utf-8")

    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT)])
    handler = EvidenceAuditAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_mode_c_inputs(store, session_id)

    env = _make_envelope(
        "evidence.citation_audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_citation_audit_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer recent citations" in system_message
    assert "[MOSS-injected guidance]" in system_message
