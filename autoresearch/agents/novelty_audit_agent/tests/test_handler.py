"""
Tests for NoveltyAuditAgent's EventBus handler.
NoveltyAuditAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature — same shape as the M2 FakeBusClient in
deep_research_agent/tests/test_handler.py) + FakeLLM (from this package's
tests/conftest.py) + a real tmp SpecStore. The FakeBusClient does NOT
auto-deliver events to subscribed handlers — each test calls the handler's
event method directly with a constructed Envelope.
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

from autoresearch.agents.novelty_audit_agent import core
from autoresearch.agents.novelty_audit_agent.handler import NoveltyAuditAgentHandler
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


NOVELTY_AUDIT_REPORT_VALIDATOR = _validator("novelty_audit_report.schema.yaml")

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

VALID_REPORT_INSUFFICIENT = {
    "verdict": "evidence_insufficient",
    "citation_sufficiency": "Too few citations were retrieved to judge the citation landscape.",
    "stale_citations": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)"],
    "unsupported_claims": ["Claim of large efficiency gains over prior RAG baselines"],
    "missing_work": ["Recent 2024 surveys on long-tail scientific QA"],
    "novelty_status": "uncertain",
}


def _harness_report(*, relation: str, difference_scope: str) -> dict:
    return {
        **VALID_REPORT_PASS,
        "method_signature": {
            "primary_core": {
                "core_id": "core-1",
                "intervention": "change retrieval routing",
                "causal_mechanism": "route evidence by question type",
                "pipeline_role": "retrieval",
                "expected_effect": "improve answer grounding",
            },
            "supporting_components": [{"name": "index", "role": "enabler"}],
            "composition_claim": {"material": False, "mechanism": ""},
        },
        "prior_work_comparisons": [{
            "prior_work_id": "arxiv:2005.11401",
            "primary_core_relation": relation,
            "difference_scope": difference_scope,
            "composition_relation": "not_applicable",
            "supporting_overlap": ["retrieval index"],
            "evidence": ["The supplied RAG paper provides the compared retrieval mechanism."],
        }],
        "retrieval_coverage": {
            "covered_core_ids": ["core-1"],
            "compared_prior_work_ids": ["arxiv:2005.11401"],
            "unresolved_candidates": [],
        },
    }


def _report_json(report: dict) -> str:
    return json.dumps(report)


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
        self.emitter_name = "NoveltyAuditAgent"
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


def _seed_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the three frozen inputs NoveltyAuditAgent reads,
    as upstream agents would; return their refs."""
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


@pytest.mark.parametrize("brief_search_id,keep_core", [
    ("prior-search", False),
    ("current-search", True),
])
async def test_revised_method_does_not_inherit_prior_search_core(
    store, monkeypatch, brief_search_id, keep_core,
):
    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    core_record = {
        "core_id": "prior-pipeline",
        "intervention": "prior intervention",
        "causal_mechanism": "prior mechanism",
        "pipeline_role": "base",
        "expected_effect": "prior effect",
    }
    brief_ref = store.put(
        kind="literature_brief", session_id=session_id,
        key=f"targeted-{brief_search_id}-evidence",
        obj={**VALID_LITERATURE_BRIEF, "searched_method_primary_core": core_record},
        emitter="DeepResearchAgent",
    )
    seen = []

    def audit_step(research_spec, literature_brief, spec_interpretation, llm_client, **kwargs):
        seen.append(literature_brief)
        return VALID_REPORT_PASS

    monkeypatch.setattr(core, "audit_step", audit_step)
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    await handler.handle_requested(client, _make_envelope(
        "novelty.audit_requested", session_id,
        {**refs, "literature_brief_ref": brief_ref,
         "targeted_search_id": "current-search"},
        "ResearchOrchestrator",
    ))

    assert len([p for p in client.published if p.event_type == "novelty.audit_ready"]) == 1
    assert ("searched_method_primary_core" in seen[0]) is keep_core
    assert store.get(brief_ref)["searched_method_primary_core"] == core_record


# ---------------------------------------------------------------------------
# (a) + (b): novelty.audit_requested -> novelty.audit_ready, ref resolves to
# a schema-valid NoveltyAuditReport; all three inputs are read from the
# STORE (provenance), never from the event payload (which carries only refs).
# ---------------------------------------------------------------------------


async def test_gen_mode_auto_passes_without_llm(store, monkeypatch):
    """OPT-IN auto-pass (AUTORESEARCH_NOVELTY_AUTOPASS=1): the novelty audit passes WITHOUT
    the LLM. By DEFAULT the audit now does the real second search even in GEN mode; the old
    skip is only reachable behind this explicit flag. A FakeLLM with an empty queue would raise
    if called -> a clean pass proves the LLM was skipped."""
    monkeypatch.setenv("AUTORESEARCH_NOVELTY_AUTOPASS", "1")
    llm = FakeLLM([])  # empty -> raises on .chat if the handler ever calls it
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope("novelty.audit_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert len(ready) == 1 and ready[0].payload["verdict"] == "pass"
    assert not [p for p in client.published if p.event_type == "agent.failed"]
    report = store.get(ready[0].payload["novelty_audit_report_ref"])
    assert list(NOVELTY_AUDIT_REPORT_VALIDATOR.iter_errors(report)) == []
    assert report["novelty_status"] == "uncertain"


async def test_targeted_reaudit_runs_real_search_but_is_informational_in_gen(
    store, monkeypatch
):
    """GEN still runs targeted re-audit, but thin literature must not block experiments."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_NOVELTY_AUTOPASS", "1")
    llm = FakeLLM([_report_json(VALID_REPORT_INSUFFICIENT)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_inputs(store, session_id)
    payload["targeted_search_id"] = "target-strict-1"

    await handler.handle_requested(
        client,
        _make_envelope("novelty.audit_requested", session_id, payload,
                       "ResearchOrchestrator"),
    )

    [ready] = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert len(llm.calls) == 1
    assert ready.payload["verdict"] == "pass"
    assert ready.payload["targeted_search_id"] == "target-strict-1"


@pytest.mark.parametrize("feasibility_status", ["uncertain", "infeasible"])
async def test_gen_informational_novelty_ignores_deprecated_feasibility_field(
    store, monkeypatch, feasibility_status
):
    """Feasibility is established by the real experiment, not this literature audit."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    report = {
        **VALID_REPORT_INSUFFICIENT,
        "feasibility_status": feasibility_status,
    }
    llm = FakeLLM([_report_json(report)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())

    await handler.handle_requested(
        client,
        _make_envelope(
            "novelty.audit_requested", session_id, _seed_inputs(store, session_id),
            "ResearchOrchestrator",
        ),
    )

    [ready] = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert ready.payload["verdict"] == "pass"


async def test_strict_novelty_mode_preserves_targeted_nonpass(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_NOVELTY_STRICT", "1")
    llm = FakeLLM([_report_json(VALID_REPORT_INSUFFICIENT)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_inputs(store, session_id)
    payload["targeted_search_id"] = "target-strict-2"

    await handler.handle_requested(
        client,
        _make_envelope("novelty.audit_requested", session_id, payload,
                       "ResearchOrchestrator"),
    )

    [ready] = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert ready.payload["verdict"] == "evidence_insufficient"


async def test_gen_informational_novelty_can_relax_known_but_feasible_method(
    store, monkeypatch
):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    report = {
        **VALID_REPORT_INSUFFICIENT,
        "novelty_status": "known",
        "feasibility_status": "feasible",
    }
    llm = FakeLLM([_report_json(report)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())

    await handler.handle_requested(
        client,
        _make_envelope(
            "novelty.audit_requested", session_id, _seed_inputs(store, session_id),
            "ResearchOrchestrator",
        ),
    )

    [ready] = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert ready.payload["verdict"] == "pass"
    saved = store.get(ready.payload["novelty_audit_report_ref"])
    assert saved["verdict"] == "evidence_insufficient"
    assert saved["novelty_status"] == "known"


async def test_gen_preserves_not_novel_finding_while_allowing_experiment(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    llm = FakeLLM([_report_json(_harness_report(
        relation="equivalent", difference_scope="supporting_only"))])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    method_ref = store.put(
        kind="method", session_id=session_id, key="known-method",
        obj={"title": "Known method", "summary": "Implement the known method"},
        emitter="MethodAgent",
    )
    await handler.handle_requested(
        client,
        _make_envelope("novelty.audit_requested", session_id,
                       {**_seed_inputs(store, session_id), "method_ref": method_ref},
                       "ResearchOrchestrator"),
    )
    [ready] = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert ready.payload["verdict"] == "pass"
    saved = store.get(ready.payload["novelty_audit_report_ref"])
    assert saved["verdict"] == "not_novel"
    assert saved["novelty_status"] == "known"


async def test_targeted_redelivery_cannot_overwrite_frozen_novelty_report(store):
    llm = FakeLLM([
        _report_json(VALID_REPORT_PASS),
        _report_json(VALID_REPORT_INSUFFICIENT),
    ])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_inputs(store, session_id)
    payload["targeted_search_id"] = "target-novelty-redelivery"

    for _ in range(2):
        await handler.handle_requested(
            client,
            _make_envelope("novelty.audit_requested", session_id, payload,
                           "ResearchOrchestrator"),
        )

    first_ready, second_ready = [
        item for item in client.published if item.event_type == "novelty.audit_ready"]
    first_ref = first_ready.payload["novelty_audit_report_ref"]
    second_ref = second_ready.payload["novelty_audit_report_ref"]
    assert first_ref != second_ref
    assert store.get(first_ref)["verdict"] == "pass"
    assert store.get(second_ref)["verdict"] == "evidence_insufficient"


async def test_distinct_methods_cannot_reuse_or_overwrite_novelty_report(store):
    llm = FakeLLM([
        _report_json(_harness_report(relation="distinct", difference_scope="material_core")),
        _report_json(_harness_report(relation="equivalent", difference_scope="supporting_only")),
    ])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_inputs(store, session_id)

    for key, title in (("method-a", "Method A"), ("method-b", "Method B")):
        method_ref = store.put(
            kind="method",
            session_id=session_id,
            key=key,
            obj={"title": title, "summary": f"Implement {title}"},
            emitter="MethodAgent",
        )
        await handler.handle_requested(
            client,
            _make_envelope(
                "novelty.audit_requested",
                session_id,
                {**payload, "method_ref": method_ref},
                "ResearchOrchestrator",
            ),
        )

    first_ready, second_ready = [
        item for item in client.published if item.event_type == "novelty.audit_ready"]
    first_ref = first_ready.payload["novelty_audit_report_ref"]
    second_ref = second_ready.payload["novelty_audit_report_ref"]
    assert first_ref != second_ref
    assert store.get(first_ref)["verdict"] == "pass"
    assert store.get(second_ref)["verdict"] == "not_novel"


async def test_requested_publishes_audit_ready_with_resolvable_ref(store):
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "novelty.audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert len(ready) == 1
    payload = ready[0].payload
    assert set(payload.keys()) == {"verdict", "novelty_audit_report_ref"}
    assert payload["verdict"] == "pass"

    report = store.get(payload["novelty_audit_report_ref"])
    errors = list(NOVELTY_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "pass"


async def test_requested_reads_all_three_inputs_from_store_not_payload(store):
    """The event payload carries ONLY the three refs (per the
    novelty.audit_requested schema) — the handler must resolve the full
    objects via the store, never expect them inline in the payload."""
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "novelty.audit_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {
        "research_spec_ref",
        "literature_brief_ref",
        "spec_interpretation_ref",
    }

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert len(ready) == 1

    # The user message given to the LLM must reflect content that only
    # exists in the STORE (not in the event payload).
    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_LITERATURE_BRIEF["summary"] in user_message
    assert VALID_SPEC_INTERPRETATION["real_open_problem"] in user_message


# ---------------------------------------------------------------------------
# (c) both verdicts are simply forwarded from the report.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canned_report",
    [VALID_REPORT_PASS, VALID_REPORT_INSUFFICIENT],
    ids=["pass", "evidence_insufficient"],
)
async def test_both_verdicts_forwarded(store, canned_report):
    llm = FakeLLM([_report_json(canned_report)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "novelty.audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert len(ready) == 1
    assert ready[0].payload["verdict"] == canned_report["verdict"]


# ---------------------------------------------------------------------------
# (d) core error (malformed LLM JSON) -> agent.failed, no crash, no
# novelty.audit_ready.
# ---------------------------------------------------------------------------


async def test_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "novelty.audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "NoveltyAuditAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "novelty.audit_ready"]


# ---------------------------------------------------------------------------
# (e) causation_id is set on published events
# ---------------------------------------------------------------------------


async def test_causation_id_set_on_audit_ready(store):
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "novelty.audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    ready = [p for p in client.published if p.event_type == "novelty.audit_ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (f) MOSS prompt_patch: config_root/agent_config/NoveltyAuditAgent/prompt_patch.md
# gets read by self.config.prompt_patch() and injected into the system message.
# ---------------------------------------------------------------------------


async def test_prompt_patch_from_config_root_injected_into_system_message(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "NoveltyAuditAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])
    handler = NoveltyAuditAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "novelty.audit_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
