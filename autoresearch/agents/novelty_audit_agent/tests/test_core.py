"""
Tests for NoveltyAuditAgent's pure audit core. / NoveltyAuditAgent 纯审计核心的测试。

No EventBus / Store / Redis / network / search backend — only a FakeLLM
(canned JSON queue). ``audit_step`` is PURE: it takes the three input dicts
(ResearchSpec, LiteratureBrief, SpecInterpretation) and an llm_client, and
returns a schema-valid NoveltyAuditReport dict.
不涉及 EventBus / Store / Redis / 网络 / 检索后端 —— 只用 FakeLLM（预设 JSON
响应队列）。``audit_step`` 是纯函数：接收三份输入 dict（ResearchSpec、
LiteratureBrief、SpecInterpretation）和 llm_client，返回 schema 合法的
NoveltyAuditReport dict。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.novelty_audit_agent.core import (
    MAX_RETRIES,
    NoveltyAuditParseError,
    _load_system_prompt,
    audit_step,
)
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


REPORT_VALIDATOR = _validator("novelty_audit_report.schema.yaml")

RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

LITERATURE_BRIEF = {
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

SPEC_INTERPRETATION = {
    "disposition": "in_scope",
    "rationale": "The proposed direction remains within the original research scope.",
}


def _report_json(report: dict) -> str:
    return json.dumps(report)


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


# ---------------------------------------------------------------------------
# (a) valid path -> schema-valid NoveltyAuditReport; verdict readable.
# ---------------------------------------------------------------------------


def test_valid_path_returns_schema_valid_report():
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])

    report = audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert isinstance(report, dict)
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "pass"
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) both verdicts round-trip through the core unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canned_report",
    [VALID_REPORT_PASS, VALID_REPORT_INSUFFICIENT],
    ids=["pass", "evidence_insufficient"],
)
def test_both_verdicts_round_trip(canned_report):
    llm = FakeLLM([_report_json(canned_report)])

    report = audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == canned_report["verdict"]
    assert report["novelty_status"] == canned_report["novelty_status"]


# ---------------------------------------------------------------------------
# (c) malformed LLM JSON -> bounded retry -> typed error.
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(NoveltyAuditParseError):
        audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (d) invalid `verdict` enum value -> schema validation fails -> retry ->
#     typed error. The bad report must never be returned.
# ---------------------------------------------------------------------------


def test_invalid_verdict_enum_retries_then_raises_typed_error():
    bad_report = {**VALID_REPORT_PASS, "verdict": "maybe"}
    llm = FakeLLM([_report_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(NoveltyAuditParseError):
        audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (d) invalid `novelty_status` enum value -> schema validation fails ->
#     retry -> typed error. The bad report must never be returned.
# ---------------------------------------------------------------------------


def test_invalid_novelty_status_enum_retries_then_raises_typed_error():
    bad_report = {**VALID_REPORT_PASS, "novelty_status": "definitely_new"}
    llm = FakeLLM([_report_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(NoveltyAuditParseError):
        audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Recovery: a malformed first response followed by a valid one succeeds
# without exhausting all retries.
# ---------------------------------------------------------------------------


def test_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _report_json(VALID_REPORT_PASS)])

    report = audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert report["verdict"] == "pass"
    assert len(llm.calls) == 2


def test_invalid_harness_packet_retry_names_frozen_core_and_allowed_papers():
    frozen_core = {
        "core_id": "gpu_tree_search",
        "intervention": "pruned GPU tree traversal",
        "causal_mechanism": "fewer distance checks",
        "pipeline_role": "nearest neighbor lookup",
        "expected_effect": "lower runtime",
    }
    brief = {**LITERATURE_BRIEF, "searched_method_primary_core": frozen_core}
    bad_report = {
        **VALID_REPORT_PASS,
        "method_signature": {"primary_core": {**frozen_core, "core_id": "renamed_core"}},
        "prior_work_comparisons": [{
            "prior_work_id": "https://example.com/unlisted-paper",
            "primary_core_relation": "unknown",
            "evidence": ["Unlisted source"],
        }],
        "retrieval_coverage": {
            "covered_core_ids": ["renamed_core"],
            "compared_prior_work_ids": ["https://example.com/unlisted-paper"],
            "unresolved_candidates": [],
        },
    }
    llm = FakeLLM([_report_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(NoveltyAuditParseError):
        audit_step(RESEARCH_SPEC, brief, SPEC_INTERPRETATION, llm, method={"approach": "GPU tree"})

    assert len(llm.calls) == MAX_RETRIES + 1
    first_prompt = llm.calls[0][1]["content"]
    retry_prompt = llm.calls[-1][-1]["content"]
    for prompt in (first_prompt, retry_prompt):
        assert json.dumps(frozen_core, ensure_ascii=False) in prompt
        assert '"arxiv:2005.11401"' in prompt
        assert "Use only these exact LiteratureBrief prior_work_id values" in prompt


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])

    audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_system_prompt()


def test_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])

    audit_step(
        RESEARCH_SPEC,
        LITERATURE_BRIEF,
        SPEC_INTERPRETATION,
        llm,
        prompt_patch="STEER: prefer X",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_prompt_separates_novelty_from_unmeasured_effectiveness():
    prompt = _load_system_prompt()

    assert "Do not require papers to prove the method's future measured effect" in prompt
    assert "Feasibility and metric improvement are established by the real experiment" in prompt


# ---------------------------------------------------------------------------
# EvidenceAuditAgent Mode A co-audit: an evidence_audit_report, when given, is
# included in the user message so the novelty verdict is evidence-informed.
# Absent (default None) leaves the message exactly as before (asserted by
# test_no_prompt_patch_leaves_system_message_unchanged's sibling coverage
# above for the system message; here we check the user message specifically).
# ---------------------------------------------------------------------------


EVIDENCE_AUDIT_REPORT = {
    "claim_citation_matrix": [
        {
            "claim": "RAG methods generalize to long-tail scientific QA.",
            "citations": ["arxiv:2005.11401"],
            "support": "strong",
        }
    ],
    "missing_papers": [],
    "expired_citations": [],
    "evidence_strength": "strong",
}


def test_evidence_audit_report_included_in_user_message_when_given():
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])

    audit_step(
        RESEARCH_SPEC,
        LITERATURE_BRIEF,
        SPEC_INTERPRETATION,
        llm,
        evidence_audit_report=EVIDENCE_AUDIT_REPORT,
    )

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "EvidenceAuditReport" in user_message
    assert "RAG methods generalize to long-tail scientific QA." in user_message


def test_no_evidence_audit_report_leaves_user_message_unchanged():
    llm = FakeLLM([_report_json(VALID_REPORT_PASS)])

    audit_step(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "EvidenceAuditReport" not in user_message
