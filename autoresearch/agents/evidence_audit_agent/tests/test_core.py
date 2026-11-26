"""
Tests for EvidenceAuditAgent's pure audit core (Mode A + Mode C).
EvidenceAuditAgent 纯审计核心（Mode A + Mode C）的测试。

No EventBus / Store / Redis / network / search backend — only a FakeLLM
(canned JSON queue). Both ``audit_literature_support`` (Mode A) and
``audit_citation_regression`` (Mode C) are PURE: they take input dicts and an
llm_client, and return a schema-valid report dict.
不涉及 EventBus / Store / Redis / 网络 / 检索后端 —— 只用 FakeLLM（预设 JSON
响应队列）。``audit_literature_support``（Mode A）和
``audit_citation_regression``（Mode C）均为纯函数：接收输入 dict 和
llm_client，返回 schema 合法的报告 dict。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evidence_audit_agent.core import (
    MAX_RETRIES,
    EvidenceAuditLLMError,
    EvidenceAuditParseError,
    _build_citation_regression_user_message,
    _load_citation_regression_system_prompt,
    _load_literature_support_system_prompt,
    audit_citation_regression,
    audit_literature_support,
)
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def test_citation_view_excerpts_oversized_diagnostics_without_mutating_evidence():
    diagnostic = "infrastructure failure: " + "traceback line\n" * 300000 + "; frozen_baseline=3.0"
    memory = {"verified_findings": ["Measured candidate score: 0.5"],
              "negative_results": [diagnostic, "A measured negative result"],
              "best_result_ref": "spec://verified-result"}
    message = _build_citation_regression_user_message(memory, {"topic": "test"})
    assert len(message) < 12000
    assert "Measured candidate score: 0.5" in message
    assert "A measured negative result" in message
    assert "spec://verified-result" in message
    assert "infrastructure failure" in message and "frozen_baseline=3.0" in message
    assert "has not been citation-audited" in message
    assert "state this limitation in notes" in message
    assert memory["negative_results"][0] == diagnostic


def test_citation_view_preserves_normal_negative_observations():
    memory = {"negative_results": ["A causes a verified regression", "B fails validation"]}
    message = _build_citation_regression_user_message(memory, {})
    assert json.dumps(memory, ensure_ascii=False, indent=2) in message
    assert "Diagnostic excerpt" not in message


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EVIDENCE_AUDIT_REPORT_VALIDATOR = _validator("evidence_audit_report.schema.yaml")
CITATION_AUDIT_REPORT_VALIDATOR = _validator("citation_audit_report.schema.yaml")

# ---------------------------------------------------------------------------
# Mode A fixtures
# ---------------------------------------------------------------------------

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

RESEARCH_MEMORY = {
    "verified_findings": ["RAG with dense retrieval beats closed-book QA on the long-tail set."],
    "negative_results": [],
    "excluded_hypotheses": [],
    "consolidated_insights": ["Retrieval quality dominates end-task accuracy."],
    "divergence_seeds": [],
    "next_steps": [],
    "best_metric": None,
    "best_result_ref": None,
}

VALID_CITATION_AUDIT_REPORT_WITH_UNSUPPORTED = {
    "unsupported_claims": [
        {
            "claim": "Retrieval quality dominates end-task accuracy.",
            "why": "No citation in the ResearchSpec or ResearchMemory backs this generalization.",
        }
    ],
    "checked_count": 2,
    "notes": "One insight lacks direct citation support.",
}

VALID_CITATION_AUDIT_REPORT_CLEAN = {
    "unsupported_claims": [],
    "checked_count": 2,
    "notes": None,
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Mode A: audit_literature_support
# ---------------------------------------------------------------------------


def test_mode_a_valid_path_returns_schema_valid_report():
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])

    report = audit_literature_support(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert isinstance(report, dict)
    errors = list(EVIDENCE_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["evidence_strength"] == "strong"
    assert len(llm.calls) == 1


def test_mode_a_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(EvidenceAuditParseError):
        audit_literature_support(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_mode_a_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(VALID_EVIDENCE_AUDIT_REPORT)])

    report = audit_literature_support(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert report["evidence_strength"] == "strong"
    assert len(llm.calls) == 2


def test_mode_a_invalid_support_enum_retries_then_raises_typed_error():
    bad_report = json.loads(_json(VALID_EVIDENCE_AUDIT_REPORT))
    bad_report["claim_citation_matrix"][0]["support"] = "definitely"
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(EvidenceAuditParseError):
        audit_literature_support(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_mode_a_infra_error_raises_llm_error():
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    with pytest.raises(EvidenceAuditLLMError):
        audit_literature_support(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, RaisingLLM())


def test_mode_a_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])

    audit_literature_support(RESEARCH_SPEC, LITERATURE_BRIEF, SPEC_INTERPRETATION, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_literature_support_system_prompt()


def test_mode_a_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])

    audit_literature_support(
        RESEARCH_SPEC,
        LITERATURE_BRIEF,
        SPEC_INTERPRETATION,
        llm,
        prompt_patch="STEER: prefer recent citations",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer recent citations" in system_message
    assert "[MOSS-injected guidance]" in system_message


# ---------------------------------------------------------------------------
# Mode C: audit_citation_regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canned_report",
    [VALID_CITATION_AUDIT_REPORT_WITH_UNSUPPORTED, VALID_CITATION_AUDIT_REPORT_CLEAN],
    ids=["with_unsupported", "clean"],
)
def test_mode_c_valid_path_returns_schema_valid_report(canned_report):
    llm = FakeLLM([_json(canned_report)])

    report = audit_citation_regression(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert isinstance(report, dict)
    errors = list(CITATION_AUDIT_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["checked_count"] == canned_report["checked_count"]
    assert report["unsupported_claims"] == canned_report["unsupported_claims"]
    assert len(llm.calls) == 1


def test_mode_c_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(EvidenceAuditParseError):
        audit_citation_regression(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_mode_c_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(VALID_CITATION_AUDIT_REPORT_CLEAN)])

    report = audit_citation_regression(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert report["unsupported_claims"] == []
    assert len(llm.calls) == 2


def test_mode_c_missing_required_field_retries_then_raises_typed_error():
    bad_report = {"unsupported_claims": []}  # missing checked_count
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(EvidenceAuditParseError):
        audit_citation_regression(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_mode_c_infra_error_raises_llm_error():
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    with pytest.raises(EvidenceAuditLLMError):
        audit_citation_regression(RESEARCH_MEMORY, RESEARCH_SPEC, RaisingLLM())


def test_mode_c_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT_CLEAN)])

    audit_citation_regression(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_citation_regression_system_prompt()


def test_mode_c_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(VALID_CITATION_AUDIT_REPORT_CLEAN)])

    audit_citation_regression(
        RESEARCH_MEMORY,
        RESEARCH_SPEC,
        llm,
        prompt_patch="STEER: prefer recent citations",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer recent citations" in system_message
    assert "[MOSS-injected guidance]" in system_message
