"""
Tests for MemoryAgent's pure core. / MemoryAgent 纯核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
``consolidate`` is PURE: it takes an EvaluationReport, a RunRecord, an
optional prior ResearchMemory, an optional VerificationReport, and an
llm_client, and returns a tuple ``(research_memory, routing_verdict)`` where
``research_memory`` is a schema-valid ResearchMemory dict.

The DETERMINISTIC best-tracking guard (crown jewel here): ``best_metric`` is
NEVER trusted from the LLM. It is COMPUTED in code:
  - if ``evaluation_report["verdict"] == "new_best"`` AND
    ``verification_report is not None`` AND
    ``verification_report["verdict"] == "verified"``:
    ``best_metric = run_record["metric_value"]`` (and ``best_result_ref`` is
    left ``None`` — no ref source is available to the pure core).
  - else: both are carried forward from ``prior_research_memory`` (or
    ``None`` if there is no prior).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.memory_agent.core import (
    MAX_RETRIES,
    MemoryParseError,
    _load_consolidate_system_prompt,
    consolidate,
)
from autoresearch.agents.memory_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


RESEARCH_MEMORY_VALIDATOR = _validator("research_memory.schema.yaml")

EVALUATION_REPORT_NEW_BEST = {
    "verdict": "new_best",
    "metric_value": 0.91,
    "baseline_metric": 0.80,
    "is_improvement": True,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": "Metric improved over baseline.",
}

EVALUATION_REPORT_VALID_NOT_BEST = {
    "verdict": "valid_not_best",
    "metric_value": 0.75,
    "baseline_metric": 0.80,
    "is_improvement": False,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": "Metric did not improve over baseline.",
}

RUN_RECORD = {
    "experiment_id": "exp-1",
    "succeeded": True,
    "exit_code": 0,
    "metric_value": 0.91,
    "stdout_excerpt": "training complete",
    "stderr_excerpt": None,
    "environment": "python 3.11, cuda 12.1",
    "seed": "42",
    "runtime_seconds": 120.5,
    "artifacts": [],
}

VERIFICATION_REPORT_VERIFIED = {
    "verdict": "verified",
    "checks": ["re-ran on held-out split"],
    "anti_cheat_flags": [],
    "reasoning": "Reproduced.",
}

VERIFICATION_REPORT_FAILED = {
    "verdict": "failed",
    "checks": ["re-ran on held-out split"],
    "anti_cheat_flags": ["metric mismatch on re-run"],
    "reasoning": "Could not reproduce.",
}

PRIOR_RESEARCH_MEMORY = {
    "best_metric": 0.80,
    "best_result_ref": "research_store://run_record/prior-best",
    "verified_findings": ["Baseline established at 0.80."],
    "negative_results": [],
    "excluded_hypotheses": [],
    "consolidated_insights": ["Larger batch size helps."],
    "divergence_seeds": [],
    "next_steps": [],
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _llm_research_memory(**overrides: object) -> dict:
    memory = {
        "verified_findings": ["New finding: metric improved with technique X."],
        "negative_results": [],
        "excluded_hypotheses": [],
        "consolidated_insights": ["Technique X is promising."],
        "divergence_seeds": ["Try technique Y next."],
        "next_steps": ["Run technique Y."],
    }
    memory.update(overrides)
    return memory


# ---------------------------------------------------------------------------
# (a) valid -> schema-valid ResearchMemory + a valid routing_verdict.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("routing_verdict", ["continue_explore", "report", "need_experiment"])
def test_consolidate_valid_returns_schema_valid_memory_and_verdict(routing_verdict):
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": routing_verdict,
    }
    llm = FakeLLM([_json(response)])

    research_memory, result_verdict = consolidate(
        EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm
    )

    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(research_memory))
    assert errors == [], [e.message for e in errors]
    assert result_verdict == routing_verdict
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) new_best + verified -> best_metric COMPUTED from run_record, not the
# LLM. Feed the LLM a bogus best_metric to prove the override wins.
# ---------------------------------------------------------------------------


def test_consolidate_new_best_verified_computes_best_metric_overriding_llm():
    response = {
        "research_memory": _llm_research_memory(best_metric=-999.0),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    research_memory, _ = consolidate(
        EVALUATION_REPORT_NEW_BEST,
        RUN_RECORD,
        PRIOR_RESEARCH_MEMORY,
        VERIFICATION_REPORT_VERIFIED,
        llm,
    )

    assert research_memory["best_metric"] == RUN_RECORD["metric_value"]
    assert research_memory["best_metric"] != -999.0


# ---------------------------------------------------------------------------
# (c) NOT new_best (or no verification) -> best_metric carried from prior (or
# None when no prior).
# ---------------------------------------------------------------------------


def test_consolidate_not_new_best_carries_forward_prior_best_metric():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    research_memory, _ = consolidate(
        EVALUATION_REPORT_VALID_NOT_BEST,
        RUN_RECORD,
        PRIOR_RESEARCH_MEMORY,
        None,
        llm,
    )

    assert research_memory["best_metric"] == PRIOR_RESEARCH_MEMORY["best_metric"]
    assert research_memory["best_result_ref"] == PRIOR_RESEARCH_MEMORY["best_result_ref"]


def test_consolidate_new_best_without_verification_report_carries_forward_prior():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    research_memory, _ = consolidate(
        EVALUATION_REPORT_NEW_BEST,
        RUN_RECORD,
        PRIOR_RESEARCH_MEMORY,
        None,
        llm,
    )

    assert research_memory["best_metric"] == PRIOR_RESEARCH_MEMORY["best_metric"]


def test_consolidate_new_best_with_failed_verification_carries_forward_prior():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    research_memory, _ = consolidate(
        EVALUATION_REPORT_NEW_BEST,
        RUN_RECORD,
        PRIOR_RESEARCH_MEMORY,
        VERIFICATION_REPORT_FAILED,
        llm,
    )

    assert research_memory["best_metric"] == PRIOR_RESEARCH_MEMORY["best_metric"]


def test_consolidate_no_prior_memory_yields_none_best_metric():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    research_memory, _ = consolidate(
        EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm
    )

    assert research_memory["best_metric"] is None
    assert research_memory["best_result_ref"] is None


# ---------------------------------------------------------------------------
# (d) invalid routing_verdict enum / malformed JSON -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_consolidate_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(MemoryParseError):
        consolidate(EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_consolidate_invalid_routing_verdict_enum_retries_then_raises_typed_error():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "maybe_later",
    }
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(MemoryParseError):
        consolidate(EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_consolidate_schema_invalid_memory_retries_then_raises_typed_error():
    bad_memory = _llm_research_memory()
    del bad_memory["verified_findings"]
    response = {"research_memory": bad_memory, "routing_verdict": "continue_explore"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(MemoryParseError):
        consolidate(EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_consolidate_non_dict_research_memory_retries_then_raises_typed_error():
    response = {"research_memory": "not-an-object", "routing_verdict": "continue_explore"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(MemoryParseError):
        consolidate(EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_consolidate_recovers_after_one_bad_response():
    good_response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "report",
    }
    llm = FakeLLM(["not json at all", _json(good_response)])

    research_memory, verdict = consolidate(
        EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm
    )

    assert verdict == "report"
    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(research_memory))
    assert errors == []
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_consolidate_no_prompt_patch_leaves_system_message_unchanged():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    consolidate(EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_consolidate_system_prompt()


def test_consolidate_prompt_patch_appended_to_system_message():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    llm = FakeLLM([_json(response)])

    consolidate(
        EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm,
        prompt_patch="STEER: prefer X",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


# ---------------------------------------------------------------------------
# (e) real-LLM shape: response wrapped in a markdown ```json fence must still
# parse (proves core.py routes through extract_json, not a bare json.loads).
# ---------------------------------------------------------------------------


def test_consolidate_parses_response_wrapped_in_markdown_json_fence():
    response = {
        "research_memory": _llm_research_memory(),
        "routing_verdict": "continue_explore",
    }
    fenced = f"```json\n{_json(response)}\n```"
    llm = FakeLLM([fenced])

    research_memory, routing_verdict = consolidate(
        EVALUATION_REPORT_VALID_NOT_BEST, RUN_RECORD, None, None, llm
    )

    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(research_memory))
    assert errors == [], [e.message for e in errors]
    assert routing_verdict == "continue_explore"
