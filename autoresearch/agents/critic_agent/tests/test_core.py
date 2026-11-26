"""
Tests for CriticAgent's pure critique core. / CriticAgent 纯批判核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
``critique_direction`` and ``critique_hypotheses`` are PURE: they take input
dicts and an llm_client, and return a schema-valid CritiqueReport dict.
不涉及 EventBus / Store / Redis / 网络 —— 只用 FakeLLM（预设 JSON 响应队列）。
``critique_direction``、``critique_hypotheses`` 是纯函数：接收输入 dict 和
llm_client，返回 schema 合法的 CritiqueReport dict。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.critic_agent.core import (
    MAX_RETRIES,
    CriticParseError,
    _load_direction_system_prompt,
    _load_hypotheses_system_prompt,
    critique_direction,
    critique_hypotheses,
)
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


REPORT_VALIDATOR = _validator("critique_report.schema.yaml")

RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

PROBLEM_MODEL = {
    "key_problems": ["Long-tail entities are poorly retrieved by dense retrievers."],
    "opportunities": ["Hybrid sparse+dense retrieval may close the gap."],
    "risks": ["Benchmark may already be saturated."],
    "verifiable_paths": ["Ablate retriever type on a long-tail QA benchmark."],
}

HYPOTHESIS_BANK = {
    "hypotheses": [
        {
            "id": "h1",
            "statement": "Hybrid sparse+dense retrieval improves long-tail QA accuracy.",
            "kind": "idea",
            "verifiable": True,
        },
        {
            "id": "h2",
            "statement": "Larger context windows alone fix long-tail QA.",
            "kind": "idea",
            "verifiable": True,
        },
    ]
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _validate(report: dict) -> None:
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# critique_direction: both verdicts round-trip, schema-valid, target ==
# "direction".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", ["pass", "fail"])
def test_critique_direction_verdicts_round_trip(verdict):
    report = {
        "target": "direction",
        "verdict": verdict,
        "issues": [] if verdict == "pass" else ["Scope creep beyond the frozen ResearchSpec."],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(report)])

    result = critique_direction(PROBLEM_MODEL, RESEARCH_SPEC, llm)

    _validate(result)
    assert result["target"] == "direction"
    assert result["verdict"] == verdict
    assert len(llm.calls) == 1


def test_critique_direction_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_direction(PROBLEM_MODEL, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_critique_direction_invalid_verdict_enum_retries_then_raises_typed_error():
    bad_report = {
        "target": "direction",
        "verdict": "maybe",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_direction(PROBLEM_MODEL, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# critique_hypotheses: some_pass -> passing_hypothesis_ids subset of bank ids,
# non-empty; all_rejected -> empty ids.
# ---------------------------------------------------------------------------


def test_critique_hypotheses_some_pass_returns_ids_subset_of_bank():
    report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": ["h2 is too narrow / trivially confounded."],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM([_json(report)])

    result = critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    _validate(result)
    assert result["target"] == "hypotheses"
    assert result["verdict"] == "some_pass"
    bank_ids = {h["id"] for h in HYPOTHESIS_BANK["hypotheses"]}
    assert result["passing_hypothesis_ids"]
    assert set(result["passing_hypothesis_ids"]).issubset(bank_ids)
    assert len(llm.calls) == 1


def test_critique_hypotheses_all_rejected_returns_empty_ids():
    report = {
        "target": "hypotheses",
        "verdict": "all_rejected",
        "issues": ["h1 is not verifiable.", "h2 is not verifiable."],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(report)])

    result = critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    _validate(result)
    assert result["verdict"] == "all_rejected"
    assert result["passing_hypothesis_ids"] == []
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# critique_hypotheses: LLM returns a passing id NOT in the bank -> typed
# error (grounding guard, mirrors deep_research_agent's key_papers check).
# ---------------------------------------------------------------------------


def test_critique_hypotheses_ungrounded_passing_id_retries_then_raises_typed_error():
    bad_report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": ["h1", "h999-not-in-bank"],
    }
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# critique_hypotheses: verdict/ids inconsistency -> typed error.
# ---------------------------------------------------------------------------


def test_critique_hypotheses_some_pass_with_empty_ids_raises_typed_error():
    bad_report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_critique_hypotheses_all_rejected_with_nonempty_ids_raises_typed_error():
    bad_report = {
        "target": "hypotheses",
        "verdict": "all_rejected",
        "issues": [],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# critique_hypotheses: malformed JSON / invalid verdict enum -> retry ->
# typed error.
# ---------------------------------------------------------------------------


def test_critique_hypotheses_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_critique_hypotheses_invalid_verdict_enum_retries_then_raises_typed_error():
    bad_report = {
        "target": "hypotheses",
        "verdict": "maybe",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(CriticParseError):
        critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Recovery: a malformed first response followed by a valid one succeeds
# without exhausting all retries.
# ---------------------------------------------------------------------------


def test_critique_hypotheses_recovers_after_one_bad_response():
    good_report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM(["not json at all", _json(good_report)])

    result = critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    assert result["verdict"] == "some_pass"
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected, for both
# critique_direction and critique_hypotheses.
# ---------------------------------------------------------------------------


def test_critique_direction_no_prompt_patch_leaves_system_message_unchanged():
    report = {
        "target": "direction",
        "verdict": "pass",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(report)])

    critique_direction(PROBLEM_MODEL, RESEARCH_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_direction_system_prompt()


def test_critique_direction_prompt_patch_appended_to_system_message():
    report = {
        "target": "direction",
        "verdict": "pass",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(report)])

    critique_direction(PROBLEM_MODEL, RESEARCH_SPEC, llm, prompt_patch="STEER: prefer X")

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_critique_hypotheses_no_prompt_patch_leaves_system_message_unchanged():
    report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM([_json(report)])

    critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_hypotheses_system_prompt()


def test_critique_hypotheses_prompt_patch_appended_to_system_message():
    report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM([_json(report)])

    critique_hypotheses(
        HYPOTHESIS_BANK, RESEARCH_SPEC, llm, prompt_patch="STEER: prefer X"
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


# ---------------------------------------------------------------------------
# critique_hypotheses: RankingJudgeAgent milestone Task 4 -- an optional
# ``ranking_report`` is included in the user message (rank-informed critique)
# when given, and omitted (unchanged behavior) when absent/None.
# ---------------------------------------------------------------------------

RANKING_REPORT = {
    "candidate_kind": "hypothesis",
    "pairwise_results": [{"a": "h1", "b": "h2", "winner": "h1", "rationale": "h1 is stronger."}],
    "btl_scores": [{"id": "h1", "score": 0.7}, {"id": "h2", "score": 0.3}],
    "ranked_shortlist": ["h1", "h2"],
    "diversity_status": "diverse",
}


def test_critique_hypotheses_ranking_report_included_in_user_message():
    report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM([_json(report)])

    critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm, ranking_report=RANKING_REPORT)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "RankingReport" in user_message
    assert "diverse" in user_message
    assert "0.7" in user_message


def test_critique_hypotheses_no_ranking_report_omits_it_from_user_message():
    report = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": [],
        "passing_hypothesis_ids": ["h1"],
    }
    llm = FakeLLM([_json(report)])

    critique_hypotheses(HYPOTHESIS_BANK, RESEARCH_SPEC, llm)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "RankingReport" not in user_message
