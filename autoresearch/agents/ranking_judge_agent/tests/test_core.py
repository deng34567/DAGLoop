"""
Tests for RankingJudgeAgent's pure ranking core.
RankingJudgeAgent 纯排序核心的测试。

No EventBus / Store / Redis / network / search backend -- only a FakeLLM
(canned JSON queue). ``rank_candidates`` is PURE: it takes a candidate list
and an llm_client, and returns a schema-valid RankingReport dict. The LLM
ONLY produces pairwise judgments -- the aggregate BTL ranking is computed
DETERMINISTICALLY by ``_btl_scores``, which is unit-tested directly for
determinism, dominance, and cycle-handling.
不涉及 EventBus / Store / Redis / 网络 / 检索后端 —— 只用 FakeLLM（预设 JSON
响应队列）。``rank_candidates`` 为纯函数：接收候选列表和 llm_client，返回
schema 合法的 RankingReport dict。LLM 只产出成对判断 —— 聚合 BTL 排序由
``_btl_scores`` 确定性计算，对其确定性、压倒性优势、环形关系分别做了直接
单测。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.ranking_judge_agent.core import (
    MAX_RETRIES,
    RankingLLMError,
    RankingParseError,
    _btl_scores,
    _load_pairwise_ranking_system_prompt,
    rank_candidates,
)
from autoresearch.agents.ranking_judge_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"

RANKING_REPORT_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "ranking_report.schema.yaml").read_text(encoding="utf-8"))
)


def _json(obj: dict) -> str:
    return json.dumps(obj)


CANDIDATES_3 = [
    {"id": "A", "summary": "Use dense retrieval with a reranker."},
    {"id": "B", "summary": "Use sparse retrieval only."},
    {"id": "C", "summary": "Skip retrieval, rely on parametric memory."},
]

# A beats B, A beats C, B beats C -> expected order A, B, C
PAIRWISE_A_DOMINANT = {
    "pairwise_results": [
        {"a": "A", "b": "B", "winner": "A", "rationale": "A is more grounded."},
        {"a": "A", "b": "C", "winner": "A", "rationale": "A is more grounded."},
        {"a": "B", "b": "C", "winner": "B", "rationale": "B still retrieves."},
    ],
    "diversity_status": "diverse",
}


# ---------------------------------------------------------------------------
# (a) rank_candidates: canned pairwise -> deterministic BTL shortlist
# ---------------------------------------------------------------------------


def test_rank_candidates_valid_path_returns_schema_valid_report():
    llm = FakeLLM([_json(PAIRWISE_A_DOMINANT)])

    report = rank_candidates(CANDIDATES_3, "hypothesis", llm)

    assert isinstance(report, dict)
    errors = list(RANKING_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["ranked_shortlist"] == ["A", "B", "C"]
    assert report["diversity_status"] == "diverse"
    assert report["candidate_kind"] == "hypothesis"
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) _btl_scores: determinism + dominance
# ---------------------------------------------------------------------------


def test_btl_scores_is_deterministic_across_calls():
    ids = ["A", "B", "C"]
    scores_1 = _btl_scores(PAIRWISE_A_DOMINANT["pairwise_results"], ids)
    scores_2 = _btl_scores(PAIRWISE_A_DOMINANT["pairwise_results"], ids)

    assert scores_1 == scores_2


def test_btl_scores_clear_dominance_gives_highest_score_to_undefeated():
    # A beats everyone, never loses.
    ids = ["A", "B", "C"]
    pairwise = [
        {"a": "A", "b": "B", "winner": "A"},
        {"a": "A", "b": "C", "winner": "A"},
        {"a": "B", "b": "C", "winner": "B"},
    ]

    scores = _btl_scores(pairwise, ids)
    by_id = {s["id"]: s["score"] for s in scores}

    assert all(math.isfinite(s) for s in by_id.values())
    assert by_id["A"] > by_id["B"] > by_id["C"]


# ---------------------------------------------------------------------------
# (c) _btl_scores: cycle -> finite, stable, deterministic
# ---------------------------------------------------------------------------


def test_btl_scores_cycle_gives_finite_stable_deterministic_scores():
    ids = ["A", "B", "C"]
    # A beats B, B beats C, C beats A -- a perfect cycle.
    pairwise = [
        {"a": "A", "b": "B", "winner": "A"},
        {"a": "B", "b": "C", "winner": "B"},
        {"a": "C", "b": "A", "winner": "C"},
    ]

    scores_1 = _btl_scores(pairwise, ids)
    scores_2 = _btl_scores(pairwise, ids)

    assert scores_1 == scores_2
    for s in scores_1:
        assert math.isfinite(s["score"])
        assert not math.isnan(s["score"])


# ---------------------------------------------------------------------------
# (d) trivial 0/1-candidate short circuit
# ---------------------------------------------------------------------------


def test_rank_candidates_single_candidate_short_circuits_no_llm_call():
    llm = FakeLLM([])

    report = rank_candidates([{"id": "A", "summary": "only one"}], "hypothesis", llm)

    assert report["ranked_shortlist"] == ["A"]
    assert report["pairwise_results"] == []
    assert report["btl_scores"] == [{"id": "A", "score": 1.0}]
    assert llm.calls == []
    errors = list(RANKING_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]


def test_rank_candidates_zero_candidates_short_circuits_empty_shortlist():
    llm = FakeLLM([])

    report = rank_candidates([], "hypothesis", llm)

    assert report["ranked_shortlist"] == []
    assert report["pairwise_results"] == []
    assert report["btl_scores"] == []
    assert llm.calls == []
    errors = list(RANKING_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# (e) prompt_patch injected into system message
# ---------------------------------------------------------------------------


def test_rank_candidates_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(PAIRWISE_A_DOMINANT)])

    rank_candidates(
        CANDIDATES_3,
        "hypothesis",
        llm,
        prompt_patch="STEER: favor novel branches",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: favor novel branches" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_rank_candidates_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(PAIRWISE_A_DOMINANT)])

    rank_candidates(CANDIDATES_3, "hypothesis", llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_pairwise_ranking_system_prompt()


# ---------------------------------------------------------------------------
# (f) malformed LLM JSON -> retry; llm_client.chat raises -> RankingLLMError
# ---------------------------------------------------------------------------


def test_rank_candidates_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(RankingParseError):
        rank_candidates(CANDIDATES_3, "hypothesis", llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_rank_candidates_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(PAIRWISE_A_DOMINANT)])

    report = rank_candidates(CANDIDATES_3, "hypothesis", llm)

    assert report["ranked_shortlist"] == ["A", "B", "C"]
    assert len(llm.calls) == 2


def test_rank_candidates_infra_error_raises_llm_error():
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    with pytest.raises(RankingLLMError):
        rank_candidates(CANDIDATES_3, "hypothesis", RaisingLLM())
