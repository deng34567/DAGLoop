"""
Tests for DataAnalysisAgent's pure core.
DataAnalysisAgent 纯核心的测试。

No EventBus / Store / Redis / network / execution -- only a FakeLLM (canned
JSON queue). ``analyze_external_data`` is PURE: it takes external data +
protocol + strictness + an llm_client, and returns a schema-valid
ExternalDataAnalysisReport dict. The LLM ONLY produces N INDEPENDENT
trajectory analyses -- the aggregate ``consensus_status`` verdict
("converged"/"diverged") is computed DETERMINISTICALLY by
``_consensus_status``, which is unit-tested directly for determinism and for
honoring the strictness-raises-the-bar rule.
不涉及 EventBus / Store / Redis / 网络 / 执行 —— 只用 FakeLLM（预设 JSON 响应
队列）。``analyze_external_data`` 为纯函数：接收外部数据 + 协议 + strictness
+ llm_client，返回 schema 合法的 ExternalDataAnalysisReport dict。LLM 只产出
N 条独立的轨迹分析 —— 聚合的 ``consensus_status`` 判定（"converged"/
"diverged"）由 ``_consensus_status`` 确定性计算，对其确定性与
strictness-抬高门槛规则分别做了直接单测。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.data_analysis_agent.core import (
    BASE_THRESHOLD,
    MAX_RETRIES,
    DataAnalysisLLMError,
    DataAnalysisParseError,
    _consensus_status,
    _load_data_analysis_system_prompt,
    analyze_external_data,
)
from autoresearch.agents.data_analysis_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"

REPORT_VALIDATOR = Draft202012Validator(
    yaml.safe_load(
        (_CONTRACTS_DIR / "external_data_analysis_report.schema.yaml").read_text(
            encoding="utf-8"
        )
    )
)


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _trajectories(conclusions: list[str]) -> list[dict]:
    return [
        {"trajectory_id": f"t{i}", "conclusion": conclusion, "notes": None}
        for i, conclusion in enumerate(conclusions)
    ]


def _llm_response(conclusions: list[str]) -> dict:
    return {
        "qc_summary": "Data passed basic QC checks; no missing samples.",
        "trajectories": _trajectories(conclusions),
        "consensus_conclusion": conclusions[0],
        "divergence_points": [],
        "robustness": "high",
        "need_human_review": False,
    }


EXTERNAL_DATA = "sample_id,expr\nS1,12.3\nS2,9.8\n"
EXTERNAL_PROTOCOL = "qPCR protocol v2: normalize against housekeeping gene."


# ---------------------------------------------------------------------------
# (a) mostly-agreeing trajectories -> converged
# ---------------------------------------------------------------------------

MOSTLY_AGREE = [
    "Gene X upregulated",
    "Gene X upregulated",
    "gene x upregulated",
    "Gene X  upregulated",  # extra whitespace, normalizes to the same bucket
    "No effect observed",
]


def test_analyze_external_data_mostly_agreeing_trajectories_converge():
    llm = FakeLLM([_json(_llm_response(MOSTLY_AGREE))])

    report = analyze_external_data(
        EXTERNAL_DATA, EXTERNAL_PROTOCOL, 0, llm, n_trajectories=5
    )

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["consensus_status"] == "converged"
    assert len(report["trajectories"]) == 5
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) evenly split trajectories -> diverged
# ---------------------------------------------------------------------------

EVENLY_SPLIT = [
    "Gene X upregulated",
    "Gene X upregulated",
    "Gene X downregulated",
    "Gene X downregulated",
    "No effect observed",
]


def test_analyze_external_data_evenly_split_trajectories_diverge():
    llm = FakeLLM([_json(_llm_response(EVENLY_SPLIT))])

    report = analyze_external_data(
        EXTERNAL_DATA, EXTERNAL_PROTOCOL, 0, llm, n_trajectories=5
    )

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["consensus_status"] == "diverged"


# ---------------------------------------------------------------------------
# (c) _consensus_status determinism
# ---------------------------------------------------------------------------


def test_consensus_status_is_deterministic_across_calls():
    trajectories = _trajectories(MOSTLY_AGREE)

    result_1 = _consensus_status(trajectories, 0)
    result_2 = _consensus_status(trajectories, 0)

    assert result_1 == result_2 == "converged"


# ---------------------------------------------------------------------------
# (d) strictness raises the bar: top_share 0.6 -> converged at strictness 0,
#     diverged at strictness 1
# ---------------------------------------------------------------------------

BORDERLINE = [
    "Gene X upregulated",
    "Gene X upregulated",
    "Gene X upregulated",
    "No effect observed",
    "Gene X downregulated",
]  # top bucket 3/5 = 0.6


def test_consensus_status_strictness_raises_threshold():
    trajectories = _trajectories(BORDERLINE)

    result_lenient = _consensus_status(trajectories, 0)
    result_strict = _consensus_status(trajectories, 1)

    assert result_lenient == "converged"  # 0.6 >= BASE_THRESHOLD (0.6)
    assert result_strict == "diverged"  # 0.6 < BASE_THRESHOLD + 0.1 (0.7)

    # deterministic under repetition too
    assert _consensus_status(trajectories, 0) == result_lenient
    assert _consensus_status(trajectories, 1) == result_strict


def test_consensus_status_threshold_formula_matches_base_plus_strictness():
    assert BASE_THRESHOLD == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# (e) prompt_patch injected into system message
# ---------------------------------------------------------------------------


def test_analyze_external_data_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(_llm_response(MOSTLY_AGREE))])

    analyze_external_data(
        EXTERNAL_DATA,
        EXTERNAL_PROTOCOL,
        0,
        llm,
        n_trajectories=5,
        prompt_patch="STEER: flag protocol deviations",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: flag protocol deviations" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_analyze_external_data_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(_llm_response(MOSTLY_AGREE))])

    analyze_external_data(EXTERNAL_DATA, EXTERNAL_PROTOCOL, 0, llm, n_trajectories=5)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_data_analysis_system_prompt()


# ---------------------------------------------------------------------------
# (f) malformed LLM JSON -> retry; llm_client.chat raises -> DataAnalysisLLMError
# ---------------------------------------------------------------------------


def test_analyze_external_data_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(DataAnalysisParseError):
        analyze_external_data(EXTERNAL_DATA, EXTERNAL_PROTOCOL, 0, llm, n_trajectories=5)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_analyze_external_data_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(_llm_response(MOSTLY_AGREE))])

    report = analyze_external_data(EXTERNAL_DATA, EXTERNAL_PROTOCOL, 0, llm, n_trajectories=5)

    assert report["consensus_status"] == "converged"
    assert len(llm.calls) == 2


def test_analyze_external_data_infra_error_raises_llm_error():
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    with pytest.raises(DataAnalysisLLMError):
        analyze_external_data(EXTERNAL_DATA, EXTERNAL_PROTOCOL, 0, RaisingLLM(), n_trajectories=5)
