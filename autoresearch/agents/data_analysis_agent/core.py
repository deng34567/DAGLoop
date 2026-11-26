"""
DataAnalysisAgent core: the pure N-trajectory analysis logic.
DataAnalysisAgent 核心：与传输层无关的纯"N 轨迹分析"逻辑。

``analyze_external_data`` calls an injected LLM client to produce N
INDEPENDENT analysis trajectories over an externally-uploaded dataset, but
the LLM is used ONLY to produce those trajectories (and the accompanying QC
summary / robustness / need_human_review verdicts). The aggregate
``consensus_status`` -- whether the trajectories "converged" or "diverged"
-- is computed DETERMINISTICALLY by ``_consensus_status`` (pure, no LLM
involvement), so the consensus verdict cannot be gamed by a misbehaving or
adversarial LLM response beyond the individual trajectory conclusions it
renders.

``analyze_external_data`` validates its assembled result against
``data-contracts/external_data_analysis_report.schema.yaml``, with NO
EventBus / Redis / Store access. The handler (a later milestone) is
responsible for resolving refs, persisting the report, and publishing
events.

``analyze_external_data`` 调用注入的 LLM 客户端针对外部上传数据集产出 N 条
独立分析轨迹，但 LLM 仅用于产出这些轨迹（以及随附的 QC 摘要 /
robustness / need_human_review 判定）。聚合的 ``consensus_status``——轨迹是
"converged" 还是 "diverged"——由 ``_consensus_status`` 确定性计算（纯函数，
不涉及 LLM），因此共识判定不会被 LLM 单次轨迹产出之外的行为操纵。

``analyze_external_data`` 对照
``data-contracts/external_data_analysis_report.schema.yaml`` 校验其组装
结果 —— 不含任何 EventBus/Redis/Store 访问。handler（后续里程碑）负责解析
ref、持久化报告、发布事件。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class DataAnalysisError(Exception):
    """Base for DataAnalysisAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "data_analysis_failed"
    #: transient | permanent
    classification: str = "transient"


class DataAnalysisParseError(DataAnalysisError):
    """LLM never returned a well-formed, schema-valid report after retries.
    重试用尽后 LLM 仍未返回合法、schema 校验通过的报告。"""

    code = "data_analysis_parse_error"
    classification = "permanent"


class DataAnalysisLLMError(DataAnalysisError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "data_analysis_llm_error"
    classification = "transient"


# ---------------------------------------------------------------------------
# LLM caller protocol / LLM 调用协议
# ---------------------------------------------------------------------------


class LLMResultLike(Protocol):
    content: str | None
    usage: dict
    cost_delta: str


class LLMClientLike(Protocol):
    """Subset of autoresearch.llm.HKGAIClient used here. / 这里用到的客户端子集。"""

    def chat(
        self,
        *,
        messages: list[dict],
        response_format: str | None = ...,
        max_tokens: int | None = ...,
    ) -> LLMResultLike: ...


# ---------------------------------------------------------------------------
# Static resources: system prompt + schema validator
# 静态资源：system prompt + schema 校验器
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_data_analysis_system_prompt() -> str:
    return (_PROMPTS_DIR / "data_analysis_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _external_data_analysis_report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "external_data_analysis_report.schema.yaml").read_text(
            encoding="utf-8"
        )
    )
    return Draft202012Validator(schema)


def _validate_report(report: dict) -> list[str]:
    return [e.message for e in _external_data_analysis_report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_user_message(
    external_data: str, external_protocol: str, strictness: int, n_trajectories: int
) -> str:
    return "\n".join(
        [
            "External data:",
            str(external_data),
            "",
            "External protocol:",
            str(external_protocol),
            "",
            f"Strictness level: {strictness}",
            "",
            f"Produce N={n_trajectories} independent analysis trajectories and "
            "respond with a single JSON object matching the shape described in "
            "the system prompt.",
        ]
    )


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object matching the "
            "shape described in the system prompt."
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic consensus aggregation (the anti-gaming core)
# 确定性共识聚合（反操纵核心）
# ---------------------------------------------------------------------------

#: consensus threshold at strictness=0. Higher strictness demands MORE
#: agreement (see ``_consensus_status``).
#: strictness=0 时的共识门槛。strictness 越高要求的一致性越高
#: （见 ``_consensus_status``）。
BASE_THRESHOLD = 0.6
_THRESHOLD_STEP = 0.1
_THRESHOLD_CAP = 0.99  # kept strictly < 1.0: unanimity should never be *required*


def _normalize_conclusion(conclusion: str) -> str:
    """Lowercase + strip + collapse internal whitespace, so trajectories that
    differ only in case/spacing bucket together.
    小写化 + 去首尾空白 + 折叠内部空白，使仅大小写/空格不同的轨迹归入同一
    桶。"""
    return re.sub(r"\s+", " ", conclusion.strip().lower())


def _consensus_status(trajectories: list[dict], strictness: int) -> str:
    """Compute the "converged"/"diverged" consensus verdict from trajectory
    conclusions. PURE and DETERMINISTIC: no ``random``, no time -- the same
    inputs always produce identical outputs. The LLM does NOT decide this.

    根据轨迹结论计算 "converged"/"diverged" 共识判定。纯函数且确定性：无
    random、无时间依赖 —— 相同输入总是产出相同输出。此判定不由 LLM 决定。

    Args:
        trajectories: list of ``{"conclusion": str, ...}``.
        strictness: non-negative int; higher values raise the agreement
            threshold required to call the trajectories "converged".

    Returns:
        ``"converged"`` if the largest bucket of (normalized) identical
        conclusions covers at least ``threshold`` of all trajectories,
        else ``"diverged"``. Threshold = ``BASE_THRESHOLD + 0.1 *
        strictness``, capped below 1.0.
    """
    if not trajectories:
        return "diverged"

    buckets: dict[str, int] = {}
    for trajectory in trajectories:
        key = _normalize_conclusion(trajectory["conclusion"])
        buckets[key] = buckets.get(key, 0) + 1

    top_share = max(buckets.values()) / len(trajectories)
    threshold = min(BASE_THRESHOLD + _THRESHOLD_STEP * strictness, _THRESHOLD_CAP)

    return "converged" if top_share >= threshold else "diverged"


# ---------------------------------------------------------------------------
# LLM response parsing/validation
# LLM 响应的解析/校验
# ---------------------------------------------------------------------------


def _parse_analysis_response(content: str) -> tuple[dict | None, str | None]:
    """Parse+structurally-validate the LLM's raw trajectory-analysis JSON.
    Returns ``(obj, None)`` on success, or ``(None, reason)`` on failure.
    解析并结构校验 LLM 原始轨迹分析 JSON。成功返回 ``(obj, None)``，失败
    返回 ``(None, reason)``。"""
    try:
        obj = json.loads(extract_json(content))
    except (json.JSONDecodeError, TypeError):
        return None, "response was not valid JSON"

    if not isinstance(obj, dict):
        return None, "response JSON is not an object"

    if not isinstance(obj.get("qc_summary"), str):
        return None, "qc_summary is missing or not a string"

    trajectories = obj.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        return None, "trajectories is missing, not a list, or empty"

    for item in trajectories:
        if not isinstance(item, dict):
            return None, "trajectories item is not an object"
        if not isinstance(item.get("trajectory_id"), str):
            return None, "trajectories item missing string trajectory_id"
        if not isinstance(item.get("conclusion"), str):
            return None, "trajectories item missing string conclusion"
        notes = item.get("notes")
        if notes is not None and not isinstance(notes, str):
            return None, "trajectories item's notes must be a string or null"

    if not isinstance(obj.get("consensus_conclusion"), str):
        return None, "consensus_conclusion is missing or not a string"

    divergence_points = obj.get("divergence_points")
    if not isinstance(divergence_points, list) or not all(
        isinstance(item, str) for item in divergence_points
    ):
        return None, "divergence_points is missing or not a list of strings"

    if obj.get("robustness") not in ("high", "moderate", "low"):
        return None, "robustness must be 'high', 'moderate', or 'low'"

    if not isinstance(obj.get("need_human_review"), bool):
        return None, "need_human_review is missing or not a boolean"

    return obj, None


# ---------------------------------------------------------------------------
# analyze_external_data
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def analyze_external_data(
    external_data: str,
    external_protocol: str,
    strictness: int,
    llm_client: LLMClientLike,
    *,
    n_trajectories: int = 5,
    prompt_patch: str | None = None,
) -> dict:
    """Analyze ``external_data`` via N LLM-produced independent trajectories
    + deterministic consensus aggregation (pure, no transport).
    通过 LLM 产出的 N 条独立轨迹 + 确定性共识聚合分析 ``external_data``
    （纯，无传输层）。

    Args:
        external_data: the uploaded dataset (raw contents or description).
        external_protocol: the experiment protocol the data was collected
            against.
        strictness: non-negative int; passed through to the LLM (for
            context) and to ``_consensus_status`` (to raise the agreement
            threshold).
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        n_trajectories: number of independent trajectories to request.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``ExternalDataAnalysisReport`` dict.

    Raises:
        DataAnalysisParseError (permanent): malformed/unparseable LLM JSON,
            or a schema-invalid assembled report, after bounded retries.
        DataAnalysisLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_data_analysis_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_user_message(
                external_data, external_protocol, strictness, n_trajectories
            ),
        },
    ]

    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise DataAnalysisLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

        content = result.content or ""

        obj, error = _parse_analysis_response(content)
        if error:
            last_error = error
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        report = {
            "qc_summary": obj["qc_summary"],
            "trajectories": obj["trajectories"],
            "consensus_conclusion": obj["consensus_conclusion"],
            "consensus_status": _consensus_status(obj["trajectories"], strictness),
            "divergence_points": obj["divergence_points"],
            "robustness": obj["robustness"],
            "need_human_review": obj["need_human_review"],
        }

        errors = _validate_report(report)
        if errors:
            last_error = f"assembled report failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return report

    raise DataAnalysisParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


__all__ = [
    "DataAnalysisError",
    "DataAnalysisParseError",
    "DataAnalysisLLMError",
    "analyze_external_data",
    "MAX_RETRIES",
    "BASE_THRESHOLD",
]
