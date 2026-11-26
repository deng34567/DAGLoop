"""
ProblemModelingAgent core: the pure ReinterpretingSpec reasoning + scope gate.
ProblemModelingAgent 核心：与传输层无关的纯 ReinterpretingSpec 推理 + scope gate。

Given a frozen ResearchSpec plus the DeepResearch FieldTrajectoryBrief and
LiteratureBrief, ``reinterpret_step`` calls the injected LLM client and
returns a ``ReinterpretResult`` carrying a schema-valid ``SpecInterpretation``
(whose ``disposition`` is the scope-gate verdict: in_scope / scope_changed /
out_of_scope) and a schema-valid revised ``ResearchSpec`` — with NO EventBus /
Redis / SpecStore access. The handler is responsible for resolving refs,
persisting the results, and publishing events.

给定冻结的 ResearchSpec 以及 DeepResearch 的 FieldTrajectoryBrief 和
LiteratureBrief，``reinterpret_step`` 调用注入的 LLM 客户端，返回携带
schema 合法的 SpecInterpretation（其 ``disposition`` 即 scope-gate
判定：in_scope / scope_changed / out_of_scope）和 schema 合法的修订版
ResearchSpec 的 ``ReinterpretResult`` —— 不含任何 EventBus/Redis/SpecStore
访问。handler 负责解析 ref、持久化结果、发布事件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class ReinterpretError(Exception):
    """Base for ReinterpretingSpec core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "reinterpret_failed"
    #: transient | permanent
    classification: str = "transient"


class ReinterpretParseError(ReinterpretError):
    """LLM never returned a well-formed reinterpretation JSON after retries,
    or one of its two parts never validated against its schema.
    重试用尽后 LLM 仍未返回合法的 reinterpretation JSON，或其两部分之一
    始终未通过对应 schema 校验。"""

    code = "reinterpret_parse_error"
    classification = "permanent"


class ReinterpretLLMError(ReinterpretError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no retry.
    基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "reinterpret_llm_error"
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
# Result object / 结果对象
# ---------------------------------------------------------------------------


@dataclass
class ReinterpretResult:
    """The result of one reinterpret_step call. / 一次 reinterpret_step 调用的结果。"""

    spec_interpretation: dict[str, Any]
    revised_research_spec: dict[str, Any]

    @property
    def disposition(self) -> str:
        """Convenience accessor for the scope-gate verdict.
        scope-gate 判定的便捷访问器。"""
        return self.spec_interpretation["disposition"]


# ---------------------------------------------------------------------------
# Static resources: system prompt + schema validators
# 静态资源：system prompt + schema 校验器
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "reinterpret_system.md"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"
_SPEC_INTERPRETATION_SCHEMA_PATH = _CONTRACTS_DIR / "spec_interpretation.schema.yaml"
_RESEARCH_SPEC_SCHEMA_PATH = _CONTRACTS_DIR / "research_spec.schema.yaml"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _spec_interpretation_validator() -> Draft202012Validator:
    schema = yaml.safe_load(_SPEC_INTERPRETATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _research_spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(_RESEARCH_SPEC_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validation_errors(validator: Draft202012Validator, obj: Any) -> list[str]:
    """Return schema validation error messages (empty list = valid).
    返回 schema 校验错误信息列表（空列表 = 合法）。"""
    errors = []
    for error in validator.iter_errors(obj):
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        errors.append(f"{path}: {error.message}")
    return errors


def _evaluation_evidence_errors(
    spec_interpretation: dict, literature_brief: dict,
) -> list[str]:
    """Validate semantic evidence identifiers against the supplied artifacts."""
    semantics = spec_interpretation.get("evaluation_semantics") or {}
    papers = {
        item.get("id_or_url")
        for key in ("key_papers", "related_work")
        for item in (literature_brief.get(key) or [])
        if isinstance(item, dict) and item.get("id_or_url")
    }
    benchmarks = {
        item.get("id_or_url")
        for item in (literature_brief.get("existing_benchmarks") or [])
        if isinstance(item, dict) and item.get("id_or_url")
    }
    errors: list[str] = []
    for item in semantics.get("evidence") or []:
        source_type = item.get("source_type")
        source_id = item.get("source_id")
        valid = (
            (source_type == "research_spec" and source_id == "research_spec")
            or (source_type == "field_trajectory" and source_id == "field_trajectory")
            or (source_type == "key_paper" and source_id in papers)
            or (source_type == "existing_benchmark" and source_id in benchmarks)
        )
        if not valid:
            errors.append(
                f"evaluation evidence {source_type!r} source_id {source_id!r} "
                "is not present in the supplied research artifacts"
            )
    return errors


def _evaluation_note_errors(spec_interpretation: dict) -> list[str]:
    """Require each new reinterpretation to state its free precaution list.

    The schema keeps the field optional so imported legacy artifacts remain
    readable. New outputs from this core must include it explicitly, even when
    the honest list is empty.
    """
    semantics = spec_interpretation.get("evaluation_semantics") or {}
    if "evaluation_notes" not in semantics:
        return ["evaluation_semantics.evaluation_notes is required"]
    notes = semantics.get("evaluation_notes")
    if not isinstance(notes, list) or not all(
            isinstance(note, str) and note.strip() for note in notes):
        return ["evaluation_semantics.evaluation_notes must be a list of non-empty strings"]
    return []


def _defer_unverified_repository_direction(spec_interpretation: dict) -> None:
    """Keep an ungrounded direction provisional until the live baseline resolves it."""
    semantics = spec_interpretation.get("evaluation_semantics")
    if not isinstance(semantics, dict) or semantics.get("comparison_mode") != "repository_defined":
        return
    primary = semantics.get("primary_metric")
    if not isinstance(primary, dict) or primary.get("direction") in ("minimize", "maximize", "unresolved"):
        return
    primary["direction"] = "unresolved"
    uncertainties = semantics.get("uncertainties")
    if isinstance(uncertainties, list):
        uncertainties.append(
            "Primary metric direction awaits the official repository evaluator and reproduced baseline.")


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_user_message(
    original_user_goal: str,
    research_spec: dict,
    field_trajectory_brief: dict,
    literature_brief: dict,
) -> str:
    lines = [
        "Original user task (verbatim; highest authority):",
        original_user_goal.strip() or "(unavailable for this legacy session; use ResearchSpec)",
        "",
        "Frozen ResearchSpec (JSON):",
        json.dumps(research_spec, ensure_ascii=False, indent=2),
        "",
        "FieldTrajectoryBrief (JSON):",
        json.dumps(field_trajectory_brief, ensure_ascii=False, indent=2),
        "",
        "LiteratureBrief (JSON):",
        json.dumps(literature_brief, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)


def _retry_feedback_message(reason: str) -> dict:
    direction_hint = ""
    if "primary_metric.direction" in reason and "is not one of" in reason:
        direction_hint = (
            " The primary_metric.direction value must be 'minimize' or 'maximize' "
            "when its ordering is known. Only for comparison_mode='repository_defined', "
            "use 'unresolved' when the official evaluator has not yet established it; "
            "record this uncertainty. 'repository_defined' is a comparison_mode, not a direction."
        )
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            f"{direction_hint}\n"
            "Respond again with ONLY a single JSON object with "
            '"spec_interpretation" and "revised_research_spec" keys as '
            "described in the system prompt."
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline / 主流程
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def reinterpret_step(
    research_spec: dict,
    field_trajectory_brief: dict,
    literature_brief: dict,
    llm_client: LLMClientLike,
    *,
    original_user_goal: str = "",
    prompt_patch: str | None = None,
) -> ReinterpretResult:
    """Run one ReinterpretingSpec turn (pure, no transport).
    跑一轮 ReinterpretingSpec（纯，无传输层）。

    Args:
        research_spec: the frozen ResearchSpec to reinterpret.
        field_trajectory_brief: DeepResearchAgent's FieldTrajectoryBrief.
        literature_brief: DeepResearchAgent's LiteratureBrief.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        original_user_goal: exact initial task text; highest-authority intent.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        ReinterpretResult with a schema-valid ``spec_interpretation`` (scope
        gate verdict in ``disposition``) and a schema-valid
        ``revised_research_spec``.

    Raises:
        ReinterpretParseError (permanent): malformed/unparseable LLM JSON, or
            either part failing its schema validation (including an invalid
            ``disposition`` enum value), after bounded retries.
        ReinterpretLLMError (transient): the LLM client itself raised.
    """
    base_messages = [
        {"role": "system", "content": apply_prompt_patch(_load_system_prompt(), prompt_patch)},
        {
            "role": "user",
            "content": _build_user_message(
                original_user_goal, research_spec, field_trajectory_brief, literature_brief
            ),
        },
    ]
    messages = base_messages

    last_error = "no attempts made"

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise ReinterpretLLMError(
                f"LLM call failed: {type(exc).__name__}: {exc}"
            ) from exc

        content = result.content or ""

        try:
            obj = json.loads(extract_json(content))
        except (json.JSONDecodeError, TypeError):
            last_error = "response was not valid JSON"
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        if not isinstance(obj, dict):
            last_error = "response JSON is not an object"
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        spec_interpretation = obj.get("spec_interpretation")
        revised_research_spec = obj.get("revised_research_spec")

        if not isinstance(spec_interpretation, dict) or not isinstance(
            revised_research_spec, dict
        ):
            last_error = (
                "response JSON is missing a 'spec_interpretation' or "
                "'revised_research_spec' object"
            )
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        _defer_unverified_repository_direction(spec_interpretation)

        errors = _validation_errors(_spec_interpretation_validator(), spec_interpretation)
        if errors:
            last_error = f"spec_interpretation failed schema validation: {errors}"
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        errors = _evaluation_note_errors(spec_interpretation)
        if errors:
            last_error = f"evaluation_semantics notes failed validation: {errors}"
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        errors = _evaluation_evidence_errors(spec_interpretation, literature_brief)
        if errors:
            last_error = f"evaluation_semantics evidence failed validation: {errors}"
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        errors = _validation_errors(_research_spec_validator(), revised_research_spec)
        if errors:
            last_error = f"revised_research_spec failed schema validation: {errors}"
            messages = base_messages + [_retry_feedback_message(last_error)]
            continue

        return ReinterpretResult(
            spec_interpretation=spec_interpretation,
            revised_research_spec=revised_research_spec,
        )

    raise ReinterpretParseError(
        f"LLM did not produce a valid reinterpretation after retries: {last_error}"
    )


__all__ = [
    "ReinterpretError",
    "ReinterpretParseError",
    "ReinterpretLLMError",
    "ReinterpretResult",
    "reinterpret_step",
    "MAX_RETRIES",
]
