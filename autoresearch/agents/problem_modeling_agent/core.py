"""
ProblemModelingAgent core: the pure clarification-dialogue logic.
ProblemModelingAgent 核心：与传输层无关的纯澄清对话逻辑。

Given the user's research goal, the accumulated draft ResearchSpec, and the
answers gathered so far, ``clarify_step`` calls the injected LLM client and
returns EITHER the next clarification question OR a finalized ResearchSpec —
with NO EventBus / Redis / SpecStore access. The handler is responsible for
resolving refs, persisting the draft/answers, and publishing events.

给定用户的研究目标、累积的草稿 ResearchSpec、以及目前收集到的答案，
``clarify_step`` 调用注入的 LLM 客户端，返回下一个澄清问题或已完成的
ResearchSpec —— 不含任何 EventBus/Redis/SpecStore 访问。handler 负责解析
ref、持久化草稿/答案、发布事件。
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


class ClarifyError(Exception):
    """Base for ProblemModelingAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "clarify_failed"
    #: transient | permanent
    classification: str = "transient"


class ClarifyParseError(ClarifyError):
    """LLM never returned a well-formed ask/finalize JSON after retries, or a
    finalize's spec never validated against the ResearchSpec schema.
    重试用尽后 LLM 仍未返回合法的 ask/finalize JSON，或 finalize 的 spec
    始终未通过 ResearchSpec schema 校验。"""

    code = "clarify_parse_error"
    classification = "permanent"


class ClarifyLLMError(ClarifyError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no retry.
    基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "clarify_llm_error"
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
# Decision object / 决策对象
# ---------------------------------------------------------------------------


@dataclass
class ClarifyDecision:
    """The result of one clarify_step call. / 一次 clarify_step 调用的结果。"""

    done: bool
    question: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Static resources: system prompt + ResearchSpec schema validator
# 静态资源：system prompt + ResearchSpec schema 校验器
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "clarify_system.md"
_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "data-contracts" / "research_spec.schema.yaml"
)

_SPEC_FIELDS = (
    "topic",
    "research_goal",
    "task_mode",
    "budget",
    "deliverable_form",
    "editable_scope",
    "forbidden_items",
    "open_questions",
)


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validate_spec(spec: dict) -> list[str]:
    """Return schema validation error messages (empty list = valid).
    返回 schema 校验错误信息列表（空列表 = 合法）。"""
    return [e.message for e in _spec_validator().iter_errors(spec)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_user_message(
    research_goal_text: str,
    draft_spec: dict,
    answered: list[dict],
    force_finalize: bool,
) -> str:
    lines = [
        f"Research goal (from the user): {research_goal_text}",
        "",
        "Current draft ResearchSpec (JSON, fields may be null/empty):",
        json.dumps(draft_spec, ensure_ascii=False, indent=2),
        "",
        f"Clarification answers gathered so far ({len(answered)}):",
        json.dumps(answered, ensure_ascii=False, indent=2),
    ]
    if force_finalize:
        lines += [
            "",
            "IMPORTANT: The maximum number of clarification questions has "
            'been reached. You MUST respond with {"action": "finalize", '
            '"spec": {...}} now, using your best-effort understanding of '
            "the goal and answers gathered so far. Do NOT ask another "
            "question.",
        ]
    return "\n".join(lines)


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object matching the "
            'required "ask" or "finalize" shape described in the system '
            "prompt."
        ),
    }


# ---------------------------------------------------------------------------
# Validation helpers for the "ask" branch / "ask" 分支的校验辅助
# ---------------------------------------------------------------------------


def _is_valid_question(question: Any) -> bool:
    if not isinstance(question, dict):
        return False
    if not isinstance(question.get("question_id"), str) or not question["question_id"]:
        return False
    if not isinstance(question.get("question_text"), str) or not question["question_text"]:
        return False
    if not isinstance(question.get("allow_free_text"), bool):
        return False
    options = question.get("options")
    if options is not None and not (
        isinstance(options, list) and all(isinstance(o, str) for o in options)
    ):
        return False
    return True


def _build_fallback_spec(draft_spec: dict, research_goal_text: str) -> dict:
    """Best-effort ResearchSpec assembled from the draft when the LLM keeps
    trying to ask past the max-question guard. Only carries the 8 allowed
    keys (additionalProperties: false safe).
    当 LLM 在 max-question guard 之后仍试图提问时，从草稿组装的兜底
    ResearchSpec。只携带 8 个允许的字段（additionalProperties: false 安全）。
    """
    spec = {field: draft_spec.get(field) for field in _SPEC_FIELDS}
    if not spec.get("topic"):
        spec["topic"] = research_goal_text[:80] or "untitled"
    if not spec.get("research_goal"):
        spec["research_goal"] = research_goal_text
    spec["editable_scope"] = spec.get("editable_scope") or []
    spec["forbidden_items"] = spec.get("forbidden_items") or []
    spec["open_questions"] = spec.get("open_questions") or []
    return spec


# ---------------------------------------------------------------------------
# Pipeline / 主流程
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def clarify_step(
    research_goal_text: str,
    draft_spec: dict,
    answered: list[dict],
    llm_client: LLMClientLike,
    max_questions: int = 5,
    *,
    prompt_patch: str | None = None,
) -> ClarifyDecision:
    """Run one clarification turn (pure, no transport).
    跑一轮澄清（纯，无传输层）。

    Args:
        research_goal_text: the user's original research goal.
        draft_spec: the ResearchSpec draft accumulated so far (may have
            null/empty fields).
        answered: list of ``{question_id, answer_text}`` gathered so far.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        max_questions: guard — once ``len(answered) >= max_questions`` the
            LLM is instructed to finalize, and an "ask" reply is coerced
            into a best-effort finalize instead of looping further.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        ClarifyDecision — either ``done=False`` with a well-formed
        ``question``, or ``done=True`` with a schema-valid ``spec``.

    Raises:
        ClarifyParseError (permanent): malformed/unparseable LLM JSON, or an
            invalid finalize ``spec``, after bounded retries.
        ClarifyLLMError (transient): the LLM client itself raised.
    """
    force_finalize = len(answered) >= max_questions

    messages = [
        {"role": "system", "content": apply_prompt_patch(_load_system_prompt(), prompt_patch)},
        {
            "role": "user",
            "content": _build_user_message(
                research_goal_text, draft_spec, answered, force_finalize
            ),
        },
    ]

    last_error = "no attempts made"

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise ClarifyLLMError(
                f"LLM call failed: {type(exc).__name__}: {exc}"
            ) from exc

        content = result.content or ""

        try:
            obj = json.loads(extract_json(content))
        except (json.JSONDecodeError, TypeError):
            last_error = "response was not valid JSON"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        if not isinstance(obj, dict) or "action" not in obj:
            last_error = "response JSON is missing the required 'action' field"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        action = obj.get("action")

        if action == "finalize":
            spec = obj.get("spec")
            if not isinstance(spec, dict):
                last_error = "'finalize' action is missing a 'spec' object"
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue
            errors = _validate_spec(spec)
            if errors:
                last_error = f"spec failed ResearchSpec schema validation: {errors}"
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue
            return ClarifyDecision(done=True, question=None, spec=spec)

        if action == "ask":
            if force_finalize:
                # Max-question guard: coerce a finalize instead of looping.
                fallback = _build_fallback_spec(draft_spec, research_goal_text)
                errors = _validate_spec(fallback)
                if errors:
                    raise ClarifyParseError(
                        "max_questions reached but no valid fallback spec "
                        f"could be built from the draft: {errors}"
                    )
                return ClarifyDecision(done=True, question=None, spec=fallback)

            question = obj.get("question")
            if not _is_valid_question(question):
                last_error = "'ask' action has a malformed 'question' object"
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue
            return ClarifyDecision(done=False, question=question, spec=None)

        last_error = f"unrecognized action {action!r}"
        messages.append({"role": "assistant", "content": content})
        messages.append(_retry_feedback_message(last_error))

    raise ClarifyParseError(
        f"LLM did not produce a valid clarify response after retries: {last_error}"
    )


__all__ = [
    "ClarifyError",
    "ClarifyParseError",
    "ClarifyLLMError",
    "ClarifyDecision",
    "clarify_step",
    "MAX_RETRIES",
]
