"""
ProblemModelingAgent core: the pure ModelingProblem reasoning.
ProblemModelingAgent 核心：与传输层无关的纯 ModelingProblem 推理。

Given a frozen ResearchSpec plus the DeepResearch LiteratureBrief and
NoveltyAuditAgent's NoveltyAuditReport, ``model_step`` calls the injected LLM
client and returns a schema-valid ``ProblemModel`` — with NO EventBus /
Redis / SpecStore access. The handler is responsible for resolving refs,
persisting the result, and publishing events.

给定冻结的 ResearchSpec 以及 DeepResearch 的 LiteratureBrief 和
NoveltyAuditAgent 的 NoveltyAuditReport，``model_step`` 调用注入的 LLM
客户端，返回 schema 合法的 ``ProblemModel`` —— 不含任何 EventBus/Redis/
SpecStore 访问。handler 负责解析 ref、持久化结果、发布事件。
"""

from __future__ import annotations

import json
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


class ModelError(Exception):
    """Base for ModelingProblem core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "model_failed"
    #: transient | permanent
    classification: str = "transient"


class ModelParseError(ModelError):
    """LLM never returned a well-formed ProblemModel JSON after retries, or
    it never validated against its schema.
    重试用尽后 LLM 仍未返回合法的 ProblemModel JSON，或其始终未通过
    schema 校验。"""

    code = "model_parse_error"
    classification = "permanent"


class ModelLLMError(ModelError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no retry.
    基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "model_llm_error"
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

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "model_system.md"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"
_PROBLEM_MODEL_SCHEMA_PATH = _CONTRACTS_DIR / "problem_model.schema.yaml"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _problem_model_validator() -> Draft202012Validator:
    schema = yaml.safe_load(_PROBLEM_MODEL_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validation_errors(validator: Draft202012Validator, obj: Any) -> list[str]:
    """Return schema validation error messages (empty list = valid).
    返回 schema 校验错误信息列表（空列表 = 合法）。"""
    return [e.message for e in validator.iter_errors(obj)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_user_message(
    research_spec: dict,
    literature_brief: dict,
    novelty_audit_report: dict | None,
    human_idea: str | None = None,
) -> str:
    lines = []
    if human_idea:
        # the human REDIRECTED the direction at the checkpoint ("you got it wrong, do X" / a new
        # idea to incorporate). This is an explicit instruction that OVERRIDES the prior modeling --
        # honor it and re-shape the direction around it (meeting 3.4).
        lines += [
            "IMPORTANT — the human has injected this explicit redirection you MUST honor and build "
            "the problem model around (it overrides any earlier direction):",
            human_idea.strip(), "",
        ]
    lines += [
        "Frozen ResearchSpec (JSON):",
        json.dumps(research_spec, ensure_ascii=False, indent=2),
        "",
        "LiteratureBrief (JSON):",
        json.dumps(literature_brief, ensure_ascii=False, indent=2),
    ]
    # 3.3: the novelty audit now runs after the method, so it is absent when the problem is first
    # modelled -- include the block only when a report exists (e.g. a later re-model under a loop).
    if novelty_audit_report is not None:
        lines += [
            "",
            "NoveltyAuditReport (JSON):",
            json.dumps(novelty_audit_report, ensure_ascii=False, indent=2),
        ]
    return "\n".join(lines)


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object with "
            '"key_problems", "opportunities", "risks", and '
            '"verifiable_paths" keys as described in the system prompt.'
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline / 主流程
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def model_step(
    research_spec: dict,
    literature_brief: dict,
    novelty_audit_report: dict | None,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
    human_idea: str | None = None,
) -> dict:
    """Run one ModelingProblem turn (pure, no transport).
    跑一轮 ModelingProblem（纯，无传输层）。

    Args:
        research_spec: the frozen ResearchSpec.
        literature_brief: DeepResearchAgent's LiteratureBrief.
        novelty_audit_report: NoveltyAuditAgent's NoveltyAuditReport.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``ProblemModel`` dict.

    Raises:
        ModelParseError (permanent): malformed/unparseable LLM JSON, or the
            response failing schema validation, after bounded retries.
        ModelLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {"role": "system", "content": apply_prompt_patch(_load_system_prompt(), prompt_patch)},
        {
            "role": "user",
            "content": _build_user_message(
                research_spec, literature_brief, novelty_audit_report, human_idea
            ),
        },
    ]

    last_error = "no attempts made"

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise ModelLLMError(
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

        if not isinstance(obj, dict):
            last_error = "response JSON is not an object"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        errors = _validation_errors(_problem_model_validator(), obj)
        if errors:
            last_error = f"response failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return obj

    raise ModelParseError(
        f"LLM did not produce a valid ProblemModel after retries: {last_error}"
    )


__all__ = [
    "ModelError",
    "ModelParseError",
    "ModelLLMError",
    "model_step",
    "MAX_RETRIES",
]
