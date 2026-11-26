"""
MemoryAgent core: the pure memory-consolidation logic. MemoryAgent 核心：
与传输层无关的纯"记忆沉淀"逻辑。

Given the frozen EvaluationReport + RunRecord (and, optionally, a prior
ResearchMemory and a VerificationReport), ``consolidate`` calls an injected
LLM client to merge them into an updated ``ResearchMemory`` — consolidating
verified findings, negative results, excluded hypotheses, and insights
("沉淀"), plus divergence seeds and next steps ("发散") — and to pick a
``routing_verdict`` for how the exploration loop should proceed next.
Validates against the shared data-contract, with NO EventBus / Redis / Store
access. The handler (a later milestone) is responsible for resolving refs,
persisting the artifact, and publishing events.

给定冻结的 EvaluationReport + RunRecord（以及可选的既往 ResearchMemory 和
VerificationReport），``consolidate`` 调用注入的 LLM 客户端将其合并为更新后
的 ResearchMemory —— 沉淀已验证发现、负面结果、排除的假设与洞见（"沉淀"），
以及发散方向与下一步（"发散"）—— 并选定探索循环下一步该如何路由的
``routing_verdict``。对照 data-contract 校验 —— 不含任何 EventBus/Redis/Store
访问。handler（后续里程碑）负责解析 ref、持久化产物、发布事件。

Deterministic best-tracking guard / 确定性最佳追踪护栏
-------------------------------------------------------
``research_memory["best_metric"]`` (and ``best_result_ref``) is NEVER trusted
from the LLM's response — it is COMPUTED HERE IN CODE, overwriting whatever
the LLM returned (or omitted):

- if ``evaluation_report.get("verdict") == "new_best"`` AND
  ``verification_report is not None`` AND
  ``verification_report.get("verdict") == "verified"``: the new result is the
  best so far — ``best_metric = run_record.get("metric_value")`` and
  ``best_result_ref = None`` (the pure core has no ref source to attach; the
  handler, a later milestone, is responsible for that).
- otherwise: both fields are carried forward unchanged from
  ``prior_research_memory`` (``None``/``None`` if there is no prior) — the
  new (non-best, or unverified) result must not silently overwrite an
  established best.

This mirrors ``evaluator_agent/core.py``'s ``evaluate_experiment``
anti-reward-hacking guard: the number that matters is computed in code, the
LLM's own claim about it is ignored.

这镜像 ``evaluator_agent/core.py`` 中 ``evaluate_experiment`` 的反作弊护栏：
关键数字在代码中计算，LLM 自身对该数字的说法一律忽略。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Base for MemoryAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "memory_failed"
    #: transient | permanent
    classification: str = "transient"


class MemoryParseError(MemoryError):
    """LLM never returned a well-formed, schema-valid ResearchMemory with a
    valid routing_verdict after retries. 重试用尽后 LLM 仍未返回合法、schema
    校验通过、routing_verdict 合法的 ResearchMemory。"""

    code = "memory_parse_error"
    classification = "permanent"


class MemoryLLMError(MemoryError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "memory_llm_error"
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

_ROUTING_VERDICTS = ("continue_explore", "report", "need_experiment")


@lru_cache(maxsize=1)
def _load_consolidate_system_prompt() -> str:
    return (_PROMPTS_DIR / "consolidate_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _research_memory_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "research_memory.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_research_memory(memory: dict) -> list[str]:
    return [e.message for e in _research_memory_validator().iter_errors(memory)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_consolidate_user_message(
    evaluation_report: dict,
    run_record: dict,
    prior_research_memory: dict | None,
    verification_report: dict | None,
) -> str:
    return "\n".join(
        [
            "EvaluationReport (JSON):",
            json.dumps(evaluation_report, ensure_ascii=False, indent=2),
            "",
            "RunRecord (JSON):",
            json.dumps(run_record, ensure_ascii=False, indent=2),
            "",
            "Prior ResearchMemory (JSON):",
            (
                json.dumps(prior_research_memory, ensure_ascii=False, indent=2)
                if prior_research_memory is not None
                else "none"
            ),
            "",
            "VerificationReport (JSON):",
            (
                json.dumps(verification_report, ensure_ascii=False, indent=2)
                if verification_report is not None
                else "none"
            ),
            "",
            "Consolidate the ResearchMemory and decide the routing_verdict; "
            "respond with a single JSON object matching the shape described "
            'in the system prompt: {"research_memory": {...}, '
            '"routing_verdict": "continue_explore|report|need_experiment"}.',
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
# Shared LLM-call-then-parse loop / 共享的"调用 LLM 再解析"循环
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def _run_llm_loop(
    messages: list[dict],
    llm_client: LLMClientLike,
    *,
    process: Callable[[dict], tuple[object, str | None]],
) -> object:
    """Call the LLM, parse the JSON response as an object, and run
    ``process`` (returning ``(result, None)`` on success or
    ``(None, error_message)`` on failure) to validate/extract the final
    result. Retries (bounded) on any failure, then raises
    ``MemoryParseError``.
    调用 LLM，把 JSON 响应解析为 object，再跑 process（成功返回
    ``(result, None)``，失败返回 ``(None, error_message)``）做校验/提取。
    任何失败均有限重试，用尽后抛 MemoryParseError。"""
    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise MemoryLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

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

        parsed, error = process(obj)
        if error is not None:
            last_error = error
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return parsed

    raise MemoryParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# consolidate / 沉淀+发散
# ---------------------------------------------------------------------------


def consolidate(
    evaluation_report: dict,
    run_record: dict,
    prior_research_memory: dict | None,
    verification_report: dict | None,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> tuple[dict, str]:
    """Consolidate the ResearchMemory and pick a routing verdict (pure, no
    transport). 沉淀 ResearchMemory 并选定路由结果（纯，无传输层）。

    Args:
        evaluation_report: the frozen EvaluationReport for the latest run.
        run_record: the frozen RunRecord the evaluation was based on.
        prior_research_memory: the prior ResearchMemory, or ``None`` if this
            is the first consolidation cycle.
        verification_report: the frozen VerificationReport if the run was
            re-verified, else ``None``.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A tuple ``(research_memory, routing_verdict)`` where
        ``research_memory`` is a schema-valid ``ResearchMemory`` dict and
        ``routing_verdict`` is one of ``"continue_explore"``, ``"report"``,
        ``"need_experiment"``.

    Raises:
        MemoryParseError (permanent): malformed/unparseable LLM JSON, a
            missing/non-dict ``research_memory``, a missing/invalid
            ``routing_verdict``, or a schema-invalid assembled
            ``ResearchMemory``, after bounded retries.
        MemoryLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_consolidate_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_consolidate_user_message(
                evaluation_report, run_record, prior_research_memory, verification_report
            ),
        },
    ]

    # GUARD (deterministic best-tracking, never trusted from the LLM): see
    # module docstring. Computed once, up front, from the frozen inputs.
    if (
        evaluation_report.get("verdict") == "new_best"
        and verification_report is not None
        and verification_report.get("verdict") == "verified"
    ):
        best_metric = run_record.get("metric_value")
        best_result_ref = None
    else:
        prior = prior_research_memory or {}
        best_metric = prior.get("best_metric")
        best_result_ref = prior.get("best_result_ref")

    def process(obj: dict) -> tuple[tuple[dict, str] | None, str | None]:
        routing_verdict = obj.get("routing_verdict")
        if routing_verdict not in _ROUTING_VERDICTS:
            return None, (
                f"routing_verdict must be one of {_ROUTING_VERDICTS}, "
                f"got {routing_verdict!r}"
            )

        research_memory = obj.get("research_memory")
        if not isinstance(research_memory, dict):
            return None, "research_memory must be a JSON object"

        # Overwrite best_metric/best_result_ref with the computed values,
        # regardless of anything the LLM put there.
        research_memory = dict(research_memory)
        research_memory["best_metric"] = best_metric
        research_memory["best_result_ref"] = best_result_ref
        # Component history is host-derived in the repository path. A generic LLM consolidation
        # may preserve it but may never rewrite or discard it.
        research_memory["component_history"] = list(
            (prior_research_memory or {}).get("component_history") or [])

        errors = _validate_research_memory(research_memory)
        if errors:
            return None, f"assembled ResearchMemory failed schema validation: {errors}"

        return (research_memory, routing_verdict), None

    return _run_llm_loop(messages, llm_client, process=process)  # type: ignore[return-value]


__all__ = [
    "MemoryError",
    "MemoryParseError",
    "MemoryLLMError",
    "consolidate",
    "MAX_RETRIES",
]
