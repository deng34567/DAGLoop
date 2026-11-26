"""
HypothesisAgent core: the pure hypothesis-generation logic.
HypothesisAgent 核心：与传输层无关的纯"假设生成"逻辑。

Given a ResearchSpec and a LiteratureBrief, ``generate_step`` calls an
injected LLM client to produce a ``HypothesisBank`` ({"hypotheses": [...]})
of candidate hypotheses, and validates it against its data-contract schema —
with NO EventBus / Redis / Store access. The handler (a later task) is
responsible for resolving refs, persisting the bank, and publishing events.

给定 ResearchSpec 和 LiteratureBrief，``generate_step`` 调用注入的 LLM 客户端
产出一份 HypothesisBank（候选假设列表），并对照 data-contract schema 校验 ——
不含任何 EventBus/Redis/Store 访问。handler（后续任务）负责解析 ref、持久化
bank、发布事件。
"""

from __future__ import annotations

import json
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


class HypothesisError(Exception):
    """Base for HypothesisAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "hypothesis_failed"
    #: transient | permanent
    classification: str = "transient"


class HypothesisParseError(HypothesisError):
    """LLM never returned a well-formed, schema-valid HypothesisBank (with
    at least one hypothesis and distinct ids) after retries. 重试用尽后 LLM
    仍未返回合法、schema 校验通过（且至少一条、id 互不相同）的
    HypothesisBank。"""

    code = "hypothesis_parse_error"
    classification = "permanent"


class HypothesisLLMError(HypothesisError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "hypothesis_llm_error"
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
# Static resources: system prompt + bank schema validator
# 静态资源：system prompt + bank schema 校验器
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "generate_system.md"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _bank_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "hypothesis_bank.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_bank(bank: dict) -> list[str]:
    return [e.message for e in _bank_validator().iter_errors(bank)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _prior_findings_block(prior_research_memory: dict | None) -> list[str]:
    """A prompt section listing what earlier idea<->experiment loop rounds already found, so
    the next round proposes SOMETHING NEW rather than repeating a tried direction. Empty on
    the first pass (no prior memory)."""
    rm = prior_research_memory or {}
    verified = rm.get("verified_findings") or []
    negative = rm.get("negative_results") or []
    excluded = rm.get("excluded_hypotheses") or []
    if not (verified or negative or excluded):
        return []
    lines = ["", "Accumulated findings from earlier experiment rounds (build on these; do NOT "
             "re-propose a direction already tried):"]
    if verified:
        lines.append("- Verified so far: " + "; ".join(map(str, verified)))
    if negative:
        lines.append("- Negative results so far: " + "; ".join(map(str, negative)))
    if excluded:
        lines.append("- Already tried / excluded: " + "; ".join(map(str, excluded)))
    lines.append("Propose hypotheses that are genuinely DIFFERENT from what was already tried.")
    return lines


def _build_user_message(research_spec: dict, literature_brief: dict, n: int,
                        prior_research_memory: dict | None = None) -> str:
    return "\n".join(
        [
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "LiteratureBrief (JSON):",
            json.dumps(literature_brief, ensure_ascii=False, indent=2),
        ]
        + _prior_findings_block(prior_research_memory)
        + [
            "",
            f"Propose approximately {n} varied candidate hypotheses and respond "
            "with a single JSON object matching the HypothesisBank shape "
            "described in the system prompt.",
        ]
    )


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object matching the "
            "HypothesisBank shape described in the system prompt. Include at "
            "least one hypothesis, and make sure every `id` is unique."
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline / 主流程
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def generate_step(
    research_spec: dict,
    literature_brief: dict,
    llm_client: LLMClientLike,
    n: int = 4,
    *,
    prompt_patch: str | None = None,
    prior_research_memory: dict | None = None,
) -> list[dict]:
    """Run one hypothesis-generation turn (pure, no transport).
    跑一轮假设生成（纯，无传输层）。

    Args:
        research_spec: the (dict-shaped) ResearchSpec driving generation.
        literature_brief: DeepResearchAgent's LiteratureBrief.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        n: target number of candidate hypotheses to ask the LLM for.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A list of schema-valid Hypothesis dicts (validated as a whole
        ``{"hypotheses": [...]}`` HypothesisBank), with at least one entry
        and all distinct ``id``s.

    Raises:
        HypothesisParseError (permanent): malformed/unparseable LLM JSON, a
            schema-invalid bank (including an invalid ``kind`` enum value),
            an empty ``hypotheses`` list, or duplicate ``id``s, after bounded
            retries.
        HypothesisLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {"role": "system", "content": apply_prompt_patch(_load_system_prompt(), prompt_patch)},
        {
            "role": "user",
            "content": _build_user_message(research_spec, literature_brief, n, prior_research_memory),
        },
    ]

    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise HypothesisLLMError(
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

        errors = _validate_bank(obj)
        if errors:
            last_error = f"bank failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        hypotheses = obj["hypotheses"]

        if len(hypotheses) < 1:
            last_error = "hypotheses list was empty"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        ids = [h["id"] for h in hypotheses]
        if len(ids) != len(set(ids)):
            last_error = f"hypothesis ids were not unique: {ids}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return hypotheses

    raise HypothesisParseError(
        f"LLM did not produce a valid HypothesisBank after retries: {last_error}"
    )


__all__ = [
    "HypothesisError",
    "HypothesisParseError",
    "HypothesisLLMError",
    "generate_step",
    "MAX_RETRIES",
]
