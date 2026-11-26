"""
EvidenceAuditAgent core: the pure evidence-grounding audit logic (Mode A +
Mode C). EvidenceAuditAgent 核心：与传输层无关的纯"证据溯源审计"逻辑
（Mode A + Mode C）。

- ``audit_literature_support`` (Mode A, co-audits at
  ``AuditingLiteratureNovelty``): given a ResearchSpec, a LiteratureBrief,
  and a SpecInterpretation, calls an injected LLM client to produce an
  ``EvidenceAuditReport`` — a per-claim citation-support matrix, missing
  papers, expired citations, and an overall evidence-strength verdict — that
  feeds the (evidence-informed) NoveltyAudit.
- ``audit_citation_regression`` (Mode C, a ``Reporting`` pre-check): given a
  ResearchMemory and a ResearchSpec, calls an injected LLM client to produce
  a ``CitationAuditReport`` — claims lacking citation support — that feeds
  ReportAgent so it can drop/flag them, reinforcing the honest-report
  invariant.

Both validate their result against their data-contract schema, with NO
EventBus / Redis / Store access. The handler (a later milestone) is
responsible for resolving refs, persisting the report, and publishing
events.

- ``audit_literature_support``（Mode A，在 ``AuditingLiteratureNovelty``
  联合审计）：给定 ResearchSpec、LiteratureBrief 和 SpecInterpretation，调用
  注入的 LLM 客户端产出 EvidenceAuditReport —— 逐条 claim 的引用支持矩阵、
  缺失论文、过期引用，以及整体证据强度判断 —— 供（证据知情的）NoveltyAudit
  使用。
- ``audit_citation_regression``（Mode C，``Reporting`` 前置检查）：给定
  ResearchMemory 和 ResearchSpec，调用注入的 LLM 客户端产出
  CitationAuditReport —— 缺乏引用支持的 claim 列表 —— 供 ReportAgent
  丢弃/标记之，强化诚实报告不变式。

两者均对照各自 data-contract schema 校验 —— 不含任何 EventBus/Redis/Store
访问。handler（后续里程碑）负责解析 ref、持久化报告、发布事件。
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


class EvidenceAuditError(Exception):
    """Base for EvidenceAuditAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "evidence_audit_failed"
    #: transient | permanent
    classification: str = "transient"


class EvidenceAuditParseError(EvidenceAuditError):
    """LLM never returned a well-formed, schema-valid report after retries.
    重试用尽后 LLM 仍未返回合法、schema 校验通过的报告。"""

    code = "evidence_audit_parse_error"
    classification = "permanent"


class EvidenceAuditLLMError(EvidenceAuditError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "evidence_audit_llm_error"
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
# Static resources: system prompts + schema validators
# 静态资源：system prompt + schema 校验器
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_literature_support_system_prompt() -> str:
    return (_PROMPTS_DIR / "literature_support_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_citation_regression_system_prompt() -> str:
    return (_PROMPTS_DIR / "citation_regression_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _evidence_audit_report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "evidence_audit_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _citation_audit_report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "citation_audit_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_evidence_audit_report(report: dict) -> list[str]:
    return [e.message for e in _evidence_audit_report_validator().iter_errors(report)]


def _validate_citation_audit_report(report: dict) -> list[str]:
    return [e.message for e in _citation_audit_report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_literature_support_user_message(
    research_spec: dict, literature_brief: dict, spec_interpretation: dict
) -> str:
    return "\n".join(
        [
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "LiteratureBrief (JSON):",
            json.dumps(literature_brief, ensure_ascii=False, indent=2),
            "",
            "SpecInterpretation (JSON):",
            json.dumps(spec_interpretation, ensure_ascii=False, indent=2),
            "",
            "Audit the ResearchSpec's claims against the LiteratureBrief and "
            "respond with a single JSON object matching the EvidenceAuditReport "
            "shape described in the system prompt.",
        ]
    )


def _build_citation_regression_user_message(
    research_memory: dict, research_spec: dict
) -> str:
    # Failure diagnostics can contain multi-megabyte tracebacks. Keep the source
    # artifact intact and expose explicit excerpts in this citation-only view.
    memory = dict(research_memory)
    negatives = memory.get("negative_results")
    excerpted = False
    if isinstance(negatives, list):
        memory["negative_results"] = []
        for value in negatives:
            if isinstance(value, str) and len(value) > 8000:
                value = value[:4000] + (
                    "\n[Diagnostic excerpt: omitted text remains in the source "
                    "ResearchMemory artifact and has not been citation-audited.]\n"
                ) + value[-4000:]
                excerpted = True
            memory["negative_results"].append(value)
    return "\n".join(
        [
            "ResearchMemory (JSON):",
            json.dumps(memory, ensure_ascii=False, indent=2),
            "",
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "Check which claims in the ResearchMemory lack citation support "
            "and respond with a single JSON object matching the "
            "CitationAuditReport shape described in the system prompt.",
            ("Some failure diagnostics are excerpted. Audit only the displayed claims "
             "and state this limitation in notes; do not claim the omitted text was checked."
             if excerpted else ""),
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


def _run_llm_loop(messages: list[dict], llm_client: LLMClientLike, *, validate) -> dict:
    """Call the LLM, parse the JSON response as an object, and validate it
    against the given schema validator (returning a list of error messages,
    empty if valid). Retries (bounded) on any failure, then raises
    ``EvidenceAuditParseError``.
    调用 LLM，把 JSON 响应解析为 object，再用给定 schema 校验器校验（返回
    错误信息列表，合法时为空）。任何失败均有限重试，用尽后抛
    EvidenceAuditParseError。"""
    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise EvidenceAuditLLMError(
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

        errors = validate(obj)
        if errors:
            last_error = f"report failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return obj

    raise EvidenceAuditParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# Mode A: audit_literature_support
# ---------------------------------------------------------------------------


def audit_literature_support(
    research_spec: dict,
    literature_brief: dict,
    spec_interpretation: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> dict:
    """Run one literature-support (claim-citation) audit turn (pure, no
    transport). Mode A — co-audits at ``AuditingLiteratureNovelty``.
    跑一轮文献支持度（claim-citation）审计（纯，无传输层）。
    Mode A —— 在 ``AuditingLiteratureNovelty`` 联合审计。

    Args:
        research_spec: the (dict-shaped) ResearchSpec being audited.
        literature_brief: DeepResearchAgent's LiteratureBrief.
        spec_interpretation: ProblemModelingAgent's SpecInterpretation.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``EvidenceAuditReport`` dict.

    Raises:
        EvidenceAuditParseError (permanent): malformed/unparseable LLM JSON,
            or a schema-invalid report, after bounded retries.
        EvidenceAuditLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(
                _load_literature_support_system_prompt(), prompt_patch
            ),
        },
        {
            "role": "user",
            "content": _build_literature_support_user_message(
                research_spec, literature_brief, spec_interpretation
            ),
        },
    ]

    return _run_llm_loop(messages, llm_client, validate=_validate_evidence_audit_report)


# ---------------------------------------------------------------------------
# Mode C: audit_citation_regression
# ---------------------------------------------------------------------------


def audit_citation_regression(
    research_memory: dict,
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> dict:
    """Run one citation-regression audit turn (pure, no transport). Mode C —
    a ``Reporting`` pre-check.
    跑一轮引用回归审计（纯，无传输层）。Mode C —— ``Reporting`` 前置检查。

    Args:
        research_memory: MemoryAgent's consolidated ResearchMemory.
        research_spec: the (dict-shaped) ResearchSpec the memory relates to.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``CitationAuditReport`` dict.

    Raises:
        EvidenceAuditParseError (permanent): malformed/unparseable LLM JSON,
            or a schema-invalid report, after bounded retries.
        EvidenceAuditLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(
                _load_citation_regression_system_prompt(), prompt_patch
            ),
        },
        {
            "role": "user",
            "content": _build_citation_regression_user_message(research_memory, research_spec),
        },
    ]

    return _run_llm_loop(messages, llm_client, validate=_validate_citation_audit_report)


__all__ = [
    "EvidenceAuditError",
    "EvidenceAuditParseError",
    "EvidenceAuditLLMError",
    "audit_literature_support",
    "audit_citation_regression",
    "MAX_RETRIES",
]
