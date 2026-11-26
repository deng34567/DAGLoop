"""
NoveltyAuditAgent core: the pure citation + novelty audit logic.
NoveltyAuditAgent 核心：与传输层无关的纯"引用充分性 + 新颖性审计"逻辑。

Given a ResearchSpec, a LiteratureBrief, and a SpecInterpretation,
``audit_step`` calls an injected LLM client to produce a
``NoveltyAuditReport`` judging citation sufficiency and novelty, and
validates it against its data-contract schema — with NO EventBus / Redis /
Store access. The handler (a later milestone) is responsible for resolving
refs, persisting the report, and publishing events.

给定 ResearchSpec、LiteratureBrief 和 SpecInterpretation，``audit_step``
调用注入的 LLM 客户端产出一份 NoveltyAuditReport（判断引用充分性和新颖性），
并对照 data-contract schema 校验 —— 不含任何 EventBus/Redis/Store 访问。
handler（后续里程碑）负责解析 ref、持久化报告、发布事件。
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
from autoresearch.agents.novelty_audit_agent.harness import (
    NoveltyHarnessInputError,
    apply_novelty_harness,
)

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class NoveltyAuditError(Exception):
    """Base for NoveltyAuditAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "novelty_audit_failed"
    #: transient | permanent
    classification: str = "transient"


class NoveltyAuditParseError(NoveltyAuditError):
    """LLM never returned a well-formed, schema-valid NoveltyAuditReport
    after retries. 重试用尽后 LLM 仍未返回合法、schema 校验通过的
    NoveltyAuditReport。"""

    code = "novelty_audit_parse_error"
    classification = "permanent"


class NoveltyAuditLLMError(NoveltyAuditError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "novelty_audit_llm_error"
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
# Static resources: system prompt + report schema validator
# 静态资源：system prompt + report schema 校验器
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "audit_system.md"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "novelty_audit_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_report(report: dict) -> list[str]:
    return [e.message for e in _report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _grounding_constraints(literature_brief: dict) -> str:
    lines = []
    primary = literature_brief.get("searched_method_primary_core")
    if isinstance(primary, dict):
        lines.append(
            "Copy method_signature.primary_core exactly from this frozen object: "
            + json.dumps(primary, ensure_ascii=False)
        )
    paper_ids = sorted({
        value.strip()
        for field in ("key_papers", "related_work")
        for paper in literature_brief.get(field) or []
        if isinstance(paper, dict)
        for value in (paper.get("id_or_url"), paper.get("title"))
        if isinstance(value, str) and value.strip()
    })
    if paper_ids:
        lines.append(
            "Use only these exact LiteratureBrief prior_work_id values: "
            + json.dumps(paper_ids, ensure_ascii=False)
        )
    return "\n".join(lines)


def _build_user_message(
    research_spec: dict,
    literature_brief: dict,
    spec_interpretation: dict,
    evidence_audit_report: dict | None = None,
    method: dict | None = None,
) -> str:
    parts = [
        "ResearchSpec (JSON):",
        json.dumps(research_spec, ensure_ascii=False, indent=2),
        "",
        "LiteratureBrief (JSON):",
        json.dumps(literature_brief, ensure_ascii=False, indent=2),
        "",
        "SpecInterpretation (JSON):",
        json.dumps(spec_interpretation, ensure_ascii=False, indent=2),
        "",
    ]
    if evidence_audit_report is not None:
        parts += [
            "EvidenceAuditReport (JSON):",
            json.dumps(evidence_audit_report, ensure_ascii=False, indent=2),
            "",
        ]
    if method is not None:
        # 3.3: the audit judges the novelty of the proposed METHOD (not the bare task). When a
        # method is supplied, it is the primary subject of the novelty verdict.
        parts += [
            "Proposed METHOD to audit for novelty (JSON):",
            json.dumps(method, ensure_ascii=False, indent=2),
            "",
            _grounding_constraints(literature_brief),
            "",
        ]
    subject = "the proposed METHOD" if method is not None else "the ResearchSpec"
    parts.append(
        f"Audit the novelty of {subject} against the LiteratureBrief and respond "
        "with a single JSON object matching the NoveltyAuditReport shape "
        "described in the system prompt."
    )
    return "\n".join(parts)


def _retry_feedback_message(reason: str, literature_brief: dict) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            f"{_grounding_constraints(literature_brief)}\n"
            "Respond again with ONLY a single JSON object matching the "
            "NoveltyAuditReport shape described in the system prompt."
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline / 主流程
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def audit_step(
    research_spec: dict,
    literature_brief: dict,
    spec_interpretation: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
    evidence_audit_report: dict | None = None,
    method: dict | None = None,
) -> dict:
    """Run one citation + novelty audit turn (pure, no transport).
    跑一轮引用充分性 + 新颖性审计（纯，无传输层）。

    Args:
        research_spec: the (dict-shaped) ResearchSpec being audited.
        literature_brief: DeepResearchAgent's LiteratureBrief.
        spec_interpretation: ProblemModelingAgent's SpecInterpretation.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).
        evidence_audit_report: optional EvidenceAuditAgent Mode A
            EvidenceAuditReport; when given, its claim-citation matrix is
            included in the prompt so the novelty verdict is
            evidence-informed. ``None`` (the default) behaves exactly as
            before -- backward-safe.

    Returns:
        A schema-valid ``NoveltyAuditReport`` dict. Callers read
        ``report["verdict"]`` (``pass`` / ``evidence_insufficient`` /
        ``not_feasible`` -- meeting item 9 added the feasibility verdict).

    Raises:
        NoveltyAuditParseError (permanent): malformed/unparseable LLM JSON,
            or a schema-invalid report (including an invalid ``verdict`` or
            ``novelty_status`` enum value), after bounded retries.
        NoveltyAuditLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {"role": "system", "content": apply_prompt_patch(_load_system_prompt(), prompt_patch)},
        {
            "role": "user",
            "content": _build_user_message(
                research_spec, literature_brief, spec_interpretation, evidence_audit_report, method
            ),
        },
    ]

    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise NoveltyAuditLLMError(
                f"LLM call failed: {type(exc).__name__}: {exc}"
            ) from exc

        content = result.content or ""

        try:
            obj = json.loads(extract_json(content))
        except (json.JSONDecodeError, TypeError):
            last_error = "response was not valid JSON"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error, literature_brief))
            continue

        if not isinstance(obj, dict):
            last_error = "response JSON is not an object"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error, literature_brief))
            continue

        if method is not None:
            try:
                # The model supplies only structured leaf relations.  The host validates their
                # LiteratureBrief provenance and overwrites both final fields from a fixed truth
                # table, so a fluent holistic verdict cannot bypass the novelty policy.
                obj = apply_novelty_harness(obj, literature_brief)
            except NoveltyHarnessInputError as exc:
                last_error = f"novelty harness packet was invalid: {exc}"
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error, literature_brief))
                continue

        errors = _validate_report(obj)
        if errors:
            last_error = f"report failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error, literature_brief))
            continue

        return obj

    raise NoveltyAuditParseError(
        f"LLM did not produce a valid NoveltyAuditReport after retries: {last_error}"
    )


__all__ = [
    "NoveltyAuditError",
    "NoveltyAuditParseError",
    "NoveltyAuditLLMError",
    "audit_step",
    "MAX_RETRIES",
]
