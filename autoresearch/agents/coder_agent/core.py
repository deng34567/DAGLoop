"""
CoderAgent core: the pure code-generation logic. CoderAgent 核心：与传输层
无关的纯"代码生成"逻辑。

Given a frozen ExperimentPlan (the declarative WHAT/WHY produced by
ExperimentPlannerAgent) and the frozen VerifierSpec, ``implement_patch``
calls an injected LLM client to produce an ``ExperimentPatch`` — the ACTUAL
CODE (a diff or new file content, per ``changes[].patch``) that implements
the plan. Validates against the shared data-contract, with NO EventBus /
Redis / Store access. The handler (a later milestone) is responsible for
resolving refs, persisting the artifact, and publishing events.

给定冻结的 ExperimentPlan（ExperimentPlannerAgent 产出的声明式 WHAT/WHY）
与冻结的 VerifierSpec，``implement_patch`` 调用注入的 LLM 客户端产出一份
ExperimentPatch —— 即实现该方案的实际代码（``changes[].patch`` 为 diff 或新
文件内容）。对照 data-contract 校验 —— 不含任何 EventBus/Redis/Store 访问。
handler（后续里程碑）负责解析 ref、持久化产物、发布事件。

KEY DESIGN POINT: no editable_region enforcement here / 关键设计：此处不做
editable_region 护栏
-----------------------------------------------------------------------
Unlike ``experiment_planner_agent.core.plan_experiment`` (which enforces an
editable_region/forbidden_files scope guard in code), this module
DELIBERATELY does NOT police ``verifier_spec["editable_region"]`` /
``verifier_spec["forbidden_files"]``. The system prompt asks the LLM to
comply (see ``prompts/implement_patch_system.md``), but CoderAgent is not
trusted to self-police: a patch that touches an out-of-region or forbidden
file is still returned SUCCESSFULLY here. The orchestrator gate (a later
milestone) is the single, independent enforcement point — it reads the
frozen VerifierSpec itself and checks the returned ``touched_files``,
default-deny. This mirrors the design philosophy "don't trust the coder; the
harness enforces."

与 ``experiment_planner_agent.core.plan_experiment``（在代码中强制
editable_region/forbidden_files 范围护栏）不同，本模块刻意不对
``verifier_spec["editable_region"]``／``verifier_spec["forbidden_files"]``
做任何自我审查。system prompt 会要求 LLM 遵守（见
``prompts/implement_patch_system.md``），但 CoderAgent 不被信任会自我审查：
即便补丁触碰了范围外或禁止的文件，这里依然会成功返回。编排器侧的门禁（后续
里程碑）才是唯一、独立的强制点 —— 它会读取冻结的 VerifierSpec 并检查返回的
``touched_files``，默认拒绝。这与"不信任 coder；由 harness 强制"的设计理念
一致。

touched_files is core-derived / touched_files 由核心计算得出
-----------------------------------------------------------------------
The LLM is never asked to supply ``touched_files``; even if it did, this
module ignores it. ``touched_files`` is always computed here as
``sorted({c["file"] for c in changes})`` -- an honest reflection of what the
returned ``changes[]`` actually touch.

LLM 从不被要求提供 ``touched_files``；即便它提供了，本模块也会忽略。
``touched_files`` 始终在此计算为 ``sorted({c["file"] for c in changes})`` ——
如实反映返回的 ``changes[]`` 实际触及的文件。
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


class CoderError(Exception):
    """Base for CoderAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "coder_failed"
    #: transient | permanent
    classification: str = "transient"


class CoderParseError(CoderError):
    """LLM never returned a well-formed, schema-valid ExperimentPatch after
    retries. 重试用尽后 LLM 仍未返回合法、schema 校验通过的
    ExperimentPatch。"""

    code = "coder_parse_error"
    classification = "permanent"


class CoderLLMError(CoderError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "coder_llm_error"
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
def _load_implement_patch_system_prompt() -> str:
    return (_PROMPTS_DIR / "implement_patch_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _experiment_patch_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "experiment_patch.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_experiment_patch(patch: dict) -> list[str]:
    return [e.message for e in _experiment_patch_validator().iter_errors(patch)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_implement_patch_user_message(experiment_plan: dict, verifier_spec: dict) -> str:
    editable_region = verifier_spec.get("editable_region", [])
    forbidden_files = verifier_spec.get("forbidden_files", [])
    return "\n".join(
        [
            "ExperimentPlan (JSON):",
            json.dumps(experiment_plan, ensure_ascii=False, indent=2),
            "",
            "VerifierSpec (JSON):",
            json.dumps(verifier_spec, ensure_ascii=False, indent=2),
            "",
            # meeting 2026-07-25 item 4: the editable_region is NO LONGER a pre-declared whitelist
            # that restricts the implementer up front. It is a HINT of where the relevant
            # implementation lives -- edit whatever the method actually needs.
            "editable_region (where the relevant implementation lives -- a starting hint, "
            "not a restriction; edit whatever the method needs):",
            json.dumps(editable_region, ensure_ascii=False, indent=2),
            "",
            "forbidden_files (the trusted evaluator/data -- editing them is cheating, not a "
            "real improvement, so leave them alone):",
            json.dumps(forbidden_files, ensure_ascii=False, indent=2),
            "",
            "Implement the ExperimentPlan and respond with a single JSON "
            "object matching the ExperimentPatch shape described in the "
            "system prompt.",
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
    ``CoderParseError``.
    调用 LLM，把 JSON 响应解析为 object，再跑 process（成功返回
    ``(result, None)``，失败返回 ``(None, error_message)``）做校验/提取。
    任何失败均有限重试，用尽后抛 CoderParseError。"""
    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise CoderLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

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

    raise CoderParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# implement_patch / 实现补丁
# ---------------------------------------------------------------------------


def implement_patch(
    experiment_plan: dict,
    verifier_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> dict:
    """Implement the frozen ExperimentPlan as an ExperimentPatch (pure, no
    transport). 把冻结的 ExperimentPlan 实现为 ExperimentPatch（纯，无传输
    层）。

    Does NOT enforce ``verifier_spec``'s ``editable_region`` /
    ``forbidden_files`` -- see module docstring. ``touched_files`` is always
    computed here from ``changes[].file``, never trusted from the LLM.
    不强制 ``verifier_spec`` 的 ``editable_region``／``forbidden_files`` ——
    见模块文档。``touched_files`` 始终在此从 ``changes[].file`` 计算得出，
    绝不信任 LLM 提供的值。

    Args:
        experiment_plan: the frozen ExperimentPlan.
        verifier_spec: the frozen VerifierSpec (editable_region, forbidden_files),
            relayed to the LLM as guidance only.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``ExperimentPatch`` dict with a core-derived
        ``touched_files`` (may include files outside editable_region or in
        forbidden_files -- this module does not filter or reject those).

    Raises:
        CoderParseError (permanent): malformed/unparseable LLM JSON, an
            empty ``changes[]``, or a schema-invalid ExperimentPatch, after
            bounded retries.
        CoderLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_implement_patch_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_implement_patch_user_message(experiment_plan, verifier_spec),
        },
    ]

    def process(obj: dict) -> tuple[dict | None, str | None]:
        changes = obj.get("changes")
        if not changes:
            return None, "response must include at least one change in changes[]"

        try:
            touched_files = sorted({change["file"] for change in changes})
        except (TypeError, KeyError):
            return None, "each changes[] item must be an object with a 'file' key"

        artifact: dict = {
            "summary": obj.get("summary"),
            "changes": changes,
            "touched_files": touched_files,
        }
        if "hypothesis_id" in obj:
            artifact["hypothesis_id"] = obj["hypothesis_id"]

        errors = _validate_experiment_patch(artifact)
        if errors:
            return None, f"response failed ExperimentPatch schema validation: {errors}"

        return artifact, None

    return _run_llm_loop(messages, llm_client, process=process)  # type: ignore[return-value]


__all__ = [
    "CoderError",
    "CoderParseError",
    "CoderLLMError",
    "implement_patch",
    "MAX_RETRIES",
]
