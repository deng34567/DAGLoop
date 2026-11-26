"""
ExperimentPlannerAgent core: the pure experiment-planning logic.
ExperimentPlannerAgent 核心：与传输层无关的纯"实验规划"逻辑。

Given a selected Hypothesis (from the frozen HypothesisBank), the frozen
VerifierSpec, BenchmarkSpec, and ResearchSpec, ``plan_experiment`` calls an
injected LLM client to produce a DECLARATIVE ``ExperimentPlan`` — what to
change and why, never actual code — constrained to the VerifierSpec's
``editable_region``. Validates against the shared data-contract, with NO
EventBus / Redis / Store access. The handler (a later milestone) is
responsible for resolving refs, persisting the artifact, and publishing
events.

给定所选 Hypothesis（来自冻结的 HypothesisBank）、冻结的 VerifierSpec、
BenchmarkSpec、ResearchSpec，``plan_experiment`` 调用注入的 LLM 客户端产出一
份声明式的 ExperimentPlan —— 只说"改什么、为什么"，绝不含实际代码 —— 且限定
在 VerifierSpec 的 editable_region 之内。对照 data-contract 校验 —— 不含任何
EventBus/Redis/Store 访问。handler（后续里程碑）负责解析 ref、持久化产物、
发布事件。

Editable-region + no-code guards / editable_region 与无代码护栏
-----------------------------------------------------------------
The returned ``ExperimentPlan`` is checked, at parse time, against TWO
invariants that are enforced HERE IN CODE — independent of whatever the
system prompt asked for (the prompt asks nicely; this module does not trust
that it was obeyed). Mirrors ``evaluator_agent/core.py``'s
``prepare_verifier`` verifier-first guards.

1. **editable_region scope**: every ``changes[i]["file"]`` MUST be a member
   of ``set(verifier_spec["editable_region"])`` AND NOT in
   ``set(verifier_spec["forbidden_files"])``. A plan that proposes touching a
   file outside the frozen editable_region (or one explicitly forbidden)
   would let the candidate solution smuggle changes into out-of-scope code —
   including, in the worst case, the evaluator/harness itself.
2. **no-code (declarative) guard**: none of ``_FORBIDDEN_FIELDS`` ("code",
   "diff", "patch", "implementation", "code_diff") may appear anywhere in the
   parsed object (top-level or inside a ``changes[]`` item). The plan
   declares WHAT should change and WHY; actual code is CoderAgent's job
   downstream. The schema's ``additionalProperties: false`` already blocks
   these structurally on ``changes[]`` items and at the top level — this
   check is belt-and-suspenders plus a precise, actionable error message.

Either violation is treated exactly like malformed JSON — the LLM is given
corrective feedback and retried (bounded), and after retries are exhausted a
``PlannerParseError`` is raised.

违反上述任一护栏都与格式错误的 JSON 同等处理 —— 给 LLM 纠错反馈并有限重试，
用尽后抛出 ``PlannerParseError``。
"""

from __future__ import annotations

import json
import re
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


class ExperimentPlannerError(Exception):
    """Base for ExperimentPlannerAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "experiment_planner_failed"
    #: transient | permanent
    classification: str = "transient"


class PlannerParseError(ExperimentPlannerError):
    """LLM never returned a well-formed, schema-valid, guard-passing
    ExperimentPlan after retries. 重试用尽后 LLM 仍未返回合法、schema 校验
    通过且通过护栏的 ExperimentPlan。"""

    code = "planner_parse_error"
    classification = "permanent"


class PlannerLLMError(ExperimentPlannerError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "planner_llm_error"
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
def _load_plan_experiment_system_prompt() -> str:
    return (_PROMPTS_DIR / "plan_experiment_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _experiment_plan_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "experiment_plan.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_experiment_plan(plan: dict) -> list[str]:
    return [e.message for e in _experiment_plan_validator().iter_errors(plan)]


def _normalize_component_label(value: object) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _component_target_error(selected_hypothesis: dict, plan: dict) -> str | None:
    """Require a component-scoped plan to identify the one frozen DAG node it targets."""
    if selected_hypothesis.get("kind") != "confirmed_method_component":
        return None
    component = selected_hypothesis.get("current_component")
    if not isinstance(component, dict):
        return "selected component context is missing current_component"
    labels = [
        _normalize_component_label(component.get("component_id")),
        _normalize_component_label(component.get("name")),
    ]
    labels = [label for label in labels if label]
    summary = str(plan.get("change_summary") or "")
    declared_target = re.match(
        r"\s*(?:implement|improve|revise|evaluate|test)\s+(?:the\s+)?(.+?)\s+component\b",
        summary, re.IGNORECASE,
    )
    if declared_target and not any(
            label in _normalize_component_label(declared_target.group(1)) for label in labels):
        return (
            "change_summary declares a different component as its target; "
            f"current component is {component.get('component_id')!r} / {component.get('name')!r}"
        )
    # The opening clause states the experiment's target. A later mention of the
    # current component as a dependency must not make a stale plan acceptable.
    target_clause = summary.split(":", 1)[0].split(";", 1)[0].split(".", 1)[0]
    target = _normalize_component_label(target_clause)
    if labels and any(label in target for label in labels):
        return None
    return (
        "change_summary must explicitly name the current frozen component_id or component name; "
        f"current component is {component.get('component_id')!r} / {component.get('name')!r}"
    )


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_plan_experiment_user_message(
    selected_hypothesis: dict,
    verifier_spec: dict,
    benchmark_spec: dict,
    research_spec: dict,
    baseline_result: dict | None = None,
    plan_review: dict | None = None,
) -> str:
    editable_region = verifier_spec.get("editable_region", [])
    forbidden_files = verifier_spec.get("forbidden_files", [])
    scope_explanation = (
        "empty editable_region means no predeclared file boundary; the whole repository is "
        "provisionally editable, except forbidden_files"
        if not editable_region else
        "editable_region is the ONLY set of files changes[].file may target"
    )
    return "\n".join(
        [
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "BenchmarkSpec (JSON):",
            json.dumps(benchmark_spec, ensure_ascii=False, indent=2),
            "",
            "VerifierSpec (JSON):",
            json.dumps(verifier_spec, ensure_ascii=False, indent=2),
            "",
            "Frozen BaselineResult (JSON; authoritative observed runtime evidence, if available):",
            json.dumps(baseline_result or {}, ensure_ascii=False, indent=2),
            "",
            # meeting 甲: this is the approach to implement -- either a hypothesis (lab_in_the_loop)
            # or the CONFIRMED METHOD (kind=confirmed_method) on the benchmark mainline.
            "Selected approach to implement (JSON — a hypothesis, or the confirmed method):",
            json.dumps(selected_hypothesis, ensure_ascii=False, indent=2),
            "",
            f"editable_region ({scope_explanation}):",
            json.dumps(editable_region, ensure_ascii=False, indent=2),
            "",
            "forbidden_files (changes[].file must NEVER target any of these):",
            json.dumps(forbidden_files, ensure_ascii=False, indent=2),
            "",
            "Plan an experiment that IMPLEMENTS the selected approach (the confirmed method, or "
            "the hypothesis) and respond with a single JSON object matching the ExperimentPlan "
            "shape described in the system prompt.",
            *( [
                "",
                "Previous ExperimentPlan review (JSON):",
                json.dumps(plan_review, ensure_ascii=False, indent=2),
                "This is a REPLAN after rejection. Address every issue above and do not repeat a "
                "rejected component or comparison shortcut unchanged.",
            ] if plan_review else [] ),
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
# Editable-region + no-code guards
# editable_region 与无代码护栏
# ---------------------------------------------------------------------------

#: Declarative-plan guard: none of these keys may appear anywhere in the
#: parsed ExperimentPlan (top-level or inside a changes[] item). A
#: declarative plan says WHAT should change and WHY; actual code belongs to
#: CoderAgent, downstream.
#: 声明式方案护栏：这些键不得出现在解析后的 ExperimentPlan 的任何位置
#: （顶层或 changes[] 条目内部）。声明式方案只说"改什么、为什么"；实际代码
#: 属于下游 CoderAgent。
_FORBIDDEN_FIELDS = {"code", "diff", "patch", "implementation", "code_diff"}


def _find_forbidden_fields(obj: object) -> list[str]:
    """Forbidden keys found anywhere (recursively) in ``obj``. Empty = clean
    (OK). 递归查找 obj 中任何位置出现的禁止键。空列表 = 干净（合规）。"""
    found: set[str] = set()

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _FORBIDDEN_FIELDS:
                    found.add(key)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    return sorted(found)


def _find_scope_violations(
    changes: list, editable_region: list, forbidden_files: list
) -> list[str]:
    """``changes[].file`` values that are NOT in ``editable_region`` or ARE
    in ``forbidden_files``. Empty = every change is in-scope (OK).
    不在 editable_region 中或位于 forbidden_files 中的 changes[].file 值。
    空列表 = 所有改动都在范围内（合规）。"""
    editable = set(editable_region)
    forbidden = set(forbidden_files)
    violations = []
    for change in changes:
        file_path = change.get("file") if isinstance(change, dict) else None
        # meeting #1: an EMPTY editable_region means no boundary was pre-declared -> PERMISSIVE
        # (the whole repo is editable); only forbidden_files still blocks. A declared region is
        # still enforced. Anti-cheat is preserved by the trusted evaluator, not this whitelist.
        if (editable and file_path not in editable) or file_path in forbidden:
            violations.append(file_path)
    return violations


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
    ``PlannerParseError``.
    调用 LLM，把 JSON 响应解析为 object，再跑 process（成功返回
    ``(result, None)``，失败返回 ``(None, error_message)``）做校验/提取。
    任何失败均有限重试，用尽后抛 PlannerParseError。"""
    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise PlannerLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

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

    raise PlannerParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# plan_experiment / 规划实验
# ---------------------------------------------------------------------------


def plan_experiment(
    selected_hypothesis: dict,
    verifier_spec: dict,
    benchmark_spec: dict,
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
    baseline_result: dict | None = None,
    plan_review: dict | None = None,
) -> dict:
    """Plan a declarative experiment to test the selected hypothesis (pure,
    no transport). 规划一份声明式实验以检验所选假设（纯，无传输层）。

    Args:
        selected_hypothesis: one Hypothesis from the frozen HypothesisBank.
        verifier_spec: the frozen VerifierSpec (editable_region, forbidden_files).
        benchmark_spec: the frozen BenchmarkSpec.
        research_spec: the frozen ResearchSpec.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid, declarative ``ExperimentPlan`` dict whose
        ``changes[].file`` values are all within ``verifier_spec``'s
        ``editable_region`` and none are in its ``forbidden_files``.

    Raises:
        PlannerParseError (permanent): malformed/unparseable LLM JSON, a
            schema-invalid ExperimentPlan, a changes[].file outside
            editable_region (or inside forbidden_files), or a forbidden
            code/diff/patch/implementation field anywhere in the response,
            after bounded retries.
        PlannerLLMError (transient): the LLM client itself raised.
    """
    if (selected_hypothesis.get("kind") == "confirmed_method_revalidation"
            and selected_hypothesis.get("reconciled_result_ref")
            and not selected_hypothesis.get("applied_method_feedback")
            and (plan_review or {}).get("verdict") != "fail"):
        return {
            "hypothesis_id": selected_hypothesis["id"],
            "change_summary": "Restore and fully revalidate the exact retained source: "
                              + str(selected_hypothesis["reconciled_result_ref"]),
            "changes": [],
            "expected_outcome": (
                "Honor any bound current-Method implementation repair by completing only "
                "already-declared missing behavior and preserving completed repairs; otherwise "
                "evaluate the retained treatment unchanged. Obtain a fresh host-bound result; "
                "its historical score does not establish this run's outcome."),
            "success_criteria": (
                "Run the frozen official evaluator and correctness gates, and review the "
                "whole Method's implementation and task compliance. Preserve the exact "
                "candidate identity, evaluator, metric and comparison. Record failures "
                "or no gain honestly; pending components gain no individual acceptance."),
        }
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_plan_experiment_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_plan_experiment_user_message(
                selected_hypothesis, verifier_spec, benchmark_spec, research_spec,
                baseline_result, plan_review
            ),
        },
    ]

    editable_region = verifier_spec.get("editable_region", [])
    forbidden_files = verifier_spec.get("forbidden_files", [])

    def process(obj: dict) -> tuple[dict | None, str | None]:
        # GUARD (no-code / declarative): checked first, ahead of schema
        # validation, so a forbidden field yields a precise, actionable
        # error rather than a generic "additionalProperties" schema message.
        forbidden_hits = _find_forbidden_fields(obj)
        if forbidden_hits:
            return None, (
                "response must be a DECLARATIVE plan (what to change + why), "
                "never actual code; forbidden field(s) present: "
                f"{forbidden_hits}"
            )

        errors = _validate_experiment_plan(obj)
        if errors:
            return None, f"response failed ExperimentPlan schema validation: {errors}"

        target_error = _component_target_error(selected_hypothesis, obj)
        if target_error:
            return None, target_error

        # GUARD (editable_region scope): every changes[].file must be
        # in-scope, enforced here, not trusted from the prompt.
        violations = _find_scope_violations(
            obj.get("changes", []), editable_region, forbidden_files
        )
        if violations:
            return None, (
                "every changes[].file must be a member of "
                "verifier_spec.editable_region and must not be in "
                f"verifier_spec.forbidden_files; offending files: {violations}"
            )

        return obj, None

    return _run_llm_loop(messages, llm_client, process=process)  # type: ignore[return-value]


__all__ = [
    "ExperimentPlannerError",
    "PlannerParseError",
    "PlannerLLMError",
    "plan_experiment",
    "MAX_RETRIES",
]
