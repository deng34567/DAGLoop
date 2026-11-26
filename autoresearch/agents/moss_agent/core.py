"""
MOSS decide core: the pure Sync-Intervention decision logic. MOSS 核心：
与传输层无关的纯"同步介入决策"逻辑。

Given a stuck-pattern ``reason`` (and an optional counter snapshot for
diagnostic context), ``decide`` calls an injected LLM client to propose an
``action``. The ``guard`` (a ``MossActionGuard`` from
``autoresearch.orchestrator.moss_matrix``) is INJECTED here, never
constructed — it is the sole source of truth for which actions are legal for
a given reason, and for the conservative fallback action. No EventBus / Redis
/ Store access; the handler (a later milestone) is responsible for delivering
patches and persisting the artifact.

给定卡顿模式 ``reason``（及可选的计数器快照作诊断上下文），``decide`` 调用
注入的 LLM 客户端提出一个 ``action``。``guard``（来自
``autoresearch.orchestrator.moss_matrix`` 的 ``MossActionGuard``）在此处是
被注入的，绝不在核心内构造——它是"某 reason 下哪些 action 合法"以及"保守兜底
action"的唯一真源。不含任何 EventBus/Redis/Store 访问；handler（后续里程碑）
负责投递 patch 与持久化产物。

Never-hard-stall fallback / 永不硬卡死的兜底
---------------------------------------------
The ONE hard invariant: the returned ``action`` MUST be legal for ``reason``
per ``guard.is_legal``. If the LLM cannot produce a legal, leak-free decision
within the bounded retry budget (``MAX_RETRIES``), ``decide`` does **not**
raise — that would dump the whole session into a hard failure over an LLM
hiccup. Instead it DEGRADES to ``guard.fallback(reason)`` (a conservative,
guaranteed-legal action), marks the decision ``degraded=True``, and returns
normally. Only when ``strict=True`` (intended for tests / stricter callers)
does retry-exhaustion raise ``MossQualityExhausted`` instead.

This mirrors ``memory_agent/core.py``'s retry-then-typed-error pattern, but
replaces the "raise on exhaustion" ending with a legal, observable degrade —
the MOSS-specific requirement that a supervisor which is itself malfunctioning
must never be the thing that stalls the system it supervises.

这镜像 ``memory_agent/core.py`` 的"重试后抛类型化错误"模式，但把"用尽后抛出"
的结局换成了一个合法、可观测的降级——这是 MOSS 特有的要求：监督者自身出故障
时，绝不能成为卡死被监督系统的那个环节。
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
from autoresearch.agents.reflection_agent.core import METHOD_DIAGNOSTIC_DIMENSIONS

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class MossError(Exception):
    """Base for MOSS core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "moss_failed"
    #: transient | permanent
    classification: str = "transient"


class MossLLMError(MossError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "moss_llm_error"
    classification = "transient"


class MossQualityExhausted(MossError):
    """``strict=True`` only: retries exhausted without a legal, leak-free
    decision. Permanent. Note that in normal (non-strict) operation
    ``decide`` does NOT raise this — it degrades to a conservative fallback
    action instead (see module docstring).
    仅 ``strict=True`` 时：重试用尽仍未得到合法、无泄露的决策。permanent。
    注意常规（非 strict）操作下 ``decide`` 不会抛出此异常——而是降级为保守兜底
    action（见模块文档）。"""

    code = "moss_quality_exhausted"
    classification = "permanent"


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


class ActionGuardLike(Protocol):
    """Subset of autoresearch.orchestrator.moss_matrix.MossActionGuard used
    here. / 这里用到的 guard 子集。"""

    def legal_actions(self, reason: str) -> list[str]: ...

    def is_legal(self, reason: str, action: str) -> bool: ...

    def fallback(self, reason: str) -> str: ...


# ---------------------------------------------------------------------------
# Static resources: system prompt + schema validator
# 静态资源：system prompt + schema 校验器
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


@lru_cache(maxsize=1)
def _load_sync_intervention_system_prompt() -> str:
    return (_PROMPTS_DIR / "sync_intervention_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _moss_decision_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "moss_decision.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_moss_decision(decision: dict) -> list[str]:
    return [e.message for e in _moss_decision_validator().iter_errors(decision)]


# ---------------------------------------------------------------------------
# Self-check leak guard / 自检泄露护栏
# ---------------------------------------------------------------------------
#
# A prompt_patch's content must be GENERAL guidance — it must never reveal
# orchestration details (that MOSS intervened, its retry counters/thresholds).
# Task acceptance thresholds and algorithmic counters are legitimate feedback,
# not orchestration leaks. Do not reject them by an unqualified keyword match.

_LEAK_KEYWORDS = ("moss", "thrash")
_LEAK_ROUND_RE = re.compile(r"第.*轮")
_LEAK_SCHEDULER_RE = re.compile(
    r"\b(?:session|retry|retries|revision|orchestrator|orchestration)[ _-]+"
    r"(?:(?:failure|attempt|revision)[ _-]+)?(?:counters?|thresholds?|limits?|budgets?)\b"
    r"|(?:会话|重试|编排|调度)[的 _-]*(?:计数器?|阈值|上限|预算)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Patchable-agent whitelist / 可打补丁 agent 白名单
# ---------------------------------------------------------------------------
#
# A prompt_patch may only target one of these 10 LLM agents -- never
# RunnerAgent/KnowledgeIndexAgent (non-LLM, no prompt to patch) or MOSS
# itself (MOSS does not patch its own prompt through this mechanism).

PATCHABLE_AGENTS = {
    "ProblemModelingAgent",
    "DeepResearchAgent",
    "NoveltyAuditAgent",
    "EvidenceAuditAgent",
    "HypothesisAgent",
    "CriticAgent",
    "EvaluatorAgent",
    "ExperimentPlannerAgent",
    "CoderAgent",
    "MemoryAgent",
    "ReportAgent",
    "RankingJudgeAgent",
    "DataAnalysisAgent",
}


# For a FIXABLE stuck pattern (one an LLM writer/designer agent is responsible for and
# can correct on its NEXT attempt), the responsible agent MOSS should feed corrective
# guidance to. This is how MOSS "gets in and FIXES" the problem while respecting the
# anti-cheat isolation: MOSS never edits code/artifacts itself -- it routes SPECIFIC
# corrective guidance (a prompt_patch) to the writer, who fixes it on the next attempt.
# Only reasons with an unambiguous responsible writer are mapped; the rest (cost/
# verifier/agent_failure/theory/benchmark scope) have no single writer to feed. Every
# target MUST be in PATCHABLE_AGENTS.
_RESPONSIBLE_AGENT: dict[str, str] = {
    "direction_thrashing": "ProblemModelingAgent",
    "literature_novelty_audit_failed": "DeepResearchAgent",
    "baseline_reproduction_thrashing": "CoderAgent",
    "hypothesis_stagnation": "HypothesisAgent",
    "search_diversity_collapse": "HypothesisAgent",
    "experiment_design_thrashing": "ExperimentPlannerAgent",
    "implementation_thrashing": "CoderAgent",
    "result_suspicious": "CoderAgent",
    "data_analysis_consensus_failed": "DataAnalysisAgent",
    "protocol_violation": "CoderAgent",
}


def _patch_content_leaks(content: str) -> bool:
    low = content.lower()
    if any(keyword in low for keyword in _LEAK_KEYWORDS):
        return True
    return bool(_LEAK_ROUND_RE.search(content) or _LEAK_SCHEDULER_RE.search(content))


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_decide_user_message(
    reason: str, counter_snapshot: dict | None, legal_actions: list[str],
    responsible: str | None = None, research_memory: dict | None = None,
    diagnostic: dict | None = None,
    method_context: dict | None = None,
    execution_context: dict | None = None,
) -> str:
    lines = [
        f"MOSS intervention reason: {reason}",
        f"Legal actions for this reason (you MUST pick exactly one of these): "
        f"{legal_actions}",
        "",
        "Counter snapshot (JSON):",
        (
            json.dumps(counter_snapshot, ensure_ascii=False, indent=2)
            if counter_snapshot is not None
            else "none"
        ),
        "",
    ]
    # The ACTUAL artifact that triggered this intervention: either a failed RunRecord or the
    # CritiqueReport that rejected a plan.  MOSS still never edits the solution artifact; it
    # feeds the responsible writer.
    if diagnostic:
        if "scope" in diagnostic:
            lines += [
                "Triggering evidence identity (host bound; not inferred from current Method):",
                json.dumps(diagnostic["scope"], ensure_ascii=False),
                "Current failure classification and measurements:",
                json.dumps(diagnostic.get("result") or {}, ensure_ascii=False),
            ]
        lines += [
            "Actual failure evidence from the triggering artifact (context only):",
            json.dumps(diagnostic.get("record", diagnostic), ensure_ascii=False, indent=2)[:4000],
            "",
        ]
    # 3.7: feed the consolidated ResearchMemory (verified/negative findings accumulated so far)
    # as diagnostic context so MOSS's advice is informed by what the run has already learned --
    # NOT trusted for legality (the guard remains the sole source of legal actions).
    if research_memory is not None:
        candidates = research_memory.get("reconciliation_candidates") or []
        if candidates:
            lines += [
                "Earlier off-Method observations may contradict a claimed method ceiling. "
                "They are not accepted results: compare their actual mechanisms and evidence "
                "before claiming that the current mechanism is impossible or reusing a score.",
                "Recent off-Method candidate refs and insights:",
                json.dumps({
                    "candidates": candidates[-3:],
                    "insights": (research_memory.get("consolidated_insights") or [])[-5:],
                }, ensure_ascii=False)[:2500],
            ]
        lines += [
            "Consolidated ResearchMemory so far (verified/negative findings; context only):",
            "This memory can span older Method versions. Unversioned prose and old component "
            "contracts do not override the current Method or triggering evidence identity.",
            json.dumps(research_memory, ensure_ascii=False, indent=2),
            "",
        ]
    if method_context:
        lines += [
            "Current Method version and frozen task requirements (design intent, not empirical "
            "proof). Check allowed edits, prohibited inputs, interfaces, evaluation semantics, "
            "and fixed settings before suggesting a Method or component change:",
            json.dumps(method_context, ensure_ascii=False, indent=2),
            "",
        ]
        if method_context.get("method_ref"):
            lines += [
                "Bind your decision to the current Method, not an older same-named component.",
                "Return method_ref exactly: " + method_context["method_ref"],
                "Return diagnostic_ref exactly: " + json.dumps(
                    (diagnostic or {}).get("scope", {}).get("ref")),
                "Historical or unscoped evidence cannot establish that the current design failed. "
                "Use it only as a labeled historical observation. A revised contract supersedes "
                "the old contract's repair instructions; do not copy those instructions into a new patch.",
                "For change_plan, revise_component, or discard_hypothesis with component contracts, "
                "also return plan_diagnosis: {component_id, plan_quote}. Quote the exact current "
                "mechanism or check you are diagnosing, not a prior version's wording. "
                "For an execution-only repair, choose an appropriate implementation action.",
                "",
            ]
    if reason == "method_not_working":
        lines += [
            "The legal action is a Method-level replacement after failed experiments. "
            "Use the existing eight diagnostic perspectives to explain the observed failure "
            "in one evidence-grounded hypothesis; do not perform a second reflection call or "
            "mistake an untested idea for a measured result:",
            json.dumps(METHOD_DIAGNOSTIC_DIMENSIONS, ensure_ascii=False),
            "In rationale, identify the relevant perspective, the supplied evidence that "
            "supports it, and a different testable mechanism for the next Method. Preserve "
            "valid prior observations and the frozen task metric and constraints. If evidence "
            "cannot distinguish a mechanism failure from an implementation or environment "
            "failure, say so rather than asserting the Method was falsified.",
            "",
        ]
    if execution_context is not None:
        lines += [
            "Configured sandbox execution interface (host configuration):",
            json.dumps(execution_context, ensure_ascii=False),
            "When sandbox_native_tool_calling is false, the sandbox uses the "
            "SDK-emulated function interface, not native tool calls. Follow the SDK's "
            "provided syntax; do not prohibit its markup or instruct a switch to native "
            "calls. When true, use the provided native tool interface. A prompt_patch "
            "does not authorize changing the execution backend or its configuration.",
            "",
        ]
    if responsible is not None:
        # CLOSE THE LOOP: this is a fixable quality problem. MOSS must not merely report
        # it -- it must get in and FIX it by feeding the responsible writer SPECIFIC
        # corrective guidance so the NEXT attempt fixes the root cause. MOSS never edits
        # the code/artifact itself (that would break the writer/ruler isolation); it
        # feeds the writer. The guidance must still be GENERAL (leak-safe).
        lines.append(
            f"This is a FIXABLE problem: the {responsible} is responsible for the "
            "artifact at fault. Do NOT just report it -- CLOSE THE LOOP: besides picking "
            f"an action, you MUST propose a prompt_patch targeting {responsible} with "
            "SPECIFIC, concrete corrective guidance that would make its next attempt fix "
            "the root cause. You never edit the code/artifact yourself -- you feed the "
            "writer. The guidance must be GENERAL advice (never mention this session's "
            "counters, thresholds, round numbers, or MOSS). Task-defined acceptance "
            "thresholds and algorithmic counters are allowed and must remain faithful "
            "to the supplied task evidence; do not lower the task gate to obtain a pass."
        )
        lines.append("")
    response_shape = {"action": "...", "prompt_patch": {
        "agent": "...", "content": "..."}, "rationale": "..."}
    if (method_context or {}).get("method_ref"):
        response_shape.update({
            "method_ref": method_context["method_ref"],
            "diagnostic_ref": (diagnostic or {}).get("scope", {}).get("ref"),
        })
    lines.append(
        "Diagnose the root cause of this stuck pattern, pick one legal action "
        "from the list above, and "
        + ("propose the corrective prompt_patch described above"
           if responsible is not None
           else "optionally propose a GENERAL prompt_patch "
                "(never session/counter/threshold-specific) for another agent")
        + '. Respond with a single JSON object: '
        + json.dumps(response_shape, ensure_ascii=False)
        + ' Use null for prompt_patch when no patch is appropriate. '
        'Include top-level method_ref, diagnostic_ref, and action-dependent plan_diagnosis '
        'when required above; the base shape does not replace those evidence fields.'
    )
    return "\n".join(lines)


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
# decide / 决策
# ---------------------------------------------------------------------------


# E13 anti-livelock: per reason, MOSS actions that RE-RUN / RE-VERIFY the same
# artifact (and can therefore be picked forever). They are removed from the choice
# set so MOSS must take a converging action instead. The human can still request a
# re-run via the escalate_to_user -> AwaitingUserDecision path.
_NO_AUTO_RERUN: dict[str, frozenset] = {
    "result_suspicious": frozenset({"strict_rerun", "add_holdout"}),
}


def decide(
    reason: str,
    counter_snapshot: dict | None,
    guard: ActionGuardLike,
    llm_client: LLMClientLike,
    *,
    strict: bool = False,
    research_memory: dict | None = None,
    diagnostic: dict | None = None,
    method_context: dict | None = None,
    execution_context: dict | None = None,
) -> dict:
    """Decide the Sync-Intervention action for ``reason`` (pure, no
    transport). 为 ``reason`` 决定同步介入 action（纯，无传输层）。

    Args:
        reason: the stuck pattern MOSS is intervening on (e.g.
            ``"direction_thrashing"``).
        counter_snapshot: the raw counter state that triggered the
            intervention, or ``None``. Diagnostic context only — never
            trusted for anything beyond that.
        guard: an injected ``MossActionGuard`` — the sole source of truth for
            legal actions and the conservative fallback action.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        strict: if ``True``, raise ``MossQualityExhausted`` on retry
            exhaustion instead of degrading to the fallback action. Intended
            for tests / callers that want hard-fail semantics.

    Returns:
        A schema-valid ``MossDecision`` dict:
        ``{"reason", "action", "prompt_patch", "rationale", "degraded"}``.
        ``action`` is guaranteed legal for ``reason`` (either the LLM's own
        legal pick, or ``guard.fallback(reason)`` on degrade).

    Raises:
        MossLLMError (transient): the LLM client itself raised. Not
            internally retried.
        MossQualityExhausted (permanent): only when ``strict=True`` and
            retries were exhausted without a legal, leak-free decision.
    """
    # E13 anti-livelock: for some reasons an action re-runs / re-verifies the SAME
    # artifact and can be picked over and over. In a real run MOSS kept choosing
    # strict_rerun (-> RunningExperiment) / add_holdout (-> VerifyingBestResult) on a
    # result_suspicious result, churning EvaluatingExperiment<->MossIntervening for
    # ~1h. We drop those looping actions from the choices so MOSS must instead take a
    # CONVERGING one (discard_result or escalate_to_user -> the human/conservative
    # policy adjudicates). Non-looping actions (incl. discard) are untouched.
    blocked = _NO_AUTO_RERUN.get(reason, frozenset())
    legal_actions = [a for a in guard.legal_actions(reason) if a not in blocked]
    if reason == "experiment_design_thrashing":
        # Plan-review failures repair the existing component contract first. Core
        # falsification still has the measured method/ablation failure routes.
        blocked_plan_action = ("back_to_hypothesis" if (method_context or {}).get("components")
                               else "revise_component")
        legal_actions = [a for a in legal_actions if a != blocked_plan_action]
    preparation = (diagnostic or {}).get("record", diagnostic or {})
    invalid_component_plan = (
        reason == "component_ablation_failed"
        and preparation.get("underlying_reason_code") == "component_plan_invalid")
    if invalid_component_plan:
        legal_actions = [a for a in legal_actions if a != "revise_method"]
    result = (diagnostic or {}).get("result") or {}
    component = result.get("component_execution") or {}
    host_execution_blocked = (
        reason == "implementation_thrashing"
        # A stale or mismatched attempt reference is an evidence-binding failure,
        # not proof that the current candidate failed to launch.
        and result.get("diagnosis") != "attempt_accounting_violation"
        and component.get("status") in {"execution_failed", "infrastructure_exhausted"}
        and component.get("execution_succeeded") is False
        and component.get("infrastructure_attempts", 0) > 0
        and result.get("component_execution_started") is not True
        and not result.get("component_attempt_id") and not result.get("component_failed_attempt_id"))
    if host_execution_blocked:
        legal_actions = [action for action in legal_actions if action in {"shrink_patch", "escalate_to_user"}]
    review = result.get("method_implementation_review") or {}
    unverified_implementation = (
        reason == "implementation_thrashing"
        and (review.get("verdict") == "insufficient_evidence"
             or (review.get("task_contract_review") or {}).get("verdict")
                 == "insufficient_evidence"))
    if unverified_implementation:
        legal_actions = [action for action in legal_actions if action != "discard_hypothesis"]
    if not legal_actions:
        raise ValueError(f"no legal actions for reason {reason!r}")

    # If a writer/designer agent is responsible for this fixable problem, MOSS closes the
    # loop by feeding IT specific corrective guidance (a prompt_patch) rather than only
    # reporting -- it never edits the artifact itself (writer/ruler isolation).
    responsible = _RESPONSIBLE_AGENT.get(reason)
    if host_execution_blocked:
        responsible = None

    messages = [
        {"role": "system", "content": _load_sync_intervention_system_prompt()},
        {
            "role": "user",
            "content": _build_decide_user_message(
                reason, counter_snapshot, legal_actions, responsible, research_memory,
                diagnostic, method_context, execution_context
            ),
        },
    ]

    last_error = "no attempts made"
    if invalid_component_plan:
        messages[1]["content"] += (
            "\nThe host rejected component preparation before a new experiment ran. "
            "Repair the ComponentPlan or dependency join; this is not an ablation that "
            "refutes the core Method. Preserve the frozen baseline and qualified checkpoints.")
    if unverified_implementation:
        messages[1]["content"] += (
            "\nHost review has not established implementation or task-contract compliance. "
            "Missing restore, measurement, or correctness evidence does not falsify the Method. "
            "Repair and bind that evidence, or locally revise an evidenced component-contract "
            "conflict; preserve qualified checkpoints. Do not discard the core hypothesis "
            "on this unverified observation.")
    if host_execution_blocked:
        messages[1]["content"] += (
            "\nHost evidence shows no candidate execution began. This is not an algorithmic failure. "
            "Do not patch candidate code or revise/discard its Method to repair host launch/transport. "
            "Use a bounded execution retry only after checking existing receipts and the cause; "
            "otherwise escalate for host configuration repair. Return prompt_patch=null.")
    elif (reason == "implementation_thrashing"
          and result.get("diagnosis") == "attempt_accounting_violation"):
        messages[1]["content"] += (
            "\nAn attempt-accounting violation may cite a receipt from an older repository. "
            "Do not infer a current host launch failure from that stale receipt. "
            "Keep the current result invalid; separately assess any prior admissible "
            "measurements and whether the current component contract needs revision.")
    last_error_code = "no_attempts"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise MossLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

        content = result.content or ""

        try:
            obj = json.loads(extract_json(content))
        except (json.JSONDecodeError, TypeError):
            last_error_code = "invalid_json"
            last_error = "response was not valid JSON"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        if not isinstance(obj, dict):
            last_error_code = "non_object_json"
            last_error = "response JSON is not an object"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        action = obj.get("action")
        scope = {}
        plan_diagnosis = None
        if (method_context or {}).get("method_ref"):
            scope = {"method_ref": method_context["method_ref"],
                     "diagnostic_ref": (diagnostic or {}).get("scope", {}).get("ref")}
            if any(obj.get(key) != value for key, value in scope.items()):
                last_error_code = "stale_evidence_scope"
                last_error = ("Decision refers to a different Method or diagnostic. "
                              "Diagnose the current contract using exactly these refs: "
                              + json.dumps(scope))
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue
            nodes = method_context.get("components") or []
            if nodes and action in {"change_plan", "revise_component", "discard_hypothesis"}:
                from autoresearch.experiments.repo_improve.component_plan import _component_id
                plan_diagnosis = obj.get("plan_diagnosis")
                node = next((item for item in nodes if isinstance(item, dict)
                             and isinstance(plan_diagnosis, dict)
                             and plan_diagnosis.get("component_id") in {
                                 item.get("component_id") or item.get("name"), item.get("name"),
                                 _component_id(item.get("component_id") or item.get("name"), 0)
                                 if item.get("component_id") or item.get("name") else None}), None)
                quote = (plan_diagnosis or {}).get("plan_quote") if isinstance(plan_diagnosis, dict) else None
                if (not node or not isinstance(quote, str) or not quote.strip()
                        or not any(quote in str(node.get(field) or "") for field in (
                            "mechanism", "implementation_check", "ablation_test",
                            "functional_check", "downstream_contract"))):
                    last_error_code = "stale_component_contract"
                    last_error = ("Plan diagnosis must identify a current component and quote its "
                                  "current mechanism or check verbatim. An old contract is not current evidence. "
                                  "Current contract fields: " + json.dumps([
                                      {"component_id": item.get("component_id") or item.get("name"),
                                       **{field: item[field] for field in (
                                           "mechanism", "implementation_check", "ablation_test",
                                           "functional_check", "downstream_contract") if item.get(field)}}
                                      for item in ([node] if node else nodes) if isinstance(item, dict)
                                  ], ensure_ascii=False))
                    messages.append({"role": "assistant", "content": content})
                    messages.append(_retry_feedback_message(last_error))
                    continue
        # accept only actions in the (livelock-filtered) legal set: matrix-legal AND
        # not a dropped re-run/re-verify looper.
        if action not in legal_actions:
            last_error_code = "illegal_action"
            last_error = (
                f"action {action!r} is not legal for reason {reason!r}; "
                f"must be one of {legal_actions}"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        prompt_patch = obj.get("prompt_patch")
        if host_execution_blocked and prompt_patch is not None:
            last_error_code = "wrong_repair_layer"
            last_error = "Host launch failure cannot be repaired by a candidate-agent prompt patch. Return prompt_patch=null."
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue
        if prompt_patch is not None:
            if (
                not isinstance(prompt_patch, dict)
                or "agent" not in prompt_patch
                or "content" not in prompt_patch
            ):
                last_error_code = "invalid_patch_shape"
                last_error = (
                    "prompt_patch must be a JSON object with 'agent' and "
                    "'content', or null"
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue

            if prompt_patch.get("agent") not in PATCHABLE_AGENTS:
                last_error_code = "invalid_patch_agent"
                last_error = (
                    f"prompt_patch.agent {prompt_patch.get('agent')!r} is not a "
                    f"patchable agent; must be one of {sorted(PATCHABLE_AGENTS)}"
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue

            if _patch_content_leaks(str(prompt_patch.get("content", ""))):
                last_error_code = "patch_scope_violation"
                last_error = (
                    "prompt_patch.content leaks session/counter/threshold-specific "
                    "orchestration detail (retry counters/thresholds, round numbers, "
                    "MOSS/thrashing). Remove that detail, not the task's own acceptance "
                    "criteria or measured failure evidence. Do not weaken the task gate."
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(_retry_feedback_message(last_error))
                continue

        rationale = obj.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""

        decision = {
            "reason": reason,
            "action": action,
            "prompt_patch": prompt_patch,
            "rationale": rationale,
            "degraded": False,
            **scope,
            **({"plan_diagnosis": plan_diagnosis} if plan_diagnosis else {}),
        }

        errors = _validate_moss_decision(decision)
        if errors:
            last_error_code = "schema_validation_failed"
            last_error = f"assembled MossDecision failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return decision

    # Retries exhausted without a legal, leak-free decision.
    if strict:
        raise MossQualityExhausted(
            f"MOSS decide({reason!r}) exhausted {MAX_RETRIES} retries: {last_error}"
        )

    fallback_decision = {
        "reason": reason,
        "action": guard.fallback(reason),
        "prompt_patch": None,
        "rationale": ("MOSS degraded to conservative fallback after exhausting retries; "
                      f"last_failure_category={last_error_code}"),
        "degraded": True,
        **({"method_ref": method_context["method_ref"],
            "diagnostic_ref": (diagnostic or {}).get("scope", {}).get("ref")}
           if (method_context or {}).get("method_ref") else {}),
    }
    errors = _validate_moss_decision(fallback_decision)
    assert not errors, f"fallback MossDecision failed schema validation: {errors}"
    return fallback_decision


__all__ = [
    "MossError",
    "MossLLMError",
    "MossQualityExhausted",
    "decide",
    "MAX_RETRIES",
    "PATCHABLE_AGENTS",
]
