"""
CriticAgent core: the pure direction + hypothesis critique logic.
CriticAgent 核心：与传输层无关的纯"方向 + 假设批判"逻辑。

Given a ProblemModel + ResearchSpec, ``critique_direction`` calls an injected
LLM client to produce a ``CritiqueReport`` (target="direction") judging
whether the direction is sound. Given a HypothesisBank + ResearchSpec,
``critique_hypotheses`` calls an injected LLM client to produce a
``CritiqueReport`` (target="hypotheses") judging each hypothesis. Both
validate against the shared ``critique_report.schema.yaml`` data-contract —
with NO EventBus / Redis / Store access. The handler (a later milestone) is
responsible for resolving refs, persisting the report, and publishing events.

给定 ProblemModel + ResearchSpec，``critique_direction`` 调用注入的 LLM 客户端
产出一份 CritiqueReport（target="direction"），判断方向是否站得住脚。给定
HypothesisBank + ResearchSpec，``critique_hypotheses`` 调用注入的 LLM 客户端
产出一份 CritiqueReport（target="hypotheses"），逐条评判假设。两者都对照共享
的 ``critique_report.schema.yaml`` data-contract 校验 —— 不含任何
EventBus/Redis/Store 访问。handler（后续里程碑）负责解析 ref、持久化报告、
发布事件。

Anti-hallucination grounding policy / 抗幻觉溯源策略
----------------------------------------------------
``critique_hypotheses``'s ``passing_hypothesis_ids`` MUST cite only ids that
actually appear in the given ``hypothesis_bank``. This is enforced here, at
parse time, as an EXPLICIT check independent of the system prompt (the
prompt asks nicely; this module does not trust that it was obeyed): after
parsing the LLM's JSON, every ``passing_hypothesis_ids`` entry must appear in
``{h["id"] for h in hypothesis_bank["hypotheses"]}``. An id that fails this
check is treated exactly like malformed JSON — the LLM is given corrective
feedback and retried (bounded), and after retries are exhausted a
``CriticParseError`` is raised. Mirrors
``deep_research_agent/core.py``'s ``key_papers`` grounding check.

``critique_hypotheses`` also enforces verdict/ids CONSISTENCY:
``verdict == "some_pass"`` iff ``passing_hypothesis_ids`` is non-empty, and
``verdict == "all_rejected"`` iff it is empty. An inconsistent report is
treated exactly like malformed JSON — retried (bounded), then a typed error.
"""

from __future__ import annotations

import json
import posixpath
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


class CriticError(Exception):
    """Base for CriticAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "critic_failed"
    #: transient | permanent
    classification: str = "transient"


class CriticParseError(CriticError):
    """LLM never returned a well-formed, schema-valid, (for hypotheses)
    grounded-and-consistent CritiqueReport after retries. 重试用尽后 LLM
    仍未返回合法、schema 校验通过（对 hypotheses 而言还需溯源且一致）的
    CritiqueReport。"""

    code = "critic_parse_error"
    classification = "permanent"


class CriticLLMError(CriticError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "critic_llm_error"
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
# Static resources: system prompts + report schema validator
# 静态资源：system prompt + report schema 校验器
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_direction_system_prompt() -> str:
    return (_PROMPTS_DIR / "critique_direction_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_hypotheses_system_prompt() -> str:
    return (_PROMPTS_DIR / "critique_hypotheses_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_experiment_plan_system_prompt() -> str:
    return (_PROMPTS_DIR / "critique_experiment_plan_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "critique_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_report(report: dict) -> list[str]:
    return [e.message for e in _report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_direction_user_message(problem_model: dict, research_spec: dict) -> str:
    return "\n".join(
        [
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "ProblemModel (JSON):",
            json.dumps(problem_model, ensure_ascii=False, indent=2),
            "",
            "Critique this research direction and respond with a single JSON "
            "object matching the CritiqueReport shape described in the "
            "system prompt (target=\"direction\").",
        ]
    )


def _build_hypotheses_user_message(
    hypothesis_bank: dict, research_spec: dict, ranking_report: dict | None = None
) -> str:
    parts = [
        "ResearchSpec (JSON):",
        json.dumps(research_spec, ensure_ascii=False, indent=2),
        "",
        "HypothesisBank (JSON):",
        json.dumps(hypothesis_bank, ensure_ascii=False, indent=2),
        "",
    ]
    if ranking_report is not None:
        parts += [
            "RankingReport (JSON) -- RankingJudgeAgent's pairwise ranking of "
            "these hypotheses; use it to inform (not replace) your own "
            "critique, e.g. weighing whether top-ranked hypotheses truly "
            "deserve to pass and whether the ranking's diversity_status "
            "signals near-duplicate candidates:",
            json.dumps(ranking_report, ensure_ascii=False, indent=2),
            "",
        ]
    parts.append(
        "Critique each hypothesis in the bank and respond with a single "
        "JSON object matching the CritiqueReport shape described in the "
        "system prompt (target=\"hypotheses\"). `passing_hypothesis_ids` "
        "must only contain `id`s that appear in the HypothesisBank above."
    )
    return "\n".join(parts)


def _build_experiment_plan_user_message(
    experiment_plan: dict, verifier_spec: dict, benchmark_spec: dict | None = None,
    baseline_result: dict | None = None,
    method: dict | None = None,
    research_memory: dict | None = None,
    method_feedback: dict | None = None,
) -> str:
    editable_region = verifier_spec.get("editable_region", [])
    forbidden_files = verifier_spec.get("forbidden_files", [])
    scope_explanation = (
        "empty editable_region means no predeclared file boundary; the whole repository is "
        "provisionally editable, except forbidden_files and any narrower limits in the "
        "authoritative repository task contract supplied with the baseline"
        if not editable_region else
        "editable_region bounds changes[].file; authoritative repository target_files may narrow it"
    )
    memory = research_memory or {}
    from autoresearch.experiments.repo_improve.component_plan import method_fingerprint
    progress = {
        "current_method_fingerprint": method_fingerprint(method) if method else None,
        "best_metric": memory.get("best_metric"),
        "best_result_ref": memory.get("best_result_ref"),
        "component_history": [
            {key: (value[:1000] if key == "changes_summary" and isinstance(value, str) else value)
             for key, value in item.items() if key in {
                 "sequence", "component_id", "mode", "method_fingerprint", "status", "metric",
                 "contract_passed", "mir_verdict", "verification_verdict", "became_best",
                 "checkpoint", "selected_checkpoint", "run_ref", "changes_summary", "failure_reason",
                 "parent_checkpoint", "parent_checkpoints", "acceptance_result",
             }}
            for item in memory.get("component_history", []) if isinstance(item, dict)
        ],
    }
    return "\n".join(
        [
            "BenchmarkSpec (JSON; authoritative metric, comparison semantics and budget):",
            json.dumps(benchmark_spec or {}, ensure_ascii=False, indent=2),
            "",
            "VerifierSpec (JSON):",
            json.dumps(verifier_spec, ensure_ascii=False, indent=2),
            "",
            "Frozen BaselineResult (JSON; authoritative observed runtime evidence, if available):",
            json.dumps(baseline_result or {}, ensure_ascii=False, indent=2),
            "",
            "ExperimentPlan (JSON):",
            json.dumps(experiment_plan, ensure_ascii=False, indent=2),
            "",
            "Confirmed Method (JSON; frozen component designs, dependencies and acceptance contracts):",
            json.dumps(method or {}, ensure_ascii=False, indent=2),
            "",
            "Prior Method feedback (JSON; recommendations, not an applied Method revision):",
            json.dumps(method_feedback or {}, ensure_ascii=False, indent=2),
            "",
            "Research progress (JSON; stored observations and qualification, not a new task contract):",
            json.dumps(progress, ensure_ascii=False, indent=2),
            "",
            f"editable_region ({scope_explanation}):",
            json.dumps(editable_region, ensure_ascii=False, indent=2),
            "",
            "forbidden_files (changes[].file must NEVER target any of these):",
            json.dumps(forbidden_files, ensure_ascii=False, indent=2),
            "",
            "Review this experiment plan and respond with a single JSON "
            "object matching the CritiqueReport shape described in the "
            "system prompt (target=\"experiment_plan\").",
        ]
    )


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object matching the "
            "CritiqueReport shape described in the system prompt."
        ),
    }


# ---------------------------------------------------------------------------
# Grounding + consistency checks (critique_hypotheses only)
# 溯源 + 一致性校验（仅 critique_hypotheses）
# ---------------------------------------------------------------------------


def _find_ungrounded_passing_ids(passing_ids: list, bank_ids: set[str]) -> list[str]:
    """Return any passing_hypothesis_ids entries not present in the bank's
    ids. Empty list = fully grounded. 返回未出现在 bank id 集合中的
    passing_hypothesis_ids 条目。空列表 = 完全溯源。"""
    return [pid for pid in passing_ids if pid not in bank_ids]


def _verdict_ids_consistent(verdict: str, passing_ids: list) -> bool:
    if verdict == "some_pass":
        return len(passing_ids) > 0
    if verdict == "all_rejected":
        return len(passing_ids) == 0
    return True


# ---------------------------------------------------------------------------
# editable_region gate (critique_experiment_plan only)
# editable_region 硬门（仅 critique_experiment_plan）
# ---------------------------------------------------------------------------


def _find_editable_region_violations(
    changes: list, editable_region: list, forbidden_files: list
) -> list[str]:
    """``changes[].file`` values that are NOT in ``editable_region`` or ARE
    in ``forbidden_files``. Empty = every change is in-scope (OK). Mirrors
    ``experiment_planner_agent/core.py``'s ``_find_scope_violations``.
    不在 editable_region 中或位于 forbidden_files 中的 changes[].file 值。
    空列表 = 所有改动都在范围内（合规）。镜像
    experiment_planner_agent/core.py 的 _find_scope_violations。"""
    editable = set(editable_region)
    forbidden = set(forbidden_files)
    violations = []
    for change in changes:
        file_path = change.get("file") if isinstance(change, dict) else None
        # meeting #1: an EMPTY editable_region means no boundary was pre-declared -> PERMISSIVE
        # (whole repo editable); only forbidden_files still blocks. Mirrors the planner's fix, so a
        # missing pre-baseline verifier can't make the critic reject-loop every plan.
        if (editable and file_path not in editable) or file_path in forbidden:
            violations.append(file_path)
    return violations


def _repository_target_paths(baseline_result: dict | None) -> list[str]:
    """Resolve explicit task config target_files to paths in the outer repository."""
    if not isinstance(baseline_result, dict) or baseline_result.get("source_kind") != "live_repo":
        return []
    contract = baseline_result.get("repository_contract") or {}
    paths: set[str] = set()
    for source in contract.get("sources") or []:
        if not isinstance(source, dict) or source.get("truncated") is True:
            continue
        source_path = source.get("path")
        if not isinstance(source_path, str) or posixpath.basename(source_path) != "config.json":
            continue
        try:
            config = json.loads(source.get("content") or "")
        except (TypeError, ValueError):
            continue
        targets = config.get("target_files") if isinstance(config, dict) else None
        if not isinstance(targets, list) or not targets:
            continue
        roots = [posixpath.dirname(source_path)]
        if isinstance(config.get("repo_dir"), str):
            roots.append(config["repo_dir"])
        for target in targets:
            if not isinstance(target, str) or not target.strip():
                continue
            normalized = posixpath.normpath(target)
            if normalized == ".." or normalized.startswith("../"):
                continue
            if posixpath.isabs(normalized):
                paths.add(normalized)
            else:
                paths.update(posixpath.normpath(posixpath.join(root, normalized))
                             for root in roots)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Shared LLM-call-then-parse loop / 共享的"调用 LLM 再解析"循环
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def _run_llm_loop(
    messages: list[dict],
    llm_client: LLMClientLike,
    *,
    expected_target: str,
    check_extra: Callable[[dict], str | None],
) -> dict:
    """Call the LLM, parse/validate the JSON response as a CritiqueReport
    with ``target == expected_target``, and run ``check_extra`` (returning an
    error-message string on failure, ``None`` on success) for any additional
    (grounding/consistency) checks. Retries (bounded) on any failure, then
    raises ``CriticParseError``.
    调用 LLM，解析/校验 JSON 响应为 target 为 expected_target 的
    CritiqueReport，并跑 check_extra（失败返回错误信息字符串，成功返回
    None）做额外（溯源/一致性）检查。任何失败均有限重试，用尽后抛
    CriticParseError。"""
    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise CriticLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

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

        errors = _validate_report(obj)
        if errors:
            last_error = f"report failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        if obj.get("target") != expected_target:
            last_error = (
                f"report 'target' must be {expected_target!r}, got {obj.get('target')!r}"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        extra_error = check_extra(obj)
        if extra_error is not None:
            last_error = extra_error
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return obj

    raise CriticParseError(
        f"LLM did not produce a valid CritiqueReport after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# critique_direction / 批判方向
# ---------------------------------------------------------------------------


def critique_direction(
    problem_model: dict,
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> dict:
    """Run one direction-critique turn (pure, no transport).
    跑一轮方向批判（纯，无传输层）。

    Args:
        problem_model: the (dict-shaped) ProblemModel being critiqued.
        research_spec: the frozen ResearchSpec.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``CritiqueReport`` dict with ``target == "direction"``.
        Callers read ``report["verdict"]`` (``pass`` / ``fail``).

    Raises:
        CriticParseError (permanent): malformed/unparseable LLM JSON, or a
            schema-invalid report (including an invalid ``verdict`` enum or
            wrong ``target``), after bounded retries.
        CriticLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_direction_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_direction_user_message(problem_model, research_spec),
        },
    ]

    def check_extra(obj: dict) -> str | None:
        if obj.get("verdict") not in ("pass", "fail"):
            return f"target=direction verdict must be 'pass' or 'fail', got {obj.get('verdict')!r}"
        return None

    return _run_llm_loop(messages, llm_client, expected_target="direction", check_extra=check_extra)


# ---------------------------------------------------------------------------
# critique_hypotheses / 批判假设
# ---------------------------------------------------------------------------


def critique_hypotheses(
    hypothesis_bank: dict,
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
    ranking_report: dict | None = None,
) -> dict:
    """Run one hypotheses-critique turn (pure, no transport).
    跑一轮假设批判（纯，无传输层）。

    Args:
        hypothesis_bank: the (dict-shaped) HypothesisBank being critiqued,
            i.e. ``{"hypotheses": [{"id": ..., ...}, ...]}``.
        research_spec: the frozen ResearchSpec.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).
        ranking_report: optional RankingJudgeAgent RankingReport (dict) for
            this same HypothesisBank; when given, it is included in the user
            message so the critique is rank-informed. ``None`` (the default)
            reproduces the pre-RankingJudgeAgent behavior exactly.

    Returns:
        A schema-valid ``CritiqueReport`` dict with ``target == "hypotheses"``,
        ``passing_hypothesis_ids`` a subset of the bank's hypothesis ids, and
        verdict/ids consistency (``some_pass`` iff non-empty ids,
        ``all_rejected`` iff empty ids).

    Raises:
        CriticParseError (permanent): malformed/unparseable LLM JSON, a
            schema-invalid report (including an invalid ``verdict`` enum or
            wrong ``target``), a ``passing_hypothesis_ids`` entry not present
            in the bank (grounding), or a verdict/ids inconsistency, after
            bounded retries.
        CriticLLMError (transient): the LLM client itself raised.
    """
    bank_ids = {h["id"] for h in hypothesis_bank.get("hypotheses", [])}

    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_hypotheses_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_hypotheses_user_message(hypothesis_bank, research_spec, ranking_report),
        },
    ]

    def check_extra(obj: dict) -> str | None:
        verdict = obj.get("verdict")
        if verdict not in ("some_pass", "all_rejected"):
            return (
                "target=hypotheses verdict must be 'some_pass' or "
                f"'all_rejected', got {verdict!r}"
            )

        passing_ids = obj.get("passing_hypothesis_ids", [])

        # EXPLICIT anti-hallucination grounding check — the enforceable
        # guard; do NOT trust that the prompt's instruction was followed.
        ungrounded = _find_ungrounded_passing_ids(passing_ids, bank_ids)
        if ungrounded:
            return (
                "passing_hypothesis_ids cited ids not present in the "
                f"HypothesisBank: {ungrounded}. Only cite ids from the "
                "HypothesisBank; do not invent ids."
            )

        if not _verdict_ids_consistent(verdict, passing_ids):
            return (
                f"verdict {verdict!r} is inconsistent with "
                f"passing_hypothesis_ids {passing_ids!r}: 'some_pass' requires "
                "a non-empty list, 'all_rejected' requires an empty list."
            )

        return None

    return _run_llm_loop(
        messages, llm_client, expected_target="hypotheses", check_extra=check_extra
    )


# ---------------------------------------------------------------------------
# critique_experiment_plan / 批判实验方案
# ---------------------------------------------------------------------------


def critique_experiment_plan(
    experiment_plan: dict,
    verifier_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
    benchmark_spec: dict | None = None,
    baseline_result: dict | None = None,
    method: dict | None = None,
    research_memory: dict | None = None,
    method_feedback: dict | None = None,
) -> dict:
    """Run one experiment-plan-review turn (pure, no transport).
    跑一轮实验方案评审（纯，无传输层）。

    The LLM reviews the plan for fairness / runnability / evaluability /
    anti-cheat / budget. Independent of whatever the LLM says, the
    ``editable_region`` GATE below is enforced HERE IN CODE: if any
    ``experiment_plan["changes"][i]["file"]`` is not a member of
    ``verifier_spec["editable_region"]`` or is in
    ``verifier_spec["forbidden_files"]``, the returned ``verdict`` is forced
    to ``"fail"`` (overriding the LLM's verdict) and ``issues`` is extended
    with a message naming the offending file(s). This makes CriticAgent a
    hard enforcement point for the editable_region invariant, independent of
    the LLM or the upstream planner. Mirrors
    ``evaluator_agent/core.py``'s ``prepare_verifier`` verifier-first guards
    and ``experiment_planner_agent/core.py``'s own scope guard.

    Args:
        experiment_plan: the (dict-shaped) ExperimentPlan being reviewed.
        verifier_spec: the frozen VerifierSpec (editable_region, forbidden_files).
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``CritiqueReport`` dict with ``target ==
        "experiment_plan"``. Callers read ``report["verdict"]`` (``pass`` /
        ``fail``); ``fail`` is guaranteed whenever any ``changes[].file`` is
    out of the editable_region scope. An explicit target_files list in a captured
    live repository task config is also enforced when present.

    Raises:
        CriticParseError (permanent): malformed/unparseable LLM JSON, or a
            schema-invalid report (including an invalid ``verdict`` enum or
            wrong ``target``), after bounded retries.
        CriticLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_experiment_plan_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_experiment_plan_user_message(
                experiment_plan, verifier_spec, benchmark_spec, baseline_result, method,
                research_memory, method_feedback),
        },
    ]

    def check_extra(obj: dict) -> str | None:
        if obj.get("verdict") not in ("pass", "fail"):
            return (
                f"target=experiment_plan verdict must be 'pass' or 'fail', "
                f"got {obj.get('verdict')!r}"
            )
        for issue in obj.get("issues", []):
            conclusion = re.search(
                r"(?:^|[.!?]\s+)\s*(?:therefore[,:]?\s*)?(?:the\s+)?verdict\s*"
                r"(?:is\s+|should\s+be\s+|:\s*)[\"']?(pass|fail)[\"']?\s*[.!]?\s*$",
                issue, re.IGNORECASE)
            if conclusion and conclusion.group(1).lower() != obj["verdict"]:
                return ("The verdict field contradicts an explicit verdict statement in issues. "
                        "Return one consistent verdict; list only concrete blocking problems "
                        "for fail, or pass with empty issues when none exists.")
        return None

    report = _run_llm_loop(
        messages, llm_client, expected_target="experiment_plan", check_extra=check_extra
    )

    # GATE (parse-time, in code — CriticAgent is the enforcement point, NOT
    # trusting that the LLM or the upstream planner honored editable_region).
    violations = _find_editable_region_violations(
        experiment_plan.get("changes", []),
        verifier_spec.get("editable_region", []),
        verifier_spec.get("forbidden_files", []),
    )
    if violations:
        report = dict(report)
        report["verdict"] = "fail"
        report["issues"] = [
            *report.get("issues", []),
            "editable_region gate: changes[].file outside verifier_spec.editable_region "
            f"or within verifier_spec.forbidden_files: {violations}",
        ]

    target_paths = _repository_target_paths(baseline_result)
    if target_paths:
        outside_targets = [
            change.get("file") for change in experiment_plan.get("changes", [])
            if isinstance(change, dict)
            and posixpath.normpath(str(change.get("file") or "")) not in target_paths
        ]
        if outside_targets:
            report = dict(report)
            report["verdict"] = "fail"
            report["issues"] = [
                *report.get("issues", []),
                f"repository target_files gate: changes[].file outside task config: {outside_targets}",
            ]

    return report


__all__ = [
    "CriticError",
    "CriticParseError",
    "CriticLLMError",
    "critique_direction",
    "critique_hypotheses",
    "critique_experiment_plan",
    "MAX_RETRIES",
]
