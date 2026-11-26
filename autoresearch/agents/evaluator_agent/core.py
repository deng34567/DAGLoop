"""
EvaluatorAgent core: the pure benchmark-selection + verifier-preparation
logic. EvaluatorAgent 核心：与传输层无关的纯"基准选择 + 验证器准备"逻辑。

Given a ResearchSpec + LiteratureBrief, ``select_benchmark`` calls an
injected LLM client to produce a ``BenchmarkSpec`` (a reproducible benchmark
+ baseline + metric + run command + forbidden files). Given a BenchmarkSpec
+ ResearchSpec, ``prepare_verifier`` calls an injected LLM client to freeze a
``VerifierSpec`` — the verifier-first artifact (evaluator hash, metric, data
split, run budget, seed policy, editable_region whitelist, forbidden_files
blacklist, result_parse_schema) BEFORE any candidate solution exists. Both
validate against the shared data-contracts — with NO EventBus / Redis /
Store access. The handler (a later milestone) is responsible for resolving
refs, persisting the artifacts, and publishing events.

给定 ResearchSpec + LiteratureBrief，``select_benchmark`` 调用注入的 LLM
客户端产出一份 BenchmarkSpec（可复现的基准 + baseline + 指标 + 运行命令 +
禁止修改文件）。给定 BenchmarkSpec + ResearchSpec，``prepare_verifier``
调用注入的 LLM 客户端冻结一份 VerifierSpec —— 即"验证器先行"产物（evaluator
哈希、指标、数据切分、运行预算、随机种子策略、editable_region 白名单、
forbidden_files 黑名单、result_parse_schema），在任何候选方案存在之前就
冻结。两者都对照各自的 data-contract 校验 —— 不含任何 EventBus/Redis/Store
访问。handler（后续里程碑）负责解析 ref、持久化产物、发布事件。

Verifier-first guards / 验证器先行护栏
---------------------------------------
``prepare_verifier``'s returned ``verifier_spec`` is checked, at parse time,
against TWO invariants that are enforced HERE IN CODE — independent of
whatever the system prompt asked for (the prompt asks nicely; this module
does not trust that it was obeyed):

1. **Disjointness**: ``set(verifier_spec["editable_region"])`` and
   ``set(verifier_spec["forbidden_files"])`` MUST be disjoint. A path that is
   simultaneously "the candidate may edit this" and "the candidate must not
   touch this" is a contradiction that would let a candidate solution smuggle
   changes into a nominally-forbidden path.
2. **Frozen-core invariant**: no path in ``editable_region`` may reference
   the evaluator/harness/verifier/metric machinery itself — checked via a
   small, documented, case-insensitive substring token list:
   ``eval``, ``evaluat``, ``verif``, ``harness``, ``grade``, ``metric``. If a
   candidate could edit the evaluator's own code, it could game the metric
   instead of improving the underlying solution — the evaluator MUST stay
   frozen.

Either violation is treated exactly like malformed JSON — the LLM is given
corrective feedback and retried (bounded), and after retries are exhausted an
``EvaluatorParseError`` is raised. Mirrors ``critic_agent/core.py``'s
``critique_hypotheses`` grounding check.

违反上述任一护栏都与格式错误的 JSON 同等处理 —— 给 LLM 纠错反馈并有限重试，
用尽后抛出 ``EvaluatorParseError``。镜像 ``critic_agent/core.py`` 中
``critique_hypotheses`` 的溯源检查模式。
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class EvaluatorError(Exception):
    """Base for EvaluatorAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "evaluator_failed"
    #: transient | permanent
    classification: str = "transient"


class EvaluatorParseError(EvaluatorError):
    """LLM never returned a well-formed, schema-valid, (for
    prepare_verifier) guard-passing response after retries. 重试用尽后 LLM
    仍未返回合法、schema 校验通过（对 prepare_verifier 而言还需通过护栏）的
    响应。"""

    code = "evaluator_parse_error"
    classification = "permanent"


class EvaluatorLLMError(EvaluatorError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "evaluator_llm_error"
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
def _load_select_benchmark_system_prompt() -> str:
    return (_PROMPTS_DIR / "select_benchmark_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_prepare_verifier_system_prompt() -> str:
    return (_PROMPTS_DIR / "prepare_verifier_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_evaluate_experiment_system_prompt() -> str:
    return (_PROMPTS_DIR / "evaluate_experiment_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _benchmark_spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "benchmark_spec.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _verifier_spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "verifier_spec.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _evaluation_report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "evaluation_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _verification_report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "verification_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_benchmark_spec(spec: dict) -> list[str]:
    return [e.message for e in _benchmark_spec_validator().iter_errors(spec)]


def _validate_verifier_spec(spec: dict) -> list[str]:
    return [e.message for e in _verifier_spec_validator().iter_errors(spec)]


def _validate_evaluation_report(report: dict) -> list[str]:
    return [e.message for e in _evaluation_report_validator().iter_errors(report)]


def _validate_verification_report(report: dict) -> list[str]:
    return [e.message for e in _verification_report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _build_select_benchmark_user_message(
    research_spec: dict,
    literature_brief: dict,
    evaluation_semantics: dict | None = None,
) -> str:
    return "\n".join(
        [
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "LiteratureBrief (JSON):",
            json.dumps(literature_brief, ensure_ascii=False, indent=2),
            "",
            "Frozen EvaluationSemantics (JSON; defines the scientific comparison):",
            (
                json.dumps(evaluation_semantics, ensure_ascii=False, indent=2)
                if evaluation_semantics is not None else "legacy session: unavailable"
            ),
            "",
            "Select a reproducible benchmark and respond with a single JSON "
            "object matching the BenchmarkSpec shape described in the "
            "system prompt.",
        ]
    )


def _build_prepare_verifier_user_message(benchmark_spec: dict, research_spec: dict) -> str:
    return "\n".join(
        [
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "BenchmarkSpec (JSON):",
            json.dumps(benchmark_spec, ensure_ascii=False, indent=2),
            "",
            "Freeze a verifier for this benchmark and respond with a single "
            "JSON object matching the shape described in the system prompt: "
            '{"verifier_spec": {...}, "verdict": "ok|incomplete|invalid"}.',
        ]
    )


def _build_evaluate_experiment_user_message(
    run_record: dict,
    verifier_spec: dict,
    benchmark_spec: dict,
    prior_research_memory: dict | None = None,
) -> str:
    return "\n".join(
        [
            "RunRecord (JSON):",
            json.dumps(run_record, ensure_ascii=False, indent=2),
            "",
            "VerifierSpec (JSON):",
            json.dumps(verifier_spec, ensure_ascii=False, indent=2),
            "",
            "BenchmarkSpec (JSON):",
            json.dumps(benchmark_spec, ensure_ascii=False, indent=2),
            "",
            "Prior ResearchMemory (JSON):",
            (
                json.dumps(prior_research_memory, ensure_ascii=False, indent=2)
                if prior_research_memory is not None
                else "none"
            ),
            "",
            "Evaluate this run and respond with a single JSON object matching "
            "the shape described in the system prompt: "
            '{"verdict": ..., "suspicious": ..., "anti_cheat_flags": [...], '
            '"reasoning": ...}.',
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
# Verifier-first guards (prepare_verifier only)
# 验证器先行护栏（仅 prepare_verifier）
# ---------------------------------------------------------------------------

#: Small, documented, case-insensitive substring token list. A path in
#: editable_region containing any of these tokens is rejected: the
#: evaluator/harness/verifier/metric machinery must stay frozen so a
#: candidate solution cannot edit its own grading logic.
#: 小而明确记录的、大小写不敏感的子串 token 列表：editable_region 中的路径
#: 若包含其中任一 token 则拒绝 —— evaluator/harness/verifier/metric 机制必须
#:保持冻结，候选方案不能编辑自己的评分逻辑。
_FROZEN_CORE_TOKENS = ("eval", "evaluat", "verif", "harness", "grade", "metric")


def _find_disjointness_violations(editable_region: list, forbidden_files: list) -> list[str]:
    """Paths present in BOTH editable_region and forbidden_files. Empty =
    disjoint (OK). 同时出现在 editable_region 和 forbidden_files 中的路径。
    空列表 = 不相交（合规）。"""
    return sorted(set(editable_region) & set(forbidden_files))


def _find_frozen_core_violations(editable_region: list) -> list[str]:
    """editable_region paths that reference the evaluator/harness/verifier
    itself (case-insensitive substring match on _FROZEN_CORE_TOKENS). Empty
    = the frozen core is untouched (OK). editable_region 中引用
    evaluator/harness/verifier 本身的路径（对 _FROZEN_CORE_TOKENS 做大小写
    不敏感子串匹配）。空列表 = 冻结核心未被触碰（合规）。"""
    return [
        path
        for path in editable_region
        if any(token in path.lower() for token in _FROZEN_CORE_TOKENS)
    ]


# ---------------------------------------------------------------------------
# Shared LLM-call-then-parse loop / 共享的"调用 LLM 再解析"循环
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def _run_llm_loop(
    messages: list[dict],
    llm_client: LLMClientLike,
    *,
    process: Callable[[dict], tuple[object, str | None]],
    stream: bool | None = None,
) -> object:
    """Call the LLM, parse the JSON response as an object, and run
    ``process`` (returning ``(result, None)`` on success or
    ``(None, error_message)`` on failure) to validate/extract the final
    result. Retries (bounded) on any failure, then raises
    ``EvaluatorParseError``.
    调用 LLM，把 JSON 响应解析为 object，再跑 process（成功返回
    ``(result, None)``，失败返回 ``(None, error_message)``）做校验/提取。
    任何失败均有限重试，用尽后抛 EvaluatorParseError。"""
    last_error = "no attempts made"
    # Retries must NOT grow the context: the old loop appended the full (bad) assistant response +
    # feedback to `messages` every retry, so each subsequent call carried a bigger prompt -> slower
    # and more stall-prone (this compounded verify stalls on BOTH flash and glm, unrelated to the
    # model). Instead, keep the ORIGINAL prompt fixed and re-issue it with a SINGLE short correction
    # appended -- constant size every attempt. Also don't mutate the caller's `messages` list.
    base = list(messages)
    correction: dict | None = None

    for _attempt in range(MAX_RETRIES + 1):
        msgs = base if correction is None else base + [correction]
        try:
            # Only forward stream= when explicitly set, so fake/mock clients (tests) whose chat()
            # lacks a stream kwarg keep working -- default callers pass nothing (client default).
            _ck: dict[str, object] = {"messages": msgs, "response_format": "json_object"}
            if stream is not None:
                _ck["stream"] = stream
            result = llm_client.chat(**_ck)
        except Exception as exc:  # any client exception is infra/transient
            raise EvaluatorLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

        content = result.content or ""

        try:
            obj = json.loads(extract_json(content))
        except (json.JSONDecodeError, TypeError):
            last_error = "response was not valid JSON"
            correction = _retry_feedback_message(last_error)
            continue

        if not isinstance(obj, dict):
            last_error = "response JSON is not an object"
            correction = _retry_feedback_message(last_error)
            continue

        parsed, error = process(obj)
        if error is not None:
            last_error = error
            correction = _retry_feedback_message(last_error)
            continue

        return parsed

    raise EvaluatorParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# select_benchmark / 选择基准
# ---------------------------------------------------------------------------


def select_benchmark(
    research_spec: dict,
    literature_brief: dict,
    llm_client: LLMClientLike,
    *,
    evaluation_semantics: dict | None = None,
    prompt_patch: str | None = None,
) -> dict:
    """Select a reproducible benchmark (pure, no transport).
    选择一个可复现的基准（纯，无传输层）。

    Args:
        research_spec: the frozen ResearchSpec.
        literature_brief: the frozen LiteratureBrief.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``BenchmarkSpec`` dict.

    Raises:
        EvaluatorParseError (permanent): malformed/unparseable LLM JSON, or a
            schema-invalid BenchmarkSpec, after bounded retries.
        EvaluatorLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_select_benchmark_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_select_benchmark_user_message(
                research_spec, literature_brief, evaluation_semantics),
        },
    ]

    def process(obj: dict) -> tuple[dict | None, str | None]:
        candidate = dict(obj)
        candidate.pop("evaluation_semantics", None)
        baseline = candidate.get("baseline")
        if isinstance(baseline, dict) and baseline.get("reported_metric") is None:
            candidate["baseline"] = dict(baseline)
            candidate["baseline"].pop("reported_metric", None)
        errors = _validate_benchmark_spec(candidate)
        if errors:
            return None, f"response failed BenchmarkSpec schema validation: {errors}"
        if evaluation_semantics is not None:
            candidate["evaluation_semantics"] = json.loads(json.dumps(
                evaluation_semantics, ensure_ascii=False))
        return candidate, None

    return _run_llm_loop(messages, llm_client, process=process)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# prepare_verifier / 准备验证器
# ---------------------------------------------------------------------------


def prepare_verifier(
    benchmark_spec: dict,
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> tuple[dict, str]:
    """Freeze a verifier for the given benchmark (pure, no transport).
    为给定基准冻结一份验证器（纯，无传输层）。

    Args:
        benchmark_spec: the frozen BenchmarkSpec.
        research_spec: the frozen ResearchSpec.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A tuple ``(verifier_spec, verdict)`` where ``verifier_spec`` is a
        schema-valid ``VerifierSpec`` dict and ``verdict`` is one of
        ``"ok"``, ``"incomplete"``, ``"invalid"``.

    Raises:
        EvaluatorParseError (permanent): malformed/unparseable LLM JSON, a
            missing/invalid ``verdict``, a schema-invalid ``verifier_spec``,
            or a verifier-first guard violation (editable_region /
            forbidden_files overlap, or an editable_region path touching the
            frozen evaluator/harness/verifier core), after bounded retries.
        EvaluatorLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_prepare_verifier_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_prepare_verifier_user_message(benchmark_spec, research_spec),
        },
    ]

    def process(obj: dict) -> tuple[tuple[dict, str] | None, str | None]:
        verdict = obj.get("verdict")
        if verdict not in ("ok", "incomplete", "invalid"):
            return None, (
                "verdict must be one of 'ok', 'incomplete', 'invalid', "
                f"got {verdict!r}"
            )

        verifier_spec = obj.get("verifier_spec")
        if not isinstance(verifier_spec, dict):
            return None, "verifier_spec must be a JSON object"

        errors = _validate_verifier_spec(verifier_spec)
        if errors:
            return None, f"verifier_spec failed schema validation: {errors}"

        editable_region = verifier_spec.get("editable_region", [])
        forbidden_files = verifier_spec.get("forbidden_files", [])

        # GUARD 1 (verifier-first): editable_region and forbidden_files MUST
        # be disjoint — enforced here, not trusted from the prompt.
        overlap = _find_disjointness_violations(editable_region, forbidden_files)
        if overlap:
            return None, (
                "editable_region and forbidden_files must be disjoint; "
                f"paths present in both: {overlap}"
            )

        # GUARD 2 (verifier-first): the evaluator/harness/verifier itself
        # must stay frozen — no editable_region path may reference it.
        frozen_core_hits = _find_frozen_core_violations(editable_region)
        if frozen_core_hits:
            return None, (
                "editable_region must not include the evaluator/harness/"
                "verifier itself (frozen-core invariant); offending paths: "
                f"{frozen_core_hits}"
            )

        return (verifier_spec, verdict), None

    return _run_llm_loop(messages, llm_client, process=process)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# evaluate_experiment / 评估实验
# ---------------------------------------------------------------------------


def _degeneracy_reason(metric: object, baseline: object) -> str | None:
    """Deterministic degeneracy signals a run's metrics show that reproduction alone cannot
    vouch for -- return a short flag string, or None if the result looks clean.

      - a non-finite metric (NaN/Inf) that slipped through as a number;
      - a method metric EXACTLY equal to the baseline (the experiment produced no
        distinguishing power -- e.g. degenerate instances, or method == baseline code).

    Deliberately narrow / high-precision: it does NOT flag ordinary wins or honest ties by
    value alone (a legitimately-optimal method reaching, say, 0.0 is not flagged) -- only the
    two unambiguous signals above, which route to MOSS for review rather than auto-failing."""
    def _num(x: object) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if _num(metric) and not math.isfinite(float(metric)):  # type: ignore[arg-type]
        return "degenerate_result:non_finite_metric"
    if _num(metric) and _num(baseline) and metric == baseline:
        return "degenerate_result:no_separation_from_baseline"
    return None


def _repo_evaluation_report(environment: str) -> dict | None:
    """If ``environment`` marks a repo_improve run (see RunnerAgent._map_repo_result), TRUST its
    mapped repository-run verdict and return a schema-valid EvaluationReport reflecting it,
    with NO self-written-experiment anti-cheat flags. Returns None for a normal run (fall through
    to the LLM evaluation). FAIL-OPEN: an unparseable repo marker -> None (normal evaluation)."""
    if not isinstance(environment, str) or "repo_improve|" not in environment:
        return None
    fields: dict[str, str] = {}
    for part in environment.split("|"):
        if "=" in part:
            k, _, v = part.partition("=")
            fields[k.strip()] = v.strip()
    flow_verdict = fields.get("verdict")
    if flow_verdict not in ("invalid", "valid_not_best", "new_best"):
        return None

    def _num(s: str | None):
        try:
            return float(s) if s not in (None, "", "None") else None
        except (TypeError, ValueError):
            return None

    return {
        "verdict": flow_verdict,
        "is_improvement": flow_verdict == "new_best",
        "suspicious": flow_verdict == "invalid",
        "anti_cheat_flags": [],
        "metric_value": _num(fields.get("method")),
        "baseline_metric": _num(fields.get("baseline")),
        "reasoning": (
            "repo_improve run: Runner mapped the baseline and method values from the completed "
            "repository-run result. The scientific comparison semantics were frozen before "
            "execution; this evaluation records the measured outcome, and the following "
            "verification step checks the exact metric or failure cause. The 34-state "
            "self-written-experiment protocol checks do not apply to this repository-owned "
            "benchmark path; no additional replication count is claimed here."
        ),
    }


def evaluate_experiment(
    run_record: dict,
    verifier_spec: dict,
    benchmark_spec: dict,
    llm_client: LLMClientLike,
    *,
    prior_research_memory: dict | None = None,
    prompt_patch: str | None = None,
) -> dict:
    """Evaluate a completed run against its verifier/benchmark (pure, no
    transport). 对照 verifier/benchmark 评估一次已完成的运行（纯，无传输层）。

    Args:
        run_record: the frozen RunRecord produced by RunnerAgent.
        verifier_spec: the frozen VerifierSpec.
        benchmark_spec: the frozen BenchmarkSpec (its ``baseline`` may or may
            not carry a ``reported_metric``).
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prior_research_memory: the prior ``ResearchMemory`` (M13), if one
            exists, used ONLY to source the running-best threshold and to
            ground the LLM's ``conflict`` judgment against
            ``verified_findings``. ``None`` when there is no prior memory
            (e.g. the first-ever run, or an M12 caller).
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``EvaluationReport`` dict.

    Deterministic anti-reward-hacking guard / 确定性反作弊护栏
    -----------------------------------------------------------
    Whether the run is an improvement is COMPUTED HERE IN CODE from
    ``run_record["metric_value"]`` vs. a threshold — the LLM's response is
    never trusted for this number, and ``is_improvement``, ``metric_value``,
    ``baseline_metric`` on the returned report always reflect the computed
    values, overwriting anything the LLM claimed. The threshold is the
    RUNNING BEST when one is known (``prior_research_memory["best_metric"]``),
    else ``benchmark_spec["baseline"].get("reported_metric")`` (M12 fallback
    behavior). ``baseline_metric`` on the report always reflects
    ``benchmark_spec``'s baseline, regardless of which threshold was used.

    The LLM's ``verdict`` is then cross-checked against the computed
    ``is_improvement`` (and against ``anti_cheat_flags``) — any of the
    following is treated exactly like malformed JSON: retried (bounded),
    then raises ``EvaluatorParseError``:
      - ``verdict == "new_best"`` while the computed ``is_improvement`` is
        not ``True``.
      - ``verdict == "valid_not_best"`` while the computed
        ``is_improvement`` is not ``False``.
      - ``verdict == "suspicious"`` while ``anti_cheat_flags`` is empty.
      - ``verdict == "conflict"`` while ``anti_cheat_flags`` is empty (must
        name the contradicted ``prior_research_memory.verified_findings``
        entry).
    ``verdict == "invalid"`` is allowed regardless of ``is_improvement``
    (e.g. the run failed or produced no metric).

    Raises:
        EvaluatorParseError (permanent): malformed/unparseable LLM JSON, a
            missing/invalid ``verdict``/``suspicious``/``anti_cheat_flags``,
            a verdict/is_improvement (or verdict/anti_cheat_flags)
            inconsistency, or a schema-invalid assembled ``EvaluationReport``,
            after bounded retries.
        EvaluatorLLMError (transient): the LLM client itself raised.
    """
    # Repo-improve short-circuit: a run cloned+modified a REAL repo and was scored by the repo's
    # repository-run result. The scientific comparison is frozen before execution; re-checking the
    # result against a self-written VerifierSpec or an after-the-fact semantic opinion spuriously flags
    # wrong_benchmark / metric_mismatch / missing-held-out. So we TRUST the repo's verdict here.
    repo_report = _repo_evaluation_report(run_record.get("environment") or "")
    if repo_report is not None:
        return repo_report

    metric = run_record.get("metric_value")
    baseline = (benchmark_spec.get("baseline") or {}).get("reported_metric")
    current_best = (prior_research_memory or {}).get("best_metric")
    threshold = current_best if current_best is not None else baseline
    is_improvement = metric is not None and threshold is not None and metric > threshold
    # Deterministic degeneracy signal (routes to MOSS via result_suspicious below): a result
    # reproduction alone cannot vouch for -- a non-finite metric, or a method metric exactly
    # equal to the baseline (the experiment produced no distinguishing power). None if clean.
    degeneracy = _degeneracy_reason(metric, baseline)

    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(
                _load_evaluate_experiment_system_prompt(), prompt_patch
            ),
        },
        {
            "role": "user",
            "content": _build_evaluate_experiment_user_message(
                run_record, verifier_spec, benchmark_spec, prior_research_memory
            ),
        },
    ]

    def process(obj: dict) -> tuple[dict | None, str | None]:
        verdict = obj.get("verdict")
        if verdict not in (
            "invalid", "valid_not_best", "new_best", "suspicious", "conflict",
        ):
            return None, (
                "verdict must be one of 'invalid', 'valid_not_best', "
                f"'new_best', 'suspicious', 'conflict', got {verdict!r}"
            )

        suspicious = obj.get("suspicious")
        if not isinstance(suspicious, bool):
            return None, f"suspicious must be a boolean, got {suspicious!r}"

        anti_cheat_flags = obj.get("anti_cheat_flags")
        if not isinstance(anti_cheat_flags, list) or not all(
            isinstance(flag, str) for flag in anti_cheat_flags
        ):
            return None, "anti_cheat_flags must be a list of strings"

        # GUARD (deterministic, anti-reward-hacking): is_improvement was
        # computed above from metric vs. the running-best/baseline threshold
        # — never trusted from the LLM. The LLM's verdict must be consistent
        # with it.
        if verdict == "new_best" and is_improvement is not True:
            return None, (
                "verdict 'new_best' requires the computed is_improvement to "
                f"be True (metric={metric!r} > threshold={threshold!r}); "
                f"computed is_improvement={is_improvement}"
            )
        if verdict == "valid_not_best" and is_improvement is not False:
            return None, (
                "verdict 'valid_not_best' requires the computed "
                f"is_improvement to be False; computed is_improvement="
                f"{is_improvement} (metric={metric!r}, threshold={threshold!r})"
            )
        if verdict == "suspicious" and not anti_cheat_flags:
            return None, (
                "verdict 'suspicious' requires a non-empty anti_cheat_flags "
                "list"
            )
        if verdict == "conflict" and not anti_cheat_flags:
            return None, (
                "verdict 'conflict' requires a non-empty anti_cheat_flags "
                "list naming the contradicted verified finding"
            )

        report = {
            "verdict": verdict,
            "metric_value": metric,
            "baseline_metric": baseline,
            "is_improvement": is_improvement,
            "suspicious": suspicious,
            "anti_cheat_flags": anti_cheat_flags,
            "reasoning": obj.get("reasoning"),
        }

        # DETERMINISTIC suspicion guard: FORCE verdict="suspicious" (-> MOSS review) whenever
        # there is ANY anti-cheat concern -- a degeneracy signal (no separation / non-finite),
        # a non-empty anti_cheat_flags list, or the evaluator's own suspicious=True. Otherwise
        # the LLM can name a real concern (e.g. "metric directionality confusion") yet pick a
        # milder verdict (valid_not_best / new_best), so it never reaches MOSS and the concern
        # is silently buried. Mirrors the is_improvement guard: computed in code, not trusted
        # from the LLM. A clean result (no degeneracy, no flags, suspicious False) is untouched.
        flags = list(report["anti_cheat_flags"])
        if degeneracy and degeneracy not in flags:
            flags.append(degeneracy)
        # Only the NON-MOSS verdicts are overridden; "suspicious" and "conflict" already route
        # to MOSS (result_suspicious / result_inconsistency) with their own semantics -- leave
        # them. So a real concern under a milder verdict is escalated, without clobbering a
        # legitimate conflict finding (which necessarily carries flags too).
        if report["verdict"] in ("invalid", "valid_not_best", "new_best") and (
            flags or report["suspicious"]
        ):
            report["verdict"] = "suspicious"
            report["suspicious"] = True
            report["anti_cheat_flags"] = flags or ["evaluator_flagged_suspicious"]

        errors = _validate_evaluation_report(report)
        if errors:
            return None, f"assembled EvaluationReport failed schema validation: {errors}"

        return report, None

    return _run_llm_loop(messages, llm_client, process=process)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# verify_best_result / 验证最佳结果
# ---------------------------------------------------------------------------


def _verify_recorded_result(
    evaluation_report: dict,
    run_record: dict,
) -> dict:
    """Verify one attempt's recorded outcome without judging scientific confidence.

    Successful attempts must bind one finite metric consistently.  Failed/invalid
    attempts must bind an observable failure reason.  Statistical significance,
    replication, ablation strength and cheating opinions are deliberately outside
    this check; they belong to reporting or the separate comparison guardian.
    """
    mapped = _repo_evaluation_report(run_record.get("environment") or "")
    flags: list[str] = []
    checks: list[str] = []

    def finite_number(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    run_metric = run_record.get("metric_value")
    report_metric = evaluation_report.get("metric_value")
    mapped_metric = mapped.get("metric_value") if mapped is not None else report_metric
    report_baseline = evaluation_report.get("baseline_metric")
    mapped_baseline = mapped.get("baseline_metric") if mapped is not None else report_baseline
    evaluation_verdict = str(evaluation_report.get("verdict") or "")
    expected_verdict = str(mapped.get("verdict") or "") if mapped is not None else evaluation_verdict

    if mapped is not None and evaluation_verdict != expected_verdict:
        flags.append("outcome_mismatch:evaluation_report/repo_result")

    successful_measurement = evaluation_verdict in {"new_best", "valid_not_best"}
    recorded_failure = evaluation_verdict == "invalid"

    if successful_measurement:
        if run_record.get("succeeded") is not True or run_record.get("exit_code") != 0:
            flags.append("run_not_successful")
        if not all(finite_number(v) for v in (run_metric, report_metric, mapped_metric)):
            flags.append("method_metric_missing_or_non_finite")
        elif not (float(run_metric) == float(report_metric) == float(mapped_metric)):
            flags.append(
                "method_metric_mismatch:run_record/evaluation_report/repo_result")
        else:
            checks.append(
                "metric_matches_run_record: recorded method metric is identical across the "
                "completed attempt records")
        if mapped is not None:
            if not all(finite_number(v) for v in (report_baseline, mapped_baseline)):
                flags.append("baseline_metric_missing_or_non_finite")
            elif float(report_baseline) != float(mapped_baseline):
                flags.append("baseline_metric_mismatch:evaluation_report/repo_result")
            else:
                checks.append(
                    "baseline_matches_run_record: recorded baseline metric is identical across "
                    "the attempt records")
        if evaluation_verdict == "new_best" and evaluation_report.get("is_improvement") is not True:
            flags.append("new_best_without_improvement")
        if (evaluation_verdict == "valid_not_best"
                and evaluation_report.get("is_improvement") is not False):
            flags.append("valid_not_best_marked_as_improvement")
    elif recorded_failure:
        diagnostic_parts = [
            str(evaluation_report.get("reasoning") or "").strip(),
            str(run_record.get("stderr_excerpt") or "").strip(),
            str(run_record.get("stdout_excerpt") or "").strip(),
        ]
        diagnostic_parts = [part for part in diagnostic_parts if part]
        if not diagnostic_parts:
            flags.append("failure_reason_missing")
        else:
            checks.append("failure_confirmed: attempt recorded an invalid outcome and a cause")
    else:
        flags.append(f"unsupported_evaluation_outcome:{evaluation_verdict or 'missing'}")

    frozen_digest = evaluation_report.get("run_record_digest")
    if frozen_digest is not None and frozen_digest != artifact_digest(run_record):
        flags.append("run_record_digest_mismatch")

    if flags:
        return {
            "verdict": "failed",
            "checks": checks or ["attempt record consistency: failed"],
            "anti_cheat_flags": flags,
            "reasoning": (
                "The attempt outcome is not consistently bound to its RunRecord and "
                "EvaluationReport."
            ),
        }

    if recorded_failure:
        reason_parts = [
            str(evaluation_report.get("reasoning") or "").strip(),
            str(run_record.get("stderr_excerpt") or "").strip(),
            str(run_record.get("stdout_excerpt") or "").strip(),
        ]
        failure_reason = " | ".join(part for part in reason_parts if part)
        return {
            "verdict": "verified",
            "checks": checks,
            "anti_cheat_flags": [],
            "reasoning": f"Recorded failure cause: {failure_reason}",
        }

    return {
        "verdict": "verified",
        "checks": ["run_completed: succeeded with exit_code=0", *checks],
        "anti_cheat_flags": [],
        "reasoning": (
            "The recorded metric is internally consistent with the completed attempt. "
            "Replication count, effect size, noise and cheating opinions are outside this "
            "record-verification step."
        ),
    }


def verify_best_result(
    evaluation_report: dict,
    run_record: dict,
    verifier_spec: dict,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> dict:
    """Verify one attempt's recorded result or recorded failure cause.

    The LLM/verifier inputs remain in the public signature for handler/API
    compatibility, but this decision is deterministic.  Verification does not
    judge cheating, statistical significance, seed count, ablations, or whether
    the method is scientifically convincing.
    """
    del verifier_spec, llm_client, prompt_patch
    report = _verify_recorded_result(evaluation_report, run_record)
    errors = _validate_verification_report(report)
    if errors:
        raise EvaluatorParseError(
            f"assembled VerificationReport failed schema validation: {errors}")
    return report


__all__ = [
    "EvaluatorError",
    "EvaluatorParseError",
    "EvaluatorLLMError",
    "select_benchmark",
    "prepare_verifier",
    "evaluate_experiment",
    "verify_best_result",
    "MAX_RETRIES",
]
