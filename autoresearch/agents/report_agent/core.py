"""
ReportAgent core: the pure honest-report generation logic. ReportAgent 核心：
与传输层无关的纯"诚实报告"生成逻辑。

Given the frozen ResearchMemory (and, optionally, the original ResearchSpec),
``generate_report`` calls an injected LLM client to classify every claim into
8 honest categories (``verified_findings``, ``negative_results``,
``unverified_ideas``, ``experiment_not_run``, ``replication_failed``,
``external_data_incomplete``, ``analysis_caveats``, ``known_risks``), build a
per-experiment ``experiment_ledger``, and write a ``summary`` — producing the
final ``Report`` for the ``Reporting`` state (AutoResearch state machine
V1.10). Validates against the shared data-contract, with NO EventBus / Redis
/ Store access. The handler (a later milestone) is responsible for resolving
refs, persisting the artifact, and publishing events.

给定冻结的 ResearchMemory（以及可选的原始 ResearchSpec），``generate_report``
调用注入的 LLM 客户端将每条论断分类到 8 个诚实类别中（已验证发现、负面结果、
未验证想法、未执行实验、复现失败、外部数据不完整、分析局限、已知风险），
构建按实验的 ``experiment_ledger``，并撰写 ``summary`` —— 生成 ``Reporting``
状态（AutoResearch 状态机 V1.10）的最终 ``Report``。对照 data-contract 校验
—— 不含任何 EventBus/Redis/Store 访问。handler（后续里程碑）负责解析 ref、
持久化产物、发布事件。

Deterministic grounding guard (honest-report anti-hallucination core)
确定性接地护栏（诚实报告反幻觉核心）
-----------------------------------------------------------------------
``report["verified_findings"]`` and ``report["negative_results"]`` are NEVER
trusted at face value from the LLM's response — they are CHECKED HERE IN
CODE against the frozen ``research_memory`` that was given as input:

- ``set(report["verified_findings"]) <= set(research_memory.get("verified_findings", []))``
  every claimed verified finding MUST already be present, verbatim, in the
  ResearchMemory's own ``verified_findings``. Any finding not in that set is
  treated as a fabrication.
- ``set(report["negative_results"]) <= set(research_memory.get("negative_results", []))``
  likewise for negative results.

If either check fails, the response is rejected exactly like a schema
violation: bounded retry, then a typed, permanent ``ReportParseError``. The
other 6 classification arrays (``unverified_ideas``, ``experiment_not_run``,
``replication_failed``, ``external_data_incomplete``, ``analysis_caveats``,
``known_risks``) are LLM judgment calls and are NOT grounded against
ResearchMemory.

This mirrors ``memory_agent/core.py``'s ``consolidate`` anti-cheat guard: the
LLM is never trusted to police its own claims about what was actually
established — the check is deterministic and lives in code.

这镜像 ``memory_agent/core.py`` 中 ``consolidate`` 的反作弊护栏：绝不信任 LLM
自我核查其论断是否真的成立 —— 校验是确定性的，写在代码里。
"""

from __future__ import annotations

import json
import math
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


class ReportError(Exception):
    """Base for ReportAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "report_failed"
    #: transient | permanent
    classification: str = "transient"


class ReportParseError(ReportError):
    """LLM never returned a well-formed, schema-valid, grounded Report after
    retries. 重试用尽后 LLM 仍未返回合法、schema 校验通过、接地校验通过的
    Report。"""

    code = "report_parse_error"
    classification = "permanent"


class ReportLLMError(ReportError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "report_llm_error"
    classification = "transient"


# ---------------------------------------------------------------------------
# LLM caller protocol / LLM 调用协议
# ---------------------------------------------------------------------------


class LLMResultLike(Protocol):
    content: str | None
    usage: dict
    cost_delta: str
    finish_reason: str | None


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
def _load_generate_report_system_prompt() -> str:
    return (_PROMPTS_DIR / "generate_report_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _report_validator() -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / "report.schema.yaml").read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validate_report(report: dict) -> list[str]:
    return [e.message for e in _report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


_PROMPT_HISTORY_LIMIT = 32
_PROMPT_HISTORY_TEXT_LIMIT = 1200
_PROMPT_HISTORY_FIELDS = (
    "sequence", "component_id", "mode", "plan_version", "round", "role",
    "status", "attempt", "operator", "execution_succeeded", "dependency_closure",
    "metric", "contract_passed", "became_best", "mir_verdict",
    "verification_verdict", "changes_summary", "failure_reason",
)
_PROMPT_ACCEPTANCE_FIELDS = (
    "passed", "measurement", "actual", "expected", "acceptance_check_result",
    "parent_metric", "candidate_metric", "reason",
)


def _prompt_text(value: object) -> object:
    if not isinstance(value, str) or len(value) <= _PROMPT_HISTORY_TEXT_LIMIT:
        return value
    return value[:_PROMPT_HISTORY_TEXT_LIMIT] + "... [prompt excerpt; full evidence preserved]"


def _compact_research_memory_for_prompt(research_memory: dict) -> dict:
    """Bound unbounded experiment history in the LLM-only report view.

    The frozen ResearchMemory remains authoritative and unchanged for grounding,
    number traceability, report post-processing and the persisted evidence bundle.
    """
    compact = dict(research_memory)

    negatives = research_memory.get("negative_results")
    if isinstance(negatives, list):
        compact["negative_results"] = [
            _prompt_text(item) for item in negatives[-_PROMPT_HISTORY_LIMIT:]
        ]
        omitted = len(negatives) - len(compact["negative_results"])
        if omitted:
            compact["negative_results_omitted_from_prompt"] = omitted

    history = research_memory.get("component_history")
    if isinstance(history, list):
        compact_history = []
        for entry in history[-_PROMPT_HISTORY_LIMIT:]:
            if not isinstance(entry, dict):
                compact_history.append(_prompt_text(entry))
                continue
            item = {
                key: _prompt_text(entry[key])
                for key in _PROMPT_HISTORY_FIELDS
                if key in entry
            }
            acceptance = entry.get("acceptance_result")
            if isinstance(acceptance, dict):
                item["acceptance_result"] = {
                    key: _prompt_text(acceptance[key])
                    for key in _PROMPT_ACCEPTANCE_FIELDS
                    if key in acceptance
                }
            compact_history.append(item)
        compact["component_history"] = compact_history
        omitted = len(history) - len(compact_history)
        if omitted:
            compact["component_history_omitted_from_prompt"] = omitted

    return compact


def _build_generate_report_user_message(
    research_memory: dict,
    research_spec: dict | None,
    research_context: dict | None = None,
) -> str:
    prompt_memory = _compact_research_memory_for_prompt(research_memory)
    return "\n".join(
        [
            "ResearchMemory (JSON):",
            json.dumps(prompt_memory, ensure_ascii=False, indent=2),
            "",
            "ResearchSpec (JSON):",
            (
                json.dumps(research_spec, ensure_ascii=False, indent=2)
                if research_spec is not None
                else "none"
            ),
            "",
            "Reviewed research context (JSON; use only supplied sources and preserve audit caveats):",
            json.dumps(research_context or {}, ensure_ascii=False, indent=2),
            "",
            "Produce the final Report; respond with a single JSON object "
            "matching the shape described in the system prompt.",
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


_REPORT_BODY_SYSTEM = """You write the complete human-facing AutoResearch report.
Return Markdown only, without JSON or a wrapping code fence. Build the scientific
body from the supplied ResearchSpec, reviewed literature, confirmed Method, audits,
ResearchMemory, and machine audit. Do not merely expand the risk lists.

Every report must position the question and result against the supplied literature,
separate prior work from this session's contribution, and cite only supplied titles,
identifiers, or URLs. Never invent a citation or claim that an unexecuted check ran.
The machine audit is authoritative about execution and verification status. If
ResearchMemory supplies `selected_result`, use its bound evidence and reviews for
that selected attempt's status; older failures retain their own attempt scope.
Do not let a stale machine-audit caveat override the selected result's evidence,
and do not infer verification beyond that evidence's recorded scope. If
`verified_findings` is empty, do not label any theorem or proposition as proved,
verified, or "已证"; label it as a proof sketch pending independent review. If
`best_metric` is null, there is no verified complete-Method metric; this does not
negate executed and verified component observations. Preserve those observations
with their component scope, even after a later failure. A check listed under
`experiment_not_run`/`next_steps` must not be described as executed;
do not turn expected/example numbers into measured results.
Keep numerical verification, implementation fidelity, task compliance, and component
acceptance separate, using evidence from the same attempt/checkpoint. In particular,
no improvement does not imply off-method: a faithful implementation can fail its gain acceptance check.
Report an official reward only when it is present in evidence bound to the selected
attempt. A locally reproduced baseline is a measured comparison value; it does not
replace fixed scoring anchors in the target evaluator. If the selected attempt has
no bound reward receipt, report its measured metric and say the official reward is
unverified; do not calculate or present a normalized reward in the report body.
When a reproduced baseline and selected method metric are both supplied, derive any
unqualified speedup or percentage change from exactly those two values. If a bound
evaluator receipt reports a different ratio because it uses a fixed scoring anchor,
label that ratio explicitly as fixed-anchor scoring and never attribute it to the
reproduced baseline.
Check each attempt's effective evaluator settings against its bound result. An
unchanged evaluator may adapt its workload to the candidate; disclose differing
settings instead of claiming that baseline and method used identical settings.
Do not transfer an earlier attempt's contract violation to a later reviewed candidate;
identify which attempt each finding belongs to, and do not erase the earlier violation.
When attribution or compliance evidence is missing or conflicting, state that gap
instead of inferring a verdict from the performance or the session's last failure.
Source history is cumulative across revisions and candidate rollback. List allowed
references and their role. Prohibited exposure cannot be erased by reverting code;
do not describe reference-assisted or unaudited work as independently discovered.

For an experimental task include: literature/community baseline positioning;
operational question, metric and comparison semantics; repository, data and
environment; reproduced baseline; method mechanism and actual implementation;
chronological attempts and failure diagnoses; component checks/ablations; verified
baseline-versus-method values; conclusions and limitations.

For a theoretical task include: research question; assumptions/axioms; definitions;
propositions/theorems; derivations or proof sketches; counterexamples and boundary
cases; component/assumption checks; implications; limitations; and optional light
survey or synthetic checks. Clearly label what is proved, sketched, calibrated, or
unverified. Theory needs no benchmark or numeric metric. A survey can calibrate an
assumption but is not a proof.

Be substantive and specific, normally 2,500-5,000 Chinese characters for a Chinese
task or comparable detail in the user's language. Avoid repetition and padding."""


_RATIO_TOKEN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:[xX×]|倍(?:加速)?)")
_PERCENT_TOKEN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
_FIXED_ANCHOR_WORDS = re.compile(
    r"fixed[- ]?(?:scoring|score|benchmark|reference)?\s*(?:anchor|baseline)"
    r"|scoring\s+(?:anchor|baseline)|reference\s+baseline|task\.toml"
    r"|固定(?:计分|评分|参考|任务)?(?:锚点|基准)|(?:计分|评分|参考)基准|任务声明的基线",
    re.I,
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _selected_comparison_semantics(research_memory: dict) -> dict:
    """Return host-derived comparison values for the selected result.

    Evaluators sometimes expose a speedup relative to a fixed scoring anchor while
    AutoResearch also has a freshly reproduced baseline.  Both values are legitimate,
    but they are different comparisons and must not be conflated in prose.
    """
    selected = research_memory.get("selected_result")
    if not isinstance(selected, dict):
        return {}
    baseline = _finite_number(selected.get("baseline_metric"))
    method = _finite_number(selected.get("method_metric"))
    if baseline is None or method is None:
        return {}
    lower_is_better = selected.get("lower_is_better", True) is not False
    measured_speedup = None
    if baseline > 0 and method > 0:
        measured_speedup = baseline / method if lower_is_better else method / baseline
    measured_change = None
    if baseline != 0:
        delta = baseline - method if lower_is_better else method - baseline
        measured_change = 100.0 * delta / abs(baseline)

    evaluator_speedups: list[float] = []
    for receipt in selected.get("attempt_review_evidence") or []:
        if not isinstance(receipt, dict):
            continue
        fields = receipt.get("verified_output_fields")
        if not isinstance(fields, dict):
            continue
        for path, value in fields.items():
            if not re.search(r"(?:^|/)speedup$", str(path), re.I):
                continue
            number = _finite_number(value)
            if number is not None and number not in evaluator_speedups:
                evaluator_speedups.append(number)
    return {
        "reproduced_baseline_metric": baseline,
        "selected_method_metric": method,
        "lower_is_better": lower_is_better,
        "measured_baseline_speedup": measured_speedup,
        "measured_baseline_relative_improvement_percent": measured_change,
        "bound_evaluator_reported_speedups": evaluator_speedups,
        "reporting_rule": (
            "Unqualified speedup and percentage change use the reproduced baseline. "
            "A differing evaluator-reported ratio must be explicitly labelled as using "
            "a fixed scoring anchor, not the reproduced baseline."
        ),
    }


def _close_derived(value: float, expected: float, *, abs_tol: float) -> bool:
    return math.isclose(value, expected, rel_tol=0.01, abs_tol=abs_tol)


def _report_body_comparison_errors(body: str, semantics: dict) -> list[str]:
    baseline = _finite_number(semantics.get("reproduced_baseline_metric"))
    method = _finite_number(semantics.get("selected_method_metric"))
    expected_speedup = _finite_number(semantics.get("measured_baseline_speedup"))
    expected_percent = _finite_number(
        semantics.get("measured_baseline_relative_improvement_percent"))
    evaluator_speedups = [
        number for number in (
            _finite_number(value)
            for value in semantics.get("bound_evaluator_reported_speedups") or [])
        if number is not None
    ]
    if baseline is None or method is None:
        return []

    errors: list[str] = []
    # Sentence and Markdown-table-row scope is deliberate: a ratio may validly
    # describe prior work elsewhere in the report, but not this measured pair.
    units = [unit.strip() for unit in re.split(
        r"(?:\r?\n)+|(?<=[。！？；;])\s*|(?<=[.!?])\s+", body)
             if unit.strip()]
    baseline_text = format(baseline, ".15g")
    method_text = format(method, ".15g")
    for unit in units:
        has_pair = baseline_text in unit and method_text in unit
        fixed_anchor = bool(_FIXED_ANCHOR_WORDS.search(unit))
        for match in _RATIO_TOKEN.finditer(unit):
            value = float(match.group(1))
            if expected_speedup is None:
                continue
            if _close_derived(value, expected_speedup, abs_tol=0.01):
                continue
            is_evaluator_value = any(
                _close_derived(value, official, abs_tol=0.01)
                for official in evaluator_speedups
            )
            if is_evaluator_value and fixed_anchor:
                continue
            if has_pair or is_evaluator_value:
                errors.append(
                    f"speedup {value:g}x is not the measured-baseline ratio "
                    f"{expected_speedup:.6f}x for baseline={baseline_text} and "
                    f"method={method_text}; a differing evaluator value must be "
                    "labelled as relative to its fixed scoring anchor"
                )
        if has_pair and expected_percent is not None and not fixed_anchor:
            for match in _PERCENT_TOKEN.finditer(unit):
                value = float(match.group(1))
                if not _close_derived(value, expected_percent, abs_tol=0.15):
                    errors.append(
                        f"percentage {value:g}% is not the reproduced-baseline change "
                        f"{expected_percent:.6f}% for baseline={baseline_text} and "
                        f"method={method_text}"
                    )
    return list(dict.fromkeys(errors))


def _repair_report_body_comparisons(body: str, semantics: dict) -> str:
    """Repair only host-auditable derived comparisons in an exhausted draft.

    The prose remains LLM-authored.  This fallback changes a ratio or percentage
    only in the same sentence/table row as the exact reproduced baseline and
    selected method values, where the host can derive the replacement without
    interpretation.  Evaluator ratios outside such a pair are only labelled with
    their already-bound fixed-anchor scope.
    """
    baseline = _finite_number(semantics.get("reproduced_baseline_metric"))
    method = _finite_number(semantics.get("selected_method_metric"))
    expected_speedup = _finite_number(semantics.get("measured_baseline_speedup"))
    expected_percent = _finite_number(
        semantics.get("measured_baseline_relative_improvement_percent"))
    evaluator_speedups = [
        number for number in (
            _finite_number(value)
            for value in semantics.get("bound_evaluator_reported_speedups") or [])
        if number is not None
    ]
    if baseline is None or method is None:
        return body

    baseline_text = format(baseline, ".15g")
    method_text = format(method, ".15g")
    units = [unit.strip() for unit in re.split(
        r"(?:\r?\n)+|(?<=[。！？；;])\s*|(?<=[.!?])\s+", body)
             if unit.strip()]
    repaired_body = body
    for unit in units:
        has_pair = baseline_text in unit and method_text in unit
        fixed_anchor = bool(_FIXED_ANCHOR_WORDS.search(unit))

        def repair_ratio(match: re.Match[str]) -> str:
            value = float(match.group(1))
            if expected_speedup is None or _close_derived(
                    value, expected_speedup, abs_tol=0.01):
                return match.group(0)
            if has_pair:
                return format(expected_speedup, ".6f") + match.group(0)[len(match.group(1)):]
            is_evaluator_value = any(
                _close_derived(value, official, abs_tol=0.01)
                for official in evaluator_speedups
            )
            if is_evaluator_value and not fixed_anchor:
                return match.group(0) + " (fixed scoring anchor)"
            return match.group(0)

        repaired = _RATIO_TOKEN.sub(repair_ratio, unit)

        if has_pair and expected_percent is not None and not fixed_anchor:
            def repair_percent(match: re.Match[str]) -> str:
                value = float(match.group(1))
                if _close_derived(value, expected_percent, abs_tol=0.15):
                    return match.group(0)
                return format(expected_percent, ".6f") + match.group(0)[len(match.group(1)):]

            repaired = _PERCENT_TOKEN.sub(repair_percent, repaired)

        if repaired != unit:
            repaired_body = repaired_body.replace(unit, repaired, 1)
    return repaired_body


def _academic_paper_skill() -> str:
    """Load the small project-local writing skill used only by the long report call."""
    path = Path(__file__).with_name("skills") / "academic-paper-writing" / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    return text.strip()


def _generate_report_body(
    research_memory: dict,
    research_spec: dict | None,
    research_context: dict,
    machine_report: dict,
    llm_client: LLMClientLike,
) -> str:
    comparison_semantics = _selected_comparison_semantics(research_memory)
    prompt_memory = _compact_research_memory_for_prompt(research_memory)
    user = "\n\n".join([
        "ResearchSpec:\n" + json.dumps(research_spec or {}, ensure_ascii=False, indent=2),
        "Reviewed research context:\n" + json.dumps(
            research_context, ensure_ascii=False, indent=2),
        "ResearchMemory:\n" + json.dumps(prompt_memory, ensure_ascii=False, indent=2),
        "Machine audit:\n" + json.dumps(machine_report, ensure_ascii=False, indent=2),
        "Host-derived comparison semantics (authoritative for derived numbers):\n"
        + json.dumps(comparison_semantics, ensure_ascii=False, indent=2),
    ])
    writing_skill = _academic_paper_skill()
    system = _REPORT_BODY_SYSTEM
    if writing_skill:
        system += "\n\nPROJECT-LOCAL ACADEMIC WRITING SKILL:\n" + writing_skill
    last_error = "no attempts made"
    saw_response = False
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    messages = list(base_messages)
    last_body = ""
    comparison_errors: list[str] = []
    truncated_prefix = ""
    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(
                messages=messages,
                response_format=None,
                max_tokens=262144,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        saw_response = True
        raw_body = str(result.content or "")
        provider_restarted = bool(
            truncated_prefix and raw_body.lstrip().startswith(("#", "```"))
        )
        combined_body = (
            raw_body if provider_restarted
            else truncated_prefix + raw_body if truncated_prefix
            else raw_body
        )
        body = combined_body.strip()
        if body.startswith("```") and body.endswith("```"):
            lines = body.splitlines()
            body = "\n".join(lines[1:-1]).strip()
        finish_reason = getattr(result, "finish_reason", None)
        if finish_reason == "length" and raw_body:
            # Rewriting a report that hit the provider output ceiling repeats
            # the same truncation and can accept an incomplete document.
            # Preserve the prefix and request only the missing suffix.
            if truncated_prefix and raw_body.lstrip().startswith("#"):
                truncated_prefix = raw_body
            else:
                truncated_prefix = combined_body
            messages = list(base_messages)
            messages.append({"role": "assistant", "content": truncated_prefix})
            messages.append({"role": "user", "content": (
                "The Markdown report was cut off by the output-length limit. "
                "Continue exactly from its final character and return only the missing "
                "suffix needed to complete the same report. Do not restart, repeat, "
                "summarize, or wrap it in a code fence."
            )})
            last_body = body
            last_error = "response was truncated by the output-length limit"
            continue
        last_body = body
        comparison_errors = _report_body_comparison_errors(body, comparison_semantics)
        if len(body) >= 800 and not comparison_errors:
            return body
        if truncated_prefix:
            # A provider may ignore the continuation request and restart a
            # complete report. Accept it alone only through the same audits.
            standalone = raw_body.strip()
            standalone_errors = _report_body_comparison_errors(
                standalone, comparison_semantics)
            if len(standalone) >= 800 and not standalone_errors:
                return standalone
        if len(body) < 800:
            last_error = "response was empty or too short"
        else:
            last_error = "numeric comparison audit failed: " + "; ".join(comparison_errors)
        truncated_prefix = ""
        messages = list(base_messages) + [
            {"role": "assistant", "content": body},
            {"role": "user", "content": (
                "The previous Markdown draft failed validation: " + last_error + ". "
                "Rewrite the complete report. Keep reproduced-baseline comparisons separate "
                "from evaluator values that use a fixed scoring anchor."
            )},
        ]
    if len(last_body) >= 800 and comparison_errors:
        repaired = _repair_report_body_comparisons(last_body, comparison_semantics)
        if not _report_body_comparison_errors(repaired, comparison_semantics):
            return repaired
    if not saw_response:
        raise ReportLLMError(
            f"long report body call failed after retries: {last_error}")
    raise ReportParseError(
        f"long report body was invalid after retries: {last_error}")


def _fallback_machine_report(research_memory: dict) -> dict:
    """Build a conservative audit when the LLM cannot emit JSON.

    The long Markdown writer still runs afterwards.  Keep this object strictly
    grounded in frozen memory and leave the experiment ledger empty rather than
    guessing structure from free text.
    """
    return {
        "summary": "Machine-readable audit reconstructed from frozen ResearchMemory.",
        "verified_findings": [
            item for item in (research_memory.get("verified_findings") or [])
            if isinstance(item, str)
        ],
        "negative_results": [
            item for item in (research_memory.get("negative_results") or [])
            if isinstance(item, str)
        ],
        "unverified_ideas": [],
        "experiment_not_run": [
            item for item in (research_memory.get("next_steps") or [])
            if isinstance(item, str)
        ],
        "replication_failed": [],
        "external_data_incomplete": [],
        "analysis_caveats": [
            "The LLM did not return a parseable machine-audit JSON object; "
            "this audit was conservatively reconstructed from frozen ResearchMemory."
        ],
        "known_risks": [],
        "experiment_ledger": [],
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
    ``ReportParseError``.
    调用 LLM，把 JSON 响应解析为 object，再跑 process（成功返回
    ``(result, None)``，失败返回 ``(None, error_message)``）做校验/提取。
    任何失败均有限重试，用尽后抛 ReportParseError。"""
    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise ReportLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

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

    raise ReportParseError(
        f"LLM did not produce a valid response after retries: {last_error}"
    )


# ---------------------------------------------------------------------------
# generate_report / 生成诚实报告
# ---------------------------------------------------------------------------


def generate_report(
    research_memory: dict,
    research_spec: dict | None,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
    citation_audit_report: dict | None = None,
    reviewer_llm: LLMClientLike | None = None,
    research_context: dict | None = None,
) -> dict:
    """Generate the final honest Report (pure, no transport).
    生成最终诚实 Report（纯，无传输层）。

    Args:
        research_memory: the frozen ResearchMemory to report on.
        research_spec: the original ResearchSpec, or ``None`` if unavailable.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).
        citation_audit_report: optional EvidenceAuditAgent Mode C
            CitationAuditReport (``{unsupported_claims: [{claim, why}, ...],
            checked_count}``). When given, reinforces the grounding guard
            above: any claim it flags is DROPPED from ``verified_findings``
            / ``negative_results`` (a citation-unsupported claim is not
            honestly reportable as established), regardless of what the LLM
            or ResearchMemory said. ``None`` (the default) behaves exactly
            as before this parameter existed.

    Returns:
        A schema-valid ``Report`` dict.

    Raises:
        ReportParseError (permanent): malformed/unparseable LLM JSON, a
            non-dict response, a fabricated ``verified_findings`` or
            ``negative_results`` entry not present verbatim in
            ``research_memory`` (the grounding guard), or a schema-invalid
            Report, after bounded retries.
        ReportLLMError (transient): the LLM client itself raised.
    """
    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_generate_report_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_generate_report_user_message(
                research_memory, research_spec, research_context),
        },
    ]

    # GUARD (deterministic grounding, never trusted from the LLM): see module
    # docstring. Computed once, up front, from the frozen input.
    allowed_verified_findings = set(research_memory.get("verified_findings", []) or [])
    allowed_negative_results = set(research_memory.get("negative_results", []) or [])

    def process(obj: dict) -> tuple[dict | None, str | None]:
        verified_findings = obj.get("verified_findings")
        if verified_findings is not None and not isinstance(verified_findings, list):
            return None, "verified_findings must be a JSON array"

        negative_results = obj.get("negative_results")
        if negative_results is not None and not isinstance(negative_results, list):
            return None, "negative_results must be a JSON array"

        # The model may paraphrase a grounded finding even after explicit retry feedback.
        # Dropping those entries is both safer and more robust than failing the whole final
        # report: the handler deterministically merges the exact ResearchMemory strings back
        # after this function returns.  Non-string entries are ungrounded as well.
        if verified_findings is not None:
            obj["verified_findings"] = [
                item for item in verified_findings
                if isinstance(item, str) and item in allowed_verified_findings
            ]

        if negative_results is not None:
            obj["negative_results"] = [
                item for item in negative_results
                if isinstance(item, str) and item in allowed_negative_results
            ]

        errors = _validate_report(obj)
        if errors:
            return None, f"assembled Report failed schema validation: {errors}"

        return obj, None

    try:
        report = _run_llm_loop(messages, llm_client, process=process)  # type: ignore[assignment]
    except ReportParseError as exc:
        # A complete human-facing report has a separate Markdown generation
        # pass.  If only the preliminary machine audit repeatedly contains no
        # parseable JSON, use a deterministic, memory-grounded audit instead
        # of failing the already-completed research session.  Schema-invalid
        # JSON still fails normally so contract errors are not hidden.
        if not research_context or "response was not valid JSON" not in str(exc):
            raise
        report = _fallback_machine_report(research_memory)
    if citation_audit_report is not None:
        report = _drop_citation_unsupported_claims(report, citation_audit_report)
    # T8: group negative results by failure mode (deterministic; attached only when >=1 matched).
    clusters = _cluster_failure_modes(report.get("negative_results") or [])
    if clusters:
        report["failure_clusters"] = clusters
    # GUARANTEED blind-review gate (L0 annotate): when a reviewer LLM is provided, a fresh,
    # blind pass (it sees ONLY the report's fields, not the author's reasoning) flags a
    # free-text SUMMARY that over-claims a positive result absent from verified_findings, and
    # annotates ``report["review"]``. Non-blocking, fail-open. Borrowed from openscience's
    # code-level reviewer gate ("prompt MANDATORY != code-enforced").
    if reviewer_llm is not None:
        review = _blind_review(reviewer_llm, report)
        # T5: DETERMINISTIC number-traceability -- every metric-like number in the summary must
        # appear in the grounded ResearchMemory; an untraceable number is presumed fabricated.
        untraceable = _untraceable_numbers(report.get("summary") or "", research_memory)
        if untraceable:
            review["findings"] = list(review.get("findings") or []) + [
                "summary contains number(s) not traceable to any experiment metric in "
                f"ResearchMemory (presumed fabricated): {untraceable}"]
            review["verdict"] = "FLAGGED"
        report["review"] = review
    if research_context:
        report["report_body"] = _generate_report_body(
            research_memory, research_spec, research_context, report, llm_client)
        errors = _validate_report(report)
        if errors:
            raise ReportParseError(
                f"Report with human body failed schema validation: {errors}")
    return report


def _drop_citation_unsupported_claims(report: dict, citation_audit_report: dict) -> dict:
    """Drop claims flagged by a CitationAuditReport from the grounded
    classification arrays (``verified_findings``, ``negative_results``).
    从接地分类数组（``verified_findings``、``negative_results``）中剔除
    CitationAuditReport 标记的论断。

    Mirrors the module's fabrication guard: a claim lacking citation
    support is not honestly reportable as verified/established, so it is
    removed rather than trusted.
    """
    unsupported = {
        item["claim"]
        for item in citation_audit_report.get("unsupported_claims", []) or []
    }
    if not unsupported:
        return report
    report["verified_findings"] = [
        claim for claim in report.get("verified_findings", []) if claim not in unsupported
    ]
    report["negative_results"] = [
        claim for claim in report.get("negative_results", []) if claim not in unsupported
    ]
    return report


# T8: cluster negative results by FAILURE MODE (borrowed from agents-cli eval-analyze) so the
# report surfaces WHY things failed, grouped, instead of a flat list. Deterministic, no LLM.
_FAILURE_CATEGORIES = [
    ("validation_and_leakage",
     re.compile(r"held.?out|validation|leakage|train[\s/_-]?test|overfit", re.I)),
    ("reproducibility",
     re.compile(r"\bseed\b|reproduc|non.?determin|not fixed|seed mismatch", re.I)),
    ("anti_cheat_violation",
     re.compile(r"forbidden|touched|smuggl|tamper|cheat|protected file", re.I)),
    ("degenerate_or_implausible_metric",
     re.compile(r"improbabl|degenerate|non.?finite|baseline\s*=?\s*0|no separation|implausible", re.I)),
    ("result_format_or_schema",
     re.compile(r"schema|parse|result format|format violation", re.I)),
    ("protocol_or_data_mismatch",
     re.compile(r"protocol|synthetic|mismatch|incomplete|budget|insufficient", re.I)),
]


def _cluster_failure_modes(negative_results: list) -> dict:
    """Cluster negative-result strings by failure MODE (validation/leakage, reproducibility,
    anti-cheat, degenerate-metric, schema, protocol/data) so the report shows WHY things failed,
    grouped. Returns {category: [matching strings]} for the categories that matched (empty when
    nothing matched -- e.g. plain domain negatives). A string may land in multiple categories."""
    clusters: dict = {}
    for nr in negative_results or []:
        s = str(nr)
        for name, pat in _FAILURE_CATEGORIES:
            if pat.search(s):
                clusters.setdefault(name, []).append(s)
    return clusters


_REVIEW_SYSTEM = (
    "You are a BLIND reviewer of a finished research report. You see ONLY the report's own "
    "fields (summary, verified_findings, negative_results), NOT the author's reasoning. Your "
    "single job: check whether the free-text SUMMARY over-claims. An over-claim is the summary "
    "asserting that the research CONFIRMED / ESTABLISHED / PROVED / SHOWED / DEMONSTRATED a "
    "positive result (an improvement, an effect, that a method works/wins) that is NOT listed "
    "in verified_findings. If verified_findings is empty, the summary must make NO such "
    "confirmed positive claim (honest framing like 'attempted to determine', 'was flagged "
    "suspicious', 'could not be verified', 'no verified findings' is fine and CLEAN). Reply "
    'with ONE JSON object: {"verdict": "CLEAN" | "FLAGGED", "findings": ["<over-claim>", ...]}.'
)


_SUMMARY_NUMBER = re.compile(r"-?\d+\.\d+")


def _untraceable_numbers(summary: str, research_memory: dict) -> list[str]:
    """Return the metric-like DECIMAL numbers in ``summary`` that do NOT appear anywhere in the
    grounded ResearchMemory (verbatim substring) -- an untraceable number in the prose is
    presumed fabricated. Only decimals are checked: bare integers (k=5, seeds, years, counts)
    are too noisy to flag. De-duplicated, order-preserving."""
    grounded = json.dumps(research_memory or {}, ensure_ascii=False)
    out: list[str] = []
    for n in _SUMMARY_NUMBER.findall(summary or ""):
        if n not in grounded and n.lstrip("-") not in grounded and n not in out:
            out.append(n)
    return out


def _blind_review(reviewer_llm: LLMClientLike, report: dict) -> dict:
    """Guaranteed, code-run BLIND review of the final report: flag a SUMMARY that asserts a
    confirmed positive result absent from ``verified_findings``. L0 (annotate-only, never
    blocks). FAIL-OPEN: any error -> ``{"verdict": "unreviewed", "findings": []}``."""
    payload = {"summary": report.get("summary"),
               "verified_findings": report.get("verified_findings"),
               "negative_results": report.get("negative_results")}
    try:
        result = reviewer_llm.chat(
            messages=[{"role": "system", "content": _REVIEW_SYSTEM},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format="json_object")
        obj = json.loads(extract_json(result.content or ""))
        if not isinstance(obj, dict):
            return {"verdict": "unreviewed", "findings": []}
        findings = [str(f) for f in (obj.get("findings") or []) if str(f).strip()]
        verdict = obj.get("verdict")
        if verdict not in ("CLEAN", "FLAGGED"):
            verdict = "FLAGGED" if findings else "CLEAN"
        return {"verdict": verdict, "findings": findings}
    except Exception:  # noqa: BLE001 -- annotate-only gate: a flaky reviewer never blocks
        return {"verdict": "unreviewed", "findings": []}


__all__ = [
    "ReportError",
    "ReportParseError",
    "ReportLLMError",
    "generate_report",
    "MAX_RETRIES",
]
