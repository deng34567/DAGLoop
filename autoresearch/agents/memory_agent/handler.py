"""
MemoryAgent EventBus handler: the thin transport wrapper around core.
MemoryAgent EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``memory.consolidation_requested`` (payload ``{evaluation_report_ref,
run_record_ref, research_memory_ref?, verification_report_ref?}``) -- reads
the EvaluationReport and RunRecord from the Spec Store by ref (provenance --
NEVER from the event payload); reads the prior ResearchMemory / a
VerificationReport too, IF their refs are present in the payload, else passes
``None``; runs ``core.consolidate`` (which returns a ``(research_memory,
routing_verdict)`` tuple); persists the resulting ResearchMemory
(kind="research_memory", key="v1"); and publishes ``memory.consolidated``
with its ref and the routing verdict.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅一个事件：memory.consolidation_requested（按 ref 从 Spec Store 读回
EvaluationReport、RunRecord，若 payload 中给出对应 ref 则一并读回既往
ResearchMemory / VerificationReport，否则传 None；跑一次
core.consolidate（返回 (research_memory, routing_verdict) 元组），持久化
ResearchMemory，发布 memory.consolidated）。core 的任何失败（或未预期错误）
都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch
from autoresearch.agents.method_implementation_agent.core import explicit_import_violation

import asyncio
from autoresearch.experiments.repo_improve.runtime_evidence import (
    can_revalidate_accepted_whole_result, can_revalidate_reconciliation_candidate, comparison_epsilon)
import json
import logging
import math
import re
import os
from typing import Any

from autoresearch.agents.memory_agent import core
from autoresearch.agents.memory_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store import refs
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_RESEARCH_MEMORY_KEY = "v1"

_GEN_ENV_FLAG = "AUTORESEARCH_GEN_EXPERIMENT"
_GEN_RESULT_KIND = "gen_experiment_result"
_GEN_CURRENT_RESULT_KEY = "current"
_TRUSTED_PHASE_DIAGNOSES = {
    "execution_failure", "baseline_drift", "candidate_contract_violation",
    "attempt_accounting_violation", "component_contract_retry",
}

_RECENT_COMPONENT_HISTORY_LIMIT = 20
_HISTORICAL_BEST_FIELDS = (
    "sequence", "component_id", "mode", "plan_version", "method_fingerprint", "round",
    "role", "status", "operator", "dependency_closure", "parent_checkpoint", "baseline_digest",
    "parent_checkpoints", "checkpoint", "diff_ref", "metric", "contract_passed",
    "acceptance_result",
    "mir_verdict", "verification", "run_ref", "changes_summary", "failure_reason",
)

# A shielded thread review may outlive cancellation of its delivery task.
_ACTIVE_COMPLETE_REVIEWS: dict[str, asyncio.Task] = {}


def _result_import_violation(result: dict) -> dict | None:
    baseline_diff = result.get("baseline_relative_diff")
    diff = (baseline_diff if isinstance(baseline_diff, str)
            and not result.get("baseline_diff_error") else str(result.get("diff") or ""))
    return explicit_import_violation(result.get("repository_contract"), diff)


def _formal_best_eligible(result: dict) -> bool:
    execution = result.get("component_execution") or {}
    # A whole-Method revalidation can contain a paired component adoption decision.  When
    # the task metric is the exact candidate scalar whose frozen direct-gain gate failed,
    # that checkpoint is an observation, not the retained Method selected by the decision.
    # Recheck old immutable receipts here so a result produced before the Runner guard cannot
    # remain the formal best during later consolidation.
    method_metric = result.get("method_metric")
    measurement = result.get("measurement_evidence") or {}
    attempt_id = str(measurement.get("attempt_id") or "")
    report = result.get("ablation_report")
    if (execution.get("mode") is None and isinstance(report, list)
            and isinstance(method_metric, (int, float)) and not isinstance(method_metric, bool)
            and math.isfinite(method_metric) and attempt_id.startswith("attempt-")):
        from autoresearch.experiments.repo_improve.search_memory import check_expectation
        decision = check_expectation(report)
        for item in report:
            if not isinstance(item, dict):
                continue
            actual = item.get("actual")
            evidence_ref = str((item.get("evidence") or {}).get("ref") or "")
            if (str(item.get("type") or "").strip().lower() in {"direct_gain", "gain"}
                    and str(item.get("component") or "") in decision["failing"]
                    and isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    and math.isfinite(actual) and actual == method_metric
                    and attempt_id in evidence_ref):
                return False
    if execution.get("mode") != "integration":
        return True
    if (result.get("candidate_admissibility") or {}).get("verdict") in {
            "violation", "insufficient_evidence"}:
        return False
    review = result.get("method_implementation_review") or {}
    contract = execution.get("attempt_contract_result") or {}
    return (not result.get("component_evidence_error")
            and contract.get("passed") is not False
            and execution.get("all_components_accepted") is True
            and review.get("verdict") == "faithful"
            and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
            and not _result_import_violation(result))


def _method_feedback_component_evidence(memory: dict) -> tuple[list[dict], list[dict]]:
    """Keep recent detail plus compact, older trustworthy best checkpoints."""
    history = [
        dict(item) for item in (memory.get("component_history") or [])
        if isinstance(item, dict)
    ]
    recent_start = max(0, len(history) - _RECENT_COMPONENT_HISTORY_LIMIT)
    recent = [
        {**item, "evidence_id": f"component_history[{index}]"}
        for index, item in enumerate(history[recent_start:], recent_start)
    ]
    older = history[:-_RECENT_COMPONENT_HISTORY_LIMIT]
    historical_best = []
    for index, item in enumerate(older):
        metric = item.get("metric")
        if not (
            item.get("became_best") is True
            and item.get("execution_succeeded") is True
            and item.get("contract_passed") is True
            and item.get("status") in {"accepted", "qualified", "improving"}
            and item.get("mir_verdict") in {"faithful", "partial"}
            and isinstance(metric, (int, float))
            and not isinstance(metric, bool)
            and math.isfinite(metric)
        ):
            continue
        historical_best.append({
            **{key: item.get(key) for key in _HISTORICAL_BEST_FIELDS
               if key != "baseline_digest" or item.get(key) is not None},
            "evidence_id": f"historical_best_component_evidence[{len(historical_best)}]",
            "source_history_index": index,
        })
    return recent, historical_best


def _independent_of_failed_result(candidate: dict, current: dict) -> bool:
    """A failed extension cannot revoke different, previously verified source."""
    if (current.get("component_execution") or {}).get("status") not in {
            "execution_failed", "implementation_failed"}:
        return True
    old_diff, new_diff = candidate.get("baseline_relative_diff"), current.get("baseline_relative_diff")
    return (bool(candidate.get("checkpoint")) and bool(current.get("checkpoint"))
            and candidate["checkpoint"] != current["checkpoint"]
            and candidate.get("run_record_snapshot_ref") != current.get("run_record_snapshot_ref")
            and not candidate.get("baseline_diff_error") and not current.get("baseline_diff_error")
            and isinstance(old_diff, str) and isinstance(new_diff, str) and old_diff != new_diff)


def _qualified_closeout_components(memory: dict, baseline_digest: str | None, *,
                                  store=None, incumbent=None, current=None) -> list[dict]:
    """Retain verified component bests, including indirect checks without a task metric."""
    if not baseline_digest:
        return []
    revalidated_source = None
    if (isinstance(current, dict) and current.get("baseline_digest") == baseline_digest
            and _formal_best_eligible(current) and can_revalidate_accepted_whole_result(current)):
        try:
            method_ref = current["method_ref"]
            method = store.get(method_ref)
            if (refs.decode(method_ref).kind == "method"
                    and refs.decode(method_ref).key == "snapshot-" + artifact_digest(method).split(":", 1)[1]):
                revalidated_source = method.get("reconciled_result_ref")
        except (KeyError, TypeError, ValueError):
            pass
    best = {}
    for index, item in enumerate(memory.get("component_history") or []):
        if (not isinstance(item, dict) or item.get("mode") != "component"
                or item.get("baseline_digest") != baseline_digest
                or item.get("became_best") is not True
                or item.get("execution_succeeded") is not True
                or item.get("contract_passed") is not True
                or item.get("verification_verdict") != "verified"
                or item.get("mir_verdict") != "faithful"
                or item.get("status") not in {"accepted", "qualified", "improving"}
                or not item.get("checkpoint") or not item.get("run_ref")):
            continue
        if incumbent is not None or current is not None:
            from autoresearch.spec_store.identity import result_for_run
            try:
                saved = result_for_run(store, item["run_ref"])
            except (KeyError, ValueError, TypeError):
                continue
            if (saved.get("checkpoint") != item["checkpoint"]
                    or saved.get("baseline_digest") != baseline_digest
                    or saved.get("accepted") is not True
                    or saved.get("component_evidence_error")
                    or _result_import_violation(saved)
                    or (saved.get("method_implementation_review") or {}).get("verdict") != "faithful"
                    or (current is not None and not _independent_of_failed_result(saved, current))):
                continue
            # Its exact source already received a qualified whole evaluation.
            # A noisier fresh metric must not nominate the old local scalar again.
            if revalidated_source and saved.get("candidate_snapshot_ref") == revalidated_source:
                continue
            if incumbent is not None:
                from autoresearch.experiments.repo_improve.runtime_evidence import (
                    can_revalidate_reconciliation_candidate)
                measurement = saved.get("measurement_evidence") or {}
                old_measurement = incumbent.get("measurement_evidence") or {}
                metric, best_metric = saved.get("method_metric"), incumbent.get("method_metric")
                # A bound local component can nominate its source for a fresh whole
                # evaluation. Missing whole-protocol evidence must still prevent
                # score promotion, but must not suppress this existing recovery route.
                local_source = ((measurement.get("protocol") or {}).get("kind") == "component_local"
                    and measurement.get("protocol_fingerprint") is None
                    and can_revalidate_reconciliation_candidate(saved, allow_qualified=True))
                if (incumbent.get("accepted") is not True or not _formal_best_eligible(incumbent)
                        or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                               or not math.isfinite(value) for value in (metric, best_metric))
                        or type(saved.get("lower_is_better")) is not bool
                        or any(saved.get(key) != incumbent.get(key) for key in (
                            "baseline_digest", "metric_name", "metric_unit", "lower_is_better"))
                        or measurement.get("valid") is not True or measurement.get("value") != metric
                        or not measurement.get("stdout_sha256")
                        or old_measurement.get("valid") is not True
                        or old_measurement.get("value") != best_metric
                        or not (local_source or (measurement.get("protocol_fingerprint")
                            and measurement["protocol_fingerprint"] == old_measurement.get("protocol_fingerprint")))):
                    continue
                gain = best_metric - metric if saved["lower_is_better"] else metric - best_metric
                if gain <= comparison_epsilon(saved, incumbent):
                    continue
        best[(item["component_id"], item.get("method_fingerprint"))] = {
            **{key: item.get(key) for key in _HISTORICAL_BEST_FIELDS},
            "evidence_id": f"component_history[{index}]",
        }
    return list(best.values())


def _qualified_integration_parent(store, gen: dict, memory: dict) -> dict | None:
    """Resolve comparable component evidence for the actual integration starting checkpoints."""
    from autoresearch.spec_store.identity import result_for_run
    from autoresearch.experiments.repo_improve.runtime_evidence import finite
    execution = gen.get("component_execution") or {}
    measurement = gen.get("measurement_evidence") or {}
    delta = gen.get("baseline_relative_diff")
    if (execution.get("mode") != "integration" or not _formal_best_eligible(gen)
            or execution.get("execution_succeeded") is not True or not gen.get("checkpoint")
            or measurement.get("valid") is not True
            or measurement.get("value") != gen.get("method_metric")
            or not measurement.get("stdout_sha256") or not measurement.get("protocol_fingerprint")
            or (gen.get("candidate_admissibility") or {}).get("verdict") != "pass"
            or not gen.get("baseline_digest") or gen.get("baseline_diff_error")
            or not isinstance(delta, str) or type(gen.get("lower_is_better")) is not bool):
        return None
    parents = execution.get("parent_checkpoints") or []
    candidates = []
    for item in memory.get("component_history") or []:
        if (not isinstance(item, dict) or item.get("mode") != "component"
                or item.get("checkpoint") not in parents
                or item.get("contract_passed") is not True
                or item.get("verification_verdict") != "verified"):
            continue
        try:
            saved = result_for_run(store, item.get("run_ref"))
            finite(gen.get("method_metric"))
            finite(saved.get("method_metric"))
        except (KeyError, TypeError, ValueError):
            continue
        old_measurement = saved.get("measurement_evidence") or {}
        old_execution = saved.get("component_execution") or {}
        contract = old_execution.get("attempt_contract_result") or old_execution.get("contract_result") or {}
        old_delta = saved.get("baseline_relative_diff")
        if (saved.get("accepted") is not True or saved.get("component_evidence_error")
                or old_execution.get("mode") != "component" or contract.get("passed") is not True
                or saved.get("checkpoint") != item.get("checkpoint")
                or old_measurement.get("valid") is not True
                or old_measurement.get("value") != saved.get("method_metric")
                or not old_measurement.get("stdout_sha256")
                or old_measurement.get("protocol_fingerprint") != measurement["protocol_fingerprint"]
                or any(saved.get(key) != gen.get(key) for key in (
                    "baseline_digest", "metric_name", "metric_unit", "lower_is_better"))
                or saved.get("baseline_diff_error") or not isinstance(old_delta, str)
                or old_delta == delta):
            continue
        candidates.append(saved)
    select = min if gen["lower_is_better"] else max
    return select(candidates, key=lambda result: result["method_metric"]) if candidates else None


def _method_review_trigger(
    gen: dict, memory: dict, diagnosis: str, incumbent: dict | None,
    baseline: dict | None = None,
    integration_parent: dict | None = None,
    closeout_requested: bool = False,
    user_closeout_requested: bool = False,
    closeout_components: list[dict] | None = None,
) -> str | None:
    """Select a measured Method question without treating a component as a complete win."""
    execution = gen.get("component_execution") or {}
    review = gen.get("method_implementation_review") or {}
    task_review = review.get("task_contract_review") or {}
    from autoresearch.llm.errors import is_resumable_provider_error
    error = str(gen.get("error") or "")
    infrastructure_failure = (
        error.startswith("clone failed:") or is_resumable_provider_error(error)
        or any(code in error for code in (
            "llm_provider_timeout", "llm_provider_content_filter", "llm_provider_rate_limit")))
    exhausted_extension = (execution.get("mode") == "component"
                           and execution.get("status") == "exhausted")
    partial_failure = (
        execution.get("mode") in {"component", "integration"}
        and execution.get("status") == "implementation_failed"
        and review.get("verdict") == "partial" and review.get("issues")
        and gen.get("method_ref") and review.get("method_ref") == gen.get("method_ref")
        and not infrastructure_failure)
    qualified = closeout_components
    if qualified is None:
        qualified = ([] if memory.get("best_result_ref") else
                     _qualified_closeout_components(memory, gen.get("baseline_digest")))
    if ((exhausted_extension or closeout_requested or user_closeout_requested)
            and not infrastructure_failure
            and execution.get("mode") in {None, "component", "integration"}
            and isinstance(baseline, dict) and baseline.get("reproduced") is True
            and gen.get("baseline_digest") == artifact_digest(baseline)
            and qualified
            and (closeout_components is not None or (
                not partial_failure and execution.get("status") != "execution_failed"))):
        if all(item.get("role") == "indirect_gain" for item in qualified):
            # Intermediate checks must not hide a measured implementation's source binding.
            retained_trigger = _method_review_trigger(
                gen, memory, diagnosis, incumbent, baseline, integration_parent,
                closeout_requested, user_closeout_requested, closeout_components=[])
            if retained_trigger in {"legal_off_method_gain", "component_method_reconciliation"}:
                return retained_trigger
        return "qualified_component_closeout"
    if (diagnosis == "component_contract_retry" and execution.get("status") == "execution_failed"
            and execution.get("execution_succeeded") is False
            and gen.get("component_execution_started") is not True
            and gen.get("diagnosis") in {"execution_failure", "attempt_accounting_violation"}):
        return None  # Let the existing retained-checkpoint decision repair or advance this refinement.
    if (execution.get("mode") in {"component", "integration"}
            and gen.get("diagnosis") in {"execution_failure", "attempt_accounting_violation"}
            and execution.get("status") == "execution_failed"
            and execution.get("execution_succeeded") is False):
        if not infrastructure_failure:
            return "implementation_failure"
    partial_trigger = "implementation_failure" if partial_failure else None
    if ((task_review.get("verdict") == "violation" or _result_import_violation(gen))
            and execution.get("mode") in {"component", "integration"}):
        return "mir_task_contract_violation"
    from autoresearch.experiments.repo_improve.runtime_evidence import (
        can_revalidate_partial_integration)
    if can_revalidate_partial_integration(gen):
        # A legal, officially measured integration can be useful even when it
        # implements only a strict subset of the frozen Method.  Keep its exact
        # checkpoint as source for Method reconciliation and a fresh official
        # run; do not promote the historical scalar as a formal best.
        return "component_method_reconciliation"
    if integration_parent is not None:
        metric, parent_metric = gen["method_metric"], integration_parent["method_metric"]
        loss = metric - parent_metric if gen["lower_is_better"] else parent_metric - metric
        if loss > comparison_epsilon(gen, integration_parent):
            return "complete_metric_regression"
    if (gen.get("accepted") is True and _formal_best_eligible(gen)
            and gen.get("checkpoint")
            and diagnosis in {"valid_improvement", "inconclusive"}
            and (not execution or (execution.get("mode") == "integration"
                                   and execution.get("all_components_accepted") is True))
            and memory.get("best_result_ref")):
        return "complete_result"
    metric = gen.get("method_metric")
    if (not isinstance(metric, (int, float)) or isinstance(metric, bool)
            or not math.isfinite(metric) or not gen.get("checkpoint")):
        return partial_trigger
    comparison = gen.get("repository_comparison_review") or {}
    if (task_review.get("verdict") != "compliant"
            or comparison.get("verdict") not in {None, "pass"}):
        return partial_trigger
    off_method = gen.get("off_method_observation") or {}
    baseline_incumbent = (
        not memory.get("best_result_ref") and isinstance(baseline, dict)
        and baseline.get("reproduced") is True
        and (baseline.get("measurement_evidence") or {}).get("valid") is True
        and baseline.get("metric_value") == gen.get("baseline_metric")
        and all(baseline.get(key) == gen.get(key) for key in (
            "metric_name", "metric_unit", "lower_is_better"))
    )
    accepted_incumbent = (
        isinstance(incumbent, dict) and incumbent.get("accepted") is True
        and incumbent.get("method_metric") == memory.get("best_metric")
        and all(incumbent.get(key) == gen.get(key) for key in (
            "metric_name", "metric_unit", "lower_is_better", "baseline_metric"))
    )
    comparison_target = baseline if baseline_incumbent else incumbent if accepted_incumbent else None
    measurement = gen.get("measurement_evidence") or {}
    admissibility = gen.get("candidate_admissibility") or {}
    contract = execution.get("attempt_contract_result") or execution.get("contract_result") or {}
    parent_metric = contract.get("parent_metric")
    minimum = comparison_epsilon(gen)
    # A legal, measured checkpoint that failed only component attribution or
    # parent binding is still valuable source code.  Route a genuinely better
    # one through Method reconciliation and fresh whole-candidate revalidation;
    # never promote its historical component-local score directly.
    if (diagnosis in {"method_description_reconciliation", "candidate_contract_violation",
                      "component_contract_retry", "component_expectation_failed"}
            # Source-only observations are retained during component execution.  Revisit
            # them only after a formal whole-result incumbent exists (or explicit closeout),
            # so an ahead-of-plan implementation cannot preempt the frozen ComponentPlan.
            and comparison_target is not None
            and (accepted_incumbent or closeout_requested or user_closeout_requested)
            and can_revalidate_reconciliation_candidate(gen)
            and all(gen.get(key) == comparison_target.get(key) for key in (
                "metric_name", "metric_unit", "lower_is_better", "baseline_metric"))
            and type(gen.get("lower_is_better")) is bool):
        target_metric = comparison_target.get(
            "method_metric", comparison_target.get("metric_value"))
        if (isinstance(target_metric, (int, float)) and not isinstance(target_metric, bool)
                and math.isfinite(target_metric)
                and (target_metric - metric > minimum if gen["lower_is_better"]
                     else metric - target_metric > minimum)):
            return "component_method_reconciliation"
    if (review.get("verdict") in {"drifted", "partial"}
            and diagnosis in {"method_description_reconciliation", "candidate_contract_violation",
                              "component_contract_retry"}
            and execution.get("mode") == "component" and execution.get("role") == "direct_gain"
            and execution.get("execution_succeeded") is True
            and not gen.get("component_evidence_error")
            and comparison.get("verdict") == "pass"
            and contract.get("passed") is True and contract.get("candidate_metric") == metric
            and (contract.get("parent_evidence") or {}).get("host_bound") is True
            and measurement.get("valid") is True and measurement.get("value") == metric
            and (measurement.get("protocol") or {}).get("kind") == "component_local"
            and measurement.get("stdout_sha256")
            and isinstance(measurement.get("attempt_id"), str)
            and measurement["attempt_id"].startswith("attempt-")
            and measurement["attempt_id"] == admissibility.get("attempt_id")
            and contract.get("measurement_evidence") == measurement
            and admissibility.get("verdict") in {"pass", "insufficient_evidence"}
            and isinstance(parent_metric, (int, float)) and not isinstance(parent_metric, bool)
            and math.isfinite(parent_metric)
            and isinstance(minimum, (int, float)) and not isinstance(minimum, bool)
            and math.isfinite(minimum) and minimum >= 0
            and type(gen.get("lower_is_better")) is bool
            and (parent_metric - metric > minimum if gen["lower_is_better"]
                 else metric - parent_metric > minimum)):
        return "component_method_reconciliation"
    if partial_trigger:
        return partial_trigger
    drifted_gain = (
        diagnosis in {"method_description_reconciliation", "candidate_contract_violation"}
        and review.get("verdict") == "drifted"
        and off_method.get("valid") is True
        and off_method.get("improved") is True
        and (off_method.get("admissibility") or {}).get("verdict") == "pass"
        and off_method.get("metric") == metric
    )
    partial_gain = (
        review.get("verdict") == "partial"
        and task_review.get("verdict") == "compliant"
        and execution.get("mode") == "integration"
        and execution.get("all_components_accepted") is True
        and execution.get("execution_succeeded") is True
        and admissibility.get("verdict") == "pass"
        and measurement.get("valid") is True
        and measurement.get("value") == metric
        and isinstance(measurement.get("attempt_id"), str)
        and measurement["attempt_id"].startswith("attempt-")
        and measurement.get("attempt_id") == admissibility.get("attempt_id")
    )
    if ((drifted_gain or partial_gain)
            and comparison.get("verdict") == "pass"
            and comparison_target is not None
            and measurement.get("protocol_fingerprint")
            and measurement.get("protocol_fingerprint")
                == (comparison_target.get("measurement_evidence") or {}).get("protocol_fingerprint")
            and type(gen.get("lower_is_better")) is bool):
        best = (baseline["metric_value"] if baseline_incumbent else memory["best_metric"])
        minimum = comparison_epsilon(gen)
        if (isinstance(best, (int, float)) and not isinstance(best, bool)
                and math.isfinite(best) and isinstance(minimum, (int, float))
                and not isinstance(minimum, bool) and math.isfinite(minimum)
                and minimum >= 0 and
                (best - metric > minimum if gen["lower_is_better"] else metric - best > minimum)):
            return "legal_off_method_gain"
    if review.get("verdict") not in {"faithful", "partial"}:
        return None
    contract = execution.get("attempt_contract_result") or execution.get("contract_result") or {}
    expectation = contract.get("acceptance_check_result") or {}
    if (execution.get("mode") == "component" and execution.get("role") == "direct_gain"
            and diagnosis in {"component_accepted", "component_progress"}
            and gen.get("accepted") is True and contract.get("passed") is True
            and expectation.get("passed") is False):
        return "cap_expectation_shortfall"
    if (execution.get("mode") == "integration"
            and execution.get("all_components_accepted") is True
            and diagnosis == "component_combination_retry"
            and execution.get("execution_succeeded") is True
            and gen.get("verdict") == "valid_not_best"
            and (gen.get("candidate_admissibility") or {}).get("verdict") == "pass"
            and isinstance(gen.get("baseline_metric"), (int, float))
            and not isinstance(gen["baseline_metric"], bool)
            and math.isfinite(gen["baseline_metric"])
            and type(gen.get("lower_is_better")) is bool):
        baseline = gen["baseline_metric"]
        if (metric > baseline if gen["lower_is_better"] else metric < baseline):
            return "complete_metric_regression"
        if (comparison_target is not None
                and measurement.get("valid") is True and measurement.get("value") == metric):
            target_metric = comparison_target.get("method_metric", comparison_target.get("metric_value"))
            gain = target_metric - metric if gen["lower_is_better"] else metric - target_metric
            if gain <= comparison_epsilon(gen, comparison_target):
                return "complete_verified_no_gain"
    if (execution.get("mode") == "component"
            and diagnosis == "component_progress"
            and gen.get("accepted") is True and contract.get("passed") is True
            and execution.get("became_best") is False):
        last_two = (memory.get("component_history") or [])[-2:]
        if (len(last_two) == 2 and all(
                item.get("mode") == "component"
                and item.get("verification_verdict") == "verified"
                and item.get("execution_succeeded") is True
                and item.get("contract_passed") is True
                and item.get("became_best") is False
                and isinstance(item.get("metric"), (int, float))
                and not isinstance(item["metric"], bool)
                and math.isfinite(item["metric"])
                for item in last_two)
                and all(last_two[0].get(key) == last_two[1].get(key)
                        for key in ("component_id", "plan_version", "method_fingerprint", "operator"))
                and last_two[0].get("component_id")
                and last_two[0].get("method_fingerprint")
                and last_two[0].get("operator")
                and last_two[0].get("diff_ref") and last_two[1].get("diff_ref")
                and last_two[0]["diff_ref"] != last_two[1]["diff_ref"]):
            return "repeated_verified_no_gain"
    return None


def _decision_retains_revalidation_source(decision: dict, result: dict) -> bool:
    """A source-only closeout must actually carry its exact checkpoint forward."""
    checkpoint = str(result.get("checkpoint") or "")
    token = checkpoint.rsplit("/", 1)[-1]
    retained = {
        key: decision.get(key)
        for key in ("retain", "modify", "add", "reuse", "validation_plan",
                    "expected_observation")
    }
    text = json.dumps(retained, ensure_ascii=False, sort_keys=True)
    return bool(
        checkpoint and token
        and decision.get("route") == "revise_method"
        and decision.get("requires_core_reconstruction") is not True
        and token in text
    )


def _classify_setting_evidence(items: Any) -> list[str]:
    """Return independently verified setting mismatches; ignore free-form self-reports."""
    if not isinstance(items, list):
        return []
    mismatches = []
    for item in items:
        if not isinstance(item, dict):
            continue
        setting = str(item.get("setting") or "").strip()
        actual = item.get("actual")
        expected = item.get("expected")
        evidence = item.get("evidence")
        if (not setting or actual is None or expected is None
                or not isinstance(evidence, dict)
                or not str(evidence.get("source") or "").strip()
                or not str(evidence.get("ref") or "").strip()):
            continue
        if actual != expected:
            mismatches.append(
                f"{setting}: observed {actual!r}, expected {expected!r} ({evidence!r})")
    return mismatches


def _has_bound_failed_quality_gate(gen: dict) -> bool:
    """A measured official rejection requests diagnosis, never result acceptance."""
    execution = gen.get("component_execution") or {}
    evidence = gen.get("measurement_evidence") or {}
    admissibility = gen.get("candidate_admissibility") or {}
    attempt_id = str(gen.get("component_attempt_id") or "")
    metric = gen.get("method_metric")
    return bool(
        execution.get("mode") == "component"
        and execution.get("status") in {"execution_failed", "contract_failed"}
        and not gen.get("error")
        and (gen.get("repository_comparison_review") or {}).get("verdict") in {None, "pass"}
        and isinstance(metric, (int, float)) and not isinstance(metric, bool)
        and math.isfinite(metric) and evidence.get("value") == metric
        and attempt_id.startswith("attempt-")
        and evidence.get("attempt_id") == admissibility.get("attempt_id") == attempt_id
        and re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("stdout_sha256") or ""))
        and admissibility.get("verdict") == "fail" and evidence.get("valid") is False
        and isinstance(evidence.get("gates"), dict)
        and any(passed is False for passed in evidence["gates"].values())
    )


def _classify_component_evidence(gen: dict) -> tuple[str | None, list[str]]:
    """Recompute check-expectation routing from structured measurements on the trusted host."""
    review = gen.get("method_implementation_review") or {}
    revision = review.get("plan_revision") or {}
    if (revision and gen.get("diagnosis") == "component_expectation_failed"
            and gen.get("method_ref") and review.get("method_ref") == gen["method_ref"]):
        # Runner already grounded the implicated contract and source quote.
        # Revision is not result acceptance or scientific falsification.
        return "component_expectation_failed", [
            revision["component_id"], revision["plan_quote"], revision["reason"],
            *revision["evidence"]]
    comparison = gen.get("repository_comparison_review") or {}
    if comparison.get("verdict") == "baseline_invalid":
        return "baseline_drift", list(gen.get("diagnosis_evidence") or comparison.get("issues") or [])
    method_metric = gen.get("method_metric")
    has_finite_metric = (
        isinstance(method_metric, (int, float))
        and not isinstance(method_metric, bool)
        and math.isfinite(float(method_metric))
    )
    component_execution = gen.get("component_execution")
    measurement = gen.get("measurement_evidence") or {}
    if (not component_execution and has_finite_metric
            and gen.get("accepted") is False and gen.get("verdict") == "valid_not_best"
            and gen.get("method_ref") and review.get("method_ref") == gen["method_ref"]
            and review.get("verdict") == "partial" and review.get("issues")
            and not review.get("review_error") and not revision
            and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
            and (gen.get("candidate_admissibility") or {}).get("verdict") == "pass"
            and measurement.get("valid") is True and measurement.get("value") == method_metric
            and measurement.get("stdout_sha256") and measurement.get("protocol_fingerprint")):
        return "execution_failure", [
            "Whole-result measurement is preserved, but the bound Method implementation needs repair.",
            *[str(item) for item in review["issues"]],
        ]
    preflight_failure = (
        not has_finite_metric
        and gen.get("accepted") is not True
        and gen.get("verdict") in {"invalid", "failed", "fail"}
    )
    if isinstance(component_execution, dict):
        mode = component_execution.get("mode")
        status = str(component_execution.get("status") or "")
        component_id = str(component_execution.get("component_id") or "unknown component")
        contract = component_execution.get("contract_result") or {}
        review = gen.get("method_implementation_review") or {}
        if status == "infrastructure_exhausted":
            return "component_execution_blocked", [
                f"{component_id}: infrastructure retry budget exhausted; no algorithmic counterexample; retain checkpoints for recovery"]
        if (mode == "component" and status == "execution_failed"
                and gen.get("diagnosis") == "execution_failure" and not has_finite_metric
                and component_execution.get("execution_succeeded") is False
                and type(component_execution.get("infrastructure_attempts")) is int
                and component_execution["infrastructure_attempts"] > 0
                and gen.get("component_execution_started") is not True
                and not gen.get("component_attempt_id")
                and not gen.get("component_failed_attempt_id")
                and (str(gen.get("error") or "").startswith("clone failed:")
                     or any(code in str(gen.get("error") or "") for code in (
                         "llm_provider_timeout", "llm_provider_content_filter")))):
            # Failure before cloning/execution does not refute the component contract.
            return "execution_failure", [str(gen["error"])]
        if (mode == "integration" and preflight_failure
                and component_execution.get("execution_succeeded") is False
                and gen.get("component_execution_started") is not True
                and not gen.get("component_attempt_id")
                and not gen.get("component_failed_attempt_id")
                and str(gen.get("error") or "").startswith("clone failed:")):
            # No integrated candidate ran, so this cannot diagnose component interaction.
            return "execution_failure", [str(gen["error"])]
        if _has_bound_failed_quality_gate(gen):
            retained = component_execution.get("selected_contract_result") or {}
            qualified = component_execution.get("qualified_attempts")
            if (mode == "component" and component_execution.get("role") == "direct_gain"
                    and component_execution.get("operator") in {"improve", "debug"}
                    and type(qualified) is int and qualified > 0
                    and component_execution.get("checkpoint")
                    and retained.get("passed") is True
                    and (retained.get("measurement_evidence") or {}).get("valid") is True):
                # This variant failed, but it does not refute the retained valid component.
                # The existing bounded choice may advance its best or try another useful edit.
                return "component_progress", [
                    f"{component_id}: refinement failed the quality/correctness gate; "
                    "retain the previously qualified checkpoint and reassess whether another "
                    "bounded refinement is worthwhile. The failed variant remains invalid."]
            return "component_expectation_failed", [
                component_id,
                "Bound official evaluation completed but rejected the quality/correctness gate; "
                "review this component's implementation and design before repeating it. "
                "The result remains invalid; this does not establish whole-method falsification."]
        if (mode == "component" and status == "execution_failed"
                and component_execution.get("operator") in {"improve", "debug"}
                and component_execution.get("qualified_attempts", 0) > 0
                and component_execution.get("checkpoint")
                and component_execution.get("execution_succeeded") is False
                and gen.get("component_execution_started") is not True
                and not gen.get("component_attempt_id")
                and not gen.get("component_failed_attempt_id")):
            return "component_contract_retry", [
                f"{component_id}: refinement was not executed; reassess its premise from the "
                "retained qualified checkpoint, without treating it as a measured component failure"]
        if (mode == "integration" and gen.get("diagnosis") == "candidate_contract_violation"
                and any("non-permitted file" in str(issue).lower()
                        or "outside the permitted edit scope" in str(issue).lower()
                        for issue in review.get("issues", []))):
            # The same MIR cannot call an extra edited file non-permitted and then use its
            # separately generated task-contract verdict to promote the candidate as legal.
            return "candidate_contract_violation", list(
                gen.get("diagnosis_evidence") or review.get("issues") or [])
        if mode == "integration" and gen.get("component_evidence_error"):
            return "component_combination_retry", [
                "Integrated observation retained, but its exact parent-control evidence is invalid: "
                + str(gen["component_evidence_error"])]
        # Integration can discover a legal alternative too. Preserve its evidence for
        # formal revision before the generic composition retry can consume it.
        if mode == "integration" and review.get("verdict") == "drifted":
            off_method = gen.get("off_method_observation") or {}
            metric = gen.get("method_metric")
            if (off_method.get("valid") is True
                    and isinstance(metric, (int, float)) and not isinstance(metric, bool)
                    and math.isfinite(metric)
                    and (review.get("task_contract_review") or {}).get("verdict")
                        not in {"violation", "insufficient_evidence"}
                    and comparison.get("verdict") in {None, "pass"}):
                return "method_description_reconciliation", [
                    f"{component_id}: legal integrated alternative retained; formally revise the Method"]
        # A failed probe can be labelled drifted solely because its planned acceptance
        # predicate cannot hold. Keep it invalid, but allow grounded CHECK-EXPECTATION
        # evidence to request a plan repair rather than another identical DEBUG attempt.
        drifted_failed_probe = (
            status == "drifted" and gen.get("method_metric") is None
            and component_execution.get("execution_succeeded") is True
            and isinstance(contract, dict) and contract.get("passed") is False
            and isinstance(review, dict)
            and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
            and bool(gen.get("ablation_report"))
        )
        # The ledger replaces the last attempt's status with exhausted at its cap.
        # Preserve a legal measured alternative's revision path even on that attempt;
        # this requests a formal revision, never acceptance or another component retry.
        if mode == "component" and (status == "drifted" or (
                status == "exhausted" and review.get("verdict") == "drifted"
                and component_execution.get("execution_succeeded") is True)):
            off_method = gen.get("off_method_observation")
            premature_implementation = any(
                isinstance(item, dict) and item.get("status") == "prematurely_implemented"
                for item in (review.get("boundary_reviews") or []))
            # A legal indirect probe can establish its intermediate property without
            # producing the task's primary metric. Revise its design, not its legality;
            # this does not accept the component or promote a performance result.
            evidence = contract.get("evidence") or {}
            attempt_id = str(gen.get("component_attempt_id") or "")
            legal_indirect_probe = (
                component_execution.get("role") == "indirect_gain"
                and component_execution.get("execution_succeeded") is True
                and contract.get("measurement") == "indirect_gain_check"
                and contract.get("passed") is True
                and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
                and evidence.get("source") == "runtime_attempt"
                and attempt_id.startswith("attempt-")
                # Match the Runner's evidence-reference grammar, including annotations,
                # but reject references to more than one distinct attempt.
                and set(re.findall(r"\battempt-[A-Za-z0-9-]+\b",
                                   str(evidence.get("ref") or ""))) == {attempt_id}
            )
            if not premature_implementation and (legal_indirect_probe or (
                    isinstance(off_method, dict) and off_method.get("valid") is True)):
                return "method_description_reconciliation", [
                    f"{component_id}: legal off-method observation retained; formally revise the implicated component before replanning",
                    *[str(issue) for issue in review.get("issues", []) if issue]]
            if status == "drifted" and not drifted_failed_probe:
                return "component_contract_retry", [
                    f"{component_id}: repair premature non-target implementation within the frozen plan"
                    if premature_implementation else
                    f"{component_id}: drifted observation failed or lacked the host-owned task-contract review"]
        if mode == "component" and status == "accepted":
            return "component_accepted", [
                f"{component_id}: {contract.get('reason') or 'frozen component contract passed'}"]
        if mode == "component" and status in {"qualified", "improving"}:
            return "component_progress", [
                f"{component_id}: contract passed; continue IMPROVE from its best checkpoint"]
        if mode == "component" and status == "exhausted":
            measured_contract = component_execution.get("attempt_contract_result") or contract
            valid_negative = (
                component_execution.get("execution_succeeded") is True
                and review.get("verdict") in {"faithful", "partial"}
                and (review.get("task_contract_review") or {}).get("verdict")
                    not in {"violation", "insufficient_evidence"}
                and measured_contract.get("measurement") in {
                    "signed_metric_improvement", "indirect_gain_check"}
                and measured_contract.get("passed") is False
            )
            if valid_negative:
                return "component_expectation_failed", [
                    f"{component_id}: measured component expectation failed within the attempt budget"]
            return "component_execution_blocked", [
                f"{component_id}: attempt budget exhausted without an accepted checkpoint; "
                "the final attempt did not establish a valid negative measurement"]
        # A failed component may carry host-grounded CHECK-EXPECTATION evidence proving that the
        # frozen component contract itself is wrong.  Let that evidence reach the preflight
        # classifier below instead of treating every such failure as another implementation retry.
        # Only defer when the diagnostic classifier below will actually run.
        # An ordinary measured attempt must retain its host component failure,
        # even when its report claims a gain against a different baseline.
        has_preflight_evidence = bool(gen.get("ablation_report")) and (
            gen.get("operator") == "check-expectation" or preflight_failure)
        if mode == "component" and status and not (
                (status in {"execution_failed", "implementation_failed", "contract_failed"}
                 or drifted_failed_probe)
                and has_preflight_evidence):
            return "component_contract_retry", [
                f"{component_id}: {contract.get('reason') or status}"]
        if (mode == "integration"
                and component_execution.get("all_components_accepted") is True
                and gen.get("accepted") is not True
                and gen.get("verdict") not in ("new_best", "pass", "verified")):
            return "component_combination_retry", [
                "all components passed individually, but the integrated method did not meet the frozen official evaluator expectation; revise only the composition, interfaces, or ordering"]
    # A confirmed component can be disproved during implementation preflight (for example, its
    # assumed baseline gap is absent or its required transformation violates repository semantics).
    # Treat that structured negative evidence like CHECK-EXPECTATION only before any candidate
    # metric exists. Completed runs remain governed by their measured result unless the dedicated
    # check-expectation operator was selected.
    if gen.get("operator") != "check-expectation" and not preflight_failure:
        return None, []
    report = gen.get("ablation_report")
    try:
        if isinstance(report, str):
            report = json.loads(report)
    except json.JSONDecodeError:
        return None, []
    if isinstance(report, dict):
        report = ([report] if report.get("component")
                  else report.get("components") or report.get("report"))
    if not isinstance(report, list) or not report:
        return None, []
    if (isinstance(component_execution, dict)
            and component_execution.get("mode") == "component"
            and component_execution.get("execution_succeeded") is False):
        # Crashes cannot establish a measured zero. A host-bound exit-1 check
        # can still report an unmet property; retain only its exact component/ID.
        failed_check_id = str(gen.get("component_failed_attempt_id") or "")
        bound_failed_check = bool(
            status == "contract_failed"
            and contract.get("measurement") == "indirect_gain_check"
            and contract.get("passed") is False
            and failed_check_id.startswith("attempt-"))
        report = [item for item in report if not (
            isinstance(item, dict)
            and isinstance(item.get("evidence"), dict)
            and item["evidence"].get("source") == "runtime_attempt"
            and not (bound_failed_check and item.get("component") == component_id
                     and set(re.findall(r"\battempt-[A-Za-z0-9-]+\b",
                                        str(item["evidence"].get("ref") or "")))
                         == {failed_check_id}))]
        if not report:
            return "component_contract_retry", [
                f"{component_id}: execution failed; runtime observations cannot establish component expectation failure"]
    from autoresearch.experiments.repo_improve.search_memory import check_expectation
    decision = check_expectation(report)
    gain_components = decision.get("gain_components") or []
    # A preflight can expose a repairable bad component, but if every component whose declared
    # purpose is metric gain is unavailable, there is no gain mechanism left to revise.  That is a
    # whole-method falsification and must use the large method-swap loop.
    if (preflight_failure and len(gain_components) > 1
            and all(component in decision["failing"] for component in gain_components)):
        return "method_hypothesis_falsified", list(gain_components)
    if decision["failing"]:
        return "component_expectation_failed", list(decision["failing"])
    if (isinstance(component_execution, dict)
            and component_execution.get("mode") == "component"
            and status in {"execution_failed", "implementation_failed", "contract_failed"}):
        # Passing local probes do not erase missing implementation/measurement evidence,
        # nor let the generic memory judgment declare the entire method falsified.
        return "component_contract_retry", [
            f"{component_id}: no failed component expectation was established; "
            "repair the current component's execution or evidence before proceeding"]
    # A local component probe (or a missing final scalar) cannot establish that
    # the assembled Method failed to improve, even when every reported probe passed.
    if (has_finite_metric
            and not (isinstance(component_execution, dict)
                     and component_execution.get("mode") == "component")
            and gain_components
            and all(component in decision.get("met", []) for component in gain_components)
            and gen.get("verdict") not in ("new_best", "pass", "verified")):
        return "method_hypothesis_falsified", [
            "all measured gain components met their contracts but the end metric did not improve"]
    return None, []


def _append_component_history(
    prior: dict, gen: dict, run_record_ref: str,
    verification_report: dict | None,
) -> list[dict]:
    """Append one host-derived component/integration attempt without asking the LLM."""
    history = [
        dict(item) for item in (prior.get("component_history") or [])
        if isinstance(item, dict)
    ]
    execution = gen.get("component_execution")
    if not isinstance(execution, dict) or execution.get("mode") not in {
        "component", "integration",
    }:
        return history
    attempt_identity = (
        run_record_ref,
        str(execution.get("method_fingerprint") or ""),
        str(execution.get("component_id") or "__integration__"),
        str(execution.get("mode") or ""),
        max(1, int(execution.get("round") or 1)),
        max(0, int(execution.get("attempts") or 0)),
        str(execution.get("operator") or gen.get("operator") or ""),
    )
    if any((
        item.get("run_ref"),
        str(item.get("method_fingerprint") or ""),
        str(item.get("component_id") or "__integration__"),
        str(item.get("mode") or ""),
        max(1, int(item.get("round") or 1)),
        max(0, int(item.get("attempt") or 0)),
        str(item.get("operator") or ""),
    ) == attempt_identity for item in history):
        return history
    contract = execution.get("attempt_contract_result") or execution.get("contract_result") or {}
    if not isinstance(contract, dict):
        contract = {}
    review = gen.get("method_implementation_review") or {}
    reasons = gen.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [reasons]
    verification = str((verification_report or {}).get("reasoning") or "").strip()
    metric = gen.get("method_metric")
    if not isinstance(metric, (int, float)) or isinstance(metric, bool) or not math.isfinite(metric):
        metric = None
    # The task metric stays in metric; the component check remains in acceptance_result.
    execution_succeeded = execution.get("execution_succeeded")
    if not isinstance(execution_succeeded, bool):
        execution_succeeded = metric is not None and str(execution.get("status") or "") not in {
            "execution_failed", "implementation_failed", "checkpoint_missing", "drifted",
        }
    entry = {
        "sequence": len(history) + 1,
        "component_id": str(execution.get("component_id") or "__integration__"),
        "mode": str(execution.get("mode")),
        "plan_version": max(1, int(execution.get("plan_version") or 1)),
        "method_fingerprint": str(execution.get("method_fingerprint") or ""),
        "baseline_digest": gen.get("baseline_digest"),
        "round": max(1, int(execution.get("round") or 1)),
        "role": str(execution.get("role") or ""),
        "status": str(execution.get("status") or gen.get("verdict") or "unknown"),
        "attempt": max(0, int(execution.get("attempts") or 0)),
        "operator": str(execution.get("operator") or gen.get("operator") or ""),
        "execution_succeeded": execution_succeeded,
        "dependency_closure": [
            str(item) for item in (execution.get("dependency_closure") or []) if str(item)
        ],
        "parent_checkpoint": str(execution.get("parent_checkpoint") or ""),
        "parent_checkpoints": [
            str(item) for item in (execution.get("parent_checkpoints") or []) if str(item)
        ],
        "checkpoint": str(execution.get("attempt_checkpoint")
                          if "attempt_checkpoint" in execution else
                          (gen.get("checkpoint") or gen.get("repo_path")
                           or execution.get("checkpoint") or "")),
        "diff_ref": str(execution.get("attempt_diff_ref")
                        if "attempt_diff_ref" in execution else
                        (execution.get("diff_ref") or "")),
        "selected_checkpoint": str(execution.get("checkpoint") or ""),
        "selected_diff_ref": str(execution.get("diff_ref") or ""),
        "metric": metric,
        "acceptance_result": contract,
        "contract_passed": (
            contract.get("passed") if isinstance(contract.get("passed"), bool) else None
        ),
        "became_best": bool(execution.get("became_best")),
        "mir_verdict": str(review.get("verdict") or ""),
        "verification": verification[:1200],
        "verification_verdict": str((verification_report or {}).get("verdict") or ""),
        "run_ref": run_record_ref,
        "changes_summary": str(gen.get("changes_summary") or "")[:1200],
        "failure_reason": "; ".join(str(item) for item in reasons if item)[:1200],
    }
    history.append(entry)
    return history


class MemoryAgentHandler:
    """Wires core.consolidate to a Spec Store + LLM + EventBus client.
    把 core.consolidate 接到 Spec Store + LLM + EventBus 客户端。"""

    def __init__(
        self,
        *,
        spec_store: Any,
        llm_client: Any,
        config_root: Any | None = None,
    ) -> None:
        self.spec_store = spec_store
        self.llm_client = llm_client
        self.config = ConfigLoader(config_root)
        self._request_inputs = {}

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler's event on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的事件。"""
        client.subscribe(
            "memory.consolidation_requested",
            self._make_eventbus_handler(client, self.handle_consolidation_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- memory.consolidation_requested ----

    async def handle_consolidation_requested(self, client: Any, env: Any) -> None:
        """Resolve the EvaluationReport + RunRecord (+ optional prior
        ResearchMemory + VerificationReport) from the store, run one
        consolidate, persist the ResearchMemory, publish memory.consolidated.
        从 store 解析 EvaluationReport + RunRecord（+ 可选既往
        ResearchMemory + VerificationReport），跑一次 consolidate，持久化
        ResearchMemory，发布 memory.consolidated。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        applied_feedback_ref = payload.get("applied_method_feedback_ref")
        def feedback_already_applied(ref):
            if not ref or not applied_feedback_ref:
                return False
            if ref == applied_feedback_ref:
                return True
            try:
                return self.spec_store.get(ref) == self.spec_store.get(applied_feedback_ref)
            except KeyError:
                return False
        self._request_inputs[(session_id, causation_id)] = dict(payload)
        receipt_ref = refs.encode("audit", session_id, "memory-request-" + str(causation_id))
        try:
            receipt = self.spec_store.get(receipt_ref)
        except KeyError:
            receipt = None
        if receipt:
            if receipt["request_digest"] != artifact_digest(payload):
                raise ValueError("memory request identity changed")
            await client.publish(event_type="memory.consolidated", session_id=session_id,
                                 payload=receipt["payload"], causation_id=causation_id)
            return
        structured_result = self._has_current_gen_result(session_id)

        try:
            theory_method_ref = payload.get("theory_method_ref")
            interpretation_ref = payload.get("spec_interpretation_ref")
            interpretation = (
                self.spec_store.get(interpretation_ref) if interpretation_ref else None)
            validation_study = (
                interpretation.get("validation_study")
                if isinstance(interpretation, dict) else None)
            if (isinstance(validation_study, dict)
                    and payload.get("run_record_ref")):
                method = (
                    self.spec_store.get(theory_method_ref) if theory_method_ref else {})
                problem_model_ref = payload.get("problem_model_ref")
                problem_model = (
                    self.spec_store.get(problem_model_ref) if problem_model_ref else None)
                prior_ref = payload.get("research_memory_ref")
                prior = self.spec_store.get(prior_ref) if prior_ref else None
                run_record = self.spec_store.get(payload["run_record_ref"])
                verification_ref = payload.get("verification_report_ref")
                verification = (
                    self.spec_store.get(verification_ref) if verification_ref else None)
                try:
                    validation_result = self.spec_store.get(refs.encode(
                        "validation_study_result", session_id, "v1"))
                except KeyError:
                    validation_result = None
                research_memory = self._consolidate_validation(
                    method=method,
                    problem_model=problem_model,
                    interpretation=interpretation,
                    run_record_ref=payload["run_record_ref"],
                    run_record=run_record,
                    validation_result=validation_result,
                    verification_report=verification,
                    prior_research_memory=prior,
                )
                await self._publish_consolidated(
                    client, session_id, causation_id, research_memory, "report",
                    diagnosis=research_memory["diagnosis"],
                    diagnosis_evidence=list(research_memory["diagnosis_evidence"]),
                )
                return
            if theory_method_ref:
                method = self.spec_store.get(theory_method_ref)
                problem_model_ref = payload.get("problem_model_ref")
                problem_model = (
                    self.spec_store.get(problem_model_ref) if problem_model_ref else None)
                prior_ref = payload.get("research_memory_ref")
                prior = self.spec_store.get(prior_ref) if prior_ref else None
                research_memory = self._consolidate_theory(
                    method=method,
                    problem_model=problem_model,
                    interpretation=interpretation,
                    prior_research_memory=prior,
                )
                await self._publish_consolidated(
                    client, session_id, causation_id, research_memory, "report",
                    diagnosis="inconclusive",
                    diagnosis_evidence=[
                        "Theory method passed method deliberation, evidence audit and "
                        "novelty audit; theorem correctness still requires proof review."
                    ],
                )
                return
            if (payload.get("failure_context")
                    or (not payload.get("evaluation_report_ref")
                        and (not payload.get("run_record_ref")
                             or not (self._gen_branch() or structured_result)))):
                prior_ref = payload.get("research_memory_ref")
                prior = self.spec_store.get(prior_ref) if prior_ref else None
                run_ref = payload.get("run_record_ref")
                baseline_ref = payload.get("baseline_result_ref")
                run_record = self.spec_store.get(run_ref) if run_ref else None
                baseline = self.spec_store.get(baseline_ref) if baseline_ref else None
                baseline_attempt_ref = payload.get("baseline_attempt_ref")
                baseline_attempt = (self.spec_store.get(baseline_attempt_ref)
                                    if baseline_attempt_ref else baseline)
                evaluation_ref = payload.get("evaluation_report_ref")
                evaluation = self.spec_store.get(evaluation_ref) if evaluation_ref else None
                gen_result = None
                if run_ref:
                    from autoresearch.spec_store.identity import result_for_run
                    try:
                        gen_result = result_for_run(self.spec_store, run_ref)
                    except KeyError:
                        pass
                research_memory = self._consolidate_failure(
                    run_record=run_record,
                    baseline_result=baseline,
                    failure_context=payload.get("failure_context"),
                    prior_research_memory=prior,
                    evaluation_report=evaluation,
                    gen_result=gen_result,
                    baseline_attempt=baseline_attempt,
                )
                feedback_ref = None
                if gen_result:
                    verification_ref = payload.get("verification_report_ref")
                    verification = (self.spec_store.get(verification_ref)
                                    if verification_ref else None)
                    feedback_ref = await self._review_retained_off_method(
                        session_id, run_ref, research_memory, baseline_ref,
                        closeout_requested=bool(payload.get("qualified_component_closeout_requested")))
                    if feedback_already_applied(feedback_ref):
                        feedback_ref = None
                    if not feedback_ref:
                        feedback_ref = await self._review_complete_result(
                            session_id, run_ref, research_memory, verification,
                            research_memory["diagnosis"], baseline_ref,
                            review_method_ref=payload.get("method_ref"),
                            closeout_requested=bool(payload.get("qualified_component_closeout_requested")))
                if feedback_already_applied(feedback_ref):
                    feedback_ref = None
                if feedback_ref:
                    research_memory["diagnosis"] = "method_improvement_proposed"
                    research_memory["diagnosis_evidence"] = [feedback_ref]
                await self._publish_consolidated(
                    client, session_id, causation_id, research_memory,
                    "need_experiment" if feedback_ref else "report",
                    diagnosis=research_memory["diagnosis"],
                    diagnosis_evidence=list(research_memory["diagnosis_evidence"]),
                )
                return
            if (self._gen_branch() or structured_result) and payload.get("run_record_ref"):
                prior_rm_ref = payload.get("research_memory_ref")
                prior_rm = self.spec_store.get(prior_rm_ref) if prior_rm_ref else None
                verification_ref = payload.get("verification_report_ref")
                verification = (self.spec_store.get(verification_ref)
                                if verification_ref else None)
                research_memory, routing_verdict, diagnosis, diagnosis_evidence = self._consolidate_gen(
                    payload.get("run_record_ref"),
                    payload.get("evaluation_report_ref"),
                    prior_research_memory=prior_rm,
                    verification_report=verification,
                )
                closing_component_advanced = False
                applied_ref = payload.get("applied_method_feedback_ref")
                if applied_ref and diagnosis in {"component_progress", "component_accepted"}:
                    applied = self.spec_store.get(applied_ref)
                    try:
                        closing = self.spec_store.get(refs.encode(
                            "human_idea_seed", session_id, "closeout-v1"))
                    except KeyError:
                        closing = {}
                    if (closing.get("closeout") is True
                            and applied.get("status") == "completed"
                            and (applied.get("user_closeout") or {}).get("closeout") is True):
                        from autoresearch.spec_store.identity import result_for_run
                        current = result_for_run(self.spec_store, payload["run_record_ref"])
                        if ((current.get("method_implementation_review") or {}).get("verdict") == "faithful"
                                and (current.get("component_execution") or {}).get("status")
                                in {"qualified", "improving", "accepted"}):
                            closing_component_advanced = await self._decide_efficient_component(
                                session_id, payload["run_record_ref"], research_memory,
                                verification, payload.get("baseline_result_ref")) is True
                try:
                    feedback_ref = None if closing_component_advanced else await self._review_retained_off_method(
                        session_id, payload["run_record_ref"], research_memory,
                        payload.get("baseline_result_ref"),
                        closeout_requested=bool(payload.get("qualified_component_closeout_requested")))
                    if feedback_already_applied(feedback_ref):
                        feedback_ref = None
                    if not feedback_ref and not closing_component_advanced:
                        feedback_ref = await self._review_complete_result(
                            session_id, payload["run_record_ref"], research_memory, verification,
                            diagnosis, payload.get("baseline_result_ref"),
                            review_method_ref=payload.get("method_ref"),
                            closeout_requested=bool(payload.get("qualified_component_closeout_requested")))
                except Exception:  # retain the verified result while exposing the failed decision
                    logger.exception("complete-method review unavailable for %s", session_id)
                    raise
                if feedback_already_applied(feedback_ref):
                    feedback_ref = None
                if feedback_ref:
                    diagnosis = "method_improvement_proposed"
                    diagnosis_evidence = [feedback_ref]
                    research_memory["diagnosis"] = diagnosis
                    research_memory["diagnosis_evidence"] = diagnosis_evidence
                    routing_verdict = "need_experiment"
                elif diagnosis in {"component_progress", "component_accepted",
                                   "component_contract_retry"}:
                    try:
                        advanced = await self._decide_efficient_component(
                            session_id, payload["run_record_ref"], research_memory,
                            verification, payload.get("baseline_result_ref"))
                        if diagnosis == "component_contract_retry" and advanced is True:
                            # The failed refinement remains invalid. The retained qualified
                            # checkpoint was accepted, so planning must read that ledger choice.
                            diagnosis = "component_accepted"
                            diagnosis_evidence = diagnosis_evidence + [
                                "Retained qualified checkpoint accepted; failed refinement "
                                "preserved. Proceed to the next component or whole evaluation."]
                            research_memory["diagnosis"] = diagnosis
                            research_memory["diagnosis_evidence"] = diagnosis_evidence
                            routing_verdict = "need_experiment"
                    except Exception:
                        logger.exception("qualified-component choice unavailable for %s", session_id)
                        raise
                await self._publish_consolidated(
                    client, session_id, causation_id, research_memory, routing_verdict,
                    diagnosis=diagnosis, diagnosis_evidence=diagnosis_evidence,
                )
                return

            evaluation_report = self.spec_store.get(payload.get("evaluation_report_ref"))
            run_record = self.spec_store.get(payload.get("run_record_ref"))

            research_memory_ref = payload.get("research_memory_ref")
            prior_research_memory = (
                self.spec_store.get(research_memory_ref) if research_memory_ref else None
            )

            verification_report_ref = payload.get("verification_report_ref")
            verification_report = (
                self.spec_store.get(verification_report_ref) if verification_report_ref else None
            )

            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            research_memory, routing_verdict = await asyncio.to_thread(
                core.consolidate,
                evaluation_report,
                run_record,
                prior_research_memory,
                verification_report,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "MemoryAgent", self.config.prompt_patch()),
            )
        except core.MemoryError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        await self._publish_consolidated(
            client, session_id, causation_id, research_memory, routing_verdict
        )

    def _consolidate_theory(
        self, *, method: dict, problem_model: dict | None,
        interpretation: dict | None, prior_research_memory: dict | None,
    ) -> dict:
        """Deterministically preserve a reviewed theory Method for honest reporting.

        Method deliberation/audits establish that this is the selected, evidence-aware
        direction; they do not magically prove every theorem.  Therefore theorem sketches
        live in ``consolidated_insights`` rather than ``verified_findings``.
        """
        prior = prior_research_memory or {}
        method = method or {}
        insights = list(prior.get("consolidated_insights") or [])
        approach = str(method.get("approach") or "").strip()
        if approach and approach not in insights:
            insights.append(approach)
        for item in method.get("changes_outline") or []:
            text = str(item).strip()
            if text and text not in insights:
                insights.append(text)

        risks = list(prior.get("negative_results") or [])
        for item in method.get("assumptions") or []:
            text = f"Unverified assumption or boundary: {str(item).strip()}"
            if text != "Unverified assumption or boundary:" and text not in risks:
                risks.append(text)
        for item in (problem_model or {}).get("risks") or []:
            text = f"Known theoretical risk: {str(item).strip()}"
            if text != "Known theoretical risk:" and text not in risks:
                risks.append(text)

        next_steps = list(prior.get("next_steps") or [])
        for component in method.get("components") or []:
            if not isinstance(component, dict):
                continue
            check = str(
                component.get("functional_check")
                or component.get("ablation_test")
                or ""
            ).strip()
            if check:
                step = f"Check {component.get('name') or 'component'}: {check}"
                if step not in next_steps:
                    next_steps.append(step)

        disposition = str((interpretation or {}).get("disposition") or "").strip()
        scope_note = str((interpretation or {}).get("scope_reason") or "").strip()
        if disposition or scope_note:
            scope = f"Scope assessment: {disposition or 'unspecified'}; {scope_note}".strip()
            if scope not in insights:
                insights.append(scope)

        memory = {
            "best_metric": None,
            "best_result_ref": None,
            "verified_findings": list(prior.get("verified_findings") or []),
            "negative_results": risks,
            "excluded_hypotheses": list(prior.get("excluded_hypotheses") or []),
            "consolidated_insights": insights,
            "divergence_seeds": list(prior.get("divergence_seeds") or []),
            "next_steps": next_steps,
            "diagnosis": "inconclusive",
            "diagnosis_evidence": [
                "The selected theory is preserved as proof sketches and explicit assumptions; "
                "no executable benchmark result is claimed."
            ],
            "component_history": list(prior.get("component_history") or []),
        }
        errors = core._validate_research_memory(memory)
        if errors:
            raise core.MemoryParseError(
                f"theory ResearchMemory failed schema validation: {errors}")
        return memory

    def _consolidate_validation(
        self, *, method: dict, problem_model: dict | None,
        interpretation: dict | None, run_record_ref: str, run_record: dict,
        validation_result: dict | None, verification_report: dict | None,
        prior_research_memory: dict | None,
    ) -> dict:
        """Attach one verified survey result to the existing theory memory."""
        memory = self._consolidate_theory(
            method=method,
            problem_model=problem_model,
            interpretation=interpretation,
            prior_research_memory=prior_research_memory,
        )
        result = validation_result or {}
        conclusion = str(result.get("conclusion") or "inconclusive")
        metric_name = str(result.get("metric_name") or "study_statistic")
        metric_value = result.get("metric_value")
        summary = str(result.get("analysis_summary") or "No analysis summary was produced.")
        sources = ", ".join(map(str, result.get("data_sources") or []))
        evidence = (
            f"survey_validation conclusion={conclusion}; {metric_name}={metric_value!r}; "
            f"samples={len(result.get('sample_ids') or [])}; sources={sources}; {summary}")
        verified = (
            (verification_report or {}).get("verdict") == "verified"
            and run_record.get("succeeded") is True)
        if conclusion == "supports" and verified:
            if evidence not in memory["verified_findings"]:
                memory["verified_findings"].append(evidence)
        else:
            negative = evidence if verified else "Unverified " + evidence
            if negative not in memory["negative_results"]:
                memory["negative_results"].append(negative)
        if evidence not in memory["consolidated_insights"]:
            memory["consolidated_insights"].append(evidence)
        memory["best_metric"] = metric_value if verified else None
        memory["best_result_ref"] = run_record_ref if verified else None
        memory["diagnosis"] = "inconclusive"
        memory["diagnosis_evidence"] = [evidence]
        errors = core._validate_research_memory(memory)
        if errors:
            raise core.MemoryParseError(
                f"survey-validation ResearchMemory failed schema validation: {errors}")
        return memory

    def _consolidate_failure(
        self, *, run_record: dict | None, baseline_result: dict | None,
        failure_context: dict | None, prior_research_memory: dict | None,
        evaluation_report: dict | None = None, gen_result: dict | None = None,
        baseline_attempt: dict | None = None,
    ) -> dict:
        """Preserve failure evidence; only a bound evaluation can falsify a Method."""
        prior = prior_research_memory or {}
        run = run_record or {}
        context = failure_context or {}
        terminal_diagnoses = {
            "literature_novelty_audit_failed": "literature_novelty_exhausted",
            "baseline_reproduction_thrashing": "baseline_reproduction_failed",
            "experiment_design_thrashing": "inconclusive",
            "method_revision_format_exhausted": "inconclusive",
            "method_contract_repair_exhausted": "inconclusive",
            "component_plan_review_format_exhausted": "inconclusive",
        }
        if context.get("action") == "discard_hypothesis":
            evaluation = evaluation_report or {}
            gen = gen_result or {}
            metric = run.get("metric_value")
            # Exit zero, a component probe, or another Method's evaluation cannot
            # falsify the current Method. Retain actual evaluated negative results.
            evaluated_negative = (
                run.get("succeeded") is True
                and isinstance(metric, (int, float)) and not isinstance(metric, bool)
                and math.isfinite(metric)
                and evaluation.get("metric_value") == metric
                and evaluation.get("verdict") == "valid_not_best"
                and evaluation.get("is_improvement") is False
                and evaluation.get("suspicious") is False
                and not evaluation.get("anti_cheat_flags")
                and evaluation.get("run_record_digest") == artifact_digest(run)
                and (gen.get("component_execution") or {}).get("mode") != "component"
                and not (gen.get("method_implementation_review") or {}).get("plan_revision")
                and (not context.get("method_ref")
                     or gen.get("method_ref") == context["method_ref"])
            )
            diagnosis = ("method_hypothesis_falsified" if evaluated_negative
                         else "component_execution_blocked")
        else:
            diagnosis = terminal_diagnoses.get(
                context.get("reason"), "execution_failure")
        detail = (context.get("detail") or run.get("stderr_excerpt") or run.get("stdout_excerpt")
                  or context.get("reason") or "candidate produced no evaluable result")
        baseline_metric = (baseline_result or {}).get("metric_value")
        evidence = (
            f"{context.get('reason') or 'execution failure'}"
            f"; action={context.get('action') or 'none'}"
            f"; exit_code={run.get('exit_code')!r}"
            f"; detail={detail}"
            f"; frozen_baseline={baseline_metric!r}"
        )
        if baseline_attempt:
            evidence += "; baseline_attempt (diagnostic, not a qualified comparison): " + json.dumps({
                key: baseline_attempt.get(key) for key in (
                    "metric_value", "metric_name", "reproduced", "exit_code", "measurement_evidence")},
                ensure_ascii=False)
        negatives = list(prior.get("negative_results") or [])
        if evidence not in negatives:
            negatives.append(evidence)
        memory = {
            "best_metric": prior.get("best_metric"),
            "best_result_ref": prior.get("best_result_ref"),
            "verified_findings": list(prior.get("verified_findings") or []),
            "negative_results": negatives,
            "excluded_hypotheses": list(prior.get("excluded_hypotheses") or []),
            "consolidated_insights": list(prior.get("consolidated_insights") or []),
            "divergence_seeds": list(prior.get("divergence_seeds") or []),
            "next_steps": list(prior.get("next_steps") or []),
            "diagnosis": diagnosis,
            "diagnosis_evidence": [evidence],
            "component_history": list(prior.get("component_history") or []),
        }
        errors = core._validate_research_memory(memory)
        if errors:
            raise core.MemoryParseError(
                f"failure ResearchMemory failed schema validation: {errors}")
        return memory

    async def _publish_consolidated(
        self, client: Any, session_id: str, causation_id: str | None,
        research_memory: dict, routing_verdict: str, *,
        diagnosis: str | None = None, diagnosis_evidence: list[str] | None = None,
    ) -> None:
        from autoresearch.experiments.repo_improve.source_policy import merge_source_history
        try:
            input_payload = getattr(self, "_request_inputs", {}).get((session_id, causation_id))
            prior_ref = (input_payload.get("research_memory_ref") if input_payload is not None
                         else refs.encode("research_memory", session_id, _RESEARCH_MEMORY_KEY))
            previous = self.spec_store.get(prior_ref) if prior_ref else {}
        except KeyError:
            previous = {}
        feedback_refs = list(dict.fromkeys(list((previous or {}).get("method_feedback_refs") or [])
                                          + list(research_memory.get("method_feedback_refs") or [])))
        if feedback_refs:
            research_memory["method_feedback_refs"] = feedback_refs
        if routing_verdict == "report":
            await self._review_selected_whole_for_report(session_id, research_memory)
        # Baseline v1 is mutable. New runners publish snapshots; use those rather
        # than erasing finite but infeasible observations with the final failure.
        baseline_refs = self.spec_store.list_refs(session_id, "baseline_result")
        snapshots = [ref for ref in baseline_refs if refs.decode(ref).key.startswith("snapshot-")]
        observations = list((previous or {}).get("baseline_observations") or [])
        baseline_sources = []
        for ref in snapshots or baseline_refs:
            baseline = self.spec_store.get(ref)
            if snapshots and refs.decode(ref).key != "snapshot-" + artifact_digest(baseline).split(":", 1)[1]:
                continue
            baseline_sources.extend(baseline.get("source_history") or [])
            if ref in {item.get("ref") for item in observations}:
                continue
            observations.append({"ref": ref, **{key: baseline.get(key) for key in (
                "metric_value", "metric_name", "metric_unit", "reproduced", "exit_code",
                "measurement_evidence", "repo_url", "repo_commit")}})
        if observations:
            research_memory["baseline_observations"] = observations
        history = merge_source_history((previous or {}).get("source_history"), research_memory.get("source_history"), baseline_sources)
        if history:
            research_memory["source_history"] = history
        candidates = merge_source_history((previous or {}).get("reconciliation_candidates"),
                                          research_memory.get("reconciliation_candidates"))
        if candidates:
            research_memory["reconciliation_candidates"] = candidates
        research_memory_ref = self.spec_store.put(
            kind="research_memory",
            session_id=session_id,
            key=_RESEARCH_MEMORY_KEY,
            obj=research_memory,
            emitter=AGENT_NAME,
        )
        if hasattr(self.spec_store, "snapshot"):
            research_memory_ref = self.spec_store.snapshot(research_memory_ref)
        event_payload = {
            "research_memory_ref": research_memory_ref,
            "routing_verdict": routing_verdict,
        }
        if diagnosis:
            event_payload["diagnosis"] = diagnosis
            event_payload["diagnosis_evidence"] = list(diagnosis_evidence or [])
        self.spec_store.put(kind="audit", session_id=session_id,
                            key="memory-request-" + str(causation_id),
                            obj={"request_digest": artifact_digest(
                                input_payload or {}),
                                 "payload": event_payload}, emitter=AGENT_NAME)
        await client.publish(
            event_type="memory.consolidated",
            session_id=session_id,
            payload=event_payload,
            causation_id=causation_id,
        )

    # ---- P1 general-mode branch (env gated, default OFF) ----

    def _gen_branch(self) -> bool:
        """Whether to seed the ResearchMemory DETERMINISTICALLY from the P1 result
        instead of the (loop-risk H6) LLM. True only under
        AUTORESEARCH_GEN_EXPERIMENT=1."""
        return os.environ.get(_GEN_ENV_FLAG) == "1"

    def _has_current_gen_result(self, session_id: str) -> bool:
        """Detect the structured result actually written by a repository run.

        Agentic repository execution predates the general-experiment feature flag and can run
        without it.  The persisted sibling artifact is the authoritative discriminator; ordinary
        runs have no such artifact and retain the LLM consolidation path.
        """
        try:
            self.spec_store.get(refs.encode(
                _GEN_RESULT_KIND, session_id, _GEN_CURRENT_RESULT_KEY))
        except KeyError:
            return False
        return True

    def _consolidate_gen(self, run_record_ref: str,
                         evaluation_report_ref: str | None = None,
                         prior_research_memory: dict | None = None,
                         verification_report: dict | None = None,
                         ) -> tuple[dict, str, str, list[str]]:
        """Recover the P1 verdict+metrics (from the sibling gen_experiment_result
        ref RunnerAgent wrote), seed a ResearchMemory grounded on the REAL numbers,
        ACCUMULATE it into ``prior_research_memory`` (so findings grow across
        idea<->experiment loop rounds), and route the exploration loop.

        Routing -- STOP-ON-WIN (the loop is also BOUNDED at the engine, which caps
        ``exploration_loop_round`` at AUTORESEARCH_LOOP_MAX_ROUNDS and forces ``report`` at
        the cap; the watchdog session budget backstops it):
          - a trusted WIN this round (a reproduced, isolated improvement) -> ``report``:
            we found an improvement, stop early.
          - a ``suspicious`` result -> ``report``: don't build the next hypothesis on a
            distrusted result.
          - an honest NON-win (valid-but-not-best / no-improvement / invalid/failed) ->
            ``continue_explore``: keep trying different ideas up to the round cap (default
            3), then report the ACCUMULATED findings.

        HONESTY: the P1 verdict is NOT the last word. The 34-state EvaluatorAgent (+ MOSS)
        is the trusted arbiter -- if it flagged the result ``suspicious`` or ``invalid``
        (impossible metric, seed/holdout violation, ...), the finding is NOT verified even
        when P1 said "pass": it lands in negative_results WITH the evaluator's flags. This
        stops a MOSS-discarded/suspicious result from being reported as a clean win."""
        parts = refs.decode(run_record_ref)
        # Runner keeps two artifacts: ``v1`` is best-so-far, while ``current`` is the round that
        # just produced this RunRecord. Consolidation must classify the current round; otherwise a
        # marginal old win survives a failed strict rerun and is incorrectly reported as verified.
        from autoresearch.spec_store.identity import result_for_run
        gen = result_for_run(self.spec_store, run_record_ref)
        verification_digest = (verification_report or {}).get("run_record_digest")
        if verification_digest and verification_digest != artifact_digest(self.spec_store.get(run_record_ref)):
            raise ValueError("verification report belongs to a different run record")
        verdict = gen.get("verdict")
        accepted = gen.get("accepted")
        pb = gen.get("baseline_metric")
        pm = gen.get("method_metric")
        reasons = gen.get("reasons") or []

        # PAPER-REPRODUCTION LINE (opt-in): surface how our live-reproduced baseline compares
        # to the paper's PUBLISHED value as a neutral insight (not a win/negative), so the
        # report shows the faithfulness check. Absent unless AUTORESEARCH_PAPER_REPRODUCTION=1.
        paper_note = None
        pr = gen.get("paper_reproduction")
        if pr:
            st_ = pr.get("status")
            if st_ == "reproduced":
                paper_note = (f"论文复现线: 实机复现 baseline={pr.get('measured')} 与论文报告值"
                              f"={pr.get('published')}(来源: {pr.get('source') or '未标注'})一致"
                              f"(相对误差 {pr.get('rel_error'):.1%} ≤ 容差 {pr.get('tolerance'):.0%}).")
            elif st_ == "mismatch":
                paper_note = (f"论文复现线: 实机复现 baseline={pr.get('measured')} 与论文报告值"
                              f"={pr.get('published')}不一致(相对误差 {pr.get('rel_error'):.1%} > 容差 "
                              f"{pr.get('tolerance'):.0%}; 来源: {pr.get('source') or '未标注'}).")
            elif st_ == "no_published_baseline":
                paper_note = "论文复现线: 文献中未找到可引用的 published baseline;以实机复现值为准."

        # The trusted 34-state evaluator's verdict (suspicious/invalid) vetoes a P1 "pass".
        eval_report = self.spec_store.get(evaluation_report_ref) if evaluation_report_ref else None
        eval_verdict = (eval_report or {}).get("verdict")
        eval_flags = (eval_report or {}).get("anti_cheat_flags") or []
        component_execution = gen.get("component_execution") or {}
        # A component attempt is evidence about that component, not yet an end-to-end
        # Method result.  Keep every measured scalar in component_history, but only let a
        # completed integration (or the legacy non-component path) update the task-level
        # best_metric / verified_findings used by the final report.  Otherwise a locally good
        # direct-gain component can incorrectly outrank the assembled Method even when the
        # repository's final correctness gates reject that intermediate checkpoint.
        primary_metric_eligible = component_execution.get("mode") != "component"
        contract_violation = _result_import_violation(gen)
        comparison_only_suspicion = (
            eval_verdict == "suspicious"
            and isinstance(eval_flags, list)
            and bool(eval_flags)
            and all(
                isinstance(flag, str)
                and flag.startswith("comparison_semantics_changed:")
                for flag in eval_flags
            )
            and isinstance(pm, (int, float))
            and not isinstance(pm, bool)
            and math.isfinite(float(pm))
        )

        verified_findings: list[str] = []
        negative_results: list[str] = []
        if contract_violation:
            negative_results.append(
                "Task-contract violation in measured candidate: "
                + "; ".join(contract_violation["violations"]))
        if not primary_metric_eligible:
            # The scalar belongs to the component contract/history, not to the task-level
            # baseline-vs-method comparison.  A fully evidenced accepted component is still
            # a verified performance fact, though: retain it with deliberately limited
            # attribution so a later failed component cannot erase a real official result.
            execution_contract = (
                component_execution.get("attempt_contract_result")
                or component_execution.get("contract_result") or {}
            )
            implementation_review = gen.get("method_implementation_review") or {}
            if (
                (component_execution.get("status") == "accepted"
                 or (component_execution.get("status") in {"qualified", "improving"}
                     and accepted is True))
                and eval_verdict not in {"invalid", "suspicious"}
                and component_execution.get("execution_succeeded") is True
                and isinstance(execution_contract, dict)
                and execution_contract.get("passed") is True
                and isinstance(implementation_review, dict)
                and implementation_review.get("verdict") in {"faithful", "partial"}
                and (verification_report or {}).get("verdict") == "verified"
                and not any("failure_confirmed" in str(check) for check in
                            ((verification_report or {}).get("checks") or []))
                and isinstance(pm, (int, float))
                and not isinstance(pm, bool)
                and math.isfinite(float(pm))
            ):
                verified_findings.append(
                    "Verified component observation: "
                    f"component={component_execution.get('component_id') or 'unknown'}, "
                    f"baseline={pb}, metric={pm}, role={component_execution.get('role') or 'unknown'}, "
                    f"implementation={implementation_review.get('verdict')}. "
                    "This verifies the measured component checkpoint only; it does not validate "
                    "unimplemented components or the complete frozen Method."
                )
        elif contract_violation:
            pass  # Already recorded above; no verified full-Method finding.
        elif eval_verdict in ("suspicious", "invalid"):
            negative_results.append(
                f"Result flagged {eval_verdict} by the trusted evaluator -- finding NOT "
                f"verified: baseline={pb} vs method={pm} (metric value); "
                f"flags: {'; '.join(map(str, eval_flags)) or '(none)'}."
            )
        elif verdict == "pass" or accepted is True:
            # A trusted-accepted win. ``verdict`` vocabulary differs by experiment path
            # (general -> "pass"; repo_improve -> "new_best"); both set ``accepted=True``
            # only for a reproduced, isolated improvement, so key off that too. (In the
            # general path verdict=="pass" <=> accepted is True, so this is a no-op there.)
            measurement = gen.get("measurement_evidence") or {}
            measurement_note = (
                "official single-run observation; repeated-run significance and resource isolation are not inferred"
                if (measurement.get("protocol") or {}).get("kind") == "official_single"
                else f"frozen aggregate over {len(measurement['samples'])} samples"
                if measurement.get("samples") else "reproduced, run-time isolated")
            finite_pair = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                              and math.isfinite(v) for v in (pb, pm))
            improved = finite_pair and (pm < pb if gen.get("lower_is_better", True) else pm > pb)
            relation = "beats" if improved else "does not beat"
            finding = f"Method {relation} baseline: baseline={pb} vs method={pm} (metric value), " + measurement_note + "."
            control = measurement.get("control_metric")
            if isinstance(control, (int, float)) and not isinstance(control, bool) and math.isfinite(control):
                finding += f" Paired control={control}; its contrast is separate from the frozen baseline."
            verified_findings.append(finding)

        else:
            negative_results.append(
                f"Verdict={verdict}: baseline={pb} vs method={pm} (metric value), "
                f"reproduced, run-time isolated; reasons: {'; '.join(map(str, reasons))}."
            )

        # ACCUMULATE across loop rounds: append this round's findings to the prior memory
        # (order-preserving dedupe), keep the best metric, carry the exploration ledger.
        prior = prior_research_memory or {}
        verification_failed = (verification_report or {}).get("verdict") == "failed"
        invalidate_prior = verification_failed
        prior_ref = prior.get("best_result_ref")
        if verification_failed and isinstance(prior_ref, str):
            try:
                prior_parts = refs.decode(prior_ref)
                saved = self.spec_store.get(prior_ref)
                current_record = self.spec_store.get(run_record_ref)
                if (prior_parts.kind == _GEN_RESULT_KIND
                        and prior_parts.session_id == refs.decode(run_record_ref).session_id
                        and prior_parts.key == "snapshot-" + artifact_digest(saved).split(":", 1)[1]):
                    saved_record = self.spec_store.get(saved["run_record_snapshot_ref"])
                    # Only a failed replication of this candidate can revoke it.
                    # A missing diff, or a failure before launch, is not that proof.
                    executed = gen.get("component_execution_started") is not False
                    same_protocol = ((saved.get("measurement_evidence") or {}).get("protocol_fingerprint")
                                     == (gen.get("measurement_evidence") or {}).get("protocol_fingerprint"))
                    same_source = (bool(saved.get("checkpoint")) and saved.get("checkpoint") == gen.get("checkpoint")) or (
                        bool(saved.get("diff")) and saved.get("diff") == gen.get("diff"))
                    invalidate_prior = saved_record == current_record or (executed and same_protocol and same_source)

            except (KeyError, ValueError, TypeError):
                pass  # Unbound legacy claims cannot be certified as independent evidence.
        verification_checks = (verification_report or {}).get("checks") or []
        verified_failure_reason = None
        if ((verification_report or {}).get("verdict") == "verified"
                and isinstance(verification_checks, list)
                and any(isinstance(check, str) and "failure_confirmed" in check
                        for check in verification_checks)):
            verified_failure_reason = str(
                (verification_report or {}).get("reasoning") or "").strip() or None
            if verified_failure_reason:
                negative_results.append(verified_failure_reason)
        if verification_failed:
            checks = "; ".join(map(str, (verification_report or {}).get("checks") or []))
            negative_results.append(
                ("Prior claimed improvement verification failed and is no longer verified"
                 if invalidate_prior else "Current candidate verification failed; independent prior result retained")
                + (f": {checks}." if checks else ".")
            )

        def _merge(prior_list, new_list):
            out = list(prior_list or [])
            for x in new_list:
                if x not in out:
                    out.append(x)
            return out

        # ``repo_improve`` labels a candidate ``new_best`` relative to the frozen baseline of
        # that sandbox.  Across outer-loop rounds that does not necessarily beat the best value
        # already in ResearchMemory, so preserve the actual running best using the task's metric
        # direction when the runner supplied it.
        best_metric = None if invalidate_prior else prior.get("best_metric")
        best_result_ref = None if invalidate_prior else prior.get("best_result_ref")
        invalid_prior_pair = None
        prior_best = None
        if best_result_ref:
            try:
                prior_best = self.spec_store.get(best_result_ref)
            except (KeyError, ValueError, TypeError):
                prior_best = None
            if isinstance(prior_best, dict) and (
                    not _formal_best_eligible(prior_best)
                    or ((prior_best.get("method_implementation_review") or {}).get("verdict")
                        in {"partial", "drifted"}
                        and can_revalidate_accepted_whole_result(gen)
                        and (verification_report or {}).get("verdict") == "verified"
                        and eval_verdict not in ("invalid", "suspicious"))):
                invalid_prior_pair = (
                    prior_best.get("baseline_metric"), prior_best.get("method_metric"))
                negative_results.append(
                    "Prior measured integration is not a formal Method best: "
                    "its implementation, task-contract or candidate-admissibility review did not pass. "
                    f"Observation retained at {best_result_ref}."
                )
                best_metric = None
                best_result_ref = None
        measurement = gen.get("measurement_evidence") or {}
        minimum = measurement.get("minimum_improvement", 0)
        comparable = True
        if measurement and best_metric is not None:
            try:
                incumbent = self.spec_store.get(best_result_ref)
                minimum = comparison_epsilon(gen, incumbent)
                comparable = all(incumbent.get(key) == gen.get(key) for key in (
                    "metric_name", "metric_unit", "lower_is_better", "baseline_metric")) and (
                    (incumbent.get("measurement_evidence") or {}).get("protocol_fingerprint")
                    == measurement.get("protocol_fingerprint"))
            except (KeyError, ValueError, TypeError):
                comparable = False
        finite_metric = (
            primary_metric_eligible and _formal_best_eligible(gen) and comparable
            and not ((gen.get("method_implementation_review") or {}).get("verdict")
                     in {"partial", "drifted"}
                     and can_revalidate_accepted_whole_result(prior_best))
            and (not measurement or (accepted is True and
                 (verification_report or {}).get("verdict") == "verified"))
            and
            isinstance(pm, (int, float))
            and not isinstance(pm, bool)
            and math.isfinite(float(pm))
        )
        if (best_result_ref is None and accepted is True and not verification_failed
                and finite_metric and eval_verdict not in ("invalid", "suspicious")
                and prior.get("reconciliation_candidates")
                and not any(not str(item).startswith("Verified component observation:")
                            for item in prior.get("verified_findings") or [])):
            # An earlier measured but unaccepted off-method result may have left a
            # scalar without a bound incumbent. The first accepted result establishes
            # the formal best; retain the earlier observation separately.
            best_metric = pm
        elif (best_metric is None and not verification_failed and finite_metric
                and eval_verdict not in ("invalid", "suspicious")):
            # Preserve the previous first-round behavior: even an honest non-win is the best
            # measured candidate when no earlier candidate exists.
            best_metric = pm
        elif (not verification_failed and accepted is True and finite_metric
              and eval_verdict not in ("invalid", "suspicious")):
            lower_is_better = gen.get("lower_is_better")
            if lower_is_better is True and best_metric - pm > minimum:
                best_metric = pm
            elif lower_is_better is False and pm - best_metric > minimum:
                best_metric = pm
            elif lower_is_better not in (True, False):
                # Backwards compatibility for older result artifacts without direction.
                best_metric = pm

        if best_metric != prior.get("best_metric"):
            # Never attach an older result's citation to a newly selected scalar.
            best_result_ref = None
        if (finite_metric and pm == best_metric
                and best_result_ref is None
                and (verification_report or {}).get("verdict") == "verified"
                and eval_verdict not in ("invalid", "suspicious")):
            snapshot_ref = gen.get("run_record_snapshot_ref")
            if isinstance(snapshot_ref, str) and snapshot_ref:
                try:
                    snapshot = self.spec_store.get(snapshot_ref)
                except KeyError:
                    snapshot = None
                # Bind only the host snapshot of this verified run, not a model-supplied ref
                # or an unrelated historical observation with the same numeric value.
                if (snapshot is not None
                        and snapshot == self.spec_store.get(run_record_ref)
                        and snapshot.get("metric_value") == pm):
                    best_result_ref = snapshot_ref
                    candidate_ref = gen.get("candidate_snapshot_ref")
                    if isinstance(candidate_ref, str) and candidate_ref:
                        try:
                            candidate_snapshot = self.spec_store.get(candidate_ref)
                        except KeyError:
                            candidate_snapshot = None
                        expected_candidate = {k: v for k, v in gen.items()
                                              if k != "candidate_snapshot_ref"}
                        if candidate_snapshot == expected_candidate and accepted is True:
                            try:
                                candidate_parts = refs.decode(candidate_ref)
                                if (candidate_parts.kind == _GEN_RESULT_KIND
                                        and candidate_parts.session_id == parts.session_id
                                        and candidate_parts.key == "snapshot-" + artifact_digest(
                                            candidate_snapshot).split(":", 1)[1]):
                                    best_result_ref = candidate_ref
                            except ValueError:
                                pass

        explicit_diagnosis = gen.get("diagnosis")
        explicit_evidence = [str(item) for item in (gen.get("diagnosis_evidence") or []) if item]
        evaluator_vetoed = eval_verdict in ("invalid", "suspicious")
        component_diagnosis, component_evidence = _classify_component_evidence(gen)
        finite_candidate = (
            isinstance(pm, (int, float))
            and not isinstance(pm, bool)
            and math.isfinite(float(pm))
        )
        implementation_review = gen.get("method_implementation_review")
        legacy_diagnostic_review_veto = (
            gen.get("operator") == "check-expectation"
            and explicit_diagnosis == "candidate_contract_violation"
            and eval_verdict == "invalid"
            and not eval_flags
            and isinstance(implementation_review, dict)
            and implementation_review.get("verdict") not in (None, "faithful")
        )
        # The evaluator remains authoritative for every measured candidate.  With no candidate
        # metric, however, its comparison verdict can only say that performance is unmeasurable;
        # it does not invalidate structured evidence that a confirmed component's prerequisite is
        # absent.  Keeping that preflight diagnosis is conservative: it can only discard/redesign
        # the method, never accept a result or claim an improvement.  Also recognize diagnostic
        # artifacts produced by older runners that incorrectly applied the whole-method fidelity
        # gate to A/B component probes.  This exception is deliberately narrow: any comparison
        # flag still vetoes the evidence, and the result can only trigger redesign, never a win.
        # An invalid metric cannot erase a host execution-budget stop. Keeping
        # this diagnosis reports the failure; it never accepts the measurement.
        if (evaluator_vetoed and finite_candidate and not legacy_diagnostic_review_veto
                and component_diagnosis != "component_execution_blocked"
                and not (component_diagnosis == "component_expectation_failed"
                         and (implementation_review or {}).get("plan_revision"))
                and not (_has_bound_failed_quality_gate(gen) and not eval_flags)):
            component_diagnosis, component_evidence = None, []
        setting_evidence = (
            [] if evaluator_vetoed
            else _classify_setting_evidence(gen.get("setting_evidence")))
        host_reflection = gen.get("host_reflection")
        reflection_evidence = []
        if (not evaluator_vetoed
                and isinstance(host_reflection, dict)
                and set(host_reflection) == {"route", "verdict", "method_note"}
                and host_reflection.get("route") == "revise_method"):
            note = str(host_reflection.get("method_note") or "").strip()
            if note:
                reflection_evidence = [note]
        if contract_violation:
            diagnosis = "candidate_contract_violation"
            diagnosis_evidence = contract_violation["violations"] + contract_violation["evidence"]
        elif component_diagnosis:
            diagnosis = component_diagnosis
            diagnosis_evidence = component_evidence
        elif verified_findings:
            diagnosis = "valid_improvement"
            diagnosis_evidence = [verified_findings[0]]
        elif setting_evidence:
            diagnosis = "setting_issue"
            diagnosis_evidence = setting_evidence
        elif explicit_diagnosis == "candidate_contract_violation" and explicit_evidence:
            diagnosis = explicit_diagnosis
            diagnosis_evidence = explicit_evidence
        elif reflection_evidence:
            diagnosis = (
                "method_description_reconciliation"
                if finite_candidate
                and isinstance(implementation_review, dict)
                and implementation_review.get("verdict") == "drifted"
                else "method_hypothesis_falsified"
            )
            diagnosis_evidence = reflection_evidence
        elif explicit_diagnosis in _TRUSTED_PHASE_DIAGNOSES and explicit_evidence:
            diagnosis = explicit_diagnosis
            diagnosis_evidence = explicit_evidence
        elif pm is None or (
                isinstance(gen.get("component_execution"), dict)
                and gen["component_execution"].get("mode") == "component"
                and gen["component_execution"].get("status") == "execution_failed"
                and gen["component_execution"].get("execution_succeeded") is False):
            # A claimed finite scalar does not repair failed host execution/binding.
            # Preserve its cause for retry, not a scientific no-improvement conclusion.
            diagnosis = "execution_failure"
            diagnosis_evidence = (
                [verified_failure_reason] if verified_failure_reason
                else ([str(reason) for reason in reasons] or ["no finite method metric"])
            )
        else:
            diagnosis = "inconclusive"
            diagnosis_evidence = [
                f"valid comparison did not improve: baseline={pb}, method={pm}"]

        off_method = gen.get("off_method_observation")
        off_method_note = None
        if (not contract_violation and isinstance(off_method, dict)
                and off_method.get("valid") is True):
            off_method_note = (
                "Off-method observation retained for comparison: "
                f"metric={off_method.get('metric')!r}; improved={off_method.get('improved')!r}; "
                "the measured diff is not accepted as the frozen Method until implementation "
                "alignment or formal method-description revision completes."
            )
        insight_updates = [item for item in (paper_note, off_method_note) if item]
        prior_verified_findings = list(prior.get("verified_findings") or [])
        if invalid_prior_pair:
            pair = (f"baseline={invalid_prior_pair[0]} vs "
                    f"method={invalid_prior_pair[1]}")
            prior_verified_findings = [
                item for item in prior_verified_findings
                if not (isinstance(item, str)
                        and item.startswith("Method beats baseline: ")
                        and pair in item)]
        if invalidate_prior:
            # A failed verification invalidates a prior whole-Method claim when that result is
            # being rerun, but it must not erase independently verified component checkpoints.
            prior_verified_findings = [
                item for item in prior_verified_findings
                if isinstance(item, str) and item.startswith("Verified component observation:")
            ]
        research_memory = {
            **({"method_feedback_refs": list(prior["method_feedback_refs"])}
               if "method_feedback_refs" in prior else {}),
            "best_metric": best_metric,
            "best_result_ref": best_result_ref,
            "verified_findings": _merge(
                prior_verified_findings, verified_findings),
            "negative_results": _merge(prior.get("negative_results"), negative_results),
            "excluded_hypotheses": list(prior.get("excluded_hypotheses") or []),
            "consolidated_insights": _merge(
                prior.get("consolidated_insights"), insight_updates),
            "divergence_seeds": list(prior.get("divergence_seeds") or []),
            "next_steps": list(prior.get("next_steps") or []),
            "diagnosis": diagnosis,
            "diagnosis_evidence": diagnosis_evidence,
            "component_history": _append_component_history(
                prior, gen, run_record_ref, verification_report),
        }
        from autoresearch.experiments.repo_improve.source_policy import merge_source_history
        research_memory["source_history"] = merge_source_history(
            prior.get("source_history"), gen.get("source_history"))
        candidates = list(prior.get("reconciliation_candidates") or [])
        off_method_candidate = gen.get("off_method_observation") or {}
        if (gen.get("checkpoint") and gen.get("candidate_snapshot_ref") and (
                (diagnosis == "method_description_reconciliation"
                 and off_method_candidate.get("valid") is True
                 and off_method_candidate.get("improved") is True)
                or can_revalidate_reconciliation_candidate(gen))):
            candidate = {"result_ref": gen["candidate_snapshot_ref"], "checkpoint": gen["checkpoint"]}
            if candidate not in candidates:
                candidates.append(candidate)
        research_memory["reconciliation_candidates"] = candidates
        errors = core._validate_research_memory(research_memory)
        if errors:
            raise core.MemoryParseError(
                f"P1 ResearchMemory failed schema validation: {errors}"
            )
        # Loop routing -- STOP-ON-WIN:
        #   - a trusted WIN this round (verified_findings non-empty: a reproduced, isolated
        #     improvement the 34-state evaluator did NOT flag) -> report. We found what we
        #     were after; no need to burn more rounds.
        #   - a SUSPICIOUS result (anti-cheat / dishonest) -> report. Don't build the next
        #     idea on a distrusted result, and don't keep probing it.
        #   - otherwise (an honest non-win: valid-but-not-best, no-improvement, or an
        #     invalid/failed run) -> continue_explore. Keep trying DIFFERENT ideas; the
        #     engine caps this at AUTORESEARCH_LOOP_MAX_ROUNDS (default 3) rounds, then reports
        #     the accumulated findings.
        if diagnosis in {"component_accepted", "component_progress", "component_contract_retry",
                         "component_combination_retry", "method_description_reconciliation"}:
            routing = "need_experiment"
        elif diagnosis == "component_execution_blocked" or verified_findings:
            routing = "report"
        elif diagnosis == "setting_issue" or comparison_only_suspicion:
            routing = "need_experiment"
        elif eval_verdict == "suspicious":
            routing = "report"
        else:
            routing = "continue_explore"
        return research_memory, routing, diagnosis, diagnosis_evidence

    async def _decide_efficient_component(self, session_id, run_ref, memory,
                                          verification, baseline_ref=None):
        """Let one evidence-bound choice release a qualified component or continue it."""
        from autoresearch.experiments.repo_improve.execution_policy import efficient_execution
        if not efficient_execution() or (verification or {}).get("verdict") != "verified":
            return
        from autoresearch.spec_store.identity import result_for_run
        from autoresearch.experiments.repo_improve.component_plan import (
            build_component_plan, ledger_for_workroot)
        from autoresearch.experiments.repo_improve.search_memory import session_workroot
        from autoresearch.agents.reflection_agent.core import decide_component_progress

        gen = result_for_run(self.spec_store, run_ref)
        execution = gen.get("component_execution") or {}
        if (execution.get("mode") != "component"
                or execution.get("status") not in {
                    "qualified", "improving", "accepted", "execution_failed",
                    "contract_failed", "implementation_failed", "checkpoint_missing",
                    "drifted"}
                or not gen.get("candidate_snapshot_ref")
                or verification.get("candidate_snapshot_ref") != gen["candidate_snapshot_ref"]
                or verification.get("run_record_ref") != run_ref):
            return
        run = self.spec_store.get(run_ref)
        method = self.spec_store.get(run["experiment_id"])
        baseline_ref = baseline_ref or refs.encode("baseline_result", session_id, "v1")
        baseline = self.spec_store.get(baseline_ref)
        plan = build_component_plan(method)
        ledger = ledger_for_workroot(
            session_workroot(session_id), plan,
            execution_identity=artifact_digest(baseline) if baseline else None)
        component_id = execution["component_id"]
        entry = ledger.entry(component_id)
        result_ref = gen["candidate_snapshot_ref"]
        node = plan.node(component_id)
        if entry.status == "accepted" and node.role == "direct_gain":
            # The ledger has already retained the host-best checkpoint at exhaustion.
            # Only indirect nodes need a final semantic checkpoint selection.
            return bool(entry.checkpoint and entry.qualified_attempts > 0
                        and entry.contract_result.get("passed") is True)
        at_limit = (entry.status == "accepted" and node.role == "indirect_gain"
                    and execution.get("status") == "accepted")
        try:
            attempt_limit = max(1, int(os.environ.get("AUTORESEARCH_COMPONENT_MAX_ATTEMPTS", "6")))
        except ValueError:
            attempt_limit = 6
        at_limit = at_limit or entry.attempts >= attempt_limit or entry.infrastructure_attempts >= attempt_limit
        if (not (entry.status == "active" or at_limit) or not entry.checkpoint
                or entry.search_decision.get("result_ref") == result_ref):
            return
        try:
            closeout_requested = self.spec_store.get(refs.encode(
                "human_idea_seed", session_id, "closeout-v1")).get("closeout") is True
        except KeyError:
            closeout_requested = False
        must_advance = at_limit or closeout_requested
        best = {"checkpoint": entry.checkpoint, "task_metric": entry.metric_value,
                "local_quality": entry.quality_value,
                "contract_result": entry.contract_result,
                "qualified_attempts": entry.qualified_attempts,
                "attempts": entry.attempts, "attempt_limit": attempt_limit,
                "remaining_attempts": max(0, attempt_limit - entry.attempts),
                "infrastructure_attempts": entry.infrastructure_attempts,
                "remaining_infrastructure_attempts": max(0, attempt_limit - entry.infrastructure_attempts),
                "attempt_limit_reached": at_limit,
                "closeout_requested": closeout_requested}
        latest = {"checkpoint": gen.get("checkpoint"),
                  "status": execution.get("status"),
                  "error": gen.get("error"),
                  "task_metric": gen.get("method_metric"),
                  "local_quality": execution.get("attempt_quality_value",
                                                 execution.get("quality_value")),
                  "contract_result": execution.get("attempt_contract_result"),
                  "changes_summary": gen.get("changes_summary"),
                  "observed_bottleneck": str(gen.get("bottleneck") or "")[:2000],
                  "result_ref": result_ref}
        history = [item for item in memory.get("component_history") or []
                   if item.get("component_id") == component_id
                   and item.get("method_fingerprint") == plan.method_fingerprint
                   and item.get("plan_version") == plan.version
                   and item.get("baseline_digest") == artifact_digest(baseline)]
        choice = await asyncio.to_thread(
            decide_component_progress, node.to_dict(), best, latest,
            entry.qualified_results, history,
            self.llm_client)
        if choice["action"] == "unavailable" and not must_advance:
            raise RuntimeError(choice["reason"])
        ledger.decide_qualified(
            component_id, result_ref=result_ref,
            action="advance" if must_advance else choice["action"],
            reason=(("explicit user closeout; " if closeout_requested else
                     "attempt limit reached; " if at_limit and choice["action"] != "advance"
                     else "") + choice.get("reason", "")),
            next_experiment="" if must_advance else choice.get("next_experiment", ""),
            selected_checkpoint=choice.get("selected_checkpoint", "best"))
        retained = ledger.entry(component_id)
        return bool(retained.status == "accepted" and retained.checkpoint
                    and retained.qualified_attempts > 0
                    and retained.contract_result.get("passed") is True)

    async def _review_retained_off_method(self, session_id, current_run_ref, memory,
                                          baseline_ref=None, *, closeout_requested=False):
        """Revisit legal observations without preempting healthy component progress."""
        from autoresearch.experiments.repo_improve.execution_policy import method_feedback_limit
        from autoresearch.spec_store.identity import result_for_run
        if not method_feedback_limit():
            return None
        current = result_for_run(self.spec_store, current_run_ref)
        execution = current.get("component_execution") or {}
        try:
            closing = self.spec_store.get(refs.encode("human_idea_seed", session_id, "closeout-v1"))
        except KeyError:
            closing = {}
        closeout_requested = (closeout_requested or
            execution.get("status") == "exhausted" or closing.get("closeout") is True)
        current_review = current.get("method_implementation_review") or {}
        if (not closeout_requested and not execution and current.get("accepted") is True
                and current_review.get("verdict") == "partial" and current_review.get("issues")
                and current.get("method_ref")
                and current.get("method_ref") == current_review.get("method_ref")
                and (current_review.get("task_contract_review") or {}).get("verdict") == "compliant"
                and (current.get("candidate_admissibility") or {}).get("verdict") == "pass"
                and (current.get("repository_comparison_review") or {}).get("verdict") == "pass"):
            # Review the new full measurement with the existing complete-result budget.
            # An older component must not repeatedly preempt its implementation repair.
            return None
        contract = execution.get("attempt_contract_result") or execution.get("contract_result") or {}
        parent = contract.get("parent_evidence") or {}
        invalid_parent = (execution.get("role") == "direct_gain"
                          and execution.get("execution_succeeded") is True
                          and contract.get("passed") is False
                          and parent.get("host_bound") is True and parent.get("failed_gates"))
        if (not closeout_requested and execution.get("mode") == "component"
                and (invalid_parent or (
                    execution.get("status") in {"qualified", "accepted", "improving"}
                    and execution.get("execution_succeeded") is True and contract.get("passed") is True))):
            return None
        best_ref = memory.get("best_result_ref")
        incumbent = self.spec_store.get(best_ref) if best_ref else None
        baseline = self.spec_store.get(baseline_ref) if baseline_ref else None
        if incumbent is None and not (baseline or {}).get("reproduced"):
            return None
        method_ref = self.spec_store.get(current_run_ref).get("experiment_id")
        if not method_ref:
            return None
        method = self.spec_store.get(method_ref)
        qualified_metrics = [
            item["metric"] for item in _qualified_closeout_components(
                memory, artifact_digest(baseline) if baseline else current.get("baseline_digest"))
            if item.get("role") == "direct_gain"
            and isinstance(item.get("metric"), (int, float))
            and not isinstance(item["metric"], bool) and math.isfinite(item["metric"])
        ]
        entries = list(memory.get("reconciliation_candidates") or [])
        for item in memory.get("component_history") or []:
            if (not (item.get("status") == "drifted"
                     or (item.get("status") == "implementation_failed"
                         and item.get("mir_verdict") == "partial")
                     or (item.get("mode") == "integration"
                         and item.get("status") in {"new_best", "valid_not_best"}
                         and item.get("mir_verdict") == "partial"))
                    or item.get("contract_passed") is not True
                    or item.get("verification_verdict") != "verified" or not item.get("run_ref")):
                continue
            try:
                observed = result_for_run(self.spec_store, item["run_ref"])
                entries.append({"result_ref": observed["candidate_snapshot_ref"],
                                "checkpoint": observed.get("checkpoint")})
            except (KeyError, TypeError, ValueError):
                continue
        candidates = []
        covering_results = (current, incumbent)
        for entry in entries:
            try:
                candidate_ref = entry["result_ref"]
                parts = refs.decode(candidate_ref)
                candidate = self.spec_store.get(candidate_ref)
                run_ref = candidate["run_record_snapshot_ref"]
                run_parts = refs.decode(run_ref)
                run = self.spec_store.get(run_ref)
                if (parts.kind != _GEN_RESULT_KIND or parts.session_id != session_id
                        or parts.key != "snapshot-" + artifact_digest(candidate).split(":", 1)[1]
                        or run_parts.kind != "run_record" or run_parts.session_id != session_id
                        or run_parts.key != "snapshot-" + artifact_digest(run).split(":", 1)[1]
                        or candidate.get("checkpoint") != entry.get("checkpoint")
                        # This observation already supplied the current Method's source.
                        # A failed revalidation needs repair, not the same adoption again.
                        or (not closeout_requested
                            and candidate_ref == method.get("reconciled_result_ref"))
                        or (run_ref != current_run_ref
                            and not _independent_of_failed_result(candidate, current))
                        or _method_review_trigger(
                            {**candidate, "candidate_snapshot_ref": candidate_ref}, memory,
                            "method_description_reconciliation", incumbent, baseline)
                            not in {"legal_off_method_gain", "component_method_reconciliation"}
                        or (baseline is not None and candidate.get("baseline_digest")
                            != artifact_digest(baseline))):
                    continue
                # Accepted whole results cover retained source at the same or
                # a worse metric, including when the latest Method failed.
                if any(isinstance(covered, dict) and _formal_best_eligible(covered)
                        and can_revalidate_accepted_whole_result(covered)
                        and all(candidate.get(key) == covered.get(key) for key in (
                            "baseline_digest", "baseline_metric", "metric_name",
                            "metric_unit", "lower_is_better"))
                        and (candidate["method_metric"] >= covered["method_metric"]
                             if covered["lower_is_better"] else
                             candidate["method_metric"] <= covered["method_metric"])
                        for covered in covering_results):
                    continue
                # A qualified component already covers an equal or better observation.
                if qualified_metrics and (
                        candidate["method_metric"] >= min(qualified_metrics)
                        if current["lower_is_better"] else
                        candidate["method_metric"] <= max(qualified_metrics)):
                    continue
                candidates.append((candidate["method_metric"], run_ref, candidate_ref, run))
            except (KeyError, TypeError, ValueError):
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=not current["lower_is_better"])
        candidate_by_run = {item[1]: item for item in candidates}
        pending = set(candidate_by_run)
        verified = {}
        for report_ref in self.spec_store.list_refs(session_id, "verification_report"):
            try:
                parts = refs.decode(report_ref)
                if not parts.key.startswith("snapshot-"):
                    continue
                report = self.spec_store.get(report_ref)
                run_ref = report.get("run_record_ref")
                if run_ref not in pending or report.get("verdict") != "verified":
                    continue
                _, _, candidate_ref, run = candidate_by_run[run_ref]
                evaluation_ref = report.get("evaluation_report_ref")
                evaluation_parts = refs.decode(evaluation_ref)
                evaluation = self.spec_store.get(evaluation_ref)
                digest = artifact_digest(run)
                source_only = can_revalidate_reconciliation_candidate(candidate)
                if (parts.key != "snapshot-" + artifact_digest(report).split(":", 1)[1]
                        or report.get("candidate_snapshot_ref") != candidate_ref
                        or report.get("run_record_digest") != digest
                        or (not source_only and any(
                            "failure_confirmed" in str(check)
                            for check in report.get("checks") or []))
                        or evaluation_parts.kind != "evaluation_report"
                        or evaluation_parts.session_id != session_id
                        or evaluation_parts.key != "snapshot-" + artifact_digest(evaluation).split(":", 1)[1]
                        or evaluation.get("run_record_ref") != run_ref
                        or evaluation.get("run_record_digest") != digest
                        or (not source_only
                            and evaluation.get("verdict") in {"invalid", "suspicious"})
                        or evaluation.get("anti_cheat_flags")
                        or report.get("anti_cheat_flags")):
                    continue
                verified[run_ref] = report
                pending.remove(run_ref)
                if not pending:
                    break
            except (KeyError, TypeError, ValueError):
                continue
        for _, run_ref, _, _ in candidates:
            if run_ref in verified:
                review_ref = await self._review_complete_result(
                    session_id, run_ref, memory, verified[run_ref],
                    "method_description_reconciliation", baseline_ref,
                    review_method_ref=method_ref, closeout_requested=closeout_requested,
                    retained_source_review=True)
                if review_ref:
                    return review_ref
        return None

    async def _review_selected_whole_for_report(self, session_id, memory):
        """A component closeout or later Method cannot replace the final result's reflection."""
        from autoresearch.experiments.repo_improve.execution_policy import method_feedback_limit
        selected_ref = memory.get("best_result_ref")
        if not method_feedback_limit() or not selected_ref:
            return
        selected = self.spec_store.get(selected_ref)
        if not (can_revalidate_accepted_whole_result(selected)
                and _formal_best_eligible(selected)):
            return

        def reviewed():
            return any(
                (review := self.spec_store.get(ref)).get("status") == "completed"
                and review.get("trigger") in {"complete_result", "complete_metric_regression"}
                and review.get("result_ref") == selected_ref
                and review.get("method_ref") == selected.get("method_ref")
                for ref in memory.get("method_feedback_refs") or [])

        if reviewed():
            return
        run_ref = selected["run_record_snapshot_ref"]
        run_digest = artifact_digest(self.spec_store.get(run_ref))
        verification = None
        for ref in self.spec_store.list_refs(session_id, "verification_report"):
            report = self.spec_store.get(ref)
            if (refs.decode(ref).key != "snapshot-" + artifact_digest(report).split(":", 1)[1]
                    or report.get("verdict") != "verified" or report.get("anti_cheat_flags")
                    or report.get("candidate_snapshot_ref") != selected_ref
                    or report.get("run_record_ref") != run_ref
                    or report.get("run_record_digest") != run_digest):
                continue
            evaluation_ref = report.get("evaluation_report_ref")
            evaluation = self.spec_store.get(evaluation_ref) if evaluation_ref else {}
            if (not evaluation_ref
                    or refs.decode(evaluation_ref).key != "snapshot-" + artifact_digest(evaluation).split(":", 1)[1]
                    or evaluation.get("run_record_ref") != run_ref
                    or evaluation.get("run_record_digest") != run_digest
                    or evaluation.get("verdict") in {"invalid", "suspicious"}
                    or evaluation.get("anti_cheat_flags")):
                continue
            verification = report
            break
        if verification is None:
            raise RuntimeError("selected whole result has no bound verified evaluation for reflection")
        await self._review_complete_result(
            session_id, run_ref, memory, verification, "valid_improvement",
            selected.get("baseline_result_ref"), closeout_requested=True,
            selected_result_closeout=True)
        if not reviewed():
            raise RuntimeError("selected whole result reflection is unavailable; reporting cannot proceed")

    async def _review_complete_result(self, session_id, run_ref, memory, verification,
                                      diagnosis, baseline_ref=None, review_method_ref=None,
                                      closeout_requested=False, selected_result_closeout=False,
                                      retained_source_review=False):
        """Review bound Method evidence within the existing budget; replay is safe."""
        from autoresearch.experiments.repo_improve.execution_policy import (
            efficient_execution, method_feedback_limit)
        from autoresearch.spec_store.identity import result_for_run
        from autoresearch.agents.reflection_agent.core import review_complete_method
        limit = method_feedback_limit()
        if not limit or (verification or {}).get("verdict") != "verified":
            return None
        gen = result_for_run(self.spec_store, run_ref)
        import_violation = _result_import_violation(gen)
        baseline = self.spec_store.get(baseline_ref) if baseline_ref else None
        best_ref = memory.get("best_result_ref")
        best = self.spec_store.get(best_ref) if best_ref else None
        closeout_ref = refs.encode("human_idea_seed", session_id, "closeout-v1")
        try:
            closeout = self.spec_store.get(closeout_ref)
        except KeyError:
            closeout = {}
        user_closeout = closeout.get("closeout") is True
        closeout_snapshot_ref = self.spec_store.snapshot(closeout_ref) if user_closeout else None
        execution = gen.get("component_execution") or {}
        closeout_components = None
        if closeout_requested or user_closeout or execution.get("status") == "exhausted":
            closeout_components = _qualified_closeout_components(
                memory, artifact_digest(baseline) if isinstance(baseline, dict) else None,
                store=self.spec_store, incumbent=best,
                current=gen if (can_revalidate_accepted_whole_result(gen)
                    or execution.get("status") in {"execution_failed", "implementation_failed"}) else None)
        if selected_result_closeout:
            closeout_components = []  # Reflect the final selection; source adoption is already over.
        integration_parent = _qualified_integration_parent(self.spec_store, gen, memory)
        trigger = _method_review_trigger(
            gen, memory, diagnosis, best, baseline, integration_parent, closeout_requested,
            user_closeout, closeout_components)
        terminal_whole_closeout = bool((closeout_requested or user_closeout)
                and trigger in {
                    "complete_result", "complete_metric_regression",
                    "complete_verified_no_gain"}
                and (gen.get("method_implementation_review") or {}).get("verdict") == "faithful")
        closing = (terminal_whole_closeout or trigger == "qualified_component_closeout" or (
            trigger in {"component_method_reconciliation", "legal_off_method_gain"}
            and (closeout_requested or user_closeout or execution.get("status") == "exhausted")))
        if trigger == "repeated_verified_no_gain" and efficient_execution():
            # The bounded component choice owns improve/advance and retains its best.
            return None
        if not trigger or (trigger == "complete_result" and (
                not isinstance(best, dict) or best.get("accepted") is not True
                or best.get("method_metric") != memory.get("best_metric"))):
            return None
        if trigger == "complete_verified_no_gain" and best is None:
            best_ref, best = baseline_ref, baseline
        if trigger in {"implementation_failure", "qualified_component_closeout"}:
            run = self.spec_store.get(run_ref)
            if (verification.get("anti_cheat_flags")
                    or verification.get("run_record_ref") != run_ref
                    or verification.get("run_record_digest") != artifact_digest(run)
                    or verification.get("candidate_snapshot_ref") != gen.get("candidate_snapshot_ref")
                    or (trigger == "implementation_failure"
                        and (gen.get("method_implementation_review") or {}).get("verdict") == "partial"
                        and gen["method_implementation_review"].get("method_ref") != run.get("experiment_id"))
                    or (trigger == "implementation_failure" and review_method_ref
                        and review_method_ref != run.get("experiment_id"))):
                return None
            if best is None and isinstance(baseline, dict) and baseline.get("reproduced") is True:
                best_ref, best = baseline_ref, baseline
        if trigger in {"mir_task_contract_violation", "component_method_reconciliation"}:
            review = gen.get("method_implementation_review") or {}
            run = self.spec_store.get(run_ref)
            if (review.get("method_ref") != run.get("experiment_id")
                    or (trigger == "mir_task_contract_violation"
                        and review.get("method_ref") != (review_method_ref or run.get("experiment_id")))
                    or verification.get("anti_cheat_flags")
                    or verification.get("run_record_ref") != run_ref
                    or verification.get("run_record_digest") != artifact_digest(run)
                    or verification.get("candidate_snapshot_ref") != gen.get("candidate_snapshot_ref")):
                return None
            if best is None and isinstance(baseline, dict) and baseline.get("reproduced") is True:
                best_ref, best = baseline_ref, baseline
        if trigger == "legal_off_method_gain" and best is None:
            if (verification.get("anti_cheat_flags") or any(
                    "failure_confirmed" in str(check)
                    for check in verification.get("checks") or [])):
                return None
            evaluation_ref = verification.get("evaluation_report_ref")
            evaluation = self.spec_store.get(evaluation_ref) if evaluation_ref else None
            if (not isinstance(evaluation, dict)
                    or evaluation.get("verdict") in {"invalid", "suspicious"}
                    or evaluation.get("anti_cheat_flags")
                    or evaluation.get("run_record_ref") != run_ref
                    or evaluation.get("run_record_digest") != artifact_digest(
                        self.spec_store.get(run_ref))
                    or verification.get("candidate_snapshot_ref") != gen.get("candidate_snapshot_ref")):
                return None
            if best is None:
                best_ref, best = baseline_ref, baseline
        run = self.spec_store.get(run_ref)
        source_method_ref = run.get("experiment_id")
        method_ref = review_method_ref or source_method_ref
        if not method_ref:
            return None
        method = self.spec_store.get(method_ref)
        current_ref = gen.get("candidate_snapshot_ref")
        if not current_ref or self.spec_store.get(current_ref) != {
                k: v for k, v in gen.items() if k != "candidate_snapshot_ref"}:
            return None
        review_for_evidence = gen.get("method_implementation_review") or {}
        if import_violation:
            review_for_evidence = {
                **review_for_evidence, "task_contract_review": import_violation}
        existing = self.spec_store.list_refs(session_id, "method_improvement_review")
        if trigger in {"component_method_reconciliation", "legal_off_method_gain"}:
            prior = [saved for ref in existing
                     if (saved := self.spec_store.get(ref)).get("status") == "completed"][:3]
            closing = closing or any(
                saved.get("closeout_requested") is True
                or saved.get("trigger") == "qualified_component_closeout"
                for saved in prior)
        def unavailable(saved):
            decision = saved.get("decision") or {}
            return (decision.get("route") == "unavailable"
                    or (saved.get("trigger") in {"component_method_reconciliation",
                                                 "qualified_component_closeout"}
                        and decision.get("route") == "report")
                    or str(decision.get("reason") or "").startswith("review unavailable:"))

        for ref in existing:
            saved = self.spec_store.get(ref)
            if saved.get("result_ref") == current_ref and saved.get("status") == "started":
                owner = _ACTIVE_COMPLETE_REVIEWS.get(ref)
                if owner is not None and not owner.done():
                    raise RuntimeError("complete-method review is unfinished; decision unavailable")
                saved = {**saved, "status": "failed", "decision": {
                    "route": "unavailable", "reason": "review unavailable: interrupted owner"}}
                self.spec_store.put(
                    kind="method_improvement_review", session_id=session_id,
                    key=refs.decode(ref).key, obj=saved, emitter=AGENT_NAME)
            if (saved.get("result_ref") == current_ref
                    and saved.get("incumbent_ref") == best_ref
                    and (saved.get("method_ref") == method_ref
                         or (retained_source_review
                             and closing and trigger in {
                                 "qualified_component_closeout",
                                 "component_method_reconciliation", "legal_off_method_gain"}
                             and saved.get("baseline_ref") == baseline_ref))
                    and saved.get("trigger") == trigger
                    and (saved.get("user_closeout") or {}).get("ref") == closeout_snapshot_ref
                    and (trigger != "qualified_component_closeout"
                         or saved.get("qualified_closeout_components") == closeout_components)
                    and (terminal_whole_closeout or (
                        bool(saved.get("closeout_requested"))
                        or saved.get("trigger") == "qualified_component_closeout") == closing)):
                if saved.get("status") in {"completed", "failed"} and unavailable(saved):
                    # Preserve protocol failures, then allow a repaired review on event replay.
                    continue
                memory["method_feedback_refs"] = list(dict.fromkeys(
                    list(memory.get("method_feedback_refs") or []) + [ref]))
                # A historical source already reviewed against this incumbent must
                # not preempt the current result merely because Method metadata changed.
                return ref if saved.get("method_ref") == method_ref and not terminal_whole_closeout and saved.get("status") == "completed" and saved.get(
                    "decision", {}).get("route") == "revise_method" else None
        if (closing
                and not terminal_whole_closeout
                # Closeout is idempotent per retained candidate, not merely per current
                # Method.  A different safe source discovered earlier must still receive
                # the terminal incumbent comparison after that Method's prior closeout.
                and not (trigger == "legal_off_method_gain"
                         or (trigger == "component_method_reconciliation"
                             and can_revalidate_reconciliation_candidate(gen)))
                and any(
                (self.spec_store.get(ref).get("closeout_requested") is True
                 or self.spec_store.get(ref).get("trigger") == "qualified_component_closeout")
                and self.spec_store.get(ref).get("status") == "completed"
                and not unavailable(self.spec_store.get(ref))
                and (self.spec_store.get(ref).get("user_closeout") or {}).get("ref")
                    == closeout_snapshot_ref
                and (trigger != "qualified_component_closeout"
                     or (self.spec_store.get(ref).get("trigger") == trigger
                         and self.spec_store.get(ref).get("qualified_closeout_components")
                             == closeout_components))
                and any(self.spec_store.get(ref).get(key)
                        and self.spec_store.get(self.spec_store.get(ref)[key]) == method
                        for key in ("method_ref", "source_method_ref"))
                for ref in existing)):
            return None  # One pass per unchanged Method when there is no distinct safe candidate.
        # The configured feedback rounds bound completed whole-method reviews.
        # Intermediate diagnostic reviews and protocol failures do not consume them.
        completed_full_reviews = len({
            artifact_digest(saved)
            for saved in (self.spec_store.get(ref) for ref in existing)
            if saved.get("status") == "completed"
            and saved.get("trigger", "complete_result") in {
                "complete_result", "complete_metric_regression",
                "complete_verified_no_gain"}
            and not unavailable(saved)})
        # Final delivery auditing cannot open another research round.
        if (trigger in {
                "complete_result", "complete_metric_regression",
                "complete_verified_no_gain"} and completed_full_reviews >= limit
                and not selected_result_closeout):
            return None
        incumbent_metric = (best.get("method_metric", best.get("metric_value"))
                            if isinstance(best, dict) else None)
        record = {"status": "started", "method_ref": method_ref,
                  "source_method_ref": source_method_ref,
                  "trigger": trigger,
                  "result_ref": current_ref, "incumbent_ref": best_ref,
                  "incumbent_metric": incumbent_metric,
                  "baseline_ref": baseline_ref}
        if closing:
            record["closeout_requested"] = True
        if terminal_whole_closeout:
            record["terminal_whole_closeout"] = True
        if selected_result_closeout:
            record["selected_result_closeout"] = True
        key = f"slot-{len(existing) + 1}"
        ref = self.spec_store.put(kind="method_improvement_review", session_id=session_id,
                                  key=key, obj=record, emitter=AGENT_NAME)
        import time
        started = time.monotonic()
        evidence = {current_ref: {
            k: gen.get(k) for k in ("changes_summary", "method_metric", "baseline_metric",
                                   "ablation_report", "method_implementation_review", "reasons",
                                   "off_method_observation", "component_execution",
                                   "measurement_evidence")}}
        if user_closeout:
            evidence["user_closeout"] = {**closeout, "ref": closeout_snapshot_ref}
            record["user_closeout"] = evidence["user_closeout"]
        if closing:
            evidence["closeout_requested"] = True
        if terminal_whole_closeout:
            evidence["terminal_whole_closeout"] = True
        evidence[current_ref]["method_implementation_review"] = review_for_evidence
        def excerpt(value, limit, full_ref):
            content = str(value or "")
            if len(content) <= limit:
                return content
            half = limit // 2
            return (content[:half] + f"\n[truncated; full content in {full_ref}]\n"
                    + content[-half:])

        evidence[current_ref]["implementation_feedback"] = {
            "evidence_status": "runner-reported; not independent profiling",
            "rationale": excerpt(gen.get("rationale"), 4000, current_ref),
            "bottleneck": excerpt(gen.get("bottleneck"), 4000, current_ref),
            "config": excerpt(json.dumps(
                gen.get("config") or {}, ensure_ascii=False), 6000, current_ref),
            "diff": excerpt(gen.get("diff"), 12000, current_ref),
            "parent_relative_diff": excerpt(gen.get("parent_relative_diff"), 6000, current_ref),
            "run_record_ref": run_ref,
            "stdout_excerpt": excerpt(run.get("stdout_excerpt"), 8000, run_ref),
            "stderr_excerpt": excerpt(run.get("stderr_excerpt"), 4000, run_ref),
        }
        if isinstance(best, dict):
            evidence["incumbent_implementation"] = {
                "ref": best_ref, "checkpoint": best.get("checkpoint"),
                "diff": excerpt(best.get("diff"), 12000, best_ref),
                "measurement_evidence": best.get("measurement_evidence"),
                "config": excerpt(json.dumps(best.get("config") or {}, ensure_ascii=False), 6000, best_ref),
            }
        if isinstance(baseline, dict):
            contract = baseline.get("repository_contract") or {}
            sources = contract.get("sources") or []
            evidence["task_requirements"] = {
                "baseline_ref": baseline_ref,
                "metric_name": baseline.get("metric_name"),
                "metric_value": baseline.get("metric_value"),
                "metric_unit": baseline.get("metric_unit"),
                "lower_is_better": baseline.get("lower_is_better"),
                "fixed_config": baseline.get("fixed_config"),
                "repo_commit": baseline.get("repo_commit"),
                "benchmark_contract": excerpt(json.dumps(
                    baseline.get("benchmark_contract"), ensure_ascii=False), 6000, baseline_ref),
                "execution_contract": excerpt(json.dumps(
                    contract.get("execution_contract"), ensure_ascii=False), 6000, baseline_ref),
                "repository_sources": [{
                    "path": source.get("path"), "sha256": source.get("sha256"),
                    "content": excerpt(source.get("content"), 6000, baseline_ref),
                } for source in sources[:8] if isinstance(source, dict)],
            }
            try:
                semantics = self.spec_store.get(refs.encode(
                    "spec_interpretation", session_id, "v1")).get("evaluation_semantics")
            except (KeyError, TypeError, ValueError):
                semantics = None
            if semantics:
                evidence["task_requirements"]["evaluation_semantics"] = excerpt(
                    json.dumps(semantics, ensure_ascii=False), 6000,
                    refs.encode("spec_interpretation", session_id, "v1"))
            record["task_requirements"] = evidence["task_requirements"]
        evidence["trigger"] = {"kind": trigger, "result_ref": current_ref,
                               "diagnosis": diagnosis,
                               "source_method_ref": source_method_ref,
                               "review_method_ref": method_ref}
        if trigger == "qualified_component_closeout":
            evidence["qualified_closeout_components"] = closeout_components
            if not evidence["qualified_closeout_components"]:
                raise RuntimeError("closeout components do not match the frozen baseline")
            record["qualified_closeout_components"] = evidence["qualified_closeout_components"]
        if trigger == "complete_metric_regression" and integration_parent is not None:
            parent_ref = integration_parent["candidate_snapshot_ref"]
            evidence["integration_parent"] = {
                "ref": parent_ref, "checkpoint": integration_parent["checkpoint"],
                "method_metric": integration_parent["method_metric"],
                "measurement_evidence": integration_parent.get("measurement_evidence"),
                "implementation_changed": True,
                "baseline_relative_diff": excerpt(
                    integration_parent["baseline_relative_diff"], 12000, parent_ref),
                "complete_baseline_relative_diff": excerpt(gen.get("baseline_relative_diff"), 12000, current_ref)}
            record["integration_parent"] = evidence["integration_parent"]
        if trigger == "component_method_reconciliation":
            evidence["trigger"].update({
                "score_admissible": False,
                "validation_required": "Reconcile the frozen Method and reverify the affected "
                                       "components and complete official result before adoption."})
        recent_history, historical_best = _method_feedback_component_evidence(memory)
        evidence["component_history"] = recent_history
        evidence["prior_method_reflections"] = [
            {"ref": old_ref, **{key: saved.get(key) for key in (
                "method_ref", "result_ref", "trigger", "decision", "closeout_requested")}}
            for old_ref in existing
            if (saved := self.spec_store.get(old_ref)).get("status") == "completed"
        ][:3]
        if historical_best:
            evidence["historical_best_component_evidence"] = historical_best
        if trigger == "mir_task_contract_violation":
            evidence["mir_contract_violation"] = {
                "review": review_for_evidence,
                "result_ref": current_ref,
                "run_record_ref": run_ref,
                "diff_ref": (gen.get("component_execution") or {}).get(
                    "attempt_diff_ref", (gen.get("component_execution") or {}).get("diff_ref")),
                "diff": excerpt(gen.get("diff"), 12000, current_ref),
                "parent_relative_diff": excerpt(gen.get("parent_relative_diff"), 6000, current_ref),
                "stdout_excerpt": excerpt(run.get("stdout_excerpt"), 8000, run_ref),
                "stderr_excerpt": excerpt(run.get("stderr_excerpt"), 4000, run_ref),
                "verification": {key: verification.get(key) for key in (
                    "verdict", "checks", "reasoning", "evaluation_report_ref")},
                "score_admissible": False,
            }
            record["mir_violation_evidence"] = evidence["mir_contract_violation"]
        if trigger in {"legal_off_method_gain", "component_method_reconciliation"}:
            def observation(result, result_ref):
                result_run_ref = result.get("run_record_snapshot_ref")
                try:
                    result_run = self.spec_store.get(result_run_ref) if result_run_ref else {}
                except KeyError:
                    result_run = {}
                execution = result.get("component_execution") or {}
                return {
                    "result_ref": result_ref,
                    "run_record_ref": result_run_ref,
                    "method_ref": result_run.get("experiment_id"),
                    "checkpoint": result.get("checkpoint"),
                    "diff_ref": execution.get("attempt_diff_ref", execution.get("diff_ref")),
                    "diff": excerpt(result.get("diff"), 20000, result_ref),
                    "parent_relative_diff": excerpt(
                        result.get("parent_relative_diff"), 8000, result_ref),
                    "changes_summary": result.get("changes_summary"),
                    "rationale": result.get("rationale"),
                    "bottleneck": result.get("bottleneck"),
                    "metric": result.get("method_metric", result.get("metric_value")),
                    "stdout_excerpt": excerpt(
                        result_run.get("stdout_excerpt") or result.get("log_excerpt"),
                        16000, result_run_ref or result_ref),
                    "stderr_excerpt": excerpt(
                        result_run.get("stderr_excerpt"), 8000, result_run_ref),
                    "exit_code": result_run.get("exit_code"),
                }

            comparison = {
                "candidate": observation(gen, current_ref),
                "incumbent": observation(best, best_ref),
                "verification": {
                    key: verification.get(key) for key in (
                        "verdict", "checks", "anti_cheat_flags", "reasoning",
                        "run_record_ref", "run_record_digest", "evaluation_report_ref",
                        "candidate_snapshot_ref")},
            }
            if source_method_ref and source_method_ref != method_ref:
                try:
                    source_method = self.spec_store.get(source_method_ref)
                except KeyError:
                    source_method = None
                if source_method is not None:
                    source_text = json.dumps(source_method, ensure_ascii=False)
                    comparison["source_method"] = {
                        "ref": source_method_ref,
                        "content": excerpt(source_text, 30000, source_method_ref),
                    }
            evaluation_ref = verification.get("evaluation_report_ref")
            if evaluation_ref:
                try:
                    evaluation = self.spec_store.get(evaluation_ref)
                except KeyError:
                    evaluation = None
                if evaluation is not None:
                    comparison["evaluation"] = {
                        "ref": evaluation_ref,
                        "report": excerpt(
                            json.dumps(evaluation, ensure_ascii=False), 12000, evaluation_ref),
                    }
            evidence["off_method_comparison"] = comparison
            record["off_method_evidence"] = comparison
        task = asyncio.create_task(asyncio.to_thread(
            review_complete_method, method,
            {"ref": best_ref, "metric": incumbent_metric,
             "checkpoint": (best or {}).get("checkpoint"),
             "lower_is_better": gen.get("lower_is_better")},
            evidence, self.llm_client))
        _ACTIVE_COMPLETE_REVIEWS[ref] = task
        try:
            decision = await asyncio.shield(task)
        finally:
            if task.done():
                _ACTIVE_COMPLETE_REVIEWS.pop(ref, None)
        source_closeout_repair = bool(
            closing and trigger == "component_method_reconciliation"
            and can_revalidate_reconciliation_candidate(gen)
            and decision.get("route") in {"revise_method", "report"}
            and not _decision_retains_revalidation_source(decision, gen)
        )
        if (closing and not terminal_whole_closeout and (
                (decision.get("route") == "unavailable"
                 and decision.get("validation_failed") is True)
                or source_closeout_repair)):
            # Only bind evidence already checked above; this policy does not accept a score.
            # A component-parent or Method-attribution failure can invalidate the old scalar,
            # but cannot silently discard legal source that passed the source-safety gate.
            record["advisory_review_failure"] = decision
            record["decision_source"] = (
                "host_revalidation_closeout_policy" if source_closeout_repair
                else "host_closeout_policy")
            decision = {
                "route": "revise_method", "dimensions": ["mechanism_understanding"],
                "primary_hypothesis": (
                    "The exact task-compliant measured checkpoint must be reconciled as source "
                    "and freshly validated; its historical component score remains unaccepted."
                ),
                "evidence_refs": ["qualified_closeout_components"]
                                 if trigger == "qualified_component_closeout" else [current_ref],
                "retain": [
                    "The exact retained implementation at " + str(checkpoint)
                    + " and its frozen task constraints."
                    for checkpoint in (
                        list(dict.fromkeys(item["checkpoint"] for item in closeout_components))
                        if trigger == "qualified_component_closeout" else [gen["checkpoint"]]
                    )
                ],
                "modify": ["Bind the Method and component contracts to the retained implementation; "
                           "remove unimplemented extensions without adding mechanisms."],
                "add": [], "remove": [], "affected_dependencies": [],
                "reuse": list(dict.fromkeys(item["checkpoint"] for item in closeout_components))
                         if trigger == "qualified_component_closeout" else [gen["checkpoint"]],
                "requires_core_reconstruction": False,
                "expected_observation": "Full official validation establishes the actual complete result.",
                "validation_plan": "Restore the exact retained sources and available runtime artifacts; "
                                   "reverify affected components and run complete official evaluation with "
                                   "all task, measurement, and evidence binding gates unchanged.",
                "stop_condition": "Report the verified result or the concrete validation failure; "
                                  "do not resume exploration.",
            }
        record.update(status=("failed" if decision.get("route") == "unavailable"
                              else "completed"), decision=decision,
                      elapsed_seconds=time.monotonic() - started)
        self.spec_store.put(kind="method_improvement_review", session_id=session_id,
                            key=key, obj=record, emitter=AGENT_NAME)
        memory["method_feedback_refs"] = list(dict.fromkeys(
            list(memory.get("method_feedback_refs") or []) + [ref]))
        if decision.get("route") == "unavailable":
            raise RuntimeError(decision.get("error_detail") or decision.get("reason"))
        return ref if decision.get("route") == "revise_method" else None

    # ---- failure path / 失败路径 ----

    async def _fail(
        self,
        client: Any,
        session_id: str,
        causation_id: str | None,
        reason_classification: str,
        underlying_reason_code: str,
        underlying_reason: str,
    ) -> None:
        await client.publish(
            event_type="agent.failed",
            session_id=session_id,
            payload={
                "agent_name": AGENT_NAME,
                "reason_classification": reason_classification,
                "underlying_reason": underlying_reason,
                "underlying_reason_code": underlying_reason_code,
            },
            causation_id=causation_id,
        )


__all__ = ["MemoryAgentHandler", "SCHEMA_VERSION"]
