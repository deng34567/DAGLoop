"""
EvaluatorAgent EventBus handler: the thin transport wrapper around core.
EvaluatorAgent EventBus handler：core 之上的薄传输层。

Subscribes FOUR events:

1. ``benchmark.selection_requested`` (payload ``{research_spec_ref,
   literature_brief_ref}``) -- reads the ResearchSpec + LiteratureBrief from
   the Spec Store by ref (provenance -- NEVER from the event payload), runs
   ``core.select_benchmark``, persists the resulting BenchmarkSpec
   (kind="benchmark_spec", key="v1"), and publishes ``benchmark.spec_ready``
   with its benchmark_id + ref.
2. ``verifier.preparation_requested`` (payload ``{benchmark_spec_ref,
   research_spec_ref}``) -- reads the BenchmarkSpec + ResearchSpec from the
   Spec Store by ref, runs ``core.prepare_verifier`` (which returns a
   ``(verifier_spec, verdict)`` tuple), persists the resulting VerifierSpec
   (kind="verifier_spec", key="v1"), and publishes ``verifier.spec_frozen``
   with its verdict + ref.
3. ``experiment.evaluation_requested`` (payload ``{run_record_ref,
   verifier_spec_ref, benchmark_spec_ref, research_memory_ref?}``) -- reads
   the RunRecord + VerifierSpec + BenchmarkSpec (+ prior ResearchMemory, if
   ``research_memory_ref`` is truthy) from the Spec Store by ref, runs
   ``core.evaluate_experiment``, persists the resulting EvaluationReport
   (kind="evaluation_report", key="v1"), and publishes ``experiment.evaluated``
   with its verdict + ref.
4. ``result.verification_requested`` (payload ``{evaluation_report_ref,
   run_record_ref, verifier_spec_ref}``) -- reads the EvaluationReport +
   RunRecord + VerifierSpec from the Spec Store by ref, runs
   ``core.verify_best_result``, persists the resulting VerificationReport
   (kind="verification_report", key="v1"), and publishes ``result.verified``
   with its verdict + ref.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅四个事件：benchmark.selection_requested（按 ref 从 Spec Store 读回
ResearchSpec + LiteratureBrief，跑一次 core.select_benchmark，持久化
BenchmarkSpec，发布 benchmark.spec_ready）、verifier.preparation_requested
（按 ref 读回 BenchmarkSpec + ResearchSpec，跑一次 core.prepare_verifier
（返回 (verifier_spec, verdict) 元组），持久化 VerifierSpec，发布
verifier.spec_frozen）、experiment.evaluation_requested（按 ref 读回
RunRecord + VerifierSpec + BenchmarkSpec，跑一次 core.evaluate_experiment，
持久化 EvaluationReport，发布 experiment.evaluated）、
result.verification_requested（按 ref 读回 EvaluationReport + RunRecord +
VerifierSpec，跑一次 core.verify_best_result，持久化 VerificationReport，
发布 result.verified）。core 的任何失败（或未预期错误）都转成
agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
import math
from typing import Any

from autoresearch.agents.evaluator_agent import core
from autoresearch.agents.comparison_guardian_agent import core as comparison_guardian
from autoresearch.agents.evaluator_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.experiments.repo_improve.benchmark_contract import (
    reconcile_repository_evaluation_semantics,
)
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.spec_store import refs

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_BENCHMARK_SPEC_KEY = "v1"
_VERIFIER_SPEC_KEY = "v1"
_EVALUATION_REPORT_KEY = "v1"
_VERIFICATION_REPORT_KEY = "v1"
_VALIDATION_RESULT_KIND = "validation_study_result"
_GEN_RESULT_KIND = "gen_experiment_result"
_GEN_CURRENT_RESULT_KEY = "current"


def _finite_number(value: Any) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        return None
    return float(value)


def _verify_indirect_component(
    evaluation_report: dict,
    run_record: dict,
    gen_result: dict | None,
) -> dict | None:
    """Verify an intermediate check's recorded outcome, not task performance."""
    if not isinstance(gen_result, dict):
        return None
    component = gen_result.get("component_execution")
    review = gen_result.get("method_implementation_review")
    if not isinstance(component, dict) or not isinstance(review, dict):
        return None
    contract = component.get("attempt_contract_result")
    if not isinstance(contract, dict):
        return None
    passed_check = (
        component.get("status") in {"qualified", "improving", "accepted"}
        and contract.get("passed") is True
    )
    failed_check = (
        component.get("status") == "contract_failed"
        and contract.get("passed") is False
        and _finite_number(contract.get("actual")) is not None
        and isinstance(contract.get("expected"), dict)
        and bool(str(contract.get("reason") or "").strip())
    )
    if not (
        component.get("mode") == "component"
        and component.get("role") == "indirect_gain"
        and component.get("execution_succeeded") is True
        and (passed_check or failed_check)
        and contract.get("measurement") == "indirect_gain_check"
        and review.get("verdict") in {"faithful", "partial"}
        and run_record.get("succeeded") is True
        and run_record.get("exit_code") == 0
        and run_record.get("metric_value") is None
        and evaluation_report.get("verdict") == "valid_not_best"
        and evaluation_report.get("metric_value") is None
        and evaluation_report.get("is_improvement") is False
    ):
        return None
    report_baseline = _finite_number(evaluation_report.get("baseline_metric"))
    result_baseline = _finite_number(gen_result.get("baseline_metric"))
    if report_baseline is None or result_baseline is None or report_baseline != result_baseline:
        return None
    evidence = contract.get("evidence")
    if not isinstance(evidence, dict) or not str(evidence.get("ref") or "").strip():
        return None
    frozen_digest = evaluation_report.get("run_record_digest")
    if frozen_digest is not None and frozen_digest != artifact_digest(run_record):
        return None
    if failed_check:
        return {
            "verdict": "verified",
            "checks": [
                "indirect_component_execution_succeeded",
                "indirect_component_failure_confirmed_with_grounded_evidence",
                "primary_task_metric_intentionally_not_claimed",
            ],
            "anti_cheat_flags": [],
            "reasoning": (
                f"Recorded component failure: {contract['reason']}; "
                f"actual={contract['actual']}, expected={contract['expected']}; "
                f"evidence={evidence['ref']}. This verifies the failed check's record, "
                "not component acceptance or task-metric improvement."
            ),
        }
    return {
        "verdict": "verified",
        "checks": [
            "indirect_component_execution_succeeded",
            "method_implementation_review_passed",
            "indirect_component_contract_passed_with_grounded_evidence",
            "primary_task_metric_intentionally_not_claimed",
        ],
        "anti_cheat_flags": [],
        "reasoning": (
            "The Runner-owned component result proves the indirect component's declared "
            "intermediate contract. No primary task-metric improvement is attributed to this "
            "component attempt."
        ),
    }


class EvaluatorAgentHandler:
    """Wires core.select_benchmark/prepare_verifier to a Spec Store + LLM +
    EventBus client.
    把 core.select_benchmark/prepare_verifier 接到 Spec Store + LLM +
    EventBus 客户端。"""

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

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler's four events on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的四个事件。"""
        client.subscribe(
            "benchmark.selection_requested",
            self._make_eventbus_handler(client, self.handle_benchmark_selection),
        )
        client.subscribe(
            "verifier.preparation_requested",
            self._make_eventbus_handler(client, self.handle_verifier_preparation),
        )
        client.subscribe(
            "experiment.evaluation_requested",
            self._make_eventbus_handler(client, self.handle_evaluation_requested),
        )
        client.subscribe(
            "result.verification_requested",
            self._make_eventbus_handler(client, self.handle_verification_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- benchmark.selection_requested ----

    async def handle_benchmark_selection(self, client: Any, env: Any) -> None:
        """Resolve the ResearchSpec + LiteratureBrief from the store, run one
        select_benchmark, persist the BenchmarkSpec, publish
        benchmark.spec_ready.
        从 store 解析 ResearchSpec + LiteratureBrief，跑一次
        select_benchmark，持久化 BenchmarkSpec，发布 benchmark.spec_ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            literature_brief = self.spec_store.get(payload.get("literature_brief_ref"))
            spec_interpretation = self.spec_store.get(payload.get("spec_interpretation_ref"))
            evaluation_semantics = spec_interpretation.get("evaluation_semantics")
            if not isinstance(evaluation_semantics, dict):
                raise ValueError("SpecInterpretation is missing evaluation_semantics")
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            benchmark_spec = await asyncio.to_thread(
                core.select_benchmark,
                research_spec, literature_brief, self.llm_client,
                evaluation_semantics=evaluation_semantics,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvaluatorAgent", self.config.prompt_patch()),
            )
        except core.EvaluatorError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        benchmark_spec_ref = self.spec_store.put(
            kind="benchmark_spec",
            session_id=session_id,
            key=_BENCHMARK_SPEC_KEY,
            obj=benchmark_spec,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="benchmark.spec_ready",
            session_id=session_id,
            payload={
                "benchmark_spec_ref": benchmark_spec_ref,
                "benchmark_id": benchmark_spec["benchmark_id"],
            },
            causation_id=causation_id,
        )

    # ---- verifier.preparation_requested ----

    async def handle_verifier_preparation(self, client: Any, env: Any) -> None:
        """Resolve the BenchmarkSpec + ResearchSpec from the store, run one
        prepare_verifier, persist the VerifierSpec, publish
        verifier.spec_frozen.
        从 store 解析 BenchmarkSpec + ResearchSpec，跑一次 prepare_verifier，
        持久化 VerifierSpec，发布 verifier.spec_frozen。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            benchmark_spec = self.spec_store.get(payload.get("benchmark_spec_ref"))

            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            verifier_spec, verdict = await asyncio.to_thread(
                core.prepare_verifier,
                benchmark_spec, research_spec, self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvaluatorAgent", self.config.prompt_patch()),
            )
        except core.EvaluatorError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        verifier_spec_ref = self.spec_store.put(
            kind="verifier_spec",
            session_id=session_id,
            key=_VERIFIER_SPEC_KEY,
            obj=verifier_spec,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="verifier.spec_frozen",
            session_id=session_id,
            payload={
                "verifier_spec_ref": verifier_spec_ref,
                "verdict": verdict,
            },
            causation_id=causation_id,
        )

    # ---- experiment.evaluation_requested ----

    async def handle_evaluation_requested(self, client: Any, env: Any) -> None:
        """Resolve the RunRecord + VerifierSpec + BenchmarkSpec from the
        store, run one evaluate_experiment, persist the EvaluationReport,
        publish experiment.evaluated.
        从 store 解析 RunRecord + VerifierSpec + BenchmarkSpec，跑一次
        evaluate_experiment，持久化 EvaluationReport，发布
        experiment.evaluated。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            run_record = self.spec_store.get(payload.get("run_record_ref"))
            verifier_spec = ((self.spec_store.get(payload.get("verifier_spec_ref")) or {}) if payload.get("verifier_spec_ref") else {})
            benchmark_ref = payload.get("benchmark_spec_ref")
            benchmark_spec = self.spec_store.get(benchmark_ref) if benchmark_ref else {}
            interpretation_ref = payload.get("spec_interpretation_ref")
            interpretation = (
                self.spec_store.get(interpretation_ref) if interpretation_ref else {})
            validation_study = (
                interpretation.get("validation_study")
                if isinstance(interpretation, dict) else None)

            baseline_result_ref = payload.get("baseline_result_ref")
            baseline_result = (
                self.spec_store.get(baseline_result_ref) if baseline_result_ref else None
            )
            if isinstance(benchmark_spec, dict):
                benchmark_spec = dict(benchmark_spec)
                benchmark_spec["evaluation_semantics"] = (
                    reconcile_repository_evaluation_semantics(
                        benchmark_spec.get("evaluation_semantics"), baseline_result))

            research_memory_ref = payload.get("research_memory_ref")
            prior_research_memory = (
                self.spec_store.get(research_memory_ref) if research_memory_ref else None
            )

            # Repository runs already carry the Runner-owned, measured outcome in their
            # environment marker.  Their scientific comparison was frozen before execution;
            # asking an LLM to reinterpret timing/config text after the run can only replace a
            # real recorded result with an opinion.  In particular, a final training step may
            # cross the nominal timer by a few seconds while remaining the repository harness's
            # completed observation.  Let the deterministic repo evaluator classify it, then let
            # result verification check the exact metric or recorded failure cause.
            repository_run = (
                isinstance(run_record.get("environment"), str)
                and "repo_improve|" in run_record["environment"]
            )
            if isinstance(validation_study, dict):
                try:
                    validation_result = self.spec_store.get(refs.encode(
                        _VALIDATION_RESULT_KIND, session_id, "v1"))
                except KeyError:
                    validation_result = None
                run_metric = _finite_number(run_record.get("metric_value"))
                result_metric = _finite_number(
                    validation_result.get("metric_value")
                    if isinstance(validation_result, dict) else None)
                valid = bool(
                    run_record.get("succeeded") is True
                    and run_record.get("exit_code") == 0
                    and run_record.get("environment") == "survey_validation|host_rerun"
                    and result_metric is not None
                    and run_metric == result_metric
                    and isinstance(validation_result.get("sample_ids"), list)
                    and bool(validation_result["sample_ids"])
                    and validation_result.get("conclusion") in {
                        "supports", "contradicts", "inconclusive"})
                evaluation_report = {
                    "verdict": "valid_not_best" if valid else "invalid",
                    "metric_value": result_metric if valid else None,
                    "baseline_metric": None,
                    "is_improvement": False,
                    "suspicious": False,
                    "anti_cheat_flags": [],
                    "reasoning": (
                        "survey_validation:" + validation_result["conclusion"]
                        if valid else "survey_validation:execution_or_evidence_invalid"),
                }
            elif (run_record.get("environment") == "component runner failure"
                  and run_record.get("succeeded") is False
                  and run_record.get("metric_value") is None):
                evaluation_report = {
                    "verdict": "invalid", "metric_value": None,
                    "baseline_metric": _finite_number((baseline_result or {}).get("metric_value")),
                    "is_improvement": False, "suspicious": False, "anti_cheat_flags": [],
                    "reasoning": "component_runner_execution_failed: " + str(
                        run_record.get("stderr_excerpt") or "no measured result"),
                }
            elif repository_run:
                evaluation_report = await asyncio.to_thread(
                    core.evaluate_experiment,
                    run_record, verifier_spec, benchmark_spec, self.llm_client,
                    prior_research_memory=prior_research_memory,
                    prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvaluatorAgent", self.config.prompt_patch()),
                )
            else:
                comparison_audit = await asyncio.to_thread(
                    comparison_guardian.audit_comparison_integrity,
                    run_record, benchmark_spec, self.llm_client,
                    baseline_result=baseline_result,
                )
                if comparison_audit["verdict"] == "pass":
                    evaluation_report = await asyncio.to_thread(
                        core.evaluate_experiment,
                        run_record, verifier_spec, benchmark_spec, self.llm_client,
                        prior_research_memory=prior_research_memory,
                        prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvaluatorAgent", self.config.prompt_patch()),
                    )
                else:
                    evaluation_report = comparison_guardian.suspicious_evaluation_report(
                        comparison_audit, run_record)
            run_record_digest = artifact_digest(run_record)
        except comparison_guardian.ComparisonGuardianError as exc:
            await self._fail(
                client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except core.EvaluatorError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        evaluation_report = dict(evaluation_report)
        evaluation_report["run_record_digest"] = run_record_digest
        evaluation_report["run_record_ref"] = payload["run_record_ref"]
        evaluation_report_ref = self.spec_store.put(
            kind="evaluation_report",
            session_id=session_id,
            key=_EVALUATION_REPORT_KEY,
            obj=evaluation_report,
            emitter=AGENT_NAME,
        )

        evaluation_report_ref = self.spec_store.put(
            kind="evaluation_report", session_id=session_id,
            key="snapshot-" + artifact_digest(evaluation_report).split(":", 1)[1],
            obj=evaluation_report, emitter=AGENT_NAME)

        await client.publish(
            event_type="experiment.evaluated",
            session_id=session_id,
            payload={
                "verdict": evaluation_report["verdict"],
                "evaluation_report_ref": evaluation_report_ref,
                "run_record_digest": run_record_digest,
            },
            causation_id=causation_id,
        )

    # ---- result.verification_requested ----

    async def handle_verification_requested(self, client: Any, env: Any) -> None:
        """Resolve one attempt's records, verify its metric or failure cause,
        persist the VerificationReport, and publish ``result.verified``."""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            evaluation_report = self.spec_store.get(payload.get("evaluation_report_ref"))
            run_record = self.spec_store.get(payload.get("run_record_ref"))
            verifier_spec = ((self.spec_store.get(payload.get("verifier_spec_ref")) or {}) if payload.get("verifier_spec_ref") else {})
            interpretation_ref = payload.get("spec_interpretation_ref")
            interpretation = (
                self.spec_store.get(interpretation_ref) if interpretation_ref else {})
            validation_study = (
                interpretation.get("validation_study")
                if isinstance(interpretation, dict) else None)
            try:
                from autoresearch.spec_store.identity import result_for_run
                current_gen_result = result_for_run(self.spec_store, payload.get("run_record_ref"))
            except KeyError:
                current_gen_result = None
            indirect_verification = _verify_indirect_component(
                evaluation_report, run_record, current_gen_result)
            if (evaluation_report.get("run_record_digest")
                    and evaluation_report["run_record_digest"] != artifact_digest(run_record)):
                verification_report = {"verdict": "failed", "checks": ["evaluation_run_digest_mismatch"],
                                       "anti_cheat_flags": ["record_consistency_error"],
                                       "reasoning": "Evaluation report belongs to a different run record."}
            elif indirect_verification is not None:
                verification_report = indirect_verification
            elif isinstance(validation_study, dict):
                try:
                    validation_result = self.spec_store.get(refs.encode(
                        _VALIDATION_RESULT_KIND, session_id, "v1"))
                except KeyError:
                    validation_result = None
                run_metric = _finite_number(run_record.get("metric_value"))
                result_metric = _finite_number(
                    validation_result.get("metric_value")
                    if isinstance(validation_result, dict) else None)
                success_consistent = bool(
                    run_record.get("succeeded") is True
                    and run_record.get("exit_code") == 0
                    and evaluation_report.get("verdict") == "valid_not_best"
                    and run_metric is not None
                    and run_metric == result_metric
                    and isinstance(validation_result.get("sample_ids"), list)
                    and bool(validation_result["sample_ids"])
                    and isinstance(validation_result.get("retrieved_at"), str)
                    and bool(validation_result["retrieved_at"].strip())
                    and isinstance(validation_result.get("artifacts"), list)
                    and bool(validation_result["artifacts"]))
                failure_consistent = bool(
                    run_record.get("succeeded") is False
                    and run_record.get("exit_code") != 0
                    and (run_record.get("stderr_excerpt")
                         or run_record.get("stdout_excerpt")))
                if success_consistent:
                    verification_report = {
                        "verdict": "verified",
                        "checks": [
                            "metric_matches_host_run_record",
                            "sample_ids_and_retrieval_date_present",
                            "raw_analysis_artifacts_declared",
                        ],
                        "anti_cheat_flags": [],
                        "reasoning": (
                            "The stored statistic and provenance fields match the host-rerun "
                            "survey result."),
                    }
                elif failure_consistent:
                    verification_report = {
                        "verdict": "verified",
                        "checks": ["failure_confirmed_from_exit_code_and_output"],
                        "anti_cheat_flags": [],
                        "reasoning": str(
                            run_record.get("stderr_excerpt")
                            or run_record.get("stdout_excerpt")),
                    }
                else:
                    verification_report = {
                        "verdict": "failed",
                        "checks": ["survey_result_does_not_match_run_record"],
                        "anti_cheat_flags": ["record_consistency_error"],
                        "reasoning": "Survey result provenance or metric could not be reproduced.",
                    }
            else:
                verification_report = await asyncio.to_thread(
                    core.verify_best_result,
                    evaluation_report, run_record, verifier_spec, self.llm_client,
                    prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvaluatorAgent", self.config.prompt_patch()),
                )
        except core.EvaluatorError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        verification_report = dict(verification_report)
        verification_report["run_record_digest"] = artifact_digest(run_record)
        verification_report["run_record_ref"] = payload["run_record_ref"]
        verification_report["evaluation_report_ref"] = self.spec_store.put(
            kind="evaluation_report", session_id=session_id,
            key="snapshot-" + artifact_digest(evaluation_report).split(":", 1)[1],
            obj=evaluation_report, emitter=AGENT_NAME)
        if current_gen_result and current_gen_result.get("candidate_snapshot_ref"):
            verification_report["candidate_snapshot_ref"] = current_gen_result["candidate_snapshot_ref"]
        verification_report_ref = self.spec_store.put(
            kind="verification_report",
            session_id=session_id,
            key=_VERIFICATION_REPORT_KEY,
            obj=verification_report,
            emitter=AGENT_NAME,
        )

        verification_report_ref = self.spec_store.put(
            kind="verification_report", session_id=session_id,
            key="snapshot-" + artifact_digest(verification_report).split(":", 1)[1],
            obj=verification_report, emitter=AGENT_NAME)

        await client.publish(
            event_type="result.verified",
            session_id=session_id,
            payload={
                "verdict": verification_report["verdict"],
                "verification_report_ref": verification_report_ref,
            },
            causation_id=causation_id,
        )

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


__all__ = ["EvaluatorAgentHandler", "SCHEMA_VERSION"]
