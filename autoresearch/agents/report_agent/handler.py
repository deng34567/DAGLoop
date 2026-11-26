"""
ReportAgent EventBus handler: the thin transport wrapper around core.
ReportAgent EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``report.requested`` (payload ``{research_memory_ref, research_spec_ref?,
citation_audit_report_ref?}``) -- reads the ResearchMemory from the Spec
Store by ref (provenance -- NEVER from the event payload); reads the
ResearchSpec too, IF its ref is present in the payload, else passes ``None``;
reads the CitationAuditReport too, IF its ref is present in the payload
(EvidenceAuditAgent's Mode C citation-regression audit), else passes
``None``; runs ``core.generate_report`` (which returns the Report dict
directly); persists the resulting Report (kind="report", key="v1"); and
publishes ``report.ready`` with its ref.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅一个事件：report.requested（按 ref 从 Spec Store 读回 ResearchMemory，若
payload 中给出 research_spec_ref 则一并读回 ResearchSpec，否则传 None；若
payload 中给出 citation_audit_report_ref（EvidenceAuditAgent 的 Mode C 引用
回归审计）则一并读回，否则传 None；跑一次 core.generate_report（直接返回
Report dict），持久化 Report，发布 report.ready）。core 的任何失败（或未预期
错误）都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import json
import logging
import math
import os
from typing import Any

from autoresearch.agents.report_agent import core
from autoresearch.agents.report_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store import refs
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_REPORT_KEY = "v1"

_NO_RESEARCH_MEMORY_NOTE = (
    "No ResearchMemory was available for this report: the research session "
    "reached the Reporting state without any consolidated findings. There "
    "are no verified findings or negative results to report."
)


def _empty_research_memory() -> dict:
    """A minimal, schema-valid ResearchMemory used when research_memory_ref
    is missing (e.g. a decision/MOSS path skipped ConsolidatingMemory) -- lets
    core.generate_report still produce an honest "research incomplete"
    Report instead of crashing on a None ref.
    当 research_memory_ref 缺失时（例如某条决策/MOSS 路径跳过了
    ConsolidatingMemory）使用的最小、schema 合法 ResearchMemory —— 让
    core.generate_report 仍能生成诚实的"研究未完成"报告，而非在 None ref 上
    崩溃。"""
    return {
        "verified_findings": [],
        "negative_results": [],
        "excluded_hypotheses": [],
        "consolidated_insights": [_NO_RESEARCH_MEMORY_NOTE],
        "divergence_seeds": [],
        "next_steps": [],
        "best_metric": None,
        "best_result_ref": None,
    }


def _apply_latest_verified_structured_result(
        report: dict, research_memory: dict, result: dict | None, *,
        best_result: dict | None = None) -> dict:
    """Make the best accepted structured comparison authoritative in the report.

    ResearchMemory intentionally accumulates failed attempts.  Once a later attempt is
    measured, those historical negatives remain useful evidence.  A later accepted attempt may
    also be worse than an earlier accepted best, so it must not replace ``best_metric`` in the
    headline or ledger.  Acceptance is reported separately from the numeric relation so a partial
    implementation cannot be called a verified win.
    """
    latest = result
    retained_history = bool(
        isinstance(best_result, dict)
        and best_result.get("accepted") is True
        and best_result.get("method_metric") == research_memory.get("best_metric")
        and (best_result.get("component_execution") or {}).get("mode") != "component"
    )
    if retained_history:
        result = best_result
    if not isinstance(result, dict):
        result = {}
    component_findings = [
        finding for finding in research_memory.get("verified_findings", [])
        if isinstance(finding, str) and finding.startswith("Verified component observation:")
    ]
    baseline = result.get("baseline_metric")
    method = result.get("method_metric")
    if (not isinstance(baseline, (int, float)) or isinstance(baseline, bool)
            or not math.isfinite(float(baseline))
            or not isinstance(method, (int, float)) or isinstance(method, bool)
            or not math.isfinite(float(method))):
        if component_findings and research_memory.get("best_metric") is None:
            return {**report, "summary": (
                "Verified component observations are retained, but no verified complete "
                "Method result is available. A later incomplete or failed run does not erase "
                "the measured component results. " + " ".join(component_findings))}
        return report

    metric = str(result.get("metric_name") or "primary metric").strip() or "primary metric"
    direction = "lower is better" if result.get("lower_is_better", True) else "higher is better"
    out = dict(report)
    accepted = (
        result.get("accepted") is True
        and (retained_history or research_memory.get("diagnosis") == "valid_improvement")
    )
    if not accepted:
        improved = method < baseline if result.get("lower_is_better", True) else method > baseline
        relation = "numerically better than" if improved else "not better than"
        reasons = [str(item) for item in (result.get("reasons") or []) if str(item).strip()]
        reason = "; ".join(reasons) or str(result.get("verdict") or "not accepted")
        out["summary"] = (
            f"Latest measured result: method={method} is {relation} baseline={baseline} "
            f"on {metric} ({direction}), but it was not accepted: {reason}."
        )
        if component_findings:
            out["summary"] += (
                " Separately verified component observations remain valid within their "
                "recorded scope; they are not complete-Method results. "
                + " ".join(component_findings))
        out["experiment_ledger"] = [
            {
                "experiment_id": "baseline",
                "metric_value": baseline,
                "verdict": "pass",
                "notes": "Baseline from the latest structured comparison.",
            },
            {
                "experiment_id": "method",
                "metric_value": method,
                "verdict": "inconclusive",
                "notes": f"Latest measured method was not accepted: {reason}",
            },
        ]
        out.pop("review", None)
        return out

    selected_method = method
    best_metric = research_memory.get("best_metric")
    if not retained_history and best_metric is not None and best_metric != method:
        out["analysis_caveats"] = list(out.get("analysis_caveats") or []) + [
            "A historical best number is recorded without a verified candidate snapshot; "
            "it is not promoted using the latest candidate's acceptance evidence."]
    latest_metric = latest.get("method_metric") if isinstance(latest, dict) else None
    latest_accepted = bool(isinstance(latest, dict) and latest.get("accepted") is True
                           and isinstance(latest_metric, (int, float))
                           and not isinstance(latest_metric, bool) and math.isfinite(latest_metric))
    relation = "beats" if (method < baseline if result.get("lower_is_better", True) else method > baseline) else "does not beat"
    out["summary"] = (
        f"Best verified result: method={selected_method} {relation} baseline={baseline} "
        f"on {metric} ({direction})."
    )
    control = (result.get("measurement_evidence") or {}).get("control_metric")
    if isinstance(control, (int, float)) and not isinstance(control, bool) and math.isfinite(control):
        out["summary"] += f" Paired control={control}; this is a separate contrast."
    implementation_review = result.get("method_implementation_review")
    partial_implementation = (
        isinstance(implementation_review, dict)
        and implementation_review.get("verdict") == "partial"
    )
    if partial_implementation:
        measured_outcome = "improvement" if relation == "beats" else "result"
        out["summary"] += (
            f" This is an accepted measured {measured_outcome} from a partial implementation; only "
            "the implemented frozen components are credited."
        )
        caveats = list(out.get("analysis_caveats") or [])
        caveat = (
            "MethodImplementationReview was partial: the metric is accepted, while unimplemented "
            "frozen components are not claimed as validated."
        )
        if caveat not in caveats:
            caveats.append(caveat)
        out["analysis_caveats"] = caveats
    if retained_history and latest_accepted and latest_metric != method:
        out["summary"] += (
            f" The latest accepted attempt measured {latest_metric}; it does not replace the earlier "
            "separately verified result."
        )
    elif retained_history and not latest_accepted:
        out["analysis_caveats"] = list(out.get("analysis_caveats") or []) + [
            "The latest attempt did not establish an accepted result; the separately verified "
            "historical candidate is retained with its own comparison and implementation scope."]
    if research_memory.get("negative_results") and not retained_history:
        out["summary"] += (
            " Historical negative results below document earlier attempts; they do not "
            "override this latest accepted result."
        )
        caveat = (
            "Historical negative results describe earlier attempts and do not invalidate "
            "the latest verified structured comparison."
        )
        caveats = list(out.get("analysis_caveats") or [])
        if caveat not in caveats:
            caveats.append(caveat)
        out["analysis_caveats"] = caveats
    out["experiment_ledger"] = [
        {
            "experiment_id": "baseline",
            "metric_value": baseline,
            "verdict": "pass",
            "notes": ("Baseline from the selected verified candidate snapshot." if retained_history
                      else "Baseline from the latest verified structured comparison."),
        },
        {
            "experiment_id": "method",
            "metric_value": selected_method,
            "verdict": "pass",
            "notes": (
                "Best accepted partial implementation retained; only implemented frozen "
                "components are credited."
                if partial_implementation
                else "Best accepted method retained from verified ResearchMemory."
            ),
        },
    ]
    if retained_history and latest_accepted and latest_metric != method:
        out["experiment_ledger"].append({
            "experiment_id": "method_latest_accepted",
            "metric_value": latest_metric,
            "verdict": "pass",
            "notes": "Latest accepted attempt, recorded separately from the selected candidate.",
        })
    # The blind review covered the discarded LLM headline, not this host-derived one.
    out.pop("review", None)
    return out


class ReportAgentHandler:
    """Wires core.generate_report to a Spec Store + LLM + EventBus client.
    把 core.generate_report 接到 Spec Store + LLM + EventBus 客户端。"""

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
        """Subscribe this handler's event on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的事件。"""
        client.subscribe(
            "report.requested",
            self._make_eventbus_handler(client, self.handle_report_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- report.requested ----

    async def handle_report_requested(self, client: Any, env: Any) -> None:
        """Resolve the ResearchMemory (+ optional ResearchSpec + optional
        CitationAuditReport) from the store, run one generate_report,
        persist the Report, publish report.ready.
        从 store 解析 ResearchMemory（+ 可选 ResearchSpec + 可选
        CitationAuditReport），跑一次 generate_report，持久化 Report，发布
        report.ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        receipt_ref = refs.encode("audit", session_id, "report-request-" + str(causation_id))
        try:
            receipt = self.spec_store.get(receipt_ref)
        except KeyError:
            receipt = None
        if receipt:
            if receipt["request_digest"] != artifact_digest(payload):
                raise ValueError("report request identity changed")
            if receipt.get("report_ref"):
                await client.publish(event_type="report.ready", session_id=session_id,
                                     payload={"report_ref": receipt["report_ref"]}, causation_id=causation_id)
            else:
                await self._finish_report(client, env, receipt["report"])
            return

        try:
            research_memory_ref = payload.get("research_memory_ref")
            research_memory = (
                self.spec_store.get(research_memory_ref)
                if research_memory_ref
                else _empty_research_memory()
            )

            research_spec_ref = payload.get("research_spec_ref")
            research_spec = (
                self.spec_store.get(research_spec_ref) if research_spec_ref else None
            )

            citation_audit_report_ref = payload.get("citation_audit_report_ref")
            citation_audit_report = (
                self.spec_store.get(citation_audit_report_ref)
                if citation_audit_report_ref
                else None
            )

            research_context = {}
            for key, ref_key in (
                ("literature_brief", "literature_brief_ref"),
                ("confirmed_method", "method_ref"),
                ("evidence_audit", "evidence_audit_report_ref"),
                ("novelty_audit", "novelty_audit_report_ref"),
            ):
                ref = payload.get(ref_key)
                if ref:
                    research_context[key] = self.spec_store.get(ref)
            if citation_audit_report is not None:
                research_context["citation_audit"] = citation_audit_report

            try:
                current_ref = payload.get("result_ref")
                if "result_ref" not in payload:  # Compatibility for callers predating request binding.
                    current_ref = refs.encode("gen_experiment_result", session_id, "current")
                current_result = self.spec_store.get(current_ref) if current_ref else None
            except KeyError:
                current_result = None
            structured_result = isinstance(current_result, dict)
            best_result = None
            best_ref = research_memory.get("best_result_ref")
            if structured_result and isinstance(best_ref, str):
                try:
                    parts = refs.decode(best_ref)
                    saved = self.spec_store.get(best_ref)
                    if (parts.kind == "gen_experiment_result" and parts.session_id == session_id
                            and parts.key == "snapshot-" + artifact_digest(saved).split(":", 1)[1]):
                        best_result = saved
                except (KeyError, ValueError, TypeError):
                    pass  # Legacy/unbound numbers do not certify a candidate's eligibility.
            report_memory = research_memory
            if structured_result:
                selected, selected_ref = current_result, current_ref
                if (isinstance(best_result, dict) and best_result.get("accepted") is True
                        and best_result.get("method_metric") == research_memory.get("best_metric")
                        and (best_result.get("component_execution") or {}).get("mode") != "component"):
                    selected, selected_ref = best_result, best_ref
                # The writer needs the selected attempt's evidence before classifying older failures.
                report_memory = {**research_memory, "selected_result": {
                    "result_ref": selected_ref,
                    **{key: selected[key] for key in (
                        "accepted", "verdict", "reasons", "baseline_metric", "method_metric",
                        "lower_is_better", "metric_name", "measurement_evidence",
                        "attempt_review_evidence",
                        "candidate_admissibility", "repository_comparison_review",
                        "method_implementation_review", "component_execution",
                        "run_record_snapshot_ref", "checkpoint", "changes_summary", "config",
                        "ablation_report",
                    ) if key in selected},
                }}
                method_ref = (selected.get("method_implementation_review") or {}).get("method_ref")
                if isinstance(method_ref, str):
                    try:
                        parts = refs.decode(method_ref)
                        measured_method = self.spec_store.get(method_ref)
                        if (parts.kind == "method" and parts.session_id == session_id
                                and parts.key == "snapshot-" + artifact_digest(measured_method).split(":", 1)[1]):
                            research_context["confirmed_method"] = measured_method
                    except (KeyError, ValueError, TypeError):
                        pass  # An unbound Method cannot replace the supplied research context.

            report = await asyncio.to_thread(  # ④ concurrency: offload the blocking LLM call off the shared web event loop
                core.generate_report,
                report_memory, research_spec, self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "ReportAgent", self.config.prompt_patch()),
                citation_audit_report=citation_audit_report,
                reviewer_llm=self.llm_client,  # guaranteed blind-review gate (fresh call)
                research_context=research_context,
            )
        except core.ReportError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        # Structured repository/general experiment: the ResearchMemory's verified_findings /
        # negative_results are already GROUNDED (our deterministic gen-seed carries
        # the honest headline, e.g. "method beats baseline: baseline=… vs method=…").
        # The LLM report sometimes files that into the ledger/summary and leaves the
        # arrays empty; the grounding guard only DROPS hallucinations, it never ADDS
        # grounded findings the LLM omitted. Force-merge so the honest result is
        # always in verified_findings/negative_results.  Key off the result artifact rather than a
        # legacy launch flag: the agentic repository path writes the same artifact but does not
        # require that flag.
        if structured_result or os.environ.get("AUTORESEARCH_GEN_EXPERIMENT") == "1":
            for field in ("verified_findings", "negative_results"):
                grounded = research_memory.get(field) or []
                merged = list(report.get(field) or [])
                for item in grounded:
                    if item not in merged:
                        merged.append(item)
                report[field] = merged
        if structured_result:
            report = _apply_latest_verified_structured_result(
                report, research_memory, current_result, best_result=best_result)

        source_history = research_memory.get("source_history") or []
        observations = research_memory.get("baseline_observations") or []
        if observations:
            note = ("Baseline attempt observations (reproduced=false is not a qualified baseline): "
                    + json.dumps(observations, ensure_ascii=False))
            report["analysis_caveats"] = list(report.get("analysis_caveats") or []) + [note]
            finite_observations = []
            for observation in observations:
                value = observation.get("metric_value")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    finite_observations.append({"metric": observation.get("metric_name"), "value": value,
                                                "reproduced": observation.get("reproduced"), "ref": observation["ref"]})
                    report.setdefault("experiment_ledger", []).append({
                        "experiment_id": observation["ref"], "metric_value": value,
                        "verdict": "pass" if observation.get("reproduced") is True else "inconclusive",
                        "notes": "Historical baseline attempt; quality and identity: "
                                 + json.dumps(observation, ensure_ascii=False)})
            if finite_observations and not structured_result and research_memory.get("diagnosis") in {
                "baseline_reproduction_failed", "component_execution_blocked", "execution_failure",
            }:
                report["summary"] = (
                    "Finite baseline observations were recorded; measurement and qualified reproduction "
                    "are separate: " + json.dumps(finite_observations, ensure_ascii=False)
                    + ". Current termination diagnosis: " + research_memory["diagnosis"] + ".")
            if report.get("report_body"):
                report["report_body"] += "\n\n" + note
            report.pop("review", None)
        if source_history:
            from autoresearch.experiments.repo_improve.source_policy import attribution
            report["summary"] = str(report.get("summary") or "") + " " + attribution(source_history)
            report["analysis_caveats"] = list(report.get("analysis_caveats") or []) + [
                "Source history: " + json.dumps(source_history, ensure_ascii=False)]
            if report.get("report_body"):
                report["report_body"] += "\n\n" + attribution(source_history) + "\n" + json.dumps(source_history, ensure_ascii=False)
        await self._finish_report(client, env, report)

    async def _finish_report(self, client, env, report):
        session_id, payload = env.session_id, env.payload or {}
        receipt = {"request_digest": artifact_digest(payload), "report": report}
        key = "report-request-" + str(env.event_id)
        self.spec_store.put(kind="audit", session_id=session_id, key=key,
                            obj=receipt, emitter=AGENT_NAME)
        try:
            if payload.get("result_ref"):
                result = self.spec_store.get(payload["result_ref"])
                native_bundle = result.get("delivery_bundle")
                if native_bundle is not None:
                    if not isinstance(native_bundle, dict) or not native_bundle.get("files"):
                        raise ValueError("selected experiment lacks its original source bundle")
                    bundle = {**native_bundle, "result_ref": payload["result_ref"],
                              "run_record_snapshot_ref": result.get("run_record_snapshot_ref")}
                else:
                    # Export the host-retained complete patch against the frozen baseline.
                    patch = result.get("baseline_relative_diff")
                    if patch is None or result.get("baseline_diff_error"):
                        raise ValueError("selected implementation lacks a baseline-relative export")
                    baseline_ref = result.get("baseline_result_ref")
                    baseline = self.spec_store.get(baseline_ref) if baseline_ref else {}
                    manifest = {"result_ref": payload["result_ref"],
                                "checkpoint": result.get("checkpoint"),
                                "repo_commit": baseline.get("repo_commit") or result.get("repo_commit"),
                                "repo_url": result.get("repo_url") or baseline.get("repo_url")
                                    or ((result.get("repository_contract") or {}).get("repo_url")),
                                "run_manifest": result.get("run_manifest"),
                                "repository_contract": result.get("repository_contract") or baseline.get("repository_contract"),
                                "runtime_provenance": result.get("runtime_provenance"),
                                "measurement_evidence": result.get("measurement_evidence"),
                                "baseline_command": baseline.get("command"),
                                "review_receipts": result.get("review_receipts"),
                                "baseline_metric": result.get("baseline_metric"),
                                "method_metric": result.get("method_metric")}
                    # The measured commands, source identity and patch are exported verbatim.
                    files = {"candidate.patch": patch, "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
                             "README.md": "Apply candidate.patch to repo_commit of repo_url in manifest.json. "
                                "Reproduce using the saved execution commands and unchanged evaluator. "
                                "An empty patch means the exported implementation equals that baseline.\n"}
                    bundle = {**manifest, "files": files}
                bundle_ref = self.spec_store.put(kind="deliverable_bundle", session_id=session_id,
                    key="snapshot-" + artifact_digest(bundle).split(":", 1)[1], obj=bundle, emitter=AGENT_NAME)
                report = {**report, "result_ref": payload["result_ref"], "delivery_bundle_ref": bundle_ref}
        except (KeyError, ValueError, TypeError) as exc:
            await self._fail(client, session_id, env.event_id, "permanent",
                             "required_delivery_unavailable", str(exc))
            return
        report = {**report, "request_digest": artifact_digest(payload)}
        self.spec_store.put(kind="report", session_id=session_id, key=_REPORT_KEY,
                            obj=report, emitter=AGENT_NAME)
        report_ref = self.spec_store.put(kind="report", session_id=session_id,
            key="snapshot-" + artifact_digest(report).split(":", 1)[1], obj=report, emitter=AGENT_NAME)
        self.spec_store.put(kind="audit", session_id=session_id, key=key,
                            obj={**receipt, "report_ref": report_ref}, emitter=AGENT_NAME)
        await client.publish(event_type="report.ready", session_id=session_id,
                             payload={"report_ref": report_ref}, causation_id=env.event_id)

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


__all__ = ["ReportAgentHandler", "SCHEMA_VERSION"]
