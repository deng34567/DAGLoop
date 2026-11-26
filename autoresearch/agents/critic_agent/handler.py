"""
CriticAgent EventBus handler: the thin transport wrapper around core.
CriticAgent EventBus handler：core 之上的薄传输层。

Subscribes THREE events:

1. ``direction.critique_requested`` (payload ``{problem_model_ref,
   research_spec_ref}``) -- reads the ProblemModel + ResearchSpec from the
   Spec Store by ref (provenance -- NEVER from the event payload), runs
   ``core.critique_direction``, persists the resulting CritiqueReport
   (kind="critique_report", key="direction-v1"), and publishes
   ``direction.critiqued`` with its verdict + ref.
2. ``hypothesis.critique_requested`` (payload ``{hypothesis_bank_ref,
   research_spec_ref, ranking_report_ref?}``) -- reads the HypothesisBank +
   ResearchSpec from the Spec Store by ref (and the RankingReport too, if
   ``ranking_report_ref`` is present -- RankingJudgeAgent milestone Task 4,
   optional/backward-safe), runs ``core.critique_hypotheses``, persists the
   resulting CritiqueReport (kind="critique_report", key="hypotheses-v1"),
   and publishes ``hypothesis.critiqued`` with its verdict + ref +
   passing_hypothesis_ids.
3. ``experiment.review_requested`` (payload ``{experiment_plan_ref,
   verifier_spec_ref}``) -- reads the ExperimentPlan + VerifierSpec from the
   Spec Store by ref, runs ``core.critique_experiment_plan`` (which
   re-enforces the editable_region gate), persists the resulting
   CritiqueReport (kind="critique_report", key="experiment_plan-v1"), and
   publishes ``experiment.plan_reviewed`` with its verdict + ref.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅三个事件：direction.critique_requested（按 ref 从 Spec Store 读回
ProblemModel + ResearchSpec，跑一次 core.critique_direction，持久化
CritiqueReport，发布 direction.critiqued）、hypothesis.critique_requested
（按 ref 读回 HypothesisBank + ResearchSpec，跑一次 core.critique_hypotheses，
持久化 CritiqueReport，发布 hypothesis.critiqued）、
experiment.review_requested（按 ref 读回 ExperimentPlan + VerifierSpec，跑
一次 core.critique_experiment_plan，持久化 CritiqueReport，发布
experiment.plan_reviewed）。core 的任何失败（或未预期错误）都转成
agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
import os
from typing import Any

from autoresearch.agents.critic_agent import core
from autoresearch.agents.critic_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store import refs
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

# P1 general honest-experiment gate: in GEN mode the state-machine's
# ExperimentPlan is UNUSED (the CoderAgent gen-branch emits a trivial patch and
# the RunnerAgent gen-branch runs the real P1 pipeline that re-designs from the
# question), so a strict LLM plan-review only THRASHES on a document nothing
# consumes (observed: experiment_design_thrashing livelock). When set, the
# experiment-plan review auto-passes (no LLM). The coder patch's editable_region
# gate still runs independently at ImplementingExperiment.
_GEN_ENV_FLAG = "AUTORESEARCH_GEN_EXPERIMENT"
# Opt-in (web only): when set, a HUMAN reviews the concrete P1 design at
# ReviewingExperimentPlan, so the critic DEFERS the experiment-plan review to the
# human (it does not emit experiment.plan_reviewed; the human's web approval does).
# Unset (default, incl. headless/tests): the GEN auto-pass below is unchanged.
_GEN_REVIEW_FLAG = "AUTORESEARCH_GEN_REVIEW_DESIGN"

SCHEMA_VERSION = 1

_DIRECTION_REPORT_KEY = "direction-v1"
_HYPOTHESES_REPORT_KEY = "hypotheses-v1"


class CriticAgentHandler:
    """Wires core.critique_direction/critique_hypotheses to a Spec Store +
    LLM + EventBus client.
    把 core.critique_direction/critique_hypotheses 接到 Spec Store + LLM +
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
        """Subscribe this handler's three events on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的三个事件。"""
        client.subscribe(
            "direction.critique_requested",
            self._make_eventbus_handler(client, self.handle_direction_critique),
        )
        client.subscribe(
            "hypothesis.critique_requested",
            self._make_eventbus_handler(client, self.handle_hypothesis_critique),
        )
        client.subscribe(
            "experiment.review_requested",
            self._make_eventbus_handler(client, self.handle_experiment_plan_review),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- direction.critique_requested ----

    async def handle_direction_critique(self, client: Any, env: Any) -> None:
        """Resolve the ProblemModel + ResearchSpec from the store, run one
        critique_direction, persist the report, publish direction.critiqued.
        从 store 解析 ProblemModel + ResearchSpec，跑一次 critique_direction，
        持久化报告，发布 direction.critiqued。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            problem_model = self.spec_store.get(payload.get("problem_model_ref"))
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            report = await asyncio.to_thread(
                core.critique_direction,
                problem_model,
                research_spec,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "CriticAgent", self.config.prompt_patch()),
            )
        except core.CriticError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        critique_report_ref = self.spec_store.put(
            kind="critique_report",
            session_id=session_id,
            key=_DIRECTION_REPORT_KEY,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="direction.critiqued",
            session_id=session_id,
            payload={
                "verdict": report["verdict"],
                "critique_report_ref": critique_report_ref,
            },
            causation_id=causation_id,
        )

    # ---- hypothesis.critique_requested ----

    async def handle_hypothesis_critique(self, client: Any, env: Any) -> None:
        """Resolve the HypothesisBank + ResearchSpec (+ RankingReport, if
        referenced) from the store, run one critique_hypotheses, persist the
        report, publish hypothesis.critiqued.
        从 store 解析 HypothesisBank + ResearchSpec（如有引用，还有
        RankingReport），跑一次 critique_hypotheses，持久化报告，发布
        hypothesis.critiqued。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            hypothesis_bank = self.spec_store.get(payload.get("hypothesis_bank_ref"))
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            ranking_report_ref = payload.get("ranking_report_ref")
            ranking_report = (
                self.spec_store.get(ranking_report_ref) if ranking_report_ref else None
            )
            report = await asyncio.to_thread(
                core.critique_hypotheses,
                hypothesis_bank,
                research_spec,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "CriticAgent", self.config.prompt_patch()),
                ranking_report=ranking_report,
            )
        except core.CriticError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        critique_report_ref = self.spec_store.put(
            kind="critique_report",
            session_id=session_id,
            key=_HYPOTHESES_REPORT_KEY,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="hypothesis.critiqued",
            session_id=session_id,
            payload={
                "verdict": report["verdict"],
                "critique_report_ref": critique_report_ref,
                "passing_hypothesis_ids": report.get("passing_hypothesis_ids", []),
            },
            causation_id=causation_id,
        )

    # ---- experiment.review_requested ----

    async def handle_experiment_plan_review(self, client: Any, env: Any) -> None:
        """Resolve the ExperimentPlan + VerifierSpec from the store, run one
        critique_experiment_plan (re-enforcing the editable_region gate),
        persist the report, publish experiment.plan_reviewed.
        从 store 解析 ExperimentPlan + VerifierSpec，跑一次
        critique_experiment_plan（再次强制 editable_region 硬门），持久化
        报告，发布 experiment.plan_reviewed。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        receipt_key = "plan-review-request-" + str(causation_id)
        try:
            completed = self.spec_store.get(refs.encode("audit", session_id, receipt_key))
        except KeyError:
            completed = None
        if completed is not None:
            if completed["request_digest"] != artifact_digest(payload):
                raise ValueError("completed plan review request identity mismatch")
            await client.publish(event_type="experiment.plan_reviewed", session_id=session_id,
                                 payload=completed["payload"], causation_id=causation_id)
            return

        baseline_result = {}
        if payload.get("baseline_result_ref"):
            try:
                baseline_result = self.spec_store.get(payload["baseline_result_ref"]) or {}
            except Exception:  # handled by the normal review path below
                baseline_result = {}
        live_repo_baseline = (
            isinstance(baseline_result, dict)
            and baseline_result.get("source_kind") == "live_repo"
        )
        try:
            experiment_plan = self.spec_store.get(payload.get("experiment_plan_ref"))
        except Exception:
            experiment_plan = None
        survey_validation = (
            isinstance(experiment_plan, dict)
            and experiment_plan.get("task_class") == "survey_validation")

        # Design-review gate (opt-in, web): defer the review to the HUMAN, who inspects
        # the concrete P1 design at ReviewingExperimentPlan and approves/rejects it via
        # the web (that emits experiment.plan_reviewed). The critic emits nothing here,
        # so the state pauses until the human acts (or the watchdog times out).
        if (os.environ.get(_GEN_ENV_FLAG) == "1"
                and os.environ.get(_GEN_REVIEW_FLAG) == "1"
                and not live_repo_baseline
                and not survey_validation):
            return

        try:
            if survey_validation:
                issues = []
                if not experiment_plan.get("changes"):
                    issues.append("survey validation has no executable analysis script")
                if not str(experiment_plan.get("success_criteria") or "").strip():
                    issues.append("survey validation has no evidence-completeness criterion")
                report = {
                    "target": "experiment_plan",
                    "verdict": "fail" if issues else "pass",
                    "issues": issues,
                }
            elif os.environ.get(_GEN_ENV_FLAG) == "1" and not live_repo_baseline:
                # GEN mode: the plan is unused downstream -> auto-pass (no LLM), so
                # a strict reviewer can't livelock the run before the real P1
                # experiment. Schema: {target, verdict, issues} only.
                report = {"target": "experiment_plan", "verdict": "pass", "issues": []}
            else:
                experiment_plan = self.spec_store.get(payload.get("experiment_plan_ref"))
                verifier_spec = ((self.spec_store.get(payload.get("verifier_spec_ref")) or {}) if payload.get("verifier_spec_ref") else {})
                benchmark_spec = ((self.spec_store.get(payload.get("benchmark_spec_ref")) or {})
                                  if payload.get("benchmark_spec_ref") else {})
                method = (self.spec_store.get(payload["method_ref"])
                          if payload.get("method_ref") else None)
                method_feedback = (self.spec_store.get(payload["method_feedback_ref"])
                                   if payload.get("method_feedback_ref") else None)
                if (isinstance(method_feedback, dict)
                        and (method_feedback.get("decision") or {}).get("route")
                        == "revise_method"):
                    method_feedback = {
                        "method_ref": method_feedback.get("method_ref"),
                        "source_method_ref": method_feedback.get("source_method_ref"),
                        "trigger": method_feedback.get("trigger"),
                        "reason": method_feedback.get("reason"),
                        "decision": method_feedback.get("decision"),
                    }
                else:
                    method_feedback = None
                research_memory = (self.spec_store.get(payload["research_memory_ref"])
                                   if payload.get("research_memory_ref") else None)
                report = await asyncio.to_thread(
                    core.critique_experiment_plan,
                    experiment_plan,
                    verifier_spec,
                    self.llm_client,
                    prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "CriticAgent", self.config.prompt_patch()),
                    benchmark_spec=benchmark_spec,
                    baseline_result=baseline_result,
                    method=method,
                    research_memory=research_memory,
                    method_feedback=method_feedback,
                )
        except core.CriticError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        review_report_ref = self.spec_store.put(
            kind="critique_report",
            session_id=session_id,
            key="snapshot-" + artifact_digest(report).split(":", 1)[1],
            obj=report,
            emitter=AGENT_NAME,
        )

        completion_payload = {"verdict": report["verdict"], "review_report_ref": review_report_ref}
        self.spec_store.put(kind="audit", session_id=session_id, key=receipt_key,
                            obj={"request_digest": artifact_digest(payload), "payload": completion_payload},
                            emitter=AGENT_NAME)
        await client.publish(
            event_type="experiment.plan_reviewed",
            session_id=session_id,
            payload=completion_payload,
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


__all__ = ["CriticAgentHandler", "SCHEMA_VERSION"]
