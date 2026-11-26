"""
ExperimentPlannerAgent EventBus handler: the thin transport wrapper around
core.
ExperimentPlannerAgent EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``experiment.planning_requested`` (payload ``{hypothesis_bank_ref,
selected_hypothesis_id, verifier_spec_ref, benchmark_spec_ref,
research_spec_ref}``) -- reads the HypothesisBank, VerifierSpec,
BenchmarkSpec, and ResearchSpec from the Spec Store by ref (provenance --
NEVER from the event payload), selects the hypothesis within the bank whose
``id`` matches ``selected_hypothesis_id``, runs ``core.plan_experiment``,
persists the resulting ExperimentPlan at a content-addressed snapshot, and
publishes ``experiment.plan_ready`` with its ref.

If no hypothesis in the bank matches ``selected_hypothesis_id``, that is
treated as a failure -- ``agent.failed`` is published WITHOUT calling the LLM.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅一个事件：experiment.planning_requested（按 ref 从 Spec Store 读回
HypothesisBank、VerifierSpec、BenchmarkSpec、ResearchSpec，在 bank 中按 id
选出所选 hypothesis，跑一次 core.plan_experiment，持久化 ExperimentPlan，
发布 experiment.plan_ready）。若 bank 中没有与 selected_hypothesis_id 匹配的
hypothesis，视为失败 —— 直接发布 agent.failed，不调用 LLM。core 的任何失败
（或未预期错误）都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
import os
from typing import Any

from autoresearch.agents.experiment_planner_agent import core
from autoresearch.agents.experiment_planner_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.spec_store import refs

# P1 general honest-experiment: in GEN mode we ALSO generate the concrete experiment
# DESIGN (task/method/baseline/metric/instances) here, at PlanningExperiment, and store
# it as ``gen_design`` -- so it EXISTS to be reviewed at ReviewingExperimentPlan and the
# RunnerAgent runs THAT approved design (instead of re-generating a different one).
_GEN_ENV_FLAG = "AUTORESEARCH_GEN_EXPERIMENT"
_GEN_DESIGN_KIND = "gen_design"

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_EXPERIMENT_PLAN_KEY = "v1"


def _component_memory_packet(
    memory: dict | None, component_id: str, dependency_closure: tuple[str, ...],
) -> dict:
    """Represent every prior round with detailed active-lineage evidence and sibling summaries."""
    from autoresearch.experiments.repo_improve.component_plan import component_memory_packet

    history = [
        dict(item) for item in ((memory or {}).get("component_history") or [])
        if isinstance(item, dict)
    ]
    return component_memory_packet(
        history, component_id=component_id, dependency_closure=dependency_closure)


class ExperimentPlannerAgentHandler:
    """Wires core.plan_experiment to a Spec Store + LLM + EventBus client.
    把 core.plan_experiment 接到 Spec Store + LLM + EventBus 客户端。"""

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
            "experiment.planning_requested",
            self._make_eventbus_handler(client, self.handle_planning_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- experiment.planning_requested ----

    async def handle_planning_requested(self, client: Any, env: Any) -> None:
        """Resolve the HypothesisBank + VerifierSpec + BenchmarkSpec +
        ResearchSpec from the store, select the hypothesis by id, run one
        plan_experiment, persist the ExperimentPlan, publish
        experiment.plan_ready.
        从 store 解析 HypothesisBank + VerifierSpec + BenchmarkSpec +
        ResearchSpec，按 id 选出 hypothesis，跑一次 plan_experiment，持久化
        ExperimentPlan，发布 experiment.plan_ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        receipt_key = "planner-request-" + str(causation_id)
        try:
            completed = self.spec_store.get(refs.encode("audit", session_id, receipt_key))
        except KeyError:
            completed = None
        if completed is not None:
            if completed["request_digest"] != artifact_digest(payload):
                raise ValueError("completed planning request identity mismatch")
            await client.publish(event_type="experiment.plan_ready", session_id=session_id,
                                 payload=completed["payload"], causation_id=causation_id)
            return

        try:
            # 甲: on the benchmark mainline there is no hypothesis bank (hypothesis_bank_ref is None) --
            # don't resolve a None ref (the store may raise on it); the method branch below supplies
            # the thing to plan.
            hb_ref = payload.get("hypothesis_bank_ref")
            hypothesis_bank = self.spec_store.get(hb_ref) if hb_ref else None
            verifier_spec = ((self.spec_store.get(payload.get("verifier_spec_ref")) or {}) if payload.get("verifier_spec_ref") else {})
            benchmark_ref = payload.get("benchmark_spec_ref")
            benchmark_spec = self.spec_store.get(benchmark_ref) if benchmark_ref else {}
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            interpretation_ref = payload.get("spec_interpretation_ref")
            interpretation = (
                self.spec_store.get(interpretation_ref) if interpretation_ref else {})
            validation_study = (
                interpretation.get("validation_study")
                if isinstance(interpretation, dict) else None)
            baseline_result = (
                self.spec_store.get(payload.get("baseline_result_ref"))
                if payload.get("baseline_result_ref") else None
            )
            plan_review = (
                self.spec_store.get(payload.get("plan_review_ref"))
                if payload.get("plan_review_ref") else None
            )
            research_memory = (
                self.spec_store.get(payload.get("research_memory_ref"))
                if payload.get("research_memory_ref") else None
            )
            method_feedback = (
                self.spec_store.get(payload.get("method_feedback_ref"))
                if payload.get("method_feedback_ref") else None
            )

            selected_hypothesis_id = payload.get("selected_hypothesis_id")
            selected_hypothesis = next(
                (
                    hyp
                    for hyp in (hypothesis_bank or {}).get("hypotheses", [])
                    if hyp.get("id") == selected_hypothesis_id
                ),
                None,
            )
            # meeting 2026-07-25 甲: on the benchmark mainline there is no hypothesis portfolio --
            # the CONFIRMED METHOD (A') is the thing to implement. When no hypothesis was selected,
            # resolve the method and plan HOW TO IMPLEMENT it (a method is structurally an approach,
            # so it slots in where a hypothesis would).
            # A validation study may be supporting evidence for a benchmark Method; it is the
            # primary experiment only for the explicit validation-only route.  In particular,
            # never let an auxiliary study replace a confirmed Method/baseline comparison.
            primary_validation_study = (
                isinstance(validation_study, dict)
                and research_spec.get("task_mode") == "experiment_validation"
                and not payload.get("method_ref")
                and not payload.get("baseline_result_ref")
            )
            if primary_validation_study:
                experiment_plan = {
                    "hypothesis_id": "validation-study",
                    "change_summary": validation_study["question"],
                    "changes": [{
                        "file": "validation_study.py",
                        "description": (
                            "Implement and run the frozen survey validation study: "
                            + validation_study["analysis"]),
                    }],
                    "expected_outcome": (
                        "Record whether the observations support, contradict, or leave the "
                        "claim inconclusive; a null or negative finding is valid."),
                    "resource_budget": validation_study["budget"],
                    "success_criteria": (
                        "The script exits successfully and preserves source provenance, retrieval "
                        "date, sample identifiers, analysis output, and reproducible artifacts."),
                    "task_class": "survey_validation",
                }
            elif selected_hypothesis is None and payload.get("method_ref"):
                method = self.spec_store.get(payload.get("method_ref"))
                if isinstance(method, dict) and method:
                    selected_hypothesis = {"id": payload.get("method_ref"),
                                           "kind": "confirmed_method", **method}
                    if isinstance(method.get("components"), list) and method["components"]:
                        from autoresearch.experiments.repo_improve.component_plan import (
                            build_component_plan, ledger_for_workroot)
                        from autoresearch.experiments.repo_improve.search_memory import (
                            session_workroot)

                        component_plan = build_component_plan(method)
                        ledger = ledger_for_workroot(
                            session_workroot(session_id), component_plan,
                            execution_identity=(
                                artifact_digest(baseline_result) if baseline_result else None
                            ),
                        )
                        node = ledger.next_ready()
                        if method.get("reconciled_result_ref") and not ledger.all_accepted():
                            selected_hypothesis = {
                                **selected_hypothesis, "kind": "confirmed_method_revalidation",
                                "instruction": (
                                    "Plan full validation of the exact retained source bound by "
                                    "reconciled_result_ref, using the frozen official evaluator "
                                    "and correctness gates. Runner must recheck its host binding. "
                                    "Do not require an already implemented component to introduce "
                                    "a new delta. Pending components remain unaccepted; the whole "
                                    "Method requires fresh implementation and task-contract review."
                                ),
                            }
                        elif node is not None:
                            closure = component_plan.dependency_closure(node.component_id)
                            entry = ledger.entry(node.component_id)
                            selected_hypothesis = {
                                "id": payload.get("method_ref"),
                                "kind": "confirmed_method_component",
                                "approach": method.get("approach"),
                                "current_component": node.to_dict(),
                                "dependency_components": [
                                    component_plan.node(item).to_dict() for item in closure
                                ],
                                "component_memory": _component_memory_packet(
                                    research_memory, node.component_id, closure),
                                "component_search_decision": entry.search_decision,
                                "instruction": (
                                    "Plan only the current ready component in its exact dependency "
                                    "context; do not implement a later or unrelated DAG branch. "
                                    "Name its exact component_id or component name in change_summary. "
                                    "If a verified continuation decision is present, test its "
                                    "specific next experiment within the frozen Method contract."
                                ),
                            }
                        elif ledger.all_accepted():
                            selected_hypothesis = {
                                "id": payload.get("method_ref"),
                                "kind": "confirmed_method_integration",
                                "approach": method.get("approach"),
                                "terminal_components": list(component_plan.terminal_components),
                                "components": [item.to_dict() for item in component_plan.components],
                                "component_memory": _component_memory_packet(
                                    research_memory, "__integration__",
                                    component_plan.topological_order),
                                "instruction": (
                                    "Plan only terminal-branch integration and evidence-driven "
                                    "joint tuning; do not invent a new component."
                                ),
                            }
                elif method:
                    selected_hypothesis = {"id": payload.get("method_ref"),
                                           "kind": "confirmed_method", "description": str(method)}
                if (isinstance(selected_hypothesis, dict)
                        and isinstance(method_feedback, dict)
                        and (method_feedback.get("decision") or {}).get("route")
                        == "revise_method"):
                    selected_hypothesis = dict(selected_hypothesis)
                    selected_hypothesis["current_method_changes_outline"] = method.get(
                        "changes_outline") or []
                    selected_hypothesis["applied_method_feedback"] = {
                        "trigger": method_feedback.get("trigger"),
                        "reason": method_feedback.get("reason"),
                        "decision": method_feedback.get("decision"),
                    }
                    selected_hypothesis["instruction"] = (
                        str(selected_hypothesis.get("instruction") or "").strip()
                        + " Retain applicable execution-only modify/add/validation steps from "
                          "applied_method_feedback. The current frozen Method and the rejecting "
                          "plan review take precedence over conflicting older feedback. Honor "
                          "current_method_changes_outline and the current component contracts; "
                          "do not restore superseded thresholds, gates or stopping rules. "
                          "Do not import later components or their validation gates into the "
                          "current component qualification."
                    ).strip()
            if not primary_validation_study and selected_hypothesis is None:
                await self._fail(
                    client,
                    session_id,
                    causation_id,
                    "permanent",
                    "hypothesis_not_found",
                    f"no hypothesis with id {selected_hypothesis_id!r} in the HypothesisBank "
                    f"and no confirmed method to plan",
                )
                return

            from autoresearch.experiments.repo_improve.execution_policy import (
                efficient_execution, EFFICIENT_EXPERIMENT)
            if efficient_execution() and isinstance(selected_hypothesis, dict):
                selected_hypothesis["execution_policy"] = EFFICIENT_EXPERIMENT
            if not primary_validation_study:
                experiment_plan = await asyncio.to_thread(  # ④ concurrency: offload the blocking LLM call off the shared web event loop
                    core.plan_experiment,
                    selected_hypothesis, verifier_spec, benchmark_spec, research_spec, self.llm_client,
                    prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "ExperimentPlannerAgent", self.config.prompt_patch()),
                    baseline_result=baseline_result,
                    plan_review=plan_review,
                )
        except core.ExperimentPlannerError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        experiment_plan_ref = self.spec_store.put(
            kind="experiment_plan",
            session_id=session_id,
            key="snapshot-" + artifact_digest(experiment_plan).split(":", 1)[1],
            obj=experiment_plan,
            emitter=AGENT_NAME,
        )

        # GEN mode: generate the concrete P1 design now so it can be human-reviewed at
        # ReviewingExperimentPlan and run as-is by the RunnerAgent. A live repository
        # baseline takes the separate repo-improve branch, which already uses the
        # repository's evaluator; generating a self-written GEN design there is unused
        # work and can delay the repository state machine behind an unrelated LLM call.
        live_repo_baseline = (
            isinstance(baseline_result, dict)
            and baseline_result.get("source_kind") == "live_repo"
        )
        if (os.environ.get(_GEN_ENV_FLAG) == "1" and not live_repo_baseline
                and not primary_validation_study):
            try:
                from autoresearch.experiments.general.design import design_experiment
                question = research_spec.get("research_goal") or ""
                gen_design = await asyncio.to_thread(design_experiment, self.llm_client, question)
                self.spec_store.put(
                    kind=_GEN_DESIGN_KIND, session_id=session_id, key=_EXPERIMENT_PLAN_KEY,
                    obj=gen_design, emitter=AGENT_NAME,
                )
            except Exception:  # noqa: BLE001 -- never fail planning on the P1-design nicety
                logger.exception("GEN design pre-generation failed (session %s)", session_id)

        completion_payload = {"experiment_plan_ref": experiment_plan_ref}
        self.spec_store.put(kind="audit", session_id=session_id, key=receipt_key,
                            obj={"request_digest": artifact_digest(payload), "payload": completion_payload},
                            emitter=AGENT_NAME)
        await client.publish(
            event_type="experiment.plan_ready",
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


__all__ = ["ExperimentPlannerAgentHandler", "SCHEMA_VERSION"]
