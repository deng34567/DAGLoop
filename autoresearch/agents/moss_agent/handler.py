"""
MOSS EventBus handler: the thin transport wrapper around core.
MOSS EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``moss.intervention_requested`` (payload ``{reason, counter_snapshot?}``) --
builds a ``MossActionGuard`` ONCE (in ``__init__``, from
``load_moss_action_matrix()``), runs ``core.decide`` with the requested
``reason`` (+ optional ``counter_snapshot``), persists the resulting
MossDecision (kind="moss_intervention"), and publishes
``moss.intervention_completed`` with its ref.

Never-hard-stall at the handler layer, per core's contract: a QUALITY
problem (the LLM can't produce a legal action) DEGRADES inside
``core.decide`` and still yields a normal ``MossDecision``
(``degraded=True``) -- this still becomes a ``moss.intervention_completed``,
never an ``agent.failed``. Only an INFRA failure (``core.MossLLMError``) or
an unexpected error is converted into ``agent.failed``, never a crash.

订阅一个事件：moss.intervention_requested（payload ``{reason,
counter_snapshot?}``）——在 __init__ 中用 load_moss_action_matrix() 构建一次
MossActionGuard，用请求的 reason（+ 可选 counter_snapshot）跑一次
core.decide，持久化得到的 MossDecision（kind="moss_intervention"），发布
moss.intervention_completed 及其 ref。

遵循 core 的契约，handler 层同样永不硬卡死：质量问题（LLM 给不出合法
action）会在 core.decide 内部降级，仍返回正常的 MossDecision
（degraded=True）——这仍然是一次 moss.intervention_completed，绝不是
agent.failed。只有基础设施失败（core.MossLLMError）或未预期错误才会转成
agent.failed，绝不崩溃。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from autoresearch.agents.moss_agent import core
from autoresearch.agents.moss_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.orchestrator.moss_matrix import MossActionGuard, load_moss_action_matrix
from autoresearch.experiments.repo_improve.component_plan import method_fingerprint
from autoresearch.spec_store.identity import artifact_digest, result_for_run
from autoresearch.spec_store import refs

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class MossAgentHandler:
    """Wires core.decide to a Spec Store + LLM + EventBus client.
    把 core.decide 接到 Spec Store + LLM + EventBus 客户端。"""

    def __init__(
        self,
        *,
        spec_store: Any,
        llm_client: Any,
        config_root: Any | None = None,
        matrix_path: Any | None = None,
    ) -> None:
        self.spec_store = spec_store
        self.llm_client = llm_client
        self.config = ConfigLoader(config_root)
        self.guard = MossActionGuard(load_moss_action_matrix(matrix_path))

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler's event on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的事件。"""
        client.subscribe(
            "moss.intervention_requested",
            self._make_eventbus_handler(client, self.handle_intervention_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- moss.intervention_requested ----

    async def handle_intervention_requested(self, client: Any, env: Any) -> None:
        """Run one Sync-Intervention decision for the requested reason,
        persist the MossDecision, publish moss.intervention_completed.
        为请求的 reason 跑一次同步介入决策，持久化 MossDecision，发布
        moss.intervention_completed。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        reason = payload.get("reason")
        counter_snapshot = payload.get("counter_snapshot")
        # 3.7: if a consolidated ResearchMemory exists, feed it to MOSS as diagnostic context so
        # its advice is informed by the run's accumulated findings. Best-effort: a missing/unreadable
        # ref must never block the intervention (MOSS still decides from reason + guard).
        research_memory = None
        memory_ref = payload.get("research_memory_ref")
        if memory_ref:
            try:
                research_memory = self.spec_store.get(memory_ref)
            except Exception as exc:  # noqa: BLE001 -- context is optional
                logger.warning("MOSS could not load research_memory_ref=%r: %s: %s",
                               memory_ref, type(exc).__name__, exc)

        # Component ownership lives in Method, not in a failed run's output tail.
        method_context = None
        method_ref = payload.get("method_ref")
        if method_ref:
            try:
                method = self.spec_store.get(method_ref)
                method_context = {key: method[key] for key in (
                    "approach", "changes_outline", "why_it_should_win", "assumptions", "components")
                                  if key in method}
                method_context.update(method_ref=method_ref,
                                      method_fingerprint=method_fingerprint(method))
            except Exception as exc:  # noqa: BLE001 -- context is optional
                logger.warning("MOSS could not load method_ref=%r: %s: %s",
                               method_ref, type(exc).__name__, exc)

        if method_context:
            baseline_ref = payload.get("baseline_result_ref")
            if baseline_ref:
                try:
                    baseline = self.spec_store.get(baseline_ref)
                    method_context["task_requirements"] = {key: baseline[key] for key in (
                        "metric_name", "lower_is_better", "fixed_config", "benchmark_contract",
                        "repository_contract") if key in baseline}
                except Exception as exc:  # noqa: BLE001 -- older sessions may lack a frozen baseline
                    logger.warning("MOSS could not load baseline_result_ref=%r: %s: %s",
                                   baseline_ref, type(exc).__name__, exc)
            spec_ref = payload.get("spec_interpretation_ref")
            if spec_ref:
                try:
                    semantics = self.spec_store.get(spec_ref).get("evaluation_semantics")
                    if semantics:
                        method_context.setdefault("task_requirements", {})["evaluation_semantics"] = semantics
                except Exception as exc:  # noqa: BLE001 -- context is optional
                    logger.warning("MOSS could not load spec_interpretation_ref=%r: %s: %s",
                                   spec_ref, type(exc).__name__, exc)

        # The ACTUAL failure evidence that triggered this intervention.  Execution failures carry
        # a RunRecord; plan-review failures happen before any run and instead carry the rejected
        # CritiqueReport.  Prefer the critique when present because a stale earlier RunRecord must
        # not hide the current plan failure.  This context is best-effort and never blocks MOSS.
        diagnostic = None
        diagnostic_ref = payload.get("critique_report_ref") or payload.get("run_record_ref")
        if reason == "baseline_reproduction_thrashing":
            # Baseline failures precede method RunRecords and must not be installed as the
            # frozen comparator merely to make their actual error visible to MOSS.
            diagnostic_ref = payload.get("baseline_attempt_ref") or diagnostic_ref
        if diagnostic_ref:
            try:
                diagnostic = self.spec_store.get(diagnostic_ref)
            except Exception as exc:  # noqa: BLE001 -- context is optional
                logger.warning("MOSS could not load diagnostic_ref=%r: %s: %s",
                               diagnostic_ref, type(exc).__name__, exc)

        # The orchestrator supplies this only for the current pre-run component failure.
        # Its earlier successful RunRecord is not the experiment that failed here.
        failure = payload.get("agent_failure")
        if isinstance(failure, dict) and failure.get("underlying_reason"):
            diagnostic = failure

        if method_context:
            # Preserve identities before the prompt's bounded output excerpt.
            # Never infer the producing Method from the currently selected Method.
            scope = {"ref": diagnostic_ref, "relation": "unscoped",
                     "method_ref": None, "method_fingerprint": None}
            result_context = {}
            if diagnostic_ref == payload.get("run_record_ref") and diagnostic_ref:
                try:
                    if refs.decode(diagnostic_ref).key.startswith("snapshot-"):
                        result = result_for_run(self.spec_store, diagnostic_ref)
                        fingerprint = (result.get("component_execution") or {}).get("method_fingerprint")
                        producer = result.get("method_ref")
                        scope.update(method_ref=producer, method_fingerprint=fingerprint)
                        if producer or fingerprint:
                            scope["relation"] = (
                                "current" if (not producer or producer == method_ref)
                                and (not fingerprint or fingerprint == method_context["method_fingerprint"])
                                else "historical")
                        result_context = {key: result.get(key) for key in (
                            "candidate_snapshot_ref", "method_metric", "accepted",
                            "diagnosis", "diagnosis_evidence", "component_execution",
                            "component_execution_started", "component_attempt_id",
                            "component_failed_attempt_id", "method_implementation_review")}
                except (KeyError, ValueError, TypeError):
                    pass  # Missing immutable provenance stays explicitly unscoped.
            if isinstance(failure, dict) and failure.get("underlying_reason"):
                scope.update(ref=None, relation="current_preparation",
                             method_ref=method_ref,
                             method_fingerprint=method_context["method_fingerprint"])
                result_context = {}
            diagnostic = {"scope": scope, "result": result_context,
                          "record": diagnostic or {}}

        try:
            # ④ concurrency: core.decide is a BLOCKING LLM round-trip. On the web
            # DeliveringBus (one shared event loop dispatching every session) running it
            # inline serializes all sessions -- one slow decision starves the others.
            # Offload it to a worker thread (the same pattern _adapter/broker_bus.py uses).
            decision = await asyncio.to_thread(
                core.decide, reason, counter_snapshot, self.guard, self.llm_client,
                research_memory=research_memory, diagnostic=diagnostic,
                method_context=method_context,
                execution_context={"sandbox_native_tool_calling": os.environ.get(
                    "AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", "1") != "0"})
        except core.MossLLMError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient",
                f"{type(exc).__name__}: {exc}",
            )
            return

        decision = dict(decision)
        decision["scope"] = {"session_id": session_id,
                             "method_fingerprint": (method_context or {}).get("method_fingerprint")}
        self.spec_store.put(
            kind="moss_intervention",
            session_id=session_id,
            key=f"{reason}-v1",
            obj=decision,
            emitter=AGENT_NAME,
        )
        # A later intervention of the same reason must not replace the decision
        # referenced by an in-flight completion event.
        moss_intervention_ref = self.spec_store.put(
            kind="moss_intervention", session_id=session_id,
            key="snapshot-" + artifact_digest(decision).split(":", 1)[1],
            obj=decision, emitter=AGENT_NAME)

        # Session-specific guidance travels by immutable ref, never a shared prompt file.

        await client.publish(
            event_type="moss.intervention_completed",
            session_id=session_id,
            payload={
                "reason": reason,
                "action": decision["action"],
                "moss_intervention_ref": moss_intervention_ref,
                "degraded": decision["degraded"],
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
        underlying_reason: str,
    ) -> None:
        await client.publish(
            event_type="agent.failed",
            session_id=session_id,
            payload={
                "agent_name": AGENT_NAME,
                "reason_classification": reason_classification,
                "underlying_reason": underlying_reason,
            },
            causation_id=causation_id,
        )


__all__ = ["MossAgentHandler", "SCHEMA_VERSION"]
