"""
DataAnalysisAgent EventBus handler: the thin transport wrapper around core.
DataAnalysisAgent EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``data_analysis.requested`` (payload ``{external_data_upload_ref,
external_experiment_package_ref, strictness}``) -- reads the uploaded
external dataset and the frozen external experiment package from the Spec
Store by their refs (provenance -- NEVER from the event payload beyond the
refs themselves), runs ``core.analyze_external_data``, persists the
resulting ExternalDataAnalysisReport (kind="external_data_analysis_report",
key="v1"), and publishes ``data_analysis.ready`` with its ref and the
report's ``consensus_status``.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅一个事件：data_analysis.requested（按 external_data_upload_ref /
external_experiment_package_ref 从 Spec Store 读回外部上传数据集与冻结的外部
实验包，跑一次 core.analyze_external_data，持久化 ExternalDataAnalysisReport，
发布 data_analysis.ready）。core 的任何失败（或未预期错误）都转成
agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
from typing import Any

from autoresearch.agents.data_analysis_agent import core
from autoresearch.agents.data_analysis_agent.kernel import AGENT_NAME, ConfigLoader

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_EXTERNAL_DATA_ANALYSIS_REPORT_KEY = "v1"


class DataAnalysisAgentHandler:
    """Wires core.analyze_external_data to a Spec Store + LLM + EventBus client.
    把 core.analyze_external_data 接到 Spec Store + LLM + EventBus 客户端。"""

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
        """Subscribe this handler's one event on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的一个事件。"""
        client.subscribe(
            "data_analysis.requested",
            self._make_eventbus_handler(client, self.handle_data_analysis_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- data_analysis.requested ----

    async def handle_data_analysis_requested(self, client: Any, env: Any) -> None:
        """Resolve the external data + protocol from the Spec Store, run one
        analyze_external_data, persist the ExternalDataAnalysisReport,
        publish data_analysis.ready.
        从 Spec Store 解析外部数据 + 协议，跑一次 analyze_external_data，
        持久化 ExternalDataAnalysisReport，发布 data_analysis.ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        strictness = payload.get("strictness")

        try:
            external_data = self.spec_store.get(payload.get("external_data_upload_ref"))
            external_protocol = self.spec_store.get(
                payload.get("external_experiment_package_ref")
            )
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            report = await asyncio.to_thread(
                core.analyze_external_data,
                external_data,
                external_protocol,
                strictness,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "DataAnalysisAgent", self.config.prompt_patch()),
            )
        except core.DataAnalysisError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        external_data_analysis_report_ref = self.spec_store.put(
            kind="external_data_analysis_report",
            session_id=session_id,
            key=_EXTERNAL_DATA_ANALYSIS_REPORT_KEY,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="data_analysis.ready",
            session_id=session_id,
            payload={
                "external_data_analysis_report_ref": external_data_analysis_report_ref,
                "consensus_status": report["consensus_status"],
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


__all__ = ["DataAnalysisAgentHandler", "SCHEMA_VERSION"]
