"""
RankingJudgeAgent EventBus handler: the thin transport wrapper around core.
RankingJudgeAgent EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``ranking.requested`` (payload ``{candidate_kind, candidate_ids?,
source_ref}``) -- reads the hypothesis_bank artifact from the Spec Store by
``source_ref`` (provenance -- NEVER from the event payload beyond the ref
itself), builds the candidate set (all bank hypotheses, or the
``candidate_ids`` subset if given), runs ``core.rank_candidates``, persists
the resulting RankingReport (kind="ranking_report", key="v1"), and publishes
``ranking.ready`` with its ref.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅一个事件：ranking.requested（按 source_ref 从 Spec Store 读回
hypothesis_bank，组装候选集合——全部或 candidate_ids 子集，跑一次
core.rank_candidates，持久化 RankingReport，发布 ranking.ready）。core 的
任何失败（或未预期错误）都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
from typing import Any

from autoresearch.agents.ranking_judge_agent import core
from autoresearch.agents.ranking_judge_agent.kernel import AGENT_NAME, ConfigLoader

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_RANKING_REPORT_KEY = "v1"


class RankingJudgeAgentHandler:
    """Wires core.rank_candidates to a Spec Store + LLM + EventBus client.
    把 core.rank_candidates 接到 Spec Store + LLM + EventBus 客户端。"""

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
            "ranking.requested",
            self._make_eventbus_handler(client, self.handle_ranking_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- ranking.requested ----

    def _resolve_candidates(self, payload: dict) -> list[dict]:
        """Build ``candidates = [{id, summary}]`` from the hypothesis_bank at
        ``source_ref``, filtered to ``candidate_ids`` if given (preserving
        bank order), else all bank hypotheses.
        从 ``source_ref`` 处的 hypothesis_bank 组装 ``candidates``，若给定
        ``candidate_ids`` 则过滤为该子集（保持 bank 顺序），否则用全部。"""
        bank = self.spec_store.get(payload.get("source_ref"))
        hypotheses = bank.get("hypotheses", [])
        candidate_ids = payload.get("candidate_ids")
        if candidate_ids:
            wanted = set(candidate_ids)
            hypotheses = [h for h in hypotheses if h["id"] in wanted]
        return [{"id": h["id"], "summary": h["statement"]} for h in hypotheses]

    async def handle_ranking_requested(self, client: Any, env: Any) -> None:
        """Resolve candidates from the hypothesis_bank, run one
        rank_candidates, persist the RankingReport, publish ranking.ready.
        从 hypothesis_bank 解析候选集合，跑一次 rank_candidates，持久化
        RankingReport，发布 ranking.ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        candidate_kind = payload.get("candidate_kind")

        try:
            candidates = self._resolve_candidates(payload)
            report = await asyncio.to_thread(  # ④ concurrency: offload the blocking LLM call off the shared web event loop
                core.rank_candidates,
                candidates,
                candidate_kind,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "RankingJudgeAgent", self.config.prompt_patch()),
            )
        except core.RankingError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        ranking_report_ref = self.spec_store.put(
            kind="ranking_report",
            session_id=session_id,
            key=_RANKING_REPORT_KEY,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="ranking.ready",
            session_id=session_id,
            payload={
                "ranking_report_ref": ranking_report_ref,
                "candidate_kind": candidate_kind,
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


__all__ = ["RankingJudgeAgentHandler", "SCHEMA_VERSION"]
