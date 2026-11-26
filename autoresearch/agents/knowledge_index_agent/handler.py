"""
KnowledgeIndexAgent EventBus handler: the thin transport wrapper around core.
KnowledgeIndexAgent EventBus handler：core 之上的薄传输层。

Subscribes THREE events:

1. ``literature.brief_ready`` (payload ``{literature_brief_ref,
   field_trajectory_brief_ref}``) -- reads the LiteratureBrief from the Spec
   Store by ``literature_brief_ref`` (provenance -- NEVER from the payload),
   maps its ``key_papers`` (Paper ``{title, id_or_url, year?, note?}``) into
   IndexEntryInputs (``path_or_ref = id_or_url``, ``summary = title`` [+
   ``" — " + note`` if present]), and auto-ingests them via
   ``core.index_entries``. No result event is published.
2. ``knowledge.ingest_requested`` (payload ``{items}``) -- ingests the given
   IndexEntryInput items directly. No result event is published.
3. ``knowledge.retrieve_requested`` (payload ``{query, top_k?}``) -- ranks
   the session's index against ``query`` via the injected retrieval backend
   and publishes ``knowledge.retrieve_result``.

Each session's index is persisted as one blob (kind="knowledge_index",
key="v1") in the Spec Store, merged in place across all three events. Any
core failure (or unexpected error) is converted into ``agent.failed``, never
a crash.

订阅三个事件：literature.brief_ready（按 ref 从 Spec Store 读回
LiteratureBrief，映射 key_papers 自动入库，无结果事件）、
knowledge.ingest_requested（直接入库给定条目，无结果事件）、
knowledge.retrieve_requested（用注入的检索后端对 query 排序，发布
knowledge.retrieve_result）。每个 session 的索引以单个 blob
（kind="knowledge_index", key="v1"）持久化，三个事件共享合并。core 的
任何失败（或未预期错误）都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

import logging
from typing import Any

from autoresearch.agents.knowledge_index_agent import core
from autoresearch.agents.knowledge_index_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store import refs as spec_refs

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_INDEX_KEY = "v1"
_DEFAULT_TOP_K = 5


class KnowledgeIndexAgentHandler:
    """Wires core.index_entries/retrieve to a Spec Store + retrieval backend
    + EventBus client.
    把 core.index_entries/retrieve 接到 Spec Store + 检索后端 + EventBus
    客户端。"""

    def __init__(
        self,
        *,
        spec_store: Any,
        backend: Any,
        config_root: Any | None = None,
    ) -> None:
        self.spec_store = spec_store
        self.backend = backend
        self.config = ConfigLoader(config_root)

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler's three events on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的三个事件。"""
        client.subscribe(
            "literature.brief_ready",
            self._make_eventbus_handler(client, self.handle_brief_ready),
        )
        client.subscribe(
            "knowledge.ingest_requested",
            self._make_eventbus_handler(client, self.handle_ingest_requested),
        )
        client.subscribe(
            "knowledge.retrieve_requested",
            self._make_eventbus_handler(client, self.handle_retrieve_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- per-session index blob / 每 session 索引 blob ----

    def _load_index(self, session_id: str) -> list[dict]:
        ref = spec_refs.encode("knowledge_index", session_id, _INDEX_KEY)
        try:
            blob = self.spec_store.get(ref)
        except KeyError:
            return []
        return blob.get("entries", [])

    def _save_index(self, session_id: str, entries: list[dict]) -> str:
        return self.spec_store.put(
            kind="knowledge_index",
            session_id=session_id,
            key=_INDEX_KEY,
            obj={"entries": entries},
            emitter=AGENT_NAME,
        )

    # ---- literature.brief_ready ----

    async def handle_brief_ready(self, client: Any, env: Any) -> None:
        """Resolve the LiteratureBrief from the store, map its key_papers into
        IndexEntryInputs, and auto-ingest them.
        从 store 解析 LiteratureBrief，把 key_papers 映射为 IndexEntryInput
        并自动入库。"""
        session_id = env.session_id
        causation_id = env.event_id
        literature_brief_ref = (env.payload or {}).get("literature_brief_ref")

        try:
            literature_brief = self.spec_store.get(literature_brief_ref)
            inputs = [
                self._paper_to_index_entry_input(paper)
                for paper in literature_brief.get("key_papers", [])
            ]
            existing = self._load_index(session_id)
            merged = core.index_entries(
                existing,
                inputs,
                produced_by_state="DeepResearching",
                source="literature.brief_ready",
                entry_type="paper",
            )
        except core.KnowledgeIndexError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self._save_index(session_id, merged)

    @staticmethod
    def _paper_to_index_entry_input(paper: dict) -> dict:
        summary = paper.get("title")
        note = paper.get("note")
        if summary and note:
            summary = f"{summary} — {note}"
        return {"path_or_ref": paper.get("id_or_url"), "summary": summary}

    # ---- knowledge.ingest_requested ----

    async def handle_ingest_requested(self, client: Any, env: Any) -> None:
        """Ingest the given IndexEntryInput items directly.
        直接入库给定的 IndexEntryInput 条目。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        items = payload.get("items", [])
        produced_by_state = payload.get("produced_by_state", "external")

        try:
            existing = self._load_index(session_id)
            merged = core.index_entries(
                existing,
                items,
                produced_by_state=produced_by_state,
                source="knowledge.ingest_requested",
                entry_type="artifact",
            )
        except core.KnowledgeIndexError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self._save_index(session_id, merged)

    # ---- knowledge.retrieve_requested ----

    async def handle_retrieve_requested(self, client: Any, env: Any) -> None:
        """Rank the session's index against ``query`` and publish
        knowledge.retrieve_result.
        对 session 索引按 query 排序并发布 knowledge.retrieve_result。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        query = payload.get("query", "")
        top_k = payload.get("top_k") or _DEFAULT_TOP_K

        try:
            entries = self._load_index(session_id)
            matched = core.retrieve(query, entries, self.backend, top_k)
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        await client.publish(
            event_type="knowledge.retrieve_result",
            session_id=session_id,
            payload={"query": query, "entries": matched, "hit_count": len(matched)},
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


__all__ = ["KnowledgeIndexAgentHandler", "SCHEMA_VERSION"]
