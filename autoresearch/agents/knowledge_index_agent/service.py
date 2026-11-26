"""
KnowledgeIndexAgent service: wires the handler to a real EventBusClient +
SpecStore (+ retrieval backend), and provides the standalone process entry
point.
KnowledgeIndexAgent 服务：把 handler 接到真实的 EventBusClient + SpecStore
（+ 检索后端），并提供独立进程入口。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from autoresearch.agents.knowledge_index_agent.handler import KnowledgeIndexAgentHandler
from autoresearch.agents.knowledge_index_agent.kernel import AGENT_NAME

logger = logging.getLogger(__name__)


def build_knowledge_index_service(
    *,
    client: Any,
    spec_store: Any,
    backend: Any | None = None,
    config_root: str | Path | None = None,
) -> KnowledgeIndexAgentHandler:
    """Assemble the KnowledgeIndexAgent handler and subscribe it on ``client``.
    装配 KnowledgeIndexAgent handler 并在 ``client`` 上完成订阅。

    ``backend`` defaults to the real ``KeywordBackend()`` (local, deterministic
    lexical ranking, NO LLM/embeddings/network).
    ``backend`` 默认使用真实 ``KeywordBackend()``（本地确定性词形排序，无
    LLM/embedding/网络依赖）。
    """
    if backend is None:
        from autoresearch.agents._reuse.knowledge_index_backend import KeywordBackend

        backend = KeywordBackend()

    handler = KnowledgeIndexAgentHandler(
        spec_store=spec_store, backend=backend, config_root=config_root
    )
    handler.register(client)
    return handler


async def run_knowledge_index_agent() -> None:
    """Standalone process entry point: build a real EventBusClient + SpecStore
    (+ default KeywordBackend) and run forever.
    独立进程入口：建真实 EventBusClient + SpecStore（+ 默认 KeywordBackend），
    run_forever。"""
    from autoresearch.eventbus.broker import EventBusClient
    from autoresearch.spec_store import SpecStore

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381")
    sqlite_path = os.environ.get(
        "KNOWLEDGE_INDEX_AGENT_SQLITE",
        f"/tmp/autoresearch-knowledge-index-{uuid.uuid4().hex[:8]}.db",
    )
    spec_store_root = os.environ.get(
        "AUTORESEARCH_SPEC_STORE_ROOT", "/tmp/autoresearch-spec-store"
    )
    # service.py -> knowledge_index_agent -> agents -> autoresearch; eventbus
    # lives at autoresearch/eventbus.
    default_eventbus_root = Path(__file__).resolve().parents[2] / "eventbus"
    eventbus_root = Path(
        os.environ.get("AUTORESEARCH_EVENTBUS_ROOT", str(default_eventbus_root))
    )

    client = EventBusClient(
        redis_url=redis_url,
        sqlite_path=sqlite_path,
        emitter_name=AGENT_NAME,
        eventbus_root=eventbus_root,
    )
    await client.start()
    spec_store = SpecStore(spec_store_root)
    build_knowledge_index_service(client=client, spec_store=spec_store)
    logger.info("KnowledgeIndexAgent service started; entering run_forever()")
    await client.run_forever()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_knowledge_index_agent())


if __name__ == "__main__":
    main()


__all__ = ["build_knowledge_index_service", "run_knowledge_index_agent", "main"]
