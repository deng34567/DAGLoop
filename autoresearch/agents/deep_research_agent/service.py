"""
DeepResearchAgent service: wires the handler to a real EventBusClient +
SpecStore (+ search backend + LLM), and provides the standalone process
entry point.
DeepResearchAgent 服务：把 handler 接到真实的 EventBusClient + SpecStore
（+ 搜索后端 + LLM），并提供独立进程入口。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from autoresearch.agents.deep_research_agent.handler import DeepResearchAgentHandler
from autoresearch.agents.deep_research_agent.kernel import AGENT_NAME

logger = logging.getLogger(__name__)


def build_deep_research_service(
    *,
    client: Any,
    spec_store: Any,
    backend: Any | None = None,
    llm_client: Any | None = None,
    config_root: str | Path | None = None,
) -> DeepResearchAgentHandler:
    """Assemble the DeepResearchAgent handler and subscribe it on ``client``.
    装配 DeepResearchAgent handler 并在 ``client`` 上完成订阅。

    ``backend`` defaults to the opt-in, NOT-wired ``PaperQA2Backend()`` (the
    production default — never invoked in tests, which inject ``FakeBackend``).
    ``llm_client`` defaults to a real ``HKGAIClient()``; tests inject a fake
    exposing ``.chat(messages=..., response_format=...)``.
    ``backend`` 默认使用 opt-in、未接线的 ``PaperQA2Backend()``（生产默认——
    测试从不调用，测试注入 ``FakeBackend``）。``llm_client`` 默认真实
    ``HKGAIClient()``；测试注入带 ``.chat(...)`` 的 fake。
    """
    if backend is None:
        from autoresearch.agents._reuse.deep_research_backend import PaperQA2Backend

        backend = PaperQA2Backend()

    if llm_client is None:
        from autoresearch.llm import HKGAIClient

        llm_client = HKGAIClient()

    handler = DeepResearchAgentHandler(
        spec_store=spec_store, backend=backend, llm_client=llm_client, config_root=config_root
    )
    handler.register(client)
    return handler


async def run_deep_research_agent() -> None:
    """Standalone process entry point: build a real EventBusClient + SpecStore
    (+ default PaperQA2Backend + HKGAIClient) and run forever.
    独立进程入口：建真实 EventBusClient + SpecStore（+ 默认 PaperQA2Backend +
    HKGAIClient），run_forever。"""
    from autoresearch.eventbus.broker import EventBusClient
    from autoresearch.spec_store import SpecStore

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381")
    sqlite_path = os.environ.get(
        "DEEP_RESEARCH_AGENT_SQLITE",
        f"/tmp/autoresearch-deep-research-{uuid.uuid4().hex[:8]}.db",
    )
    spec_store_root = os.environ.get(
        "AUTORESEARCH_SPEC_STORE_ROOT", "/tmp/autoresearch-spec-store"
    )
    # service.py -> deep_research_agent -> agents -> autoresearch; eventbus
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
    build_deep_research_service(client=client, spec_store=spec_store)
    logger.info("DeepResearchAgent service started; entering run_forever()")
    await client.run_forever()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_deep_research_agent())


if __name__ == "__main__":
    main()


__all__ = ["build_deep_research_service", "run_deep_research_agent", "main"]
