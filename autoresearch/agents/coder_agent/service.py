"""
CoderAgent service: wires the handler to a real EventBusClient + SpecStore
(+ LLM), and provides the standalone process entry point.
CoderAgent 服务：把 handler 接到真实的 EventBusClient + SpecStore（+ LLM），
并提供独立进程入口。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from autoresearch.agents.coder_agent.handler import CoderAgentHandler
from autoresearch.agents.coder_agent.kernel import AGENT_NAME

logger = logging.getLogger(__name__)


def build_coder_service(
    *,
    client: Any,
    spec_store: Any,
    llm_client: Any | None = None,
    config_root: str | Path | None = None,
) -> CoderAgentHandler:
    """Assemble the CoderAgent handler and subscribe it on ``client``.
    装配 CoderAgent handler 并在 ``client`` 上完成订阅。

    ``llm_client`` defaults to a real ``HKGAIClient()``; tests inject a fake
    exposing ``.chat(messages=..., response_format=...)``.
    ``llm_client`` 默认真实 ``HKGAIClient()``；测试注入带 ``.chat(...)`` 的
    fake。
    """
    if llm_client is None:
        from autoresearch.llm import HKGAIClient

        llm_client = HKGAIClient()

    handler = CoderAgentHandler(
        spec_store=spec_store, llm_client=llm_client, config_root=config_root
    )
    handler.register(client)
    return handler


async def run_coder_agent() -> None:
    """Standalone process entry point: build a real EventBusClient + SpecStore
    (+ default HKGAIClient) and run forever.
    独立进程入口：建真实 EventBusClient + SpecStore（+ 默认 HKGAIClient），
    run_forever。"""
    from autoresearch.eventbus.broker import EventBusClient
    from autoresearch.spec_store import SpecStore

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381")
    sqlite_path = os.environ.get(
        "CODER_AGENT_SQLITE",
        f"/tmp/autoresearch-coder-{uuid.uuid4().hex[:8]}.db",
    )
    spec_store_root = os.environ.get(
        "AUTORESEARCH_SPEC_STORE_ROOT", "/tmp/autoresearch-spec-store"
    )
    # service.py -> coder_agent -> agents -> autoresearch; eventbus lives at
    # autoresearch/eventbus.
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
    build_coder_service(client=client, spec_store=spec_store)
    logger.info("CoderAgent service started; entering run_forever()")
    await client.run_forever()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_coder_agent())


if __name__ == "__main__":
    main()


__all__ = ["build_coder_service", "run_coder_agent", "main"]
