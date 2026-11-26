"""
RunnerAgent service: wires the handler to a real EventBusClient + SpecStore
(+ RunnerBackend), and provides the standalone process entry point.
RunnerAgent 服务：把 handler 接到真实的 EventBusClient + SpecStore
（+ RunnerBackend），并提供独立进程入口。

Unlike every other agent's service module, ``build_runner_service`` injects a
RunnerBackend, not an LLM client -- and its default is ALWAYS
``FakeRunnerBackend`` (in-memory, canned result): real execution backends
(E2BRunnerBackend / LocalSubprocessRunnerBackend) are deferred/opt-in past
M10 and are NEVER wired here.
与其他 agent 的 service 模块不同，``build_runner_service`` 注入的是
RunnerBackend 而非 LLM client —— 且默认值恒为 ``FakeRunnerBackend``
（内存、预设结果）：真实执行后端推迟到 M10 之后、按需接入，本模块绝不
在此接入它们。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.agents.runner_agent.kernel import AGENT_NAME

logger = logging.getLogger(__name__)


def build_runner_service(
    *,
    client: Any,
    spec_store: Any,
    backend: Any | None = None,
    config_root: str | Path | None = None,
    p1_backend: Any | None = None,
    p1_llm: Any | None = None,
    repository_search_backend: Any | None = None,
) -> RunnerAgentHandler:
    """Assemble the RunnerAgent handler and subscribe it on ``client``.
    装配 RunnerAgent handler 并在 ``client`` 上完成订阅。

    ``backend`` defaults to ``FakeRunnerBackend()`` -- NEVER a real backend;
    tests (and this default) inject/use the in-memory fake.
    ``backend`` 默认 ``FakeRunnerBackend()`` —— 绝不是真实后端；测试（以及
    这个默认值）都用内存版 fake。

    ``p1_backend`` / ``p1_llm`` are the OPTIONAL P1 general honest-experiment
    execution backend + LLM, forwarded verbatim to ``RunnerAgentHandler``.
    DEFAULT-OFF: both default to None so the gen branch stays inert and the
    handler runs byte-identically -- the branch only fires when a driver
    injects a ``p1_backend`` AND sets ``AUTORESEARCH_GEN_EXPERIMENT=1``.
    """
    if backend is None:
        from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend

        backend = FakeRunnerBackend()

    handler = RunnerAgentHandler(
        spec_store=spec_store,
        backend=backend,
        config_root=config_root,
        p1_backend=p1_backend,
        p1_llm=p1_llm,
        repository_search_backend=repository_search_backend,
    )
    handler.register(client)
    return handler


async def run_runner_agent() -> None:
    """Standalone process entry point: build a real EventBusClient + SpecStore
    (+ default FakeRunnerBackend) and run forever.
    独立进程入口：建真实 EventBusClient + SpecStore（+ 默认
    FakeRunnerBackend），run_forever。"""
    from autoresearch.eventbus.broker import EventBusClient
    from autoresearch.spec_store import SpecStore

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381")
    sqlite_path = os.environ.get(
        "RUNNER_AGENT_SQLITE",
        f"/tmp/autoresearch-runner-{uuid.uuid4().hex[:8]}.db",
    )
    spec_store_root = os.environ.get(
        "AUTORESEARCH_SPEC_STORE_ROOT", "/tmp/autoresearch-spec-store"
    )
    # service.py -> runner_agent -> agents -> autoresearch; eventbus lives at
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
    build_runner_service(client=client, spec_store=spec_store)
    logger.info("RunnerAgent service started; entering run_forever()")
    await client.run_forever()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_runner_agent())


if __name__ == "__main__":
    main()


__all__ = ["build_runner_service", "run_runner_agent", "main"]
