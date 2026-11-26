"""
MOSS service: wires the handler to a real EventBusClient + SpecStore (+ LLM),
and provides the standalone process entry point.
MOSS 服务：把 handler 接到真实的 EventBusClient + SpecStore（+ LLM），并提供
独立进程入口。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from autoresearch.agents.moss_agent.handler import MossAgentHandler
from autoresearch.agents.moss_agent.kernel import AGENT_NAME
from autoresearch.agents.moss_agent.observation_store import MossObservationStore
from autoresearch.agents.moss_agent.observer import MossObserver

logger = logging.getLogger(__name__)


def build_moss_service(
    *,
    client: Any,
    spec_store: Any,
    llm_client: Any | None = None,
    config_root: str | Path | None = None,
    matrix_path: str | Path | None = None,
    observation_db_path: str | Path | None = None,
    modeb_threshold: int | None = None,
) -> MossAgentHandler:
    """Assemble the MOSS handler (Mode A) + observer (Mode B) and subscribe
    both on ``client``.
    装配 MOSS handler（Mode A）+ observer（Mode B），并在 ``client`` 上完成
    订阅。

    ``llm_client`` defaults to a real ``HKGAIClient()``; tests inject a fake
    exposing ``.chat(messages=..., response_format=...)``. ``matrix_path``
    optionally overrides the MOSS action-matrix YAML the guard is built from.
    ``llm_client`` 默认真实 ``HKGAIClient()``；测试注入带 ``.chat(...)`` 的
    fake。``matrix_path`` 可选覆盖 guard 所依据的 MOSS 动作矩阵 YAML。

    ``observation_db_path`` / ``modeb_threshold`` configure Mode B (the
    cross-session observation substrate): a ``MossObservationStore`` is
    opened at ``observation_db_path`` (falling back to ``spec_store.root /
    "moss.db"`` when ``spec_store`` has a ``.root``, then env
    ``AUTORESEARCH_MOSS_DB``, then a tmp default) and a ``MossObserver`` is
    subscribed to ``agent.failed`` + ``moss.intervention_completed``
    ALONGSIDE (not instead of) Mode A's existing
    ``moss.intervention_requested`` subscription. IMPORTANT: a real
    deployment MUST pass a PERSISTENT ``observation_db_path`` -- Mode B's
    cross-session pattern detection only works if the same DB is reused
    across runs; a per-run tmp path only ever "learns" within that one run
    and is discarded afterwards.
    ``observation_db_path`` / ``modeb_threshold`` 配置 Mode B（跨会话观察
    子系统）：``MossObservationStore`` 在 ``observation_db_path`` 处打开
    （回退到 ``spec_store.root / "moss.db"``，若无则 env
    ``AUTORESEARCH_MOSS_DB``，再无则 tmp 默认路径），``MossObserver`` 订阅
    ``agent.failed`` + ``moss.intervention_completed``，与 Mode A 既有的
    ``moss.intervention_requested`` 订阅并存。重要：生产部署必须传入
    PERSISTENT 的 ``observation_db_path`` —— Mode B 的跨会话模式检测依赖
    同一个数据库在多次运行间被复用；每次运行都用临时路径的话，只能在单次
    运行内“学习”，运行结束后即丢失。
    """
    if llm_client is None:
        from autoresearch.llm import HKGAIClient

        llm_client = HKGAIClient()

    handler = MossAgentHandler(
        spec_store=spec_store,
        llm_client=llm_client,
        config_root=config_root,
        matrix_path=matrix_path,
    )
    handler.register(client)

    db_path = observation_db_path
    if db_path is None:
        spec_store_root = getattr(spec_store, "root", None)
        if spec_store_root is not None:
            db_path = Path(spec_store_root) / "moss.db"
        else:
            db_path = os.environ.get(
                "AUTORESEARCH_MOSS_DB", "/tmp/autoresearch-moss.db"
            )
    observation_store = MossObservationStore(db_path)
    observer = MossObserver(observation_store, threshold_sessions=modeb_threshold)

    # Mode B: additive subscriptions alongside Mode A's
    # moss.intervention_requested (registered above). Subscribed to these
    # TWO SPECIFIC event types (not "*") -- the v1 classifier only handles
    # these two, and specific subscriptions are robust and sufficient.
    client.subscribe("agent.failed", observer.on_event)
    client.subscribe("moss.intervention_completed", observer.on_event)

    return handler


async def run_moss_agent() -> None:
    """Standalone process entry point: build a real EventBusClient + SpecStore
    (+ default HKGAIClient) and run forever.
    独立进程入口：建真实 EventBusClient + SpecStore（+ 默认 HKGAIClient），
    run_forever。"""
    from autoresearch.eventbus.broker import EventBusClient
    from autoresearch.spec_store import SpecStore

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381")
    sqlite_path = os.environ.get(
        "MOSS_AGENT_SQLITE",
        f"/tmp/autoresearch-moss-agent-{uuid.uuid4().hex[:8]}.db",
    )
    spec_store_root = os.environ.get(
        "AUTORESEARCH_SPEC_STORE_ROOT", "/tmp/autoresearch-spec-store"
    )
    # service.py -> moss_agent -> agents -> autoresearch;
    # eventbus lives at autoresearch/eventbus.
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
    build_moss_service(client=client, spec_store=spec_store)
    logger.info("MOSS service started; entering run_forever()")
    await client.run_forever()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_moss_agent())


if __name__ == "__main__":
    main()


__all__ = ["build_moss_service", "run_moss_agent", "main"]
