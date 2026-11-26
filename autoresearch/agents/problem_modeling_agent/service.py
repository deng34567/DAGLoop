"""
ProblemModelingAgent service: wires the handler to a real EventBusClient +
SpecStore (+ LLM), and provides the standalone process entry point.
ProblemModelingAgent 服务：把 handler 接到真实的 EventBusClient + SpecStore
（+ LLM），并提供独立进程入口。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from autoresearch.agents.problem_modeling_agent.handler import ProblemModelingAgentHandler
from autoresearch.agents.problem_modeling_agent.kernel import AGENT_NAME

logger = logging.getLogger(__name__)


def build_problem_modeling_service(
    *,
    client: Any,
    spec_store: Any,
    llm_client: Any | None = None,
    semantic_guardian_llm: Any | None = None,
    config_root: str | Path | None = None,
) -> ProblemModelingAgentHandler:
    """Assemble the ProblemModelingAgent handler and subscribe it on ``client``.
    装配 ProblemModelingAgent handler 并在 ``client`` 上完成订阅。

    ``llm_client`` defaults to a real ``HKGAIClient()``; tests inject a fake
    exposing ``.chat(messages=..., response_format=...)``.
    ``llm_client`` 默认真实 ``HKGAIClient()``；测试注入带 ``.chat(...)`` 的 fake。
    """
    if llm_client is None:
        from autoresearch.llm import HKGAIClient

        llm_client = HKGAIClient()

    handler = ProblemModelingAgentHandler(
        spec_store=spec_store, llm_client=llm_client,
        semantic_guardian_llm=semantic_guardian_llm,
        config_root=config_root,
    )
    # Subscribes all four: problem_modeling.requested, human.clarification_answer
    # (M1 ClarifyingResearchGoal), problem_modeling.reinterpret_requested
    # (ReinterpretingSpec), and problem_modeling.model_requested
    # (ModelingProblem) — see ProblemModelingAgentHandler.register().
    handler.register(client)
    return handler


async def run_problem_modeling_agent() -> None:
    """Standalone process entry point: build a real EventBusClient + SpecStore
    (+ default HKGAIClient) and run forever.
    独立进程入口：建真实 EventBusClient + SpecStore（+ 默认 HKGAIClient），
    run_forever。"""
    from autoresearch.eventbus.broker import EventBusClient
    from autoresearch.spec_store import SpecStore

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381")
    sqlite_path = os.environ.get(
        "PROBLEM_MODELING_AGENT_SQLITE",
        f"/tmp/autoresearch-problem-modeling-{uuid.uuid4().hex[:8]}.db",
    )
    spec_store_root = os.environ.get(
        "AUTORESEARCH_SPEC_STORE_ROOT", "/tmp/autoresearch-spec-store"
    )
    # service.py -> problem_modeling_agent -> agents -> autoresearch; eventbus
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
    build_problem_modeling_service(client=client, spec_store=spec_store)
    logger.info("ProblemModelingAgent service started; entering run_forever()")
    await client.run_forever()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_problem_modeling_agent())


if __name__ == "__main__":
    main()


__all__ = ["build_problem_modeling_service", "run_problem_modeling_agent", "main"]
