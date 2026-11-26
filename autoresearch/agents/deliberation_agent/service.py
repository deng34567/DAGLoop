"""DeliberationAgent service: wire the handler to an EventBusClient + SpecStore + LLM."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoresearch.agents.deliberation_agent.handler import DeliberationAgentHandler

logger = logging.getLogger(__name__)


def build_deliberation_service(
    *,
    client: Any,
    spec_store: Any,
    llm_client: Any | None = None,
    config_root: str | Path | None = None,
) -> DeliberationAgentHandler:
    """Assemble the DeliberationAgent handler and subscribe it on ``client``.

    ``llm_client`` defaults to a real ``HKGAIClient()``; tests inject a fake exposing
    ``.chat(messages=..., response_format=...)``.
    """
    if llm_client is None:
        from autoresearch.llm import HKGAIClient

        llm_client = HKGAIClient()

    handler = DeliberationAgentHandler(
        spec_store=spec_store, llm_client=llm_client, config_root=config_root
    )
    handler.register(client)
    return handler


__all__ = ["build_deliberation_service"]
