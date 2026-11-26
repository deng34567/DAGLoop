"""MethodAgent service: wire the handler to an EventBusClient + SpecStore (+ LLM)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.agents.method_agent.handler import MethodAgentHandler


def build_method_service(*, client: Any, spec_store: Any, llm_client: Any | None = None,
                         config_root: str | Path | None = None) -> MethodAgentHandler:
    """Assemble the MethodAgent handler and subscribe it. ``llm_client`` defaults to a real
    HKGAIClient; tests inject a fake exposing ``.chat(messages=..., response_format=...)``."""
    if llm_client is None:
        from autoresearch.llm import HKGAIClient
        llm_client = HKGAIClient()
    handler = MethodAgentHandler(spec_store=spec_store, llm_client=llm_client, config_root=config_root)
    handler.register(client)
    return handler


__all__ = ["build_method_service"]
