"""
DeliberationAgent EventBus handler: the thin transport wrapper around core.

Subscribes ``deliberation.requested`` (payload ``{research_spec_ref,
literature_brief_ref?}``), resolves the frozen specs from the Spec Store by ref
(provenance -- never from the payload), runs ``core.deliberate`` (bounded N-angle
cross-reference), persists the resulting ``DeliberationReport``
(kind="deliberation_report"), and publishes ``deliberation.ready`` with its ref.

FAIL-OPEN: deliberation is a DEEPENING step, never a gate. Any failure (LLM down,
bad refs, unexpected error) is degraded into a MINIMAL report + a normal
``deliberation.ready`` so the flow always proceeds to SelectingBenchmark -- it never
emits ``agent.failed`` and never stalls the session.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from autoresearch.agents.deliberation_agent import core
from autoresearch.agents.deliberation_agent.kernel import AGENT_NAME, ConfigLoader

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_REPORT_KEY = "final"

# N angles: bounded (professor's convergence worry). Env-overridable for experiments.
_N_ANGLES_ENV = "AUTORESEARCH_DELIBERATION_ANGLES"


class DeliberationAgentHandler:
    """Wires core.deliberate to a Spec Store + LLM + EventBus client."""

    def __init__(self, *, spec_store: Any, llm_client: Any, config_root: Any | None = None) -> None:
        self.spec_store = spec_store
        self.llm_client = llm_client
        self.config = ConfigLoader(config_root)

    def register(self, client: Any) -> None:
        client.subscribe(
            "deliberation.requested",
            self._make_eventbus_handler(client, self.handle_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    async def handle_requested(self, client: Any, env: Any) -> None:
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        import os
        try:
            n_angles = int(os.environ.get(_N_ANGLES_ENV, "3"))
        except ValueError:
            n_angles = 3
        n_angles = max(2, min(n_angles, 3))

        try:
            research_spec = self.spec_store.get(payload.get("research_spec_ref")) or {}
            literature_brief = None
            if payload.get("literature_brief_ref"):
                literature_brief = self.spec_store.get(payload["literature_brief_ref"])
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            result = await asyncio.to_thread(
                core.deliberate,
                research_spec, literature_brief, self.llm_client,
                n_angles=n_angles, prompt_patch=self.config.prompt_patch(),
            )
            report = result.deliberation_report
        except Exception as exc:  # noqa: BLE001 -- FAIL-OPEN: degrade, never stall
            logger.warning("deliberation degraded to minimal report (session %s): %s",
                           session_id, exc)
            report = core.minimal_report(f"deliberation unavailable: {type(exc).__name__}")

        report_ref = self.spec_store.put(
            kind="deliberation_report", session_id=session_id, key=_REPORT_KEY,
            obj=report, emitter=AGENT_NAME,
        )
        await client.publish(
            event_type="deliberation.ready",
            session_id=session_id,
            payload={"deliberation_report_ref": report_ref},
            causation_id=causation_id,
        )


__all__ = ["DeliberationAgentHandler", "SCHEMA_VERSION"]
