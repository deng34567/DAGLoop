"""
Cross-cutting concerns for the AutoResearch orchestrator.

M0 PLACEHOLDER. The minimal (M0) OrchestratorEngine does NOT route
``agent.failed``, so nothing here is wired yet. This module reserves the seam:
vibe's ``cross_cutting.handle_agent_failed`` implements the 3-branch agent-failure
classification (transient -> G[state]++/retry, permanent -> AwaitingUserDecision,
policy_violation -> MOSS then AwaitingUserDecision), which for AutoResearch depends
on the A–M counters, MOSS routing, and the agent-failure-routing skill file — all
DEFERRED to the later integration milestone.

Port vibe's full routing here when those pieces land; ``tick`` will then dispatch
``agent.failed`` to ``handle_agent_failed(engine, state, event)``.
"""

from __future__ import annotations

# Reason classifications carried on the agent.failed payload (V1.10). Kept as a
# reference constant for the future port; not consumed at M0.
AGENT_FAILURE_CLASSIFICATIONS = ("transient", "permanent", "policy_violation")


async def handle_agent_failed(engine, state, event) -> None:  # pragma: no cover - M0 stub
    """DEFERRED: route an ``agent.failed`` event per its reason_classification.

    Not implemented at M0 (needs counters + MOSS). See module docstring.
    """
    raise NotImplementedError(
        "agent.failed routing is deferred to the integration milestone "
        "(needs A-M counters + MOSS); port from vibe/orchestrator/cross_cutting.py"
    )
