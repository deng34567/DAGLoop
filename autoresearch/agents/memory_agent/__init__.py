"""
MemoryAgent: consolidates verified findings, negative results, excluded
hypotheses, and insights (plus divergence seeds and next steps) from the
frozen EvaluationReport + RunRecord (and, optionally, a prior ResearchMemory
and VerificationReport), and picks a routing verdict for the exploration
loop -- ConsolidatingMemory (AutoResearch state machine V1.10).
MemoryAgent：把冻结的 EvaluationReport + RunRecord（以及可选的既往
ResearchMemory 和 VerificationReport）沉淀为已验证发现、负面结果、排除的
假设与洞见（加上发散方向与下一步），并选定探索循环的路由结果 ——
ConsolidatingMemory。

Public surface / 公开接口:
  - ``core.consolidate``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.MemoryAgentHandler``: EventBus wrapper over the Spec Store + LLM.
  - ``service.build_memory_service`` / ``run_memory_agent``: wiring + process
    entry point.
"""

from __future__ import annotations

from autoresearch.agents.memory_agent import core, kernel
from autoresearch.agents.memory_agent.core import consolidate
from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
from autoresearch.agents.memory_agent.service import (
    build_memory_service,
    run_memory_agent,
)

__all__ = [
    "core",
    "kernel",
    "consolidate",
    "MemoryAgentHandler",
    "build_memory_service",
    "run_memory_agent",
]
