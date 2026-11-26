"""
DeepResearchAgent: runs the two-pass literature search (seminal + frontier)
against a frozen ResearchSpec and synthesizes a LiteratureBrief +
FieldTrajectoryBrief (AutoResearch state machine V1.10 §1, DeepResearching).
DeepResearchAgent：针对冻结的 ResearchSpec 跑两轮文献搜索（seminal +
frontier），合成 LiteratureBrief + FieldTrajectoryBrief。

Public surface / 公开接口:
  - ``core.deep_research_step``: pure two-pass-search + synthesis logic (no
    transport).
  - ``handler.DeepResearchAgentHandler``: EventBus wrapper over the Spec
    Store + search backend + LLM.
  - ``service.build_deep_research_service`` / ``run_deep_research_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.deep_research_agent import core, kernel
from autoresearch.agents.deep_research_agent.core import DeepResearchResult, deep_research_step
from autoresearch.agents.deep_research_agent.handler import DeepResearchAgentHandler
from autoresearch.agents.deep_research_agent.service import (
    build_deep_research_service,
    run_deep_research_agent,
)

__all__ = [
    "core",
    "kernel",
    "deep_research_step",
    "DeepResearchResult",
    "DeepResearchAgentHandler",
    "build_deep_research_service",
    "run_deep_research_agent",
]
