"""
HypothesisAgent: generates candidate hypotheses grounded in a frozen
ResearchSpec and LiteratureBrief, producing a HypothesisBank (AutoResearch
state machine V1.10 §1, GeneratingHypotheses).
HypothesisAgent：基于冻结的 ResearchSpec 和 LiteratureBrief 生成候选假设，
产出 HypothesisBank。

Public surface / 公开接口:
  - ``core.generate_step``: pure hypothesis-generation logic (no transport).
  - ``handler.HypothesisAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_hypothesis_service`` / ``run_hypothesis_agent``: wiring +
    process entry point.
"""

from __future__ import annotations

from autoresearch.agents.hypothesis_agent import core, kernel
from autoresearch.agents.hypothesis_agent.core import generate_step
from autoresearch.agents.hypothesis_agent.handler import HypothesisAgentHandler
from autoresearch.agents.hypothesis_agent.service import (
    build_hypothesis_service,
    run_hypothesis_agent,
)

__all__ = [
    "core",
    "kernel",
    "generate_step",
    "HypothesisAgentHandler",
    "build_hypothesis_service",
    "run_hypothesis_agent",
]
