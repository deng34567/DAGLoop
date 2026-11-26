"""
ProblemModelingAgent: drives the ClarifyingResearchGoal dialogue (AutoResearch
state machine V1.10 §1), confirming topic/goal/task-mode/budget/deliverable-
form/scope with the user and freezing the result as a ResearchSpec in the
Spec Store.
ProblemModelingAgent：驱动 ClarifyingResearchGoal 对话，与用户确认
主题/目标/任务模式/预算/交付形式/范围，并把结果冻结为 ResearchSpec 写入
Spec Store。

Public surface / 公开接口:
  - ``core.clarify_step``: pure clarification-turn logic (no transport).
  - ``reinterpret_core.reinterpret_step``: pure ReinterpretingSpec / scope-gate
    logic (no transport).
  - ``model_core.model_step``: pure ModelingProblem reasoning (no transport).
  - ``handler.ProblemModelingAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_problem_modeling_service`` / ``run_problem_modeling_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.problem_modeling_agent import core, kernel, model_core, reinterpret_core
from autoresearch.agents.problem_modeling_agent.core import ClarifyDecision, clarify_step
from autoresearch.agents.problem_modeling_agent.handler import ProblemModelingAgentHandler
from autoresearch.agents.problem_modeling_agent.model_core import model_step
from autoresearch.agents.problem_modeling_agent.reinterpret_core import (
    ReinterpretResult,
    reinterpret_step,
)
from autoresearch.agents.problem_modeling_agent.service import (
    build_problem_modeling_service,
    run_problem_modeling_agent,
)

__all__ = [
    "core",
    "kernel",
    "model_core",
    "reinterpret_core",
    "clarify_step",
    "ClarifyDecision",
    "reinterpret_step",
    "ReinterpretResult",
    "model_step",
    "ProblemModelingAgentHandler",
    "build_problem_modeling_service",
    "run_problem_modeling_agent",
]
