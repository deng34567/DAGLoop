"""
ExperimentPlannerAgent: plans a declarative, no-code experiment to test a
selected hypothesis, grounded in the frozen HypothesisBank, VerifierSpec,
BenchmarkSpec, and ResearchSpec (PlanningExperiment) — AutoResearch state
machine V1.10 §1.
ExperimentPlannerAgent：针对所选 hypothesis、冻结的 HypothesisBank、
VerifierSpec、BenchmarkSpec、ResearchSpec，规划一份声明式、无代码的实验
（PlanningExperiment）。

Public surface / 公开接口:
  - ``core.plan_experiment``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.ExperimentPlannerAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_experiment_planner_service`` / ``run_experiment_planner_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.experiment_planner_agent import core, kernel
from autoresearch.agents.experiment_planner_agent.core import plan_experiment
from autoresearch.agents.experiment_planner_agent.handler import ExperimentPlannerAgentHandler
from autoresearch.agents.experiment_planner_agent.service import (
    build_experiment_planner_service,
    run_experiment_planner_agent,
)

__all__ = [
    "core",
    "kernel",
    "plan_experiment",
    "ExperimentPlannerAgentHandler",
    "build_experiment_planner_service",
    "run_experiment_planner_agent",
]
