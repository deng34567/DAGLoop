"""
CriticAgent: critiques the frozen ProblemModel (CritiquingDirection), the
frozen HypothesisBank (CritiquingHypotheses), and the frozen ExperimentPlan
against the frozen VerifierSpec (ReviewingExperimentPlan), producing
CritiqueReports (AutoResearch state machine V1.10 §1).
CriticAgent：分别针对冻结的 ProblemModel（CritiquingDirection）、冻结的
HypothesisBank（CritiquingHypotheses），以及针对冻结的 VerifierSpec 评审冻结
的 ExperimentPlan（ReviewingExperimentPlan），做批判，产出 CritiqueReport。

Public surface / 公开接口:
  - ``core.critique_direction`` / ``core.critique_hypotheses`` /
    ``core.critique_experiment_plan``: pure critique logic (no transport).
  - ``handler.CriticAgentHandler``: EventBus wrapper over the Spec Store +
    LLM.
  - ``service.build_critic_service`` / ``run_critic_agent``: wiring +
    process entry point.
"""

from __future__ import annotations

from autoresearch.agents.critic_agent import core, kernel
from autoresearch.agents.critic_agent.core import (
    critique_direction,
    critique_experiment_plan,
    critique_hypotheses,
)
from autoresearch.agents.critic_agent.handler import CriticAgentHandler
from autoresearch.agents.critic_agent.service import build_critic_service, run_critic_agent

__all__ = [
    "core",
    "kernel",
    "critique_direction",
    "critique_hypotheses",
    "critique_experiment_plan",
    "CriticAgentHandler",
    "build_critic_service",
    "run_critic_agent",
]
