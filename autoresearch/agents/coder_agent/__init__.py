"""
CoderAgent: implements a frozen ExperimentPlan (the declarative WHAT/WHY
produced by ExperimentPlannerAgent) as an ExperimentPatch — actual code
(diffs/new file content) that a later milestone (RunnerAgent + the
orchestrator gate) will execute and score — AutoResearch state machine
V1.10 §1.
CoderAgent：把 ExperimentPlannerAgent 产出的冻结 ExperimentPlan（声明式
WHAT/WHY）实现为 ExperimentPatch —— 实际代码（diff/新文件内容），供后续
里程碑（RunnerAgent + 编排器门禁）执行与评分。

Public surface / 公开接口:
  - ``core.implement_patch``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.CoderAgentHandler``: EventBus wrapper over the Spec Store + LLM.
  - ``service.build_coder_service`` / ``run_coder_agent``: wiring + process
    entry point.
"""

from __future__ import annotations

from autoresearch.agents.coder_agent import core, kernel
from autoresearch.agents.coder_agent.core import implement_patch
from autoresearch.agents.coder_agent.handler import CoderAgentHandler
from autoresearch.agents.coder_agent.service import build_coder_service, run_coder_agent

__all__ = [
    "core",
    "kernel",
    "implement_patch",
    "CoderAgentHandler",
    "build_coder_service",
    "run_coder_agent",
]
