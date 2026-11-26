"""
MOSS: the Sync-Intervention supervisor that diagnoses stuck/looping patterns
in the ResearchOrchestrator run and decides a legal corrective action --
Sync-Intervention (AutoResearch state machine V1.10).
MOSS：诊断 ResearchOrchestrator 运行中卡顿/循环模式并决定一个合法纠正 action
的同步介入监督者 —— Sync-Intervention。

Public surface / 公开接口:
  - ``core.decide``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.MossAgentHandler``: EventBus wrapper over the Spec Store + LLM.
  - ``service.build_moss_service`` / ``run_moss_agent``: wiring + process
    entry point.
"""

from __future__ import annotations

from autoresearch.agents.moss_agent import core, kernel
from autoresearch.agents.moss_agent.core import decide
from autoresearch.agents.moss_agent.handler import MossAgentHandler
from autoresearch.agents.moss_agent.service import build_moss_service, run_moss_agent

__all__ = [
    "core",
    "kernel",
    "decide",
    "MossAgentHandler",
    "build_moss_service",
    "run_moss_agent",
]
