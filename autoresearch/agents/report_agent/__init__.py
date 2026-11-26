"""
ReportAgent: composes the final honest Report -- a summary plus 8 required
classification arrays (verified findings, negative results, unverified ideas,
experiments not run, failed replications, incomplete external data, analysis
caveats, known risks) and a per-experiment ledger -- from the frozen
ResearchMemory (and, optionally, the original ResearchSpec) -- Reporting
(AutoResearch state machine V1.10).
ReportAgent：从冻结的 ResearchMemory（以及可选的原始 ResearchSpec）出发，
撰写最终诚实 Report —— 一段摘要，加上 8 个必需的分类数组（已验证发现、负面
结果、未验证想法、未执行实验、复现失败、外部数据不完整、分析局限、已知
风险）与按实验的台账 —— Reporting。

Public surface / 公开接口:
  - ``core.generate_report``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.ReportAgentHandler``: EventBus wrapper over the Spec Store + LLM.
  - ``service.build_report_service`` / ``run_report_agent``: wiring + process
    entry point.
"""

from __future__ import annotations

from autoresearch.agents.report_agent import core, kernel
from autoresearch.agents.report_agent.core import generate_report
from autoresearch.agents.report_agent.handler import ReportAgentHandler
from autoresearch.agents.report_agent.service import (
    build_report_service,
    run_report_agent,
)

__all__ = [
    "core",
    "kernel",
    "generate_report",
    "ReportAgentHandler",
    "build_report_service",
    "run_report_agent",
]
