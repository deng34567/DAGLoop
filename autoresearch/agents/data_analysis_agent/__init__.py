"""
DataAnalysisAgent: analyzes an externally-uploaded dataset against a frozen
external experiment package via N INDEPENDENT LLM-produced analysis
trajectories, aggregated into a DETERMINISTICALLY-computed consensus verdict
(``consensus_status``: "converged"/"diverged").
DataAnalysisAgent：针对外部上传数据集与冻结外部实验包，通过 LLM 产出的 N 条
独立分析轨迹，聚合为确定性计算的共识判定（``consensus_status``：
"converged"/"diverged"）。

Public surface / 公开接口:
  - ``core.analyze_external_data``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.DataAnalysisAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_data_analysis_service`` / ``run_data_analysis_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.data_analysis_agent import core, kernel
from autoresearch.agents.data_analysis_agent.core import analyze_external_data
from autoresearch.agents.data_analysis_agent.handler import DataAnalysisAgentHandler
from autoresearch.agents.data_analysis_agent.service import (
    build_data_analysis_service,
    run_data_analysis_agent,
)

__all__ = [
    "core",
    "kernel",
    "analyze_external_data",
    "DataAnalysisAgentHandler",
    "build_data_analysis_service",
    "run_data_analysis_agent",
]
