"""
RunnerAgent: reproduces a benchmark's previously-reported baseline against
the frozen BenchmarkSpec + VerifierSpec (ReproducingBaseline) -- AutoResearch
state machine V1.10 §1.
RunnerAgent：针对冻结的 BenchmarkSpec + VerifierSpec 复现基准先前报告的基线
（ReproducingBaseline）。

Public surface / 公开接口:
  - ``core.reproduce_baseline``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.RunnerAgentHandler``: EventBus wrapper over the Spec Store + a
    RunnerBackend.
  - ``service.build_runner_service`` / ``run_runner_agent``: wiring + process
    entry point.
"""

from __future__ import annotations

from autoresearch.agents.runner_agent import core, kernel
from autoresearch.agents.runner_agent.core import reproduce_baseline
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.agents.runner_agent.service import (
    build_runner_service,
    run_runner_agent,
)

__all__ = [
    "core",
    "kernel",
    "reproduce_baseline",
    "RunnerAgentHandler",
    "build_runner_service",
    "run_runner_agent",
]
