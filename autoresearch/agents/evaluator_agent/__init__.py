"""
EvaluatorAgent: selects a reproducible benchmark against the frozen
ResearchSpec + LiteratureBrief (SelectingBenchmark), then freezes a
verifier-first VerifierSpec against the frozen BenchmarkSpec
(PreparingVerifier) — AutoResearch state machine V1.10 §1.
EvaluatorAgent：针对冻结的 ResearchSpec + LiteratureBrief 选择可复现的基准
（SelectingBenchmark），再针对冻结的 BenchmarkSpec 冻结一份"验证器先行"的
VerifierSpec（PreparingVerifier）。

Public surface / 公开接口:
  - ``core.select_benchmark`` / ``core.prepare_verifier``: pure core logic
    (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.EvaluatorAgentHandler``: EventBus wrapper over the Spec Store +
    LLM.
  - ``service.build_evaluator_service`` / ``run_evaluator_agent``: wiring +
    process entry point.
"""

from __future__ import annotations

from autoresearch.agents.evaluator_agent import core, kernel
from autoresearch.agents.evaluator_agent.core import prepare_verifier, select_benchmark
from autoresearch.agents.evaluator_agent.handler import EvaluatorAgentHandler
from autoresearch.agents.evaluator_agent.service import (
    build_evaluator_service,
    run_evaluator_agent,
)

__all__ = [
    "core",
    "kernel",
    "select_benchmark",
    "prepare_verifier",
    "EvaluatorAgentHandler",
    "build_evaluator_service",
    "run_evaluator_agent",
]
