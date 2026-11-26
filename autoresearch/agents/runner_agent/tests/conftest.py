"""
Shared fixtures for RunnerAgent tests. / RunnerAgent 测试共享 fixture。

No Redis / network / EventBus / SpecStore / LLM. RunnerAgent's core is
backend-driven only -- FakeRunnerBackend (from
autoresearch.agents._reuse.runner_backend) is imported directly by the
tests that need it; this conftest holds only shared spec fixtures.
不涉及 Redis / 网络 / EventBus / SpecStore / LLM。RunnerAgent 核心仅由
backend 驱动 —— 需要 FakeRunnerBackend 的测试直接从
autoresearch.agents._reuse.runner_backend 导入；本 conftest 只放共享的
spec fixture。
"""

from __future__ import annotations

import pytest


@pytest.fixture
def benchmark_spec() -> dict:
    return {
        "benchmark_id": "bench-1",
        "name": "Some Benchmark",
        "metric": "accuracy",
        "run_command": "python run.py",
        "baseline": {"command": "python baseline.py", "reported_metric": 0.5},
    }


@pytest.fixture
def verifier_spec() -> dict:
    return {
        "evaluator_hash": "abc123",
        "metric": "accuracy",
        "data_split": {"in_loop": "dev", "held_out": "test"},
        "editable_region": ["src/"],
        "forbidden_files": ["harness/"],
    }
