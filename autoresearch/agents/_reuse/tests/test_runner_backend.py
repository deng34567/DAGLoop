"""Unit tests for the RunnerBackend adapter interface (M10 Task 2).

FakeRunnerBackend must return its canned RawRunResult dict from
run_baseline (deterministic, call-counted). The opt-in real backend stubs
(E2BRunnerBackend, LocalSubprocessRunnerBackend) must NEVER actually run a
subprocess or network call -- they raise NotImplementedError.
"""
from __future__ import annotations

import pytest

from autoresearch.agents._reuse.runner_backend import (
    E2BRunnerBackend,
    FakeRunnerBackend,
    LocalSubprocessRunnerBackend,
    RunnerBackend,
)

BENCHMARK_SPEC = {
    "benchmark_id": "bench-1",
    "name": "Some Benchmark",
    "metric": "accuracy",
    "run_command": "python run.py",
    "baseline": {"command": "python baseline.py", "reported_metric": 0.5},
}

VERIFIER_SPEC = {
    "evaluator_hash": "abc123",
    "metric": "accuracy",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "editable_region": ["src/"],
    "forbidden_files": ["harness/"],
}


def test_fake_runner_backend_default_returns_successful_run():
    backend = FakeRunnerBackend()

    result = backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC)

    assert isinstance(result, dict)
    assert result["exit_code"] == 0
    assert result["metric_value"] is not None
    assert isinstance(result["environment"], str)
    assert isinstance(result["command"], str)


def test_fake_runner_backend_records_call_counts():
    backend = FakeRunnerBackend()

    assert backend.calls == 0
    backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC)
    backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC)

    assert backend.calls == 2


def test_fake_runner_backend_returns_canned_raw_result_passed_to_constructor():
    canned = {
        "metric_value": 0.42,
        "environment": "py3.11/cuda12.1",
        "command": "python baseline.py",
        "seed": "1234",
        "runtime_seconds": 12.5,
        "exit_code": 0,
        "log_excerpt": "all good",
    }
    backend = FakeRunnerBackend(raw_result=canned)

    assert backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC) == canned


def test_fake_runner_backend_convenience_kwargs():
    backend = FakeRunnerBackend(
        metric_value=0.99,
        exit_code=1,
        command="python weird.py",
        environment="weird-env",
    )

    result = backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC)

    assert result["metric_value"] == 0.99
    assert result["exit_code"] == 1
    assert result["command"] == "python weird.py"
    assert result["environment"] == "weird-env"


def test_fake_runner_backend_satisfies_protocol():
    backend: RunnerBackend = FakeRunnerBackend()
    assert hasattr(backend, "run_baseline")


def test_e2b_backend_stub_raises_not_implemented():
    backend = E2BRunnerBackend()
    with pytest.raises(NotImplementedError):
        backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC)


def test_local_subprocess_backend_is_now_real_requires_benchmark_dir():
    # LocalSubprocessRunnerBackend is now a REAL opt-in backend (P3); it requires
    # a benchmark_dir. Its real behavior is covered by
    # test_local_subprocess_backend.py. Here we just assert it's no longer a
    # zero-arg raising stub.
    with pytest.raises(TypeError):
        LocalSubprocessRunnerBackend()


EXPERIMENT_PATCH = {
    "hypothesis_id": "hyp-1",
    "diff": "--- a/src/foo.py\n+++ b/src/foo.py\n",
}


def test_fake_runner_backend_run_experiment_default_includes_stdout_and_stderr():
    backend = FakeRunnerBackend()

    result = backend.run_experiment(EXPERIMENT_PATCH, VERIFIER_SPEC)

    assert isinstance(result, dict)
    assert result["exit_code"] == 0
    assert result["metric_value"] is not None
    assert isinstance(result["environment"], str)
    assert isinstance(result["stdout"], str)
    assert "stderr" in result


def test_fake_runner_backend_run_experiment_convenience_kwargs():
    backend = FakeRunnerBackend(exit_code=1, metric_value=0.77, stdout="custom stdout")

    result = backend.run_experiment(EXPERIMENT_PATCH, VERIFIER_SPEC)

    assert result["exit_code"] == 1
    assert result["metric_value"] == 0.77
    assert result["stdout"] == "custom stdout"


def test_fake_runner_backend_run_experiment_does_not_affect_run_baseline():
    backend = FakeRunnerBackend()

    baseline_result = backend.run_baseline(BENCHMARK_SPEC, VERIFIER_SPEC)
    experiment_result = backend.run_experiment(EXPERIMENT_PATCH, VERIFIER_SPEC)

    assert "stdout" not in baseline_result
    assert "stderr" not in baseline_result
    assert "stdout" in experiment_result


def test_fake_runner_backend_run_experiment_records_call_counts():
    backend = FakeRunnerBackend()

    assert backend.calls == 0
    backend.run_experiment(EXPERIMENT_PATCH, VERIFIER_SPEC)
    backend.run_experiment(EXPERIMENT_PATCH, VERIFIER_SPEC)

    assert backend.calls == 2


def test_e2b_backend_run_experiment_stub_raises_not_implemented():
    backend = E2BRunnerBackend()
    with pytest.raises(NotImplementedError):
        backend.run_experiment(EXPERIMENT_PATCH, VERIFIER_SPEC)

