"""
Tests for RunnerAgent's pure core. / RunnerAgent 纯核心的测试。

RunnerAgent has NO LLM and NO retry loop: ``reproduce_baseline`` is a single
deterministic call to an injected RunnerBackend followed by a comparison and
schema validation. Only FakeRunnerBackend is used here -- NO real execution,
NO subprocess, NO LLM, NO network.
RunnerAgent 无 LLM、无重试循环：``reproduce_baseline`` 是对注入的
RunnerBackend 的一次确定性调用，随后做比较和 schema 校验。这里只用
FakeRunnerBackend —— 无真实执行、无子进程、无 LLM、无网络。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.core import (
    ComponentPlanExhaustedError,
    ComponentPlanInvalidError,
    RunnerError,
    RunnerExecutionError,
    RunnerParseError,
    reproduce_baseline,
    run_experiment,
)

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _baseline_result_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "baseline_result.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


BASELINE_RESULT_VALIDATOR = _baseline_result_validator()


def test_component_plan_exhaustion_has_a_distinct_failure_code():
    assert ComponentPlanExhaustedError.code == "component_plan_exhausted"
    assert ComponentPlanExhaustedError.classification == "permanent"
    assert ComponentPlanInvalidError.code == "component_plan_invalid"
    assert ComponentPlanInvalidError.classification == "permanent"


def _assert_schema_valid(result: dict) -> None:
    errors = list(BASELINE_RESULT_VALIDATOR.iter_errors(result))
    assert not errors, f"BaselineResult schema errors: {[e.message for e in errors]}"


def test_reproduced_true_when_exit_zero_and_metric_within_tolerance(
    benchmark_spec, verifier_spec
):
    backend = FakeRunnerBackend(metric_value=0.505, exit_code=0)

    result = reproduce_baseline(benchmark_spec, verifier_spec, backend, tolerance=0.02)

    assert result["reproduced"] is True
    assert result["benchmark_id"] == benchmark_spec["benchmark_id"]
    assert result["metric_value"] == 0.505
    assert result["reported_metric"] == 0.5
    assert result["tolerance"] == 0.02
    _assert_schema_valid(result)


def test_reproduced_false_when_metric_outside_tolerance(benchmark_spec, verifier_spec):
    backend = FakeRunnerBackend(metric_value=0.9, exit_code=0)

    result = reproduce_baseline(benchmark_spec, verifier_spec, backend, tolerance=0.02)

    assert result["reproduced"] is False
    _assert_schema_valid(result)


def test_reproduced_false_when_exit_code_nonzero_even_if_metric_matches(
    benchmark_spec, verifier_spec
):
    backend = FakeRunnerBackend(metric_value=0.5, exit_code=1)

    result = reproduce_baseline(benchmark_spec, verifier_spec, backend, tolerance=0.02)

    assert result["reproduced"] is False
    assert result["exit_code"] == 1
    _assert_schema_valid(result)


def test_reproduced_matches_exit_code_when_no_reported_metric(verifier_spec):
    benchmark_spec_no_reported = {
        "benchmark_id": "bench-2",
        "name": "No Reported Metric Benchmark",
        "metric": "accuracy",
        "run_command": "python run.py",
        "baseline": {"command": "python baseline.py"},
    }
    backend = FakeRunnerBackend(metric_value=None, exit_code=0)

    result = reproduce_baseline(
        benchmark_spec_no_reported, verifier_spec, backend, tolerance=0.02
    )

    assert result["reported_metric"] is None
    assert result["reproduced"] is True
    _assert_schema_valid(result)

    backend_fail = FakeRunnerBackend(metric_value=None, exit_code=1)
    result_fail = reproduce_baseline(
        benchmark_spec_no_reported, verifier_spec, backend_fail, tolerance=0.02
    )
    assert result_fail["reproduced"] is False
    _assert_schema_valid(result_fail)


class _RaisingBackend:
    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        raise RuntimeError("backend blew up")


def test_backend_exception_raises_runner_execution_error(benchmark_spec, verifier_spec):
    backend = _RaisingBackend()

    with pytest.raises(RunnerExecutionError) as excinfo:
        reproduce_baseline(benchmark_spec, verifier_spec, backend)

    assert excinfo.value.classification == "transient"
    assert isinstance(excinfo.value, RunnerError)


def test_returned_baseline_result_validates_against_schema(benchmark_spec, verifier_spec):
    backend = FakeRunnerBackend(metric_value=0.5, exit_code=0)

    result = reproduce_baseline(benchmark_spec, verifier_spec, backend)

    _assert_schema_valid(result)


def test_live_repo_baseline_provenance_validates_against_schema():
    """Catch removal/rejection of the provenance needed to freeze a repo baseline."""
    result = {
        "benchmark_id": "repo:nanochat",
        "reproduced": True,
        "command": "uv run train.py",
        "exit_code": 0,
        "environment": "DGX-120 GPU 0; CUDA 13.0",
        "metric_value": 1.818624,
        "reported_metric": None,
        "tolerance": None,
        "seed": "0",
        "runtime_seconds": 300.0,
        "log_excerpt": "val_bpb: 1.818624",
        "source_kind": "live_repo",
        "repo_url": "https://github.com/example/nanochat",
        "repo_commit": "0123456789abcdef",
        "metric_name": "val_bpb",
        "lower_is_better": True,
        "fixed_config": "depth=12, device_batch_size=16, max_steps=1000",
        "environment_fingerprint": "sha256:environment",
        "baseline_runs": [1.818624],
        "artifacts": ["repo_setup://session/baseline"],
    }

    _assert_schema_valid(result)


def test_parse_error_is_permanent_classification():
    assert RunnerParseError.classification == "permanent"


# ---------------------------------------------------------------------------
# run_experiment (M11 Task 2) / 运行实验
# ---------------------------------------------------------------------------

_RUN_RECORD_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _run_record_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_RUN_RECORD_CONTRACTS_DIR / "run_record.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


RUN_RECORD_VALIDATOR = _run_record_validator()


def _assert_run_record_schema_valid(result: dict) -> None:
    errors = list(RUN_RECORD_VALIDATOR.iter_errors(result))
    assert not errors, f"RunRecord schema errors: {[e.message for e in errors]}"


EXPERIMENT_PATCH = {
    "hypothesis_id": "hyp-1",
    "diff": "--- a/src/foo.py\n+++ b/src/foo.py\n",
}


def test_run_experiment_succeeded_true_when_exit_code_zero(verifier_spec):
    backend = FakeRunnerBackend(exit_code=0, metric_value=0.5)

    result = run_experiment(EXPERIMENT_PATCH, verifier_spec, backend)

    assert result["succeeded"] is True
    assert result["exit_code"] == 0
    assert result["experiment_id"] == "hyp-1"
    _assert_run_record_schema_valid(result)


def test_run_experiment_succeeded_false_when_exit_code_nonzero(verifier_spec):
    backend = FakeRunnerBackend(exit_code=1, metric_value=0.5)

    result = run_experiment(EXPERIMENT_PATCH, verifier_spec, backend)

    assert result["succeeded"] is False
    assert result["exit_code"] == 1
    _assert_run_record_schema_valid(result)


class _RaisingExperimentBackend:
    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        raise RuntimeError("backend blew up")


def test_run_experiment_backend_exception_raises_runner_execution_error(verifier_spec):
    backend = _RaisingExperimentBackend()

    with pytest.raises(RunnerExecutionError) as excinfo:
        run_experiment(EXPERIMENT_PATCH, verifier_spec, backend)

    assert excinfo.value.classification == "transient"
    assert isinstance(excinfo.value, RunnerError)


class _NoMetricBackend:
    """Returns a raw result with metric_value explicitly None (e.g. crashed
    before producing a metric), to exercise the schema's nullable field."""

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        return {
            "metric_value": None,
            "exit_code": 0,
            "stdout": "ran, but no metric produced",
            "stderr": None,
            "environment": "py3.11/cpu",
            "seed": None,
            "runtime_seconds": None,
        }


def test_run_experiment_metric_value_none_validates(verifier_spec):
    backend = _NoMetricBackend()

    result = run_experiment(EXPERIMENT_PATCH, verifier_spec, backend)

    assert result["metric_value"] is None
    _assert_run_record_schema_valid(result)


def test_run_experiment_without_hypothesis_id_omits_experiment_id(verifier_spec):
    patch_without_hypothesis_id = {"diff": "--- a/src/foo.py\n+++ b/src/foo.py\n"}
    backend = FakeRunnerBackend(exit_code=0, metric_value=0.5)

    result = run_experiment(patch_without_hypothesis_id, verifier_spec, backend)

    assert "experiment_id" not in result
    _assert_run_record_schema_valid(result)
