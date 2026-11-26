"""
RunnerAgent core: the pure baseline-reproduction logic.
RunnerAgent 核心：与传输层无关的纯"基线复现"逻辑。

Unlike every other agent in this codebase, RunnerAgent has NO LLM: given a
frozen BenchmarkSpec + VerifierSpec, ``reproduce_baseline`` calls an injected
RunnerBackend to run the baseline command, compares the observed metric
against the benchmark's previously-reported metric (within a tolerance), and
returns a schema-valid ``BaselineResult`` -- with NO EventBus / Redis / Store
/ network access. The handler (a later milestone) is responsible for
resolving refs, persisting the artifact, and publishing events.

给定冻结的 BenchmarkSpec + VerifierSpec，``reproduce_baseline`` 调用注入的
RunnerBackend 运行基线命令，将观测到的指标与基准先前报告的指标做比较（容差
内），并返回 schema 合法的 ``BaselineResult`` —— 不含任何 EventBus/Redis/
Store/网络访问。handler（后续里程碑）负责解析 ref、持久化产物、发布事件。

This is a single deterministic call, not a retry loop: unlike the LLM-backed
agents, there is nothing to retry against (the backend either ran or it
didn't), so there is no ``_run_llm_loop`` analogue here.
这是一次确定性调用，而非重试循环：与 LLM 驱动的 agent 不同，这里没有可重试
的对象（backend 要么跑了要么没跑），因此没有类似 ``_run_llm_loop`` 的机制。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml
from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class RunnerError(Exception):
    """Base for RunnerAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "runner_failed"
    #: transient | permanent
    classification: str = "transient"


class RunnerExecutionError(RunnerError):
    """The injected RunnerBackend raised while running the baseline.
    注入的 RunnerBackend 在运行基线时抛出异常。"""

    code = "runner_execution_error"
    classification = "transient"


class RunnerParseError(RunnerError):
    """The assembled BaselineResult failed schema validation.
    组装出的 BaselineResult 未通过 schema 校验。"""

    code = "runner_parse_error"
    classification = "permanent"


class ComponentPlanExhaustedError(RunnerError):
    """The component DAG has unfinished nodes but none can run again."""

    code = "component_plan_exhausted"
    classification = "permanent"


class ComponentPlanInvalidError(RunnerError):
    """A structured confirmed ComponentPlan is not executable as declared."""

    code = "component_plan_invalid"
    classification = "permanent"


# ---------------------------------------------------------------------------
# Backend protocol / 后端协议
# ---------------------------------------------------------------------------


class RunnerBackendLike(Protocol):
    """Subset of autoresearch.agents._reuse.runner_backend.RunnerBackend
    used here. / 这里用到的后端子集。"""

    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict: ...

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict: ...


# ---------------------------------------------------------------------------
# Static resources: schema validator / 静态资源：schema 校验器
# ---------------------------------------------------------------------------

_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _baseline_result_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "baseline_result.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_baseline_result(result: dict) -> list[str]:
    return [e.message for e in _baseline_result_validator().iter_errors(result)]


@lru_cache(maxsize=1)
def _run_record_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "run_record.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_run_record(result: dict) -> list[str]:
    return [e.message for e in _run_record_validator().iter_errors(result)]


# ---------------------------------------------------------------------------
# reproduce_baseline / 复现基线
# ---------------------------------------------------------------------------


def reproduce_baseline(
    benchmark_spec: dict,
    verifier_spec: dict,
    backend: RunnerBackendLike,
    tolerance: float = 0.02,
) -> dict:
    """Reproduce a benchmark's reported baseline (pure, no transport).
    复现基准报告的基线（纯，无传输层）。

    Args:
        benchmark_spec: the frozen BenchmarkSpec.
        verifier_spec: the frozen VerifierSpec.
        backend: object exposing ``.run_baseline(benchmark_spec, verifier_spec)``.
        tolerance: allowed absolute deviation between the observed metric and
            the benchmark's reported metric for ``reproduced`` to be true.

    Returns:
        A schema-valid ``BaselineResult`` dict.

    Raises:
        RunnerExecutionError (transient): the backend itself raised.
        RunnerParseError (permanent): the assembled BaselineResult failed
            schema validation.
    """
    try:
        raw = backend.run_baseline(benchmark_spec, verifier_spec)
    except Exception as exc:  # any backend exception is infra/transient
        raise RunnerExecutionError(
            f"backend.run_baseline failed: {type(exc).__name__}: {exc}"
        ) from exc

    reported = (benchmark_spec.get("baseline") or {}).get("reported_metric")
    metric = raw.get("metric_value")

    reproduced = bool(
        raw["exit_code"] == 0
        and (reported is None or (metric is not None and abs(metric - reported) <= tolerance))
    )

    result = {
        "benchmark_id": benchmark_spec["benchmark_id"],
        "reproduced": reproduced,
        "metric_value": metric,
        "reported_metric": reported,
        "tolerance": tolerance,
        "environment": raw["environment"],
        "command": raw["command"],
        "seed": raw.get("seed"),
        "runtime_seconds": raw.get("runtime_seconds"),
        "exit_code": raw["exit_code"],
        "log_excerpt": raw.get("log_excerpt"),
    }

    errors = _validate_baseline_result(result)
    if errors:
        raise RunnerParseError(f"BaselineResult failed schema validation: {errors}")

    return result


# ---------------------------------------------------------------------------
# run_experiment / 运行实验
# ---------------------------------------------------------------------------


def run_experiment(
    experiment_patch: dict,
    verifier_spec: dict,
    backend: RunnerBackendLike,
) -> dict:
    """Run an experiment patch (pure, no transport).
    运行一个实验补丁（纯，无传输层）。

    Args:
        experiment_patch: the frozen ExperimentPatch.
        verifier_spec: the frozen VerifierSpec.
        backend: object exposing
            ``.run_experiment(experiment_patch, verifier_spec)``.

    Returns:
        A schema-valid ``RunRecord`` dict.

    Raises:
        RunnerExecutionError (transient): the backend itself raised.
        RunnerParseError (permanent): the assembled RunRecord failed schema
            validation.
    """
    try:
        raw = backend.run_experiment(experiment_patch, verifier_spec)
    except Exception as exc:  # any backend exception is infra/transient
        raise RunnerExecutionError(
            f"backend.run_experiment failed: {type(exc).__name__}: {exc}"
        ) from exc

    succeeded = raw["exit_code"] == 0

    result = {
        "succeeded": succeeded,
        "exit_code": raw["exit_code"],
        "metric_value": raw.get("metric_value"),
        "stdout_excerpt": raw["stdout"],
        "stderr_excerpt": raw.get("stderr"),
        "environment": raw["environment"],
        "seed": raw.get("seed"),
        "runtime_seconds": raw.get("runtime_seconds"),
        "artifacts": raw.get("artifacts", []),
    }
    # experiment_id is optional but NOT nullable in the schema (unlike
    # metric_value/stderr_excerpt/seed/runtime_seconds) -- omit rather than
    # emit null when there's no hypothesis_id to attach.
    experiment_id = experiment_patch.get("hypothesis_id") or None
    if experiment_id is not None:
        result["experiment_id"] = experiment_id

    errors = _validate_run_record(result)
    if errors:
        raise RunnerParseError(f"RunRecord failed schema validation: {errors}")

    return result


__all__ = [
    "RunnerError",
    "RunnerExecutionError",
    "RunnerParseError",
    "ComponentPlanExhaustedError", "ComponentPlanInvalidError",
    "reproduce_baseline",
    "run_experiment",
]
