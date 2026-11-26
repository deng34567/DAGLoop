"""c-layer wiring: the env+task_class gate that routes RunningExperiment through
``MonteCarloExperimentRunner`` instead of the default RunnerAgent/CoderAgent
patch path.

``MonteCarloExperimentRunner`` does NOT implement the ``RunnerBackend`` Protocol
(it runs a container agent that self-writes ``method.py``, not a CoderAgent
patch), so this is a DEDICATED branch, not a backend swap. It is DEFAULT-OFF:
the branch fires only when BOTH ``AUTORESEARCH_MC_EXPERIMENT=1`` is set AND the
plan/patch carries ``task_class == "mc_variance_reduction"``. When either is
absent the existing RunnerAgent path runs, byte-identical.
"""
from __future__ import annotations

import os

MC_TASK_CLASS = "mc_variance_reduction"
MC_ENV_FLAG = "AUTORESEARCH_MC_EXPERIMENT"


def mc_branch_selected(task_class, *, env=None) -> bool:
    """True iff the MC branch should run: env flag == "1" AND task_class matches."""
    env = os.environ if env is None else env
    return env.get(MC_ENV_FLAG) == "1" and task_class == MC_TASK_CLASS


def build_run_record(baseline_result: dict, experiment_result: dict) -> dict:
    """Map MC baseline+experiment results into a RunRecord-equivalent dict
    (RunnerAgent's ``run_record`` schema shape)."""
    ratio = experiment_result.get("metric_value")
    over_budget = bool(experiment_result.get("over_budget"))
    succeeded = ratio is not None and ratio > 0 and not over_budget
    summary = (
        "MC variance reduction: "
        f"baseline_rmse={baseline_result.get('rmse_baseline')} "
        f"proposed_rmse={experiment_result.get('rmse_proposed')} "
        f"ratio={ratio} over_budget={over_budget} "
        f"hardcoded_suspected={experiment_result.get('hardcoded_suspected')} "
        f"agent_status={experiment_result.get('agent_status')}"
    )
    return {
        "succeeded": bool(succeeded),
        "exit_code": 0 if succeeded else 1,
        "metric_value": ratio,
        "stdout_excerpt": summary,
        "stderr_excerpt": None,
        "environment": "openhands-docker/mc_variance_reduction",
        "seed": None,
        "runtime_seconds": None,
        "artifacts": [],
    }


__all__ = ["MC_TASK_CLASS", "MC_ENV_FLAG", "mc_branch_selected", "build_run_record"]
