"""Task 8: ExperimentPlan may OPTIONALLY carry ``task_class`` + ``problem_ref``.

These are backward-safe: absent by default (old plans validate byte-identically),
present only when a plan opts into a dedicated task class (e.g. the MC branch).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
PLAN_VALIDATOR = Draft202012Validator(
    yaml.safe_load(
        (_CONTRACTS_DIR / "experiment_plan.schema.yaml").read_text(encoding="utf-8")
    )
)

_BASE_PLAN = {
    "hypothesis_id": "hyp-1",
    "change_summary": "Reduce Monte-Carlo estimator variance.",
    "changes": [{"file": "method.py", "description": "antithetic variates"}],
    "expected_outcome": "Lower RMSE at equal sample budget.",
}


def test_plan_without_task_class_is_valid_backward_identical():
    # old behavior: a plan lacking task_class/problem_ref still validates
    assert list(PLAN_VALIDATOR.iter_errors(_BASE_PLAN)) == []
    assert "task_class" not in _BASE_PLAN


def test_plan_with_task_class_and_problem_ref_is_valid_and_exposed():
    plan = {**_BASE_PLAN,
            "task_class": "mc_variance_reduction",
            "problem_ref": "problem/oscillatory_ratio_4d"}
    errors = list(PLAN_VALIDATOR.iter_errors(plan))
    assert errors == [], [e.message for e in errors]
    assert plan["task_class"] == "mc_variance_reduction"
    assert plan["problem_ref"] == "problem/oscillatory_ratio_4d"
