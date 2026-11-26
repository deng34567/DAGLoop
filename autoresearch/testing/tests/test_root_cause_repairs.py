"""Regression cases derived from the cross-task workflow audit (no benchmark data)."""
import pytest

from autoresearch.agents.memory_agent.handler import _classify_component_evidence
from autoresearch.agents.report_agent.handler import _apply_latest_verified_structured_result
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.experiments.repo_improve.component_plan import (
    ComponentLedger, ComponentNode, ComponentPlan, ComponentPlanFormatError,
    evaluate_component_contract, validate_component_acceptance,
)


def node(check="signed_metric_improvement > 0"):
    return ComponentNode("gain", "gain", "direct_gain", (), "improve", ("parent",),
                         ("candidate",), (), "change implementation", "lower loss",
                         check, "inspect implementation", "same mechanism", "final result")


@pytest.mark.parametrize("check,parent,candidate,lower,passed,met_expectation", [
    ("signed_metric_improvement > 3", 14.5, 15.1, False, True, False),
    ("signed_metric_improvement >= 3", 14.5, 17.5, False, True, True),
    ("candidate_metric <= 4620", 5132, 5000, True, True, False),
    ("candidate_metric <= 4620", 5132, 4618, True, True, True),
    ("candidate_metric >= 0.95", 0.8, 0.96, False, True, True),
    ("candidate_metric <= 4620", 4000, 4500, True, False, True),
])
def test_declared_direct_threshold_is_recorded_separately_from_real_gain(
        check, parent, candidate, lower, passed, met_expectation):
    result = evaluate_component_contract(node(check), parent_metric=parent,
                                        candidate_metric=candidate, lower_is_better=lower)
    assert result["passed"] is passed
    assert result["acceptance_check_result"]["measurement"] == check.split()[0]
    assert result["acceptance_check_result"]["passed"] is met_expectation
    if passed and not met_expectation:
        assert "expected gain was not reached" in result["reason"]


@pytest.mark.parametrize("check", [
    "matched ablation > 3 and cache-off control > 1",
    "signed_metric_improvement > 0; matched ablation required",
    "signed_metric_improvement > 1e999",
])
def test_unsupported_direct_claim_is_not_silently_replaced_by_positive_delta(check):
    result = evaluate_component_contract(node(check), parent_metric=5,
                                        candidate_metric=1, lower_is_better=True)
    assert result["passed"] is False
    assert result["plan_invalid"] is True
    with pytest.raises(ComponentPlanFormatError):
        validate_component_acceptance(ComponentPlan("method", (node(check),), ("gain",)))


@pytest.mark.parametrize("force,second_status", [(False, "accepted"), (True, "active")])
def test_existing_force_search_flag_applies_to_component_budget(tmp_path, monkeypatch, force, second_status):
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "1" if force else "0")
    plan = ComponentPlan("method", (node(),), ("gain",))
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("gain")
    ledger.record_attempt("gain", execution_succeeded=True, contract_passed=True,
                          checkpoint="A", metric_value=1.0, quality_value=1.0, max_attempts=3)
    second = ledger.record_attempt("gain", execution_succeeded=True, contract_passed=True,
                                   checkpoint="B", metric_value=1.01, quality_value=1.01, max_attempts=3)
    assert second.status == second_status
    assert second.checkpoint == "A"
    if force:
        third = ledger.record_attempt("gain", execution_succeeded=True, contract_passed=True,
                                      checkpoint="C", metric_value=0.9, quality_value=0.9, max_attempts=3)
        assert third.status == "accepted"
        assert third.checkpoint == "C"


@pytest.mark.parametrize("legal,metric,task_verdict,expected", [
    (True, 0.9, "compliant", "method_description_reconciliation"),
    (False, 0.9, "compliant", "component_combination_retry"),
    (True, None, "compliant", "component_combination_retry"),
    (True, float("nan"), "compliant", "component_combination_retry"),
    (True, 0.9, "violation", "component_combination_retry"),
])
def test_integration_alternative_routes_to_revision_only_with_legal_measurement(
        legal, metric, task_verdict, expected):
    gen = {
        "verdict": "valid_not_best", "accepted": False, "method_metric": metric,
        "method_implementation_review": {"verdict": "drifted",
                                         "task_contract_review": {"verdict": task_verdict}},
        "off_method_observation": {"valid": legal},
        "component_execution": {"mode": "integration", "status": "valid_not_best",
                                "all_components_accepted": True},
    }
    assert _classify_component_evidence(gen)[0] == expected


@pytest.mark.parametrize("lower,values", [(True, [1.0, 1.1, 1.0, 0.9]),
                                          (False, [1.0, 0.9, 1.0, 1.1])])
def test_integration_best_survives_restart_and_ignores_ties_or_regressions(tmp_path, lower, values):
    plan = ComponentPlan("test-method", (node(),), ("gain",))
    path = str(tmp_path / "ledger.json")
    handler = RunnerAgentHandler.__new__(RunnerAgentHandler)
    results = []
    for i, metric in enumerate(values):
        ledger = ComponentLedger(path, plan)
        handler._active_component_context = {"s": {"mode": "integration", "ledger": ledger,
                                                   "plan": plan}}
        result = handler._finalize_component_attempt("s", {
            "verdict": "new_best", "accepted": True, "method_metric": metric,
            "metric_name": "loss", "metric_unit": "unit", "lower_is_better": lower,
            "baseline_metric": 10.0 if lower else 0.0, "repo_path": f"candidate-{i}"})
        results.append(result)
    assert [r["component_execution"]["became_best"] for r in results] == [True, False, False, True]
    assert [r["verdict"] for r in results] == ["new_best", "valid_not_best", "valid_not_best", "new_best"]
    assert all(r["accepted"] for r in results)
    assert ledger.snapshot()["integration_best"]["checkpoint"] == "candidate-3"


@pytest.mark.parametrize("changes,comparable", [
    ({"accepted": False}, True),
    ({"method_metric": float("nan")}, True),
    ({"method_metric": True}, True),
    ({"repo_path": ""}, True),
    ({"metric_name": "different_loss"}, False),
    ({"baseline_metric": 20.0}, False),
    ({"lower_is_better": False}, False),
])
def test_ineligible_integration_cannot_replace_incumbent(tmp_path, changes, comparable):
    plan = ComponentPlan("test-method", (node(),), ("gain",))
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    handler = RunnerAgentHandler.__new__(RunnerAgentHandler)
    handler._active_component_context = {"s": {"mode": "integration", "ledger": ledger,
                                               "plan": plan}}
    first = {"verdict": "new_best", "accepted": True, "method_metric": 1.0,
             "baseline_metric": 10.0, "metric_name": "loss", "metric_unit": "unit",
             "lower_is_better": True, "repo_path": "incumbent"}
    handler._finalize_component_attempt("s", first)
    result = handler._finalize_component_attempt(
        "s", {**first, "method_metric": 0.1, "repo_path": "candidate", **changes})
    assert result["component_execution"]["became_best"] is False
    assert result["component_execution"]["best_comparison_available"] is comparable
    assert ledger.snapshot()["integration_best"]["checkpoint"] == "incumbent"


def test_non_best_accepted_integration_is_not_a_composition_failure():
    gen = {"verdict": "valid_not_best", "accepted": True,
           "component_execution": {"mode": "integration", "all_components_accepted": True}}
    assert _classify_component_evidence(gen)[0] is None


def test_component_observation_corrects_null_final_result_summary():
    finding = "Verified component observation: component=merge, metric=0.15."
    memory = {"best_metric": None, "verified_findings": [finding]}
    report = {"summary": "No component achieved an accepted checkpoint.",
              "verified_findings": [finding]}
    result = _apply_latest_verified_structured_result(
        report, memory, {"method_metric": None, "accepted": False})
    assert "No component achieved" not in result["summary"]
    assert "component" in result["summary"].lower()
    assert "complete" in result["summary"].lower()
    assert result["verified_findings"] == [finding]


def test_metricless_latest_run_preserves_the_best_whole_method_headline():
    memory = {"best_metric": 0.1, "verified_findings": [], "diagnosis": "execution_failure"}
    best = {"accepted": True, "baseline_metric": 1.0, "method_metric": 0.1,
            "metric_name": "loss", "lower_is_better": True}
    result = _apply_latest_verified_structured_result(
        {"summary": "latest failed"}, memory, {"method_metric": None}, best_result=best)
    assert "0.1" in result["summary"]
