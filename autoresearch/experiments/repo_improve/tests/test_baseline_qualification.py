"""An observed zero-score starter is a control, not a passing method candidate."""
import json

import pytest

from autoresearch.experiments.repo_improve import agentic_improve as core
from autoresearch.experiments.repo_improve.runtime_evidence import measured_result, save_output


OUTCOME = {"feasible": False, "official_score": 0,
           "quality_observations": {"accuracy": 0.7},
           "reason": "Official accuracy eligibility threshold was not met."}
CONTRACT = {"quality_gates": [{"pointer": "/accuracy", "operator": ">=", "threshold": 0.9}],
            "required_values": {"/split": "test"}}


def _record(tmp_path, payload, rc=0):
    return {"attempt_id": "attempt-measured", "phase": "baseline", "launched": True,
            "finished": True, "rc": rc,
            **save_output(tmp_path, "attempt-measured", json.dumps(payload), "")}


def _payload():
    return {"baseline": 1234, "baseline_valid": True, "accuracy": 0.7,
            "split": "test", "baseline_outcome": dict(OUTCOME)}


def test_external_reward_is_not_read_or_exported_as_research_feedback(tmp_path):
    payload = _payload()
    payload["external_reward"] = "must never be parsed as a number"
    payload["baseline_outcome"] = {
        **payload["baseline_outcome"], "official_score": "must never be parsed",
        "quality_observations": {"accuracy": 0.7, "official_reward": 0.5},
        "reason": "official reward is 0.5"}
    measured = measured_result(_record(tmp_path, payload),
        {**CONTRACT, "score_pointer": "/external_reward"}, "baseline")
    assert measured["valid"] is True
    assert measured["value"] == 1234
    assert "official_score" not in json.dumps(measured)
    assert "official_reward" not in json.dumps(measured)
    assert "official reward is" not in json.dumps(measured)
    assert measured["quality_observations"]["/accuracy"] == 0.7


def test_baseline_feasibility_uses_declared_quality_not_reward_eligibility(tmp_path):
    payload = _payload()
    measured = measured_result(_record(tmp_path, payload), {}, "baseline")
    assert measured["baseline_outcome"]["feasible"] is True
    assert measured["valid"] is True


def test_legacy_reward_ineligibility_is_not_inferred_into_method_prompt():
    prompt = core._method_task_prompt("improve", "/repo", "/repo/ar_result.json", 100,
        {"metric_name": "parameters", "metric_value": 1234, "lower_is_better": True,
         "measurement_evidence": {"baseline_outcome": OUTCOME,
                                  "gates": {"/baseline_valid": True}}})
    assert '"feasible": true' in prompt
    assert '"feasible": false' not in prompt


@pytest.mark.parametrize("field", ["official_reward", "OfficialReward", "OFFICIAL_SCORE"])
def test_legacy_qualification_prompt_does_not_carry_nested_reward(field):
    prompt = core._method_task_prompt("improve", "/repo", "/repo/ar_result.json", 100,
        {"metric_name": "parameters", "metric_value": 1234, "lower_is_better": True,
         "measurement_evidence": {"baseline_outcome": {
             **OUTCOME, "quality_observations": {"accuracy": 0.7, field: 0.5},
             "reason": "official reward is 0.5"}}})
    assert json.dumps(field) not in prompt
    assert "official reward is" not in prompt
    assert '"accuracy": 0.7' in prompt


@pytest.mark.parametrize("field", ["official_reward", "official_score"])
@pytest.mark.parametrize("use", ["metric", "quality", "objective"])
def test_external_benchmark_fields_cannot_be_research_objectives_or_gates(tmp_path, field, use):
    payload = _payload()
    payload[field] = "must not be parsed"
    config = {"metric_pointer": "/" + field} if use == "metric" else {
        "quality_gates": [{"pointer": "/" + field, "operator": ">=", "threshold": 0}]}
    if use == "objective":
        config = {}
    with pytest.raises(ValueError, match="external benchmark reward"):
        measured_result(_record(tmp_path, payload), config, "baseline",
                        expected_metric_name=field if use == "objective" else None)


def test_zero_score_starter_freezes_local_objective_and_host_evidence(tmp_path):
    payload = {**_payload(), "metric": {"name": "parameters", "unit": "count",
                                       "lower_is_better": True}, "quality_gate_passed": False}
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps(payload))
    record = _record(tmp_path, payload)
    log = tmp_path / "attempts.tsv"
    log.write_text("\n".join(json.dumps({**record, "event": event}) for event in ("launch", "finish")))
    result = core._build_result_from_file(str(path), "example", None, None,
                                          lambda *args: None, "baseline")
    result["repository_contract"] = {"execution_contract": {"evaluation": CONTRACT}}
    result = core._finalize_relaxed_phase_result(result, "baseline", None,
                                                  attempt_log=str(log), repo_path=str(tmp_path))
    assert result["baseline"].valid is True
    assert result["baseline"].metric_value == 1234
    evidence = result["measurement_evidence"]
    assert evidence["baseline_outcome"]["feasible"] is False
    assert evidence["baseline_outcome"]["quality_observations"] == {"/accuracy": 0.7}
    assert "official_score" not in json.dumps(evidence)
    assert evidence["gates"]["quality:/accuracy>=0.9"] is False
    assert evidence["attempt_id"] == "attempt-measured"
    # The adapter already preserves this evidence in the frozen BaselineResult.
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    _, measured = _map_repo_result("example", {
        **result, "baseline": {"metric": 1234, "valid": True}, "candidate": {}})
    assert measured["measurement_evidence"] == evidence


def test_ineligible_baseline_without_visible_official_score_keeps_raw_objective(tmp_path):
    payload = _payload()
    payload["baseline_outcome"].pop("official_score")
    measured = measured_result(_record(tmp_path, payload), CONTRACT, "baseline")
    assert measured["valid"] is True
    assert measured["value"] == 1234
    assert measured["baseline_outcome"]["feasible"] is False
    assert "official_score" not in measured


def test_external_score_does_not_change_raw_baseline_qualification(tmp_path):
    payload = _payload()
    payload["baseline_outcome"].pop("official_score")
    payload["reward"] = 0.5
    measured = measured_result(_record(tmp_path, payload),
                               {**CONTRACT, "score_pointer": "/reward"}, "baseline")
    assert measured["valid"] is True
    assert measured["baseline_outcome"]["feasible"] is False
    assert "official_score" not in json.dumps(measured)


def test_zero_score_cannot_replace_explicitly_measured_objective(tmp_path):
    payload = {**_payload(), "baseline": 0,
               "metric": {"name": "total_params", "unit": "count", "lower_is_better": True},
               "baseline_outcome": {**OUTCOME, "quality_observations": {
                   "measured_total_params": 17988}}}
    with pytest.raises(ValueError, match="disagrees with measured_total_params"):
        measured_result(_record(tmp_path, payload), CONTRACT, "baseline")
    payload["baseline"] = 17988
    passing = tmp_path / "passing"
    passing.mkdir()
    assert measured_result(_record(passing, payload), CONTRACT, "baseline")["value"] == 17988


@pytest.mark.parametrize("failure", ["execution", "identity", "implementation", "missing_objective",
                                     "false_feasibility", "missing_reason"])
def test_baseline_qualification_never_waives_execution_or_identity(tmp_path, failure):
    payload = _payload()
    if failure == "identity":
        payload["split"] = "another"
    elif failure == "implementation":
        payload["baseline_valid"] = False
    elif failure == "missing_objective":
        payload["baseline"] = None
    elif failure == "false_feasibility":
        payload["baseline_outcome"]["feasible"] = True
    elif failure == "missing_reason":
        payload["baseline_outcome"].pop("reason")
    record = _record(tmp_path, payload, rc=1 if failure == "execution" else 0)
    if failure in {"identity", "implementation"}:
        measured = measured_result(record, CONTRACT, "baseline")
        assert measured["valid"] is False
        assert measured["baseline_outcome"]["feasible"] is False
    else:
        with pytest.raises(ValueError):
            measured_result(record, CONTRACT, "baseline")


def test_same_outcome_cannot_waive_method_gate(tmp_path):
    payload = {**_payload(), "method": 100, "candidate_valid": True}
    measured = measured_result(_record(tmp_path, payload), CONTRACT, "method")
    assert measured["valid"] is False
    assert "baseline_outcome" not in measured
    assert core._explicit_candidate_validity({"candidate_valid": True,
                                               "quality_gate_passed": False}) is False


def test_unclassified_failed_quality_gate_keeps_legacy_rejection(tmp_path):
    payload = _payload()
    payload.pop("baseline_outcome")
    assert measured_result(_record(tmp_path, payload), CONTRACT, "baseline")["valid"] is False


def test_baseline_correctness_is_not_overridden_by_feasibility_record():
    assert core._explicit_baseline_validity({**_payload(), "official_correctness": False}, "baseline") is False


def test_method_prompt_keeps_qualification_without_baseline_reward():
    prompt = core._method_task_prompt("improve", "/repo", "/repo/ar_result.json", 100,
        {"metric_name": "parameters", "metric_value": 1234, "lower_is_better": True,
         "measurement_evidence": {"baseline_outcome": OUTCOME,
                                  "gates": {"quality:/accuracy>=0.9": False}}})
    assert '"feasible": false' in prompt
    assert '"official_score"' not in prompt
    assert "does not relax any candidate correctness or quality gate" in prompt


def test_baseline_prompt_resolves_provisional_hints_without_switching_to_reward():
    prompt = core._baseline_task_prompt("optimize repository task", "/repo", "/repo/ar_result.json",
                                        "repository-defined", False)
    assert "Resolve explicitly provisional metric/direction hints" in prompt
    assert "do not switch to reward merely to match" in prompt
    assert "keep the measured objective in baseline" in prompt.lower()
    assert "do not invent a target or an official score" in prompt
