"""Small offline counterexamples for the cross-layer Round 2 fixes."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.experiments.repo_improve.runtime_evidence import (
    execution_file_evidence, measured_result, save_output,
)
from autoresearch.spec_store import SpecStore


def record(tmp_path, **payload):
    return {"attempt_id": "a", "launched": True, "finished": True, "rc": 0,
            **save_output(tmp_path, "a", json.dumps(payload), "")}


@pytest.mark.parametrize("metric,actual,valid", [(0, 0.9, False), (7, 0.15, False), (7, 0.10, True)])
def test_quality_feasibility_is_not_exit_zero_or_a_self_reported_boolean(tmp_path, metric, actual, valid):
    run = record(tmp_path, baseline=metric, baseline_valid=True, accuracy=actual)
    config = {"quality_gates": [
        {"pointer": "/accuracy", "operator": "<=", "threshold": 0.12}]}
    result = measured_result(run, config, "baseline")
    assert result["value"] == metric
    assert result["valid"] is valid
    assert result["quality_observations"]["/accuracy"] == actual


@pytest.mark.parametrize("field,expected,observed", [
    ("metric", "composite_score", "throughput"),
    ("prime", 17, 19), ("aggregation", "sum", "mean"),
    ("selection_split", "dev", "test"), ("model", "model-v2", "model-v1"),
])
def test_explicit_shared_contract_rejects_identity_drift(tmp_path, field, expected, observed):
    run = record(tmp_path, method=1.0, candidate_valid=True, **{field: observed})
    result = measured_result(run, {"required_values": {"/" + field: expected}}, "method")
    assert result["valid"] is False
    assert result["value"] == 1.0


def test_empty_gate_list_cannot_disable_correctness(tmp_path):
    run = record(tmp_path, method=1.0, candidate_valid=True)
    with pytest.raises(ValueError, match="nonempty"):
        measured_result(run, {"gate_pointers": []}, "method")


@pytest.mark.parametrize("wrapper", [[], ["timeout", "60s"],
    ["/usr/bin/timeout", "--foreground", "--kill-after=2s", "60s"]])
def test_frozen_evaluator_accepts_transparent_timeout_wrappers(tmp_path, wrapper):
    command = ["python", "evaluate.py"]
    run = record(tmp_path, method=1.0, candidate_valid=True)
    run["argv"] = wrapper + command
    assert measured_result(run, {"argv": command}, "method")["valid"] is True
    with pytest.raises(ValueError, match="frozen evaluator"):
        measured_result(run, {"argv": ["python", "another_evaluator.py"]}, "method")


def test_parent_entry_point_comparison_ignores_only_timeout_wrapper(tmp_path):
    from autoresearch.experiments.repo_improve.tests.test_historical_component_evidence import receipt
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    command = ["python", "evaluate.py"]
    parent = receipt(tmp_path, "parent", tmp_path / "control", {"method": 2.0},
                     argv=["timeout", "--foreground", "60s", *command],
                     execution_protocol={"argv": command})
    candidate = receipt(tmp_path, "candidate", tmp_path / "candidate", {"method": 1.0},
                        argv=command, execution_protocol={"argv": command},
                        dependency_parent_cwd=parent["cwd"],
                        dependency_parent_state={"repo_commit": "commit", "tracked_diff_hash_start": "diff"})
    report = [{"component": "change", "actual": 1.0, "evidence": {"json_pointer": "/method"},
               "parent_metric": 2.0, "parent_evidence": {"ref": "parent", "json_pointer": "/method"}}]
    assert bind_component_measurements(report, candidate, {"parent": parent})[0]["parent_metric"] == 2.0
    parent["argv"] = ["env", "MODE=other", *command]
    with pytest.raises(ValueError, match="frozen evaluator"):
        bind_component_measurements(report, candidate, {"parent": parent})


def test_implementation_can_change_while_evaluator_and_input_identity_stay_frozen(tmp_path):
    evaluator = tmp_path / "verify.py"
    evaluator.write_text("print('independent evaluator')\n")
    source = tmp_path / "candidate.py"
    source.write_text("algorithm = 'new'\n")
    config = {"immutable_files": {"verify.py": hashlib.sha256(evaluator.read_bytes()).hexdigest()},
              "source_files": {"candidate.py": "implementation", "verify.py": "evaluator"}}
    before = execution_file_evidence(tmp_path, config)
    source.write_text("algorithm = 'better'\n")
    after = execution_file_evidence(tmp_path, config)
    assert before["candidate.py"] != after["candidate.py"]
    assert before["verify.py"] == after["verify.py"]
    evaluator.write_text("print('skip checks')\n")
    with pytest.raises(ValueError, match="frozen evaluator/input file changed"):
        execution_file_evidence(tmp_path, config)


def test_runtime_input_identity_must_be_unchanged_and_host_bound(tmp_path):
    run = record(tmp_path, method=1.0, candidate_valid=True)
    config = {"source_files": {"candidate.py": "implementation"}}
    with pytest.raises(ValueError, match="host file evidence"):
        measured_result(run, config, "method")
    run["execution_files_start"] = run["execution_files_end"] = {
        "candidate.py": {"sha256": "a" * 64, "role": "implementation"}}
    result = measured_result(run, config, "method")
    assert result["execution_sources"] == run["execution_files_start"]


def test_repeated_measurement_keeps_same_parent_samples_and_explicit_noise_floor(tmp_path):
    run = record(tmp_path, method=10.0, candidate_valid=True,
                 samples=[9.9, 10.1], control_samples=[10.1, 9.9])
    result = measured_result(run, {"measurement": {
        "kind": "repeated", "repeats": 2, "aggregation": "mean", "minimum_improvement": 0.2}}, "method")
    assert result["control_metric"] == result["value"] == 10.0
    assert result["minimum_improvement"] == 0.2


def test_real_memory_event_schema_accepts_method_scoped_failure_and_last_baseline():
    from autoresearch.eventbus.broker.registry import Registry
    from autoresearch.orchestrator.engine import OrchestratorEngine
    from autoresearch.orchestrator.session_state import SessionState
    engine = object.__new__(OrchestratorEngine)
    engine.spec_store = None
    state = SessionState(session_id="s", method_ref="current", previous_state="MossIntervening",
                         last_baseline_attempt_ref="spec://baseline_result/s/failed",
                         moss_interventions=[{"reason": "implementation_thrashing", "action": "discard_hypothesis"}])
    payload = engine._memory_consolidation_payload(state)
    validator = Registry(Path(__file__).resolve().parents[2] / "eventbus/eventbus-registry.yaml").get_payload_validator(
        "memory.consolidation_requested", 1)
    assert list(validator.iter_errors(payload)) == []
    assert payload["baseline_result_ref"] is None
    assert payload["baseline_attempt_ref"].endswith("/failed")
    assert payload["failure_context"]["method_ref"] == "current"


async def test_finite_infeasible_baseline_survives_later_empty_attempt(tmp_path):
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
    with closing(SpecStore(tmp_path / "store")) as store:
        runner = object.__new__(RunnerAgentHandler)
        runner.spec_store = store
        first = runner._persist_baseline_attempt("s", {
            "metric_value": 7.0, "reproduced": False, "exit_code": 0,
            "measurement_evidence": {"gates": {"quality:/accuracy<=0.12": False}}})
        runner._persist_baseline_attempt("s", {"metric_value": None, "reproduced": False, "exit_code": 1})
        assert store.get(first)["metric_value"] == 7.0
        handler = object.__new__(MemoryAgentHandler)
        handler.spec_store = store
        events = []
        async def publish(**kwargs):
            events.append(kwargs)
        memory = handler._consolidate_failure(
            run_record=None, baseline_result=None, prior_research_memory=None,
            failure_context={"reason": "baseline_reproduction_thrashing"})
        await handler._publish_consolidated(SimpleNamespace(publish=publish), "s", None, memory,
                                           "report", diagnosis=memory["diagnosis"])
        saved = store.get(events[-1]["payload"]["research_memory_ref"])
    assert len(saved["baseline_observations"]) == 2
    observed = next(item for item in saved["baseline_observations"] if item["ref"] == first)
    assert observed["metric_value"] == 7.0 and observed["reproduced"] is False
    assert saved["best_metric"] is None and saved["verified_findings"] == []


def test_truncated_capture_cannot_be_used_to_assert_component_absence():
    from autoresearch.agents.method_implementation_agent import core
    from autoresearch.agents.moss_agent.tests.conftest import FakeLLM
    answer = {"verdict": "drifted", "approach_evidence": [],
              "component_reviews": [{"name": "query", "status": "missing", "evidence": []}],
              "issues": ["implementation absent"], "remediation": "implement query", "reasoning": "not seen"}
    result = core.review_implementation(
        {"approach": "query", "components": [{"name": "query", "mechanism": "query"}]},
        "# SOURCE_CAPTURE_INCOMPLETE: unexpanded query.py", FakeLLM([json.dumps(answer)] * 6))
    assert result["verdict"] == "insufficient_evidence"


def test_host_startup_failure_cannot_trigger_method_discard_or_coder_patch():
    from autoresearch.agents.moss_agent import core
    from autoresearch.agents.moss_agent.tests.conftest import FakeLLM
    from autoresearch.orchestrator.moss_matrix import MossActionGuard, load_moss_action_matrix
    wrong = {"action": "discard_hypothesis", "rationale": "Algorithm failed"}
    misrouted = {"action": "shrink_patch", "rationale": "Repair host", "prompt_patch": {
        "agent": "CoderAgent", "content": "Rewrite the candidate to fix the host socket."}}
    correct = {"action": "escalate_to_user", "rationale": "Repair host configuration before retry."}
    llm = FakeLLM(list(map(json.dumps, (wrong, misrouted, correct))))
    result = core.decide("implementation_thrashing", None,
                         MossActionGuard(load_moss_action_matrix()), llm,
                         diagnostic={"result": {"component_execution": {
                             "status": "execution_failed", "execution_succeeded": False,
                             "infrastructure_attempts": 2}}})
    assert result["action"] == "escalate_to_user"
    assert result["degraded"] is False and len(llm.calls) == 3


def test_parent_quality_failure_is_preserved_in_bound_control_evidence(tmp_path):
    from autoresearch.experiments.repo_improve.tests.test_historical_component_evidence import receipt
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    config = {"quality_gates": [{"pointer": "/accuracy", "operator": "<=", "threshold": 0.12}]}
    parent = receipt(tmp_path, "parent", tmp_path / "control", {"method": 0, "accuracy": 0.9}, execution_protocol=config)
    candidate = receipt(tmp_path, "candidate", tmp_path / "candidate", {"method": 7, "accuracy": 0.1},
                        execution_protocol=config, dependency_parent_cwd=parent["cwd"],
                        dependency_parent_state={"repo_commit": "commit", "tracked_diff_hash_start": "diff"})
    report = [{"component": "attack", "actual": 7, "evidence": {"json_pointer": "/method"},
               "parent_metric": 0, "parent_evidence": {"ref": "parent", "json_pointer": "/method"}}]
    bound = bind_component_measurements(report, candidate, {"parent": parent})[0]
    assert bound["actual"] == 7 and bound["parent_metric"] == 0
    assert bound["parent_evidence"]["failed_gates"] == ["quality:/accuracy<=0.12"]


def test_large_parent_delta_keeps_actual_component_code_without_inherited_fallback():
    from autoresearch.agents.method_implementation_agent.core import bounded_parent_diff
    code = "diff --git a/query.py b/query.py\n--- a/query.py\n+++ b/query.py\n@@ -1 +1 @@\n-old()\n+new()\n"
    data = "diff --git a/result.json b/result.json\n@@ -0,0 +1 @@\n+" + "x" * 130000
    result = bounded_parent_diff(data + "\n" + code)
    assert code in result
    assert "SOURCE_CAPTURE_INCOMPLETE" in result and "result.json" in result
    assert len(result) <= 120000


def test_runtime_inventory_distinguishes_loaded_code_from_later_disk_edits(tmp_path):
    from types import ModuleType
    from autoresearch.spec_store.identity import runtime_identity
    module = ModuleType("autoresearch.fixture_module")
    source = tmp_path / "fixture_module.py"
    source.write_text("def f():\n    return 1\n")
    module.__file__ = str(source)
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    first = runtime_identity({module.__name__: module})["modules"][module.__name__]
    source.write_text("def f():\n    return 2\n")
    second = runtime_identity({module.__name__: module})["modules"][module.__name__]
    assert first["loaded_functions_sha256"] == second["loaded_functions_sha256"]
    assert first["source_sha256"] != second["source_sha256"]
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    third = runtime_identity({module.__name__: module})["modules"][module.__name__]
    assert second["loaded_functions_sha256"] != third["loaded_functions_sha256"]
    assert second["source_sha256"] == third["source_sha256"]


def test_opt_in_runtime_and_human_feedback_are_bound_to_immutable_provenance(tmp_path, monkeypatch):
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.spec_store import identity
    from autoresearch.experiments.repo_improve.source_policy import attribution
    monkeypatch.setattr(identity, "runtime_identity", lambda: {"process_id": 1, "modules": {}})
    with closing(SpecStore(tmp_path / "store")) as store:
        store.put(kind="human_idea_seed", session_id="s", key="v1", obj={"idea": "external correction"}, emitter="Human")
        runner = object.__new__(RunnerAgentHandler)
        runner.spec_store = store
        result = runner._persist_execution_provenance("s", {
            "runtime_provenance": {"process_id": 2, "modules": {}},
            "repository_contract": {"execution_contract": {"record_runtime_identity": True,
                "experiment_provenance": {"campaign": "paired", "arm": "open"}}}})
        versions = [item for item in result["source_history"] if item["role"] == "runtime_version"]
        saved = store.get(versions[0]["ref"])
    assert "/snapshot-" in versions[0]["ref"]
    assert saved["runner"]["process_id"] == 1 and saved["execution"]["process_id"] == 2
    assert saved["declared_experiment"]["arm"] == "open"
    assert "Human-assisted" in attribution(result["source_history"])


@pytest.mark.parametrize("before", [True, False])
def test_attempt_broker_rejects_evaluator_tampering_before_and_during_execution(tmp_path, before):
    from autoresearch.experiments.repo_improve import agentic_improve as ai
    repo = tmp_path / "repo"
    repo.mkdir()
    evaluator = repo / "verify.py"
    evaluator.write_text("trusted")
    config = {"immutable_files": {"verify.py": hashlib.sha256(evaluator.read_bytes()).hexdigest()}}
    bindir, log_path = ai._install_attempt_guard(str(tmp_path), str(repo), str(repo / "ar_result.json"))
    calls = []
    def execute(*args, **kwargs):
        command = args[1]
        calls.append(command)
        if "AUTORESEARCH_ATTEMPT_ACTIVE=1" in command:
            evaluator.write_text("tampered")
        return SimpleNamespace(exit_code=0, stdout='{"method":7,"candidate_valid":true}', stderr="")
    authority = object.__new__(ai._AttemptAuthority)
    authority.uncertain_attempt = None  # Constructor state for this focused fixture.
    authority.repo_path, authority.phase = str(repo), "method"
    authority.log_path, authority.last_attempt_path = log_path, str(Path(bindir).parent / "last_attempt_id")
    authority.backend, authority.handle = SimpleNamespace(execute_command=execute), None
    authority.control_path, authority.artifact_paths = None, []
    authority.execution_protocol, authority.dependency_parent_state = config, {}
    authority.repository_contract = None
    if before:
        evaluator.write_text("tampered")
    request = json.dumps({"argv": ["python", "verify.py"], "cwd": str(repo)}).encode() + b"\n"
    result = authority._handle(SimpleNamespace(recv=lambda _: request))
    assert result["rc"] == 126
    assert sum("AUTORESEARCH_ATTEMPT_ACTIVE=1" in command for command in calls) == (0 if before else 1)
    if before:
        assert result["attempt_id"] is None
    else:
        assert '"method":7' in result["stdout"]
        assert ai._attempt_records(log_path)[result["attempt_id"]]["rc"] == 126


@pytest.mark.parametrize("name,unit,direction,valid", [
    ("latency", "milliseconds", True, True),
    ("throughput", "ms", True, False), ("latency", "s", True, False),
    ("latency", "ms", False, False),
])
def test_baseline_cannot_relabel_the_frozen_objective(tmp_path, monkeypatch, name, unit, direction, valid):
    from autoresearch.experiments.repo_improve import agentic_improve as ai
    run = record(tmp_path, baseline=7, baseline_valid=True)
    run["phase"] = "baseline"
    monkeypatch.setattr(ai, "_attempt_records", lambda _: {"a": run})
    result = ai._finalize_relaxed_phase_result({
        "_metric_attempt_id": "a", "_reported_metric": name, "_reported_metric_unit": unit,
        "_reported_lower_is_better": direction, "metric": name,
        "baseline": ai.RunResult(metric_name=name, metric_value=7.0, valid=True),
        "baseline_validity": True, "repository_contract": {"execution_contract": {"evaluation": {
            "metric_name": "latency", "metric_unit": "ms", "lower_is_better": True}}}},
        "baseline", None, attempt_log="log", repo_path=str(tmp_path))
    assert result["baseline"].metric_value == 7.0
    assert result["baseline"].valid is valid
    if not valid:
        assert "objective identity" in result["error"]


def test_method_prompt_preserves_explicit_contract_without_repository_documents(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt
    contract = {"sources": [], "execution_contract": {"evaluation": {
        "required_values": {"/selection_split": "dev"}, "argv": ["python", "fixed_evaluator.py"]}}}
    prompt = _method_task_prompt("improve", str(tmp_path), str(tmp_path / "ar_result.json"), 60,
                                 {"metric_value": 1.0, "metric_name": "latency", "repository_contract": contract})
    assert '"/selection_split": "dev"' in prompt
    assert "fixed_evaluator.py" in prompt
