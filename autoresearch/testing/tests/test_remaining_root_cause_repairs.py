"""Cross-layer counterexamples for the remaining R1–R11 workflow repairs."""
import json
from pathlib import Path
import subprocess

import pytest

from autoresearch.agents.deliberation_agent.core import _repository_review_problems
from autoresearch.experiments.repo_improve import agentic_improve as runner
from autoresearch.experiments.repo_improve.checkpoint_artifacts import (
    snapshot_checkpoint, validate_checkpoint, restore_external)
from autoresearch.experiments.repo_improve.component_plan import ComponentPlan, ComponentLedger
from autoresearch.experiments.repo_improve.runtime_evidence import (
    save_output, read_output, measured_result, bind_component_measurements)
from autoresearch.experiments.repo_improve.source_policy import (
    permitted, merge_source_history, attribution)
from autoresearch.testing.tests.test_root_cause_repairs import node


def output(tmp_path, payload, name="attempt-candidate", **changes):
    record = {"attempt_id": name, "launched": True, "finished": True, "rc": 0,
              "repo_commit": "base", "repo_commit_end": "base", "cwd": "/repo",
              "tracked_diff_hash_start": "parent", "tracked_diff_hash_end": "parent"}
    record.update(save_output(tmp_path, name, json.dumps(payload), ""))
    record.update(changes)
    return record


def grounded_review():
    method = {"components": [{"component_id": "gain", "type": "direct_gain", "inputs": ["data"]}]}
    baseline = {"repository_contract": {"sources": [{"path": "task.md", "content":
        "Forward transform. Online API. Precomputed keys. One CPU thread. Deterministic evaluator. No runtime weights. No reference input."}]}}
    critic = {"problems": [], "repository_checks": [
        {"topic": topic, "status": "supported", "source_path": "task.md", "quote": quote}
        for topic, quote in zip(("task_direction", "interface_and_inputs", "existing_mechanism", "resource_scale",
                                 "measurement_protocol", "artifact_scope", "source_boundary"),
                               ("Forward transform.", "Online API.", "Precomputed keys.", "One CPU thread.",
                                "Deterministic evaluator.", "No runtime weights.", "No reference input."))],
        "component_checks": [{"component_id": "gain", "kind": "intervention",
                              "independently_executable": True, "parent_control_available": True,
                              "input_bindings": {"data": "baseline"}}]}
    return method, baseline, critic


@pytest.mark.parametrize("topic", ["task_direction", "interface_and_inputs", "existing_mechanism", "resource_scale"])
def test_r1_empty_problem_list_does_not_override_contradicted_repository_fact(topic):
    method, baseline, critic = grounded_review()
    assert not _repository_review_problems(critic, baseline, method)
    next(check for check in critic["repository_checks"] if check["topic"] == topic)["status"] = "contradicted"
    assert _repository_review_problems(critic, baseline, method)


def test_r1_invented_quote_and_missing_review_do_not_freeze_plan():
    method, baseline, critic = grounded_review()
    critic["repository_checks"][0]["quote"] = "Reverse transform"
    assert _repository_review_problems(critic, baseline, method)
    assert _repository_review_problems({"problems": []}, baseline, method)


@pytest.mark.parametrize("change", [
    {"source_path": "baseline_result.repository_contract.sources[0]"},
    {"quote": "Reverse transform"},
])
def test_repository_citation_mismatch_requests_binding_repair_not_new_experiment(change):
    method, baseline, critic = grounded_review()
    critic["repository_checks"][0].update(change)
    problems = _repository_review_problems(critic, baseline, method)
    assert len(problems) == 1
    problem = problems[0]
    assert "evidence binding mismatch" in problem["issue"]
    assert "task.md" in problem["suggestion"]
    assert "baseline_result" in problem["suggestion"]
    assert "Critic" in problem["suggestion"]
    assert "Measure missing facts" not in problem["suggestion"]
    critic["repository_checks"][0].update(source_path="task.md", quote="Forward transform.")
    assert not _repository_review_problems(critic, baseline, method)


@pytest.mark.parametrize("change", [{"kind": "measurement"}, {"independently_executable": False},
                                   {"parent_control_available": False}, {"input_bindings": {}}])
def test_r2_r3_measurement_only_or_inseparable_gain_node_cannot_pass_review(change):
    method, baseline, critic = grounded_review()
    critic["component_checks"][0].update(change)
    assert _repository_review_problems(critic, baseline, method)


def test_r2_measurement_node_is_allowed_without_claiming_performance_gain():
    method, baseline, critic = grounded_review()
    method["components"][0]["type"] = "indirect_gain"
    critic["component_checks"][0]["kind"] = "measurement"
    assert not _repository_review_problems(critic, baseline, method)


def test_abbreviated_input_binding_reports_protocol_mismatch_not_inseparable_component():
    method, baseline, critic = grounded_review()
    exact_input = "data — fixed validation samples"
    method["components"][0]["inputs"] = [exact_input]
    problems = _repository_review_problems(critic, baseline, method)
    assert len(problems) == 1
    assert "input binding mismatch" in problems[0]["issue"]
    assert exact_input in problems[0]["issue"]
    assert "Critic" in problems[0]["suggestion"]
    assert "independently implemented" not in problems[0]["issue"]
    critic["component_checks"][0]["input_bindings"] = {exact_input: "baseline"}
    assert not _repository_review_problems(critic, baseline, method)


def test_abbreviated_ancestor_output_reports_exact_available_output():
    method, baseline, critic = grounded_review()
    exact_output = "features — normalized representation"
    method["components"].insert(0, {"component_id": "parent", "type": "indirect_gain",
                                   "inputs": ["task"], "outputs": [exact_output]})
    method["components"][1]["depends_on"] = ["parent"]
    critic["component_checks"].append({"component_id": "parent", "kind": "measurement",
        "independently_executable": True, "parent_control_available": True,
        "input_bindings": {"task": "baseline"}})
    critic["component_checks"][0]["input_bindings"] = {"data": "parent:features"}
    problems = _repository_review_problems(critic, baseline, method)
    assert len(problems) == 1
    assert "input binding mismatch" in problems[0]["issue"]
    assert exact_output in problems[0]["issue"]
    critic["component_checks"][0]["input_bindings"] = {"data": "parent:" + exact_output}
    assert not _repository_review_problems(critic, baseline, method)


@pytest.mark.parametrize("sources", [[{"path": "public/source.py", "source_id": "other.py"}], "public/source.py"])
def test_reference_source_error_explains_the_required_json_element_type(sources):
    from autoresearch.experiments.repo_improve.source_policy import source_findings
    with pytest.raises(ValueError, match="JSON array of strings"):
        source_findings({"reference_sources": sources}, {})
    findings = source_findings({"reference_sources": ["public/source.py"]}, {})
    assert findings[0]["path"] == "public/source.py"


@pytest.mark.parametrize("description_field", ["usage", "relevance", "used_for", "supports"])
def test_structured_method_source_keeps_policy_checks_and_original_metadata(description_field):
    from autoresearch.experiments.repo_improve.source_policy import source_findings
    record = {"source_id": "restricted/reference.py", "source_type": "code", description_field: "comparison"}
    method = {"reference_sources": [record, "public/source.py"]}
    findings = source_findings(method, {"execution_contract": {"sources": {"forbidden": ["restricted/"]}}})
    assert findings == [
        {"path": "restricted/reference.py", "allowed": False, "role": "method_reference"},
        {"path": "public/source.py", "allowed": True, "role": "method_reference"}]
    assert method["reference_sources"][0] == record


def test_path_source_record_preserves_metadata_and_checks_forbidden_source():
    from autoresearch.experiments.repo_improve.source_policy import source_findings
    record = {"path": "restricted/instruction.md", "role": "task_contract",
              "evidence": "The task defines the execution boundary."}
    method = {"reference_sources": [record, "public/source.py"]}
    findings = source_findings(method, {"execution_contract": {"sources": {"forbidden": ["restricted/"]}}})
    assert findings == [
        {"path": "restricted/instruction.md", "allowed": False, "role": "method_reference"},
        {"path": "public/source.py", "allowed": True, "role": "method_reference"}]
    assert method["reference_sources"][0] == record


@pytest.mark.parametrize("record", [{"source_id": ""}, {"source_id": 7},
    {"source_id": "public/source.py", "path": "restricted/answer.py"},
    {"path": ""}, {"path": 7}, {"path": "public/source.py", "url": "restricted/answer.py"}])
def test_ambiguous_structured_sources_are_not_silently_dropped(record):
    from autoresearch.experiments.repo_improve.source_policy import source_findings
    with pytest.raises(ValueError, match="source_id"):
        source_findings({"reference_sources": [record]}, {})


@pytest.mark.parametrize("use_unrelated_branch", [False, True])
def test_component_can_be_checked_with_shared_ancestors_but_not_an_unrelated_branch(use_unrelated_branch):
    method, baseline, critic = grounded_review()
    nodes = [
        ("root", [], ["data"], {"data": "baseline"}),
        ("left", ["root"], ["features"], {"features": "root:output"}),
        ("right", ["root"], ["features"], {"features": "root:output"}),
        ("unrelated", [], ["data"], {"data": "baseline"}),
        ("join", ["left", "right"], ["lhs", "rhs", "source"],
         {"lhs": "left:output", "rhs": "right:output",
          "source": "unrelated:output" if use_unrelated_branch else "root:output"}),
    ]
    method["components"] = [
        {"component_id": cid, "depends_on": parents, "inputs": inputs,
         "outputs": ["output"], "type": "indirect_gain"}
        for cid, parents, inputs, _ in nodes
    ]
    critic["component_checks"] = [
        {"component_id": cid, "kind": "measurement", "independently_executable": True,
         "parent_control_available": True, "input_bindings": bindings}
        for cid, _, _, bindings in nodes
    ]
    problems = _repository_review_problems(critic, baseline, method)
    assert bool(problems) is use_unrelated_branch
    if problems:
        assert all(problem["component"] == "join" for problem in problems)


def test_r4_component_actual_cannot_disagree_with_runtime_output(tmp_path):
    record = output(tmp_path, {"measurements": {"swaps": 16256}})
    report = [{"component": "swaps", "actual": 16384,
               "evidence": {"source": "runtime_attempt", "ref": record["attempt_id"]}}]
    with pytest.raises(ValueError, match="disagrees"):
        bind_component_measurements(report, record, {record["attempt_id"]: record})
    report[0]["actual"] = 16256
    bound = bind_component_measurements(report, record, {record["attempt_id"]: record})
    assert bound[0]["_host_bound"] is True
    assert bound[0]["actual"] == 16256


def test_r4_all_declared_quality_gates_are_required_even_with_better_metric(tmp_path):
    record = output(tmp_path, {"parameters": 2081, "passed_constraints": False, "correct": True})
    config = {"metric_pointer": "/parameters", "gate_pointers": ["/passed_constraints", "/correct"]}
    assert measured_result(record, config, "method")["valid"] is False
    config["gate_pointers"].append("/missing_gate")
    with pytest.raises(ValueError, match="missing"):
        measured_result(record, config, "method")


def test_r4_failed_evaluator_or_changed_saved_output_never_certifies_old_json(tmp_path):
    record = output(tmp_path, {"method": 0, "candidate_valid": True}, rc=1)
    with pytest.raises(ValueError, match="successfully"):
        read_output(record)
    record["rc"] = 0
    assert measured_result(record, {}, "method")["valid"] is True  # legal zero is allowed
    Path(record["stdout_path"]).write_text('{"method": -1, "candidate_valid": true}')
    with pytest.raises(ValueError, match="changed"):
        read_output(record)


def test_validation_only_output_cannot_be_bound_as_test_metric(tmp_path):
    payload = {"method": 0.51, "candidate_valid": True,
               "val_acc_mean": 0.51, "val_exit_code": 0}
    validation = output(tmp_path, payload)
    with pytest.raises(ValueError, match="validation-only val_acc_mean"):
        measured_result(validation, {}, "method", expected_metric_name="test_acc_mean")
    assert measured_result(validation, {}, "method")["value"] == 0.51
    payload["test_acc_mean"] = 0.51
    test = output(tmp_path, payload, name="attempt-test")
    assert measured_result(test, {}, "method", expected_metric_name="test_acc_mean")["valid"]


def test_runtime_output_accepts_one_multiline_json_surrounded_by_logs(tmp_path):
    payload = {"failed_check_count": 0, "checks": [{"passed": True}],
               "measurements": {"component": {"failed_check_count": 0}}}
    raw = "[check] starting\n[check] exit 0\n" + json.dumps(payload, indent=2) + "\n[check] done\n"
    record = {"launched": True, "finished": True, "rc": 0,
              **save_output(tmp_path, "probe", raw, "")}
    assert read_output(record) == payload


@pytest.mark.parametrize("raw", [
    'log\n{"metric": 1}\n{"metric": 2}\n',
    'log\n{"metric": 1} {"metric": 2}\n',
    'log\n{broken\n{"metric": 1}\n}\n',
    'log\n{\n  "nested": {"metric": 1}\n',
    'log without a result\n',
])
def test_runtime_output_never_selects_one_of_ambiguous_or_broken_results(tmp_path, raw):
    record = {"launched": True, "finished": True, "rc": 0,
              **save_output(tmp_path, "probe", raw, "")}
    with pytest.raises(ValueError, match="unambiguous JSON"):
        read_output(record)


def test_r5_control_must_resolve_to_the_exact_parent_and_output(tmp_path):
    parent = output(tmp_path, {"method": 2}, "attempt-parent")
    candidate = output(tmp_path, {"method": 1}, dependency_parent_state={
        "repo_commit": "base", "tracked_diff_hash_start": "parent"})
    report = [{"component": "gain", "parent_metric": 2,
               "parent_evidence": {"ref": "attempt-parent", "json_pointer": "/method"}}]
    records = {"attempt-parent": parent, candidate["attempt_id"]: candidate}
    assert bind_component_measurements(report, candidate, records)[0]["parent_evidence"]["host_bound"]
    with pytest.raises(ValueError):
        bind_component_measurements(report, candidate, {})
    parent["tracked_diff_hash_start"] = "different-ancestor"
    with pytest.raises(ValueError, match="exact dependency"):
        bind_component_measurements(report, candidate, records)


def test_control_pointer_error_names_parent_field_not_candidate(tmp_path):
    protocol = {"metric_pointer": "/baseline_value"}
    parent = output(tmp_path, {"baseline_value": 2}, "attempt-parent",
                    execution_protocol=protocol)
    candidate = output(tmp_path, {"method": 1}, dependency_parent_state={
        "repo_commit": "base", "tracked_diff_hash_start": "parent"},
        execution_protocol=protocol)
    report = [{"component": "gain", "parent_metric": 2,
               "parent_evidence": {"ref": "attempt-parent"}}]
    records = {"attempt-parent": parent, candidate["attempt_id"]: candidate}
    with pytest.raises(ValueError, match="parent_evidence.json_pointer") as error:
        bind_component_measurements(report, candidate, records)
    assert "baseline_value" in str(error.value)
    report[0]["parent_evidence"]["json_pointer"] = "/baseline_value"
    assert bind_component_measurements(report, candidate, records)[0]["parent_metric"] == 2


@pytest.mark.parametrize("expected_cwd", ["/control/.", "/control/"])
@pytest.mark.parametrize("mismatch", [None, "cwd", "repo_commit", "tracked_diff_hash_end", "execution_protocol"])
def test_r5_parent_path_spelling_does_not_weaken_identity_checks(tmp_path, expected_cwd, mismatch):
    parent = output(tmp_path, {"method": 2}, "attempt-parent", cwd="/control", execution_protocol={})
    candidate = output(tmp_path, {"method": 1}, cwd="/candidate", execution_protocol={},
                       dependency_parent_cwd=expected_cwd, dependency_parent_state={
                           "repo_commit": "base", "tracked_diff_hash_start": "parent"})
    report = [{"component": "gain", "parent_metric": 2,
               "parent_evidence": {"ref": "attempt-parent", "json_pointer": "/method"}}]
    records = {"attempt-parent": parent, candidate["attempt_id"]: candidate}
    if mismatch:
        parent[mismatch] = {"changed": True} if mismatch == "execution_protocol" else "/different"
        with pytest.raises(ValueError, match="exact dependency"):
            bind_component_measurements(report, candidate, records)
    else:
        assert bind_component_measurements(report, candidate, records)[0]["parent_evidence"]["host_bound"]


def test_r8_extreme_sample_is_not_the_declared_aggregate(tmp_path):
    config = {"measurement": {"kind": "repeated", "repeats": 3, "aggregation": "median",
                              "minimum_improvement": 0.05}}
    bad = output(tmp_path, {"method": 0.1, "candidate_valid": True, "samples": [1, 1.1, 0.1],
                            "control_samples": [1.2, 1.3, 1.2]}, "attempt-bad")
    with pytest.raises(ValueError, match="aggregate"):
        measured_result(bad, config, "method")
    good = output(tmp_path, {"method": 1, "candidate_valid": True, "samples": [1, 1.1, 0.1],
                             "control_samples": [1.2, 1.3, 1.2]}, "attempt-good")
    evidence = measured_result(good, config, "method")
    assert evidence["control_metric"] == 1.2
    assert evidence["samples"] == [1, 1.1, 0.1]
    assert evidence["minimum_improvement"] == 0.05


def test_r9_infrastructure_is_bounded_without_erasing_scientific_budget(tmp_path):
    plan = ComponentPlan("method", (node(),), ("gain",))
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("gain")
    for _ in range(3):
        failed = ledger.record_attempt("gain", execution_started=False, execution_succeeded=False,
                                       max_attempts=3, error="checkout unavailable")
    assert failed.attempts == 0
    assert failed.infrastructure_attempts == 3
    assert failed.last_outcome == "infrastructure_exhausted"
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    recovered = ledger.record_attempt("gain", execution_succeeded=True, contract_passed=True,
                                      checkpoint="A", quality_value=1, max_attempts=3)
    assert recovered.attempts == 1
    assert recovered.checkpoint == "A"


def repository(tmp_path, name="repo"):
    root = tmp_path / name
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "solve.py").write_text("version = 'base'\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@local",
                    "commit", "-qm", "base"], check=True)
    return root


def test_r10_best_code_and_declared_weights_survive_shared_output_overwrite(tmp_path):
    root = repository(tmp_path)
    (root / "solve.py").write_text("version = 'A'\n")
    (root / "weights.bin").write_bytes(b"weight-A")
    external = tmp_path / "shared-output.bin"
    external.write_bytes(b"external-A")
    checkpoint = snapshot_checkpoint(str(root), "attempt-A", ["weights.bin", str(external)])
    (root / "solve.py").write_text("version = 'B'\n")
    (root / "weights.bin").write_bytes(b"weight-B")
    external.write_bytes(b"external-B")
    target = tmp_path / "next"
    subprocess.run(["git", "clone", "--quiet", str(root), str(target)], check=True)
    runner._apply_prior_worktree(str(target), checkpoint)
    assert (target / "weights.bin").read_bytes() == b"weight-A"
    assert (target / "solve.py").read_text() == "version = 'A'\n"
    class Local:
        def execute_command(self, handle, command, cwd, timeout):
            from types import SimpleNamespace
            done = subprocess.run(command, shell=True, cwd=cwd, timeout=timeout)
            return SimpleNamespace(exit_code=done.returncode)
    restore_external(checkpoint, Local(), None, str(target))
    assert external.read_bytes() == b"external-A"
    (Path(checkpoint) / "solve.py").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        runner._apply_prior_worktree(str(target), checkpoint)


def test_r11_document_selection_respects_forbidden_and_allowed_sources(tmp_path):
    (tmp_path / "task.toml").write_text('[autoresearch.sources]\nforbidden = ["private", "solution/answer"]\nallowed = ["solution/public"]\n')
    for directory in ("private", "solution/answer", "solution/public", "design"):
        folder = tmp_path / directory
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "README.md").write_text("source fixture")
    assert not permitted("private/README.md", {"forbidden": ["private"]})
    assert permitted("design/README.md")
    contract = runner._repository_task_contract(str(tmp_path), "private answer")
    assert all("private" not in s["path"] and "solution/answer" not in s["path"] for s in contract["sources"])


def test_r11_method_revision_and_rollback_do_not_erase_reference_exposure():
    exposure = [{"path": "solution/reference.md", "allowed": False, "role": "method_reference"}]
    carried = merge_source_history(exposure, [], exposure)
    assert carried == exposure
    assert "independent" in attribution(carried)
    method, baseline, critic = grounded_review()
    method["reference_sources"] = ["solution/reference.md"]
    baseline["repository_contract"]["execution_contract"] = {"sources": {"forbidden": ["solution"]}}
    assert any("prohibited" in p["issue"] for p in _repository_review_problems(critic, baseline, method))


def test_r1_grounded_contradiction_routes_cap_to_method_revision():
    from autoresearch.agents.deliberation_agent import core
    from autoresearch.agents.deliberation_agent.tests.test_method_debate import _RoleLLM, _PM, _LIT, _DRAFT
    _, baseline, critic = grounded_review()
    critic["repository_checks"][0]["status"] = "contradicted"
    critic["method_revision_required"] = True
    critic["method_revision_reason"] = (
        "The confirmed Method requires a reverse transform, but task.md requires Forward transform."
    )
    llm = _RoleLLM(conv=[{"converged": True}], critic=[critic])
    result = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, baseline_result=baseline,
                                    inner_rounds=1, outer_rounds=3, component_plan_audit=True)
    assert result.report["accepted"] is False
    assert result.report["method_revision_required"] is True
    assert result.report["outer_rounds"] == 1


def test_r2_inputs_cannot_bind_to_a_missing_ancestor_output():
    method, baseline, critic = grounded_review()
    method["components"].insert(0, {"component_id": "parent", "type": "indirect_gain",
                                   "inputs": ["task"], "outputs": ["keys"]})
    method["components"][1]["depends_on"] = ["parent"]
    critic["component_checks"].append({"component_id": "parent", "kind": "measurement",
        "independently_executable": True, "parent_control_available": True, "input_bindings": {"task": "baseline"}})
    critic["component_checks"][0]["input_bindings"] = {"data": "parent:keys"}
    assert not _repository_review_problems(critic, baseline, method)
    critic["component_checks"][0]["input_bindings"] = {"data": "parent:missing"}
    assert _repository_review_problems(critic, baseline, method)


def test_r4_external_score_is_not_research_feedback(tmp_path):
    record = output(tmp_path, {"method": 10, "candidate_valid": True, "official_score": 0.7},
                    argv=["timeout", "10s", "python", "evaluate.py"])
    config = {"argv": ["python", "evaluate.py"], "score_pointer": "/official_score",
              "score_contract": {"anchor_source": "official task contract", "anchor": 30}}
    evidence = measured_result(record, config, "method")
    assert evidence["value"] == 10
    assert "official_score" not in evidence
    assert "score_contract" not in evidence
    config["argv"] = ["python", "cached.py"]
    with pytest.raises(ValueError, match="entry point"):
        measured_result(record, config, "method")


def test_r4_exit_one_diagnostic_is_bound_but_cannot_be_a_success(tmp_path):
    record = output(tmp_path, {"measurements": {"check": 0}}, rc=1)
    report = [{"component": "check", "actual": 0, "verdict": "unmet"}]
    bound = bind_component_measurements(report, record, {}, diagnostic=True)
    assert bound[0]["_host_bound"] is True
    with pytest.raises(ValueError, match="successfully"):
        measured_result(record, {}, "method")
    report[0]["actual"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        bind_component_measurements(report, record, {}, diagnostic=True)


def test_r8_qualified_near_tie_retains_prior_and_preserves_observation(tmp_path):
    plan = ComponentPlan("method", (node(),), ("gain",))
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("gain")
    ledger.record_attempt("gain", execution_succeeded=True, contract_passed=True,
                          checkpoint="A", metric_value=1, quality_value=1, max_attempts=3)
    result = ledger.record_attempt("gain", execution_succeeded=True, contract_passed=True,
        checkpoint="B", metric_value=0.999, quality_value=0.999, max_attempts=3,
        minimum_metric_improvement=0.01)
    # Parent admission and comparison with the incumbent are distinct comparisons.
    # Preserve both observations, but do not swap the selected best within tolerance.
    assert result.checkpoint == "A"
    assert result.qualified_results[-1]["checkpoint"] == "B"


def test_r10_delayed_run_cannot_borrow_new_current_and_snapshot_is_immutable(tmp_path):
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest, result_for_run
    store = SpecStore(tmp_path / "store")
    def snapshot(kind, obj):
        return store.put(kind=kind, session_id="s", key="snapshot-" + artifact_digest(obj).split(":")[1],
                         obj=obj, emitter="RunnerAgent")
    run_a = snapshot("run_record", {"metric_value": 1})
    candidate_a = {"run_record_snapshot_ref": run_a, "method_metric": 1}
    ref_a = snapshot("gen_experiment_result", candidate_a)
    run_b = snapshot("run_record", {"metric_value": 2})
    candidate_b = {"run_record_snapshot_ref": run_b, "method_metric": 2}
    snapshot("gen_experiment_result", candidate_b)
    store.put(kind="gen_experiment_result", session_id="s", key="current", obj=candidate_b, emitter="RunnerAgent")
    assert result_for_run(store, run_a)["candidate_snapshot_ref"] == ref_a
    assert result_for_run(store, run_a)["method_metric"] == 1
    with pytest.raises(ValueError):
        store.put(kind="gen_experiment_result", session_id="s", key=ref_a.rsplit("/", 1)[1],
                  obj=candidate_b, emitter="RunnerAgent")
    missing = snapshot("run_record", {"metric_value": 3})
    with pytest.raises(KeyError):
        result_for_run(store, missing)


def test_r11_confirmed_source_policy_applies_before_document_selection(tmp_path):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "README.md").write_text("Allowed reference")
    (tmp_path / "task.md").write_text("Public task")
    (tmp_path / "README.md").symlink_to(tmp_path / "solution" / "README.md")
    hidden = runner._repository_task_contract(str(tmp_path), "solution", {"sources": {"forbidden": ["solution"]}})
    assert all(s["path"] != "README.md" for s in hidden["sources"])
    allowed = runner._repository_task_contract(str(tmp_path), "solution", {"sources": {"allowed": ["solution"]}})
    assert any(s["path"] == "solution/README.md" for s in allowed["sources"])


@pytest.mark.asyncio
async def test_r6_r11_failure_memory_keeps_reconciliation_seed_and_exposure(tmp_path):
    from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
    from autoresearch.agents.memory_agent.tests.test_gen_handler import FakeBusClient
    from autoresearch.spec_store import SpecStore, refs
    store = SpecStore(tmp_path / "store")
    previous = {"source_history": [{"path": "reference.py", "allowed": True, "role": "method_reference"}],
                "reconciliation_candidates": [{"result_ref": "saved", "checkpoint": "A"}]}
    store.put(kind="research_memory", session_id="s", key="v1", obj=previous, emitter="MemoryAgent")
    handler = MemoryAgentHandler.__new__(MemoryAgentHandler)
    handler.spec_store = store
    await handler._publish_consolidated(FakeBusClient(), "s", None, {"diagnosis": "execution_failure"}, "report")
    current = store.get(refs.encode("research_memory", "s", "v1"))
    assert current["source_history"] == previous["source_history"]
    assert current["reconciliation_candidates"] == previous["reconciliation_candidates"]


def test_r5_r10_control_uses_parent_external_weights_and_restores_candidate(tmp_path):
    import os
    from types import SimpleNamespace
    root = repository(tmp_path)
    external = tmp_path / "weights.bin"
    external.write_bytes(b"parent-A")
    parent = snapshot_checkpoint(str(root), "attempt-parent", [str(external)])
    external.write_bytes(b"candidate-B")
    control = tmp_path / "control"
    subprocess.run(["git", "clone", "--quiet", str(root), str(control)], check=True)
    class Local:
        def execute_command(self, handle, command, cwd, timeout):
            done = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            return SimpleNamespace(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)
    authority = runner._AttemptAuthority.__new__(runner._AttemptAuthority)
    authority.uncertain_attempt = None  # Constructor state for this focused fixture.
    authority.repository_contract = {}
    authority.backend, authority.handle = Local(), None
    authority.repo_path, authority.control_path = str(root), str(control)
    authority.control_checkpoints = [parent]
    authority.control_is_frozen_baseline = False
    authority.phase = "method"
    authority.artifact_paths = [str(external)]
    authority.execution_protocol = {}
    authority.dependency_parent_state = {}
    bindir, authority.log_path = runner._install_attempt_guard(
        str(tmp_path), str(root), str(root / "ar_result.json"), authoritative=True)
    authority.last_attempt_path = str(Path(bindir).parent / "last_attempt_id")
    script = "import pathlib,json;print(json.dumps({'method':1,'candidate_valid':True,'weights':pathlib.Path(" + repr(str(external)) + ").read_text()}))"
    class Request:
        def recv(self, size):
            return json.dumps({"cwd": str(control), "argv": [os.sys.executable, "-c", script]}).encode() + b"\n"
    response = authority._handle(Request())
    assert response["rc"] == 0
    assert json.loads(response["stdout"])["weights"] == "parent-A"
    assert external.read_bytes() == b"candidate-B"
    record = runner._attempt_records(authority.log_path)[response["attempt_id"]]
    saved = validate_checkpoint(record["checkpoint"])
    assert (Path(record["checkpoint"]) / saved["external"][str(external)]).read_bytes() == b"parent-A"


@pytest.mark.parametrize("deny_readable_copy", [False, True])
def test_private_sandbox_artifact_remains_recoverable_by_host(
        tmp_path, monkeypatch, deny_readable_copy):
    import builtins
    import shlex
    from types import SimpleNamespace
    root = repository(tmp_path)
    external = tmp_path / "model.bin"
    external.write_bytes(b"trained-model")
    external.chmod(0o600)
    original_open = builtins.open

    def host_open(path, mode="r", *args, **kwargs):
        # The controller does not share the sandbox file owner's UID.
        if "external-files" in Path(path).parts and "r" in mode:
            if not Path(path).stat().st_mode & 0o004:
                raise PermissionError("sandbox-owned runtime output is not host-readable")
        return original_open(path, mode, *args, **kwargs)

    class Local:
        def execute_command(self, handle, command, cwd, timeout):
            if deny_readable_copy and shlex.split(command)[0] == "chmod":
                return SimpleNamespace(exit_code=1)
            done = subprocess.run(command, shell=True, cwd=cwd, timeout=timeout,
                                  capture_output=True, text=True)
            return SimpleNamespace(exit_code=done.returncode)

    monkeypatch.setattr(runner, "_prepare_checkpoint_handoff", lambda *_: None)
    monkeypatch.setattr(builtins, "open", host_open)
    if deny_readable_copy:
        with pytest.raises(ValueError, match="cannot read preserved sandbox artifact"):
            snapshot_checkpoint(str(root), "attempt-model", [str(external)], backend=Local())
    else:
        saved = snapshot_checkpoint(str(root), "attempt-model", [str(external)], backend=Local())
        manifest = validate_checkpoint(saved)
        assert (Path(saved) / manifest["external"][str(external)]).read_bytes() == b"trained-model"
    assert external.stat().st_mode & 0o777 == 0o600
    assert external.read_bytes() == b"trained-model"


def test_r7_r8_memory_best_uses_same_protocol_and_improvement_margin(tmp_path):
    from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    store = SpecStore(tmp_path / "store")
    handler = MemoryAgentHandler.__new__(MemoryAgentHandler)
    handler.spec_store = store
    def snapshot(kind, obj):
        return store.put(kind=kind, session_id="s", key="snapshot-" + artifact_digest(obj).split(":")[1],
                         obj=obj, emitter="RunnerAgent")
    memory = None
    for value, protocol, expected in ((0.9, "p1", 0.9), (0.899, "p1", 0.9),
                                      (0.5, "p2", 0.9), (0.8, "p1", 0.8)):
        run = snapshot("run_record", {"metric_value": value})
        gen = {"run_record_snapshot_ref": run, "verdict": "new_best", "accepted": True,
               "baseline_metric": 1, "method_metric": value, "metric_name": "loss", "metric_unit": "nats",
               "lower_is_better": True, "measurement_evidence": {"value": value,
                   "protocol_fingerprint": protocol, "minimum_improvement": 0.01,
                   "protocol": {"kind": "official_single"}}}
        snapshot("gen_experiment_result", gen)
        memory, *_ = handler._consolidate_gen(run, prior_research_memory=memory,
                                              verification_report={"verdict": "verified"})
        assert memory["best_metric"] == expected
        assert store.get(memory["best_result_ref"])["method_metric"] == expected
    assert "significance" in memory["verified_findings"][0]


def test_r6_revised_method_restores_legal_seed_without_granting_acceptance(tmp_path, monkeypatch):
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    root = repository(tmp_path)
    (root / "solve.py").write_text("version = 'alternative'\n")
    checkpoint = snapshot_checkpoint(str(root), "attempt-alternative")
    store = SpecStore(tmp_path / "store")
    candidate = {"checkpoint": checkpoint, "accepted": False, "method_metric": 1,
                 "off_method_observation": {"valid": True}}
    ref = store.put(kind="gen_experiment_result", session_id="s", obj=candidate,
                    key="snapshot-" + artifact_digest(candidate).split(":")[1], emitter="RunnerAgent")
    plan = ComponentPlan("revised-method", (node(),), ("gain",))
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    handler = RunnerAgentHandler.__new__(RunnerAgentHandler)
    handler.spec_store = store
    handler._active_component_context = {"s": {"mode": "integration", "ledger": ledger, "plan": plan}}
    monkeypatch.setattr(handler, "_latest_method", lambda _: {"reconciled_result_ref": ref})
    selected = handler._prior_worktree("s")
    assert selected == checkpoint
    target = tmp_path / "new-method"
    subprocess.run(["git", "clone", "--quiet", str(root), str(target)], check=True)
    runner._apply_prior_worktree(str(target), selected)
    assert "alternative" in (target / "solve.py").read_text()
    assert ledger.snapshot().get("integration_best") is None
    assert store.get(ref)["accepted"] is False


def test_r10_external_entrypoint_edit_is_visible_to_fidelity_review(tmp_path):
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import external_source_diff
    root = repository(tmp_path)
    external = tmp_path / "serving.py"
    external.write_text("mode = 'baseline'\n")
    baseline = snapshot_checkpoint(str(root), "attempt-base", [str(external)])
    external.write_text("mode = 'candidate'\n")
    candidate = snapshot_checkpoint(str(root), "attempt-candidate", [str(external)])
    diff = external_source_diff(candidate, baseline)
    assert "-mode = 'baseline'" in diff
    assert "+mode = 'candidate'" in diff
    assert str(external) in diff


def test_r11_prohibited_component_input_cannot_hide_behind_missing_reference_list():
    method, baseline, critic = grounded_review()
    baseline["repository_contract"]["execution_contract"] = {"sources": {"forbidden": ["solution"]}}
    method["components"][0]["inputs"] = ["copy solution/reference.py"]
    critic["component_checks"][0]["input_bindings"] = {"copy solution/reference.py": "baseline"}
    assert any("prohibited" in p["issue"] for p in _repository_review_problems(critic, baseline, method))


@pytest.mark.parametrize("directory", ["solution", "solutions", "reference", "oracle", "gold"])
def test_source_policy_has_no_implicit_answer_directory_blacklist(directory):
    assert permitted(directory + "/implementation.py")
    assert not permitted(directory + "/implementation.py", {"forbidden": [directory]})


def test_checkpoint_preserves_nested_repository_sources(tmp_path):
    root = repository(tmp_path)
    nested = repository(root, "nested")
    (nested / "solve.py").write_text("version = 'A'\n")
    (nested / "new.py").write_text("helper = 'A'\n")
    checkpoint = snapshot_checkpoint(str(root), "attempt-nested")
    manifest = validate_checkpoint(checkpoint)
    assert "nested/solve.py" in manifest["files"]
    assert "nested/new.py" in manifest["files"]
    (nested / "solve.py").write_text("version = 'B'\n")
    target = tmp_path / "restored"
    subprocess.run(["git", "clone", "--quiet", str(root), str(target)], check=True)
    runner._apply_prior_worktree(str(target), checkpoint)
    assert (target / "nested/solve.py").read_text() == "version = 'A'\n"
    assert (target / "nested/new.py").read_text() == "helper = 'A'\n"
