import pytest
import json
import subprocess
from pathlib import Path
from autoresearch.experiments.repo_improve.runtime_evidence import save_output, pointer
from autoresearch.experiments.repo_improve import agentic_improve as core


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("phase", ["baseline", "method"])
@pytest.mark.parametrize("artifact_kind", ["runtime", "source", "snapshot_source"])
def test_late_runtime_declaration_requires_files_in_measured_checkpoint(
        tmp_path, monkeypatch, sealed, absolute, phase, artifact_kind):
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("VALUE = 1\n")
    (repo / ".gitignore").write_text("weights.bin\nar_result.json\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], check=True)
    (repo / "weights.bin").write_bytes(b"measured weights")
    name = "main.py" if artifact_kind != "runtime" else "weights.bin"
    artifact = str(repo / name) if absolute else name
    state = core._tracked_repo_state(str(repo))["tracked_diff_hash"]
    commit = core._repo_commit(str(repo))
    saved = snapshot_checkpoint(str(repo), "attempt-measured", [artifact] if sealed else [])
    if artifact_kind == "snapshot_source" and absolute:
        artifact = str(Path(saved) / name)
    record = dict(attempt_id="attempt-measured", launched=True, finished=True, rc=0,
                  phase=phase, cwd=str(repo), repo_commit=commit, repo_commit_end=commit,
                  tracked_diff_hash_start=state, tracked_diff_hash_end=state, checkpoint=saved)
    record.update(save_output(tmp_path, "attempt-measured",
                              '{"baseline":0.5,"method":0.6,"baseline_valid":true,"candidate_valid":true}', ""))
    # The sandbox adds this declaration after the host sealed the measured attempt.
    result_path = repo / "ar_result.json"
    result_path.write_text(json.dumps({
        "baseline": 0.5, "method": 0.6, "attempt_id": "attempt-measured",
        "baseline_valid": True, "candidate_valid": True,
        "metric": {"name": "accuracy", "unit": "ratio", "lower_is_better": False},
        "artifact_paths": [artifact],
    }))
    monkeypatch.setattr(core, "_attempt_records", lambda _: {"attempt-measured": record})
    feedback = core._result_completion_request(
        str(result_path), execution_phase=phase, attempt_log="ledger")
    result = core._build_result_from_file(
        str(result_path), "runtime-artifacts", "accuracy", False, lambda *_: None, phase)
    result["repo_commit"] = commit
    result = core._finalize_relaxed_phase_result(
        result, phase, {"metric_name": "accuracy", "metric_unit": "ratio",
                        "metric_value": 0.5, "lower_is_better": False},
        attempt_log="ledger", repo_path=str(repo))
    outcome = result["baseline" if phase == "baseline" else "candidate"]
    assert outcome.metric_value == (0.5 if phase == "baseline" else 0.6)
    if sealed or artifact_kind != "runtime":
        assert feedback is None
        assert outcome.valid is True
        assert result["checkpoint"] == saved
    else:
        assert "declared runtime artifact is absent from measured checkpoint" in feedback
        assert "declared runtime artifact is absent from measured checkpoint" in result["error"]
        assert "weights.bin" in result["error"]
        assert outcome.valid is False
        assert not (Path(saved) / "weights.bin").exists()


@pytest.mark.parametrize("change", ["stage_only", "edit", "new_source", "missing",
                                    "tampered_checkpoint", "during_execution"])
def test_late_staging_requires_exact_measured_snapshot_bytes(tmp_path, change):
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], check=True)
    (repo / "helper.py").write_text("HELPER = 2\n")
    measured = core._tracked_repo_state(str(repo))["tracked_diff_hash"]
    commit = core._repo_commit(str(repo))
    saved = snapshot_checkpoint(str(repo), "attempt-measured")
    record = dict(attempt_id="attempt-measured", launched=True, finished=True, rc=0,
                  phase="method", cwd=str(repo), repo_commit=commit, repo_commit_end=commit,
                  tracked_diff_hash_start=measured, tracked_diff_hash_end=measured,
                  checkpoint=saved)
    record.update(save_output(tmp_path, "attempt-measured", '{"method":0.6}', ""))
    subprocess.run(["git", "-C", str(repo), "add", "helper.py"], check=True)
    assert core._tracked_repo_state(str(repo))["tracked_diff_hash"] != measured
    if change == "edit":
        (repo / "helper.py").write_text("HELPER = 3\n")
    elif change == "new_source":
        (repo / "new.py").write_text("UNMEASURED = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "new.py"], check=True)
    elif change == "missing":
        record.pop("checkpoint")
    elif change == "tampered_checkpoint":
        (Path(saved) / "helper.py").write_text("TAMPERED = True\n")
    elif change == "during_execution":
        record["tracked_diff_hash_start"] = "different-at-launch"
    result = {"_metric_attempt_id": "attempt-measured"}
    error = core._bind_component_evidence_attempt(
        result, {"attempt-measured": record}, str(repo), "method")
    if change == "stage_only":
        assert error is None
        assert result["_metric_attempt_id"] == "attempt-measured"
        assert result["checkpoint"] == saved
        assert record["tracked_diff_hash_start"] == record["tracked_diff_hash_end"] == measured
    else:
        assert "tracked diff fingerprint" in error
        assert not result.get("_metric_attempt_id")


def test_failed_execution_claiming_met_reports_host_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    record = dict(attempt_id="attempt-real", launched=True, finished=True, rc=1,
                  phase="method", cwd=str(tmp_path), repo_commit="commit",
                  repo_commit_end="commit", tracked_diff_hash_start="diff",
                  tracked_diff_hash_end="diff")
    record.update(save_output(tmp_path, "attempt-real", '{"failed_check_count":0}',
                              "NameError: name 'false' is not defined"))
    result = {"ablation_report": [{"component": "selection", "actual": 0,
               "verdict": "met", "evidence": {"source": "runtime_attempt",
               "ref": "attempt-real", "json_pointer": "/measurements/selection"}}]}

    error = core._bind_component_evidence_attempt(
        result, {"attempt-real": record}, str(tmp_path), "method")

    assert "exited with code 1" in error
    assert record["stderr_path"] in error
    assert "DEBUG" in error
    assert result["component_execution_started"] is True
    assert not result.get("_metric_attempt_id")
    assert not result.get("component_failed_attempt_id")
    assert not result["ablation_report"][0].get("_host_bound")


def test_cited_attempt_missing_from_host_ledger_reports_authority_cause(tmp_path):
    result = {"ablation_report": [{
        "component": "feature", "type": "indirect_gain", "actual": 0,
        "evidence": {
            "source": "runtime_attempt", "ref": "attempt-sandbox-local",
            "json_pointer": "/measurements/feature",
        },
    }]}

    error = core._bind_component_evidence_attempt(
        result, {}, str(tmp_path), "method")

    assert "attempt-sandbox-local" in error
    assert "not present in the host authority ledger" in error
    assert "Do not unset, replace, or redirect" in error
    assert "sandbox-local attempt log is not execution evidence" in error
    assert not result.get("_metric_attempt_id")


def test_mistyped_attempt_id_reports_exact_host_receipt_without_binding(tmp_path):
    result = {"_metric_attempt_id": "attempt-123-abc", "ablation_report": []}
    records = {"attempt-123-abcdef": {"phase": "method", "finished": True}}

    error = core._bind_component_evidence_attempt(result, records, str(tmp_path), "method")

    assert "attempt-123-abc" in error
    assert "attempt-123-abcdef" in error
    assert "do not infer identity" in error
    assert not result.get("_metric_attempt_id")


@pytest.mark.parametrize("mismatch", [None, "cwd", "repo_commit_end",
                                      "tracked_diff_hash_end", "execution_protocol",
                                      "phase", "missing_identity", "rc", "missing_ref",
                                      "direct_gain", "tampered_output"])
def test_separate_component_probe_binds_only_same_candidate_state(tmp_path, mismatch):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    from pathlib import Path

    identity = dict(launched=True, finished=True, rc=0, cwd="/candidate",
                    phase="method", repo_commit="commit", repo_commit_end="commit",
                    tracked_diff_hash_start="diff", tracked_diff_hash_end="diff",
                    execution_protocol={})
    candidate = dict(identity, attempt_id="attempt-eval")
    candidate.update(save_output(tmp_path, "attempt-eval", '{"method":0.6}', ""))
    probe = dict(identity, attempt_id="attempt-probe")
    probe.update(save_output(tmp_path, "attempt-probe", '{"measurements":{"init":0}}', ""))
    report = [{"component": "init", "type": "indirect_gain", "actual": 0,
               "evidence": {"source": "runtime_attempt", "ref": "attempt-probe",
                            "json_pointer": "/measurements/init"}}]
    records = {"attempt-eval": candidate, "attempt-probe": probe}
    if mismatch == "missing_identity":
        probe.pop("execution_protocol")
    elif mismatch == "missing_ref":
        records.pop("attempt-probe")
    elif mismatch == "direct_gain":
        report[0]["type"] = "direct_gain"
    elif mismatch == "tampered_output":
        Path(probe["stdout_path"]).write_text('{"measurements":{"init":1}}')
    elif mismatch:
        probe[mismatch] = 124 if mismatch == "rc" else "different"
    if mismatch:
        with pytest.raises(ValueError) as exc:
            bind_component_measurements(report, candidate, records)
        if mismatch not in {"rc", "tampered_output"}:
            message = str(exc.value)
            assert "component init" in message
            assert "attempt-probe" in message and "attempt-eval" in message
            assert "not proof that the candidate command never ran" in message
    else:
        bound = bind_component_measurements(report, candidate, records)[0]
        assert bound["actual"] == 0
        assert bound["_host_bound"] is True
        assert bound["evidence"]["ref"] == "attempt-probe"
        assert bound["evidence"]["sha256"] == probe["stdout_sha256"]
        assert Path(candidate["stdout_path"]).read_text() == '{"method":0.6}'


def test_nested_brokered_attempt_rejected_before_socket_wait(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log = core._install_attempt_guard(str(tmp_path), str(repo), str(repo / "ar_result.json"))
    marker = repo / "executed"
    env = dict(os.environ, AUTORESEARCH_ATTEMPT_ACTIVE="1",
               AUTORESEARCH_ATTEMPT_SOCKET=str(tmp_path / "broker.sock"),
               AUTORESEARCH_ATTEMPT_LOG=log)
    # A busy serial broker cannot accept its own child's request. The launcher
    # must reject nesting without sending anything to that broker.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as broker:
        broker.bind(env["AUTORESEARCH_ATTEMPT_SOCKET"])
        broker.listen(1)
        result = subprocess.run(
            [str(Path(bindir) / "ar-attempt"), sys.executable, "-c",
             f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            env=env, capture_output=True, text=True, timeout=2)
        broker.settimeout(0.05)
        with pytest.raises(socket.timeout):
            broker.accept()
    assert result.returncode == 126
    assert "nested ar-attempt" in result.stderr
    assert "separate top-level" in result.stderr
    assert not marker.exists()
    assert core._attempt_count(log) == 0


def test_nested_attempt_counts_outer_failure_and_allows_next_top_level_run(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log = core._install_attempt_guard(str(tmp_path), str(repo), str(repo / "ar_result.json"))
    launcher = str(Path(bindir) / "ar-attempt")
    env = dict(os.environ, AUTORESEARCH_ATTEMPT_LOG=log,
               AUTORESEARCH_EXECUTION_PHASE="method", AUTORESEARCH_AGENTIC_VARIANT_CAP="2")
    for key in ("AUTORESEARCH_ATTEMPT_SOCKET", "AUTORESEARCH_ATTEMPT_ACTIVE"):
        env.pop(key, None)
    result = subprocess.run([launcher, launcher, "true"], env=env,
                            capture_output=True, text=True, timeout=2)
    assert result.returncode == 126
    assert "nested ar-attempt" in result.stderr
    assert core._attempt_count(log) == 1
    result = subprocess.run([launcher, "true"], env=env,
                            capture_output=True, text=True, timeout=2)
    assert result.returncode == 0
    assert core._attempt_count(log) == 2


def test_missing_result_field_identifies_json_pointer_not_disk_path():
    with pytest.raises(ValueError) as exc:
        pointer({"value": 1.0}, "/control")
    assert "JSON field pointer" in str(exc.value)
    assert "not a filesystem path" in str(exc.value)
    assert "/control" in str(exc.value)


def test_malformed_json_reports_syntax_location_without_rewriting_output(tmp_path):
    from pathlib import Path
    from autoresearch.experiments.repo_improve.runtime_evidence import read_output

    raw = '{""phase":"candidate","method":0.5,"candidate_valid":true"}\n'
    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    record.update(save_output(tmp_path, "candidate", raw, ""))
    with pytest.raises(ValueError, match="one unambiguous JSON result") as exc:
        read_output(record)
    assert "line 1 column" in str(exc.value)
    assert "JSON serializer" in str(exc.value)
    assert "do not edit saved output" in str(exc.value)
    assert Path(record["stdout_path"]).read_text() == raw


@pytest.mark.parametrize("gate,value", [("candidate_valid", False), ("baseline_valid", False),
                                       ("candidate_valid", True), ("baseline_valid", True)])
@pytest.mark.parametrize("role", ["direct_gain", "indirect_gain"])
def test_explicitly_invalid_parent_is_not_a_performance_control(tmp_path, gate, value, role):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    state = {"repo_commit": "commit", "tracked_diff_hash_start": "hash"}
    parent = {"attempt_id": "parent", "launched": True, "finished": True, "rc": 0,
              "cwd": str(tmp_path), **state, "repo_commit_end": "commit",
              "tracked_diff_hash_end": "hash", "execution_protocol": {}}
    parent.update(save_output(tmp_path, "parent", json.dumps({"method": 148, gate: value}), ""))
    candidate = {**parent, "attempt_id": "candidate", "dependency_parent_state": state,
                 "dependency_parent_cwd": str(tmp_path)}
    candidate.update(save_output(tmp_path, "candidate", json.dumps({"method": 1119}), ""))
    report = [{"component": "change", "type": role, "parent_metric": 148,
               "parent_evidence": {"ref": "parent", "json_pointer": "/method"}}]
    bound = bind_component_measurements(report, candidate, {"parent": parent})[0]["parent_evidence"]
    assert bound["host_bound"]
    assert bound.get("failed_gates", []) == ([] if value else [gate])


def test_missing_control_binding_does_not_blame_successful_candidate(tmp_path):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    record.update(save_output(tmp_path, "candidate", json.dumps({"method": 0.6}), ""))
    report = [{"component": "change", "actual": 0.6,
               "evidence": {"json_pointer": "/method"}, "parent_metric": 0.5,
               "parent_evidence": {"ref": "frozen baseline file description"}}]
    with pytest.raises(ValueError, match="parent/control evidence is invalid") as exc:
        bind_component_measurements(report, record, {"candidate": record})
    assert "candidate execution was read successfully" in str(exc.value)
    # A failed candidate still receives the original execution failure diagnostic.
    record["rc"] = 1
    with pytest.raises(ValueError, match="measurement requires a successfully finished"):
        bind_component_measurements(report, record, {"candidate": record})


@pytest.mark.parametrize("conflict", [None, "value", "gate", "type"])
def test_repeated_runtime_result_requires_identical_payloads(tmp_path, conflict):
    from pathlib import Path
    from autoresearch.experiments.repo_improve.runtime_evidence import read_output

    first = {"measurements": {"probe": 0}, "candidate_valid": True}
    second = {"candidate_valid": True, "measurements": {"probe": 0}}
    if conflict == "value":
        second["measurements"]["probe"] = 1
    elif conflict == "gate":
        second["candidate_valid"] = False
    elif conflict == "type":
        second["measurements"]["probe"] = False
    raw = "progress log\n" + json.dumps(first, indent=2) + "\n" + json.dumps(second) + "\n"
    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    record.update(save_output(tmp_path, "candidate", raw, ""))
    if conflict:
        with pytest.raises(ValueError, match="one unambiguous JSON result"):
            read_output(record)
    else:
        assert read_output(record) == first
        record["rc"] = 1
        with pytest.raises(ValueError, match="successfully finished"):
            read_output(record)
        record["rc"] = 0
        record["stdout_sha256"] = "incorrect"
        with pytest.raises(ValueError, match="content changed"):
            read_output(record)
    assert Path(record["stdout_path"]).read_text() == raw


def test_stale_control_reference_reports_current_control_ids_without_accepting(tmp_path):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    candidate = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0,
                 "cwd": "/candidate", "dependency_parent_cwd": "/control"}
    candidate.update(save_output(tmp_path, "candidate", json.dumps({"method": 0.6}), ""))
    records = {
        "candidate": candidate,
        "new-control": {"attempt_id": "new-control", "cwd": "/control",
                        "launched": True, "finished": True, "rc": 0},
        "failed-control": {"attempt_id": "failed-control", "cwd": "/control",
                           "launched": True, "finished": True, "rc": 124},
        "unrelated-control": {"attempt_id": "unrelated-control", "cwd": "/other",
                              "launched": True, "finished": True, "rc": 0},
    }
    report = [{"component": "change", "actual": 0.6,
               "evidence": {"json_pointer": "/method"}, "parent_metric": 0.5,
               "parent_evidence": {"ref": "old-control#/baseline"}}]
    with pytest.raises(ValueError, match="parent/control evidence is invalid") as exc:
        bind_component_measurements(report, candidate, records)
    message = str(exc.value)
    assert "old-control#/baseline" in message
    assert "new-control" in message
    assert "failed-control" not in message and "unrelated-control" not in message
    assert "not automatically accepted" in message
    assert "not registered in the current sandbox ledger" in message
    assert "does not establish that the historical run failed" in message
    assert report[0]["parent_evidence"]["ref"] == "old-control#/baseline"


def test_missing_component_pointer_requests_rebinding_saved_field_without_guessing(tmp_path):
    from pathlib import Path
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    payload = {"test_accuracy": 0.9, "validation_accuracy": 0.8}
    record.update(save_output(tmp_path, "candidate", json.dumps(payload), ""))
    report = [{"component": "train", "actual": 0.9, "evidence": {"ref": "candidate"}}]
    with pytest.raises(ValueError, match="available top-level fields") as exc:
        bind_component_measurements(report, record, {})
    assert "test_accuracy" in str(exc.value)
    assert "validation_accuracy" in str(exc.value)
    assert "evidence.json_pointer" in str(exc.value)
    assert "do not rewrite saved output" in str(exc.value)
    # The host does not pick a same-valued field or silently pass the result.
    report[0]["evidence"]["json_pointer"] = "/test_accuracy"
    assert bind_component_measurements(report, record, {})[0]["actual"] == 0.9
    report[0]["evidence"]["json_pointer"] = "/validation_accuracy"
    with pytest.raises(ValueError, match="scalar disagrees"):
        bind_component_measurements(report, record, {})
    assert json.loads(Path(record["stdout_path"]).read_text()) == payload


@pytest.mark.parametrize("saved,reported,accepted", [
    (0.006316666666666748, 0.006316666666666858, True),
    (0.9, 0.9000000000000001, True),
    (1e-71, 1.0000000000000002e-71, True),
    (0.006316666666666748, 0.006317, False),
    (0.9, 0.900001, False),
    (0.0, 1e-71, False),
    (1e-71, 0.0, False),
    (1e-71, 1e-20, False),
    (100, 101, False),
])
def test_component_scalar_roundoff_preserves_host_value(tmp_path, saved, reported, accepted):
    from pathlib import Path
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    raw_output = json.dumps({"measurements": {"change": saved}})
    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    record.update(save_output(tmp_path, "candidate", raw_output, ""))
    report = [{"component": "change", "actual": reported,
               "evidence": {"source": "runtime_attempt", "ref": "candidate",
                            "json_pointer": "/measurements/change"}}]
    if accepted:
        bound = bind_component_measurements(report, record, {})[0]
        assert bound["actual"] == saved
        assert bound["_host_bound"] is True
    else:
        with pytest.raises(ValueError, match="scalar disagrees"):
            bind_component_measurements(report, record, {})
    assert Path(record["stdout_path"]).read_text() == raw_output
    assert report[0]["actual"] == reported


def test_scalar_mismatch_identifies_selected_field_without_guessing(tmp_path):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    payload = {"measurements": {"change": 0}, "latency": 12.5}
    record.update(save_output(tmp_path, "candidate", json.dumps(payload), ""))
    report = [{"component": "change", "actual": 12.5, "evidence": {"ref": "candidate"}}]
    with pytest.raises(ValueError, match="scalar disagrees") as exc:
        bind_component_measurements(report, record, {})
    assert "/measurements/change" in str(exc.value)
    assert "saved=0.0, reported=12.5" in str(exc.value)
    assert "do not substitute" in str(exc.value)
    report[0]["evidence"]["json_pointer"] = "/latency"
    assert bind_component_measurements(report, record, {})[0]["actual"] == 12.5


def test_control_binding_reports_exact_required_checkout_without_relaxing_identity(tmp_path):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    parent = {"attempt_id": "parent", "launched": True, "finished": True, "rc": 0,
              "repo_commit": "commit", "repo_commit_end": "commit",
              "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff",
              "cwd": "/candidate/subdir", "execution_protocol": {}}
    parent.update(save_output(tmp_path, "parent", json.dumps({"method": 2}), ""))
    candidate = {**parent, "attempt_id": "candidate",
                 "dependency_parent_state": {"repo_commit": "commit", "tracked_diff_hash_start": "diff"},
                 "dependency_parent_cwd": "/control/subdir"}
    candidate.update(save_output(tmp_path, "candidate", json.dumps({"measurements": {"change": 1}}), ""))
    report = [{"component": "change", "actual": 1, "evidence": {"ref": "candidate"},
               "parent_metric": 2, "parent_evidence": {"ref": "parent"}}]
    with pytest.raises(ValueError, match="exact dependency parent") as exc:
        bind_component_measurements(report, candidate, {"parent": parent})
    assert "expected cwd='/control/subdir'" in str(exc.value)
    assert "observed cwd='/candidate/subdir'" in str(exc.value)
    parent["cwd"] = "/control/subdir"
    assert bind_component_measurements(report, candidate, {"parent": parent})[0]["parent_evidence"]["host_bound"]


@pytest.mark.parametrize('feasible,failed_gate,allowed', [
    (False, 'candidate_valid', True), (True, 'candidate_valid', True),
    (True, 'baseline_valid', False), (None, 'candidate_valid', False),
])
def test_control_binding_marks_only_host_owned_ineligible_frozen_baseline(
        tmp_path, feasible, failed_gate, allowed):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    outcome = {"feasible": feasible,
               "quality_observations": {"accuracy": 0.79},
               "reason": "official eligibility threshold was not met"}
    parent = {
        "attempt_id": "parent", "launched": True, "finished": True, "rc": 0,
        "repo_commit": "base", "repo_commit_end": "base",
        "tracked_diff_hash_start": "parent", "tracked_diff_hash_end": "parent",
        "cwd": "/control", "execution_protocol": {},
        "frozen_baseline_control": True, "frozen_baseline_outcome": outcome,
    }
    parent.update(save_output(tmp_path, "parent", json.dumps({
        "method": 17988, failed_gate: False,
    }), ""))
    candidate = {
        **parent, "attempt_id": "candidate", "cwd": "/candidate",
        "frozen_baseline_control": False, "frozen_baseline_outcome": None,
        "dependency_parent_state": {
            "repo_commit": "base", "tracked_diff_hash_start": "parent"},
        "dependency_parent_cwd": "/control",
    }
    candidate.update(save_output(tmp_path, "candidate", json.dumps({
        "method": 14913, "candidate_valid": True,
        "measurements": {"change": 14913},
    }), ""))
    report = [{
        "component": "change", "actual": 14913,
        "evidence": {"ref": "candidate"}, "parent_metric": 17988,
        "parent_evidence": {"ref": "parent"},
    }]
    bound = bind_component_measurements(report, candidate, {"parent": parent})
    evidence = bound[0]['parent_evidence']
    assert evidence["failed_gates"] == [failed_gate]
    assert bool(evidence.get("reproduced_ineligible_baseline")) is allowed
    from autoresearch.experiments.repo_improve.component_plan import (
        build_component_plan, evaluate_component_contract)
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    node = build_component_plan({'components': [_component('change')]}).node('change')
    result = evaluate_component_contract(node, parent_metric=None, candidate_metric=14913,
        lower_is_better=True, ablation_report=bound)
    assert result['passed'] is allowed

    parent.pop("frozen_baseline_control")
    evidence = bind_component_measurements(report, candidate, {"parent": parent})[0][
        "parent_evidence"]
    assert "reproduced_ineligible_baseline" not in evidence


@pytest.mark.parametrize("parent_value,candidate_value,lower", [(10.0, 2.0, True), (2.0, 10.0, False)])
def test_direct_gain_keeps_raw_evidence_separate_from_derived_acceptance(
    tmp_path, parent_value, candidate_value, lower,
):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.runtime_evidence import (
        bind_component_measurements, measured_result,
    )
    from autoresearch.experiments.repo_improve.component_plan import evaluate_component_contract

    parent = {"attempt_id": "parent", "launched": True, "finished": True, "rc": 0,
              "repo_commit": "base", "repo_commit_end": "base",
              "tracked_diff_hash_start": "parent", "tracked_diff_hash_end": "parent",
              "cwd": "/control", "execution_protocol": {}}
    parent.update(save_output(tmp_path, "parent", json.dumps({"method": parent_value}), ""))
    candidate = {**parent, "attempt_id": "candidate", "cwd": "/candidate",
                 "dependency_parent_state": {"repo_commit": "base", "tracked_diff_hash_start": "parent"},
                 "dependency_parent_cwd": "/control"}
    candidate.update(save_output(tmp_path, "candidate", json.dumps({
        "method": candidate_value, "candidate_valid": True,
        "measurements": {"change": candidate_value},
    }), ""))
    report = [{"component": "change", "type": "direct_gain", "actual": 8.0,
               "evidence": {"ref": "candidate"}, "parent_metric": parent_value,
               "parent_evidence": {"ref": "parent"}}]
    with pytest.raises(ValueError, match="ablation_report.actual"):
        bind_component_measurements(report, candidate, {"parent": parent})
    report[0]["actual"] = candidate_value
    bound = bind_component_measurements(report, candidate, {"parent": parent})
    measured = measured_result(candidate, {}, "method")
    node = SimpleNamespace(role="direct_gain", component_id="change", name="change",
                           acceptance_check="signed_metric_improvement > 0")
    result = evaluate_component_contract(node, parent_metric=None,
        candidate_metric=measured["value"], lower_is_better=lower, ablation_report=bound)
    assert measured["valid"] and result["passed"]
    assert bound[0]["actual"] == candidate_value
    assert result["actual"] == 8.0


def test_component_numeric_pointer_accepts_explicit_fragment_not_nested_guess(tmp_path):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements
    record = {"attempt_id": "candidate", "launched": True, "finished": True, "rc": 0}
    payload = {"measurements": {"transform": {"residual": 0.01, "count": 50}}}
    record.update(save_output(tmp_path, "candidate", json.dumps(payload), ""))
    report = [{"component": "transform", "actual": 0.01,
               "evidence": {"ref": "candidate#/measurements/transform/residual"}}]
    bound = bind_component_measurements(report, record, {})
    assert bound[0]["evidence"]["json_pointer"] == "/measurements/transform/residual"
    report[0]["evidence"] = {"ref": "candidate"}
    with pytest.raises(ValueError, match="selects dict") as exc:
        bind_component_measurements(report, record, {})
    assert "residual" in str(exc.value)
    assert "do not guess" in str(exc.value)


@pytest.mark.parametrize("case", ["valid", "explicit_pointer", "unknown", "failed",
                                  "wrong_cwd", "changed_output", "missing_field", "wrong_scalar"])
def test_parent_reference_fragment_uses_same_binding_rules_as_candidate(tmp_path, case):
    from pathlib import Path
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    parent = {"attempt_id": "attempt-parent", "launched": True, "finished": True, "rc": 0,
              "repo_commit": "base", "repo_commit_end": "base",
              "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff",
              "cwd": "/control", "execution_protocol": {}}
    parent.update(save_output(tmp_path, "attempt-parent", json.dumps({"baseline": 2}), ""))
    candidate = {**parent, "attempt_id": "attempt-candidate", "cwd": "/candidate",
                 "dependency_parent_state": {"repo_commit": "base", "tracked_diff_hash_start": "diff"},
                 "dependency_parent_cwd": "/control"}
    candidate.update(save_output(tmp_path, "attempt-candidate", json.dumps({"method": 1}), ""))
    ref = {"ref": "attempt-parent#/baseline"}
    if case == "explicit_pointer":
        ref = {"ref": "attempt-parent#/ignored", "json_pointer": "/baseline"}
    elif case == "unknown":
        ref["ref"] = "attempt-missing#/baseline"
    elif case == "failed":
        parent["rc"] = 1
    elif case == "wrong_cwd":
        parent["cwd"] = "/candidate"
    elif case == "changed_output":
        Path(parent["stdout_path"]).write_text('{"baseline":2,"altered":true}')
    elif case == "missing_field":
        ref["ref"] = "attempt-parent#/not_measured"
    report = [{"component": "change", "actual": 1,
               "evidence": {"ref": "attempt-candidate#/method"},
               "parent_metric": 2, "parent_evidence": ref}]
    if case == "wrong_scalar":
        report[0]["parent_metric"] = 3
    if case not in {"valid", "explicit_pointer"}:
        with pytest.raises(ValueError) as exc:
            bind_component_measurements(report, candidate, {"attempt-parent": parent})
        if case == "wrong_scalar":
            message = str(exc.value)
            assert "attempt-parent" in message and "/baseline" in message
            assert "saved=2.0" in message and "reported=3" in message
            assert "frozen baseline" in message and "do not rewrite saved output" in message
            assert report[0]["parent_metric"] == 3
            assert json.loads(Path(parent["stdout_path"]).read_text()) == {"baseline": 2}
    else:
        result = bind_component_measurements(report, candidate, {"attempt-parent": parent})[0]
        assert result["actual"] == 1 and result["parent_metric"] == 2
        assert result["parent_evidence"]["ref"] == "attempt-parent"
        assert result["parent_evidence"]["host_bound"]
        assert result["parent_evidence"]["sha256"] == parent["stdout_sha256"]


@pytest.mark.parametrize("pointer_path,protocol,accepted", [
    ("/failed_check_count", {}, False),
    ("/failed_check_count", {"metric_pointer": "/latency"}, False),
    ("/method", {}, True),
    ("/baseline", {}, True),
    ("/latency", {"metric_pointer": "/latency"}, True),
    ("/method", {"metric_pointer": "/latency"}, False),
])
def test_parent_comparison_binds_primary_quantity_not_intermediate_check(
    tmp_path, pointer_path, protocol, accepted,
):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    state = {"repo_commit": "base", "repo_commit_end": "base",
             "tracked_diff_hash_start": "parent", "tracked_diff_hash_end": "parent",
             "execution_protocol": protocol}
    # Equal numbers are still different quantities; a check count is not latency.
    parent = dict(state, attempt_id="parent", phase="method", launched=True,
                  finished=True, rc=0, cwd="/control")
    parent.update(save_output(tmp_path, "parent", json.dumps({
        "method": 2, "baseline": 2, "latency": 2, "failed_check_count": 2,
    }), ""))
    candidate = dict(state, attempt_id="candidate", phase="method", launched=True,
                     finished=True, rc=0, cwd="/candidate",
                     dependency_parent_cwd="/control", dependency_parent_state=state)
    candidate.update(save_output(tmp_path, "candidate", json.dumps({"method": 1}), ""))
    report = [{"component": "change", "actual": 1,
               "evidence": {"ref": "candidate", "json_pointer": "/method"},
               "parent_metric": 2,
               "parent_evidence": {"ref": "parent", "json_pointer": pointer_path}}]
    if accepted:
        bound = bind_component_measurements(report, candidate, {"parent": parent})
        assert bound[0]["parent_metric"] == 2
        assert bound[0]["parent_evidence"]["host_bound"] is True
    else:
        with pytest.raises(ValueError, match="primary performance quantity"):
            bind_component_measurements(report, candidate, {"parent": parent})
        assert "host_bound" not in report[0]["parent_evidence"]
    assert json.loads(Path(parent["stdout_path"]).read_text())["failed_check_count"] == 2


@pytest.mark.parametrize("phase,wrong_pointer", [
    ("method", "/baseline"), ("baseline", "/method"),
])
def test_parent_alias_cannot_substitute_another_treatment_scalar(tmp_path, phase, wrong_pointer):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    state = {"repo_commit": "base", "repo_commit_end": "base",
             "tracked_diff_hash_start": "parent", "tracked_diff_hash_end": "parent",
             "execution_protocol": {}}
    primary = "/baseline" if phase == "baseline" else "/method"
    parent = dict(state, attempt_id="parent", phase=phase, launched=True,
                  finished=True, rc=0, cwd="/control")
    parent.update(save_output(tmp_path, "parent", json.dumps({
        primary[1:]: 2, wrong_pointer[1:]: 10,
    }), ""))
    candidate = dict(state, attempt_id="candidate", phase="method", launched=True,
                     finished=True, rc=0, cwd="/candidate",
                     dependency_parent_cwd="/control", dependency_parent_state=state)
    candidate.update(save_output(tmp_path, "candidate", json.dumps({"method": 1}), ""))
    report = [{"component": "change", "actual": 1,
               "evidence": {"ref": "candidate", "json_pointer": "/method"},
               "parent_metric": 10,
               "parent_evidence": {"ref": "parent", "json_pointer": wrong_pointer}}]
    with pytest.raises(ValueError, match="primary performance quantity"):
        bind_component_measurements(report, candidate, {"parent": parent})
    report[0]["parent_metric"] = 2
    report[0]["parent_evidence"]["json_pointer"] = primary
    assert bind_component_measurements(report, candidate, {"parent": parent})[0]["parent_metric"] == 2
    assert json.loads(Path(parent["stdout_path"]).read_text())[wrong_pointer[1:]] == 10


@pytest.mark.parametrize("case", ["valid", "spec_uri", "unknown", "failed", "unfinished", "stale_diff", "wrong_phase"])
def test_explicit_attempt_id_must_match_successful_host_record(tmp_path, monkeypatch, case):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    record = {"attempt_id": "attempt-real", "launched": True, "finished": True, "rc": 0,
              "phase": "method", "repo_commit": "commit", "repo_commit_end": "commit",
              "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff",
              "cwd": str(tmp_path)}
    record.update(save_output(tmp_path, "attempt-real", json.dumps({"method": 0.1, "candidate_valid": True}), ""))
    explicit = "attempt-real"
    if case == "spec_uri":
        explicit = "spec://run_record/session/v1"
    elif case == "unknown":
        explicit = "attempt-not-in-ledger"
    elif case == "failed":
        record["rc"] = 1
    elif case == "unfinished":
        record["finished"] = False
    elif case == "stale_diff":
        record["tracked_diff_hash_end"] = "older-diff"
    elif case == "wrong_phase":
        record["phase"] = "baseline"
    result = {"_metric_attempt_id": explicit}
    core._bind_component_evidence_attempt(result, {"attempt-real": record}, str(tmp_path), "method")
    assert result.get("_metric_attempt_id") == ("attempt-real" if case == "valid" else None)
    assert bool(result.get("component_execution_started")) is (case not in {"spec_uri", "unknown", "wrong_phase"})


@pytest.mark.parametrize("case", ["malformed_output", "nonzero", "not_launched", "unknown"])
def test_failed_output_keeps_host_launch_for_accounting_not_result_acceptance(tmp_path, monkeypatch, case):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    record = {"attempt_id": "attempt-real", "launched": case != "not_launched",
              "finished": True, "rc": 2 if case == "nonzero" else 0,
              "phase": "method", "cwd": str(tmp_path), "repo_commit": "commit",
              "repo_commit_end": "commit", "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff"}
    record.update(save_output(tmp_path, "attempt-real", "not valid numeric output", ""))
    monkeypatch.setattr(core, "_attempt_records", lambda _: {} if case == "unknown" else {"attempt-real": record})
    result = {"component_execution_started": True, "_metric_attempt_id": "attempt-real",
              "_reported_metric": "latency", "_reported_metric_unit": "ms",
              "candidate": core.RunResult(metric_name="latency", metric_value=0.1, valid=True),
              "candidate_validity": True}
    out = core._finalize_relaxed_phase_result(result, "method", {
        "metric_name": "latency", "metric_unit": "ms", "metric_value": 1.0,
        "lower_is_better": True}, attempt_log="host-ledger", repo_path=str(tmp_path))
    assert bool(out.get("component_execution_started")) is (case in {"malformed_output", "nonzero"})
    assert out["candidate"].valid is False
    assert out["verdict"]["improved"] is False


def test_sandbox_cannot_claim_execution_started_without_host_ledger():
    out = core._finalize_relaxed_phase_result({"component_execution_started": True}, "method", {})
    assert not out.get("component_execution_started")


def test_unbound_finite_result_is_retained_without_success_or_best_eligibility():
    result = {"metric": "latency", "_reported_metric": "latency", "_reported_metric_unit": "ms",
              "candidate": core.RunResult(metric_name="latency", metric_value=0.1, valid=True),
              "candidate_validity": True}
    frozen = {"metric_name": "latency", "metric_unit": "ms", "metric_value": 1.0,
              "lower_is_better": True}
    out = core._finalize_relaxed_phase_result(result, "method", frozen)
    assert out["candidate"].metric_value == 0.1
    assert out["candidate"].valid is False
    assert out["verdict"]["improved"] is False
    assert out["candidate_admissibility"]["verdict"] == "insufficient_evidence"
    assert out["diagnosis"] == "attempt_accounting_violation"


@pytest.mark.parametrize("other_change", [
    {}, {"finished": False}, {"rc": 1}, {"phase": "baseline"},
    {"cwd": "/another-checkout"}, {"repo_commit_end": "other-commit"},
    {"tracked_diff_hash_end": "stale"}, {"execution_protocol": {"budget": 2}},
])
def test_stale_attempt_feedback_lists_matching_receipt_without_auto_binding(
        tmp_path, monkeypatch, other_change):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "current"})
    old = {"attempt_id": "attempt-old", "launched": True, "finished": True, "rc": 0,
           "phase": "method", "cwd": str(tmp_path), "repo_commit": "commit",
           "repo_commit_end": "commit", "tracked_diff_hash_start": "old",
           "tracked_diff_hash_end": "old", "execution_protocol": {"budget": 1}}
    newer = dict(old, attempt_id="attempt-current", tracked_diff_hash_start="current",
                 tracked_diff_hash_end="current")
    newer.update(other_change)
    result = {"_metric_attempt_id": "attempt-old"}

    error = core._bind_component_evidence_attempt(
        result, {"attempt-old": old, "attempt-current": newer}, str(tmp_path), "method")

    assert "tracked diff fingerprint" in error
    assert ("attempt-current" in error) is (not other_change)
    assert "diagnostic hints, not accepted evidence" in error
    assert "do not rewrite saved output" in error
    assert result.get("_metric_attempt_id") is None
    assert result.get("measurement_evidence") is None
    assert result["component_execution_started"] is True


@pytest.mark.parametrize("record_change", [
    {}, {"checkpoint": None}, {"checkpoint_error": "snapshot failed"},
    {"rc": 1}, {"tracked_diff_hash_end": "changed-during-run"},
])
def test_stale_attempt_feedback_identifies_restorable_history_without_accepting_it(
        tmp_path, monkeypatch, record_change):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "current"})
    checkpoint = str(tmp_path / "checkpoint-historical")
    record = {"attempt_id": "attempt-old", "launched": True, "finished": True, "rc": 0,
              "phase": "method", "cwd": str(tmp_path), "repo_commit": "commit",
              "repo_commit_end": "commit", "tracked_diff_hash_start": "old",
              "tracked_diff_hash_end": "old", "checkpoint": checkpoint}
    record.update(record_change)
    result = {"_metric_attempt_id": "attempt-old"}

    error = core._bind_component_evidence_attempt(
        result, {"attempt-old": record}, str(tmp_path), "method")

    assert "tracked diff fingerprint" in error
    assert (checkpoint in error) is (not record_change)
    if not record_change:
        assert "restore its exact code and required runtime artifacts" in error
        assert "normal evidence checks still apply" in error
    assert result.get("measurement_evidence") is None
    assert result["component_execution_started"] is True


def test_admissibility_preserves_binding_failure_for_next_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "after-control"})
    record = {"attempt_id": "attempt-real", "launched": True, "finished": True, "rc": 0,
              "phase": "method", "repo_commit": "commit", "repo_commit_end": "commit",
              "tracked_diff_hash_start": "measured", "tracked_diff_hash_end": "measured",
              "cwd": str(tmp_path)}
    monkeypatch.setattr(core, "_attempt_records", lambda _: {"attempt-real": record})
    result = {"_metric_attempt_id": "attempt-real", "_reported_metric": "latency",
              "_reported_metric_unit": "ms", "candidate_validity": True,
              "candidate": core.RunResult(metric_name="latency", metric_value=0.1, valid=True)}
    frozen = {"metric_name": "latency", "metric_unit": "ms", "metric_value": 1.0,
              "lower_is_better": True}
    out = core._finalize_relaxed_phase_result(
        result, "method", frozen, attempt_log="ledger", repo_path=str(tmp_path))
    assert out["candidate"].metric_value == 0.1
    assert out["candidate"].valid is False
    assert out["candidate_admissibility"]["verdict"] == "insufficient_evidence"
    assert out["candidate_admissibility"]["reason"] == out["error"]
    assert "tracked diff fingerprint" in out["candidate_admissibility"]["reason"]
    assert "finish editing/staging" in out["candidate_admissibility"]["reason"]


@pytest.mark.parametrize("inside_repo", [True, False])
def test_indirect_attempt_reports_wrong_working_directory(tmp_path, monkeypatch, inside_repo):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    record = {"attempt_id": "attempt-real", "launched": True, "finished": True, "rc": 0,
              "phase": "method", "repo_commit": "commit", "repo_commit_end": "commit",
              "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff",
              "cwd": str(tmp_path if inside_repo else tmp_path.parent)}
    record.update(save_output(tmp_path, "attempt-real", json.dumps({"measurements": {"feature": 0}}), ""))
    monkeypatch.setattr(core, "_attempt_records", lambda _: {"attempt-real": record})
    result = {"metric": "latency", "_reported_metric": "latency", "_reported_metric_unit": "ms",
              "candidate": core.RunResult(metric_name="latency", metric_value=None, valid=False),
              "candidate_validity": True, "diagnosis": "execution_failure",
              "ablation_report": [{"component": "feature", "type": "indirect_gain",
                                   "actual": 0, "expected": {"operator": "==", "value": 0},
                                   "evidence": {"source": "runtime_attempt",
                                                "ref": "attempt-real: check passed"}}]}
    frozen = {"metric_name": "latency", "metric_unit": "ms", "metric_value": 1.0,
              "lower_is_better": True}
    out = core._finalize_relaxed_phase_result(
        result, "method", frozen, attempt_log="ledger", repo_path=str(tmp_path))
    assert out["candidate"].metric_value is None
    assert out["verdict"]["improved"] is False
    if inside_repo:
        assert out["component_attempt_id"] == "attempt-real"
        assert not out.get("error")
    else:
        assert out["component_attempt_id"] is None
        assert "working directory" in out["error"]
        assert "cd" in out["error"]
        assert out["diagnosis"] == "attempt_accounting_violation"
