"""Regression receipts for cross-sandbox component and parent evidence handoff."""
import hashlib
import json
from pathlib import Path

import pytest

from autoresearch.experiments.repo_improve import agentic_improve as core
from autoresearch.experiments.repo_improve.runtime_evidence import (
    bind_component_measurements, read_output, save_output,
)


def snapshot(root, files, attempt):
    data = {"attempt_id": attempt, "files": {
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()
    }, "external": {}}
    raw = json.dumps(data, sort_keys=True).encode()
    path = root / ("checkpoint-" + hashlib.sha256(raw).hexdigest())
    path.mkdir()
    for name, value in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    (path / ".autoresearch-checkpoint.json").write_bytes(raw)
    return str(path)


def receipt(root, aid, cwd, payload, **kwargs):
    result = dict(attempt_id=aid, cwd=str(cwd), phase="method", launched=True,
                  finished=True, rc=0, repo_commit="commit", repo_commit_end="commit",
                  tracked_diff_hash_start="diff", tracked_diff_hash_end="diff",
                  execution_protocol={})
    result.update(save_output(root, aid, json.dumps(payload), ""))
    result.update(kwargs)
    return result


@pytest.mark.parametrize("same_session", [True, False])
@pytest.mark.parametrize("old_round", ["repo", "repo-1"])
def test_historical_primary_handoff_requires_current_validation_with_accurate_guidance(
        tmp_path, same_session, old_round):
    session = tmp_path / "session"
    repo = session / "repo-2"
    repo.mkdir(parents=True)
    old_repo = (session if same_session else tmp_path / "other-session") / old_round
    old_repo.mkdir(parents=True)
    old_authority = tmp_path / f".{old_round}-attempt-authority-old"
    current_authority = tmp_path / ".repo-2-attempt-authority-new"
    old_authority.mkdir()
    current_authority.mkdir()
    old = receipt(old_authority, "attempt-incumbent", old_repo,
                  {"method": 1.3, "candidate_valid": True})
    (old_authority / "attempts.jsonl").write_text(
        json.dumps({"event": "launch", "attempt_id": old["attempt_id"],
                    "phase": "method", "cwd": str(old_repo)}) + "\n"
        + json.dumps({"event": "finish", **old}) + "\n")
    log = current_authority / "attempts.jsonl"
    log.write_text("")
    path = repo / "ar_result.json"
    path.write_text(json.dumps({"baseline": 1.0, "method": 1.3, "candidate_valid": True,
        "attempt_id": "attempt-incumbent", "metric": {
            "name": "quality", "unit": "ratio", "lower_is_better": False}}))
    before = path.read_bytes()
    feedback = core._result_completion_request(str(path), execution_phase="method", attempt_log=str(log))
    assert feedback is not None
    if same_session:
        assert "HISTORICAL CANDIDATE NEEDS CURRENT VALIDATION" in feedback
        assert "not a misspelled attempt ID" in feedback
        assert "negative" in feedback
        parsed = core._build_result_from_file(str(path), "test", None, None, lambda *a: None, "method")
        records = core._evidence_records(str(log), parsed, str(repo))
        assert "outside the current repository" in core._bind_component_evidence_attempt(parsed, records, str(repo), "method")
    else:
        assert "HISTORICAL CANDIDATE" not in feedback
        assert "not an exact host record" in feedback
    assert path.read_bytes() == before
    assert read_output(old)["method"] == 1.3


@pytest.mark.parametrize("variant_attempt", [None, "attempt-better"])
def test_unbound_summary_cannot_replace_explicit_primary_measurement(tmp_path, variant_attempt):
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({"baseline": 1.0, "method": 0.99,
        "attempt_id": "attempt-selected", "candidate_valid": True,
        "metric": {"name": "quality", "unit": "ratio", "lower_is_better": False}}))
    (tmp_path / "analysis").mkdir()
    variant = {"variant": "three-run summary", "method": 1.08, "mean": 1.08,
               "candidate_valid": True, "attempt_ids": ["attempt-a", "attempt-b", "attempt-c"]}
    if variant_attempt:
        variant["attempt_id"] = variant_attempt
    (tmp_path / "analysis/variants.jsonl").write_text(json.dumps(variant) + "\n")
    result = core._build_result_from_file(str(path), "test", None, None, lambda *a: None, "method")
    assert result["candidate"].metric_value == 0.99
    assert result["_metric_attempt_id"] == "attempt-selected"
    assert json.loads((tmp_path / "analysis/variants.jsonl").read_text())["mean"] == 1.08


@pytest.mark.parametrize("phase", ["method", "baseline"])
@pytest.mark.parametrize("case", ["missing", "null", "explicit_null", "different",
                                  "boolean", "string", "nan", "object"])
def test_legacy_control_null_placeholder_keeps_explicit_parent_binding(tmp_path, phase, case):
    primary = "method" if phase == "method" else "baseline"
    measured = "baseline" if phase == "method" else "method"
    payload = {measured: 2}
    if case != "missing":
        payload[primary] = {"null": None, "explicit_null": None, "different": 8,
                            "boolean": False, "string": "2", "nan": float("nan"),
                            "object": {"value": None}}[case]
    protocol = {"metric_pointer": "/" + primary} if case == "explicit_null" else {}
    parent = receipt(tmp_path, "attempt-parent", tmp_path / "control", payload,
                     phase=phase, execution_protocol=protocol)
    candidate = receipt(tmp_path, "attempt-candidate", tmp_path / "repo", {"method": 1},
                        execution_protocol=protocol,
                        dependency_parent_cwd=parent["cwd"],
                        dependency_parent_state={"repo_commit": "commit",
                                                 "tracked_diff_hash_start": "diff"})
    report = [{"component": "update", "type": "direct_gain", "actual": 1,
               "evidence": {"source": "runtime_attempt", "ref": "attempt-candidate",
                            "json_pointer": "/method"},
               "parent_metric": 2, "parent_evidence": {"ref": "attempt-parent",
                                                       "json_pointer": "/" + measured}}]
    original = Path(parent["stdout_path"]).read_bytes()
    if case in {"missing", "null"}:
        bound = bind_component_measurements(report, candidate, {"attempt-parent": parent})[0]
        assert bound["parent_metric"] == 2
        assert bound["parent_evidence"]["host_bound"] is True
        assert bound["parent_evidence"]["json_pointer"] == "/" + measured
        assert bound["parent_evidence"]["sha256"] == parent["stdout_sha256"]
    else:
        with pytest.raises(ValueError):
            bind_component_measurements(report, candidate, {"attempt-parent": parent})
    assert Path(parent["stdout_path"]).read_bytes() == original
    assert "host_bound" not in report[0]["parent_evidence"]


@pytest.mark.parametrize("fault", [None, "changed_file", "tampered_checkpoint", "protocol", "failed"])
def test_historical_component_reads_its_own_receipt_only_if_preserved(tmp_path, fault):
    old = receipt(tmp_path, "attempt-old", tmp_path / "repo-1",
                  {"measurements": {"features": 0}})
    old["checkpoint"] = snapshot(tmp_path, {"features.py": "original"}, "attempt-old")
    new_files = {"features.py": "changed" if fault == "changed_file" else "original", "head.py": "new"}
    new = receipt(tmp_path, "attempt-new", tmp_path / "repo-2", {"method": .8})
    new["checkpoint"] = snapshot(tmp_path, new_files, "attempt-new")
    if fault == "tampered_checkpoint":
        (Path(old["checkpoint"]) / "features.py").write_text("tampered")
    elif fault == "protocol":
        old["execution_protocol"] = {"metric_pointer": "/other"}
    elif fault == "failed":
        old["rc"] = 124
    report = [{"component": "features", "type": "indirect_gain", "actual": 0,
               "evidence": {"source": "runtime_attempt", "ref": "attempt-old",
                            "json_pointer": "/measurements/features"}}]
    if fault:
        with pytest.raises(ValueError):
            bind_component_measurements(report, new, {"attempt-old": old})
    else:
        bound = bind_component_measurements(report, new, {"attempt-old": old})[0]
        assert bound["evidence"]["ref"] == "attempt-old"
        assert bound["_host_bound"] and bound["actual"] == 0
        assert read_output(new) == {"method": .8}


@pytest.mark.parametrize("fault", [None, "changed_parent", "missing_snapshot", "ambiguous_ref"])
def test_relocated_parent_requires_preserved_files_not_only_equal_diff(tmp_path, fault):
    control = tmp_path / "control-new"
    control.mkdir()
    (control / "model.py").write_text("wrong" if fault == "changed_parent" else "parent")
    old = receipt(tmp_path, "attempt-parent", tmp_path / "control-old", {"baseline": .5})
    if fault != "missing_snapshot":
        old["checkpoint"] = snapshot(tmp_path, {"model.py": "parent"}, "attempt-parent")
    new = receipt(tmp_path, "attempt-new", tmp_path / "candidate", {"method": .6},
                  dependency_parent_cwd=str(control),
                  dependency_parent_state={"repo_commit": "commit", "tracked_diff_hash_start": "diff"})
    report = [{"component": "update", "type": "direct_gain", "actual": .6,
               "evidence": {"json_pointer": "/method"}, "parent_metric": .5,
               "parent_evidence": {"ref": "attempt-parent (saved control)", "json_pointer": "/baseline"}}]
    if fault == "ambiguous_ref":
        report[0]["parent_evidence"]["ref"] += " or attempt-other"
    if fault:
        with pytest.raises(ValueError):
            bind_component_measurements(report, new, {"attempt-parent": old})
    else:
        bound = bind_component_measurements(report, new, {"attempt-parent": old})[0]
        assert bound["parent_evidence"]["ref"] == "attempt-parent"
        assert bound["parent_evidence"]["host_bound"]
        assert bound["parent_evidence"]["json_pointer"] == "/baseline"
        assert bind_component_measurements([bound], new, {"attempt-parent": old}) == [bound]


def test_historical_direct_component_keeps_its_own_parent(tmp_path):
    parent = receipt(tmp_path, "attempt-parent", tmp_path / "control-old", {"baseline": .5})
    old = receipt(tmp_path, "attempt-old", tmp_path / "repo-1", {"method": .6},
                  dependency_parent_cwd=str(tmp_path / "control-old"),
                  dependency_parent_state={"repo_commit": "commit", "tracked_diff_hash_start": "diff"})
    old["checkpoint"] = snapshot(tmp_path, {"model.py": "old component"}, "attempt-old")
    new = receipt(tmp_path, "attempt-new", tmp_path / "repo-2", {"method": .7},
                  dependency_parent_cwd=str(tmp_path / "control-final"),
                  dependency_parent_state={"repo_commit": "different", "tracked_diff_hash_start": "different"})
    new["checkpoint"] = snapshot(tmp_path, {"model.py": "old component", "head.py": "new"}, "attempt-new")
    report = [{"component": "update", "type": "direct_gain", "actual": .6,
               "evidence": {"source": "runtime_attempt", "ref": "attempt-old", "json_pointer": "/method"},
               "parent_metric": .5,
               "parent_evidence": {"ref": "attempt-parent", "json_pointer": "/baseline"}}]
    bound = bind_component_measurements(report, new, {"attempt-parent": parent, "attempt-old": old})[0]
    assert bound["actual"] == .6 and bound["parent_metric"] == .5
    assert bound["evidence"]["ref"] == "attempt-old"


@pytest.mark.parametrize("old_round", ["repo", "repo-1"])
def test_history_loading_is_session_scoped_and_does_not_change_attempt_count(tmp_path, old_round):
    session = tmp_path / "session"
    (session / "repo-2").mkdir(parents=True)
    current = tmp_path / ".repo-2-attempt-authority-new" / "attempts.jsonl"
    old = tmp_path / f".{old_round}-attempt-authority-old" / "attempts.jsonl"
    for path, events in [(current, [{"event": "launch", "attempt_id": "attempt-new",
                                    "cwd": str(session / "repo-2")}]),
                         (old, [{"event": "launch", "attempt_id": "attempt-old",
                                 "cwd": str(session / "control-1")},
                                {"event": "finish", "attempt_id": "attempt-old", "rc": 0},
                                {"event": "launch", "attempt_id": "attempt-other-task",
                                 "cwd": str(tmp_path / "other-session" / "repo-1")}])]:
        path.parent.mkdir()
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result = {"ablation_report": [{"evidence": {"ref": "attempt-old"},
                                   "parent_evidence": {"ref": "attempt-other-task"}}]}
    records = core._evidence_records(str(current), result, str(session / "repo-2"))
    assert set(records) == {"attempt-new", "attempt-old"}
    assert records["attempt-old"]["finished"] and records["attempt-old"]["rc"] == 0
    assert core._attempt_count(str(current)) == 1


def test_primary_measurement_survives_missing_component_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    record = receipt(tmp_path, "attempt-measured", tmp_path,
                     {"method": .6, "candidate_valid": True})
    monkeypatch.setattr(core, "_attempt_records", lambda _: {"attempt-measured": record})
    result = {"_metric_attempt_id": "attempt-measured", "_reported_metric": "accuracy",
              "_reported_metric_unit": "fraction", "candidate_validity": True,
              "candidate": core.RunResult(metric_name="accuracy", metric_value=.6, valid=True),
              "ablation_report": [{"component": "features", "type": "indirect_gain", "actual": 0,
                                   "evidence": {"source": "runtime_attempt", "ref": "attempt-measured"}}]}
    out = core._finalize_relaxed_phase_result(result, "method", {
        "metric_name": "accuracy", "metric_unit": "fraction", "metric_value": .5,
        "lower_is_better": False}, attempt_log=str(tmp_path / "ledger"), repo_path=str(tmp_path))
    assert out["measurement_evidence"]["value"] == .6
    assert out["candidate"].valid and out["candidate_admissibility"]["verdict"] == "pass"
    assert "missing /measurements/features" in out["component_evidence_error"]
    assert not out["ablation_report"][0].get("_host_bound")
    from autoresearch.agents.runner_agent.handler import _res_to_improve_json
    assert _res_to_improve_json(out)["component_evidence_error"] == out["component_evidence_error"]


@pytest.mark.parametrize("official", [False, True])
def test_component_official_claim_requires_frozen_command_and_gate(tmp_path, monkeypatch, official):
    monkeypatch.setattr(core, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(core, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    record = receipt(tmp_path, "attempt-local", tmp_path,
                     {"method": 7156, "candidate_valid": True})
    evaluation = {"metric_name": "total_params", "lower_is_better": True}
    if official:
        record["argv"] = ["python", "official_eval.py"]
        evaluation.update(argv=record["argv"], gate_pointers=["/candidate_valid"])
    monkeypatch.setattr(core, "_attempt_records", lambda _: {"attempt-local": record})
    result = {"_metric_attempt_id": "attempt-local", "_reported_metric": "total_params",
              "_reported_metric_unit": "count", "_reported_lower_is_better": True,
              "candidate_validity": True,
              "candidate": core.RunResult(metric_name="total_params", metric_value=7156,
                                          valid=True)}
    out = core._finalize_relaxed_phase_result(result, "method", {
        "metric_name": "total_params", "metric_unit": "count", "metric_value": 17988,
        "lower_is_better": True, "repository_contract": {"execution_contract": {
            "evaluation": evaluation}}},
        attempt_log=str(tmp_path / "ledger"), repo_path=str(tmp_path), component_stage=True)

    assert out["candidate"].valid is True
    assert out["measurement_evidence"]["value"] == 7156
    assert out["measurement_evidence"]["protocol"]["kind"] == (
        "official_single" if official else "component_local")
    assert out["measurement_evidence"]["official_gate_verified"] is official
    assert out["candidate_admissibility"]["verdict"] == (
        "pass" if official else "insufficient_evidence")
    if not official:
        assert "final eligibility remains unverified" in out["candidate_admissibility"]["reason"]


def test_later_timeout_does_not_destroy_prior_success_or_become_success(tmp_path):
    old = receipt(tmp_path, "attempt-completed", tmp_path, {"method": .34})
    timed_out = receipt(tmp_path, "attempt-timeout", tmp_path, {"method": .9}, rc=124)
    assert read_output(old)["method"] == .34
    with pytest.raises(ValueError, match="successfully finished"):
        read_output(timed_out)
