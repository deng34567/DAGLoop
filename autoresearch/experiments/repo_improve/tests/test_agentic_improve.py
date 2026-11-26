"""No-infer agentic improvement (2026-07-30): clone -> agent does everything -> read ar_result.json.

FakeBackend (no docker): its run_agent_task simulates the in-container agent by writing an
ar_result.json into the (bind-mounted) repo dir. Asserts the result maps to a run_repo_improve-
shaped dict and that every failure degrades to an honest invalid (never a fabricated number).
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import stat
import subprocess
import time
import shlex
from pathlib import Path

import pytest

from autoresearch.agents._reuse.openhands_backend import InfraError
from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve


@pytest.mark.parametrize("component_stage", [False, True])
def test_method_scope_is_independent_of_parent_worktree(tmp_path, monkeypatch, component_stage):
    from autoresearch.experiments.repo_improve import agentic_improve as core
    observed = []
    original = core._finalize_relaxed_phase_result

    def finalize(*args, **kwargs):
        observed.append(kwargs["component_stage"])
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_finalize_relaxed_phase_result", finalize)
    result = run_agentic_improve(
        "q", repo_url="https://x/r", workroot=str(tmp_path),
        backend=_FakeBackend(), clone_fn=_fake_clone,
        execution_phase="method", enforce_fairness=False,
        frozen_baseline={"metric_value": 1.8},
        component_parent_worktree=[], component_stage=component_stage)
    assert observed == [component_stage], result.get("error")


def test_whole_method_does_not_become_component_when_parent_is_restored(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as core
    observed = []
    original = core._finalize_relaxed_phase_result

    def finalize(*args, **kwargs):
        observed.append(kwargs["component_stage"])
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_finalize_relaxed_phase_result", finalize)
    result = run_agentic_improve(
        "q", repo_url="https://x/r", workroot=str(tmp_path),
        backend=_FakeBackend(), clone_fn=_fake_clone,
        execution_phase="method", enforce_fairness=False,
        frozen_baseline={"metric_value": 1.8})
    assert observed == [False], result.get("error")


@pytest.mark.parametrize("source_size", [64, 14052, 24001])
def test_attempt_command_source_capture_reads_without_executing_and_detects_changes(tmp_path, source_size):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            return SimpleNamespace(stdout=p.stdout)

    script = tmp_path / "check space.py"
    script.write_text("from pathlib import Path\nPath('executed').touch()\n")
    script.write_text(script.read_text() + "#" * (source_size - script.stat().st_size))
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    argv = ["env", "MODE=unit", "python3", script.name]
    before = authority._capture_command_sources(argv, str(tmp_path))
    assert before == [{"path": str(script), "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                       "source": script.read_text()[:24000], "truncated": source_size > 24000}]
    assert not (tmp_path / "executed").exists()
    script.write_text(script.read_text() + "# changed\n")
    after = authority._capture_command_sources(argv, str(tmp_path))
    assert after[0]["sha256"] != before[0]["sha256"]
    assert authority._capture_command_sources(["./compiled-check"], str(tmp_path)) == []


@pytest.mark.parametrize("suffix", [".go", ".rs", ".cpp", ".cu"])
def test_attempt_source_capture_binds_explicit_compiled_probe_sources(tmp_path, suffix):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            result = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert result.returncode == 0, result.stderr
            return SimpleNamespace(stdout=result.stdout)

    wrapper = tmp_path / "run.sh"
    wrapper.write_text("#!/bin/sh\ntouch executed\n")
    probes = [tmp_path / f"probe {i}{suffix}" for i in range(4)]
    for i, probe in enumerate(probes):
        probe.write_text(f"// compiled probe {i} reads the candidate implementation\n")
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    before = authority._capture_command_sources(
        ["bash", str(wrapper), *map(str, probes)], str(tmp_path))
    assert [item["path"] for item in before] == [str(wrapper), *map(str, probes)]
    for item in before:
        assert item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
    assert not (tmp_path / "executed").exists()

    probes[0].write_text(probes[0].read_text() + "// changed after measurement\n")
    after = authority._capture_command_sources(
        [item["path"] for item in before], str(tmp_path), discover_dependencies=False)
    assert [item["path"] for item in after] == [item["path"] for item in before]
    assert after[1]["sha256"] != before[1]["sha256"]
    assert after[2:] == before[2:]


@pytest.mark.parametrize("reference", ["argument", "variable", "dollar_chain", "brace_chain",
    "env_prefix", "env_prefix_interpreter", "timeout", "timeout_options"])
def test_attempt_source_capture_binds_literal_wrapper_dependencies(tmp_path, reference):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert p.returncode == 0, p.stderr
            return SimpleNamespace(stdout=p.stdout)

    probe = tmp_path / "probe space.py"
    probe.write_text("from pathlib import Path\nPath('executed').touch()\n")
    wrapper = tmp_path / "run.sh"
    if reference == "argument":
        source = f"python3 {shlex.quote(str(probe))}\n"
    elif reference == "variable":
        source = f"PROBE={shlex.quote(str(probe))}\npython3 \"$PROBE\"\n"
    elif reference in {"timeout", "timeout_options"}:
        prefix = "timeout 1200s" if reference == "timeout" else "timeout --foreground --kill-after=5s 1200s"
        source = (f'ROOT={shlex.quote(str(tmp_path))}\nPROBE="$ROOT/probe space.py"\n'
                  f'INTERP=python3\n{prefix} "$INTERP" "$PROBE" > check.json 2> check.err\n')
    elif reference in {"env_prefix", "env_prefix_interpreter"}:
        interpreter = '"$INTERP"' if reference == "env_prefix_interpreter" else "python3"
        source = (f'ROOT={shlex.quote(str(tmp_path))}\nPROBE="$ROOT/probe space.py"\n'
                  f'INTERP=python3\nDATA_DIR="$ROOT/data" APP_DIR="$ROOT" {interpreter} "$PROBE"\n')
    else:
        root = "$ROOT" if reference == "dollar_chain" else "${ROOT}"
        source = f'ROOT={shlex.quote(str(tmp_path))}\nPROBE="{root}/probe space.py"\npython3 "$PROBE"\n'
    wrapper.write_text("#!/bin/bash\n" + source)
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None

    before = authority._capture_command_sources(["bash", str(wrapper)], str(tmp_path))

    assert [item["path"] for item in before] == [str(wrapper), str(probe)]
    assert before[1]["source"] == probe.read_text()
    assert not (tmp_path / "executed").exists()
    probe.write_text(probe.read_text() + "# changed\n")
    after = authority._capture_command_sources(["bash", str(wrapper)], str(tmp_path))
    assert before[1]["sha256"] != after[1]["sha256"]


@pytest.mark.parametrize("assignment", [
    'PROBE="$(touch executed)/probe.py"',
    'PROBE="$UNDECLARED/probe.py"',
    "PROBE='$ROOT/probe.py'",
    'PROBE="\\$ROOT/probe.py"',
    'PROBE=\\$ROOT/probe.py',
    'ROOT="$(touch executed)"\nPROBE="$ROOT/probe.py"',
])
def test_attempt_source_capture_does_not_evaluate_dynamic_shell_or_expand_single_quotes(tmp_path, assignment):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert p.returncode == 0, p.stderr
            return SimpleNamespace(stdout=p.stdout)

    (tmp_path / "probe.py").write_text("raise RuntimeError('must not execute')\n")
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(f'ROOT={shlex.quote(str(tmp_path))}\n{assignment}\npython3 "$PROBE"\n')
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    captured = authority._capture_command_sources(["bash", str(wrapper)], str(tmp_path))
    assert [item["path"] for item in captured] == [str(wrapper)]
    assert not (tmp_path / "executed").exists()


def test_attempt_source_end_rehash_keeps_the_start_set_when_wrapper_creates_a_file(tmp_path):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert p.returncode == 0, p.stderr
            return SimpleNamespace(stdout=p.stdout)

    generated = tmp_path / "generated.py"
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(f'GENERATED={shlex.quote(str(generated))}\ncat > "$GENERATED" <<EOF\nprint(1)\nEOF\n')
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    before = authority._capture_command_sources(["bash", str(wrapper)], str(tmp_path))
    assert [item["path"] for item in before] == [str(wrapper)]
    generated.write_text("print(1)\n")
    after = authority._capture_command_sources(
        ["bash", str(wrapper), *(item["path"] for item in before)], str(tmp_path), False)
    assert after == before


def test_attempt_source_capture_excludes_copy_inputs_and_destinations(tmp_path):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert p.returncode == 0, p.stderr
            return SimpleNamespace(stdout=p.stdout)

    source = tmp_path / "candidate.py"; source.write_text("print('candidate')\n")
    destination = tmp_path / "active.py"; destination.write_text("print('previous')\n")
    probe = tmp_path / "probe.py"; probe.write_text("print('check')\n")
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(f'SRC={shlex.quote(str(source))}\nDST={shlex.quote(str(destination))}\n'
                       f'PROBE={shlex.quote(str(probe))}\ncp "$SRC" "$DST"\n'
                       f'cp {shlex.quote(str(source))} {shlex.quote(str(destination))}\n'
                       'timeout --foreground --kill-after=5s 20s cp "$SRC" "$DST"\n'
                       'python3 "$PROBE"\n')
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    before = authority._capture_command_sources(["bash", str(wrapper)], str(tmp_path))
    assert [item["path"] for item in before] == [str(wrapper), str(probe)]
    destination.write_bytes(source.read_bytes())
    after = authority._capture_command_sources(
        ["bash", str(wrapper), *(item["path"] for item in before)], str(tmp_path), False)
    assert after == before


@pytest.mark.parametrize("substitution", ["dollar", "backtick"])
@pytest.mark.parametrize("change_executed_source", [False, True])
def test_source_capture_distinguishes_hash_assignment_data_from_executed_script(
        tmp_path, substitution, change_executed_source):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert p.returncode == 0, p.stderr
            return SimpleNamespace(stdout=p.stdout)

    source = tmp_path / "candidate.py"
    source.write_text("# candidate treatment\n")
    deployed = tmp_path / "active.py"
    deployed.write_text("# previously deployed parent\n")
    probe = tmp_path / "evaluate.py"
    probe.write_text("raise RuntimeError('source capture must not execute the probe')\n")
    runner = tmp_path / "measure.sh"
    lines = [f'SRC={shlex.quote(str(source))}', f'DST={shlex.quote(str(deployed))}',
             f'PROBE={shlex.quote(str(probe))}', 'cp "$SRC" "$DST"']
    for name, var in [("ACTIVE_SHA", "DST"), ("SOURCE_SHA", "SRC")]:
        command = f'sha256sum "${var}" | cut -d\' \' -f1'
        value = "$(" + command + ")" if substitution == "dollar" else "`" + command + "`"
        lines.append(name + "=" + value)
    runner.write_text("\n".join(lines) + '\npython3 "$PROBE"\n')
    wrapper = tmp_path / "launch.sh"
    wrapper.write_text(shlex.join(["bash", str(runner)]) + "\n")
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    argv = ["bash", str(wrapper)]

    before = authority._capture_command_sources(argv, str(tmp_path))
    assert [item["path"] for item in before] == [str(wrapper), str(runner), str(probe)]
    deployed.write_bytes(source.read_bytes())
    if change_executed_source:
        probe.write_text("# actual evaluation source changed\n")
    after = authority._capture_command_sources(
        [*argv, *(item["path"] for item in before)], str(tmp_path), False)
    assert (after == before) is (not change_executed_source)


@pytest.mark.parametrize("substitution", ["dollar", "backtick"])
def test_source_capture_preserves_programs_executed_in_command_substitution(tmp_path, substitution):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    class Backend:
        def execute_command(self, handle, command, **kwargs):
            p = subprocess.run(shlex.split(command), capture_output=True, text=True, **kwargs)
            assert p.returncode == 0, p.stderr
            return SimpleNamespace(stdout=p.stdout)

    probe = tmp_path / "evaluate.py"
    probe.write_text("# executed evaluation program\n")
    command = f'python3 {shlex.quote(str(probe))} | cat'
    value = "$(" + command + ")" if substitution == "dollar" else "`" + command + "`"
    wrapper = tmp_path / "measure.sh"
    wrapper.write_text("RESULT=" + value + "\n")
    authority = object.__new__(_AttemptAuthority)
    authority.backend, authority.handle = Backend(), None
    argv = ["bash", str(wrapper)]
    before = authority._capture_command_sources(argv, str(tmp_path))
    assert [item["path"] for item in before] == [str(wrapper), str(probe)]
    probe.write_text("# actual execution source changed\n")
    after = authority._capture_command_sources(
        [*argv, *(item["path"] for item in before)], str(tmp_path), False)
    assert after[1]["sha256"] != before[1]["sha256"]


def test_attempt_review_evidence_preserves_candidate_test_sequence(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_review_evidence

    log = tmp_path / "attempts.jsonl"
    rows = []
    for index, (setting, score) in enumerate(((0.004, 0.88), (0.002, 0.90)), 1):
        output = tmp_path / f"attempt-{index}.stdout"
        output.write_text(json.dumps({"measurements": {"test_score": score},
                                      "config": {"epsilon": setting}}))
        rows.extend((
            {"event": "launch", "attempt_id": f"attempt-{index}",
             "argv": ["python", "evaluate.py", "--split", "test"],
             "command_sources": [{"path": "evaluate.py", "source": "actual candidate call"}],
             "tracked_diff_hash_start": f"sha256:{index}"},
            {"event": "finish", "attempt_id": f"attempt-{index}",
             "rc": 0, "stdout_path": str(output), "command_sources_stable": True},
        ))
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))

    trace = _attempt_review_evidence(str(log))

    assert [row["attempt_id"] for row in trace] == ["attempt-1", "attempt-2"]
    assert [row["stdout_fields"]["measurements"]["test_score"] for row in trace] == [0.88, 0.90]
    assert trace[1]["stdout_fields"]["config"]["epsilon"] == 0.002
    assert trace[1]["command_sources"] == rows[2]["command_sources"]
    assert trace[1]["command_sources_stable"] is True


@pytest.mark.parametrize("tampered", [False, True])
def test_attempt_review_exposes_verified_normalization_evidence(tmp_path, tampered):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_review_evidence
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    payload = {"method": 1.25, "candidate_valid": True, "measurements": {"rate": 125},
               "notes": {"formula": "candidate_rate / reference_rate",
                         "reference_policy": "stored reference from an earlier run",
                         "reference_raw": {"rate": 100},
                         "official_reward": 0.99}}
    saved = save_output(tmp_path, "attempt-normalization", json.dumps(payload), "")
    rows = [{"event": "launch", "attempt_id": "attempt-normalization", "phase": "method"},
            {"event": "finish", "attempt_id": "attempt-normalization", "rc": 0, **saved}]
    log = tmp_path / "attempts.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if tampered:
        Path(saved["stdout_path"]).write_text(json.dumps({"notes": {"formula": "replacement"}}))

    trace = _attempt_review_evidence(str(log))

    if tampered:
        assert not trace[0].get("verified_output_fields")
    else:
        fields = trace[0]["verified_output_fields"]
        assert fields["/notes/formula"] == payload["notes"]["formula"]
        assert fields["/notes/reference_policy"] == payload["notes"]["reference_policy"]
        assert fields["/notes/reference_raw/rate"] == 100
        assert fields["/method"] == 1.25
        assert trace[0]["stdout_sha256"] == saved["stdout_sha256"]
        assert "/notes/official_reward" not in fields


def test_attempt_review_preserves_complete_diagnostic_source_evidence(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_review_evidence
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    source = "fn check(candidate: &Candidate) { assert_eq!(candidate.count(), 0); }\n" * 140
    payload = {"measurements": {"failure_count": 0},
               "checks": {str(index): True for index in range(100)},
               "notes": {"probe_source": source, "main_source": source,
                         "contract_gate": {"candidate_sha256": "a" * 64,
                                           "build_passed": True}}}
    saved = save_output(tmp_path, "attempt-diagnostic", json.dumps(payload), "")
    rows = [{"event": "launch", "attempt_id": "attempt-diagnostic", "phase": "method"},
            {"event": "finish", "attempt_id": "attempt-diagnostic", "rc": 0, **saved}]
    log = tmp_path / "attempts.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))

    trace = _attempt_review_evidence(str(log))

    fields = trace[0]["verified_output_fields"]
    assert fields["/measurements/failure_count"] == 0
    assert fields["/checks/99"] is True
    assert fields["/notes/probe_source"] == source
    assert fields["/notes/main_source"] == source
    assert fields["/notes/contract_gate/candidate_sha256"] == "a" * 64
    assert fields["/notes/contract_gate/build_passed"] is True
    assert trace[0]["stdout_sha256"] == saved["stdout_sha256"]
    assert not trace[0].get("verified_output_truncated")


def test_attempt_review_marks_oversized_diagnostic_source_as_truncated(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_review_evidence
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    payload = {"measurements": {"failure_count": 0},
               "notes": {"probe_source": "fn check() {}\n" * 10000}}
    saved = save_output(tmp_path, "attempt-large-diagnostic", json.dumps(payload), "")
    rows = [{"event": "launch", "attempt_id": "attempt-large-diagnostic", "phase": "method"},
            {"event": "finish", "attempt_id": "attempt-large-diagnostic", "rc": 0, **saved}]
    log = tmp_path / "attempts.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))

    trace = _attempt_review_evidence(str(log))

    assert trace[0]["verified_output_truncated"] is True
    assert trace[0]["verified_output_fields"]["/measurements/failure_count"] == 0
    assert len(json.dumps(trace[0]["verified_output_fields"], ensure_ascii=False)) <= 65536


@pytest.mark.parametrize("error_type", [RuntimeError, InfraError])
def test_agent_error_preserves_unmeasured_worktree_for_debug(tmp_path, error_type):
    class Backend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            repo = Path(self._mounts[0]) / "repo"
            (repo / "partial.py").write_text("# unfinished assigned implementation\n")
            raise error_type("model request timed out")
    backend = Backend(write_result=False, record_attempt=False)
    result = run_agentic_improve(
        "q", repo_url="https://x/r", workroot=str(tmp_path),
        backend=backend, clone_fn=_fake_clone)
    assert Path(result["repo_path"], "partial.py").is_file()
    assert f"workspace: {result['repo_path']}" in result["diagnosis_evidence"]
    assert result["candidate"] is None
    assert not result["verdict"]["improved"]
    assert backend.closed


class _Res:
    def __init__(self, status="FINISHED"):
        self.status = status
        self.cost = "0"


def test_repository_task_contract_selects_request_matching_task_directory(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    first = tmp_path / "tasks" / "first_task"
    target = tmp_path / "tasks" / "target_task"
    first.mkdir(parents=True)
    target.mkdir(parents=True)
    (first / "instruction.md").write_text("Rule for another task.\n", encoding="utf-8")
    (target / "instruction.md").write_text(
        "Do not use prohibited algorithm Q.\n", encoding="utf-8")
    (target / "task.toml").write_text(
        '[optimization]\nmetric = "quality"\n', encoding="utf-8")

    contract = _repository_task_contract(
        str(tmp_path), "Improve the official target_task benchmark")

    assert [item["path"] for item in contract["sources"]] == [
        "tasks/target_task/instruction.md", "tasks/target_task/task.toml"]
    assert "prohibited algorithm Q" in contract["sources"][0]["content"]
    assert all(len(item["sha256"]) == 64 for item in contract["sources"])


@pytest.mark.parametrize('confirmed_task', [False, True])
def test_task_rules_are_not_displaced_by_explicit_candidate_data_and_code(tmp_path, confirmed_task):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    task = tmp_path / 'tasks/target_task'
    environment = task / 'environment'
    environment.mkdir(parents=True)
    instruction = 'Rewrite candidate.txt to maximize quality. Do not edit the evaluator.'
    (task / 'instruction.md').write_text(instruction)
    (task / 'task.toml').write_text('[optimization]\nmetric="quality"\ndirection="higher"\n')
    (environment / 'candidate.txt').write_text('42\n' * 9000)
    (environment / 'solve.py').write_text('# implementation\n' * 1000)
    (environment / 'Dockerfile').write_text('COPY candidate.txt /app/candidate.txt\n')
    contract = _repository_task_contract(str(tmp_path),
        'Optimize target_task. Read tasks/target_task/environment/candidate.txt '
        'and tasks/target_task/environment/solve.py.',
        {'experiment_provenance': {'task': 'target_task'}} if confirmed_task else None)
    sources = {source['path']: source for source in contract['sources']}
    assert sources['tasks/target_task/instruction.md']['content'] == instruction
    assert sources['tasks/target_task/instruction.md']['truncated'] is False
    assert sources['tasks/target_task/task.toml']['truncated'] is False
    assert sources['tasks/target_task/environment/Dockerfile']['truncated'] is False
    assert contract['execution_contract']['evaluation'] == {
        'metric_name': 'quality', 'lower_is_better': False}
    assert not contract['capture_complete']  # Auxiliary data still obeys the capture budget.


def test_confirmed_task_selects_its_manifest_despite_other_task_in_recovery_text(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    for name, metric in (("alpha_task", "latency"), ("omega_task", "accuracy")):
        task = tmp_path / "tasks" / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            f'[optimization]\nmetric = "{metric}"\ndirection = "lower"\n')
    request = "Reproduce alpha_task; prior failed attempt mentioned omega_task"
    with pytest.raises(ValueError, match="conflicting optimization objectives"):
        _repository_task_contract(str(tmp_path), request)

    contract = _repository_task_contract(str(tmp_path), request, {
        "experiment_provenance": {"task": "alpha_task"}})
    assert [source["path"] for source in contract["sources"]] == [
        "tasks/alpha_task/task.toml"]
    assert contract["execution_contract"]["evaluation"]["metric_name"] == "latency"


@pytest.mark.parametrize('layout', ['declared_subtask', 'repository_root'])
def test_confirmed_task_root_rules_precede_nested_legacy_manifest(tmp_path, layout):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    task = tmp_path / 'tasks/target_task' if layout == 'declared_subtask' else tmp_path
    (task / 'tests').mkdir(parents=True)
    (task / 'instruction.md').write_text('Edit only the documented candidate.')
    (task / 'task.toml').write_text('[optimization]\nmetric="quality"\ndirection="higher"\n')
    (task / 'tests/task.toml').write_text('[optimization]\nmetric="obsolete_metric"\n')
    prefix = 'tasks/target_task/' if layout == 'declared_subtask' else ''
    contract = _repository_task_contract(str(tmp_path),
        'Optimize target_task; prior setup referenced ' + prefix + 'tests/task.toml',
        {'experiment_provenance': {'task': 'target_task'}} if prefix else None)
    assert {s['path'] for s in contract['sources']} == {
        prefix + 'instruction.md', prefix + 'task.toml'}
    assert contract['execution_contract']['evaluation'] == {
        'metric_name': 'quality', 'lower_is_better': False}


@pytest.mark.parametrize("reported,valid", [
    ("runtime_seconds", True),
    ("wall_clock_runtime_seconds", False),
])
def test_task_manifest_metric_binds_baseline_identity(tmp_path, reported, valid):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        RunResult, _finalize_relaxed_phase_result, _repository_task_contract,
    )

    task = tmp_path / "tasks" / "blur"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        '[optimization]\nmetric = "runtime_seconds"\ndirection = "lower"\n')
    contract = _repository_task_contract(str(tmp_path), "Optimize the blur task")
    assert contract["execution_contract"]["evaluation"] == {
        "metric_name": "runtime_seconds", "lower_is_better": True}

    result = _finalize_relaxed_phase_result({
        "metric": reported,
        "_reported_metric": reported,
        "_reported_metric_unit": "seconds",
        "_reported_lower_is_better": True,
        "baseline": RunResult(metric_name=reported, metric_value=0.34, valid=True),
        "baseline_validity": True,
        "repository_contract": contract,
    }, "baseline", None)

    assert result["baseline"].valid is valid
    if not valid:
        assert "expected name='runtime_seconds'" in result["error"]
        assert "reported name='wall_clock_runtime_seconds'" in result["error"]


def test_repository_contract_survives_recovery_symlink_without_allowing_escape(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    checkout = tmp_path / "original" / "repo"
    checkout.mkdir(parents=True)
    (checkout / "task.toml").write_text('[optimization]\nmetric="bytes"\ndirection="lower"\n')
    outside = tmp_path / "outside.md"
    outside.write_text("This is not a task document.")
    (checkout / "instruction.md").symlink_to(outside)
    recovery = tmp_path / "recovered"
    recovery.symlink_to(checkout, target_is_directory=True)

    direct = _repository_task_contract(str(checkout), "optimize bytes")
    restored = _repository_task_contract(str(recovery), "optimize bytes")
    assert restored == direct
    assert [source["path"] for source in restored["sources"]] == ["task.toml"]


def test_task_contract_keeps_selected_task_packaging_context(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    for task in ("chosen_task", "other_task"):
        root = tmp_path / "tasks" / task
        (root / "container").mkdir(parents=True)
        (root / "instruction.md").write_text("Edit only /app/solve.py.")
        (root / "container" / "Dockerfile").write_text("WORKDIR /app\nCOPY . /app/\n")
        (root / "container" / ".dockerignore").write_text("scratch/\n")
    # A nested, separately documented task must not leak into the selected context.
    nested = tmp_path / "tasks/chosen_task/examples/another"
    nested.mkdir(parents=True)
    (nested / "task.md").write_text("Different task.")
    (nested / "Containerfile").write_text("COPY secret /app/\n")
    result = _repository_task_contract(
        str(tmp_path), "Optimize tasks/chosen_task/instruction.md")
    assert [s["path"] for s in result["sources"]] == [
        "tasks/chosen_task/instruction.md",
        "tasks/chosen_task/container/.dockerignore",
        "tasks/chosen_task/container/Dockerfile",
    ]
    assert result["sources"][2]["content"] == "WORKDIR /app\nCOPY . /app/\n"
    assert result["capture_complete"]


def test_task_packaging_context_respects_capture_budget_and_symlink_boundary(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "instruction.md").write_text("Use the documented submission scope.")
    outside = tmp_path / "external"
    outside.write_text("COPY outside /app/\n")
    (root / "Dockerfile").symlink_to(outside)
    for i in range(9):
        folder = root / f"build{i}"
        folder.mkdir()
        (folder / "Containerfile").write_text("COPY . /app/\n")
    result = _repository_task_contract(str(root), "Optimize the program")
    assert len(result["sources"]) == 8
    assert result["sources"][0]["path"] == "instruction.md"
    assert all(s["path"] != "Dockerfile" for s in result["sources"])
    assert not result["capture_complete"]
    assert len(result["omitted_sources"]) == 2


@pytest.mark.parametrize("mutation,expected", [
    (None, True), ("packaging", False), ("rules", False), ("new_file", False),
])
def test_frozen_review_contract_only_adds_commit_pinned_packaging(tmp_path, mutation, expected):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _repository_task_contract, _contract_with_packaging_evidence,
    )
    (tmp_path / "instruction.md").write_text("Edit only /app/solve.py.")
    (tmp_path / "Dockerfile").write_text("COPY environment /app\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    captured = _repository_task_contract(str(tmp_path), "Optimize the program")
    frozen = {**captured, "sources": [s for s in captured["sources"] if s["path"] == "instruction.md"]}
    before = json.dumps(frozen, sort_keys=True)
    if mutation == "packaging":
        (tmp_path / "Dockerfile").write_text("COPY . /app\n")
    elif mutation == "rules":
        (tmp_path / "instruction.md").write_text("Any edit is allowed.")
    elif mutation == "new_file":
        (tmp_path / "Dockerfile").unlink()
        (tmp_path / "Containerfile").write_text("COPY . /app\n")
    fresh = _repository_task_contract(str(tmp_path), "Optimize the program")
    result = _contract_with_packaging_evidence(str(tmp_path), {
        "repository_contract": frozen, "repo_commit": commit,
    }, fresh)
    assert json.dumps(frozen, sort_keys=True) == before
    assert result["sources"][0] == frozen["sources"][0]
    assert result["execution_contract"] == frozen["execution_contract"]
    assert len(result["sources"]) == (2 if expected else 1)
    if expected:
        assert result["sources"][1]["path"] == "Dockerfile"
        assert result["sources"][1]["content"] == "COPY environment /app\n"


@pytest.mark.parametrize("mutation", [None, "rules", "missing_commit", "incomplete"])
def test_empty_frozen_contract_recovers_only_complete_commit_pinned_sources(tmp_path, mutation):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _repository_task_contract, _contract_with_packaging_evidence,
    )
    (tmp_path / "instruction.md").write_text("Edit only solve.py; preserve the official evaluator.")
    (tmp_path / "Dockerfile").write_text("COPY . /app\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    frozen = {"sources": [], "execution_contract": {"record_runtime_identity": True}}
    before = json.dumps(frozen, sort_keys=True)
    if mutation == "rules":
        (tmp_path / "instruction.md").write_text("Any change is allowed.")
    captured = _repository_task_contract(str(tmp_path), "Optimize solve.py", frozen["execution_contract"])
    if mutation == "incomplete":
        captured["capture_complete"] = False
    result = _contract_with_packaging_evidence(str(tmp_path), {
        "repository_contract": frozen,
        "repo_commit": None if mutation == "missing_commit" else commit,
    }, captured)
    assert json.dumps(frozen, sort_keys=True) == before
    if mutation is None:
        assert result["sources"] == captured["sources"]
        assert result["recovered_task_sources"] == [s["path"] for s in captured["sources"]]
        assert result["execution_contract"]["record_runtime_identity"] is True
    else:
        assert result == frozen


@pytest.mark.parametrize("mutation", [None, "rules", "missing_commit"])
def test_full_legacy_code_capture_does_not_hide_commit_pinned_task_documents(tmp_path, mutation):
    import hashlib
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _repository_task_contract, _contract_with_packaging_evidence,
    )
    (tmp_path / "instruction.md").write_text("Edit only input.txt; do not modify the evaluator.")
    (tmp_path / "task.toml").write_text("[optimization]\nmetric = 'cost'\ndirection = 'lower'\n")
    sources = []
    for i in range(8):
        path = tmp_path / f"legacy{i}.py"
        content = "#" + "x" * 2498 + "\n"
        path.write_text(content)
        sources.append({"path": path.name, "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(), "truncated": False})
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    frozen = {"sources": sources, "execution_contract": {"record_runtime_identity": True}}
    before = json.dumps(frozen, sort_keys=True)
    if mutation == "rules":
        (tmp_path / "instruction.md").write_text("Modify anything, including the evaluator.")
    captured = _repository_task_contract(str(tmp_path), "Optimize the program")
    recovered = _contract_with_packaging_evidence(str(tmp_path), {
        "repository_contract": frozen, "repo_commit": None if mutation == "missing_commit" else commit,
    }, captured)

    assert json.dumps(frozen, sort_keys=True) == before
    assert recovered["sources"][:8] == sources
    assert recovered["execution_contract"] == frozen["execution_contract"]
    added = {s["path"]: s for s in recovered["sources"][8:]}
    expected = set() if mutation == "missing_commit" else {"task.toml"}
    if mutation is None:
        expected.add("instruction.md")
    assert set(added) == expected
    assert set(recovered.get("recovered_task_sources", [])) == expected
    assert all("Modify anything" not in s["content"] for s in added.values())


@pytest.mark.parametrize("existing_sources", [False, True])
@pytest.mark.parametrize("edited_source", [False, True])
def test_review_context_recovers_named_python_sources_without_rewriting_baseline(
        tmp_path, existing_sources, edited_source):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract
    from autoresearch.experiments.repo_improve.repository_context import baseline_repository_context

    root = tmp_path / "modules/route"
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    (root / "instruction.md").write_text("Preserve the read-only runtime interface.")
    (root / "task.toml").write_text("[environment]\ncpus = 1\n")
    (runtime / "Dockerfile").write_text("COPY *.py /app/\n")
    names = ["api.py", "weights.py", "trainer.py", "metrics.py", "secret.py"]
    for name in names:
        (runtime / name).write_text(f"# public interface {name}\ndef load_model(path):\n    return path\n")
    policy = {"experiment_provenance": {"task": "route"},
              "sources": {"forbidden": ["modules/route/runtime/secret.py"]}}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    original = _repository_task_contract(str(tmp_path), "Optimize route", policy)
    frozen = {**original, "sources": original["sources"] if existing_sources else []}
    baseline = {"repo_commit": commit, "repository_contract": frozen,
                "artifacts": [str(tmp_path / "result.json")]}
    before = json.dumps(baseline, sort_keys=True)
    if edited_source:
        (runtime / "api.py").write_text("# candidate-edited interface must not become baseline evidence\n")
    request = "Inspect " + " ".join(
        f"https://example.invalid/owner/repo/blob/main/modules/route/runtime/{name}" for name in names)
    recovered = baseline_repository_context(baseline, request)
    sources = {source["path"]: source for source in recovered["repository_contract"]["sources"]}
    assert json.dumps(baseline, sort_keys=True) == before
    assert "modules/route/runtime/secret.py" not in sources
    if edited_source:
        assert "modules/route/runtime/api.py" not in sources
    else:
        assert len(sources) == 7
        assert "def load_model(path):" in sources["modules/route/runtime/api.py"]["content"]
        assert "modules/route/instruction.md" in sources
        assert all(not source["truncated"] for source in sources.values())


def test_repository_resource_limits_come_only_from_selected_task_manifest(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _repository_resource_limits,
        _repository_task_contract,
    )

    other = tmp_path / "tasks" / "other_task"
    target = tmp_path / "tasks" / "target_task"
    other.mkdir(parents=True)
    target.mkdir(parents=True)
    (other / "task.toml").write_text(
        "[environment]\ncpus = 64\nmemory_mb = 99999\n", encoding="utf-8")
    (target / "task.toml").write_text(
        "[environment]\ncpus = 2\nmemory_mb = 4096\n", encoding="utf-8")

    contract = _repository_task_contract(
        str(tmp_path), "Improve the official target_task benchmark")

    assert _repository_resource_limits(contract) == {"cpus": 2, "memory_mb": 4096}


def test_repository_resource_limits_do_not_guess_without_manifest():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _repository_resource_limits,
    )

    assert _repository_resource_limits({"sources": []}) == {}


def test_task_contract_keeps_explicit_json_yaml_and_named_sibling(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    for rel in ("configs/tasks/vision.yaml", "ml_tasks/vision/config.json",
                "ml_tasks/vision/prompt.json", "ml_tasks/other/prompt.json",
                "ml_tasks/vision/unmentioned.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"task": "real repository-defined objective"}')
    request = "Use configs/tasks/vision.yaml and ml_tasks/vision/config.json、prompt.json."
    result = _repository_task_contract(str(tmp_path), request)
    assert {s["path"] for s in result["sources"]} == {
        "configs/tasks/vision.yaml", "ml_tasks/vision/config.json", "ml_tasks/vision/prompt.json"}
    assert all("repository-defined objective" in s["content"] for s in result["sources"])


def test_task_contract_does_not_follow_explicit_document_outside_repo(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "private.json"
    external.write_text('{"private": true}')
    (root / "linked.json").symlink_to(external)
    assert _repository_task_contract(str(root), "Use ./linked.json or ../private.json") is None


def test_capture_repo_diff_includes_large_untracked_source_but_not_result_protocol(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
                   check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    source = "\n".join(f"value_{index} = {index}" for index in range(1600)) + "\n"
    assert len(source.encode()) > 20_000
    (tmp_path / "method_train.py").write_text(source, encoding="utf-8")
    (tmp_path / "ar_result.json").write_text('{"method": -999}\n', encoding="utf-8")

    diff = _capture_repo_diff(str(tmp_path))

    assert "# 新增文件 method_train.py:" in diff
    assert "value_1599 = 1599" in diff
    assert "ar_result.json" not in diff
    assert '"method": -999' not in diff


def test_capture_repo_diff_separates_host_manifest_from_candidate_files(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import (
        MANIFEST, snapshot_checkpoint,
    )

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "algorithm.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "algorithm.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    (repo / "algorithm.py").write_text("value = 2\n")
    (repo / "run_candidate.py").write_text("import algorithm\n")
    checkpoint = snapshot_checkpoint(str(repo), "attempt-1")

    diff = _capture_repo_diff(checkpoint, host_checkpoint=True)
    assert "+value = 2" in diff
    assert "run_candidate.py" in diff
    assert MANIFEST not in diff

    # A same-named file in a candidate checkout is not host-owned evidence.
    (repo / MANIFEST).write_text('{"candidate_authored": true}')
    assert "candidate_authored" in _capture_repo_diff(str(repo))
    # Nor does the root manifest exemption hide a file in a candidate subdirectory.
    nested = Path(checkpoint) / "candidate"
    nested.mkdir()
    (nested / MANIFEST).write_text('{"nested_candidate_authored": true}')
    assert "nested_candidate_authored" in _capture_repo_diff(checkpoint, host_checkpoint=True)


def test_capture_repo_diff_expands_source_files_inside_untracked_directory(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
                   check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    method_dir = tmp_path / "_method"
    method_dir.mkdir()
    (method_dir / "train_method.py").write_text(
        "class VarianceGatedResidualReadout:\n    pass\n", encoding="utf-8")

    diff = _capture_repo_diff(str(tmp_path))

    assert "# 新增文件 _method/train_method.py:" in diff
    assert "class VarianceGatedResidualReadout:" in diff


def test_capture_repo_diff_does_not_silently_drop_later_component_files(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for component in range(6):
        folder = tmp_path / f"component_{component}"
        folder.mkdir()
        for part in range(4):
            (folder / f"part_{part}.py").write_text(
                f"def check_{component}_{part}():\n    return True\n", encoding="utf-8")
    diff = _capture_repo_diff(str(tmp_path))
    for component in range(6):
        assert f"def check_{component}_3():" in diff


def test_capture_repo_diff_preserves_small_later_sources_when_content_budget_is_full(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("# first large source\n" + "# padding\n" * 9000)
    (tmp_path / "b.py").write_text("# second large source\n" + "# padding\n" * 9000)
    (tmp_path / "z.py").write_text("def check_final_component():\n    return True\n")
    diff = _capture_repo_diff(str(tmp_path))
    assert "def check_final_component():" in diff
    assert "# 新增文件(大文件/目录,未展开): b.py" in diff
    assert "SOURCE_CAPTURE_INCOMPLETE" in diff
    assert len(diff) < 120_000


@pytest.mark.parametrize("cmake_build", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_capture_repo_diff_cmake_outputs_do_not_hide_component_sources(tmp_path, cmake_build, nested):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "--allow-empty", "-qm", "base"],
                   check=True)
    build = tmp_path / "build_lib"
    if nested:
        build = build / "nested_build"
    generated = build / "CMakeFiles" / "CompilerIdCUDA"
    generated.mkdir(parents=True)
    if cmake_build:
        (build / "CMakeCache.txt").write_text("# CMake cache\n", encoding="utf-8")
    for index in range(25 if cmake_build else 1):
        (generated / f"generated_{index:02}.cu").write_text(
            "// generated compiler identification\n", encoding="utf-8")
    component = tmp_path / "component"
    component.mkdir()
    (component / "kernel.cuh").write_text(
        "__device__ int component_kernel() { return 1; }\n", encoding="utf-8")

    diff = _capture_repo_diff(str(tmp_path))

    assert "component/kernel.cuh" in diff
    assert "__device__ int component_kernel()" in diff
    assert ("generated compiler identification" in diff) is not cmake_build


def test_capture_repo_diff_includes_staged_and_unstaged_source_without_binary(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_repo_diff

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
                   check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    source = tmp_path / "component.cpp"
    source.write_text("int component_size = 256;\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "component.cpp"], check=True)
    source.write_text("int component_size = 256;\nfloat decay = 0.99;\n", encoding="utf-8")
    (tmp_path / "compiled_probe").write_bytes(b"\x7fELF\x00BINARY_PAYLOAD")

    diff = _capture_repo_diff(str(tmp_path))

    assert "+int component_size = 256;" in diff
    assert "+float decay = 0.99;" in diff
    assert "compiled_probe" in diff
    assert "BINARY_PAYLOAD" not in diff
    assert "\x00" not in diff


def test_repo_commit_reads_container_owned_checkpoint_via_resolved_path(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _repo_commit

    source = tmp_path / "checkpoint"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "--allow-empty", "-qm", "base"],
                   check=True)
    expected = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    alias = tmp_path / "recovery-checkpoint"
    alias.symlink_to(source, target_is_directory=True)
    # Git's own ownership test seam avoids changing filesystem ownership in tests.
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "")
    rejected = subprocess.run(["git", "-C", str(alias), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "dubious ownership" in rejected.stderr
    assert _repo_commit(str(alias)) == expected


def test_repo_commit_does_not_invent_commit_for_non_repository(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _repo_commit

    assert _repo_commit(str(tmp_path / "missing")) is None


@pytest.mark.parametrize("hard_cap", [7200, 28800])
def test_method_prompt_without_soft_budget_discloses_actual_hard_deadline(monkeypatch, hard_cap):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", str(hard_cap))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_FINALIZE_RESERVE_S", "180")
    prompt = _method_task_prompt("improve", "/repo", "/repo/ar_result.json", None, {
        "metric_name": "loss", "metric_value": 1.0, "lower_is_better": True,
    })
    assert "no fixed total wall-clock budget" not in prompt
    assert f"{hard_cap}s total sandbox hard limit" in prompt
    assert "final 180s" in prompt
    assert "before further analysis or another attempt" in prompt


@pytest.mark.parametrize("suffix", [".s", ".S", ".asm"])
def test_assembly_source_survives_component_diff_checkpoint_and_restore(tmp_path, suffix):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _capture_component_diff, _capture_repo_diff, _apply_prior_worktree)
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    source = repo / "src" / ("kernel" + suffix)
    source.parent.mkdir()
    source.write_text(".text\n.globl kernel\nkernel:\n  ret\n")
    assert ".globl kernel" in _capture_component_diff(str(repo), None)
    assert ".globl kernel" in _capture_repo_diff(str(repo))
    saved = Path(snapshot_checkpoint(str(repo), "attempt-assembly"))
    assert (saved / source.relative_to(repo)).read_bytes() == source.read_bytes()
    target = tmp_path / "next"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    _apply_prior_worktree(str(target), str(saved))
    assert (target / source.relative_to(repo)).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("change_shared", [False, True])
@pytest.mark.parametrize("changed_parent", [False, True])
def test_component_diff_excludes_reused_starter_lines_in_copied_sources(
    tmp_path, monkeypatch, change_shared, changed_parent,
):
    from autoresearch.experiments.repo_improve import agentic_improve
    from autoresearch.agents.method_implementation_agent.core import _diff_evidence_candidates

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    starter = repo / "starter.py"
    original = "\n".join(
        f"def baseline_{i}(value):\n    return value + {i}\n" for i in range(20))
    starter.write_text(original)
    subprocess.run(["git", "-C", str(repo), "add", "starter.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "starter implementation"], check=True)
    candidate = repo / "candidate.py"
    content = original + "\ndef target(value):\n    return value * 2\n"
    if change_shared:
        content = content.replace("return value + 3\n", "return value + 300\n")
    candidate.write_text(content)
    if change_shared:
        candidate.chmod(0o755)
    subprocess.run(["git", "-C", str(repo), "add", "candidate.py"], check=True)
    if changed_parent:
        parent_content = original.replace("return value + 0\n", "return value + 1000\n")
        starter.write_text(parent_content)
        monkeypatch.setattr(agentic_improve, "_dependency_worktree_changes",
                            lambda *_: {"starter.py": (parent_content.encode(), 0o644)})

    raw = agentic_improve._capture_repo_diff(str(repo))
    patch = agentic_improve._capture_component_diff(str(repo), ["parent"] if changed_parent else [])
    assert agentic_improve._capture_repo_diff(str(repo)) == raw
    added = [line for _, sign, line in _diff_evidence_candidates(patch) if sign == "+"]
    assert any(line.startswith("def baseline_") for line in added) is changed_parent
    assert "def target(value):" in added
    assert ("return value + 300" in added) is change_shared
    assert ("copy from starter.py" in patch) is not changed_parent

    target = tmp_path / "target"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    if changed_parent:
        (target / "starter.py").write_text(parent_content)
    applied = subprocess.run(["git", "-C", str(target), "apply", "-"],
                             input=patch, text=True, capture_output=True)
    assert applied.returncode == 0, applied.stderr
    assert (target / "candidate.py").read_text() == content
    assert bool((target / "candidate.py").stat().st_mode & 0o111) is change_shared
    assert (target / "starter.py").read_text() == (parent_content if changed_parent else original)


@pytest.mark.parametrize("deleted", [False, True])
@pytest.mark.parametrize("executable", [False, True])
@pytest.mark.parametrize("final_newline", [False, True])
def test_component_diff_round_trips_added_and_deleted_source(
    tmp_path, deleted, executable, final_newline,
):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_component_diff

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    source = repo / "kernel.c"
    content = "int kernel(void) { return 7; }" + ("\n" if final_newline else "")
    if deleted:
        source.write_text(content)
        source.chmod(0o755 if executable else 0o644)
        subprocess.run(["git", "-C", str(repo), "add", "kernel.c"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "add source"], check=True)
    target = tmp_path / "target"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    if deleted:
        source.unlink()
    else:
        source.write_text(content)
        source.chmod(0o755 if executable else 0o644)
    patch = _capture_component_diff(str(repo), None)
    applied = subprocess.run(["git", "-C", str(target), "apply", "-"],
                             input=patch, text=True, capture_output=True)
    assert applied.returncode == 0, applied.stderr
    if deleted:
        assert not (target / "kernel.c").exists()
    else:
        assert (target / "kernel.c").read_text() == content
        assert bool((target / "kernel.c").stat().st_mode & 0o111) is executable


@pytest.mark.parametrize("deleted", [False, True])
def test_component_diff_records_tracked_binary_identity_without_losing_source(tmp_path, deleted):
    import hashlib
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_component_diff

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    binary = repo / "program"
    before = b"\x7fELF\0old"
    binary.write_bytes(before)
    subprocess.run(["git", "-C", str(repo), "add", "program"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "-qm", "tracked build output"], check=True)
    assert _capture_component_diff(str(repo), None) == ""
    if deleted:
        binary.unlink()
    else:
        binary.write_bytes(b"\x7fELF\0new")
    (repo / "program.c").write_text("int main(void) { return 0; }\n")
    delta = _capture_component_diff(str(repo), None)
    assert "Binary files differ (content not expanded): program" in delta
    assert hashlib.sha256(before).hexdigest() in delta
    assert ("after absent" if deleted else hashlib.sha256(binary.read_bytes()).hexdigest()) in delta
    assert "+int main(void)" in delta


def test_component_diff_still_rejects_binary_disguised_as_source(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_component_diff

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    (repo / "program.c").write_bytes(b"\x7fELF\0not source")
    with pytest.raises(ValueError, match="cannot expand binary source"):
        _capture_component_diff(str(repo), None)


def test_component_diff_excludes_inherited_shared_file_edits_across_improvements(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _apply_prior_worktree, _capture_component_diff)
    base = tmp_path / "base"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    prefix = "".join(f"# unchanged line {i}\n" for i in range(10000))
    (base / "solve.py").write_text(prefix + "PARENT = 0\nCURRENT = 0\n")
    subprocess.run(["git", "-C", str(base), "add", "."], check=True)
    subprocess.run(["git", "-C", str(base), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "-qm", "base"], check=True)
    parent, target = tmp_path / "parent", tmp_path / "target"
    for repo in (parent, target):
        subprocess.run(["git", "clone", "-q", str(base), str(repo)], check=True)
    (parent / "solve.py").write_text(prefix + "PARENT = 1\nCURRENT = 0\n")
    (parent / "helper.py").write_text("def encode(x):\n    return x\n")
    _apply_prior_worktree(str(target), str(parent))
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    (cache / "private.py").write_text("MUST_NOT_BE_READ = True\n")
    (target / "third_party_runtime").symlink_to(cache, target_is_directory=True)
    for version in (1, 2):
        (target / "solve.py").write_text(prefix + f"PARENT = 1\nCURRENT = {version}\n")
        (target / "helper.py").write_text("def encode(x):\n    return x\n\ndef new_loss(x):\n    return x*x\n")
        subprocess.run(["git", "-C", str(target), "add", "solve.py"], check=True)
        delta = _capture_component_diff(str(target), [str(parent)])
        assert "+PARENT = 1" not in delta
        assert "-PARENT = 0" not in delta
        assert "+def encode" not in delta
        assert "+def new_loss" in delta
        assert "-CURRENT = 0" in delta
        assert f"+CURRENT = {version}" in delta
        assert "MUST_NOT_BE_READ" not in delta
        assert "third_party_runtime" not in delta
    assert (parent / "solve.py").read_text() == prefix + "PARENT = 1\nCURRENT = 0\n"


@pytest.mark.parametrize("committed", [False, True])
def test_component_diff_and_checkpoint_include_untracked_nested_repository(tmp_path, committed):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _apply_prior_worktree, _capture_component_diff)

    parent, target = tmp_path / "parent", tmp_path / "target"
    parent.mkdir()
    subprocess.run(["git", "init", "-q", str(parent)], check=True)
    (parent / "main.py").write_text("BASE = 1\n")
    subprocess.run(["git", "-C", str(parent), "add", "."], check=True)
    subprocess.run(["git", "-C", str(parent), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "-qm", "base"], check=True)
    subprocess.run(["git", "clone", "-q", str(parent), str(target)], check=True)
    nested = parent / "method" / "library"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "kernel.cuh").write_text("int reduce() { return 1; }\n")
    (nested / "tests").mkdir()
    (nested / "tests" / "evaluator.py").write_text("DO_NOT_CARRY = True\n")
    subprocess.run(["git", "-C", str(nested), "add", "."], check=True)
    subprocess.run(["git", "-C", str(nested), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "-qm", "library"], check=True)
    (nested / "kernel.cuh").write_text("int reduce() { return 2; }\n")
    if committed:
        subprocess.run(["git", "-C", str(nested), "add", "."], check=True)
        subprocess.run(["git", "-C", str(nested), "-c", "user.email=test@example.com",
                        "-c", "user.name=Test", "commit", "-qm", "component"], check=True)
    delta = _capture_component_diff(str(parent), [])
    assert "method/library/kernel.cuh" in delta
    assert "+int reduce() { return 2; }" in delta
    assert "DO_NOT_CARRY" not in delta
    assert _apply_prior_worktree(str(target), str(parent)) == 1
    assert (target / "method/library/kernel.cuh").read_text() == "int reduce() { return 2; }\n"
    assert not (target / "method/library/.git").exists()
    assert not (target / "method/library/tests").exists()
    assert _capture_component_diff(str(target), str(parent)) == ""
    (target / "method/library/kernel.cuh").write_text("int reduce() { return 3; }\n")
    delta = _capture_component_diff(str(target), str(parent))
    assert "-int reduce() { return 2; }" in delta
    assert "+int reduce() { return 3; }" in delta


def test_apply_prior_worktree_carries_repository_code_but_not_evaluator(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree

    source = tmp_path / "repo"
    target = tmp_path / "repo-2"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    for relative, content in {
        "environment/solve.py": "BASE = 1\n",
        "tests/evaluator.py": "OFFICIAL = 1\n",
        "solution/answer.py": "SECRET = 1\n",
        "task.toml": "metric = 'x'\n",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "base"], check=True)
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True)

    (source / "environment/solve.py").write_text("BASE = 2\n")
    (source / "tests/evaluator.py").write_text("OFFICIAL = 2\n")
    (source / "solution/answer.py").write_text("SECRET = 2\n")
    (source / "task.toml").write_text("metric = 'changed'\n")
    (source / "environment/helper.py").write_text("HELPER = True\n")

    assert _apply_prior_worktree(str(target), str(source)) == 3
    assert (target / "environment/solve.py").read_text() == "BASE = 2\n"
    assert (target / "environment/helper.py").read_text() == "HELPER = True\n"
    assert (target / "tests/evaluator.py").read_text() == "OFFICIAL = 1\n"
    assert (target / "solution/answer.py").read_text() == "SECRET = 2\n"
    assert (target / "task.toml").read_text() == "metric = 'x'\n"


@pytest.mark.parametrize("staged", [False, True])
def test_checkpoint_carries_analysis_implementation_without_old_results(tmp_path, staged):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _apply_prior_worktree, _capture_component_diff)

    source, target = tmp_path / "parent", tmp_path / "next"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "main.py").write_text("BASE = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "-qm", "base"], check=True)
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True)
    analysis = source / "analysis"
    analysis.mkdir()
    (analysis / "scan.py").write_text("def scan():\n    return 1\n")
    (analysis / "run_scan.sh").write_text("#!/bin/sh\npython3 analysis/scan.py\n")
    (analysis / "run_scan.sh").chmod(0o755)
    for name in ("result.json", "variants.jsonl", "direction.txt", "timings.csv"):
        (analysis / name).write_text("OLD_RESULT\n")
    (analysis / "tests").mkdir()
    (analysis / "tests/evaluator.py").write_text("CHANGED_EVALUATOR = True\n")
    if staged:
        subprocess.run(["git", "-C", str(source), "add", "analysis"], check=True)

    assert _apply_prior_worktree(str(target), str(source)) == 2
    assert (target / "analysis/scan.py").read_text() == (analysis / "scan.py").read_text()
    assert (target / "analysis/run_scan.sh").stat().st_mode & 0o111
    assert sorted(p.name for p in (target / "analysis").iterdir()) == ["run_scan.sh", "scan.py"]
    assert _capture_component_diff(str(target), str(source)) == ""

    (target / "analysis/scan.py").write_text("def scan():\n    return 2\n")
    (target / "analysis/check_next.py").write_text("NEXT_CHECK = True\n")
    (target / "analysis/result.json").write_text('{"value": 123}\n')
    diff = _capture_component_diff(str(target), str(source))
    assert "analysis/scan.py" in diff and "+    return 2" in diff
    assert "analysis/check_next.py" in diff and "+NEXT_CHECK = True" in diff
    assert "analysis/run_scan.sh" not in diff
    assert "result.json" not in diff


@pytest.mark.parametrize("change", ["unchanged", "target_edit", "retarget", "escape"])
def test_component_diff_compares_original_alias_to_materialized_checkpoint(tmp_path, change):
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_component_diff
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint

    source, candidate = tmp_path / "source", tmp_path / "candidate"
    _fake_clone("unused", str(source))
    (source / "helper.py").write_text("VALUE = 1\n")
    (source / "other.py").write_text("VALUE = 3\n")
    (source / "alias.py").symlink_to("helper.py")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.email=test@example.com",
                    "-c", "user.name=Test", "commit", "-qm", "internal alias"], check=True)
    checkpoint = snapshot_checkpoint(str(source), "baseline-attempt")
    assert not (Path(checkpoint) / "alias.py").is_symlink()
    subprocess.run(["git", "clone", "-q", str(source), str(candidate)], check=True)
    (candidate / "component.py").write_text("COMPONENT = True\n")
    if change == "target_edit":
        (candidate / "helper.py").write_text("VALUE = 2\n")
    elif change in {"retarget", "escape"}:
        (candidate / "alias.py").unlink()  # Only replace this test-owned alias.
        destination = "other.py"
        if change == "escape":
            external = tmp_path / "external.py"
            external.write_text("PRIVATE = True\n")
            destination = str(external)
        (candidate / "alias.py").symlink_to(destination)
        with pytest.raises(ValueError, match="symlink|escapes repository"):
            _capture_component_diff(str(candidate), checkpoint)
        return
    delta = _capture_component_diff(str(candidate), checkpoint)
    assert "+COMPONENT = True" in delta
    assert (candidate / "alias.py").is_symlink()
    if change == "target_edit":
        assert "a/alias.py" in delta and "+VALUE = 2" in delta
    else:
        assert "alias.py" not in delta


def test_apply_prior_worktree_materializes_all_independent_parent_branches(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree

    base = tmp_path / "base"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.name", "Test"], check=True)
    (base / "shared.py").write_text("SHARED = 0\n")
    (base / "e.py").write_text("E = 0\n")
    (base / "f.py").write_text("F = 0\n")
    (base / "g.py").write_text("G = 0\n")
    subprocess.run(["git", "-C", str(base), "add", "."], check=True)
    subprocess.run(["git", "-C", str(base), "commit", "-qm", "base"], check=True)

    parents = []
    for name, file_name in (("e", "e.py"), ("f", "f.py"), ("g", "g.py")):
        parent = tmp_path / f"repo-{name}"
        subprocess.run(["git", "clone", "-q", str(base), str(parent)], check=True)
        (parent / "shared.py").write_text("SHARED = 1\n")
        (parent / file_name).write_text(f"{name.upper()} = 1\n")
        (parent / "analysis").mkdir()
        (parent / "analysis/check.json").write_text(json.dumps({"branch": name}))
        parents.append(str(parent))
    target = tmp_path / "repo-target"
    subprocess.run(["git", "clone", "-q", str(base), str(target)], check=True)

    assert _apply_prior_worktree(str(target), parents) == 4
    assert (target / "shared.py").read_text() == "SHARED = 1\n"
    assert (target / "e.py").read_text() == "E = 1\n"
    assert (target / "f.py").read_text() == "F = 1\n"
    assert (target / "g.py").read_text() == "G = 1\n"
    assert not (target / "analysis/check.json").exists()
    from autoresearch.experiments.repo_improve.agentic_improve import _capture_component_diff
    assert _capture_component_diff(str(target), parents) == ""
    (target / "join.py").write_text("JOIN = E + F + G\n")
    delta = _capture_component_diff(str(target), parents)
    assert "+JOIN = E + F + G" in delta
    assert "+SHARED" not in delta
    # JSON may also be implementation configuration: do not suppress its conflicts.
    for i, parent in enumerate(parents):
        (Path(parent) / "settings.json").write_text(json.dumps({"width": i + 1}))
    with pytest.raises(ValueError, match="dependency checkpoint conflict for settings.json"):
        _apply_prior_worktree(str(target), parents)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("change", ["implement", "delete", "conflict", "content_and_mode",
                                    "mode_conflict", "delete_and_mode"])
def test_apply_prior_worktree_compares_branches_to_shared_checkpoint(tmp_path, reverse, change):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree

    base = tmp_path / "base"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.name", "Test"], check=True)
    (base / "README.md").write_text("repository baseline\n")
    subprocess.run(["git", "-C", str(base), "add", "."], check=True)
    subprocess.run(["git", "-C", str(base), "commit", "-qm", "base"], check=True)
    ancestor, left, right, target = [tmp_path / name for name in
                                     ("ancestor", "left", "right", "target")]
    for path in (ancestor, left, right, target):
        subprocess.run(["git", "clone", "-q", str(base), str(path)], check=True)
    for path in (ancestor, left, right):
        (path / "shared.py").write_text("def transform(): pass\n")
    (left / "independent.py").write_text("LEFT = 1\n")
    if change == "delete":
        (right / "shared.py").unlink()
    else:
        (right / "shared.py").write_text("def transform(): return 1\n")
    if change == "conflict":
        (left / "shared.py").write_text("def transform(): return 2\n")
    if change in {"content_and_mode", "mode_conflict", "delete_and_mode"}:
        (ancestor / "shared.py").chmod(0o755)
        (left / "shared.py").chmod(0o777)
        (right / "shared.py").chmod(0o700 if change == "mode_conflict" else 0o777)
    if change == "delete_and_mode":
        (right / "shared.py").unlink()
    parents = [str(left), str(right)]
    if reverse:
        parents.reverse()
    if change == "implement":
        # Without host-provided ancestry, the merger must not guess which snapshot is newer.
        with pytest.raises(ValueError, match="dependency checkpoint conflict"):
            _apply_prior_worktree(str(target), parents)
    prior = {"checkpoints": parents, "merge_bases": [str(ancestor)]}
    if change in {"conflict", "mode_conflict", "delete_and_mode"}:
        with pytest.raises(ValueError, match="dependency checkpoint conflict"):
            _apply_prior_worktree(str(target), prior)
        return
    _apply_prior_worktree(str(target), prior)
    assert (target / "independent.py").read_text() == "LEFT = 1\n"
    if change == "delete":
        assert not (target / "shared.py").exists()
    else:
        assert (target / "shared.py").read_text() == "def transform(): return 1\n"
        if change == "content_and_mode":
            assert (target / "shared.py").stat().st_mode & 0o777 == 0o777


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_dependency_join_merges_disjoint_text_edits_only(tmp_path, reverse, overlap):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _apply_prior_worktree, _capture_component_diff,
    )

    base = tmp_path / "base"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.name", "Test"], check=True)
    original = "".join(f"VALUE_{i} = 0\n" for i in range(30))
    (base / "shared.py").write_text(original)
    subprocess.run(["git", "-C", str(base), "add", "."], check=True)
    subprocess.run(["git", "-C", str(base), "commit", "-qm", "base"], check=True)
    parents = []
    for i, line in enumerate([2, 15, 27]):
        parent = tmp_path / f"branch-{i}"
        subprocess.run(["git", "clone", "-q", str(base), str(parent)], check=True)
        if overlap:
            line = 2
        (parent / "shared.py").write_text(original.replace(f"VALUE_{line} = 0", f"VALUE_{line} = {i + 1}"))
        parents.append(str(parent))
    target = tmp_path / "target"
    subprocess.run(["git", "clone", "-q", str(base), str(target)], check=True)
    prior = {"checkpoints": parents[::-1] if reverse else parents, "merge_bases": [str(base)]}
    if overlap:
        with pytest.raises(ValueError, match="dependency checkpoint conflict"):
            _apply_prior_worktree(str(target), prior)
        assert (target / "shared.py").read_text() == original
        return
    assert _apply_prior_worktree(str(target), prior) == 1
    expected = original
    for i, line in enumerate([2, 15, 27]):
        expected = expected.replace(f"VALUE_{line} = 0", f"VALUE_{line} = {i + 1}")
    assert (target / "shared.py").read_text() == expected
    assert _capture_component_diff(str(target), prior) == ""


def test_apply_prior_worktree_ignores_untracked_symlinks_into_runtime_cache(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree

    source = tmp_path / "repo"
    target = tmp_path / "repo-next"
    cache = tmp_path / "shared-cache"
    source.mkdir()
    cache.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"],
                   check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "train.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "base"], check=True)
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True)

    (source / "train.py").write_text("VALUE = 2\n")
    (cache / "generated.py").write_text("CACHE = True\n")
    os.symlink(cache / "generated.py", source / "generated.py")

    assert _apply_prior_worktree(str(target), str(source)) == 1
    assert (target / "train.py").read_text() == "VALUE = 2\n"
    assert not (target / "generated.py").exists()


def test_dump_ablation_report_preserves_complete_structured_json():
    from autoresearch.experiments.repo_improve.agentic_improve import _dump_ablation_report

    report = [
        {
            "component": f"component-{index}",
            "type": "functional",
            "expected": "x" * 700,
            "actual": 0.0,
            "meets_expectation": False,
        }
        for index in range(8)
    ]

    dumped = _dump_ablation_report(report)

    assert len(dumped) > 4000
    assert json.loads(dumped) == report


class _FakeBackend:
    """Simulates the agent by writing a canned ar_result.json into <workroot>/repo/."""

    def __init__(self, *, result: dict | None = None, fail: str | None = None,
                 status="FINISHED", write_result=True, record_attempt=True,
                 measurement_stdout="metric=1.8 steps=1000\n", measurement_stderr="",
                 measurement_rc=0, attempt_argv=None,
                 default_candidate_valid=True, default_baseline_valid=True):
        self._result = result if result is not None else {
            "metric": "val_loss", "baseline": 1.80, "method": 1.60,
            "lower_is_better": True, "notes": "tuned lr"}
        self._fail = fail
        self._status = status
        self._write = write_result
        self._record_attempt = record_attempt
        self.task = None
        self.llm = "unset"
        self.timeout = "unset"
        self.closed = False
        self.opened = False
        self.measurement_stdout = measurement_stdout
        self.measurement_stderr = measurement_stderr
        self.measurement_rc = measurement_rc
        self.attempt_argv = attempt_argv
        self.default_candidate_valid = default_candidate_valid
        self.default_baseline_valid = default_baseline_valid
        self.executed = []
        self.attempt_executed = []

    def open_sandbox(self, *, cuda_device=None, mounts=None, env=None):
        if self._fail == "open":
            raise InfraError("no docker")
        self.opened = True
        self._mounts = mounts
        self._env = env
        return object()

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.task = task
        self.llm = llm
        self.timeout = timeout
        payload = json.loads(json.dumps(self._result))
        phase = self._env.get("AUTORESEARCH_EXECUTION_PHASE")
        if phase == "baseline":
            if self.default_baseline_valid:
                payload.setdefault("baseline_valid", True)
            supplied_measurement = payload.get("measurement")
            payload.setdefault("measurement", _measurement_spec())
            if payload.get("baseline") is not None:
                if supplied_measurement is None:
                    self.measurement_stdout = f"metric={payload['baseline']} steps=1000\n"
        elif phase == "method" and self._record_attempt:
            repo = Path(self._mounts[0]) / "repo"
            (repo / "README.md").write_text("method treatment\n", encoding="utf-8")
            if payload.get("method") is not None:
                self.measurement_stdout = f"metric={payload['method']} steps=1000\n"
        if (phase == "method" and self.default_candidate_valid
                and payload.get("method") is not None
                and "candidate_valid" not in payload):
            # Most canned method fixtures represent a completed official evaluator.
            # Tests for missing gate evidence opt out explicitly.
            payload["candidate_valid"] = True
        if self._record_attempt and phase in {"baseline", "method"}:
            emitted = dict(payload)
            emitted["measurements"] = {item["component"]: item["actual"]
                                       for item in payload.get("ablation_report", [])
                                       if isinstance(item, dict) and "actual" in item}
            self.attempt_stdout = self.measurement_stdout + "\n" + json.dumps(emitted) + "\n"
            repo = Path(self._mounts[0]) / "repo"
            guard = Path(self._env["PATH"].split(os.pathsep)[0])
            python_link = guard / "python"
            if not python_link.exists():
                python_link.symlink_to("/usr/bin/python3")
            launcher = guard / "ar-attempt"
            measurement = payload.get("measurement")
            declared_argv = ((measurement or {}).get("execution") or {}).get("argv")
            if not isinstance(declared_argv, list):
                declared_argv = None
            attempt_argv = list(self.attempt_argv or declared_argv or ["python", "train.py"])
            if attempt_argv and os.path.basename(attempt_argv[0]) == "torchrun":
                torchrun = guard / "torchrun"
                torchrun.write_text("#!/bin/sh\nexec python train.py\n", encoding="utf-8")
                torchrun.chmod(0o755)
            subprocess.run(
                [str(launcher), "timeout", "10s", *attempt_argv], env=self._env, cwd=repo,
                check=True, capture_output=True, text=True,
            )
            attempt_id = Path(self._env["AUTORESEARCH_LAST_ATTEMPT"]).read_text(
                encoding="utf-8",
            ).strip()
            payload["attempt_id"] = attempt_id
            if isinstance(payload.get("run_manifest"), dict):
                payload["run_manifest"]["attempt_id"] = attempt_id
        if self._write:
            # the "container" writes into the bind-mounted repo dir (host path == mount path)
            root = self._mounts[0]
            repo = Path(root) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "ar_result.json").write_text(json.dumps(payload), encoding="utf-8")
        # A TRANSIENT error at the very end of the agent session (glm hiccup, sandbox teardown
        # race) must NOT discard the baseline/method the agent ALREADY wrote to disk.
        self._agent_finished = True
        if self._fail == "run_after_write":
            raise RuntimeError("agent session ended with a transient error")
        if self._fail == "infra_after_write":
            raise InfraError("sandbox teardown race")
        return _Res(self._status)

    def close_sandbox(self, handle):
        self.closed = True

    def _execute_command(self, cmd, *, cwd, timeout, attempt):
        from types import SimpleNamespace
        (self.attempt_executed if attempt else self.executed).append(
            {"cmd": cmd, "cwd": cwd, "timeout": timeout})
        cmd_parts = shlex.split(cmd)
        if "train.py" in cmd_parts or "train" in cmd_parts:
            Path(cwd, "metrics.json").write_text("{}\n", encoding="utf-8")
            now = time.time_ns() / 1_000_000_000
            os.utime(Path(cwd, "metrics.json"), (now, now))
        return SimpleNamespace(
            exit_code=self.measurement_rc,
            stdout=getattr(self, "attempt_stdout", self.measurement_stdout) if attempt else self.measurement_stdout,
            stderr=self.measurement_stderr,
            timed_out=False,
        )

    def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        if "suffixes, names, blocked" in cmd or "source=data[:24000]" in cmd:
            # This host-owned filesystem handoff is not an evaluator invocation.
            proc = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True,
                                  text=True, timeout=timeout)
            from types import SimpleNamespace
            return SimpleNamespace(exit_code=proc.returncode, stdout=proc.stdout,
                                   stderr=proc.stderr)
        return self._execute_command(cmd, cwd=cwd, timeout=timeout, attempt=False)

    def execute_attempt_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        if getattr(self, "_agent_finished", False):
            # The host probe now uses the one-shot transport too. Keep its
            # independent canned output distinct from the agent's attempt echo.
            cmd = cmd.removeprefix("env AUTORESEARCH_HOST_NATIVE_EXECUTION=1 ")
            return self.execute_command(handle, cmd, cwd=cwd, timeout=timeout)
        return self._execute_command(cmd, cwd=cwd, timeout=timeout, attempt=True)


def _fake_clone(repo_url, dest):
    os.makedirs(dest, exist_ok=True)
    Path(dest, "README.md").write_text("repo\n", encoding="utf-8")
    Path(dest, "data_config.txt").write_text("canonical data\n", encoding="utf-8")
    Path(dest, "evaluate.py").write_text(
        "print('metric=1.8 steps=1000')\n", encoding="utf-8")
    Path(dest, "train.py").write_text(
        "from pathlib import Path\nPath('metrics.json').write_text('{}\\n')\n",
        encoding="utf-8")
    Path(dest, "PROTOCOL.md").write_text(
        "Official command: python train.py\n"
        "Trusted probe: prepare.evaluate_bpb prepare.py\n"
        "Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n",
        encoding="utf-8")
    subprocess.run(["git", "init", "-q", dest], check=True)
    subprocess.run(["git", "-C", dest, "config", "user.name", "Test User"], check=True)
    subprocess.run(
        ["git", "-C", dest, "config", "user.email", "test@example.invalid"], check=True,
    )
    subprocess.run(
        ["git", "-C", dest, "add", "README.md", "data_config.txt", "evaluate.py",
         "train.py", "PROTOCOL.md"],
        check=True,
    )
    subprocess.run(["git", "-C", dest, "commit", "-q", "-m", "baseline"], check=True)


def _nongit_fake_clone(repo_url, dest):
    os.makedirs(dest, exist_ok=True)
    Path(dest, "README.md").write_text("repo\n", encoding="utf-8")


def _benchmark_contract():
    return {
        "version": 1,
        "benchmark_kind": "official",
        "objective": {"metric": "val_loss", "direction": "minimize", "mode": "quality"},
        "protocol": {"kind": "fixed_work", "budget": {"steps": 1000}},
        "controls": {
            "resources": {"accelerator_count": 0},
            "data": {"dataset": "canonical", "split": "validation"},
            "evaluator": {"name": "canonical"},
            "workload": {"steps": 1000},
        },
        "allowed_treatment_axes": ["implementation"],
        "provenance": [{"source": "official_harness", "detail": "canonical script"}],
        "deviations": [],
    }


def _measurement_spec(value_pattern=r"metric=([0-9.]+)"):
    return {
        "execution": {"argv": ["python", "train.py"], "cwd": "."},
        "trusted_probe": {
            "kind": "external_scorer",
            "argv": ["python", "evaluate.py", "--frozen", "{artifact}"], "cwd": ".",
            "artifact": {"path": "metrics.json", "producer_sha256_before": None,
                         "required_new": True},
            "parser": {"stream": "stdout", "regex": value_pattern, "group": 1},
            "protocol_parser": {
                "stream": "stdout", "regex": r"steps=([0-9.]+)",
                "group": 1, "unit": "steps",
            },
        },
    }


def _v2_benchmark_contract(value=1.8):
    contract = _benchmark_contract()
    contract.update({
        "version": 2,
        "protocol": {"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        "observed_outcomes": {
            "quality": {"metric": "val_loss", "value": value}, "steps": 1000,
        },
        "verified_evidence": {
            "source": "agent proposal", "artifact": "pending host measurement",
            "observations": {"quality": {"metric": "val_loss", "value": value}},
        },
    })
    contract["controls"]["evaluator"].update({
        "immutable_paths": ["evaluate.py", "PROTOCOL.md"],
        "output_paths": ["metrics.json"],
        "measurement_spec": _measurement_spec(),
        "official_source_evidence": _official_evidence(_measurement_spec()),
    })
    contract["controls"]["data"]["immutable_paths"] = ["data_config.txt"]
    contract["controls"]["treatment"] = {"allowed_paths": ["README.md", "train.py"]}
    return contract


def _local_git_clone(tmp_path):
    """Return a no-network clone seam and the source repository's real HEAD."""
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test User"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True,
    )
    (source / "README.md").write_text("repo\n", encoding="utf-8")
    (source / "data_config.txt").write_text("canonical data\n", encoding="utf-8")
    (source / "evaluate.py").write_text("print('metric=1.8 steps=1000')\n", encoding="utf-8")
    (source / "PROTOCOL.md").write_text(
        "Official command: python train.py\n"
        "Trusted probe: prepare.evaluate_bpb prepare.py\n"
        "Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n",
        encoding="utf-8")
    (source / "prepare.py").write_text(
        "def evaluate_bpb():\n    return 1.75\n", encoding="utf-8")
    (source / "train.py").write_text(
        "from prepare import evaluate_bpb\n"
        "from pathlib import Path\n"
        "Path('metrics.json').write_text('{}\\n')\n"
        "print('editable fake metric=-999 steps=1')\n"
        "evaluate_bpb()\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(source), "add", "README.md", "data_config.txt", "evaluate.py",
         "PROTOCOL.md", "prepare.py", "train.py"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "baseline"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    def clone(_repo_url, dest):
        subprocess.run(["git", "clone", "-q", str(source), dest], check=True)

    return clone, commit


def _source_evidence(path: str, content: bytes, command: str) -> dict:
    return {
        "path": path, "line_start": 1, "line_end": len(content.splitlines()),
        "file_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "excerpt_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "command": command,
    }


def _canonical_sha(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _official_evidence(measurement, content=None) -> list[dict]:
    content = content or (
        b"Official command: python train.py\n"
        b"Trusted probe: prepare.evaluate_bpb prepare.py\n"
        b"Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n")
    base = {
        "path": "PROTOCOL.md", "line_start": 1,
        "line_end": len(content.splitlines()),
        "file_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "excerpt_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
    }
    execution_identity = measurement["execution"]
    probe_identity = measurement["trusted_probe"]
    return [
        {**base, "command": json.dumps(
            execution_identity, sort_keys=True, separators=(",", ":"))},
        {**base, "command": json.dumps(
            probe_identity, sort_keys=True, separators=(",", ":"))},
    ]


def _callable_measurement(argv=None, module="prepare", source_path="prepare.py") -> dict:
    return {
        "execution": {"argv": argv or ["python", "train.py"], "cwd": "."},
        "trusted_probe": {
            "kind": "python_callable", "module": module,
            "callable": "evaluate_bpb", "source_path": source_path,
        },
    }


def _external_measurement(execution_argv=None, protocol_regex=r"steps=([0-9.]+)") -> dict:
    return {
        "execution": {"argv": execution_argv or ["python", "train.py"], "cwd": "."},
        "trusted_probe": {
            "kind": "external_scorer",
            "argv": ["python", "evaluate.py", "--frozen", "{artifact}"], "cwd": ".",
            "artifact": {"path": "metrics.json", "producer_sha256_before": None,
                         "required_new": True},
            "parser": {"stream": "stdout", "regex": r"metric=([0-9.]+)", "group": 1},
            "protocol_parser": {
                "stream": "stdout", "regex": protocol_regex, "group": 1, "unit": "steps",
            },
        },
    }


def _round4_contract(measurement=None, *, protocol=None, command="python train.py") -> dict:
    measurement = measurement or _callable_measurement()
    contract = _v2_benchmark_contract(1.75)
    contract["protocol"] = protocol or {
        "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 0.01},
        "timing_scope": "process", "timer_authority": "host",
    }
    contract["controls"]["workload"] = {"training_time_s": 0.01}
    probe_source = ("evaluate.py"
                    if measurement["trusted_probe"]["kind"] == "external_scorer"
                    else "prepare.py")
    source_content = (
        f"Official command: {command}\n"
        "Trusted probe: prepare.evaluate_bpb prepare.py\n"
        "Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n"
    ).encode()
    contract["controls"]["evaluator"] = {
        "immutable_paths": [probe_source, "PROTOCOL.md"],
        "output_paths": ["metrics.json"],
        "measurement_spec": measurement,
        "official_source_evidence": _official_evidence(measurement, source_content),
    }
    return contract


def _clone_with_protocol_command(tmp_path, command):
    clone, _commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    (source / "PROTOCOL.md").write_text(
        f"Official command: {command}\n"
        "Trusted probe: prepare.evaluate_bpb prepare.py\n"
        "Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n",
        encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "PROTOCOL.md"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return clone, commit


class _LocalCommandBackend(_FakeBackend):
    """Keep agent I/O fake but execute the post-agent host probe in the local clone."""

    def _execute_local_command(self, cmd, *, cwd, timeout, attempt):
        from types import SimpleNamespace
        (self.attempt_executed if attempt else self.executed).append(
            {"cmd": cmd, "cwd": cwd, "timeout": timeout})
        completed = subprocess.run(
            cmd, shell=True, cwd=cwd, env=self._env,
            capture_output=True, text=True, timeout=timeout,
        )
        return SimpleNamespace(
            exit_code=completed.returncode, stdout=completed.stdout,
            stderr=completed.stderr, timed_out=False,
        )

    def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        if "suffixes, names, blocked" in cmd or "source=data[:24000]" in cmd:
            return super().execute_command(handle, cmd, cwd=cwd, timeout=timeout)
        return self._execute_local_command(cmd, cwd=cwd, timeout=timeout, attempt=False)

    def execute_attempt_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        if getattr(self, "_agent_finished", False):
            return self.execute_command(handle, cmd, cwd=cwd, timeout=timeout)
        return self._execute_local_command(cmd, cwd=cwd, timeout=timeout, attempt=True)


def _expected_immutable_hashes():
    return {
        "data_config.txt": "sha256:" + hashlib.sha256(b"canonical data\n").hexdigest(),
        "evaluate.py": "sha256:" + hashlib.sha256(
            b"print('metric=1.8 steps=1000')\n").hexdigest(),
        "PROTOCOL.md": "sha256:" + hashlib.sha256(
            b"Official command: python train.py\n"
            b"Trusted probe: prepare.evaluate_bpb prepare.py\n"
            b"Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n"
        ).hexdigest(),
    }


def _frozen_host_attestation(contract=None, measurement=None):
    contract = contract or _v2_benchmark_contract()
    measurement = measurement or contract["controls"]["evaluator"]["measurement_spec"]
    claims = contract["controls"]["evaluator"]["official_source_evidence"]
    verified_claims = []
    for role, identity, claim in zip(
            ("execution", "trusted_probe"),
            (measurement["execution"], measurement["trusted_probe"]), claims):
        verified_claims.append({
            "authority": "repository", "role": role,
            "identity_sha256": _canonical_sha(identity), **claim,
        })
    immutable_hashes = _expected_immutable_hashes()
    return {
        "measurement_spec": measurement,
        "immutable_file_hashes": immutable_hashes,
        "official_source_evidence": verified_claims,
        "resource_evidence": {
            "accelerator_count": 0, "device_models": [],
            "source": "allocation_count_only",
            "caveat": "precision, seed, and model scale are contract-frozen, not host-observed",
        },
        "actual_controls": {
            "version": 1,
            "execution": {
                **measurement["execution"], "source_paths": ["train.py"],
            },
            "immutable_file_hashes": immutable_hashes,
            "bindings": [],
            "caveat": (
                "Host observes canonical execution argv/cwd, resolved tracked sources, immutable "
                "file hashes, and declared structured bindings; it does not observe arbitrary "
                "internal runtime state or prove mathematical equivalence of undeclared "
                "batch/model/seed/precision controls."
            ),
        },
    }


def _callable_frozen_attestation(contract):
    attestation = _frozen_host_attestation(contract)
    attestation["immutable_file_hashes"] = {
        "data_config.txt": "sha256:" + hashlib.sha256(b"canonical data\n").hexdigest(),
        "prepare.py": "sha256:" + hashlib.sha256(
            b"def evaluate_bpb():\n    return 1.75\n").hexdigest(),
        "PROTOCOL.md": "sha256:" + hashlib.sha256(
            b"Official command: python train.py\n"
            b"Trusted probe: prepare.evaluate_bpb prepare.py\n"
            b"Trusted probe: external_scorer python evaluate.py --frozen {artifact}\n"
        ).hexdigest(),
    }
    attestation["actual_controls"]["immutable_file_hashes"] = dict(
        attestation["immutable_file_hashes"])
    return attestation


def test_happy_path_improvement(tmp_path):
    be = _FakeBackend()
    res = run_agentic_improve("lower val loss", repo_url="https://x/nanoGPT",
                              workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)
    assert res["benchmark"] == "nanoGPT"
    assert res["baseline"].metric_value == 1.80 and res["baseline"].valid
    assert res["candidate"].metric_value == 1.60 and res["candidate"].valid
    assert res["metric"] == "val_loss" and res["lower_is_better"] is True
    assert res["verdict"]["verdict"] == "pass" and res["verdict"]["improved"] is True
    assert be.closed and "error" not in res


def test_inherited_checkpoint_is_writable_before_agent_execution(tmp_path):
    parent = tmp_path / "parent"
    _fake_clone("unused", str(parent))
    (parent / "component").mkdir(mode=0o755)
    source = parent / "component" / "implementation.py"
    source.write_text("value = 1\n")
    source.chmod(0o644)
    observed = []

    class InspectBackend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            inherited = Path(self._mounts[0]) / "repo" / "component"
            observed.append((stat.S_IMODE(inherited.stat().st_mode),
                             stat.S_IMODE((inherited / "implementation.py").stat().st_mode)))
            return super().run_agent_task(handle, task, **kwargs)

    run_agentic_improve("improve the implementation", repo_url="https://example.org/repo",
                        workroot=str(tmp_path), backend=InspectBackend(),
                        clone_fn=lambda url, dest: subprocess.run(
                            ["git", "clone", "--quiet", str(parent), dest], check=True),
                        prior_worktree=str(parent))
    assert observed and observed[0][0] & 0o333 == 0o333
    assert observed[0][1] & 0o222 == 0o222
    assert stat.S_IMODE(source.stat().st_mode) == 0o644
    assert stat.S_IMODE(source.parent.stat().st_mode) == 0o755


def test_selected_task_manifest_resources_are_forwarded_to_sandbox(tmp_path):
    def clone_with_manifest(url, destination):
        _fake_clone(url, destination)
        task = Path(destination) / "tasks" / "target_task"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            "[environment]\ncpus = 2\nmemory_mb = 4096\n", encoding="utf-8")

    backend = _FakeBackend()

    run_agentic_improve(
        "Improve the official target_task benchmark",
        repo_url="https://x/benchmark",
        workroot=str(tmp_path),
        backend=backend,
        clone_fn=clone_with_manifest,
    )

    assert backend._env["AUTORESEARCH_TASK_CPUS"] == "2"
    assert backend._env["AUTORESEARCH_TASK_MEMORY_MB"] == "4096"


def test_declared_cuda_devel_toolkit_is_mounted_and_preflighted(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def clone_with_cuda_manifest(url, destination):
        _fake_clone(url, destination)
        task = Path(destination) / "tasks" / "target_task"
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_text(
            "[environment]\ncpus = 2\nmemory_mb = 4096\ngpus = 1\n",
            encoding="utf-8")
        (task / "environment" / "Dockerfile").write_text(
            "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04\n",
            encoding="utf-8")

    toolkit = tmp_path / "cuda-12.4.1"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.4, V12.4.131'\n")
    nvcc.chmod(0o755)
    (toolkit / "lib64").mkdir()
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_CUDA_ROOT", str(toolkit))

    class Backend(_FakeBackend):
        def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
            if cmd == "nvcc --version":
                return SimpleNamespace(
                    exit_code=0,
                    stdout="Cuda compilation tools, release 12.4, V12.4.131\n",
                    stderr="",
                    timed_out=False,
                )
            return super().execute_command(handle, cmd, cwd=cwd, timeout=timeout)

    backend = Backend()
    result = run_agentic_improve(
        "Improve the official target_task benchmark",
        repo_url="https://x/benchmark",
        workroot=str(tmp_path / "run"),
        backend=backend,
        clone_fn=clone_with_cuda_manifest,
        cuda_device=0,
    )

    assert "error" not in result
    assert (str(toolkit), str(toolkit) + ":ro") in backend._mounts
    assert backend._env["CUDA_HOME"] == str(toolkit)
    assert backend._env["CUDA_PATH"] == str(toolkit)
    assert backend._env["PATH"].split(os.pathsep)[1] == str(toolkit / "bin")
    assert backend._env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(toolkit / "lib64")


def test_declared_cuda_devel_fails_before_agent_without_matching_toolkit(
        tmp_path, monkeypatch):
    def clone_with_cuda_manifest(url, destination):
        _fake_clone(url, destination)
        task = Path(destination) / "tasks" / "target_task"
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_text("[environment]\ngpus = 1\n", encoding="utf-8")
        (task / "environment" / "Dockerfile").write_text(
            "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04\n",
            encoding="utf-8")

    monkeypatch.setenv("AUTORESEARCH_SANDBOX_CUDA_ROOT", str(tmp_path / "missing"))
    backend = _FakeBackend()
    result = run_agentic_improve(
        "Improve the official target_task benchmark",
        repo_url="https://x/benchmark",
        workroot=str(tmp_path / "run"),
        backend=backend,
        clone_fn=clone_with_cuda_manifest,
        cuda_device=0,
    )

    assert not backend.opened
    assert "requires CUDA 12.4" in result["error"]
    assert "AUTORESEARCH_SANDBOX_CUDA_ROOT" in result["error"]


def test_declared_cuda_devel_is_provisioned_after_initial_download_authorization(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve import agentic_improve

    def clone_with_cuda_manifest(url, destination):
        _fake_clone(url, destination)
        task = Path(destination) / "tasks" / "target_task"
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_text(
            "[environment]\ngpus = 1\nallow_internet = false\n", encoding="utf-8")
        (task / "environment" / "Dockerfile").write_text(
            "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04\n", encoding="utf-8")

    toolkit = tmp_path / "cache" / "cuda-12.4.1"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.4, V12.4.131'\n")
    nvcc.chmod(0o755)
    monkeypatch.setattr(
        agentic_improve, "_matching_cuda_toolkit_root", lambda version: (None, None))
    calls = []
    monkeypatch.setattr(
        agentic_improve, "_provision_cuda_toolkit",
        lambda image, version, workroot, emit: (
            calls.append((image, version, workroot)) or (str(toolkit), "12.4")),
        raising=False)

    class Backend(_FakeBackend):
        def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
            if cmd == "nvcc --version":
                return SimpleNamespace(
                    exit_code=0,
                    stdout="Cuda compilation tools, release 12.4, V12.4.131\n",
                    stderr="",
                    timed_out=False,
                )
            return super().execute_command(handle, cmd, cwd=cwd, timeout=timeout)

    backend = Backend()
    run_root = tmp_path / "run"
    result = run_agentic_improve(
        "Improve the official target_task benchmark",
        repo_url="https://x/benchmark",
        workroot=str(run_root),
        backend=backend,
        clone_fn=clone_with_cuda_manifest,
        cuda_device=0,
        confirmed_boundary={"download_authorization": {"approved": True}},
    )

    assert "error" not in result
    assert calls == [("nvidia/cuda:12.4.1-devel-ubuntu22.04", "12.4.1", str(run_root))]
    assert (str(toolkit), str(toolkit) + ":ro") in backend._mounts


def test_cuda_provisioner_uses_declared_image_and_shared_data_root(tmp_path, monkeypatch):
    import subprocess
    from autoresearch.experiments.repo_improve import agentic_improve

    data_root = tmp_path / "shared-data"
    data_root.mkdir()
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_DATA_ROOT", str(data_root))
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_CUDA_ROOT", raising=False)
    monkeypatch.setattr(agentic_improve.shutil, "which", lambda name: "docker")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        if str(argv[0]).endswith("/bin/nvcc") and Path(argv[0]).is_file():
            return subprocess.CompletedProcess(
                argv, 0, "Cuda compilation tools, release 12.4, V12.4.131\n", "")
        if str(argv[0]).endswith("/bin/nvcc"):
            raise FileNotFoundError(argv[0])
        if argv[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(argv, 0, "container-id\n", "")
        if argv[:2] == ["docker", "cp"]:
            destination = Path(argv[-1])
            (destination / "bin").mkdir(parents=True)
            nvcc = destination / "bin" / "nvcc"
            nvcc.write_text(
                "#!/bin/sh\necho 'Cuda compilation tools, release 12.4, V12.4.131'\n")
            nvcc.chmod(0o755)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                argv, 0, "nvidia/cuda@sha256:0123456789abcdef\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(agentic_improve.subprocess, "run", fake_run)
    emitted = []
    root, release = agentic_improve._provision_cuda_toolkit(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", "12.4.1",
        str(tmp_path / "run"), lambda event, payload: emitted.append((event, payload)))

    assert release == "12.4"
    assert Path(root).parent == data_root / "toolchains"
    assert stat.S_IMODE(Path(root).stat().st_mode) == 0o755
    pull = next(item for item in calls if item[0][:2] == ["docker", "pull"])
    assert pull[0] == [
        "docker", "pull", "--platform", "linux/amd64",
        "nvidia/cuda:12.4.1-devel-ubuntu22.04"]
    assert not any(key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
                   for key in pull[1])
    assert (Path(root) / "source-image.txt").read_text().strip() == (
        "nvidia/cuda:12.4.1-devel-ubuntu22.04")
    assert any(event == "agentic_cuda_provision" and payload["stage"] == "ready"
               for event, payload in emitted)


def test_method_resources_follow_frozen_baseline_task(tmp_path):
    def clone_with_manifests(url, destination):
        _fake_clone(url, destination)
        for name, cpus, memory in (("baseline_task", 8, 32768),
                                   ("mentioned_task", 2, 4096)):
            task = Path(destination) / "tasks" / name
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                f"[environment]\ncpus = {cpus}\nmemory_mb = {memory}\n",
                encoding="utf-8")

    backend = _FakeBackend()
    frozen = {
        "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
        "repository_contract": {"sources": [{
            "path": "tasks/baseline_task/task.toml",
            "content": "[environment]\ncpus = 8\nmemory_mb = 32768\n",
        }]},
    }

    run_agentic_improve(
        "Improve mentioned_task using the frozen baseline task",
        repo_url="https://x/benchmark", workroot=str(tmp_path),
        backend=backend, clone_fn=clone_with_manifests,
        execution_phase="method", frozen_baseline=frozen,
        enforce_fairness=False,
    )

    assert backend._env["AUTORESEARCH_TASK_CPUS"] == "8"
    assert backend._env["AUTORESEARCH_TASK_MEMORY_MB"] == "32768"


@pytest.mark.parametrize("candidate_size", [100, 24000])
def test_legacy_method_recovers_task_limits_despite_named_candidate_source(
        tmp_path, monkeypatch, candidate_size):
    from autoresearch.agents._reuse.openhands_backend import _sandbox_extra_args

    source = tmp_path / "source"
    _fake_clone("unused", str(source))
    task = source / "tasks" / "target_task"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text(
        "[environment]\ncpus = 1\nmemory_mb = 512\n")
    (task / "instruction.md").write_text("Edit environment/solve.py only.\n")
    (task / "environment" / "solve.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "task"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    frozen = {"metric_value": 1.8, "repo_commit": commit, "repository_contract": {
        "sources": [], "execution_contract": {
            "experiment_provenance": {"task": "target_task"}}}}
    before = json.dumps(frozen, sort_keys=True)

    def clone_candidate(url, destination):
        subprocess.run(["git", "clone", "--quiet", str(source), destination], check=True)
        (Path(destination) / "tasks/target_task/environment/solve.py").write_text(
            "value = 2\n#" + "x" * candidate_size)

    monkeypatch.setenv("AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB", "512")
    backend = _FakeBackend()
    run_agentic_improve(
        "Improve target_task; inspect tasks/target_task/environment/solve.py",
        repo_url="https://x/benchmark", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone_candidate,
        execution_phase="method", frozen_baseline=frozen, enforce_fairness=False)

    assert backend._env["AUTORESEARCH_TASK_CPUS"] == "1"
    assert backend._env["AUTORESEARCH_TASK_MEMORY_MB"] == "512"
    args = _sandbox_extra_args(env=backend._env)
    assert args[args.index("--cpus") + 1] == "1.0"
    assert args[args.index("--memory") + 1] == "1024m"
    assert json.dumps(frozen, sort_keys=True) == before


@pytest.mark.parametrize("with_commit", [False, True])
def test_method_uses_frozen_task_contract_when_request_mentions_another_task(
        tmp_path, monkeypatch, with_commit):
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    monkeypatch.setenv("GIT_AUTHOR_DATE", "2000-01-01T00:00:00Z")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2000-01-01T00:00:00Z")

    def clone_with_conflicting_tasks(url, destination):
        _fake_clone(url, destination)
        for name, metric in (("alpha_task", "latency"), ("omega_task", "accuracy")):
            task = Path(destination) / "tasks" / name
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                f'[optimization]\nmetric = "{metric}"\ndirection = "lower"\n')

    frozen_contract = {"sources": [{
        "path": "tasks/alpha_task/task.toml",
        "content": '[optimization]\nmetric = "latency"\ndirection = "lower"\n',
    }], "execution_contract": {"evaluation": {
        "metric_name": "latency", "lower_is_better": True}}}
    request = "Improve alpha_task; compare omega_task only as background context"
    probe = tmp_path / "probe"
    clone_with_conflicting_tasks("https://x/benchmark", str(probe))
    commit = subprocess.check_output(
        ["git", "-C", str(probe), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(ValueError, match="conflicting optimization objectives"):
        _repository_task_contract(str(probe), request)
    backend = _FakeBackend(write_result=False)
    result = run_agentic_improve(
        request,
        repo_url="https://x/benchmark", workroot=str(tmp_path),
        backend=backend, clone_fn=clone_with_conflicting_tasks,
        execution_phase="method", frozen_baseline={
            "metric_value": 2.0, "metric_name": "latency",
            "lower_is_better": True, "repository_contract": frozen_contract,
            "repo_commit": commit if with_commit else None},
        enforce_fairness=False)

    assert backend.opened
    assert "conflicting optimization objectives" not in str(result)


def test_combined_phase_hides_raw_agent_metadata(tmp_path):
    """Raw metadata preserved for phase validation is not part of the result contract."""
    be = _FakeBackend()

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone,
    )

    assert "_reported_metric" not in res
    assert "_reported_lower_is_better" not in res


def test_none_budget_still_passes_finite_hard_timeout(tmp_path, monkeypatch):
    # #46: budget_s=None must NOT reach run_agent_task as timeout=None (which runs the agent inline
    # with no reliable backstop -> a dead-connection poll hangs the orchestrator forever). A finite
    # hard cap must always be handed down so the daemon-thread timeout can abandon a hung run.
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "7200")
    be = _FakeBackend()
    run_agentic_improve("lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone, budget_s=None)
    assert be.timeout == 7200            # finite hard cap, not None


def test_repository_run_does_not_inject_path_name_access_policy(tmp_path):
    class RecordingBackend(_FakeBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None, **kwargs):
            self.extra_kwargs = kwargs
            return super().run_agent_task(handle, task, llm=llm, timeout=timeout)

    be = RecordingBackend()
    run_agentic_improve(
        "improve the official task",
        repo_url="https://github.com/example/official-benchmark/",
        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
    )

    assert be.extra_kwargs == {}


def test_explicit_budget_keeps_a_separate_result_finalization_window(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_FINALIZE_RESERVE_S", "120")
    be = _FakeBackend()
    run_agentic_improve("lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone, budget_s=900)
    assert be.timeout == 1020
    prompt = _method_task_prompt(
        "lower val loss", "/repo", "/repo/ar_result.json", 900,
        {"metric_name": "val_loss", "metric_value": 1.8,
         "lower_is_better": True, "fixed_config": "canonical"},
    )
    assert "~900s experiment budget" in prompt
    assert "final 120s are reserved only for writing" in prompt


def test_per_run_cap_env_tunable_and_rendered_in_prompt(tmp_path, monkeypatch):
    # 老师 2026-08-05: per-run wall-clock enlarged + env-tunable (was a hardcoded 300s, too small for
    # heavy repos). The prompt the agent gets must carry the configured cap.
    from autoresearch.experiments.repo_improve.agentic_improve import _per_run_cap, _task_prompt
    monkeypatch.delenv("AUTORESEARCH_AGENTIC_PER_RUN_S", raising=False)
    assert _per_run_cap() == 7200
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "invalid")
    assert _per_run_cap() == 7200
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "2400")
    assert _per_run_cap() == 2400
    prompt = _task_prompt("lower loss", "/repo", "/repo/ar_result.json", None, "val_loss", True)
    assert "timeout --foreground 2400s" in prompt and "300s" not in prompt
    assert '"expected":{"operator":"<|<=|>|>=|==","value":<finite number>}' in prompt
    assert '"expected": "<the component contract' not in prompt


@pytest.mark.parametrize("budget", [None, 900])
def test_explicit_unlimited_run_preserves_a_supplied_budget(tmp_path, monkeypatch, budget):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT", "1")
    be = _FakeBackend()
    run_agentic_improve("lower loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone, budget_s=budget)
    assert be.timeout is None if budget is None else be.timeout >= budget
    prompt = _method_task_prompt("lower loss", "/repo", "/repo/ar_result.json", budget,
        {"metric_name": "loss", "metric_value": 1.8, "lower_is_better": True})
    assert ("No framework total wall-clock limit" in prompt) == (budget is None)
    assert ("~900s experiment budget" in prompt) == (budget is not None)


def test_higher_is_better_improvement(tmp_path):
    be = _FakeBackend(result={"metric": "accuracy", "baseline": 0.90, "method": 0.94,
                              "lower_is_better": False, "notes": "x"})
    res = run_agentic_improve("raise accuracy", repo_url="https://x/cifar",
                              workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)
    assert res["verdict"]["improved"] is True and res["lower_is_better"] is False


def test_no_improvement_is_fail_but_keeps_values(tmp_path):
    be = _FakeBackend(result={"metric": "val_loss", "baseline": 1.6, "method": 1.7,
                              "lower_is_better": True, "notes": "worse"})
    res = run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                              backend=be, clone_fn=_fake_clone)
    assert res["verdict"]["improved"] is False
    assert res["baseline"].metric_value == 1.6 and res["candidate"].metric_value == 1.7


def test_baseline_only_method_null(tmp_path):
    be = _FakeBackend(result={"metric": "latency_ms", "baseline": 12.0, "method": None,
                              "lower_is_better": True, "notes": "couldn't improve"})
    res = run_agentic_improve("q", repo_url="https://x/SOL", workroot=str(tmp_path),
                              backend=be, clone_fn=_fake_clone)
    assert res["baseline"].metric_value == 12.0 and res["baseline"].valid
    assert res["candidate"].metric_value is None and res["candidate"].valid is False
    assert res["verdict"]["improved"] is False


def test_baseline_phase_accepts_real_baseline_without_candidate(tmp_path):
    """Catch a baseline-only phase being treated as a failed method experiment."""
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000", "notes": "pristine",
        "benchmark_contract": _v2_benchmark_contract(),
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
    )

    assert res["baseline"].metric_value == 1.8
    assert res["baseline"].valid is True
    assert res["candidate"].valid is False
    assert res["candidate"].metric_value is None
    assert res["verdict"] == {
        "verdict": "pass", "improved": False, "reason": "baseline measured: val_loss=1.8",
    }
    assert res["config"] == "steps=1000"
    assert res["benchmark_contract"]["selected_source"] == "official_harness"
    assert res["contract_fingerprint"].startswith("sha256:")
    assert "METHOD EXPERIMENTS ARE OUT OF SCOPE" in be.task


def test_baseline_phase_rejects_missing_benchmark_contract(tmp_path):
    """Catch a real number being frozen without a complete fairness definition."""
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000", "notes": "pristine",
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "baseline_contract_invalid"


def test_production_baseline_cannot_freeze_an_unexecuted_self_report(tmp_path):
    be = _FakeBackend(result={
        "metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
        "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000", "notes": "measured",
    }, record_attempt=False)

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        enforce_fairness=False,
    )

    assert res["baseline"].metric_value == 1.8
    assert res["baseline"].valid is False
    assert "host attempt" in res["error"]
    assert res["metric_unit"] == "nats"
    assert res["candidate"].metric_value is None
    assert res.get("benchmark_contract") is None
    assert res.get("host_attestation") is None


@pytest.mark.parametrize("gate", [False, None])
def test_finite_baseline_with_failed_or_missing_correctness_cannot_be_frozen(tmp_path, gate):
    payload = {"metric": {"name": "latency", "unit": "ms", "lower_is_better": True},
               "baseline": 0.8, "method": None}
    if gate is not None:
        payload["baseline_valid"] = gate
    be = _FakeBackend(result=payload, record_attempt=False, default_baseline_valid=False)
    result = run_agentic_improve("improve", repo_url="https://example.test/repo",
        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
        execution_phase="baseline", enforce_fairness=False)
    assert result["baseline"].metric_value == 0.8
    assert result["baseline"].valid is False
    assert result["diagnosis"] == "correctness_gate_failed"


@pytest.mark.parametrize("commit_field", ["repo_commit", "commit"])
@pytest.mark.parametrize("reported_commit,valid", [
    ("3" * 40, False), ("7" * 40, True), ("7" * 7, True), (None, True),
])
def test_baseline_reported_repository_commit_must_match_host_checkout(
        tmp_path, reported_commit, valid, commit_field):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _build_result_from_file, _finalize_relaxed_phase_result)
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({
        "metric": {"name": "latency", "unit": "ms", "lower_is_better": True},
        "baseline": 2.0, "baseline_valid": True, "method": None,
        "config": {commit_field: reported_commit},
    }))
    result = _build_result_from_file(str(path), "project", None, None,
                                     lambda *args: None, "baseline")
    result["repo_commit"] = "7" * 40
    result = _finalize_relaxed_phase_result(result, "baseline", None)
    assert result["baseline"].metric_value == 2.0
    assert result["baseline"].valid is valid
    if not valid:
        assert result["diagnosis"] == "baseline_provenance_violation"
        assert "reported repository commit" in result["error"]
        assert "nested dependency" in result["error"]
        assert "Do not change HEAD" in result["error"]


@pytest.mark.parametrize("reported_commit", ["3" * 40, "7" * 40, "7" * 7, None])
def test_baseline_completion_repairs_repository_identity_before_closing(
        tmp_path, monkeypatch, reported_commit):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.tests.test_historical_component_evidence import receipt

    path = tmp_path / "ar_result.json"
    payload = {"metric": {"name": "latency", "unit": "ms", "lower_is_better": True},
               "baseline": 2.0, "baseline_valid": True, "method": None,
               "attempt_id": "attempt-measured", "config": {"repo_commit": reported_commit}}
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    record = receipt(tmp_path, "attempt-measured", tmp_path,
                     {"baseline": 2.0, "baseline_valid": True}, phase="baseline")
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"attempt-measured": record})
    monkeypatch.setattr(module, "_repo_commit", lambda _: "7" * 40)
    feedback = module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl")
    if reported_commit == "3" * 40:
        assert "reported repository commit" in feedback
        assert "nested dependency" in feedback
        assert "Do not change HEAD or rerun training" in feedback
    else:
        assert feedback is None
    assert path.read_bytes() == original


def test_baseline_prompt_separates_selected_and_nested_repository_identity():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt("reproduce the selected task", "/selected",
                                   "/selected/ar_result.json", "accuracy", False)
    assert "config.repo_commit" in prompt
    assert "git -C /selected rev-parse HEAD" in prompt
    assert "nested dependency" in prompt
    assert "copy its exact metric identifier into metric.name" in prompt


def test_baseline_gate_cannot_borrow_candidate_validity():
    from autoresearch.experiments.repo_improve.agentic_improve import _explicit_baseline_validity
    assert _explicit_baseline_validity({"candidate_valid": True}, "baseline") is None
    assert _explicit_baseline_validity({"baseline_valid": True,
        "baseline": {"valid": False}}, "baseline") is False


def test_baseline_validity_includes_requested_task_not_only_repository_correctness(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt("evaluate the requested domain", "/repo",
                                   "/repo/ar_result.json", "latency", True)
    assert "actual measured run satisfies the explicit user requirements" in prompt
    assert "Reporting a deviation does not authorize it" in prompt

    backend = _FakeBackend(result={
        "metric": {"name": "latency", "unit": "ms", "lower_is_better": True},
        "baseline": 0.8, "baseline_valid": False, "method": None,
        "notes": "Repository correctness passed, but the requested domain is unsupported; "
                 "this measurement uses another domain and is not a qualifying baseline.",
    }, record_attempt=False, default_baseline_valid=False)
    result = run_agentic_improve("evaluate the requested domain",
        repo_url="https://example.test/repo", workroot=str(tmp_path),
        backend=backend, clone_fn=_fake_clone, execution_phase="baseline",
        enforce_fairness=False)
    assert result["baseline"].metric_value == 0.8
    assert result["baseline"].valid is False


def test_checkpoint_handoff_preserves_private_cache_and_carries_build_sources(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _prepare_checkpoint_handoff, _dependency_worktree_changes)
    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    (repo / "src").mkdir(mode=0o700)
    source = repo / "src" / "component.hxx"
    source.write_text("int component = 1;\n")
    source.chmod(0o600)
    makefile = repo / "Makefile"
    makefile.write_text("all:\n\ttrue\n")
    makefile.chmod(0o600)
    (repo / "analysis").mkdir(mode=0o700)
    scan = repo / "analysis/scan.py"
    scan.write_text("def scan():\n    return 1\n")
    scan.chmod(0o600)
    old_result = repo / "analysis/result.json"
    old_result.write_text('{"actual": 1}')
    old_result.chmod(0o600)
    secret = tmp_path / "shared.py"
    secret.write_text("private cache")
    secret.chmod(0o600)
    (repo / "linked.py").symlink_to(secret)
    be = _FakeBackend()
    errors = []
    _prepare_checkpoint_handoff(be, object(), str(repo), lambda *args: errors.append(args))
    assert not errors
    assert stat.S_IMODE(source.stat().st_mode) == 0o644
    assert stat.S_IMODE(source.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert stat.S_IMODE(scan.stat().st_mode) == 0o644
    assert stat.S_IMODE(old_result.stat().st_mode) == 0o600
    target = tmp_path / "repo-next"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    (repo / "linked.py").unlink()  # test-owned symlink: outside cache remains untouched
    changes = _dependency_worktree_changes(str(target), str(repo))
    assert "src/component.hxx" in changes and "Makefile" in changes
    assert "analysis/scan.py" in changes and "analysis/result.json" not in changes


def test_dependency_checkpoint_allows_large_repository_scan(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as core

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    target = tmp_path / "repo-next"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    original_run = core.subprocess.run
    observed_timeouts = {}

    def record_timeout(command, *args, **kwargs):
        if "ls-files" in command and "--others" in command:
            observed_timeouts["untracked"] = kwargs.get("timeout")
        if "diff" in command and "--name-only" in command:
            observed_timeouts["tracked"] = kwargs.get("timeout")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(core.subprocess, "run", record_timeout)
    core._dependency_worktree_changes(str(target), str(repo))
    assert observed_timeouts == {
        "tracked": core._GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS,
        "untracked": core._GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS,
    }
    assert set(observed_timeouts.values()) == {600}


def test_dependency_checkpoint_imports_immutable_snapshot_from_recovered_workroot(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import (
        snapshot_checkpoint, validate_checkpoint)

    old_root = tmp_path / "old-controller"
    old_root.mkdir()
    source = old_root / "repo"
    _fake_clone("unused", str(source))
    (source / "restored.py").write_text("VALUE = 'retained'\n")
    checkpoint = snapshot_checkpoint(str(source), "retained-result")

    new_root = tmp_path / "new-controller"
    new_root.mkdir()
    target = new_root / "repo"
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True)

    assert _apply_prior_worktree(str(target), checkpoint) == 1
    imported = new_root / Path(checkpoint).name
    assert imported.is_dir()
    assert validate_checkpoint(imported) == validate_checkpoint(checkpoint)
    assert (target / "restored.py").read_text() == "VALUE = 'retained'\n"


def test_tracked_repository_fingerprint_allows_large_repository_scan(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as core

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    original_run = core.subprocess.run
    observed_timeout = None

    def record_timeout(command, *args, **kwargs):
        nonlocal observed_timeout
        if "diff" in command and "--binary" in command:
            observed_timeout = kwargs.get("timeout")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(core.subprocess, "run", record_timeout)
    assert core._tracked_repo_state(str(repo))["tracked_diff_hash"]
    assert observed_timeout == core._GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS == 600


@pytest.mark.parametrize("parents", [[], None])
@pytest.mark.parametrize("initial_capture_fails", [False, True])
def test_root_component_diff_keeps_source_beyond_cumulative_preview(
        tmp_path, parents, initial_capture_fails, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as core

    original_capture = core._capture_component_diff
    calls = []

    def capture(repo, dependencies):
        calls.append(repo)
        if initial_capture_fails and len(calls) == 1:
            raise ValueError("live alias cannot be expanded before checkpoint materialization")
        return original_capture(repo, dependencies)

    monkeypatch.setattr(core, "_capture_component_diff", capture)
    class SourceBackend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            repo = Path(self._mounts[0]) / "repo"
            library = repo / "method" / "library"
            library.mkdir(parents=True)
            for index in range(25):
                (library / f"a{index:02}.hpp").write_text(f"int helper_{index};\n")
            (library / "z_component.cuh").write_text("int actual_component() { return 42; }\n")
            return super().run_agent_task(handle, task, **kwargs)
    result = run_agentic_improve(
        "improve component", repo_url="https://example.invalid/repo", workroot=str(tmp_path),
        backend=SourceBackend(), clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline={"metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True},
        enforce_fairness=False, component_parent_worktree=parents)
    assert "actual_component" in result["diff"]  # Later source files are no longer silently omitted.
    assert result.get("checkpoint")
    assert ".autoresearch-checkpoint.json" not in result["diff"]
    if parents == []:
        assert len(calls) == 2
        assert "actual_component" in result["parent_relative_diff"]
        assert "z_component.cuh" in result["parent_relative_diff"]
        assert result["parent_diff_error"] is None
    else:
        assert result["parent_relative_diff"] is None


def test_integration_preserves_delta_against_selected_component_source(tmp_path):
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint

    parent = tmp_path / "accepted"
    _fake_clone("unused", str(parent))
    (parent / "settings.py").write_text("WINDOW = 128\n")
    checkpoint = snapshot_checkpoint(str(parent), "qualified-parent")

    def clone(_url, dest):
        subprocess.run(['git', 'clone', '--quiet', str(parent), dest], check=True)

    class Backend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            repo = Path(self._mounts[0]) / "repo"
            assert (repo / "settings.py").read_text() == "WINDOW = 128\n"
            (repo / "settings.py").write_text("WINDOW = 64\n")
            return super().run_agent_task(handle, task, **kwargs)

    result = run_agentic_improve(
        "integrate qualified components", repo_url="https://example.invalid/repo",
        workroot=str(tmp_path), backend=Backend(), clone_fn=clone,
        execution_phase="method", enforce_fairness=False,
        frozen_baseline={"metric_value": 1.8}, prior_worktree=[checkpoint],
        component_parent_worktree=[], component_stage=False)

    assert result.get("checkpoint"), result.get("error")
    assert "-WINDOW = 128" in result["starting_point_diff"]
    assert "+WINDOW = 64" in result["starting_point_diff"]
    assert "-WINDOW = 128" not in result["parent_relative_diff"]
    assert result["starting_point_diff_error"] is None


@pytest.mark.parametrize("parent_kind", ["explicit", "root", "omitted", "empty_legacy"])
@pytest.mark.parametrize("source_path", ["component.py", "workspace/component.py"])
def test_parent_control_fingerprint_includes_restored_new_sources(
        tmp_path, monkeypatch, parent_kind, source_path):
    from autoresearch.experiments.repo_improve import agentic_improve as core
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint

    parent = tmp_path / "parent"
    _fake_clone("unused", str(parent))
    (parent / ".gitignore").write_text("workspace/\n")
    subprocess.run(["git", "-C", str(parent), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(parent), "commit", "-qm", "ignore runtime workspace"],
                   check=True)
    (parent / source_path).parent.mkdir(parents=True, exist_ok=True)
    (parent / source_path).write_text("def component(): return 0\n")
    subprocess.run(["git", "-C", str(parent), "add", "--force", source_path], check=True)
    (parent / "workspace").mkdir(exist_ok=True)
    (parent / "workspace/runtime.pt").write_bytes(b"runtime artifact")
    baseline_checkpoint = snapshot_checkpoint(
        str(parent), "baseline-control", artifacts=["workspace/runtime.pt"])
    (parent / source_path).write_text("def component(): return 1\n")
    subprocess.run(["git", "-C", str(parent), "add", "--force", source_path], check=True)
    observed = {}
    original_start = core._AttemptAuthority.start

    def inspect_start(authority):
        original_start(authority)
        assert authority.frozen_baseline_outcome == (
            None if parent_kind == "explicit" else {"feasible": True})
        control = authority.control_path
        observed["expected"] = authority.dependency_parent_state["tracked_diff_hash_start"]
        observed["content"] = Path(control, source_path).read_text()
        if parent_kind != "explicit":
            assert Path(control, "workspace/runtime.pt").is_file()
        assert subprocess.run(["git", "-C", control, "ls-files", "--error-unmatch",
                               "workspace/runtime.pt"], capture_output=True).returncode != 0
        # Staging the identical restored files must not invalidate the parent.
        subprocess.run(["git", "-C", control, "add", "--force", "--", source_path], check=True)
        observed["restaged"] = core._tracked_repo_state(control)["tracked_diff_hash"]
        # Real edits must still change the fingerprint.
        Path(control, source_path).write_text("def component(): return 2\n")
        observed["edited"] = core._tracked_repo_state(control)["tracked_diff_hash"]

    monkeypatch.setattr(core._AttemptAuthority, "start", inspect_start)
    original_run = subprocess.run
    control_clone_timeouts = []
    control_clone_umasks = []
    control_stage_commands = []

    def record_control_clone_timeout(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", [])
        if command[:4] == ["git", "clone", "--shared", "--quiet"]:
            control_clone_timeouts.append(kwargs.get("timeout"))
            control_clone_umasks.append(kwargs.get("umask"))
        if ("-C" in command and "add" in command
                and any(str(part).startswith(str(tmp_path / "control-"))
                        for part in command)):
            control_stage_commands.append(command)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(core.subprocess, "run", record_control_clone_timeout)

    def clone_parent(_url, dest):
        original_run(["git", "clone", "--quiet", str(parent), dest], check=True)

    backend = _FakeBackend()
    result = run_agentic_improve(
        "improve component", repo_url="https://example.invalid/repo",
        workroot=str(tmp_path), backend=backend, clone_fn=clone_parent,
        execution_phase="method", enforce_fairness=False,
        frozen_baseline={"metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
                         "checkpoint": baseline_checkpoint,
                         "measurement_evidence": {"gates": {"/baseline_valid": True},
                             "baseline_outcome": {"feasible": False, "official_score": 0}}},
        component_parent_worktree={"explicit": str(parent), "root": [], "omitted": None,
                                  "empty_legacy": ""}[parent_kind])
    assert "content" in observed, result.get("error")
    expected_value = 1 if parent_kind == "explicit" else 0
    assert observed["content"] == f"def component(): return {expected_value}\n"
    assert observed["expected"] == observed["restaged"]
    assert observed["expected"] != observed["edited"]
    assert "prepare the same pinned dependencies there without overwriting inherited source" in backend.task
    assert "Change directory to the control checkout before launching ar-attempt" in backend.task
    assert "A cd inside the child wrapper does not change the host-recorded launch directory" in backend.task
    assert "do not substitute candidate code for the parent" in backend.task
    assert control_clone_timeouts == [600]
    assert control_clone_umasks == [0]
    assert all(command[-1] != "--all" for command in control_stage_commands)
    assert all("--" in command and source_path in command[command.index("--") + 1:]
               for command in control_stage_commands)


@pytest.mark.parametrize('matching_commit', [False, True])
def test_method_exports_pinned_baseline_without_legacy_checkpoint(tmp_path, matching_commit):
    origin = tmp_path / 'origin'
    _fake_clone('unused', str(origin))
    commit = subprocess.check_output(['git', '-C', str(origin), 'rev-parse', 'HEAD'], text=True).strip()

    def clone(_url, dest):
        subprocess.run(['git', 'clone', '--quiet', str(origin), dest], check=True)

    result = run_agentic_improve(
        'improve the candidate', repo_url='https://example.invalid/repo',
        workroot=str(tmp_path / 'session'), backend=_FakeBackend(), clone_fn=clone,
        execution_phase='method', enforce_fairness=False,
        frozen_baseline={'metric_value': 1.8, 'repo_commit': commit if matching_commit else '0' * 40})

    if matching_commit:
        assert '-repo\n+method treatment\n' in result['baseline_relative_diff']
        assert '.autoresearch-checkpoint.json' not in result['baseline_relative_diff']
        assert not result.get('baseline_diff_error')
    else:
        assert not result.get('baseline_relative_diff')
        assert result.get('baseline_diff_error') or result.get('error')


def test_production_method_phase_uses_baseline_value_without_fairness_gate(tmp_path):
    be = _FakeBackend(result={
        "metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
        "baseline": None, "method": 1.7,
        "candidate_valid": True,
        "lower_is_better": True, "config": "method-selected", "notes": "measured",
    })
    frozen = {
        "metric_value": 1.8, "metric_name": "val_loss", "metric_unit": "nats",
        "lower_is_better": True,
        "repo_commit": None, "fixed_config": "baseline-selected",
        "benchmark_contract": None, "host_attestation": None,
    }

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["baseline"].metric_value == 1.8
    assert res["candidate"].metric_value == 1.7
    assert res["verdict"]["improved"] is True
    assert res.get("benchmark_contract") is None
    assert res.get("host_attestation") is None
    assert "host-enforced fairness contract" not in be.task
    assert "Frozen configuration" not in be.task
    assert "immutable file hashes" not in be.task
    assert "ar-attempt" in be.task and "Run at most 20 candidates" in be.task
    assert "one complete JSON object per physical line" in be.task
    assert "validate every non-empty line with json.loads" in be.task


def test_production_method_phase_marks_attempt_bound_official_result_admissible(tmp_path):
    be = _FakeBackend(result={
        "metric": {"name": "total_params", "unit": "count", "lower_is_better": True},
        "baseline": None, "method": 608, "candidate_valid": True,
        "lower_is_better": True, "config": "official evaluator", "notes": "measured",
    })
    frozen = {
        "metric_value": 17988, "metric_name": "total_params", "metric_unit": "count",
        "lower_is_better": True, "fixed_config": "official evaluator",
        "repository_contract": {
            "sources": [{"path": "task.toml", "content": "[verifier]\n"}],
        },
    }

    res = run_agentic_improve(
        "reduce parameters", repo_url="https://x/game", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].valid is True
    assert res["candidate_admissibility"]["verdict"] == "pass"
    assert res["candidate_admissibility"]["attempt_id"].startswith("attempt-")


def test_production_method_phase_rejects_missing_repository_gate_outcome(tmp_path):
    be = _FakeBackend(result={
        "metric": {"name": "total_params", "unit": "count", "lower_is_better": True},
        "baseline": None, "method": 458,
        "lower_is_better": True, "config": "partial evaluator",
        "notes": "a preliminary check passed but the final repository gate was not reported",
    }, default_candidate_valid=False)
    frozen = {
        "metric_value": 17988, "metric_name": "total_params", "metric_unit": "count",
        "lower_is_better": True, "repo_commit": None, "fixed_config": "official evaluator",
        "repository_contract": {
            "sources": [{"path": "task.toml", "content": "[verifier]\n"}],
        },
    }

    res = run_agentic_improve(
        "reduce parameters", repo_url="https://x/game", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].metric_value == 458
    assert res["candidate"].valid is False
    assert res["verdict"]["improved"] is False
    assert res["diagnosis"] == "candidate_validity_missing"
    assert "missing /candidate_valid" in res["verdict"]["reason"]


def test_production_method_phase_rejects_explicit_repository_gate_failure(tmp_path):
    be = _FakeBackend(result={
        "metric": {"name": "total_params", "unit": "count", "lower_is_better": True},
        "baseline": None, "method": 913, "candidate_valid": False,
        "lower_is_better": True, "config": "official evaluator",
        "notes": "official evaluator emitted a finite metric but rejected correctness",
    })
    frozen = {
        "metric_value": 17988, "metric_name": "total_params", "metric_unit": "count",
        "lower_is_better": True, "repo_commit": None, "fixed_config": "official evaluator",
    }

    res = run_agentic_improve(
        "reduce parameters", repo_url="https://x/game", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].metric_value == 913
    assert res["candidate"].valid is False
    assert res["verdict"]["improved"] is False
    assert res["diagnosis"] == "correctness_gate_failed"
    assert res["candidate_admissibility"]["verdict"] == "fail"
    assert "candidate_valid=true" in be.task


@pytest.mark.parametrize("gate", [False, None])
def test_method_without_measurement_does_not_claim_evaluator_rejection(tmp_path, gate):
    payload = {
        "metric": {"name": "accuracy", "unit": "fraction", "lower_is_better": False},
        "baseline": None, "method": None,
        "notes": "Training ran out of host memory before evaluation; no checkpoint produced.",
    }
    if gate is not None:
        payload["candidate_valid"] = gate
    backend = _FakeBackend(result=payload, record_attempt=False,
                           default_candidate_valid=False)
    result = run_agentic_improve(
        "improve accuracy", repo_url="https://example.test/repo", workroot=str(tmp_path),
        backend=backend, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline={"metric_value": 0.7, "metric_name": "accuracy",
                         "metric_unit": "fraction", "lower_is_better": False,
                         "repository_contract": {"sources": [{"path": "README.md"}]}},
        enforce_fairness=False,
    )
    assert result["candidate"].metric_value is None
    assert result["candidate"].valid is False
    assert result["verdict"]["improved"] is False
    assert result["diagnosis"] == "execution_failure"
    assert result["verdict"]["reason"] == "method phase produced no finite method value"
    assert "host memory" in result["report"]


def test_indirect_component_preserves_attempt_id_without_final_metric(tmp_path):
    be = _FakeBackend(result={
        "metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
        "baseline": None, "method": None, "lower_is_better": True,
        "config": "component check", "notes": "intermediate property measured",
        "ablation_report": [{
            "component": "feature", "type": "indirect_gain", "actual": 0,
            "expected": {"operator": "==", "value": 0},
            "evidence": {"source": "runtime_attempt", "ref": "pending"},
            "verdict": "met",
        }],
    })
    frozen = {
        "metric_value": 1.8, "metric_name": "val_loss", "metric_unit": "nats",
        "lower_is_better": True, "fixed_config": "baseline-selected",
    }

    res = run_agentic_improve(
        "check an indirect component", repo_url="https://x/nanoGPT",
        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
        execution_phase="method", frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].metric_value is None
    assert res["component_attempt_id"].startswith("attempt-")
    assert res.get("error") is None


def test_indirect_component_recovers_attempt_id_from_runtime_evidence(tmp_path):
    class EvidenceOnlyBackend(_FakeBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None):
            response = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            repo = Path(self._mounts[0]) / "repo"
            result_path = repo / "ar_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            attempt_id = payload.pop("attempt_id")
            payload["ablation_report"][0]["evidence"]["ref"] = (
                f"{attempt_id}: official component check passed")
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return response

    be = EvidenceOnlyBackend(result={
        "metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
        "baseline": None, "method": None, "lower_is_better": True,
        "config": "component check", "notes": "intermediate property measured",
        "ablation_report": [{
            "component": "feature", "type": "indirect_gain", "actual": 0,
            "expected": {"operator": "==", "value": 0},
            "evidence": {"source": "runtime_attempt", "ref": "pending"},
            "verdict": "met",
        }],
    })
    frozen = {
        "metric_value": 1.8, "metric_name": "val_loss", "metric_unit": "nats",
        "lower_is_better": True, "fixed_config": "baseline-selected",
    }

    res = run_agentic_improve(
        "check an indirect component", repo_url="https://x/nanoGPT",
        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
        execution_phase="method", frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].metric_value is None
    assert res["component_attempt_id"].startswith("attempt-")
    assert res.get("error") is None


@pytest.mark.parametrize("phase", ["baseline", "method"])
@pytest.mark.parametrize("lower_is_better", [False, True])
def test_explicit_native_result_is_not_shadowed_by_unverified_better_variant(
    tmp_path, monkeypatch, phase, lower_is_better,
):
    from autoresearch.experiments.repo_improve import agentic_improve as mod
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    selected, historical = "attempt-selected", "attempt-historical"
    baseline = 10.0 if lower_is_better else 1.0
    payload = {
        "metric": {"name": "quality", "unit": "score",
                   "lower_is_better": lower_is_better},
        "baseline": 6.0 if phase == "baseline" else baseline,
        "method": 6.0 if phase == "method" else None,
        "baseline_valid": True, "candidate_valid": True, "attempt_id": selected,
    }
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    variants = analysis / "variants.jsonl"
    variants.write_text(json.dumps({
        "variant": "historical observation", "attempt_id": historical,
        "method": 5.0 if lower_is_better else 7.0, "valid": True,
    }) + "\n", encoding="utf-8")
    original_variants = variants.read_bytes()
    records = {}
    for attempt_id, stdout in [(historical, ""), (selected, json.dumps(payload))]:
        record = {
            "attempt_id": attempt_id, "launched": True, "finished": True,
            "rc": 0, "phase": phase, "cwd": str(tmp_path),
            "repo_commit": "commit", "repo_commit_end": "commit",
            "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff",
        }
        record.update(save_output(tmp_path, attempt_id, stdout, ""))
        records[attempt_id] = record
    monkeypatch.setattr(mod, "_attempt_records", lambda _: records)
    monkeypatch.setattr(mod, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(mod, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})

    feedback = mod._result_completion_request(
        str(path), execution_phase=phase, attempt_log="ledger")
    assert feedback is None
    parsed = mod._build_result_from_file(
        str(path), "generic", None, None, lambda *_: None, phase)
    assert parsed["_metric_attempt_id"] == selected
    assert parsed["_metric_evidence_source"] == "ar_result.json"
    result = mod._finalize_relaxed_phase_result(
        parsed, phase, {"metric_name": "quality", "metric_unit": "score",
                        "metric_value": baseline, "lower_is_better": lower_is_better},
        attempt_log="ledger", repo_path=str(tmp_path))
    outcome = result["baseline" if phase == "baseline" else "candidate"]
    assert outcome.valid is True
    assert outcome.metric_value == 6.0
    assert result["measurement_evidence"]["attempt_id"] == selected
    assert result["measurement_evidence"]["official_gate_verified"] is True
    assert variants.read_bytes() == original_variants
    assert json.loads(path.read_text()) == payload


@pytest.mark.parametrize("phase", ["baseline", "method"])
@pytest.mark.parametrize("reported,gate,valid", [
    (0.3179, True, True),
    (0.3179, False, False),
    (0.3179000001, True, False),
    (0.3255, True, False),
])
def test_report_scalar_roundoff_keeps_exact_host_value(tmp_path, monkeypatch, phase, reported, gate, valid):
    from autoresearch.experiments.repo_improve import agentic_improve as mod
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    measured = 0.31789999999999996
    attempt_id = "attempt-measured"
    record = {"attempt_id": attempt_id, "launched": True, "finished": True,
              "rc": 0, "phase": phase, "cwd": str(tmp_path),
              "repo_commit": "commit", "repo_commit_end": "commit",
              "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff"}
    record.update(save_output(tmp_path, attempt_id, json.dumps({
        "baseline": measured, "method": measured,
        "baseline_valid": gate, "candidate_valid": gate,
    }), ""))
    monkeypatch.setattr(mod, "_attempt_records", lambda _: {attempt_id: record})
    monkeypatch.setattr(mod, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(mod, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "diff"})
    result_path = tmp_path / "ar_result.json"
    payload = {"metric": {"name": "quality", "unit": "ratio", "lower_is_better": True},
               "baseline": reported, "method": reported if phase == "method" else None,
               "baseline_valid": True, "candidate_valid": True, "attempt_id": attempt_id}
    result_path.write_text(json.dumps(payload))
    feedback = mod._result_completion_request(
        str(result_path), execution_phase=phase, attempt_log="ledger")
    if reported == 0.3179:
        assert feedback is None  # Gate outcomes are checked separately; no numeric mismatch.
    else:
        assert "scalar disagrees" in feedback
    parsed = mod._build_result_from_file(str(result_path), "test", None, None, lambda *_: None, phase)
    result = mod._finalize_relaxed_phase_result(
        parsed, phase, {"metric_name": "quality", "metric_unit": "ratio",
                        "metric_value": 0.8, "lower_is_better": True},
        attempt_log="ledger", repo_path=str(tmp_path))
    outcome = result["baseline" if phase == "baseline" else "candidate"]
    assert outcome.valid is valid
    if reported == 0.3179:
        assert outcome.metric_value == measured
        assert result["measurement_evidence"]["official_gate_verified"] is True
        assert result["measurement_evidence"]["value"] == measured
    else:
        assert "scalar disagrees" in result["error"]


@pytest.mark.parametrize("changed", ["commit", "diff"])
def test_indirect_attempt_reports_code_binding_failure_not_missing_metric(tmp_path, monkeypatch, changed):
    from autoresearch.experiments.repo_improve import agentic_improve as mod

    attempt_id = "attempt-current-check"
    record = {
        "launched": True, "finished": True, "rc": 0, "phase": "method",
        "cwd": str(tmp_path), "repo_commit": "original", "repo_commit_end": "original",
        "tracked_diff_hash_start": "original-diff", "tracked_diff_hash_end": "original-diff",
    }
    monkeypatch.setattr(mod, "_attempt_records", lambda _: {attempt_id: record})
    monkeypatch.setattr(mod, "_repo_commit", lambda _: "changed" if changed == "commit" else "original")
    monkeypatch.setattr(mod, "_tracked_repo_state", lambda _: {
        "tracked_diff_hash": "changed" if changed == "diff" else "original-diff"})
    result = mod._finalize_relaxed_phase_result({
        "_metric_attempt_id": attempt_id,
        "_reported_metric": "quality", "_reported_metric_unit": "score",
        "candidate": mod.RunResult(metric_name="quality", metric_value=None, valid=False),
        "candidate_validity": True,
        "ablation_report": [{"component": "feature", "actual": 0,
                             "evidence": {"source": "runtime_attempt", "ref": attempt_id}}],
    }, "method", {"metric_name": "quality", "metric_unit": "score", "metric_value": 1.0},
        attempt_log="ledger", repo_path=str(tmp_path))

    assert result["component_attempt_id"] is None
    assert result["candidate"].valid is False
    assert result["diagnosis"] == "attempt_accounting_violation"
    assert changed in result["error"]
    assert "rerun" in result["error"]
    assert "no finite method value" not in result["error"]


@pytest.mark.parametrize("rc,changed", [(1, False), (0, False), (124, False), (137, False), (1, True)])
def test_failed_component_receipt_is_not_a_successful_metric_attempt(tmp_path, monkeypatch, rc, changed):
    from autoresearch.experiments.repo_improve import agentic_improve as mod

    attempt_id = "attempt-property-check"
    record = {"launched": True, "finished": True, "rc": rc, "phase": "method",
              "cwd": str(tmp_path), "repo_commit": "commit", "repo_commit_end": "commit",
              "tracked_diff_hash_start": "diff", "tracked_diff_hash_end": "diff"}
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output
    record["attempt_id"] = attempt_id
    record.update(save_output(tmp_path, attempt_id, json.dumps({"measurements": {"feature": 1}}), ""))
    monkeypatch.setattr(mod, "_attempt_records", lambda _: {attempt_id: record})
    monkeypatch.setattr(mod, "_repo_commit", lambda _: "commit")
    monkeypatch.setattr(mod, "_tracked_repo_state", lambda _: {
        "tracked_diff_hash": "changed" if changed else "diff"})
    result = mod._finalize_relaxed_phase_result({
        "_reported_metric": "quality", "_reported_metric_unit": "score",
        "candidate": mod.RunResult(metric_name="quality", metric_value=None, valid=False),
        "candidate_validity": False,
        "component_failed_attempt_id": "attempt-forged",
        "ablation_report": [{"component": "feature", "actual": 1,
                             "expected": {"operator": "==", "value": 0}, "verdict": "unmet",
                             "evidence": {"source": "runtime_attempt", "ref": attempt_id}}],
    }, "method", {"metric_name": "quality", "metric_unit": "score", "metric_value": 1.0},
        attempt_log="ledger", repo_path=str(tmp_path))

    assert result["candidate"].valid is False
    assert result["verdict"]["improved"] is False
    if rc == 1 and not changed:
        assert result["component_failed_attempt_id"] == attempt_id
        assert result["component_attempt_id"] is None
        assert result["diagnosis"] == "component_expectation_failed"
        assert "reported unmet" in result["error"]
        assert "no finite method value" not in result["error"]
    else:
        assert not result.get("component_failed_attempt_id")


def test_production_method_phase_accepts_unit_when_legacy_baseline_unit_is_missing(tmp_path):
    """A legacy missing unit must not invalidate the same named official metric."""
    be = _FakeBackend(result={
        "metric": {"name": "runtime_ms", "unit": "milliseconds", "lower_is_better": True},
        "baseline": None, "method": 1.05,
        "lower_is_better": True, "config": "official evaluator", "notes": "measured",
    })
    frozen = {
        "metric_value": 107.0, "metric_name": "runtime_ms", "metric_unit": None,
        "lower_is_better": True, "repo_commit": None, "fixed_config": "official evaluator",
    }

    res = run_agentic_improve(
        "lower runtime", repo_url="https://x/kernel", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].valid is True
    assert res["verdict"]["improved"] is True


def test_production_method_phase_accepts_descriptive_metric_suffix_and_unit_alias(tmp_path):
    """Presentation-only labels must not reject the same official measured quantity."""
    be = _FakeBackend(result={
        "metric": {
            "name": "runtime_ms (median of 7 timed CUDA-event runs)",
            "unit": "milliseconds",
            "lower_is_better": True,
        },
        "baseline": None, "method": 0.967968,
        "lower_is_better": True, "config": "official evaluator",
    })
    frozen = {
        "metric_value": 107.013924, "metric_name": "runtime_ms", "metric_unit": "ms",
        "lower_is_better": True, "repo_commit": None,
        "fixed_config": "official evaluator",
    }

    res = run_agentic_improve(
        "lower runtime", repo_url="https://x/kernel", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].valid is True
    assert res["verdict"]["improved"] is True


@pytest.mark.parametrize("candidate_score,improved", [(1.2, True), (0.8, False)])
def test_split_phases_preserve_evaluator_defined_composite_score(tmp_path, candidate_score, improved):
    metric = {"name": "composite_score", "unit": "score", "lower_is_better": False}
    baseline = run_agentic_improve(
        "Optimize the declared composite", repo_url="https://example.test/repo",
        workroot=str(tmp_path / "baseline"), clone_fn=_fake_clone,
        backend=_FakeBackend(result={
            "metric": metric, "baseline": 1.0, "method": None,
            "config": {"raw_constituents": {"quality": 0.7, "resource": 50.0}},
        }), execution_phase="baseline", enforce_fairness=False,
    )
    assert baseline["baseline"].valid is True
    frozen = {
        "metric_name": baseline["metric"], "metric_unit": baseline["metric_unit"],
        "metric_value": baseline["baseline"].metric_value,
        "lower_is_better": baseline["lower_is_better"], "repo_commit": None,
        "fixed_config": baseline["config"],
    }
    method = run_agentic_improve(
        "Optimize the declared composite", repo_url="https://example.test/repo",
        workroot=str(tmp_path / "method"), clone_fn=_fake_clone,
        backend=_FakeBackend(result={
            "metric": metric, "baseline": None, "method": candidate_score,
            "config": {"raw_constituents": {"quality": 0.8, "resource": 60.0}},
        }), execution_phase="method", frozen_baseline=frozen, enforce_fairness=False,
    )
    assert method["metric"] == "composite_score"
    assert method["metric_unit"] == "score"
    assert method["baseline"].metric_value == 1.0
    assert method["candidate"].metric_value == candidate_score
    assert method["candidate"].valid is True
    assert method["verdict"]["improved"] is improved


def test_production_method_phase_rejects_mixed_runtime_and_speedup_units(tmp_path):
    """Never compare a baseline runtime number directly with a method speedup factor."""
    be = _FakeBackend(result={
        "metric": {"name": "speedup", "unit": "ratio", "lower_is_better": False},
        "baseline": None, "method": 1.99, "lower_is_better": False,
    })
    frozen = {
        "metric_value": 11.5, "metric_name": "runtime_ms", "metric_unit": "ms",
        "lower_is_better": True, "repo_commit": None, "fixed_config": "official evaluator",
    }

    res = run_agentic_improve(
        "improve kernel runtime", repo_url="https://x/kernel",
        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
        execution_phase="method", frozen_baseline=frozen, enforce_fairness=False,
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "metric_identity_mismatch"
    assert "does not match" in res["verdict"]["reason"]


def test_baseline_phase_builds_contract_from_confirmed_boundary(tmp_path, monkeypatch):
    """The trusted boundary must not depend on the sandbox copying JSON perfectly."""
    contract = _v2_benchmark_contract(0.997409)
    contract["objective"]["metric"] = "val_bpb"
    contract["observed_outcomes"]["quality"] = {
        "metric": "val_bpb", "value": 0.997409}
    contract["verified_evidence"]["observations"] = {
        "quality": {"metric": "val_bpb", "value": 0.997409}}
    be = _FakeBackend(result={
        "metric": "val_bpb", "baseline": 0.997409, "method": None,
        "lower_is_better": True,
        "config": {"training_seconds": 300.3, "accelerator_count": 1},
        "notes": "measured baseline",
        "benchmark_contract": contract,
    })
    boundary = {
        "metric": "val_bpb", "direction": "minimize",
        "protocol_kind": "fixed_wall_clock", "training_budget_s": 300,
        "accelerator_count": 1, "data": "repository canonical data",
        "evaluator": "repository canonical evaluator",
        "workload_accounting": "300 seconds training wall-clock",
        "allowed_treatment_axes": ["method implementation"],
    }
    ticks = iter([100.0, 400.0])
    monkeypatch.setattr(
        "autoresearch.experiments.repo_improve.agentic_improve.time.time",
        lambda: next(ticks),
    )

    res = run_agentic_improve(
        "lower val_bpb", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline", cuda_device="0",
        confirmed_boundary=boundary,
    )

    assert res["baseline"].valid is True
    assert res["benchmark_contract"]["selected_source"] == "confirmed_boundary"
    assert res["benchmark_contract"]["controls"]["resources"]["accelerator_count"] == 1
    assert res["contract_fingerprint"].startswith("sha256:")


def test_baseline_phase_merges_agent_evidence_with_confirmed_boundary(tmp_path):
    """Replacing the agent contract would discard v2 official-harness evidence."""
    agent_contract = _v2_benchmark_contract()
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "benchmark_contract": agent_contract,
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        confirmed_boundary={"metric": "val_loss", "direction": "minimize"},
    )

    assert res["baseline"].valid is True
    assert res["benchmark_contract"]["version"] == 2
    assert res["benchmark_contract"]["verified_evidence"]["source"] == "host_remeasurement"


def test_baseline_phase_rejects_confirmed_boundary_metric_mismatch(tmp_path):
    """A trusted boundary cannot freeze a baseline for a different metric."""
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000", "notes": "pristine",
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        confirmed_boundary={"metric": "accuracy", "direction": "minimize"},
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "baseline_contract_invalid"


def test_baseline_phase_rejects_confirmed_boundary_direction_mismatch(tmp_path):
    """A trusted boundary cannot freeze a baseline with the opposite direction."""
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000", "notes": "pristine",
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        confirmed_boundary={"metric": "val_loss", "direction": "maximize"},
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "baseline_contract_invalid"


def test_baseline_phase_uses_confirmed_boundary_for_missing_agent_metadata(tmp_path):
    """A trusted boundary completes baseline metadata the agent omitted."""
    be = _FakeBackend(result={
        "baseline": 1.8, "method": None,
        "config": "steps=1000", "notes": "pristine",
        "benchmark_contract": _v2_benchmark_contract(),
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        confirmed_boundary={"metric": "val_loss", "direction": "minimize"},
    )

    assert res["baseline"].valid is True
    assert res["metric"] == "val_loss"
    assert res["lower_is_better"] is True
    assert res["contract_fingerprint"].startswith("sha256:")


def test_baseline_phase_uses_boundary_direction_for_missing_agent_direction(tmp_path):
    """A confirmed maximize boundary remains usable when the agent omits direction."""
    contract = _v2_benchmark_contract(0.9)
    contract["objective"] = {"metric": "accuracy", "direction": "maximize", "mode": "quality"}
    contract["observed_outcomes"]["quality"] = {"metric": "accuracy", "value": 0.9}
    contract["verified_evidence"]["observations"] = {
        "quality": {"metric": "accuracy", "value": 0.9}}
    be = _FakeBackend(result={
        "metric": "accuracy", "baseline": 0.9, "method": None,
        "config": "steps=1000", "notes": "pristine",
        "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "raise accuracy", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        confirmed_boundary={"metric": "accuracy", "direction": "maximize"},
    )

    assert res["baseline"].valid is True
    assert res["lower_is_better"] is False
    assert res["benchmark_contract"]["objective"]["direction"] == "maximize"


def test_baseline_phase_boundary_without_result_metadata_uses_fallback(tmp_path):
    """A boundary without metric metadata continues to derive it from the measurement."""
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000", "notes": "pristine",
        "benchmark_contract": _v2_benchmark_contract(),
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
        confirmed_boundary={"protocol_kind": "fixed_work"},
    )

    assert res["baseline"].valid is True
    assert res["benchmark_contract"]["objective"] == {
        "metric": "val_loss", "direction": "minimize", "mode": "quality",
    }
    assert res["contract_fingerprint"].startswith("sha256:")


def test_baseline_prompt_carries_confirmed_user_requirements_without_fairness_contract():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "run the repository benchmark", "/repo", "/repo/ar_result.json", None, None,
        confirmed_boundary={"budget_s": 300, "accelerator_count": 1},
    )

    assert '"budget_s":300' in prompt
    assert '"accelerator_count":1' in prompt
    assert "Follow them when selecting the baseline procedure" in prompt


def test_baseline_prompt_preserves_composite_objective_instead_of_one_constituent():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "Optimize the evaluator-defined composite score", "/repo", "/repo/ar_result.json",
        "composite_score", False,
    )

    assert "single raw quantity" in prompt
    assert "preserves the declared objective's ordering" in prompt
    assert "Do not replace a composite metric with one constituent" in prompt
    assert "same declared formula and reference" in prompt
    assert "Do not invent a baseline score" in prompt


def test_baseline_prompt_keeps_objective_separate_from_feasibility_checks():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "minimize resource usage subject to the declared quality requirement",
        "/repo", "/repo/ar_result.json", "resource_usage", True,
    )
    assert "baseline algorithm's own optimization or intervention" in prompt
    assert "Do not substitute a feasibility/quality measurement for the optimization objective" in prompt
    assert "separate successful reproduction from measured eligibility" in prompt
    assert "baseline_outcome" in prompt
    assert "reference anchors are context, never substitutes for local measurements" in prompt
    assert "Method candidates still must pass every official" in prompt
    assert "Zero interventions on an untouched starting state" in prompt
    assert "not the baseline scalar" in prompt
    assert "A measured zero is valid if the target was already attained" in prompt
    assert "When baseline_valid=true, `baseline` MUST be one finite JSON number" in prompt
    assert "write baseline=null and baseline_valid=false" in prompt


def test_baseline_prompt_distinguishes_measured_stdout_from_result_envelope():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "measure the repository's declared objective", "/repo", "/repo/ar_result.json",
        "objective", True,
    )
    assert "ar_result.json alone is not measured evidence" in prompt
    assert "metric_pointer and gate_pointers" in prompt
    assert "default stdout fields are baseline and baseline_valid" in prompt
    assert "wrapper launched by ar-attempt" in prompt
    assert "must not print copied historical numbers" in prompt
    assert "keep measurement-only wrappers outside the candidate repository" in prompt


def test_baseline_prompt_does_not_require_improvement_headroom():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "minimize the official objective", "/repo", "/repo/ar_result.json",
        "objective", True,
    )
    assert "Baseline validity does not require improvement headroom" in prompt
    assert "already optimal baseline remains valid" in prompt
    assert "Do not weaken or replace it merely to make an improvement possible" in prompt


def test_baseline_prompt_requires_post_clone_repository_scenario_inspection():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "run the repository's short-budget benchmark", "/repo", "/repo/ar_result.json",
        "score", False,
    )

    assert "inspect the cloned repository itself" in prompt
    assert "task-relevant README/documentation" in prompt
    assert "official run scripts, task manifests, configs" in prompt
    assert "scenario-specific recommended recipe" in prompt
    assert "Do not treat a generic parser/config default" in prompt
    assert "REPOSITORY_INSPECTION" in prompt
    assert "selected command/config, rejected alternatives" in prompt
    assert "$AUTORESEARCH_ARTIFACT_DIR/baseline/" in prompt
    assert "raw measurements" in prompt
    assert "not container-local" in prompt


def test_restore_index_executable_bits_removes_mirror_mode_drift(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _restore_index_executable_bits,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    plain = repo / "plain.py"
    executable = repo / "run.sh"
    plain.write_text("print('plain')\n", encoding="utf-8")
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    plain.chmod(0o644)
    executable.chmod(0o755)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)

    plain.chmod(0o755)
    executable.chmod(0o644)
    _restore_index_executable_bits(str(repo))

    assert plain.stat().st_mode & 0o111 == 0
    assert executable.stat().st_mode & 0o111 == 0o111
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--summary"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "mode change" not in diff.stdout


def test_split_phase_prompts_share_literature_grounded_evaluation_semantics():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt, _method_task_prompt,
    )

    semantics = {
        "comparison_statement": (
            "On the same hardware, model scale, data, tokenizer and evaluator, give baseline "
            "and method five minutes of pure training and compare final val_loss."
        ),
        "research_question": "Does the method lower validation loss in five minutes?",
        "primary_metric": {"name": "val_loss", "direction": "minimize"},
        "comparison_mode": "quality_at_fixed_time",
        "comparison_basis": "Compare final validation loss after the stated training time.",
        "budget_interpretation": "Training time is the comparison budget.",
        "conditions_to_hold_constant": ["training time", "hardware", "dataset", "evaluator"],
        "method_freedom": ["optimizer and model implementation"],
        "evidence": [{
            "source_type": "key_paper", "source_id": "paper-1",
            "supports": "The paper reports quality at a fixed training time.",
        }],
        "uncertainties": [],
    }
    frozen = {
        "metric_name": "val_loss", "metric_value": 1.0, "lower_is_better": True,
        "fixed_config": "depth=12, device_batch_size=16, training_seconds=300",
    }
    baseline_prompt = _baseline_task_prompt(
        "goal", "/repo", "/repo/ar_result.json", "val_loss", True,
        evaluation_semantics=semantics,
    )
    method_prompt = _method_task_prompt(
        "goal", "/repo", "/repo/ar_result.json", None, frozen,
        evaluation_semantics=semantics,
    )
    canonical = json.dumps(
        semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    for prompt in (baseline_prompt, method_prompt):
        assert canonical in prompt
        assert "conditions_to_hold_constant are not score-improvement levers" in prompt
        assert "method_freedom lists legitimate method choices" in prompt
        assert "COMPARISON_AUDIT" in prompt
        assert "observed totals" in prompt
        assert "arithmetically inconsistent" in prompt

    assert "Do not create a git commit" in method_prompt
    assert "working-tree diff" in method_prompt
    assert "Do not rerun or recalibrate the baseline" in method_prompt
    assert "does not prohibit a required parent/control measurement" in method_prompt
    assert "never replaces the frozen BaselineResult" in method_prompt
    assert "A baseline artifact path is not a host attempt receipt" in method_prompt
    assert "Frozen baseline configuration" in method_prompt
    assert "depth=12, device_batch_size=16, training_seconds=300" in method_prompt
    assert "held-constant conditions" in method_prompt
    assert "training-only budget" in method_prompt
    assert "setup, compilation, and final evaluation" in method_prompt
    assert "inspect the exact layout" in method_prompt
    assert "container-local compatibility view" in method_prompt
    assert "all optimizer-step time across every phase" in method_prompt
    assert "compilation exclusion only once" in method_prompt
    assert "top-level attempt_id" in method_prompt
    assert "same candidate" in method_prompt


def test_method_prompt_preserves_runner_envelope_for_component_outputs():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {"metric_name": "score", "metric_value": 1.0, "lower_is_better": True},
    )
    assert "ComponentPlan's custom test output does not replace this result envelope" in prompt
    assert "separate evidence artifact" in prompt
    assert "candidate_valid=false" in prompt
    assert "When no new candidate command ran" in prompt
    assert "do not label them as this attempt's runtime measurement" in prompt


def test_method_prompt_separates_official_gate_from_component_acceptance():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {"metric_name": "score", "metric_value": 1.0, "lower_is_better": False},
    )
    assert "Keep repository validity separate from ComponentPlan acceptance and MIR fidelity" in prompt
    assert "Do not AND custom component assertions" in prompt
    assert "unless the repository itself requires those checks" in prompt
    assert "Record a failed component check in ablation_report" in prompt
    assert "component still cannot be accepted until its own frozen contract passes" in prompt
    assert "An official gate failure must remain false" in prompt
    assert "host-captured attempt output as well as the result file" in prompt


def test_official_gate_pass_keeps_metric_despite_unmet_component_check(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({
        "metric": "quality", "baseline": 0.4, "method": 0.6,
        "candidate_valid": True, "lower_is_better": False,
        "ablation_report": [{"component": "representation", "type": "indirect_gain",
            "actual": 0.2, "expected": {"operator": "<=", "value": 0.1},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-1"},
            "verdict": "unmet"}],
    }), encoding="utf-8")
    result = _build_result_from_file(
        str(path), "generic", "quality", False, lambda e, p: None)

    assert result["candidate"].metric_value == 0.6
    assert result["candidate"].valid is True
    assert json.loads(result["ablation_report"])[0]["verdict"] == "unmet"


def test_method_prompt_keeps_candidate_driver_source_in_reviewed_worktree():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {"metric_name": "score", "metric_value": 1.0, "lower_is_better": True},
    )

    assert "candidate implementation source, including candidate-specific drivers, inside" in prompt
    assert "current repository worktree" in prompt
    assert "host-captured diff" in prompt
    assert "task-permitted submission path in the current worktree" in prompt
    assert "temporary copy's score as the final method result" in prompt
    assert "candidate-specific drivers and outputs separately under" not in prompt
    assert "$AUTORESEARCH_ARTIFACT_DIR/baseline/" in prompt
    assert "$AUTORESEARCH_ATTEMPT_ARTIFACT_DIR" in prompt
    assert "Do not reuse a shared" in prompt
    assert "parent checkpoints as read-only inputs" in prompt
    assert "update that driver's output destination before rerunning it" in prompt
    assert "do not overwrite baseline evidence" in prompt


def test_method_prompt_reports_confirmed_component_incompatibility_as_preflight_evidence():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "score", "metric_value": 1.0,
            "lower_is_better": True, "fixed_config": "canonical evaluator",
        },
    )

    assert "do not keep debugging or substitute a different method" in prompt
    assert "structured preflight component failure" in prompt
    assert '"actual": 0' in prompt
    assert '"expected": {"operator": "==", "value": 1}' in prompt
    assert "tracked repository path+lines or failed attempt_id" in prompt


def test_method_prompt_has_no_repository_path_name_policy():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "score", "metric_value": 1.0,
            "lower_is_better": True, "fixed_config": "canonical evaluator",
        },
    )

    assert "reference answers, gold solutions, or optimized answer implementations" not in prompt
    assert "private or invisible to the agent" not in prompt
    assert "Do not inspect, execute, copy, or adapt" not in prompt


def test_method_prompt_requires_scalar_ablation_actual():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "check the confirmed method", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "score", "metric_value": 1.0,
            "lower_is_better": True, "fixed_config": "canonical evaluator",
        },
    )

    assert "actual MUST be exactly one finite JSON number" in prompt
    assert "never an object, list, or string" in prompt
    assert "auxiliary measurements" in prompt
    assert "Every ablation_report item MUST include these common fields" in prompt
    assert '"expected":{"operator":"<|<=|>|>=|==","value":<finite number>}' in prompt
    assert '"evidence":{"source":"<non-empty source>","ref":"<non-empty ref>"}' in prompt


@pytest.mark.parametrize("phase", ["combined", "method"])
def test_component_result_prompt_keeps_parent_measurement_in_ablation_item(phase):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _task_prompt, _method_task_prompt,
    )

    args = ("check the current component", "/repo", "/repo/ar_result.json", 1200)
    prompt = (
        _task_prompt(*args, None, None)
        if phase == "combined"
        else _method_task_prompt(*args, {
            "metric_name": "score", "metric_value": 1.0,
            "lower_is_better": True, "fixed_config": "canonical evaluator",
        })
    )

    assert "Each direct_gain item must additionally contain" in prompt
    assert "including final integration" in prompt
    assert "ablation_report.actual is the raw candidate metric" in prompt
    assert "not a parent-minus-candidate delta" in prompt
    assert "host computes signed_metric_improvement" in prompt
    assert "never copy contract_result.actual" in prompt
    assert "Keep baseline and method as top-level fields" in prompt
    assert "exactly one ablation_report item per component" in prompt
    assert "analysis/<component-id>/<attempt-id>/" in prompt
    assert "do not inherit a previous report as a fresh measurement" in prompt
    assert "Preserve repository-required output locations" in prompt
    assert "explicitly declared runtime artifacts" in prompt
    assert "distinguish evidence-output collisions from implementation conflicts" in prompt
    assert "do not make a second item for the parent measurement" in prompt
    assert '"parent_metric":<finite measured parent value>' in prompt
    assert '"parent_evidence":{"source":"runtime_attempt","ref":"<parent attempt ref>"}' in prompt
    assert "not only in config, notes, expected, or evidence.ref text" in prompt
    assert "indirect_gain items do not require a parent metric" in prompt
    assert "evidence.json_pointer" in prompt
    assert "parent_evidence.json_pointer" in prompt
    assert "structured stdout" in prompt
    assert "a stored number alone is not parent evidence" in prompt
    assert "successful host-registered control attempt" in prompt
    assert "reuse that receipt without rerunning" in prompt
    assert "Change cwd BEFORE invoking ar-attempt" in prompt
    assert "changing directory inside its child script is too late" in prompt
    assert "Do not restore baseline source over the dependency parent" in prompt
    assert "if it lacks a stored metric" not in prompt
    assert "not a filesystem path" in prompt
    assert "MUST use this exact trusted shape" not in prompt


@pytest.mark.parametrize("phase", ["combined", "baseline", "method"])
def test_accelerator_preflight_forbids_silent_cpu_fallback(phase):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _task_prompt, _baseline_task_prompt, _method_task_prompt,
    )

    args = ("use the assigned accelerator", "/repo", "/repo/ar_result.json")
    if phase == "baseline":
        prompt = _baseline_task_prompt(*args, "accuracy", False)
    elif phase == "method":
        prompt = _method_task_prompt(*args, 1200, {
            "metric_name": "accuracy", "metric_value": 0.5,
            "lower_is_better": False,
        })
    else:
        prompt = _task_prompt(*args, 1200, "accuracy", False)
    assert "DEVICE PREFLIGHT" in prompt
    assert "exact interpreter and runtime" in prompt
    assert "device enumeration alone" in prompt
    assert "do not silently fall back to CPU" in prompt
    assert "hard dependency pin" in prompt
    assert "re-establish the baseline" in prompt


def test_method_prompt_avoids_malformed_terminal_recovery_calls():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt, _method_task_prompt,
    )

    method = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "score", "metric_value": 1.0,
            "lower_is_better": True, "fixed_config": "canonical evaluator",
        },
    )
    baseline = _baseline_task_prompt("reproduce", "/repo", "/repo/ar_result.json", "score", True)

    for prompt in (baseline, method):
        assert "start the command with nohup and &" in prompt
        assert "Never use an input flag, an empty/whitespace command" in prompt
        assert "tool-call markup as chat text to wait" in prompt
        assert "timeout field to at least" not in prompt


def test_all_execution_prompts_preserve_pipeline_failure_status():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt, _method_task_prompt, _task_prompt,
    )

    prompts = (
        _task_prompt("optimize", "/repo", "/repo/result.json", 1200, "loss", True),
        _baseline_task_prompt("optimize", "/repo", "/repo/result.json", "loss", True),
        _method_task_prompt("optimize", "/repo", "/repo/result.json", 1200,
                            {"metric_name": "loss", "metric_value": 1.0,
                             "lower_is_better": True}),
    )
    for prompt in prompts:
        assert "set -e -o pipefail" in prompt
        assert "tee/head/tail succeeding does not prove" in prompt
        assert "complete training/evaluation logs" in prompt
        assert "readiness polling must check that child's exit status" in prompt
        assert "unresolved failed scoring requests" in prompt
        assert "do not drop failed samples or change the evaluator" in prompt
        assert "smoke-test the actual scoring entry point" in prompt


def test_execution_prompts_keep_external_benchmark_reward_outside_research():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt, _method_task_prompt, _task_prompt,
    )

    prompts = (
        _task_prompt("study the task objective", "/repo", "/repo/result.json", 1200, "loss", True),
        _baseline_task_prompt("reproduce the task", "/repo", "/repo/result.json", "loss", True),
        _method_task_prompt("test the method", "/repo", "/repo/result.json", 1200,
                            {"metric_name": "loss", "metric_value": 1.0,
                             "lower_is_better": True}),
    )
    for prompt in prompts:
        assert "Do not read, infer, or optimize external benchmark reward" in prompt
        assert "raw task measurements and implementation correctness feedback" in prompt
        assert "official score, bind and record it separately" not in prompt
        assert "mounted persistent session storage" in prompt
        assert "verify them again after restoration" in prompt
        assert "unhandled main-process exception" in prompt
        assert "give scoring its own finite timeout" in prompt
        assert "caught exception alone is not authority to kill" in prompt
        assert "Record resumable progress in artifact_paths" in prompt
        assert "remaining total task budget" in prompt
        assert "partial progress is not a completed result" in prompt


def test_long_setup_commands_use_polled_background_jobs(monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt,
        _method_task_prompt,
    )

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "1200")
    baseline = _baseline_task_prompt(
        "reproduce benchmark", "/repo", "/repo/ar_result.json", "loss", True,
    )
    method = _method_task_prompt(
        "implement the confirmed method", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "loss", "metric_value": 1.0,
            "lower_is_better": True, "fixed_config": "canonical evaluator",
        },
    )

    for prompt in (baseline, method):
        assert "start the command with nohup and &" in prompt
        assert "poll those files with new terminal commands that wait at most 60 seconds" in prompt
        assert "do not claim completion before seeing the exit status" in prompt
        assert "timeout field to at least" not in prompt


def test_split_phase_prompts_preflight_shared_venv_import_provenance():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt,
        _method_task_prompt,
    )

    prompts = (
        _baseline_task_prompt(
            "reproduce benchmark", "/current/repo", "/current/repo/ar_result.json",
            "loss", True,
        ),
        _method_task_prompt(
            "implement the confirmed method", "/current/repo",
            "/current/repo/ar_result.json", None,
            {
                "metric_name": "loss", "metric_value": 1.0,
                "lower_is_better": True, "fixed_config": "canonical evaluator",
            },
        ),
    )

    for prompt in prompts:
        assert "resolves inside the current repository worktree" in prompt
        assert "prior round's editable install" in prompt
        assert "do not modify the shared environment or package cache" in prompt
        assert 'uv venv "$UV_PROJECT_ENVIRONMENT"' in prompt
        assert 'uv pip --python "$UV_PROJECT_ENVIRONMENT/bin/python"' in prompt
        assert "never use bare `pip`" in prompt


def test_analysis_sidecar_does_not_overwrite_direction_from_diagnostic_only_result(
        tmp_path, monkeypatch):
    """A method=null CHECK-EXPECTATION artifact is routed by trusted Memory/MOSS, not by the
    advisory variant-analysis sidecar.  Treating it as a new candidate erased the prior faithful
    run and caused the same disproved method to be launched again."""
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    direction = Path(ai._analysis_direction_path(str(repo)))
    direction.parent.mkdir(parents=True)
    direction.write_text("prior faithful run was measured and failed", encoding="utf-8")
    (repo / ai._RESULT_BASENAME).write_text(json.dumps({
        "metric": "val_loss",
        "baseline": 1.0,
        "method": None,
        "changes_summary": "CHECK-EXPECTATION diagnostic only; no new candidate",
        "bottleneck": "confirmed component contract failed",
        "ablation_report": [{
            "component": "core step", "type": "functional", "actual": 0,
            "expected": {"operator": "==", "value": 1},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-1"},
            "verdict": "unmet",
        }],
    }), encoding="utf-8")
    called = {"n": 0}

    def _fake_llm(user, timeout=90):
        called["n"] += 1
        return "repeat the disproved method"

    monkeypatch.setattr(ai, "_call_analysis_llm", _fake_llm)

    last = ai._analyze_once(
        str(repo), "confirmed method", "val_loss", True, lambda e, p: None, last=7)

    assert last == 7
    assert called["n"] == 0
    assert direction.read_text(encoding="utf-8") == "prior faithful run was measured and failed"


def test_baseline_prompt_respects_sizeable_download_approval_policy():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt("reproduce nanochat", "/repo", "/repo/ar_result.json",
                                   "val_bpb", True)

    assert "download a missing repository-required dependency or dataset directly" in prompt
    assert "only when the Research goal, confirmed user requirements" in prompt
    assert "report the URL, expected size, purpose, and cache status" in prompt
    assert "wait for supervisor approval" in prompt
    assert "Never weaken repository signature verification" in prompt
    assert "mark an unverified source trusted" in prompt
    assert "Never clean, prune, delete, or replace a shared package/data cache" in prompt
    assert "Treat mounted shared datasets and pre-existing cache contents as read-only inputs" in prompt
    assert "checkpoints, logs, and temporary outputs" in prompt
    assert "current session workroot" in prompt
    assert "Install dependencies into `$UV_PROJECT_ENVIRONMENT`" in prompt
    assert 'uv venv "$UV_PROJECT_ENVIRONMENT"' in prompt
    assert 'uv pip --python "$UV_PROJECT_ENVIRONMENT/bin/python"' in prompt
    assert "never use bare `pip`" in prompt
    assert "If `$AUTORESEARCH_DATA_ROOT` is set" in prompt
    assert "inspect the exact layout" in prompt
    assert "container-local compatibility view" in prompt
    assert "native/CUDA extension" in prompt
    assert "smallest supported forward/backward" in prompt
    assert "compatibility metadata" in prompt
    assert "Never fake compatibility" in prompt
    assert "data.immutable_paths and evaluator.immutable_paths" not in prompt
    assert "protocol_parser" not in prompt


def test_method_prompt_respects_sizeable_download_approval_policy():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "improve benchmark", "/repo", "/repo/ar_result.json", None,
        {"metric_name": "loss", "metric_value": 1.0, "lower_is_better": True},
    )

    assert "download a missing repository-required dependency or dataset" in prompt
    assert "only when the Research goal, confirmed user requirements" in prompt
    assert "report the URL, expected size, purpose, and cache" in prompt
    assert "wait for supervisor approval" in prompt
    assert "Never weaken repository signature verification" in prompt
    assert "mark an unverified source trusted" in prompt


def test_baseline_prompt_leaves_training_and_evaluation_unwrapped(monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "1200")
    prompt = _baseline_task_prompt(
        "reproduce benchmark", "/repo", "/repo/ar_result.json", "val_loss", True,
        confirmed_boundary={"protocol_kind": "fixed_wall_clock", "training_budget_s": 300},
    )

    assert "Do not wrap the canonical baseline command in a command-level timeout" in prompt
    assert "`timeout 1200s`" not in prompt

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "60")
    long_budget_prompt = _baseline_task_prompt(
        "reproduce benchmark", "/repo", "/repo/ar_result.json", "val_loss", True,
        confirmed_boundary={"protocol_kind": "fixed_wall_clock", "training_budget_s": 300},
    )
    assert "`timeout 315s`" not in long_budget_prompt
    assert "overall sandbox safety cap" in long_budget_prompt


def test_baseline_prompt_registers_scoring_attempt_without_timeout():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt(
        "reproduce benchmark", "/repo", "/repo/ar_result.json", "val_loss", True,
    )

    assert "`ar-attempt <canonical baseline command>`" in prompt
    assert "tiny-train smoke test" in prompt
    assert "do not use unledgered training output" in prompt
    assert "include that emitted `attempt_id`" in prompt
    assert "Do not put `timeout` inside or outside this baseline scoring command" in prompt
    assert "`baseline` MUST be one finite JSON number" in prompt
    assert "never an object, array, or string" in prompt
    assert "same directly measured quantity" in prompt
    assert "derived ratio" in prompt


def test_measurement_timeout_leaves_room_outside_official_training_timer(monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _measurement_timeout

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "1200")
    training_contract = {
        "protocol": {
            "kind": "quality_at_fixed_time",
            "budget": {"wall_clock_s": 300},
            "timing_scope": "training",
            "timer_authority": "official_harness",
        },
    }
    process_contract = {
        "protocol": {
            "kind": "quality_at_fixed_time",
            "budget": {"wall_clock_s": 300},
            "timing_scope": "process",
            "timer_authority": "host",
        },
    }

    assert _measurement_timeout({"benchmark_contract": training_contract}, None) == 1200
    assert _measurement_timeout({"benchmark_contract": process_contract}, None) == 315

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "60")
    assert _measurement_timeout({"benchmark_contract": training_contract}, None) == 315
    assert _measurement_timeout({"benchmark_contract": process_contract}, None) == 315


@pytest.mark.parametrize("rc,finished,phase,included", [
    (-1, True, "method", True), (124, True, "method", True),
    (126, True, "method", True), (0, True, "method", True),
    (None, False, "method", False), (1, True, "baseline", False),
])
def test_missing_result_failure_notes_preserve_host_facts_only(
        tmp_path, monkeypatch, rc, finished, phase, included):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    record = dict(attempt_id="attempt-real", launched=True, finished=finished,
                  phase=phase, rc=rc, started_at=100, ended_at=7310,
                  stdout_path=str(tmp_path / "output"), stderr_path=str(tmp_path / "error"),
                  argv=["env", "API_KEY=private-secret", "train"],
                  detail="provider-private-secret")
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"attempt-real": record})
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "7200")
    notes = module._failed_attempt_notes("host-ledger", "method")
    assert bool(notes) is included
    if included:
        assert '"rc": ' + str(rc) in notes[0]
        assert '"host_command_cap_s": 7200' in notes[0]
        assert '"ended_at": 7310' in notes[0]
        assert str(tmp_path / "error") in notes[0]
        assert "private-secret" not in notes[0]
        if rc == 0:
            assert "successful command" in notes[0]
            assert "not an accepted result" in notes[0]


def test_failed_attempt_notes_include_bounded_redacted_stderr(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    error = tmp_path / "error"
    secret = "configured-private-credential"
    error.write_text("old log\n" * 2000 + "token=" + secret +
                     "\nValueError: could not convert string to float: ''\n")
    monkeypatch.setenv("OPENROUTER_KEY", secret)
    record = dict(attempt_id="attempt-failed", launched=True, finished=True,
                  phase="method", rc=1, stderr_path=str(error))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"attempt-failed": record})
    note = module._failed_attempt_notes("host-ledger", "method")[0]
    assert "ValueError: could not convert string to float" in note
    assert secret not in note
    assert "[REDACTED]" in note
    assert len(note) < 3000


def test_missing_handoff_keeps_success_after_earlier_failure(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    records = {
        "failed": dict(attempt_id="failed", phase="method", launched=True,
                       finished=True, rc=1),
        "completed": dict(attempt_id="completed", phase="method", launched=True,
                          finished=True, rc=0, stdout_path=str(tmp_path / "measured.stdout"),
                          stdout_sha256="a" * 64, checkpoint=str(tmp_path / "checkpoint")),
    }
    monkeypatch.setattr(module, "_attempt_records", lambda _: records)
    notes = module._failed_attempt_notes("host-ledger", "method")
    assert len(notes) == 2
    assert '"rc": 1' in notes[0]
    assert '"rc": 0' in notes[1]
    assert str(tmp_path / "checkpoint") in notes[1]
    assert str(tmp_path / "measured.stdout") in notes[1]
    assert "not an accepted result" in notes[1]
    assert "a" * 64 in notes[1]
    assert "method_metric" not in notes[1]


def test_missing_result_propagates_host_failure_notes_to_next_attempt(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    note = "Host-recorded execution failure: attempt-real rc=-1"
    monkeypatch.setattr(module, "_failed_attempt_notes", lambda *args: [note])
    res = run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                             backend=_FakeBackend(write_result=False), clone_fn=_fake_clone)
    assert note in res["error"] and note in res["verdict"]["reason"]
    assert note in res["diagnosis_evidence"]
    assert res["candidate"] is None and not res["verdict"]["improved"]


@pytest.mark.parametrize("optimizer", ["SGD", "Adam", "AdamW"])
def test_optimizer_updates_require_a_registered_attempt(tmp_path, optimizer):
    import sys
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve import agentic_improve as core
    pytest.importorskip("torch")

    program = r'''
import json, os
from pathlib import Path
import torch
torch.set_num_threads(1)
weight = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.__OPTIMIZER__([weight], lr=0.1)
print("IMPORTS_READY", flush=True)
weight.square().backward()
optimizer.step()
payload = {"baseline": float(weight.detach().square()), "baseline_valid": True,
           "method": None, "metric": {"name": "weight_square", "unit": "dimensionless", "lower_is_better": True},
           "attempt_id": os.environ.get("AUTORESEARCH_ATTEMPT_ID")}
Path(os.environ["AUTORESEARCH_ATTEMPT_RESULT"]).write_text(json.dumps(payload))
print(json.dumps(payload))
'''.replace("__OPTIMIZER__", optimizer)

    class Backend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            repo = Path(self._mounts[0]) / "repo"
            bare = subprocess.run([sys.executable, "-c", program], env=self._env, cwd=repo,
                                  capture_output=True, text=True, timeout=45)
            assert "IMPORTS_READY" in bare.stdout
            assert bare.returncode != 0, "unregistered optimizer updated the candidate"
            assert "ar-attempt" in bare.stderr
            assert not (repo / "ar_result.json").exists()
            guard = Path(self._env["PATH"].split(os.pathsep)[0])
            registered = subprocess.run(
                [str(guard / "ar-attempt"), sys.executable, "-c", program],
                env=self._env, cwd=repo, capture_output=True, text=True, timeout=60)
            assert registered.returncode == 0, registered.stderr
            return SimpleNamespace(status="FINISHED", event_summary="")

        def execute_attempt_command(self, handle, cmd, *, cwd, timeout=None):
            result = subprocess.run(shlex.split(cmd), env=self._env, cwd=cwd,
                                    capture_output=True, text=True, timeout=45)
            return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout,
                                   stderr=result.stderr)

    result = run_agentic_improve(
        "measure weight square", repo_url="https://x/r", workroot=str(tmp_path / "run"),
        backend=Backend(), clone_fn=_fake_clone, execution_phase="baseline", enforce_fairness=False)
    assert result["baseline"].valid is True, result.get("error")
    assert 0 < result["baseline"].metric_value < 1
    records = core._attempt_records(str(next(tmp_path.glob(".repo-attempt-authority-*/attempts.jsonl"))))
    assert len(records) == 1
    assert next(iter(records.values()))["rc"] == 0


@pytest.mark.parametrize("failure", [None, "run_after_write", "infra_after_write"])
def test_method_phase_uses_frozen_comparator_on_every_result_path(tmp_path, failure):
    """Catch normal or salvaged results trusting the agent-reported comparator."""
    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    fingerprint = contract_fingerprint(contract)
    frozen = {
        "metric_value": 1.8,
        "metric_name": "val_loss",
        "lower_is_better": True,
        "fixed_config": "steps=1000",
        "repo_commit": commit,
        "benchmark_contract": contract, "contract_fingerprint": fingerprint,
        "host_attestation": _frozen_host_attestation(contract),
    }
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.6, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000", "notes": "candidate",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, fail=failure)

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline=frozen,
        budget_s=900,
    )

    assert "If `$AUTORESEARCH_DATA_ROOT` is set" in be.task

    assert res["baseline"].metric_value == 1.8
    assert res["candidate"].metric_value == 1.7
    assert res["metric"] == "val_loss"
    assert res["lower_is_better"] is True
    assert res["verdict"]["verdict"] == "pass"
    assert res["verdict"]["improved"] is True
    assert res["config"] == "steps=1000"
    assert commit not in be.task
    assert "host-enforced fairness contract" not in be.task
    assert "Frozen immutable file hashes" not in be.task
    assert "~900s experiment budget" in be.task
    assert "reserved only for writing" in be.task


@pytest.mark.parametrize("lower_is_better", [True, False])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_method_phase_rejects_nonfinite_candidate(tmp_path, lower_is_better, value):
    """A non-finite candidate is never a valid measurement or a passing improvement."""
    clone, commit = _local_git_clone(tmp_path)
    be = _FakeBackend(result={
        "metric": "score", "baseline": 2.0, "method": value,
        "lower_is_better": lower_is_better, "config": "steps=10",
    })

    res = run_agentic_improve(
        "improve score", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 2.0, "metric_name": "score",
            "lower_is_better": lower_is_better,
            "fixed_config": "steps=10", "repo_commit": commit,
        },
    )

    assert res["candidate"].valid is False
    assert res["verdict"]["verdict"] == "fail"
    assert res["verdict"]["improved"] is False


@pytest.mark.parametrize("failure", ["run_after_write", "infra_after_write"])
def test_method_phase_salvages_candidate_without_agent_baseline(tmp_path, failure):
    """Method salvage uses the trusted frozen baseline, not an omitted agent baseline."""
    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    fingerprint = contract_fingerprint(contract)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000", "notes": "candidate complete",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, fail=failure)

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["baseline"].metric_value == 1.8
    assert res["baseline"].valid is True
    assert res["candidate"].metric_value == 1.7
    assert res["candidate"].valid is True
    assert res["verdict"]["verdict"] == "pass"
    assert res["verdict"]["improved"] is True


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_method_phase_rejects_nonfinite_frozen_metric_before_sandbox(tmp_path, value):
    """Catch a new repository method run proceeding without the hard-gate artifact."""
    be = _FakeBackend()

    with pytest.raises(ValueError, match="finite metric_value"):
        run_agentic_improve(
            "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
            backend=be, clone_fn=_fake_clone, execution_phase="method",
            frozen_baseline={"metric_value": value},
        )

    assert be.opened is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_baseline_phase_rejects_nonfinite_baseline_and_candidate(tmp_path, value):
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": value, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["candidate"].valid is False
    assert res["verdict"]["verdict"] == "fail"
    assert "finite baseline" in res["verdict"]["reason"]


def test_method_phase_does_not_trust_agent_config_string(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": 1.7,
        "lower_is_better": True, "config": "steps=1001",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["baseline"].valid is True
    assert res["candidate"].valid is True
    assert res["config"] == "steps=1000"
    assert "baseline drift" not in res["verdict"]["reason"]


def test_method_phase_accepts_matching_structured_contract(tmp_path):
    """Catch a fair candidate being rejected despite an exact frozen run manifest."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000", "notes": "candidate",
        "run_manifest": {
            "contract_fingerprint": fingerprint,
            "controls": contract["controls"],
        },
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "host_attestation": _frozen_host_attestation(contract),
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
        },
    )

    assert res["baseline"].valid is True
    assert res["candidate"].valid is True
    assert res["candidate"].metric_value == 1.7
    assert res["verdict"]["improved"] is True
    assert res["contract_fingerprint"] == fingerprint
    assert fingerprint not in be.task


def test_method_phase_rejects_mismatched_contract_fingerprint_without_rebaselining(tmp_path):
    """A wrong frozen-contract identity is candidate misconduct, not baseline drift."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.6,
        "lower_is_better": True, "config": "steps=1000", "notes": "two GPUs",
        "run_manifest": {
            "contract_fingerprint": "sha256:" + "0" * 64,
            "controls": {"invented": True},
        },
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
                "fixed_config": "steps=1000", "repo_commit": commit,
                "benchmark_contract": contract, "contract_fingerprint": fingerprint,
                "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.8
    assert res["candidate"].valid is False
    assert res["diagnosis"] == "candidate_contract_violation"
    assert "baseline drift" not in res["verdict"]["reason"]


def test_method_phase_matching_commit_opens_sandbox(tmp_path):
    clone, commit = _local_git_clone(tmp_path)
    be = _FakeBackend(result={
        "metric": "loss", "baseline": 2.0, "method": 1.9,
        "lower_is_better": True, "config": "steps=10",
    })

    run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 2.0, "metric_name": "loss", "lower_is_better": True,
            "fixed_config": "steps=10", "repo_commit": commit,
        },
    )

    assert be.opened is True


def test_method_phase_fetches_frozen_commit_missing_from_fresh_shallow_clone(tmp_path, monkeypatch):
    _, frozen_commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    subprocess.run(["git", "-C", str(source), "commit", "--allow-empty", "-m", "new upstream HEAD"],
                   check=True, capture_output=True)
    missing_before_fetch = []

    def shallow_clone(_url, dest):
        subprocess.run(["git", "clone", "-q", "--depth", "1", source.as_uri(), dest], check=True)
        probe = subprocess.run(["git", "-C", dest, "cat-file", "-e", frozen_commit],
                               capture_output=True)
        missing_before_fetch.append(probe.returncode != 0)

    backend = _FakeBackend()
    events = []
    checkout_timeouts = []
    original_run = subprocess.run

    def record_checkout_timeout(args, *pargs, **kwargs):
        if isinstance(args, (list, tuple)) and "checkout" in args and "--detach" in args:
            checkout_timeouts.append(kwargs.get("timeout"))
        return original_run(args, *pargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_checkout_timeout)
    run_agentic_improve(
        "run method", repo_url=source.as_uri(), workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=shallow_clone, execution_phase="method",
        frozen_baseline={"metric_value": 2.0, "metric_name": "loss",
                         "lower_is_better": True, "repo_commit": frozen_commit},
        emit=lambda event, payload: events.append((event, payload)),
    )

    assert missing_before_fetch == [True]
    assert backend.opened is True
    head = subprocess.check_output(["git", "-C", str(tmp_path / "run/repo"),
                                    "rev-parse", "HEAD"], text=True).strip()
    assert head == frozen_commit
    assert checkout_timeouts == [300]
    assert ("agentic_frozen_commit_fetch", {"repo_commit": frozen_commit, "exit_code": 0}) in events


def test_method_skips_checkout_when_clone_is_already_at_frozen_commit(tmp_path, monkeypatch):
    clone, frozen_commit = _local_git_clone(tmp_path)
    original_run = subprocess.run

    def reject_redundant_checkout(args, *pargs, **kwargs):
        if isinstance(args, (list, tuple)) and "checkout" in args and "--detach" in args:
            raise AssertionError("checkout of the existing frozen HEAD is redundant")
        return original_run(args, *pargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", reject_redundant_checkout)
    backend = _FakeBackend()
    run_agentic_improve(
        "run method", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="method",
        frozen_baseline={"metric_value": 2.0, "metric_name": "loss",
                         "lower_is_better": True, "repo_commit": frozen_commit},
    )

    assert backend.opened is True


def test_method_phase_commit_mismatch_fails_before_sandbox(tmp_path):
    clone, commit = _local_git_clone(tmp_path)
    be = _FakeBackend()

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 2.0, "metric_name": "loss", "lower_is_better": True,
            "fixed_config": "steps=10", "repo_commit": "0" * len(commit),
        },
    )

    assert be.opened is False
    assert "baseline drift" in res["error"]
    assert res["repo_commit"] == commit


def test_method_phase_unparseable_commit_fails_before_sandbox(tmp_path):
    be = _FakeBackend()

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=be, clone_fn=_nongit_fake_clone, execution_phase="method", frozen_baseline={
            "metric_value": 2.0, "metric_name": "loss", "lower_is_better": True,
            "fixed_config": "steps=10", "repo_commit": "a" * 40,
        },
    )

    assert be.opened is False
    assert "could not resolve repository commit" in res["error"]


def test_missing_result_degrades_to_invalid(tmp_path):
    be = _FakeBackend(write_result=False)          # agent wrote no ar_result.json
    res = run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                              backend=be, clone_fn=_fake_clone)
    assert res["baseline"] is None and res["candidate"] is None
    assert res["error"] == "no result" and be.closed
    assert "agent status=FINISHED" in res["verdict"]["reason"]
    assert res["repo_path"] == str(tmp_path / "repo")
    assert res["diagnosis_evidence"] == [
        "workspace: " + str(tmp_path / "repo"),
        "attempt_ledger: " + str(tmp_path / ".sandbox-control/repo/attempts.tsv"),
    ]


@pytest.mark.parametrize("phase,write_result,output_case", [
    ("baseline", True, "complete"), ("method", True, "complete"),
    ("baseline", False, "complete"), ("method", False, "complete"),
    ("baseline", False, "preflight"), ("method", False, "preflight"),
    ("baseline", False, "ambiguous"), ("method", False, "wrong_id"),
    ("baseline", False, "failed_rc"), ("method", False, "failed_gate"),
])
def test_provider_timeout_waits_for_owned_attempt_before_teardown(
        tmp_path, phase, write_result, output_case):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve import agentic_improve as core

    started, failed, release, closed = (Event() for _ in range(4))

    class Backend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            repo = Path(self._mounts[0]) / "repo"
            if phase == "method":
                (repo / "README.md").write_text("measured treatment\n")
            guard = Path(self._env["PATH"].split(os.pathsep)[0])
            self.child = subprocess.Popen(
                [str(guard / "ar-attempt"), "python", "train.py"],
                env=self._env, cwd=repo, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            assert started.wait(5)
            failed.set()
            raise RuntimeError("Conversation error code: LLMTimeoutError; llm_provider_timeout")

        def execute_attempt_command(self, handle, cmd, *, cwd, timeout=None):
            started.set()
            assert release.wait(10)
            if closed.is_set():
                return SimpleNamespace(exit_code=143, stdout="", stderr="sandbox stopped")
            attempt_id = next(p.split("=", 1)[1] for p in shlex.split(cmd)
                              if p.startswith("AUTORESEARCH_ATTEMPT_ID="))
            payload = {
                "baseline": 1.8, "baseline_valid": True,
                "method": 1.6 if phase == "method" else None,
                "candidate_valid": phase == "method", "attempt_id": attempt_id,
                "metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
            }
            if write_result:
                self.original_result = json.dumps({**payload, "notes": "original handoff metadata"})
                Path(cwd, "ar_result.json").write_text(self.original_result)
            else:
                payload.pop("attempt_id")
            if output_case == "preflight":
                payload = {"setup_ok": True}
            elif output_case == "wrong_id":
                payload["attempt_id"] = "attempt-unrelated"
            elif output_case == "failed_gate":
                payload["candidate_valid"] = False
            stdout = json.dumps(payload)
            if output_case == "ambiguous":
                stdout += "\n" + json.dumps({**payload, "baseline": 0.1})
            return SimpleNamespace(exit_code=1 if output_case == "failed_rc" else 0,
                                   stdout=stdout, stderr="")

        def close_sandbox(self, handle):
            closed.set()
            super().close_sandbox(handle)

    backend = Backend()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_agentic_improve, "q", repo_url="https://x/r", workroot=str(tmp_path / "run"),
            backend=backend, clone_fn=_fake_clone, execution_phase=phase,
            enforce_fairness=False,
            frozen_baseline={"metric_value": 1.8, "metric_name": "val_loss",
                             "lower_is_better": True} if phase == "method" else None)
        try:
            assert failed.wait(5)
            prematurely_closed = closed.wait(3)
        finally:
            release.set()
        result = future.result(timeout=10)
    backend.child.wait(timeout=5)

    assert not prematurely_closed, "provider timeout stopped an owned running experiment"
    assert closed.is_set()
    measured = result["candidate" if phase == "method" else "baseline"]
    if output_case in {"preflight", "ambiguous", "wrong_id", "failed_rc"}:
        assert measured is None
        assert "LLMTimeoutError" in result["error"]
    else:
        assert measured.valid is (output_case != "failed_gate")
        assert measured.metric_value == (1.6 if phase == "method" else 1.8)
        if output_case == "failed_gate":
            assert result["verdict"]["improved"] is False
    if write_result:
        assert (tmp_path / "run/repo/ar_result.json").read_text() == backend.original_result
    records = core._attempt_records(str(next(
        tmp_path.glob(".repo-attempt-authority-*/attempts.jsonl"))))
    assert len(records) == 1
    assert next(iter(records.values()))["rc"] == (1 if output_case == "failed_rc" else 0)


@pytest.mark.parametrize("failure", [None, "run_after_write", "infra_after_write"])
@pytest.mark.parametrize("launched", [False, True])
def test_missing_result_counts_launched_method_even_when_agent_fails(tmp_path, failure, launched):
    backend = _FakeBackend(write_result=False, record_attempt=launched, fail=failure)
    result = run_agentic_improve(
        "check current component", repo_url="https://example.org/repo",
        workroot=str(tmp_path), backend=backend, clone_fn=_fake_clone,
        execution_phase="method", enforce_fairness=False,
        frozen_baseline={"metric_value": 1.8, "metric_name": "val_loss",
                         "metric_unit": "nats", "lower_is_better": True})
    assert result.get("component_execution_started", False) is launched
    assert result["candidate"] is None
    assert result["verdict"]["improved"] is False
    assert backend.closed


@pytest.mark.parametrize("case", ["changed", "wrong_metric", "failed_gate", "tampered_checkpoint"])
def test_provider_timeout_restores_reported_measured_source_without_rerun(tmp_path, case):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve import agentic_improve as core

    class Backend(_FakeBackend):
        def run_agent_task(self, handle, task, **kwargs):
            super().run_agent_task(handle, task, **kwargs)
            repo = Path(self._mounts[0]) / "repo"
            self.original_result = (repo / "ar_result.json").read_bytes()
            payload = json.loads(self.original_result)
            self.reported_attempt = payload["attempt_id"]
            record = core._attempt_records(str(next(
                tmp_path.glob(".repo-attempt-authority-*/attempts.jsonl"))))[self.reported_attempt]
            if case == "wrong_metric":
                payload["method"] = 0.01
                (repo / "ar_result.json").write_text(json.dumps(payload))
            if case == "tampered_checkpoint":
                Path(record["checkpoint"], "README.md").write_text("tampered\n")
            (repo / "README.md").write_text("unmeasured follow-up\n")
            guard = Path(self._env["PATH"].split(os.pathsep)[0])
            subprocess.run([str(guard / "ar-attempt"), "inspect-only"], env=self._env,
                           cwd=repo, check=True, capture_output=True, text=True)
            raise RuntimeError("Conversation error code: LLMTimeoutError; llm_provider_timeout")

        def execute_attempt_command(self, handle, cmd, **kwargs):
            if "inspect-only" in shlex.split(cmd):
                return SimpleNamespace(exit_code=0, stdout='{"setup_ok":true}', stderr="")
            return super().execute_attempt_command(handle, cmd, **kwargs)

    backend = Backend(result={"metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
                              "baseline": 1.8, "method": 1.6,
                              "lower_is_better": True, "candidate_valid": case != "failed_gate"})
    events = []
    result = run_agentic_improve(
        "q", repo_url="https://x/r", workroot=str(tmp_path / "run"), backend=backend,
        clone_fn=_fake_clone, execution_phase="method", enforce_fairness=False,
        emit=lambda *args: events.append(args),
        frozen_baseline={"metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True})
    records = core._attempt_records(str(next(tmp_path.glob(".repo-attempt-authority-*/attempts.jsonl"))))
    assert len(records) == 2, "recovery launched another train/eval"
    if case != "changed":
        assert not getattr(result.get("candidate"), "valid", False)
        assert (tmp_path / "run/repo/README.md").read_text() == "unmeasured follow-up\n"
        return
    assert result["candidate"].valid is True, (result.get("candidate_admissibility"), events[-8:])
    assert result["candidate"].metric_value == 1.6
    assert result["measurement_evidence"]["attempt_id"] == backend.reported_attempt
    assert (tmp_path / "run/repo/README.md").read_text() == "method treatment\n"
    assert (tmp_path / "run/repo/ar_result.json").read_bytes() == backend.original_result
    recovered = next(data for _, data in events if data.get("stage") == "measured_checkpoint_restored")
    assert Path(recovered["unmeasured_checkpoint"], "README.md").read_text() == "unmeasured follow-up\n"


@pytest.mark.parametrize("temporary_mode_denied", [False, True])
def test_checkpoint_materialization_replaces_foreign_owned_file_atomically(
        tmp_path, monkeypatch, temporary_mode_denied):
    from autoresearch.experiments.repo_improve import agentic_improve as core
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
    source = tmp_path / "source"
    _fake_clone("unused", str(source))
    (source / "train.py").write_text("MEASURED = True\n")
    checkpoint = snapshot_checkpoint(str(source), "attempt-measured")
    target = tmp_path / "target"
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True)
    destination = target / "train.py"
    destination.write_text("UNMEASURED = True\n")
    original_chmod = os.chmod
    def chmod(path, mode, *args, **kwargs):
        if Path(path) == destination or temporary_mode_denied:
            raise PermissionError("cannot chmod another user's existing file")
        return original_chmod(path, mode, *args, **kwargs)
    monkeypatch.setattr(core.os, "chmod", chmod)
    if temporary_mode_denied:
        with pytest.raises(PermissionError):
            core._apply_prior_worktree(str(target), checkpoint)
        assert destination.read_text() == "UNMEASURED = True\n"
    else:
        assert core._apply_prior_worktree(str(target), checkpoint) == 1
        assert destination.read_text() == "MEASURED = True\n"
    assert not list(target.glob("*.ar-restore-*"))


@pytest.mark.parametrize("location,phase,launched,expected", [
    ("repo", "method", True, True),
    ("repo/subtask", "method", True, True),
    ("repo-control", "method", True, False),
    ("repo-other", "method", True, False),
    ("repo", "baseline", True, False),
    ("repo", "method", False, False),
])
def test_failed_execution_accounting_uses_launch_not_metric(
        tmp_path, monkeypatch, location, phase, launched, expected):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    # A command still running when the agent disconnects has no finish/metric.
    record = {"launched": launched, "phase": phase, "cwd": str(tmp_path / location)}
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"attempt-real": record})
    evidence = module._failed_execution_accounting("host-ledger", str(tmp_path / "repo"), "method")
    assert evidence.get("component_execution_started", False) is expected
    assert "component_attempt_id" not in evidence
    assert "method_metric" not in evidence


def test_missing_result_preserves_agent_explanation_without_accepting_claimed_metric(tmp_path):
    class ExplainingBackend(_FakeBackend):
        def run_agent_task(self, *args, **kwargs):
            result = super().run_agent_task(*args, **kwargs)
            result.event_summary = (
                "Agent finish message (unverified): Claimed accuracy=0.99; evaluator not run.")
            return result
    backend = ExplainingBackend(write_result=False)
    result = run_agentic_improve(
        "q", repo_url="https://x/r", workroot=str(tmp_path),
        backend=backend, clone_fn=_fake_clone)
    assert "evaluator not run" in result["report"]
    assert result["baseline"] is None and result["candidate"] is None
    assert result["verdict"]["verdict"] == "fail"
    assert result["verdict"]["improved"] is False


def test_infra_failure_degrades(tmp_path):
    be = _FakeBackend(fail="open")
    res = run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                              backend=be, clone_fn=_fake_clone)
    assert res["baseline"] is None and "error" in res


def test_clone_repo_retries_then_full_clone(tmp_path, monkeypatch):
    """fresh clone 可靠性: shallow clone 大仓库从某些 host 会截断成空目录(git rc=0)。_clone_repo
    要重试,最后一次用全克隆(无 --depth)兜底。"""
    from autoresearch.experiments.repo_improve import agentic_improve as ai
    monkeypatch.setenv("AUTORESEARCH_CLONE_MIRROR_DIR", str(tmp_path / "no_mirror"))

    class _R:
        returncode = 0
        stderr = ""
        stdout = ""

    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        if "--depth" not in args:                      # 只有全克隆才真的产出文件
            os.makedirs(args[-1], exist_ok=True)
            open(os.path.join(args[-1], "README"), "w").close()
        return _R()

    monkeypatch.setattr(ai.subprocess, "run", fake_run)
    dest = str(tmp_path / "repo")
    ai._clone_repo("https://github.com/x/bigrepo", dest)
    assert len(calls) == 3                              # 2 次 shallow 空 + 1 次全克隆
    assert "--depth" not in calls[2]                    # 最后一次是全克隆兜底
    assert os.path.exists(os.path.join(dest, "README"))


@pytest.mark.parametrize("directory_mode,file_mode", [(0o755, 0o644), (0o700, 0o600)])
def test_writable_workspace_skips_redundant_chmod_and_external_symlinks(
        tmp_path, monkeypatch, directory_mode, file_mode):
    from autoresearch.experiments.repo_improve.agentic_improve import _make_workspace_writable

    root = tmp_path / "repo"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "source.c"
    source.write_text("int x;\n")
    external = tmp_path / "external.c"
    external.write_text("not part of this workspace\n")
    external.chmod(0o444)
    (nested / "linked.c").symlink_to(external)
    root.chmod(directory_mode)
    nested.chmod(directory_mode)
    source.chmod(file_mode)
    calls = []
    real_chmod = os.chmod

    def track_chmod(path, mode):
        calls.append(Path(path))
        real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", track_chmod)
    _make_workspace_writable(str(root))
    assert calls.count(nested) == 1
    assert set(calls) == {root, nested, source}
    assert stat.S_IMODE(nested.stat().st_mode) == 0o777
    assert stat.S_IMODE(source.stat().st_mode) == 0o666
    assert stat.S_IMODE(external.stat().st_mode) == 0o444
    calls.clear()
    _make_workspace_writable(str(root))
    assert calls == []


def test_writable_dependency_paths_do_not_scan_unrelated_checkout(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _make_workspace_paths_writable,
    )

    root = tmp_path / "control"
    changed = root / "method" / "component.py"
    unrelated = root / "large" / "data.bin"
    changed.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    changed.write_text("x = 1\n")
    unrelated.write_bytes(b"data")
    for path in (root, changed.parent, unrelated.parent):
        path.chmod(0o755)
    changed.chmod(0o644)
    unrelated.chmod(0o444)
    calls = []
    real_chmod = os.chmod

    def track_chmod(path, mode):
        calls.append(Path(path))
        real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", track_chmod)
    _make_workspace_paths_writable(str(root), ["method/component.py"])

    assert changed.stat().st_mode & 0o002
    assert changed.parent.stat().st_mode & 0o002
    assert unrelated.stat().st_mode & 0o002 == 0
    assert unrelated not in calls


def test_cloned_workspace_is_writable_by_sandbox_user(tmp_path):
    """A host-owned read-only clone must be made writable before it is bind-mounted.

    The official OpenHands image runs as a different uid, so leaving the copied repository at
    the mirror's 0755/0644 modes makes both lockfile updates and source edits fail in-container.
    """
    repo_path = tmp_path / "run" / "repo"

    def read_only_clone(_url, dest):
        repo = Path(dest)
        repo.mkdir(parents=True)
        source = repo / "train.py"
        source.write_text("print('baseline')\n", encoding="utf-8")
        source.chmod(0o444)
        repo.chmod(0o555)

    try:
        result = run_agentic_improve(
            "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
            backend=_FakeBackend(), clone_fn=read_only_clone,
        )
    finally:
        # Keep pytest's tmpdir cleanup reliable even while this test is red.
        repo_path.chmod(0o755)

    assert result["candidate"].metric_value == 1.60
    assert (repo_path / "train.py").stat().st_mode & 0o002


@pytest.mark.parametrize("dirname", ["artifacts", ".pkg-cache"])
def test_reused_writable_runtime_directory_does_not_require_host_chmod(
        tmp_path, monkeypatch, dirname):
    """A prior sandbox may own a runtime dir even when it is already mode 777."""
    if dirname == ".pkg-cache":
        monkeypatch.setenv("AUTORESEARCH_SANDBOX_PKG_CACHE_DIR", "")
    runtime_dir = tmp_path / dirname
    runtime_dir.mkdir()
    runtime_dir.chmod(0o777)
    real_chmod = os.chmod

    def reject_redundant_chmod(path, mode):
        if Path(path) == runtime_dir:
            raise PermissionError("sandbox-owned directory")
        return real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", reject_redundant_chmod)
    result = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path),
        backend=_FakeBackend(), clone_fn=_fake_clone,
    )
    assert result["candidate"].metric_value == 1.60


def test_cloned_repository_has_no_answer_path_name_policy(tmp_path):
    repo_path = tmp_path / "run" / "repo"

    def clone_with_answer(_url, dest):
        repo = Path(dest)
        (repo / "solution").mkdir(parents=True)
        (repo / "solution" / "answer.py").write_text("secret\n", encoding="utf-8")
        (repo / "environment").mkdir()
        (repo / "environment" / "starter.py").write_text("starter\n", encoding="utf-8")

    result = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=_FakeBackend(), clone_fn=clone_with_answer,
    )

    assert result["candidate"].metric_value == 1.60
    assert stat.S_IMODE((repo_path / "solution").stat().st_mode) & 0o004
    assert stat.S_IMODE((repo_path / "solution" / "answer.py").stat().st_mode) & 0o004
    assert (repo_path / "environment" / "starter.py").stat().st_mode & 0o002


def test_clone_failure_degrades(tmp_path):
    def _boom(url, dest):
        raise RuntimeError("git clone failed")
    be = _FakeBackend()
    res = run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                              backend=be, clone_fn=_boom)
    assert res["baseline"] is None and "clone failed" in res["error"]


def test_task_prompt_and_llm_isolation(tmp_path):
    be = _FakeBackend()
    run_agentic_improve("lower the rmsnorm CUDA kernel latency", repo_url="https://x/SOL",
                        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
                        budget_s=300, metric_hint="latency_ms", lower_is_better_hint=True)
    assert "rmsnorm CUDA kernel" in be.task            # the raw goal reaches the agent
    assert "ar_result.json" in be.task                 # told where to write the result
    assert "300s" in be.task                           # wall-clock budget stated
    assert "latency_ms" in be.task                     # metric hint surfaced
    assert be.llm is None                              # never hand our wrapper to the SDK agent
    # A: a worse-than-baseline measured result is an honest negative -- the agent must record its
    # real number in "method", not leave it null (micrograd left method=null despite a 0.9225 run).
    assert "honest negative" in be.task
    assert "worse-than-baseline included" in be.task
    # Within-session analysis (老师 2026-08-07): the agent is told to LOG each variant and READ the
    # analysis agent's direction before designing the next one (directed, not blind, iteration).
    assert "variants.jsonl" in be.task
    assert "direction.txt" in be.task
    assert "one complete JSON object per physical line" in be.task
    assert "validate every non-empty line with json.loads" in be.task


def test_all_repo_prompts_forbid_solution_answer_material():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt,
        _method_task_prompt,
        _task_prompt,
    )

    prompts = [
        _task_prompt("goal", "/repo", "/repo/ar_result.json", None, "loss", True),
        _baseline_task_prompt(
            "goal", "/repo", "/repo/ar_result.json", "loss", True),
        _method_task_prompt(
            "goal", "/repo", "/repo/ar_result.json", None,
            {
                "metric_name": "loss",
                "metric_unit": "nats",
                "metric_value": 1.0,
                "lower_is_better": True,
            },
        ),
    ]

    for prompt in prompts:
        assert "directory name is `solution` or `solutions`" in prompt
        assert "recover through Git history" in prompt


@pytest.mark.parametrize("evaluation_rule", [
    "Local evaluation is allowed. Prepare fresh fixtures with the official setup script.",
    "Final test fixtures belong to the isolated evaluator. Do not create them in the editing sandbox.",
])
def test_method_prompt_scopes_evaluation_setup_to_repository_permissions(evaluation_rule):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "improve the official metric", "/repo", "/repo/ar_result.json", 1200,
        {"metric_name": "score", "metric_value": 1.0, "lower_is_better": False,
         "repository_contract": {
             "sources": [{"path": "instruction.md", "content": evaluation_rule}]}},
    )

    assert evaluation_rule in prompt
    assert "When the task or official repository reserves a final test for isolated evaluation" in prompt
    assert "If the repository authorizes local evaluation" in prompt
    assert "prepare fresh fixtures using its unchanged frozen setup source" in prompt
    assert "This permission does not override protected-path or data-access rules" in prompt
    assert "Only the official evaluator may consume evaluation-only examples and labels" in prompt
    assert "Do not run setup or data-generation commands in the editing sandbox when they create" not in prompt


def test_method_prompt_forbids_candidate_selection_on_held_out_labels():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "improve the official metric",
        "/repo",
        "/repo/ar_result.json",
        1200,
        {
            "metric_name": "score",
            "metric_value": 1.0,
            "lower_is_better": False,
            "fixed_config": "official",
        },
    )

    assert "held-out/test labels" in prompt
    assert "deterministic train-derived validation split" in prompt
    assert "its score on that split is an in-sample diagnostic" in prompt
    assert "permits a development validation set for model selection or calibration" in prompt
    assert "validation feedback is not an independent final-test result" in prompt
    assert "does not permit label-dependent answers at final inference" in prompt
    assert "visible to Git at this outer repository root" in prompt
    assert "git update-index --add --cacheinfo" in prompt
    assert "verify `git diff HEAD -- <files>`" in prompt
    assert "Do not add datasets, weights, environments, or change HEAD" in prompt


def test_sandbox_dependency_contract_uses_direct_network_and_real_wheelhouse_only(
        tmp_path, monkeypatch):
    """A missing default /opt/ar-wheels must never make uv fail before it can use indexes.

    The sandbox must also be insulated from inherited proxy variables: downloads on DGX are
    direct, while the persistent package cache remains mounted for reuse.
    """
    cache = tmp_path / "pkg-cache"
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_PKG_CACHE_DIR", str(cache))
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_PIP_MIRROR", "1")
    monkeypatch.delenv("AR_TORCH_WHEELS", raising=False)
    be = _FakeBackend()

    workroot = tmp_path / "run"
    run_agentic_improve("lower loss", repo_url="https://x/nanochat",
                        workroot=str(workroot), backend=be, clone_fn=_fake_clone)

    assert be._env["NO_PROXY"] == "*" and be._env["no_proxy"] == "*"
    # A sandbox may temporarily redirect UV_CACHE_DIR while repairing one dependency.  Symlink
    # mode would then leave the shared venv pointing into that container's /tmp after it exits;
    # hardlink mode stays self-contained (and uv safely falls back to copying across filesystems).
    assert be._env["UV_LINK_MODE"] == "hardlink"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        assert be._env[key] == ""
    assert be._env["GIT_TERMINAL_PROMPT"] == "0"
    assert be._env["PIP_NO_INPUT"] == "1"
    assert "/opt/ar-wheels" not in be._env["UV_FIND_LINKS"]
    assert "mirrors.aliyun.com/pytorch-wheels" in be._env["UV_FIND_LINKS"]
    assert str(cache) in be._mounts
    assert "PRE-INSTALLED TORCH" not in be.task
    # Both paths must survive a fresh container in the same outer research session. In
    # particular, a shared venv whose interpreter symlink points into container-local
    # ~/.local/share/uv/python is broken as soon as that container is replaced.
    assert be._env["UV_PYTHON_INSTALL_DIR"] == str(workroot / ".uv-python")
    assert be._env["XDG_CACHE_HOME"] == str(workroot / ".cache")
    assert be._env["NLTK_DATA"] == str(workroot / ".cache" / "nltk_data")
    assert Path(be._env["NLTK_DATA"]).is_dir()
    assert Path(be._env["NLTK_DATA"]).stat().st_mode & 0o777 == 0o777
    assert be._env["AUTORESEARCH_ARTIFACT_DIR"] == str(workroot / "artifacts")
    assert (workroot / "artifacts").is_dir()
    assert (workroot / "artifacts").stat().st_mode & 0o777 == 0o777
    assert str(workroot) in be._mounts
    assert "Do NOT start a sizeable network pull" in be.task
    assert "it will still finish" not in be.task


def test_public_sandbox_defaults_are_session_local_and_use_official_indexes(
        tmp_path, monkeypatch):
    """A clean public install must not inherit the authors' cluster paths or mirrors."""
    for key in (
        "AUTORESEARCH_SANDBOX_PYTHON_ROOT",
        "AUTORESEARCH_SANDBOX_PKG_CACHE_DIR",
        "AUTORESEARCH_SANDBOX_HF_HOME",
        "AUTORESEARCH_SANDBOX_PIP_MIRROR",
        "AUTORESEARCH_CLONE_MIRROR_DIR",
        "AR_TORCH_WHEELS",
    ):
        monkeypatch.delenv(key, raising=False)
    be = _FakeBackend()

    run_agentic_improve("evaluate text", repo_url="https://example.test/repo",
                        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)

    assert be._env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / ".uv-python")
    assert be._env["UV_CACHE_DIR"] == str(tmp_path / ".pkg-cache")
    assert be._env["HF_HOME"] == str(tmp_path / ".cache" / "huggingface")
    for key in ("PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_TRUSTED_HOST",
                "UV_INDEX", "UV_INDEX_STRATEGY", "UV_FIND_LINKS"):
        assert key not in be._env
    assert "/srv/shared/" not in json.dumps({"env": be._env, "mounts": be._mounts})


def test_nltk_data_survives_a_new_sandbox_without_downloading(tmp_path):
    first = _FakeBackend()
    run_agentic_improve("evaluate text", repo_url="https://example.test/repo",
                        workroot=str(tmp_path), backend=first, clone_fn=_fake_clone)
    data = Path(first._env["NLTK_DATA"])
    fixture = data / "local-evaluator-resource.txt"
    fixture.write_text("already downloaded resource\n")
    second = _FakeBackend()
    run_agentic_improve("evaluate text", repo_url="https://example.test/repo",
                        workroot=str(tmp_path), backend=second, clone_fn=_fake_clone)
    assert second._env["NLTK_DATA"] == str(data)
    assert str(tmp_path) in second._mounts
    assert fixture.read_text() == "already downloaded resource\n"


def test_recovered_session_mounts_canonical_workspace_for_attempt_broker(tmp_path):
    original = tmp_path / "original-session"
    original.mkdir()
    recovered = tmp_path / "recovered-session"
    recovered.symlink_to(original, target_is_directory=True)
    be = _FakeBackend()

    run_agentic_improve("reduce evaluation cost", repo_url="https://example.org/repo",
                        workroot=str(recovered), backend=be, clone_fn=_fake_clone)

    assert str(recovered) in be._mounts
    assert str(original) in be._mounts
    assert str(tmp_path) not in be._mounts
    assert os.path.samefile(recovered, original)


@pytest.mark.parametrize("recovered", [False, True])
def test_sandbox_mounts_held_controller_lock_read_only_at_each_alias(tmp_path, recovered):
    import fcntl
    from autoresearch.agents._reuse.openhands_backend import bind_mount_args

    original = tmp_path / "original-session"
    original.mkdir()
    workroot = original
    if recovered:
        workroot = tmp_path / "recovered-session"
        workroot.symlink_to(original, target_is_directory=True)
    lock = original / ".agentic_run.lock"
    lock.write_text("controller lock identity")
    inode = lock.stat().st_ino
    be = _FakeBackend()
    with lock.open() as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        run_agentic_improve("reduce evaluation cost", repo_url="https://example.org/repo",
                            workroot=str(workroot), backend=be, clone_fn=_fake_clone)
        args = bind_mount_args(be._mounts)
        for root in {original, workroot}:
            assert f"{lock}:{root / '.agentic_run.lock'}:ro" in args
        with lock.open() as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert lock.stat().st_ino == inode
    assert lock.read_text() == "controller lock identity"


def test_sandbox_package_cache_falls_back_when_configured_path_is_unusable(
        tmp_path, monkeypatch):
    unusable = tmp_path / "not-a-directory"
    unusable.write_text("occupied")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_PKG_CACHE_DIR", str(unusable))
    be = _FakeBackend()

    workroot = tmp_path / "run"
    run_agentic_improve("lower loss", repo_url="https://x/nanochat",
                        workroot=str(workroot), backend=be, clone_fn=_fake_clone)

    fallback = str(workroot / ".pkg-cache")
    assert be._env["UV_CACHE_DIR"] == fallback
    assert be._env["PIP_CACHE_DIR"] == fallback
    assert fallback in be._mounts


def test_sandbox_reuses_existing_shared_python_runtime_across_sessions(
        tmp_path, monkeypatch):
    shared_python = tmp_path / "shared-python"
    shared_python.mkdir()
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_PYTHON_ROOT", str(shared_python))
    be = _FakeBackend()

    workroot = tmp_path / "run"
    run_agentic_improve("lower loss", repo_url="https://x/project",
                        workroot=str(workroot), backend=be, clone_fn=_fake_clone)

    assert be._env["UV_PYTHON_INSTALL_DIR"] == str(shared_python)
    assert str(shared_python) in be._mounts
    assert be._env["XDG_CACHE_HOME"] == str(workroot / ".cache")


def test_sandbox_uses_container_local_cuda_ordinals(tmp_path):
    """A reduced device set is renumbered densely inside the container."""
    single = _FakeBackend()
    run_agentic_improve("lower loss", repo_url="https://x/nanochat",
                        workroot=str(tmp_path / "single"), backend=single,
                        clone_fn=_fake_clone, cuda_device=7)
    assert single._env["CUDA_VISIBLE_DEVICES"] == "0"

    multi = _FakeBackend()
    run_agentic_improve("lower loss", repo_url="https://x/nanochat",
                        workroot=str(tmp_path / "multi"), backend=multi,
                        clone_fn=_fake_clone, cuda_device="3,5")
    assert multi._env["CUDA_VISIBLE_DEVICES"] == "0,1"

    for value, suffix in (([3, 5], "list"), ((3, 5), "tuple")):
        sequence = _FakeBackend()
        run_agentic_improve("lower loss", repo_url="https://x/nanochat",
                            workroot=str(tmp_path / suffix), backend=sequence,
                            clone_fn=_fake_clone, cuda_device=value)
        assert sequence._env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_explicit_existing_wheelhouse_is_mounted_and_preferred(tmp_path, monkeypatch):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    monkeypatch.setenv("AR_TORCH_WHEELS", str(wheelhouse))
    be = _FakeBackend()

    run_agentic_improve("lower loss", repo_url="https://x/nanochat",
                        workroot=str(tmp_path / "run"), backend=be, clone_fn=_fake_clone)

    assert be._env["UV_FIND_LINKS"].split()[0] == str(wheelhouse)
    assert str(wheelhouse) in be._mounts


def test_configured_data_root_is_project_neutral(tmp_path, monkeypatch):
    data_root = tmp_path / "datasets"
    data_root.mkdir()
    hf = tmp_path / "hf-cache"
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_HF_HOME", str(hf))
    be = _FakeBackend()

    run_agentic_improve("reduce latency", repo_url="https://x/arbitrary-project",
                        workroot=str(tmp_path / "run"), backend=be, clone_fn=_fake_clone)

    assert be._env["AUTORESEARCH_DATA_ROOT"] == str(data_root)
    assert be._env["HF_HOME"] == str(hf)
    assert str(data_root) in be._mounts and str(hf) in be._mounts
    assert "NANOCHAT_BASE_DIR" not in be._env
    assert "DATA_PATH" not in be._env
    assert all(not (isinstance(mount, tuple) and mount[1] == "/data")
               for mount in be._mounts)
    assert "HF_ENDPOINT" not in be._env  # direct official host unless explicitly configured


@pytest.mark.parametrize("repository_owned", [False, True])
@pytest.mark.parametrize("launcher", [False, True])
def test_candidate_fingerprint_separates_handoff_from_source(tmp_path, repository_owned, launcher):
    import ast
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    (repo / "analysis").mkdir()
    outputs = [repo / "ar_result.json", repo / "analysis/variants.jsonl"]
    for path in outputs:
        path.write_text("{}\n")
    if repository_owned:
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=test",
                        "-c", "user.email=test@example.com", "commit", "-m", "repository outputs"],
                       check=True, capture_output=True)

    fingerprint = lambda: ai._tracked_repo_state(str(repo))["tracked_diff_hash"]
    if launcher:
        bindir, _ = ai._install_attempt_guard(str(tmp_path), str(repo), str(outputs[0]))
        tree = ast.parse((Path(bindir) / "ar-attempt").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "repo_state")
        namespace = {"hashlib": hashlib, "subprocess": subprocess}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "launcher", "exec"), namespace)
        fingerprint = lambda: namespace["repo_state"](str(repo))[1]
    before = fingerprint()
    for path in outputs:
        path.write_text('{"method": 8}\n')
        subprocess.run(["git", "-C", str(repo), "add", str(path.relative_to(repo))], check=True)
    after = fingerprint()
    assert (after != before) is repository_owned
    assert after == ai._tracked_repo_state(str(repo))["tracked_diff_hash"]
    # Do not exempt arbitrary JSON configs or sibling analysis code.
    for filename in ("analysis/config.json", "analysis/run.py"):
        (repo / filename).write_text("changed\n")
        subprocess.run(["git", "-C", str(repo), "add", filename], check=True)
        assert fingerprint() != after
        after = fingerprint()


@pytest.mark.parametrize("settings, expected", [
    ({"AUTORESEARCH_SANDBOX_REASONING_EFFORT": "max",
      "AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS": "262144",
      "AUTORESEARCH_LLM_TEMPERATURE": "0.7",
      "AUTORESEARCH_SANDBOX_LLM_TIMEOUT": "1800"},
     {"reasoning_effort": "max", "max_tokens": 262144, "temperature": 0.7, "timeout": 1800}),
    ({"VIBE_REASONING_EFFORT": "high", "AUTORESEARCH_LLM_MAX_TOKENS": "65536"},
     {"reasoning_effort": "high", "max_tokens": 65536, "temperature": 0.3, "timeout": 90}),
    ({"AUTORESEARCH_SANDBOX_REASONING_EFFORT": "default", "VIBE_REASONING_EFFORT": "max"},
     {"temperature": 0.3, "timeout": 90}),
    ({"AUTORESEARCH_SANDBOX_REASONING_EFFORT": "none",
      "AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS": "2048"},
     {"reasoning_effort": "none", "max_tokens": 2048, "temperature": 0.3, "timeout": 90}),
])
def test_analysis_llm_forwards_explicit_runtime_settings(monkeypatch, settings, expected):
    import urllib.request
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    for name in ("AUTORESEARCH_SANDBOX_REASONING_EFFORT", "VIBE_REASONING_EFFORT",
                 "AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS", "AUTORESEARCH_LLM_MAX_TOKENS",
                 "AUTORESEARCH_LLM_TEMPERATURE", "AUTORESEARCH_SANDBOX_LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {"OPENROUTER_KEY": "test-key", "OPENROUTER_BASE_URL": "http://example.invalid/v1",
                        "AUTORESEARCH_SANDBOX_MODEL": "test-model", **settings}.items():
        monkeypatch.setenv(name, value)
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"bounded direction"}}]}'

    class Opener:
        def open(self, request, timeout):
            observed.update(json.loads(request.data), timeout=timeout)
            return Response()

    def build_opener(handler):
        assert handler.proxies == {}
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    assert ai._call_analysis_llm("analyze measured evidence") == "bounded direction"
    assert observed["model"] == "test-model"
    assert {key: observed[key] for key in expected} == expected
    for key in ("reasoning_effort", "max_tokens"):
        if key not in expected:
            assert key not in observed


def test_analysis_feedback_does_not_change_candidate_fingerprint(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    (repo / "analysis").mkdir()
    legacy = repo / "analysis" / "direction.txt"
    legacy.write_text("previous direction")
    subprocess.run(["git", "-C", str(repo), "add", "analysis/direction.txt"], check=True)
    (repo / ai._RESULT_BASENAME).write_text(json.dumps({
        "baseline": 10, "method": 8, "changes_summary": "implemented the component",
    }))
    monkeypatch.setattr(ai, "_call_analysis_llm", lambda *a, **k: "new measured feedback")
    before = ai._tracked_repo_state(str(repo))
    ai._analyze_once(str(repo), "improve", "runtime", True, lambda *a: None)

    assert ai._tracked_repo_state(str(repo)) == before
    assert legacy.read_text() == "previous direction"
    feedback = tmp_path / ".sandbox-control" / "repo" / "direction.txt"
    assert feedback.read_text() == "new measured feedback"
    assert str(feedback) in ai._task_prompt(
        "improve", str(repo), str(repo / ai._RESULT_BASENAME), None, "runtime", True)
    legacy.write_text("real tracked edit")
    assert ai._tracked_repo_state(str(repo)) != before


def test_analysis_sidecar_writes_direction_from_variants(tmp_path, monkeypatch):
    """The within-session analysis agent turns >=2 logged variants + baseline into a direction.txt.
    The LLM call is monkeypatched; we assert the sidecar reads variants.jsonl and persists a
    direction, and that the prompt it builds carries the goal + baseline + variants."""
    import threading
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / ai._RESULT_BASENAME).write_text(
        json.dumps({"metric": "step_ms", "baseline": 76.8}), encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant": "bigger tile", "result": 74.1, "note": "helped"}\n'
        '{"variant": "fp8", "result": 90.0, "note": "diverged"}\n', encoding="utf-8")

    seen = {}

    def _fake_llm(user, timeout=90):
        seen["user"] = user
        return "Conclusion: bigger tile fits. Next: fuse rmsnorm."

    monkeypatch.setattr(ai, "_call_analysis_llm", _fake_llm)
    stop = threading.Event()
    t = threading.Thread(
        target=ai._analysis_sidecar,
        args=(str(repo), "make each training step faster", "step_ms", True, stop,
              lambda e, p: None),
        daemon=True)
    t.start()
    dfile = Path(ai._analysis_direction_path(str(repo)))
    for _ in range(100):  # poll up to ~5s for the first pass to write
        if dfile.exists():
            break
        time.sleep(0.05)
    stop.set()
    t.join(timeout=5)
    assert dfile.exists()
    assert (repo / "analysis").stat().st_mode & 0o777 == 0o777
    assert "fuse rmsnorm" in dfile.read_text(encoding="utf-8")
    assert "make each training step faster" in seen["user"]  # goal reaches the analysis LLM
    assert "76.8" in seen["user"]                            # baseline surfaced
    assert "bigger tile" in seen["user"]                     # variants surfaced


def test_sidecar_anchors_to_confirmed_method(tmp_path, monkeypatch):
    """用户 2026-08-09: sidecar 诊断不能脱离讨论出的 method。请求含 CONFIRMED METHOD 时,分析 LLM 的
    prompt 必须带上该 method,并要求下一步锚定它(不飘到无关通用调参)。"""
    from autoresearch.experiments.repo_improve import agentic_improve as ai
    # _extract_confirmed_method 抽块 + 返回 goal
    q = ("★ CONFIRMED METHOD -- the team agreed: Newton-Muon optimizer with gradient-based head "
         "clustering. changes_outline: cluster heads after warmup. Clone and improve modded-nanogpt "
         "to reduce time to val_loss 3.28.")
    method, goal = ai._extract_confirmed_method(q)
    assert "Newton-Muon" in method and "head clustering" in method
    assert "CONFIRMED METHOD" not in goal or "modded-nanogpt" in goal   # goal 保留实际任务

    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / ai._RESULT_BASENAME).write_text(
        json.dumps({"metric": "seconds", "baseline": 76.8}), encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant": "lr_mul=1.05", "result": 76.84, "note": "no improvement"}\n'
        '{"variant": "cooldown=0.5", "result": 76.85, "note": "no improvement"}\n', encoding="utf-8")
    seen = {}
    monkeypatch.setattr(ai, "_call_analysis_llm",
                        lambda user, timeout=90: seen.setdefault("user", user) or "next: implement Newton-Muon head clustering")
    ai._analyze_once(str(repo), q, "seconds", True, lambda e, p: None)
    assert "CONFIRMED METHOD" in seen["user"]          # method 喂进了分析 prompt
    assert "Newton-Muon" in seen["user"]               # 具体 method 内容在
    assert "faithfully advances" in seen["user"].lower() or "advance" in seen["user"].lower()
    assert "Preserve any measured off-method alternative" in seen["user"]
    assert "formal host review" in seen["user"]
    assert "method mismatch alone is not proof of a dead end" in ai._ANALYSIS_SYS.lower()


def test_sidecar_keeps_late_method_components_and_host_corrections(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    method = {
        "approach": "Retain the measured data mix. " + "coverage evidence " * 120,
        "components": [{"name": "data-mix"}, {"name": "reward-normalization"},
                       {"name": "length-correction"}],
    }
    correction = "HOST CORRECTION: supply reward-normalization implementation evidence."
    request = ("★ CONFIRMED METHOD — The confirmed method:\n"
               + json.dumps(method) + "\n\n" + "Existing task constraint. " * 80
               + correction + "\nGoal: improve the frozen metric.")
    extracted, goal = ai._extract_confirmed_method(request)
    assert "reward-normalization" in extracted
    assert "length-correction" in extracted
    assert correction in goal

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ar_result.json").write_text(json.dumps({
        "baseline": 0.5, "method": 0.6, "changes_summary": "measured the data mix",
    }))
    seen = {}
    monkeypatch.setattr(ai, "_call_analysis_llm",
                        lambda user, timeout=90: seen.setdefault("user", user))
    ai._analyze_once(str(repo), request, "accuracy", False, lambda *_: None)
    assert "reward-normalization" in seen["user"]
    assert "length-correction" in seen["user"]
    assert correction in seen["user"]


@pytest.mark.parametrize("carry", ["", "Candidate accepted; do not implement more components."])
def test_carry_advice_cannot_override_current_host_repair(carry):
    from autoresearch.experiments.repo_improve.agentic_improve import _task_prompt

    prompt = _task_prompt(
        "HOST CORRECTION: implement the missing component.", "/repo", "/repo/ar_result.json",
        None, "accuracy", False, carry_direction=carry)
    assert ("PREVIOUS ROUND'S CONCLUSION" in prompt) is bool(carry)
    assert "advisory" in prompt.lower()
    assert "current host" in prompt.lower() and "take precedence" in prompt.lower()
    assert "cannot accept a candidate" in prompt


def test_analysis_sidecar_no_variants_writes_nothing(tmp_path, monkeypatch):
    """With <2 variants there is nothing to analyze -- the sidecar must not call the LLM or write."""
    import threading
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant": "only one", "result": 1.0}\n', encoding="utf-8")

    called = {"n": 0}

    def _fake_llm(user, timeout=90):
        called["n"] += 1
        return "x"

    monkeypatch.setattr(ai, "_call_analysis_llm", _fake_llm)
    stop = threading.Event()
    t = threading.Thread(
        target=ai._analysis_sidecar,
        args=(str(repo), "goal", None, None, stop, lambda e, p: None), daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=5)
    assert called["n"] == 0
    assert not Path(ai._analysis_direction_path(str(repo))).exists()


def test_prompt_env_rule_lockfile_no_global_cuda(tmp_path):
    """SOL 卡死根因(2026-08-08): agent 没用仓库 uv.lock、自己全局装 CUDA-13 nvcc 撞沙箱 ptxas-12
    → 全 kernel 编译失败。prompt 必须(通用地)要求: 有 lockfile 就 uv sync + uv run、禁止自己装
    torch/CUDA、缺 nvcc 别全局重装 CUDA 栈、部分 baseline 也算有效结果。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _task_prompt
    p = _task_prompt("lower kernel latency", "/repo", "/repo/ar_result.json", None,
                     "latency_ms", True)
    assert "uv.lock" in p and "uv run" in p                 # 强制用仓库锁环境
    assert "do NOT" in p and "CUDA" in p                    # 禁止自己重建工具链
    assert "nvidia-cuda-nvcc" in p                          # 点名禁止手装 nvcc(观察到的坑)
    assert "partial baseline" in p                          # 部分编译成功也算有效结果


def test_method_salvaged_from_variants_when_agent_left_null(tmp_path):
    """#67 (真机 speedrun 实锤): agent 测了 8 个变体(variants.jsonl 全有真实 76.8~76.93s 值)却因
    '没改进' 把 method 留 null。_build_result_from_file 必须从 variants.jsonl 捞最优非 baseline 变体
    值进 method —— 跑过就该有值(哪怕不如 baseline)。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file
    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "seconds", "baseline": 76.8, "method": None, "lower_is_better": True}),
        encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant": "Baseline (unmodified)", "result": 76.8}\n'
        '{"variant": "cooldown=0.5", "result": 76.85}\n'
        '{"variant": "lr_mul=1.05", "result": 76.84}\n'          # 最优非baseline变体
        '{"variant": "adam=0.012", "result": 76.91}\n', encoding="utf-8")
    res = _build_result_from_file(str(repo / "ar_result.json"), "modded-nanogpt",
                                  "seconds", True, lambda e, p: None)
    assert res["candidate"].metric_value == 76.84          # 捞到最优非baseline变体
    assert res["candidate"].valid                          # method 不再是 null
    assert res["verdict"]["improved"] is False             # 诚实: 不如 baseline
    assert "salvaged" in res["report"]


def test_variant_salvage_skips_faster_candidate_rejected_by_repository_gate(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "runtime", "baseline": 10.0, "method": None,
        "lower_is_better": True,
    }), encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant":"fast but wrong","result":1.0,"valid":false}\n'
        '{"variant":"slower and correct","result":2.0,"valid":true}\n',
        encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic", "runtime", True, lambda e, p: None)

    assert res["candidate"].metric_value == 2.0
    assert res["candidate"].valid is True


def test_embedded_comparison_audit_gate_failure_invalidates_finite_metric(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir()
    audit = {"observed_outcomes": {
        "official_correctness": False, "correctness_gate_passed": False}}
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "total_params", "baseline": 17988, "method": 913,
        "lower_is_better": True, "notes": "COMPARISON_AUDIT: " + json.dumps(audit),
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic", "total_params", True,
        lambda e, p: None)

    assert res["candidate"].metric_value == 913
    assert res["candidate"].valid is False
    assert res["verdict"]["improved"] is False


def test_selected_candidate_gate_is_not_overridden_by_failed_history(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir()
    audit = {"observed_outcomes": {
        "attempt_1": "accuracy=0.94 candidate_valid=false",
        "attempt_2": "accuracy=0.952 candidate_valid=true",
    }}
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "total_params", "baseline": 17988, "method": 913,
        "candidate_valid": True, "lower_is_better": True,
        "notes": "COMPARISON_AUDIT: " + json.dumps(audit),
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic", "total_params", True,
        lambda e, p: None)

    assert res["candidate"].metric_value == 913
    assert res["candidate"].valid is True
    assert res["verdict"]["improved"] is True


def test_prefixed_measurement_notes_gate_failure_invalidates_finite_metric(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": {"name": "total_params", "unit": "count",
                   "lower_is_better": True},
        "baseline": 17988, "method": 1128,
        "notes": (
            "MEASUREMENT_CONTEXT: official run completed. COMPARISON_AUDIT: "
            '{"observed_outcomes":["accuracy=0.772689 params=1128 reward=0.0 '
            'correctness=False"]}'
        ),
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic", "total_params", True,
        lambda e, p: None)

    assert res["candidate"].metric_value == 1128
    assert res["candidate"].valid is False
    assert res["verdict"]["improved"] is False
    assert "correctness/quality gate" in res["verdict"]["reason"]


def test_named_gate_met_failure_invalidates_finite_metric(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "total_params", "baseline": 17988, "method": 9520,
        "lower_is_better": True,
        "notes": (
            'COMPARISON_AUDIT: {"accuracy_gate_met":false,"accuracy":0.87} '
            "The parameter count is finite but the task gate failed."
        ),
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic", "total_params", True,
        lambda e, p: None)

    assert res["candidate"].metric_value == 9520
    assert res["candidate"].valid is False
    assert res["verdict"]["improved"] is False
    assert "correctness/quality gate" in res["verdict"]["reason"]


def test_method_salvaged_from_variant_method_field(tmp_path):
    """A measured variant may use the same top-level ``method`` field as ar_result.json."""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "quality", "baseline": 1.0, "method": None, "lower_is_better": True,
    }), encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(json.dumps({
        "variant": "confirmation", "method": 1.1, "attempt_id": "attempt-1",
    }) + "\n", encoding="utf-8")

    res = _build_result_from_file(str(repo / "ar_result.json"), "generic-repo",
                                  "quality", True, lambda e, p: None)

    assert res["candidate"].metric_value == 1.1
    assert res["candidate"].valid
    assert res["_metric_attempt_id"] == "attempt-1"


def test_best_variant_overrides_stale_nonnull_method(tmp_path):
    """真机 nanochat 2026-08-09 实锤: agent 在 14:59 写了 method=1.03595(V2 首个改进)后,继续把
    V3~V9(全更好,最优 1.014969)只 append 到 variants.jsonl,再没回写 ar_result.json。旧 salvage
    只在 method==None 时触发 -> 非空的过期 method(1.03595)盖住了 variants 里更优的 1.014969,把
    2.5% 的真实改进低报成 0.4%(6 倍差)。_build_result_from_file 必须始终对齐 {已记 method, variants
    最优} 取更优(按方向),而不只在 method==None 时。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file
    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({          # agent 只写了首个改进,没更新
        "metric": "val_bpb", "baseline": 1.040402, "method": 1.03595, "lower_is_better": True}),
        encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant": "V2: grad_clip", "result": 1.03595}\n'
        '{"variant": "V4: matrix_lr=0.025", "result": 1.01752}\n'
        '{"variant": "V9: matrix_lr=0.03", "result": 1.014969}\n'   # 最优,但 ar_result 没记
        '{"variant": "V6: gelu", "result": 1.036772}\n', encoding="utf-8")
    res = _build_result_from_file(str(repo / "ar_result.json"), "nanochat",
                                  "val_bpb", True, lambda e, p: None)
    assert res["candidate"].metric_value == 1.014969       # 取 variants 最优,不是过期的 1.03595
    assert res["verdict"]["improved"] is True


def test_recorded_method_kept_when_better_than_variants(tmp_path):
    """对称保护: 若 agent 已把最优值写进 method(variants 里没有更好的),不能被 variants 里的次优覆盖。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file
    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "val_bpb", "baseline": 1.04, "method": 1.010, "lower_is_better": True}),
        encoding="utf-8")
    (repo / "analysis" / "variants.jsonl").write_text(
        '{"variant": "V1", "result": 1.02}\n{"variant": "V2", "result": 1.015}\n', encoding="utf-8")
    res = _build_result_from_file(str(repo / "ar_result.json"), "nanochat",
                                  "val_bpb", True, lambda e, p: None)
    assert res["candidate"].metric_value == 1.010          # 保留已记的更优 method


def test_prompt_budget_discipline_curbs_overexploration(tmp_path, monkeypatch):
    """#67 过度探索: SOL 真机 agent 重跑 baseline 133 次不收尾、method=null。prompt 必须逼它
    限制 baseline 重跑、尽早写 method、时间过大半就收尾。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _task_prompt
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", "6")
    p = _task_prompt("lower latency", "/repo", "/repo/ar_result.json", None, "latency_ms", True)
    assert "BUDGET DISCIPLINE" in p
    assert "FIRST THIRD" in p                                # baseline 只占前 1/3
    assert "more than TWICE" in p                            # baseline 别重跑超 2 次
    assert "STOP" in p                                       # 时间过大半就收尾
    assert "AT MOST 6 train/eval attempts" in p               # 单沙箱最多 6 次，失败也算
    assert "crashes, times out, or produces no metric" in p


def test_attempt_guard_counts_failed_runs_and_blocks_seventh(tmp_path):
    """A crashed/timeout attempt consumes budget even when no variants metric was produced."""
    import subprocess
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    result = repo / "ar_result.json"
    bindir, log_path = _install_attempt_guard(str(tmp_path), str(repo), str(result))
    wrapper = str(Path(bindir) / "timeout")
    # Host creates controls; sandbox runs as uid 10001 and must be able to append the log.
    import stat
    assert stat.S_IMODE(os.stat(Path(bindir).parent).st_mode) == 0o777
    assert stat.S_IMODE(os.stat(log_path).st_mode) == 0o666
    env = dict(os.environ, AUTORESEARCH_ATTEMPT_RESULT=str(result),
               AUTORESEARCH_ATTEMPT_LOG=log_path, AUTORESEARCH_AGENTIC_VARIANT_CAP="6")

    # Baseline happens before ar_result.json exists and is not part of the six method attempts.
    assert subprocess.run([wrapper, "1s", "sh", "-c", "exit 9"], env=env).returncode == 9
    assert _attempt_count(log_path) == 0
    result.write_text('{"baseline": 1.0, "method": null}', encoding="utf-8")

    for _ in range(6):
        assert subprocess.run([wrapper, "1s", "sh", "-c", "exit 7"], env=env).returncode == 7
    assert _attempt_count(log_path) == 6
    assert subprocess.run([wrapper, "1s", "true"], env=env).returncode == 125
    assert _attempt_count(log_path) == 6


def test_sandbox_guard_refuses_destructive_shared_uv_cache_cleanup(tmp_path):
    """The package cache is shared across sessions and must never be agent-deletable."""
    import subprocess
    from autoresearch.experiments.repo_improve.agentic_improve import _install_attempt_guard

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, _ = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )

    completed = subprocess.run(
        [str(Path(bindir) / "uv"), "cache", "clean"],
        capture_output=True, text=True,
    )

    assert completed.returncode == 126
    assert "shared package cache" in completed.stderr


def test_ar_attempt_counts_at_launch_and_blocks_seventh(tmp_path):
    """Catch failed/no-metric commands escaping the cap because accounting happened at exit."""
    import subprocess
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    result = repo / "ar_result.json"
    bindir, log_path = _install_attempt_guard(str(tmp_path), str(repo), str(result))
    launcher = str(Path(bindir) / "ar-attempt")
    env = dict(
        os.environ, AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_EXECUTION_PHASE="method",
    )

    for _ in range(6):
        assert subprocess.run([launcher, "sh", "-c", "exit 7"], env=env).returncode == 7
    assert _attempt_count(log_path) == 6
    assert subprocess.run([launcher, "true"], env=env).returncode == 125
    assert _attempt_count(log_path) == 6


@pytest.mark.parametrize("configured", [None, "-1", "invalid", "20"])
def test_default_attempt_cap_keeps_twenty_receipts_and_blocks_twenty_first(
        tmp_path, monkeypatch, configured):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _attempt_cap_ready_to_pause, _install_attempt_guard,
        _task_prompt, _variant_cap,
    )
    if configured is None:
        monkeypatch.delenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", configured)
    assert _variant_cap() == 20
    repo = tmp_path / "repo"
    repo.mkdir()
    result = repo / "ar_result.json"
    bindir, log_path = _install_attempt_guard(str(tmp_path), str(repo), str(result))
    env = dict(os.environ, AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_EXECUTION_PHASE="method", AUTORESEARCH_REPO_PATH=str(repo))
    for _ in range(20):
        completed = subprocess.run([str(Path(bindir) / "ar-attempt"), "false"],
            env=env, capture_output=True, text=True)
        assert completed.returncode == 1
    assert subprocess.run([str(Path(bindir) / "ar-attempt"), "true"],
        env=env, capture_output=True, text=True).returncode == 125
    assert _attempt_count(log_path) == 20
    assert _attempt_cap_ready_to_pause(log_path, 20, grace_s=0)
    prompt = _task_prompt("improve", str(repo), str(result), None, "seconds", True)
    assert "AT MOST 20" in prompt
    assert "AT MOST 0" not in prompt and "AT MOST 6" not in prompt


@pytest.mark.parametrize("bound", [False, True])
def test_attempt_preserves_reported_runtime_files_before_next_trial(tmp_path, bound):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority, _attempt_records
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import validate_checkpoint

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    (repo / ".gitignore").write_text("weights.bin\nrequired.bin\nar_result.json\n")
    (repo / "required.bin").write_bytes(b"required by frozen contract")

    class Backend(_FakeBackend):
        trial = 0

        def execute_attempt_command(self, handle, cmd, *, cwd, timeout):
            self.trial += 1
            attempt_id = next(x.split("=", 1)[1] for x in shlex.split(cmd)
                              if x.startswith("AUTORESEARCH_ATTEMPT_ID="))
            (repo / "weights.bin").write_bytes(f"trial-{self.trial}".encode())
            (repo / "ar_result.json").write_text(json.dumps({
                "attempt_id": attempt_id if bound else "attempt-stale",
                "method": 0.56, "artifact_paths": ["weights.bin"],
            }))
            return SimpleNamespace(exit_code=0, stdout="method=0.56\n", stderr="")

    log = tmp_path / "attempts.jsonl"
    authority = _AttemptAuthority(Backend(), None, str(repo), "method", str(log),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt"))
    authority.artifact_paths = ["required.bin"]
    payload = json.dumps({"argv": ["true"], "cwd": str(repo)}).encode() + b"\n"
    try:
        first = authority._handle(SimpleNamespace(recv=lambda _: payload))
        record = _attempt_records(str(log))[first["attempt_id"]]
        checkpoint = Path(record["checkpoint"])
        manifest = validate_checkpoint(checkpoint)
        assert ("weights.bin" in manifest["runtime_paths"]) is bound
        assert (checkpoint / "required.bin").read_bytes() == b"required by frozen contract"
        second = authority._handle(SimpleNamespace(recv=lambda _: payload))
        assert first["rc"] == second["rc"] == 0
        assert (repo / "weights.bin").read_bytes() == b"trial-2"
        if bound:
            assert (checkpoint / "weights.bin").read_bytes() == b"trial-1"
            validate_checkpoint(checkpoint)
        assert len(_attempt_records(str(log))) == 2
    finally:
        authority._socket.close()


@pytest.mark.parametrize("changed", [False, True])
def test_host_attempt_binds_auxiliary_measurement_source(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority, _attempt_records

    repo = tmp_path / "repo"
    repo.mkdir()
    log = tmp_path / "attempts.jsonl"
    authority = _AttemptAuthority(_FakeBackend(), None, str(repo), "method", str(log),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt"))
    source = {"path": "/tmp/check.py", "sha256": "a" * 64, "source": "candidate.check()"}
    snapshots = iter([[source], [{**source, "sha256": "b" * 64}] if changed else [source]])
    monkeypatch.setattr(authority, "_capture_command_sources", lambda *_: next(snapshots))
    payload = json.dumps({"argv": ["python3", "/tmp/check.py"], "cwd": str(repo)}).encode() + b"\n"
    try:
        reply = authority._handle(SimpleNamespace(recv=lambda _: payload))
        record = next(iter(_attempt_records(str(log)).values()))
        assert reply["rc"] == (126 if changed else 0)
        assert record["command_sources"] == [source]
        assert record["command_sources_stable"] is (not changed)
        assert record["finished"] is True
        if changed:
            assert source["path"] in reply["stderr"]
            assert "staging before ar-attempt" in reply["stderr"]
    finally:
        authority._socket.close()


def test_host_authority_twenty_attempt_cap_preserves_all_launches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _AttemptAuthority, _attempt_records,
    )
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", "20")
    repo = tmp_path / "repo"
    repo.mkdir()
    log = tmp_path / "attempts.jsonl"
    authority = _AttemptAuthority(_FakeBackend(), None, str(repo), "method", str(log),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt"))
    payload = json.dumps({"argv": ["true"], "cwd": str(repo)}).encode() + b"\n"
    try:
        replies = [authority._handle(SimpleNamespace(recv=lambda _: payload)) for _ in range(20)]
        assert all(reply['attempt_id'] and reply['rc'] == 0 for reply in replies)
        blocked = authority._handle(SimpleNamespace(recv=lambda _: payload))
        assert blocked['rc'] == 125 and blocked['attempt_id'] is None
        records = _attempt_records(str(log))
        assert len(records) == 20 and all(record['finished'] for record in records.values())
    finally:
        authority._socket.close()


@pytest.mark.parametrize("authoritative", [False, True])
def test_zero_attempt_cap_keeps_receipts_beyond_twenty(tmp_path, monkeypatch, authoritative):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _AttemptAuthority, _attempt_count, _attempt_cap_ready_to_pause,
        _install_attempt_guard, _task_prompt, _variant_cap,
    )
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", "0")
    assert _variant_cap() == 0
    repo = tmp_path / "repo"
    repo.mkdir()
    result = repo / "ar_result.json"
    if authoritative:
        log_path = str(tmp_path / "attempts.jsonl")
        authority = _AttemptAuthority(_FakeBackend(), None, str(repo), "method", log_path,
            str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt"))
        payload = json.dumps({"argv": ["true"], "cwd": str(repo)}).encode() + b"\n"
        try:
            for _ in range(22):
                reply = authority._handle(SimpleNamespace(recv=lambda _: payload))
                assert reply["rc"] == 0 and reply["attempt_id"]
        finally:
            authority._socket.close()
    else:
        bindir, log_path = _install_attempt_guard(str(tmp_path), str(repo), str(result))
        env = dict(os.environ, AUTORESEARCH_ATTEMPT_LOG=log_path,
            AUTORESEARCH_EXECUTION_PHASE="method", AUTORESEARCH_REPO_PATH=str(repo))
        for _ in range(22):
            assert subprocess.run([str(Path(bindir) / "ar-attempt"), "false"],
                env=env, capture_output=True).returncode == 1
    assert _attempt_count(log_path) == 22
    assert not _attempt_cap_ready_to_pause(log_path, 0, grace_s=0)
    prompt = _task_prompt("improve", str(repo), str(result), None, "seconds", True)
    assert "There is no fixed sandbox attempt cap" in prompt
    assert "AT MOST 0" not in prompt and "AT MOST 20" not in prompt


@pytest.mark.parametrize("outside", ["workspace", "repo-old", "escaping-link"])
def test_attempt_wrong_cwd_names_authorized_checkout_without_executing(tmp_path, outside):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    repo = tmp_path / "repo"
    repo.mkdir()
    other = tmp_path / "workspace"
    other.mkdir()
    cwd = tmp_path / outside
    if outside == "escaping-link":
        cwd = repo / "escaping-link"
        cwd.symlink_to(other, target_is_directory=True)
    authority = object.__new__(_AttemptAuthority)
    authority.uncertain_attempt = None  # Constructor state for this focused fixture.
    authority.repo_path = str(repo)
    authority.control_path = str(tmp_path / "control")
    payload = json.dumps({"argv": ["python", "evaluate.py"], "cwd": str(cwd)}).encode() + b"\n"
    # No backend or attempt log: a rejected request must not reach execution or accounting.
    result = authority._handle(SimpleNamespace(recv=lambda _: payload))
    assert result["rc"] == 126 and result["attempt_id"] is None
    assert str(cwd.resolve()) in result["stderr"]
    assert str(repo) in result["stderr"] and authority.control_path in result["stderr"]
    assert "cwd" in result["stderr"] and "ar-attempt" in result["stderr"]


def test_attempt_authority_recovers_expired_uncertain_execution(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "60")
    repo = tmp_path / "repo"
    repo.mkdir()
    attempt_id = "attempt-expired"
    log_path = tmp_path / "attempts.jsonl"
    log_path.write_text(
        json.dumps({"event": "launch", "attempt_id": attempt_id,
                    "started_at": time.time() - 400}) + "\n" +
        json.dumps({"event": "finish", "attempt_id": attempt_id,
                    "execution_uncertain": True, "rc": 126}) + "\n")
    marker = tmp_path / ".autoresearch-uncertain-execution.json"
    marker.write_text(json.dumps({"attempt_id": attempt_id, "log_path": str(log_path)}))
    authority = _AttemptAuthority(
        None, None, str(repo), "method", str(tmp_path / "current.jsonl"),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt_id"))
    try:
        assert authority.uncertain_attempt is None
        assert not marker.exists()
        resolved = list(tmp_path.glob(".autoresearch-uncertain-execution.resolved-*.json"))
        assert len(resolved) == 1
        assert json.loads(resolved[0].read_text())["attempt_id"] == attempt_id
    finally:
        authority._socket.close()


@pytest.mark.parametrize(
    "command_timeout, age, uncertain_finish, no_time_limit, recovered",
    [
        (8100, 19000, True, True, True),
        (8100, 200, True, True, False),
        (8100, 400, True, False, False),
        (8100, 19000, False, True, False),
        (None, 19000, True, True, False),
        (0, 19000, True, True, False),
        (float("inf"), 19000, True, True, False),
    ],
)
def test_attempt_authority_reconciles_recorded_command_deadline(
        tmp_path, monkeypatch, command_timeout, age, uncertain_finish,
        no_time_limit, recovered):
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "60")
    monkeypatch.setenv("AUTORESEARCH_UNCERTAIN_RECOVERY_GRACE_S", "300")
    if no_time_limit:
        monkeypatch.setenv("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT", "1")
    else:
        monkeypatch.delenv("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    attempt_id = "attempt-command-deadline"
    launch = {"event": "launch", "attempt_id": attempt_id,
              "started_at": time.time() - age}
    if command_timeout is not None:
        launch["command_timeout_s"] = command_timeout
    events = [launch]
    if uncertain_finish:
        events.append({"event": "finish", "attempt_id": attempt_id,
                       "execution_uncertain": True, "rc": 126})
    log_path = tmp_path / "original.jsonl"
    log_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    marker = tmp_path / ".autoresearch-uncertain-execution.json"
    marker.write_text(json.dumps({"attempt_id": attempt_id, "log_path": str(log_path)}))
    authority = _AttemptAuthority(
        None, None, str(repo), "method", str(tmp_path / "current.jsonl"),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt_id"))
    try:
        assert (authority.uncertain_attempt is None) is recovered
        assert marker.exists() is not recovered
        if recovered:
            archived = list(tmp_path.glob(".autoresearch-uncertain-execution.resolved-*.json"))
            assert len(archived) == 1
            assert json.loads(archived[0].read_text())["attempt_id"] == attempt_id
    finally:
        authority._socket.close()


def test_attempt_authority_keeps_recent_uncertain_execution_blocked(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "60")
    repo = tmp_path / "repo"
    repo.mkdir()
    attempt_id = "attempt-recent"
    log_path = tmp_path / "attempts.jsonl"
    log_path.write_text(
        json.dumps({"event": "launch", "attempt_id": attempt_id,
                    "started_at": time.time() - 200}) + "\n" +
        json.dumps({"event": "finish", "attempt_id": attempt_id,
                    "execution_uncertain": True, "rc": 126}) + "\n")
    marker = tmp_path / ".autoresearch-uncertain-execution.json"
    marker.write_text(json.dumps({"attempt_id": attempt_id, "log_path": str(log_path)}))
    authority = _AttemptAuthority(
        None, None, str(repo), "method", str(tmp_path / "current.jsonl"),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt_id"))
    try:
        assert authority.uncertain_attempt == attempt_id
        assert marker.exists()
    finally:
        authority._socket.close()


def test_attempt_authority_releases_uncertain_execution_after_sandbox_close(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "3600")
    repo = tmp_path / "repo"
    repo.mkdir()
    attempt_id = "attempt-closed-sandbox"
    log_path = tmp_path / "attempts.jsonl"
    log_path.write_text(
        json.dumps({"event": "launch", "attempt_id": attempt_id,
                    "started_at": time.time() - 10}) + "\n" +
        json.dumps({"event": "finish", "attempt_id": attempt_id,
                    "execution_uncertain": True, "rc": 126}) + "\n")
    marker = tmp_path / ".autoresearch-uncertain-execution.json"
    marker.write_text(json.dumps({"attempt_id": attempt_id, "log_path": str(log_path)}))
    authority = _AttemptAuthority(
        None, None, str(repo), "method", str(log_path),
        str(tmp_path / "authority.sock"), str(tmp_path / "last_attempt_id"))
    try:
        authority.resolve_uncertain_after_sandbox_close(False)
        assert authority.uncertain_attempt == attempt_id
        assert marker.exists()

        authority.resolve_uncertain_after_sandbox_close(True)
        assert authority.uncertain_attempt is None
        assert not marker.exists()
        resolved = list(tmp_path.glob(".autoresearch-uncertain-execution.resolved-*.json"))
        assert len(resolved) == 1
        assert "definitively closed" in json.loads(resolved[0].read_text())["resolution"]
    finally:
        authority._socket.close()


def test_attempt_guidance_forbids_bypassing_host_authority():
    from autoresearch.experiments.repo_improve.agentic_improve import _ATTEMPT_INVOCATION_GUIDANCE

    assert "Never unset, replace or override AUTORESEARCH_ATTEMPT_SOCKET" in _ATTEMPT_INVOCATION_GUIDANCE
    assert "locally written attempt log is not host-registered evidence" in _ATTEMPT_INVOCATION_GUIDANCE


def test_sandbox_close_waits_for_delayed_uncertain_finish(tmp_path, monkeypatch):
    import threading
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority

    repo = tmp_path / "repo"
    repo.mkdir()
    log_path = tmp_path / "attempts.jsonl"
    authority = _AttemptAuthority(None, None, str(repo), "method", str(log_path),
                                  str(tmp_path / "authority.sock"), str(tmp_path / "last"))
    attempt_id = "attempt-reply-lost-during-close"
    marker = tmp_path / ".autoresearch-uncertain-execution.json"
    release = threading.Event()

    def finish_after_close():
        release.wait()
        log_path.write_text(json.dumps({"event": "launch", "attempt_id": attempt_id}) + "\n"
                            + json.dumps({"event": "finish", "attempt_id": attempt_id,
                                          "execution_uncertain": True, "rc": 126}) + "\n")
        marker.write_text(json.dumps({"attempt_id": attempt_id, "log_path": str(log_path)}))
        authority.uncertain_attempt = attempt_id

    authority._thread = threading.Thread(target=finish_after_close)
    join = authority._thread.join

    def release_and_join(*args, **kwargs):
        release.set()
        return join(*args, **kwargs)

    monkeypatch.setattr(authority._thread, "join", release_and_join)
    authority._thread.start()
    try:
        authority.resolve_uncertain_after_sandbox_close(True)
        assert not authority._thread.is_alive()
        assert authority.uncertain_attempt is None
        assert not marker.exists()
        archived = list(tmp_path.glob(".autoresearch-uncertain-execution.resolved-*.json"))
        assert len(archived) == 1
        assert json.loads(archived[0].read_text())["attempt_id"] == attempt_id
    finally:
        release.set()
        join()
        authority._socket.close()


@pytest.mark.parametrize("brokered", [False, True])
def test_attempt_outputs_do_not_overwrite_prior_checkpoint(tmp_path, brokered):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _AttemptAuthority, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline = artifacts / "baseline.txt"
    baseline.write_text("frozen baseline")
    bindir, log_path = _install_attempt_guard(str(tmp_path), str(repo), str(repo / "ar_result.json"))
    env = dict(os.environ, AUTORESEARCH_ARTIFACT_DIR=str(artifacts),
               AUTORESEARCH_ATTEMPT_LOG=log_path, AUTORESEARCH_EXECUTION_PHASE="method")
    child = (
        "import os,json; from pathlib import Path; "
        "p=Path(os.environ['AUTORESEARCH_ATTEMPT_ARTIFACT_DIR'])/'best_model.bin'; "
        "p.write_text(os.environ['AUTORESEARCH_ATTEMPT_ID']); "
        "print(json.dumps({'path':str(p),'id':os.environ['AUTORESEARCH_ATTEMPT_ID']}))"
    )
    results = []
    for _ in range(2):
        argv = ["python3", "-c", child]
        if brokered:
            def execute(_handle, cmd, *, cwd, timeout):
                run = subprocess.run(shlex.split(cmd), cwd=cwd, env=env,
                                     capture_output=True, text=True, timeout=timeout)
                return SimpleNamespace(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)

            authority = object.__new__(_AttemptAuthority)
            authority.uncertain_attempt = None  # Constructor state for this focused fixture.
            authority.repo_path, authority.phase = str(repo), "method"
            authority.log_path = log_path
            authority.last_attempt_path = str(Path(bindir).parent / "last_attempt_id")
            authority.backend = SimpleNamespace(execute_command=execute)
            authority.handle = None
            authority.control_path = None
            authority.artifact_paths = []
            authority.execution_protocol = {}
            authority.repository_contract = None
            authority.dependency_parent_state = {}
            request = json.dumps({"argv": argv, "cwd": str(repo)}).encode() + b"\n"
            result = authority._handle(SimpleNamespace(recv=lambda _size: request))
            assert result["rc"] == 0, result
            output = result["stdout"]
        else:
            result = subprocess.run([str(Path(bindir) / "ar-attempt"), *argv],
                                    env=env, cwd=repo, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            output = result.stdout
        results.append(json.loads(output))

    assert results[0]["path"] != results[1]["path"]
    for result in results:
        path = Path(result["path"])
        assert path.parent == artifacts / "attempts" / result["id"]
        assert path.read_text() == result["id"]
    assert baseline.read_text() == "frozen baseline"


def test_ar_attempt_help_does_not_consume_method_budget(tmp_path):
    """Inspecting launcher usage is not a train/eval attempt and must not spend the cap."""
    import subprocess
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    launcher = str(Path(bindir) / "ar-attempt")
    env = dict(
        os.environ, AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_LAST_ATTEMPT=str(Path(log_path).with_name("last_attempt_id")),
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_EXECUTION_PHASE="method",
        AUTORESEARCH_REPO_PATH=str(repo),
    )

    for flag in ("--help", "-h"):
        completed = subprocess.run(
            [launcher, flag], env=env, capture_output=True, text=True,
        )
        assert completed.returncode == 0
        assert "usage: ar-attempt" in completed.stdout
        assert "ar-attempt env NAME=VALUE" in completed.stdout
        assert "do not inherit terminal exports" in completed.stdout
    assert _attempt_count(log_path) == 0
    assert not Path(env["AUTORESEARCH_LAST_ATTEMPT"]).read_text().strip()


def test_method_timeout_setup_does_not_spend_candidate_budget(tmp_path):
    """A bounded dependency probe remains possible after candidate attempts are exhausted."""
    import sys
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    env = dict(
        os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
        AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_EXECUTION_PHASE="method",
        AUTORESEARCH_REPO_PATH=str(repo),
    )
    probe = [str(Path(bindir) / "timeout"), "1s", sys.executable,
             "-c", "import urllib.request; print('dependency import succeeded')"]
    for _ in range(7):
        assert subprocess.run(probe, env=env, capture_output=True).returncode == 0
    assert _attempt_count(log_path) == 0
    launcher = str(Path(bindir) / "ar-attempt")
    for _ in range(6):
        assert subprocess.run([launcher, "false"], env=env, capture_output=True).returncode == 1
    assert subprocess.run([launcher, "true"], env=env, capture_output=True).returncode == 125
    assert subprocess.run(probe, env=env, capture_output=True).returncode == 0
    assert _attempt_count(log_path) == 6


def test_method_explicit_timeout_counts_failures_and_blocks_seventh(tmp_path):
    """Bounded explicit experiments count failures/124 and refuse a seventh launch."""
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    wrapper = str(Path(bindir) / "timeout")
    env = dict(
        os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
        AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_LAST_ATTEMPT=str(Path(log_path).with_name("last_attempt_id")),
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_EXECUTION_PHASE="method",
        AUTORESEARCH_REPO_PATH=str(repo),
    )
    launcher = str(Path(bindir) / "ar-attempt")
    assert subprocess.run(
        [launcher, wrapper, "0.01s", "sh", "-c", "sleep 0.1"], env=env,
        capture_output=True, text=True,
    ).returncode == 124
    for _ in range(5):
        assert subprocess.run(
            [launcher, wrapper, "1s", "sh", "-c", "exit 7"], env=env,
            capture_output=True, text=True,
        ).returncode == 7
    assert _attempt_count(log_path) == 6
    assert subprocess.run(
        [launcher, wrapper, "1s", "true"], env=env, capture_output=True, text=True,
    ).returncode == 125
    assert _attempt_count(log_path) == 6


def test_explicit_ar_attempt_timeout_is_counted_once(tmp_path):
    """The timeout shim nested below ar-attempt must not create a second ledger record."""
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _attempt_records, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    launcher = str(Path(bindir) / "ar-attempt")
    env = dict(
        os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
        AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_LAST_ATTEMPT=str(Path(log_path).with_name("last_attempt_id")),
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_EXECUTION_PHASE="method",
        AUTORESEARCH_REPO_PATH=str(repo),
    )

    assert subprocess.run(
        [launcher, "timeout", "1s", "true"], env=env,
        capture_output=True, text=True,
    ).returncode == 0
    records = _attempt_records(log_path)
    assert _attempt_count(log_path) == 1
    assert len(records) == 1
    assert next(iter(records.values()))["argv"] == ["timeout", "1s", "true"]


def test_timeout_guard_keeps_pty_child_in_foreground(tmp_path):
    """Keep PTY foreground behavior while owning a process group for descendant cleanup."""
    from autoresearch.experiments.repo_improve.agentic_improve import _install_attempt_guard

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, _log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )

    script = (Path(bindir) / "timeout-cleanup").read_text()
    assert '["/usr/bin/timeout", "--foreground", *sys.argv[1:]]' in script
    assert "start_new_session=True" in script


def test_timeout_guard_terminates_children_after_parent_failure(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _install_attempt_guard

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, _log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    child_pid = tmp_path / "child.pid"
    command = f"sleep 60 & echo $! > {shlex.quote(str(child_pid))}; exit 7"

    completed = subprocess.run(
        [str(Path(bindir) / "timeout"), "10s", "sh", "-c", command],
        cwd=repo, check=False, timeout=5,
    )

    assert completed.returncode == 7
    pid = int(child_pid.read_text(encoding="utf-8"))
    for _ in range(20):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(0.05)
    assert not Path(f"/proc/{pid}").exists()


def test_timeout_guard_terminates_children_when_helper_is_terminated(tmp_path):
    import signal
    from autoresearch.experiments.repo_improve.agentic_improve import _install_attempt_guard

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, _ = _install_attempt_guard(str(tmp_path), str(repo), str(repo / "ar_result.json"))
    child_pid = tmp_path / "child.pid"
    helper = subprocess.Popen([
        str(Path(bindir) / "timeout-cleanup"), "60s", "sh", "-c",
        f"echo $$ > {shlex.quote(str(child_pid))}; exec sleep 60",
    ])
    pid = None
    try:
        for _ in range(100):
            if child_pid.exists() and child_pid.read_text().strip():
                break
            time.sleep(0.02)
        pid = int(child_pid.read_text())
        helper.terminate()
        helper.wait(timeout=5)
        for _ in range(50):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.02)
        assert not Path(f"/proc/{pid}").exists(), "terminated helper left its command running"
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=5)
        if pid is not None and Path(f"/proc/{pid}").exists():
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_method_prompt_requires_foreground_timeout(monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "900")
    prompt = _method_task_prompt(
        "improve repository", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "loss",
            "metric_value": 1.0,
            "lower_is_better": True,
            "fixed_config": "canonical",
        },
    )

    assert "`ar-attempt timeout --foreground 900s <command>`" in prompt
    assert "setup, compilation, training, and final evaluation" in prompt
    assert "optional experiment tracking or telemetry" in prompt
    assert "never wait for interactive stdin" in prompt


def test_method_prompt_does_not_charge_inspection_or_editing_as_candidate_attempts():
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    prompt = _method_task_prompt(
        "improve repository", "/repo", "/repo/ar_result.json", None,
        {
            "metric_name": "loss",
            "metric_value": 1.0,
            "lower_is_better": True,
            "fixed_config": "canonical",
        },
    )

    assert "ordinary shell commands for repository inspection" in prompt
    assert "diagnostic profiling, ablation, sweep" in prompt
    assert "outside `ar-attempt`" in prompt
    assert "do NOT prefix those non-experiment commands with `ar-attempt`" in prompt
    assert "Only a command that actually launches candidate training" in prompt


def test_attempt_identity_does_not_strip_clean_environment_wrapper():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _normalized_attempt_argv,
    )

    raw = ["env", "--ignore-environment", "python", "train.py"]

    assert _normalized_attempt_argv(raw) == raw


def test_baseline_timeout_attempts_do_not_consume_method_cap(tmp_path):
    """Baseline work and method variants share a ledger but have separate budget semantics."""
    from autoresearch.experiments.repo_improve.agentic_improve import _install_attempt_guard

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    wrapper = str(Path(bindir) / "timeout")
    launcher = str(Path(bindir) / "ar-attempt")
    base_env = dict(
        os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
        AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_LAST_ATTEMPT=str(Path(log_path).with_name("last_attempt_id")),
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_REPO_PATH=str(repo),
    )
    baseline_env = dict(base_env, AUTORESEARCH_EXECUTION_PHASE="baseline")
    method_env = dict(base_env, AUTORESEARCH_EXECUTION_PHASE="method")

    for _ in range(6):
        assert subprocess.run(
            [wrapper, "1s", "true"], env=baseline_env,
            capture_output=True, text=True,
        ).returncode == 0
    # Explicit baseline ar-attempt remains attestable, but must not reduce method capacity.
    assert subprocess.run(
        [launcher, "true"], env=baseline_env, capture_output=True, text=True,
    ).returncode == 0
    for _ in range(6):
        assert subprocess.run(
            [launcher, wrapper, "1s", "true"], env=method_env,
            capture_output=True, text=True,
        ).returncode == 0
    assert subprocess.run(
        [launcher, wrapper, "1s", "true"], env=method_env,
        capture_output=True, text=True,
    ).returncode == 125


def test_ar_attempt_cap_check_and_launch_append_are_atomic(tmp_path):
    """Concurrent candidate launches cannot all observe the same stale ledger count."""
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _attempt_count, _install_attempt_guard,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    launcher = str(Path(bindir) / "ar-attempt")
    env = dict(
        os.environ, AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_LAST_ATTEMPT=str(Path(log_path).with_name("last_attempt_id")),
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6", AUTORESEARCH_EXECUTION_PHASE="method",
        AUTORESEARCH_REPO_PATH=str(repo),
    )

    children = [subprocess.Popen(
        [launcher, "sh", "-c", "sleep 0.05"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) for _ in range(12)]
    returncodes = [child.wait() for child in children]

    assert returncodes.count(0) == 6
    assert returncodes.count(125) == 6
    assert _attempt_count(log_path) == 6


def test_structured_method_without_attempt_ledger_is_invalid(tmp_path):
    """Catch a direct command bypass being accepted with an unbounded candidate result."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, record_attempt=False)

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
        },
    )

    assert res["baseline"].valid is True
    assert res["candidate"].valid is False
    assert res["diagnosis"] == "execution_failure"


def test_attempt_cap_gives_final_result_write_a_grace_period(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_cap_ready_to_pause

    log = tmp_path / "attempts.tsv"
    log.write_text("7\tfailed\n" * 6, encoding="utf-8")
    assert not _attempt_cap_ready_to_pause(str(log), 6, grace_s=60)
    old = time.time() - 61
    os.utime(log, (old, old))
    assert _attempt_cap_ready_to_pause(str(log), 6, grace_s=60)


def test_attempt_cap_does_not_pause_until_a_parseable_result_exists(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_cap_ready_to_pause

    log = tmp_path / "attempts.tsv"
    result = tmp_path / "ar_result.json"
    log.write_text("7\tfailed\n" * 6, encoding="utf-8")
    old = time.time() - 600
    os.utime(log, (old, old))

    assert not _attempt_cap_ready_to_pause(
        str(log), 6, result_path=str(result), grace_s=0)
    result.write_text("{not-json", encoding="utf-8")
    assert not _attempt_cap_ready_to_pause(
        str(log), 6, result_path=str(result), grace_s=0)
    result.write_text(json.dumps({"metric": "loss", "method": 1.25}), encoding="utf-8")
    assert _attempt_cap_ready_to_pause(
        str(log), 6, result_path=str(result), grace_s=0)

    result.write_text(json.dumps({"metric": "loss", "method_metric": 1.25}),
                      encoding="utf-8")
    assert _attempt_cap_ready_to_pause(
        str(log), 6, result_path=str(result), grace_s=0)


@pytest.mark.parametrize("metric", ["loss", {"name": "loss", "unit": "nats", "lower_is_better": True}])
@pytest.mark.parametrize("value", [1.25, None])
def test_attempt_cap_accepts_current_metric_envelope_only_after_child_finishes(tmp_path, metric, value):
    from autoresearch.experiments.repo_improve.agentic_improve import _attempt_cap_ready_to_pause

    log = tmp_path / "attempts.jsonl"
    result = tmp_path / "ar_result.json"
    events = [{"event": "launch", "attempt_id": "attempt-current"}]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result.write_text(json.dumps({"metric": metric, "method": value}))
    # A previous/partial result must not interrupt the last running command.
    assert not _attempt_cap_ready_to_pause(str(log), 1, result_path=str(result))
    events.append({"event": "finish", "attempt_id": "attempt-current", "rc": 0 if value else 1})
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    assert _attempt_cap_ready_to_pause(str(log), 1, result_path=str(result))
    assert not _attempt_cap_ready_to_pause(str(log), 2, result_path=str(result))
    result.write_text(json.dumps({"metric": {}, "method": value}))
    assert not _attempt_cap_ready_to_pause(str(log), 1, result_path=str(result))


def test_combined_prompt_does_not_inject_retired_fairness_protocol(tmp_path):
    """Fairness semantics belong to problem reinterpretation, not the execution prompt."""
    from autoresearch.experiments.repo_improve.agentic_improve import _task_prompt
    p = _task_prompt("lower val_bpb", "/repo", "/repo/ar_result.json", None, "val_bpb", True)
    low = p.lower()
    assert "controlled comparison" not in low
    assert "exact same configuration" not in low
    assert "only thing allowed to differ" not in low
    assert "byte-for-byte identical" not in low
    assert "baseline" in low and "method" in low


def test_carry_direction_persisted_to_workroot(tmp_path, monkeypatch):
    """#75 跨轮引导: 当分析产出 direction,除了写 per-repo analysis/direction.txt,还要落到 workroot
    级 carry_direction.txt(外层 loop 每轮换 repo 子目录,只有 workroot 级下一轮才读得到)。"""
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    repo = tmp_path / "repo"                       # workroot=tmp_path, repo 是其直接子目录
    (repo / "analysis").mkdir(parents=True)
    (repo / ai._RESULT_BASENAME).write_text(
        json.dumps({"metric": "loss", "baseline": 2.0,
                    "changes_summary": "replaced MLP with a 1-layer transformer decoder"}),
        encoding="utf-8")
    monkeypatch.setattr(ai, "_call_analysis_llm",
                        lambda user, timeout=90: "Next: widen n_embd to 64 and add cosine LR.")
    ai._analyze_once(str(repo), "lower the loss", "loss", True, lambda e, p: None)

    carry = tmp_path / ai._CARRY_BASENAME
    assert carry.exists()                          # workroot 级 carry 落盘
    assert "widen n_embd" in carry.read_text(encoding="utf-8")
    assert Path(ai._analysis_direction_path(str(repo))).exists()


def test_carry_direction_seeded_and_injected_next_round(tmp_path):
    """#75: 下一轮起跑时,workroot 级 carry 要 (a) 预置到本轮 repo/analysis/direction.txt(agent 被
    要求读该文件) + (b) 内联进 task prompt(flash 读文件不可靠)。"""
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    (tmp_path / ai._CARRY_BASENAME).write_text(
        "Conclusion: fp8 diverged; next widen the tile and fuse rmsnorm.", encoding="utf-8")
    be = _FakeBackend()
    run_agentic_improve("lower step time", repo_url="https://x/nanoGPT",
                        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)

    assert "PREVIOUS ROUND'S CONCLUSION" in be.task          # (b) 内联进 prompt
    assert "fuse rmsnorm" in be.task                         # 上一轮结论文本进了 prompt
    seeded = Path(ai._analysis_direction_path(str(tmp_path / "repo")))
    assert seeded.exists() and "fuse rmsnorm" in seeded.read_text(encoding="utf-8")  # (a) 预置文件


def test_carry_direction_isolated_between_methods(tmp_path):
    """A revised method must not inherit the prior method's directed next step."""
    from autoresearch.experiments.repo_improve import agentic_improve as ai

    old_scope = "spec://method/session/sha256-old"
    new_scope = "spec://method/session/sha256-new"
    (tmp_path / ai._carry_basename(old_scope)).write_text(
        "Old method: add gradient accumulation.", encoding="utf-8")

    old_backend = _FakeBackend()
    run_agentic_improve(
        "improve", repo_url="https://x/project", workroot=str(tmp_path),
        backend=old_backend, clone_fn=_fake_clone, memory_scope=old_scope,
    )
    assert Path(ai._analysis_direction_path(str(tmp_path / "repo"))).read_text(
        encoding="utf-8") == "Old method: add gradient accumulation."

    new_backend = _FakeBackend()
    run_agentic_improve(
        "improve", repo_url="https://x/project", workroot=str(tmp_path),
        backend=new_backend, clone_fn=_fake_clone, memory_scope=new_scope,
    )
    assert not Path(ai._analysis_direction_path(str(tmp_path / "repo-2"))).exists()


def test_split_prompts_reject_shared_result_metadata_as_current_evidence():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _baseline_task_prompt, _method_task_prompt)

    baseline = _baseline_task_prompt("goal", "/repo", "/repo/ar_result.json", "loss", True)
    method = _method_task_prompt(
        "goal", "/repo", "/repo/ar_result.json", None,
        {"metric_name": "loss", "metric_value": 1.0, "lower_is_better": True},
    )

    for prompt in (baseline, method):
        assert "do not treat result, analysis, checkpoint, or direction metadata" in prompt.lower()
        assert "current method" in prompt.lower()


def test_carry_direction_absent_first_round_is_noop(tmp_path):
    """首轮无 carry -> prompt 不含跨轮块,行为与从前一致(向后兼容)。"""
    be = _FakeBackend()
    run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone)
    assert "PREVIOUS ROUND'S CONCLUSION" not in be.task


def test_carry_direction_disabled_by_env(tmp_path, monkeypatch):
    """AUTORESEARCH_CARRY_DIRECTION=0 关掉跨轮引导:即使 carry 存在也不注入。"""
    from autoresearch.experiments.repo_improve import agentic_improve as ai
    monkeypatch.setenv("AUTORESEARCH_CARRY_DIRECTION", "0")
    (tmp_path / ai._CARRY_BASENAME).write_text("some prior direction", encoding="utf-8")
    be = _FakeBackend()
    run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone)
    assert "PREVIOUS ROUND'S CONCLUSION" not in be.task


def test_tool_repos_cloned_and_surfaced_in_prompt(tmp_path):
    """#76 多仓库联动: 额外工具仓库(显式参数)clone 进 workroot/tools/<name>,并在 prompt 里作为
    instrument 告知 agent(用来评测/增强主仓库,不是要改的目标)。"""
    be = _FakeBackend()
    res = run_agentic_improve(
        "compute % of H800 speed-of-light using SOLAR", repo_url="https://x/SOL-ExecBench",
        workroot=str(tmp_path), backend=be, clone_fn=_fake_clone,
        tool_repo_urls=("https://github.com/NVlabs/SOLAR",))
    tool_dir = tmp_path / "tools" / "SOLAR"
    assert tool_dir.exists() and (tool_dir / "README.md").exists()   # clone 进 tools/
    assert "TOOL REPOSITORIES" in be.task                            # prompt 有工具块
    assert "SOLAR" in be.task and str(tool_dir) in be.task           # 名字+路径都告诉 agent
    assert res["baseline"].valid                                     # 主改进流程照常


def test_tool_repos_from_env(tmp_path, monkeypatch):
    """无显式参数时,从 AUTORESEARCH_TOOL_REPOS(逗号分隔)读取,驱动本地路径无需改 handler。"""
    monkeypatch.setenv("AUTORESEARCH_TOOL_REPOS", "https://x/toolA, https://x/toolB")
    be = _FakeBackend()
    run_agentic_improve("q", repo_url="https://x/main", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone)
    assert (tmp_path / "tools" / "toolA").exists()
    assert (tmp_path / "tools" / "toolB").exists()
    assert "toolA" in be.task and "toolB" in be.task


def test_tool_repo_clone_failure_is_non_fatal(tmp_path):
    """工具仓库 clone 失败不该毁掉主改进:主仓库照常 clone+跑,只是 prompt 里没有那个工具。"""
    calls = {"n": 0}

    def _clone_main_ok_tool_boom(url, dest):
        # 主仓库(workroot/repo)成功;工具仓库(workroot/tools/*)失败
        if os.path.join("tools", "") in dest or "/tools/" in dest:
            calls["n"] += 1
            raise RuntimeError("tool clone disconnect")
        _fake_clone(url, dest)

    be = _FakeBackend()
    res = run_agentic_improve("q", repo_url="https://x/main", workroot=str(tmp_path),
                              backend=be, clone_fn=_clone_main_ok_tool_boom,
                              tool_repo_urls=("https://x/brokenTool",))
    assert calls["n"] == 1                          # 确实尝试了工具 clone
    assert res["baseline"].valid                    # 主改进没被工具失败拖垮
    assert "TOOL REPOSITORIES" not in be.task       # 失败的工具不进 prompt


def test_extract_repo_urls_splits_main_and_tools():
    """#76 一等公民: 请求里多个 github URL 按序抽出,第一个=主仓库,其余=工具仓库。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _extract_repo_urls
    q = ("Improve https://github.com/NVIDIA/SOL-ExecBench using the "
         "https://github.com/NVlabs/SOLAR tool to compute speed-of-light.")
    urls = _extract_repo_urls(q)
    assert urls[0] == "https://github.com/NVIDIA/SOL-ExecBench"   # 主仓库
    assert urls[1] == "https://github.com/NVlabs/SOLAR"           # 工具仓库
    # 去重 + 剥尾部标点
    assert _extract_repo_urls("see https://github.com/a/b, and https://github.com/a/b again") == \
        ["https://github.com/a/b"]


def test_tool_repo_urls_union_explicit_and_env(monkeypatch):
    """#76: tool_repo_urls = 显式(从请求抽的)∪ 环境变量,去重保序。"""
    from autoresearch.experiments.repo_improve.agentic_improve import _tool_repo_urls
    monkeypatch.setenv("AUTORESEARCH_TOOL_REPOS", "https://x/envtool, https://x/shared")
    got = _tool_repo_urls(("https://x/fromreq", "https://x/shared"))
    assert got == ("https://x/fromreq", "https://x/shared", "https://x/envtool")  # 去重(shared只一次)


def test_no_tool_repos_is_noop(tmp_path):
    """无工具仓库(默认)-> prompt 不含工具块,单仓库行为完全不变。"""
    be = _FakeBackend()
    run_agentic_improve("q", repo_url="https://x/r", workroot=str(tmp_path),
                        backend=be, clone_fn=_fake_clone)
    assert "TOOL REPOSITORIES" not in be.task
    assert not (tmp_path / "tools").exists()


def test_salvage_baseline_when_run_raises_after_write(tmp_path):
    """BUG (win-loss, 2026-07-31): a TRANSIENT run_agent_task error at the end of the agent
    session must NOT discard the baseline+method the agent already wrote to ar_result.json.
    Repro: glm 6-concurrent had 5/6 rounds return base=None despite real baselines on disk."""
    be = _FakeBackend(result={"metric": "val_loss", "baseline": 1.47, "method": 1.44,
                              "lower_is_better": True, "notes": "x"},
                      fail="run_after_write")
    res = run_agentic_improve("lower val loss", repo_url="https://x/nanoGPT",
                              workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)
    # the measured values survive the transient error
    assert res["baseline"].metric_value == 1.47 and res["baseline"].valid
    assert res["candidate"].metric_value == 1.44 and res["candidate"].valid


def test_salvage_baseline_when_infra_raises_after_write(tmp_path):
    """Same salvage for an InfraError (sandbox teardown race) after the agent wrote a baseline."""
    be = _FakeBackend(result={"metric": "acc", "baseline": 87.2, "method": None,
                              "lower_is_better": False, "notes": "y"},
                      fail="infra_after_write")
    res = run_agentic_improve("improve", repo_url="https://x/micrograd",
                              workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)
    assert res["baseline"].metric_value == 87.2 and res["baseline"].valid


def test_combined_salvage_still_requires_agent_baseline(tmp_path):
    """Method-phase salvage relaxation must not change the legacy combined contract."""
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
    }, fail="run_after_write")

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/nanoGPT", workroot=str(tmp_path),
        backend=be, clone_fn=_fake_clone,
    )

    assert res["baseline"] is None
    assert res["candidate"] is None
    assert "transient error" in res["error"]


def test_run_error_no_file_still_degrades_to_none(tmp_path):
    """If run_agent_task errors AND the agent wrote nothing, degrade honestly to invalid."""
    be = _FakeBackend(fail="run_after_write", write_result=False)
    res = run_agentic_improve("x", repo_url="https://x/repo",
                              workroot=str(tmp_path), backend=be, clone_fn=_fake_clone)
    assert res["baseline"] is None and "error" in res


def test_prompt_caps_variants_without_injecting_fairness_protocol(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _task_prompt
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", "6")
    p = _task_prompt("improve speedrun", "/repo", "/repo/ar_result.json", None, "seconds", True)
    low = p.lower()
    assert "canonical benchmark protocol" not in low
    assert "at most 6" in low and "attempt cap" in low     # 尝试上限=6，失败也计数
    assert "past 6" in low
    assert "past 3" not in low                              # 不得残留旧的三次上限
    assert "~2000 iters" not in p and "a few hundred to" not in p  # 漂移授权已删


def test_result_carries_config_for_reporting(tmp_path):
    """The agent's descriptive config remains available to reports and search memory."""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "val_bpb", "baseline": 1.05, "method": 1.02, "lower_is_better": True,
        "config": "depth=12, 2000 iters, data=fineweb, eval=val_bpb over 1M tokens"}),
        encoding="utf-8")
    res = _build_result_from_file(str(repo / "ar_result.json"), "nanochat",
                                  "val_bpb", True, lambda e, p: None)
    assert "2000 iters" in res["config"]


def test_result_accepts_method_metric_alias_from_repository_agent(tmp_path):
    """Do not discard a real method value when an agent uses the Runner field name."""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "pure_train_time_seconds",
        "baseline": 196.682,
        "method_metric": 196.721,
        "lower_is_better": True,
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic-speedrun",
        "pure_train_time_seconds", True, lambda e, p: None,
    )

    assert res["candidate"].metric_value == 196.721
    assert res["candidate"].valid is True
    assert res["verdict"]["improved"] is False


def test_method_result_accepts_value_inside_structured_metric(tmp_path):
    """A method artifact may use metric.value as its sole candidate value."""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": {
            "name": "runtime_seconds", "value": 8.565626, "unit": "s",
            "lower_is_better": True,
        },
        "baseline": 62.55084,
        "method": None,
        "attempt_id": "attempt-real-method",
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic-runtime-task",
        "runtime_seconds", True, lambda e, p: None, execution_phase="method",
    )

    assert res["candidate"].metric_value == 8.565626
    assert res["candidate"].valid is True
    assert res["verdict"]["improved"] is True
    assert res["_metric_attempt_id"] == "attempt-real-method"


@pytest.mark.parametrize("phase", [None, "baseline", "combined"])
def test_structured_metric_value_is_not_a_method_outside_method_phase(tmp_path, phase):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({
        "metric": {"name": "runtime_seconds", "value": 12.0, "unit": "s"},
        "baseline": 12.0,
    }), encoding="utf-8")
    result = _build_result_from_file(
        str(path), "generic-task", "runtime_seconds", True, lambda e, p: None,
        execution_phase=phase)

    assert result["baseline"].metric_value == 12.0
    assert result["candidate"].metric_value is None
    assert result["verdict"]["improved"] is False


def test_result_accepts_structured_speedrun_baseline_from_repository_agent(tmp_path):
    """A repository agent may attach threshold/eval evidence beside its numeric baseline."""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": {
            "name": "elapsed_seconds", "value": 83.5, "unit": "seconds",
            "threshold": 3.28, "lower_is_better": True,
        },
        "baseline": {
            "value": 83.5, "unit": "seconds", "val_loss": 3.2771,
            "attempt_id": "attempt-real-baseline",
        },
        "method": None,
        "lower_is_better": True,
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic-speedrun",
        "elapsed_seconds", True, lambda e, p: None,
    )

    assert res["metric"] == "elapsed_seconds"
    assert res["baseline"].metric_value == 83.5
    assert res["baseline"].valid is True
    assert res["_metric_attempt_id"] == "attempt-real-baseline"


def test_result_carries_structured_comparison_audit_for_semantic_review(tmp_path):
    """Do not discard the method-side conditions that the sandbox actually reported."""
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    audit = {
        "metric_measured": "validation_loss",
        "baseline_budget_seconds": 60,
        "method_budget_seconds": 60.2,
        "conditions_held_constant": ["one accelerator", "same validation evaluator"],
        "method_factors_changed": ["optimizer"],
    }
    (repo / "ar_result.json").write_text(json.dumps({
        "metric": "validation_loss", "baseline": 1.0, "method": 1.1,
        "lower_is_better": True, "comparison_audit": audit,
    }), encoding="utf-8")

    res = _build_result_from_file(
        str(repo / "ar_result.json"), "generic-benchmark",
        "validation_loss", True, lambda e, p: None,
    )

    assert res["comparison_audit"] == audit


class _ExecutingAttemptBackend(_FakeBackend):
    """Run the real host-installed attempt wrapper, then persist a cited result."""

    def __init__(self, *, result, commands=("true",), variant_from_attempt=None):
        super().__init__(result=result, record_attempt=False)
        self._commands = commands
        self._variant_from_attempt = variant_from_attempt
        if result.get("method") is not None:
            self.measurement_stdout = f"metric={result['method']} steps=1000\n"

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.task = task
        self.llm = llm
        self.timeout = timeout
        repo = Path(self._mounts[0]) / "repo"
        guard = Path(self._env["PATH"].split(os.pathsep)[0])
        python_link = guard / "python"
        if not python_link.exists():
            python_link.symlink_to("/usr/bin/python3")
        launcher = guard / "ar-attempt"
        attempt_ids = []
        for command in self._commands:
            subprocess.run(
                command, shell=True, env=self._env, cwd=repo,
                check=False, capture_output=True, text=True,
            )
            subprocess.run(
                [str(launcher), "timeout", "10s", "python", "train.py"], env=self._env,
                cwd=repo, check=False, capture_output=True, text=True,
            )
            attempt_ids.append(
                Path(self._env["AUTORESEARCH_LAST_ATTEMPT"]).read_text(
                    encoding="utf-8",
                ).strip()
            )

        payload = json.loads(json.dumps(self._result))
        payload.setdefault("attempt_id", attempt_ids[-1])
        if isinstance(payload.get("run_manifest"), dict):
            payload["run_manifest"].setdefault("attempt_id", attempt_ids[-1])
        (repo / "ar_result.json").write_text(json.dumps(payload), encoding="utf-8")
        if self._variant_from_attempt is not None:
            index, value = self._variant_from_attempt
            analysis = repo / "analysis"
            analysis.mkdir(exist_ok=True)
            (analysis / "variants.jsonl").write_text(json.dumps({
                "variant": "measured candidate",
                "result": value,
                "attempt_id": attempt_ids[index],
            }) + "\n", encoding="utf-8")
        self._agent_finished = True
        return _Res()


class _OmittedAttemptCitationBackend(_ExecutingAttemptBackend):
    """Model a successful sandbox run whose final JSON omits host attempt references."""

    def __init__(self, *, mutate_after_attempt=False, **kwargs):
        super().__init__(**kwargs)
        self._mutate_after_attempt = mutate_after_attempt

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
        repo = Path(self._mounts[0]) / "repo"
        payload = json.loads((repo / "ar_result.json").read_text(encoding="utf-8"))
        payload.pop("attempt_id", None)
        payload.pop("run_manifest", None)
        (repo / "ar_result.json").write_text(json.dumps(payload), encoding="utf-8")
        if self._mutate_after_attempt:
            (repo / "README.md").write_text("different post-attempt diff\n", encoding="utf-8")
        return result


def test_method_binds_omitted_citation_to_matching_host_attempt(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _OmittedAttemptCitationBackend(
        result={
            "metric": "val_loss", "baseline": None, "method": 1.7,
            "lower_is_better": True, "config": "steps=1000",
        },
        commands=("printf 'method change\\n' > README.md",),
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is True
    assert res["candidate"].metric_value == 1.7
    evidence = res["host_attestation"]["agent_attempt_evidence"]
    assert evidence["authority"] == "host_execution_broker"
    assert evidence["attempt_id"].startswith("attempt-")


def test_method_does_not_guess_omitted_citation_across_multiple_candidates(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _OmittedAttemptCitationBackend(
        result={
            "metric": "val_loss", "baseline": None, "method": 1.7,
            "lower_is_better": True, "config": "steps=1000",
        },
        commands=(
            "printf 'first candidate\\n' > README.md",
            "printf 'second candidate\\n' > README.md",
        ),
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert "agent attempt evidence invalid" in res["_host_measurement_error"]


def test_method_does_not_bind_attempt_after_tracked_diff_changes(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _OmittedAttemptCitationBackend(
        result={
            "metric": "val_loss", "baseline": None, "method": 1.7,
            "lower_is_better": True, "config": "steps=1000",
        },
        commands=("printf 'method change\\n' > README.md",),
        mutate_after_attempt=True,
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert "agent attempt evidence invalid" in res["_host_measurement_error"]


def test_attempt_wrapper_records_stable_completed_attestation_for_failure(tmp_path):
    """Removing finish/rc recording would make a crashed launch indistinguishable from a run."""
    from autoresearch.experiments.repo_improve.agentic_improve import _install_attempt_guard

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    bindir, log_path = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"),
    )
    last_attempt = str(Path(log_path).with_name("last_attempt_id"))
    env = dict(
        os.environ,
        AUTORESEARCH_ATTEMPT_LOG=log_path,
        AUTORESEARCH_LAST_ATTEMPT=last_attempt,
        AUTORESEARCH_AGENTIC_VARIANT_CAP="6",
        AUTORESEARCH_EXECUTION_PHASE="method",
        AUTORESEARCH_REPO_PATH=str(repo),
    )

    completed = subprocess.run(
        [str(Path(bindir) / "ar-attempt"), "sh", "-c", "exit 7"], env=env,
        check=False,
    )

    events = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    assert completed.returncode == 7
    assert [event["event"] for event in events] == ["launch", "finish"]
    assert events[0]["attempt_id"] == events[1]["attempt_id"]
    assert events[0]["argv"] == ["sh", "-c", "exit 7"]
    assert events[0]["started_at"] <= events[1]["ended_at"]
    assert events[1]["rc"] == 7
    assert Path(last_attempt).read_text().strip() == events[0]["attempt_id"]


def test_baseline_phase_rejects_tracked_worktree_mutation_and_attests_it(tmp_path):
    """A baseline number measured after editing tracked source is not an unmodified baseline."""
    clone, _commit = _local_git_clone(tmp_path)
    be = _ExecutingAttemptBackend(
        result={
            "metric": "val_loss", "baseline": 1.8, "method": None,
            "lower_is_better": True, "config": "steps=1000",
            "measurement": _measurement_spec(),
            "benchmark_contract": _v2_benchmark_contract(),
        },
        commands=("printf 'changed\\n' > README.md",),
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "baseline_provenance_violation"
    assert res["run_attestation"]["host_issued"] is True
    assert res["run_attestation"]["tracked_diff_present"] is True


def test_method_phase_attests_matching_attempt_and_real_tracked_diff(tmp_path):
    """The accepted metric must expose its exact command and repository state for persistence."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(
        result={
            "metric": "val_loss", "baseline": None, "method": 1.7,
            "lower_is_better": True, "config": "steps=1000",
            "run_manifest": {
                "contract_fingerprint": fingerprint, "controls": contract["controls"],
            },
        },
        commands=("printf 'method change\\n' > README.md",),
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )
    attestation = res["run_attestation"]
    assert res["candidate"].valid is True
    assert attestation["host_issued"] is True
    assert attestation["attempt_id"]
    assert attestation["argv"] == ["python", "train.py"]
    assert attestation["rc"] == 0
    assert attestation["repo_commit"] == commit
    assert attestation["tracked_diff_present"] is True
    assert attestation["metric_evidence"]["source"] == "host_parser"
    assert attestation["metric_evidence"]["metric"] == "val_loss"
    assert attestation["metric_evidence"]["value"] == 1.7


@pytest.mark.parametrize(
    ("reported_metric", "reported_direction"),
    [("accuracy", True), ("val_loss", False)],
)
def test_method_phase_rejects_explicit_frozen_objective_conflict(
    tmp_path, reported_metric, reported_direction,
):
    """An explicit agent objective cannot be silently rewritten to the frozen comparator."""
    clone, commit = _local_git_clone(tmp_path)
    be = _ExecutingAttemptBackend(result={
        "metric": reported_metric, "baseline": None, "method": 1.7,
        "lower_is_better": reported_direction, "config": "steps=1000",
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "candidate_contract_violation"
    assert "objective" in res["verdict"]["reason"]


def test_method_phase_rejects_variant_metric_spliced_with_other_attempt_manifest(tmp_path):
    """A better variants.jsonl number cannot inherit a different attempt's manifest."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(
        result={
            "metric": "val_loss", "baseline": None, "method": 1.7,
            "lower_is_better": True, "config": "steps=1000",
            "run_manifest": {
                "contract_fingerprint": fingerprint, "controls": contract["controls"],
            },
        },
        commands=("true", "true"), variant_from_attempt=(0, 1.6),
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "execution_failure"


def test_baseline_uses_host_remeasurement_not_agent_metric(tmp_path, monkeypatch):
    """Changing the agent JSON number cannot change the accepted host-parsed baseline."""
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "7200")
    monkeypatch.delenv("AUTORESEARCH_AGENTIC_PER_RUN_S", raising=False)
    clone, commit = _local_git_clone(tmp_path)
    be = _FakeBackend(
        result={
            "metric": "val_loss", "baseline": 999.0, "method": None,
            "lower_is_better": True, "config": "steps=1000",
            "measurement": _measurement_spec(),
            "benchmark_contract": _v2_benchmark_contract(999.0),
        },
        measurement_stdout="setup metric=9.9\nFINAL metric=1.75 steps=1000\n",
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].metric_value == 1.75
    assert res["baseline"].valid is True
    assert be.executed == [
        {"cmd": "python train.py", "cwd": str(tmp_path / "run" / "repo"),
         "timeout": 7200},
        {"cmd": "python evaluate.py --frozen " + str(
            tmp_path / "run" / "repo" / "metrics.json"),
         "cwd": str(tmp_path / "run" / "repo"),
         "timeout": 7200},
    ]
    attestation = res["host_attestation"]
    assert attestation["source"] == "host_remeasurement"
    assert attestation["repo_commit"] == commit
    assert attestation["measurement_spec"] == _measurement_spec()
    assert attestation["metric_evidence"]["value"] == 1.75
    assert res["benchmark_contract"]["verified_evidence"]["source"] == "host_remeasurement"
    assert res["benchmark_contract"]["observed_outcomes"]["quality"]["value"] == 1.75
    observations = res["benchmark_contract"]["verified_evidence"]["observations"]
    assert observations["immutable_file_hashes"] == _expected_immutable_hashes()
    assert observations["protocol_evidence"] == {
        "source": "host_parser", "unit": "steps", "value": 1000.0,
    }
    assert all(item["authority"] == "repository"
               for item in observations["official_source_evidence"])
    assert observations["trusted_probe_evidence"] == {
        "kind": "external_scorer",
        "argv": ["python", "evaluate.py", "--frozen", "{artifact}"], "cwd": ".",
    }


def test_method_uses_frozen_host_measurement_and_ignores_agent_replacement(tmp_path):
    """Method cannot replace the frozen evaluator/parser with an easier scorer."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract(1.8)
    frozen_spec = _measurement_spec(r"OFFICIAL=([0-9.]+)")
    contract["controls"]["evaluator"]["measurement_spec"] = frozen_spec
    contract["controls"]["evaluator"]["official_source_evidence"] = _official_evidence(
        frozen_spec)
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(
        result={
            "metric": "val_loss", "baseline": None, "method": -100.0,
            "lower_is_better": True, "config": "steps=1000",
            "measurement": _measurement_spec(r"FAKE=([0-9.]+)"),
            "run_manifest": {
                "contract_fingerprint": fingerprint, "controls": contract["controls"],
            },
        },
        commands=("printf 'method change\\n' > README.md",),
    )
    be.measurement_stdout = "FAKE=0.01\nOFFICIAL=1.7 steps=1000\n"

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract, frozen_spec),
        },
    )

    assert res["candidate"].metric_value == 1.7
    assert res["candidate"].valid is True
    assert res["host_attestation"]["measurement_spec"] == frozen_spec
    assert [item["cmd"] for item in be.executed] == [
        "python train.py", "python evaluate.py --frozen " + str(
            tmp_path / "run" / "repo" / "metrics.json")]


@pytest.mark.parametrize("bad_spec", [
    {"execution": {"argv": "python train.py", "cwd": "."},
     "trusted_probe": _measurement_spec()["trusted_probe"]},
    {"execution": {"argv": ["python", "train.py"], "cwd": "../escape"},
     "trusted_probe": _measurement_spec()["trusted_probe"]},
    {"execution": _measurement_spec()["execution"], "trusted_probe": {
        **_measurement_spec()["trusted_probe"],
        "parser": {"stream": "stdout", "regex": "(", "group": 1}}},
])
def test_host_remeasurement_rejects_unsafe_measurement_spec(tmp_path, bad_spec):
    clone, _commit = _local_git_clone(tmp_path)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": bad_spec, "benchmark_contract": _v2_benchmark_contract(),
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "measurement_spec_invalid"
    assert be.executed == []


def test_baseline_uses_v2_contract_measurement_when_top_level_is_omitted(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _measurement_spec()
    contract = _v2_benchmark_contract()
    contract["controls"]["evaluator"]["measurement_spec"] = measurement

    class _OmitTopLevelMeasurement(_FakeBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None):
            response = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            result_path = Path(self._mounts[0]) / "repo" / "ar_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload.pop("measurement", None)
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return response

    backend = _OmitTopLevelMeasurement(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8 steps=1000\n")

    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )

    assert produced["baseline"].valid is True, produced
    assert produced["host_attestation"]["measurement_spec"] == measurement


def test_baseline_rejects_conflicting_top_level_and_contract_measurements(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    conflicting = _measurement_spec(r"other_metric=([0-9.]+)")
    backend = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": conflicting, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8 steps=1000\n")

    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )

    assert produced["baseline"].valid is False
    assert "agent measurement differs from contract evaluator" in produced[
        "_host_measurement_error"]
    assert backend.executed == []


def test_confirmed_boundary_requires_agent_v2_and_carries_fixed_time_budget(
    tmp_path, monkeypatch,
):
    """Boundary runs cannot discard evidence or lose the confirmed 300s budget."""
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    contract["protocol"] = {
        "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 300},
        "timing_scope": "process", "timer_authority": "host",
    }
    contract["controls"]["workload"] = {"training_time_s": 300}
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "seconds=300",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    })
    ticks = iter([100.0, 400.0])
    monkeypatch.setattr(
        "autoresearch.experiments.repo_improve.agentic_improve.time.time",
        lambda: next(ticks),
    )

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", confirmed_boundary={
            "metric": "val_loss", "direction": "minimize",
            "protocol_kind": "fixed_wall_clock", "training_budget_s": 300,
        },
    )

    assert res["baseline"].valid is True
    assert res["benchmark_contract"]["version"] == 2
    assert res["benchmark_contract"]["protocol"]["kind"] == "quality_at_fixed_time"
    assert res["benchmark_contract"]["protocol"]["budget"]["wall_clock_s"] == 300


def test_confirmed_boundary_rejects_agent_v1_contract(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(), "benchmark_contract": _benchmark_contract(),
    })

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
        confirmed_boundary={"metric": "val_loss", "direction": "minimize"},
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "baseline_contract_invalid"


def test_method_null_with_real_tracked_diff_uses_host_measurement(tmp_path):
    clone, commit = _local_git_clone(tmp_path)
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, commands=("printf 'method change\\n' > README.md",))
    be.measurement_stdout = "metric=1.7 steps=1000\n"

    res = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is True
    assert res["candidate"].metric_value == 1.7
    assert [item["cmd"] for item in be.executed] == [
        "python train.py", "python evaluate.py --frozen " + str(
            tmp_path / "run" / "repo" / "metrics.json")]


def test_confirmed_boundary_rejects_agent_python_c_fake_measurement(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    fake = _measurement_spec()
    fake["execution"]["argv"] = [
        "python", "-c", "print('metric=0.0 steps=1000')"]
    contract = _v2_benchmark_contract()
    contract["controls"]["evaluator"]["measurement_spec"] = fake
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 0.0, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": fake, "benchmark_contract": contract,
    }, measurement_stdout="metric=0.0 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
        confirmed_boundary={"metric": "val_loss", "direction": "minimize"},
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "measurement_spec_invalid"
    assert be.executed == []


def test_confirmed_boundary_measurement_spec_is_authoritative(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    official = _measurement_spec(r"OFFICIAL=([0-9.]+)")
    agent = _measurement_spec(r"FAKE=([0-9.]+)")
    contract = _v2_benchmark_contract()
    contract["controls"]["evaluator"]["measurement_spec"] = agent
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 0.0, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": agent, "benchmark_contract": contract,
    }, measurement_stdout="FAKE=0.0 OFFICIAL=1.75 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", confirmed_boundary={
            "metric": "val_loss", "direction": "minimize", "measurement_spec": official,
        },
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.75
    assert res["host_attestation"]["measurement_spec"] == official


def test_structured_boundary_measurement_can_replace_repository_source_evidence(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    official = _measurement_spec()
    contract = _v2_benchmark_contract()
    contract["controls"]["evaluator"].pop("official_source_evidence")
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": official, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", confirmed_boundary={
            "metric": "val_loss", "direction": "minimize", "measurement_spec": official,
        },
    )

    assert res["baseline"].valid is True
    source = res["host_attestation"]["official_source_evidence"]
    assert [item["role"] for item in source] == ["execution", "trusted_probe"]
    assert all(item["authority"] == "user" for item in source)


@pytest.mark.parametrize("run_cmd", [
    "python evaluate.py --frozen; python -c 'print(0)'",
    "python evaluate.py --frozen | cat",
])
def test_confirmed_boundary_run_cmd_rejects_shell_control_syntax(tmp_path, run_cmd):
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 0.0, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", confirmed_boundary={
            "metric": "val_loss", "direction": "minimize", "run_cmd": run_cmd,
            "metric_parser": _measurement_spec()["trusted_probe"]["parser"],
        },
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "measurement_spec_invalid"
    assert be.executed == []


def test_confirmed_boundary_run_cmd_and_parser_are_authoritative(tmp_path):
    command = "python evaluate.py --frozen"
    clone, _commit = _clone_with_protocol_command(tmp_path, "python train.py")
    contract = _v2_benchmark_contract()
    boundary_spec = {
        "execution": {"argv": ["python", "train.py"], "cwd": "."},
        "trusted_probe": {
            "kind": "external_scorer", "argv": shlex.split(command), "cwd": ".",
            "parser": {"stream": "stdout", "regex": r"OFFICIAL=([0-9.]+)", "group": 1},
            "protocol_parser": _measurement_spec()["trusted_probe"]["protocol_parser"],
            "artifact": {"path": "metrics.json", "producer_sha256_before": None,
                         "required_new": True},
        },
    }
    boundary_spec["trusted_probe"]["argv"].append("{artifact}")
    contract["controls"]["evaluator"]["measurement_spec"] = boundary_spec
    contract["controls"]["evaluator"]["official_source_evidence"] = _official_evidence(
        boundary_spec, ("Official command: python train.py\n"
                        "Trusted probe: prepare.evaluate_bpb prepare.py\n"
                        "Trusted probe: external_scorer python evaluate.py --frozen "
                        "{artifact}\n").encode())
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 0.0, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(r"FAKE=([0-9.]+)"),
        "benchmark_contract": contract,
    }, measurement_stdout="FAKE=0.0 OFFICIAL=1.75 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", confirmed_boundary={
            "metric": "val_loss", "direction": "minimize",
            "run_cmd": command,
            "metric_parser": {
                "stream": "stdout", "regex": r"OFFICIAL=([0-9.]+)", "group": 1,
            },
            "protocol_parser": _measurement_spec()["trusted_probe"]["protocol_parser"],
        },
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.75
    assert be.executed[0]["cmd"] == "python train.py"
    assert shlex.split(be.executed[1]["cmd"])[:-1] == shlex.split(command)


def test_method_rejects_modified_immutable_evaluator(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)

    class _Tamper(_ExecutingAttemptBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None):
            result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            repo = Path(self._mounts[0]) / "repo"
            (repo / "evaluate.py").write_text("print('metric=0 steps=1000')\n")
            return result

    be = _Tamper(result={
        "metric": "val_loss", "baseline": None, "method": 0.0,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=("printf 'method change\\n' > README.md",))
    be.measurement_stdout = "metric=0 steps=1000\n"

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "immutable_evaluator_violation"
    assert be.executed == []


def test_method_rejects_modified_immutable_data(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)

    class _TamperData(_ExecutingAttemptBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None):
            result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            repo = Path(self._mounts[0]) / "repo"
            (repo / "data_config.txt").write_text("easier data\n", encoding="utf-8")
            return result

    be = _TamperData(result={
        "metric": "val_loss", "baseline": None, "method": 0.0,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=("printf 'method change\\n' > README.md",))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "immutable_evaluator_violation"
    assert be.executed == []


def test_method_allows_nonimmutable_tracked_treatment(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=("printf 'method change\\n' > README.md",))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is True
    assert res["candidate"].metric_value == 1.7


def test_fixed_work_rejects_actual_work_below_budget(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8 steps=10\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "protocol_evidence_violation"


def test_official_fixed_time_protocol_parser_applies_positive_multiplier(tmp_path):
    clone, _commit = _clone_with_protocol_command(tmp_path, "python train.py")
    spec = _external_measurement(protocol_regex=r"minutes=([0-9.]+)")
    spec["trusted_probe"]["protocol_parser"].update({
        "unit": "seconds", "multiplier": 60,
    })
    contract = _round4_contract(
        spec,
        protocol={
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 300},
            "timing_scope": "training", "timer_authority": "official_harness",
        },
    )
    contract["controls"]["workload"] = {"training_time_s": 300}
    backend = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "training_time_s=300",
        "measurement": spec, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 minutes=5\n")

    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )

    assert produced["baseline"].valid is True, produced
    assert produced["host_attestation"]["protocol_evidence"] == {
        "source": "host_parser", "unit": "seconds", "value": 300.0,
    }
    assert produced["host_attestation"]["measurement_spec"]["trusted_probe"][
        "protocol_parser"] == {
            "stream": "stdout", "regex": r"minutes=([0-9.]+)", "group": 1,
            "unit": "seconds", "multiplier": 60,
        }


def test_official_fixed_time_uses_minimum_five_second_upper_grace(tmp_path):
    clone, _commit = _clone_with_protocol_command(tmp_path, "python train.py")
    spec = _external_measurement(protocol_regex=r"seconds=([0-9.]+)")
    spec["trusted_probe"]["protocol_parser"]["unit"] = "seconds"
    contract = _round4_contract(
        spec,
        protocol={
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 1},
            "timing_scope": "training", "timer_authority": "official_harness",
        },
    )
    contract["controls"]["workload"] = {"training_time_s": 1}
    backend = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "training_time_s=1",
        "measurement": spec, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 seconds=7\n")

    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )

    assert produced["baseline"].valid is True, produced
    assert produced["host_attestation"]["protocol_evidence"]["value"] == 7.0


def test_protocol_parser_without_multiplier_keeps_legacy_exact_shape():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _normalize_measurement_spec,
    )

    spec = _measurement_spec()

    assert _normalize_measurement_spec(spec) == spec
    assert set(spec["trusted_probe"]["protocol_parser"]) == {
        "stream", "regex", "group", "unit",
    }


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_protocol_parser_rejects_nonpositive_or_nonfinite_multiplier(multiplier):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _normalize_measurement_spec,
    )

    spec = _measurement_spec()
    spec["trusted_probe"]["protocol_parser"]["multiplier"] = multiplier

    with pytest.raises(ValueError, match="measurement protocol_parser is invalid"):
        _normalize_measurement_spec(spec)


def test_fixed_time_rejects_short_echo_and_accepts_budget_elapsed(tmp_path, monkeypatch):
    clone, _commit = _local_git_clone(tmp_path)
    spec = _measurement_spec()
    spec["trusted_probe"].pop("protocol_parser")
    contract = _v2_benchmark_contract()
    contract["protocol"] = {
        "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 10},
        "timing_scope": "process", "timer_authority": "host",
    }
    contract["controls"]["workload"] = {"training_time_s": 10}
    contract["controls"]["evaluator"]["measurement_spec"] = spec
    contract["controls"]["evaluator"]["official_source_evidence"] = _official_evidence(spec)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "seconds=10",
        "measurement": spec, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8\n")

    ticks = iter([100.0, 101.0])
    monkeypatch.setattr("autoresearch.experiments.repo_improve.agentic_improve.time.time",
                        lambda: next(ticks))
    short = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "short"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )
    assert short["baseline"].valid is False
    assert short["diagnosis"] == "protocol_evidence_violation"

    be2 = _FakeBackend(result=be._result, measurement_stdout="metric=1.8\n")
    ticks = iter([200.0, 209.6])
    monkeypatch.setattr("autoresearch.experiments.repo_improve.agentic_improve.time.time",
                        lambda: next(ticks))
    fair = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "fair"),
        backend=be2, clone_fn=clone, execution_phase="baseline",
    )
    assert fair["baseline"].valid is True
    evidence = fair["host_attestation"]["protocol_evidence"]
    assert evidence["source"] == "host_elapsed"
    assert evidence["unit"] == "seconds"
    assert evidence["value"] == pytest.approx(9.6)


def test_confirmed_300_second_budget_rejects_agent_fixed_work_echo(tmp_path):
    """The boundary protocol, not the agent's fixed-work JSON, governs host evidence."""
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "seconds=300",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", confirmed_boundary={
            "metric": "val_loss", "direction": "minimize",
            "protocol_kind": "fixed_wall_clock", "training_budget_s": 300,
        },
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "protocol_evidence_violation"


def test_round4_rejects_tracked_fake_entry_without_official_command_evidence(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement(argv=["python", "train.py"])
    contract = _round4_contract(measurement)
    contract["controls"]["evaluator"]["official_source_evidence"][0]["command"] = (
        json.dumps({"argv": ["python", "fake.py"], "cwd": "."},
                   sort_keys=True, separators=(",", ":")))
    be = _LocalCommandBackend(result={
        "metric": "val_loss", "baseline": -999.0, "method": None,
        "lower_is_better": True, "config": "seconds=0.01",
        "measurement": measurement, "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "measurement_spec_invalid"
    assert be.executed == []


def test_round4_provenance_string_cannot_replace_official_source_evidence(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)
    contract["controls"]["evaluator"].pop("official_source_evidence")
    be = _LocalCommandBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "seconds=0.01",
        "measurement": measurement, "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "baseline_contract_invalid"
    assert be.executed == []


def test_round4_callable_probe_ignores_editable_stdout_and_attests_source(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)
    be = _LocalCommandBackend(result={
        "metric": "val_loss", "baseline": -999.0, "method": None,
        "lower_is_better": True, "config": "seconds=0.01",
        "measurement": measurement, "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.75
    attestation = res["host_attestation"]
    assert attestation["trusted_probe_evidence"]["kind"] == "python_callable"
    assert attestation["trusted_probe_evidence"]["call_count"] == 1
    assert all(item["authority"] == "repository"
               for item in attestation["official_source_evidence"])
    assert all(item["path"] == "PROTOCOL.md"
               for item in attestation["official_source_evidence"])


def test_callable_metric_uses_execution_training_minutes_protocol_parser(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    (source / "train.py").write_text(
        "from prepare import evaluate_bpb\n"
        "from pathlib import Path\n"
        "Path('metrics.json').write_text('{}\\n')\n"
        "print('training_minutes=5')\n"
        "evaluate_bpb()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
    subprocess.run([
        "git", "-C", str(source), "commit", "-q", "--amend", "--no-edit",
    ], check=True)
    measurement = _callable_measurement()
    measurement["trusted_probe"]["protocol_parser"] = {
        "stream": "stdout", "regex": r"training_minutes=([0-9.]+)", "group": 1,
        "unit": "seconds", "multiplier": 60,
    }
    contract = _round4_contract(
        measurement,
        protocol={
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 300},
            "timing_scope": "training", "timer_authority": "official_harness",
        },
    )
    contract["controls"]["workload"] = {"training_time_s": 300}
    backend = _LocalCommandBackend(result={
        "metric": "val_loss", "baseline": -999.0, "method": None,
        "lower_is_better": True, "config": "training_time_s=300",
        "measurement": measurement, "benchmark_contract": contract,
    })

    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )

    assert produced["baseline"].valid is True, produced
    assert produced["baseline"].metric_value == 1.75
    assert produced["host_attestation"]["protocol_evidence"] == {
        "source": "host_parser", "unit": "seconds", "value": 300.0,
    }
    assert produced["host_attestation"]["trusted_probe_evidence"]["kind"] == (
        "python_callable")


@pytest.mark.parametrize("stream", ["stderr", "combined"])
def test_callable_protocol_parser_rejects_non_stdout_stream(stream):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _normalize_measurement_spec,
    )

    measurement = _callable_measurement()
    measurement["trusted_probe"]["protocol_parser"] = {
        "stream": stream, "regex": r"seconds=([0-9.]+)", "group": 1,
        "unit": "seconds",
    }

    with pytest.raises(ValueError, match="python_callable protocol_parser requires stdout"):
        _normalize_measurement_spec(measurement)


def test_callable_probe_marker_cannot_supply_training_seconds(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    measurement["trusted_probe"]["protocol_parser"] = {
        "stream": "stderr", "regex": r'"value":([0-9.]+)', "group": 1,
        "unit": "seconds", "multiplier": 60,
    }
    contract = _round4_contract(
        measurement,
        protocol={
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 105},
            "timing_scope": "training", "timer_authority": "official_harness",
        },
    )
    contract["controls"]["workload"] = {"training_time_s": 105}
    backend = _LocalCommandBackend(result={
        "metric": "val_loss", "baseline": -999.0, "method": None,
        "lower_is_better": True, "config": "training_time_s=105",
        "measurement": measurement, "benchmark_contract": contract,
    })

    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )

    assert produced["baseline"].valid is False
    assert produced["diagnosis"] == "measurement_spec_invalid"
    assert backend.executed == []


def test_round4_callable_probe_rejects_changed_host_bootstrap(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)

    class _MutateBootstrap(_LocalCommandBackend):
        def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
            measured = super().execute_command(handle, cmd, cwd=cwd, timeout=timeout)
            pythonpath = next(
                token.split("=", 1)[1] for token in shlex.split(cmd)
                if token.startswith("PYTHONPATH="))
            bootstrap = Path(pythonpath) / "sitecustomize.py"
            bootstrap.chmod(0o644)
            bootstrap.write_text("# changed\n", encoding="utf-8")
            return measured

    be = _MutateBootstrap(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "seconds=0.01",
        "measurement": measurement, "benchmark_contract": contract,
    })
    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "immutable_evaluator_violation"


def test_round4_method_may_edit_entry_but_not_fake_probe_metric(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)
    fingerprint = contract_fingerprint(contract)

    class _EditTrain(_LocalCommandBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None):
            result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            repo = Path(self._mounts[0]) / "repo"
            (repo / "train.py").write_text(
                "from prepare import evaluate_bpb\n"
                "print('metric=-999999')\n"
                "evaluate_bpb()\n",
                encoding="utf-8",
            )
            return result

    be = _EditTrain(result={
        "metric": "val_loss", "baseline": None, "method": -999999.0,
        "lower_is_better": True, "config": "seconds=0.01",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    })
    frozen_hashes = {
        "data_config.txt": "sha256:" + hashlib.sha256(b"canonical data\n").hexdigest(),
        "prepare.py": "sha256:" + hashlib.sha256(
            b"def evaluate_bpb():\n    return 1.75\n").hexdigest(),
        "PROTOCOL.md": "sha256:" + hashlib.sha256(
            b"Official command: python train.py\n").hexdigest(),
    }

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "seconds=0.01", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _callable_frozen_attestation(contract),
        },
    )

    assert res["candidate"].valid is True
    assert res["candidate"].metric_value == 1.75


def test_round4_method_rejects_modified_callable_source(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)
    fingerprint = contract_fingerprint(contract)

    class _EditPrepare(_LocalCommandBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None):
            result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            repo = Path(self._mounts[0]) / "repo"
            (repo / "prepare.py").write_text(
                "def evaluate_bpb():\n    return -999.0\n", encoding="utf-8")
            return result

    be = _EditPrepare(result={
        "metric": "val_loss", "baseline": None, "method": -999.0,
        "lower_is_better": True, "config": "seconds=0.01",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    })
    hashes = {
        "data_config.txt": "sha256:" + hashlib.sha256(b"canonical data\n").hexdigest(),
        "prepare.py": "sha256:" + hashlib.sha256(
            b"def evaluate_bpb():\n    return 1.75\n").hexdigest(),
        "PROTOCOL.md": "sha256:" + hashlib.sha256(
            b"Official command: python train.py\n").hexdigest(),
    }

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "seconds=0.01", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _callable_frozen_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "immutable_evaluator_violation"
    assert be.executed == []


@pytest.mark.parametrize("argv,command", [
    (["python", "-m", "train"], "python -m train"),
    (["torchrun", "--nproc-per-node=1", "-m", "train"],
     "torchrun --nproc-per-node=1 -m train"),
])
def test_round4_python_and_torchrun_modules_resolve_to_tracked_entry(
    tmp_path, argv, command,
):
    clone, _commit = _clone_with_protocol_command(tmp_path, command)
    measurement = _external_measurement(execution_argv=argv)
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command=command,
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.75


def test_round4_external_scorer_ignores_execution_stdout(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _external_measurement()
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
    )
    contract["controls"]["workload"] = {"steps": 1000}

    class _ExternalOutput(_FakeBackend):
        def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
            if "suffixes, names, blocked" in cmd or "source=data[:24000]" in cmd:
                return super().execute_command(handle, cmd, cwd=cwd, timeout=timeout)
            from types import SimpleNamespace
            self.executed.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
            if cmd == "python train.py":
                Path(cwd, "metrics.json").write_text("{}\n", encoding="utf-8")
                os.utime(Path(cwd, "metrics.json"), (2_000_000_000, 2_000_000_000))
                return SimpleNamespace(
                    exit_code=0, stdout="metric=-999 steps=1000\n", stderr="", timed_out=False)
            return SimpleNamespace(
                exit_code=0, stdout="metric=1.75 steps=1000\n", stderr="", timed_out=False)

    be = _ExternalOutput(result={
        "metric": "val_loss", "baseline": -999.0, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    })
    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.75
    attestation = res["host_attestation"]
    assert attestation["trusted_probe_evidence"]["kind"] == "external_scorer"
    assert attestation["artifact_evidence"]["path"] == "metrics.json"
    assert attestation["artifact_evidence"]["sha256"].startswith("sha256:")
    assert "{artifact}" not in be.executed[1]["cmd"]
    assert str(tmp_path / "run" / "repo" / "metrics.json") in be.executed[1]["cmd"]


def test_round4_fixed_work_rejects_width_as_fake_steps(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _external_measurement(protocol_regex=r"actual_steps=([0-9.]+)")
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
    )
    contract["controls"]["workload"] = {"steps": 1000, "width": 1000}
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000,width=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 width=1000 actual_steps=1\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "protocol_evidence_violation"


def test_external_scorer_rejects_nonnull_prior_artifact_policy(tmp_path):
    """The safe no-move policy cannot claim to authenticate a prior artifact hash."""
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _external_measurement()
    measurement["trusted_probe"]["artifact"]["producer_sha256_before"] = (
        "sha256:" + "0" * 64)
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "measurement_spec_invalid"
    assert be.executed == []


def test_fixed_work_uses_execution_evidence_not_scorer_echo(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    measurement = _external_measurement(protocol_regex=r"steps=([0-9.]+)")
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
    )
    contract["controls"]["workload"] = {"steps": 1000}

    class _DifferentWorkEvidence(_FakeBackend):
        def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
            from types import SimpleNamespace
            self.executed.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
            if cmd == "python train.py":
                Path(cwd, "metrics.json").write_text("{}\n", encoding="utf-8")
                os.utime(Path(cwd, "metrics.json"), (2_000_000_000, 2_000_000_000))
                return SimpleNamespace(
                    exit_code=0, stdout="steps=1\n", stderr="", timed_out=False)
            return SimpleNamespace(
                exit_code=0, stdout="metric=1.75 steps=1000\n", stderr="", timed_out=False)

    be = _DifferentWorkEvidence(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    })
    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "protocol_evidence_violation"


def test_baseline_v2_without_attempt_ledger_is_invalid(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8 steps=1000\n", record_attempt=False)

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "execution_failure"
    assert be.executed == []


def test_method_resource_evidence_must_match_baseline(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    contract["controls"]["resources"]["accelerator_count"] = 2
    fingerprint = contract_fingerprint(contract)
    frozen_attestation = _frozen_host_attestation(contract)
    frozen_attestation["resource_evidence"] = {
        "accelerator_count": 2, "device_models": ["unknown", "unknown"],
        "source": "allocation_count_only",
        "caveat": ("precision, seed, and model scale are contract-frozen, not host-observed; "
                   "device model not verified (nvidia-smi unavailable)"),
    }
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=("printf 'method change\\n' > README.md",))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", cuda_device="1",
        frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": frozen_attestation,
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "resource_evidence_violation"
    assert be.executed == []


@pytest.mark.parametrize("command", [
    "printf 'def evaluate_bpb():\\n    return 1.76\\n' > prepare.py",
    "printf 'AUTORESEARCH_PROBE_FAKE\\n' > README.md",
])
def test_method_rejects_outside_treatment_and_reserved_probe_diff(tmp_path, command):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=(command,))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "immutable_evaluator_violation"
    assert be.executed == []


def test_method_checks_out_detached_frozen_commit_when_upstream_head_moved(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, frozen_commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    (source / "README.md").write_text("new upstream head\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "move head"], check=True)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=("printf 'method change\\n' > README.md",))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": frozen_commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is True
    assert res["repo_commit"] == frozen_commit


def test_baseline_prompt_omits_retired_probe_contract():
    from autoresearch.experiments.repo_improve.agentic_improve import _baseline_task_prompt

    prompt = _baseline_task_prompt("goal", "/repo", "/repo/ar_result.json", None, None)
    assert "prepare.evaluate_bpb" not in prompt
    assert "<trusted_metric_callable>" not in prompt
    assert "controls.treatment.allowed_paths" not in prompt
    assert "producer_sha256_before:null" not in prompt


def test_resource_identity_ignores_ordinal_when_models_match():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _resource_evidence, _resource_pair_matches,
    )

    baseline = _resource_evidence("0", query_fn=lambda ids: [" NVIDIA   L20 "])
    method = _resource_evidence("3", query_fn=lambda ids: ["nvidia l20"])

    assert baseline == {
        "accelerator_count": 1,
        "device_models": ["nvidia l20"],
        "source": "nvidia-smi",
        "caveat": "precision, seed, and model scale are contract-frozen, not host-observed",
    }
    assert method == baseline
    assert _resource_pair_matches(method, baseline) is True


@pytest.mark.parametrize("current", [
    {"accelerator_count": 1, "device_models": ["nvidia a100"],
     "source": "nvidia-smi", "caveat": "x"},
    {"accelerator_count": 2, "device_models": ["nvidia l20", "nvidia l20"],
     "source": "nvidia-smi", "caveat": "x"},
])
def test_resource_identity_rejects_different_model_or_count(current):
    from autoresearch.experiments.repo_improve.agentic_improve import _resource_pair_matches

    baseline = {
        "accelerator_count": 1, "device_models": ["nvidia l20"],
        "source": "nvidia-smi", "caveat": "x",
    }
    assert _resource_pair_matches(current, baseline) is False


def test_resource_evidence_marks_nvidia_smi_unavailable_as_unknown():
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _resource_evidence, _resource_pair_matches,
    )

    def unavailable(_ids):
        raise FileNotFoundError("nvidia-smi")

    baseline = _resource_evidence("0", query_fn=unavailable)
    method = _resource_evidence("3", query_fn=unavailable)

    assert baseline["device_models"] == ["unknown"]
    assert baseline["source"] == "allocation_count_only"
    assert "not verified" in baseline["caveat"]
    assert _resource_pair_matches(method, baseline) is True
    assert _resource_pair_matches(
        _resource_evidence("2,3", query_fn=unavailable), baseline) is False


@pytest.mark.parametrize("field", ["device_model", "resource_class"])
def test_explicit_contract_model_fails_closed_when_host_model_unknown(field):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _validate_resource_contract,
    )

    contract = {"controls": {"resources": {
        "accelerator_count": 1, field: "NVIDIA L20",
    }}}
    evidence = {
        "accelerator_count": 1, "device_models": ["unknown"],
        "source": "allocation_count_only", "caveat": "device model not verified",
    }

    with pytest.raises(ValueError, match=f"{field} cannot be verified"):
        _validate_resource_contract(contract, evidence)


def test_baseline_rejects_contract_accelerator_count_mismatch(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    contract["controls"]["resources"]["accelerator_count"] = 1
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    }, measurement_stdout="metric=1.8 steps=1000\n")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline", cuda_device=None,
    )

    assert res["baseline"].valid is False
    assert res["diagnosis"] == "resource_evidence_violation"
    assert be.executed == []


@pytest.mark.parametrize("observed_model,expected_valid", [
    ("NVIDIA L20", True),
    ("NVIDIA A100-SXM4-80GB", False),
])
def test_method_resource_pair_uses_model_not_ordinal(
    tmp_path, monkeypatch, observed_model, expected_valid,
):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    from autoresearch.experiments.repo_improve.agentic_improve import _resource_evidence

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    contract["controls"]["resources"].update({
        "accelerator_count": 1, "device_model": "NVIDIA L20",
    })
    fingerprint = contract_fingerprint(contract)
    frozen_attestation = _frozen_host_attestation(contract)
    frozen_attestation["resource_evidence"] = _resource_evidence(
        "0", query_fn=lambda ids: ["NVIDIA L20"])
    monkeypatch.setattr(
        "autoresearch.experiments.repo_improve.agentic_improve._query_device_models",
        lambda ids: [observed_model],
    )
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint,
                         "controls": contract["controls"]},
    }, commands=("printf 'method change\\n' > README.md",))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", cuda_device="3",
        frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": frozen_attestation,
        },
    )

    assert res["candidate"].valid is expected_valid
    if not expected_valid:
        assert res["diagnosis"] == "resource_evidence_violation"
        assert be.executed == []


class _DeclaredArtifactAttemptBackend(_FakeBackend):
    """Run the declared execution through the real ledger before host verification."""

    def __init__(self, *, result, execution_argv, prior_artifact=None, attempts=1):
        super().__init__(result=result, record_attempt=False)
        self._execution_argv = execution_argv
        self._prior_artifact = prior_artifact
        self._attempts = attempts

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.task = task
        self.llm = llm
        self.timeout = timeout
        repo = Path(self._mounts[0]) / "repo"
        artifact = repo / "metrics.json"
        if self._prior_artifact == "stale":
            artifact.write_text("stale\n", encoding="utf-8")
            os.utime(artifact, (1, 1))
        elif self._prior_artifact == "symlink":
            artifact.symlink_to("README.md")
        launcher = Path(self._env["PATH"].split(os.pathsep)[0]) / "ar-attempt"
        for _index in range(self._attempts):
            completed = subprocess.run(
                [str(launcher), "timeout", "10s", *self._execution_argv],
                env=self._env, cwd=repo, check=False, capture_output=True, text=True,
            )
            assert completed.returncode == 0
        attempt_id = Path(self._env["AUTORESEARCH_LAST_ATTEMPT"]).read_text(
            encoding="utf-8").strip()
        payload = json.loads(json.dumps(self._result))
        payload["attempt_id"] = attempt_id
        if isinstance(payload.get("run_manifest"), dict):
            payload["run_manifest"]["attempt_id"] = attempt_id
        (repo / "ar_result.json").write_text(json.dumps(payload), encoding="utf-8")
        return _Res()


class _LedgerTamperBackend(_FakeBackend):
    def __init__(self, *, result, mode):
        super().__init__(result=result, record_attempt=False)
        self.mode = mode
        self.seventh_rc = None

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        repo = Path(self._mounts[0]) / "repo"
        guard = Path(self._env["PATH"].split(os.pathsep)[0])
        launcher = guard / "ar-attempt"
        attempts = 6 if self.mode == "truncate_cap" else 1
        for _index in range(attempts):
            subprocess.run(
                [str(launcher), "true"], env=self._env, cwd=repo,
                check=False, capture_output=True, text=True,
            )
        public_log = guard.parent / "attempts.tsv"
        forged_id = "attempt-forged"
        empty_diff_hash = "sha256:" + hashlib.sha256(b"").hexdigest()
        forged_events = [
            {"event": "launch", "attempt_id": forged_id, "phase": self._env[
                "AUTORESEARCH_EXECUTION_PHASE"], "argv": ["python3", "train.py"],
             "cwd": str(repo), "started_at": time.time() - 0.1,
             "repo_commit": subprocess.run(
                 ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                 capture_output=True, text=True).stdout.strip(),
             "tracked_diff_hash_start": empty_diff_hash},
            {"event": "finish", "attempt_id": forged_id, "phase": self._env[
                "AUTORESEARCH_EXECUTION_PHASE"], "ended_at": time.time(), "rc": 0,
             "repo_commit_end": subprocess.run(
                 ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                 capture_output=True, text=True).stdout.strip(),
             "tracked_diff_hash_end": empty_diff_hash},
        ]
        try:
            public_log.write_text(
                "\n".join(json.dumps(event) for event in forged_events) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        if self.mode == "truncate_cap":
            self.seventh_rc = subprocess.run(
                [str(launcher), "true"], env=self._env, cwd=repo,
                check=False, capture_output=True, text=True,
            ).returncode
        payload = json.loads(json.dumps(self._result))
        payload["attempt_id"] = forged_id
        if isinstance(payload.get("run_manifest"), dict):
            payload["run_manifest"]["attempt_id"] = forged_id
        (repo / "ar_result.json").write_text(json.dumps(payload), encoding="utf-8")
        return _Res()


def test_forged_truncated_sandbox_ledger_cannot_authorize_measurement(tmp_path):
    clone, _commit = _clone_with_protocol_command(tmp_path, "python3 train.py")
    measurement = _external_measurement(execution_argv=["python3", "train.py"])
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command="python3 train.py",
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _LedgerTamperBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, mode="forge")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert "agent attempt evidence invalid" in res["_host_measurement_error"]


def test_truncating_sandbox_ledger_cannot_reset_host_method_cap(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_VARIANT_CAP", "6")

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _LedgerTamperBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint},
    }, mode="truncate_cap")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert be.seventh_rc == 125
    assert res["candidate"].valid is False


def test_external_artifact_from_cited_attempt_is_quarantined_before_host_rerun(tmp_path):
    """The real agent measurement output must not collide with host recreation."""
    clone, _commit = _clone_with_protocol_command(tmp_path, "python3 train.py")
    measurement = _external_measurement(execution_argv=["python3", "train.py"])
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command="python3 train.py",
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _DeclaredArtifactAttemptBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, execution_argv=["python3", "train.py"])

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"] is not None, res
    assert res["baseline"].valid is True
    prior = res["host_attestation"]["prior_artifact_evidence"]
    assert prior["state"] == "quarantined"
    assert prior["source"] == "cited_agent_attempt"
    assert prior["path"] == "metrics.json"
    assert prior["sha256"].startswith("sha256:")
    assert Path(prior["quarantine_path"]).is_file()
    assert (tmp_path / "run" / "repo" / "metrics.json").is_file()
    attempt = res["host_attestation"]["agent_attempt_evidence"]
    assert attempt["authority"] == "host_execution_broker"
    assert attempt["normalized_argv"] == ["python3", "train.py"]
    assert attempt["execution_argv"] == ["python3", "train.py"]
    assert attempt["cwd"] == str(tmp_path / "run" / "repo")
    assert attempt["count"] == 1


def test_successful_unrelated_attempt_cannot_authorize_host_measurement(tmp_path, monkeypatch):
    """A successful `ar-attempt true` is not evidence for the frozen benchmark command."""
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "7200")
    clone, _commit = _local_git_clone(tmp_path)
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": _measurement_spec(),
        "benchmark_contract": _v2_benchmark_contract(),
    }, attempt_argv=["true"])

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert "does not invoke frozen measurement execution" in res["_host_measurement_error"]
    assert be.executed == []
    assert be.attempt_executed[0]["cmd"].endswith("timeout 10s true")
    assert be.attempt_executed[0]["timeout"] == 7210
    command = shlex.split(be.attempt_executed[0]["cmd"])
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        assert ["-u", key] == command[command.index(key) - 1:command.index(key) + 1]
    cleanup = next((i for i, token in enumerate(command)
                    if token.endswith("/timeout-cleanup")), None)
    assert cleanup is not None, "host deadline must also bound the sandbox process group"
    assert command[cleanup + 1:cleanup + 3] == ["--kill-after=2s", "7200s"]


def test_baseline_reproduction_attempts_are_not_charged_to_method_cap(tmp_path):
    clone, _commit = _clone_with_protocol_command(tmp_path, "python3 train.py")
    measurement = _external_measurement(execution_argv=["python3", "train.py"])
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command="python3 train.py",
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _DeclaredArtifactAttemptBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, execution_argv=["python3", "train.py"], attempts=7)

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is True
    assert res["host_attestation"]["agent_attempt_evidence"]["count"] == 7


def test_baseline_reproduction_is_not_paused_by_method_stop_predicate(tmp_path):
    class StopPredicateBackend(_DeclaredArtifactAttemptBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None,
                           stop_predicate=None):
            self.stop_predicate = stop_predicate
            return super().run_agent_task(
                handle, task, llm=llm, timeout=timeout)

    clone, _commit = _clone_with_protocol_command(tmp_path, "python3 train.py")
    measurement = _external_measurement(execution_argv=["python3", "train.py"])
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command="python3 train.py",
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = StopPredicateBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, execution_argv=["python3", "train.py"], attempts=7)

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is True
    assert be.stop_predicate is None


def test_method_created_commit_is_candidate_violation_not_baseline_drift(tmp_path):
    """A correct frozen checkout remains valid when the candidate commits its treatment."""
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, commands=(
        "printf 'candidate commit\\n' > README.md && git add README.md && "
        "git -c user.name=Candidate -c user.email=candidate@example.invalid "
        "commit -qm candidate",
    ))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["baseline"].valid is True
    assert res["baseline"].metric_value == 1.8
    assert res["candidate"].valid is False
    assert res["diagnosis"] == "candidate_contract_violation"
    assert "candidate created a commit" in res["error"]


def test_stale_declared_artifact_is_quarantined_and_rejected_distinctly(tmp_path):
    clone, _commit = _clone_with_protocol_command(tmp_path, "python3 evaluate.py")
    measurement = _external_measurement(execution_argv=["python3", "evaluate.py"])
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command="python3 evaluate.py",
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _DeclaredArtifactAttemptBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, execution_argv=["python3", "evaluate.py"], prior_artifact="stale")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"] is not None, res
    assert res["baseline"].valid is False
    assert "stale pre-attempt artifact" in res["_host_measurement_error"]
    prior = res["artifact_quarantine_evidence"]
    assert prior["state"] == "quarantined"
    assert prior["source"] == "stale_pre_attempt"
    assert Path(prior["quarantine_path"]).is_file()
    assert not (tmp_path / "run" / "repo" / "metrics.json").exists()


def test_declared_artifact_symlink_is_rejected_without_following_it(tmp_path):
    clone, _commit = _clone_with_protocol_command(tmp_path, "python3 evaluate.py")
    measurement = _external_measurement(execution_argv=["python3", "evaluate.py"])
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command="python3 evaluate.py",
    )
    contract["controls"]["workload"] = {"steps": 1000}
    be = _DeclaredArtifactAttemptBackend(result={
        "metric": "val_loss", "baseline": 1.75, "method": None,
        "lower_is_better": True, "config": "steps=1000",
        "measurement": measurement, "benchmark_contract": contract,
    }, execution_argv=["python3", "evaluate.py"], prior_artifact="symlink")

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"] is not None, res
    assert res["baseline"].valid is False
    assert "artifact symlink is unsafe" in res["_host_measurement_error"]
    assert (tmp_path / "run" / "repo" / "metrics.json").is_symlink()


def _clone_with_common_cli_sources(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    (source / "bench.sh").write_text("#!/bin/sh\nexec python train.py\n", encoding="utf-8")
    (source / "Makefile").write_text("bench:\n\tpython train.py\n", encoding="utf-8")
    (source / "Bench.mk").write_text("bench:\n\tpython train.py\n", encoding="utf-8")
    (source / "build").mkdir()
    (source / "build" / "Bench.mk").write_text(
        "bench:\n\tpython ../train.py\n", encoding="utf-8")
    (source / "bench.env").write_text("SEED=17\n", encoding="utf-8")
    (source / "requirements-bench.txt").write_text("pytest==8.0.0\n", encoding="utf-8")
    (source / "tests").mkdir()
    (source / "tests" / "test_bench.py").write_text(
        "def test_metric():\n    assert True\n", encoding="utf-8")
    (source / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (source / "Cargo.toml").write_text(
        "[package]\nname='bench-fixture'\nversion='0.1.0'\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='bench-fixture'\nversion='0.1.0'\n"
        "[project.scripts]\nbench-cli='train:main'\n",
        encoding="utf-8",
    )
    subprocess.run([
        "git", "-C", str(source), "add", "bench.sh", "Makefile", "tests/test_bench.py",
        "Bench.mk", "bench.env", "requirements-bench.txt", "pytest.ini", "Cargo.toml",
        "pyproject.toml", "build/Bench.mk",
    ], check=True)
    subprocess.run([
        "git", "-C", str(source), "commit", "-q", "-m", "cli fixtures",
    ], check=True)
    return clone


@pytest.mark.parametrize("argv,expected", [
    (["bash", "bench.sh"], "bench.sh"),
    (["make", "bench"], "Makefile"),
    (["pytest", "-q", "-c", "pytest.ini", "tests/test_bench.py::test_metric"],
     "tests/test_bench.py"),
    (["cargo", "bench"], "Cargo.toml"),
    (["uv", "run", "--project", ".", "bench-cli"], "train.py"),
])
def test_common_cli_execution_forms_resolve_to_tracked_sources(tmp_path, argv, expected):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_entry

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    assert _execution_entry(str(repo), {"argv": argv, "cwd": "."}) == expected


@pytest.mark.parametrize("argv,reason", [
    (["bash", "-c", "python train.py"], "shell -c is not a safe measurement entrypoint"),
    (["make"], "make measurement requires one explicit target"),
    (["pytest", "-q"], "pytest measurement requires an explicit tracked test entry"),
    (["cargo", "bench", "--manifest-path", "../Cargo.toml"],
     "cargo manifest escapes repository"),
    (["uv", "run", "--project", ".", "not-declared"],
     "uv console script is not declared in tracked pyproject.toml"),
])
def test_common_cli_execution_forms_fail_closed_when_source_is_unsafe(
    tmp_path, argv, reason,
):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_entry

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    with pytest.raises(ValueError, match=re.escape(reason)):
        _execution_entry(str(repo), {"argv": argv, "cwd": "."})


def test_make_long_file_option_binds_the_explicit_tracked_makefile(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    assert _execution_source_paths(
        str(repo), {"argv": ["make", "--file=Bench.mk", "bench"], "cwd": "."},
    ) == ["Bench.mk"]


def test_make_joined_file_option_binds_the_explicit_tracked_makefile(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    assert _execution_source_paths(
        str(repo), {"argv": ["make", "-fBench.mk", "bench"], "cwd": "."},
    ) == ["Bench.mk"]


def test_make_binds_every_repeated_explicit_makefile(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    assert _execution_source_paths(
        str(repo),
        {"argv": ["make", "-fMakefile", "-fBench.mk", "bench"], "cwd": "."},
    ) == ["Makefile", "Bench.mk"]


def test_make_joined_directory_option_binds_entrypoint_under_that_directory(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    assert _execution_source_paths(
        str(repo),
        {"argv": ["make", "-Cbuild", "-fBench.mk", "bench"], "cwd": "."},
    ) == ["build/Bench.mk"]


@pytest.mark.parametrize("eval_option", [
    "--eval", "--eval=bench: ; python train.py", "-E", "-Ebench: ; python train.py",
])
def test_make_eval_fragment_fails_closed(tmp_path, eval_option):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    argv = ["make", eval_option]
    if eval_option in {"--eval", "-E"}:
        argv.append("bench: ; python train.py")
    argv.append("bench")
    with pytest.raises(ValueError, match="make --eval is not a verifiable entrypoint"):
        _execution_source_paths(str(repo), {"argv": argv, "cwd": "."})


@pytest.mark.parametrize("option", ["-Iincludes", "--include-dir=includes", "--mystery"])
def test_make_ambiguous_source_options_fail_closed(tmp_path, option):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))

    with pytest.raises(ValueError, match="unsupported make option"):
        _execution_source_paths(
            str(repo), {"argv": ["make", option, "bench"], "cwd": "."},
        )


@pytest.mark.parametrize("option,path", [
    ("--env-file", "bench.env"),
    ("--with-requirements", "requirements-bench.txt"),
])
def test_uv_run_binds_tracked_environment_and_requirement_inputs(tmp_path, option, path):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _execution_source_paths, _immutable_file_hashes,
    )

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))
    command = {
        "argv": ["uv", "run", "--project", ".", option, path, "bench-cli"], "cwd": ".",
    }
    measurement = _measurement_spec()
    measurement["execution"] = command
    contract = _v2_benchmark_contract()
    contract["controls"]["evaluator"]["measurement_spec"] = measurement

    assert path in _execution_source_paths(str(repo), command)
    assert {"pyproject.toml", path} <= set(
        _immutable_file_hashes(str(repo), contract, measurement))


@pytest.mark.parametrize("option", [
    "--with", "--with=pytest", "-w", "-wpytest", "--with-editable", "--with-editable=.",
])
def test_uv_run_rejects_unbound_dependency_overrides(tmp_path, option):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))
    argv = ["uv", "run", "--project", ".", option]
    if option in {"--with", "-w", "--with-editable"}:
        argv.append("pytest" if option == "--with" else ".")
    argv.append("bench-cli")

    with pytest.raises(ValueError, match="uv dependency override is not host-verifiable"):
        _execution_source_paths(str(repo), {"argv": argv, "cwd": "."})


@pytest.mark.parametrize("option,value", [
    ("--python", "3.12"),
    ("--python=3.12", None),
    ("-p", "3.12"),
    ("-p3.12", None),
    ("--python-platform", "x86_64-unknown-linux-gnu"),
    ("--python-platform=x86_64-unknown-linux-gnu", None),
    ("--managed-python", None),
    ("--no-managed-python", None),
    ("--no-python-downloads", None),
    ("--index", "https://example.invalid/simple"),
    ("--index=https://example.invalid/simple", None),
    ("--default-index", "https://example.invalid/simple"),
    ("--default-index=https://example.invalid/simple", None),
    ("--index-url", "https://example.invalid/simple"),
    ("--extra-index-url=https://example.invalid/simple", None),
    ("-i", "https://example.invalid/simple"),
    ("-ihttps://example.invalid/simple", None),
    ("--find-links", "wheels"),
    ("--find-links=wheels", None),
    ("-f", "wheels"),
    ("-fwheels", None),
    ("--no-index", None),
    ("--index-strategy", "first-index"),
    ("--keyring-provider=disabled", None),
])
def test_uv_run_rejects_unbound_python_and_index_selectors(tmp_path, option, value):
    from autoresearch.experiments.repo_improve.agentic_improve import _execution_source_paths

    clone = _clone_with_common_cli_sources(tmp_path)
    repo = tmp_path / "repo-cli"
    clone("https://x/repo", str(repo))
    argv = ["uv", "run", "--project", ".", option]
    if value is not None:
        argv.append(value)
    argv.append("bench-cli")

    with pytest.raises(
        ValueError, match="uv environment/dependency selector is not host-verifiable",
    ):
        _execution_source_paths(str(repo), {"argv": argv, "cwd": "."})


def _clone_with_structured_control_files(tmp_path):
    clone, _commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    (source / "controls.json").write_text(
        '{"training":{"seed":17}}\n', encoding="utf-8")
    (source / "controls.toml").write_text(
        "[training]\nprecision='bf16'\n", encoding="utf-8")
    (source / "controls.py").write_text("MODEL_SCALE = 125_000_000\n", encoding="utf-8")
    subprocess.run([
        "git", "-C", str(source), "add", "controls.json", "controls.toml", "controls.py",
    ], check=True)
    subprocess.run([
        "git", "-C", str(source), "commit", "-q", "-m", "structured controls",
    ], check=True)
    return clone


def _structured_control_bindings():
    return [
        {"name": "seed", "path": "controls.json", "format": "json",
         "json_pointer": "/training/seed"},
        {"name": "precision", "path": "controls.toml", "format": "toml",
         "toml_key": "training.precision"},
        {"name": "model_scale", "path": "controls.py", "format": "python_literal",
         "python_literal_name": "MODEL_SCALE"},
    ]


def test_host_attests_structured_control_bindings_and_config_hashes(tmp_path):
    clone = _clone_with_structured_control_files(tmp_path)
    contract = _v2_benchmark_contract()
    contract["controls"]["control_bindings"] = _structured_control_bindings()
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "agent prose is not evidence",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is True
    actual = res["host_attestation"]["actual_controls"]
    assert actual["execution"] == {
        "argv": ["python", "train.py"], "cwd": ".", "source_paths": ["train.py"],
    }
    assert {binding["name"]: binding["value"] for binding in actual["bindings"]} == {
        "seed": 17, "precision": "bf16", "model_scale": 125_000_000,
    }
    assert set(actual["immutable_file_hashes"]) >= {
        "controls.json", "controls.toml", "controls.py",
    }
    assert "does not observe arbitrary internal runtime state" in actual["caveat"]


def test_unextractable_declared_control_binding_fails_closed(tmp_path):
    clone = _clone_with_structured_control_files(tmp_path)
    contract = _v2_benchmark_contract()
    contract["controls"]["control_bindings"] = [{
        "name": "seed", "path": "controls.json", "format": "json",
        "json_pointer": "/training/missing",
    }]
    be = _FakeBackend(result={
        "metric": "val_loss", "baseline": 1.8, "method": None,
        "lower_is_better": True, "config": "seed=made-up",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    })

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="baseline",
    )

    assert res["baseline"].valid is False
    assert "control binding seed could not be extracted" in res["_host_measurement_error"]
    assert be.executed == []


def test_agent_fixed_config_and_manifest_prose_cannot_override_host_controls(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    frozen_attestation = _frozen_host_attestation(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "invented controls",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": {"invented": True},
        },
    }, commands=("printf 'method change\\n' > README.md",))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": frozen_attestation,
        },
    )

    assert res["candidate"].valid is True
    assert res["candidate"].metric_value == 1.7
    assert res["config"] == "steps=1000"
    assert res["host_attestation"]["actual_controls"] == frozen_attestation["actual_controls"]


@pytest.mark.parametrize("filename,content", [
    ("helper.py", "VALUE = 1\n"),
    ("candidate-config.json", '{"batch":999}\n'),
])
def test_method_rejects_undeclared_untracked_code_or_config(
    tmp_path, filename, content,
):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    command = (
        "printf 'method change\\n' > README.md && "
        f"printf {shlex.quote(content)} > {shlex.quote(filename)}"
    )
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, commands=(command,))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert res["diagnosis"] == "candidate_contract_violation"
    assert filename in res["error"]


def test_method_allows_declared_untracked_treatment_and_output_paths(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    contract["controls"]["treatment"]["allowed_paths"].append("helper.py")
    fingerprint = contract_fingerprint(contract)
    frozen_attestation = _frozen_host_attestation(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {
            "contract_fingerprint": fingerprint, "controls": contract["controls"],
        },
    }, commands=(
        "printf 'method change\\n' > README.md && printf 'VALUE = 1\\n' > helper.py",
    ))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": frozen_attestation,
        },
    )

    assert res["candidate"].valid is True
    assert res["host_attestation"]["untracked_files"]["added_allowed"] == [
        "ar_result.json", "helper.py", "metrics.json",
    ]


def test_method_rejects_undeclared_added_binary_input(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint},
    }, commands=(
        "printf 'method change\\n' > README.md && printf weights > weights.bin",
    ))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert "weights.bin" in res["error"]


def test_method_rejects_modification_of_preexisting_undeclared_untracked_evidence(tmp_path):
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    base_clone, commit = _local_git_clone(tmp_path)

    def clone(repo_url, dest):
        base_clone(repo_url, dest)
        Path(dest, "weights.bin").write_bytes(b"original weights")

    contract = _v2_benchmark_contract()
    fingerprint = contract_fingerprint(contract)
    be = _ExecutingAttemptBackend(result={
        "metric": "val_loss", "baseline": None, "method": 1.7,
        "lower_is_better": True, "config": "steps=1000",
        "run_manifest": {"contract_fingerprint": fingerprint},
    }, commands=(
        "printf 'method change\\n' > README.md && printf tampered > weights.bin",
    ))

    res = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=be, clone_fn=clone, execution_phase="method", frozen_baseline={
            "metric_value": 1.8, "metric_name": "val_loss", "lower_is_better": True,
            "fixed_config": "steps=1000", "repo_commit": commit,
            "benchmark_contract": contract, "contract_fingerprint": fingerprint,
            "host_attestation": _frozen_host_attestation(contract),
        },
    )

    assert res["candidate"].valid is False
    assert "weights.bin" in res["error"]


@pytest.mark.parametrize("content", [None, "{broken", "[]", "null", '{"method": null}'])
def test_result_completion_request_only_repairs_missing_protocol(tmp_path, content):
    from autoresearch.experiments.repo_improve.agentic_improve import _result_completion_request
    path = tmp_path / "ar_result.json"
    if content is not None:
        path.write_text(content)
    request = _result_completion_request(str(path))
    if content == '{"method": null}':
        assert request is None  # A structured negative result needs no retry.
    else:
        assert str(path) in request
        assert "Do not launch" in request
        assert "null" in request
        assert "existing" in request


def test_completion_requests_read_access_without_rewriting_result(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    path = tmp_path / "ar_result.json"
    content = '{"baseline": 9, "method": 4, "candidate_valid": true}'
    path.write_text(content)
    with monkeypatch.context() as patch:
        def denied(*args, **kwargs):
            raise PermissionError("container-owned result is not host-readable")
        patch.setattr(module, "open", denied, raising=False)
        request = module._result_completion_request(str(path))
    assert "RESULT FILE NOT READABLE" in request
    assert str(path) in request
    assert "read permission" in request
    assert "Do not rewrite" in request
    assert "Do not launch" in request
    assert "recursive" in request
    assert path.read_text() == content
    assert module._result_completion_request(str(path)) is None


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_completion_resumes_unstarted_task_instead_of_forcing_empty_handoff(tmp_path, monkeypatch, phase):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    monkeypatch.setattr(module, "_attempt_records", lambda _: {})
    path = tmp_path / "ar_result.json"
    request = module._result_completion_request(
        str(path), execution_phase=phase, attempt_log="host.jsonl")
    assert "TASK EXECUTION INCOMPLETE" in request
    assert "original" in request and "remaining" in request
    assert "ar-attempt" in request
    assert "do not duplicate" in request
    assert "Do not launch training" not in request
    assert not path.exists()  # Feedback only; never fabricate a result.

    path.write_text('{"baseline": null, "method": null, "notes": "Dependency unavailable"}')
    assert module._result_completion_request(
        str(path), execution_phase=phase, attempt_log="host.jsonl") is None


@pytest.mark.parametrize("record", [{"phase": "method", "launched": True},
                                    {"phase": "method", "rc": 1},
                                    {"phase": "method", "rc": 0}, {"rc": 0}])
def test_missing_result_with_existing_attempt_keeps_serialization_only(tmp_path, monkeypatch, record):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"run": record})
    request = module._result_completion_request(
        str(tmp_path / "ar_result.json"), execution_phase="method", attempt_log="host.jsonl")
    assert "RESULT HANDOFF INCOMPLETE" in request
    assert "Do not launch training" in request


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_missing_result_after_control_probe_keeps_assigned_execution_available(tmp_path, phase):
    from autoresearch.experiments.repo_improve import agentic_improve as module

    repo = tmp_path / "candidate"
    repo.mkdir()
    path = repo / "ar_result.json"
    ledger = tmp_path / "attempts.jsonl"
    control = {"attempt_id": "control-only", "phase": phase,
               "cwd": str(tmp_path / "parent-control")}
    events = [{**control, "event": "launch"}, {**control, "event": "finish", "rc": 1}]
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    check = lambda: module._result_completion_request(
        str(path), execution_phase=phase, attempt_log=str(ledger))
    assert check().startswith("TASK EXECUTION INCOMPLETE:")
    assert "Do not launch training" not in check()
    assert not path.exists()  # Continue the original assignment; do not fabricate its result.

    candidate = {"attempt_id": "candidate-done", "phase": phase, "cwd": str(repo)}
    events.extend([{**candidate, "event": "launch"},
                   {**candidate, "event": "finish", "rc": 1}])
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert check().startswith("RESULT HANDOFF INCOMPLETE:")
    assert "Do not launch training" in check()  # Never replay a completed candidate.


@pytest.mark.parametrize("null_result", [False, True])
def test_runner_attempt_cap_waits_for_evidence_handoff(tmp_path, monkeypatch, null_result):
    from autoresearch.experiments.repo_improve import agentic_improve as module

    monkeypatch.setattr(module, "_variant_cap", lambda: 1)
    checks = []

    class CappedBackend(_FakeBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None,
                           completion_check=None, stop_predicate=None):
            result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            path = Path(self._mounts[0]) / "repo/ar_result.json"
            payload = json.loads(path.read_text())
            original = dict(payload)
            if null_result:
                payload["method"] = None
                payload["candidate_valid"] = False
                path.write_text(json.dumps(payload))
                assert completion_check() is None
                assert stop_predicate()  # A genuine failure needs no new experiment.
            else:
                payload["attempt_id"] = "stale-reference"
                path.write_text(json.dumps(payload))
                assert "RESULT EVIDENCE HANDOFF INCOMPLETE" in completion_check()
                assert not stop_predicate()  # Keep the same sandbox for citation repair.
                assert json.loads(path.read_text()) == payload
                path.write_text(json.dumps(original))
                assert completion_check() is None
                assert stop_predicate()
            assert len(self.attempt_executed) == 1  # Repair did not repeat evaluation.
            checks.append(True)
            return result

    backend = CappedBackend(result={
        "metric": {"name": "val_loss", "unit": "nats", "lower_is_better": True},
        "method": 1.6, "baseline": None, "lower_is_better": True})
    run_agentic_improve(
        "lower val loss", repo_url="https://x/r", workroot=str(tmp_path),
        backend=backend, clone_fn=_fake_clone, execution_phase="method",
        frozen_baseline={"metric_value": 1.8}, enforce_fairness=False)
    assert checks == [True]
    assert backend.closed


def test_runner_supplies_completion_check_without_restarting_experiment(tmp_path):
    checks = []
    class CompletionBackend(_FakeBackend):
        def run_agent_task(self, handle, task, *, llm=None, timeout=None,
                           completion_check=None):
            result = super().run_agent_task(handle, task, llm=llm, timeout=timeout)
            assert completion_check is not None
            assert "ar-attempt records only its child command's stdout" in task
            assert "ar-attempt bash run_and_export.sh" in task
            assert "Finish every candidate source edit" in task
            assert "do not edit or stage repository files" in task
            assert "Smoke-test the result serializer before expensive training" in task
            assert "json.dumps" in task and "JSON true/false/null are not Python literals" in task
            assert "Save resumable progress periodically during long experiments" in task
            assert "Partial progress is not a passing final measurement" in task
            request = completion_check()
            assert "ar_result.json" in request
            path = Path(self._mounts[0]) / "repo/ar_result.json"
            path.write_text(json.dumps({"baseline": None, "method": None,
                                       "notes": "No experiment completed."}))
            assert completion_check() is None
            checks.append(True)
            return result
    backend = CompletionBackend(write_result=False, record_attempt=False)
    result = run_agentic_improve(
        "q", repo_url="https://x/r", workroot=str(tmp_path),
        backend=backend, clone_fn=_fake_clone)
    assert result["candidate"].metric_value is None
    assert result["candidate"].valid is False
    assert result["verdict"]["improved"] is False
    assert backend.closed
    assert checks == [True]


@pytest.mark.parametrize("saved_null", [False, True])
def test_completion_keeps_registered_background_attempt_alive(tmp_path, saved_null):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    path = tmp_path / "ar_result.json"
    if saved_null:
        path.write_text(json.dumps({"method": None, "candidate_valid": False}))
    ledger = tmp_path / "attempts.jsonl"
    launch = {"event": "launch", "attempt_id": "still-training", "phase": "method"}
    ledger.write_text(json.dumps(launch) + "\n")
    check = lambda: module._result_completion_request(
        str(path), execution_phase="method", attempt_log=str(ledger))
    assert check().startswith("TASK EXECUTION RUNNING:")
    # A completed failure is evidence, not permission to wait or start another experiment.
    with ledger.open("a") as output:
        output.write(json.dumps({**launch, "event": "finish", "rc": 1}) + "\n")
    assert check() is None if saved_null else check().startswith("RESULT HANDOFF INCOMPLETE:")


def test_completion_does_not_wait_on_uncertain_execution(tmp_path):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    ledger = tmp_path / "attempts.jsonl"
    ledger.write_text('\n'.join(json.dumps(item) for item in [
        {"event": "launch", "attempt_id": "lost", "phase": "method"},
        {"event": "finish", "attempt_id": "lost", "phase": "method",
         "execution_uncertain": True}]) + '\n')
    assert module._result_completion_request(
        str(tmp_path / "ar_result.json"), execution_phase="method",
        attempt_log=str(ledger)).startswith("RESULT HANDOFF INCOMPLETE:")


@pytest.mark.parametrize("alive", [False, True])
@pytest.mark.parametrize("saved_null", [False, True])
def test_completion_checks_pending_attempt_authority_liveness(tmp_path, alive, saved_null):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve import agentic_improve as module
    path = tmp_path / "ar_result.json"
    if saved_null:
        path.write_text('{"method": null, "candidate_valid": false}')
    ledger = tmp_path / "attempts.jsonl"
    original = json.dumps({"event": "launch", "attempt_id": "pending",
                           "phase": "method"}) + "\n"
    ledger.write_text(original)
    authority = SimpleNamespace(_thread=SimpleNamespace(is_alive=lambda: alive))
    def check():
        return module._result_completion_request(
            str(path), execution_phase="method", attempt_log=str(ledger),
            attempt_authority=authority)
    if alive:
        assert check().startswith("TASK EXECUTION RUNNING:")
    else:
        with pytest.raises(InfraError, match="attempt authority stopped.*pending"):
            check()
    assert ledger.read_text() == original
    assert path.exists() is saved_null


def test_completion_checks_measurement_binding_before_sandbox_close(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    records = {}
    for aid, value in (("train", 0.7), ("test", 0.6)):
        records[aid] = dict(attempt_id=aid, phase="baseline", launched=True,
                            finished=True, rc=0, **save_output(
                                tmp_path, aid, json.dumps({"baseline": value,
                                                          "baseline_valid": True}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: records)
    path = tmp_path / "ar_result.json"
    payload = {"baseline": 0.6, "method": None, "baseline_valid": True,
               "metric": {"name": "quality", "unit": "ratio"}}
    path.write_text(json.dumps(payload))
    check = lambda: module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl")
    request = check()
    assert "multiple successful baseline attempts" in request
    assert '"attempt_id": "test", "value": 0.6, "valid": true' in request
    assert '"attempt_id": "train", "value": 0.7, "valid": true' in request
    assert "TOP-LEVEL attempt_id" in request and "os.replace atomically" in request
    assert json.loads(path.read_text()) == payload  # Only feedback, never host-fabricated ar_result.
    payload["attempt_id"] = "test"
    path.write_text(json.dumps(payload))
    assert check() is None
    payload["attempt_id"] = "train"
    path.write_text(json.dumps(payload))
    assert "disagrees" in check()  # Never silently bind the latest/best attempt.


@pytest.mark.parametrize("primary", [{"baseline": 0.7, "baseline_valid": False}, {}])
def test_completion_shows_host_read_output_fields_without_exposing_private_files(
        tmp_path, monkeypatch, primary):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    output = {**primary, "measurements": {"count/zero~": 0, "samples": [False, 2]},
              "official_reward": 918273645, "notes": "not a measurement"}
    raw = json.dumps(output)
    record = dict(attempt_id="run", phase="baseline", launched=True, finished=True,
                  rc=0, **save_output(tmp_path, "run", raw, ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"run": record})
    path = tmp_path / "ar_result.json"
    submitted = json.dumps({"baseline": 0.6, "attempt_id": "run",
                            "metric": {"name": "quality", "unit": "ratio"}})
    path.write_text(submitted)

    request = module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl")

    assert '"output_fields"' in request
    assert '"/measurements/count~1zero~0": 0' in request
    assert '"/measurements/samples/0": false' in request
    assert '"/measurements/samples/1": 2' in request
    assert "host-only" in request
    assert "918273645" not in request and "not a measurement" not in request
    assert path.read_text() == submitted
    assert Path(record["stdout_path"]).read_text() == raw


def test_completion_does_not_summarize_changed_host_output(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    record = dict(attempt_id="run", phase="baseline", launched=True, finished=True,
                  rc=0, **save_output(tmp_path, "run", '{"baseline":0.7}', ""))
    Path(record["stdout_path"]).write_text('{"baseline":0.8}')
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"run": record})
    path = tmp_path / "ar_result.json"
    path.write_text('{"baseline":0.6,"attempt_id":"run"}')
    request = module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl")
    assert "saved execution output content changed" in request
    assert '"output_fields"' not in request


def test_completion_repairs_manifest_metric_name_without_repeating_measurement(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    record = dict(attempt_id="measured", phase="baseline", launched=True,
                  finished=True, rc=0, **save_output(
                      tmp_path, "measured", json.dumps({
                          "baseline": 0.6, "baseline_valid": True}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"measured": record})
    path = tmp_path / "ar_result.json"
    payload = {"baseline": 0.6, "baseline_valid": True, "method": None,
               "attempt_id": "measured", "metric": {
                   "name": "wall_clock_runtime_seconds", "unit": "seconds",
                   "lower_is_better": True}}
    path.write_text(json.dumps(payload))
    check = lambda: module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl",
        evaluation={"metric_name": "runtime_seconds", "lower_is_better": True})

    request = check()
    assert "expected {'metric_name': 'runtime_seconds'" in request
    assert "reported name='wall_clock_runtime_seconds'" in request
    assert "do not change the measurement or rerun the evaluator" in request
    assert json.loads(path.read_text()) == payload
    payload["metric"]["name"] = "runtime_seconds"
    path.write_text(json.dumps(payload))
    assert check() is None


@pytest.mark.parametrize("defect", ["stale_code", "wrong_cwd", "changed_during_run"])
def test_completion_checks_current_source_before_reusing_saved_measurement(tmp_path, monkeypatch, defect):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    commit = module._repo_commit(str(repo))
    fingerprint = module._tracked_repo_state(str(repo))["tracked_diff_hash"]
    records = {}
    for aid in ("attempt-old", "attempt-current"):
        records[aid] = dict(
            attempt_id=aid, phase="method", launched=True, finished=True, rc=0,
            cwd=str(repo), repo_commit=commit, repo_commit_end=commit,
            tracked_diff_hash_start=fingerprint, tracked_diff_hash_end=fingerprint,
            **save_output(tmp_path, aid, json.dumps({"method": 0.6, "candidate_valid": True}), ""))
    if defect == "wrong_cwd":
        records["attempt-old"]["cwd"] = str(tmp_path / "control")
    elif defect == "stale_code":
        records["attempt-old"]["tracked_diff_hash_start"] = "older-source"
        records["attempt-old"]["tracked_diff_hash_end"] = "older-source"
    else:
        records["attempt-old"]["tracked_diff_hash_end"] = "changed-during-run"
    monkeypatch.setattr(module, "_attempt_records", lambda _: records)
    path = repo / "ar_result.json"
    payload = {"method": 0.6, "candidate_valid": True, "attempt_id": "attempt-old",
               "metric": {"name": "quality", "unit": "ratio"}}
    path.write_text(json.dumps(payload))
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    check = lambda: module._result_completion_request(
        str(path), execution_phase="method", attempt_log="host.jsonl")

    request = check()
    assert request and "RESULT EVIDENCE HANDOFF INCOMPLETE" in request
    assert "outside the current repository" in request if defect == "wrong_cwd" else "fingerprint" in request
    assert "do not retrain" in request
    assert json.loads(path.read_text()) == payload  # No implicit choice of the matching receipt.
    assert {p: p.read_bytes() for p in before} == before
    payload["attempt_id"] = "attempt-current"
    path.write_text(json.dumps(payload))
    assert check() is None  # Correcting the citation needs no additional execution.
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_completion_checks_selected_checkpoint_artifacts_before_close(tmp_path, monkeypatch, phase):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    repo = tmp_path / "repo"
    _fake_clone("unused", str(repo))
    selected_artifact = tmp_path / "selected.bin"
    selected_artifact.write_bytes(b"measured candidate runtime artifact")
    checkpoint = snapshot_checkpoint(str(repo), "attempt-selected", [str(selected_artifact)])
    commit = module._repo_commit(str(repo))
    fingerprint = module._tracked_repo_state(str(repo))["tracked_diff_hash"]
    field, valid = ("baseline", "baseline_valid") if phase == "baseline" else ("method", "candidate_valid")
    record = dict(
        attempt_id="attempt-selected", phase=phase, launched=True, finished=True, rc=0,
        cwd=str(repo), repo_commit=commit, repo_commit_end=commit,
        tracked_diff_hash_start=fingerprint, tracked_diff_hash_end=fingerprint,
        checkpoint=checkpoint,
        **save_output(tmp_path, "attempt-selected", json.dumps({field: 0.6, valid: True}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"attempt-selected": record})
    later_artifact = tmp_path / "later-attempt.json"
    later_artifact.write_text('{"method":0.7}')
    path = repo / "ar_result.json"
    payload = {field: 0.6, valid: True, "attempt_id": "attempt-selected",
               "metric": {"name": "quality", "unit": "ratio"},
               "artifact_paths": [str(selected_artifact), str(later_artifact)]}
    path.write_text(json.dumps(payload))
    original_output = Path(record["stdout_path"]).read_bytes()
    check = lambda: module._result_completion_request(
        str(path), execution_phase=phase, attempt_log="host.jsonl")

    request = check()
    assert request and "RESULT EVIDENCE HANDOFF INCOMPLETE" in request
    assert "declared runtime artifact is absent from measured checkpoint" in request
    assert str(later_artifact) in request
    assert json.loads(path.read_text()) == payload
    payload["artifact_paths"] = [str(selected_artifact)]
    path.write_text(json.dumps(payload))
    assert check() is None  # Repair only the cross-attempt attachment, preserving this measurement.
    assert Path(record["stdout_path"]).read_bytes() == original_output
    assert selected_artifact.read_bytes() == b"measured candidate runtime artifact"


def test_completion_requests_exact_attempt_id_before_sandbox_close(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    record = dict(attempt_id="attempt-123-abcdef", phase="method", launched=True, cwd=str(tmp_path),
                  finished=True, rc=0, **save_output(tmp_path, "actual", json.dumps({
                      "method": 0.6, "candidate_valid": True}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {record["attempt_id"]: record})
    path = tmp_path / "ar_result.json"
    payload = {"method": 0.6, "attempt_id": "attempt-123-abc",
               "metric": {"name": "quality", "unit": "ratio"}}
    path.write_text(json.dumps(payload))

    request = module._result_completion_request(
        str(path), execution_phase="method", attempt_log="host.jsonl")

    assert "attempt-123-abc" in request
    assert "attempt-123-abcdef" in request
    assert "infer identity from a similar ID" in request
    assert json.loads(path.read_text()) == payload
    payload["attempt_id"] = record["attempt_id"]
    path.write_text(json.dumps(payload))
    assert module._result_completion_request(
        str(path), execution_phase="method", attempt_log="host.jsonl") is None


@pytest.mark.parametrize("metric", [None, "quality", {"name": "quality"}])
def test_completion_requests_missing_metric_metadata_before_close(tmp_path, monkeypatch, metric):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    record = dict(attempt_id="measured", phase="method", launched=True, finished=True, cwd=str(tmp_path),
                  rc=0, **save_output(tmp_path, "measured", json.dumps({
                      "method": 0.6, "candidate_valid": True}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"measured": record})
    path = tmp_path / "ar_result.json"
    payload = {"method": 0.6, "attempt_id": "measured", "metric": metric}
    path.write_text(json.dumps(payload))
    check = lambda: module._result_completion_request(
        str(path), execution_phase="method", attempt_log="host.jsonl")

    request = check()
    assert "metric.name" in request and "metric.unit" in request
    assert "do not retrain" in request
    assert json.loads(path.read_text()) == payload
    payload["metric"] = {"name": "quality", "unit": "ratio"}
    path.write_text(json.dumps(payload))
    assert check() is None
    assert record["rc"] == 0


def test_completion_feedback_keeps_indirect_output_paths(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output
    record = dict(attempt_id="attempt-check", phase="method", launched=True, finished=True,
                  rc=0, checkpoint=str(tmp_path / "checkpoint"),
                  **save_output(tmp_path, "probe", '{"measurements":{"property_check":0}}', ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"attempt-check": record})
    path = tmp_path / "ar_result.json"
    request = module._result_completion_request(
        str(path), execution_phase="method", attempt_log="host.jsonl")
    assert record["stdout_path"] in request
    assert record["stdout_sha256"] in request
    assert record["checkpoint"] in request
    assert "indirect-component check" in request
    assert '"value": 0' not in request  # Never promote a check into the task metric.
    assert not path.exists()


def test_completion_feedback_for_missing_stdout_preserves_candidate(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    record = dict(attempt_id="test", phase="baseline", launched=True, finished=True,
                  rc=0, **save_output(tmp_path, "test", "Accuracy: 0.6\n", ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"test": record})
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({"baseline": 0.6, "attempt_id": "test"}))
    request = module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl")
    assert "unambiguous JSON result" in request
    assert '"evidence_error"' in request
    assert "existing checkpoint" in request
    assert "inside the same ar-attempt child command" in request
    assert "not after ar-attempt returns" in request
    assert "do not retrain" in request
    assert record["rc"] == 0  # Feedback does not rewrite execution evidence.
    path.write_text(json.dumps({"baseline": None, "method": None}))
    assert module._result_completion_request(
        str(path), execution_phase="baseline", attempt_log="host.jsonl") is None


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_completion_distinguishes_nested_output_from_missing_execution(tmp_path, monkeypatch, phase):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    record = dict(attempt_id="measured", phase=phase, launched=True, finished=True,
                  rc=0, **save_output(tmp_path, "measured", json.dumps({
                      "task": {"means": {"quality": 0.6}}}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"measured": record})
    path = tmp_path / "ar_result.json"
    payload = {phase: 0.6, "attempt_id": "measured",
               "metric": {"name": "quality", "unit": "ratio"}}
    path.write_text(json.dumps(payload))
    request = module._result_completion_request(
        str(path), execution_phase=phase, attempt_log="host.jsonl")
    assert "already contains JSON but the declared metric/gate pointers are missing" in request
    assert "Editing ar_result.json alone cannot change captured stdout" in request
    assert "Never invent a passing gate" in request
    assert "do not retrain" in request
    assert json.loads(path.read_text()) == payload
    assert record["rc"] == 0


def test_measurement_attempt_rejects_cross_phase_but_accepts_single_baseline():
    from autoresearch.experiments.repo_improve.agentic_improve import _measurement_attempt
    record = {"phase": "baseline", "rc": 0}
    assert _measurement_attempt({"one": record}, None, "baseline") is record
    with pytest.raises(ValueError, match="different execution phase"):
        _measurement_attempt({"one": record}, "one", "method")


@pytest.mark.parametrize("defect,expected", [
    ("component_pointer", "execution output is missing"),
    ("parent_ref", "parent_evidence.ref"),
    ("parent_pointer", "cannot resolve parent_evidence.json_pointer"),
    ("parent_cwd", "control attempt is not the exact dependency parent"),
    ("parent_end", "'tracked_diff_hash_end': 'changed-during-run'"),
])
@pytest.mark.parametrize("indirect", [False, True])
def test_completion_checks_component_bindings_before_closing(tmp_path, monkeypatch, defect, expected, indirect):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    state = {"repo_commit": "abc", "repo_commit_end": "abc",
             "tracked_diff_hash_start": "clean", "tracked_diff_hash_end": "clean",
             "execution_protocol": {}}
    records = {
        "candidate": dict(state, attempt_id="candidate", phase="method", launched=True,
                          finished=True, rc=0, cwd=str(tmp_path), dependency_parent_cwd="/control",
                          dependency_parent_state=state, **save_output(tmp_path, "candidate",
                              json.dumps({"method": None if indirect else 0.6, "candidate_valid": True,
                                          "measurements": {"component": 0.6}}), "")),
        "control": dict(state, attempt_id="control", phase="control", launched=True,
                        finished=True, rc=0, cwd="/control", **save_output(tmp_path, "control",
                            json.dumps({"baseline": 0.7, "baseline_valid": True}), "")),
    }
    monkeypatch.setattr(module, "_attempt_records", lambda _: records)
    monkeypatch.setattr(module, "_repo_commit", lambda _: "abc")
    monkeypatch.setattr(module, "_tracked_repo_state", lambda _: {"tracked_diff_hash": "clean"})
    item = {"component": "component", "type": "indirect_gain" if indirect else "direct_gain", "actual": 0.6,
            "evidence": {"source": "runtime_attempt", "ref": "candidate",
                         "json_pointer": "/measurements/component"},
            "parent_metric": 0.7,
            "parent_evidence": {"source": "runtime_attempt", "ref": "control",
                                "json_pointer": "/baseline"}}
    payload = {"baseline": 0.7, "method": None if indirect else 0.6, "candidate_valid": True,
               "attempt_id": "candidate", "ablation_report": [item],
               "metric": {"name": "quality", "unit": "ratio"}}
    good = json.dumps(payload)
    if defect == "component_pointer":
        item["evidence"]["json_pointer"] = "/measurements/absent"
    elif defect == "parent_ref":
        item["parent_evidence"]["ref"] = "control.json"
    elif defect == "parent_pointer":
        item["parent_evidence"]["json_pointer"] = "/method"
    elif defect == "parent_end":
        records["control"]["tracked_diff_hash_end"] = "changed-during-run"
    else:
        records["control"]["cwd"] = "/candidate"
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps(payload))
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    check = lambda: module._result_completion_request(
        str(path), execution_phase="method", attempt_log="host.jsonl")
    request = check()
    assert request and expected in request
    assert "parent-control" in request and "do not retrain" in request
    assert {p: p.read_bytes() for p in before} == before
    path.write_text(good)
    records["control"]["cwd"] = "/control"
    records["control"]["tracked_diff_hash_end"] = "clean"
    assert check() is None
    if indirect:
        records["candidate"]["rc"] = 1
        assert check() is None  # An honest failed property check does not trigger a rerun.



@pytest.mark.parametrize("direction", ["false", "true", 0, 1])
def test_result_parser_rejects_non_boolean_direction(tmp_path, direction):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({"baseline": 10, "method": 11, "lower_is_better": direction}))
    events = []
    result = _build_result_from_file(str(path), "test", "runtime", True,
                                    lambda *args: events.append(args))
    assert result is None
    assert path.exists()
    assert "JSON boolean" in events[0][1]["error"]



@pytest.mark.parametrize("failed_command", ["python train.py", "python evaluate.py"])
def test_host_measurement_reply_loss_executes_once(tmp_path, failed_command):
    clone, _ = _local_git_clone(tmp_path)
    calls = []
    class LostReply(_FakeBackend):
        def execute_attempt_command(self, handle, cmd, **kwargs):
            if getattr(self, "_agent_finished", False):
                cmd = cmd.removeprefix("env AUTORESEARCH_HOST_NATIVE_EXECUTION=1 ")
                calls.append(cmd)
                if cmd.startswith(failed_command):
                    exc = InfraError("ConnectionResetError after command was submitted")
                    exc.execution_uncertain = True
                    raise exc
            return super().execute_attempt_command(handle, cmd, **kwargs)
    backend = LostReply(result={"metric": "val_loss", "baseline": 99, "method": None,
                               "lower_is_better": True, "measurement": _measurement_spec(),
                               "benchmark_contract": _v2_benchmark_contract()})
    result = run_agentic_improve("lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
                                backend=backend, clone_fn=clone, execution_phase="baseline")
    assert len([cmd for cmd in calls if cmd.startswith(failed_command)]) == 1
    assert result["baseline"].valid is False
    assert "ConnectionResetError" in str(result)


@pytest.mark.parametrize('manifest,expected', [
    ('[agent]\ntimeout_sec = 14400\n[verifier]\ntimeout_sec = 1800\n', 16200),
    ('[agent]\ntimeout_sec = 3600\n', 7200),
    ('[agent]\ntimeout_sec = true\n', 7200),
    ('[agent]\ntimeout_sec = -1\n', 7200),
])
def test_selected_manifest_command_allowance(manifest, expected, monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _per_run_cap

    monkeypatch.setenv('AUTORESEARCH_AGENTIC_PER_RUN_S', '7200')
    contract = {'sources': [{'path': 'task/task.toml', 'content': manifest}]}
    assert _per_run_cap(contract) == expected


@pytest.mark.parametrize('manifest,expected', [
    ('[agent]\ntimeout_sec = 7200\n[verifier]\ntimeout_sec = 900\n', 900),
    ('[verifier]\ntimeout_sec = true\n', None),
    ('[verifier]\ntimeout_sec = 0\n', None),
    ('[agent]\ntimeout_sec = 7200\n', None),
])
def test_selected_manifest_verifier_timeout(manifest, expected):
    from autoresearch.experiments.repo_improve.agentic_improve import _verifier_timeout

    contract = {'sources': [{'path': 'task/task.toml', 'content': manifest}]}
    assert _verifier_timeout(contract) == expected


def test_method_prompt_enforces_manifest_verifier_substage_timeout(monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _method_task_prompt

    monkeypatch.setenv('AUTORESEARCH_AGENTIC_PER_RUN_S', '7200')
    contract = {'sources': [{'path': 'task/task.toml', 'content':
        '[agent]\ntimeout_sec=7200\n[verifier]\ntimeout_sec=900\n'}]}
    prompt = _method_task_prompt('q', '/repo', '/result.json', None, {
        'metric_name': 'runtime', 'metric_value': 1.0,
        'repository_contract': contract,
    })

    assert 'VERIFIER SUBSTAGE LIMIT' in prompt
    assert 'timeout --foreground --kill-after=2s 900s <official verifier command>' in prompt
    assert 'outer 8100s ar-attempt cap' in prompt
    assert 'verifier timeout (exit 124) is an invalid attempt' in prompt
    assert '$AUTORESEARCH_VERIFIER_TIMEOUT_S' in prompt


def test_conflicting_selected_manifest_timeouts_keep_configured_cap(monkeypatch):
    from autoresearch.experiments.repo_improve.agentic_improve import _per_run_cap

    monkeypatch.setenv('AUTORESEARCH_AGENTIC_PER_RUN_S', '7200')
    contract = {'sources': [
        {'path': 'one/task.toml', 'content': '[agent]\ntimeout_sec=14400\n'},
        {'path': 'two/task.toml', 'content': '[agent]\ntimeout_sec=3600\n'},
    ]}
    assert _per_run_cap(contract) == 7200


def test_repository_allowance_reaches_attempt_executor_and_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoresearch.experiments.repo_improve import agentic_improve as core

    monkeypatch.setenv('AUTORESEARCH_AGENTIC_PER_RUN_S', '7200')
    repo = tmp_path / 'repo'
    repo.mkdir()
    calls = []

    def execute(handle, command, *, cwd, timeout):
        calls.append((command, cwd, timeout))
        return SimpleNamespace(exit_code=124, stdout='', stderr='timeout')

    authority = core._AttemptAuthority(
        SimpleNamespace(execute_command=execute), None, str(repo), 'method',
        str(tmp_path / 'attempts.jsonl'), str(tmp_path / 'authority.sock'),
        str(tmp_path / 'last_attempt_id'))
    contract = {'sources': [{'path': 'task/task.toml', 'content':
        '[agent]\ntimeout_sec=14400\n[verifier]\ntimeout_sec=1800\n'}]}
    authority.repository_contract = contract
    payload = json.dumps({'argv': ['timeout', '--foreground', '14400s', 'train'],
                          'cwd': str(repo)}).encode() + b'\n'
    try:
        outcome = authority._handle(SimpleNamespace(recv=lambda _: payload))
        assert outcome['rc'] == 124
        assert len(calls) == 1
        command, cwd, timeout = calls[0]
        assert timeout == 16210 and '16200s' in shlex.split(command)
        assert 'AUTORESEARCH_VERIFIER_TIMEOUT_S=1800' in shlex.split(command)
        record = core._attempt_records(authority.log_path)[outcome['attempt_id']]
        assert record['command_timeout_s'] == 16200
        assert record['verifier_timeout_s'] == 1800
        monkeypatch.setenv('AUTORESEARCH_AGENTIC_PER_RUN_S', '1800')
        notes = core._failed_attempt_notes(authority.log_path, 'method')
        assert '"host_command_cap_s": 16200' in notes[0]
        prompt = core._method_task_prompt('q', str(repo), '/result.json', None,
            {'metric_name': 'quality', 'metric_value': 1.0,
             'repository_contract': contract})
        assert '16200s' in prompt
    finally:
        authority._socket.close()
