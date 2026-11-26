"""The broker executes argv, not the calling interactive shell's environment."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import signal
from types import SimpleNamespace
import pytest

from autoresearch.experiments.repo_improve.agentic_improve import (
    _AttemptAuthority, _attempt_records, _baseline_task_prompt,
    _install_attempt_guard, _method_task_prompt, _task_prompt,
)


def test_attempt_authority_stop_before_start_is_safe(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    bindir, log = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "result.json"))
    authority = _AttemptAuthority(
        SimpleNamespace(), None, str(repo), "method", log,
        str(tmp_path / "attempt.sock"),
        str(Path(bindir).parent / "last_attempt_id"),
    )

    authority.stop()


def test_attempt_handoff_precedes_result_artifact_discovery(tmp_path, monkeypatch):
    """A private atomic result file must not lose its runtime declarations."""
    import builtins
    from autoresearch.experiments.repo_improve import agentic_improve as core
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import validate_checkpoint

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README.md").write_text("starter\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "commit", "-qm", "starter",
    ], check=True)
    bindir, log = _install_attempt_guard(
        str(tmp_path), str(repo), str(repo / "ar_result.json"))

    def execute(_handle, command, *, cwd, timeout):
        completed = subprocess.run(
            command, shell=True, cwd=cwd, timeout=timeout,
            capture_output=True, text=True)
        return SimpleNamespace(exit_code=completed.returncode,
                               stdout=completed.stdout, stderr=completed.stderr)

    prepared = []
    monkeypatch.setattr(
        core, "_prepare_checkpoint_handoff",
        lambda _backend, _handle, path, _emit: prepared.append(path))
    real_open = builtins.open
    result_path = repo / "ar_result.json"

    def private_result_open(path, *args, **kwargs):
        if os.fspath(path) == str(result_path) and not prepared:
            raise PermissionError("sandbox UID owns atomic result")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", private_result_open)
    authority = _AttemptAuthority(
        SimpleNamespace(execute_command=execute), None, str(repo), "method", log,
        str(tmp_path / "attempt.sock"), str(Path(bindir).parent / "last_attempt_id"))
    code = (
        "import json,os; "
        "open('runtime.bin','wb').write(b'model'); "
        "json.dump({'attempt_id':os.environ['AUTORESEARCH_ATTEMPT_ID'],"
        "'artifact_paths':['runtime.bin']},open('ar_result.json','w'))"
    )
    payload = json.dumps({"argv": [sys.executable, "-c", code],
                          "cwd": str(repo)}).encode() + b"\n"
    try:
        outcome = authority._handle(SimpleNamespace(recv=lambda _: payload))
        record = _attempt_records(log)[outcome["attempt_id"]]
        manifest = validate_checkpoint(record["checkpoint"])
        assert manifest["runtime_paths"] == ["runtime.bin"]
        assert prepared == [str(repo)]
    finally:
        authority._socket.close()


def test_brokered_attempt_receives_explicit_env_not_terminal_exports(tmp_path_factory):
    # Keep the real Unix socket path below its platform length limit.
    root = tmp_path_factory.mktemp("env")
    repo = root / "repo"
    repo.mkdir()
    bindir, log = _install_attempt_guard(str(root), str(repo), str(repo / "result.json"))

    def execute(_handle, command, *, cwd, timeout):
        run = subprocess.run(shlex.split(command), cwd=cwd, timeout=timeout,
                             env={"PATH": os.defpath}, capture_output=True, text=True)
        return SimpleNamespace(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)

    socket_path = str(root / "attempt.sock")
    authority = _AttemptAuthority(SimpleNamespace(execute_command=execute), None,
                                  str(repo), "method", log, socket_path,
                                  str(Path(bindir).parent / "last_attempt_id"))
    probe = [sys.executable, "-c",
             "import os,json; print(json.dumps({'value':os.getenv('CAP_PROBE_VALUE')}))"]
    client_env = {"PATH": os.defpath, "CAP_PROBE_VALUE": "terminal-only",
                  "AUTORESEARCH_ATTEMPT_SOCKET": socket_path}
    authority.start()
    try:
        outputs = []
        for command in (probe, ["env", "CAP_PROBE_VALUE=explicit-value", *probe]):
            run = subprocess.run([sys.executable, str(Path(bindir) / "ar-attempt"), *command],
                                 cwd=repo, env=client_env, capture_output=True, text=True,
                                 timeout=15)
            assert run.returncode == 0, run.stderr
            outputs.append(json.loads(run.stdout)["value"])
        assert outputs == [None, "explicit-value"]
        records = _attempt_records(log)
        assert len(records) == 2
        assert all(record["rc"] == 0 for record in records.values())
        assert any(record["argv"][:2] == ["env", "CAP_PROBE_VALUE=explicit-value"]
                   for record in records.values())
    finally:
        authority.stop()


@pytest.mark.parametrize("nested_timeout", [False, True])
def test_broker_deadline_stops_command_before_transport_timeout(tmp_path_factory, monkeypatch, nested_timeout):
    from autoresearch.experiments.repo_improve import agentic_improve as core

    root = tmp_path_factory.mktemp("deadline")
    repo = root / "repo"
    repo.mkdir()
    bindir, log = _install_attempt_guard(str(root), str(repo), str(repo / "result.json"))
    pid_file = root / "command.pid"
    observed = []

    def execute(_handle, command, *, cwd, timeout):
        child = subprocess.Popen(shlex.split(command), cwd=cwd, start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = child.communicate(timeout=5)
            observed.append(child.returncode)
            return SimpleNamespace(exit_code=child.returncode, stdout=stdout, stderr=stderr)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=3)

    monkeypatch.setattr(core, "_per_run_cap", lambda _contract=None: 1)
    authority = _AttemptAuthority(SimpleNamespace(execute_command=execute), None, str(repo),
                                  "method", log, str(root / "attempt.sock"),
                                  str(Path(bindir).parent / "last_attempt_id"))
    argv = ["sh", "-c", f"echo $$ > {shlex.quote(str(pid_file))}; exec sleep 60"]
    if nested_timeout:
        argv = [str(Path(bindir) / "timeout"), "60s", *argv]
    payload = json.dumps({"argv": argv, "cwd": str(repo)}).encode() + b"\n"
    authority.start()
    try:
        result = authority._handle(SimpleNamespace(recv=lambda _: payload))
        assert observed == [124], result
        assert result["rc"] == 124
        pid = int(pid_file.read_text())
        assert not Path(f"/proc/{pid}").exists(), "broker returned with live timed-out work"
        records = _attempt_records(log)
        assert len(records) == 1
        assert next(iter(records.values()))["argv"] == argv
        assert next(iter(records.values()))["rc"] == 124
    finally:
        authority.stop()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_all_run_prompts_explain_broker_environment_and_interpreter():
    prompts = [
        _task_prompt("task", "/repo", "/repo/result.json", 600, "score", False),
        _baseline_task_prompt("task", "/repo", "/repo/result.json", "score", False),
        _method_task_prompt("task", "/repo", "/repo/result.json", None,
                            {"metric_name": "score", "metric_value": 1,
                             "lower_is_better": False}),
    ]
    for prompt in prompts:
        assert "ar-attempt env NAME=VALUE" in prompt
        assert "does not inherit temporary exports" in prompt
        assert "absolute interpreter path" in prompt
        assert "top-level `baseline_drift`" in prompt
        assert "Do not bury this recovery request only inside notes" in prompt
        assert "request itself is not a verified finding" in prompt


def test_recovered_control_alias_is_canonical_and_does_not_expand_authorization(tmp_path_factory):
    from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements

    root = tmp_path_factory.mktemp("alias")
    real = root / "original"
    repo = real / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "init"], check=True)
    control = real / "control"
    subprocess.run(["git", "clone", "-q", str(repo), str(control)], check=True)
    for path in (repo, control):
        (path / "nested").mkdir()
    alias = root / "recovered"
    alias.symlink_to(real, target_is_directory=True)
    outside = real / "control-other"
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    bindir, log = _install_attempt_guard(str(root), str(repo), str(repo / "result.json"))

    def execute(_handle, command, *, cwd, timeout):
        run = subprocess.run(shlex.split(command), cwd=cwd, timeout=timeout,
                             env={"PATH": os.defpath}, capture_output=True, text=True)
        return SimpleNamespace(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)

    authority = _AttemptAuthority(SimpleNamespace(execute_command=execute), None,
                                  str(alias / "repo"), "method", log,
                                  str(root / "attempt.sock"), str(Path(bindir).parent / "last_attempt_id"))
    authority.control_path = str(alias / "control")
    authority.start()
    try:
        env = {"PATH": os.defpath, "AUTORESEARCH_ATTEMPT_SOCKET": authority.socket_path}
        command = [sys.executable, str(Path(bindir) / "ar-attempt"), sys.executable,
                   "-c", "import json; print(json.dumps({'method': 2, 'measurements': {'gain': 2}}))"]
        for cwd in (alias / "control/nested", alias / "repo/nested"):
            run = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=15)
            assert run.returncode == 0, run.stderr
        records = _attempt_records(log)
        parent, candidate = records.values()
        assert parent["cwd"] == candidate["dependency_parent_cwd"] == str(control / "nested")
        bound = bind_component_measurements([{
            "component": "gain", "actual": 2, "parent_metric": 2,
            "evidence": {"source": "runtime_attempt", "ref": candidate["attempt_id"]},
            "parent_evidence": {"source": "runtime_attempt", "ref": parent["attempt_id"]},
        }], candidate, records)
        assert bound[0]["parent_evidence"]["host_bound"] is True
        for cwd in (outside, repo / "escape"):
            run = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=15)
            assert run.returncode == 126
            assert "outside the authorized checkouts" in run.stderr
        assert len(_attempt_records(log)) == 2
    finally:
        authority.stop()
