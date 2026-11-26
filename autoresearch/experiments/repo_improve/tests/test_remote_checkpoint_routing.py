import subprocess

import pytest

from autoresearch.experiments.repo_improve import checkpoint_artifacts as checkpoint


@pytest.fixture
def remote_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTORESEARCH_REPO_REMOTE", raising=False)
    monkeypatch.setenv("AUTORESEARCH_GPU_HOST", "configured-host")
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE_DIR", "/configured/runtime")
    monkeypatch.setenv("AUTORESEARCH_SSH_CONFIG", "/configured/ssh.conf")
    monkeypatch.setenv("AUTORESEARCH_REPO_SPLIT", "1")
    return str(tmp_path / "remote-checkpoint")


@pytest.mark.parametrize("mode", ["auto", "remote"])
def test_remote_checkpoint_uses_current_routing_and_ssh_config(monkeypatch, remote_config, mode):
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", mode)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(checkpoint.subprocess, "run", run)
    assert checkpoint.checkpoint_recoverable(remote_config)
    assert calls[0][:3] == ["ssh", "-F", "/configured/ssh.conf"]
    assert calls[0][-2] == "configured-host"
    assert "validate_checkpoint" in calls[0][-1]
    assert remote_config in calls[0][-1]


def test_remote_checkpoint_rejects_failed_manifest_validation(monkeypatch, remote_config):
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setattr(checkpoint.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1))
    assert not checkpoint.checkpoint_recoverable(remote_config)


def test_remote_checkpoint_uses_checkpoint_operation_timeout(monkeypatch, remote_config):
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_CHECKPOINT_OPERATION_TIMEOUT_S", "777")
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(checkpoint.subprocess, "run", run)
    assert checkpoint.checkpoint_recoverable(remote_config)
    assert calls[0]["timeout"] == 777


@pytest.mark.parametrize("local_only", ["local", "split_disabled", "present_invalid"])
def test_local_only_or_invalid_local_checkpoint_does_not_fall_through_to_ssh(
    monkeypatch, remote_config, local_only,
):
    from pathlib import Path

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    if local_only == "local":
        monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "local")
    elif local_only == "split_disabled":
        monkeypatch.setenv("AUTORESEARCH_REPO_SPLIT", "0")
    else:
        Path(remote_config).mkdir()

    def run(*args, **kwargs):
        pytest.fail("local-only or invalid local checkpoint must not use SSH")

    monkeypatch.setattr(checkpoint.subprocess, "run", run)
    assert not checkpoint.checkpoint_recoverable(remote_config)


@pytest.mark.parametrize("damage", [None, "unfinished", "uncertain", "stdout", "stderr", "missing"])
def test_review_recovery_preserves_failed_control_without_qualifying_it(tmp_path, damage):
    from pathlib import Path
    from autoresearch.experiments.repo_improve.runtime_evidence import read_output, save_output

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solve.py").write_text("candidate = True\n")
    for args in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    saved = checkpoint.snapshot_checkpoint(str(repo), "measured")
    candidate = dict(launched=True, finished=True, rc=0,
                     **save_output(tmp_path, "candidate", '{"method":0.9}', ""))
    control = dict(launched=True, finished=True, rc=126,
                   **save_output(tmp_path, "control", "control failed\n", "permission denied\n"))
    if damage == "unfinished":
        control["finished"] = False
    elif damage == "uncertain":
        control["execution_uncertain"] = True
    elif damage in {"stdout", "stderr"}:
        Path(control[damage + "_path"]).write_text("changed\n")
    elif damage == "missing":
        Path(control["stderr_path"]).unlink()

    assert checkpoint.checkpoint_recoverable(saved, runtime_receipts=[candidate, control]) is (damage is None)
    # Recovering the review never turns a failed control into numeric evidence.
    with pytest.raises(ValueError, match="successfully finished"):
        read_output(control)
