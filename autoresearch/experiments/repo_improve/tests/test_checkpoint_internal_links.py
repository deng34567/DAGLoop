import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from autoresearch.experiments.repo_improve import checkpoint_artifacts as checkpoint


def make_repo(tmp_path, target):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.toml").write_text("threshold = 1\n")
    if target is not None:
        (repo / "alias.toml").symlink_to(target)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Offline Test",
        "-c", "user.email=offline@example.invalid", "commit", "-qm", "fixture"
    ], check=True)
    return repo


def test_repeated_snapshot_reuses_identical_valid_checkpoint(tmp_path):
    repo = make_repo(tmp_path, "config.toml")
    first = checkpoint.snapshot_checkpoint(str(repo), "same-attempt")
    manifest = (Path(first) / checkpoint.MANIFEST).read_bytes()
    assert checkpoint.snapshot_checkpoint(str(repo), "same-attempt") == first
    assert (Path(first) / checkpoint.MANIFEST).read_bytes() == manifest
    checkpoint.validate_checkpoint(first)


@pytest.mark.parametrize("declared", ["inside", "pattern", "outside", "escape", "unsealed", "ambiguous"])
def test_absolute_source_artifact_requires_bound_root_and_sealed_inventory(tmp_path, declared):
    repo = make_repo(tmp_path, None)
    (repo / "config-other.toml").write_text("threshold = 2\n")
    saved = checkpoint.snapshot_checkpoint(str(repo), "attempt-source")
    (repo / "config.toml").write_text("CHANGED AFTER MEASUREMENT\n")
    (repo / "new.bin").write_bytes(b"unsealed")
    paths = {"inside": repo / "config.toml", "pattern": repo / "config.t*",
             "outside": tmp_path / "config.toml", "escape": repo / ".." / "config.toml",
             "unsealed": repo / "new.bin", "ambiguous": repo / "config*.toml"}
    artifact = str(paths[declared])
    if declared in {"inside", "pattern"}:
        checkpoint.validate_checkpoint(saved, artifact_paths=[artifact], repo_path=str(repo))
        with pytest.raises(ValueError, match="declared runtime artifact is absent"):
            checkpoint.validate_checkpoint(saved, artifact_paths=[artifact])
    else:
        with pytest.raises(ValueError, match="declared runtime artifact is absent"):
            checkpoint.validate_checkpoint(saved, artifact_paths=[artifact], repo_path=str(repo))


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("declared", ["weight[0].bin", "weight-a.*", "missing.bin", "weight-*.bin"])
def test_checkpoint_checks_declared_runtime_files_against_sealed_inventory(tmp_path, absolute, declared):
    repo = make_repo(tmp_path, None)
    folder = tmp_path if absolute else repo
    names = ["weight[0].bin", "weight-a.bin", "weight-b.bin"]
    paths = [str(folder / name) if absolute else name for name in names]
    for name in names:
        (folder / name).write_bytes(b"measured runtime file")
    saved = checkpoint.snapshot_checkpoint(str(repo), "attempt-sealed", paths)
    required = str(folder / declared) if absolute else declared
    # Current runtime files are irrelevant: only immutable, measured bytes count.
    for name in names:
        (folder / name).unlink()
    if declared in {"missing.bin", "weight-*.bin"}:
        with pytest.raises(ValueError, match="declared runtime artifact is absent"):
            checkpoint.validate_checkpoint(saved, artifact_paths=[required])
    else:
        checkpoint.validate_checkpoint(saved, artifact_paths=[required])


def test_absolute_snapshot_source_is_sealed_but_another_snapshot_is_not(tmp_path):
    repo = make_repo(tmp_path, None)
    first = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-first"))
    second = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-second"))
    checkpoint.validate_checkpoint(second, artifact_paths=[str(second / "config.toml")])
    with pytest.raises(ValueError, match="declared runtime artifact is absent"):
        checkpoint.validate_checkpoint(second, artifact_paths=[str(first / "config.toml")],
                                       repo_path=str(repo))


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("matches", [0, 1, 2])
def test_checkpoint_resolves_only_unique_artifact_filename_patterns(
        tmp_path, monkeypatch, sandbox, absolute, matches):
    from autoresearch.experiments.repo_improve import agentic_improve
    repo = make_repo(tmp_path, None)
    logs = (tmp_path if absolute else repo) / "run logs"
    logs.mkdir()
    for index in range(matches):
        (logs / f"run-{index}.log").write_text(f"measured log {index}\n")
    declared = str(logs / "run-*.log") if absolute else "run logs/run-*.log"
    backend = None
    if sandbox:
        def execute(handle, command, cwd, timeout):
            result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
            return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        backend = SimpleNamespace(execute_command=execute)
        monkeypatch.setattr(agentic_improve, "_prepare_checkpoint_handoff", lambda *args: None)
    if matches != 1:
        with pytest.raises(ValueError, match="exactly one file"):
            checkpoint.snapshot_checkpoint(str(repo), "attempt-logs", [declared],
                                           backend=backend, handle=object())
        return
    saved = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-logs", [declared],
                                              backend=backend, handle=object()))
    manifest = checkpoint.validate_checkpoint(saved)
    if absolute:
        assert set(manifest["external"]) == {str(logs / "run-0.log")}
        preserved = saved / manifest["external"][str(logs / "run-0.log")]
    else:
        assert manifest["runtime_paths"] == ["run logs/run-0.log"]
        preserved = saved / "run logs/run-0.log"
    assert preserved.read_text() == "measured log 0\n"


def test_restore_external_retries_with_noninteractive_sudo_in_unprivileged_sandbox(tmp_path):
    repo = make_repo(tmp_path, None)
    runtime_file = tmp_path / "task-root" / "output" / "adapter_config.json"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text('{"adapter": true}\n')
    saved = checkpoint.snapshot_checkpoint(str(repo), "attempt-model", [str(runtime_file)])
    calls = []

    def execute(_handle, command, *, cwd, timeout):
        calls.append((command, cwd, timeout))
        return SimpleNamespace(exit_code=0 if "sudo -n" in command else 1)

    backend = SimpleNamespace(execute_command=execute)
    checkpoint.restore_external(saved, backend, object(), str(repo))

    assert len(calls) == 2
    assert "sudo" not in calls[0][0]
    assert "sudo -n mkdir -p" in calls[1][0]
    assert "sudo -n cp --" in calls[1][0]
    assert str(runtime_file) in calls[1][0]


def test_restore_external_prefers_backend_root_seam_for_absolute_artifact(tmp_path):
    repo = make_repo(tmp_path, None)
    runtime_file = tmp_path / "task-root" / "output" / "adapter_config.json"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text('{"adapter": true}\n')
    saved = checkpoint.snapshot_checkpoint(str(repo), "attempt-model", [str(runtime_file)])
    calls = []

    def execute(_handle, command, *, cwd, timeout):
        calls.append(("user", command, cwd, timeout))
        return SimpleNamespace(exit_code=1)

    def execute_as_root(_handle, command, *, cwd, timeout):
        calls.append(("root", command, cwd, timeout))
        return SimpleNamespace(exit_code=0)

    backend = SimpleNamespace(
        execute_command=execute, execute_command_as_root=execute_as_root)
    checkpoint.restore_external(saved, backend, object(), str(repo))

    assert [call[0] for call in calls] == ["user", "root"]
    assert "sudo" not in calls[1][1]
    assert str(runtime_file) in calls[1][1]


def test_restore_external_uses_verified_checkpoint_import_beside_current_checkout(tmp_path):
    old_root = tmp_path / "old-controller"
    old_root.mkdir()
    old_repo = make_repo(old_root, None)
    runtime_file = old_root / "runtime.bin"
    runtime_file.write_bytes(b"sealed runtime\n")
    old_checkpoint = Path(checkpoint.snapshot_checkpoint(
        str(old_repo), "attempt-old", [str(runtime_file)]))

    new_root = tmp_path / "new-controller"
    new_root.mkdir()
    new_repo = make_repo(new_root, None)
    imported = new_root / old_checkpoint.name
    shutil.copytree(old_checkpoint, imported, symlinks=True)
    commands = []

    def execute(_handle, command, *, cwd, timeout):
        commands.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    checkpoint.restore_external(
        str(old_checkpoint), SimpleNamespace(execute_command=execute),
        object(), str(new_repo))

    assert len(commands) == 1
    assert str(imported) in commands[0]
    assert str(old_checkpoint) not in commands[0]


def test_checkpoint_preserves_literal_metacharacters_in_artifact_paths(tmp_path):
    repo = make_repo(tmp_path, None)
    logs = repo / "logs[0]"
    logs.mkdir()
    (logs / "run[0].log").write_text("literal filename\n")
    saved = Path(checkpoint.snapshot_checkpoint(
        str(repo), "attempt-literal", ["logs[0]/run[0].log"]))
    checkpoint.validate_checkpoint(saved)
    assert (saved / "logs[0]/run[0].log").read_text() == "literal filename\n"


def test_checkpoint_rejects_index_only_source_until_materialized(tmp_path):
    repo = make_repo(tmp_path, "config.toml")
    source = repo / "engine.py"
    implementation = "def predict():\n    return 2\n"
    source.write_text(implementation)
    subprocess.run(["git", "-C", str(repo), "add", "engine.py"], check=True)
    source.unlink()
    # A staged addition followed by a working-tree deletion has no HEAD diff.
    assert subprocess.check_output([
        "git", "-C", str(repo), "diff", "HEAD", "--name-only"]) == b""

    with pytest.raises(ValueError, match="staged.*missing.*engine.py"):
        checkpoint.snapshot_checkpoint(str(repo), "attempt-index-only")
    assert not list(tmp_path.glob("checkpoint-*"))

    source.write_text(implementation)
    saved = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-materialized"))
    assert (saved / "engine.py").read_text() == implementation
    checkpoint.validate_checkpoint(saved)


def test_completion_repairs_index_only_source_before_closing_sandbox(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve as module
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output

    repo = make_repo(tmp_path, None)
    saved = checkpoint.snapshot_checkpoint(str(repo), "measured")
    commit = module._repo_commit(str(repo))
    measured_diff = module._tracked_repo_state(str(repo))["tracked_diff_hash"]
    source = repo / "engine.py"
    source.write_text("implemented = True\n")
    subprocess.run(["git", "-C", str(repo), "add", "engine.py"], check=True)
    source.unlink()
    record = dict(attempt_id="measured", phase="method", launched=True, finished=True,
                  cwd=str(repo), rc=0, checkpoint=saved,
                  repo_commit=commit, repo_commit_end=commit,
                  tracked_diff_hash_start=measured_diff, tracked_diff_hash_end=measured_diff,
                  **save_output(tmp_path, "measured", json.dumps({
                      "method": 0.6, "candidate_valid": True}), ""))
    monkeypatch.setattr(module, "_attempt_records", lambda _: {"measured": record})
    result = repo / "ar_result.json"
    payload = {"method": 0.6, "attempt_id": "measured",
               "metric": {"name": "quality", "unit": "ratio"}}
    result.write_text(json.dumps(payload))

    request = module._result_completion_request(
        str(result), execution_phase="method", attempt_log="host.jsonl")

    assert request is not None
    assert "engine.py" in request and "staged" in request
    assert "ar-attempt" in request and "actual checkout" in request
    assert json.loads(result.read_text()) == payload
    assert record["rc"] == 0 and not source.exists()


def test_repeated_snapshot_does_not_replace_corrupted_checkpoint(tmp_path):
    repo = make_repo(tmp_path, "config.toml")
    first = Path(checkpoint.snapshot_checkpoint(str(repo), "same-attempt"))
    (first / "config.toml").write_text("corrupted\n")
    with pytest.raises(ValueError):
        checkpoint.snapshot_checkpoint(str(repo), "same-attempt")
    assert (first / "config.toml").read_text() == "corrupted\n"


def test_frozen_baseline_delta_excludes_inherited_generated_source(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _apply_prior_worktree, _capture_component_diff, _capture_repo_diff)

    repo = make_repo(tmp_path, "config.toml")
    generated = repo / "generated_cache" / "Compiled.py"
    generated.parent.mkdir()
    generated.write_text("# runtime generated\n")
    baseline = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-baseline"))
    candidate = tmp_path / "candidate"
    subprocess.run(["git", "clone", "-q", str(repo), str(candidate)], check=True)
    _apply_prior_worktree(str(candidate), str(baseline))
    (candidate / "config.toml").write_text("threshold = 2\n")
    (candidate / "new_component.py").write_text("candidate_authored = True\n")

    assert "generated_cache/Compiled.py" in _capture_repo_diff(str(candidate))
    delta = _capture_component_diff(str(candidate), [str(baseline)])
    assert "config.toml" in delta
    assert "new_component.py" in delta
    assert "generated_cache/Compiled.py" not in delta


def test_checkpoint_handles_sandbox_uid_without_global_git_trust(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, "config.toml")
    (repo / "config.toml").write_text("threshold = 2\n")
    (repo / "candidate.py").write_text("value = 2\n")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    rejected = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                              capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "dubious ownership" in rejected.stderr

    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-sandbox-uid"))

    checkpoint.validate_checkpoint(snapshot)
    assert (snapshot / "config.toml").read_text() == "threshold = 2\n"
    assert (snapshot / "candidate.py").read_text() == "value = 2\n"
    # The scoped exception must not persist into unrelated subsequent git calls.
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                          capture_output=True).returncode != 0


@pytest.mark.parametrize("tracked", [False, True])
def test_cargo_cache_is_not_promoted_to_source_but_tracked_files_remain_checked(tmp_path, tracked):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _dependency_worktree_changes, _tracked_repo_state)

    repo = make_repo(tmp_path, "config.toml")
    cache = repo / "custom-build"
    fingerprint = cache / "release/.fingerprint/package/bin-app.json"
    fingerprint.parent.mkdir(parents=True)
    marker = cache / ".rustc_info.json"
    marker.write_text('{"rustc_fingerprint": 1}')
    fingerprint.write_text('{"rustc": 1}')
    if tracked:
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Offline Test",
                        "-c", "user.email=offline@example.invalid", "commit", "-qm",
                        "repository-owned metadata"], check=True)
        marker.write_text('{"rustc_fingerprint": 2}')
        fingerprint.write_text('{"rustc": 2}')
    # Generated code and task configuration must not disappear with cache metadata.
    (cache / "generated.rs").write_text("fn generated() {}\n")
    (cache / "settings.json").write_text('{"width": 8}')
    plain = repo / ".fingerprint/config.json"
    plain.parent.mkdir()
    plain.write_text('{"threshold": 2}')
    child = tmp_path / "control"
    subprocess.run(["git", "clone", "-q", str(repo), str(child)], check=True)
    # Also cover restoring older snapshots that already contain untracked caches.
    changes = _dependency_worktree_changes(str(child), str(repo))
    assert (str(marker.relative_to(repo)) in changes) == tracked
    assert (str(fingerprint.relative_to(repo)) in changes) == tracked
    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-cache"))
    for source in (marker, fingerprint):
        restored = snapshot / source.relative_to(repo)
        assert restored.exists() == tracked
        if tracked:
            assert restored.read_bytes() == source.read_bytes()
    for relative in ("custom-build/generated.rs", "custom-build/settings.json",
                     ".fingerprint/config.json"):
        assert relative in changes
        assert (snapshot / relative).read_bytes() == (repo / relative).read_bytes()
    assert marker.exists()  # No cleanup of the original runtime cache.
    if tracked:
        before = _tracked_repo_state(str(snapshot))
        (snapshot / marker.relative_to(repo)).write_text('{"rustc_fingerprint": 3}')
        assert _tracked_repo_state(str(snapshot)) != before


def test_tracked_internal_link_materialized_without_changing_source(tmp_path):
    repo = make_repo(tmp_path, "config.toml")
    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-test"))
    manifest = checkpoint.validate_checkpoint(snapshot)
    assert (repo / "alias.toml").is_symlink()
    assert not (snapshot / "alias.toml").is_symlink()
    assert (snapshot / "alias.toml").read_bytes() == (repo / "config.toml").read_bytes()
    assert manifest["files"]["alias.toml"] == manifest["files"]["config.toml"]


def test_checkpoint_validation_does_not_resolve_each_regular_file(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, "config.toml")
    for index in range(20):
        (repo / f"source_{index}.py").write_text(f"VALUE = {index}\n")
    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-many-files"))

    original = Path.resolve
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted)
    checkpoint.validate_checkpoint(snapshot)

    assert len(calls) < 10


def test_checkpoint_clone_allows_large_repositories_more_than_one_minute(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, "config.toml")
    original = subprocess.run
    clone_timeouts = []

    def record_clone_timeout(args, *positional, **kwargs):
        if "clone" in args and "--shared" in args:
            clone_timeouts.append(kwargs.get("timeout"))
        return original(args, *positional, **kwargs)

    monkeypatch.setattr(checkpoint.subprocess, "run", record_clone_timeout)
    checkpoint.snapshot_checkpoint(str(repo), "attempt-large-repo")

    assert clone_timeouts == [600]


@pytest.mark.parametrize("alias_value", [1, 2])
def test_restore_materialized_alias_does_not_write_through_checkout_link(tmp_path, alias_value):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree

    repo = make_repo(tmp_path, "config.toml")
    # Once materialized, the alias is an independent file in the accepted state.
    (repo / "alias.toml").unlink()
    (repo / "alias.toml").write_text(f"threshold = {alias_value}\n")
    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-updated-parent"))
    child = tmp_path / "control"
    subprocess.run(["git", "clone", "-q", str(repo), str(child)], check=True)
    assert (child / "alias.toml").is_symlink()

    _apply_prior_worktree(str(child), str(snapshot))

    assert not (child / "alias.toml").is_symlink()
    assert (child / "alias.toml").read_text() == f"threshold = {alias_value}\n"
    assert (child / "config.toml").read_text() == "threshold = 1\n"
    checkpoint.validate_checkpoint(snapshot)


def test_restore_refuses_alias_pointing_outside_checkout(tmp_path):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree

    repo = make_repo(tmp_path, "config.toml")
    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-parent"))
    child = tmp_path / "control"
    subprocess.run(["git", "clone", "-q", str(repo), str(child)], check=True)
    outside = tmp_path / "outside.toml"
    outside.write_text("outside = true\n")
    (child / "alias.toml").unlink()
    (child / "alias.toml").symlink_to(outside)

    with pytest.raises(ValueError, match="target escapes repository"):
        _apply_prior_worktree(str(child), str(snapshot))

    assert (child / "alias.toml").is_symlink()
    assert outside.read_text() == "outside = true\n"


def test_snapshot_replacement_does_not_write_through_external_head_alias(tmp_path):
    outside = tmp_path / "outside.toml"
    outside.write_text("outside = true\n")
    repo = make_repo(tmp_path, "../outside.toml")
    (repo / "alias.toml").unlink()
    (repo / "alias.toml").write_text("local = true\n")

    with pytest.raises(ValueError, match="snapshot destination escapes"):
        checkpoint.snapshot_checkpoint(str(repo), "attempt-local-replacement")

    assert outside.read_text() == "outside = true\n"


def test_external_link_is_not_imported(tmp_path):
    (tmp_path / "outside.toml").write_text("private = true\n")
    repo = make_repo(tmp_path, "../outside.toml")
    with pytest.raises(ValueError, match="inside its snapshot"):
        checkpoint.snapshot_checkpoint(str(repo), "attempt-test")
    assert (repo / "alias.toml").is_symlink()


def test_dangling_link_fails_without_changing_source(tmp_path):
    repo = make_repo(tmp_path, "missing.toml")
    with pytest.raises(ValueError, match="inside its snapshot"):
        checkpoint.snapshot_checkpoint(str(repo), "attempt-test")
    assert (repo / "alias.toml").is_symlink()


def test_untracked_external_source_reports_relative_path_for_repair(tmp_path):
    repo = make_repo(tmp_path, "config.toml")
    outside = tmp_path / "previous.py"
    outside.write_text("print('previous attempt')\n")
    relative = "analysis/probe/staging/source.py"
    link = repo / relative
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError) as error:
        checkpoint.snapshot_checkpoint(str(repo), "attempt-current")
    assert relative in str(error.value)
    assert "snapshot source escapes the repository" in str(error.value)
    assert link.is_symlink()
    assert outside.read_text() == "print('previous attempt')\n"


def test_attempt_snapshot_prepares_sandbox_sources_before_reading(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve

    repo = make_repo(tmp_path, "config.toml")
    generated = repo / "generated.py"
    generated.write_text("print('measured')\n")
    backend, handle = object(), object()
    prepared = []
    copy = checkpoint.shutil.copy2

    def prepare(actual_backend, actual_handle, path, emit):
        assert (actual_backend, actual_handle, path) == (backend, handle, str(repo))
        prepared.append(path)

    def copy_after_handoff(src, dst, *args, **kwargs):
        if Path(src) == generated and not prepared:
            raise PermissionError("sandbox UID owns generated source")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(agentic_improve, "_prepare_checkpoint_handoff", prepare)
    monkeypatch.setattr(checkpoint.shutil, "copy2", copy_after_handoff)
    saved = Path(checkpoint.snapshot_checkpoint(
        str(repo), "attempt-readable", backend=backend, handle=handle))
    assert prepared == [str(repo)]
    assert (saved / "generated.py").read_bytes() == generated.read_bytes()
    checkpoint.validate_checkpoint(saved)


@pytest.mark.parametrize("intent_to_add", [False, True])
@pytest.mark.parametrize("nested_repo", [False, True])
def test_checkpoint_keeps_force_added_ignored_source_visible_to_review_and_restore(
        tmp_path, intent_to_add, nested_repo):
    from autoresearch.experiments.repo_improve.agentic_improve import (
        _capture_component_diff, _apply_prior_worktree)

    repo = make_repo(tmp_path, "config.toml")
    (repo / ".gitignore").write_text("workspace/\n")
    subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Offline Test",
                    "-c", "user.email=offline@example.invalid", "commit", "-qm", "ignore workspace"], check=True)
    relative = "workspace/upstream/feature.py"
    source = repo / relative
    source.parent.mkdir(parents=True)
    if nested_repo:
        subprocess.run(["git", "init", "-q", str(source.parent)], check=True)
        source.write_text("def feature(x):\n    return x\n")
        subprocess.run(["git", "-C", str(source.parent), "add", "feature.py"], check=True)
        subprocess.run(["git", "-C", str(source.parent), "-c", "user.name=Offline Test",
                        "-c", "user.email=offline@example.invalid", "commit", "-qm", "upstream"], check=True)
    source.write_text("def feature(x):\n    return x * 2\n")
    subprocess.run(["git", "-C", str(repo), "add", "-f",
                    *(["--intent-to-add"] if intent_to_add else []), relative], check=True)
    if nested_repo:
        # git add can silently ignore a path owned by a nested checkout. Explicit
        # outer-index registration keeps the real file visible without moving HEAD.
        blob = subprocess.check_output(["git", "-C", str(repo), "hash-object", "-w", relative],
                                       text=True).strip()
        subprocess.run(["git", "-C", str(repo), "update-index", "--add", "--cacheinfo",
                        "100644", blob, relative], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"])
    before = subprocess.check_output(["git", "-C", str(repo), "diff", "--cached"])

    snapshot = Path(checkpoint.snapshot_checkpoint(str(repo), "attempt-ignored-source"))
    assert relative in checkpoint.validate_checkpoint(snapshot)["files"]
    assert subprocess.check_output(["git", "-C", str(snapshot), "rev-parse", "HEAD"]) == head
    assert "return x * 2" in _capture_component_diff(str(snapshot), [])
    child = tmp_path / "child"
    subprocess.run(["git", "clone", "-q", str(repo), str(child)], check=True)
    _apply_prior_worktree(str(child), str(snapshot))
    assert (child / relative).read_bytes() == source.read_bytes()
    (child / relative).write_text("def feature(x):\n    return x * 3\n")
    next_snapshot = Path(checkpoint.snapshot_checkpoint(str(child), "attempt-child"))
    assert relative in checkpoint.validate_checkpoint(next_snapshot)["files"]
    assert "+    return x * 3" in _capture_component_diff(
        str(next_snapshot), str(snapshot))
    assert subprocess.check_output(["git", "-C", str(repo), "diff", "--cached"]) == before
