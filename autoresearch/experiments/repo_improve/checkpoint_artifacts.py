"""Recoverable code snapshots plus only explicitly declared runtime artifacts."""
from __future__ import annotations

import glob
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import uuid

MANIFEST = ".autoresearch-checkpoint.json"


def _checkpoint_operation_timeout() -> int:
    """Timeout for checkpoint clone/index operations on shared filesystems."""
    try:
        return max(60, int(os.environ.get("AUTORESEARCH_CHECKPOINT_OPERATION_TIMEOUT_S", "600")))
    except (TypeError, ValueError):
        return 600


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint(directory, *, artifact_paths=(), repo_path=None):
    root = Path(directory)
    root_resolved = root.resolve()
    manifest = root / MANIFEST
    if not manifest.exists():
        raise ValueError("checkpoint has no immutable content manifest")
    raw = manifest.read_bytes()
    if root.name != "checkpoint-" + hashlib.sha256(raw).hexdigest():
        raise ValueError("checkpoint manifest identity changed")
    data = json.loads(raw)
    present = set()
    links = {}
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                relative = str(path.relative_to(root))
                resolved = path.resolve()
                if (not resolved.is_relative_to(root_resolved) or not resolved.is_dir()
                        or ".git" in resolved.relative_to(root_resolved).parts
                        or resolved == path.parent or resolved in path.parents):
                    raise ValueError("checkpoint directory link escapes or cycles inside its snapshot")
                links[relative] = os.readlink(path)
        present.update(str((Path(directory) / name).relative_to(root)) for name in names
                       if str((Path(directory) / name).relative_to(root)) != MANIFEST)
    if links != data.get("directory_links", {}):
        raise ValueError("checkpoint directory link inventory changed")
    if present != set(data["files"]):
        raise ValueError("checkpoint file inventory changed")
    for relative, expected in data["files"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("checkpoint path escapes its snapshot")
        path = root / relative_path
        # os.walk already inventories every entry and validates directory links above.
        # Resolving every regular file repeats a full ancestor walk for large repositories;
        # reject file links directly without weakening the complete content-hash check.
        if path.is_symlink():
            raise ValueError("checkpoint path escapes its snapshot")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"checkpoint content changed: {relative}")
    if not isinstance(artifact_paths, (list, tuple)):
        raise ValueError("artifact_paths must be a list of file paths")
    for raw in artifact_paths:
        if not isinstance(raw, str) or not raw:
            raise ValueError("artifact_paths entries must name files")
        declared = Path(os.path.normpath(raw))
        inventory = data.get("external", {}) if declared.is_absolute() else data["files"]
        names = {os.path.normpath(name) for name in inventory}
        if declared.is_absolute():
            snapshot_names = {str(root_resolved / name) for name in data["files"]}
            names.update(snapshot_names)
        if declared.is_absolute() and repo_path:
            source = Path(os.path.normpath(repo_path))
            if source.is_absolute():
                # A host-bound repository source may already be sealed by its
                # relative path. Only the snapshot inventory supplies its bytes.
                names.update(str(source / name) for name in data["files"])
        matched = str(declared) in names
        if not matched and glob.has_magic(declared.name) and not glob.has_magic(str(declared.parent)):
            matched = sum(Path(name).parent == declared.parent
                          and fnmatch.fnmatchcase(Path(name).name, declared.name)
                          for name in names) == 1
        if not matched:
            raise ValueError(
                f"declared runtime artifact is absent from measured checkpoint: {raw}; "
                "declare the required files in ar_result.json before ar-attempt exits, "
                "then rerun the same evaluator with those files; preserve the previous "
                "measurement and do not modify its immutable checkpoint")
    return data


def _validate_review_receipt_output(receipt):
    from .runtime_evidence import read_output
    if receipt.get("rc") == 0:
        read_output(receipt)
        return
    # A failed control remains failure evidence, not a numeric measurement.
    # Review recovery still requires its original, completed host output.
    if (not receipt.get("launched") or not receipt.get("finished")
            or receipt.get("execution_uncertain") or type(receipt.get("rc")) is not int):
        raise ValueError("review requires a completed host attempt")
    for stream in ("stdout", "stderr"):
        path = receipt.get(stream + "_path")
        if not isinstance(path, str) or not path or digest(path) != receipt.get(stream + "_sha256"):
            raise ValueError("saved execution output content changed")


def checkpoint_recoverable(directory, *, runtime_receipts=()):
    """Validate a checkpoint where it lives, including remote GPU workspaces."""
    if not isinstance(directory, (str, os.PathLike)) or not str(directory):
        return False
    try:
        validate_checkpoint(directory)
        for receipt in runtime_receipts:
            _validate_review_receipt_output(receipt)
        return True
    except (OSError, ValueError, TypeError):
        # A locally present but invalid snapshot must never be accepted through SSH.
        if Path(directory).exists() or not Path(directory).is_absolute():
            return False
    mode = os.environ.get("AUTORESEARCH_REPO_MODE")
    if mode == "local" or os.environ.get("AUTORESEARCH_REPO_SPLIT", "1") == "0":
        return False
    if mode not in ("auto", "remote") and os.environ.get("AUTORESEARCH_REPO_REMOTE") != "1":
        return False
    host = os.environ.get("AUTORESEARCH_GPU_HOST", "").strip()
    runtime = os.environ.get("AUTORESEARCH_REPO_REMOTE_DIR", "").strip()
    if not host or not runtime:
        return False
    remote = shlex.join([
        "env", f"PYTHONPATH={runtime}", "python3", "-c",
        "import sys,json; from autoresearch.experiments.repo_improve.checkpoint_artifacts "
        "import validate_checkpoint, _validate_review_receipt_output; "
        "validate_checkpoint(sys.argv[1]); "
        "[_validate_review_receipt_output(r) for r in json.loads(sys.argv[2])]",
        str(directory), json.dumps(runtime_receipts),
    ])
    control_path = os.environ.get(
        "AUTORESEARCH_SSH_CONTROLPATH", os.path.expanduser("~/.ssh/cm-ar-%C"))
    command = ["ssh"]
    if os.environ.get("AUTORESEARCH_SSH_CONFIG"):
        command += ["-F", os.environ["AUTORESEARCH_SSH_CONFIG"]]
    try:
        result = subprocess.run(command + [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={control_path}",
            "-o", "ControlPersist=120", host, remote,
        ], capture_output=True, text=True, timeout=_checkpoint_operation_timeout(),
            env={key: value for key, value in os.environ.items() if key.casefold() not in {
                "http_proxy", "https_proxy", "all_proxy"}})
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def validate_staged_additions(repo_path):
    """Staging an external worktree must not hide source absent from this checkout."""
    source = Path(repo_path)
    raw = subprocess.check_output([
        "git", "-c", f"safe.directory={source.resolve()}", "-C", str(source),
        "diff", "--cached", "--name-only", "--diff-filter=A", "-z",
    ], timeout=_checkpoint_operation_timeout())
    missing = [os.fsdecode(name) for name in raw.split(b"\0") if name
               and not (source / os.fsdecode(name)).exists()
               and not (source / os.fsdecode(name)).is_symlink()]
    if missing:
        raise ValueError(
            "staged additions missing from actual checkout: " + ", ".join(missing)
            + "; materialize the exact implementation files here, stage this checkout, "
            "then rerun the same candidate through ar-attempt; preserve previous measurements")


def snapshot_checkpoint(repo_path, attempt_id, artifacts=(), *, backend=None, handle=None,
                        handoff_prepared=False):
    from .agentic_improve import (
        _carry_path_allowed, _CARRY_SOURCE_SUFFIXES, _CARRY_SOURCE_NAMES,
        _expand_untracked_repositories, _prepare_checkpoint_handoff,
        _untracked_compiler_metadata,
    )
    if backend is not None and not handoff_prepared:
        # Each attempt snapshots before the final sandbox handoff. Generated
        # sources may still be readable only by the sandbox's different UID.
        _prepare_checkpoint_handoff(backend, handle, str(repo_path), lambda *_: None)
    source = Path(repo_path)
    validate_staged_additions(source)
    target = source.parent / ("checkpoint-" + uuid.uuid4().hex)
    # Sandboxes may create the checkout under a different UID. Trust only this
    # resolved checkout for these commands, without changing global Git config.
    source_git = ["git", "-c", f"safe.directory={source.resolve()}"]
    git_dir = subprocess.check_output(
        [*source_git, "-C", str(source), "rev-parse", "--absolute-git-dir"], text=True).strip()
    # Clone's upload-pack subprocess drops the caller's -c settings.
    upload_pack = shlex.join(["git", "-c", f"safe.directory={git_dir}", "upload-pack"])
    subprocess.run([*source_git, "clone", "--shared", "--no-hardlinks", "--quiet",
                    f"--upload-pack={upload_pack}", str(source), str(target)],
                   capture_output=True, check=True, timeout=_checkpoint_operation_timeout())
    def paths(*args):
        raw = subprocess.check_output([*source_git, "-C", str(source), *args, "-z"])
        return [os.fsdecode(p) for p in raw.split(b"\0") if p]
    changed = paths("diff", "--name-only", "HEAD")
    untracked = _expand_untracked_repositories(
        str(source), paths("ls-files", "--others", "--exclude-standard"))
    selected = changed + [p for p in untracked if _carry_path_allowed(p)
        and not _untracked_compiler_metadata(str(source), p) and (
        Path(p).suffix.casefold() in _CARRY_SOURCE_SUFFIXES or Path(p).name in _CARRY_SOURCE_NAMES)]
    for relative in selected:
        src, dst = source / relative, target / relative
        if src.is_symlink() or not src.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"snapshot source escapes the repository: {relative}; "
                             "keep the required source inside this checkout instead of "
                             "linking another attempt")
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                if not dst.resolve().is_relative_to(target.resolve()):
                    raise ValueError("snapshot destination escapes the repository")
                # Copy the source's regular file, not through the cloned HEAD alias.
                dst.unlink()
            shutil.copy2(src, dst)
        elif dst.is_file():
            dst.unlink()
    if changed:
        # A forced-added source can live under an ignored workspace. Cloning and
        # copying bytes alone loses its index entry, hiding the change from MIR
        # and later dependency restores. Preserve visibility without moving HEAD.
        subprocess.run(["git", "-c", f"safe.directory={target.resolve()}",
                        "-C", str(target), "add", "--force", "--all", "--", *changed],
                       capture_output=True, check=True, timeout=_checkpoint_operation_timeout())
    external = {}
    runtime_paths = []
    for raw in artifacts:
        if not isinstance(raw, str) or not raw:
            raise ValueError("artifact_paths entries must name files")
        original = Path(raw)
        if glob.has_magic(original.name) and not glob.has_magic(str(original.parent)):
            if not original.is_absolute() and ".." in original.parts:
                raise ValueError("artifact path escapes repository")
            if backend is None:
                candidate = original if original.is_absolute() else source / original
                matches = ([str(candidate)] if candidate.is_file() else
                           [p for p in sorted(glob.glob(str(candidate))) if Path(p).is_file()])
                if not original.is_absolute():
                    matches = [str(Path(p).relative_to(source)) for p in matches]
            else:
                code = ("import glob,json,os,sys; p=sys.argv[1]; "
                        "print(json.dumps([p] if os.path.isfile(p) else "
                        "[v for v in sorted(glob.glob(p)) if os.path.isfile(v)]))")
                result = backend.execute_command(handle, shlex.join(["python3", "-c", code, raw]),
                                                 cwd=str(source), timeout=_checkpoint_operation_timeout())
                if result.exit_code != 0:
                    raise ValueError(f"cannot resolve declared sandbox artifact: {raw}")
                matches = json.loads(result.stdout)
            if len(matches) != 1:
                raise ValueError(f"artifact pattern must match exactly one file: {raw}")
            raw = matches[0]
            original = Path(raw)
        if original.is_absolute():
            relative = "external-files/" + hashlib.sha256(raw.encode()).hexdigest()
            destination = target / relative
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError("artifact destination escapes its snapshot through a link")
            destination.parent.mkdir(exist_ok=True)
            os.chmod(target, 0o777)
            os.chmod(destination.parent, 0o777)
            if backend is None:
                shutil.copy2(original, destination)
            else:
                result = backend.execute_command(handle, shlex.join([
                    "cp", "--", raw, str(destination)]), cwd=str(source),
                    timeout=_checkpoint_operation_timeout())
                if result.exit_code != 0:
                    raise ValueError(f"cannot preserve declared sandbox artifact: {raw}")
                result = backend.execute_command(handle, shlex.join([
                    "chmod", "a+r", "--", str(destination)]), cwd=str(source),
                    timeout=_checkpoint_operation_timeout())
                if result.exit_code != 0:
                    raise ValueError(f"cannot read preserved sandbox artifact: {raw}")
            external[raw] = relative
        else:
            if ".." in original.parts:
                raise ValueError("artifact path escapes repository")
            relative = raw
            destination = target / relative
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError("artifact destination escapes its snapshot through a link")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
            runtime_paths.append(relative)
    files = {}
    directory_links = {}
    for root, dirs, names in os.walk(target):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in dirs:
            path = Path(root) / name
            if path.is_symlink():
                resolved = path.resolve()
                if (not resolved.is_relative_to(target.resolve()) or not resolved.is_dir()
                        or ".git" in resolved.relative_to(target.resolve()).parts
                        or resolved == path.parent or resolved in path.parents):
                    raise ValueError("checkpoint directory link must resolve inside its snapshot without cycles")
                directory_links[str(path.relative_to(target))] = os.readlink(path)
        for name in names:
            path = Path(root) / name
            if path.is_symlink():
                resolved = path.resolve()
                if not resolved.is_relative_to(target.resolve()) or not resolved.is_file():
                    raise ValueError("checkpoint link must resolve to a regular file inside its snapshot")
                # Materialize the snapshot's internal alias, not the source checkout.
                path.unlink()
                shutil.copy2(resolved, path)
            files[str(path.relative_to(target))] = digest(path)
    data = {"attempt_id": attempt_id, "files": files, "external": external,
            "runtime_paths": runtime_paths, "directory_links": directory_links}
    raw = json.dumps(data, sort_keys=True).encode()
    (target / MANIFEST).write_bytes(raw)
    final = source.parent / ("checkpoint-" + hashlib.sha256(raw).hexdigest())
    try:
        target.rename(final)
    except OSError as exc:
        import errno
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        if validate_checkpoint(final) != data:
            raise ValueError("existing checkpoint does not match the captured content") from exc
        shutil.rmtree(target)
    return str(final)


def restore_external(parents, backend, handle, repo_path):
    if isinstance(parents, dict):
        parents = parents.get("checkpoints")
    parents = parents if isinstance(parents, (list, tuple)) else [parents] if parents else []
    selected = {}
    for parent in parents:
        data = validate_checkpoint(parent)
        # Source restoration may already have imported an immutable checkpoint
        # from an earlier controller beside the current checkout. Runtime files
        # must use that same mounted copy; the original controller path is not
        # necessarily visible inside the new sandbox.
        imported = Path(repo_path).resolve().parent / Path(parent).name
        if imported.resolve() != Path(parent).resolve() and (imported / MANIFEST).is_file():
            if validate_checkpoint(imported) != data:
                raise ValueError(f"imported dependency checkpoint identity mismatch: {parent}")
            parent = str(imported)
        for destination, relative in data.get("external", {}).items():
            saved = str(Path(parent) / relative)
            prior = selected.get(destination)
            if prior and digest(prior) != digest(saved):
                raise ValueError(f"dependency checkpoint conflict for external artifact {destination}")
            selected[destination] = saved
    for destination, saved in selected.items():
        command = shlex.join(["mkdir", "-p", "--", str(Path(destination).parent)]) + " && " + shlex.join([
            "cp", "--", saved, destination])
        result = backend.execute_command(
            handle, command, cwd=repo_path, timeout=_checkpoint_operation_timeout())
        if result.exit_code != 0 and Path(destination).is_absolute():
            # Task-owned absolute paths such as /app/output commonly live outside the
            # unprivileged OpenHands user's writable roots.  They were explicitly
            # declared by a measured attempt and the source bytes are sealed by the
            # checkpoint manifest, so retry only this restore through the backend's
            # root maintenance seam. Older/test backends retain the sudo fallback.
            root_execute = getattr(backend, "execute_command_as_root", None)
            if callable(root_execute):
                result = root_execute(
                    handle, command, cwd=repo_path,
                    timeout=_checkpoint_operation_timeout())
            else:
                command = shlex.join([
                    "sudo", "-n", "mkdir", "-p", "--", str(Path(destination).parent)
                ]) + " && " + shlex.join([
                    "sudo", "-n", "cp", "--", saved, destination
                ])
                result = backend.execute_command(
                    handle, command, cwd=repo_path,
                    timeout=_checkpoint_operation_timeout())
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout or "").strip().replace("\n", " ")[:300]
            suffix = f": {detail}" if detail else ""
            raise ValueError(f"could not restore declared artifact {destination}{suffix}")


def external_source_diff(checkpoint, parents=()):
    """Expose declared out-of-repository source edits to implementation review."""
    import difflib
    from .agentic_improve import _CARRY_SOURCE_SUFFIXES, _CARRY_SOURCE_NAMES
    current = validate_checkpoint(checkpoint)
    if isinstance(parents, dict):
        parents = parents.get("checkpoints") or []
    parents = parents if isinstance(parents, (list, tuple)) else [parents] if parents else []
    before = {}
    for parent in parents:
        if not (Path(parent) / MANIFEST).is_file():
            continue
        for original, relative in validate_checkpoint(parent).get("external", {}).items():
            before[original] = Path(parent) / relative
    chunks = []
    for original, relative in current.get("external", {}).items():
        path = Path(original)
        if path.suffix.casefold() not in _CARRY_SOURCE_SUFFIXES and path.name not in _CARRY_SOURCE_NAMES:
            continue
        old = before[original].read_text(errors="replace") if original in before else ""
        new = (Path(checkpoint) / relative).read_text(errors="replace")
        chunks.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/external:" + original, tofile="b/external:" + original))
    return "".join(chunks)
