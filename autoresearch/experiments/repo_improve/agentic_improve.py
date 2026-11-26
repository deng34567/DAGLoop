"""No-infer, full-OpenHands repo improvement (2026-07-30 architecture decision).

Unlike ``pipeline.run_repo_improve`` -- which first runs ``discover``/``infer_benchmark`` to
derive a rigid benchmark boundary (metric, run_cmd, editable/protected files) and then only lets
the agent edit the pre-declared editable files -- this path hands the WHOLE job to an in-container
OpenHands code-agent:

    clone the repo -> the agent explores it, runs the repo's OWN benchmark to establish the
    BASELINE, improves the code, re-runs the benchmark, and writes the two measured numbers.

There is NO ``infer_benchmark`` gate (so repos it fails-closed on -- SOL's CUDA-kernel-in-JSON,
Docker/big-data repos -- reach a capable agent instead of returning None) and NO anti-cheat
machinery (the professor deemed protected-file/harness enforcement not worth its complexity):
we trust the numbers the agent measured with the repo's own scorer.

Returns the SAME dict shape as ``run_repo_improve`` so ``_res_to_improve_json`` / ``_map_repo_result``
consume it unchanged. Any infra failure or a missing/garbled agent result degrades to an honest
``{"error": ...}`` (baseline/method -> None -> invalid), never a fabricated value.
"""
from __future__ import annotations

import ast
import difflib
import errno
import fcntl
import json
import hashlib
import inspect
import math
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

from autoresearch.experiments.repo_improve.runner import RunResult
from autoresearch.experiments.repo_improve.benchmark_contract import (
    BenchmarkContractError,
    contract_from_confirmed_boundary,
    contract_fingerprint,
    merge_benchmark_contracts,
    normalize_contract,
)

_RESULT_BASENAME = "ar_result.json"
# #75 跨轮引导: workroot 级的分析结论文件。外层 R5/method loop 用同一个 workroot 重跑,每轮取新的
# repo 子目录(repo, repo-2, ...),所以 per-repo 的 analysis/direction.txt 下一轮读不到 -- 把结论
# 也落到 workroot 级这里,下一轮再读回来,补上跨轮那一环。
_CARRY_BASENAME = "carry_direction.txt"


def _repository_task_contract(repo_path: str, request: str, execution_contract: dict | None = None) -> dict[str, Any] | None:
    """Capture the authoritative task documents most relevant to this repository run.

    Selection is repository-agnostic: task terms from the request are matched against document
    paths, then the best matching task directory is frozen verbatim.  The model does not get to
    omit an inconvenient rule from a document it already used to run the benchmark.
    """
    # Recovery owners may expose the same checkout through a symlink. Compare
    # canonical paths on both sides; an in-checkout document is not an escape.
    repo_path = os.path.realpath(repo_path)
    request_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(request).casefold())
        if len(token) >= 3
    }
    # Explicit task documents need not use our legacy README/task.toml names.
    text_file = r"[A-Za-z0-9_.-]+\.(?:json|ya?ml|toml|md|txt|py)\b"
    referenced_paths = {os.path.normpath(path).replace(os.sep, "/") for path in
                        re.findall(r"(?:[A-Za-z0-9_.-]+/)+" + text_file, str(request))}
    referenced_dirs = {os.path.dirname(path) for path in referenced_paths}
    referenced_names = set(re.findall(text_file, str(request)))
    from .source_policy import permitted
    source_policy = {}
    try:
        if os.path.islink(os.path.join(repo_path, "task.toml")):
            raise OSError("task manifest must not alias another source")
        with open(os.path.join(repo_path, "task.toml"), "rb") as declaration:
            source_policy = (tomllib.load(declaration).get("autoresearch", {}).get("sources") or {})
    except (OSError, tomllib.TOMLDecodeError):
        pass
    if execution_contract is not None:
        source_policy = execution_contract.get("sources", source_policy)
    candidates: list[tuple[int, str, str]] = []
    packaging: list[tuple[int, str, str]] = []
    ignored_dirs = {
        ".git", ".venv", "node_modules", "vendor", "dist", "build", "__pycache__",
    }
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [name for name in dirs if name not in ignored_dirs and not name.startswith(".")
                   and permitted(os.path.relpath(os.path.join(root, name), repo_path), source_policy, directory=True)]
        for name in files:
            folded = name.casefold()
            path = os.path.join(root, name)
            rel = os.path.relpath(path, repo_path).replace(os.sep, "/")
            explicitly_named = rel in referenced_paths or (rel.endswith(".py") and any(
                named.endswith("/" + rel) for named in referenced_paths)) or (
                os.path.dirname(rel) in referenced_dirs and name in referenced_names)
            is_packaging = folded in {"dockerfile", "containerfile", ".dockerignore"}
            if not (
                    is_packaging or explicitly_named or folded in {"instruction.md", "instructions.md", "task.md", "task.toml"}
                    or folded.startswith("readme")):
                continue
            if not permitted(rel, source_policy):
                continue
            if not os.path.realpath(path).startswith(os.path.realpath(repo_path) + os.sep):
                continue
            if not permitted(os.path.relpath(os.path.realpath(path), repo_path), source_policy):
                continue
            path_tokens = set(re.findall(r"[a-z0-9]+", rel.casefold()))
            score = sum(len(token) for token in request_tokens & path_tokens)
            (packaging if is_packaging else candidates).append((score, rel, path))
    if not candidates:
        return {"sources": [], "execution_contract": execution_contract} if execution_contract else None
    task_document_paths = {rel for _, rel, _ in candidates if (
        os.path.basename(rel).casefold() in {
            "instruction.md", "instructions.md", "task.md", "task.toml"}
        or os.path.basename(rel).casefold().startswith("readme"))}
    declared_task = ((execution_contract or {}).get("experiment_provenance") or {}).get("task")
    if isinstance(declared_task, str) and declared_task.strip():
        matching = [item for item in candidates
                    if not item[1].endswith(".py")
                    and declared_task.strip() in item[1].split("/")[:-1]]
        if matching:
            root_dirs = {os.path.dirname(rel) for _, rel, _ in matching
                         if rel in task_document_paths
                         and os.path.basename(os.path.dirname(rel)) == declared_task.strip()}
            if root_dirs:
                matching = [item for item in matching if item[1] not in task_document_paths
                            or os.path.dirname(item[1]) in root_dirs]
            candidates = matching + [item for item in candidates if item[1].endswith(".py")]
    elif "task.toml" in task_document_paths and any(
            name in task_document_paths for name in {"instruction.md", "instructions.md", "task.md"}):
        candidates = [item for item in candidates if item[1] not in task_document_paths
                      or not os.path.dirname(item[1])]
    documents = [item for item in candidates if not item[1].endswith(".py")]
    task_documents = [item for item in documents if item[1] in task_document_paths]
    if any(score > 0 for score, _, _ in task_documents):
        documents = task_documents  # Explicit input/code paths must not select the task's rules.
    best_score = max((score for score, _, _ in documents), default=0)
    if best_score > 0:
        best_dirs = {
            os.path.dirname(rel) for score, rel, _ in documents if score == best_score
        }
        selected = [item for item in documents if os.path.dirname(item[1]) in best_dirs]
    else:
        # With no task/path match, only root-level instructions are unambiguously authoritative.
        selected = [item for item in documents if os.path.dirname(item[1]) in {"", "."}]
    explicit = [item for item in candidates if item[1] in referenced_paths or (item[1].endswith(".py") and any(
        named.endswith("/" + item[1]) for named in referenced_paths)) or (
        os.path.dirname(item[1]) in referenced_dirs and os.path.basename(item[1]) in referenced_names)]
    selected = sorted(selected, key=lambda item: item[1])
    # Supply actual COPY/WORKDIR context without letting build files select a task
    # or displace its rules. A separately documented subtree owns its own packaging.
    document_dirs = ({os.path.dirname(rel) for _, rel, _ in documents}
                     | {os.path.dirname(rel) for rel in task_document_paths})
    selected_dirs = {os.path.dirname(rel) for _, rel, _ in selected}
    for item in sorted(packaging, key=lambda item: item[1]):
        directory = os.path.dirname(item[1])
        while directory and directory not in document_dirs:
            directory = os.path.dirname(directory)
        if directory in selected_dirs:
            selected.append(item)
    selected.extend(item for item in sorted(explicit, key=lambda item: item[1])
                    if item not in selected)
    sources: list[dict[str, Any]] = []
    remaining = 20_000
    omitted_sources = []
    for index, (_, rel, path) in enumerate(selected):
        if index >= 8 or remaining <= 0:
            omitted_sources.append({"path": rel, "reason": "capture_budget"})
            continue
        try:
            digest = hashlib.sha256()
            size = 0
            excerpt = bytearray()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    excerpt.extend(chunk[:max(0, remaining - len(excerpt))])
        except OSError:
            omitted_sources.append({"path": rel, "reason": "unreadable"})
            continue
        content = bytes(excerpt).decode("utf-8", errors="replace")
        remaining -= len(excerpt)
        sources.append({
            "path": rel, "sha256": digest.hexdigest(), "content": content,
            "size_bytes": size, "excerpt_bytes": len(excerpt), "excerpt_offset": 0,
            "excerpt_sha256": hashlib.sha256(excerpt).hexdigest(),
            "truncated": len(excerpt) != size,
        })
    declarations = []
    objectives = []
    for source in sources:
        if not source.get("truncated") and os.path.basename(source["path"]) == "task.toml":
            try:
                manifest = tomllib.loads(source["content"])
            except tomllib.TOMLDecodeError:
                continue
            declaration = manifest.get("autoresearch", {})
            if declaration:
                declarations.append(declaration)
            optimization = manifest.get("optimization")
            if isinstance(optimization, dict) and isinstance(optimization.get("metric"), str):
                metric = optimization["metric"].strip()
                if metric:
                    objective = {"metric_name": metric}
                    direction = optimization.get("direction")
                    if direction in {"lower", "minimize"}:
                        objective["lower_is_better"] = True
                    elif direction in {"higher", "maximize"}:
                        objective["lower_is_better"] = False
                    objectives.append(objective)
    if declarations and any(item != declarations[0] for item in declarations):
        raise ValueError("selected task documents declare conflicting execution contracts")
    if objectives and any(item != objectives[0] for item in objectives):
        raise ValueError("selected task documents declare conflicting optimization objectives")
    declared = declarations[0] if declarations else {}
    if execution_contract is not None:
        declared = {**declared, **execution_contract}
    if objectives:
        evaluation = dict(declared.get("evaluation") or {})
        for key, value in objectives[0].items():
            evaluation.setdefault(key, value)
        declared = {**declared, "evaluation": evaluation}
    return {"sources": sources, "execution_contract": declared,
            "capture_complete": not omitted_sources and not any(s["truncated"] for s in sources),
            "omitted_sources": omitted_sources} if sources or declared or omitted_sources else None


def _contract_with_packaging_evidence(repo_path, frozen_baseline, captured):
    """Supplement legacy review inputs without replacing frozen task rules."""
    frozen = (frozen_baseline or {}).get("repository_contract")
    if not frozen:
        return captured
    commit = (frozen_baseline or {}).get("repo_commit")
    sources = list(frozen.get("sources") or [])
    current = {s["path"]: s for s in (captured or {}).get("sources", [])}
    recover_sources = not sources
    if (not commit or (recover_sources and (
            not current or (captured or {}).get("capture_complete") is not True)) or any(
            not s.get("sha256") or (s["path"] in current
                and current[s["path"]].get("sha256") != s["sha256"])
            for s in sources)):
        return frozen
    paths = {s["path"] for s in sources}
    # Keep the frozen capture intact and allow one bounded supplemental capture.
    # Legacy code excerpts must not consume the missing task documents' budget.
    size = 0
    added = []
    recovered_rules = []
    for path, source in current.items():
        name = os.path.basename(path).casefold()
        is_rule = name in {"instruction.md", "instructions.md", "task.md", "task.toml"} or name.startswith("readme")
        if (path in paths or (not recover_sources and os.path.basename(path).casefold() not in
                {"dockerfile", "containerfile", ".dockerignore"} and not path.endswith(".py")
                and not is_rule)
                or source.get("truncated")):
            continue
        if len(added) >= 8 or size + len(source["content"].encode("utf-8")) > 20000:
            continue
        try:
            original = subprocess.check_output(
                ["git", "show", f"{commit}:{path}"], cwd=repo_path,
                stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if hashlib.sha256(original).hexdigest() != source.get("sha256"):
            continue
        sources.append(source)
        added.append(path)
        if is_rule:
            recovered_rules.append(path)
        size += len(source["content"].encode("utf-8"))
    if recover_sources and len(added) != len(current):
        return frozen  # A partial or candidate-edited ruleset cannot replace missing authority.
    if recover_sources and added:
        return {**captured, **frozen, "sources": sources, "recovered_task_sources": added,
                "execution_contract": {**(captured.get("execution_contract") or {}),
                                       **(frozen.get("execution_contract") or {})}}
    code_sources = [path for path in added if path.endswith(".py")]
    return ({**frozen, "sources": sources,
             "supplemental_packaging_sources": [path for path in added
                 if path not in code_sources and path not in recovered_rules],
             **({"recovered_task_sources": recovered_rules} if recovered_rules else {}),
             **({"supplemental_reference_sources": code_sources} if code_sources else {})}
            if added else frozen)


def validate_execution_contract(contract):
    """Opt-in publication preflight; do not invent scientific controls for legacy runs."""
    declaration = (contract or {}).get("execution_contract") or {}
    if not declaration.get("require_evaluation_contract"):
        return
    evaluation = declaration.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("strict execution contract requires evaluation configuration")
    required = ("argv", "metric_pointer", "gate_pointers", "required_values", "measurement", "immutable_files")
    missing = [field for field in required if not evaluation.get(field)]
    if missing:
        raise ValueError("strict evaluation contract missing: " + ", ".join(missing))
    if (contract or {}).get("capture_complete") is False:
        raise ValueError("task contract capture is incomplete; resolve truncated/omitted sources before launch")
    if not isinstance(evaluation["argv"], list) or not all(
            isinstance(token, str) and token for token in evaluation["argv"]):
        raise ValueError("evaluation argv must be a nonempty command token list")
    if not isinstance(evaluation["required_values"], dict) or not isinstance(evaluation["immutable_files"], dict):
        raise ValueError("evaluation identities and immutable files must be mappings")
    if (not isinstance(evaluation["metric_pointer"], str) or not evaluation["metric_pointer"].startswith("/")
            or not isinstance(evaluation["gate_pointers"], list)
            or not all(isinstance(v, str) and v.startswith("/") for v in evaluation["gate_pointers"])):
        raise ValueError("evaluation metric and gates need explicit JSON pointers")
    policy = evaluation["measurement"]
    if not isinstance(policy, dict) or policy.get("kind") not in {"official_single", "deterministic", "repeated"}:
        raise ValueError("evaluation requires an explicit measurement policy")
    if policy["kind"] == "repeated" and (
            type(policy.get("repeats")) is not int or policy["repeats"] < 2
            or policy.get("aggregation") not in {"mean", "median"}):
        raise ValueError("repeated measurement needs repeats >= 2 and mean/median aggregation")
    if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
           for h in evaluation["immutable_files"].values()):
        raise ValueError("immutable evaluator/input files require full SHA256 values")


def _repository_resource_limits(contract: dict[str, Any] | None) -> dict[str, int]:
    """Read Docker-enforceable limits from the uniquely selected repository task manifest.

    The task-document selector above already resolves the request to a task directory.  This
    helper consumes only its captured ``task.toml``; it does not infer defaults from task names,
    repository identity, or benchmark type.  Conflicting selected manifests fail open.
    """
    parsed: list[dict[str, int]] = []
    for source in (contract or {}).get("sources", []):
        if not isinstance(source, dict) or os.path.basename(
                str(source.get("path") or "")).casefold() != "task.toml":
            continue
        try:
            document = tomllib.loads(str(source.get("content") or ""))
        except (TypeError, tomllib.TOMLDecodeError):
            continue
        environment = document.get("environment")
        if not isinstance(environment, dict):
            continue
        limits: dict[str, int] = {}
        for manifest_key, result_key in (("cpus", "cpus"), ("memory_mb", "memory_mb")):
            value = environment.get(manifest_key)
            if isinstance(value, bool):
                continue
            try:
                normalized = int(value)
            except (TypeError, ValueError):
                continue
            if normalized > 0:
                limits[result_key] = normalized
        if limits:
            parsed.append(limits)
    if not parsed or any(item != parsed[0] for item in parsed[1:]):
        return {}
    return parsed[0]


def _repository_cuda_devel_image(
        contract: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return one selected task Dockerfile's official NVIDIA devel image and version."""
    images: set[tuple[str, str]] = set()
    pattern = re.compile(
        r"(?im)^\s*FROM(?:\s+--platform=\S+)?\s+"
        r"(nvidia/cuda:([0-9]+(?:\.[0-9]+){1,2})-devel(?:[-\w.]*)?)\s*$")
    for source in (contract or {}).get("sources", []):
        if (not isinstance(source, dict) or os.path.basename(
                str(source.get("path") or "")).casefold() not in
                {"dockerfile", "containerfile"}):
            continue
        images.update(pattern.findall(str(source.get("content") or "")))
    if len(images) > 1:
        raise ValueError("selected task packaging declares conflicting CUDA devel versions")
    return next(iter(images), (None, None))


def _repository_cuda_devel_version(contract: dict[str, Any] | None) -> str | None:
    """Return a selected task Dockerfile's declared NVIDIA CUDA devel version."""
    return _repository_cuda_devel_image(contract)[1]


def _cuda_release(nvcc: str) -> str | None:
    try:
        result = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"\brelease\s+([0-9]+\.[0-9]+)\b", result.stdout + result.stderr)
    return match.group(1) if match else None


def _matching_cuda_toolkit_root(required_version: str) -> tuple[str | None, str | None]:
    """Resolve an installed toolkit without downloading or mutating the host."""
    required_release = ".".join(required_version.split(".")[:2])
    configured = os.environ.get("AUTORESEARCH_SANDBOX_CUDA_ROOT", "").strip()
    candidates = [configured] if configured else []
    data_root = os.environ.get("AUTORESEARCH_SANDBOX_DATA_ROOT", "").strip()
    toolchain_root = os.path.join(data_root, "toolchains") if data_root else ""
    if toolchain_root and os.path.isdir(toolchain_root):
        try:
            candidates.extend(sorted(
                entry.path for entry in os.scandir(toolchain_root)
                if entry.is_dir() and entry.name.startswith("cuda-" + required_version)))
        except OSError:
            pass
    candidates.extend([
        "/usr/local/cuda-" + required_version,
        "/usr/local/cuda-" + required_release,
        "/usr/local/cuda",
    ])
    observed = None
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        root = os.path.realpath(candidate)
        nvcc = os.path.join(root, "bin", "nvcc")
        if not os.path.isfile(nvcc) or not os.access(nvcc, os.X_OK):
            continue
        observed = _cuda_release(nvcc)
        if observed == required_release:
            return root, observed
    return None, observed


def _provision_cuda_toolkit(
        image: str, required_version: str, workroot: str, emit) -> tuple[str, str]:
    """Pull one repository-declared official CUDA image and extract its toolkit cache."""
    expected = re.fullmatch(
        r"nvidia/cuda:([0-9]+(?:\.[0-9]+){1,2})-devel(?:[-\w.]*)?", image)
    if expected is None or expected.group(1) != required_version:
        raise ValueError("CUDA provisioning requires one exact official nvidia/cuda devel image")
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker is required to provision the repository-declared CUDA image")

    data_root = os.environ.get("AUTORESEARCH_SANDBOX_DATA_ROOT", "").strip()
    cache_parent = (os.path.join(data_root, "toolchains")
                    if data_root and os.path.isdir(data_root)
                    else os.path.join(workroot, ".toolchains"))
    os.makedirs(cache_parent, exist_ok=True)
    cache_key = hashlib.sha256(image.encode()).hexdigest()[:12]
    destination = os.path.join(
        cache_parent, f"cuda-{required_version}-docker-{cache_key}")
    required_release = ".".join(required_version.split(".")[:2])
    lock_path = os.path.join(cache_parent, f".cuda-{required_version}-{cache_key}.lock")

    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = _cuda_release(os.path.join(destination, "bin", "nvcc"))
        if cached == required_release:
            emit("agentic_cuda_provision", {
                "stage": "cached", "image": image, "root": destination,
                "observed_release": cached})
            return os.path.realpath(destination), cached

        direct_env = _direct_network_env()

        def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, env=direct_env)
            if result.returncode:
                detail = (result.stderr or result.stdout or "").strip()[-1000:]
                raise RuntimeError(
                    f"{' '.join(argv[:2])} failed with exit {result.returncode}: {detail}")
            return result

        emit("agentic_cuda_provision", {
            "stage": "pulling", "image": image, "root": destination,
            "direct_connection": True})
        run([docker, "pull", "--platform", "linux/amd64", image], 3600)
        digest = run(
            [docker, "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
            60).stdout.strip()
        stage = tempfile.mkdtemp(
            prefix=f".cuda-{required_version}-{cache_key}.stage.", dir=cache_parent)
        container_id = ""
        try:
            container_id = run([docker, "create", image], 120).stdout.strip()
            if not container_id:
                raise RuntimeError("docker create returned no container id")
            run([docker, "cp", f"{container_id}:/usr/local/cuda-{required_release}/.", stage],
                1800)
            Path(stage, "source-image.txt").write_text(image + "\n", encoding="utf-8")
            Path(stage, "source-image-digest.txt").write_text(digest + "\n", encoding="utf-8")
            observed = _cuda_release(os.path.join(stage, "bin", "nvcc"))
            if observed != required_release:
                raise RuntimeError(
                    f"extracted nvcc release {observed!r} does not match {required_release}")
            # tempfile.mkdtemp is 0700.  The OpenHands sandbox runs as a different
            # user, so the read-only bind mount must remain traversable there.
            os.chmod(stage, 0o755)
            os.replace(stage, destination)
        except Exception as exc:
            emit("agentic_cuda_provision", {
                "stage": "failed", "image": image, "staging_root": stage,
                "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            if container_id:
                subprocess.run(
                    [docker, "rm", "-f", container_id], capture_output=True, text=True,
                    timeout=120, env=direct_env)
        emit("agentic_cuda_provision", {
            "stage": "ready", "image": image, "image_digest": digest,
            "root": destination, "observed_release": required_release})
        return os.path.realpath(destination), required_release


def _carry_basename(memory_scope: str | None = None) -> str:
    """Return a stable per-method carry filename; preserve the legacy unscoped path."""
    scope = str(memory_scope or "").strip()
    if not scope:
        return _CARRY_BASENAME
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    return f"carry_direction.{digest}.txt"


def _direct_network_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update({"NO_PROXY": "*", "no_proxy": "*"})
    return env


def _variant_cap() -> int:
    """Sandbox launch limit; explicit zero is unlimited, invalid settings use the default."""
    try:
        cap = int(os.environ.get("AUTORESEARCH_AGENTIC_VARIANT_CAP", "20"))
        return cap if cap >= 0 else 20
    except ValueError:
        return 20


def _attempt_count(path: str) -> int:
    """Count distinct launches, including incomplete, non-zero and timed-out attempts."""
    return len(_attempt_records(path))


def _attempt_records(path: str) -> dict[str, dict]:
    """Fold the append-only launch/finish ledger into one host record per attempt.

    Old TSV rows remain countable so an in-flight session created by the previous wrapper still
    reaches its cap, but they intentionally lack the fields needed by the attestation gate.
    """
    records: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    records[f"legacy-{index}"] = {"legacy": True, "raw": line}
                    continue
                attempt_id = str(event.get("attempt_id") or "")
                if not attempt_id:
                    records[f"malformed-{index}"] = {"malformed": True, "raw": event}
                    continue
                current = records.setdefault(attempt_id, {"attempt_id": attempt_id})
                current.update(event)
                if event.get("event") == "launch":
                    current["launched"] = True
                elif event.get("event") == "finish":
                    current["finished"] = not event.get("execution_uncertain", False)
            return records
    except OSError:
        return records


def _saved_review_receipts(path: str | None, result: dict, repo_path: str,
                           attempt_id: str | None) -> list[dict]:
    """Retain host receipts needed to revalidate a review without rerunning work."""
    if not path:
        return []
    ids = {attempt_id} if attempt_id else set()
    for item in result.get("ablation_report") or []:
        if isinstance(item, dict):
            for name in ("evidence", "parent_evidence"):
                ref = (item.get(name) or {}).get("ref", "")
                ids.update(re.findall(r"\battempt-[A-Za-z0-9-]+\b", str(ref)))
    records = _evidence_records(path, result, repo_path)
    return [records[key] for key in sorted(ids) if key in records]


def _attempt_review_evidence(path: str | None) -> list[dict]:
    """Bounded host trace for task rules about how a candidate was selected."""
    if not path:
        return []
    from .runtime_evidence import pointer, read_output

    def output_fields(value, path=""):
        if isinstance(value, (dict, list)):
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                part = str(key).replace("~", "~0").replace("/", "~1")
                yield from output_fields(item, path + "/" + part)
        elif value is None or isinstance(value, (str, bool, int, float)):
            yield path, value

    summary = []
    for record in list(_attempt_records(path).values())[-16:]:
        item = {key: record.get(key) for key in (
            "attempt_id", "argv", "frozen_baseline_control", "rc",
            "tracked_diff_hash_start", "command_timeout_s", "verifier_timeout_s",
            "command_sources", "command_sources_stable")
                if record.get(key) is not None}
        stdout_path = record.get("stdout_path")
        if stdout_path:
            try:
                with open(stdout_path, encoding="utf-8", errors="replace") as handle:
                    output = handle.read(65537)
                if len(output) > 65536:
                    item["output_truncated"] = True
                    output = output[:65536]
                try:
                    parsed = json.loads(output)
                except ValueError:
                    item["stdout_excerpt"] = output[-700:]
                else:
                    if isinstance(parsed, dict):
                        item["stdout_fields"] = {key: parsed[key] for key in (
                            "measurements", "val", "test", "config", "best_config",
                            "selected_config", "grid_best", "exit_codes") if key in parsed}
                    else:
                        item["stdout_excerpt"] = output[-700:]
            except OSError:
                item["output_unavailable"] = True
            try:
                payload = read_output(record, diagnostic=True)
            except ValueError:
                pass
            else:
                fields, size = {}, 0
                for field, value in output_fields(payload):
                    try:
                        pointer(payload, field)
                    except ValueError:
                        continue
                    if isinstance(value, str) and len(value) > 65536:
                        value = value[:65536]
                        item["verified_output_truncated"] = True
                    size += len(json.dumps({field: value}, ensure_ascii=False))
                    if size > 65536:
                        item["verified_output_truncated"] = True
                        break
                    fields[field] = value
                item["verified_output_fields"] = fields
                item["stdout_sha256"] = record["stdout_sha256"]
        summary.append(item)
    return summary


def _evidence_records(path: str, result: dict, repo_path: str) -> dict[str, dict]:
    """Load explicitly cited receipts from earlier sandboxes of this session only.

    Keep launch counting and candidate selection on the current ledger. Historical
    receipts are evidence inputs, never new launches or implicit candidate choices.
    """
    from pathlib import Path
    records = _attempt_records(path)
    wanted = set(re.findall(r"\battempt-[A-Za-z0-9-]+\b", json.dumps({
        "ablation_report": result.get("ablation_report"),
        "metric_attempt_id": result.get("_metric_attempt_id"),
    }))) - records.keys()
    if not wanted or not path:
        return records
    session_root = Path(repo_path).resolve().parent
    for ledger in Path(path).parent.parent.glob(".repo*-attempt-authority-*/attempts.jsonl"):
        if ledger.resolve() == Path(path).resolve():
            continue
        for aid, record in _attempt_records(str(ledger)).items():
            if aid not in wanted or not record.get("cwd"):
                continue
            if not Path(record["cwd"]).resolve().is_relative_to(session_root):
                continue
            records[aid] = record
            wanted.remove(aid)
        if not wanted:
            break
    return records


def _attempt_cap_ready_to_pause(path: str, cap: int, grace_s: float = 60.0,
                                result_path: str | None = None) -> bool:
    """Pause at the cap only after the agent has persisted its structured result.

    ``grace_s`` preserves the legacy fallback for callers without a result path.  The live
    method path must not assume that one model turn always finishes within a fixed grace period.
    """
    if cap <= 0:
        return False
    records = _attempt_records(path)
    if len(records) < cap or any(
            record.get("launched") and not record.get("finished") for record in records.values()):
        return False
    if result_path is not None:
        try:
            with open(result_path, encoding="utf-8") as handle:
                result = json.load(handle)
            if not isinstance(result, dict):
                return False
            metric = result.get("metric")
            metric_name = metric.get("name") if isinstance(metric, dict) else metric
            return (isinstance(metric_name, str) and bool(metric_name.strip())
                    and ("method" in result or "method_metric" in result))
        except (OSError, TypeError, ValueError):
            return False
    try:
        return time.time() - os.path.getmtime(path) >= grace_s
    except OSError:
        return False


def _normalized_attempt_argv(value: Any) -> list[str]:
    """Remove only transparent launch wrappers from a recorded attempt argv."""
    if (not isinstance(value, list) or not value
            or any(not isinstance(token, str) or not token for token in value)):
        raise ValueError("attempt argv must be a non-empty string array")
    argv = list(value)
    while argv:
        executable = os.path.basename(argv[0])
        if executable == "timeout":
            index = 1
            value_options = {"-s", "--signal", "-k", "--kill-after"}
            flag_options = {"--preserve-status", "--foreground", "--verbose"}
            while index < len(argv) and argv[index].startswith("-"):
                option = argv[index]
                name = option.split("=", 1)[0]
                if name in flag_options:
                    index += 1
                elif name in value_options:
                    index += 1 if "=" in option else 2
                else:
                    raise ValueError(f"unsupported timeout wrapper option: {option}")
            if index >= len(argv):
                raise ValueError("timeout wrapper is missing a duration")
            index += 1
            if index >= len(argv):
                raise ValueError("timeout wrapper is missing a command")
            argv = argv[index:]
            continue
        # `env`, including `env -i`, changes inherited scientific controls.  It is execution
        # identity, not a transparent launcher wrapper.
        break
    return argv


def _bind_matching_host_attempt(result: dict, records: dict[str, dict], repo_path: str,
                                execution_phase: str, spec: dict,
                                frozen_baseline: dict | None) -> None:
    """Fill an omitted citation only when one successful execution is unambiguous."""
    explicit_id = str(result.get("_metric_attempt_id") or "")
    current_commit = _repo_commit(repo_path)
    current_diff = _tracked_repo_state(repo_path).get("tracked_diff_hash")
    expected_cwd = os.path.realpath(os.path.join(repo_path, spec["execution"]["cwd"]))

    execution_matching = []
    matching = []
    for record in records.values():
        try:
            execution_eligible = bool(
                record.get("launched") is True
                and record.get("finished") is True
                and record.get("rc") == 0
                and record.get("phase") == execution_phase
                and record.get("repo_commit") == current_commit
                and record.get("repo_commit_end") == current_commit
                and record.get("cwd") == expected_cwd
                and _normalized_attempt_argv(record.get("argv")) == spec["execution"]["argv"]
            )
        except ValueError:
            execution_eligible = False
        if not execution_eligible:
            continue
        execution_matching.append(record)
        if (record.get("tracked_diff_hash_start") == current_diff
                and record.get("tracked_diff_hash_end") == current_diff):
            matching.append(record)

    if explicit_id:
        chosen = next((record for record in matching
                       if record.get("attempt_id") == explicit_id), None)
    else:
        # With multiple successful candidates, the current diff does not prove that the scalar in
        # ar_result.json came from the matching candidate: it may be a historical best copied from
        # another attempt. Require an explicit citation in that case.
        chosen = matching[0] if len(execution_matching) == 1 and len(matching) == 1 else None
    if chosen is None:
        return

    attempt_id = str(chosen["attempt_id"])
    result["_metric_attempt_id"] = attempt_id
    if execution_phase != "method":
        return

    expected_fingerprint = str((frozen_baseline or {}).get("contract_fingerprint") or "")
    manifest = result.get("run_manifest")
    if manifest is not None and not isinstance(manifest, dict):
        return
    manifest = dict(manifest or {})
    if manifest.get("attempt_id") not in (None, "", attempt_id):
        return
    if expected_fingerprint and manifest.get("contract_fingerprint") not in (
            None, "", expected_fingerprint):
        return
    manifest["attempt_id"] = attempt_id
    if expected_fingerprint:
        manifest["contract_fingerprint"] = expected_fingerprint
    result["run_manifest"] = manifest


def _index_only_source_change(record: dict, repo_path: str) -> bool:
    """Prove a late Git staging change against the host's measured snapshot.

    Never infer equivalent code from a matching metric or an agent explanation.
    Newly tracked source absent from the measured snapshot remains a rejection.
    """
    from .checkpoint_artifacts import validate_checkpoint
    from .runtime_evidence import _preserved_snapshot

    checkpoint = record.get("checkpoint")
    measured = record.get("tracked_diff_hash_start")
    if (not checkpoint or record.get("rc") != 0 or not measured
            or measured != record.get("tracked_diff_hash_end")):
        return False
    try:
        manifest = validate_checkpoint(checkpoint)
        if (_repo_commit(checkpoint) != record.get("repo_commit")
                or _tracked_repo_state(checkpoint).get("tracked_diff_hash") != measured
                or not _preserved_snapshot(record, target_directory=repo_path)):
            return False
        changed = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo_path,
             "diff", "--name-only", "-z", "HEAD", "--"],
            capture_output=True, check=True, timeout=30).stdout
        # These two new handoff outputs are already excluded from source hashes.
        allowed = set(manifest["files"]) | {_RESULT_BASENAME, "analysis/variants.jsonl"}
        return all(os.fsdecode(path) in allowed for path in changed.split(b"\0") if path)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


def _bind_component_evidence_attempt(result: dict, records: dict[str, dict],
                                     repo_path: str, execution_phase: str) -> str | None:
    """Validate an explicit ID, or recover an omitted one, against the host ledger."""
    result.pop("component_failed_attempt_id", None)
    result.pop("component_execution_started", None)
    result.pop("component_evidence_error", None)
    if execution_phase != "method":
        return
    explicit_id = str(result.pop("_metric_attempt_id", "") or "").strip()
    report = result.get("ablation_report")
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except (TypeError, ValueError):
            report = None
    if isinstance(report, dict):
        report = [report]
    if not isinstance(report, list):
        report = []

    # Host validation markers are never accepted from the sandbox's result file.
    for item in report:
        if isinstance(item, dict):
            item.pop("_host_bound", None)
            if isinstance(item.get("parent_evidence"), dict):
                item["parent_evidence"].pop("host_bound", None)
    if report:
        result["ablation_report"] = report

    cited_ids: set[str] = {explicit_id} if explicit_id else set()
    for item in ([] if explicit_id else report):
        evidence = item.get("evidence") if isinstance(item, dict) else None
        if not isinstance(evidence, dict) or evidence.get("source") != "runtime_attempt":
            continue
        cited_ids.update(re.findall(r"\battempt-[A-Za-z0-9-]+\b", str(evidence.get("ref") or "")))
    if len(cited_ids) != 1:
        return

    attempt_id = next(iter(cited_ids))
    record = records.get(attempt_id)
    if not isinstance(record, dict):
        registered = [aid for aid, item in records.items()
                      if item.get("phase") == execution_phase and item.get("finished")][-6:]
        return (
            f"runtime attempt {attempt_id} is not present in the host authority ledger. "
            f"Registered finished attempt IDs (diagnostic only): {registered!r}. "
            "Check their saved output and correct the exact top-level attempt_id and "
            "matching evidence refs if this is a transcription error; do not infer "
            "identity from a similar ID or metric. "
            "Do not unset, replace, or redirect AUTORESEARCH_ATTEMPT_SOCKET or "
            "AUTORESEARCH_ATTEMPT_LOG; if no registered attempt measures this candidate, "
            "run the component check through the supplied ar-attempt command. A sandbox-local attempt "
            "log is not execution evidence."
        )
    current_commit = _repo_commit(repo_path)
    current_diff = _tracked_repo_state(repo_path).get("tracked_diff_hash")
    try:
        cwd = os.path.realpath(str(record.get("cwd") or ""))
        in_repo = os.path.commonpath([cwd, os.path.realpath(repo_path)]) == os.path.realpath(repo_path)
    except (ValueError, OSError):
        in_repo = False
    if not in_repo:
        return (f"runtime attempt {attempt_id} used working directory {cwd!r} outside "
                f"the current repository {repo_path!r}; cd into the current repository "
                "before ar-attempt and rerun the component check")
    # Launch accounting is independent of successful numeric evidence binding.
    # A malformed result or nonzero exit must not become an unstarted infra retry.
    if record.get("launched") is True and record.get("phase") == execution_phase:
        result["component_execution_started"] = True
    if not (
        record.get("launched") is True
        and record.get("finished") is True
        and record.get("rc") in (0, 1)
        and record.get("phase") == execution_phase
    ):
        return
    if os.path.exists(os.path.join(repo_path, ".git")):
        from .checkpoint_artifacts import validate_staged_additions
        try:
            validate_staged_additions(repo_path)
        except ValueError as exc:
            return str(exc)
    if (record.get("repo_commit") != current_commit
            or record.get("repo_commit_end") != current_commit):
        return (f"runtime attempt {attempt_id} commit does not match the current repository; "
                "fix the checkout before execution and rerun the component check")
    if ((record.get("tracked_diff_hash_start") != current_diff
            or record.get("tracked_diff_hash_end") != current_diff)
            and not _index_only_source_change(record, repo_path)):
        matching_ids = [
            key for key, other in records.items()
            if other.get("launched") and other.get("finished") and other.get("rc") == 0
            and other.get("phase") == execution_phase
            and os.path.realpath(str(other.get("cwd") or "")) == cwd
            and other.get("repo_commit") == other.get("repo_commit_end") == current_commit
            and other.get("tracked_diff_hash_start") == other.get("tracked_diff_hash_end") == current_diff
            and other.get("execution_protocol") == record.get("execution_protocol")
        ][-3:]
        checkpoint = record.get("checkpoint")
        restore_hint = ""
        if (isinstance(checkpoint, str) and checkpoint.strip()
                and not record.get("checkpoint_error") and record.get("rc") == 0
                and record.get("tracked_diff_hash_start") == record.get("tracked_diff_hash_end")):
            restore_hint = (
                f"Recorded checkpoint for the cited attempt (verify before reuse): {checkpoint!r}. "
                "To report that historical candidate, restore its exact code and required runtime artifacts "
                "using the checkpoint restoration procedure; normal evidence checks still apply. "
                "Do not attach that number to the current variant or rewrite saved execution output. ")
        return (f"runtime attempt {attempt_id} tracked diff fingerprint does not match the "
                "current repository (code edits or git index changes can alter it). "
                f"Registered successful attempts matching current code/cwd/phase/protocol: {matching_ids!r}. "
                "These are diagnostic hints, not accepted evidence. Inspect their saved output "
                "and declared measurement before correcting attempt_id/evidence.ref; do not rewrite saved output. "
                + restore_hint +
                "If none supplies the required measurement, finish editing/staging before "
                "ar-attempt and rerun the component check.")
    if record.get("rc") == 0:
        from .runtime_evidence import bind_component_measurements, read_output
        try:
            read_output(record)
            from .checkpoint_artifacts import validate_checkpoint
            if record.get("checkpoint_error"):
                raise ValueError("runtime checkpoint could not be preserved: " + record["checkpoint_error"])
            if record.get("checkpoint"):
                validate_checkpoint(record["checkpoint"],
                                    artifact_paths=result.get("_reported_artifact_paths", []),
                                    repo_path=repo_path)
            elif result.get("_reported_artifact_paths"):
                raise ValueError("declared runtime artifacts have no measured checkpoint")
        except ValueError as exc:
            return str(exc)
        result["_metric_attempt_id"] = attempt_id
        result["checkpoint"] = record.get("checkpoint")
        if report:
            try:
                result["ablation_report"] = bind_component_measurements(report, record, records)
            except ValueError as exc:
                # The run and its primary scalar can still be verified. Do not
                # promote the component or erase the performance observation.
                result["component_evidence_error"] = str(exc)
    elif record.get("rc") == 1 and any(
        isinstance(item, dict) and item.get("verdict") == "unmet"
        and isinstance(item.get("evidence"), dict)
        and item["evidence"].get("source") == "runtime_attempt"
        and attempt_id in re.findall(
            r"\battempt-[A-Za-z0-9-]+\b", str(item["evidence"].get("ref") or ""))
        for item in report
    ):
        # A check may intentionally return 1 for an unmet property. Keep its
        # failure provenance, never treat it as a successful metric attempt.
        from .runtime_evidence import bind_component_measurements
        try:
            result["ablation_report"] = bind_component_measurements(
                report, record, records, diagnostic=True)
        except ValueError as exc:
            return str(exc)
        result["component_failed_attempt_id"] = attempt_id
    elif record.get("rc") == 1:
        return (f"runtime attempt {attempt_id} exited with code 1, despite the component "
                "not reporting an unmet property. DEBUG the execution/result wrapper using "
                f"host stderr {record.get('stderr_path')!r} and stdout {record.get('stdout_path')!r}; "
                "rerun after fixing the error. A claimed met verdict is not successful execution.")


class _AttemptAuthority:
    """Host-owned launch ledger and execution broker for sandbox attempts."""

    def __init__(self, backend: Any, handle: Any, repo_path: str, phase: str,
                 log_path: str, socket_path: str, last_attempt_path: str) -> None:
        self.backend = backend
        self.handle = handle
        self.repo_path = os.path.realpath(repo_path)
        self.phase = phase
        self.log_path = log_path
        self.socket_path = socket_path
        self.last_attempt_path = last_attempt_path
        self.artifact_paths = []
        self.control_path = None
        self.control_checkpoints = []
        self.control_is_frozen_baseline = False
        self.frozen_baseline_outcome = None
        self.execution_protocol = {}
        self.repository_contract = None
        self.uncertain_marker = os.path.join(os.path.dirname(self.repo_path), ".autoresearch-uncertain-execution.json")
        self.uncertain_attempt = None
        if os.path.exists(self.uncertain_marker):
            with open(self.uncertain_marker, encoding="utf-8") as pending:
                uncertain = json.load(pending)
            self.uncertain_attempt = uncertain["attempt_id"]
            # A transport failure can outlive the command that caused it.  Keep the
            # marker while that command could still be running, but do not block the
            # session forever once the original host ledger records the uncertain
            # finish and its recorded command deadline has necessarily expired.
            try:
                records = _attempt_records(uncertain["log_path"])
                record = records[self.uncertain_attempt]
                command_timeout = record.get("command_timeout_s")
                if "command_timeout_s" not in record:
                    command_timeout = (_agentic_hard_cap() if os.environ.get(
                        "AUTORESEARCH_AGENTIC_NO_TIME_LIMIT") != "1" else None)
                # No overall research deadline does not disable the host-enforced
                # timeout of an individual attempt. Never substitute a shorter
                # current cap for the original command's recorded allowance.
                expired = (type(command_timeout) in (int, float)
                           and math.isfinite(command_timeout) and command_timeout > 0
                           and time.time() >= float(record["started_at"])
                           + command_timeout + _uncertain_recovery_grace())
                if (record.get("launched") is True
                        and record.get("execution_uncertain") is True and expired):
                    uncertain.update({
                        "resolved_at": time.time(),
                        "resolution": "host execution deadline expired after uncertain host finish",
                    })
                    resolved = self.uncertain_marker.replace(
                        ".json", f".resolved-{self.uncertain_attempt}.json")
                    with open(resolved, "w", encoding="utf-8") as archived:
                        json.dump(uncertain, archived, sort_keys=True)
                    os.unlink(self.uncertain_marker)
                    self.uncertain_attempt = None
            except (KeyError, OSError, StopIteration, TypeError, ValueError,
                    json.JSONDecodeError):
                # Missing or inconsistent reconciliation evidence remains blocked.
                pass
        self.dependency_parent_state = {
            "repo_commit": _repo_commit(self.repo_path),
            "tracked_diff_hash_start": _tracked_repo_state(self.repo_path).get("tracked_diff_hash")}
        self._stop = threading.Event()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(socket_path)
        os.chmod(socket_path, 0o666)
        self._socket.listen(8)
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        # Match the canonical candidate/cwd paths after a symlink-based recovery.
        if self.control_path:
            self.control_path = os.path.realpath(self.control_path)
        self._thread.start()

    def stop(self, *, wait_for_inflight: bool = False) -> None:
        self._stop.set()
        self._socket.close()
        # Cleanup can run after setup fails but before start() is reached.  Python
        # raises when join() is called on a thread that was never started; in that
        # case closing the listening socket is already sufficient cleanup.
        if self._thread.ident is not None:
            self._thread.join(timeout=None if wait_for_inflight else 2)

    def resolve_uncertain_after_sandbox_close(self, sandbox_closed: bool) -> None:
        """Release an uncertain attempt only after its owning sandbox is gone."""
        if not sandbox_closed:
            return
        # Closing the sandbox can make an in-flight RPC fail after stop()'s
        # bounded join. Drain its receipt before checking the uncertain marker.
        if self._thread.ident is not None:
            self._thread.join()
        if not self.uncertain_attempt:
            return
        attempt_id = self.uncertain_attempt
        try:
            records = _attempt_records(self.log_path)
            record = records[attempt_id]
            if not (record.get("launched") is True
                    and record.get("execution_uncertain") is True):
                return
            with open(self.uncertain_marker, encoding="utf-8") as pending:
                uncertain = json.load(pending)
            if uncertain.get("attempt_id") != attempt_id:
                return
            uncertain.update({
                "resolved_at": time.time(),
                "resolution": "owning sandbox definitively closed after uncertain host finish",
            })
            resolved = self.uncertain_marker.replace(
                ".json", f".resolved-{attempt_id}.json")
            with open(resolved, "w", encoding="utf-8") as archived:
                json.dump(uncertain, archived, sort_keys=True)
            os.unlink(self.uncertain_marker)
            self.uncertain_attempt = None
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            # Ambiguous evidence remains blocked for the ordinary expiry recovery.
            return

    def _append(self, event: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _capture_command_sources(self, argv: list[str], cwd: str,
                                 discover_dependencies: bool = True) -> list[dict]:
        """Bind explicit code-source arguments and statically named shell dependencies."""
        paths = list(dict.fromkeys(arg for arg in argv
            if arg.endswith((".py", ".sh", ".go", ".rs", ".c", ".cc", ".cpp",
                             ".cu", ".cuh", ".h", ".hpp"))
            and not arg.startswith("-") and "=" not in arg))
        if not paths:
            return []
        code = r"""import hashlib,json,re,shlex,sys
from pathlib import Path
out=[]; seen=set(); pending=json.loads(sys.argv[1]); discover=json.loads(sys.argv[2])
while pending and len(out)<8:
 p=Path(pending.pop(0)).resolve()
 if str(p) in seen: continue
 seen.add(str(p)); item={'path':str(p)}
 try:
  with p.open('rb') as f:
   data=f.read(24001); f.seek(0); digest=hashlib.sha256()
   for chunk in iter(lambda:f.read(65536),b''): digest.update(chunk)
   sha=digest.hexdigest()
  item.update(sha256=sha,source=data[:24000].decode('utf-8',errors='replace'),truncated=len(data)>24000)
  if discover and p.suffix=='.sh':
   variables={}
   def literal(value):
    value=re.sub(r'\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)\})',
                 lambda m:variables.get(m[1] or m[2],m[0]),value)
    return None if any(c in value for c in ('$', '`', '\\')) else value
   for line in item['source'].splitlines():
    try: words=shlex.split(line,comments=True)
    except ValueError: continue
    if words[:1]==['export']: words=words[1:]
    if len(words)==1 and re.fullmatch(r'[A-Za-z_]\w*=.*',words[0]):
     key,value=words[0].split('=',1)
     raw_value=line.split('=',1)[1].lstrip()
     if '\\' in raw_value:
      variables.pop(key,None); continue
     if not raw_value.startswith("'"):
      value=literal(value)
     variables.pop(key,None)
     if value is not None and not any(c in value for c in ('$', '`', '\\')):
      variables[key]=value
     continue
    substitution=re.fullmatch(r'[A-Za-z_]\w*=(?:\$\(|`)([^`)]+)',words[0]) if words else None
    if substitution:
     words[0]=substitution[1]
    # A shell invocation can begin with environment assignments.
    while words and re.fullmatch(r'[A-Za-z_]\w*=.*',words[0]): words=words[1:]
    # Assignments and copy destinations are not script invocations.
    words=[literal(word) for word in words]
    if words[:1] in (['exec'],['nohup']): words=words[1:]
    if not words or words[0] is None: continue
    command=Path(words[0]).name
    if command=='timeout':
     words=words[1:]
     while words and words[0] is not None and words[0].startswith('-'):
      flag=words.pop(0)
      if flag=='--': break
      if flag in ('-k','--kill-after','-s','--signal'):
       words=words[1:]
      elif (flag not in ('--foreground','--preserve-status','-v','--verbose')
            and not flag.startswith(('--kill-after=','--signal='))):
       words=[]; break
     if (not words or words[0] is None
         or not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)[smhd]?',words[0])): continue
     words=words[1:]
     if not words or words[0] is None: continue
     command=Path(words[0]).name
    if re.fullmatch(r'python\d*(?:\.\d+)?|bash|sh|zsh',command):
     words=words[1:]
    elif command.endswith(('.py','.sh')):
     words=words[:1]
    else: continue
    for value in words:
     if (value is not None and value.endswith(('.py','.sh')) and Path(value).is_absolute()
         and not any(c in value for c in ('$', '`', '\\', '='))
         and Path(value).is_file()):
      pending.append(value)
 except OSError as exc: item['error']=type(exc).__name__
 out.append(item)
print(json.dumps(out))
"""
        try:
            reply = self.backend.execute_command(
                self.handle, shlex.join(["python3", "-c", code, json.dumps(paths),
                                        json.dumps(discover_dependencies)]),
                cwd=cwd, timeout=30)
            captured = json.loads(reply.stdout)
            return captured if isinstance(captured, list) else []
        except Exception:  # unavailable probe source remains unverified, not fabricated
            return []

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._socket.accept()
            except (OSError, socket.timeout):
                continue
            with connection:
                response = self._handle(connection)
                try:
                    connection.sendall(json.dumps(response).encode() + b"\n")
                except OSError:
                    pass

    def _handle(self, connection: socket.socket) -> dict:
        raw = b""
        while b"\n" not in raw and len(raw) <= 1024 * 1024:
            chunk = connection.recv(65536)
            if not chunk:
                break
            raw += chunk
        try:
            if self.uncertain_attempt:
                raise ValueError("previous attempt execution status is unknown: " + self.uncertain_attempt
                                 + "; reconcile the existing job before launching another experiment")
            request = json.loads(raw.split(b"\n", 1)[0])
            argv = request["argv"]
            cwd = os.path.realpath(request["cwd"])
            if (set(request) != {"argv", "cwd"}
                    or not isinstance(argv, list) or not argv
                    or any(not isinstance(token, str) or not token for token in argv)):
                raise ValueError("invalid attempt request")
            roots = [self.repo_path] + ([self.control_path] if self.control_path else [])
            if not any(os.path.commonpath([root, cwd]) == root for root in roots):
                raise ValueError(
                    f"attempt cwd {cwd!r} is outside the authorized checkouts {roots!r}. "
                    "Change directory to the candidate checkout (or the exact parent control "
                    "for a control measurement), then run ar-attempt there. An absolute script "
                    "path does not change cwd. Do not unset AUTORESEARCH_ATTEMPT_SOCKET.")
            from .runtime_evidence import execution_file_evidence
            execution_root = (self.control_path if self.control_path and
                              os.path.commonpath([self.control_path, cwd]) == self.control_path else self.repo_path)
            execution_files = execution_file_evidence(execution_root, self.execution_protocol)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"attempt_id": None, "rc": 126, "stdout": "", "stderr": str(exc)}
        records = _attempt_records(self.log_path)
        if (_variant_cap() and self.phase in {"method", "combined"}
                and len(records) >= _variant_cap()):
            return {
                "attempt_id": None, "rc": 125, "stdout": "",
                "stderr": f"AutoResearch: sandbox attempt cap ({_variant_cap()}) reached; "
                          "refusing another train/eval run.\n",
            }
        attempt_id = f"attempt-{time.time_ns()}-{uuid.uuid4().hex}"
        started = time.time_ns() / 1_000_000_000
        execution_root = (self.control_path if self.control_path and
                          os.path.commonpath([self.control_path, cwd]) == self.control_path else self.repo_path)
        commit = _repo_commit(execution_root)
        diff_hash = _tracked_repo_state(execution_root).get("tracked_diff_hash")
        command_sources = self._capture_command_sources(argv, cwd)
        attempt_timeout = (_agentic_hard_cap() if self.phase == "baseline" else
                           _per_run_cap(self.repository_contract))
        verifier_timeout = _verifier_timeout(self.repository_contract)
        launch = {
            "event": "launch", "attempt_id": attempt_id, "phase": self.phase,
            "argv": argv, "cwd": cwd, "started_at": started,
            "command_timeout_s": attempt_timeout,
            "verifier_timeout_s": verifier_timeout,
            "repo_commit": commit, "tracked_diff_hash_start": diff_hash,
            "dependency_parent_state": self.dependency_parent_state,
            "dependency_parent_cwd": (os.path.normpath(os.path.join(self.control_path, os.path.relpath(cwd, self.repo_path)))
                                      if self.control_path and execution_root == self.repo_path else None),
            "execution_protocol": self.execution_protocol,
            "execution_files_start": execution_files,
            "command_sources": command_sources,
        }
        if execution_root == self.control_path and self.control_is_frozen_baseline:
            launch["frozen_baseline_control"] = True
            launch["frozen_baseline_outcome"] = self.frozen_baseline_outcome
        self._append(launch)
        with open(self.last_attempt_path, "w", encoding="utf-8") as handle:
            handle.write(attempt_id + "\n")
        candidate_backup = None
        execution_uncertain = False
        try:
            if execution_root == self.control_path and any(os.path.isabs(p) for p in self.artifact_paths):
                from .checkpoint_artifacts import snapshot_checkpoint, restore_external
                if not self.control_checkpoints:
                    raise ValueError("exact parent runtime artifacts are unavailable; restore the baseline checkpoint")
                candidate_backup = snapshot_checkpoint(self.repo_path, attempt_id + "-before-control",
                    self.artifact_paths, backend=self.backend, handle=self.handle)
                restore_external(self.control_checkpoints, self.backend, self.handle, execution_root)
            # Runtime outputs must not replace an earlier candidate's saved weights/results.
            output_dir = os.path.join(os.path.dirname(self.repo_path), "artifacts", "attempts", attempt_id)
            os.makedirs(output_dir, exist_ok=False)
            os.chmod(output_dir, 0o777)
            executor = getattr(
                self.backend, "execute_attempt_command", self.backend.execute_command)
            # Enforce the host budget inside the sandbox before the transport
            # deadline. Killing only the RPC shell leaves its process group alive.
            cleanup = os.path.join(os.path.dirname(self.last_attempt_path), "bin", "timeout-cleanup")
            completed = executor(
                self.handle,
                # Empty proxy values (and even NO_PROXY alone) are not equivalent to absent
                # proxy configuration for every client
                # (notably NLTK's pathsec guard).  Repository work is direct, so remove the
                # proxy variables from the actual measured subprocess rather than merely
                # overriding them with empty strings at container creation time.
                shlex.join([
                    "env",
                    "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
                    "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
                    "-u", "NO_PROXY", "-u", "no_proxy",
                    "AUTORESEARCH_ATTEMPT_ACTIVE=1",
                    f"AUTORESEARCH_ATTEMPT_ID={attempt_id}",
                    f"AUTORESEARCH_ATTEMPT_ARTIFACT_DIR={output_dir}",
                    *([f"AUTORESEARCH_VERIFIER_TIMEOUT_S={verifier_timeout}"]
                      if verifier_timeout is not None else []),
                    cleanup, "--kill-after=2s", f"{attempt_timeout}s", *argv,
                ]),
                cwd=cwd, timeout=attempt_timeout + 10)
            rc = int(getattr(completed, "exit_code", 1))
            stdout = str(getattr(completed, "stdout", "") or "")
            stderr = str(getattr(completed, "stderr", "") or "")
        except Exception as exc:  # noqa: BLE001 -- failed host execution is still an attempt
            execution_uncertain = bool(getattr(exc, "execution_uncertain", False))
            if execution_uncertain:
                self.uncertain_attempt = attempt_id
                with open(self.uncertain_marker, "w", encoding="utf-8") as pending:
                    json.dump({"attempt_id": attempt_id, "cwd": cwd, "log_path": self.log_path,
                               "reason": str(exc), "requires_reconciliation": True}, pending)
            rc, stdout, stderr = 126, "", f"{type(exc).__name__}: {exc}\n"
        end_state = _tracked_repo_state(execution_root)
        # Rehash exactly the start set; a wrapper can legitimately create new scripts.
        command_sources_end = self._capture_command_sources(
            [*argv, *(item["path"] for item in command_sources)], cwd, False)
        command_sources_stable = bool(command_sources) and all(
            item.get("sha256") for item in command_sources + command_sources_end
        ) and command_sources == command_sources_end
        try:
            if (command_sources and command_sources_end
                    and all(item.get("sha256") for item in command_sources + command_sources_end)
                    and not command_sources_stable):
                changed_paths = [item["path"] for item in command_sources
                                 if item not in command_sources_end]
                raise ValueError(
                    "measurement entrypoint source changed during the attempt: "
                    + ", ".join(changed_paths)
                    + "; finish staging before ar-attempt and keep captured scripts unchanged until it returns")
            execution_files_end = execution_file_evidence(execution_root, self.execution_protocol)
            if execution_files_end != execution_files:
                raise ValueError("declared execution sources changed during the attempt")
        except (OSError, ValueError) as exc:
            execution_files_end = {}
            rc = 126
            stderr += "\nExecution file contract: " + str(exc)
        from .runtime_evidence import save_output
        output_evidence = save_output(os.path.dirname(self.log_path), attempt_id, stdout, stderr)
        if rc == 0:
            from .checkpoint_artifacts import snapshot_checkpoint
            try:
                # A sandbox can atomically replace ar_result.json with a private
                # mkstemp file owned by its UID.  Handoff must precede the host read,
                # otherwise its artifact declarations are silently missed.
                _prepare_checkpoint_handoff(
                    self.backend, self.handle, execution_root, lambda *_: None)
                artifact_paths = list(self.artifact_paths)
                try:
                    with open(os.path.join(execution_root, _RESULT_BASENAME), encoding="utf-8") as f:
                        reported = json.load(f)
                except (OSError, ValueError):
                    reported = None
                if isinstance(reported, dict) and reported.get("attempt_id") == attempt_id:
                    extra = reported.get("artifact_paths", [])
                    if not isinstance(extra, list) or not all(isinstance(p, str) and p for p in extra):
                        raise ValueError("reported artifact_paths must be a list of file paths")
                    artifact_paths.extend(extra)
                output_evidence["checkpoint"] = snapshot_checkpoint(
                    execution_root, attempt_id, list(dict.fromkeys(artifact_paths)),
                    backend=self.backend, handle=self.handle, handoff_prepared=True)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                output_evidence["checkpoint_error"] = str(exc)
        if candidate_backup and not execution_uncertain:
            from .checkpoint_artifacts import restore_external
            try:
                restore_external(candidate_backup, self.backend, self.handle, self.repo_path)
            except (OSError, ValueError) as exc:
                output_evidence["checkpoint_error"] = "candidate artifacts were not restored after control: " + str(exc)
        self._append({
            "event": "finish", "attempt_id": attempt_id, "phase": self.phase,
            "ended_at": time.time_ns() / 1_000_000_000, "rc": rc,
            "repo_commit_end": _repo_commit(execution_root),
            "tracked_diff_hash_end": end_state.get("tracked_diff_hash"),
            "execution_files_end": execution_files_end,
            "command_sources_stable": command_sources_stable,
            "execution_uncertain": execution_uncertain,
            **output_evidence,
        })
        return {"attempt_id": attempt_id, "rc": rc, "stdout": stdout, "stderr": stderr}


def _restore_reported_attempt_checkpoint(authority, result_path, emit):
    """Recover the explicitly reported measured source after a provider failure."""
    if authority.phase != "method" or not os.path.isfile(result_path):
        return
    from .checkpoint_artifacts import snapshot_checkpoint, validate_checkpoint, restore_external
    from .runtime_evidence import measured_result
    with open(result_path, encoding="utf-8") as saved:
        payload = json.load(saved)
    if not isinstance(payload, dict):
        raise ValueError("reported result must be a JSON object")
    attempt_id = payload.get("attempt_id")
    records = _attempt_records(authority.log_path)
    record = records.get(attempt_id) or {}
    repo_path = os.path.realpath(authority.repo_path)
    cwd = os.path.realpath(str(record.get("cwd") or ""))
    if (record.get("phase") != "method" or not record.get("launched")
            or not record.get("finished") or record.get("rc") != 0
            or record.get("execution_uncertain")
            or os.path.commonpath([repo_path, cwd]) != repo_path
            or record.get("repo_commit") != _repo_commit(repo_path)
            or record.get("repo_commit_end") != record.get("repo_commit")
            or record.get("tracked_diff_hash_start") != record.get("tracked_diff_hash_end")):
        raise ValueError("reported attempt cannot restore this checkout")
    measured = measured_result(record, authority.execution_protocol, "method")
    if measured["valid"] is not True or measured["value"] != _coerce_metric(payload.get("method")):
        raise ValueError("reported value does not match a valid native measurement")
    if _bind_component_evidence_attempt({"_metric_attempt_id": attempt_id}, records,
                                        repo_path, "method") is None:
        return
    checkpoint = record.get("checkpoint")
    manifest = validate_checkpoint(checkpoint)
    if manifest.get("attempt_id") != attempt_id:
        raise ValueError("checkpoint belongs to another attempt")
    backup = snapshot_checkpoint(repo_path, "provider-unmeasured-" + attempt_id,
        authority.artifact_paths, backend=authority.backend, handle=authority.handle)
    emit("agentic_salvage", {"stage": "unmeasured_checkpoint_saved", "checkpoint": backup})
    _apply_prior_worktree(repo_path, checkpoint)
    restore_external([checkpoint], authority.backend, authority.handle, repo_path)
    error = _bind_component_evidence_attempt({"_metric_attempt_id": attempt_id}, records,
                                            repo_path, "method")
    if error:
        raise ValueError(error)
    emit("agentic_salvage", {"stage": "measured_checkpoint_restored", "attempt_id": attempt_id,
                             "checkpoint": checkpoint, "unmeasured_checkpoint": backup})


_OPTIMIZER_ATTEMPT_GUARD = r'''
import builtins
import os
import sys

if (os.environ.get("AUTORESEARCH_ATTEMPT_ACTIVE") != "1"
        and os.environ.get("AUTORESEARCH_HOST_NATIVE_EXECUTION") != "1"):
    original_import = builtins.__import__

    def require_attempt(optimizer, args, kwargs):
        raise RuntimeError(
            "AutoResearch: optimizer updates require a registered experiment. "
            "Run this training through ar-attempt; do not disable execution guards.")

    def guarded_import(*args, **kwargs):
        module = original_import(*args, **kwargs)
        optimizer = sys.modules.get("torch.optim.optimizer")
        register = getattr(optimizer, "register_optimizer_step_pre_hook", None)
        if register is not None:
            builtins.__import__ = original_import
            register(require_attempt)
        return module

    builtins.__import__ = guarded_import
'''


def _install_attempt_guard(
    workroot: str, repo_path: str, result_path: str, *, authoritative: bool = False,
) -> tuple[str, str]:
    """Install explicit and legacy launch guards for sandbox train/eval attempts.

    Explicit ``ar-attempt`` launches count failures and timeout exit 124 against the method cap.
    Plain ``timeout`` also bounds inspection/setup, so it must not classify these as experiments.
    The legacy combined phase retains its result-file-based accounting.
    """
    round_name = os.path.basename(os.path.normpath(repo_path)) or "repo"
    control = os.path.join(workroot, ".sandbox-control", round_name)
    bindir = os.path.join(control, "bin")
    if authoritative:
        host_control = tempfile.mkdtemp(
            prefix=f".{round_name}-attempt-authority-",
            dir=os.path.dirname(os.path.realpath(workroot)),
        )
        log_path = os.path.join(host_control, "attempts.jsonl")
    else:
        log_path = os.path.join(control, "attempts.tsv")
    os.makedirs(bindir, exist_ok=True)
    os.chmod(control, 0o777)
    os.chmod(bindir, 0o777)
    # The host user creates this file, while OpenHands appends as uid 10001 in the bind mount.
    # Pre-create it explicitly writable; otherwise a 755 host directory makes the live guard silent.
    with open(log_path, "a", encoding="utf-8"):
        pass
    os.chmod(log_path, 0o600 if authoritative else 0o666)
    last_attempt_path = os.path.join(control, "last_attempt_id")
    with open(last_attempt_path, "a", encoding="utf-8"):
        pass
    os.chmod(last_attempt_path, 0o666)
    if authoritative:
        python_guard = os.path.join(control, "python")
        os.makedirs(python_guard, exist_ok=True)
        with open(os.path.join(python_guard, "sitecustomize.py"), "w", encoding="utf-8") as handle:
            handle.write(_OPTIMIZER_ATTEMPT_GUARD)
    launcher = os.path.join(bindir, "ar-attempt")
    launcher_script = r'''#!/usr/bin/env python3
import fcntl
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time


def append_event(path, event):
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def launched_count(path):
    ids = set()
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    if line.strip():
                        ids.add("legacy:" + str(len(ids)))
                    continue
                if (event.get("attempt_id")
                        and event.get("phase", "method") in {"method", "combined"}):
                    ids.add(str(event["attempt_id"]))
    except OSError:
        pass
    return len(ids)


def repo_state(repo):
    empty_hash = "sha256:" + hashlib.sha256(b"").hexdigest()
    if not repo:
        return None, empty_hash
    base = ["git", "-c", "safe.directory=*", "-c", "core.fileMode=false", "-C", repo]
    try:
        head = subprocess.run(base + ["rev-parse", "HEAD"], capture_output=True,
                              timeout=10, check=False).stdout.decode(errors="replace").strip()
        outputs = ["ar_result.json", "analysis/variants.jsonl"]
        owned = subprocess.run(base + ["ls-tree", "-r", "--name-only", "HEAD", "--"] + outputs,
                               capture_output=True, timeout=10, check=True).stdout.decode().splitlines()
        excludes = [":(top,exclude,literal)" + path for path in outputs if path not in owned]
        diff = subprocess.run(base + ["diff", "--no-ext-diff", "--binary", "HEAD", "--"] + excludes,
                              capture_output=True, timeout=30, check=False).stdout
        return (head or None), "sha256:" + hashlib.sha256(diff).hexdigest()
    except Exception:
        return None, empty_hash


if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
    print("usage: ar-attempt <train-or-eval-command> [args...]")
    print("Brokered commands do not inherit terminal exports or venv activation. "
          "Use ar-attempt env NAME=VALUE /absolute/interpreter script.py, "
          "or export/activate inside the launched driver. Change cwd before ar-attempt.")
    raise SystemExit(0)


if os.environ.get("AUTORESEARCH_ATTEMPT_ACTIVE") == "1":
    print("AutoResearch: nested ar-attempt is not supported; the outer attempt "
          "already owns this execution. Run parent/control and candidate as "
          "separate top-level ar-attempt commands, then cite their saved attempt "
          "IDs. Do not call ar-attempt from inside an ar-attempt wrapper.", file=sys.stderr)
    raise SystemExit(126)

authority_socket = os.environ.get("AUTORESEARCH_ATTEMPT_SOCKET", "")
if authority_socket:
    if len(sys.argv) < 2:
        print("usage: ar-attempt <train-or-eval-command> [args...]", file=sys.stderr)
        raise SystemExit(2)
    request = json.dumps({
        "argv": sys.argv[1:], "cwd": os.path.realpath(os.getcwd()),
    }, separators=(",", ":")).encode() + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(authority_socket)
            client.sendall(request)
            response_raw = b""
            while b"\n" not in response_raw:
                chunk = client.recv(65536)
                if not chunk:
                    break
                response_raw += chunk
        response = json.loads(response_raw.split(b"\n", 1)[0])
    except Exception as exc:
        print(f"AutoResearch attempt authority unavailable: {exc}", file=sys.stderr)
        raise SystemExit(126)
    if response.get("attempt_id"):
        print(f"AutoResearch attempt_id={response['attempt_id']}", file=sys.stderr)
    sys.stdout.write(str(response.get("stdout") or ""))
    sys.stderr.write(str(response.get("stderr") or ""))
    raise SystemExit(int(response.get("rc", 126)))


log = os.environ.get("AUTORESEARCH_ATTEMPT_LOG", "")
last_path = os.environ.get("AUTORESEARCH_LAST_ATTEMPT", "")
repo = os.environ.get("AUTORESEARCH_REPO_PATH", "")
phase = os.environ.get("AUTORESEARCH_EXECUTION_PHASE", "combined")
try:
    cap = int(os.environ.get("AUTORESEARCH_AGENTIC_VARIANT_CAP", "20"))
    cap = cap if cap >= 0 else 20
except ValueError:
    cap = 20
if not log or len(sys.argv) < 2:
    print("usage: ar-attempt <train-or-eval-command> [args...]", file=sys.stderr)
    raise SystemExit(2)
attempt_id = f"attempt-{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
started = time.time()
commit, diff_hash = repo_state(repo)
with open(log, "a+", encoding="utf-8") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.seek(0)
    count = 0
    seen = set()
    for line in handle:
        try:
            event = json.loads(line)
            key = (str(event.get("attempt_id") or "")
                   if event.get("phase", "method") in {"method", "combined"} else "")
        except (TypeError, ValueError):
            key = "legacy:" + str(count) if line.strip() else ""
        if key and key not in seen:
            seen.add(key)
            count += 1
    if cap and phase in {"method", "combined"} and count >= cap:
        print(f"AutoResearch: sandbox attempt cap ({cap}) reached; refusing another train/eval run.",
              file=sys.stderr)
        raise SystemExit(125)
    event = {
        "event": "launch", "attempt_id": attempt_id, "phase": phase,
        "argv": sys.argv[1:], "cwd": os.path.realpath(os.getcwd()),
        "started_at": started, "repo_commit": commit,
        "tracked_diff_hash_start": diff_hash,
    }
    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
if last_path:
    with open(last_path, "w", encoding="utf-8") as handle:
        handle.write(attempt_id + "\n")
print(f"AutoResearch attempt_id={attempt_id}", file=sys.stderr)

child_env = os.environ.copy()
child_env["AUTORESEARCH_ATTEMPT_ACTIVE"] = "1"
child_env["AUTORESEARCH_ATTEMPT_ID"] = attempt_id
rc = 126
try:
    artifact_root = child_env.get("AUTORESEARCH_ARTIFACT_DIR")
    if artifact_root:
        output_dir = os.path.join(artifact_root, "attempts", attempt_id)
        os.makedirs(output_dir, exist_ok=False)
        child_env["AUTORESEARCH_ATTEMPT_ARTIFACT_DIR"] = output_dir
    completed = subprocess.run(sys.argv[1:], env=child_env, check=False)
    rc = completed.returncode
except FileNotFoundError:
    rc = 127
except KeyboardInterrupt:
    rc = 130
finally:
    ended = time.time()
    end_commit, end_diff_hash = repo_state(repo)
    append_event(log, {
        "event": "finish", "attempt_id": attempt_id, "phase": phase,
        "ended_at": ended, "rc": rc, "repo_commit_end": end_commit,
        "tracked_diff_hash_end": end_diff_hash,
    })
raise SystemExit(rc)
'''
    with open(launcher, "w", encoding="utf-8") as f:
        f.write(launcher_script)
    os.chmod(launcher, 0o755)

    timeout_cleanup = os.path.join(bindir, "timeout-cleanup")
    cleanup_script = r'''#!/usr/bin/python3
import os
import signal
import subprocess
import sys
import time

process = subprocess.Popen(
    ["/usr/bin/timeout", "--foreground", *sys.argv[1:]],
    start_new_session=True,
)

def interrupted(signum, _frame):
    # Default SIGTERM/SIGHUP handling skips finally and leaves the new group alive.
    raise SystemExit(128 + signum)

try:
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    returncode = process.wait()
except KeyboardInterrupt:
    returncode = 130
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    # A benchmark parent can fail while a server/worker child remains alive.  Every command in
    # this helper owns a fresh process group, so cleanup cannot touch another sandbox attempt.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
raise SystemExit(returncode)
'''
    with open(timeout_cleanup, "w", encoding="utf-8") as f:
        f.write(cleanup_script)
    os.chmod(timeout_cleanup, 0o755)

    wrapper = os.path.join(bindir, "timeout")
    script = """#!/bin/sh
set -u
result=${AUTORESEARCH_ATTEMPT_RESULT:-}
log=${AUTORESEARCH_ATTEMPT_LOG:-}
cap=${AUTORESEARCH_AGENTIC_VARIANT_CAP:-20}
if ! [ "$cap" -ge 0 ] 2>/dev/null; then
    cap=20
fi
phase=${AUTORESEARCH_EXECUTION_PHASE:-combined}
cleanup=$(dirname "$0")/timeout-cleanup
if [ "${AUTORESEARCH_ATTEMPT_ACTIVE:-}" = "1" ]; then
    exec "$cleanup" "$@"
fi
if [ "$phase" = "baseline" ]; then
    exec "$cleanup" "$@"
fi
if [ "$phase" = "method" ]; then
    exec "$cleanup" "$@"
fi
if [ -n "$result" ] && [ -s "$result" ] && [ -n "$log" ]; then
    done_count=0
    if [ -f "$log" ]; then
        done_count=$(wc -l < "$log")
    fi
    if [ "$cap" -gt 0 ] && [ "$done_count" -ge "$cap" ]; then
        printf 'AutoResearch: sandbox attempt cap (%s) reached; refusing another train/eval run.\n' "$cap" >&2
        exit 125
    fi
    "$cleanup" "$@"
    rc=$?
    printf '%s\t%s\n' "$rc" "$*" >> "$log"
    exit "$rc"
fi
exec "$cleanup" "$@"
"""
    with open(wrapper, "w", encoding="utf-8") as f:
        f.write(script)
    os.chmod(wrapper, 0o755)

    # The uv cache is shared by every AutoResearch sandbox.  An agent trying to "fix" its own
    # environment with `uv cache clean` otherwise deletes tens of gigabytes needed by other runs.
    uv_wrapper = os.path.join(bindir, "uv")
    uv_script = """#!/bin/sh
set -u
if [ "${1:-}" = "cache" ] && { [ "${2:-}" = "clean" ] || [ "${2:-}" = "prune" ]; }; then
    printf 'AutoResearch: refusing to delete the shared package cache.\n' >&2
    exit 126
fi
for real_uv in /usr/local/bin/uv /usr/bin/uv; do
    if [ -x "$real_uv" ]; then
        exec "$real_uv" "$@"
    fi
done
printf 'AutoResearch: real uv executable not found.\n' >&2
exit 127
"""
    with open(uv_wrapper, "w", encoding="utf-8") as f:
        f.write(uv_script)
    os.chmod(uv_wrapper, 0o755)
    return bindir, log_path


def _carry_enabled() -> bool:
    """#75 跨轮引导开关(默认开)。AUTORESEARCH_CARRY_DIRECTION=0 关掉(行为回到 per-round)。"""
    return os.environ.get("AUTORESEARCH_CARRY_DIRECTION", "1").strip().lower() not in ("0", "false", "")


_URL_RE = __import__("re").compile(r"https?://[\w.-]+/[\w.-]+/[\w.-]+", __import__("re").I)


def _extract_repo_urls(text: str) -> list:
    """#76 一等公民多仓库联动: 从请求/主题里按出现顺序抽出所有 github 式仓库 URL(去重、剥尾部标点)。
    调用方约定: 第一个=要改进的主仓库, 其余=工具仓库(clone 进 tools/ 供 agent 评测/增强主仓库)。
    这样"improve <A> using <B>"这类请求不必靠 env 就能触发多仓库联动。"""
    seen, out = set(), []
    for m in _URL_RE.finditer(text or ""):
        u = m.group(0).rstrip("/.,;)]")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _tool_repo_urls(explicit: tuple = ()) -> tuple:
    """#76 多仓库联动: 额外的"工具仓库"URL 列表 —— clone 进 workroot/tools/<name>,供沙箱 agent
    用来评测/增强主仓库(如 NVlabs/SOLAR 给每个 kernel 算 H800 speed-of-light,再和实测比出百分比)。
    显式参数优先,否则读环境 AUTORESEARCH_TOOL_REPOS(逗号分隔 URL)。空 -> 单仓库行为完全不变。

    只覆盖本地(L20)路径的进程内 env;远程 dgx 路径要生效需把该 env 透传到 agentic_run.py(follow-up)。
    discover 自动识别"需要哪个工具仓库" + UI 确认语义从"N 选 1"改成"1 主 + M 工具" = 产品决策,留后续。"""
    # union: 显式传入(如从请求里抽出的额外 URL)+ 环境变量,都生效、去重、保序
    raw = os.environ.get("AUTORESEARCH_TOOL_REPOS", "").strip()
    env_urls = [u.strip() for u in raw.split(",") if u.strip()] if raw else []
    seen, out = set(), []
    for u in list(explicit) + env_urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return tuple(out)


def _persist_carry(repo_path: str, direction: str,
                   memory_scope: str | None = None) -> None:
    """把本轮分析结论也写到 workroot 级 carry 文件,供下一轮读(见 _CARRY_BASENAME)。
    repo_path 恒为 workroot 的直接子目录,故 workroot = os.path.dirname(repo_path)。Fail-open。"""
    if not _carry_enabled():
        return
    try:
        carry = os.path.join(os.path.dirname(repo_path), _carry_basename(memory_scope))
        tmp = carry + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(direction)
        os.replace(tmp, carry)
    except Exception:  # noqa: BLE001 -- advisory, never crash the run
        pass


# ---- WITHIN-SESSION analysis agent (老师 2026-08-07): the sandbox agent must NOT iterate blind. --
# A SEPARATE analysis agent runs alongside the sandbox session: it reads the variants the agent has
# tried so far (analysis/variants.jsonl), and, against the goal + method + baseline, writes a
# CONCLUSION + next direction outside the candidate checkout. The sandbox reads it before each new
# variant, so its iteration is DIRECTED (not闭着眼瞎改). Fail-open: any error -> no direction, the
# agent just continues (never blocks the run).
_ANALYSIS_SYS = (
    "You are a SEPARATE analysis agent (NOT the coding agent) inside an automated ML-research loop. "
    "The coding agent is trying variants in a sandbox. When a CONFIRMED METHOD is given, the research "
    "team already deliberated and agreed on it. In this sandbox, help implement that Method; "
    "a Method change needs formal host review. Given the GOAL, the CONFIRMED METHOD (if any), "
    "and current host corrections, consider every declared component, not only the approach. "
    "Your advice cannot accept a candidate, waive a component, or override a host correction; "
    "only the host's bound review establishes acceptance or a Method revision. "
    "the BASELINE, and the "
    "VARIANTS tried so far (each: what changed + measured result), analyze like a scientist: which "
    "variant best ADVANCES the confirmed method AND moved the metric, and WHY; which variants DRIFTED "
    "off the method (e.g. blind hyperparameter search when the method calls for a structural/kernel "
    "change). A method mismatch alone is not proof of a dead end: preserve a measured alternative "
    "as a hypothesis for host legality, comparability, and Method review; variant self-reports do "
    "not establish those facts. If evidence refutes the Method's core, say which assumption failed "
    "and request formal revision rather than suggesting cosmetic edits. Otherwise give ONE concrete "
    "next design move within the confirmed Method, so the agent designs its next variant FROM your "
    "conclusion instead of guessing or drifting. Reply with a SHORT plain-text "
    "conclusion + next-direction (no JSON, <=6 sentences)."
)


def _call_analysis_llm(user: str, timeout: int = 90) -> str | None:
    """One plain flash call to the gateway for the within-session analysis. urllib (no deps),
    fail-open (returns None on any error). Uses the sandbox gateway env (same as the agent)."""
    import urllib.request
    base = os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("AUTORESEARCH_SANDBOX_BASE_URL")
    key = os.environ.get("OPENROUTER_KEY")
    model = os.environ.get("AUTORESEARCH_SANDBOX_MODEL") or os.environ.get("KIMI_MODEL")
    if not (base and key and model):
        return None
    payload = {"model": model,
               "temperature": float(os.environ.get("AUTORESEARCH_LLM_TEMPERATURE", "0.3")),
               "messages": [{"role": "system", "content": _ANALYSIS_SYS},
                            {"role": "user", "content": user}]}
    reasoning = (os.environ.get("AUTORESEARCH_SANDBOX_REASONING_EFFORT")
                 or os.environ.get("VIBE_REASONING_EFFORT"))
    if reasoning and reasoning != "default":
        payload["reasoning_effort"] = reasoning
    max_tokens = (os.environ.get("AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS")
                  or os.environ.get("AUTORESEARCH_LLM_MAX_TOKENS"))
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"}, method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=float(os.environ.get("AUTORESEARCH_SANDBOX_LLM_TIMEOUT", timeout))) as r:
            obj = json.loads(r.read().decode())
        return (obj["choices"][0]["message"]["content"] or "").strip() or None
    except Exception:  # noqa: BLE001 -- advisory only, never block the run
        return None


def _extract_confirmed_method(request: str) -> tuple:
    """从 request 抽出'CONFIRMED METHOD'块(web 流程 _method_directive 前置的、深思讨论出的 method),
    返回 (method_text, goal_without_method)。没有则 ('', request)。用户 2026-08-09: sidecar 的诊断不能
    脱离讨论出的 method —— 要把 method 喂进去,让它基于"变体结果 + 确认的 method"共同给下一步,而不是
    另开一个方向(如盲目 LR 搜索)让 agent 更困惑。bare 路径无 method 块 -> 优雅退回纯 goal。"""
    if not request:
        return "", request
    idx = request.find("CONFIRMED METHOD")
    if idx < 0:
        return "", request
    marker = "The confirmed method:"
    start = request.find(marker, idx)
    if start >= 0:
        payload = request[start + len(marker):].lstrip()
        try:
            method_obj, end = json.JSONDecoder().raw_decode(payload)
        except ValueError:
            pass
        else:
            if isinstance(method_obj, dict):
                end += len(request) - len(payload)
                return (request[idx:end].strip(),
                        (request[:idx].strip() + " " + request[end:].strip()).strip() or request)
    method = request[idx:idx + 1600].strip()             # 块=approach+changes_outline(前置在最前)
    goal = (request[:idx].strip() + " " + request[idx + 1600:].strip()).strip() or request
    return method, goal


def _analysis_direction_path(repo_path: str) -> str:
    # Host feedback changes asynchronously, including after measurement. Keep it
    # outside the candidate so it cannot invalidate the measured Git fingerprint.
    repo = os.path.abspath(repo_path)
    return os.path.join(os.path.dirname(repo), ".sandbox-control",
                        os.path.basename(repo), "direction.txt")


def _analyze_once(repo_path: str, request: str, metric_hint: str | None,
                  lower_is_better_hint: bool | None, emit, last: int = 0,
                  memory_scope: str | None = None) -> int:
    """One analysis pass. Material = the agent's tried-variants, taken from analysis/variants.jsonl
    IF the agent bothered to append them, ELSE (fallback) from ar_result.json's
    changes_summary/rationale/bottleneck -- which the agent MUST write to report its result, so it's
    far more reliable than variants.jsonl (flash agents often skip variants.jsonl entirely, or never
    even create it -- seen live 2026-08-08 on nano+speedrun). Writes host-side direction.txt + emits
    ``agentic_analysis``(source) on success / ``agentic_analysis_skip``(reason) so a no-direction run
    is diagnosable. ``last`` is a signature to avoid re-analyzing identical material. Fail-open."""
    adir = os.path.join(repo_path, "analysis")
    vfile = os.path.join(adir, "variants.jsonl")
    dfile = _analysis_direction_path(repo_path)
    rpath = os.path.join(repo_path, _RESULT_BASENAME)
    # baseline-only 占位的关键词(子串匹配): agent 只建了 baseline、还没试变体时的 changes_summary
    # (如 "No changes yet - baseline only" / "Original unmodified demo" / "unmodified baseline run"),
    # 这些不是变体,不该喂给分析。子串匹配比精确匹配稳(措辞每次不同)。
    _BASELINE_ONLY_KW = ("no change", "unmodified", "baseline only", "baseline measurement",
                         "original unmodified", "no modification")
    try:
        variants = []
        if os.path.exists(vfile):
            with open(vfile, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        variants.append(line[:400])
        baseline = "(unknown)"
        result_summary = ""
        diagnostic_only = False
        if os.path.exists(rpath):
            try:
                r = json.load(open(rpath, encoding="utf-8"))
                baseline = str(r.get("baseline"))
                diagnostic_only = (
                    r.get("method") is None
                    and bool(r.get("ablation_report"))
                )
                cs = str(r.get("changes_summary") or "").strip()
                if cs and not any(k in cs.lower() for k in _BASELINE_ONLY_KW):  # 排除 baseline-only
                    result_summary = (f"changes_summary: {cs[:1500]}\n"
                                      f"rationale: {str(r.get('rationale') or '')[:400]}\n"
                                      f"bottleneck: {str(r.get('bottleneck') or '')[:400]}").strip()
            except Exception:  # noqa: BLE001
                pass
        if diagnostic_only:
            emit("agentic_analysis_skip", {"reason": "diagnostic_only_result"})
            return last
        # material: 优先 variants.jsonl(>=2 行),否则退到 ar_result 的 changes_summary
        if len(variants) >= 2:
            material = "VARIANTS TRIED SO FAR (newest last):\n" + "\n".join(variants[-12:])
            source, sig = "variants.jsonl", len(variants)
        elif result_summary:
            material = "WHAT THE AGENT TRIED THIS ROUND (from its result summary):\n" + result_summary
            source, sig = "changes_summary", len(result_summary)
        else:
            return last   # 真没料(还在 baseline / baseline-only)
        if sig == last:
            return last
        dirn = "lower-is-better" if lower_is_better_hint else "higher-is-better"
        # 用户 2026-08-09: sidecar 诊断要锚定讨论出的 method(基于"变体结果 + 确认的 method"共同给下一步)
        method, goal = _extract_confirmed_method(request)
        method_block = (
            "CONFIRMED METHOD (the research team deliberated and agreed on THIS; your next direction "
            "for this sandbox should faithfully ADVANCE this method; a different Method requires "
            "formal host review. Preserve any measured off-method alternative for that review -- "
            "do NOT drift to unrelated generic tweaks like "
            f"blind LR/hyperparameter search unless the method itself calls for it):\n{method}\n\n"
            if method else "")
        anchor = "the CONFIRMED METHOD above" if method else "the goal"
        user = (f"GOAL AND CURRENT HOST DIRECTIVES:\n{goal}\n\n{method_block}METRIC: {metric_hint or '(the repo metric)'} "
                f"({dirn}). BASELINE: {baseline}.\n\n{material}\n\nAnalyze the tried variants AGAINST "
                f"{anchor}: which best fit it and moved the metric, which drifted off it. Then give ONE "
                f"concrete next direction within {anchor}, or identify evidence requiring formal "
                "Method review (not a generic tweak).")
        direction = _call_analysis_llm(user)
        if direction:
            os.makedirs(os.path.dirname(dfile), exist_ok=True)
            tmp = dfile + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(direction)
            os.replace(tmp, dfile)
            _persist_carry(repo_path, direction, memory_scope)  # 同一方法的下一轮再读
            emit("agentic_analysis", {"source": source, "direction": direction[:160]})
            return sig
        emit("agentic_analysis_skip", {"reason": "llm_returned_none", "source": source})
    except Exception as exc:  # noqa: BLE001 -- advisory, never crash the run
        emit("agentic_analysis_skip", {"reason": type(exc).__name__})
    return last


def _analysis_sidecar(repo_path: str, request: str, metric_hint: str | None,
                      lower_is_better_hint: bool | None, stop_event, emit,
                      memory_scope: str | None = None) -> None:
    """Background loop: while the sandbox session runs, turn the agent's tried-variants into a
    directed CONCLUSION. Polls analysis/variants.jsonl every ~15s via _analyze_once. Pure advisory +
    fail-open. NOTE: flash agents often only APPEND variants at the very END of a run, so the caller
    ALSO calls _analyze_once once more after the agent finishes (belt-and-suspenders); true
    cross-round guidance (persist direction to the next round's prompt) is a follow-up."""
    adir = os.path.join(repo_path, "analysis")
    try:
        os.makedirs(adir, exist_ok=True)
        # The host creates this bind-mounted directory, while the sandbox agent runs as uid 10001.
        # Keep it writable so completed attempts can persist variants.jsonl instead of being lost.
        os.chmod(adir, 0o777)
    except Exception:  # noqa: BLE001
        return
    last = 0
    while not stop_event.is_set():
        last = _analyze_once(
            repo_path, request, metric_hint, lower_is_better_hint, emit, last,
            memory_scope=memory_scope)
        stop_event.wait(15)


def _clone_repo(repo_url: str, dest: str) -> None:
    """Shallow-clone ``repo_url`` into ``dest`` on the host (the dir is bind-mounted into the
    sandbox at the same path). Raises on failure.

    If a local mirror exists under an explicitly configured AUTORESEARCH_CLONE_MIRROR_DIR,
    copy FROM it instead -- some big repos (e.g. modded-nanogpt, 327MB) clone
    fine from one host but truncate to an EMPTY dir with git STILL returning 0 from another (dgx's
    github path did exactly this). A pre-seeded mirror sidesteps that flaky path. After a real clone
    we VERIFY the dir is non-empty and raise otherwise, so a silent empty clone never leaves a
    half-repo for the sandbox agent to 'fix' by re-cloning (that looped on the speedrun full-clone
    disconnect)."""
    import shutil
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    name = repo_url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "") if repo_url else ""
    mirror_root = os.environ.get("AUTORESEARCH_CLONE_MIRROR_DIR", "").strip()
    mirror = os.path.join(mirror_root, name) if mirror_root else ""
    if mirror and name and os.path.isdir(mirror) and any(f != ".git" for f in os.listdir(mirror)):
        shutil.copytree(mirror, dest, dirs_exist_ok=True)
        _restore_index_executable_bits(dest)
        return

    def _nonempty(d):
        return os.path.isdir(d) and any(f != ".git" for f in os.listdir(d))

    # Fresh clone with RETRY + full-clone fallback: `git clone --depth 1` of a big repo (modded-nanogpt,
    # 327MB) from some hosts truncates to an EMPTY dir with git STILL returning 0 (dgx's github path did
    # this). Retry the shallow clone a few times; the LAST attempt does a FULL clone (no --depth), which
    # is slower but survives the mid-pack disconnect that kills the shallow one. This makes fresh
    # end-to-end clone reliable without a pre-seeded mirror (每次重新克隆模拟新任务).
    # 强制 HTTP/1.1 + 大 postBuffer: dgx->github 对大仓库(modded-nanogpt)走 HTTP/2 会
    # `curl 92 HTTP/2 stream CANCEL` 崩掉(真机实测三次重试全崩)。HTTP/1.1 绕过该 bug。
    gitc = ["git", "-c", "http.version=HTTP/1.1", "-c", "http.postBuffer=524288000"]
    last_err = ""
    for attempt in range(3):
        try:
            shutil.rmtree(dest, ignore_errors=True)
            args = (gitc + ["clone", repo_url, dest] if attempt == 2
                    else gitc + ["clone", "--depth", "1", repo_url, dest])
            r = subprocess.run(args, capture_output=True, text=True, timeout=1200,
                               env=_direct_network_env())
            if r.returncode == 0 and _nonempty(dest):
                _restore_index_executable_bits(dest)
                return
            last_err = (r.stderr or r.stdout or "empty clone")[-300:]
        except Exception as exc:  # noqa: BLE001 -- retry on any transport error
            last_err = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"git clone produced an empty/incomplete repo for {repo_url}: {last_err}")


def _restore_index_executable_bits(repo_path: str) -> None:
    """Undo file-mode drift inherited from a copied mirror before sandbox access."""
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return
    result = subprocess.run(
        ["git", "-C", repo_path, "ls-files", "--stage", "-z"],
        capture_output=True,
        check=True,
    )
    for entry in result.stdout.split(b"\0"):
        if not entry or b"\t" not in entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        path = os.path.join(repo_path, os.fsdecode(raw_path))
        if mode == b"120000" or not os.path.isfile(path) or os.path.islink(path):
            continue
        current = stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)
        wanted = current | 0o111 if mode == b"100755" else current & ~0o111
        if wanted != current:
            os.chmod(path, wanted)


def _make_workspace_writable(repo_path: str) -> None:
    """Let a different sandbox UID read/edit the checkout, even under a restrictive umask."""
    for current, dirs, files in os.walk(repo_path, followlinks=False):
        if not os.path.islink(current):
            mode = stat.S_IMODE(os.stat(current, follow_symlinks=False).st_mode)
            if mode | 0o777 != mode:
                os.chmod(current, mode | 0o777)
        for name in list(dirs):
            path = os.path.join(current, name)
            if os.path.islink(path):
                continue
            mode = stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)
            if mode | 0o777 != mode:
                os.chmod(path, mode | 0o777)
        for name in files:
            path = os.path.join(current, name)
            if os.path.islink(path):
                continue
            mode = stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)
            if mode | 0o666 != mode:
                os.chmod(path, mode | 0o666)


def _make_workspace_paths_writable(repo_path: str, relative_paths) -> None:
    """Make only restored dependency paths writable in an already-writable checkout."""
    root = os.path.realpath(repo_path)
    directories = {root}
    files = set()
    for relative in relative_paths:
        path = os.path.abspath(os.path.join(root, str(relative)))
        if not os.path.realpath(path).startswith(root + os.sep):
            raise ValueError(f"dependency checkpoint target escapes repository: {relative}")
        current = os.path.dirname(path)
        while current != root:
            directories.add(current)
            current = os.path.dirname(current)
        if os.path.lexists(path) and not os.path.islink(path):
            files.add(path)
    for path in sorted(directories, key=len):
        if os.path.isdir(path) and not os.path.islink(path):
            mode = stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)
            if mode | 0o777 != mode:
                os.chmod(path, mode | 0o777)
    for path in sorted(files):
        if os.path.isfile(path):
            mode = stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)
            if mode | 0o666 != mode:
                os.chmod(path, mode | 0o666)


def _winner_repeats() -> int:
    """借鉴 Frontis-MA1: 让 agent 把 best 变体多跑几次报 mean±std(报告可信度,不当判决门)。
    AUTORESEARCH_WINNER_REPEATS 控制,默认 3;=1 关闭(prompt 与从前一致)。"""
    try:
        return max(1, int(os.environ.get("AUTORESEARCH_WINNER_REPEATS", "3")))
    except ValueError:
        return 3


def _capture_repo_diff(repo_path: str, *, host_checkpoint: bool = False) -> str:
    """Best-effort capture of the agent's actual code changes (git diff of tracked files + CONTENT
    of new untracked files). Supplements the agent's self-reported changes_summary (which is the
    RELIABLE carry-forward -- on the remote path the sandbox agent often writes whole NEW scripts
    rather than editing tracked files, so a bare `git diff` is empty). `-c safe.directory=*` avoids
    git's dubious-ownership refusal when the container wrote files as a different UID into the
    bind-mounted repo. Best-effort -> "" on any failure (advisory, never load-bearing)."""
    def _git(*args):
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", repo_path, *args],
                              capture_output=True, text=True, timeout=30)
    try:
        tracked = _git("diff", "HEAD")
        # Agents may stage source changes before review. Compare the full worktree
        # with HEAD, not just the unstaged portion. New repos may not have HEAD yet.
        out = (tracked.stdout or "") if tracked.returncode == 0 else (
            (_git("diff", "--cached").stdout or "") + (_git("diff").stdout or ""))
        st = _git("status", "--short")
        news = [ln[3:].strip() for ln in (st.stdout or "").splitlines() if ln.startswith("??")]
        source_suffixes = {
            ".s", ".asm",
            ".c", ".cc", ".cpp", ".cu", ".cuh", ".go", ".h", ".hpp", ".java",
            ".js", ".jsx", ".json", ".md", ".py", ".rs", ".sh", ".toml", ".ts",
            ".tsx", ".txt", ".yaml", ".yml",
        }
        expanded: list[str] = []
        for fp in news:
            full = os.path.join(repo_path, fp)
            if os.path.isfile(full):
                expanded.append(fp)
                continue
            if not os.path.isdir(full) or os.path.basename(fp.rstrip("/")).startswith("."):
                expanded.append(fp)
                continue
            found_source = False
            for current, dirs, files in os.walk(full, followlinks=False):
                cmake_build = os.path.isfile(os.path.join(current, "CMakeCache.txt"))
                dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                                  and d not in {"__pycache__", "node_modules"}
                                  # CMake's generated compiler probes must not exhaust
                                  # the source-evidence budget before candidate files.
                                  and not (cmake_build and d == "CMakeFiles"))
                for name in sorted(files):
                    nested = os.path.join(current, name)
                    if (os.path.islink(nested)
                            or os.path.splitext(name)[1].lower() not in source_suffixes):
                        continue
                    expanded.append(os.path.relpath(nested, repo_path))
                    found_source = True
            if not found_source:
                expanded.append(fp)
        # A fixed file-count prefix can hide whole later DAG branches. Keep the
        # existing content budget, but capture small files first and name every
        # unexpanded file instead of silently omitting later components.
        def source_size(fp):
            try:
                full = os.path.join(repo_path, fp)
                return os.path.getsize(full) if os.path.isfile(full) else float("inf")
            except OSError:
                return float("inf")

        incomplete = False
        for fp in sorted(set(expanded), key=lambda fp: (source_size(fp), fp)):
            if host_checkpoint:
                from .checkpoint_artifacts import MANIFEST
                if fp == MANIFEST:
                    # snapshot_checkpoint wrote this inventory, not the candidate.
                    # Keep same-named candidate files visible in ordinary worktrees.
                    continue
            if os.path.basename(fp.rstrip("/")) == _RESULT_BASENAME:
                continue
            full = os.path.join(repo_path, fp)
            try:
                size = os.path.getsize(full) if os.path.isfile(full) else 0
                if size < 100_000 and len(out) + size < 120_000 and os.path.isfile(full):
                    with open(full, "rb") as fh:
                        content = fh.read()
                    if b"\x00" in content:
                        out += f"\n# 新增二进制文件(未展开): {fp}\n"
                    else:
                        out += f"\n# 新增文件 {fp}:\n{content.decode('utf-8', errors='replace')}\n"
                else:
                    out += f"\n# 新增文件(大文件/目录,未展开): {fp}\n"
                    incomplete = incomplete or os.path.isfile(full)
            except Exception:  # noqa: BLE001
                out += f"\n# 新增文件: {fp}\n"
                incomplete = True
        if incomplete:
            out += ("\n# SOURCE_CAPTURE_INCOMPLETE: named unexpanded files may contain implementation; "
                    "absence from these excerpts does not prove a component is missing.\n")
        return out
    except Exception:  # noqa: BLE001
        return ""


_CARRY_BLOCKED_COMPONENTS = frozenset({
    "test", "tests", "evaluator", "evaluators", ".sandbox-control",
})
_CARRY_SOURCE_SUFFIXES = frozenset({
    ".s", ".asm",
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".go", ".h", ".hpp", ".hxx", ".java",
    ".js", ".jsx", ".json", ".md", ".py", ".rs", ".sh", ".toml", ".ts",
    ".tsx", ".txt", ".yaml", ".yml",
})
_CARRY_SOURCE_NAMES = frozenset({"Makefile", "makefile", "GNUmakefile", "CMakeLists.txt"})
# Analysis directories may contain the actual component implementation, not only reports.
# Carry its code, but leave prior measurements and prose in their original evidence location.
_CARRY_ANALYSIS_SOURCE_SUFFIXES = _CARRY_SOURCE_SUFFIXES - {
    ".json", ".md", ".toml", ".txt", ".yaml", ".yml",
}
_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS = 600


def _checkpoint_handoff_timeout() -> int:
    """Host-to-sandbox source handoff timeout, independent of experiment runtime."""
    try:
        return max(60, int(os.environ.get("AUTORESEARCH_CHECKPOINT_HANDOFF_TIMEOUT_S", "600")))
    except (TypeError, ValueError):
        return 600


def _uncertain_recovery_grace() -> int:
    """Grace after the hard cap before an uncertain attempt may be reconciled."""
    try:
        return max(0, int(os.environ.get("AUTORESEARCH_UNCERTAIN_RECOVERY_GRACE_S", "300")))
    except (TypeError, ValueError):
        return 300


def _prepare_checkpoint_handoff(backend, handle, repo_path: str, emit) -> None:
    """Make generated checkpoint sources readable before the sandbox UID disappears.

    Only changed/untracked source files in this repository are touched. Never follow
    symlinks into shared caches, change ownership, or make files executable/writable.
    """
    script = r'''
import json, os, stat, subprocess, sys
root = os.path.realpath(sys.argv[1])
suffixes, names, blocked, analysis_suffixes = map(set, json.loads(sys.argv[2]))
paths = set()
for args in (("diff", "--name-only", "-z", "HEAD", "--"),
             ("ls-files", "--others", "--exclude-standard", "-z")):
    output = subprocess.check_output(["git", "-c", "safe.directory=" + root, "-C", root, *args])
    paths.update(os.fsdecode(p) for p in output.split(b"\0") if p)
for relative in paths:
    if any(part.casefold() in blocked for part in relative.split("/")):
        continue
    if ("analysis" in relative.casefold().split("/")
            and os.path.splitext(relative)[1].casefold() not in analysis_suffixes
            and os.path.basename(relative) not in names):
        continue
    if os.path.splitext(relative)[1].casefold() not in suffixes and os.path.basename(relative) not in names:
        continue
    path = os.path.join(root, relative)
    if os.path.islink(path) or not os.path.realpath(path).startswith(root + os.sep) or not os.path.isfile(path):
        continue
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o444 != 0o444:
        os.chmod(path, mode | 0o444)
    parent = os.path.dirname(path)
    while parent != root:
        mode = stat.S_IMODE(os.stat(parent).st_mode)
        if mode & 0o555 != 0o555:
            os.chmod(parent, mode | 0o555)
        parent = os.path.dirname(parent)
'''
    command = shlex.join(["python3", "-c", script, repo_path, json.dumps([
        sorted(_CARRY_SOURCE_SUFFIXES), sorted(_CARRY_SOURCE_NAMES),
        sorted(_CARRY_BLOCKED_COMPONENTS), sorted(_CARRY_ANALYSIS_SOURCE_SUFFIXES),
    ])])
    try:
        result = backend.execute_command(
            handle, command, cwd=repo_path, timeout=_checkpoint_handoff_timeout())
        if result.exit_code:
            emit("agentic_error", {"stage": "checkpoint_handoff", "error": "source permission handoff failed"})
    except Exception as exc:
        emit("agentic_error", {"stage": "checkpoint_handoff", "error": type(exc).__name__})


def _carry_path_allowed(relative_path: str) -> bool:
    parts = [part.casefold() for part in relative_path.replace("\\", "/").split("/") if part]
    if ("analysis" in parts
            and os.path.splitext(relative_path)[1].casefold() not in _CARRY_ANALYSIS_SOURCE_SUFFIXES
            and os.path.basename(relative_path) not in _CARRY_SOURCE_NAMES):
        return False
    return bool(parts) and not any(part in _CARRY_BLOCKED_COMPONENTS for part in parts) \
        and parts[-1] not in {_RESULT_BASENAME.casefold(), "task.toml", ".autoresearch-checkpoint.json"}


def _expand_untracked_repositories(repo_path: str, paths: list[str]) -> list[str]:
    """Outer Git lists an untracked nested checkout as a directory, not its source files."""
    root = os.path.realpath(repo_path)
    pending = list(paths)
    expanded = []
    while pending:
        relative = pending.pop()
        path = os.path.join(root, relative)
        if (not _carry_path_allowed(relative) or os.path.islink(path)
                or not os.path.realpath(path).startswith(root + os.sep)):
            expanded.append(relative)
            continue
        if os.path.isdir(path) and os.path.exists(os.path.join(path, ".git")):
            # Keep the actual worktree contents, including changes staged or committed
            # inside the nested repository. Never carry its Git metadata.
            result = subprocess.run(
                ["git", "-c", f"safe.directory={os.path.realpath(path)}", "-C", path,
                 "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                capture_output=True, timeout=30, check=True)
            pending.extend(os.path.join(relative, os.fsdecode(item))
                           for item in result.stdout.split(b"\0") if item)
        else:
            expanded.append(relative)
    return expanded


def _untracked_compiler_metadata(repo_path: str, relative: str) -> bool:
    """Recognize Cargo cache metadata, not arbitrary JSON or generated source.

    Call only for untracked files: repository-owned files remain subject to the
    ordinary source/checkpoint checks even if they have a compiler-cache name.
    """
    from pathlib import Path

    root = Path(repo_path)
    path = root / relative
    for parent in path.parents:
        if parent == root.parent:
            break
        if (parent / ".rustc_info.json").is_file():
            return (path == parent / ".rustc_info.json"
                    or ".fingerprint" in path.relative_to(parent).parts)
    return False


def _dependency_worktree_changes(
    repo_path: str, prior_worktree: str | list[str] | tuple[str, ...] | dict | None,
) -> dict[str, tuple[bytes | None, int | None]]:
    """Resolve the file states of one or more accepted parent checkpoints.

    A DAG join must contain every immediate parent's accepted implementation.  Parent worktrees
    are full snapshots, so shared ancestor edits normally appear identically in several parents;
    those are de-duplicated. With host-selected merge bases, an unchanged ancestor snapshot
    does not conflict with a branch edit. Content and permissions are merged independently;
    disjoint text edits use a three-way merge; conflicting edits still fail explicitly.

    Official tests, evaluators, task contracts and run-protocol files are deliberately excluded.
    """
    bases_raw = []
    if isinstance(prior_worktree, dict):
        bases_raw = prior_worktree["merge_bases"]
        prior_worktree = prior_worktree["checkpoints"]
    sources_raw = (
        list(prior_worktree)
        if isinstance(prior_worktree, (list, tuple))
        else ([prior_worktree] if prior_worktree else [])
    )
    if not sources_raw:
        return {}
    workroot = os.path.realpath(os.path.dirname(repo_path))
    target = os.path.realpath(repo_path)
    sources: list[str] = []
    target_commit = _repo_commit(target)
    bases: list[str] = []
    for raw_source in sources_raw + list(bases_raw):
        source = os.path.realpath(str(raw_source))
        from .checkpoint_artifacts import validate_checkpoint, MANIFEST
        if os.path.isfile(os.path.join(source, MANIFEST)):
            source_manifest = validate_checkpoint(source)
            if os.path.dirname(source) != workroot:
                # A recovered session may revalidate an immutable checkpoint from
                # an earlier controller workroot. Import that exact content-addressed
                # snapshot beside the new checkout; ordinary mutable worktrees remain
                # confined to their original workroot and are still rejected below.
                staged = os.path.join(workroot, os.path.basename(source))
                if os.path.exists(staged):
                    if validate_checkpoint(staged) != source_manifest:
                        raise ValueError(
                            f"staged dependency checkpoint identity mismatch: {raw_source}")
                else:
                    temporary = os.path.join(
                        workroot, ".checkpoint-import-" + uuid.uuid4().hex)
                    shutil.copytree(source, temporary, symlinks=True)
                    try:
                        os.rename(temporary, staged)
                    except OSError as exc:
                        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                            raise
                        shutil.rmtree(temporary)
                    if validate_checkpoint(staged) != source_manifest:
                        raise ValueError(
                            f"imported dependency checkpoint identity mismatch: {raw_source}")
                source = os.path.realpath(staged)
        if (os.path.dirname(source) != workroot or source == target
                or not os.path.isdir(source)):
            raise ValueError(f"invalid dependency checkpoint: {raw_source}")
        if _repo_commit(source) != target_commit:
            raise ValueError("dependency checkpoints do not share the frozen repository commit")
        collection = sources if raw_source in sources_raw else bases
        if source not in collection:
            collection.append(source)

    def file_state(source: str, relative: str) -> tuple[bytes | None, int | None]:
        path = os.path.join(source, relative)
        src = os.path.realpath(path)
        if not src.startswith(source + os.sep):
            raise ValueError(f"dependency checkpoint path escapes repository: {relative}")
        if os.path.lexists(path) and (os.path.islink(path) or not os.path.isfile(path)):
            raise ValueError(f"unsupported dependency checkpoint path: {relative}")
        if not os.path.isfile(src):
            return None, None
        with open(src, "rb") as handle:
            return handle.read(), stat.S_IMODE(os.stat(src).st_mode)

    # relative path -> [(source, content-or-None-for-deletion, mode)]
    variants: dict[str, list[tuple[str, bytes | None, int | None]]] = {}
    for source in sources + bases:
        changed = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", source, "diff", "--name-only",
             "--diff-filter=ACDMRTUXB", "HEAD", "--"],
            capture_output=True, text=True,
            timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS)
        if changed.returncode != 0:
            raise ValueError(f"cannot inspect dependency checkpoint: {source}")
        paths = [path for path in changed.stdout.splitlines() if _carry_path_allowed(path)]
        if os.path.isfile(os.path.join(source, MANIFEST)):
            paths.extend(validate_checkpoint(source).get("runtime_paths", []))
        untracked = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", source,
             "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True, timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS)
        if untracked.returncode == 0:
            for relative in _expand_untracked_repositories(source, [
                    os.fsdecode(raw) for raw in untracked.stdout.split(b"\0") if raw]):
                if (not _carry_path_allowed(relative)
                        or _untracked_compiler_metadata(source, relative)
                        or (os.path.splitext(relative)[1].casefold()
                            not in _CARRY_SOURCE_SUFFIXES
                            and os.path.basename(relative) not in _CARRY_SOURCE_NAMES)):
                    continue
                # Runtime package/build views may expose untracked source-looking files through
                # symlinks into a shared cache.  They are not part of the component checkpoint and
                # must not make an otherwise valid parent impossible to materialize.
                candidate = os.path.realpath(os.path.join(source, relative))
                if not candidate.startswith(source + os.sep):
                    continue
                paths.append(relative)
        for relative in dict.fromkeys(paths):
            content, mode = file_state(source, relative)
            variants.setdefault(relative, []).append((source, content, mode))

    resolved = {}
    for relative, candidates in sorted(variants.items()):
        if bases:
            # Full snapshots are required here: absence can be an intentional deletion of
            # an untracked file introduced by the shared ancestor.
            candidates = [(source, *file_state(source, relative)) for source in sources]
        distinct = {(content, mode) for _, content, mode in candidates}
        base_states = {file_state(source, relative) for source in bases}
        if len(distinct) > 1 and len(base_states) == 1:
            base_content, base_mode = next(iter(base_states))
            contents = {content for content, _ in distinct}
            modes = {mode for _, mode in distinct}
            if len(contents) > 1:
                contents.discard(base_content)
            if len(modes) > 1:
                modes.discard(base_mode)
            if (len(contents) > 1 and len(modes) == 1
                    and base_content is not None and b"\0" not in base_content
                    and all(content is not None and b"\0" not in content for content in contents)):
                # Use only host-selected ancestry, never guess from the target or a parent.
                # Stable ordering makes the result independent of DAG parent enumeration.
                with tempfile.TemporaryDirectory(prefix="ar-checkpoint-merge-") as merge_dir:
                    current, ancestor, other = [os.path.join(merge_dir, name)
                                                for name in ("current", "ancestor", "other")]
                    with open(ancestor, "wb") as handle:
                        handle.write(base_content)
                    merged = base_content
                    for content in sorted(contents):
                        with open(current, "wb") as handle:
                            handle.write(merged)
                        with open(other, "wb") as handle:
                            handle.write(content)
                        merge = subprocess.run(
                            ["git", "merge-file", "--stdout", current, ancestor, other],
                            capture_output=True, timeout=30)
                        if merge.returncode != 0:
                            break
                        merged = merge.stdout
                    else:
                        contents = {merged}
            if len(contents) == len(modes) == 1:
                distinct = {(next(iter(contents)), next(iter(modes)))}
        if len(distinct) > 1:
            parents = ", ".join(os.path.basename(source) for source, _, _ in candidates)
            raise ValueError(
                f"dependency checkpoint conflict for {relative}: {parents} produced different states")
        resolved[relative] = next(iter(distinct))
    return resolved


def _apply_prior_worktree(
    repo_path: str, prior_worktree: str | list[str] | tuple[str, ...] | dict | None,
) -> int:
    """Materialize the same resolved parent state used by component-diff review."""
    changes = _dependency_worktree_changes(repo_path, prior_worktree)
    target = os.path.realpath(repo_path)
    inherited_sources = []
    for relative, (content, mode) in changes.items():
        dst = os.path.abspath(os.path.join(target, relative))
        if not os.path.realpath(dst).startswith(target + os.sep):
            raise ValueError(f"dependency checkpoint target escapes repository: {relative}")
        # Snapshots materialize internal file aliases. Restore that regular file,
        # not bytes through the original checkout's link into another source file.
        if os.path.islink(dst):
            os.unlink(dst)
        if content is None:
            if os.path.exists(dst) and os.path.isfile(dst):
                os.unlink(dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if mode is None and os.path.isfile(dst):
            mode = os.stat(dst).st_mode & 0o777
        temporary = dst + ".ar-restore-" + uuid.uuid4().hex
        try:
            with open(temporary, "xb") as handle:
                handle.write(content)
            if mode is not None:
                os.chmod(temporary, mode)
            os.replace(temporary, dst)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if (_carry_path_allowed(relative)
                and (os.path.splitext(relative)[1].casefold() in _CARRY_SOURCE_SUFFIXES
                     or os.path.basename(relative) in _CARRY_SOURCE_NAMES)):
            inherited_sources.append(relative)
    if inherited_sources:
        git = ["git", "-c", f"safe.directory={target}", "-C", target]
        ignored = subprocess.run(
            git + ["check-ignore", "-z", "--stdin"],
            input=("\0".join(inherited_sources) + "\0").encode(),
            capture_output=True, timeout=30)
        if ignored.returncode not in (0, 1):
            raise subprocess.CalledProcessError(
                ignored.returncode, ignored.args, ignored.stdout, ignored.stderr)
        for raw in ignored.stdout.split(b"\0"):
            if not raw:
                continue
            relative = os.fsdecode(raw)
            dst = os.path.join(target, relative)
            blob = subprocess.check_output(
                git + ["hash-object", "-w", "--", relative], text=True,
                timeout=30).strip()
            file_mode = "100755" if os.stat(dst).st_mode & 0o111 else "100644"
            subprocess.run(git + ["update-index", "--add", "--cacheinfo",
                                  file_mode, blob, relative], check=True, timeout=30)
    return len(changes)


def _capture_component_diff(repo_path: str, parents) -> str:
    """Host-captured component delta against the DAG dependency closure, not last attempt.

    Reuse checkpoint materialization semantics without creating or editing a worktree.
    Improvements are still compared to the dependency parent, so already implemented
    parts of the current component remain visible to MIR on its later attempts.
    """
    inherited = _dependency_worktree_changes(repo_path, parents)
    git = ["git", "-c", f"safe.directory={os.path.realpath(repo_path)}", "-C", repo_path]
    paths = set(inherited)
    for args in (["diff", "--name-only", "-z", "HEAD", "--"],
                 ["ls-files", "--others", "--exclude-standard", "-z"]):
        result = subprocess.run(
            git + args, capture_output=True,
            timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS, check=True)
        paths.update(_expand_untracked_repositories(repo_path, [
            os.fsdecode(p) for p in result.stdout.split(b"\0") if p]))
    sections = []
    copy_sources = None
    for relative in sorted(paths):
        if not _carry_path_allowed(relative):
            continue
        path = os.path.join(repo_path, relative)
        if relative in inherited:
            before, _ = inherited[relative]
        else:
            original = subprocess.run(git + ["show", f"HEAD:{relative}"],
                                      capture_output=True, timeout=30)
            before = original.stdout if original.returncode == 0 else None
        # Generated weights/data are not source-diff evidence. Tracked changes remain visible
        # through the full cumulative diff used by the task-contract reviewer.
        if before is None and os.path.splitext(relative)[1].lower() not in _CARRY_SOURCE_SUFFIXES:
            continue
        # Apply the source scope before inspecting a runtime-directory link. Non-source,
        # untracked package views are not checkpoint code; never traverse their targets.
        if not os.path.realpath(path).startswith(os.path.realpath(repo_path) + os.sep):
            raise ValueError(f"component diff path escapes repository: {relative}")
        if os.path.islink(path):
            # Checkpoints materialize original internal file aliases. Compare the
            # same bytes without rewriting the live checkout or accepting new links.
            entry = subprocess.run(git + ["ls-tree", "HEAD", "--", relative],
                                   capture_output=True, timeout=30, check=True)
            original = subprocess.run(git + ["show", f"HEAD:{relative}"],
                                      capture_output=True, timeout=30)
            if (relative not in inherited or before is None or not os.path.isfile(path)
                    or not entry.stdout.startswith(b"120000 ") or original.returncode
                    or original.stdout != os.fsencode(os.readlink(path))):
                raise ValueError(f"component diff cannot expand symlink: {relative}")
        if os.path.isfile(path):
            # A large existing source file can have a tiny component delta. The review
            # layer bounds changed-line evidence; this is only a host read safety limit.
            if os.path.getsize(path) > 8 * 1024 * 1024:
                raise ValueError(f"component diff source exceeds evidence limit: {relative}")
            with open(path, "rb") as handle:
                after = handle.read()
        else:
            after = None
        if before == after:
            continue
        if any(b"\0" in (value or b"") for value in (before, after)):
            if os.path.splitext(relative)[1].lower() in _CARRY_SOURCE_SUFFIXES:
                raise ValueError(f"component diff cannot expand binary source: {relative}")
            # Tracked build outputs may change alongside their source. Keep their
            # identities visible for review, without treating them as source text
            # or claiming that a digest proves implementation correctness.
            identities = ["absent" if value is None else
                          f"sha256:{hashlib.sha256(value).hexdigest()} bytes={len(value)}"
                          for value in (before, after)]
            sections.append(f"diff --git a/{relative} b/{relative}\n"
                            f"Binary files differ (content not expanded): {relative}\n"
                            f"before {identities[0]}\nafter {identities[1]}\n")
            continue
        copy_from = None
        if before is None and after is not None:
            if copy_sources is None:
                detected = subprocess.run(
                    git + ["diff", "-C", "--find-copies-harder", "--name-status", "-z", "HEAD", "--"],
                    capture_output=True, timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS, check=True)
                copy_sources = {}
                records = iter(detected.stdout.split(b"\0"))
                for status in records:
                    if not status:
                        continue
                    source = os.fsdecode(next(records))
                    if status[:1] in {b"C", b"R"}:
                        destination = os.fsdecode(next(records))
                        if status.startswith(b"C"):
                            copy_sources[destination] = (source, int(status[1:]))
            source, similarity = copy_sources.get(relative, (None, None))
            # A dependency-modified source is not the repository-baseline copy
            # reported by Git. Keep its full addition visible to the review.
            if source is not None and source not in inherited and _carry_path_allowed(source):
                entry = subprocess.run(git + ["ls-tree", "HEAD", "--", source],
                                       capture_output=True, timeout=30, check=True)
                source_mode = entry.stdout.split(b" ", 1)[0].decode()
                if source_mode in {"100644", "100755"}:
                    original = subprocess.run(git + ["show", f"HEAD:{source}"],
                                              capture_output=True, timeout=30, check=True).stdout
                    if b"\0" not in original and len(original) <= 8 * 1024 * 1024:
                        before, copy_from = original, source
        delta = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(
            (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
            (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
            fromfile=f"a/{copy_from or relative}" if before is not None else "/dev/null",
            tofile=f"b/{relative}" if after is not None else "/dev/null"))
        header = f"diff --git a/{relative} b/{relative}\n"
        if copy_from is not None:
            header = (f"diff --git a/{copy_from} b/{relative}\n"
                      f"similarity index {similarity}%\ncopy from {copy_from}\ncopy to {relative}\n")
            mode = "100755" if os.stat(path).st_mode & 0o111 else "100644"
            if source_mode != mode:
                header += f"old mode {source_mode}\nnew mode {mode}\n"
        elif before is None:
            mode = "100755" if os.stat(path).st_mode & 0o111 else "100644"
            header += f"new file mode {mode}\n"
        sections.append(header + delta)
    return "\n".join(sections)


def _per_run_cap(repository_contract: dict | None = None) -> int:
    """Execution safety limit for one training/evaluation command, not its scientific budget.

    Allow the selected manifest's agent and verifier allowance when longer than
    the generic safety cap; official steps/epochs/timers and the total
    sandbox budget remain unchanged. This is independent of the LLM response timeout.
    """
    try:
        cap = max(60, int(os.environ.get("AUTORESEARCH_AGENTIC_PER_RUN_S", "7200")))
    except (TypeError, ValueError):
        cap = 7200
    allowances = []
    for source in (repository_contract or {}).get("sources", []):
        if not isinstance(source, dict) or os.path.basename(
                str(source.get("path") or "")).casefold() != "task.toml":
            continue
        try:
            manifest = tomllib.loads(str(source.get("content") or ""))
        except (TypeError, tomllib.TOMLDecodeError):
            return cap
        agent = manifest.get("agent", {})
        verifier = manifest.get("verifier", {})
        if not isinstance(agent, dict) or "timeout_sec" not in agent:
            continue
        seconds = agent["timeout_sec"]
        evaluation = verifier.get("timeout_sec", 0) if isinstance(verifier, dict) else 0
        if type(seconds) is not int or seconds <= 0 or type(evaluation) is not int or evaluation < 0:
            return cap
        allowances.append(seconds + evaluation)
    if allowances and all(value == allowances[0] for value in allowances):
        return max(cap, allowances[0])
    return cap


def _verifier_timeout(repository_contract: dict | None = None) -> int | None:
    """Return one unambiguous positive verifier timeout from selected task manifests."""
    timeouts = []
    for source in (repository_contract or {}).get("sources", []):
        if not isinstance(source, dict) or os.path.basename(
                str(source.get("path") or "")).casefold() != "task.toml":
            continue
        try:
            manifest = tomllib.loads(str(source.get("content") or ""))
        except (TypeError, tomllib.TOMLDecodeError):
            return None
        verifier = manifest.get("verifier", {})
        if not isinstance(verifier, dict) or "timeout_sec" not in verifier:
            continue
        seconds = verifier["timeout_sec"]
        if type(seconds) is not int or seconds <= 0:
            return None
        timeouts.append(seconds)
    if timeouts and all(value == timeouts[0] for value in timeouts):
        return timeouts[0]
    return None


def _agentic_hard_cap() -> int:
    """HARD wall-clock backstop (seconds) for the in-container agent loop when no explicit budget is set
    (#46). run_agent_task ONLY uses its reliable daemon-thread timeout when given a finite ``timeout``;
    with ``None`` it runs inline and a dead-connection poll (container exited) hangs the whole run FOREVER
    (seen live: micrograd's local round ran ~2h with the orchestrator stuck, never returning). A finite
    cap makes the daemon backstop abandon a hung run and salvage best-so-far. Env-tunable; default 4h so
    it sits just beyond the SDK's own 3h per-conversation timeout and never cuts a legit long train."""
    try:
        return max(60, int(os.environ.get("AUTORESEARCH_AGENTIC_HARD_CAP_S", "14400")))
    except (TypeError, ValueError):
        return 14400


def _agentic_finalize_reserve() -> int:
    """Wall-clock kept outside an explicit experiment budget for writing the final artifact."""
    try:
        return max(60, int(os.environ.get("AUTORESEARCH_AGENTIC_FINALIZE_RESERVE_S", "300")))
    except (TypeError, ValueError):
        return 300


_RESEARCH_FEEDBACK_BOUNDARY = (
    "RESEARCH FEEDBACK BOUNDARY: Do not read, infer, or optimize external benchmark reward. "
    "Use only raw task measurements and implementation correctness feedback permitted by "
    "the research goal and repository contract. Keep external reward artifacts and formulas "
    "out of Agent observations, research memory, and reports. If an evaluator also produces "
    "external reward, do not open its score artifacts or expose that output to the Agent. "
    "External evaluation tools calculate benchmark reward only after the final candidate "
    "is fixed. A task's native scientific objective remains distinct from that external reward.\n"
)


_EXECUTION_STATUS_GUIDANCE = (
    "EXECUTION STATUS: in Bash launch scripts you create, use `set -e -o pipefail` or "
    "explicitly capture and propagate the training/evaluator exit status. A pipeline's "
    "tee/head/tail succeeding does not prove the experiment succeeded. Avoid piping a live "
    "experiment into head, which may terminate it early; save complete training/evaluation logs "
    "and inspect them after the command. A zero wrapper exit code alone is not success evidence; "
    "check the actual evaluator outputs and failures before writing a successful result. "
    "If your driver starts a required child service, readiness polling must check that child's exit status "
    "and report its stderr immediately if it exits before readiness, rather than waiting the full deadline. "
    "Inspect companion service logs too: unresolved failed scoring requests invalidate an aggregate score "
    "even when the evaluator exits zero. Preserve the model and predictions for valid scoring recovery; "
    "do not drop failed samples or change the evaluator, labels or thresholds to obtain a score. "
    "Before expensive training in each fresh sandbox, smoke-test the actual scoring entry point "
    "with a small valid input in the selected environment; imports alone do not check runtime data. "
    "Verify the evaluator loads the exact candidate code, recording module origins or source hashes. "
    "For Python scripts, the script directory can precede PYTHONPATH; an environment variable alone "
    "does not prove the candidate is imported. Preserve the official evaluator when fixing launch wiring. "
    "Resources installed only in a container's temporary/home directory may disappear at the next "
    "sandbox. Keep required runtime data in mounted persistent session storage, record its paths "
    "and environment settings in artifact_paths/notes, and verify them again after restoration. "
    "A distributed evaluator can hang during shutdown after an unhandled main-process exception. "
    "In launchers you author, preserve that traceback, propagate the failure, and bound cleanup "
    "of their own workers; give scoring its own finite timeout within the remaining task budget. "
    "A logged/caught exception alone is not authority to kill a running experiment. "
    "Record resumable progress in artifact_paths and notes before an interrupted run closes. "
    "After a timeout, inspect the previous attempt's saved progress before restarting; resume "
    "only when its code, data, configuration and execution protocol are compatible and the "
    "remaining total task budget permits it. Otherwise explain why a restart is necessary. "
    "Preserve the original attempt provenance; partial progress is not a completed result.\n"
)


_DEVICE_PREFLIGHT_GUIDANCE = (
    "DEVICE PREFLIGHT: when the task requires an accelerator, before full training run a "
    "minimal supported operation on that device using the exact interpreter and runtime "
    "selected for the experiment. Successful imports or device enumeration alone do not "
    "prove its kernels execute. Record the operation, runtime, device and actual error. "
    "If it fails, repair the environment; do not silently fall back to CPU or another device "
    "outside the task's resource requirements. Inspect compatible preinstalled environments "
    "and session caches under the repository/task constraints before downloading. Do not "
    "override a hard dependency pin without resolving the conflict; report an environment "
    "blocker if the required runtime and hardware cannot be reconciled. An environment change "
    "after baseline freezing must not affect only the method: report the need to re-establish the baseline "
    "in the same compatible environment before making a new performance comparison; do not "
    "rerun baseline inside a method-only phase. In that case return a top-level `baseline_drift` "
    "object in ar_result.json describing the issue, actual evidence and required repair, with "
    "candidate_valid=false; retain real observations but use method=null when none ran. Do not "
    "bury this recovery request only inside notes. The host will review the evidence before "
    "deciding whether to re-establish the baseline; the request itself is not a verified finding.\n"
)


_ATTEMPT_INVOCATION_GUIDANCE = (
    "ATTEMPT INVOCATION: ar-attempt sends argv and cwd to a separate execution broker; "
    "it does not inherit temporary exports or venv activation from the interactive terminal. "
    "Pass task-permitted run settings explicitly as `ar-attempt env NAME=VALUE <command>` "
    "or export them inside the launched driver. `NAME=VALUE ar-attempt <command>` is not "
    "equivalent. Use the selected environment's absolute interpreter path or activate it "
    "inside that driver. Record effective settings from the actual child, not just the "
    "terminal. Change directory before ar-attempt, including before measuring a parent "
    "control; cd inside a driver cannot change the recorded launch checkout. Do not forward "
    "the full terminal environment, secrets, or proxy settings. Never unset, replace or "
    "override AUTORESEARCH_ATTEMPT_SOCKET or AUTORESEARCH_ATTEMPT_LOG. If the broker refuses "
    "a launch because an earlier execution is uncertain, report that exact error for host "
    "reconciliation; a locally written attempt log is not host-registered evidence.\n"
    "Stage measured scripts at their actual execution paths before ar-attempt; do not copy "
    "a different version into those paths inside the measured driver or edit them while it runs. "
    "Use a new driver filename for a later variant. Preparation excludes training/scoring, "
    "which must still run through ar-attempt.\n"
    "When a measured driver uses separate compiled probe sources, pass their actual staged "
    "paths as explicit command arguments, and have the driver consume or verify those same "
    "files. The host captures explicit source arguments and checks their hashes again after "
    "execution; a driver filename or a printed hash alone does not supply the probe code.\n"
    "Do not rely on printing probe code to stderr: stderr is not forwarded as source evidence.\n"
    "Before each measured command returns, atomically write its result JSON at the launch "
    "checkout's ar_result.json (not only stdout or the attempt artifact directory), with its current "
    "AUTORESEARCH_ATTEMPT_ID and an artifact_paths list of individual runtime files needed "
    "to restore that candidate (such as model weights and configuration). Use repository-relative "
    "or absolute sandbox file paths, not directories. The host snapshots those files at this "
    "attempt's completion; writing only the final conversation result or notes is too late "
    "when later trials overwrite them. Preserve any task-required files as well.\n"
    "A terminal tool response can time out while its command is still running. For a long "
    "install, build, or experiment, start the command with nohup and &, redirect stdin, "
    "write its output and final exit status under $AUTORESEARCH_ARTIFACT_DIR, and poll "
    "those files with new terminal commands that wait at most 60 seconds and check both "
    "the process and its exit-status file; stop polling as soon as it exits. "
    "Keep ar-attempt inside that job for every physical experiment. "
    "Do not use ar-attempt for echo, sleep, or readiness probes. "
    "Never use an input flag, an empty/whitespace command, "
    "or tool-call markup as chat text to wait; do not claim completion before seeing "
    "the exit status and measured output.\n"
)


def _task_prompt(request: str, repo_path: str, result_path: str, budget_s: int,
                 metric_hint: str | None, lower_is_better_hint: bool | None,
                 repeats: int = 1, carry_direction: str = "", tool_repos: tuple = (),
                 repository_contract: dict | None = None) -> str:
    """The whole task, handed to the in-container agent. No editable/protected whitelist: the
    agent decides what to change and how to build/run/measure using the repo's own tooling.

    ``repeats`` (借鉴 Frontis-MA1 的 "3 independent runs + std"): when >1, ask the agent to run its
    BEST variant and the baseline on the same seeds, then report both means, standard deviations,
    and per-seed lists. Report-only: never a judge gate."""
    metric_line = (f"The target metric is '{metric_hint}'."
                   if metric_hint else
                   "Determine the repo's own primary metric from its benchmark/README.")
    dir_line = ("Lower is better." if lower_is_better_hint is True
                else "Higher is better." if lower_is_better_hint is False
                else "Decide the direction (lower- or higher-is-better) from the repo.")
    _pr = _per_run_cap(repository_contract)  # independent of the official training budget
    repeat_line = (
        f"\n\nREPRODUCIBILITY (report-only, does NOT change the win/lose judgement): once you have "
        f"your BEST improving variant, run THAT SAME variant {repeats} times with DIFFERENT random "
        f'seeds. Set "method" to the MEAN across those runs, and ALSO write "method_std" (the '
        f'standard deviation) and "method_runs" (the list of per-seed numbers). If you only have '
        f"time for a single run, that is fine -- just omit method_std/method_runs."
        if repeats and repeats > 1 else "")
    # #75 跨轮引导: 上一轮的分析结论(carry)内联进 prompt。flash agent 读文件(direction.txt)不可靠,
    # 直接写进任务文本更稳,让它从上一轮结论接着改,而不是每轮从头重来。
    carry_block = (
        "PREVIOUS ROUND'S CONCLUSION (advisory analysis from the last round of this same task). "
        "Preserve measured working state, but verify any claimed acceptance against the host "
        f"evidence before following this suggestion:\n{carry_direction.strip()[:1200]}\n\n"
        if carry_direction and carry_direction.strip() else "")
    # #76 多仓库联动: 额外 clone 的"工具仓库"路径告诉 agent —— 它们是 instruments(用来评测/增强主
    # 仓库),不是要改进的目标,别去改它们。空 -> 不出现该块(单仓库行为不变)。
    tool_block = ""
    if tool_repos:
        _lines = "\n".join(f"  - {n} at {p}" for n, p in tool_repos)
        tool_block = (
            "TOOL REPOSITORIES (already cloned for you to USE as instruments, NOT the target to "
            "improve -- do NOT modify them; use them to evaluate/augment the main repo exactly as the "
            f"research goal asks):\n{_lines}\n\n")
    return (
        f"You are improving a cloned repository. It is at: {repo_path}\n\n"
        f"Research goal:\n{request}\n\n"
        f"{metric_line} {dir_line}\n\n"
        f"{_EXECUTION_STATUS_GUIDANCE}"
        f"{_RESEARCH_FEEDBACK_BOUNDARY}"
        "HOST REVIEW PRIORITY: analysis files, carry notes and historical run summaries are "
        "advisory. The current host Method, ComponentPlan and review corrections take precedence. "
        "Historical advice cannot accept a candidate, remove a required component or cancel a "
        "host repair. A passing task evaluator or comparison review alone does not establish "
        "Method fidelity or host acceptance.\n"
        f"{_DEVICE_PREFLIGHT_GUIDANCE}"
        f"{_ATTEMPT_INVOCATION_GUIDANCE}"
        f"{carry_block}"
        f"{tool_block}"
        "ANSWER-MATERIAL BOUNDARY: do not list, open, search, copy, import, execute, or recover "
        "through Git history any path whose directory name is `solution` or `solutions`. Treat "
        "those directories as unavailable benchmark answers, even when the public clone contains "
        "them. Develop the candidate only from the task instructions, starter, tests/evaluator, "
        "ordinary repository documentation, literature, and your own measured feedback.\n\n"
        "REPOSITORY WRITE BOUNDARY: obey the task's editable-file rule for persistent candidate "
        "changes. Keep temporary checks, helper scripts, and generated caches outside the target "
        "checkout when their paths are not allowed; configure official runs to write such outputs "
        "elsewhere before running them. Check tracked and untracked files against the rule before "
        "each official attempt and before reporting a candidate. Do not alter protected task files "
        "or the evaluator to satisfy this boundary.\n\n"
        "ENVIRONMENT: you are running INSIDE a Docker sandbox container. A nested Docker daemon is "
        "NOT available -- `docker`/`docker run`/`docker compose` fail with 'cannot connect to the "
        "Docker daemon'. If the repo's official/recommended run path is Docker-based, do NOT get "
        "stuck on it: find and use the repo's OWN non-Docker / direct alternative (most repos ship "
        "one, e.g. a plain run/eval script) -- explore its scripts/README to find it.\n\n"
        "Do ALL of the following, using the repository's OWN build/run/scoring tooling "
        "(do not invent your own scorer):\n"
        "1. Explore the repo. Figure out how to build and run its benchmark/evaluation. If the repo "
        "PINS dependency versions (requirements.txt / pyproject / setup / README / a specific torch "
        "or CUDA build), install EXACTLY those pinned versions -- do NOT install a bare `pip install "
        "torch` or let dependency versions drift: a different torch/CUDA build silently changes "
        "per-step speed and makes the BASELINE non-reproducible (a speedrun baseline swinging "
        "77s<->160s is exactly this). Prefer the repo's own env/setup script if it ships one.\n"
        "   ★ NETWORK NOTE: persistent package, Python-runtime, dataset and tool caches are mounted. "
        "Inspect and reuse those local paths first; do not wipe them or force a fresh download. Every "
        "network call must use the direct connection already enforced by the environment. Do NOT start a "
        "sizeable network pull (datasets, weights, wheels, images, or large clones). If a required large "
        "artifact is absent locally, record its URL, expected size and purpose as an environment blocker "
        "and stop; the outer supervisor must obtain approval before any such pull.\n"
        "   DEPENDENCY REALITY: torch is NOT guaranteed to be pre-installed. First run the repo's locked "
        "installer (`uv sync` when a uv.lock exists), then check it with `uv run python -c 'import "
        "torch;print(torch.__version__, torch.cuda.is_available())'`. The shared project venv and package "
        "cache may already contain the exact pinned build; use them, but never assume a system-python "
        "install or a local wheelhouse exists. Do not override the repo with an arbitrary torch/CUDA "
        "version and do not churn through several CUDA builds. If the locked install genuinely needs a "
        "missing package, install that one pinned dependency once through the configured direct indexes "
        "and let the persistent cache retain it. Prefix subsequent Python commands with `uv run`.\n"
        "   ENV RULE (critical -- do NOT reconstruct the toolchain yourself): if the repo ships a "
        "`uv.lock` / `poetry.lock` / `Pipfile.lock`, THAT lock is the correct, self-consistent "
        "environment -- run `uv sync` (or the repo's locked installer) ONCE and prefix EVERY command "
        "with `uv run`. Do NOT create your own separate venv and do NOT hand-install torch or a CUDA "
        "toolkit (`pip install torch`, `pip install nvidia-cuda-nvcc-*`, a different cuXX build): "
        "mixing in your own CUDA/torch breaks the repo's pinned, matched toolchain and then NOTHING "
        "compiles (a real failure: an agent hand-installed a CUDA-13 nvcc that clashed with the "
        "sandbox's bundled CUDA-12 ptxas -> every kernel failed to build -> no baseline). If a build "
        "tool like `nvcc` seems missing, do NOT try to globally reinstall the whole CUDA stack -- run "
        "the repo's OWN eval/data scripts via `uv run <script>` (e.g. a dataset-download script, then "
        "the benchmark runner): the repo's build/JIT path (Triton, torch, CUTLASS bring their own "
        "compilers) already handles most kernels, and a partial baseline over the kernels that DO "
        "build is a valid result -- far better than thrashing on a global toolchain reinstall.\n"
        "2. Run it UNMODIFIED to establish the BASELINE metric value. Sanity-check the baseline is "
        "in a sane range (e.g. per-step time / total time near the repo's documented figure); if it "
        "is 2x+ off the expected, the ENVIRONMENT is wrong (wrong torch build) -- fix the env and "
        "re-measure the baseline BEFORE trying any improvement, so method is compared to a real baseline.\n"
        f"3. AS SOON AS you have the baseline, WRITE {result_path} with the schema below and "
        '"method": null. This is critical: write it early so partial progress is never lost.\n'
        "4. Improve the metric. PREFER to edit the repository's OWN existing model/training/kernel "
        "files IN PLACE -- locate the file(s) that define the core method (often only a few lines) "
        "and modify those directly, rather than writing a separate standalone training script from "
        "scratch: editing THIS repo's real files keeps you improving the actual method, and the "
        "edits carry forward to the next round. Keep the scoring/timing harness as-is.\n"
        "5. Re-run the SAME benchmark to measure your modified variant's metric value, then "
        f"OVERWRITE {result_path} filling in \"method\" with THAT REAL MEASURED NUMBER. Do this "
        "for EVERY completed variant run -- keep \"method\" set to the value of your latest (or "
        "best) completed attempt. A result that is WORSE than the baseline is a VALID, honest "
        "negative finding: record its real number -- NEVER leave \"method\" null just because "
        "the change did not help.\n"
        "6. DIRECTED ITERATION (do NOT iterate blind). A SEPARATE analysis agent watches your work "
        "and helps you aim. Two rules:\n"
        f"   (a) After EACH completed variant run, APPEND one JSON line to {repo_path}/analysis/"
        "variants.jsonl describing that variant. Write one complete JSON object per physical line, "
        "terminate it with a newline, and validate every non-empty line with json.loads before "
        "starting another candidate: "
        '{"variant": "<what you changed>", "result": <the measured metric number>, '
        '"valid": <true only when every repository correctness/quality gate passed>, '
        '"note": "<1-line why it did/did not help>"}. Create the analysis/ dir if missing.\n'
        f"   (b) BEFORE you design your NEXT variant, READ {_analysis_direction_path(repo_path)} if it "
        "exists. The analysis agent writes there: given the goal, the method, the baseline, and the "
        "variants you have tried, it tells you which direction best fits the method and why. Design "
        "your next variant FROM that conclusion -- do NOT keep trying scattered variants without "
        "consulting it. (It may ALREADY hold the PREVIOUS round's conclusion at the start -- if so, "
        "start your very first variant FROM it. If absent early on, that is fine; it will (re)appear "
        "once you have logged a couple of variants.)\n\n"
        f"JSON schema for {result_path} (write with a tiny script, not by hand):\n"
        '   {"metric": "<name>", "baseline": <number>, "method": <number|null>, '
        '"candidate_valid": <true|false; copy the repository evaluator gate outcome>, '
        '"attempt_id": "<attempt id emitted by ar-attempt for this method value>", '
        '"lower_is_better": <true|false>, "notes": "<what you changed>"'
        ', "config": "<the run configuration used for this result: model size/depth, '
        '#iterations or #steps, batch size, dataset>"'
        ', "baseline_std": <number|omit>, "baseline_runs": [<per-seed numbers>|omit]'
        ', "method_std": <number|omit>, "method_runs": [<per-seed numbers>|omit]'
        ', "changes_summary": "<concretely WHAT you changed this round: the specific '
        'edits/new code, as a short pseudo-diff or bullet list; ALSO list the KEY variants you '
        'tried this round and each variant\'s measured result (e.g. \'A: +rmsnorm fuse -> 0.021ms; '
        'B: +bigger tile -> 0.019ms (best); C: fp8 -> diverged\'), so the analysis can see WHICH '
        'variant best fit the method and why>"'
        ', "rationale": "<WHY you made those changes: the hypothesis you were testing and '
        'what you expected them to fix>"'
        ', "bottleneck": "<after this round, what is the CURRENT performance bottleneck or the '
        'thing that did NOT meet expectation -- i.e. what a next round should attack>"'
        ', "comparison_audit": {"metric_measured": "<name>", '
        '"baseline_budget_seconds": <number|null>, "method_budget_seconds": <number|null>, '
        '"conditions_held_constant": ["<condition>", ...], '
        '"method_factors_changed": ["<factor>", ...], '
        '"observed_outcomes": {"<name>": "<value>"}, "deviations": ["<deviation>", ...]}'
        ', "ablation_report": [{"component":"<name>","type":"direct_gain|indirect_gain",'
        '"actual":<finite number>,"expected":{"operator":"<|<=|>|>=|==",'
        '"value":<finite number>},"evidence":{"source":"<non-empty source>",'
        '"ref":"<non-empty ref>"},"verdict":"met|partial|unmet|low_confidence"}, ...]  '
        '(ONLY when the operator preamble explicitly requests a component contract measurement; '
        'run that current component\'s ablation_test (direct gain) or intermediate-property check '
        '(indirect gain) and '
        'report it; omit or [] otherwise). Each ablation_report.actual MUST be exactly one finite '
        'JSON number, never an '
        'object, list, or string; put auxiliary measurements in evidence or notes instead. Every '
        'ablation_report item MUST include these common fields: '
        '{"component":"<name>","type":"direct_gain|indirect_gain","actual":<finite number>,'
        '"expected":{"operator":"<|<=|>|>=|==","value":<finite number>},'
        '"evidence":{"source":"<non-empty source>","ref":"<non-empty ref>"},'
        '"verdict":"met|partial|unmet|low_confidence"}. '
        'Keep baseline and method as top-level fields, not only inside the metric object. '
        'Use exactly one ablation_report item per component; do not make a second item for '
        'the parent measurement. Each direct_gain item must additionally contain '
        '"parent_metric":<finite measured parent value> and '
        '"parent_evidence":{"source":"runtime_attempt","ref":"<parent attempt ref>"} '
        'inside that ablation_report item, not only in config, notes, expected, or evidence.ref text. '
        'For direct gain, including final integration, ablation_report.actual is the raw candidate metric '
        'selected by evidence.json_pointer, not a parent-minus-candidate delta substituted for that metric. '
        'The host computes signed_metric_improvement from the bound candidate and parent_metric; '
        'never copy contract_result.actual into ablation_report.actual. '
        'Require a successful host-registered control attempt for the exact provided parent '
        'checkout, code state, and evaluator; a stored number alone is not parent evidence. '
        'Change cwd BEFORE invoking ar-attempt: the host captures the launch checkout; '
        'changing directory inside its child script is too late. '
        'Do not restore baseline source over the dependency parent: it must retain '
        'all required ancestor components unchanged. '
        'If one already exists and its measured scalar matches, reuse that receipt without rerunning; '
        'otherwise measure the exact control through ar-attempt. Cite that control attempt '
        'separately from the candidate attempt. Never '
        'substitute an expected threshold or the original baseline for an unmeasured parent. '
        'indirect_gain items do not require a parent metric. During final integration, '
        'omit a previously accepted indirect component from this round\'s ablation_report '
        'unless its declared check is rerun on the integrated candidate. Never emit a '
        'literal passing scalar in place of that check; the accepted component ledger '
        'already retains the earlier evidence.}\n'
        'Bind evidence.ref to the candidate attempt ID and evidence.json_pointer to the '
        'finite scalar in that measured command\'s structured stdout. A JSON pointer is '
        'not a filesystem path: for {"measurements":{"residual":0.01}}, use '
        '"/measurements/residual", with actual=0.01. For direct gain, also set '
        'parent_evidence.json_pointer to the actual scalar field in the separate control '
        'attempt\'s stdout. That parent field must be the frozen evaluator metric_pointer; '
        'when none is declared, use /method or /baseline, not a custom measurements alias '
        'or an intermediate check count. If the saved control output already contains that '
        'valid primary scalar, correct only the result JSON binding; do not retrain or rerun '
        'a successful measurement. Omission defaults to /measurements/<component> for the candidate '
        'and /method for the parent; do not omit when those fields do not exist. Reference '
        'existing measured output; never edit saved stdout to make a pointer resolve.\n'
        'For agent-authored component checks, save disposable JSON/text reports under '
        'analysis/<component-id>/<attempt-id>/, not beside implementation files. '
        'Keep check scripts and shell wrappers outside the candidate checkout: analysis/*.py '
        'and analysis/*.sh can be copied into the source checkpoint and violate file restrictions. '
        'Keep their evidence references; do not inherit a previous report as a fresh measurement. '
        'Preserve repository-required output locations and explicitly declared runtime artifacts. '
        'If parent checkpoints conflict on a generated report, distinguish evidence-output '
        'collisions from implementation conflicts before revising component dependencies.\n'
        "The fields changes_summary/rationale/bottleneck are REQUIRED and are read by "
        "a reflection step that checks your changes against the research method and decides the next "
        "direction -- be concrete and honest (if a change did NOT help, say so and why you think so).\n"
        "★ BUDGET DISCIPLINE (critical -- an agent before you burned its ENTIRE wall-clock re-running "
        "the baseline 100+ times and never wrote a single method value; do NOT repeat that). The "
        "point of this task is a REAL IMPROVEMENT number, not a perfect baseline. Rules: (1) spend at "
        "most the FIRST THIRD of your time establishing the baseline, and do NOT re-run the full "
        "baseline more than TWICE -- one solid baseline number is enough, stop re-measuring it. (2) "
        "Then move to IMPROVEMENT and get at least ONE real `method` number written to disk as early "
        "as possible. (3) Once a variant improves the metric, you do NOT need to keep exploring -- run "
        "your BEST variant once more to confirm, finalize `method`, and STOP. (4) When roughly "
        "TWO-THIRDS of your time is gone, immediately STOP exploring, write your best measured "
        "`method` value, and end the run -- a written improvement always beats endless re-measurement. "
        + (f"(5) ATTEMPT CAP: after the baseline, run AT MOST {_variant_cap()} train/eval attempts this "
        f"round. EVERY attempt counts, including a command that crashes, times out, or produces no "
        f"metric. Prefix every train/eval invocation with `timeout`; the sandbox counts completed "
        f"invocations and refuses attempt {_variant_cap() + 1}. If {_variant_cap()} attempts finish "
        "without beating the baseline, STOP and write your best measured `method` value (a completed "
        "worse-than-baseline number is a valid honest negative -- the outer loop will change the method "
        f"next round); do NOT keep spawning new variants past {_variant_cap()}.\n"
        if _variant_cap() else "(5) There is no fixed sandbox attempt cap. Preserve execution "
        "receipts and update the structured result after each completed candidate.\n")
        +
        "ALWAYS fill \"baseline\" once the unmodified benchmark ran even once. Set \"method\" to "
        "null ONLY if not a single modified run ever completed (e.g. every variant crashed); if "
        "ANY variant produced a number, put that number in \"method\" -- worse-than-baseline "
        "included."
        + repeat_line
        + "\n\n"
        + (
            "WALL-CLOCK: there is no HARD total cap, but you win by trying quick variants, NOT one long "
            "training. Keep EACH training/eval run bounded: wrap every run as "
            f"`timeout --foreground {_pr}s <cmd>` (~{_pr // 60} min max per run). Choose runs that can complete "
            "within that cap and prioritize getting a measured method result. Make sure at least ONE "
            "modified variant actually "
            "COMPLETES within the cap and writes a real 'method' number (a completed worse-than-baseline "
            "run is a valid honest negative; an incomplete run gives nothing). Report REAL measured "
            "numbers."
            if budget_s is None else
            f"WALL-CLOCK BUDGET: you have ~{budget_s}s TOTAL and each run should be wrapped as "
            f"`timeout --foreground <N>s <cmd>`; size the workload (iterations/steps) to finish comfortably so "
            "you get BOTH a baseline and at least one improvement attempt. Report REAL measured "
            "numbers, never guesses."
        )
    )


def _evaluation_semantics_guidance(value: dict | None) -> str:
    if value is None:
        return ""
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (
        "\nLiterature-grounded evaluation semantics (canonical JSON): " + canonical + "\n"
        "Use this as the scientific meaning of the comparison. "
        "conditions_to_hold_constant are not score-improvement levers; "
        "method_freedom lists legitimate method choices. This guidance defines what the result "
        "means; it is not a request to invent a stricter config-equality gate.\n"
        "In notes, include one `COMPARISON_AUDIT: {...}` JSON record stating the metric measured, "
        "the baseline/method budget used where known, which declared conditions were held "
        "constant, method factors changed, observed outcomes such as steps/tokens/throughput, "
        "and every deviation. Do not claim evidence you did not observe.\n"
    )


def _baseline_task_prompt(request: str, repo_path: str, result_path: str,
                          metric_hint: str | None, lower_is_better_hint: bool | None,
                          confirmed_boundary: dict | None = None,
                          evaluation_semantics: dict | None = None) -> str:
    metric = metric_hint or "the repository's canonical primary metric"
    direction = ("lower is better" if lower_is_better_hint is True else
                 "higher is better" if lower_is_better_hint is False else
                 "determine the direction from the repository")
    boundary = json.dumps(
        confirmed_boundary or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # A canonical baseline may legitimately take longer than one method-candidate run.
    # Its repository-owned timer remains authoritative; only the overall sandbox safety
    # cap bounds a lost command.  Keep this consistent with _AttemptAuthority.
    return (
        "BASELINE-ONLY PHASE. METHOD EXPERIMENTS ARE OUT OF SCOPE. Establish one real baseline.\n"
        "Human-injected ideas and retained optimization source are Method proposals, not the "
        "selected repository's baseline. Preserve them for the Method phase. If execution uses "
        "a staged/runtime copy outside the checkout, verify its origin and diff against the "
        "selected repository baseline too: a clean Git worktree alone does not prove that the "
        "executed copy excludes candidate changes. Record environment bootstrap separately.\n"
        "Baseline-only excludes the proposed new Method, not the baseline algorithm's own optimization or intervention. "
        "Run the selected repository's existing baseline procedure through the task's required outcome; "
        "evaluating its untouched starting state is insufficient when the task requires that procedure. "
        "Do not substitute a feasibility/quality measurement for the optimization objective. "
        "For a constrained objective, separate successful reproduction from measured eligibility. "
        "An unmodified official starter with a defined, actually measured objective remains a valid "
        "baseline even when it misses a quality/feasibility threshold. Do not optimize "
        "the starter just to make baseline reproduction pass. Preserve its measured objective, "
        "quality observations, and failed eligibility checks. Emit "
        "baseline_outcome={feasible: <boolean>, "
        "quality_observations: <object>, reason: <explanation citing official checks>} in the "
        "measured stdout JSON and ar_result.json when eligibility is measured. The raw objective "
        "belongs in baseline, with its metric name, unit, direction, and actual measurement evidence. "
        "Record a task-declared target separately in notes with its source; do not invent a "
        "target or an official score that the available evaluator did not return. "
        "This does not waive execution, correctness of "
        "the implementation, or task applicability. Do not relabel crashes or an undefined "
        "objective as an infeasible but reproduced baseline. Published baseline values and "
        "reference anchors are context, never substitutes for local measurements.\n"
        "Resolve explicitly provisional metric/direction hints using the repository's task "
        "definition. Keep its raw optimization objective separate from any available normalized reward "
        "and feasibility observations; do not switch to reward merely to match an earlier "
        "generated direction or audit recommendation. Inspect the source evidence first.\n"
        "For cost-to-target objectives, record whether this baseline actually attained the target. "
        "Zero interventions on an untouched starting state do not measure the cost to attain "
        "an unmet target. Keep that starting-state measurement as context, not the baseline scalar; "
        "do not replace it with a budget limit or an invented failure penalty. A measured zero "
        "is valid if the target was already attained.\n"
        "Baseline validity does not require improvement headroom: a correctly executed, task-compliant, "
        "finite, already optimal baseline remains valid. Report evidence for any optimality claim "
        "as a limitation on possible improvement, not a reproduction failure. "
        "Do not weaken or replace it merely to make an improvement possible.\n"
        f"Repository: {repo_path}\nResearch goal: {request}\n"
        f"Metric: {metric}; {direction}.\n\n"
        f"{_EXECUTION_STATUS_GUIDANCE}"
        f"{_RESEARCH_FEEDBACK_BOUNDARY}"
        f"{_DEVICE_PREFLIGHT_GUIDANCE}"
        f"{_ATTEMPT_INVOCATION_GUIDANCE}"
        f"Confirmed user requirements: {boundary}. Follow them when selecting the baseline procedure.\n"
        + _evaluation_semantics_guidance(evaluation_semantics)
        + "Do not list, open, search, copy, import, execute, or recover through Git history any "
        "path whose directory name is `solution` or `solutions`; those paths are unavailable "
        "benchmark answers even when present in the public clone. "
        + "Before choosing or launching the baseline command, inspect the cloned repository itself. "
        "Read the task-relevant README/documentation, official run scripts, task manifests, configs, "
        "and real entrypoints. Bind the user's requested experimental scenario to the repository's "
        "scenario-specific recommended recipe. Do not treat a generic parser/config default as the "
        "right task configuration when repository documentation gives a different recipe for the "
        "requested scale, budget, hardware, or evaluation mode. Explicit user requirements and frozen "
        "evaluation semantics take precedence; among repository choices, use the best-supported "
        "scenario-specific recipe and report any remaining ambiguity instead of silently guessing. "
        "In the result notes, include one `REPOSITORY_INSPECTION: {...}` JSON record naming the inspected "
        "repository paths, the scenario evidence they provide, the selected command/config, rejected "
        "alternatives, and every deviation. Only then run the baseline. "
        "Use the repository's own documented build, data and evaluator. Keep the repository source "
        "unmodified. Resolve dependencies in this order: the repository's locked environment, "
        "a compatible preinstalled environment, the session-shared venv/package/data caches, and "
        "only then a direct online install. If `$AUTORESEARCH_DATA_ROOT` is set, reuse compatible "
        "data from that mounted location before downloading it. If the repository expects a fixed "
        "path such as `/data`, inspect the exact layout it requires and create a container-local "
        "compatibility view of the existing files; do not assume the mounted root already has that "
        "layout. Never clean, prune, delete, or replace a shared package/data cache. Treat mounted "
        "shared datasets and pre-existing cache contents as read-only inputs; write checkpoints, "
        "logs, and temporary outputs under the current session workroot. "
        "Save any auxiliary benchmark driver, its dependency manifest, raw measurements "
        "and correctness outputs in `$AUTORESEARCH_ARTIFACT_DIR/baseline/`, not container-local "
        "`/tmp`; record their exact paths in ar_result.json. This writable mounted directory survives "
        "container cleanup and lets the method reuse the same evaluation. Do not replace a missing "
        "repository baseline algorithm with a self-written weaker one. An auxiliary driver must "
        "call the actual repository implementation, not reimplement or translate its algorithm "
        "and describe that port as an unmodified repository baseline. If the selected repository "
        "cannot supply the required executable path, report that limitation. Shared locations provide "
        "inputs and dependencies only: do not treat result, analysis, checkpoint, or direction "
        "metadata found outside the current repository worktree as evidence for the current method. "
        "Every network "
        "call must use the direct connection already enforced by the environment. After checking "
        "the mounted caches, download a missing repository-required dependency or dataset directly "
        "only when the Research goal, confirmed user requirements, and applicable project/session "
        "policy do not require prior approval. If any boundary requires approval for a sizeable "
        "uncached pull, do not start it: report the URL, expected size, purpose, and cache status as "
        "an environment blocker and wait for supervisor approval. Never weaken repository signature "
        "verification or mark an unverified source trusted to make setup proceed. "
        "Install dependencies into `$UV_PROJECT_ENVIRONMENT`. If that path does not exist yet, "
        "create it first with `uv venv \"$UV_PROJECT_ENVIRONMENT\"` using a repository-compatible "
        "Python, then install with `uv sync` or `uv pip --python \"$UV_PROJECT_ENVIRONMENT/bin/python\"`; "
        "never use bare `pip` against the container's system Python. Before scoring, verify that any "
        "project package imported by the benchmark resolves inside the current repository worktree, "
        "not a prior round's editable install left in the session-shared venv. If it does not, bind "
        "the current worktree for this command only; do not modify the shared environment or package "
        "cache.\n"
        "If the selected code path loads a native/CUDA extension or remotely distributed kernel, "
        "preflight that exact path with an import and the smallest supported forward/backward before "
        "a full run. Compare the repository constraints, the extension's published or locally cached "
        "compatibility metadata, native-library ABI/SONAME requirements, and available cached runtime "
        "candidates; use a version only when those sources have a compatible intersection. Never fake "
        "compatibility by renaming a variant directory, copying a single shared library, or ignoring an "
        "ABI mismatch. If no compatible local intersection exists, record the missing artifact as an "
        "environment blocker instead of churning through runtime versions.\n"
        "Run benchmark commands non-interactively. If optional experiment tracking or telemetry "
        "would request login or consent, use the repository's supported disabled/offline mode; "
        "never wait for interactive stdin.\n"
        "Run the unmodified repository until one real finite metric is produced. Any command that "
        "launches training, including a tiny-train smoke test, diagnostic, profiling run, ablation, "
        "sweep, or evaluator/scoring invocation, is a physical experiment and MUST start with "
        "`ar-attempt`; do not use unledgered training output as evidence or selection feedback. "
        "Launch the canonical baseline scoring run as `ar-attempt <canonical baseline command>` and include that emitted "
        "`attempt_id` as the top-level attempt_id in ar_result.json. "
        "The host reads the saved stdout of that exact attempt; ar_result.json alone is not measured evidence. "
        "Emit one unambiguous JSON object in measured stdout. When the repository declares "
        "evaluation metric_pointer and gate_pointers, preserve those fields; otherwise the "
        "default stdout fields are baseline and baseline_valid. If the official evaluator "
        "writes files or another output format, use a wrapper launched by ar-attempt that "
        "runs the unmodified official command and exports its fresh result and actual "
        "correctness/quality checks to that JSON. Send progress logs to stderr; do not change "
        "the evaluator or count an export script alone as a new measurement. The wrapper "
        "must not print copied historical numbers, assumed passing gates, or guessed values. "
        "When the task restricts submitted or editable files, keep measurement-only wrappers "
        "outside the candidate repository working tree (for example in "
        "$AUTORESEARCH_ARTIFACT_DIR, which is available before launch and survives container cleanup) "
        "and invoke them by absolute path. They must "
        "not appear in the candidate diff or modify protected evaluator files. "
        "Do not put `timeout` inside or "
        "outside this baseline scoring command. Do not wrap the "
        "canonical baseline command in a command-level timeout: setup, compilation and the canonical "
        "evaluation may legitimately run outside a training-only budget. Let the repository's own "
        "training timer enforce that budget and let the overall sandbox safety cap handle a genuinely "
        "lost conversation. "
        f"Then write {result_path} as JSON containing metric, "
        "baseline, baseline_valid, method=null, lower_is_better, config, and notes. "
        "Set baseline_valid=true only after the repository's implementation correctness checks pass AND "
        "the actual measured run satisfies the explicit user requirements and frozen evaluation "
        "semantics. Cite the check command, output, and task applicability evidence in notes. "
        "A failed eligibility threshold alone does not make baseline_valid=false: record "
        "it in baseline_outcome with feasible=false and the actual failed checks. Keep the measured "
        "objective in baseline; never replace it with an assumed zero reward. "
        "Any quality_observations.measured_<metric_name> must agree with baseline. "
        "If the official "
        "evaluator combines implementation correctness and eligibility into one boolean, preserve "
        "that original output and explain the separate checks; never claim that combined gate passed. "
        "A wrapper may export a completed ineligible evaluation, but must not hide a crash, missing "
        "result or failed implementation check. Method candidates still must pass every official "
        "correctness and quality gate to qualify for a score. "
        "Reporting a deviation does not authorize it: a correct run on a different problem or "
        "unsupported task configuration is not a qualifying baseline. Set baseline_valid=false "
        "if correctness or task applicability fails or remains unverified; retain any finite "
        "measurement as an observation and explain the unsupported requirement for recovery. "
        "The config is descriptive only. "
        "If config.repo_commit or config.commit is reported, it identifies the selected outer "
        f"repository shown above: read it with `git -C {shlex.quote(repo_path)} rev-parse HEAD`. "
        "Record each nested dependency repository's path and commit separately in config.dependencies; "
        "do not substitute its revision for the selected repository's identity. "
        "`metric` MUST be an object with non-empty `name`, `unit`, and `lower_is_better`, naming "
        "the measured task quantity or evaluator-defined composite score for each arm; for example "
        "`{\"name\":\"runtime_ms\",\"unit\":\"ms\",\"lower_is_better\":true}`. "
        "If the repository task manifest declares the optimization metric, copy its exact metric "
        "identifier into metric.name; describe synonymous quantities in notes, not metric.name. "
        "When baseline_valid=true, `baseline` MUST be one finite JSON number: put uncertainty, "
        "per-run statistics, and other metadata in config or notes, never an object, array, or string. "
        "If the requested objective is undefined or unmeasured, write baseline=null and "
        "baseline_valid=false; keep auxiliary observations in notes. This is an honest failed "
        "reproduction for recovery, not permission to invent a finite number. "
        "The top-level baseline and method numbers are compared directly, so they MUST represent "
        "the same quantity, unit, and direction. A derived ratio of a single raw quantity "
        "(for example, reference runtime divided by candidate runtime) can use the same directly "
        "measured quantity for both phases only when that representation preserves the declared "
        "objective's ordering. Use its corresponding direction and report the ratio only after "
        "both measurements exist; never label a raw denominator as the derived metric. "
        "Do not replace a composite metric with one constituent. For a multi-quantity score, "
        "retain the declared metric and all raw constituents, using the same declared formula "
        "and reference for baseline and method. A baseline self-reference is valid only when "
        "that formula defines it from the measured baseline. Do not invent a baseline score "
        "or reference; if required measurements or the scoring definition are missing, record "
        "the gap with baseline_valid=false. "
        "Before writing descriptive configuration, reconcile it with the executed repository source "
        "and observed totals (for example steps, tokens per step, and total tokens). If prose and "
        "runtime evidence disagree, report the discrepancy and use the executed settings; do not "
        "invent or copy an arithmetically inconsistent configuration. "
        "Stop after the baseline is recorded."
    )


def _method_task_prompt(request: str, repo_path: str, result_path: str, budget_s: int | None,
                        frozen_baseline: dict,
                        evaluation_semantics: dict | None = None) -> str:
    metric = frozen_baseline.get("metric_name")
    value = frozen_baseline.get("metric_value")
    direction = bool(frozen_baseline.get("lower_is_better"))
    fixed_config = str(frozen_baseline.get("fixed_config") or "not recorded")
    repository_contract = frozen_baseline.get("repository_contract")
    contract_line = (
        "AUTHORITATIVE REPOSITORY TASK CONTRACT (verbatim host-captured documents):\n"
        + json.dumps(repository_contract, ensure_ascii=False, sort_keys=True)
        + "\nThe candidate must obey every applicable rule in these documents. A better metric "
          "does not make a prohibited implementation valid.\n"
        if isinstance(repository_contract, dict) and (
            repository_contract.get("sources") or repository_contract.get("execution_contract")) else ""
    )
    baseline_evidence = frozen_baseline.get("measurement_evidence") or {}
    baseline_outcome = baseline_evidence.get("baseline_outcome") or {}
    baseline_gates = baseline_evidence.get("gates") or {}
    baseline_qualification = {
        "feasible": (all(baseline_gates.values()) if baseline_gates
                     and all(type(value) is bool for value in baseline_gates.values()) else None),
        "quality_observations": {
            key: value for key, value in
            (baseline_outcome.get("quality_observations") or {}).items()
            if re.sub(r"[^a-z0-9]", "", str(key).casefold()) not in {
                "officialscore", "officialreward"}},
    } if baseline_outcome else {}
    candidate_timeout = _per_run_cap(repository_contract)
    verifier_timeout = _verifier_timeout(repository_contract)
    verifier_timeout_line = (
        f"VERIFIER SUBSTAGE LIMIT: the selected task manifest gives the official verifier "
        f"{verifier_timeout}s. Inside every candidate/control driver, wrap each invocation of "
        f"that official verifier with `timeout --foreground --kill-after=2s "
        f"{verifier_timeout}s <official verifier command>`. The outer {candidate_timeout}s "
        "ar-attempt cap covers the combined setup/compile/train/evaluate command and does not "
        "replace this verifier-only limit. A verifier timeout (exit 124) is an invalid attempt: "
        "do not emit a metric or success result from it. The attempt child also receives the "
        "same value as `$AUTORESEARCH_VERIFIER_TIMEOUT_S` for use in a driver.\n"
        if verifier_timeout is not None else ""
    )
    budget_line = (f"WALL-CLOCK BUDGET: {_agentic_hard_cap()}s total sandbox hard limit, "
                   "including setup, all attempts, evaluation, and result writing. "
                   f"Keep the final {min(_agentic_finalize_reserve(), _agentic_hard_cap())}s "
                   "within that limit for recording results, not launching another candidate. "
                   "Before a new attempt, check elapsed time and whether its full run, evaluation, "
                   "and result handoff fit the remaining budget; otherwise finish with existing "
                   "measured evidence.\n"
                   if budget_s is None else
                   f"WALL-CLOCK BUDGET: you have ~{budget_s}s experiment budget; "
                   f"the final {_agentic_finalize_reserve()}s are reserved only for writing "
                   f"{result_path} and must not be used to launch another candidate.\n")
    if budget_s is None and os.environ.get("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT") == "1":
        budget_line = ("No framework total wall-clock limit is imposed. Preserve the repository's "
                       "training, data, evaluation and compute budgets.\n")
    return (
        "METHOD-ONLY PHASE. A baseline was measured in the preceding sandbox.\n"
        "Measured baseline qualification (not a published anchor): "
        + json.dumps(baseline_qualification, ensure_ascii=False, sort_keys=True)
        + ". An ineligible starter does not relax any candidate correctness or quality gate. "
        "Distinguish raw objective improvement from attaining feasibility; do not infer an "
        "unavailable normalized score.\n"
        "EVIDENCE HANDOFF: give every component measurement an explicit evidence.json_pointer "
        "to the scalar actually emitted by its command; component IDs are not output field names. "
        "Keep raw candidate metric, raw parent metric, and derived improvement separate. "
        "Cite one exact attempt ID per reference, with explanations in a different field. "
        "Retain prior component attempt/checkpoint references when integrating; do not copy their "
        "values into a new output as if they were remeasured. Changed measured dependencies need "
        "a fresh check, not necessarily retraining. Run different training configurations as "
        "separate top-level ar-attempt commands, saving each result before starting the next; "
        "do not nest ar-attempt. Required repeated measurements of one configuration remain "
        "one evaluation protocol. If a later command times out, preserve earlier completed "
        "attempts and never report the timed-out command as successful.\n"
        f"{_EXECUTION_STATUS_GUIDANCE}"
        f"{_RESEARCH_FEEDBACK_BOUNDARY}"
        f"{_DEVICE_PREFLIGHT_GUIDANCE}"
        f"{_ATTEMPT_INVOCATION_GUIDANCE}"
        f"Repository: {repo_path}\nResearch goal: {request}\n"
        f"Reference metric: {metric}={value}; lower_is_better={direction}. Use it as context for "
        "judging progress. Do not rerun or recalibrate the baseline; it is frozen and baseline "
        "attempts are out of scope. This does not prohibit a required parent/control measurement "
        "in the designated unchanged control checkout via ar-attempt. Such a control run "
        "never replaces the frozen BaselineResult; preserve its original settings and budget. "
        "Reuse an existing matching host control receipt when available. A baseline artifact path "
        "is not a host attempt receipt. "
        f"Frozen baseline configuration: {fixed_config}. Use it to preserve held-constant conditions; "
        "treat that prose as descriptive rather than higher-authority execution evidence. Reconcile "
        "it with the frozen repository source and observed totals such as steps, tokens per step, "
        "and total tokens. If they disagree, use the actually executed baseline settings and record "
        "the metadata discrepancy instead of reproducing an arithmetically inconsistent string. "
        "change only method factors allowed by the research goal. Follow the research goal and the "
        "repository's own benchmark semantics. "
        "Reuse the persisted baseline evaluation driver and fixed inputs, if any, from "
        "`$AUTORESEARCH_ARTIFACT_DIR/baseline/` without changing their scoring semantics. Keep "
        "candidate implementation source, including candidate-specific drivers, inside the "
        "current repository worktree in task-permitted paths so the host-captured diff can review "
        "and carry forward the implementation. If the task names a submission artifact, a score "
        "from a temporary copy or alternate evaluation directory is only a development observation "
        "until that exact candidate is present at the task-permitted submission path in the current "
        "worktree and the unchanged official evaluator measures it there. Do not report a temporary "
        "copy's score as the final method result. Before an attempt, verify that every changed "
        "candidate source is visible to Git at this outer repository root, not only inside a "
        "nested checkout. If an allowed source lies under an ignored directory, stage only "
        "those explicit source files in the outer index and verify `git diff HEAD -- <files>`. "
        "For a nested Git checkout, `git add` may silently leave the outer index unchanged; "
        "use the actual file's Git blob hash and mode with outer `git update-index --add --cacheinfo` "
        "when needed, then verify the diff again. Do not add datasets, weights, environments, "
        "or change HEAD. An invisible "
        "source cannot be reviewed or carried by the checkpoint. Do not keep the only candidate source in the "
        "external artifact directory. Each ar-attempt child receives its own "
        "`$AUTORESEARCH_ATTEMPT_ARTIFACT_DIR` and `$AUTORESEARCH_ATTEMPT_ID`. Read this output "
        "directory inside the launched training/evaluation driver, and save its weights, logs "
        "and result files there; record the exact paths in ar_result.json. Do not reuse a shared "
        "method/best_model path between attempts. Treat previous attempts' output paths and "
        "parent checkpoints as read-only inputs, even when an inherited driver used a shared "
        "output path; update that driver's output destination before rerunning it. "
        "do not overwrite baseline evidence or keep required reproducibility files only in `/tmp`. "
        + contract_line
        + verifier_timeout_line
        +
        "Do not list, open, search, copy, import, execute, or recover through Git history any path "
        "whose directory name is `solution` or `solutions`; those paths are unavailable benchmark "
        "answers even when present in the public clone. "
        "When the task or official repository reserves a final test for isolated evaluation, "
        "do not run setup or data-generation commands that create its examples or labels in the "
        "editing sandbox, even if the generator is public; leave that setup and access to the "
        "isolated official evaluator. If the repository authorizes local evaluation, prepare fresh "
        "fixtures using its unchanged frozen setup source and cached inputs without inspecting "
        "their contents or changing existing protected data or scoring semantics. Only the official "
        "evaluator may consume evaluation-only examples and labels. This permission does not "
        "override protected-path or data-access rules, or authorize candidate code to read labels. "
        "Record environment setup separately from the candidate treatment. A held-out split is not "
        "automatically final-only; follow the repository's declared feedback permissions. "
        "Treat final held-out/test labels and evaluator-only examples as evaluation-only data: do not "
        "load their labels or use their individual examples to tune, diagnose, rank, or select a "
        "candidate. When the original task or official repository permits a development validation "
        "set for model selection or calibration, use that set within those permissions; do not "
        "reinterpret it as a forbidden final test set. Otherwise select candidates using training "
        "data or a deterministic train-derived validation split. When reporting validation results, "
        "ensure the evaluated examples were not used to fit the evaluated model; if a final model "
        "is refit on all training examples, its score on that split is an in-sample diagnostic. "
        "Preserve the official development "
        "and final-test roles: validation feedback is not an independent final-test result, and "
        "permission to tune on validation does not permit label-dependent answers at final inference. "
        "When the task reserves a test command or score for final evaluation, do not run it or "
        "inspect its aggregate score while choosing, debugging, or ablating candidates, even if "
        "the test data are locally readable; choose with the declared development feedback and "
        "run the final test only after fixing the candidate. "
        "If a component plan names that final test score or reward as a success criterion, "
        "mark it unverified until complete-method evaluation; the plan does not grant earlier "
        "test access. Do not report a development proxy as proof that the final eligibility "
        "gate passed. If the component's direct-gain contract cannot be decided without the "
        "reserved final test, request component-plan revision rather than running the test. "
        "Invoke the official evaluator within the task's declared feedback and evaluation protocol. "
        "Focus on implementing and measuring a method variant. Do not create a git commit or "
        "change HEAD; leave candidate source changes as a working-tree diff so they can be "
        "measured against the frozen baseline commit.\n"
        + _evaluation_semantics_guidance(evaluation_semantics)
        + "Use ordinary shell commands for repository inspection, environment setup, source editing, "
        "and writing ar_result.json; do NOT prefix those non-experiment commands with `ar-attempt`. "
        "Only a command that actually launches candidate training and/or its scoring evaluation is "
        "a candidate attempt and consumes the bounded attempt budget. "
        "Exploratory or diagnostic profiling, ablation, sweep, train-derived validation, and "
        "sanity-check commands also count whenever they launch candidate training or scoring; "
        "do not run them directly from `/tmp` or otherwise outside `ar-attempt`. "
        + "Start EVERY candidate command with `ar-attempt` (for example, "
        f"`ar-attempt timeout --foreground {candidate_timeout}s <command>`); this configured "
        "per-run safety cap must cover setup, compilation, training, and final evaluation. It records "
        "the attempt at launch, so crashes, "
        "timeouts, and missing metrics consume the same budget as successes. "
        "When the comparison uses a training-only budget, keep that training timer unchanged but "
        "give the bounded candidate command enough additional time for setup, compilation, and final evaluation. "
        "For a multi-phase method, count all optimizer-step time across every phase against that "
        "single training budget and apply any compilation exclusion only once globally. "
        "Reuse mounted caches. If `$AUTORESEARCH_DATA_ROOT` is set, reuse data from that mounted "
        "location before considering any download. If the repository expects a fixed path such as "
        "`/data`, inspect the exact layout it requires and create a container-local compatibility "
        "view of the existing files; do not assume the mounted root already has that layout. "
        "Shared locations provide inputs and dependencies only: do not treat result, analysis, "
        "checkpoint, or direction metadata found outside the current repository worktree as "
        "evidence for the current method. "
        "After checking mounted caches, download a missing repository-required dependency or dataset "
        "directly only when the Research goal, confirmed user requirements, and applicable "
        "project/session policy do not require prior approval. If any boundary requires approval for "
        "a sizeable uncached pull, do not start it: report the URL, expected size, purpose, and cache "
        "status as an environment blocker and wait for supervisor approval. Never weaken repository "
        "signature verification or mark an unverified source trusted to make setup proceed. "
        "Install dependencies into `$UV_PROJECT_ENVIRONMENT`. If that path does not exist yet, create it "
        "first with `uv venv \"$UV_PROJECT_ENVIRONMENT\"` using a repository-compatible Python, then use "
        "`uv sync` or `uv pip --python \"$UV_PROJECT_ENVIRONMENT/bin/python\"`; never use bare `pip` "
        "against the container's system Python. "
        "Before scoring, verify that any project package imported by the benchmark resolves inside the "
        "current repository worktree, not a prior round's editable install left in the session-shared "
        "venv. If it does not, bind the current worktree for this command only; do not modify the shared "
        "environment or package cache. "
        "Every launched candidate counts, including crashes, "
        "timeouts, and missing metrics; retain every execution receipt. "
        + (f"Run at most {_variant_cap()} candidates.\n" if _variant_cap()
           else "There is no fixed sandbox attempt cap.\n")
        +
        "Run candidates non-interactively. If optional experiment tracking or telemetry would "
        "request login or consent, use the repository's supported disabled/offline mode; never "
        "wait for interactive stdin. If a tool call is rejected, correct its arguments on the "
        "next turn; do not write tool-call markup as chat text.\n"
        "Before repeating a failed implementation, verify that every confirmed component's stated "
        "precondition and semantic contract actually hold in the frozen repository. If tracked "
        "source or a real failed attempt proves that a confirmed component is already present, has "
        "no real delta, or cannot be implemented without changing the task semantics, do not keep "
        "debugging or substitute a different method inside this sandbox. Restore any trial edits, "
        "leave method=null, and record a structured preflight component failure in ablation_report: "
        '{"component":"<confirmed component>","type":"indirect_gain","actual": 0,'
        '"expected": {"operator": "==", "value": 1},'
        '"evidence":{"source":"tracked_repository_source|runtime_attempt",'
        '"ref":"tracked repository path+lines or failed attempt_id"},"verdict":"unmet"}. '
        "When no new candidate command ran, cite tracked_repository_source only for a prerequisite "
        "you actually inspected in the current dependency workspace. Keep historical attempt results "
        "as supporting notes; do not label them as this attempt's runtime measurement. "
        "This is a safe request for the outer method loop to redesign the method; it is not a "
        "successful result. Do not use this escape for an ordinary fixable code or environment "
        "error.\n"
        + budget_line
        + f"After every completed candidate, write {result_path} before further analysis or another attempt, "
        "including for an honest negative result. Use metric as an object whose "
        f"name is {metric!r}, unit is {frozen_baseline.get('metric_unit')!r}, and "
        f"lower_is_better is {direction!r}; write "
        f"baseline={value}, the real method metric (or null if none completed), lower_is_better, "
        "and top-level attempt_id copied exactly from the ar-attempt invocation that produced that "
        "method number. The checked-out diff, cited attempt, and method number must identify the "
        "same candidate. If you want to report an earlier best after trying later variants, restore "
        "that exact candidate diff and cite its original attempt; otherwise report the latest "
        "candidate's real number rather than combining a historical number with current code. "
        "Also write top-level candidate_valid=true only when every correctness or quality gate "
        "reported by the repository evaluator passed; write false when any gate failed. A finite "
        "primary metric with candidate_valid=false remains an honest observation but cannot qualify "
        "a component or become the accepted method. "
        "Keep repository validity separate from ComponentPlan acceptance and MIR fidelity. "
        "Do not AND custom component assertions, expected improvement, or implementation-fidelity "
        "checks into candidate_valid unless the repository itself requires those checks. "
        "Record a failed component check in ablation_report and its evidence artifact; preserve "
        "the real official metric and evaluator gate outcome even when that component is unmet. "
        "The component still cannot be accepted until its own frozen contract passes or is formally "
        "revised. An official gate failure must remain false; do not flip an old receipt after "
        "measurement or substitute an unsupported claim of validity. Emit these separate outcomes "
        "in the host-captured attempt output as well as the result file. "
        "ComponentPlan's custom test output does not replace this result envelope: preserve its "
        "raw report in a separate evidence artifact and reference it while retaining the required "
        "top-level metric, baseline, method, candidate_valid, and attempt_id fields. "
        "config, notes, changes_summary, rationale, bottleneck, and ablation_report when requested. "
        "Each ablation_report.actual MUST be exactly one finite JSON number, never an object, list, "
        "or string; put auxiliary measurements in evidence or notes instead. Every "
        "ablation_report item MUST include these common fields: "
        '{"component":"<name>","type":"direct_gain|indirect_gain","actual":<finite number>,'
        '"expected":{"operator":"<|<=|>|>=|==","value":<finite number>},'
        '"evidence":{"source":"<non-empty source>","ref":"<non-empty ref>"},'
        '"verdict":"met|partial|unmet|low_confidence"}. '
        'Keep baseline and method as top-level fields, not only inside the metric object. '
        'Use exactly one ablation_report item per component; do not make a second item for '
        'the parent measurement. Each direct_gain item must additionally contain '
        '"parent_metric":<finite measured parent value> and '
        '"parent_evidence":{"source":"runtime_attempt","ref":"<parent attempt ref>"} '
        'inside that ablation_report item, not only in config, notes, expected, or evidence.ref text. '
        'For direct gain, including final integration, ablation_report.actual is the raw candidate metric '
        'selected by evidence.json_pointer, not a parent-minus-candidate delta substituted for that metric. '
        'The host computes signed_metric_improvement from the bound candidate and parent_metric; '
        'never copy contract_result.actual into ablation_report.actual. '
        'Require a successful host-registered control attempt for the exact provided parent '
        'checkout, code state, and evaluator; a stored number alone is not parent evidence. '
        'Change cwd BEFORE invoking ar-attempt: the host captures the launch checkout; '
        'changing directory inside its child script is too late. '
        'Do not restore baseline source over the dependency parent: it must retain '
        'all required ancestor components unchanged. '
        'If one already exists and its measured scalar matches, reuse that receipt without rerunning; '
        'otherwise measure the exact control through ar-attempt. Cite that control attempt '
        'separately from the candidate attempt. Never '
        'substitute an expected threshold or the original baseline for an unmeasured parent. '
        'indirect_gain items do not require a parent metric. During final integration, '
        'omit a previously accepted indirect component from this round\'s ablation_report '
        'unless its declared check is rerun on the integrated candidate. Never emit a '
        'literal passing scalar in place of that check; the accepted component ledger '
        'already retains the earlier evidence. '
        'Bind evidence.ref to the candidate attempt ID and evidence.json_pointer to the '
        'finite scalar in that measured command\'s structured stdout. A JSON pointer is '
        'not a filesystem path: for {"measurements":{"residual":0.01}}, use '
        '"/measurements/residual", with actual=0.01. For direct gain, also set '
        'parent_evidence.json_pointer to the actual scalar field in the separate control '
        'attempt\'s stdout. That parent field must be the frozen evaluator metric_pointer; '
        'when none is declared, use /method or /baseline, not a custom measurements alias '
        'or an intermediate check count. If the saved control output already contains that '
        'valid primary scalar, correct only the result JSON binding; do not retrain or rerun '
        'a successful measurement. Omission defaults to /measurements/<component> for the candidate '
        'and /method for the parent; do not omit when those fields do not exist. Reference '
        'existing measured output; never edit saved stdout to make a pointer resolve. '
        'For agent-authored component checks, save disposable JSON/text reports under '
        'analysis/<component-id>/<attempt-id>/, not beside implementation files. '
        'Keep check scripts and shell wrappers outside the candidate checkout: analysis/*.py '
        'and analysis/*.sh can be copied into the source checkpoint and violate file restrictions. '
        'Keep their evidence references; do not inherit a previous report as a fresh measurement. '
        'Preserve repository-required output locations and explicitly declared runtime artifacts. '
        'If parent checkpoints conflict on a generated report, distinguish evidence-output '
        'collisions from implementation conflicts before revising component dependencies. '
        "Also append the measured variant to analysis/variants.jsonl as one complete JSON object per "
        "physical line with valid=true|false copied from the same repository gate outcome, "
        "terminate it with a newline, and validate every non-empty line with json.loads "
        "before starting another candidate. A worse real number is still a valid result; record it "
        "and stop when the attempt cap is reached."
    )


def _repo_commit(repo_path: str) -> str | None:
    try:
        proc = subprocess.run(
            # A container-created checkpoint may have a different UID. Trust only this
            # resolved workspace for this read, without changing global Git policy.
            ["git", "-c", f"safe.directory={os.path.realpath(repo_path)}",
             "-C", repo_path, "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10,
        )
        value = (proc.stdout or "").strip()
        return value if proc.returncode == 0 and value else None
    except Exception:  # noqa: BLE001 -- provenance is best-effort for injected non-git test clones
        return None


def _tracked_repo_state(repo_path: str) -> dict:
    """Hash tracked changes, excluding newly created protocol handoff outputs."""
    empty_hash = "sha256:" + hashlib.sha256(b"").hexdigest()
    try:
        base = ["git", "-c", "safe.directory=*", "-c", "core.fileMode=false", "-C", repo_path]
        # Writing/staging the handoff after measurement is not a code change. Only
        # exempt these protocol outputs if they were absent from the repository's HEAD;
        # existing same-name files and all other source/config changes remain checked.
        outputs = [_RESULT_BASENAME, "analysis/variants.jsonl"]
        owned = subprocess.run(base + ["ls-tree", "-r", "--name-only", "HEAD", "--"] + outputs,
                               capture_output=True, timeout=10, check=True).stdout.decode().splitlines()
        excludes = [":(top,exclude,literal)" + path for path in outputs if path not in owned]
        proc = subprocess.run(
            base + ["diff", "--no-ext-diff", "--binary", "HEAD", "--"] + excludes,
            capture_output=True, timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            return {"tracked_diff_hash": None, "tracked_diff_present": None}
        digest = "sha256:" + hashlib.sha256(proc.stdout).hexdigest()
        return {"tracked_diff_hash": digest,
                "tracked_diff_present": digest != empty_hash}
    except Exception:  # noqa: BLE001 -- an unreadable provenance state must fail closed downstream
        return {"tracked_diff_hash": None, "tracked_diff_present": None}


def _tracked_diff_paths(repo_path: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo_path,
             "diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD", "--"],
            capture_output=True, text=True,
            timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            return []
        return sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})
    except Exception:
        return []


def _validate_treatment_paths(repo_path: str, contract: dict,
                              immutable_hashes: dict[str, str]) -> list[str]:
    treatment = contract.get("controls", {}).get("treatment")
    allowed = treatment.get("allowed_paths") if isinstance(treatment, dict) else None
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("contract treatment.allowed_paths must be non-empty")
    normalized = []
    for item in allowed:
        if not isinstance(item, str) or not item or os.path.isabs(item):
            raise ValueError("treatment allowed path must be repository-relative")
        relative = os.path.normpath(item)
        if relative == ".." or relative.startswith(".." + os.sep):
            raise ValueError("treatment allowed path escapes repository")
        normalized.append(relative)
    changed = _tracked_diff_paths(repo_path)
    if any(path not in normalized for path in changed):
        raise ValueError("tracked treatment diff is outside allowed_paths")
    if any(path in immutable_hashes for path in changed):
        raise ValueError("tracked treatment diff touches immutable path")
    try:
        diff = subprocess.run(
            ["git", "-C", repo_path, "diff", "--no-ext-diff", "HEAD", "--"],
            capture_output=True, text=True,
            timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS).stdout
    except Exception:
        diff = ""
    if any(marker in diff for marker in (
            "AUTORESEARCH_HOST_PROBE", "AUTORESEARCH_PROBE_", "sitecustomize")):
        raise ValueError("tracked treatment diff contains reserved host-probe marker")
    return changed


_RISKY_UNTRACKED_SUFFIXES = {
    ".bash", ".c", ".cc", ".cfg", ".conf", ".cpp", ".cu", ".fish", ".go",
    ".h", ".hpp", ".ini", ".java", ".js", ".json", ".lua", ".pl", ".py",
    ".r", ".rb", ".rs", ".sh", ".toml", ".ts", ".yaml", ".yml", ".zsh",
}
_RISKY_UNTRACKED_NAMES = {"Dockerfile", "GNUmakefile", "Makefile", "makefile"}


def _untracked_file_inventory(repo_path: str) -> dict[str, dict]:
    paths: set[str] = set()
    for extra in ([], ["--ignored"]):
        completed = subprocess.run(
            ["git", "-C", repo_path, "ls-files", "--others", *extra,
             "--exclude-standard", "-z"],
            capture_output=True, timeout=_GIT_REPOSITORY_SCAN_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError("could not inventory untracked repository files")
        paths.update(
            os.fsdecode(item) for item in completed.stdout.split(b"\0") if item)
    inventory: dict[str, dict] = {}
    for relative in sorted(paths):
        full = os.path.abspath(os.path.join(repo_path, relative))
        if os.path.commonpath([os.path.realpath(repo_path), os.path.realpath(
                os.path.dirname(full))]) != os.path.realpath(repo_path):
            inventory[relative] = {"kind": "unsafe", "executable": False}
        elif os.path.islink(full):
            inventory[relative] = {
                "kind": "symlink", "target": os.readlink(full), "executable": False,
            }
        elif os.path.isfile(full):
            digest, stat_result = _regular_file_evidence(full)
            inventory[relative] = {
                "kind": "regular", "sha256": digest, "size": stat_result.st_size,
                "executable": bool(stat_result.st_mode & 0o111),
            }
        else:
            inventory[relative] = {"kind": "other", "executable": False}
    return inventory


def _path_declared(relative: str, declared: list[str]) -> bool:
    return any(relative == path or relative.startswith(path.rstrip(os.sep) + os.sep)
               for path in declared)


def _validate_untracked_files(before: dict[str, dict], after: dict[str, dict],
                              contract: dict, execution_phase: str) -> dict:
    evaluator = contract.get("controls", {}).get("evaluator", {})
    output_paths = evaluator.get("output_paths", []) if isinstance(evaluator, dict) else []
    treatment = contract.get("controls", {}).get("treatment", {})
    treatment_paths = (treatment.get("allowed_paths", [])
                       if execution_phase == "method" and isinstance(treatment, dict) else [])
    declared = [os.path.normpath(path) for path in [*output_paths, *treatment_paths]
                if isinstance(path, str) and path and not os.path.isabs(path)]
    added = sorted(set(after) - set(before))
    changed_or_removed = sorted(
        relative for relative in before
        if relative not in after or before[relative] != after[relative])
    known_host_outputs = {_RESULT_BASENAME}
    allowed: list[str] = []
    other: list[str] = []
    unsafe: list[str] = []
    for relative in [*added, *changed_or_removed]:
        evidence = after.get(relative, before.get(relative, {}))
        host_evidence = (relative in known_host_outputs
                         or (relative.startswith("analysis" + os.sep)
                             and os.path.splitext(relative)[1] in {".jsonl", ".txt"}))
        output_declared = _path_declared(relative, [
            os.path.normpath(path) for path in output_paths
            if isinstance(path, str) and path and not os.path.isabs(path)])
        if output_declared and evidence.get("kind") == "symlink":
            raise ValueError(f"artifact symlink is unsafe: {relative}")
        if host_evidence or _path_declared(relative, declared):
            if evidence.get("kind") == "symlink":
                unsafe.append(relative)
            else:
                allowed.append(relative)
            continue
        unsafe.append(relative)
    if unsafe:
        raise ValueError(
            "undeclared untracked additions or modifications: " + ", ".join(unsafe))
    return {
        "before": before, "after": after,
        "added_allowed": sorted(allowed), "added_other": sorted(other),
    }


def _normalize_device_model(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _query_device_models(device_ids: list[str]) -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
         ",".join(device_ids)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi model query failed")
    models = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(models) != len(device_ids):
        raise RuntimeError("nvidia-smi returned an unexpected device count")
    return models


def _resource_evidence(cuda_device: Any, query_fn: Any = None) -> dict:
    """Describe fair resource identity without treating scheduler ordinals as hardware identity."""
    devices = ([item.strip() for item in str(cuda_device).split(",") if item.strip()]
               if cuda_device is not None else [])
    caveat = "precision, seed, and model scale are contract-frozen, not host-observed"
    if not devices:
        return {
            "accelerator_count": 0, "device_models": [],
            "source": "allocation_count_only", "caveat": caveat,
        }
    try:
        raw_models = (query_fn or _query_device_models)(devices)
        models = sorted(_normalize_device_model(model) for model in raw_models)
        if len(models) != len(devices) or any(not model for model in models):
            raise ValueError("device model count mismatch")
        source = "nvidia-smi"
    except Exception:  # noqa: BLE001 -- unavailable observation is explicit, never fabricated
        models = ["unknown"] * len(devices)
        source = "allocation_count_only"
        caveat += "; device model not verified (nvidia-smi unavailable)"
    return {
        "accelerator_count": len(devices), "device_models": models,
        "source": source, "caveat": caveat,
    }


def _resource_pair_matches(current: Any, frozen: Any) -> bool:
    if not isinstance(current, dict) or not isinstance(frozen, dict):
        return False
    try:
        current_count = int(current["accelerator_count"])
        frozen_count = int(frozen["accelerator_count"])
        current_models = current["device_models"]
        frozen_models = frozen["device_models"]
    except (KeyError, TypeError, ValueError):
        return False
    if (isinstance(current.get("accelerator_count"), bool)
            or isinstance(frozen.get("accelerator_count"), bool)
            or current_count < 0 or current_count != frozen_count
            or not isinstance(current_models, list) or not isinstance(frozen_models, list)
            or len(current_models) != current_count or len(frozen_models) != frozen_count
            or any(not isinstance(model, str) or not model for model in [
                *current_models, *frozen_models])):
        return False
    normalized_current = sorted(_normalize_device_model(model) for model in current_models)
    normalized_frozen = sorted(_normalize_device_model(model) for model in frozen_models)
    if "unknown" in normalized_current or "unknown" in normalized_frozen:
        return True
    return normalized_current == normalized_frozen


def _validate_resource_contract(contract: dict, evidence: dict) -> None:
    resources = contract.get("controls", {}).get("resources")
    declared_count = resources.get("accelerator_count") if isinstance(resources, dict) else None
    if (not isinstance(declared_count, int) or isinstance(declared_count, bool)
            or declared_count < 0
            or declared_count != evidence.get("accelerator_count")):
        raise ValueError("contract accelerator_count differs from host allocation")
    models = evidence.get("device_models")
    unknown = not isinstance(models, list) or "unknown" in models
    for field in ("device_model", "resource_class"):
        declared = resources.get(field)
        if declared is None:
            continue
        expected = _normalize_device_model(declared)
        if not expected or unknown:
            raise ValueError(f"contract {field} cannot be verified from host allocation")
        if field == "device_model":
            matches = all(_normalize_device_model(model) == expected for model in models)
        else:
            matches = all(expected in _normalize_device_model(model) for model in models)
        if not matches:
            raise ValueError(f"contract {field} differs from host allocation")


def _normalize_command(value: Any, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"argv", "cwd"}:
        raise ValueError(f"{label} must contain exactly argv and cwd")
    argv = value.get("argv")
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(item, str) or not item or "\0" in item for item in argv)):
        raise ValueError(f"{label} argv must be a non-empty string array")
    for index, token in enumerate(argv[:-1]):
        if os.path.basename(token).startswith("python") and argv[index + 1] == "-c":
            raise ValueError("python -c cannot be used as a measurement entrypoint")
    cwd = value.get("cwd")
    if not isinstance(cwd, str) or os.path.isabs(cwd):
        raise ValueError(f"{label} cwd must be repository-relative")
    normalized_cwd = os.path.normpath(cwd or ".")
    if normalized_cwd == ".." or normalized_cwd.startswith(".." + os.sep):
        raise ValueError(f"{label} cwd escapes repository")
    return {"argv": list(argv), "cwd": normalized_cwd}


def _normalize_parser(value: Any, *, protocol: bool = False) -> dict:
    parser = value
    required = {"stream", "regex", "group", "unit"} if protocol else {
        "stream", "regex", "group"}
    allowed = required | ({"multiplier"} if protocol else set())
    multiplier = parser.get("multiplier", 1) if isinstance(parser, dict) else None
    try:
        multiplier_finite = math.isfinite(float(multiplier))
    except (OverflowError, TypeError, ValueError):
        multiplier_finite = False
    if (not isinstance(parser, dict)
            or not required <= set(parser) or set(parser) - allowed
            or parser.get("stream") not in {"stdout", "stderr", "combined"}
            or not isinstance(parser.get("regex"), str) or not parser.get("regex")
            or len(parser.get("regex")) > 512
            or not isinstance(parser.get("group"), (int, str))
            or isinstance(parser.get("group"), bool)
            or (protocol and parser.get("unit") not in {"seconds", "steps", "tokens"})
            or (protocol and (
                not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool)
                or not multiplier_finite or multiplier <= 0))):
        raise ValueError("measurement protocol_parser is invalid" if protocol
                         else "measurement parser is invalid")
    compiled = re.compile(parser["regex"])
    group = parser["group"]
    if ((isinstance(group, int) and not 0 <= group <= compiled.groups)
            or (isinstance(group, str) and group not in compiled.groupindex)):
        raise ValueError("measurement parser group does not exist")
    return dict(parser)


def _normalize_measurement_spec(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {"execution", "trusted_probe"}:
        raise ValueError("measurement must contain execution and trusted_probe")
    execution = _normalize_command(value.get("execution"), "measurement execution")
    probe = value.get("trusted_probe")
    if not isinstance(probe, dict):
        raise ValueError("measurement trusted_probe must be an object")
    kind = probe.get("kind")
    if kind == "external_scorer":
        if (not {"kind", "argv", "cwd", "parser", "artifact"} <= set(probe)
                or set(probe) - {
                    "kind", "argv", "cwd", "parser", "protocol_parser", "artifact"}):
            raise ValueError("external_scorer trusted_probe is invalid")
        command = _normalize_command(
            {"argv": probe.get("argv"), "cwd": probe.get("cwd")}, "external scorer")
        normalized_probe = {
            "kind": kind, **command, "parser": _normalize_parser(probe.get("parser")),
        }
        artifact = probe.get("artifact")
        if (not isinstance(artifact, dict)
                or set(artifact) != {"path", "producer_sha256_before", "required_new"}
                or not isinstance(artifact.get("path"), str) or not artifact["path"]
                or os.path.isabs(artifact["path"])
                or artifact.get("required_new") is not True
                # We deliberately implement only the safe required-new policy.  Supporting a
                # prior hash would require an atomic quarantine/restore channel outside the repo.
                or artifact.get("producer_sha256_before") is not None):
            raise ValueError("external_scorer artifact is invalid")
        artifact_path = os.path.normpath(artifact["path"])
        if artifact_path == ".." or artifact_path.startswith(".." + os.sep):
            raise ValueError("external_scorer artifact escapes repository")
        if sum(token.count("{artifact}") for token in command["argv"]) != 1:
            raise ValueError("external scorer argv requires one {artifact} token")
        normalized_probe["artifact"] = {**artifact, "path": artifact_path}
        if probe.get("protocol_parser") is not None:
            normalized_probe["protocol_parser"] = _normalize_parser(
                probe["protocol_parser"], protocol=True)
    elif kind == "python_callable":
        callable_keys = {"kind", "module", "callable", "source_path"}
        if (not callable_keys <= set(probe)
                or set(probe) - (callable_keys | {"protocol_parser"})):
            raise ValueError("python_callable trusted_probe is invalid")
        for field in ("module", "callable", "source_path"):
            if not isinstance(probe.get(field), str) or not probe[field].strip():
                raise ValueError(f"python_callable {field} must be non-empty")
        if os.path.isabs(probe["source_path"]):
            raise ValueError("python_callable source_path must be repository-relative")
        normalized_probe = dict(probe)
        normalized_probe["source_path"] = os.path.normpath(probe["source_path"])
        if probe.get("protocol_parser") is not None:
            normalized_probe["protocol_parser"] = _normalize_parser(
                probe["protocol_parser"], protocol=True)
            if normalized_probe["protocol_parser"]["stream"] != "stdout":
                raise ValueError("python_callable protocol_parser requires stdout")
    else:
        raise ValueError("trusted_probe kind must be external_scorer or python_callable")
    normalized = {"execution": execution, "trusted_probe": normalized_probe}
    return normalized


def _boundary_measurement(boundary: dict | None, proposed: Any) -> Any:
    boundary = boundary if isinstance(boundary, dict) else {}
    if isinstance(boundary.get("measurement_spec"), dict):
        return boundary["measurement_spec"]
    run_cmd = boundary.get("run_cmd")
    if run_cmd is None:
        return proposed
    parser = boundary.get("metric_parser")
    if not isinstance(run_cmd, str) or not isinstance(parser, dict):
        raise ValueError("boundary run_cmd requires a structured metric_parser")
    if any(char in run_cmd for char in ";&|><`$\n"):
        raise ValueError("boundary run_cmd contains shell control syntax")
    argv = shlex.split(run_cmd)
    command = {"argv": argv, "cwd": str(boundary.get("measurement_cwd") or ".")}
    proposed_execution = (proposed.get("execution")
                          if isinstance(proposed, dict) else None)
    proposed_probe = (proposed.get("trusted_probe", {})
                      if isinstance(proposed, dict) else {})
    if not isinstance(proposed_execution, dict):
        raise ValueError("boundary run_cmd requires an agent-discovered execution entry")
    artifact = proposed_probe.get("artifact")
    scorer_argv = [*argv, "{artifact}"] if isinstance(artifact, dict) else argv
    return {
        "execution": proposed_execution,
        "trusted_probe": {
            "kind": "external_scorer", "argv": scorer_argv, "cwd": command["cwd"],
            "parser": parser, "artifact": artifact,
            **({"protocol_parser": boundary["protocol_parser"]}
               if isinstance(boundary.get("protocol_parser"), dict) else {}),
        },
    }


def _boundary_source_claims(spec: dict) -> list[dict]:
    canonical_spec = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    digest = "sha256:" + hashlib.sha256(canonical_spec).hexdigest()
    return [{
        "path": ".confirmed-boundary", "line_start": 1, "line_end": 1,
        "file_sha256": digest, "excerpt_sha256": digest,
        "command": json.dumps(identity, sort_keys=True, separators=(",", ":")),
    } for identity in (spec["execution"], spec["trusted_probe"])]


def _tracked_regular_path(repo_path: str, relative: str, label: str) -> str:
    repo_real = os.path.realpath(repo_path)
    candidate = os.path.abspath(os.path.join(repo_path, relative))
    full = os.path.realpath(candidate)
    if (os.path.commonpath([repo_real, full]) != repo_real
            or os.path.islink(candidate) or not os.path.isfile(full)):
        raise ValueError(f"{label} must be a regular repository file")
    tracked = subprocess.run(
        ["git", "-C", repo_path, "ls-files", "--error-unmatch", "--", relative],
        capture_output=True, timeout=10)
    if tracked.returncode != 0:
        raise ValueError(f"{label} must be tracked")
    return full


def _resolve_module_path(repo_path: str, cwd: str, module: str) -> str:
    base = module.replace(".", os.sep)
    candidates = [base + ".py", os.path.join(base, "__main__.py"),
                  os.path.join(base, "__init__.py")]
    for item in candidates:
        relative = os.path.normpath(os.path.join(cwd, item))
        try:
            _tracked_regular_path(repo_path, relative, "module entrypoint")
            return relative
        except ValueError:
            continue
    raise ValueError(f"module entrypoint {module!r} does not resolve to a tracked file")


def _tracked_command_path(repo_path: str, cwd: str, path: str, label: str) -> str:
    if os.path.isabs(path):
        raise ValueError(f"{label} must be repository-relative")
    relative = os.path.normpath(os.path.join(cwd, path))
    if relative == ".." or relative.startswith(".." + os.sep):
        raise ValueError(f"{label} escapes repository")
    _tracked_regular_path(repo_path, relative, label)
    return relative


def _uv_run_command(
    repo_path: str, argv: list[str], cwd: str,
) -> tuple[list[str], str, str, list[str]]:
    if len(argv) < 3 or argv[1] != "run":
        raise ValueError("uv measurement must use `uv run`")
    value_options = {
        "--project", "--directory", "--python", "--env-file", "--with",
        "--with-editable", "--with-requirements", "--index", "--default-index",
        "--index-url", "--extra-index-url", "--find-links", "--resolution",
        "--prerelease", "--fork-strategy", "--link-mode", "--compile-bytecode-timeout",
    }
    flag_options = {
        "--isolated", "--locked", "--frozen", "--no-sync", "--no-project",
        "--all-extras", "--no-dev", "--exact", "--inexact", "--offline", "--managed-python",
        "--no-managed-python", "--no-env-file", "--no-editable", "--compile-bytecode",
        "--no-compile-bytecode", "--native-tls", "--no-cache", "--refresh", "--quiet", "-q",
        "--verbose", "-v", "--color", "--no-progress", "--no-config", "--",
    }
    unverifiable_selectors = {
        "--python", "--python-platform", "--managed-python", "--no-managed-python",
        "--no-python-downloads", "--index", "--default-index", "--index-url",
        "--extra-index-url", "--find-links", "--no-index", "--index-strategy",
        "--keyring-provider",
    }
    project = None
    directory = cwd
    config_sources: list[str] = []
    index = 2
    while index < len(argv) and argv[index].startswith("-"):
        option = argv[index]
        if option == "--":
            index += 1
            break
        name = option.split("=", 1)[0]
        if (name in {"--with", "--with-editable"}
                or option == "-w" or option.startswith("-w")):
            raise ValueError("uv dependency override is not host-verifiable")
        if (name in unverifiable_selectors
                or option in {"-p", "-i", "-f"}
                or any(option.startswith(prefix) and len(option) > len(prefix)
                       for prefix in ("-p", "-i", "-f"))):
            raise ValueError(
                "uv environment/dependency selector is not host-verifiable")
        if name in flag_options:
            index += 1
            continue
        if name not in value_options:
            raise ValueError(f"unsupported uv run option: {option}")
        if "=" in option:
            option_value = option.split("=", 1)[1]
            index += 1
        else:
            if index + 1 >= len(argv):
                raise ValueError(f"uv run option requires a value: {option}")
            option_value = argv[index + 1]
            index += 2
        if name == "--project":
            project = option_value
        elif name == "--directory":
            directory = os.path.normpath(os.path.join(cwd, option_value))
            if directory == ".." or directory.startswith(".." + os.sep):
                raise ValueError("uv run directory escapes repository")
        elif name in {"--env-file", "--with-requirements"}:
            config_sources.append(_tracked_command_path(
                repo_path, cwd, option_value, "uv configuration input"))
    if index >= len(argv):
        raise ValueError("uv run is missing a command")
    project_dir = os.path.normpath(os.path.join(cwd, project or "."))
    if project_dir == ".." or project_dir.startswith(".." + os.sep):
        raise ValueError("uv project escapes repository")
    pyproject = _tracked_command_path(
        repo_path, ".", os.path.join(project_dir, "pyproject.toml"), "uv pyproject")
    return argv[index:], directory, pyproject, config_sources


def _resolve_uv_console_script(repo_path: str, cwd: str, pyproject: str,
                               command: str) -> list[str]:
    full = _tracked_regular_path(repo_path, pyproject, "uv pyproject")
    with open(full, "rb") as handle:
        parsed = tomllib.load(handle)
    project = parsed.get("project") if isinstance(parsed, dict) else None
    scripts = project.get("scripts") if isinstance(project, dict) else None
    target = scripts.get(command) if isinstance(scripts, dict) else None
    if not isinstance(target, str) or not target.strip():
        raise ValueError("uv console script is not declared in tracked pyproject.toml")
    module = target.split(":", 1)[0].strip()
    if not module:
        raise ValueError("uv console script target is invalid")
    module_path = _resolve_module_path(repo_path, os.path.dirname(pyproject) or ".", module)
    return [pyproject, module_path]


def _resolve_make_sources(repo_path: str, argv: list[str], cwd: str) -> list[str]:
    makefiles: list[str] = []
    target_cwd = cwd
    targets: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]
        if name == "--eval" or token == "-E" or token.startswith("-E"):
            raise ValueError("make --eval is not a verifiable entrypoint")
        if token in {"-f", "--file", "--makefile", "-C", "--directory"}:
            if index + 1 >= len(argv):
                raise ValueError(f"make option requires a value: {token}")
            value = argv[index + 1]
            if token in {"-C", "--directory"}:
                target_cwd = os.path.normpath(os.path.join(cwd, value))
                if target_cwd == ".." or target_cwd.startswith(".." + os.sep):
                    raise ValueError("make directory escapes repository")
            else:
                makefiles.append(value)
            index += 2
            continue
        if name in {"--file", "--makefile", "--directory"} and "=" in token:
            value = token.split("=", 1)[1]
            if name == "--directory":
                target_cwd = os.path.normpath(os.path.join(cwd, value))
                if target_cwd == ".." or target_cwd.startswith(".." + os.sep):
                    raise ValueError("make directory escapes repository")
            else:
                makefiles.append(value)
            index += 1
            continue
        if token.startswith("-f") and len(token) > 2:
            makefiles.append(token[2:])
            index += 1
            continue
        if token.startswith("-C") and len(token) > 2:
            target_cwd = os.path.normpath(os.path.join(cwd, token[2:]))
            if target_cwd == ".." or target_cwd.startswith(".." + os.sep):
                raise ValueError("make directory escapes repository")
            index += 1
            continue
        if token.startswith("-"):
            raise ValueError(f"unsupported make option: {token}")
        if "=" not in token:
            targets.append(token)
        index += 1
    if len(targets) != 1:
        raise ValueError("make measurement requires one explicit target")
    target = targets[0]
    if not re.fullmatch(r"[A-Za-z0-9_./+%-]+", target) or ".." in target.split("/"):
        raise ValueError("make target is unsafe")
    if not makefiles:
        for name in ("GNUmakefile", "makefile", "Makefile"):
            try:
                makefiles.append(_tracked_command_path(
                    repo_path, target_cwd, name, "makefile entrypoint"))
                break
            except ValueError:
                continue
        if not makefiles:
            raise ValueError("make measurement has no tracked Makefile")
    else:
        makefiles = [
            _tracked_command_path(repo_path, target_cwd, item, "makefile entrypoint")
            for item in makefiles
        ]
    makefiles = list(dict.fromkeys(makefiles))
    sources = []
    for makefile in makefiles:
        with open(os.path.join(repo_path, makefile), encoding="utf-8",
                  errors="replace") as handle:
            sources.append(handle.read())
    if not any(re.search(rf"(?m)^{re.escape(target)}(?:\s+[^:]*)?:", source)
               for source in sources):
        raise ValueError("make target is not declared in tracked Makefile")
    return makefiles


def _resolve_pytest_sources(repo_path: str, argv: list[str], cwd: str) -> list[str]:
    sources: list[str] = []
    entries: list[str] = []
    index = 1
    value_options = {"-c", "--config-file", "--rootdir", "-k", "-m", "--tb", "--maxfail"}
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]
        if name in value_options:
            if "=" in token:
                value = token.split("=", 1)[1]
                index += 1
            else:
                if index + 1 >= len(argv):
                    raise ValueError(f"pytest option requires a value: {token}")
                value = argv[index + 1]
                index += 2
            if name in {"-c", "--config-file"}:
                sources.append(_tracked_command_path(
                    repo_path, cwd, value, "pytest config"))
            continue
        if token.startswith("-"):
            index += 1
            continue
        entry = token.split("::", 1)[0]
        if entry.endswith(".py"):
            entries.append(_tracked_command_path(
                repo_path, cwd, entry, "pytest test entrypoint"))
        index += 1
    if not entries:
        raise ValueError("pytest measurement requires an explicit tracked test entry")
    return [*sources, entries[0]]


def _resolve_cargo_sources(repo_path: str, argv: list[str], cwd: str) -> list[str]:
    if len(argv) < 2 or argv[1] != "bench":
        raise ValueError("cargo measurement must use `cargo bench`")
    manifest = "Cargo.toml"
    for index, token in enumerate(argv[2:], start=2):
        if token == "--manifest-path":
            if index + 1 >= len(argv):
                raise ValueError("cargo --manifest-path requires a value")
            manifest = argv[index + 1]
        elif token.startswith("--manifest-path="):
            manifest = token.split("=", 1)[1]
    normalized = os.path.normpath(os.path.join(cwd, manifest))
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("cargo manifest escapes repository")
    return [_tracked_command_path(repo_path, ".", normalized, "cargo manifest")]


def _execution_source_paths(repo_path: str, command: dict) -> list[str]:
    argv = list(command["argv"])
    cwd = command["cwd"]
    uv_pyproject = None
    uv_config_sources: list[str] = []
    if os.path.basename(argv[0]) == "uv":
        argv, cwd, uv_pyproject, uv_config_sources = _uv_run_command(
            repo_path, argv, cwd)
    executable = os.path.basename(argv[0])
    if executable in {"bash", "sh", "dash", "zsh"}:
        if any(token in {"-c", "--command"} for token in argv[1:]):
            raise ValueError("shell -c is not a safe measurement entrypoint")
        scripts = [token for token in argv[1:] if not token.startswith("-")]
        if not scripts:
            raise ValueError("shell measurement requires a tracked script entrypoint")
        return [_tracked_command_path(repo_path, cwd, scripts[0], "shell entrypoint")]
    if executable in {"make", "gmake"}:
        return _resolve_make_sources(repo_path, argv, cwd)
    if executable in {"pytest", "py.test"}:
        return _resolve_pytest_sources(repo_path, argv, cwd)
    if executable == "cargo":
        return _resolve_cargo_sources(repo_path, argv, cwd)
    if executable.startswith("python"):
        if "-c" in argv[1:]:
            raise ValueError("python -c is not an executable measurement entrypoint")
        if "-m" in argv[1:]:
            index = argv.index("-m", 1)
            if index + 1 >= len(argv):
                raise ValueError("python -m requires a module")
            if argv[index + 1] == "pytest":
                return _resolve_pytest_sources(
                    repo_path, ["pytest", *argv[index + 2:]], cwd)
            return [_resolve_module_path(repo_path, cwd, argv[index + 1])]
        candidates = [token for token in argv[1:] if token.endswith(".py")]
    elif executable == "torchrun":
        if "-m" in argv[1:]:
            index = argv.index("-m", 1)
            if index + 1 >= len(argv):
                raise ValueError("torchrun -m requires a module")
            return [_resolve_module_path(repo_path, cwd, argv[index + 1])]
        candidates = [token for token in argv[1:] if token.endswith(".py")]
    else:
        candidates = [token for token in argv if token.endswith(".py")]
    if not candidates:
        if uv_pyproject is not None:
            return [*uv_config_sources, *_resolve_uv_console_script(
                repo_path, cwd, uv_pyproject, argv[0])]
        if "/" in argv[0] or argv[0].startswith("."):
            return [_tracked_command_path(
                repo_path, cwd, argv[0], "execution entrypoint")]
        raise ValueError("execution command has no tracked script or module entrypoint")
    return [*uv_config_sources, _tracked_command_path(
        repo_path, cwd, candidates[0], "execution entrypoint")]


def _execution_immutable_paths(repo_path: str, command: dict) -> list[str]:
    argv = list(command["argv"])
    if os.path.basename(argv[0]) != "uv":
        return []
    _command, _cwd, pyproject, config_sources = _uv_run_command(
        repo_path, argv, command["cwd"])
    return [pyproject, *config_sources]


def _execution_entry(repo_path: str, command: dict) -> str:
    return _execution_source_paths(repo_path, command)[-1]


def _control_binding_specs(repo_path: str, contract: dict) -> list[dict]:
    raw = contract.get("controls", {}).get("control_bindings", [])
    if not isinstance(raw, list):
        raise ValueError("contract control_bindings must be an array")
    allowed_axes = contract.get("allowed_treatment_axes", [])
    treatment = contract.get("controls", {}).get("treatment", {})
    treatment_paths = treatment.get("allowed_paths", []) if isinstance(treatment, dict) else []
    seen: set[str] = set()
    normalized: list[dict] = []
    selectors = {
        "json": "json_pointer", "toml": "toml_key",
        "python_literal": "python_literal_name",
    }
    for binding in raw:
        if not isinstance(binding, dict):
            raise ValueError("control binding must be an object")
        name = binding.get("name")
        path = binding.get("path")
        fmt = binding.get("format")
        selector = selectors.get(fmt)
        expected = {"name", "path", "format", selector} if selector else set()
        treatment_axis = binding.get("treatment_axis")
        if treatment_axis is not None:
            expected.add("treatment_axis")
        if (not expected or set(binding) != expected
                or not isinstance(name, str) or not name.strip() or name in seen
                or not isinstance(path, str) or not path
                or not isinstance(binding.get(selector), str) or not binding[selector]):
            raise ValueError("control binding has an invalid shape")
        relative = _tracked_command_path(repo_path, ".", path, "control binding path")
        if treatment_axis is not None:
            if (not isinstance(treatment_axis, str) or treatment_axis not in allowed_axes
                    or relative not in [os.path.normpath(item) for item in treatment_paths
                                        if isinstance(item, str)]):
                raise ValueError("control binding treatment axis is not explicitly allowed")
        seen.add(name)
        normalized.append({**binding, "name": name.strip(), "path": relative})
    return sorted(normalized, key=lambda item: item["name"])


def _json_pointer_value(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must start with /")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise KeyError(token)
    return current


def _python_literal_value(source: str, name: str) -> Any:
    matches = []
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            matches.append(node.value)
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
              and node.target.id == name and node.value is not None):
            matches.append(node.value)
    if len(matches) != 1:
        raise ValueError("python literal assignment must exist exactly once")
    return ast.literal_eval(matches[0])


def _extract_control_bindings(repo_path: str, contract: dict) -> list[dict]:
    extracted = []
    for binding in _control_binding_specs(repo_path, contract):
        full = _tracked_regular_path(repo_path, binding["path"], "control binding path")
        try:
            if binding["format"] == "json":
                with open(full, encoding="utf-8") as handle:
                    value = _json_pointer_value(json.load(handle), binding["json_pointer"])
                selector = {"json_pointer": binding["json_pointer"]}
            elif binding["format"] == "toml":
                with open(full, "rb") as handle:
                    value = tomllib.load(handle)
                for token in binding["toml_key"].split("."):
                    if not token or not isinstance(value, dict) or token not in value:
                        raise KeyError(token)
                    value = value[token]
                selector = {"toml_key": binding["toml_key"]}
            else:
                with open(full, encoding="utf-8") as handle:
                    value = _python_literal_value(
                        handle.read(), binding["python_literal_name"])
                selector = {"python_literal_name": binding["python_literal_name"]}
            value = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
            digest, _stat_result = _regular_file_evidence(full)
        except (OSError, SyntaxError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"control binding {binding['name']} could not be extracted: {exc}") from exc
        extracted.append({
            "name": binding["name"], "value": value,
            "treatment_axis": binding.get("treatment_axis"),
            "source": {
                "path": binding["path"], "format": binding["format"],
                **selector, "file_sha256": digest,
            },
        })
    return extracted


_ACTUAL_CONTROLS_CAVEAT = (
    "Host observes canonical execution argv/cwd, resolved tracked sources, immutable file hashes, "
    "and declared structured bindings; it does not observe arbitrary internal runtime state or "
    "prove mathematical equivalence of undeclared batch/model/seed/precision controls."
)


def _actual_controls(repo_path: str, contract: dict, spec: dict,
                     immutable_hashes: dict[str, str]) -> dict:
    return {
        "version": 1,
        "execution": {
            **spec["execution"],
            "source_paths": _execution_source_paths(repo_path, spec["execution"]),
        },
        "immutable_file_hashes": dict(sorted(immutable_hashes.items())),
        "bindings": _extract_control_bindings(repo_path, contract),
        "caveat": _ACTUAL_CONTROLS_CAVEAT,
    }


def _actual_controls_pair_matches(current: Any, frozen: Any) -> bool:
    if (not isinstance(current, dict) or not isinstance(frozen, dict)
            or set(current) != {"version", "execution", "immutable_file_hashes", "bindings", "caveat"}
            or set(frozen) != set(current)
            or any(current[field] != frozen[field] for field in (
                "version", "execution", "immutable_file_hashes", "caveat"))
            or not isinstance(current["bindings"], list)
            or not isinstance(frozen["bindings"], list)):
        return False
    current_bindings = {item.get("name"): item for item in current["bindings"]
                        if isinstance(item, dict)}
    frozen_bindings = {item.get("name"): item for item in frozen["bindings"]
                       if isinstance(item, dict)}
    if len(current_bindings) != len(current["bindings"]) or set(current_bindings) != set(
            frozen_bindings):
        return False
    for name, current_binding in current_bindings.items():
        frozen_binding = frozen_bindings[name]
        axis = current_binding.get("treatment_axis")
        if axis != frozen_binding.get("treatment_axis"):
            return False
        if axis is None:
            if current_binding != frozen_binding:
                return False
            continue
        current_source = dict(current_binding.get("source") or {})
        frozen_source = dict(frozen_binding.get("source") or {})
        current_source.pop("file_sha256", None)
        frozen_source.pop("file_sha256", None)
        if current_source != frozen_source:
            return False
    return True


def _immutable_file_hashes(repo_path: str, contract: dict, spec: dict) -> dict[str, str]:
    controls = contract.get("controls", {})
    evaluator = controls.get("evaluator", {})
    evaluator_paths = evaluator.get("immutable_paths") if isinstance(evaluator, dict) else None
    data = controls.get("data", {})
    data_paths = data.get("immutable_paths") if isinstance(data, dict) else None
    workload = controls.get("workload", {})
    workload_paths = workload.get("immutable_paths", []) if isinstance(workload, dict) else []
    if not isinstance(evaluator_paths, list) or not evaluator_paths:
        raise ValueError("contract evaluator requires non-empty immutable_paths")
    if not isinstance(data_paths, list) or not data_paths:
        raise ValueError("contract data requires non-empty immutable_paths")
    if not isinstance(workload_paths, list):
        raise ValueError("contract workload immutable_paths must be an array")
    def normalize_paths(values: list, label: str) -> list[str]:
        normalized_values: list[str] = []
        for item in values:
            if not isinstance(item, str) or not item or os.path.isabs(item):
                raise ValueError(f"immutable {label} path must be repository-relative")
            relative = os.path.normpath(item)
            if relative == ".." or relative.startswith(".." + os.sep):
                raise ValueError(f"immutable {label} path escapes repository")
            normalized_values.append(relative)
        return normalized_values

    normalized_evaluator_paths = normalize_paths(evaluator_paths, "evaluator")
    paths = list(dict.fromkeys([
        *normalize_paths(data_paths, "data"),
        *normalized_evaluator_paths,
        *normalize_paths(workload_paths, "workload"),
        *_execution_immutable_paths(repo_path, spec["execution"]),
        *[binding["path"] for binding in _control_binding_specs(repo_path, contract)
          if binding.get("treatment_axis") is None],
    ]))
    normalized: list[str] = []
    hashes: dict[str, str] = {}
    for item in paths:
        relative = os.path.normpath(item)
        full = _tracked_regular_path(repo_path, relative, "immutable path")
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = "sha256:" + digest.hexdigest()
        normalized.append(relative)
    _execution_source_paths(repo_path, spec["execution"])
    probe = spec["trusted_probe"]
    if probe["kind"] == "external_scorer":
        scorer_entries = _execution_source_paths(repo_path, probe)
        if any(entry not in normalized_evaluator_paths for entry in scorer_entries):
            raise ValueError("external scorer entrypoint must be immutable")
    else:
        source_path = probe["source_path"]
        if source_path not in normalized_evaluator_paths:
            raise ValueError("python callable source_path must be immutable")
        resolved = _resolve_module_path(repo_path, spec["execution"]["cwd"], probe["module"])
        if resolved != source_path:
            raise ValueError("python callable module does not resolve to source_path")
    return hashes


def _verify_official_source_evidence(repo_path: str, contract: dict, spec: dict,
                                     confirmed_boundary: dict | None) -> list[dict]:
    if isinstance(confirmed_boundary, dict) and isinstance(
            confirmed_boundary.get("measurement_spec"), dict):
        boundary_sha = "sha256:" + hashlib.sha256(json.dumps(
            confirmed_boundary["measurement_spec"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        return [{
            "role": role, "authority": "user", "boundary_sha256": boundary_sha,
            "identity_sha256": "sha256:" + hashlib.sha256(json.dumps(
                identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        } for role, identity in (
            ("execution", spec["execution"]),
            ("trusted_probe", spec["trusted_probe"]),
        )]
    evaluator = contract.get("controls", {}).get("evaluator", {})
    evidence = evaluator.get("official_source_evidence") if isinstance(evaluator, dict) else None
    required = {"path", "line_start", "line_end", "file_sha256",
                "excerpt_sha256", "command"}
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise ValueError("evaluator requires execution and trusted_probe source evidence")
    identities = [("execution", spec["execution"]),
                  ("trusted_probe", spec["trusted_probe"])]
    verified = []
    for item, (role, identity) in zip(evidence, identities):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("official source evidence item is invalid")
        path = item["path"]
        if not isinstance(path, str) or not path or os.path.isabs(path):
            raise ValueError("official source path must be repository-relative")
        relative = os.path.normpath(path)
        evaluator_paths = evaluator.get("immutable_paths", [])
        if relative not in [os.path.normpath(value) for value in evaluator_paths
                            if isinstance(value, str)]:
            raise ValueError("official source must be in evaluator immutable_paths")
        current = _tracked_regular_path(repo_path, relative, "official source")
        shown = subprocess.run(
            ["git", "-C", repo_path, "show", f"HEAD:{relative}"],
            capture_output=True, timeout=10)
        if shown.returncode != 0:
            raise ValueError("official source is absent from frozen HEAD")
        with open(current, "rb") as handle:
            current_bytes = handle.read()
        file_digest = "sha256:" + hashlib.sha256(shown.stdout).hexdigest()
        if current_bytes != shown.stdout or item["file_sha256"] != file_digest:
            raise ValueError("official source differs from frozen HEAD")
        start, end = item["line_start"], item["line_end"]
        lines = shown.stdout.splitlines(keepends=True)
        if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int)
                or isinstance(end, bool) or start < 1 or end < start or end > len(lines)):
            raise ValueError("official source line range is invalid")
        excerpt = b"".join(lines[start - 1:end])
        if item["excerpt_sha256"] != "sha256:" + hashlib.sha256(excerpt).hexdigest():
            raise ValueError("official source excerpt hash mismatch")
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        if item["command"] != canonical:
            raise ValueError("official source identity hash mismatch")
        excerpt_text = " ".join(excerpt.decode(errors="replace").split())
        command_text = " ".join(spec["execution"]["argv"])
        probe = spec["trusted_probe"]
        probe_text = (f"{probe['module']}.{probe['callable']} {probe['source_path']}"
                      if probe["kind"] == "python_callable" else
                      " ".join(probe["argv"]))
        expected_text = command_text if role == "execution" else probe_text
        if expected_text not in excerpt_text:
            raise ValueError(f"official source excerpt does not bind {role}")
        verified.append({
            "authority": "repository", "role": role,
            "identity_sha256": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
            **item,
        })
    return verified


def _parse_protocol_evidence(contract: dict, spec: dict, stdout: str, stderr: str,
                             elapsed: float) -> dict:
    protocol = contract.get("protocol", {})
    kind = protocol.get("kind")
    if kind == "time_to_target":
        raise ValueError("time_to_target protocol is not yet host-verifiable")
    if kind == "quality_at_fixed_time":
        budget = float(protocol.get("budget", {}).get("wall_clock_s"))
        scope = protocol.get("timing_scope")
        authority = protocol.get("timer_authority")
        if scope == "process" and authority == "host":
            source, value = "host_elapsed", elapsed
        elif authority == "official_harness" and scope in {"process", "training"}:
            parser = spec["trusted_probe"].get("protocol_parser")
            if not isinstance(parser, dict) or parser.get("unit") != "seconds":
                raise ValueError("official timing requires a seconds protocol_parser")
            source, value = "host_parser", _parse_protocol_value(parser, stdout, stderr)
        else:
            raise ValueError("fixed-time timing scope/authority is not verifiable")
        grace = max(5.0, budget * 0.05) + 2.0
        if not (budget * 0.95 <= value <= budget + grace):
            raise ValueError("actual fixed-time execution does not match budget")
        return {"source": source, "unit": "seconds", "value": value}
    if kind == "quality_at_fixed_work":
        budget_obj = protocol.get("budget", {})
        if "steps" in budget_obj and "tokens" in budget_obj:
            raise ValueError("fixed-work budget cannot contain both steps and tokens")
        unit = "steps" if "steps" in budget_obj else "tokens" if "tokens" in budget_obj else None
        parser = spec["trusted_probe"].get("protocol_parser")
        if unit is None or not isinstance(parser, dict) or parser.get("unit") != unit:
            raise ValueError("fixed-work protocol requires a matching protocol_parser")
        value = _parse_protocol_value(parser, stdout, stderr)
        if value != float(budget_obj[unit]):
            raise ValueError("actual fixed work does not match budget")
        return {"source": "host_parser", "unit": unit, "value": value}
    raise ValueError("unsupported protocol for host evidence")


def _parse_protocol_value(parser: dict, stdout: str, stderr: str) -> float:
    stream = (stdout if parser["stream"] == "stdout" else stderr
              if parser["stream"] == "stderr" else stdout + "\n" + stderr)
    matches = list(re.finditer(parser["regex"], stream[-2 * 1024 * 1024:]))
    if not matches:
        raise ValueError("protocol parser found no evidence")
    value = float(matches[-1].group(parser["group"])) * float(parser.get("multiplier", 1))
    if not math.isfinite(value):
        raise ValueError("protocol evidence must be finite")
    return value


def _measurement_timeout(result: dict, frozen_baseline: dict | None) -> int:
    contract = ((frozen_baseline or {}).get("benchmark_contract")
                or result.get("benchmark_contract") or {})
    try:
        protocol = contract.get("protocol", {})
        budget = protocol.get("budget", {})
        wall_clock = float(budget.get("wall_clock_s"))
        if math.isfinite(wall_clock) and wall_clock > 0:
            budget_timeout = int(math.ceil(wall_clock + max(5.0, wall_clock * 0.05)))
            if (protocol.get("timing_scope") == "training"
                    and protocol.get("timer_authority") == "official_harness"):
                return max(_per_run_cap(), budget_timeout)
            return budget_timeout
    except (AttributeError, TypeError, ValueError):
        pass
    return _per_run_cap()


_CALLABLE_BOOTSTRAP = r'''import importlib.abc
import importlib.util
import json
import math
import os
import sys

TARGET = os.environ["AUTORESEARCH_PROBE_MODULE"]
CALLABLE = os.environ["AUTORESEARCH_PROBE_CALLABLE"]
SOURCE = os.path.realpath(os.environ["AUTORESEARCH_PROBE_SOURCE"])
NONCE = os.environ["AUTORESEARCH_PROBE_NONCE"]
for _key in ("AUTORESEARCH_PROBE_MODULE", "AUTORESEARCH_PROBE_CALLABLE",
             "AUTORESEARCH_PROBE_SOURCE", "AUTORESEARCH_PROBE_NONCE"):
    os.environ.pop(_key, None)
COUNT = 0

def wrap(module):
    global COUNT
    if os.path.realpath(getattr(module, "__file__", "")) != SOURCE:
        return
    original = getattr(module, CALLABLE, None)
    if not callable(original) or getattr(original, "__autoresearch_host_probe__", False):
        return
    def measured(*args, **kwargs):
        global COUNT
        value = original(*args, **kwargs)
        COUNT += 1
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            event = {"nonce": NONCE, "value": float(value), "call_count": COUNT}
            print("AUTORESEARCH_HOST_PROBE:" + json.dumps(event, sort_keys=True,
                  separators=(",", ":")), file=sys.stderr, flush=True)
        return value
    measured.__autoresearch_host_probe__ = True
    setattr(module, CALLABLE, measured)

class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original
    def create_module(self, spec):
        creator = getattr(self.original, "create_module", None)
        return creator(spec) if creator else None
    def exec_module(self, module):
        self.original.exec_module(module)
        wrap(module)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = Loader(spec.loader)
        return spec

if TARGET in sys.modules:
    wrap(sys.modules[TARGET])
else:
    sys.meta_path.insert(0, Finder())
'''


def _create_callable_bootstrap(repo_path: str, probe: dict) -> tuple[str, str, str]:
    """Create a post-agent probe shim with best-effort same-UID spoof resistance.

    This is a fairness/accident boundary, not an adversarial OS security boundary: code running as
    the same sandbox user can inspect its own process.  The nonce is removed from the child
    environment immediately, exactly one event is accepted, and treatment diffs containing the
    reserved probe markers are rejected.
    """
    control = os.path.join(os.path.dirname(repo_path), ".sandbox-control", "host-probes")
    os.makedirs(control, exist_ok=True)
    directory = tempfile.mkdtemp(prefix="probe-", dir=control)
    os.chmod(directory, 0o755)
    path = os.path.join(directory, "sitecustomize.py")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        os.write(fd, _CALLABLE_BOOTSTRAP.encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = "sha256:" + hashlib.sha256(_CALLABLE_BOOTSTRAP.encode()).hexdigest()
    nonce = uuid.uuid4().hex
    return directory, digest, nonce


def _parse_callable_probe(stderr: str, nonce: str) -> tuple[float | None, int]:
    prefix = "AUTORESEARCH_HOST_PROBE:"
    accepted: list[tuple[float, int]] = []
    for line in stderr[-2 * 1024 * 1024:].splitlines():
        if not line.startswith(prefix):
            continue
        try:
            event = json.loads(line[len(prefix):])
            value = float(event.get("value"))
            count = int(event.get("call_count"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if event.get("nonce") == nonce and math.isfinite(value) and count == 1:
            accepted.append((value, count))
    return accepted[0] if len(accepted) == 1 else (None, 0)


def _stream_for_parser(parser: dict, stdout: str, stderr: str) -> str:
    return (stdout if parser["stream"] == "stdout" else stderr
            if parser["stream"] == "stderr" else stdout + "\n" + stderr)


def _parse_metric(parser: dict, stdout: str, stderr: str) -> float | None:
    matches = list(re.finditer(parser["regex"], _stream_for_parser(
        parser, stdout, stderr)[-2 * 1024 * 1024:]))
    if not matches:
        return None
    try:
        value = float(matches[-1].group(parser["group"]))
    except (IndexError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _artifact_target(repo_path: str, contract: dict, probe: dict) -> tuple[str, str]:
    artifact = probe["artifact"]
    relative = artifact["path"]
    outputs = contract.get("controls", {}).get("evaluator", {}).get("output_paths", [])
    if relative not in [os.path.normpath(item) for item in outputs if isinstance(item, str)]:
        raise ValueError("artifact path must be a contract-declared evaluator output_path")
    tracked = subprocess.run(
        ["git", "-C", repo_path, "ls-files", "--error-unmatch", "--", relative],
        capture_output=True, timeout=10)
    if tracked.returncode == 0:
        raise ValueError("artifact path must be untracked")
    full = os.path.abspath(os.path.join(repo_path, relative))
    if os.path.commonpath([os.path.realpath(repo_path), os.path.realpath(
            os.path.dirname(full))]) != os.path.realpath(repo_path):
        raise ValueError("artifact path escapes repository")
    current = os.path.realpath(repo_path)
    for part in relative.split(os.sep)[:-1]:
        current = os.path.join(current, part)
        if os.path.islink(current):
            raise ValueError("artifact parent symlink is unsafe")
    return relative, full


def _regular_file_evidence(full: str) -> tuple[str, os.stat_result]:
    if os.path.islink(full):
        raise ValueError("artifact symlink is unsafe")
    if not os.path.isfile(full):
        raise ValueError("artifact must be a regular file")
    stat_result = os.stat(full, follow_symlinks=False)
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(full, flags)
    with os.fdopen(fd, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest(), stat_result


def _quarantine_prior_artifact(repo_path: str, relative: str, full: str,
                               attempt_id: str, attempt: dict) -> dict:
    """Move a declared prior output into host control before verification.

    Moving preserves the agent's evidence without letting the host execution reuse it.  A file
    older than the cited attempt is still preserved, but is identified as stale so the caller can
    reject the result instead of accidentally attributing it to that attempt.
    """
    if not os.path.lexists(full):
        return {"state": "absent", "path": relative}
    digest, stat_result = _regular_file_evidence(full)
    started = float(attempt.get("started_at"))
    ended = float(attempt.get("ended_at"))
    source = ("cited_agent_attempt"
              if stat_result.st_mtime + 1e-6 >= started
              and stat_result.st_mtime <= ended + 1.0
              else "stale_pre_attempt")
    control = os.path.join(
        os.path.dirname(repo_path), ".sandbox-control", "host-artifacts")
    os.makedirs(control, exist_ok=True)
    directory = tempfile.mkdtemp(prefix="artifact-", dir=control)
    destination = os.path.join(directory, os.path.basename(relative))
    os.replace(full, destination)
    moved_digest, moved_stat = _regular_file_evidence(destination)
    if moved_digest != digest or moved_stat.st_size != stat_result.st_size:
        raise ValueError("quarantined artifact evidence changed while moving")
    return {
        "state": "quarantined", "path": relative,
        "quarantine_path": destination, "sha256": digest,
        "size": stat_result.st_size, "mtime": stat_result.st_mtime,
        "source": source, "attempt_id": attempt_id,
    }


def _attest_new_artifact(relative: str, full: str, started: float) -> dict:
    try:
        digest, stat_result = _regular_file_evidence(full)
    except ValueError as exc:
        raise ValueError("execution did not create a regular artifact") from exc
    if stat_result.st_mtime + 1e-6 < started:
        raise ValueError("execution artifact predates measurement")
    return {
        "path": relative, "sha256": digest,
        "mtime": stat_result.st_mtime, "size": stat_result.st_size,
        "required_new": True,
    }


def _host_remeasure(backend: Any, handle: Any, repo_path: str, result: dict,
                    execution_phase: str, frozen_baseline: dict | None,
                    confirmed_boundary: dict | None = None,
                    attempt_log: str | None = None,
                    resource_evidence: dict | None = None,
                    untracked_before: dict[str, dict] | None = None) -> None:
    """Execute and parse the accepted metric on the host side of the agent boundary."""
    if execution_phase == "method" and _tracked_repo_state(repo_path).get(
            "tracked_diff_present") is not True:
        result["_host_measurement_error"] = "method worktree contains no tracked treatment diff"
        return
    records = _attempt_records(attempt_log or "")
    proposed = result.get("measurement")
    if execution_phase == "method":
        proposed = ((frozen_baseline or {}).get("host_attestation") or {}).get(
            "measurement_spec")
    elif not proposed:
        proposed_contract = result.get("benchmark_contract")
        if isinstance(proposed_contract, dict):
            proposed_controls = proposed_contract.get("controls")
            proposed_evaluator = (proposed_controls.get("evaluator")
                                  if isinstance(proposed_controls, dict) else None)
            proposed = (proposed_evaluator.get("measurement_spec")
                        if isinstance(proposed_evaluator, dict) else None)
    try:
        if execution_phase == "baseline":
            proposed = _boundary_measurement(confirmed_boundary, proposed)
        spec = _normalize_measurement_spec(proposed)
    except (TypeError, ValueError, re.error) as exc:
        result["_host_measurement_error"] = f"invalid measurement specification: {exc}"
        return
    _bind_matching_host_attempt(
        result, records, repo_path, execution_phase, spec, frozen_baseline)
    agent_attempt_id = str(result.get("_metric_attempt_id") or "")
    agent_attempt = records.get(agent_attempt_id)
    manifest = result.get("run_manifest")
    if (not agent_attempt_id or not isinstance(agent_attempt, dict)
            or agent_attempt.get("launched") is not True
            or agent_attempt.get("finished") is not True
            or agent_attempt.get("rc") != 0
            or (execution_phase == "method" and _variant_cap()
                and len(records) > _variant_cap())
            or (execution_phase == "method" and (
                not isinstance(manifest, dict)
                or manifest.get("attempt_id") != agent_attempt_id))):
        result["_host_measurement_error"] = (
            "agent attempt evidence invalid: result must cite one successful ar-attempt record")
        return
    try:
        normalized_attempt_argv = _normalized_attempt_argv(agent_attempt.get("argv"))
    except ValueError as exc:
        result["_host_measurement_error"] = f"agent attempt evidence invalid: {exc}"
        return
    if normalized_attempt_argv != spec["execution"]["argv"]:
        result["_host_measurement_error"] = (
            "agent attempt evidence invalid: cited successful attempt does not invoke "
            "frozen measurement execution")
        return
    expected_attempt_cwd = os.path.realpath(os.path.join(
        repo_path, spec["execution"]["cwd"]))
    if agent_attempt.get("cwd") != expected_attempt_cwd:
        result["_host_measurement_error"] = (
            "agent attempt evidence invalid: cited successful attempt cwd differs from "
            "frozen measurement execution")
        return
    contract = ((frozen_baseline or {}).get("benchmark_contract")
                if execution_phase == "method" else result.get("benchmark_contract"))
    if not isinstance(contract, dict) or int(contract.get("version", 0)) < 2:
        result["_host_measurement_error"] = (
            "invalid measurement specification: host measurement requires benchmark contract v2")
        return
    if execution_phase == "baseline" and isinstance(confirmed_boundary, dict):
        try:
            contract = json.loads(json.dumps(contract, allow_nan=False))
            if isinstance(confirmed_boundary.get("measurement_spec"), dict):
                evaluator = contract.setdefault("controls", {}).setdefault("evaluator", {})
                evaluator["measurement_spec"] = spec
                evaluator["official_source_evidence"] = _boundary_source_claims(spec)
            if confirmed_boundary.get("protocol_kind") == "fixed_wall_clock":
                contract.get("protocol", {}).pop("budget", None)
                contract.get("controls", {}).get("workload", {}).pop("steps", None)
            contract = merge_benchmark_contracts(
                contract,
                _confirmed_boundary_fragment(
                    confirmed_boundary,
                    str(confirmed_boundary.get("metric") or result.get("metric") or ""),
                    (confirmed_boundary.get("direction") == "minimize"
                     if confirmed_boundary.get("direction") in {"minimize", "maximize"}
                     else bool(result.get("lower_is_better")))),
            )
        except (BenchmarkContractError, TypeError, ValueError) as exc:
            result["_host_measurement_error"] = (
                f"invalid measurement specification: confirmed benchmark contract: {exc}")
            return
    boundary_owns_measurement = isinstance(confirmed_boundary, dict) and (
        isinstance(confirmed_boundary.get("measurement_spec"), dict)
        or confirmed_boundary.get("run_cmd") is not None)
    evaluator_spec = (contract.get("controls", {}).get("evaluator", {}).get(
        "measurement_spec"))
    if execution_phase == "baseline" and not boundary_owns_measurement:
        try:
            if _normalize_measurement_spec(evaluator_spec) != spec:
                raise ValueError("agent measurement differs from contract evaluator")
        except (TypeError, ValueError, re.error) as exc:
            result["_host_measurement_error"] = f"invalid measurement specification: {exc}"
            return
    try:
        untracked_agent = _untracked_file_inventory(repo_path)
        untracked_evidence = _validate_untracked_files(
            untracked_before or {}, untracked_agent, contract, execution_phase)
    except (OSError, TypeError, ValueError) as exc:
        result["_host_measurement_error"] = f"candidate contract violation: {exc}"
        return
    try:
        immutable_start = _immutable_file_hashes(repo_path, contract, spec)
        frozen_hashes = (((frozen_baseline or {}).get("host_attestation") or {}).get(
            "immutable_file_hashes") if execution_phase == "method" else None)
        if execution_phase == "method" and immutable_start != frozen_hashes:
            raise ValueError("immutable evaluator/data files differ from frozen baseline")
        changed_paths = (_validate_treatment_paths(repo_path, contract, immutable_start)
                         if execution_phase == "method" else [])
    except (TypeError, ValueError, re.error) as exc:
        result["_host_measurement_error"] = f"immutable evaluator violation: {exc}"
        return
    try:
        official_source = _verify_official_source_evidence(
            repo_path, contract, spec, confirmed_boundary)
        frozen_source = (((frozen_baseline or {}).get("host_attestation") or {}).get(
            "official_source_evidence") if execution_phase == "method" else None)
        if execution_phase == "method" and official_source != frozen_source:
            raise ValueError("official source evidence differs from frozen baseline")
    except (TypeError, ValueError, re.error) as exc:
        result["_host_measurement_error"] = f"invalid measurement specification: {exc}"
        return
    resource_evidence = dict(resource_evidence or {})
    frozen_resources = (((frozen_baseline or {}).get("host_attestation") or {}).get(
        "resource_evidence") if execution_phase == "method" else None)
    try:
        _validate_resource_contract(contract, resource_evidence)
        if execution_phase == "method" and not _resource_pair_matches(
                resource_evidence, frozen_resources):
            raise ValueError("resource identity differs from frozen baseline")
    except (TypeError, ValueError) as exc:
        result["_host_measurement_error"] = f"resource evidence invalid: {exc}"
        return
    try:
        actual_controls_start = _actual_controls(
            repo_path, contract, spec, immutable_start)
        frozen_controls = (((frozen_baseline or {}).get("host_attestation") or {}).get(
            "actual_controls") if execution_phase == "method" else None)
        if execution_phase == "method" and not _actual_controls_pair_matches(
                actual_controls_start, frozen_controls):
            raise ValueError("actual controls differ from frozen baseline")
    except (OSError, SyntaxError, TypeError, ValueError) as exc:
        result["_host_measurement_error"] = f"actual controls invalid: {exc}"
        return

    start_commit = _repo_commit(repo_path)
    start_state = _tracked_repo_state(repo_path)
    execution = spec["execution"]
    execution_cwd = os.path.realpath(os.path.join(repo_path, execution["cwd"]))
    if os.path.commonpath([os.path.realpath(repo_path), execution_cwd]) != os.path.realpath(
            repo_path):
        result["_host_measurement_error"] = "measurement cwd escapes repository"
        return
    probe = spec["trusted_probe"]
    artifact_relative = artifact_path = None
    prior_artifact_evidence = None
    if probe["kind"] == "external_scorer":
        try:
            artifact_relative, artifact_path = _artifact_target(repo_path, contract, probe)
            prior_artifact_evidence = _quarantine_prior_artifact(
                repo_path, artifact_relative, artifact_path,
                agent_attempt_id, agent_attempt)
        except ValueError as exc:
            result["_host_measurement_error"] = f"invalid measurement specification: {exc}"
            return
        if prior_artifact_evidence.get("source") == "stale_pre_attempt":
            result["artifact_quarantine_evidence"] = prior_artifact_evidence
            result["_host_measurement_error"] = (
                "artifact lifecycle invalid: stale pre-attempt artifact was quarantined")
            return
    bootstrap_path = None
    bootstrap_hash = None
    nonce = None
    command_argv = list(execution["argv"])
    if probe["kind"] == "python_callable":
        try:
            bootstrap_dir, bootstrap_hash, nonce = _create_callable_bootstrap(repo_path, probe)
        except OSError as exc:
            result["_host_measurement_error"] = f"host probe bootstrap failed: {exc}"
            return
        bootstrap_path = os.path.join(bootstrap_dir, "sitecustomize.py")
        command_argv = [
            "env", "PYTHONDONTWRITEBYTECODE=1", f"PYTHONPATH={bootstrap_dir}",
            f"AUTORESEARCH_PROBE_MODULE={probe['module']}",
            f"AUTORESEARCH_PROBE_CALLABLE={probe['callable']}",
            f"AUTORESEARCH_PROBE_SOURCE={os.path.realpath(os.path.join(repo_path, probe['source_path']))}",
            f"AUTORESEARCH_PROBE_NONCE={nonce}", *command_argv,
        ]
    started = time.time()
    try:
        execute_measurement = getattr(backend, "execute_attempt_command", backend.execute_command)
        executed = execute_measurement(
            handle, shlex.join(["env", "AUTORESEARCH_HOST_NATIVE_EXECUTION=1", *command_argv]), cwd=execution_cwd,
            timeout=_measurement_timeout(result, frozen_baseline))
    except Exception as exc:  # noqa: BLE001 -- finalizer converts host execution failures
        result["_host_measurement_error"] = f"host measurement failed: {type(exc).__name__}: {exc}"
        result["host_measurement_execution_uncertain"] = getattr(exc, "execution_uncertain", False) is True
        return
    ended = time.time()
    execution_stdout = str(getattr(executed, "stdout", "") or "")
    execution_stderr = str(getattr(executed, "stderr", "") or "")
    trusted = executed
    artifact_evidence = None
    if probe["kind"] == "external_scorer" and getattr(executed, "exit_code", 1) == 0:
        try:
            artifact_evidence = _attest_new_artifact(
                str(artifact_relative), str(artifact_path), started)
        except ValueError as exc:
            result["_host_measurement_error"] = f"artifact evidence invalid: {exc}"
            return
        scorer_cwd = os.path.realpath(os.path.join(repo_path, probe["cwd"]))
        scorer_argv = [str(artifact_path) if token == "{artifact}" else token
                       for token in probe["argv"]]
        try:
            trusted = execute_measurement(
                handle, shlex.join(scorer_argv), cwd=scorer_cwd,
                timeout=_measurement_timeout(result, frozen_baseline))
        except Exception as exc:  # noqa: BLE001
            result["_host_measurement_error"] = (
                f"host measurement failed: {type(exc).__name__}: {exc}")
            result["host_measurement_execution_uncertain"] = getattr(exc, "execution_uncertain", False) is True
            return
    stdout = str(getattr(trusted, "stdout", "") or "")
    stderr = str(getattr(trusted, "stderr", "") or "")
    end_commit = _repo_commit(repo_path)
    end_state = _tracked_repo_state(repo_path)
    try:
        immutable_end = _immutable_file_hashes(repo_path, contract, spec)
        if immutable_end != immutable_start:
            raise ValueError("immutable evaluator/data files changed during measurement")
        actual_controls_end = _actual_controls(repo_path, contract, spec, immutable_end)
        if actual_controls_end != actual_controls_start:
            raise ValueError("actual controls changed during measurement")
        untracked_evidence = _validate_untracked_files(
            untracked_before or {}, _untracked_file_inventory(repo_path),
            contract, execution_phase)
        if bootstrap_path is not None:
            with open(bootstrap_path, "rb") as handle_bootstrap:
                bootstrap_end = "sha256:" + hashlib.sha256(handle_bootstrap.read()).hexdigest()
            if bootstrap_end != bootstrap_hash:
                raise ValueError("host probe bootstrap changed during measurement")
        protocol_evidence = _parse_protocol_evidence(
            contract, spec, execution_stdout, execution_stderr, ended - started)
    except (TypeError, ValueError, re.error) as exc:
        result["_host_measurement_error"] = (
            "protocol evidence invalid: " + str(exc)
            if "budget" in str(exc) or "protocol" in str(exc) or "fixed" in str(exc)
            else "immutable evaluator violation: " + str(exc))
        return
    execution_rc = int(getattr(executed, "exit_code", 1))
    trusted_rc = int(getattr(trusted, "exit_code", 1))
    rc = execution_rc if execution_rc != 0 else trusted_rc
    if probe["kind"] == "python_callable":
        value, call_count = _parse_callable_probe(stderr, str(nonce))
        trusted_probe_evidence = {
            "kind": "python_callable", "source_path": probe["source_path"],
            "module": probe["module"], "callable": probe["callable"],
            "call_count": call_count, "bootstrap_sha256": bootstrap_hash,
        }
        metric_evidence = {
            "source": "host_callable_probe", "stream": "stderr",
            "metric": str(result.get("metric") or ""), "value": value,
            "nonce_sha256": "sha256:" + hashlib.sha256(str(nonce).encode()).hexdigest(),
        }
    else:
        parser = probe["parser"]
        value = _parse_metric(parser, stdout, stderr) if rc == 0 else None
        trusted_probe_evidence = {
            "kind": "external_scorer", "argv": probe["argv"], "cwd": probe["cwd"],
        }
        metric_evidence = {
            "source": "host_parser", "stream": parser["stream"],
            "regex": parser["regex"], "group": parser["group"],
            "metric": str(result.get("metric") or ""), "value": value,
        }
    attestation = {
        "host_issued": True,
        "source": "host_remeasurement",
        "phase": execution_phase,
        "attempt_id": "host-" + uuid.uuid4().hex,
        "measurement_spec": spec,
        "argv": execution["argv"],
        "cwd": execution["cwd"],
        "start": started,
        "end": ended,
        "rc": rc,
        "stdout_sha256": "sha256:" + hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": "sha256:" + hashlib.sha256(stderr.encode()).hexdigest(),
        "repo_commit_start": start_commit,
        "repo_commit_end": end_commit,
        "repo_commit": end_commit,
        "tracked_diff_hash_start": start_state.get("tracked_diff_hash"),
        "tracked_diff_hash_end": end_state.get("tracked_diff_hash"),
        "tracked_diff_present": end_state.get("tracked_diff_present"),
        "tracked_diff_paths": changed_paths,
        "immutable_file_hashes": immutable_end,
        "official_source_evidence": official_source,
        "trusted_probe_evidence": trusted_probe_evidence,
        "artifact_evidence": artifact_evidence,
        "prior_artifact_evidence": prior_artifact_evidence,
        "agent_attempt_evidence": {
            "authority": "host_execution_broker",
            "attempt_id": agent_attempt_id, "argv": agent_attempt.get("argv"),
            "normalized_argv": normalized_attempt_argv,
            "execution_argv": execution["argv"],
            "cwd": agent_attempt.get("cwd"),
            "start": agent_attempt.get("started_at"), "end": agent_attempt.get("ended_at"),
            "rc": agent_attempt.get("rc"), "count": len(records),
        },
        "resource_evidence": resource_evidence,
        "actual_controls": actual_controls_end,
        "untracked_files": untracked_evidence,
        "protocol_evidence": protocol_evidence,
        "metric_evidence": metric_evidence,
    }
    result["host_attestation"] = attestation
    result["run_attestation"] = attestation
    if value is None:
        result["_host_measurement_error"] = "host measurement produced no finite metric"
        return
    target = "baseline" if execution_phase == "baseline" else "candidate"
    result[target] = RunResult(
        metric_name=str(result.get("metric") or ""), metric_value=value, valid=True)


def _coerce_metric(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _host_attestation_shape_valid(attestation: dict, phase: str) -> bool:
    """Validate producer invariants before a host result becomes authoritative."""
    try:
        spec = _normalize_measurement_spec(attestation.get("measurement_spec"))
        start = float(attestation.get("start"))
        end = float(attestation.get("end"))
        protocol = attestation.get("protocol_evidence")
        protocol_value = float(protocol.get("value"))
        hashes = attestation.get("immutable_file_hashes")
        metric_evidence = attestation.get("metric_evidence")
        metric_value = float(metric_evidence.get("value"))
        source_evidence = attestation.get("official_source_evidence")
        probe_evidence = attestation.get("trusted_probe_evidence")
        actual_controls = attestation.get("actual_controls")
        untracked_files = attestation.get("untracked_files")
        attempt_evidence = attestation.get("agent_attempt_evidence")
    except (AttributeError, TypeError, ValueError, re.error):
        return False
    hash_pattern = re.compile(r"sha256:[0-9a-f]{64}\Z")
    source_valid = bool(isinstance(source_evidence, list) and source_evidence and all(
        isinstance(item, dict)
        and item.get("authority") in {"repository", "user"}
        and hash_pattern.fullmatch(str(item.get("identity_sha256") or ""))
        for item in source_evidence))
    if spec["trusted_probe"]["kind"] == "external_scorer":
        probe_valid = bool(
            isinstance(probe_evidence, dict)
            and set(probe_evidence) == {"kind", "argv", "cwd"}
            and probe_evidence.get("kind") == "external_scorer"
            and probe_evidence.get("argv") == spec["trusted_probe"]["argv"]
            and probe_evidence.get("cwd") == spec["trusted_probe"]["cwd"])
    else:
        probe_valid = bool(
            isinstance(probe_evidence, dict)
            and set(probe_evidence) == {
                "kind", "source_path", "module", "callable", "call_count",
                "bootstrap_sha256"}
            and probe_evidence.get("kind") == "python_callable"
            and all(probe_evidence.get(field) == spec["trusted_probe"][field]
                    for field in ("source_path", "module", "callable"))
            and isinstance(probe_evidence.get("call_count"), int)
            and probe_evidence.get("call_count") == 1
            and hash_pattern.fullmatch(str(
                probe_evidence.get("bootstrap_sha256") or "")))
    attempt_valid = bool(
        isinstance(attempt_evidence, dict)
        and set(attempt_evidence) == {
            "authority", "attempt_id", "argv", "normalized_argv", "execution_argv", "cwd",
            "start", "end", "rc", "count",
        }
        and attempt_evidence.get("authority") == "host_execution_broker"
        and isinstance(attempt_evidence.get("attempt_id"), str)
        and attempt_evidence["attempt_id"].startswith("attempt-")
        and attempt_evidence.get("normalized_argv") == spec["execution"]["argv"]
        and attempt_evidence.get("execution_argv") == spec["execution"]["argv"]
        and isinstance(attempt_evidence.get("cwd"), str)
        and isinstance(attempt_evidence.get("start"), (int, float))
        and isinstance(attempt_evidence.get("end"), (int, float))
        and attempt_evidence["end"] >= attempt_evidence["start"]
        and attempt_evidence.get("rc") == 0
        and isinstance(attempt_evidence.get("count"), int)
        and attempt_evidence["count"] >= 1
        and (phase != "method" or not _variant_cap()
             or attempt_evidence["count"] <= _variant_cap()))
    prior = attestation.get("prior_artifact_evidence")
    if isinstance(prior, dict) and prior.get("state") == "absent":
        prior_valid = set(prior) == {"state", "path"}
    elif isinstance(prior, dict) and prior.get("state") == "quarantined":
        prior_valid = bool(
            set(prior) == {
                "state", "path", "quarantine_path", "sha256", "size", "mtime",
                "source", "attempt_id",
            }
            and prior.get("source") == "cited_agent_attempt"
            and prior.get("attempt_id") == attempt_evidence.get("attempt_id")
            and hash_pattern.fullmatch(str(prior.get("sha256") or ""))
            and isinstance(prior.get("quarantine_path"), str)
            and isinstance(prior.get("size"), int) and prior["size"] >= 0
            and isinstance(prior.get("mtime"), (int, float)))
    else:
        prior_valid = False
    return bool(
        attestation.get("host_issued") is True
        and attestation.get("source") == "host_remeasurement"
        and attestation.get("phase") == phase
        and isinstance(attestation.get("attempt_id"), str)
        and attestation["attempt_id"].startswith("host-")
        and attestation.get("argv") == spec["execution"]["argv"]
        and attestation.get("cwd") == spec["execution"]["cwd"]
        and math.isfinite(start) and math.isfinite(end) and end >= start
        and isinstance(attestation.get("rc"), int)
        and not isinstance(attestation.get("rc"), bool)
        and isinstance(attestation.get("repo_commit"), str)
        and isinstance(attestation.get("repo_commit_start"), str)
        and attestation.get("repo_commit") == attestation.get("repo_commit_end")
        and all(isinstance(attestation.get(field), str)
                and hash_pattern.fullmatch(attestation[field])
                for field in ("stdout_sha256", "stderr_sha256",
                              "tracked_diff_hash_start", "tracked_diff_hash_end"))
        and isinstance(attestation.get("tracked_diff_present"), bool)
        and isinstance(hashes, dict) and bool(hashes)
        and all(isinstance(path, str) and path and not os.path.isabs(path)
                and isinstance(digest, str) and hash_pattern.fullmatch(digest)
                for path, digest in hashes.items())
        and source_valid
        and probe_valid
        and ((spec["trusted_probe"]["kind"] == "python_callable"
              and attestation.get("artifact_evidence") is None)
             or (spec["trusted_probe"]["kind"] == "external_scorer"
                 and isinstance(attestation.get("artifact_evidence"), dict)
                 and prior_valid))
        and attempt_valid
        and _actual_controls_pair_matches(actual_controls, actual_controls)
        and actual_controls.get("immutable_file_hashes") == hashes
        and isinstance(untracked_files, dict)
        and set(untracked_files) == {"before", "after", "added_allowed", "added_other"}
        and isinstance(untracked_files.get("before"), dict)
        and isinstance(untracked_files.get("after"), dict)
        and isinstance(untracked_files.get("added_allowed"), list)
        and isinstance(untracked_files.get("added_other"), list)
        and isinstance(attestation.get("resource_evidence"), dict)
        and set(attestation["resource_evidence"]) == {
            "accelerator_count", "device_models", "source", "caveat"}
        and attestation["resource_evidence"].get("source") in {
            "nvidia-smi", "allocation_count_only"}
        and isinstance(attestation["resource_evidence"].get("caveat"), str)
        and _resource_pair_matches(
            attestation["resource_evidence"], attestation["resource_evidence"])
        and isinstance(attestation.get("tracked_diff_paths"), list)
        and isinstance(protocol, dict)
        and set(protocol) == {"source", "unit", "value"}
        and protocol.get("source") in {"host_elapsed", "host_parser"}
        and protocol.get("unit") in {"seconds", "steps", "tokens"}
        and (protocol.get("source") != "host_elapsed" or protocol.get("unit") == "seconds")
        and math.isfinite(protocol_value)
        and isinstance(metric_evidence, dict)
        and isinstance(metric_evidence.get("metric"), str)
        and math.isfinite(metric_value)
        and (
            (spec["trusted_probe"]["kind"] == "external_scorer"
             and set(metric_evidence) == {
                 "source", "stream", "regex", "group", "metric", "value"}
             and metric_evidence.get("source") == "host_parser"
             and metric_evidence.get("stream") == spec["trusted_probe"]["parser"]["stream"]
             and metric_evidence.get("regex") == spec["trusted_probe"]["parser"]["regex"]
             and metric_evidence.get("group") == spec["trusted_probe"]["parser"]["group"])
            or
            (spec["trusted_probe"]["kind"] == "python_callable"
             and set(metric_evidence) == {
                 "source", "stream", "metric", "value", "nonce_sha256"}
             and metric_evidence.get("source") == "host_callable_probe"
             and metric_evidence.get("stream") == "stderr"
             and hash_pattern.fullmatch(str(metric_evidence.get("nonce_sha256") or ""))
            )
        )
    )


def _salvage_method_from_variants(result_path: str, lower_is_better: bool):
    """#67: agent 常在"没有改进"时把 method 留 null,却其实已在 analysis/variants.jsonl 里测了
    多个变体的真实值(如 speedrun: 8 个变体全测出 76.8~76.93s 到 val_loss 3.28 的秒数)。捞出最优
    的**非 baseline** 变体测量值作为 method —— 跑过就该有值,哪怕不如 baseline(honest 负结果)。
    返回 (value, note, attempt_id, valid)；结构化 phase 会在 host finalizer 验证该 attempt。Fail-open。"""
    try:
        vf = os.path.join(os.path.dirname(result_path), "analysis", "variants.jsonl")
        if not os.path.exists(vf):
            return None, "", None, None
        vals = []
        with open(vf, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                nm = str(o.get("variant") or "").lower()
                if "baseline" in nm or "unmodified" in nm:   # 跳过 baseline 行
                    continue
                # A repository evaluator may emit a finite primary metric even when a
                # separate correctness/quality gate fails.  Keep that observation in the
                # history, but never promote it to the best admissible method.
                if _explicit_candidate_validity(o) is False:
                    continue
                r = _coerce_metric(o.get("result"))
                if r is None:
                    r = _coerce_metric(o.get("method"))
                if r is not None:
                    vals.append((r, str(o.get("attempt_id") or "") or None,
                                 _explicit_candidate_validity(o)))
        if not vals:
            return None, "", None, None
        best, attempt_id, valid = (min(vals, key=lambda item: item[0]) if lower_is_better
                                   else max(vals, key=lambda item: item[0]))
        return best, (f" [method salvaged from variants.jsonl: best of {len(vals)} measured "
                      f"variants = {best} (agent measured variants but left method null)]"), attempt_id, valid
    except Exception:  # noqa: BLE001
        return None, "", None, None


def _explicit_candidate_validity(obj: Any) -> bool | None:
    """Return an explicitly reported repository gate result, when present.

    This is deliberately task-agnostic: repositories choose their own gate and threshold;
    the runner only consumes a boolean outcome.  ``False`` dominates conflicting fields so
    a finite speed/size/loss cannot be accepted after the official evaluator rejected it.
    Legacy artifacts without a separate gate remain supported.
    """
    if not isinstance(obj, dict):
        return None

    values: list[bool] = []

    def add(value: Any) -> None:
        if isinstance(value, bool):
            values.append(value)

    def add_text(value: Any) -> None:
        """Consume explicit boolean gate assignments embedded in result evidence text."""
        if not isinstance(value, str):
            return
        for match in re.finditer(
                r"(?i)\b(?:candidate_valid|official_correctness|correctness|"
                r"[a-z0-9_]+_gate_(?:passed|met))[\"']?\s*[:=]\s*(true|false)\b",
                value):
            values.append(match.group(1).casefold() == "true")

    for key in ("candidate_valid", "valid", "official_correctness", "correctness_gate_passed",
                "quality_gate_passed"):
        add(obj.get(key))
    for container_key in ("method", "metric"):
        container = obj.get(container_key)
        if isinstance(container, dict):
            for key in ("valid", "correctness", "correctness_gate_passed",
                        "quality_gate_passed"):
                add(container.get(key))

    # Direct fields describe the selected candidate.  Historical outcomes embedded in
    # notes may legitimately contain failed earlier attempts, so consult them only when
    # the selected result did not report a gate outcome of its own.
    if False in values:
        return False
    if True in values:
        return True

    audits = [obj.get("comparison_audit"), obj.get("benchmark_contract")]
    notes = obj.get("notes")
    add_text(notes)
    if isinstance(notes, str) and notes.startswith("COMPARISON_AUDIT:"):
        try:
            audits.append(json.loads(notes.split(":", 1)[1].strip()))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    for audit in audits:
        if not isinstance(audit, dict):
            continue
        outcomes = audit.get("observed_outcomes")
        if isinstance(outcomes, list):
            for outcome in outcomes:
                add_text(outcome)
            continue
        if not isinstance(outcomes, dict):
            continue
        for key in ("candidate_valid", "official_correctness", "correctness",
                    "correctness_gate_passed", "quality_gate_passed"):
            add(outcomes.get(key))

    if False in values:
        return False
    if True in values:
        return True
    return None


def _explicit_baseline_validity(obj: dict, execution_phase: str | None) -> bool | None:
    values = [obj.get("baseline_valid"), _explicit_candidate_validity({
        "method": obj.get("baseline"),
    })]
    if execution_phase == "baseline":
        values.append(_explicit_candidate_validity({
            k: obj[k] for k in ("official_correctness", "correctness_gate_passed",
                               "valid") if k in obj
        }))
    if any(value is False for value in values):
        return False
    return True if any(value is True for value in values) else None


def _dump_ablation_report(val: Any) -> str:
    """check-expectation 轮 agent 自报的逐组件消融报告 -> 归一化为 JSON 字符串(供 handler
    check_expectation() 解析)。接受 list / dict / 已是字符串;非法/空 -> ""。永不抛。"""
    if not val:
        return ""
    try:
        if isinstance(val, str):
            json.loads(val)          # 已是 JSON 字符串: 校验后原样
            return val
        return json.dumps(val, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return ""


def _measurement_attempt(records: dict, attempt_id: str | None, phase: str) -> dict | None:
    if attempt_id:
        record = records.get(attempt_id)
        if record and record.get("phase") != phase:
            raise ValueError("result attempt_id belongs to a different execution phase")
        return record
    if phase == "baseline":
        eligible = [r for r in records.values()
                    if r.get("phase") == phase and r.get("rc") == 0]
        if len(eligible) == 1:
            return eligible[0]
        if len(eligible) > 1:
            raise ValueError(
                "multiple successful baseline attempts: ar_result.json must cite the exact "
                "attempt_id that produced the reported metric; do not choose by recency")
    return None


def _result_completion_request(result_path: str, *, execution_phase: str | None = None,
                               attempt_log: str | None = None,
                               evaluation: dict | None = None,
                               attempt_authority: _AttemptAuthority | None = None) -> str | None:
    """Request missing result serialization, without restarting the experiment."""
    from pathlib import Path

    records = _attempt_records(attempt_log) if attempt_log else {}
    pending = [aid for aid, record in records.items()
               if record.get("phase") == execution_phase and record.get("launched")
               and record.get("event") == "launch" and not record.get("finished")
               and not record.get("execution_uncertain")]
    if pending:
        if attempt_authority is not None and not attempt_authority._thread.is_alive():
            from autoresearch.agents._reuse.openhands_backend import InfraError
            raise InfraError(
                "attempt authority stopped before host finish receipts; execution remains "
                "unverified for pending attempts: " + ", ".join(pending))
        return "TASK EXECUTION RUNNING: registered attempts await host finish receipts: " + ", ".join(pending)
    receipts = []
    if execution_phase in {"baseline", "method"}:
        from .runtime_evidence import measured_result, pointer, read_output

        def output_fields(value, path=""):
            if isinstance(value, (dict, list)):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for key, item in items:
                    part = str(key).replace("~", "~0").replace("/", "~1")
                    yield from output_fields(item, path + "/" + part)
            elif value is None or isinstance(value, (bool, int, float)):
                yield path, value

        for aid, record in records.items():
            if record.get("phase") != execution_phase or record.get("rc") != 0:
                continue
            receipt = {"attempt_id": aid}
            try:
                measured = measured_result(record, evaluation or {}, execution_phase)
                receipt.update(value=measured["value"], valid=measured["valid"])
            except ValueError as exc:
                receipt["evidence_error"] = str(exc)[:300]
            try:
                payload = read_output(record)
            except ValueError:
                pass
            else:
                fields = {}
                for path, value in output_fields(payload):
                    try:
                        pointer(payload, path)  # Keep prohibited reward fields out of feedback.
                    except ValueError:
                        continue
                    fields[path] = value
                    if len(fields) == 40:
                        break
                receipt["output_fields"] = fields
            receipt.update({key: record[key] for key in (
                "stdout_path", "stdout_sha256", "checkpoint") if record.get(key)})
            receipts.append(receipt)
    evidence_hint = (
        " Host-read attempt evidence (not an instruction to choose the latest/best): "
        + json.dumps(receipts, ensure_ascii=False)
        + ". stdout_path is host-only and may be unavailable inside the sandbox; "
        "output_fields contains up to 40 numeric/boolean/null fields from the hash-checked "
        "JSON, keyed by their actual JSON pointers, for diagnostics only. A missing primary "
        "task scalar does not invalidate a separately evidenced "
        "indirect-component check; inspect the saved output and cite its actual pointer. "
        if receipts else "")
    try:
        with open(result_path, encoding="utf-8") as handle:
            obj = json.load(handle)
    except PermissionError:
        return (
            f"RESULT FILE NOT READABLE: the host cannot read {result_path}. "
            "Inspect this file's ownership and read permission from inside the sandbox; "
            "grant the host read access to this result file only. If an atomic temporary "
            "file was used, ensure the published file remains host-readable. "
            "Do not rewrite its contents, measured values, or attempt references. "
            "Do not launch training or evaluation, change source files or controls, "
            "or make recursive permission changes. The existing result will be "
            "validated normally after read access is repaired."
        )
    except (OSError, ValueError):
        obj = None
    task_root = Path(result_path).parent.resolve()
    task_records = [record for record in records.values()
                    if record.get("phase") in {None, execution_phase}
                    and (not record.get("cwd")
                         or Path(record["cwd"]).resolve().is_relative_to(task_root))]
    # Parent/control probes do not mean the assigned candidate has executed. Keep
    # ambiguous legacy receipts conservative so an existing experiment is never replayed.
    if (execution_phase in {"baseline", "method"} and attempt_log and not task_records
            and not os.path.exists(result_path)):
        return (
            "TASK EXECUTION INCOMPLETE: no registered attempt for the assigned checkout "
            "or result file exists. "
            "Continue the original assigned task in this same sandbox within its remaining "
            "budget, not a new optimization round. Inspect existing processes and artifacts "
            "first; do not duplicate work that is still running. If execution has not begun, "
            "implement only the assigned scope and run its required check through ar-attempt, "
            f"then save the actual evidence to {result_path}. An indirect component needs its "
            "declared intermediate check, not an invented final metric. Preserve the frozen "
            "task, controls and attempt limits. If execution is genuinely blocked, write a "
            "structured failure with concrete reasons and null unmeasured values."
        )
    if isinstance(obj, dict):
        if execution_phase not in {"baseline", "method"} or not attempt_log:
            return None
        parsed = _build_result_from_file(
            result_path, "handoff", None, None, lambda *_: None, execution_phase)
        which = "baseline" if execution_phase == "baseline" else "candidate"
        reported = _coerce_metric(getattr(parsed.get(which), "metric_value", None))
        if reported is None and not (execution_phase == "method" and parsed.get("ablation_report")):
            return None  # Honest negative results do not require another experiment.
        cited_id = parsed.get("_metric_attempt_id")
        if cited_id and cited_id not in records:
            historical = _evidence_records(
                attempt_log, parsed, os.path.dirname(result_path)).get(cited_id)
            if (historical and os.path.realpath(str(historical.get("cwd") or ""))
                    != os.path.realpath(os.path.dirname(result_path))):
                return (
                    "HISTORICAL CANDIDATE NEEDS CURRENT VALIDATION: the cited receipt "
                    f"{cited_id!r} belongs to an earlier checkout in this session; it is "
                    "not a misspelled attempt ID. Preserve the historical candidate and its "
                    "original evidence as history, not as a new current-worktree result. Report "
                    "the current experiment's bound outcome even when negative. If the "
                    "selected code was reverted to that checkpoint, first inspect existing "
                    "current-worktree receipts; if none validates that selected code, run "
                    "its required check through ar-attempt from this checkout within the "
                    "remaining budget. Do not relabel the historical receipt as current, "
                    "rewrite saved output, or replace an unmeasured value with an old score."
                )
            registered = [aid for aid, item in records.items()
                          if item.get("phase") == execution_phase and item.get("finished")][-6:]
            return (
                f"RESULT EVIDENCE HANDOFF INCOMPLETE: attempt_id {cited_id!r} is not an "
                f"exact host record. Registered finished IDs (diagnostic only): {registered!r}. "
                "Inspect the saved output for the reported value and correct the top-level "
                "attempt_id and matching evidence refs in the existing result JSON. Do not "
                "infer identity from a similar ID or metric, alter receipts, or rerun an "
                "already measured candidate merely to repair a citation."
            )
        try:
            from .runtime_evidence import measured_result
            selected = _measurement_attempt(
                records, parsed.get("_metric_attempt_id"), execution_phase)
            if execution_phase == "method" and selected and selected.get("rc") == 0:
                binding_error = _bind_component_evidence_attempt(
                    dict(parsed), _evidence_records(attempt_log, parsed, os.path.dirname(result_path)),
                    os.path.dirname(result_path), execution_phase)
                if binding_error:
                    raise ValueError(binding_error)
            if execution_phase == "baseline" and parsed.get("_reported_artifact_paths"):
                from .checkpoint_artifacts import validate_checkpoint
                if not selected or not selected.get("checkpoint"):
                    raise ValueError("declared runtime artifacts have no measured checkpoint")
                validate_checkpoint(selected["checkpoint"],
                                    artifact_paths=parsed["_reported_artifact_paths"],
                                    repo_path=os.path.dirname(result_path))
            if reported is not None:
                measured = measured_result(
                    selected, evaluation or {}, execution_phase,
                    expected_metric_name=(evaluation or {}).get("metric_name")
                    or parsed.get("_reported_metric"))
                if not math.isclose(measured["value"], reported, rel_tol=1e-15, abs_tol=0.0):
                    raise ValueError("reported task scalar disagrees with the saved evaluator output")
            elif not selected or selected.get("rc") != 0:
                return None  # Preserve genuine failed checks; do not demand another experiment.
            if execution_phase == "method" and parsed.get("ablation_report"):
                from .runtime_evidence import bind_component_measurements
                report = json.loads(parsed["ablation_report"])
                if isinstance(report, dict):
                    report = [report]
                if isinstance(report, list) and all(isinstance(item, dict) for item in report):
                    # Check the same evidence before closing; do not rewrite the result or receipts.
                    bind_component_measurements(report, selected, _evidence_records(
                        attempt_log, parsed, os.path.dirname(result_path)))
            if reported is not None and any(
                    not isinstance(parsed.get(key), str) or not parsed[key].strip()
                    for key in ("_reported_metric", "_reported_metric_unit")):
                raise ValueError(
                    "measured result omitted metric.name or metric.unit; supply the metric "
                    "object using the actual evaluator quantity and unit, consistent with the "
                    "frozen task. Do not relabel a different quantity, convert or copy a score, "
                    "or change the measured value or its attempt")
            if reported is not None and evaluation and (
                    (evaluation.get("metric_name") is not None
                     and _normalized_metric_identity(parsed.get("_reported_metric"))
                     != _normalized_metric_identity(evaluation["metric_name"]))
                    or (evaluation.get("metric_unit") is not None
                        and _normalized_metric_unit(parsed.get("_reported_metric_unit"))
                        != _normalized_metric_unit(evaluation["metric_unit"]))
                    or ("lower_is_better" in evaluation
                        and parsed.get("_reported_lower_is_better") is not evaluation["lower_is_better"])):
                raise ValueError(
                    "reported objective identity differs from the frozen evaluation contract: "
                    f"expected {evaluation!r}; reported name={parsed.get('_reported_metric')!r}, "
                    f"unit={parsed.get('_reported_metric_unit')!r}, "
                    f"lower_is_better={parsed.get('_reported_lower_is_better')!r}. "
                    "Correct only the metric identity in the result JSON if it describes the same "
                    "measured quantity; do not change the measurement or rerun the evaluator")
            if execution_phase == "baseline":
                provenance_error = _baseline_provenance_error(
                    parsed.get("_reported_repo_commit"), _repo_commit(os.path.dirname(result_path)))
                if provenance_error:
                    raise ValueError(provenance_error)
        except ValueError as exc:
            return (
                f"RESULT EVIDENCE HANDOFF INCOMPLETE: {exc}. Repair {result_path} in this "
                "same sandbox before finishing. " + evidence_hint +
                "For an existing valid JSON file, use json.load, modify only the missing "
                "fields (the attempt ID is the TOP-LEVEL attempt_id), then json.dump to a "
                "temporary file and os.replace atomically. Do not rewrite long notes, "
                "reinspect host processes, or recreate the report. Cite the exact successful ar-attempt ID for "
                "the reported scalar. For parent/control errors, correct parent_evidence to the "
                "matching host attempt, or measure the unchanged parent in the designated parent-control "
                "checkout: cd there BEFORE invoking ar-attempt, not only inside its child wrapper. "
                "Do not substitute another workspace or the candidate's output for parent evidence. "
                "If its saved stdout lacks the evaluator JSON, or already contains JSON but the declared metric/gate pointers are missing, "
                "this is an output-adapter gap, not evidence that training did not run. "
                "Editing ar_result.json alone cannot change captured stdout. Never invent a passing gate "
                "or change the frozen metric/pointers to bypass verification. Rerun only "
                "the unchanged evaluator from the existing checkpoint through ar-attempt, "
                "emitting one JSON object derived from its actual output and correctness gates "
                "inside the same ar-attempt child command, not after ar-attempt returns; "
                "use a wrapper that checks evaluator success, reads its fresh output and prints JSON "
                "using the declared JSON pointers (defaults: baseline/baseline_valid or "
                "method/candidate_valid). Preserve the candidate, data, evaluator and budget; "
                "do not retrain, optimize, edit old attempt receipts, or merely echo a claimed "
                "score. If this cannot be repaired, retain the observation and report the "
                "concrete evidence gap without claiming a verified result."
            )
        return None
    return (
        f"RESULT HANDOFF INCOMPLETE: write one JSON object to {result_path} now, "
        "using existing attempt records and artifacts only. " + evidence_hint +
        "Keep this a minimal machine-readable result, not a long report; use json.dump "
        "and atomic os.replace rather than hand-escaped JSON. Do not launch training, "
        "evaluation, installation, or another experiment, and do not modify candidate "
        "code or frozen controls. Preserve the actual attempt IDs and measured values. "
        "If no valid measurement exists, write baseline/method as null as appropriate "
        "and report the concrete failure and evidence paths; never invent a score or "
        "substitute an unchanged baseline evaluation for the method. This is one bounded "
        "serialization step, not another optimization round."
    )


def _build_result_from_file(result_path: str, name: str, metric_hint: str | None,
                            lower_is_better_hint: bool | None, emit: Any,
                            execution_phase: str | None = None) -> dict | None:
    """Read the agent-written ar_result.json and map it to a ``run_repo_improve``-shaped dict.
    Returns None if the file is missing/garbled (nothing the agent measured to surface).

    Called on BOTH the success path AND the run_agent_task error path: a TRANSIENT agent/sandbox
    error at the END of the session must NOT discard a baseline/method the agent ALREADY wrote to
    disk. That was the "win-loss" bug -- glm 6-concurrent lost 5/6 real baselines because the
    except handler returned None without ever reading the file the agent had already written."""
    try:
        with open(result_path, encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("ar_result.json must contain a JSON object")
    except Exception as exc:  # noqa: BLE001 -- no/garbled file -> nothing to surface
        emit("agentic_error", {"stage": "result", "error": f"{type(exc).__name__}: {exc}"})
        return None

    raw_metric = obj.get("metric")
    reported_metric = (raw_metric.get("name")
                       if isinstance(raw_metric, dict) else raw_metric)
    reported_metric_unit = (raw_metric.get("unit")
                            if isinstance(raw_metric, dict) else None)
    reported_lower_is_better = obj.get("lower_is_better")
    if reported_lower_is_better is None and isinstance(raw_metric, dict):
        reported_lower_is_better = raw_metric.get("lower_is_better")
    metric = reported_metric or metric_hint
    lower_is_better = reported_lower_is_better
    if lower_is_better is None:
        lower_is_better = lower_is_better_hint if lower_is_better_hint is not None else True
    if type(lower_is_better) is not bool:
        emit("agentic_error", {"stage": "result", "error": "lower_is_better must be a JSON boolean"})
        return None
    raw_baseline = obj.get("baseline")
    raw_method = obj.get("method")
    base_v = _coerce_metric(
        raw_baseline.get("value") if isinstance(raw_baseline, dict) else raw_baseline)
    meth_v = _coerce_metric(
        raw_method.get("value") if isinstance(raw_method, dict) else raw_method)
    if meth_v is None:
        # Runner-facing artifacts call this field ``method_metric``.  Repository agents
        # occasionally reuse that name in ar_result.json; preserve a real finite result
        # instead of turning an honest negative experiment into an execution failure.
        meth_v = _coerce_metric(obj.get("method_metric"))
    if meth_v is None and execution_phase == "method" and isinstance(raw_metric, dict):
        # Some repository agents put the measured candidate directly in the structured
        # metric object instead of duplicating it under ``method``.  The phase disambiguates
        # this from the same shape used for a baseline measurement.
        meth_v = _coerce_metric(raw_metric.get("value"))
    notes = str(obj.get("notes") or "")[:4000]
    comparison_audit = obj.get("comparison_audit")
    if isinstance(obj.get("baseline_drift"), dict) and obj["baseline_drift"]:
        # Preserve a request for review, not a trusted diagnosis or replacement score.
        comparison_audit = {
            **(comparison_audit if isinstance(comparison_audit, dict) else {}),
            "baseline_drift": obj["baseline_drift"],
        }
    # #67 + best-so-far(nanochat 2026-08-09 实锤): 始终把 method 对齐 variants.jsonl 里的最优变体,取更优。
    # 两种情形都覆盖: (a) agent 因"没改进"把 method 留 null -> 捞最优变体(跑过就有值); (b) agent 写了一个
    # 早期次优 method(如首个改进 1.03595)后继续测出更优变体(1.014969)却只 append variants.jsonl、没回写
    # ar_result -> 非空的过期 method 会盖住真实最优,把 2.5% 改进低报成 0.4%。只在 method==None 时捞是不够的。
    metric_attempt_id = str(
        obj.get("attempt_id")
        or (raw_baseline.get("attempt_id") if isinstance(raw_baseline, dict) else "")
        or (raw_method.get("attempt_id") if isinstance(raw_method, dict) else "")
        or "") or None
    metric_evidence_source = "ar_result.json"
    candidate_validity = _explicit_candidate_validity(obj)
    _sv, _snote, _sattempt, _svalid = _salvage_method_from_variants(
        result_path, bool(lower_is_better))
    _better = _sv is not None and (meth_v is None
                                   or (_sv < meth_v if lower_is_better else _sv > meth_v))
    # Validate explicit native receipts before considering historical observations.
    if metric_attempt_id and (execution_phase in {"baseline", "method"} or not _sattempt):
        _better = False
    if _better:
        meth_v = _sv
        metric_attempt_id = _sattempt
        metric_evidence_source = "variants.jsonl"
        candidate_validity = _svalid
        notes = (notes + _snote)[:4000]
        emit("agentic_salvage", {"method": _sv, "source": "variants.jsonl"})
    # 组4(P1-3): 若 agent 报了 paired baseline/best 多-seed 复现,把 mean±std 落进 report
    # (报告可信度,不改判决)。旧结果只含 method 字段时保持兼容。
    _base_std = _coerce_metric(obj.get("baseline_std"))
    _base_runs = obj.get("baseline_runs")
    _std = _coerce_metric(obj.get("method_std"))
    _runs = obj.get("method_runs")
    if (_base_std is not None or isinstance(_base_runs, list)
            or _std is not None or isinstance(_runs, list)):
        base_n = len(_base_runs) if isinstance(_base_runs, list) else "?"
        base_runs_txt = (", ".join(str(r) for r in _base_runs)
                         if isinstance(_base_runs, list) else "")
        n = len(_runs) if isinstance(_runs, list) else "?"
        runs_txt = (", ".join(str(r) for r in _runs)) if isinstance(_runs, list) else ""
        base_txt = (f"baseline={base_v} 为 {base_n} 次不同 seed 的均值"
                    + (f", std={_base_std}" if _base_std is not None else "")
                    + (f", runs=[{base_runs_txt}]" if base_runs_txt else ""))
        notes = (f"{notes}\n[复现: {base_txt}; method={meth_v} 为 {n} 次不同 seed 的均值"
                 + (f", std={_std}" if _std is not None else "")
                 + (f", runs=[{runs_txt}]" if runs_txt else "") + "]")[:4000]
    emit("agentic_result", {"metric": metric, "baseline": base_v, "method": meth_v,
                            "method_std": _std})

    candidate_valid = meth_v is not None and candidate_validity is not False
    improved = False
    if base_v is not None and candidate_valid:
        improved = (meth_v < base_v) if lower_is_better else (meth_v > base_v)
    if base_v is None:
        verdict = {"verdict": "fail", "improved": False,
                   "reason": "agent produced no baseline value"}
    elif candidate_validity is False:
        verdict = {"verdict": "fail", "improved": False,
                   "reason": "repository evaluator rejected the candidate correctness/quality gate"}
    elif improved:
        verdict = {"verdict": "pass", "improved": True,
                   "reason": f"agent improved {metric}: {base_v} -> {meth_v}"}
    else:
        verdict = {"verdict": "fail", "improved": False,
                   "reason": f"no improvement ({metric}: baseline={base_v} method={meth_v})"}

    return {
        "benchmark": name,
        "baseline": RunResult(metric_name=str(metric or ""), metric_value=base_v,
                              valid=base_v is not None),
        "candidate": RunResult(metric_name=str(metric or ""), metric_value=meth_v,
                               valid=candidate_valid),
        "candidate_validity": candidate_validity,
        "reference_sources": obj.get("reference_sources") or [],
        # Baseline correctness is separate from the method's gate. Never borrow a
        # later method result (or variants history) to approve the baseline.
        "baseline_validity": _explicit_baseline_validity(obj, execution_phase),
        "metric": metric,
        "lower_is_better": bool(lower_is_better),
        "baseline_std": _base_std, "baseline_runs": _base_runs,
        "method_std": _std, "method_runs": _runs,
        "_reported_metric": reported_metric,
        "_reported_metric_unit": reported_metric_unit,
        "_reported_lower_is_better": reported_lower_is_better,
        "_metric_attempt_id": metric_attempt_id,
        "_metric_evidence_source": metric_evidence_source,
        "_reported_artifact_paths": (obj.get("artifact_paths", [])
                                     if metric_evidence_source == "ar_result.json" else []),
        "verdict": verdict,
        "report": notes or verdict["reason"],
        # 方法反思回路的输入: agent 自报本轮"改了啥/为什么/当前瓶颈"(比 git diff 远程更可靠)。
        "changes_summary": str(obj.get("changes_summary") or notes or "")[:2000],
        "rationale": str(obj.get("rationale") or "")[:2000],
        "bottleneck": str(obj.get("bottleneck") or "")[:2000],
        # Descriptive run configuration retained for reports and search memory.
        "config": str(obj.get("config") or ""),
        "_reported_repo_commit": (
            (obj["config"].get("repo_commit") or obj["config"].get("commit"))
            if isinstance(obj.get("config"), dict) else None),
        "benchmark_contract": obj.get("benchmark_contract"),
        "measurement": obj.get("measurement"),
        "run_manifest": obj.get("run_manifest"),
        "diagnosis": obj.get("diagnosis"),
        # Semantic-comparison supervision needs the candidate-side conditions and observed
        # budget.  Keeping this structured object avoids making the downstream guardian infer
        # them from prose or from a large git diff.
        "comparison_audit": (
            comparison_audit if isinstance(comparison_audit, dict)
            else None
        ),
        # check-expectation 轮(会议 2026-08-11): agent 逐组件消融的结构化报告(JSON 字符串), handler
        # 侧 check_expectation() 客观路由。非该算子轮时为空。
        "ablation_report": _dump_ablation_report(obj.get("ablation_report")),
    }


def _salvage_after_error(result_path: str, name: str, metric_hint: str | None,
                         lower_is_better_hint: bool | None, emit: Any,
                         err: str, execution_phase: str = "combined") -> dict | None:
    """After run_agent_task raised, try to recover a REAL measured value the agent already wrote.
    Combined/baseline results require an agent baseline; method results use the frozen baseline and
    only require a parseable phase result. The transient error is noted in the report. Returns None
    when nothing worth surfacing was measured."""
    salvaged = _build_result_from_file(result_path, name, metric_hint,
                                       lower_is_better_hint, emit,
                                       execution_phase=execution_phase)
    if salvaged is None or (execution_phase != "method" and not salvaged["baseline"].valid):
        return None
    emit("agentic_salvage", {"error": err, "baseline": salvaged["baseline"].metric_value,
                             "method": salvaged["candidate"].metric_value})
    salvaged["report"] = (f"[agent session ended with a transient error: {err}] "
                          + str(salvaged.get("report") or ""))[:4000]
    return salvaged


def _make_run_attestation(result: dict, execution_phase: str, attempt_log: str | None,
                          initial_commit: str | None, final_commit: str | None,
                          tracked_state: dict) -> tuple[dict, dict | None]:
    """Build host-owned evidence and return it with the cited wrapper record."""
    attempt_id = str(result.get("_metric_attempt_id") or "") or None
    record = (_attempt_records(attempt_log or "").get(attempt_id)
              if attempt_id is not None else None)
    metric_value = (getattr(result.get("baseline"), "metric_value", None)
                    if execution_phase == "baseline" else
                    getattr(result.get("candidate"), "metric_value", None))
    evidence = {
        "source": str(result.get("_metric_evidence_source") or "ar_result.json"),
        "attempt_id": attempt_id,
        "metric": str(result.get("metric") or ""),
        "value": _coerce_metric(metric_value),
    }
    attestation = {
        "host_issued": True,
        "phase": execution_phase,
        "attempt_id": attempt_id,
        "argv": record.get("argv") if record else None,
        "start": record.get("started_at") if record else None,
        "end": record.get("ended_at") if record else None,
        "rc": record.get("rc") if record else None,
        "repo_commit_start": (record.get("repo_commit") if record else initial_commit),
        "repo_commit": final_commit,
        "tracked_diff_hash_start": (record.get("tracked_diff_hash_start")
                                    if record else None),
        "attempt_tracked_diff_hash": (record.get("tracked_diff_hash_end")
                                      if record else None),
        "tracked_diff_hash": tracked_state.get("tracked_diff_hash"),
        "tracked_diff_present": tracked_state.get("tracked_diff_present"),
        "metric_evidence": evidence,
    }
    result["host_attestation"] = attestation
    result["run_attestation"] = attestation
    if execution_phase == "method":
        result["review_receipts"] = _saved_review_receipts(
            attempt_log, result, str((record or {}).get("cwd") or ""), attempt_id)
    return attestation, record


def _confirmed_boundary_fragment(boundary: dict, metric: str,
                                 lower_is_better: bool) -> dict:
    """Translate only explicitly confirmed fields for authority-aware contract merging."""
    fragment: dict[str, Any] = {
        "version": 1,
        "provenance": [{
            "source": "confirmed_boundary", "detail": boundary,
            "conflicts_with_official": boundary.get("conflicts_with_official") is True,
        }],
    }
    objective = {}
    if "metric" in boundary:
        objective["metric"] = metric
    if "direction" in boundary:
        objective["direction"] = "minimize" if lower_is_better else "maximize"
    if "objective_mode" in boundary:
        objective["mode"] = boundary["objective_mode"]
    if objective:
        fragment["objective"] = objective
    if "protocol_kind" in boundary:
        kind = ("quality_at_fixed_time" if boundary["protocol_kind"] == "fixed_wall_clock"
                else boundary["protocol_kind"])
        fragment["protocol"] = {"kind": kind}
        if boundary.get("training_budget_s") is not None:
            fragment["protocol"]["budget"] = {
                "wall_clock_s": boundary["training_budget_s"]}
        if kind == "quality_at_fixed_time":
            fragment["protocol"].update({
                "timing_scope": str(boundary.get("timing_scope") or "process"),
                "timer_authority": str(boundary.get("timer_authority") or "host"),
            })
    resources = {
        key: boundary[key] for key in (
            "accelerator_count", "accelerator_type", "world_size", "precision")
        if key in boundary
    }
    controls = {}
    if resources:
        controls["resources"] = resources
    if "data" in boundary:
        controls["data"] = {"requirement": boundary["data"]}
    if "evaluator" in boundary:
        controls["evaluator"] = {"requirement": boundary["evaluator"]}
    if "workload_accounting" in boundary:
        controls["workload"] = {"accounting": boundary["workload_accounting"]}
    if controls:
        fragment["controls"] = controls
    if "allowed_treatment_axes" in boundary:
        fragment["allowed_treatment_axes"] = list(boundary["allowed_treatment_axes"])
    return fragment


def _finalize_phase_result(result: dict, execution_phase: str,
                           frozen_baseline: dict | None,
                           attempt_log: str | None = None,
                           confirmed_boundary: dict | None = None,
                           initial_commit: str | None = None,
                           final_commit: str | None = None,
                           tracked_state: dict | None = None) -> dict:
    """Apply the phase contract to every parsed result, including error salvage paths."""
    if execution_phase == "combined":
        result.pop("_reported_metric", None)
        result.pop("_reported_metric_unit", None)
        result.pop("_reported_lower_is_better", None)
        return result

    metric = str(result.get("metric") or "")
    old_reason = str((result.get("verdict") or {}).get("reason") or "")
    tracked_state = tracked_state or {
        "tracked_diff_hash": None, "tracked_diff_present": None,
    }
    attestation = result.get("host_attestation")
    if not isinstance(attestation, dict):
        attestation = {}
    if execution_phase == "baseline":
        baseline = result.get("baseline")
        value = _coerce_metric(getattr(baseline, "metric_value", None))
        finite = value is not None and math.isfinite(value)
        boundary = confirmed_boundary if isinstance(confirmed_boundary, dict) else {}
        reported_metric = result.pop("_reported_metric", result.get("metric"))
        reported_metric_unit = result.pop("_reported_metric_unit", None)
        reported_lower_is_better = result.pop(
            "_reported_lower_is_better", result.get("lower_is_better"))
        if not reported_metric and boundary.get("metric") is not None:
            metric = str(boundary["metric"])
        if reported_lower_is_better is None and boundary.get("direction") is not None:
            lower_is_better = boundary["direction"] == "minimize"
        else:
            lower_is_better = bool(result.get("lower_is_better"))
        result["metric"] = metric
        result["metric_unit"] = reported_metric_unit
        result["lower_is_better"] = lower_is_better
        try:
            agent_contract = result.get("benchmark_contract")
            if boundary and (not isinstance(agent_contract, dict)
                             or int(agent_contract.get("version", 0)) < 2):
                raise BenchmarkContractError(
                    "confirmed-boundary baseline requires an evidence-bearing v2 contract")
            if isinstance(agent_contract, dict) and int(agent_contract.get("version", 0)) >= 2:
                agent_contract = json.loads(json.dumps(agent_contract))
                agent_contract["verified_evidence"] = {
                    "source": "host_remeasurement",
                    "artifact": attestation.get("attempt_id") or "host measurement",
                    "observations": {
                        "quality": {"metric": metric, "value": value},
                        "stdout_sha256": attestation.get("stdout_sha256"),
                        "stderr_sha256": attestation.get("stderr_sha256"),
                        "immutable_file_hashes": attestation.get("immutable_file_hashes"),
                        "protocol_evidence": attestation.get("protocol_evidence"),
                        "official_source_evidence": attestation.get(
                            "official_source_evidence"),
                        "trusted_probe_evidence": attestation.get(
                            "trusted_probe_evidence"),
                        "artifact_evidence": attestation.get("artifact_evidence"),
                        "agent_attempt_evidence": attestation.get(
                            "agent_attempt_evidence"),
                        "resource_evidence": attestation.get("resource_evidence"),
                        "actual_controls": attestation.get("actual_controls"),
                        "untracked_files": attestation.get("untracked_files"),
                    },
                }
                outcomes = agent_contract.setdefault("observed_outcomes", {})
                quality = outcomes.setdefault("quality", {"metric": metric})
                quality.update({"metric": metric, "value": value})
                evaluator = agent_contract.setdefault("controls", {}).setdefault(
                    "evaluator", {})
                if isinstance(evaluator, dict):
                    evaluator["measurement_spec"] = attestation.get("measurement_spec")
                    if (isinstance(boundary, dict)
                            and isinstance(boundary.get("measurement_spec"), dict)):
                        evaluator["official_source_evidence"] = _boundary_source_claims(
                            attestation["measurement_spec"])
            if boundary:
                if (boundary.get("protocol_kind") == "fixed_wall_clock"
                        and isinstance(agent_contract, dict)):
                    agent_contract.get("protocol", {}).pop("budget", None)
                    agent_contract.get("controls", {}).get("workload", {}).pop("steps", None)
                boundary_contract = contract_from_confirmed_boundary(
                    boundary, metric=metric, lower_is_better=lower_is_better,
                    fixed_config=str(result.get("config") or ""),
                )
                contract = (merge_benchmark_contracts(
                                agent_contract,
                                _confirmed_boundary_fragment(
                                    boundary, metric, lower_is_better))
                            if isinstance(agent_contract, dict) else boundary_contract)
            else:
                contract = normalize_contract(agent_contract)
        except BenchmarkContractError:
            contract = None
        empty_hash = "sha256:" + hashlib.sha256(b"").hexdigest()
        attempt_valid = bool(
            _host_attestation_shape_valid(attestation, "baseline")
            and attestation.get("rc") == 0
            and attestation.get("repo_commit_start") == initial_commit
            and attestation.get("repo_commit_end") == final_commit
            and attestation.get("tracked_diff_hash_start") == empty_hash
            and attestation.get("tracked_diff_hash_end") == empty_hash
            and _coerce_metric(
                (attestation.get("metric_evidence") or {}).get("value")) == value
        )
        provenance_clean = bool(
            initial_commit is not None
            and final_commit == initial_commit
            and tracked_state.get("tracked_diff_present") is False
        )
        valid = finite and contract is not None and attempt_valid and provenance_clean
        result["baseline"] = RunResult(
            metric_name=metric, metric_value=value if valid else None, valid=valid)
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        if valid:
            result["benchmark_contract"] = contract
            result["contract_fingerprint"] = contract_fingerprint(contract)
            reason = f"baseline measured: {metric}={value}"
            result["verdict"] = {"verdict": "pass", "improved": False, "reason": reason}
            result.pop("error", None)
        else:
            if not finite:
                reason = "baseline phase requires a finite baseline value"
                diagnosis = "baseline_contract_invalid"
            elif contract is None:
                reason = "baseline phase requires a valid benchmark contract"
                diagnosis = "baseline_contract_invalid"
            elif not provenance_clean or not attempt_valid:
                reason = "baseline provenance violation: tracked repository state is not pristine"
                host_error = str(result.get("_host_measurement_error") or "")
                diagnosis = ("protocol_evidence_violation" if "protocol evidence" in host_error
                             else "immutable_evaluator_violation"
                             if "immutable evaluator" in host_error
                             else "resource_evidence_violation"
                             if "resource evidence" in host_error
                             else "immutable_evaluator_violation"
                             if "actual controls" in host_error
                             else "execution_failure"
                             if "agent attempt evidence invalid" in host_error
                             else "measurement_spec_invalid" if host_error
                             else "baseline_provenance_violation")
            else:
                reason = "metric evidence violation: baseline metric is not bound to one successful attempt"
                diagnosis = "metric_evidence_violation"
            result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
            result["error"] = reason
            result["diagnosis"] = diagnosis
        if not result.get("report") or result.get("report") == old_reason:
            result["report"] = reason
        return result

    reported_metric = result.pop("_reported_metric", None)
    reported_metric_unit = result.pop("_reported_metric_unit", None)
    reported_direction = result.pop("_reported_lower_is_better", None)
    frozen = frozen_baseline or {}
    metric = str(frozen.get("metric_name") or metric)
    lower_is_better = bool(frozen.get("lower_is_better"))
    frozen_metric_unit = frozen.get("metric_unit")
    baseline_value = frozen["metric_value"]
    fixed_config = str(frozen.get("fixed_config") or "")
    candidate = result.get("candidate")
    candidate_value = _coerce_metric(getattr(candidate, "metric_value", None))

    result["baseline"] = RunResult(
        metric_name=metric, metric_value=baseline_value, valid=True)
    result["metric"] = metric
    result["metric_unit"] = frozen_metric_unit
    result["lower_is_better"] = lower_is_better
    if attestation:
        attestation["metric_evidence"]["metric"] = metric
    if ((reported_metric is not None and str(reported_metric) != metric)
            or (reported_direction is not None
                and (not isinstance(reported_direction, bool)
                     or reported_direction != lower_is_better))):
        reason = "candidate contract violation: explicit objective conflicts with frozen baseline"
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
        result["error"] = reason
        result["report"] = reason
        result["diagnosis"] = "candidate_contract_violation"
        return result
    if initial_commit != frozen.get("repo_commit"):
        reason = "baseline drift: method did not start from frozen repository commit"
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
        result["error"] = reason
        result["report"] = reason
        result["diagnosis"] = "baseline_drift"
        return result
    if final_commit != initial_commit:
        reason = "candidate contract violation: candidate created a commit during method attempt"
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
        result["error"] = reason
        result["report"] = reason
        result["diagnosis"] = "candidate_contract_violation"
        return result
    frozen_contract = frozen.get("benchmark_contract")
    if frozen_contract is not None:
        try:
            normalized_contract = normalize_contract(frozen_contract)
            expected_fingerprint = contract_fingerprint(normalized_contract)
        except BenchmarkContractError:
            normalized_contract = None
            expected_fingerprint = ""
        if (normalized_contract is None
                or frozen.get("contract_fingerprint") != expected_fingerprint):
            reason = "baseline drift: frozen benchmark contract is invalid"
            result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
            result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
            result["error"] = reason
            result["report"] = reason
            result["diagnosis"] = "baseline_drift"
            return result
        manifest = result.get("run_manifest")
        if (not isinstance(manifest, dict)
                or manifest.get("contract_fingerprint") != expected_fingerprint):
            reason = "candidate contract violation: run manifest does not cite frozen contract"
            result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
            result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
            result["error"] = reason
            result["report"] = reason
            result["diagnosis"] = "candidate_contract_violation"
            return result
        result["benchmark_contract"] = normalized_contract
        result["contract_fingerprint"] = expected_fingerprint

    manifest = result.get("run_manifest")
    attempt_id = attestation.get("attempt_id")
    if result.get("_host_measurement_error") and not attestation:
        reason = str(result["_host_measurement_error"])
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
        result["error"] = reason
        result["report"] = reason
        result["diagnosis"] = ("protocol_evidence_violation" if "protocol evidence" in reason
                               else "immutable_evaluator_violation"
                               if "immutable evaluator" in reason
                               else "resource_evidence_violation"
                               if "resource evidence" in reason
                               else "candidate_contract_violation"
                               if ("candidate contract violation" in reason
                                   or "actual controls" in reason
                                   or "undeclared untracked" in reason)
                               else "measurement_spec_invalid" if "specification" in reason
                               else "execution_failure")
        return result
    evidence_valid = bool(
        candidate_value is not None
        and attempt_id
        and _host_attestation_shape_valid(attestation, "method")
        and attestation.get("rc") == 0
        and attestation.get("repo_commit_start") == initial_commit
        and attestation.get("repo_commit_end") == final_commit
        and attestation.get("tracked_diff_hash_start") == attestation.get(
            "tracked_diff_hash_end")
        and attestation.get("tracked_diff_present") is True
        and _coerce_metric((attestation.get("metric_evidence") or {}).get("value")) == candidate_value
    )
    if not evidence_valid:
        reason = ("metric evidence violation: result, variant, and run manifest must cite the "
                  "same attempt and that attempt must succeed")
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
        result["error"] = reason
        result["report"] = reason
        result["diagnosis"] = "metric_evidence_violation"
        return result

    result["config"] = fixed_config
    explicit_validity = result.get("candidate_validity")
    strict_gate = bool(((frozen_baseline or {}).get("repository_contract") or {}).get("sources"))
    validity_ok = explicit_validity is True if strict_gate else explicit_validity is not False
    candidate_valid = bool(validity_ok
                           and getattr(candidate, "valid", False)
                           and candidate_value is not None
                           and math.isfinite(candidate_value))
    result["candidate"] = RunResult(
        metric_name=metric, metric_value=candidate_value, valid=candidate_valid)
    improved = candidate_valid and (
        candidate_value < baseline_value if lower_is_better else candidate_value > baseline_value)
    if improved:
        reason = f"agent improved {metric}: {baseline_value} -> {candidate_value}"
        result["verdict"] = {"verdict": "pass", "improved": True, "reason": reason}
    else:
        if explicit_validity is False:
            reason = "repository evaluator rejected the candidate correctness/quality gate"
        elif strict_gate and explicit_validity is not True:
            reason = "method result omitted the repository correctness/quality gate outcome"
        else:
            reason = (f"no improvement ({metric}: baseline={baseline_value} "
                      f"method={candidate_value})")
        result["verdict"] = {"verdict": "fail", "improved": False, "reason": reason}
        if not candidate_valid and not result.get("diagnosis"):
            result["diagnosis"] = (
                "correctness_gate_failed" if explicit_validity is False
                else "candidate_validity_missing" if strict_gate and explicit_validity is not True
                else "execution_failure")
    result["candidate_admissibility"] = {
        "verdict": "pass" if candidate_valid else "fail",
        "reason": (
            "host validated the frozen benchmark contract, allowed treatment paths, "
            "immutable evaluator, run manifest, and metric attestation"
            if candidate_valid else reason),
        "contract_fingerprint": result.get("contract_fingerprint"),
        "attempt_id": attempt_id,
    }
    result.pop("error", None)
    if not result.get("report") or result.get("report") == old_reason:
        result["report"] = reason
    result.pop("_metric_attempt_id", None)
    result.pop("_metric_evidence_source", None)
    return result


def _baseline_provenance_error(reported_commit: Any, host_commit: Any) -> str | None:
    host_commit = str(host_commit or "")
    if not reported_commit or (
            isinstance(reported_commit, str)
            and re.fullmatch(r"[0-9a-fA-F]{7,40}", reported_commit) is not None
            and host_commit.lower().startswith(reported_commit.lower())):
        return None
    return (
        f"baseline reported repository commit {reported_commit!r} differs from host checkout {host_commit!r}. "
        "config.repo_commit/config.commit identifies the selected outer repository, not a nested dependency. "
        "Verify both checkout identities: if a nested dependency revision was placed in this field, "
        "correct only that metadata and retain its path/revision separately in config.dependencies. "
        "Do not change HEAD or rerun training solely to repair a descriptive field; preserve the original "
        "measured attempt, source evidence and metric. If the actual selected source is wrong, rerun "
        "the selected repository or formally reselect it.")


def _normalized_metric_identity(value: Any) -> str:
    """Normalize presentation-only metric spelling without changing its quantity."""
    text = str(value or "").strip().lower().split("(", 1)[0]
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _normalized_metric_unit(value: Any) -> str:
    """Normalize common unit spellings while keeping dimensions distinct."""
    unit = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    aliases = {
        "millisecond": "ms",
        "milliseconds": "ms",
        "second": "s",
        "seconds": "s",
        "microsecond": "us",
        "microseconds": "us",
        "nanosecond": "ns",
        "nanoseconds": "ns",
    }
    return aliases.get(unit, unit)


def _finalize_relaxed_phase_result(result: dict, execution_phase: str,
                                   frozen_baseline: dict | None, *,
                                   attempt_log: str | None = None,
                                   repo_path: str | None = None,
                                   component_stage: bool = False) -> dict:
    """Finish the two-sandbox flow without the retired fairness/attestation gate."""
    binding_error = None
    measurement_evidence = None
    official_gate_verified = not component_stage
    result.pop("component_failed_attempt_id", None)
    result.pop("component_execution_started", None)
    if execution_phase == "method" and attempt_log and repo_path:
        binding_error = _bind_component_evidence_attempt(
            result, _evidence_records(attempt_log, result, repo_path), repo_path, execution_phase)
    component_attempt_id = str(result.get("_metric_attempt_id") or "").strip() or None
    if attempt_log and repo_path:
        from .runtime_evidence import measured_result, MissingGateEvidence
        records = _attempt_records(attempt_log)
        which = "baseline" if execution_phase == "baseline" else "candidate"
        reported = _coerce_metric(getattr(result.get(which), "metric_value", None))
        if reported is not None and not binding_error:
            try:
                selected = _measurement_attempt(records, component_attempt_id, execution_phase)
                contract = (frozen_baseline or result).get("repository_contract") or {}
                evaluation = (contract.get("execution_contract") or {}).get("evaluation") or {}
                measurement_evidence = measured_result(
                    selected, evaluation, execution_phase,
                    expected_metric_name=evaluation.get("metric_name")
                    or (frozen_baseline or {}).get("metric_name")
                    or result.get("_reported_metric"))
                measurement_evidence["official_gate_verified"] = official_gate_verified
                if component_stage:
                    # A component's train-only scalar may be useful, but inheriting
                    # an official_single policy does not prove the official gate ran.
                    # Only an explicit frozen command and gate can establish that.
                    official_gate_verified = bool(
                        evaluation.get("argv") and evaluation.get("gate_pointers"))
                    measurement_evidence["official_gate_verified"] = official_gate_verified
                    if not official_gate_verified:
                        measurement_evidence["protocol"] = {"kind": "component_local"}
                        measurement_evidence["protocol_fingerprint"] = None
                from .checkpoint_artifacts import validate_checkpoint
                if selected.get("checkpoint_error"):
                    raise ValueError("runtime checkpoint could not be preserved: " + selected["checkpoint_error"])
                checkpoint = selected.get("checkpoint")
                if checkpoint:
                    validate_checkpoint(checkpoint,
                                        artifact_paths=result.get("_reported_artifact_paths", []),
                                        repo_path=repo_path)
                    result["checkpoint"] = checkpoint
                elif result.get("_reported_artifact_paths"):
                    raise ValueError("declared runtime artifacts have no measured checkpoint")
                if not math.isclose(reported, measurement_evidence["value"],
                                    rel_tol=1e-15, abs_tol=0.0):
                    raise ValueError("reported task scalar disagrees with the saved evaluator output")
                # Tolerate only floating-point representation noise, not rounded
                # scores. All downstream comparison uses the exact host-read value.
                result[which].metric_value = measurement_evidence["value"]
                # A sandbox's True cannot override a false host-read gate.
                result["baseline_validity" if which == "baseline" else "candidate_validity"] = (
                    measurement_evidence["valid"] and result.get(
                        "baseline_validity" if which == "baseline" else "candidate_validity") is not False)
                result["measurement_evidence"] = measurement_evidence
                if measurement_evidence.get("samples"):
                    result[which + "_runs"] = measurement_evidence["samples"]
                    result[which + "_std"] = measurement_evidence["method_std"]
                if measurement_evidence.get("execution_sources"):
                    from .source_policy import merge_source_history
                    result["source_history"] = merge_source_history(result.get("source_history"), [
                        {"path": path, **evidence, "attempt_id": selected["attempt_id"],
                         "attribution": "declared execution input; file identity verified, usage not inferred"}
                        for path, evidence in measurement_evidence["execution_sources"].items()])
            except MissingGateEvidence as exc:
                binding_error = str(exc)
                result["diagnosis"] = "candidate_validity_missing"
            except ValueError as exc:
                binding_error = str(exc)
    reported_metric = result.pop("_reported_metric", None)
    reported_metric_unit = result.pop("_reported_metric_unit", None)
    reported_direction = result.pop("_reported_lower_is_better", None)
    evaluation = (((frozen_baseline or result).get("repository_contract") or {}).get("execution_contract") or {}).get("evaluation") or {}
    if ((evaluation.get("metric_name") is not None
         and _normalized_metric_identity(reported_metric) != _normalized_metric_identity(evaluation["metric_name"]))
            or (evaluation.get("metric_unit") is not None
                and _normalized_metric_unit(reported_metric_unit) != _normalized_metric_unit(evaluation["metric_unit"]))
            or ("lower_is_better" in evaluation and reported_direction is not evaluation["lower_is_better"])):
        binding_error = (
            "reported objective identity differs from the explicit frozen evaluation contract: "
            f"expected name={evaluation.get('metric_name')!r}, unit={evaluation.get('metric_unit')!r}, "
            f"lower_is_better={evaluation.get('lower_is_better')!r}; "
            f"reported name={reported_metric!r}, unit={reported_metric_unit!r}, "
            f"lower_is_better={reported_direction!r}")
    if execution_phase == "method":
        result["review_receipts"] = _saved_review_receipts(
            attempt_log, result, repo_path or "", component_attempt_id)
    result.pop("_metric_attempt_id", None)
    result.pop("_metric_evidence_source", None)
    result.pop("_reported_artifact_paths", None)
    for key in ("benchmark_contract", "contract_fingerprint", "host_attestation",
                "run_attestation", "run_manifest", "measurement"):
        result.pop(key, None)
    if execution_phase == "method" and result.get("ablation_report"):
        result["component_attempt_id"] = component_attempt_id
    metric = str(result.get("metric") or (frozen_baseline or {}).get("metric_name") or "metric")
    if execution_phase == "baseline":
        value = _coerce_metric(getattr(result.get("baseline"), "metric_value", None))
        finite = value is not None and math.isfinite(value)
        metric_identity_valid = bool(
            isinstance(reported_metric, str) and reported_metric.strip()
            and isinstance(reported_metric_unit, str) and reported_metric_unit.strip())
        gate_passed = result.get("baseline_validity") is True
        reported_commit = result.pop("_reported_repo_commit", None)
        host_commit = str(result.get("repo_commit") or "")
        provenance_error = _baseline_provenance_error(reported_commit, host_commit)
        provenance_valid = provenance_error is None
        valid = finite and metric_identity_valid and gate_passed and provenance_valid and not binding_error
        result["metric_unit"] = reported_metric_unit
        result["baseline"] = RunResult(metric_name=metric, metric_value=value, valid=valid)
        result["candidate"] = RunResult(metric_name=metric, metric_value=None, valid=False)
        reason = (binding_error if binding_error else f"baseline measured: {metric}={value}" if valid
                  else "baseline phase produced no finite baseline value" if not finite
                  else "baseline correctness/quality gate failed or was not reported"
                  if not gate_passed else
                  provenance_error
                  if not provenance_valid else
                  "baseline metric identity requires a directly measured quantity name and unit")
        result["verdict"] = {
            "verdict": "pass" if valid else "fail", "improved": False, "reason": reason}
        if valid:
            result.pop("error", None)
        else:
            result["error"] = reason
            result["diagnosis"] = ("execution_failure" if not finite
                                   else "correctness_gate_failed" if not gate_passed
                                   else "baseline_provenance_violation" if not provenance_valid
                                   else "metric_identity_invalid")
        result["report"] = result.get("report") or reason
        return result

    frozen = frozen_baseline or {}
    baseline_value = _coerce_metric(frozen.get("metric_value"))
    candidate_value = _coerce_metric(getattr(result.get("candidate"), "metric_value", None))
    lower_is_better = bool(frozen.get("lower_is_better", result.get("lower_is_better", True)))
    metric = str(frozen.get("metric_name") or metric)
    frozen_metric_unit = frozen.get("metric_unit")
    baseline_valid = baseline_value is not None and math.isfinite(baseline_value)
    metric_identity_valid = bool(
        isinstance(reported_metric, str)
        and _normalized_metric_identity(reported_metric) == _normalized_metric_identity(metric)
        and isinstance(reported_metric_unit, str) and reported_metric_unit.strip()
        and (not frozen_metric_unit
             or _normalized_metric_unit(reported_metric_unit)
             == _normalized_metric_unit(frozen_metric_unit)))
    candidate_finite = candidate_value is not None and math.isfinite(candidate_value)
    explicit_validity = result.get("candidate_validity")
    strict_gate = bool((frozen.get("repository_contract") or {}).get("sources"))
    validity_ok = explicit_validity is True if strict_gate else explicit_validity is not False
    attempt_bound = bool(component_attempt_id)
    candidate_valid = bool(candidate_finite and metric_identity_valid
                           and validity_ok and attempt_bound and not binding_error)
    result["metric"] = metric
    result["metric_unit"] = frozen_metric_unit
    result["lower_is_better"] = lower_is_better
    result["baseline"] = RunResult(
        metric_name=metric, metric_value=baseline_value, valid=baseline_valid)
    result["candidate"] = RunResult(
        metric_name=metric, metric_value=candidate_value, valid=candidate_valid)
    control = (measurement_evidence or {}).get("control_metric", baseline_value)
    minimum = (measurement_evidence or {}).get("minimum_improvement", 0)
    improved = bool(baseline_valid and candidate_valid and (
        control - candidate_value if lower_is_better else candidate_value - control) > minimum)
    failed_component_attempt = result.get("component_failed_attempt_id")
    if failed_component_attempt:
        reason = (f"component acceptance check reported unmet: runtime attempt "
                  f"{failed_component_attempt} exited with code 1")
        result["diagnosis"] = "component_expectation_failed"
    elif improved:
        reason = f"agent improved {metric}: {baseline_value} -> {candidate_value}"
    elif not candidate_finite and not binding_error:
        reason = "method phase produced no finite method value"
    elif binding_error:
        reason = binding_error
    elif explicit_validity is False:
        reason = "repository evaluator rejected the candidate correctness/quality gate"
    elif strict_gate and explicit_validity is not True:
        reason = "method result omitted the repository correctness/quality gate outcome"
    elif candidate_finite and not attempt_bound:
        reason = "method result is not bound to a successful current-worktree runtime attempt"
    elif candidate_valid:
        reason = f"no improvement ({metric}: baseline={baseline_value} method={candidate_value})"
    elif candidate_finite:
        reason = "method metric identity does not match the frozen baseline quantity and unit"
    else:
        reason = "method phase produced no finite method value"
    result["verdict"] = {
        "verdict": "pass" if improved else "fail", "improved": improved, "reason": reason}
    candidate_admissible = bool(candidate_valid and attempt_bound and official_gate_verified)
    result["candidate_admissibility"] = {
        "verdict": (
            "pass" if candidate_admissible
            else "fail" if explicit_validity is False
            else "insufficient_evidence"
        ),
        "reason": (
            "official repository gate, frozen metric identity, finite result, and bound "
            "runtime attempt all passed"
            if candidate_admissible else
            "component measurement is bound, but the official evaluator command and gate "
            "were not frozen; final eligibility remains unverified"
            if candidate_valid and attempt_bound and component_stage else reason
        ),
        "attempt_id": component_attempt_id,
    }
    indirect_component_observation = bool(
        component_attempt_id and result.get("ablation_report"))
    if baseline_valid and (candidate_valid or indirect_component_observation):
        result.pop("error", None)
        if indirect_component_observation and result.get("diagnosis") == "execution_failure":
            result.pop("diagnosis", None)
    else:
        result["error"] = reason
        if binding_error and result.get("diagnosis") != "candidate_validity_missing":
            result["diagnosis"] = "attempt_accounting_violation"
        elif not result.get("diagnosis"):
            result["diagnosis"] = (
                "execution_failure" if not candidate_finite
                else "correctness_gate_failed" if explicit_validity is False
                else "candidate_validity_missing" if strict_gate and explicit_validity is not True
                else "attempt_accounting_violation" if candidate_finite and not attempt_bound
                else "metric_identity_mismatch" if candidate_finite
                else "execution_failure")
    result["report"] = result.get("report") or reason
    return result


def _failed_attempt_notes(attempt_log, execution_phase):
    """Preserve execution facts and bounded redacted errors, not commands or secrets."""
    records = list(_attempt_records(attempt_log).values()) if attempt_log else []
    notes = []
    for record in records[-(_variant_cap() or 16):]:
        if (record.get("phase") != execution_phase or not record.get("launched")
                or not record.get("finished")):
            continue
        facts = {key: record.get(key) for key in (
            "attempt_id", "rc", "started_at", "ended_at", "stdout_path", "stderr_path")}
        facts["host_command_cap_s"] = record.get("command_timeout_s") or (
            _agentic_hard_cap() if execution_phase == "baseline" else _per_run_cap())
        if record.get("rc") == 0:
            facts.update({key: record[key] for key in (
                "stdout_sha256", "checkpoint") if record.get(key)})
            prefix = ("Host-recorded successful command (not an accepted result; "
                      "inspect saved output and checkpoint to repair the missing handoff): ")
        else:
            prefix = "Host-recorded execution failure (not a measured score): "
            try:
                with open(record.get("stderr_path") or "", "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    handle.seek(max(0, handle.tell() - 4096))
                    tail = handle.read().decode("utf-8", errors="replace")
                for key, value in os.environ.items():
                    if (key.upper().endswith(("_KEY", "TOKEN", "PASSWORD", "SECRET"))
                            and len(value) >= 8):
                        tail = tail.replace(value, "[REDACTED]")
                if tail.strip():
                    facts["stderr_tail"] = tail[-1000:]
            except OSError:
                pass  # The existing path/exit evidence remains available to the executor.
        notes.append(prefix
                     + json.dumps(facts, ensure_ascii=False, sort_keys=True))
    return notes


def _failed_execution_accounting(attempt_log, repo_path, execution_phase):
    """A missing result does not erase a host-recorded candidate launch."""
    if execution_phase != "method" or not attempt_log:
        return {}
    root = os.path.realpath(repo_path)
    for record in _attempt_records(attempt_log).values():
        if (record.get("launched") is not True or record.get("phase") != execution_phase
                or not record.get("cwd")):
            continue
        cwd = os.path.realpath(str(record["cwd"]))
        if os.path.commonpath([root, cwd]) == root:
            # This marks accounting only, not a passed check or verified score.
            return {"component_execution_started": True}
    return {}


def run_agentic_improve(
    request: str,
    *,
    repo_url: str,
    workroot: str,
    backend: Any = None,
    sandbox_llm: Any = None,
    cuda_device: Any = None,
    budget_s: int | None = 1800,
    metric_hint: str | None = None,
    lower_is_better_hint: bool | None = None,
    emit: Any = None,
    clone_fn: Any = None,
    server_image: str | None = None,
    tool_repo_urls: tuple = (),
    execution_phase: str = "combined",
    frozen_baseline: dict | None = None,
    confirmed_boundary: dict | None = None,
    evaluation_semantics: dict | None = None,
    enforce_fairness: bool = True,
    memory_scope: str | None = None,
    prior_worktree: str | list[str] | dict | None = None,
    component_parent_worktree: str | list[str] | dict | None = None,
    component_stage: bool | None = None,
) -> dict:
    """Run the no-infer agentic improvement and return a ``run_repo_improve``-shaped dict.

    ``clone_fn(repo_url, dest)`` is the test seam (defaults to a real shallow git clone).
    ``sandbox_llm`` MUST be an OpenHands-SDK LLM or None (never our client wrapper)."""
    if execution_phase not in {"combined", "baseline", "method"}:
        raise ValueError(f"unknown execution_phase: {execution_phase}")
    if component_stage is None:
        # Legacy callers supplied component parents; restored baseline controls
        # below do not establish whether this is a component or integration.
        component_stage = component_parent_worktree is not None
    if execution_phase == "method":
        value = _coerce_metric((frozen_baseline or {}).get("metric_value"))
        if value is None or not math.isfinite(value):
            raise ValueError("method phase requires a finite metric_value in frozen_baseline")
        frozen_baseline = dict(frozen_baseline or {})
        frozen_baseline["metric_value"] = value
        metric_hint = metric_hint or frozen_baseline.get("metric_name")
        if lower_is_better_hint is None:
            lower_is_better_hint = frozen_baseline.get("lower_is_better")

    emit = emit or (lambda event, payload: None)
    name = (repo_url.rstrip("/").rsplit("/", 1)[-1] or repo_url) if repo_url else "repo"
    # Use a FRESH repo dir each call: the outer R5 loop re-invokes this with the SAME workroot, and
    # `git clone` into an existing non-empty dir fails (exit 128). A container may also have written
    # root-owned files into a prior clone, so we can't reliably delete it -- pick a new name instead
    # (seen live: micrograd's round-1 baseline 94.0 lost when round-2 re-clone failed to "repo").
    repo_path = os.path.join(workroot, "repo")
    _i = 1
    while os.path.exists(repo_path):
        _i += 1
        repo_path = os.path.join(workroot, f"repo-{_i}")
    result_path = os.path.join(repo_path, _RESULT_BASENAME)

    from autoresearch.agents._reuse.openhands_backend import InfraError, _parse_cards
    if backend is None:
        from autoresearch.agents._reuse.openhands_backend import (
            DEFAULT_SERVER_IMAGE, OpenHandsBackend)
        backend = OpenHandsBackend(server_image=server_image or DEFAULT_SERVER_IMAGE)

    # 1. clone (host; bind-mounted into the sandbox at the SAME path)
    frozen_checkout_error = ""
    try:
        (clone_fn or _clone_repo)(repo_url, repo_path)
        frozen_commit = (str((frozen_baseline or {}).get("repo_commit") or "")
                         if execution_phase == "method" else "")
        if frozen_commit:
            exists = subprocess.run(
                ["git", "-C", repo_path, "cat-file", "-e", f"{frozen_commit}^{{commit}}"],
                capture_output=True, timeout=30)
            if exists.returncode != 0 and re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", frozen_commit):
                # A fresh shallow clone may no longer include the baseline commit after
                # upstream HEAD advances. Fetch that exact object before blaming a candidate
                # that has not even started. Do not substitute HEAD or fetch full history.
                fetched = subprocess.run(
                    ["git", "-c", "http.proxy=", "-c", "https.proxy=", "-C", repo_path,
                     "fetch", "--no-tags", "--depth", "1", "origin", frozen_commit],
                    capture_output=True, text=True, timeout=1200,
                    env=_direct_network_env())
                emit("agentic_frozen_commit_fetch", {
                    "repo_commit": frozen_commit, "exit_code": fetched.returncode})
                exists = subprocess.run(
                    ["git", "-C", repo_path, "cat-file", "-e", f"{frozen_commit}^{{commit}}"],
                    capture_output=True, timeout=30)
            if exists.returncode != 0:
                frozen_checkout_error = (
                    "baseline drift: could not resolve repository commit " + frozen_commit
                    + "; host failed to obtain the frozen commit before candidate execution; "
                    "do not repair this by committing candidate changes")
            else:
                head = subprocess.run(
                    ["git", "-C", repo_path, "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=30)
                if head.returncode != 0 or head.stdout.strip() != frozen_commit:
                    checkout = subprocess.run(
                        ["git", "-C", repo_path, "checkout", "--detach", frozen_commit],
                        capture_output=True, text=True, timeout=300)
                    if checkout.returncode != 0:
                        frozen_checkout_error = "baseline drift: could not checkout frozen commit"
        _make_workspace_writable(repo_path)
        emit("agentic_clone", {"repo_url": repo_url, "dest": repo_path})
    except Exception as exc:  # noqa: BLE001 -- clone failure is an honest hard error
        emit("agentic_error", {"stage": "clone", "error": f"{type(exc).__name__}: {exc}"})
        return {"benchmark": name, "baseline": None, "candidate": None,
                "metric": metric_hint, "lower_is_better": lower_is_better_hint,
                "verdict": {"verdict": "fail", "improved": False,
                            "reason": f"clone failed: {exc}"},
                "report": "", "error": f"clone failed: {exc}"}

    commit = _repo_commit(repo_path)
    if frozen_checkout_error:
        return {"benchmark": name, "baseline": None, "candidate": None,
                "metric": metric_hint, "lower_is_better": lower_is_better_hint,
                "verdict": {"verdict": "fail", "improved": False,
                            "reason": frozen_checkout_error},
                "report": "", "error": frozen_checkout_error, "repo_commit": commit,
                "diagnosis": "baseline_drift"}
    frozen_commit = (str((frozen_baseline or {}).get("repo_commit") or "")
                     if execution_phase == "method" else "")
    if frozen_commit and commit != frozen_commit:
        reason = ("baseline drift: could not resolve repository commit"
                  if commit is None else "baseline drift: repository commit changed")
        return {"benchmark": name, "baseline": None, "candidate": None,
                "metric": metric_hint, "lower_is_better": lower_is_better_hint,
                "verdict": {"verdict": "fail", "improved": False, "reason": reason},
                "report": "", "error": reason, "repo_commit": commit,
                "diagnosis": "baseline_drift"}

    if execution_phase == "method" and (frozen_baseline or {}).get("checkpoint"):
        if not prior_worktree:
            prior_worktree = frozen_baseline["checkpoint"]
        # Legacy/standalone callers may omit the control; compare to the frozen
        # baseline in that case, never to the candidate checkout itself.
        if component_parent_worktree is None or component_parent_worktree in ([], ""):
            component_parent_worktree = [frozen_baseline["checkpoint"]]
    inherited_files = _apply_prior_worktree(repo_path, prior_worktree)
    if inherited_files:
        _make_workspace_writable(repo_path)
        emit("agentic_prior_worktree", {
            "source": prior_worktree, "files": inherited_files, "dest": repo_path,
        })

    # #75 跨轮引导: 上一轮的分析结论持久化在 workroot 级 carry 文件(见 _persist_carry)。读它 ->
    # (a) 预置到本轮 checkout 外的反馈文件(agent 开跑前读取) + (b) 内联进 task
    # prompt(flash 读文件不可靠,内联更稳)。首轮无 carry -> 行为不变。Fail-open。
    carry_direction = ""
    if _carry_enabled():
        try:
            cf = os.path.join(workroot, _carry_basename(memory_scope))
            if os.path.exists(cf):
                with open(cf, encoding="utf-8", errors="ignore") as f:
                    carry_direction = f.read().strip()
            if carry_direction:
                direction_path = _analysis_direction_path(repo_path)
                os.makedirs(os.path.dirname(direction_path), exist_ok=True)
                with open(direction_path, "w", encoding="utf-8") as f:
                    f.write(carry_direction)
                emit("agentic_carry", {"chars": len(carry_direction),
                                       "direction": carry_direction[:160]})
        except Exception:  # noqa: BLE001
            carry_direction = ""

    # #76 多仓库联动: clone 额外"工具仓库"进 workroot/tools/<name>(sandbox 挂载整个 workroot,故容器
    # 内同路径可见),供 agent 评测/增强主仓库(如 SOLAR 算 speed-of-light 百分比)。用与主仓库同一个
    # _clone_repo(mirror 保护 + 非空校验,规避不走代理/大仓库静默截断)。workroot 跨轮稳定 -> tools/ 复用。
    # 单个工具仓库 clone 失败不致命(emit 警告 + 从 prompt 剔除),主改进流程照常。
    tool_repos: list = []   # list[(name, path)]
    for turl in _tool_repo_urls(tool_repo_urls):
        tname = ((turl.rstrip("/").rsplit("/", 1)[-1] or "tool").replace(".git", ""))
        tdest = os.path.join(workroot, "tools", tname)
        try:
            if not (os.path.isdir(tdest) and any(f != ".git" for f in os.listdir(tdest))):
                (clone_fn or _clone_repo)(turl, tdest)
            tool_repos.append((tname, tdest))
            emit("agentic_tool_clone", {"repo_url": turl, "dest": tdest})
        except Exception as exc:  # noqa: BLE001 -- 工具仓库缺失不该毁掉主改进
            emit("agentic_tool_clone_fail",
                 {"repo_url": turl, "error": f"{type(exc).__name__}: {exc}"})

    # Resolve the selected repository task once.  Besides freezing its textual contract for the
    # baseline result, this supplies only explicitly declared container resource limits.
    _task_contract = ((frozen_baseline or {}).get("repository_contract")
                      if execution_phase == "method" else None) or _repository_task_contract(
        repo_path, request, (confirmed_boundary or {}).get("execution_contract"))
    if execution_phase == "method" and _task_contract and (frozen_baseline or {}).get("repo_commit"):
        declared_task = ((_task_contract.get("execution_contract") or {}).get(
            "experiment_provenance") or {}).get("task")
        recovery_request = declared_task or " ".join(
            source["path"] for source in _task_contract.get("sources") or []) or request
        _task_contract = _contract_with_packaging_evidence(
            repo_path, frozen_baseline, _repository_task_contract(
                repo_path, recovery_request, _task_contract.get("execution_contract")))
    validate_execution_contract(_task_contract)
    # Method comparisons inherit the baseline task's resource contract. A later
    # component prompt can mention other task documents without changing its limits.
    _resource_contract = _task_contract
    _task_resources = _repository_resource_limits(_resource_contract)
    _required_cuda_image, _required_cuda_devel = _repository_cuda_devel_image(
        _resource_contract)

    # 2-4. hand the whole job to the in-container agent
    handle = None
    _attempt_authority: _AttemptAuthority | None = None
    # Within-session analysis agent (老师 2026-08-07): a SEPARATE thread turns the agent's tried-
    # variants into a directed CONCLUSION so its in-session iteration is aimed, not blind. Advisory +
    # fail-open; started before the run, stopped in finally. The bind mount makes repo_path/analysis/*
    # shared between this host thread and the in-container agent.
    import threading
    _stop = threading.Event()
    _sidecar = threading.Thread(
        target=_analysis_sidecar,
        args=(repo_path, request, metric_hint, lower_is_better_hint, _stop, emit, memory_scope),
        daemon=True)

    _attempt_log: str | None = None
    _untracked_before: dict[str, dict] = {}

    def finalize(result: dict) -> dict:
        if execution_phase == "method":
            result["attempt_review_evidence"] = _attempt_review_evidence(_attempt_log)
        if enforce_fairness and execution_phase in {"baseline", "method"} and handle is not None:
            _host_remeasure(
                backend, handle, repo_path, result, execution_phase, frozen_baseline,
                confirmed_boundary, attempt_log=_attempt_log,
                resource_evidence=_resource_evidence(cuda_device),
                untracked_before=_untracked_before)
        result["diff"] = _capture_repo_diff(repo_path)
        if execution_phase == "method" and (
                (frozen_baseline or {}).get("checkpoint")
                or (frozen_baseline or {}).get("repo_commit")):
            try:
                # A baseline execution may leave generated source-looking files in its
                # frozen checkpoint. Review only changes made after that checkpoint.
                baseline_checkpoint = frozen_baseline.get("checkpoint")
                if not baseline_checkpoint and _repo_commit(repo_path) != frozen_baseline["repo_commit"]:
                    raise ValueError("repository HEAD no longer matches the frozen baseline commit")
                result["baseline_relative_diff"] = _capture_component_diff(
                    repo_path, [baseline_checkpoint] if baseline_checkpoint else [])
            except Exception as exc:
                result["baseline_diff_error"] = str(exc)
                emit("baseline_diff_unavailable", {"reason": str(exc)})
        # Never accept a sandbox's self-reported parent delta as host evidence.
        result["parent_relative_diff"] = None
        result["parent_diff_error"] = None
        # [] is a root DAG component: its parent is the repository baseline, not
        # "no component review". Preserve full source delta for it just as for children.
        if execution_phase == "method" and (
                component_parent_worktree or component_parent_worktree == []):
            try:
                result["parent_relative_diff"] = _capture_component_diff(
                    repo_path, component_parent_worktree)
            except Exception as exc:
                result["parent_diff_error"] = str(exc)
                emit("component_diff_unavailable", {"reason": str(exc)})
        result["repo_path"] = repo_path
        result["repository_contract"] = _contract_with_packaging_evidence(
            repo_path, frozen_baseline, _task_contract)
        if ((result["repository_contract"] or {}).get("execution_contract") or {}).get("record_runtime_identity") is True:
            from autoresearch.spec_store.identity import runtime_identity
            result["runtime_provenance"] = runtime_identity()
        result["source_history"] = (frozen_baseline or {}).get("source_history", [])
        from .source_policy import source_findings, merge_source_history
        result["source_history"] = merge_source_history(result["source_history"], source_findings(
            result, result["repository_contract"]))
        if execution_phase == "baseline":
            result["repository_contract"] = _task_contract
            result["source_history"] = merge_source_history(result.get("source_history"), [
                {"path": s["path"], "sha256": s["sha256"], "allowed": True, "role": "task_contract"}
                for s in (_task_contract or {}).get("sources", [])])
        final_commit = _repo_commit(repo_path)
        tracked_state = _tracked_repo_state(repo_path)
        result["repo_commit"] = final_commit
        if execution_phase in {"baseline", "method"} and not enforce_fairness:
            result = _finalize_relaxed_phase_result(
                result, execution_phase, frozen_baseline,
                attempt_log=_attempt_log, repo_path=repo_path,
                component_stage=execution_phase == "method" and component_stage)
            if result.get("checkpoint"):
                from .checkpoint_artifacts import external_source_diff
                result["diff"] = _capture_repo_diff(result["checkpoint"], host_checkpoint=True) + external_source_diff(
                    result["checkpoint"], (frozen_baseline or {}).get("checkpoint"))
                if execution_phase == "method" and component_parent_worktree is not None:
                    result["parent_relative_diff"] = _capture_component_diff(
                        result["checkpoint"], component_parent_worktree) + external_source_diff(
                            result["checkpoint"], component_parent_worktree)
                    result["parent_diff_error"] = None
        else:
            result = _finalize_phase_result(
                result, execution_phase, frozen_baseline, attempt_log=_attempt_log,
                confirmed_boundary=confirmed_boundary, initial_commit=commit,
                final_commit=final_commit, tracked_state=tracked_state)
        if (execution_phase == "method" and not component_stage
                and component_parent_worktree == [] and prior_worktree):
            # Integration still compares against the frozen baseline. Independently
            # expose edits made to the accepted source selected as its starting point.
            result["starting_point_diff"] = None
            result["starting_point_diff_error"] = None
            try:
                from .checkpoint_artifacts import external_source_diff
                selected_source = result.get("checkpoint") or repo_path
                result["starting_point_diff"] = _capture_component_diff(
                    selected_source, prior_worktree) + external_source_diff(
                        selected_source, prior_worktree)
            except Exception as exc:
                result["starting_point_diff_error"] = str(exc)
                emit("integration_starting_diff_unavailable", {"reason": str(exc)})
        return result

    try:
        # Docker receives the requested host nodes via ``--device /dev/nvidiaN``, but CUDA
        # enumerates that reduced visible set densely inside the container.  Keep host ordinals
        # for passthrough/resource evidence and expose only container-local logical ordinals here.
        _claimed_cards = _parse_cards(cuda_device) if cuda_device is not None else []
        _cuda_root = None
        _cuda_release_version = None
        if _claimed_cards and _required_cuda_devel:
            _cuda_root, _cuda_release_version = _matching_cuda_toolkit_root(
                _required_cuda_devel)
            if _cuda_root is None:
                download_authorization = ((confirmed_boundary or {}).get(
                    "download_authorization") or {})
                if not isinstance(download_authorization, dict):
                    download_authorization = {}
                if (download_authorization.get("approved") is True
                        and _required_cuda_image is not None):
                    try:
                        _cuda_root, _cuda_release_version = _provision_cuda_toolkit(
                            _required_cuda_image, _required_cuda_devel, workroot, emit)
                    except Exception as exc:
                        raise InfraError(
                            "repository-declared CUDA environment provisioning failed before "
                            f"baseline: {type(exc).__name__}: {exc}") from exc
                else:
                    required_release = ".".join(_required_cuda_devel.split(".")[:2])
                    observed = (f"; configured nvcc reports {_cuda_release_version}"
                                if _cuda_release_version else "")
                    raise InfraError(
                        f"repository Dockerfile requires CUDA {required_release} devel, but no "
                        "matching host toolkit is available and initial download authorization "
                        "was not granted. Approve repository-declared dependency downloads or "
                        "set AUTORESEARCH_SANDBOX_CUDA_ROOT; do not install a different CUDA "
                        f"stack inside the experiment sandbox{observed}")
        _gpu_env = ({"CUDA_VISIBLE_DEVICES": ",".join(
                         str(index) for index in range(len(_claimed_cards))),
                     "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                    if _claimed_cards else {})
        _sandbox_env = dict(_gpu_env)
        if _task_resources.get("cpus"):
            _sandbox_env["AUTORESEARCH_TASK_CPUS"] = str(_task_resources["cpus"])
        if _task_resources.get("memory_mb"):
            _sandbox_env["AUTORESEARCH_TASK_MEMORY_MB"] = str(
                _task_resources["memory_mb"])
        # DGX downloads are direct.  Explicitly neutralise every common proxy spelling so neither
        # the SDK nor repo-owned installers can inherit the L20's metered conversation proxy.
        _sandbox_env.update({
            "NO_PROXY": "*", "no_proxy": "*",
            "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "",
            "http_proxy": "", "https_proxy": "", "all_proxy": "",
            # Missing/private repositories and interactive installers must fail promptly so the
            # agent can choose another compatible source instead of waiting forever for stdin.
            "GIT_TERMINAL_PROMPT": "0", "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        # A shared venv is only reusable when the interpreter it links to is shared too. Prefer an
        # existing host runtime cache across sessions; fall back to the session workroot when the
        # configured cache is absent. Both paths are bind-mounted below.
        _shared_python_root = os.environ.get(
            "AUTORESEARCH_SANDBOX_PYTHON_ROOT", ""
        ).strip()
        _python_install_dir = (
            _shared_python_root if _shared_python_root and os.path.isdir(_shared_python_root)
            else os.path.join(workroot, ".uv-python")
        )
        _persistent_runtime_dirs = {
            "UV_PYTHON_INSTALL_DIR": _python_install_dir,
            "XDG_CACHE_HOME": os.path.join(workroot, ".cache"),
            # NLTK does not use XDG_CACHE_HOME; its default home data disappears
            # with each container even when the Python environment is shared.
            "NLTK_DATA": os.path.join(workroot, ".cache", "nltk_data"),
        }
        for _runtime_dir in _persistent_runtime_dirs.values():
            try:
                os.makedirs(_runtime_dir, exist_ok=True)
                os.chmod(_runtime_dir, 0o777)
            except Exception:  # noqa: BLE001 -- cache persistence is an optimization
                pass
        _sandbox_env.update(_persistent_runtime_dirs)
        # Host-create a writable mounted destination for auxiliary evaluators/results. The
        # session root itself can be unwritable to the sandbox UID; /tmp disappears at close.
        artifact_dir = os.path.join(workroot, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        if stat.S_IMODE(os.stat(artifact_dir).st_mode) != 0o777:
            os.chmod(artifact_dir, 0o777)
        _sandbox_env["AUTORESEARCH_ARTIFACT_DIR"] = artifact_dir
        _extra_mounts: list[Any] = []
        if _cuda_root:
            _extra_mounts.append((_cuda_root, _cuda_root + ":ro"))
        if _python_install_dir == _shared_python_root:
            _extra_mounts.append(_shared_python_root)
        # Persist downloads across sandbox rounds. By default the cache belongs to the current
        # session; deployments may explicitly point AUTORESEARCH_SANDBOX_PKG_CACHE_DIR at a
        # machine-wide cache. AUTORESEARCH_SANDBOX_PKG_CACHE=0 disables it.
        if os.environ.get("AUTORESEARCH_SANDBOX_PKG_CACHE", "1") == "1":
            _cache_dir = os.environ.get("AUTORESEARCH_SANDBOX_PKG_CACHE_DIR",
                                        os.path.join(workroot, ".pkg-cache"))
            try:
                os.makedirs(_cache_dir, exist_ok=True)
                os.chmod(_cache_dir, 0o777)   # 容器 user 可写
            except Exception:  # noqa: BLE001 -- 缓存是优化,建不了就跳过
                pass
            if not os.path.isdir(_cache_dir):
                _cache_dir = os.path.join(workroot, ".pkg-cache")
                os.makedirs(_cache_dir, exist_ok=True)
                if stat.S_IMODE(os.stat(_cache_dir).st_mode) != 0o777:
                    os.chmod(_cache_dir, 0o777)
            _sandbox_env.update({
                "UV_CACHE_DIR": _cache_dir,
                "PIP_CACHE_DIR": _cache_dir,
                # Keep the session-shared venv valid after a sandbox exits.  Agents may redirect
                # UV_CACHE_DIR to container-local /tmp while repairing one dependency; symlink
                # mode would leave broken package files behind.  Hardlinks remain valid and uv
                # falls back to a copy when cache and environment are on different filesystems.
                "UV_LINK_MODE": "hardlink",
            })
            _extra_mounts.append(_cache_dir)
        # Reuse already-present model/dataset caches.  These mounts never initiate a download;
        # they only expose local DGX data to the sandbox.  The official HF endpoint remains the
        # default and all traffic is direct if a cache miss genuinely requires access.
        if os.environ.get("AUTORESEARCH_SANDBOX_HF_CACHE", "1") == "1":
            _hf_home = os.environ.get(
                "AUTORESEARCH_SANDBOX_HF_HOME",
                os.path.join(workroot, ".cache", "huggingface"))
            try:
                os.makedirs(_hf_home, exist_ok=True)
                os.chmod(_hf_home, 0o777)
            except Exception:  # noqa: BLE001 -- cache reuse is optional
                pass
            if os.path.isdir(_hf_home):
                _sandbox_env.update({"HF_HOME": _hf_home, "HF_HUB_DOWNLOAD_TIMEOUT": "120"})
                _hf_endpoint = os.environ.get("AUTORESEARCH_HF_ENDPOINT", "").strip()
                if _hf_endpoint:
                    _sandbox_env["HF_ENDPOINT"] = _hf_endpoint
                _extra_mounts.append(_hf_home)
        _data_root = os.environ.get("AUTORESEARCH_SANDBOX_DATA_ROOT", "").strip()
        if _data_root and os.path.isdir(_data_root):
            _sandbox_env["AUTORESEARCH_DATA_ROOT"] = _data_root
            _extra_mounts.append(_data_root)
        # An explicitly configured local wheelhouse is useful without any network mirror.
        # Mount and advertise it independently; a missing path is ignored because uv treats a
        # nonexistent find-links directory as a hard error.
        _wheelhouse = os.environ.get("AR_TORCH_WHEELS", "").strip()
        _local_find_links = ""
        if _wheelhouse and os.path.isdir(_wheelhouse):
            _local_find_links = _wheelhouse
            _extra_mounts.append(_wheelhouse)
            _sandbox_env.update({
                "PIP_FIND_LINKS": _local_find_links,
                "UV_FIND_LINKS": _local_find_links,
            })
        # Optional deployment-specific mirror. Public/default installs retain the package
        # manager's official indexes and do not add trusted hosts or unsafe index selection.
        if os.environ.get("AUTORESEARCH_SANDBOX_PIP_MIRROR", "0") == "1":
            _ali_pypi = "https://mirrors.aliyun.com/pypi/simple/"
            _ali_torch = "https://mirrors.aliyun.com/pytorch-wheels/cu130/"
            _find_links = " ".join(filter(None, (_local_find_links, _ali_torch)))
            _sandbox_env.update({
                "PIP_EXTRA_INDEX_URL": _ali_pypi,
                "PIP_FIND_LINKS": _find_links,
                "PIP_TRUSTED_HOST": "mirrors.aliyun.com pypi.org files.pythonhosted.org download.pytorch.org",
                "UV_INDEX": _ali_pypi, "UV_INDEX_STRATEGY": "unsafe-best-match",
                "UV_FIND_LINKS": _find_links,
            })
        # #75 提速: 同一 session 跨轮共享一个 workroot 级 .venv,避免每轮重装 torch(~30-40min/轮)。
        # workroot 由 session_id 派生稳定(见 scripts/dgx/agentic_run.py),故 round2+ 的 `uv sync` 命中
        # 上一轮已装好的共享 venv -> 秒过。UV_PROJECT_ENVIRONMENT 让 uv 写/读该绝对路径;再补一个
        # repo/.venv -> ../.venv-shared 符号链接,让 agent 手动引用(.venv/bin/python / activate)也解析到
        # 同一处。container 以 root 建/复用它;host 只建链接。controlled comparison 不受影响(代码仍每轮
        # fresh clone,只共享依赖)。AUTORESEARCH_SHARE_VENV=0 关。fail-open。
        if os.environ.get("AUTORESEARCH_SHARE_VENV", "1") == "1":
            _venv = os.path.join(workroot, ".venv-shared")
            # workroot 由宿主进程创建，通常不可由非 root 的沙箱用户新增同级目录。
            # 预先创建共享 venv 根目录并开放写权限，首次运行才能执行 `uv venv`；
            # 后续轮次继续复用其中内容。
            try:
                os.makedirs(_venv, exist_ok=True)
                os.chmod(_venv, 0o777)
            except OSError:
                pass
            _sandbox_env["UV_PROJECT_ENVIRONMENT"] = _venv
            # 会议 2026-08-11 修 setup thrash: agent 常用【系统 python】跑 `python -c import torch` ->
            # 系统 python 无 torch(torch 在 .venv-shared, 训练走 uv run)-> agent 误判"没装"-> 反复
            # `pip install torch`(冗余, 烧 setup 预算, GPU 空转)。根治: 把共享 venv 的 bin 前置进 PATH,
            # 让裸 `python`/`pip` 一旦 venv 建好就解析到它 -> `import torch` 直接成功、不再重装。venv 未建
            # 好时 PATH 里该项被安全忽略、fall through 到系统(uv 仍在 PATH, uv sync 照跑)。不设 VIRTUAL_ENV
            # (指向尚未建好的目录会让裸 pip 报错)。
            _sandbox_env["PATH"] = (os.path.join(_venv, "bin") +
                                    ":/home/openhands/.local/bin:/usr/local/bin:/usr/bin:/bin"
                                    ":/usr/local/games:/usr/games")
            try:
                _lnk = os.path.join(repo_path, ".venv")
                if not os.path.lexists(_lnk):
                    os.symlink("../.venv-shared", _lnk)   # 相对: repo(-N)/.venv -> workroot/.venv-shared
            except Exception:  # noqa: BLE001 -- 建链接失败就退回本轮自建 .venv(不比现状差)
                pass
        if _cuda_root:
            _sandbox_env.update({
                "CUDA_HOME": _cuda_root,
                "CUDA_PATH": _cuda_root,
                "CUDACXX": os.path.join(_cuda_root, "bin", "nvcc"),
                "PATH": os.path.join(_cuda_root, "bin") + ":" + _sandbox_env.get(
                    "PATH", "/home/openhands/.local/bin:/usr/local/bin:/usr/bin:/bin"),
                "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
                    os.path.join(_cuda_root, "lib64"),
                    os.path.join(_cuda_root, "targets", "x86_64-linux", "lib"),
                    _sandbox_env.get("LD_LIBRARY_PATH"),
                ))),
            })
        # Execution provenance is required in the normal two-phase flow too;
        # it must not depend on the retired fairness remeasurement switch.
        structured_phase = execution_phase in {"baseline", "method"}
        _guard_bin, _attempt_log = _install_attempt_guard(
            workroot, repo_path, result_path, authoritative=structured_phase)
        guard_control = os.path.dirname(_guard_bin)
        last_attempt_path = os.path.join(guard_control, "last_attempt_id")
        if structured_phase:
            _sandbox_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                os.path.join(guard_control, "python"), _sandbox_env.get("PYTHONPATH"))))
        attempt_socket_path = None
        if structured_phase:
            attempt_socket_dir = tempfile.mkdtemp(prefix="ar-attempt-sock-")
            attempt_socket_path = os.path.join(attempt_socket_dir, "authority.sock")
            _extra_mounts.append(attempt_socket_dir)
        _sandbox_env.update({
            "AUTORESEARCH_ATTEMPT_RESULT": result_path,
            "AUTORESEARCH_LAST_ATTEMPT": last_attempt_path,
            "AUTORESEARCH_REPO_PATH": repo_path,
            "AUTORESEARCH_AGENTIC_VARIANT_CAP": str(_variant_cap()),
            "AUTORESEARCH_EXECUTION_PHASE": execution_phase,
            "PATH": _guard_bin + ":" + _sandbox_env.get(
                "PATH", "/home/openhands/.local/bin:/usr/local/bin:/usr/bin:/bin"),
        })
        if attempt_socket_path is not None:
            _sandbox_env["AUTORESEARCH_ATTEMPT_SOCKET"] = attempt_socket_path
        else:
            _sandbox_env["AUTORESEARCH_ATTEMPT_LOG"] = _attempt_log
        # Recovery may expose the same session through a symlink. The host attempt
        # broker uses canonical paths, so expose that SAME directory in the sandbox
        # too; mounting only the alias leaves broker cwd/artifact paths unavailable.
        canonical_workroot = os.path.realpath(workroot)
        if canonical_workroot != os.path.abspath(workroot):
            _extra_mounts.append(canonical_workroot)
        # The controller holds this inode throughout execution. Sandbox DEBUG must
        # not delete it as a "stale" lock or change its ownership through either alias.
        for root in dict.fromkeys((os.path.abspath(workroot), canonical_workroot)):
            lock_path = os.path.join(root, ".agentic_run.lock")
            if os.path.isfile(lock_path):
                _extra_mounts.append((os.path.realpath(lock_path), lock_path + ":ro"))
        handle = backend.open_sandbox(cuda_device=cuda_device, mounts=[workroot, *_extra_mounts],
                                      env=(_sandbox_env or None))
        if _cuda_root:
            preflight = backend.execute_command(
                handle, "nvcc --version", cwd=repo_path, timeout=30)
            output = (preflight.stdout or "") + (preflight.stderr or "")
            observed = re.search(r"\brelease\s+([0-9]+\.[0-9]+)\b", output)
            required_release = ".".join(_required_cuda_devel.split(".")[:2])
            if preflight.exit_code != 0 or observed is None or observed.group(1) != required_release:
                raise InfraError(
                    "mounted CUDA toolkit failed sandbox preflight: required release "
                    f"{required_release}, exit={preflight.exit_code}, output={output[:300]!r}")
            emit("agentic_cuda_toolkit", {
                "root": _cuda_root,
                "required": _required_cuda_devel,
                "observed_release": observed.group(1),
                "read_only": True,
            })
        from .checkpoint_artifacts import restore_external
        if execution_phase == "method":
            restore_external(prior_worktree, backend, handle, repo_path)
        if attempt_socket_path is not None:
            _attempt_authority = _AttemptAuthority(
                backend, handle, repo_path, execution_phase, _attempt_log,
                attempt_socket_path, last_attempt_path)
            _attempt_authority.repository_contract = _resource_contract
            _attempt_authority.artifact_paths = (((frozen_baseline or {}).get("repository_contract")
                or _task_contract or {}).get("execution_contract") or {}).get("artifact_paths", [])
            _attempt_authority.execution_protocol = (((frozen_baseline or {}).get("repository_contract")
                or _task_contract or {}).get("execution_contract") or {}).get("evaluation", {})
            if (execution_phase == "method" and component_parent_worktree is not None
                    and os.path.exists(os.path.join(repo_path, ".git"))):
                control_path = os.path.join(workroot, "control-" + uuid.uuid4().hex)
                subprocess.run(["git", "clone", "--shared", "--quiet", repo_path, control_path],
                               capture_output=True, check=True, timeout=600, umask=0)
                _apply_prior_worktree(control_path, component_parent_worktree)
                # Restored source may live under an ignored runtime directory.
                # Register only inherited code, not ignored data or model artifacts.
                dependency_changes = _dependency_worktree_changes(
                    control_path, component_parent_worktree)
                _make_workspace_paths_writable(control_path, dependency_changes)
                inherited_sources = [p for p in dependency_changes
                    if os.path.isfile(os.path.join(control_path, p)) and (
                        os.path.splitext(p)[1].casefold() in _CARRY_SOURCE_SUFFIXES
                        or os.path.basename(p) in _CARRY_SOURCE_NAMES)]
                if inherited_sources:
                    subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.fileMode=false",
                                    "-C", control_path, "add", "--force", "--",
                                    *sorted(inherited_sources)],
                                    capture_output=True, check=True,
                                    timeout=_checkpoint_handoff_timeout())
                # Restored additions start untracked. Register the fresh control's
                # inherited files before freezing its diff fingerprint, otherwise
                # an agent's content-preserving git add looks like parent drift. Scope
                # staging to the resolved dependency delta: repository-wide `git add
                # --all` can traverse large generated datasets that are not parent code.
                _attempt_authority.control_path = control_path
                _attempt_authority.control_checkpoints = component_parent_worktree
                baseline_checkpoint = (frozen_baseline or {}).get("checkpoint")
                parent_checkpoints = (component_parent_worktree
                                      if isinstance(component_parent_worktree, list)
                                      else [component_parent_worktree])
                if (baseline_checkpoint and len(parent_checkpoints) == 1
                        and os.path.realpath(str(parent_checkpoints[0]))
                        == os.path.realpath(str(baseline_checkpoint))):
                    baseline_evidence = (frozen_baseline or {}).get("measurement_evidence") or {}
                    baseline_gates = baseline_evidence.get("gates") or {}
                    _attempt_authority.control_is_frozen_baseline = True
                    if baseline_gates and all(type(value) is bool for value in baseline_gates.values()):
                        _attempt_authority.frozen_baseline_outcome = {
                            "feasible": all(baseline_gates.values())}
                _attempt_authority.dependency_parent_state = {
                    "repo_commit": _repo_commit(control_path),
                    "tracked_diff_hash_start": _tracked_repo_state(control_path).get("tracked_diff_hash")}
                from .runtime_evidence import execution_file_evidence
                if _attempt_authority.execution_protocol.get("source_files") or _attempt_authority.execution_protocol.get("immutable_files"):
                    _attempt_authority.dependency_parent_state["execution_files_start"] = execution_file_evidence(
                        control_path, _attempt_authority.execution_protocol)
            _attempt_authority.start()
            os.chmod(os.path.dirname(attempt_socket_path), 0o555)
        if execution_phase == "baseline":
            task = _baseline_task_prompt(
                request, repo_path, result_path, metric_hint, lower_is_better_hint,
                confirmed_boundary=confirmed_boundary,
                evaluation_semantics=evaluation_semantics)
        elif execution_phase == "method":
            task = _method_task_prompt(
                request, repo_path, result_path, budget_s, frozen_baseline,
                evaluation_semantics=evaluation_semantics)
        else:
            task = _task_prompt(request, repo_path, result_path, budget_s,
                                metric_hint, lower_is_better_hint, repeats=_winner_repeats(),
                                carry_direction=carry_direction, tool_repos=tuple(tool_repos),
                                repository_contract=_resource_contract)
        task += ("\nHOST CONTROL BOUNDARY: The session directory, .agentic_run.lock, "
                 "component ledgers and .sandbox-control belong to the running controller. "
                 "Do not remove, rename, chown or chmod these controls or their ancestors, "
                 "including via sudo or recursive commands. A lock is not stale merely "
                 "because a previous attempt reported a permission error. Inspect current "
                 "conditions without disabling startup guards or setting host-owned "
                 "AUTORESEARCH_ATTEMPT_ACTIVE/AUTORESEARCH_HOST_NATIVE_EXECUTION markers. "
                 "If controller permissions still block execution, report the "
                 "infrastructure failure without changing controller files. Normal repository "
                 "edits and setup in the provided writable worktree, artifact and cache "
                 "locations remain allowed.")
        task += ("\nSOURCE BOUNDARY (applies to all inspection, planning, evaluation and carry): "
                 "Do not read prohibited reference/answer material. List every actually used "
                 "reference in reference_sources as a JSON array of path/URL strings, not objects "
                 "(keep usage descriptions in a separate field), and preserve prior exposure even after reverting. "
                 + json.dumps((((frozen_baseline or {}).get("repository_contract") or _task_contract or {})
                                .get("execution_contract") or {}).get("sources", {}), ensure_ascii=False))
        from .execution_policy import efficient_execution, EFFICIENT_EXPERIMENT
        if efficient_execution():
            task += "\n" + EFFICIENT_EXPERIMENT
        task += ("\nRUNTIME EVIDENCE: ar-attempt must emit one JSON result on stdout, from the actual "
                 "evaluator output. ar-attempt records only its child command's stdout, not later "
                 "shell commands. If the evaluator writes a file, use one wrapper that runs the "
                 "unchanged evaluator, checks its exit status and actual gates, reads the freshly "
                 "written output and prints JSON; launch the entire wrapper, for example "
                 "`ar-attempt bash run_and_export.sh`. Exporting JSON after ar-attempt returns "
                 "does not attach it to that attempt. Do not merely echo a stored/claimed score. "
                 "When the task restricts submitted or editable files, keep measurement-only "
                 "wrappers outside the candidate repository working tree (for example in "
                 "$AUTORESEARCH_ARTIFACT_DIR, which is available before launch and survives "
                 "container cleanup) and invoke them by absolute path. "
                 "They must not appear in the candidate diff or modify protected evaluator files. "
                 "Finish every candidate source edit, repository-local measurement wrapper, and "
                 "git index change before launching ar-attempt. After ar-attempt starts, do not "
                 "edit or stage repository files until its result has been written and handed off; "
                 "the host binds the measurement to that exact source/index fingerprint. "
                 "Smoke-test the result serializer before expensive training, without treating "
                 "synthetic smoke-test values as measurements. Use json.dumps on Python values "
                 "or json.loads to parse JSON; JSON true/false/null are not Python literals. "
                 "For a scoring or component-acceptance attempt, launch the required computation, "
                 "checks and result export for that frozen phase/component in one ar-attempt command. "
                 "That command must itself atomically write ar_result.json with its own "
                 "AUTORESEARCH_ATTEMPT_ID and emit the measured stdout JSON before exiting. "
                 "Do not defer required evaluation or result-file creation to a later LLM turn; "
                 "the provider may fail while background computation is still running. "
                 "Save resumable progress periodically during long experiments, in declared "
                 "runtime artifact paths, rather than only after the final loop iteration. "
                 "Include loop position, model/search state, optimizer and RNG state as applicable; "
                 "reserve time for the required evaluation and result export within the allowed budget. "
                 "After a timeout inspect the saved progress before rerunning; resume only when "
                 "the frozen protocol permits it, preserving cumulative budget and provenance. "
                 "Partial progress is not a passing final measurement; never relax the evaluator "
                 "or task requirements to finish within a timeout. "
                 "Never nest ar-attempt: measure parent/control and candidate with separate "
                 "top-level ar-attempt commands; wrappers must not call ar-attempt internally. "
                 "Default scalar keys are baseline/method, and gate keys "
                 "baseline_valid/candidate_valid. task.toml [autoresearch.evaluation] can declare "
                 "metric_pointer, gate_pointers and a measurement protocol. Every declared gate "
                 "is mandatory. Emit component scalars under measurements.<component_id>, and "
                 "cite their JSON pointers; do not copy expected thresholds into actual values. "
                 "For repeated protocols preserve all samples and matched control samples under "
                 "the unchanged evaluator configuration; the host checks the declared aggregate. "
                 "Preserve every required runtime file declared in autoresearch.artifact_paths. "
                 "Undeclared external weights cannot support a recoverable best checkpoint.")
        if _attempt_authority and _attempt_authority.control_path:
            task += ("\nEXACT PARENT CONTROL checkout: " + _attempt_authority.control_path +
                     ". First reuse an existing frozen parent measurement when its protocol and "
                     "evidence pointer match. To fill a genuinely missing parent metric, run the same "
                     "evaluator there through "
                     "ar-attempt and cite its attempt ID and JSON pointer as parent_evidence. "
                     "If that evaluator is reserved for isolated final evaluation, do not run it "
                     "in this component sandbox; request plan or evidence-boundary revision. "
                     "This is the control execution worktree, not merely a code reference. "
                     "Change directory to the control checkout before launching ar-attempt "
                     "(cd <control checkout> && ar-attempt bash <evaluation wrapper>). "
                     "A cd inside the child wrapper does not change the host-recorded launch directory "
                     "or source fingerprint; it cannot bind a candidate-launched attempt as control evidence. "
                     "If dependencies or ignored upstream files are absent, prepare the same pinned "
                     "dependencies there without overwriting inherited source; do not substitute "
                     "candidate code for the parent. "
                     "Do not rewrite a working candidate to repair a missing control measurement.")
        if execution_phase in {"baseline", "method"}:
            _untracked_before = _untracked_file_inventory(repo_path)
        _sidecar.start()
        emit("agentic_run", {"stage": "running", "budget_s": budget_s})
        # #46: ALWAYS give run_agent_task a FINITE timeout so it uses the reliable daemon-thread backstop
        # (timeout=None runs inline and a hung/dead-connection poll never returns -> orchestrator perma-
        # hang). budget_s is the soft prompt budget; the hard wall-clock cap is separate and always set.
        hard_timeout = (budget_s + _agentic_finalize_reserve()
                        if budget_s is not None else _agentic_hard_cap())
        if budget_s is None and os.environ.get("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT") == "1":
            hard_timeout = None
        run_kwargs = {"llm": sandbox_llm, "timeout": hard_timeout}
        completion_check = lambda: _result_completion_request(
            result_path, execution_phase=execution_phase, attempt_log=_attempt_log,
            evaluation=(((frozen_baseline or {}).get("repository_contract")
                         or _task_contract or {}).get("execution_contract") or {}).get("evaluation"),
            attempt_authority=_attempt_authority)
        if "completion_check" in inspect.signature(backend.run_agent_task).parameters:
            run_kwargs["completion_check"] = completion_check
        # Prompt discipline is advisory; enforce the cap from the host by pausing the conversation as
        # soon as the capped variants and their result handoff are complete.  Duck-typed backends that do
        # not expose this optional control retain their prior behavior.
        if (execution_phase in {"method", "combined"}
                and "stop_predicate" in inspect.signature(
                    backend.run_agent_task).parameters):
            run_kwargs["stop_predicate"] = lambda: (
                _attempt_cap_ready_to_pause(_attempt_log, _variant_cap(), result_path=result_path)
                and completion_check() is None)
        try:
            res = backend.run_agent_task(handle, task, **run_kwargs)
        except Exception as exc:
            from autoresearch.llm.errors import is_resumable_provider_error
            if _attempt_authority is not None and is_resumable_provider_error(exc):
                # The agent's provider can fail while its original, independently
                # bounded experiment is still running. Drain that one launch and
                # save its receipt before cleanup can stop the owning sandbox.
                emit("agentic_run", {"stage": "provider_pause_waiting_for_attempt"})
                _attempt_authority.stop(wait_for_inflight=True)
                if not os.path.lexists(result_path):
                    # A completed wrapper may emit the full result without writing
                    # its handoff file. Preserve only this host-owned attempt, then
                    # let the normal phase/provenance validators check the result.
                    try:
                        from .runtime_evidence import measured_result, read_output
                        with open(last_attempt_path, encoding="utf-8") as saved:
                            attempt_id = saved.read().strip()
                        record = _attempt_records(_attempt_log).get(attempt_id) or {}
                        cwd = os.path.realpath(str(record.get("cwd") or ""))
                        if (record.get("phase") != execution_phase
                                or record.get("execution_uncertain")
                                or os.path.commonpath([cwd, os.path.realpath(repo_path)])
                                != os.path.realpath(repo_path)):
                            raise ValueError("completed attempt does not belong to this phase/checkout")
                        payload = read_output(record)
                        key = "baseline" if execution_phase == "baseline" else "method"
                        if _coerce_metric(payload.get(key)) is None:
                            raise ValueError("attempt output has no complete phase result")
                        measured_result(record, _attempt_authority.execution_protocol, execution_phase)
                        if payload.get("attempt_id") not in (None, "", attempt_id):
                            raise ValueError("attempt output cites a different attempt")
                        payload["attempt_id"] = attempt_id
                        with tempfile.TemporaryDirectory(
                                prefix=".ar-provider-recovery-", dir=os.path.dirname(result_path)) as staging:
                            recovered_path = os.path.join(staging, _RESULT_BASENAME)
                            with open(recovered_path, "w", encoding="utf-8") as saved:
                                json.dump(payload, saved, ensure_ascii=False)
                            os.replace(recovered_path, result_path)
                        emit("agentic_salvage", {"stage": "result_file_recovered",
                                                "source": "runtime_attempt", "attempt_id": attempt_id})
                    except (OSError, TypeError, ValueError) as recovery_error:
                        emit("agentic_run", {"stage": "provider_result_recovery_unavailable",
                                             "error": str(recovery_error)})
                try:
                    _restore_reported_attempt_checkpoint(_attempt_authority, result_path, emit)
                except (OSError, TypeError, ValueError, subprocess.SubprocessError) as recovery_error:
                    emit("agentic_run", {"stage": "provider_checkpoint_recovery_unavailable",
                                         "error": str(recovery_error)})
            raise
        finally:
            _prepare_checkpoint_handoff(backend, handle, repo_path, emit)
        status = getattr(res, "status", "") or ""
        emit("agentic_run", {"stage": "agent_done", "status": status})
        # sidecar 收尾(2026-08-08 定位): flash agent 常在 run 末尾才一次性 append variants,轮询没赶
        # 上 -> 停轮询并主动补一次分析,保证 direction 至少产出 + emit 可观测(之前三题 direction 全没
        # 产出且无法诊断)。注: 这轮 direction 对已结束的 agent 无引导价值,真正的跨轮引导是 follow-up。
        _stop.set()
        try:
            _analyze_once(
                repo_path, request, metric_hint, lower_is_better_hint, emit,
                memory_scope=memory_scope)
        except Exception:  # noqa: BLE001
            pass
        result = _build_result_from_file(
            result_path, name, metric_hint, lower_is_better_hint, emit,
            execution_phase=execution_phase)
        if result is not None:
            return finalize(result)
        failure_notes = _failed_attempt_notes(_attempt_log, execution_phase)
        failure_context = "\n" + "\n".join(failure_notes) if failure_notes else ""
        return {"benchmark": name, "baseline": None, "candidate": None,
                "metric": metric_hint, "lower_is_better": lower_is_better_hint,
                "verdict": {"verdict": "fail", "improved": False,
                            "reason": ("agent wrote no parseable ar_result.json; "
                                       f"agent status={status or 'unknown'}" + failure_context)},
                "repo_path": repo_path,
                "diagnosis_evidence": [f"workspace: {repo_path}",
                                       f"attempt_ledger: {_attempt_log}"] + failure_notes,
                "report": str(getattr(res, "event_summary", "") or "")[:2000],
                "error": "no result" + failure_context,
                **_failed_execution_accounting(_attempt_log, repo_path, execution_phase)}
    except InfraError as exc:
        emit("agentic_error", {"stage": "infra", "error": str(exc)})
        salvaged = _salvage_after_error(result_path, name, metric_hint,
                                        lower_is_better_hint, emit, f"infra: {exc}",
                                        execution_phase)
        if salvaged is not None:
            return finalize(salvaged)
        return {"benchmark": name, "baseline": None, "candidate": None,
                "metric": metric_hint, "lower_is_better": lower_is_better_hint,
                "verdict": {"verdict": "fail", "improved": False,
                            "reason": f"sandbox infra error: {exc}"},
                "repo_path": repo_path,
                "diagnosis_evidence": [f"workspace: {repo_path}"] + (
                    [f"attempt_ledger: {_attempt_log}"] if _attempt_log else []),
                "report": "", "error": f"infra: {exc}",
                **_failed_execution_accounting(_attempt_log, repo_path, execution_phase)}
    except Exception as exc:  # noqa: BLE001 -- never take the run down
        emit("agentic_error", {"stage": "run", "error": f"{type(exc).__name__}: {exc}"})
        salvaged = _salvage_after_error(result_path, name, metric_hint,
                                        lower_is_better_hint, emit, str(exc), execution_phase)
        if salvaged is not None:
            return finalize(salvaged)
        return {"benchmark": name, "baseline": None, "candidate": None,
                "metric": metric_hint, "lower_is_better": lower_is_better_hint,
                "verdict": {"verdict": "fail", "improved": False, "reason": str(exc)},
                "repo_path": repo_path,
                "diagnosis_evidence": [f"workspace: {repo_path}"] + (
                    [f"attempt_ledger: {_attempt_log}"] if _attempt_log else []),
                "report": "", "error": str(exc),
                **_failed_execution_accounting(_attempt_log, repo_path, execution_phase)}
    finally:
        _stop.set()   # 停 sidecar 轮询线程(异常路径也停,避免 daemon 线程被主线程退出强杀)
        if _attempt_authority is not None:
            _attempt_authority.stop()
        sandbox_closed = False
        if handle is not None:
            try:
                closed = backend.close_sandbox(handle)
                sandbox_closed = closed is not False
            except Exception:  # noqa: BLE001
                pass
        if _attempt_authority is not None:
            _attempt_authority.resolve_uncertain_after_sandbox_close(sandbox_closed)

__all__ = ["run_agentic_improve"]
