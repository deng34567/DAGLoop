import sys, os, json, re, uuid, hashlib, base64, fcntl, socket
import subprocess
import time
from itertools import combinations
from pathlib import Path

# Support both a deployed bundle and the source-tree scripts/dgx entry point.
# Never mix a new entry script with modules from an unrelated old deployment.
_script_dir = Path(__file__).resolve().parent
if os.environ.get("AUTORESEARCH_DEPLOYMENT_ENV"):
    from dotenv import load_dotenv
    load_dotenv(os.environ["AUTORESEARCH_DEPLOYMENT_ENV"], override=True)
for _root in (_script_dir, _script_dir.parent.parent):
    if (_root / "autoresearch").is_dir():
        sys.path.insert(0, str(_root))
        break
from autoresearch.experiments.repo_improve.agentic_improve import (
    run_agentic_improve, _extract_repo_urls)
from autoresearch.agents.runner_agent.handler import _res_to_improve_json


def _acquire_session_lock(workroot: str, *, blocking: bool = True):
    """Serialize remote runs that share one persisted session workroot."""
    os.makedirs(workroot, exist_ok=True)
    handle = os.fdopen(os.open(os.path.join(workroot, ".agentic_run.lock"),
                              os.O_RDONLY | os.O_CREAT, 0o644), "r")
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(handle.fileno(), flags)
    except Exception:
        handle.close()
        raise
    return handle


def _inspect_containers(ids: list[str]) -> list[dict]:
    """Keep surviving containers when one disappears between ps and inspect."""
    try:
        output = subprocess.check_output(
            ["docker", "inspect", *ids], text=True, stderr=subprocess.PIPE, timeout=15)
    except subprocess.CalledProcessError as exc:
        errors = (exc.stderr or "").splitlines()
        if not errors or any(
            not line.startswith("Error: No such object: ")
            or line.removeprefix("Error: No such object: ") not in ids
            for line in errors
        ):
            raise
        output = exc.output
    return json.loads(output)


def _container_uses_cards(cards: set[str]) -> bool:
    """Include containers and native compute processes outside our file locks."""
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True, timeout=15).split()
    containers = _inspect_containers(ids) if ids else []
    for container in containers:
        if not container.get("State", {}).get("Running"):
            continue
        config = container.get("HostConfig", {})
        devices = {d.get("PathOnHost", "") for d in config.get("Devices") or []}
        if devices & {"/dev/nvidia" + card for card in cards}:
            return True
        for request in config.get("DeviceRequests") or []:
            if request.get("Driver") == "nvidia" or any(
                "gpu" in group for group in request.get("Capabilities") or []
            ):
                requested = set(request.get("DeviceIDs") or [])
                # Count-based / UUID requests cannot be safely resolved as numeric indices.
                if not requested or not all(c.isdigit() for c in requested) or requested & cards:
                    return True
    occupied = set(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True, timeout=15).split())
    if not occupied:
        return False
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True, timeout=15)
    by_index = dict(line.strip().split(", ", 1) for line in inventory.splitlines() if line.strip())
    return any(by_index[card] in occupied for card in cards)


def _remove_orphan_agent_containers(workroot: str) -> None:
    """Remove stale agent-server containers left by an earlier run of this session.

    The caller already owns the per-session lock, so another live agentic run cannot
    legitimately own an agent-server mounted on the same workroot.  Do not touch
    evaluator/training containers: a lost attempt may still need reconciliation.
    """
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True, timeout=15).split()
    if not ids:
        return
    containers = _inspect_containers(ids)
    target = os.path.realpath(workroot)
    stale = []
    for container in containers:
        if not container.get("State", {}).get("Running"):
            continue
        name = str(container.get("Name") or "").lstrip("/")
        if not name.startswith("agent-server-"):
            continue
        sources = {
            os.path.realpath(str(mount.get("Source") or ""))
            for mount in container.get("Mounts") or []
            if mount.get("Source")
        }
        if target in sources:
            stale.append(str(container.get("Id") or ""))
    for container_id in filter(None, stale):
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True,
                       text=True, check=True, timeout=60)


def _resolve_uncertain_attempt_without_owner(workroot: str) -> bool:
    """Archive an uncertain attempt once its owning sandbox is definitively gone.

    The caller owns the per-session lock and has already removed stale agent-server
    containers.  A remaining container mounted on this workroot may still own the
    execution, so preserve the marker in that case.  This recovers a session after
    the old wrapper and sandbox both disappeared without weakening the ordinary
    uncertain-execution guard while either can still be alive.
    """
    marker = Path(workroot) / ".autoresearch-uncertain-execution.json"
    if not marker.is_file():
        return False
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True, timeout=15).split()
    if ids:
        containers = _inspect_containers(ids)
        target = os.path.realpath(workroot)
        for container in containers:
            if not container.get("State", {}).get("Running"):
                continue
            sources = {
                os.path.realpath(str(mount.get("Source") or ""))
                for mount in container.get("Mounts") or []
                if mount.get("Source")
            }
            if target in sources:
                return False
    pending = json.loads(marker.read_text())
    attempt_id = str(pending.get("attempt_id") or "").strip()
    cwd = str(pending.get("cwd") or "").strip()
    log_path = str(pending.get("log_path") or "").strip()
    if (not pending.get("requires_reconciliation") or not attempt_id or not cwd or not log_path
            or os.path.commonpath([os.path.realpath(workroot), os.path.realpath(cwd)])
            != os.path.realpath(workroot) or not os.path.isfile(log_path)):
        return False
    launch_seen = False
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if event.get("attempt_id") == attempt_id and event.get("event") == "launch":
                launch_seen = True
                break
    if not launch_seen:
        return False
    pending.update({
        "resolved_at": time.time(),
        "resolution": "owning wrapper and sandbox absent under exclusive session lock",
    })
    resolved = marker.with_name(marker.stem + f".resolved-{attempt_id}.json")
    resolved.write_text(json.dumps(pending, sort_keys=True))
    marker.unlink()
    return True


def _acquire_gpu_locks(workroot: str, card: str | None, *,
                       allowed_cards: str | None = None, blocking: bool = True):
    """Reserve cards across Web processes and keep one session on the same cards."""
    if not card:
        return []
    cards = set(card.split(","))
    if not all(c.isdigit() for c in cards):
        raise ValueError("GPU card IDs must be numeric indices")
    pool = set(allowed_cards.split(",")) if allowed_cards is not None else cards
    if not all(c.isdigit() for c in pool) or not cards <= pool:
        raise ValueError("Requested GPU cards must belong to the numeric allowed pool")
    os.makedirs(workroot, exist_ok=True)
    affinity_path = Path(workroot) / ".gpu-card-affinity"
    if affinity_path.exists():
        affinity = set(affinity_path.read_text().strip().split(","))
        if (not affinity or not all(c.isdigit() for c in affinity)
                or len(affinity) != len(cards) or not affinity <= pool):
            raise ValueError("Persisted GPU affinity is incompatible with the requested pool")
        cards = affinity
        # An explicitly configured homogeneous pool may prefer the old card
        # without leaving another allowed card idle while that card is busy.
        if os.environ.get("AUTORESEARCH_GPU_AFFINITY", "strict") != "prefer":
            pool = affinity
    preferred = tuple(sorted(cards, key=int))
    choices = [preferred] + [
        group for group in combinations(sorted(pool, key=int), len(cards))
        if group != preferred
    ]
    lock_root = os.environ.get("AUTORESEARCH_GPU_LOCK_DIR") or os.path.join(
        os.path.dirname(os.path.realpath(workroot)), ".gpu-locks")
    # The lock root may live on a filesystem shared by several GPU hosts.  A
    # numeric CUDA index is host-local, so locks for GPU 0 on different hosts
    # must not contend with each other.
    lock_dir = os.path.join(lock_root, socket.gethostname())
    os.makedirs(lock_dir, exist_ok=True)
    waiting = False
    while True:
        for group in choices:
            handles = []
            try:
                for gpu in group:
                    handle = open(os.path.join(lock_dir, f"gpu-{gpu}.lock"), "a+")
                    handles.append(handle)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if _container_uses_cards(set(group)):
                    raise BlockingIOError("GPU still assigned to a running container")
                affinity_path.write_text(",".join(group) + "\n")
                return handles
            except Exception as exc:
                for handle in handles:
                    handle.close()
                if not isinstance(exc, BlockingIOError):
                    raise
        if not blocking:
            raise BlockingIOError("No permitted GPU group is free")
        if not waiting:
            print(f"GPU_WAIT preferred={card} allowed={','.join(sorted(pool, key=int))}", flush=True)
            waiting = True
        time.sleep(5)


def _decode_execution_contract(env) -> tuple[
    str, dict | None, dict | None, dict | None
]:
    phase = str(env.get("AR_EXECUTION_PHASE") or "combined").strip().lower()
    if phase not in {"combined", "baseline", "method"}:
        raise ValueError(f"unknown execution phase: {phase}")
    encoded = str(env.get("AR_FROZEN_BASELINE_B64") or "").strip()
    frozen = None
    if encoded:
        try:
            frozen = json.loads(base64.b64decode(encoded, validate=True).decode())
        except Exception as exc:
            raise ValueError(f"invalid frozen baseline payload: {exc}") from exc
        if not isinstance(frozen, dict):
            raise ValueError("invalid frozen baseline payload: expected object")
    if phase == "method" and frozen is None:
        raise ValueError("method execution requires a frozen baseline")
    boundary_encoded = str(env.get("AR_CONFIRMED_BOUNDARY_B64") or "").strip()
    boundary = None
    if boundary_encoded:
        try:
            boundary = json.loads(base64.b64decode(
                boundary_encoded, validate=True).decode())
        except Exception as exc:
            raise ValueError(f"invalid confirmed boundary payload: {exc}") from exc
        if not isinstance(boundary, dict):
            raise ValueError("invalid confirmed boundary payload: expected object")
    semantics_encoded = str(env.get("AR_EVALUATION_SEMANTICS_B64") or "").strip()
    semantics = None
    if semantics_encoded:
        try:
            semantics = json.loads(base64.b64decode(
                semantics_encoded, validate=True).decode())
        except Exception as exc:
            raise ValueError(f"invalid evaluation semantics payload: {exc}") from exc
        if not isinstance(semantics, dict):
            raise ValueError("invalid evaluation semantics payload: expected object")
    return phase, frozen, boundary, semantics


def _select_repo_urls(question: str, env) -> tuple[str | None, tuple[str, ...]]:
    """Prefer the structured, confirmed primary repository over URLs in prose."""
    encoded = str(env.get("AR_PRIMARY_REPO_B64") or "").strip()
    if encoded:
        try:
            primary = base64.b64decode(encoded, validate=True).decode().strip()
        except Exception as exc:
            raise ValueError(f"invalid primary repository payload: {exc}") from exc
        if not primary:
            raise ValueError("invalid primary repository payload: empty URL")
        # Reference links in the research question are not implicit tool repositories.
        # Explicit tool repositories remain available through AUTORESEARCH_TOOL_REPOS.
        return primary, ()
    urls = _extract_repo_urls(question)
    return (urls[0] if urls else None), tuple(urls[1:])


def _decode_memory_scope(env) -> str | None:
    encoded = str(env.get("AR_METHOD_SCOPE_B64") or "").strip()
    if not encoded:
        return None
    try:
        scope = base64.b64decode(encoded, validate=True).decode().strip()
    except Exception as exc:
        raise ValueError(f"invalid method memory scope: {exc}") from exc
    if not scope:
        raise ValueError("invalid method memory scope: empty value")
    return scope


def _decode_prior_worktree(env) -> str | list[str] | dict | None:
    encoded = str(env.get("AR_PRIOR_WORKTREE_B64") or "").strip()
    if not encoded:
        return None
    try:
        path = base64.b64decode(encoded, validate=True).decode().strip()
    except Exception as exc:
        raise ValueError(f"invalid prior worktree payload: {exc}") from exc
    if not path:
        raise ValueError("invalid prior worktree payload: empty value")
    if path.startswith(("[", "{", '"')):
        try:
            paths = json.loads(path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prior worktree list: {exc}") from exc
        if isinstance(paths, str):
            if not paths.strip():
                raise ValueError("invalid prior worktree payload: empty value")
            return paths.strip()
        if isinstance(paths, dict):
            if (set(paths) != {"checkpoints", "merge_bases"}
                    or any(not isinstance(paths[key], list) or not paths[key]
                           or not all(isinstance(item, str) and item.strip()
                                      for item in paths[key]) for key in paths)):
                raise ValueError("invalid prior worktree merge bases: expected non-empty path lists")
            return {key: [item.strip() for item in values] for key, values in paths.items()}
        # [] explicitly identifies a root component whose parent is the baseline.
        if (not isinstance(paths, list)
                or not all(isinstance(item, str) and item.strip() for item in paths)):
            raise ValueError("invalid prior worktree list: expected non-empty path strings")
        return [item.strip() for item in paths]
    return path


def _read_request(argv, env, stream):
    """Large dispatches use stdin, never an oversized argv or inherited env value."""
    q = argv[1] if len(argv) > 1 else ""
    if q != "--request-stdin":
        return q, env
    request = json.load(stream)
    if not isinstance(request, dict) or not isinstance(request.get("question"), str):
        raise ValueError("invalid streamed request: expected question string")
    extra = request.get("environment", {})
    allowed = {"AR_METHOD_SCOPE_B64", "AR_PRIOR_WORKTREE_B64", "AR_COMPONENT_PARENT_B64",
               "AR_FROZEN_BASELINE_B64", "AR_CONFIRMED_BOUNDARY_B64",
               "AR_EVALUATION_SEMANTICS_B64", "AR_PRIMARY_REPO_B64", "AR_COMPONENT_STAGE"}
    if (not isinstance(extra, dict) or not set(extra) <= allowed
            or not all(isinstance(value, str) for value in extra.values())):
        raise ValueError("invalid streamed request fields")
    return request["question"], {**env, **extra}


def _session_workroot(sid: str) -> str:
    base = os.environ.get("AUTORESEARCH_REMOTE_SESSION_ROOT") or "/srv/dagloop/ar_gpu_runs/agentic_remote"
    if not os.path.isabs(base):
        raise ValueError("remote session root must be absolute")
    slug = ("sess-" + hashlib.sha1(sid.encode()).hexdigest()[:12]) if sid else uuid.uuid4().hex[:10]
    return os.path.join(base, slug)


def main() -> None:
    q, request_env = _read_request(sys.argv, os.environ, sys.stdin)
    card = os.environ.get("AR_GPU_CARD")
    selection_env = request_env
    primary_source = "question"
    if len(sys.argv) > 2:
        selection_env = dict(os.environ)
        selection_env["AR_PRIMARY_REPO_B64"] = sys.argv[2]
        primary_source = "argument"
    elif request_env.get("AR_PRIMARY_REPO_B64"):
        primary_source = "environment"
    repo_url, tool_urls = _select_repo_urls(q, selection_env)
    (execution_phase, frozen_baseline, confirmed_boundary,
     evaluation_semantics) = _decode_execution_contract(request_env)
    memory_scope = _decode_memory_scope(request_env)
    prior_worktree = _decode_prior_worktree(request_env)
    component_parent = _decode_prior_worktree({
        "AR_PRIOR_WORKTREE_B64": request_env.get("AR_COMPONENT_PARENT_B64", "")})
    component_stage = request_env.get("AR_COMPONENT_STAGE")
    if component_stage not in {None, "0", "1"}:
        raise ValueError("invalid component stage: expected 0 or 1")
    # #75 稳定 workroot: 同一 session 每一轮复用同一个 workroot -> 复用 workroot 级 .venv-shared。
    sid = os.environ.get("AR_SESSION_ID", "").strip()
    workroot = _session_workroot(sid)
    os.makedirs(workroot, exist_ok=True)
    # A Web restart can orphan the old SSH child while redriving the same persisted state.  Keep
    # both processes from mutating the same repo-N/checkpoint sequence concurrently; the lock is
    # released automatically when this process exits.
    session_lock = _acquire_session_lock(workroot)
    _remove_orphan_agent_containers(workroot)
    _resolve_uncertain_attempt_without_owner(workroot)
    # Held until process exit, including sandbox shutdown and result emission.  A Web's
    # in-memory card queue alone cannot protect against another Web or an orphan container.
    gpu_locks = _acquire_gpu_locks(
        workroot, card, allowed_cards=os.environ.get("AR_ALLOWED_GPU_CARDS"))
    if gpu_locks:
        # Record and execute on the actual locked devices, not the Web's provisional choice.
        card = ",".join(os.path.basename(handle.name)[4:-5] for handle in gpu_locks)
        print(f"GPU_RESERVED cards={card}", flush=True)
    try:
        os.chmod(workroot, 0o777)
        venv_shared = os.path.join(workroot, ".venv-shared")
        os.makedirs(venv_shared, exist_ok=True)
        os.chmod(venv_shared, 0o777)
    except Exception:
        pass

    log_path = os.path.join(workroot, "agentic_emit.log")

    def log(message):
        try:
            with open(log_path, "a") as f:
                f.write("%s %s\n" % (workroot[-10:], message))
        except Exception:
            pass

    def emit(event, payload):
        try:
            line = "EMIT %s %s" % (event, json.dumps(payload)[:500])
            print(line, flush=True)
            log(line)
        except Exception:
            pass

    log("PRIMARY source=%s repo=%s" % (primary_source, repo_url))

    res = run_agentic_improve(
        q, repo_url=repo_url, workroot=workroot,
        tool_repo_urls=tool_urls,
        cuda_device=(card if card else None),
        budget_s=None, sandbox_llm=None, emit=emit,
        execution_phase=execution_phase, frozen_baseline=frozen_baseline,
        confirmed_boundary=confirmed_boundary,
        evaluation_semantics=evaluation_semantics,
        enforce_fairness=False,
        memory_scope=memory_scope,
        prior_worktree=prior_worktree,
        component_parent_worktree=component_parent,
        component_stage=(component_stage == "1" if component_stage is not None else None),
    )
    rr = _res_to_improve_json(res)
    # Keep component runtime evidence across deployments whose remote serializer predates
    # the ComponentPlan protocol.  The in-process result already contains the host-validated
    # attempt ID; dropping it here makes a successful indirect component look unexecuted.
    rr["component_attempt_id"] = res.get("component_attempt_id")
    print("RESULT_JSON " + json.dumps(rr), flush=True)
    log("RESULT bl=%s m=%s" % (rr.get("baseline"), rr.get("candidate")))
    # hangfix: agent SDK/docker/httpx 遗留非daemon线程会卡住解释器退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
