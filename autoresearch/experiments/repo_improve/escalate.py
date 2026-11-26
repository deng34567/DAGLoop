"""Tier 3 of setup healing: when reading the repo isn't enough, RUN it in a sandbox.
第三层安装自愈:读文件猜不出来时,开沙箱让 agent 真跑着摸出来。

The ladder this completes:
  tier 1 (cheap)  ``infer_benchmark``  -- read the repo, state the boundary + commands
  tier 2 (medium) ``correct_execution_from_run`` -- read the ERROR+OUTPUT, correct commands/metric (bounded)
  tier 3 (costly) this module          -- an agent iterates inside a GPU sandbox until
                                          the baseline actually runs, hard-judged

Tier 1/2 are text problems: they can only find what the repo SAYS. Some setups are not
textual at all -- nanochat needs a tokenizer trained by a script you only learn about by
tracing an assert, and on this host it also needs torch.compile disabled because the
machine lacks python headers, a fact that appears nowhere in the repo. Those surface only
by running, which is what tier 3 does.

Once a setup is verified in the sandbox, the rest of the run STAYS in that sandbox: a
setup is only valid in the environment it was verified in (the very same nanochat commands
that work in the container die on this host). So this returns a ``run_variant_fn`` bound to
the container, which the pipeline then uses for baseline, candidate and reproduce alike.
"""
from __future__ import annotations

import dataclasses
import logging
import os

logger = logging.getLogger(__name__)


# A SHARED download/dep cache, reused across runs so a heavy dataset/deps (e.g. SOL-ExecBench's
# ~15GB flashinfer-trace HF dataset) is fetched ONCE, not re-downloaded every fresh run. It lives
# OUTSIDE the per-run workroot (which callers wipe between runs) so it survives; it is bind-mounted
# into every sandbox. Deployments can override the portable user-cache default.
SHARED_SANDBOX_CACHE = os.environ.get(
    "AUTORESEARCH_SANDBOX_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "sandbox"))


def _sandbox_scratch(workroot: str):
    """A per-run scratch dir UNDER the (bind-mounted) work root, plus the env that
    points the container's heavy writes at it. ``workroot`` is mounted into the
    container at the same path, and on this box it lives on the big ``/data`` disk --
    so a repo clone, its dataset, the venv all land on ``/data`` instead of the
    container overlay. HF/torch/pip/uv CACHES point at the SHARED cross-run cache
    (``SHARED_SANDBOX_CACHE``, also mounted) so a big dataset is downloaded once.

    Returns ``(repo_dir, env)``. Dirs are pre-created host-side so the mount exposes
    them and tools that expect an existing HOME/cache don't trip.
    """
    scratch = os.path.join(workroot, "_sandbox")
    home = os.path.join(scratch, "home")
    cache = os.path.join(home, ".cache")      # per-run HOME cache (small, ephemeral)
    shared = SHARED_SANDBOX_CACHE             # cross-run cache (datasets/deps, persists)
    repo_dir = os.path.join(scratch, "repo")
    for d in (scratch, home, cache, repo_dir, shared,
              os.path.join(shared, "huggingface"), os.path.join(shared, "torch"),
              os.path.join(shared, "pip"), os.path.join(shared, "uv")):
        try:
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o777)   # the container writes as root/other UID into these
        except OSError as exc:
            logger.warning("could not pre-create sandbox scratch %s: %s", d, exc)
    env = {
        "HOME": home,
        "XDG_CACHE_HOME": cache,
        "HF_HOME": os.path.join(shared, "huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(shared, "huggingface", "hub"),
        "TORCH_HOME": os.path.join(shared, "torch"),
        "PIP_CACHE_DIR": os.path.join(shared, "pip"),
        "UV_CACHE_DIR": os.path.join(shared, "uv"),
    }
    return repo_dir, env


#: The agent's report may prefix commands with an absolute ``cd`` into the sandbox clone
#: (``cd /workspace/repo && ...``). The pipeline runs them in ITS OWN variant workspace,
#: so such a prefix would silently send the run to the wrong directory.
from autoresearch.experiments.repo_improve.discover import _strip_leading_cd

#: The agent's report may prefix commands with an absolute ``cd`` into the sandbox clone
#: (``cd /workspace/repo && ...``). The pipeline runs them in ITS OWN variant workspace,
#: so such a prefix would silently send the run to the wrong directory.
from autoresearch.experiments.repo_improve.discover import _strip_leading_cd

#: A discovery run talks to an agent-server over HTTP inside the container; on a busy host
#: that link occasionally drops mid-run ("connection reset by peer", "timed out") and the
#: whole ~20-min discovery is lost. Those are transport failures of the SUBSTRATE, not a
#: verdict about the repo, so we reopen a FRESH container and try once more before giving up.
_MAX_ESCALATE_ATTEMPTS = int(os.environ.get("AUTORESEARCH_MAX_ESCALATE_ATTEMPTS", "2"))


def escalate_to_sandbox(
    *,
    workroot: str,
    pristine: str,
    benchmark,
    request: str,
    cuda_device=None,
    llm=None,
    emit=None,
    backend=None,
    discover_fn=None,
    budget_s: int | None = None,
    sandbox_llm=None,
    server_image: str | None = None,
):
    """Open a sandbox, discover a VERIFIED setup for ``benchmark.repo_url``, and return
    ``(new_benchmark, run_variant_fn, close)`` — or ``None`` if escalation didn't work.

    ``new_benchmark`` carries only the discovered SETUP (prepare/run/metric); the
    trusted-eval boundary (editable vs protected files, budget) is untouched, because
    which files may be edited has nothing to do with how the repo is installed.

    The sandbox is torn down on every failure path; the caller owns ``close`` on success.
    Never raises: escalation is best-effort, and a host without docker/openhands must
    degrade to an honest failure rather than take the pipeline down.
    """
    emit = emit or (lambda event, payload: None)
    from autoresearch.agents._reuse.openhands_backend import InfraError
    if backend is None:
        from autoresearch.agents._reuse.openhands_backend import (
            DEFAULT_SERVER_IMAGE, OpenHandsBackend)
        # A repo may need a specific image (SOL needs its CUDA/CUTLASS one with nvcc);
        # the caller derives that from the repo, so it is never a global hard-coded pin.
        backend = OpenHandsBackend(server_image=server_image or DEFAULT_SERVER_IMAGE)
    if discover_fn is None:
        from autoresearch.agents._reuse.sandbox_setup import discover_setup_in_sandbox
        discover_fn = discover_setup_in_sandbox

    # Point the container's heavy writes (clone, dataset, venv, caches) at a scratch dir
    # on the mounted /data work root so they never fill the root-disk container overlay.
    repo_dir, sandbox_env = _sandbox_scratch(workroot)

    last_infra = None
    for attempt in range(_MAX_ESCALATE_ATTEMPTS):
        handle = None
        try:
            emit("sandbox_escalation", {"stage": "opening", "repo_url": benchmark.repo_url,
                                        "cuda_device": cuda_device, "attempt": attempt + 1})
            # Mount the work root at the SAME path so the workspaces the pipeline builds on
            # the host are directly usable inside the container.
            handle = backend.open_sandbox(cuda_device=cuda_device, mounts=[workroot, SHARED_SANDBOX_CACHE], env=sandbox_env)

            # NOT ``llm``: the pipeline's client (used for correct_execution_from_run / propose_patch) is OUR
            # LLM wrapper, while the in-container agent needs an OpenHands SDK LLM. Handing the
            # former over blows up inside get_default_agent and the escalation "fails" in 20s
            # with a misleading "no usable setup" -- seen live on nanochat. Passing None lets
            # run_agent_task build the right one (build_sandbox_llm, configured from env);
            # ``sandbox_llm`` stays available for injection.
            setup = discover_fn(
                backend, handle,
                repo_url=benchmark.repo_url,
                goal=request,
                budget_s=budget_s or benchmark.budget_s,
                llm=sandbox_llm,
                repo_dir=repo_dir,
            )
            if setup is None or not getattr(setup, "verified", False):
                # The agent RAN and found no usable setup -- a verdict about the repo, not a
                # transport hiccup, so retrying the same discovery would only burn another 20 min.
                emit("sandbox_escalation", {"stage": "rejected",
                                            "reason": getattr(setup, "notes", "no usable setup")})
                backend.close_sandbox(handle)
                return None

            new_benchmark = dataclasses.replace(
                benchmark,
                prepare_cmd=_strip_leading_cd(setup.prepare_cmd) or "true",
                run_cmd=_strip_leading_cd(setup.run_cmd),
                metric=setup.metric,
                lower_is_better=bool(setup.lower_is_better),
            )
            emit("sandbox_escalation", {"stage": "verified", "metric": setup.metric,
                                        "baseline_in_sandbox": setup.metric_value,
                                        "prepare_cmd": new_benchmark.prepare_cmd,
                                        "run_cmd": new_benchmark.run_cmd})

            from autoresearch.agents._reuse.sandbox_runner import make_sandbox_run_variant
            run_variant_fn = make_sandbox_run_variant(backend, handle)

            def _close(_h=handle):
                try:
                    backend.close_sandbox(_h)
                except Exception:  # noqa: BLE001 -- teardown is best-effort
                    logger.warning("sandbox teardown failed", exc_info=True)

            return new_benchmark, run_variant_fn, _close, backend, handle
        except InfraError as exc:
            # Transport/substrate failure (container link reset, agent-server unreachable): the
            # discovery is lost but the repo hasn't been judged. Drop the dead container and, if
            # attempts remain, reopen a fresh one -- last night's identical setup discovered fine.
            last_infra = exc
            logger.warning("escalation transient failure (attempt %d/%d): %s",
                           attempt + 1, _MAX_ESCALATE_ATTEMPTS, exc)
            emit("sandbox_escalation", {"stage": "transient_retry", "attempt": attempt + 1,
                                        "error": f"{type(exc).__name__}: {exc}"})
            if handle is not None:
                try:
                    backend.close_sandbox(handle)
                except Exception:  # noqa: BLE001
                    pass
            continue
        except Exception as exc:  # noqa: BLE001 -- no docker/openhands must not kill the run
            logger.warning("sandbox escalation unavailable: %s: %s", type(exc).__name__, exc)
            emit("sandbox_escalation", {"stage": "unavailable",
                                        "error": f"{type(exc).__name__}: {exc}"})
            if handle is not None:
                try:
                    backend.close_sandbox(handle)
                except Exception:  # noqa: BLE001
                    pass
            return None

    # every attempt died on a transport failure -- honest give-up (not a repo verdict).
    emit("sandbox_escalation", {"stage": "unavailable",
                                "error": f"InfraError after {_MAX_ESCALATE_ATTEMPTS} attempts: {last_infra}"})
    return None


def run_in_known_image(
    *, workroot: str, benchmark, cuda_device=None, emit=None, server_image: str,
    backend=None, env=None,
):
    """Open a plain container for a repo whose setup is ALREADY known (no agent discovery),
    and return ``(benchmark, run_variant_fn, close)`` -- or ``None`` if docker is unavailable.

    Used for SOL: its adapter gives a definite boundary + run command, and its image is a
    plain CUDA benchmark image (not an agent-server), so we just run the given commands in it
    with GPU passthrough. The trusted-eval boundary is unchanged; only execution moves into
    the container. Never raises -- no docker degrades to an honest failure.
    """
    emit = emit or (lambda event, payload: None)
    handle = None
    try:
        if backend is None:
            from autoresearch.agents._reuse.docker_exec import DockerExecBackend
            backend = DockerExecBackend(server_image=server_image)
        emit("docker_image_run", {"stage": "opening", "image": server_image,
                                  "cuda_device": cuda_device})
        # Cache/HOME on the mounted /data scratch too, so a recalled setup's re-download
        # (the committed image no longer carries the data) lands on /data, not root.
        _, scratch_env = _sandbox_scratch(workroot)
        scratch_env.update(env or {})   # caller env (e.g. SOL's FLASHINFER_*) wins on conflict
        handle = backend.open_sandbox(cuda_device=cuda_device, mounts=[workroot, SHARED_SANDBOX_CACHE], env=scratch_env)

        from autoresearch.agents._reuse.sandbox_runner import make_sandbox_run_variant
        run_variant_fn = make_sandbox_run_variant(backend, handle)

        def _close():
            try:
                backend.close_sandbox(handle)
            except Exception:  # noqa: BLE001
                logger.warning("container teardown failed", exc_info=True)

        return benchmark, run_variant_fn, _close, backend, handle
    except Exception as exc:  # noqa: BLE001 -- no docker must not kill the run
        logger.warning("known-image run unavailable: %s: %s", type(exc).__name__, exc)
        emit("docker_image_run", {"stage": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
        if handle is not None and backend is not None:
            try:
                backend.close_sandbox(handle)
            except Exception:  # noqa: BLE001
                pass
        return None


def open_agent_sandbox(
    *, workroot: str, benchmark, cuda_device=None, emit=None, env=None, server_image: str | None = None,
):
    """Open an AGENT-CAPABLE sandbox (OpenHandsBackend) for a repo whose setup ALREADY works, with NO
    discovery agent -- and return ``(benchmark, run_variant_fn, close, backend, handle)`` or ``None``.

    Why separate from ``run_in_known_image``: that one uses ``DockerExecBackend`` (no ``run_agent_task``),
    so its handle can't drive the in-container OpenHands agent. This opens the OpenHands backend so the
    improve step can go through ``build_sandbox_propose`` (the item-6 agent route) even for a repo whose
    baseline ran fine on the host (minGPT/nanoGPT) -- both baseline and candidate then run in the SAME
    container, keeping the comparison fair. Never raises: no docker/openhands degrades to ``None`` and
    the caller keeps whatever baseline it had.
    """
    emit = emit or (lambda event, payload: None)
    handle = None
    backend = None
    try:
        from autoresearch.agents._reuse.openhands_backend import (
            DEFAULT_SERVER_IMAGE, OpenHandsBackend)
        backend = OpenHandsBackend(server_image=server_image or DEFAULT_SERVER_IMAGE)
        _, scratch_env = _sandbox_scratch(workroot)
        scratch_env.update(env or {})
        emit("agent_sandbox", {"stage": "opening", "repo_url": benchmark.repo_url,
                               "cuda_device": cuda_device})
        handle = backend.open_sandbox(cuda_device=cuda_device,
                                      mounts=[workroot, SHARED_SANDBOX_CACHE], env=scratch_env)
        from autoresearch.agents._reuse.sandbox_runner import make_sandbox_run_variant
        run_variant_fn = make_sandbox_run_variant(backend, handle)

        def _close(_h=handle):
            try:
                backend.close_sandbox(_h)
            except Exception:  # noqa: BLE001 -- teardown is best-effort
                logger.warning("agent sandbox teardown failed", exc_info=True)

        emit("agent_sandbox", {"stage": "opened"})
        return benchmark, run_variant_fn, _close, backend, handle
    except Exception as exc:  # noqa: BLE001 -- no docker/openhands must not kill the run
        logger.warning("agent sandbox unavailable: %s: %s", type(exc).__name__, exc)
        emit("agent_sandbox", {"stage": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
        if handle is not None and backend is not None:
            try:
                backend.close_sandbox(handle)
            except Exception:  # noqa: BLE001
                pass
        return None


__all__ = ["escalate_to_sandbox", "run_in_known_image", "open_agent_sandbox"]
