"""Run a repo_improve variant INSIDE a sandbox container instead of on the host.
把 repo_improve 的一次 variant 运行搬进沙箱容器执行(宿主逻辑不变)。

``run_repo_improve`` takes a ``run_variant_fn`` seam. Swapping in the callable this
module builds moves ONLY the execution into the container -- the pipeline keeps
doing everything else on the host filesystem exactly as before: cloning, building
each isolated variant workspace, restoring protected files over the agent's patch,
comparing seeds, deciding the verdict. The container sees those same directories
because the work root is bind-mounted at the SAME path, so a host path is a valid
container path.

Why bother: a repo's verified setup is only valid in the environment it was verified
in. nanochat's baseline runs in a container without ``TORCHDYNAMO_DISABLE`` because
the image ships python headers; on this host torch.compile cannot JIT and the same
commands die. Discovering a setup in a sandbox and then running it on the host would
be unsound -- so we run where we verified.

Semantics are kept identical to ``repo_improve.runner.run_variant`` on purpose
(fail-closed, budget cap, at-cap scoring, seed env), because the pipeline's fairness
guarantees are written against those.
"""
from __future__ import annotations

import logging
import os
import shlex

logger = logging.getLogger(__name__)

#: Prepare (install / download / build) gets its own large budget and is NOT charged
#: to the training run -- same split as the host runner.
PREPARE_TIMEOUT_S = 1800


def _make_container_writable(work_dir: str) -> None:
    """Let the container write into a workspace the HOST created.

    The pipeline builds each variant workspace on the host, and the container runs as a
    different user, so the default 0755/host-owned tree is read-only from inside: ``uv
    sync`` dies with "failed to create directory" before anything trains (seen live on
    nanochat, where the sandbox had verified the very same setup in its own clone). These
    are our own scratch directories under the work root, so widening them is safe.

    Best-effort: a path we cannot chmod is not worth failing the run over -- the run
    itself will report the real problem if it turns out to matter.
    """
    for root, dirs, files in os.walk(work_dir):
        for path in (root, *(os.path.join(root, n) for n in dirs + files)):
            try:
                os.chmod(path, os.stat(path).st_mode | 0o777)
            except OSError:
                pass


def make_sandbox_run_variant(backend, handle, *, budget_margin_s: int = 60):
    """Build a ``run_variant_fn``-compatible callable that executes in ``handle``'s container.

    Signature matches ``repo_improve.runner.run_variant`` so it can be passed straight
    to ``run_repo_improve(run_variant_fn=...)``. ``cuda_device`` is accepted and ignored:
    the sandbox was already given exactly one GPU when it was opened, so the container
    sees a single device and pinning inside would be redundant.
    """
    from autoresearch.experiments.repo_improve.runner import (
        RunResult, _parse_metric, _OUTPUT_KEEP, _idempotent_prepare)

    def _run_variant(benchmark, work_dir, *, prepare: bool = True, seed=None,
                     cuda_device=None, **_ignored) -> RunResult:
        _make_container_writable(work_dir)
        try:
            if prepare and benchmark.prepare_cmd and benchmark.prepare_cmd != "true":
                pc = backend.execute_command(
                    handle, _idempotent_prepare(benchmark.prepare_cmd), cwd=work_dir, timeout=PREPARE_TIMEOUT_S
                )
                if getattr(pc, "exit_code", 1) != 0:
                    err = (getattr(pc, "stderr", "") or getattr(pc, "stdout", "") or "")[:800]
                    return RunResult(benchmark.metric, None, False, error=f"prepare failed: {err}")

            cmd = benchmark.run_cmd
            if seed is not None and getattr(benchmark, "seed_env", None):
                # export inline so the fresh shell the container spawns actually sees it
                cmd = f"{benchmark.seed_env}={shlex.quote(str(seed))} {cmd}"

            rc = backend.execute_command(
                handle, cmd, cwd=work_dir, timeout=benchmark.budget_s + budget_margin_s
            )
            out = (getattr(rc, "stdout", "") or "") + "\n" + (getattr(rc, "stderr", "") or "")

            # A cap-kill is not a crash: an unbounded training script never exits on its
            # own, so the fair result is the metric it had reached at the shared cap.
            # Both baseline and candidate are capped identically (host runner does the same).
            capped = bool(getattr(rc, "timed_out", False))
            if getattr(rc, "exit_code", 1) != 0 and not capped:
                return RunResult(benchmark.metric, None, False,
                                 error=f"run failed (exit {getattr(rc, 'exit_code', '?')})",
                                 stdout_tail=out[-800:])

            value = _parse_metric(benchmark.metric, out)
            if value is None:
                reason = ("hit the wall-clock cap without emitting a parseable metric"
                          if capped else f"could not find metric {benchmark.metric!r} in output")
                return RunResult(benchmark.metric, None, False,
                                 error=(f"run {reason}" if capped else reason),
                                 stdout_tail=out[-800:], output=out[-_OUTPUT_KEEP:])
            return RunResult(benchmark.metric, value, True, stdout_tail=out[-800:])
        except Exception as exc:  # noqa: BLE001 -- fail closed, never crash the pipeline
            return RunResult(benchmark.metric, None, False, error=f"{type(exc).__name__}: {exc}")

    return _run_variant


__all__ = ["make_sandbox_run_variant", "PREPARE_TIMEOUT_S"]
