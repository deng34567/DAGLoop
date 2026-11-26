"""Orchestrate the repo-improvement flow.

    request -> discover benchmark -> clone repo -> run pristine baseline ->
    agent proposes a constrained patch -> run patched candidate (same trusted eval,
    same budget) -> honest verdict -> report

Same philosophy as the general honest-experiment pipeline (propose -> trusted/isolated
run -> honest verdict -> report); only the two ends differ: the "method" is a patch to a
REAL cloned baseline, and the "ruler" is the repo's own trusted evaluator on its pinned
validation data. ``run_variant_fn`` / ``clone_fn`` are injected so the orchestration is
fully testable without a GPU or network.
"""
from __future__ import annotations

import dataclasses as _dc
import math
import os
from pathlib import Path

from autoresearch.experiments.repo_improve import clone as _clone
from autoresearch.experiments.repo_improve import runner as _runner
from autoresearch.experiments.repo_improve.discover import discover, infer_benchmark
from autoresearch.experiments.repo_improve.discover import correct_execution_from_run as _correct_run
from autoresearch.experiments.repo_improve.escalate import (
    escalate_to_sandbox, run_in_known_image, open_agent_sandbox)
from autoresearch.experiments.repo_improve.improve import propose_patch
from autoresearch.experiments.repo_improve.optimize_loop import optimize_in_sandbox, build_llm_propose
from autoresearch.experiments.repo_improve.sandbox_propose import build_sandbox_propose
from autoresearch.experiments.repo_improve.implementation_discussion import discuss_implementation
from autoresearch.experiments.repo_improve.setup_memory import (
    recall_setup, remember_setup, setup_image_tag)

# phase 1b: how many times to let correct_execution_from_run heal a failing baseline (missing data/deps/
# build/Docker) before giving up with an honest fail. Bounded so it can't loop forever.
_MAX_SETUP_REPAIRS = int(os.environ.get("AUTORESEARCH_MAX_SETUP_REPAIRS", "2"))

# 边界推断是一次性 LLM 调用,偶发返回 None(实测同仓库重跑抖动过,或 LLM 端点间歇 502/503);
# 重试远比放弃便宜。可配 AUTORESEARCH_MAX_INFER_ATTEMPTS -- 端点抖时调高能扛过更多坏窗口。
_MAX_INFER_ATTEMPTS = int(os.environ.get("AUTORESEARCH_MAX_INFER_ATTEMPTS", "3"))

# 沙箱内优化循环的迭代上限(每轮=一次可信重跑,预算敏感)。env 可覆盖(测试时调小加速)。
_MAX_OPTIMIZE_ROUNDS = max(1, int(os.environ.get("AUTORESEARCH_MAX_OPTIMIZE_ROUNDS", "3")))


def _resolve_repo_image(spec: str, pristine: str, emit) -> str | None:
    """Resolve an inferred ``benchmark.image`` to a usable LOCAL docker image tag, or None.

    Two forms, both general (no repo hard-coded):
      * an existing image tag        -> used as-is when present locally.
      * ``build:<Dockerfile-relpath>`` -> ``docker build`` from the cloned repo, tagged
        deterministically by the Dockerfile's content (so a repeat reuses the built image and
        docker's layer cache -- a repo whose CUDA image was built once is instant next time).

    Best-effort: any failure returns None and the caller degrades to the default sandbox.
    """
    import hashlib
    import subprocess
    spec = (spec or "").strip()
    if not spec:
        return None

    def _exists(tag: str) -> bool:
        try:
            return subprocess.run(["docker", "image", "inspect", tag],
                                  capture_output=True, timeout=30).returncode == 0
        except Exception:  # noqa: BLE001
            return False

    if not spec.startswith("build:"):
        return spec if _exists(spec) else None

    df_rel = spec[len("build:"):].strip()
    df_path = os.path.join(pristine, df_rel)
    if not os.path.exists(df_path):
        return None
    try:
        digest = hashlib.sha1(open(df_path, "rb").read()).hexdigest()[:12]
    except OSError:
        digest = hashlib.sha1(df_rel.encode()).hexdigest()[:12]
    tag = f"autoresearch-repo-{digest}"
    if _exists(tag):
        emit("repo_image_build", {"stage": "cached", "image": tag})
        return tag
    emit("repo_image_build", {"stage": "building", "image": tag, "dockerfile": df_rel})
    # Many repo Dockerfiles create a user from build args (HOST_UID/HOST_GID/HOST_USER, or the
    # commonly-named UID/GID/USER) so container writes match the host owner. Pass them so such a
    # Dockerfile builds; a Dockerfile that ignores them is unaffected (unused build args are fine).
    import getpass
    _uid, _gid, _user = os.getuid(), os.getgid(), getpass.getuser()
    build_args: list[str] = []
    for _k, _v in (("HOST_UID", _uid), ("HOST_GID", _gid), ("HOST_USER", _user),
                   ("UID", _uid), ("GID", _gid), ("USER", _user)):
        build_args += ["--build-arg", f"{_k}={_v}"]
    try:
        r = subprocess.run(["docker", "build", *build_args, "-t", tag, "-f", df_path, pristine],
                           capture_output=True, text=True, timeout=3600)
    except Exception as exc:  # noqa: BLE001 -- a broken build must not take the run down
        emit("repo_image_build", {"stage": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return None
    if r.returncode == 0:
        emit("repo_image_build", {"stage": "built", "image": tag})
        return tag
    emit("repo_image_build", {"stage": "failed", "error": (r.stderr or "")[-300:]})
    return None


def _heal_once(pristine, benchmark, base_res, request, llm):
    """ONE general self-heal step: re-infer the EXECUTION side (commands / metric / budget) from the
    run's ACTUAL error + output + the repo's files. No failure-type routing and no per-repo hard-code
    -- whatever is wrong is corrected from the real run evidence. Returns a repaired Benchmark or None
    (no change). The trusted-eval boundary (editable/protected) is never touched."""
    # Prefer the richer retained output (the full trailing traces) over the 800-char tail so the
    # LLM sees the ACTUAL metric-bearing lines -- e.g. SOL's nested `evaluation.performance.latency_ms`
    # trace, which a short tail can cut off, leaving the metric name un-fixable.
    evidence = base_res.output or base_res.stdout_tail or ""
    return _correct_run(pristine, benchmark, base_res.error or "", evidence, request, llm)


def _heal_sandbox_baseline(benchmark, base_res, *, base_ws, pristine, run_variant_fn, request, llm,
                           cuda_device, emit, where):
    """Bounded self-heal for a baseline that RAN in a sandbox but produced no parseable metric --
    almost always a metric-NAME mismatch (the discovery agent named the metric differently than the
    trusted runner reads it: SOL's inferred `kernel_latency_us` vs the trace's actual
    `evaluation.performance.latency_ms`). Re-infer the execution side from the ACTUAL output; when
    ONLY the metric name/path changed, RE-PARSE the retained output instead of re-running the
    (tens-of-minutes) sandbox baseline. Returns the (possibly repaired) ``(benchmark, base_res)``.

    The host and known-image baseline paths already loop like this inline; the escalate/force-sandbox
    paths call HERE so all four share one metric-name self-heal (with the re-parse fast path)."""
    rnd = 0
    while (not base_res.valid) and rnd < _MAX_SETUP_REPAIRS:
        repaired = _heal_once(pristine, benchmark, base_res, request, llm)
        if repaired is None:
            break
        rnd += 1
        metric_only = (repaired.prepare_cmd == benchmark.prepare_cmd
                       and repaired.run_cmd == benchmark.run_cmd)
        benchmark = repaired
        emit("setup_repair", {"round": rnd, "in": where, "prepare_cmd": benchmark.prepare_cmd,
                              "run_cmd": benchmark.run_cmd, "metric": benchmark.metric,
                              "metric_only": metric_only})
        if metric_only:
            val = _runner.reparse_metric(base_res, benchmark.metric)
            if val is not None:   # corrected name found in the retained output -> no re-run needed
                base_res = _runner.RunResult(benchmark.metric, val, True,
                                             stdout_tail=base_res.stdout_tail, output=base_res.output)
                continue
            # retained output didn't contain the corrected metric -> fall through to a real re-run
        _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
        base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0, cuda_device=cuda_device)
    return benchmark, base_res


def _is_better(candidate: float, baseline: float, lower_is_better: bool) -> bool:
    return candidate < baseline if lower_is_better else candidate > baseline


def _report(benchmark, base_res, cand_res, patch, verdict, base_rep=None, cand_rep=None) -> str:
    protected = "、".join(benchmark.protected_files) or "(无)"
    trusted = (f"可信重评(仓库自带评测器在固定验证分片上重算,agent 自报数不采信)"
               if benchmark.trusted_eval_cmd else "仓库自带评测器 + 固定验证分片")
    # reproduce is only a genuine SEED swap when the script actually reads a seed env var;
    # otherwise it is an honest same-seed rerun (checks run-to-run variance, not seed-robustness).
    seeded = bool(benchmark.seed_env)
    r0, r1 = ("seed0", "seed1") if seeded else ("run0", "run1")
    lines = [
        "# 仓库改进诚实报告 (repo-improve)", "",
        f"## 基准\n{benchmark.name} — {benchmark.repo_url}",
        f"指标 {benchmark.metric}(越低越好={benchmark.lower_is_better}), 预算 {benchmark.budget_s}s", "",
        "## 基线(未改动仓库)",
        f"{r0}: {benchmark.metric} = {base_res.metric_value}  (valid={base_res.valid})",
    ]
    if base_rep is not None:
        lines.append(f"{r1}: {benchmark.metric} = {base_rep.metric_value}  (valid={base_rep.valid})")
    lines += ["", "## 改进(agent 受限补丁)",
              f"改动: {len(patch.get('edits') or [])} 处定向编辑,"
              f"文件: {'、'.join((patch.get('patched_sources') or {}).keys()) or '(无)'}",
              f"理由: {patch.get('rationale') or '(无)'}",
              f"{r0}: {benchmark.metric} = {cand_res.metric_value if cand_res else None}  "
              f"(valid={cand_res.valid if cand_res else False})"]
    if cand_rep is not None:
        lines.append(f"{r1}: {benchmark.metric} = {cand_rep.metric_value}  (valid={cand_rep.valid})")
    repro_note = (f"换 seed(环境变量 {benchmark.seed_env})复现" if seeded
                  else "同种子重跑复现(检验运行间方差;该仓库无 seed 钩子可切)")
    lines += ["", "## 判决", f"{verdict['verdict']} — {verdict['reason']}", "",
              "## 诚实声明",
              f"改进由可信层用{trusted}判分;agent 只改自身受限文件;"
              f"受信文件({protected})每次从原始克隆恢复,agent 动不了。数字为真跑所得,{repro_note}。"
              + ("" if benchmark.trusted_eval_cmd else
                 " 残余(留后续):当前采信运行脚本打印的可信指标,checkpoint 级独立重评为进一步加固。")]
    return "\n".join(lines)


def run_repo_improve(llm, request, *, workroot, emit=None, run_variant_fn=None,
                     clone_fn=None, reproduce=True, cuda_device=None,
                     boundary_overrides=None, advice: str = "",
                     sandbox_escalation: bool = True, setup_memory=None,
                     force_sandbox: bool | None = None,
                     execution_phase: str = "combined",
                     frozen_baseline: dict | None = None) -> dict:
    """Run the full repo-improvement flow for ``request``. Returns a result dict with the
    benchmark, baseline/candidate metrics, the patch, verdict and report. Never raises on
    an expected failure (unmatched request, clone/run failure, bad patch) -- those become an
    honest non-improvement verdict. ``boundary_overrides`` are USER-CONFIRMED boundary fields
    (run_cmd/metric/budget/editable...) that win over the LLM's inference. ``advice`` is an
    optional MOSS suggestion (what to change / how to fix a prior failure) fed to the patcher.

    ``sandbox_escalation`` (tier 3) lets a baseline that host-side healing could not fix be
    handed to an agent in a GPU sandbox; the rest of the run then continues inside that
    sandbox. Set it False to keep a run strictly host-side and fail honestly instead.

    ``force_sandbox`` makes the improve step ALWAYS go through the in-container OpenHands agent
    (item 6 route), even for a repo whose baseline ran fine on the host (minGPT/nanoGPT): an
    agent-capable sandbox is opened and BOTH baseline and candidate run in it. Defaults to the
    ``AUTORESEARCH_FORCE_SANDBOX`` env flag so the route can be turned on without a code change.

    Thin wrapper whose only job is guaranteeing the escalated sandbox is torn down on EVERY
    exit path -- including the early returns and exceptions in the flow below.
    """
    if execution_phase not in {"combined", "baseline", "method"}:
        raise ValueError(f"unknown execution_phase: {execution_phase}")
    if execution_phase == "method":
        try:
            frozen_value = float((frozen_baseline or {}).get("metric_value"))
        except (TypeError, ValueError):
            frozen_value = None
        if frozen_value is None or not math.isfinite(frozen_value):
            raise ValueError("method phase requires a finite metric_value in frozen_baseline")
        frozen_baseline = dict(frozen_baseline or {})
        frozen_baseline["metric_value"] = frozen_value
    if force_sandbox is None:
        force_sandbox = os.environ.get("AUTORESEARCH_FORCE_SANDBOX", "") not in ("", "0", "false")
    ctx: dict = {"close": None}
    try:
        return _run_repo_improve(
            llm, request, workroot=workroot, emit=emit, run_variant_fn=run_variant_fn,
            clone_fn=clone_fn, reproduce=reproduce, cuda_device=cuda_device,
            boundary_overrides=boundary_overrides, advice=advice,
            sandbox_escalation=sandbox_escalation, setup_memory=setup_memory,
            force_sandbox=force_sandbox, execution_phase=execution_phase,
            frozen_baseline=frozen_baseline, _ctx=ctx,
        )
    finally:
        if ctx["close"] is not None:
            ctx["close"]()


def _run_repo_improve(llm, request, *, workroot, emit=None, run_variant_fn=None,
                      clone_fn=None, reproduce=True, cuda_device=None,
                      boundary_overrides=None, advice: str = "",
                      sandbox_escalation: bool = True, setup_memory=None,
                      force_sandbox: bool = False, execution_phase: str = "combined",
                      frozen_baseline: dict | None = None, _ctx=None) -> dict:
    """The actual flow. ``_ctx["close"]`` is set when a sandbox is opened so the wrapper
    above can tear it down no matter how this returns."""
    _ctx = _ctx if _ctx is not None else {}
    emit = emit or (lambda event, payload: None)
    run_variant_fn = run_variant_fn or _runner.run_variant
    clone_fn = clone_fn or _clone.clone_repo

    # 1. discover the benchmark ("AI finds the repo")
    benchmark = discover(request, llm)
    if benchmark is None:
        return {"error": "no benchmark matched the request", "request": request}
    emit("benchmark_selected", {"name": benchmark.name, "repo_url": benchmark.repo_url,
                                "metric": benchmark.metric, "budget_s": benchmark.budget_s})

    # 2. clone the real repo (trusted)
    pristine = os.path.join(workroot, "pristine")
    clone_fn(benchmark.repo_url, pristine)
    emit("cloned", {"repo_url": benchmark.repo_url, "dir": pristine})

    # 2b. DYNAMIC benchmark (no registered boundary): infer the trusted-eval boundary -- which
    #     files are editable vs the protected evaluator/data, the metric, run commands, budget,
    #     and (for repos that ship their own CUDA/heavy image) which image to run in -- from the
    #     freshly-cloned repo. ONE general path: no per-repo special-casing (SOL's JSON-embedded
    #     kernel + harness scoring is inferred the same way as a train.py repo). Registered
    #     benchmarks already carry their boundary.
    sandbox_image = None
    sandbox_env = None
    if not benchmark.editable_files:
        # One flaky LLM call must not kill the whole run: the SAME repo inferred fine on a
        # previous run and came back None on the next (seen live on tinygrad). Inference is
        # a single cheap call relative to a full improve round, so retry before giving up.
        stub = benchmark
        for _attempt in range(_MAX_INFER_ATTEMPTS):
            benchmark = infer_benchmark(pristine, request, stub, llm,
                                        overrides=boundary_overrides)
            if benchmark is not None and benchmark.editable_files:
                break
            emit("boundary_infer_retry", {"attempt": _attempt + 1})
        if benchmark is None or not benchmark.editable_files:
            # meeting 2026-07-25 item 4: don't refuse the task when a clean editable/protected
            # boundary can't be inferred -- degrade to a PERMISSIVE boundary (the repo's own code is
            # editable; a weak/empty run_cmd is repaired downstream by correct_execution_from_run).
            benchmark = infer_benchmark(pristine, request, stub, llm,
                                        overrides=boundary_overrides, permissive=True)
            if benchmark is None or not benchmark.editable_files:
                return {"error": "cloned repo has no editable code file to improve",
                        "request": request}
            emit("boundary_permissive_fallback", {"editable": list(benchmark.editable_files)})
        emit("boundary_inferred", {"editable": list(benchmark.editable_files),
                                   "protected": list(benchmark.protected_files),
                                   "metric": benchmark.metric, "run_cmd": benchmark.run_cmd,
                                   "image": benchmark.image})

    # 2c. If the repo needs its OWN image (inferred: a CUDA/CUTLASS Dockerfile a plain host can't
    #     provide), resolve it -- build it from the repo's Dockerfile (or reuse an existing tag) --
    #     and route the whole run into it. General: driven by benchmark.image, no repo name pinned.
    if benchmark.image:
        sandbox_image = _resolve_repo_image(benchmark.image, pristine, emit)
        if sandbox_image is None:
            emit("repo_image_unavailable", {"spec": benchmark.image})  # fall back to default sandbox

    baseline_sources = _clone.read_editable(pristine, benchmark)
    program_md = ""
    pm = Path(pristine) / "program.md"
    if pm.exists():
        program_md = pm.read_text(encoding="utf-8")

    # RECALL: a setup tier 3 already proved for THIS repo under THIS budget+GPU. Only a
    # HINT -- we still run it, and a repo that has moved on since simply fails and falls
    # through to the healing ladder below, exactly as if we had never remembered anything.
    hit = None
    if setup_memory is not None:
        hit = recall_setup(setup_memory, repo_url=benchmark.repo_url,
                           budget_s=benchmark.budget_s, gpu=cuda_device)
        if hit:
            benchmark = _dc.replace(
                benchmark, prepare_cmd=hit["prepare_cmd"], run_cmd=hit["run_cmd"],
                metric=hit["metric"], lower_is_better=bool(hit["lower_is_better"]))
            emit("setup_recalled", {"repo_url": benchmark.repo_url, "run_cmd": benchmark.run_cmd,
                                    "recorded_at": hit.get("recorded_at"),
                                    "verified_in": hit.get("verified_in", "sandbox"),
                                    "previous_metric": hit.get("evidence", {}).get("metric_value")})
    if execution_phase == "method":
        frozen = frozen_baseline or {}
        benchmark = _dc.replace(
            benchmark,
            metric=str(frozen.get("metric_name") or benchmark.metric),
            lower_is_better=(bool(frozen["lower_is_better"])
                             if "lower_is_better" in frozen else benchmark.lower_is_better),
        )
    # A recalled setup verified inside a sandbox uses container-only paths (a ``.venv`` this
    # host lacks, a JIT it can't run); trying it on the host first is a guaranteed failure that
    # burns the repair ladder before we re-open the sandbox anyway. Route it straight to its
    # image instead -- the commands are already known, so no agent re-discovery is needed.
    # (Legacy entries carry no ``verified_in``; every remembered setup so far was sandbox-proven,
    # so absent means sandbox.)
    recalled_sandbox = bool(hit) and hit.get("verified_in", "sandbox") == "sandbox" \
        and sandbox_image is None
    # The frozen image of that verified container, if we have one. WITHOUT it, the two
    # remembered commands alone can't rebuild what the agent installed at the container level
    # (a compiler, an apt lib), so the recall would only go stale -- skip the image run and
    # re-discover directly. WITH it, the recall is reliable (see run_in_known_image below).
    recalled_image = hit.get("image") if recalled_sandbox else None

    # 3. pristine baseline (runs prepare = data + tokenizer, cached for the candidate).
    base_ws = os.path.join(workroot, "baseline_ws")
    if execution_phase == "method":
        base_res = _runner.RunResult(
            benchmark.metric, frozen_baseline["metric_value"], True)
    elif sandbox_image is not None:
        # This repo can ONLY run in its own image (SOL needs nvcc/CUTLASS the host lacks), so
        # trying on the host + text-healing it would only waste LLM calls before failing. Skip
        # straight to sandbox escalation with a placeholder-invalid baseline.
        base_res = _runner.RunResult(benchmark.metric, None, False, error="requires sandbox image")
    elif recalled_sandbox:
        # Sandbox-verified recall: don't touch the host -- go straight to the known-image run below.
        base_res = _runner.RunResult(benchmark.metric, None, False, error="recalled sandbox setup")
    else:
        _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
        base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0, cuda_device=cuda_device)
        # transient-retry: the UNCHANGED baseline is deterministic in principle, so a single failure
        # is almost always transient (partial data download, a busy host, a flaky import). Re-prepare
        # a fresh workspace and try once more before declaring the whole round dead.
        if not base_res.valid:
            emit("baseline_retry", {"error": base_res.error})
            _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
            base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0, cuda_device=cuda_device)
    # SELF-HEAL (phase 1b): a baseline that STILL fails is usually a SETUP gap the transient retry
    # can't fix -- missing data the script reads, deps/package not installed, a native/CUDA kernel
    # that must be built, or a repo that requires Docker. Let correct_execution_from_run READ the actual error +
    # the repo's setup/build files and propose a corrected prepare_cmd/run_cmd, then re-prepare +
    # retry. Bounded. The repaired benchmark (setup only -- metric/boundary unchanged) is reused for
    # the candidate too, so both variants run in the same fixed environment.
    repair_round = 0
    while (not base_res.valid) and sandbox_image is None and not recalled_sandbox \
            and repair_round < _MAX_SETUP_REPAIRS:
        repaired = _heal_once(pristine, benchmark, base_res, request, llm)
        if repaired is None:
            break  # no new execution-side proposal -> stop (honest fail, not an endless loop)
        repair_round += 1
        benchmark = repaired
        emit("setup_repair", {"round": repair_round, "prepare_cmd": benchmark.prepare_cmd,
                              "run_cmd": benchmark.run_cmd, "metric": benchmark.metric})
        _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
        base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0, cuda_device=cuda_device)

    # REPO WITH ITS OWN IMAGE (inferred, e.g. a CUDA/CUTLASS kernel benchmark): open that image
    # and run the inferred prepare/run commands in it. No agent discovery needed. The commands are
    # LLM-inferred so they can be slightly off (e.g. `pip` when the image only has `uv`); the SAME
    # the self-heal the host path gets now runs INSIDE the image -- read the error + the
    # repo's setup files, correct prepare_cmd/run_cmd, re-run in the same container. Bounded.
    if (not base_res.valid) and sandbox_image is not None:
        ran = run_in_known_image(
            workroot=workroot, benchmark=benchmark, cuda_device=cuda_device,
            emit=emit, server_image=sandbox_image, env=sandbox_env,
        )
        if ran is not None:
            benchmark, run_variant_fn, _ctx["close"], _ctx["backend"], _ctx["handle"] = ran
            _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
            base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0,
                                      cuda_device=cuda_device)
            img_repair = 0
            while (not base_res.valid) and img_repair < _MAX_SETUP_REPAIRS:
                repaired = _heal_once(pristine, benchmark, base_res, request, llm)
                if repaired is None:
                    break
                img_repair += 1
                benchmark = repaired
                emit("setup_repair", {"round": img_repair, "in": "image",
                                      "prepare_cmd": benchmark.prepare_cmd,
                                      "run_cmd": benchmark.run_cmd, "metric": benchmark.metric})
                _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
                base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0,
                                          cuda_device=cuda_device)

    # RECALLED SANDBOX SETUP: the setup was proved in a prior run AND frozen into an image,
    # so we skip the ~20-min discovery agent and just re-open THAT image and rerun the known
    # commands -- the fast path the whole setup memory exists for. The frozen image carries the
    # container-level state (compilers, apt libs, caches) the commands can't rebuild, so unlike
    # a bare base image it reproduces the verified environment. If it still fails (repo moved,
    # image gone) we drop it and fall through to full re-discovery, as the recall contract promises.
    if (not base_res.valid) and recalled_image:
        ran = run_in_known_image(
            workroot=workroot, benchmark=benchmark, cuda_device=cuda_device,
            emit=emit, server_image=recalled_image,
        )
        if ran is not None:
            benchmark, run_variant_fn, _ctx["close"], _ctx["backend"], _ctx["handle"] = ran
            _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
            base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0,
                                      cuda_device=cuda_device)
        if not base_res.valid:
            emit("setup_recall_stale", {"repo_url": benchmark.repo_url, "image": recalled_image})
            if _ctx.get("close"):
                _ctx["close"]()            # tear down the stale container before re-discovering
                _ctx["close"] = _ctx["backend"] = _ctx["handle"] = None
            recalled_sandbox = False       # let the escalation below re-remember the fresh setup

    # ESCALATE (tier 3): tiers 1-2 can only find what the repo SAYS. A setup that is not
    # textual at all -- a tokenizer some assert expects, a host that cannot JIT-compile --
    # only surfaces by RUNNING. Hand the repo to an agent in a GPU sandbox; if it finds a
    # setup that survives our hard judge, the rest of the run continues INSIDE that sandbox
    # (a setup is only valid in the environment it was verified in).
    if (not base_res.valid) and sandbox_image is None and sandbox_escalation:
        escalated = escalate_to_sandbox(
            workroot=workroot, pristine=pristine, benchmark=benchmark, request=request,
            cuda_device=cuda_device, llm=llm, emit=emit, server_image=sandbox_image,
        )
        if escalated is not None:
            benchmark, run_variant_fn, _ctx["close"], _ctx["backend"], _ctx["handle"] = escalated
            _clone.prepare_variant_workspace(pristine, base_ws, benchmark, {})
            base_res = run_variant_fn(benchmark, base_ws, prepare=True, seed=0,
                                      cuda_device=cuda_device)
            # SELF-HEAL (the escalate path was MISSING this -- host/known-image both have it): an
            # escalated baseline that RUNS but has no parseable metric is a metric-NAME mismatch;
            # re-infer + re-parse (no re-run) via the shared helper.
            benchmark, base_res = _heal_sandbox_baseline(
                benchmark, base_res, base_ws=base_ws, pristine=pristine,
                run_variant_fn=run_variant_fn, request=request, llm=llm,
                cuda_device=cuda_device, emit=emit, where="sandbox")
            # REMEMBER: only once the escalated setup produced a real baseline here. First
            # FREEZE the verified container into an image so a later recall reruns the SAME
            # environment (the commands alone can't rebuild the agent's container-level
            # installs). The image name is stored as evidence alongside the metric value; if
            # the commit failed, image=None and the next recall simply re-discovers.
            if setup_memory is not None and base_res.valid:
                image_tag = setup_image_tag(benchmark.repo_url, benchmark.budget_s, cuda_device)
                committed = False
                try:
                    committed = bool(_ctx["backend"].commit_sandbox(_ctx["handle"], image_tag))
                except Exception:  # noqa: BLE001 -- freezing is best-effort
                    committed = False
                emit("setup_image_committed", {"image": image_tag, "ok": committed})
                remember_setup(setup_memory, repo_url=benchmark.repo_url,
                               prepare_cmd=benchmark.prepare_cmd, run_cmd=benchmark.run_cmd,
                               metric=benchmark.metric,
                               lower_is_better=benchmark.lower_is_better,
                               metric_value=base_res.metric_value,
                               budget_s=benchmark.budget_s, gpu=cuda_device,
                               verified_in="sandbox",
                               image=image_tag if committed else None)
                emit("setup_remembered", {"repo_url": benchmark.repo_url,
                                          "metric_value": base_res.metric_value,
                                          "image": image_tag if committed else None})
    # FORCE-SANDBOX (item 6 mandate: SOL / nanoGPT / minGPT all improve via the in-container OpenHands
    # agent, never the one-shot LLM patcher). If a valid baseline was obtained WITHOUT a sandbox handle
    # (a host CPU repo), open an agent-capable sandbox now and RE-RUN the baseline in it so baseline and
    # candidate share one environment; the improve step below then routes through build_sandbox_propose.
    # A metric-name mismatch in the container is healed by the same shared self-heal (re-parse fast path).
    if (force_sandbox and execution_phase != "method" and base_res.valid
            and _ctx.get("handle") is None and sandbox_escalation):
        opened = open_agent_sandbox(workroot=workroot, benchmark=benchmark, cuda_device=cuda_device,
                                    emit=emit, env=sandbox_env)
        if opened is not None:
            host_base_res, host_benchmark = base_res, benchmark
            _orig_run_variant_fn = run_variant_fn      # host runner, restored if the sandbox can't repro
            sb_benchmark, run_variant_fn, _ctx["close"], _ctx["backend"], _ctx["handle"] = opened
            _clone.prepare_variant_workspace(pristine, base_ws, sb_benchmark, {})
            sb_base_res = run_variant_fn(sb_benchmark, base_ws, prepare=True, seed=0,
                                         cuda_device=cuda_device)
            sb_benchmark, sb_base_res = _heal_sandbox_baseline(
                sb_benchmark, sb_base_res, base_ws=base_ws, pristine=pristine,
                run_variant_fn=run_variant_fn, request=request, llm=llm,
                cuda_device=cuda_device, emit=emit, where="force_sandbox")
            emit("force_sandbox", {"host_baseline": host_base_res.metric_value,
                                   "sandbox_baseline": sb_base_res.metric_value,
                                   "valid": sb_base_res.valid})
            if sb_base_res.valid:
                benchmark, base_res = sb_benchmark, sb_base_res   # both variants now run in-sandbox
            else:
                # the sandbox couldn't reproduce the baseline -> tear it down and keep the honest HOST
                # result + host-side improve route (never fail a good baseline just to force the route).
                if _ctx.get("close"):
                    _ctx["close"]()
                _ctx["close"] = _ctx["backend"] = _ctx["handle"] = None
                run_variant_fn = _orig_run_variant_fn
                benchmark, base_res = host_benchmark, host_base_res

    if execution_phase == "baseline":
        try:
            baseline_value = float(base_res.metric_value)
        except (TypeError, ValueError):
            baseline_value = None
        if not base_res.valid or baseline_value is None or not math.isfinite(baseline_value):
            base_res = _dc.replace(
                base_res, metric_value=None, valid=False,
                error="baseline phase requires a finite baseline value")
        else:
            base_res = _dc.replace(base_res, metric_value=baseline_value)

    emit("baseline_done", {"metric": benchmark.metric, "value": base_res.metric_value,
                           "valid": base_res.valid, "error": base_res.error})
    if not base_res.valid:
        verdict = {"verdict": "fail", "improved": False,
                   "reason": f"基线未能跑通: {base_res.error}"}
        emit("verdict", verdict)
        return {"benchmark": benchmark.name, "baseline": base_res, "candidate": None,
                "metric": benchmark.metric, "lower_is_better": benchmark.lower_is_better,
                "patch": None, "verdict": verdict,
                "report": _report(benchmark, base_res, None, {}, verdict)}

    if execution_phase == "baseline":
        verdict = {"verdict": "pass", "improved": False,
                   "reason": f"baseline measured: {benchmark.metric}={base_res.metric_value}"}
        emit("verdict", verdict)
        return {"benchmark": benchmark.name, "baseline": base_res, "candidate": None,
                "metric": benchmark.metric, "lower_is_better": benchmark.lower_is_better,
                "patch": None, "verdict": verdict, "report": verdict["reason"]}

    # 4. IMPROVE. ONE general loop for every repo when a sandbox is available (SOL / nanochat /
    #    CPU all reach here inside their sandbox): an agent iterates editing ONLY the editable
    #    files, and we re-score each attempt in a clean workspace with protected files restored
    #    (AIDE-style search + MLE-bench's 'agent never controls scoring'). No per-task patch logic
    #    -- SOL's JSON-embedded kernel is edited by the agent itself, not by our find/replace.
    #    When there is no sandbox (host baseline, e.g. minGPT, or no docker in a unit test) we fall
    #    back to the one-shot propose_patch.
    cand_ws = os.path.join(workroot, "candidate_ws")
    if _ctx.get("backend") is not None:
        def _rescore(bench, edits, label):
            ws = os.path.join(workroot, f"opt_{label}")
            _clone.prepare_variant_workspace(pristine, ws, bench, edits)   # protected restored
            r = run_variant_fn(bench, ws, prepare=True, seed=0, cuda_device=cuda_device)
            return r.metric_value, r.valid
        # item 6 (repo path): with a live sandbox handle, an in-container OpenHands agent edits the
        # editable file(s) INCREMENTALLY (small diffs it can sanity-run) -- instead of one LLM call
        # regenerating the whole file as JSON, which timed out silently on large files (SOL/cifar
        # 0-edit rounds). Fall back to the one-shot LLM proposer when there is no handle (host
        # baseline, e.g. minGPT, or a unit test without docker).
        # meeting #3 (C row): BEFORE the agent edits, run a lightweight 3-angle implementation
        # discussion (impact / simplicity / pitfalls) + synthesis into a short plan, and hand it to
        # the agent as advice. Best-effort: "" on any failure -> the agent starts cold as before.
        impl_advice = discuss_implementation(llm, benchmark, request, base_res.metric_value, emit=emit)
        _handle = _ctx.get("handle")
        if _handle is not None:
            # Agent PER-ROUND task budget != per-TRAINING cap. The agent needs to WRITE + run a
            # training (each hard-capped at benchmark.budget_s via the prompt's `timeout` wrapper) +
            # observe + fix -- so a per-round budget equal to one training (benchmark.budget_s) is too
            # short (the agent times out mid-training every round, leaving an orphaned run to drag on;
            # live: a sandbox chargpt ran ~45 min after its 300s agent task "timed out"). Give it room
            # for ~3 capped runs + thinking, bounded so a round can't run away.
            _agent_round_budget = min(1800, 3 * benchmark.budget_s + 300)
            _propose = build_sandbox_propose(_ctx["backend"], _handle, benchmark, pristine, workroot,
                                             llm=llm, budget_s=_agent_round_budget, emit=emit,
                                             advice=impl_advice)
        else:
            _propose = build_llm_propose(llm, benchmark, emit)
        opt = optimize_in_sandbox(benchmark, baseline_metric=base_res.metric_value,
                                  rescore=_rescore, propose=_propose,
                                  initial_editable=baseline_sources,
                                  max_rounds=_MAX_OPTIMIZE_ROUNDS)
        emit("optimize_done", {"improved": opt.improved, "best": opt.best_metric,
                               "rounds": opt.rounds})
        # when nothing beat the baseline, still surface the BEST edit that RAN + its real metric
        # (the agent tried and lost -- the user needs the number, not a bare None).
        edits = opt.best_edits or opt.best_attempt_edits or {}
        patch = {"edits": [{"file": f} for f in edits], "patched_sources": edits or None,
                 "rationale": "agent iterative in-sandbox optimization",
                 "error": None if opt.improved else "no improving edit beat the baseline"}
        _clone.prepare_variant_workspace(pristine, cand_ws, benchmark, edits)
        if opt.improved:
            cand_res = _runner.RunResult(benchmark.metric, opt.best_metric, True)
        elif opt.best_attempt_metric is not None:
            # ran, produced a real metric, but didn't beat baseline -> valid number, not improved.
            cand_res = _runner.RunResult(benchmark.metric, opt.best_attempt_metric, True,
                                         error="did not beat baseline")
        else:
            cand_res = _runner.RunResult(benchmark.metric, None, False,
                                         error="no valid edit produced")
    else:
        # FALLBACK (no sandbox): one-shot patch. The agent decides what to change; MOSS may advise.
        patch = propose_patch(llm, benchmark, program_md, baseline_sources,
                              base_res.metric_value, goal=request, advice=advice)
        emit("patch_ready", {"n_edits": len(patch.get("edits") or []),
                             "files": list((patch.get("patched_sources") or {}).keys()),
                             "rationale": patch.get("rationale"), "error": patch.get("error")})
        if patch.get("patched_sources") is None:
            verdict = {"verdict": "fail", "improved": False,
                       "reason": f"agent 补丁未能应用: {patch.get('error')}"}
            emit("verdict", verdict)
            return {"benchmark": benchmark.name, "baseline": base_res, "candidate": None,
                    "metric": benchmark.metric, "lower_is_better": benchmark.lower_is_better,
                    "patch": patch, "verdict": verdict,
                    "report": _report(benchmark, base_res, None, patch, verdict)}
        _clone.prepare_variant_workspace(pristine, cand_ws, benchmark, patch["patched_sources"])
        cand_res = run_variant_fn(benchmark, cand_ws, prepare=True, seed=0, cuda_device=cuda_device)
    emit("candidate_done", {"metric": benchmark.metric, "value": cand_res.metric_value,
                            "valid": cand_res.valid, "error": cand_res.error})

    # 5b. reproduce at a second seed: baseline AND candidate, so a win must hold twice
    base_rep = cand_rep = None
    if reproduce and cand_res.valid and execution_phase != "method":
        base_rep = run_variant_fn(benchmark, base_ws, prepare=False, seed=1, cuda_device=cuda_device)
        cand_rep = run_variant_fn(benchmark, cand_ws, prepare=False, seed=1, cuda_device=cuda_device)
        emit("reproduce_done", {"baseline": base_rep.metric_value,
                                "candidate": cand_rep.metric_value,
                                "valid": bool(base_rep.valid and cand_rep.valid)})

    # 6. honest verdict: candidate must beat baseline on the primary seed AND (if we
    #    reproduced) on the second seed too -- a one-seed win that doesn't reproduce fails.
    lower = benchmark.lower_is_better
    # honest run labels: a real seed swap only when the script reads a seed env var (else rerun)
    r0, r1 = ("seed0", "seed1") if benchmark.seed_env else ("run0", "run1")
    primary_ok = cand_res.valid and _is_better(cand_res.metric_value, base_res.metric_value, lower)
    if not cand_res.valid:
        verdict = {"verdict": "fail", "improved": False, "reason": f"改后未能跑通: {cand_res.error}"}
    elif not primary_ok:
        verdict = {"verdict": "fail", "improved": False,
                   "reason": f"未改进: {benchmark.metric} {base_res.metric_value} -> {cand_res.metric_value}"}
    elif reproduce and execution_phase != "method":
        rep_ok = (base_rep and cand_rep and base_rep.valid and cand_rep.valid
                  and _is_better(cand_rep.metric_value, base_rep.metric_value, lower))
        if rep_ok:
            verdict = {"verdict": "pass", "improved": True,
                       "reason": f"改进且复现: {benchmark.metric} {base_res.metric_value}->"
                                 f"{cand_res.metric_value} ({r0}), "
                                 f"{base_rep.metric_value}->{cand_rep.metric_value} ({r1})"}
        else:
            # P6: record the ACTUAL reproduce-run numbers, not just "did not replicate", so the
            # result is auditable after the fact (how far off was the second run?).
            rep_vals = (f"{base_rep.metric_value}->{cand_rep.metric_value}"
                        if (base_rep and cand_rep) else "run failed")
            verdict = {"verdict": "fail", "improved": False,
                       "reason": f"{r0} 改进({base_res.metric_value}->{cand_res.metric_value})"
                                 f"但{r1}未复现({rep_vals})(不稳健)"}
    else:
        verdict = {"verdict": "pass", "improved": True,
                   "reason": f"改进: {benchmark.metric} {base_res.metric_value} -> {cand_res.metric_value}"}
    emit("verdict", verdict)

    report = _report(benchmark, base_res, cand_res, patch, verdict, base_rep, cand_rep)
    emit("report", {"report": report})
    return {"benchmark": benchmark.name, "baseline": base_res, "candidate": cand_res,
            "metric": benchmark.metric, "lower_is_better": benchmark.lower_is_better,
            "baseline_reproduce": base_rep, "candidate_reproduce": cand_rep,
            "patch": patch, "verdict": verdict, "report": report}


__all__ = ["run_repo_improve"]
