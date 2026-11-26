"""OpenHands-agent proposer for the repo improve loop (meeting 2026-07-25 item 6, repo path).

Drop-in replacement for ``build_llm_propose``'s ``propose(current, baseline, best) -> {path: new}``,
but the "search step" is an in-container OpenHands agent that edits the repo's source file(s)
INCREMENTALLY instead of a single LLM call that must regenerate the ENTIRE editable file as JSON.

Why: the one-shot proposer was untenable for a large editable file -- glm couldn't finish rewriting
a 10KB JSON-embedded CUDA kernel within the stream deadline, so every round silently produced 0
edits. An agent that makes small, targeted edits (and can run the code to sanity-check) is the
right tool.

ROBUST READ-BACK (fix after live run): the agent's edits are captured by DIFFING the whole seeded
workspace against pristine and keeping every CHANGED source file that is NOT protected -- rather
than reading only ``benchmark.editable_files``. A live glm run inferred a WRONG editable path
(``SOL-ExecBench/kernels/rmsnorm/`` -- absent; the repo actually uses ``examples/cuda_cpp/rmsnorm/``)
so pointing the agent at that path and reading only it yielded 0 edits even though the plumbing
worked. Diffing-vs-pristine decouples from the fragile editable-path inference.

Trusted-eval boundary is PRESERVED: protected files (the evaluator + data + benchmark definition)
are EXCLUDED from the read-back, so an agent that edits the scorer can't get that edit into the
patch -- the rescore then applies the patch to a clean pristine tree, leaving protected files
pristine at scoring time. Fully defensive: ANY failure returns ``{}`` (round skipped), never worse
than the old silent-empty behavior.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("autoresearch.repo_improve.sandbox_propose")

from autoresearch.experiments.repo_improve import clone as _clone
from autoresearch.experiments.repo_improve.optimize_loop import build_llm_propose

# The in-container path the agent naturally works in: OpenHands' Conversation workspace is
# /workspace and its file_editor is /workspace-relative, so the agent edits HERE -- NOT at an
# external host-mounted path (a live run proved it edited /workspace/*, so reading back a mounted
# agent_round dir captured nothing). We put the repo here, let the agent edit it, and export it back.
_REPO_IN = "/workspace/repo"

# Build artifacts / lockfiles the agent's `uv sync` etc. touch -- not source improvements.
_SKIP_NAMES = {"uv.lock", "poetry.lock", "package-lock.json", "Cargo.lock", ".DS_Store",
               "ws_export.tgz"}
# Binary / data / compiled extensions -- never a readable source edit.
_SKIP_EXT = {".bin", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".idx", ".safetensors", ".pkl",
             ".pyc", ".so", ".o", ".a", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg"}
_SKIP_DIR_PARTS = {".git", "__pycache__", ".cache", ".nv", "node_modules", ".venv",
                   # run OUTPUTS (checkpoints, logs) an agent's sanity-run writes -- not source edits
                   "out", "outputs", "checkpoints", "runs", "wandb", "logs"}


def _is_protected(rel: str, protected) -> bool:
    """True iff ``rel`` is (or is under) a protected file/dir -- the trusted evaluator/data that the
    agent must never influence."""
    for p in protected:
        p = str(p).strip().rstrip("/")
        if p and (rel == p or rel.startswith(p + "/")):
            return True
    return False


def _editable_hint(benchmark, ws_abs: str) -> str:
    """The editable paths that ACTUALLY exist in the workspace, as a location HINT (not a hard
    restriction -- inference may name a wrong path)."""
    hints = [f for f in benchmark.editable_files
             if os.path.exists(os.path.join(ws_abs, f))]
    return ", ".join(hints)


def _agent_task(benchmark, repo_dir: str, baseline_metric, best_metric, hint: str = "",
                advice: str = "") -> str:
    goal = "LOWER" if benchmark.lower_is_better else "RAISE"
    lines = [
        f"The repository is at {repo_dir}. Improve it to {goal} the metric '{benchmark.metric}' "
        f"({'lower' if benchmark.lower_is_better else 'higher'} is better).",
        f"Baseline (unmodified repo): {benchmark.metric} = {baseline_metric}.",
    ]
    if best_metric is not None:
        lines.append(f"Best so far: {benchmark.metric} = {best_metric}. Beat it.")
    lines += ["", f"The repo is run with: {benchmark.run_cmd}"]
    if hint:
        lines.append(f"The implementation to improve is most likely in: {hint} (a hint -- if the "
                     "real code lives elsewhere under the repo, edit there instead).")
    if advice:
        # meeting #3: the C-row implementation discussion's synthesized plan, so the agent starts
        # from a considered approach instead of a cold prompt.
        lines += ["", "A short implementation plan was discussed for this task -- follow it unless "
                  "the code clearly calls for something else:", advice]
    lines += [
        f"Edit the relevant SOURCE file(s) under {repo_dir} to improve the metric. Do NOT modify the "
        "evaluator, scoring harness, dataset, or benchmark definition -- those are restored from a "
        "clean copy before scoring, so changing them is useless and will be discarded.",
        # P8 (safety guard): a destructive rewrite that breaks the run scores nothing (even the
        # baseline can fail to run). Keep the code RUNNABLE at every step.
        "Keep the code RUNNABLE: make SMALL, targeted, reversible edits (e.g. a learning-rate "
        "schedule, a hyperparameter, a normalization) rather than restructuring the model or "
        "rewriting large files. If you change something substantial, run the code first to confirm "
        "it still executes and prints the metric before finishing.",
        # P7 (convergence): do not turn this into an unbounded hyperparameter search or blow up the
        # training scale -- the run is wall-clock budgeted, so a huge max_iters just gets cut off.
        "Do NOT run a long hyperparameter search or massively increase the training length/scale; "
        "the run has a fixed wall-clock budget. Make ONE or TWO concrete, well-motivated edits, "
        "sanity-check them, and FINISH -- do not keep exploring.",
        # FAIR WALL-CLOCK BUDGET (enforced, not advisory): the trusted evaluator re-runs BOTH the
        # baseline and the agent's edited code under a HARD budget_s cap (_capped_run), identical for
        # both -- so "improving" only by training longer scores nothing. Tell the agent the truth so
        # it stops burning wall-clock on long test runs (live: a sandbox chargpt ran ~45 min).
        "The final scoring re-runs baseline and your code under a HARD "
        f"{benchmark.budget_s}s wall-clock cap (same for both), so training LONGER scores no better "
        f"-- it is cut off at the cap. Test your code as `timeout {benchmark.budget_s}s {benchmark.run_cmd}`; "
        f"a run exceeding {benchmark.budget_s}s is killed and wasted. Improve the metric WITHIN "
        f"{benchmark.budget_s}s, not by training longer.",
        # P2 (terminal constraint): the sandbox terminal executes ONE command per call.
        "When you run shell commands, run ONE command per call (no '&&', ';', or multi-line "
        "scripts -- the terminal rejects multiple commands at once).",
    ]
    return "\n".join(lines)


def _collect_changed(ws: str, pristine: str, protected, *, cap_bytes: int = 400_000) -> dict[str, str]:
    """Every text source file under ``ws`` that DIFFERS from ``pristine`` (or is new), excluding
    protected files, VCS/cache/build noise, binaries, and lockfiles. These are the agent's edits."""
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_PARTS]
        for f in files:
            if f in _SKIP_NAMES or os.path.splitext(f)[1].lower() in _SKIP_EXT:
                continue
            ap = os.path.join(root, f)
            rel = os.path.relpath(ap, ws)
            if _is_protected(rel, protected):
                continue
            try:
                if os.path.getsize(ap) > cap_bytes:
                    continue
                new = open(ap, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            try:
                old = open(os.path.join(pristine, rel), encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                old = None  # a file the agent newly created
            if new.strip() and new != old:
                out[rel] = new
    return out


# The in-container agent runtime (OpenHands SDK) occasionally crashes mid-run -- observed live on a
# CUDA-image sandbox: ``RuntimeError: TmuxPanePool is not initialized or already closed`` + a remote
# conversation failure -- which the SDK does not recover from within the call. Re-open the whole
# agent round a couple of times; if it STILL crashes, fall back to the one-shot LLM proposer (which
# needs no in-container runtime) so a flaky agent server never permanently blocks a method attempt.
_AGENT_ATTEMPTS = 2


def build_sandbox_propose(backend, handle, benchmark, pristine: str, workroot: str,
                          *, llm=None, sandbox_llm=None, budget_s: int = 1800, emit=None,
                          advice: str = ""):
    """Return ``propose(current, baseline_metric, best_metric) -> {rel: new_content}`` backed by an
    in-container agent.

    The agent works in its Conversation workspace (``/workspace``); a live run showed it edits
    ``/workspace/*`` and never the external host-mounted round dir, so we PUT the seeded repo at
    ``/workspace/repo`` (via the mounted round dir), point the agent there, then EXPORT that dir
    back to the host and diff it against pristine. Only changed non-protected source files are
    kept -- the evaluator/data stay pristine at scoring time.

    Robustness (double insurance): each round retries the agent up to ``_AGENT_ATTEMPTS`` times on
    an SDK/infra crash; if it still crashes, and ``llm`` is given, it falls back to the one-shot LLM
    proposer for that round. Any remaining failure -> {} (honest no progress).
    """
    emit = emit or (lambda event, payload: None)
    protected = list(benchmark.protected_files)
    llm_fallback = build_llm_propose(llm, benchmark, emit) if llm is not None else None
    counter = {"i": 0}

    def _run_agent_round(seed, exported, baseline_metric, best_metric, feedback=""):
        """One agent attempt: (re)fill /workspace/repo, run the agent, export it back. Raises on any
        infra/SDK error so the caller can retry or fall back."""
        backend.execute_command(
            handle, f"rm -rf {_REPO_IN} && mkdir -p {_REPO_IN} && cp -a {seed}/. {_REPO_IN}/",
            timeout=300)
        hint = _editable_hint(benchmark, seed)
        # Meeting 2026-07-29: carry the previous round's diagnosis (找规律) into this round's advice so
        # the agent adjusts from what actually happened, not a cold re-attempt.
        round_advice = (advice + "\n\n" + feedback).strip() if feedback else advice
        task = _agent_task(benchmark, _REPO_IN, baseline_metric, best_metric, hint, round_advice)
        res = backend.run_agent_task(handle, task, llm=sandbox_llm, timeout=budget_s)
        backend.export_workspace(handle, exported, src=_REPO_IN)
        return getattr(res, "status", "") or ""

    def _propose(current, baseline_metric, best_metric, feedback=""):
        counter["i"] += 1
        seed = os.path.join(workroot, f"agent_round{counter['i']}")      # host, bind-mounted
        exported = os.path.join(workroot, f"agent_export{counter['i']}")  # host, export target
        _clone.prepare_variant_workspace(pristine, seed, benchmark, current)
        status = ""
        for attempt in range(_AGENT_ATTEMPTS):
            try:
                status = _run_agent_round(seed, exported, baseline_metric, best_metric, feedback)
                break
            except Exception as exc:  # noqa: BLE001 -- SDK/infra crash: retry, then fall back
                logger.warning("sandbox agent round %d crashed (attempt %d/%d): %s: %s",
                               counter["i"], attempt + 1, _AGENT_ATTEMPTS, type(exc).__name__, exc)
                emit("propose_result", {"ok": False, "reason": f"agent_error:{type(exc).__name__}",
                                        "attempt": attempt + 1})
                if attempt + 1 < _AGENT_ATTEMPTS:
                    continue
                # double insurance: the in-container agent keeps crashing -> one-shot LLM propose.
                if llm_fallback is not None:
                    logger.warning("sandbox agent round %d exhausted %d attempts -> one-shot LLM "
                                   "fallback (weak for large files)", counter["i"], _AGENT_ATTEMPTS)
                    emit("propose_result", {"ok": False, "reason": "fallback_to_llm_propose"})
                    return llm_fallback(current, baseline_metric, best_metric, feedback)
                logger.warning("sandbox agent round %d exhausted %d attempts, no llm fallback -> {}",
                               counter["i"], _AGENT_ATTEMPTS)
                return {}
        # capture EVERY changed non-protected source file (protected excluded -> scorer pristine).
        edits = _collect_changed(exported, pristine, protected)
        if not edits:
            # The agent "finished" but produced no edit. This is the SWALLOWED-crash case: the SDK
            # catches a terminal crash (TmuxPanePool) as an error OBSERVATION and lets the agent
            # stop, so run_agent_task doesn't raise -- but nothing was written. Treat it like a
            # crash: fall back to the one-shot LLM proposer so a method attempt still happens.
            logger.warning("sandbox agent round %d finished (status=%s) but exported NO changed "
                           "source file -> treating as swallowed-crash / no-op", counter["i"], status)
            emit("propose_result", {"ok": False, "reason": "agent_no_change", "status": status})
            if llm_fallback is not None:
                emit("propose_result", {"ok": False, "reason": "fallback_to_llm_propose",
                                        "after": "no_edit"})
                return llm_fallback(current, baseline_metric, best_metric)
            return {}
        emit("propose_result", {"ok": True, "files": list(edits.keys())[:10],
                                "n": len(edits), "status": status})
        return edits

    return _propose


__all__ = ["build_sandbox_propose"]
