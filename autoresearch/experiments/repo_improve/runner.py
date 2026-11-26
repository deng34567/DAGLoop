"""Trusted runner: prepare (once) + train under the budget + read the official metric.

The runner is trusted code. It runs the benchmark's ``prepare_cmd`` (data + tokenizer, per
isolated variant workspace) and ``run_cmd`` (trains under ``budget_s``) in a variant workspace,
then reads the metric the run reports. Because the variant workspace is pristine except for
the editable file (see ``clone.prepare_variant_workspace``), the metric is computed by the
benchmark's OWN trusted evaluator on its pinned validation data.

``executor`` is injected (defaults to ``subprocess.run``) so the orchestration is testable
without a GPU. Any failure fails CLOSED: ``valid=False`` with the captured error.

Residual (documented, hardened next): the editable ``train.py`` both trains AND prints the
metric via the trusted ``evaluate_bpb``; a maximally-adversarial train.py could compute the
honest value but print a different one. Closing that needs a separate trusted re-eval of the
saved checkpoint (load the candidate model class + pristine ``evaluate_bpb`` + pinned val
shard). The workspace/metric plumbing here is built to accept that trusted re-eval step.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from autoresearch.experiments.repo_improve.benchmark import Benchmark


@dataclass
class RunResult:
    metric_name: str
    metric_value: float | None
    valid: bool
    error: str | None = None
    stdout_tail: str = ""
    #: A LARGER slice of the run's real output, retained only when the metric couldn't be parsed.
    #: It feeds the self-heal (so the LLM sees the actual metric-bearing lines, not just an 800-char
    #: tail) AND lets a metric-NAME-only correction be applied by RE-PARSING this output instead of
    #: re-running an expensive training (a SOL baseline is ~tens of minutes). Empty on success/other
    #: failures where re-parse can't help.
    output: str = ""


#: Cap for ``RunResult.output``. Big enough to hold the trailing JSONL traces a run prints
#: (SOL emits one ``json.dumps`` trace per workload), small enough not to bloat what we pass around.
_OUTPUT_KEEP = 16000


def reparse_metric(res: "RunResult", new_metric: str) -> float | None:
    """Try to read ``new_metric`` out of a prior run's retained ``output`` -- WITHOUT re-running.

    Used when the self-heal changes ONLY the metric name/path (not the commands): the run already
    happened and its output is retained, so the corrected name can be parsed directly. Returns the
    value or ``None`` if the retained output doesn't contain it (then the caller must re-run)."""
    if not new_metric or not res.output:
        return None
    return _parse_metric(new_metric, res.output)


class _CappedTimeout(subprocess.CompletedProcess):
    """A CompletedProcess that also flags the run was cut off at the wall-clock cap (``timed_out``)
    while STILL carrying the partial stdout/stderr captured before the kill."""
    def __init__(self, cmd, returncode, stdout, stderr):
        super().__init__(cmd, returncode, stdout, stderr)
        self.timed_out = True


def _capped_run(cmd, *, cwd, shell, env, capture_output, text, timeout):
    """Like subprocess.run(timeout=...), but HARD-caps the whole process TREE at ``timeout``.

    Anti-cheat / fair-budget: ``subprocess.run(shell=True, timeout=...)`` on timeout only kills the
    direct child (the shell) -- the real training grandchild (``python train.py``) is orphaned and
    keeps running, so a method can quietly train PAST its budget and 'win' on extra compute (seen
    live: minGPT ran ~13 min on a '5 min' budget). Running in a fresh process GROUP and killing the
    whole group on timeout makes the wall-clock budget actually enforced -- both baseline and method
    get exactly the same real training time; an overrun is CUT OFF at the cap (never allowed to run
    longer).

    On a cap-kill we RETURN a ``_CappedTimeout`` carrying the partial output (not raise): a training
    script that runs unbounded (e.g. minGPT's chargpt.py) never exits on its own, so the fair result
    is 'the metric it had reached AT the shared wall-clock cap'. run_variant decides whether that
    partial output contains a usable metric; the budget is still equally enforced for both variants."""
    proc = subprocess.Popen(cmd, cwd=cwd, shell=shell, env=env,
                            stdout=subprocess.PIPE if capture_output else None,
                            stderr=subprocess.PIPE if capture_output else None,
                            text=text, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # kill the ENTIRE process group (shell + all training grandchildren), then reap + keep output.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        out, err = proc.communicate()
        return _CappedTimeout(cmd, proc.returncode, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


#: Statuses a per-workload trace may report and still count as a real measurement.
_OK_STATUSES = {"passed", "pass", "ok", "success", "succeeded"}


def _dig(obj, path: str):
    """Follow a dotted path through nested dicts; ``None`` if any hop is missing."""
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _find_status(obj) -> str | None:
    """First ``status`` string anywhere in the record (traces nest it under evaluation)."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "status" and isinstance(val, str):
                return val
            found = _find_status(val)
            if found is not None:
                return found
    return None


def _parse_jsonl_metric(metric: str, text: str) -> float | None:
    """Average a dotted-path value over the JSON records a run printed.

    Some benchmarks don't print ``metric: number`` at all -- SOL-ExecBench emits one JSON
    trace per workload with the number at ``evaluation.performance.latency_ms``. Naming the
    metric by its path lets those work with no special-casing in the pipeline.

    ANTI-CHEAT: a record whose status is not a passing one INVALIDATES the whole run rather
    than being skipped. A kernel that is blazing fast on the workloads it gets right and
    simply fails the rest must never come out ahead of a correct one -- so a single failed
    workload means no score, not a better average.
    """
    if "." not in metric:
        return None
    values: list[float] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        val = _dig(rec, metric)
        if val is None:
            continue
        status = _find_status(rec)
        if status is not None and status.lower() not in _OK_STATUSES:
            return None      # correctness failed somewhere -> the run has no valid score
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            return None
    if not values:
        return None
    return sum(values) / len(values)


def _parse_metric(metric: str, text: str) -> float | None:
    """Last occurrence of ``<metric> [:=] <number>`` in the run's output (case-insensitive).

    Separators WITHIN the metric name -- space / underscore / slash / hyphen -- are treated as
    interchangeable, so an inferred ``val_loss`` still matches a repo that actually prints
    ``val loss`` or ``val/loss``. This closes a real mismatch: infer_benchmark/the LLM tends to
    normalize a printed ``val loss`` into ``val_loss``, which ``re.escape`` would then fail to
    find in the output -- fail-closing an otherwise valid baseline (seen live on nanoGPT)."""
    tokens = [t for t in re.split(r"[\s_/-]+", (metric or "").strip()) if t]
    if not tokens:
        return None
    name_pat = r"[\s_/-]+".join(re.escape(t) for t in tokens)
    pat = re.compile(rf"{name_pat}\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I)
    matches = pat.findall(text or "")
    if not matches:
        # Fallback for runs that emit structured traces instead of a printed metric line.
        return _parse_jsonl_metric(metric, text)
    try:
        return float(matches[-1])
    except ValueError:
        return None


#: A venv-creation (`uv venv .venv`, `python -m venv X`, `virtualenv X`) FAILS if the target dir
#: already exists ("a virtual environment already exists"), which sinks the baseline on a re-prepare
#: (transient retry, or the setup agent re-running its own prepare). Make it idempotent.
_VENV_CREATE_RE = re.compile(r"\b(?:uv\s+venv|python[0-9.]*\s+-m\s+venv|virtualenv)\s+([^\s&;|]+)")


def _idempotent_prepare(cmd: str | None) -> str | None:
    """Prepend a removal of the venv dir so a prepare command that creates one is safe to re-run.
    Returns the command unchanged when it creates no venv (or is already guarded)."""
    if not cmd:
        return cmd
    m = _VENV_CREATE_RE.search(cmd)
    if not m:
        return cmd
    venv_dir = m.group(1)
    if venv_dir in ("", ".", "/") or f"rm -rf {venv_dir}" in cmd:
        return cmd
    return f"rm -rf {venv_dir}; {cmd}"


def run_variant(
    benchmark: Benchmark, work_dir: str, *,
    executor=_capped_run, prepare: bool = True, budget_margin_s: int = 60,
    cuda_device: str | None = None, seed: int | None = None,
) -> RunResult:
    """Run one variant workspace and return its official metric. ``prepare`` runs the
    (cached) data/tokenizer prep first; ``cuda_device`` pins the run to one GPU (single-card
    budget); ``seed`` (via the benchmark's ``seed_env``) varies the run for reproduce. When
    the benchmark defines a TRUSTED re-eval, its output -- not train.py's printed number --
    is authoritative. Fails CLOSED on any error.

    Fair-budget (anti-cheat): the training ``run_cmd`` is HARD-capped at ``budget_s +
    budget_margin_s`` via ``_capped_run`` (kills the whole process group on timeout so a training
    grandchild can't outlive its budget). ``budget_margin_s`` is a small cold-start allowance (60s),
    NOT a near-doubling -- both baseline and method get the same real wall-clock; an overrun is
    killed -> invalid, never a winner. ``prepare`` (install/build/download) is a SEPARATE, larger
    budget and is not charged to the training run."""
    env = dict(os.environ)
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    if seed is not None and benchmark.seed_env:
        env[benchmark.seed_env] = str(seed)
    try:
        if prepare:
            pc = executor(_idempotent_prepare(benchmark.prepare_cmd), cwd=work_dir, shell=True, env=env,
                          capture_output=True, text=True, timeout=1800)
            if pc.returncode != 0:
                return RunResult(benchmark.metric, None, False,
                                 error=f"prepare failed: {(pc.stderr or pc.stdout)[:800]}")
        rc = executor(benchmark.run_cmd, cwd=work_dir, shell=True, env=env,
                      capture_output=True, text=True, timeout=benchmark.budget_s + budget_margin_s)
        out = (rc.stdout or "") + "\n" + (rc.stderr or "")
        # A cap-kill (``timed_out``) is NOT a crash: a script that trains unbounded (minGPT's
        # chargpt.py) never exits on its own, so the FAIR result is the metric it had reached AT the
        # shared wall-clock cap. Both variants are capped identically, so this can't 'win by training
        # longer'. A genuine non-zero exit (not a cap-kill) is still an honest failure.
        capped = getattr(rc, "timed_out", False)
        if rc.returncode != 0 and not capped:
            return RunResult(benchmark.metric, None, False,
                             error=f"run failed (exit {rc.returncode})", stdout_tail=out[-800:])
        # TRUSTED re-eval (authoritative): write our eval files AFTER the agent's patch,
        # run our eval command, take ITS metric. The agent's printed number is ignored.
        if benchmark.trusted_eval_cmd:
            for rel, content in benchmark.trusted_eval_files:
                p = Path(work_dir) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            ev = executor(benchmark.trusted_eval_cmd, cwd=work_dir, shell=True, env=env,
                          capture_output=True, text=True, timeout=benchmark.budget_s + budget_margin_s)
            ev_out = (ev.stdout or "") + "\n" + (ev.stderr or "")
            if ev.returncode != 0:
                return RunResult(benchmark.metric, None, False,
                                 error=f"trusted eval failed (exit {ev.returncode})",
                                 stdout_tail=ev_out[-800:])
            out = ev_out  # authoritative source for the metric
        value = _parse_metric(benchmark.metric, out)
        if value is None:
            reason = ("hit the wall-clock cap without emitting a parseable metric"
                      if capped else f"could not find metric {benchmark.metric!r} in output")
            return RunResult(benchmark.metric, None, False,
                             error=f"run {reason}" if capped else reason,
                             stdout_tail=out[-800:], output=out[-_OUTPUT_KEEP:])
        return RunResult(benchmark.metric, value, True, stdout_tail=out[-800:])
    except subprocess.TimeoutExpired:
        return RunResult(benchmark.metric, None, False, error="run timed out")
    except Exception as exc:  # noqa: BLE001 -- fail closed, never crash the pipeline
        return RunResult(benchmark.metric, None, False, error=f"{type(exc).__name__}: {exc}")


__all__ = ["RunResult", "run_variant", "_parse_metric"]
