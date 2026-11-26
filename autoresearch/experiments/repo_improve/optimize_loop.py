"""General optimization loop: an LLM proposes edits, WE score them. Backend-agnostic.
通用优化循环:LLM 提改法,我们重跑评分。不依赖任何特定执行后端。

The one loop that replaces per-task adapters + one-shot ``propose_patch``. Borrowed from
AIDE (a task-agnostic tree/loop over code variants) and MLE-bench (the agent never controls
scoring): each round the LLM proposes a new version of the EDITABLE files to improve the
metric, and WE re-score it in a clean workspace with the PROTECTED files restored -- so a
cheating edit that touches the evaluator is undone before it counts.

Deliberately does NOT drive an in-container agent runtime: scoring already runs the code
(``rescore``), so the "observe" step is our trusted re-score, not the agent's self-report.
That keeps the loop independent of the sandbox image -- it works whether the run happens in
an OpenHands sandbox or a plain ``docker exec`` container (SOL). The same loop handles minGPT
(edit train.py), SOL (edit the CUDA kernel inside a solution JSON), a CPU repo, anything: the
only difference is the task text handed to the LLM, never this control flow.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from autoresearch.agents._reuse.json_extract import extract_json

logger = logging.getLogger(__name__)


@dataclass
class OptimizeResult:
    best_edits: dict | None      # editable-file contents that scored best (None if none beat baseline)
    best_metric: float | None    # its TRUSTED re-scored metric
    baseline_metric: float
    improved: bool
    rounds: int
    history: list                # [(round, trusted_metric, valid)]
    # The best edit that RAN and produced a valid metric, even if it did NOT beat the baseline.
    # Lets the pipeline report the method's REAL measured value ("tried, lost to X") instead of a
    # bare None -- an agent that edited + ran but lost is very different from one that produced
    # nothing, and the user needs to see the number.
    best_attempt_edits: dict | None = None
    best_attempt_metric: float | None = None


def _is_better(candidate: float, baseline: float, lower_is_better: bool) -> bool:
    return candidate < baseline if lower_is_better else candidate > baseline


def _propose_prompt(benchmark, current, baseline_metric, best_metric, feedback=""):
    goal = "LOWER" if benchmark.lower_is_better else "RAISE"
    lines = [
        f"You are optimizing a repository to {goal} its metric by editing code.",
        f"Metric: {benchmark.metric} ({'lower' if benchmark.lower_is_better else 'higher'} is better).",
        f"Baseline (unmodified repo): {benchmark.metric} = {baseline_metric}.",
    ]
    if best_metric is not None:
        lines.append(f"Best so far: {benchmark.metric} = {best_metric}. Beat it.")
    if feedback:
        # Meeting 2026-07-29: don't blindly retry. Feed the PREVIOUS round's diagnosis (what was
        # changed, what it scored vs baseline) so the next change LEARNS from the pattern -- like a
        # human tuning params by experience -- instead of re-proposing blind.
        lines += ["", feedback]
    lines += [
        "",
        "You may ONLY change these files (everything else -- the evaluator, the metric, the "
        f"dataset, the run command -- is restored before scoring, so editing it is useless): "
        f"{list(benchmark.editable_files)}.",
        f"The repo is run with: {benchmark.run_cmd}",
        "",
        "Current contents of the editable file(s):",
    ]
    for f, content in current.items():
        lines += [f"=== {f} ===", content, ""]
    lines += [
        "Propose ONE concrete change likely to improve the metric while keeping the run correct.",
        "Return a single JSON object mapping each file you changed to its COMPLETE new contents:",
        '{"<path>": "<full new file text>", ...}. Only include files you actually changed. '
        "Keep every other file untouched. If the file is itself JSON (e.g. a solution spec with an "
        "embedded kernel), return it as valid JSON text with only the intended part changed.",
    ]
    return "\n".join(lines)


def build_llm_propose(llm, benchmark, emit=None):
    """Default proposer: ask the LLM for the full new contents of the files it wants to change.

    Returns ``propose(current, baseline_metric, best_metric) -> {path: new_content}``. A malformed
    reply or one that names non-editable files yields ``{}`` (that round is skipped), never a crash.

    ``emit`` (optional): a ``(event, payload)`` callback that records WHY a round produced no edit,
    so a silent empty round is diagnosable. Distinguishes an LLM error (e.g. a streaming timeout on
    a too-large one-shot whole-file rewrite), an unparseable reply, and a reply that named only
    non-editable paths -- the difference between "the model is too slow/weak for one-shot propose"
    and "a path/format mismatch", which used to be invisible."""
    editable = set(benchmark.editable_files)
    emit = emit or (lambda event, payload: None)

    def _propose(current, baseline_metric, best_metric, feedback=""):
        prompt = _propose_prompt(benchmark, current, baseline_metric, best_metric, feedback)
        try:
            res = llm.chat(messages=[{"role": "user", "content": prompt}],
                           response_format="json_object")
        except Exception as exc:  # noqa: BLE001 -- a failed call just means no progress this round
            emit("propose_result", {"ok": False, "reason": f"llm_error:{type(exc).__name__}"})
            return {}
        raw = res.content or ""
        try:
            obj = json.loads(extract_json(raw))
        except Exception:  # noqa: BLE001 -- reply wasn't the requested JSON object
            emit("propose_result", {"ok": False, "reason": "unparseable_reply",
                                    "reply_head": raw[:200], "reply_len": len(raw)})
            return {}
        if not isinstance(obj, dict):
            emit("propose_result", {"ok": False, "reason": "not_a_json_object"})
            return {}
        # keep only real edits to editable files with string content
        kept = {k: v for k, v in obj.items()
                if k in editable and isinstance(v, str) and v.strip()}
        if not kept:
            emit("propose_result", {"ok": False, "reason": "no_editable_keys",
                                    "got_keys": list(obj.keys())[:10], "editable": list(editable)})
        else:
            emit("propose_result", {"ok": True, "files": list(kept.keys())})
        return kept

    return _propose


def optimize_in_sandbox(benchmark, *, baseline_metric: float, rescore, propose,
                        initial_editable: dict, max_rounds: int = 3) -> OptimizeResult:
    """Run the bounded optimize loop; return the best TRUSTED-rescored edit.

    ``propose(current, baseline_metric, best_metric) -> {path: new_content}`` is the search step
    (LLM by default). ``rescore(benchmark, edits, label) -> (metric, valid)`` applies ``edits`` to a
    CLEAN workspace (protected restored) and runs -- that number, not anything the proposer says, is
    what we trust. ``initial_editable`` is the pristine editable-file contents. Bounded by
    ``max_rounds``; keeps only versions whose trusted re-score beats the baseline; honest
    ``improved=False`` when none do.
    """
    lower = benchmark.lower_is_better
    best_edits, best_metric = None, None
    best_attempt_edits, best_attempt_metric = None, None   # best VALID run, even if it lost
    current = dict(initial_editable)
    history = []
    # Meeting 2026-07-29: don't blindly retry the plan. After each round, diagnose what happened
    # (what was changed, did it beat baseline, did it even run) and feed that to the NEXT proposer so
    # it LEARNS from the pattern (找规律) -- like a human tuning params by experience.
    feedback = ""

    for round_i in range(max_rounds):
        edits = propose(current, baseline_metric, best_metric, feedback)
        if not edits:
            history.append((round_i, None, False))
            feedback = ("The previous round produced NO usable edit. Make a DIFFERENT, concrete, "
                        "well-formed change to the editable file(s) this time.")
            continue
        # HARD SCORE: the proposer's claim is irrelevant -- only this trusted re-run counts.
        metric, valid = rescore(benchmark, edits, f"round{round_i}")
        history.append((round_i, metric, valid))
        if not valid or metric is None:
            feedback = (f"The previous round's edit to {list(edits)} did NOT run to a valid "
                        f"{benchmark.metric} (it crashed or printed no parseable metric). Diagnose "
                        "WHY it broke and fix that first -- keep the code runnable before optimizing.")
            continue
        _beat = _is_better(metric, baseline_metric, lower)
        feedback = (
            f"The previous round edited {list(edits)} and scored a trusted {benchmark.metric}="
            f"{metric} vs baseline {baseline_metric} -- it "
            f"{'BEAT the baseline (build on this direction)' if _beat else 'did NOT beat the baseline (that lever did not help -- diagnose why and try a DIFFERENT one)'}. "
            "Find the pattern; do not repeat the same change.")
        # record the best edit that RAN + produced a metric, regardless of beating the baseline --
        # so a "tried, ran, but lost" round still surfaces its real number, not None.
        if best_attempt_metric is None or _is_better(metric, best_attempt_metric, lower):
            best_attempt_edits, best_attempt_metric = dict(edits), metric
        if _is_better(metric, baseline_metric, lower) and (
                best_metric is None or _is_better(metric, best_metric, lower)):
            best_edits, best_metric = dict(edits), metric
            current = {**current, **edits}   # improve from the new best next round

    return OptimizeResult(best_edits=best_edits, best_metric=best_metric,
                          baseline_metric=baseline_metric, improved=best_edits is not None,
                          rounds=len(history), history=history,
                          best_attempt_edits=best_attempt_edits,
                          best_attempt_metric=best_attempt_metric)


__all__ = ["optimize_in_sandbox", "OptimizeResult", "build_llm_propose"]
