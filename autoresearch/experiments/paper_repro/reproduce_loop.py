"""Reproduce-until-close (meeting 2026-08-03): retry a PAPER reproduction until the measured metric
APPROACHES the paper's reported value, bounded by a max number of attempts.

The professor's task setting for the paper line: "如果差值很大，就重新复现（重新写），直到正确率接近论文里的值"。
When the synthesized repo reproduces a metric that is far from the paper's reported number, re-run the
reproduction (the agent rewrites/fixes) with the gap fed back, until it is within tolerance or the
attempt budget is exhausted.

Pure + injectable so it stays DECOUPLED from the runner and is offline-testable: the caller injects
``attempt_fn(feedback) -> measured`` (one reproduction attempt) plus the paper target; this module owns
ONLY the gap math + the bounded-retry decision. Loose coupling: no Store / EventBus / runner import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


def reproduction_gap(measured, target, lower_is_better: bool):
    """Relative gap |measured - target| / |target| (a fraction, e.g. 0.05 == 5%).

    Returns None when it cannot be computed (measured/target missing, or target == 0). ``lower_is_better``
    is accepted for signature symmetry with the rest of the paper line (the magnitude of the relative
    gap is direction-agnostic)."""
    if measured is None or target is None:
        return None
    try:
        m, t = float(measured), float(target)
    except (TypeError, ValueError):
        return None
    if t == 0:
        return None
    return abs(m - t) / abs(t)


def is_close(measured, target, lower_is_better: bool, tol: float) -> bool:
    """True when ``measured`` reproduces ``target`` well enough to stop.

    Two ways to be "close": (1) within relative tolerance ``tol`` of the paper's value, or (2) on the
    GOOD side of it -- reproducing a metric that is already >= the paper's value (higher-is-better) or
    <= it (lower-is-better) means we reproduced or beat the paper, so there is nothing to chase."""
    gap = reproduction_gap(measured, target, lower_is_better)
    if gap is None:
        return False
    if gap <= tol:
        return True
    m, t = float(measured), float(target)
    return (m <= t) if lower_is_better else (m >= t)


def _feedback(measured, target, metric: str, lower_is_better: bool, gap) -> str:
    dirn = "lower" if lower_is_better else "higher"
    pct = f"{gap * 100:.1f}%" if gap is not None else "large"
    return (f"复现值 {metric}={measured} 与论文报告值 {target} 相差 {pct}（{dirn}-is-better）。"
            f"请重新复现/修正实现，使 {metric} 更接近论文的 {target}。")


@dataclass
class ReproduceResult:
    best_measured: float | None
    best_attempt: object                     # whatever attempt_fn returned alongside the metric
    attempts: int
    close: bool
    target: float | None
    history: list = field(default_factory=list)   # [(attempt_no, measured, gap, close)]


def reproduce_until_close(
    attempt_fn: Callable[[str], tuple],
    target,
    metric: str,
    lower_is_better: bool,
    *,
    tol: float = 0.05,
    max_attempts: int = 3,
    emit: Callable[[str, dict], None] | None = None,
) -> ReproduceResult:
    """Run ``attempt_fn(feedback) -> (payload, measured)`` up to ``max_attempts`` times, stopping as
    soon as ``measured`` is close to ``target`` (see :func:`is_close`). Each retry is fed a gap
    feedback string. Keeps the attempt whose metric is CLOSEST to the target. Bounded (never loops
    forever); fail-soft (an attempt returning measured=None just doesn't win). When ``target`` is
    unknown, runs exactly once (nothing to chase). ``emit`` (optional) mirrors the runner's event
    sink for observability."""
    emit = emit or (lambda ev, pl: None)
    best_measured = None
    best_attempt = None
    best_gap = None
    history: list = []
    feedback = ""
    n = max(1, int(max_attempts)) if target is not None else 1

    for i in range(n):
        payload, measured = attempt_fn(feedback)
        gap = reproduction_gap(measured, target, lower_is_better)
        close = is_close(measured, target, lower_is_better, tol)
        history.append({"attempt": i + 1, "measured": measured, "gap": gap, "close": close})
        emit("paper_reproduce_attempt",
             {"attempt": i + 1, "measured": measured, "target": target, "gap": gap, "close": close})
        # keep the attempt closest to the target (a valid measured always beats None)
        if measured is not None and (best_gap is None or (gap is not None and gap < best_gap)):
            best_measured, best_attempt, best_gap = measured, payload, gap
        if close:
            return ReproduceResult(best_measured, best_attempt, i + 1, True, target, history)
        feedback = _feedback(measured, target, metric, lower_is_better, gap)

    return ReproduceResult(best_measured, best_attempt, len(history),
                           is_close(best_measured, target, lower_is_better, tol), target, history)


__all__ = ["reproduction_gap", "is_close", "reproduce_until_close", "ReproduceResult"]
