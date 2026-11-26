"""Task 5: minimal P1 acceptance verdict for a general honest experiment.

PURE function (no container / LLM / network): the result carries numbers, so the
verdict is computed by fixed RULES (reproducible, tamper-resistant), in order:

  1. illegal solution (primary ``valid`` False)               -> ``fail``
  2. method NOT better than baseline (by metric direction+margin) -> ``fail``
  3. improvement does NOT reproduce on the fresh seed          -> ``inconclusive``
  4. isolation broken (``isolated`` is False; evidence from Task 4) -> ``inconclusive``
  5. otherwise                                                 -> ``pass``

``lower_is_better`` selects the metric direction (default True, e.g. bins used);
the improvement margin is module-level and env-overridable. A ``pass`` here means
ONLY that an AI-chosen method beat an AI-chosen baseline on an AI-chosen metric,
reproducibly and isolated -- NOT that the method is genuinely good or the harness
fair (see report.py's honesty note).
"""
from __future__ import annotations

import os

# Relative margin the method must beat the baseline by to count as an improvement
# (0.0 = strictly better). Env-overridable.
IMPROVEMENT_MARGIN = float(os.environ.get("AUTORESEARCH_GEN_IMPROVEMENT_MARGIN", "0.0"))


def _verdict(verdict: str, reasons: list[str], accepted: bool) -> dict:
    return {"verdict": verdict, "accepted": accepted, "reasons": reasons}


def _failure_detail(res: dict) -> str:
    """A human, LEAK-SAFE reason for an invalid run, from P1 diagnostics (method's
    own error / validate rejection + validity rate). Falls back to the old generic
    message when no diagnostics are present."""
    diag = (res or {}).get("diagnostics") or {}
    ff = diag.get("first_failure") or {}
    parts = []
    if ff.get("detail"):
        idx = ff.get("index")
        where = f"(实例#{idx}) " if idx is not None else ""
        parts.append(f"{where}{ff['detail']}")
    n, v = diag.get("n_instances"), diag.get("n_valid")
    if n:
        parts.append(f"合法率 {v}/{n}")
    if diag.get("timed_out"):
        parts.append("方法超时")
    return "; ".join(parts) if parts else "方法未产出合法解"


def _improved(baseline, method, lower_is_better: bool, margin: float) -> bool:
    # NOTE(P2): a NEGATIVE margin would loosen the bar (accept a slightly-worse
    # method). Left as P2: default 0.0 fails closed; operators set it knowingly.
    if baseline is None or method is None:
        return False
    # Defense: a non-numeric metric (e.g. the harness returned a dict) is never an
    # improvement -- guard so decide() can't crash on ``dict * float`` (the
    # orchestrator now coerces to a scalar upstream; this is belt-and-suspenders).
    if isinstance(baseline, bool) or isinstance(method, bool) \
            or not isinstance(baseline, (int, float)) or not isinstance(method, (int, float)):
        return False
    if lower_is_better:
        return method < baseline * (1.0 - margin)
    return method > baseline * (1.0 + margin)


def decide(primary: dict, reproduce, *, isolated: bool,
           lower_is_better: bool = True) -> dict:
    """Return ``{verdict, accepted, reasons}`` for a primary + reproduce run(s).

    ``reproduce`` may be a single run dict OR a LIST of K reproduce runs (B5): the
    improvement must hold on a strict MAJORITY of the (valid) reproduces, so a
    single flaky/lucky seed neither passes nor sinks a real result.
    """
    margin = IMPROVEMENT_MARGIN

    # 1) illegal solution
    if not primary.get("valid", False):
        return _verdict("fail", [f"非法解(valid=False):{_failure_detail(primary)}"], False)

    pb, pm = primary.get("baseline_metric"), primary.get("method_metric")

    # 2) improvement over the baseline
    if not _improved(pb, pm, lower_is_better, margin):
        return _verdict(
            "fail", [f"方法未优于基线(baseline={pb}, method={pm})"], False
        )

    # 3) reproducibility -- MAJORITY of the reproduce seeds must be valid + improved
    reproduces = [r for r in (reproduce if isinstance(reproduce, list) else [reproduce]) if r]
    n = len(reproduces)
    n_ok = sum(
        1 for r in reproduces
        if r.get("valid", False)
        and _improved(r.get("baseline_metric"), r.get("method_metric"), lower_is_better, margin)
    )
    need = (n // 2) + 1 if n else 1  # strict majority
    if n_ok < need:
        return _verdict(
            "inconclusive",
            [f"复现不住:{n_ok}/{n} 个复现种子未复现提升"],
            False,
        )

    # 4) isolation evidence (Task 4) -- broken isolation is not a clean pass
    if isolated is False:
        return _verdict(
            "inconclusive", ["隔离被破坏:harness 绕过可信隔离器直接跑 method"], False
        )

    return _verdict(
        "pass",
        [f"方法优于基线且在新 seed 复现、运行期隔离(baseline={pb}, method={pm})"],
        True,
    )


__all__ = ["decide", "IMPROVEMENT_MARGIN"]
