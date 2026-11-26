"""Deterministic acceptance verdict for the MC variance-reduction experiment.

This is a PURE function (no container / LLM / network): the MC metric is a
number, so the verdict is computed by fixed RULES rather than handed to an LLM
(more tamper-resistant than an NL "sentinel", and reproducible).

Rules, in order:
  1. anti-cheat HARD-NO -> ``fail`` (over_budget or trusted-file tamper).
  2. improvement -> ``fail`` if the RMSE-reduction ratio is below the margin.
  3. reproducibility -> ``inconclusive`` if the reproduce ratio is missing, not
     an improvement (<=1), or collapses far below the primary ratio.
  4. ``hardcoded_suspected`` -> ``inconclusive`` (a SIGNAL; needs human/later
     acceptance, not a hard reject here).
  5. otherwise -> ``pass``.

Thresholds are HEURISTIC (tuned to the ratio noise of the default 8-seed
problem) and env-overridable.
"""
from __future__ import annotations

import os

# Require >= 10% RMSE reduction (ratio >= 1.10) to count as a real improvement.
IMPROVEMENT_MARGIN = float(os.environ.get("AUTORESEARCH_MC_IMPROVEMENT_MARGIN", "1.10"))
# On fresh seeds the reduction must stay > 1x AND keep at least this fraction of
# the primary ratio (else the win looks seed-specific / unstable).
REPRODUCE_TOL = float(os.environ.get("AUTORESEARCH_MC_REPRODUCE_TOL", "0.5"))


def _verdict(verdict: str, reasons: list[str], accepted: bool) -> dict:
    return {"verdict": verdict, "accepted": accepted, "reasons": reasons}


def decide(exp_result) -> dict:
    """Return ``{verdict, accepted, reasons}`` for a ``run_experiment`` result."""
    ac = exp_result.get("anti_cheat") or {}
    ratio = exp_result.get("metric_value") or 0.0
    rep = exp_result.get("reproduce_metric_value")

    # 1) anti-cheat hard-no
    reasons: list[str] = []
    if ac.get("over_budget"):
        reasons.append("over_budget(超采样预算)")
    if ac.get("tamper_detected"):
        reasons.append("tamper(改动可信文件)")
    if reasons:
        return _verdict("fail", reasons, accepted=False)

    # 2) improvement margin
    if ratio < IMPROVEMENT_MARGIN:
        return _verdict("fail", [f"未达提升阈值(比值 {ratio:.2f} < {IMPROVEMENT_MARGIN})"], False)

    # 3) reproducibility on fresh seeds
    if rep is None or rep <= 1.0 or rep < REPRODUCE_TOL * ratio:
        return _verdict("inconclusive", [f"复现不稳(主 {ratio:.2f} vs 复现 {rep})"], False)

    # 4) hardcode suspicion -> not a reject, but not a clean pass either
    if ac.get("hardcoded_suspected"):
        return _verdict("inconclusive", ["疑似写死(存疑,需人工/后续验收)"], False)

    return _verdict(
        "pass",
        [f"RMSE 缩减比 {ratio:.2f}×、复现 {rep:.2f}×、无作弊信号"],
        accepted=True,
    )
