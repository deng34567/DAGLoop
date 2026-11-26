"""组3 P1-1: best-so-far multi-factor tie-break (score tier + progress margin);
#63: noise-aware absolute-vs-margin two-stage tie-break."""
from __future__ import annotations

import pytest

from autoresearch.agents.runner_agent.handler import (
    _best_so_far_overwrite, _bestsofar_eps, _gen_result_margin,
    _gen_result_metric, _gen_result_quality, _infer_lower_is_better)


def _win(base, method):
    return {"verdict": "new_best", "accepted": True,
            "baseline_metric": base, "method_metric": method}


# --- tiers unchanged (anti-clobber invariant preserved) --------------------------------

def test_quality_tiers_unchanged():
    assert _gen_result_quality(None) == -1
    assert _gen_result_quality({"method_metric": None, "baseline_metric": None}) == 0
    assert _gen_result_quality({"method_metric": None, "baseline_metric": 1.7}) == 1
    assert _gen_result_quality({"verdict": "invalid", "method_metric": 1.0,
                                "baseline_metric": 1.0}) == 2
    assert _gen_result_quality({"verdict": "valid_not_best", "method_metric": 1.0,
                                "baseline_metric": 0.9}) == 3
    assert _gen_result_quality(_win(1.7, 1.5)) == 4


def test_unbound_parent_control_cannot_pin_best_so_far():
    old = {**_win(4080, 1283), "component_evidence_error": "wrong parent control"}
    new = _win(4080, 1400)
    assert _gen_result_quality(old) == 2
    assert _best_so_far_overwrite(
        _gen_result_quality(new), _gen_result_quality(old),
        _gen_result_margin(new), _gen_result_margin(old),
        method_new=1400, method_prev=1283, lower_is_better=True,
    )


def test_margin_is_abs_improvement():
    assert _gen_result_margin(_win(1.7, 1.5)) == pytest.approx(0.2)
    assert _gen_result_margin({"method_metric": None, "baseline_metric": 1.7}) == 0.0
    assert _gen_result_margin(None) == 0.0


# --- the tie-break decision ------------------------------------------------------------

def test_higher_tier_always_wins():
    assert _best_so_far_overwrite(4, 2, 0.0, 9.9) is True    # win beats invalid regardless of margin
    assert _best_so_far_overwrite(1, 4, 9.9, 0.0) is False   # lower tier never clobbers a win


def test_same_win_tier_bigger_improvement_wins():
    # prev win improved by 0.2; new win improves only 0.05 -> keep prior (don't clobber better win)
    assert _best_so_far_overwrite(4, 4, 0.05, 0.20) is False
    # new win improves more -> overwrite
    assert _best_so_far_overwrite(4, 4, 0.30, 0.20) is True
    # equal margin -> overwrite (>= keeps later, harmless idempotent)
    assert _best_so_far_overwrite(4, 4, 0.20, 0.20) is True


def test_same_non_win_tier_keeps_numerically_better_result():
    # Live AutoResearch regression: both rounds carried real lower-is-better values, but
    # the later 1.082210 incorrectly overwrote the earlier/better 1.076792 merely because
    # both were evaluator-invalid/non-winning results.
    assert _best_so_far_overwrite(
        2, 2, 0.084991, 0.079573,
        method_new=1.082210, method_prev=1.076792,
        lower_is_better=True,
    ) is False
    assert _best_so_far_overwrite(
        3, 3, 0.01, 0.02,
        method_new=0.81, method_prev=0.80,
        lower_is_better=False,
    ) is True


def test_same_low_tier_without_comparable_metrics_keeps_later_semantics():
    assert _best_so_far_overwrite(1, 1, 0.0, 0.0) is True


def test_worse_later_win_does_not_clobber_better_earlier_win():
    # the bug this fixes, end-to-end via the two helpers (no noise info -> pure margin):
    prev = _win(1.70, 1.59)     # improvement 0.11
    new = _win(1.70, 1.65)      # improvement 0.05 (still a win, but worse)
    keep = _best_so_far_overwrite(
        _gen_result_quality(new), _gen_result_quality(prev),
        _gen_result_margin(new), _gen_result_margin(prev))
    assert keep is False        # the better earlier win (1.59) is preserved


# --- #63: noise-aware absolute-vs-margin two-stage tie-break ---------------------------

def _over(new, prev, eps):
    """Full noise-aware guard call from two gen_result dicts."""
    return _best_so_far_overwrite(
        _gen_result_quality(new), _gen_result_quality(prev),
        _gen_result_margin(new), _gen_result_margin(prev),
        method_new=_gen_result_metric(new), method_prev=_gen_result_metric(prev),
        lower_is_better=bool(new.get("lower_is_better", True)), eps=eps)


def test_absolute_better_beyond_noise_wins_even_with_smaller_margin():
    # THE #63 fix: new has a LOWER absolute (1.30) than prev (1.44), gap 0.14 >> ε, but a
    # SMALLER margin (0.10 vs 0.20). Beyond noise -> absolute wins (overwrite to 1.30).
    new = _win(1.40, 1.30)      # margin 0.10
    prev = _win(1.64, 1.44)     # margin 0.20
    assert _over(new, prev, eps=0.01) is True


def test_within_noise_falls_back_to_bigger_margin():
    # the live nano case: 1.4412 vs 1.4454 differ 0.0042 (within ε), so the bigger paired
    # improvement (prev, margin 0.0204) is kept over the marginally-lower absolute.
    new = _win(1.4569, 1.4412)   # margin 0.0157
    prev = _win(1.4658, 1.4454)  # margin 0.0204
    assert _over(new, prev, eps=0.0072) is False    # KEEP-PRIOR (within-noise -> margin)
    # but if ε is tiny (trust the tiny absolute gap), the lower absolute wins:
    assert _over(new, prev, eps=0.0) is True         # zero tolerance selects the better metric


@pytest.mark.parametrize("configured", [None, "invalid", "nan", "inf"])
def test_default_selection_has_no_relative_floor(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("AUTORESEARCH_BESTSOFAR_EPS_REL", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_BESTSOFAR_EPS_REL", configured)
    assert _bestsofar_eps(_win(2000, 1543), _win(2000, 1549)) == 0


@pytest.mark.parametrize("lower,previous,current", [
    (True, _win(2000, 1549), _win(1800, 1543)),
    (False, _win(1000, 1543), _win(1200, 1549)),
])
def test_zero_tolerance_selects_metric_despite_baseline_drift(lower, previous, current):
    current = {**current, "lower_is_better": lower}
    assert _over(current, previous, eps=0.0) is True
    previous = {**previous, "lower_is_better": lower}
    assert _over(previous, current, eps=0.0) is False


def test_higher_is_better_direction():
    # accuracy (higher better): new 99.0 vs prev 98.0, gap 1.0 > ε -> new wins despite smaller margin
    new = {"verdict": "new_best", "accepted": True, "baseline_metric": 97.5,
           "method_metric": 99.0, "lower_is_better": False}     # margin 1.5
    prev = {"verdict": "new_best", "accepted": True, "baseline_metric": 96.0,
            "method_metric": 98.0, "lower_is_better": False}    # margin 2.0
    assert _over(new, prev, eps=0.3) is True


def test_eps_prefers_std_then_relative_default(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_BESTSOFAR_EPS_REL", "0.01")
    # no std -> relative default: 1% of max(|1.44|,|1.45|) ~= 0.0145
    new, prev = _win(1.46, 1.44), _win(1.47, 1.45)
    assert _bestsofar_eps(new, prev) == pytest.approx(0.0145, rel=1e-3)
    # A model-written std is preserved for reporting but cannot change selection.
    new2 = dict(new, method_std=0.05)
    assert _bestsofar_eps(new2, prev) == pytest.approx(0.0145)
    # Host-bound repeated output supplies the tolerance.
    measured = dict(new2, measurement_evidence={"stdout_sha256":"host", "samples":[1.4, 1.5]})
    assert _bestsofar_eps(measured, prev) == pytest.approx(2 ** -0.5 * 0.1)


# --- #63 direction fix: general-path gen_result omits lower_is_better ---------------------

def _win_acc(base, method):
    # a general/discover-path win: NO "lower_is_better" key (accuracy, higher-is-better)
    return {"verdict": "pass", "accepted": True, "baseline_metric": base, "method_metric": method}


def test_infer_lower_is_better():
    assert _infer_lower_is_better({"lower_is_better": True}) is True        # explicit wins
    assert _infer_lower_is_better({"lower_is_better": False}) is False
    assert _infer_lower_is_better(_win_acc(0.85, 0.97)) is False            # win, method>base -> higher-better
    assert _infer_lower_is_better(_win_acc(1.7, 1.4)) is True               # win, method<base -> lower-better
    assert _infer_lower_is_better({}) is True                               # unknown -> loss default
    assert _infer_lower_is_better(None) is True


def test_accuracy_general_path_does_not_keep_worse_round():
    # THE live bug (sklearn-digits, general path, force-4 round2): both are wins, higher-is-better,
    # but gen_result carries NO lower_is_better -> old code defaulted to lower-is-better and KEPT the
    # WORSE round. With direction inferred, the higher-accuracy round wins.
    prev = _win_acc(0.8583, 0.9722)   # round1
    new = _win_acc(0.7777, 0.9777)    # round2 -- higher accuracy (0.9777 > 0.9722)
    over = _best_so_far_overwrite(
        _gen_result_quality(new), _gen_result_quality(prev),
        _gen_result_margin(new), _gen_result_margin(prev),
        method_new=_gen_result_metric(new), method_prev=_gen_result_metric(prev),
        lower_is_better=_infer_lower_is_better(new),
        eps=_bestsofar_eps(new, prev))
    assert over is True               # keep the BETTER accuracy (0.9777), not the worse (0.9722)
