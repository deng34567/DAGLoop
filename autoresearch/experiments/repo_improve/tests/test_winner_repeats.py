"""组4 P1-3: winner mean±std -- prompt asks for N-seed repeat; result parses std/runs into report."""
from __future__ import annotations

import json
import os

from autoresearch.experiments.repo_improve import agentic_improve as ai


def test_winner_repeats_env(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_WINNER_REPEATS", raising=False)
    assert ai._winner_repeats() == 3                       # default
    monkeypatch.setenv("AUTORESEARCH_WINNER_REPEATS", "5")
    assert ai._winner_repeats() == 5
    monkeypatch.setenv("AUTORESEARCH_WINNER_REPEATS", "1")
    assert ai._winner_repeats() == 1                       # off
    monkeypatch.setenv("AUTORESEARCH_WINNER_REPEATS", "junk")
    assert ai._winner_repeats() == 3                       # bad value -> default


def test_prompt_asks_for_repeats_when_gt1():
    p = ai._task_prompt("goal", "/repo", "/r.json", None, "acc", False, repeats=3)
    assert "REPRODUCIBILITY" in p and "3 times" in p
    assert "method_std" in p and "method_runs" in p
    assert "same seed list" not in p
    assert "respective MEANS" not in p


def test_prompt_omits_repeats_when_1():
    p = ai._task_prompt("goal", "/repo", "/r.json", None, "acc", False, repeats=1)
    assert "REPRODUCIBILITY" not in p                       # single-run -> identical to before


def _write_result(tmp_path, obj) -> str:
    p = os.path.join(str(tmp_path), "ar_result.json")
    with open(p, "w") as fh:
        json.dump(obj, fh)
    return p


def test_build_result_folds_std_and_runs_into_report(tmp_path):
    p = _write_result(tmp_path, {
        "metric": "val_loss", "baseline": 1.70, "method": 1.60, "lower_is_better": True,
        "notes": "tuned lr",
        "baseline_std": 0.01, "baseline_runs": [1.69, 1.70, 1.71],
        "method_std": 0.02, "method_runs": [1.58, 1.61, 1.61]})
    res = ai._build_result_from_file(p, "nano", "val_loss", True, lambda e, pl: None)
    assert res["candidate"].metric_value == 1.60           # mean is the reported method value
    assert "复现" in res["report"] and "std=0.02" in res["report"]
    assert "1.58" in res["report"]                          # per-seed runs listed
    assert "baseline=1.7" in res["report"] and "std=0.01" in res["report"]
    assert "1.69" in res["report"]


def test_build_result_without_std_is_unchanged(tmp_path):
    p = _write_result(tmp_path, {
        "metric": "val_loss", "baseline": 1.70, "method": 1.60, "lower_is_better": True,
        "notes": "tuned lr"})
    res = ai._build_result_from_file(p, "nano", "val_loss", True, lambda e, pl: None)
    assert "复现" not in res["report"]                       # no repeat data -> no repro line
    assert res["report"] == "tuned lr"
