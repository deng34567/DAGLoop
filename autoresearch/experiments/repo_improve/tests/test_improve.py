"""constrained-patch application + the propose_patch agent step."""
from __future__ import annotations

import json

import pytest

from autoresearch.experiments.repo_improve.benchmark import EXAMPLE_BENCHMARK
from autoresearch.experiments.repo_improve.improve import PatchError, apply_edits, propose_patch


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        content = self._responses.pop(0) if self._responses else ""
        return type("R", (), {"content": content})()


def test_apply_edits_replaces_unique_find():
    out = apply_edits({"train.py": "lr = 0.001\nsteps = 100\n"},
                      [{"file": "train.py", "find": "lr = 0.001", "replace": "lr = 0.002"}])
    assert out["train.py"] == "lr = 0.002\nsteps = 100\n"


def test_apply_edits_rejects_missing_find():
    with pytest.raises(PatchError):
        apply_edits({"a.py": "a = 1\n"}, [{"file": "a.py", "find": "b = 2", "replace": "b = 3"}])


def test_apply_edits_rejects_nonunique_find():
    with pytest.raises(PatchError):
        apply_edits({"x.py": "x\nx\n"}, [{"file": "x.py", "find": "x", "replace": "y"}])


def test_apply_edits_rejects_unknown_or_protected_file():
    # an edit naming a file we didn't show (e.g. a protected evaluator) is rejected
    with pytest.raises(PatchError):
        apply_edits({"a.py": "a = 1\n"},
                    [{"file": "eval/score.py", "find": "a = 1", "replace": "cheat"}])


def test_propose_patch_applies_clean_multi_file_patch():
    srcs = {"train.py": "learning_rate = 0.001\n# train\n", "model.py": "width = 64\n"}
    resp = json.dumps({"rationale": "raise LR + widen", "edits": [
        {"file": "train.py", "find": "learning_rate = 0.001", "replace": "learning_rate = 0.003"},
        {"file": "model.py", "find": "width = 64", "replace": "width = 96"}]})
    llm = _FakeLLM([resp])
    out = propose_patch(llm, EXAMPLE_BENCHMARK, "improve", srcs, 0.94, goal="improve accuracy")
    assert out["error"] is None
    assert "learning_rate = 0.003" in out["patched_sources"]["train.py"]
    assert "width = 96" in out["patched_sources"]["model.py"]      # multi-file
    assert out["rationale"] == "raise LR + widen"


def test_propose_patch_retries_then_gives_up_on_bad_find():
    srcs = {"train.py": "learning_rate = 0.001\n"}
    bad = json.dumps({"rationale": "x", "edits": [
        {"file": "train.py", "find": "NONEXISTENT", "replace": "z"}]})
    llm = _FakeLLM([bad, bad])           # both attempts bad
    out = propose_patch(llm, EXAMPLE_BENCHMARK, "instr", srcs, 0.94)
    assert out["patched_sources"] is None and out["error"]
    assert len(llm.calls) == 2           # retried once
