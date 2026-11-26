"""通用优化循环:LLM 提改法,我们重跑评分,只留真改进的版本。

统一 minGPT/SOL/... —— 借 AIDE 的循环,借 MLE-bench 的"agent 不控制 scoring"。
后端无关:只需 propose(提改法)+ rescore(可信重跑)。纯假件。
"""
from __future__ import annotations

import json

from autoresearch.experiments.repo_improve.benchmark import Benchmark
from autoresearch.experiments.repo_improve.optimize_loop import (
    build_llm_propose,
    optimize_in_sandbox,
)

BENCH = Benchmark(
    name="minGPT", description="", repo_url="u",
    editable_files=("train.py",), protected_files=("evaluate.py",),
    prepare_cmd="true", run_cmd="python train.py",
    metric="loss", lower_is_better=True, budget_s=300)

INIT = {"train.py": "original"}


def _propose_seq(edits_list):
    """propose 依次返回预设的 edits(模拟 LLM 每轮的提议)。"""
    seq = list(edits_list)
    calls = []
    def _p(current, baseline, best, feedback=""):
        calls.append((dict(current), baseline, best))
        return seq.pop(0) if seq else {}
    _p.calls = calls
    return _p


def _rescore_seq(score_list):
    """rescore 依次返回预设的可信分(模拟我们在干净工作区重跑)。"""
    seq = list(score_list)
    seen = []
    def _r(bench, edits, label):
        seen.append((dict(edits), label))
        v = seq.pop(0) if seq else None
        return v, (v is not None)
    _r.seen = seen
    return _r


def test_keeps_only_versions_that_improve_on_TRUSTED_rescore():
    # propose 声称两轮都好;但可信重跑:第1轮 0.9(改进)、第2轮 1.5(变差)-> 只留第1轮
    propose = _propose_seq([{"train.py": "v1"}, {"train.py": "v2"}])
    rescore = _rescore_seq([0.9, 1.5])
    out = optimize_in_sandbox(BENCH, baseline_metric=1.0, rescore=rescore,
                              propose=propose, initial_editable=INIT, max_rounds=2)
    assert out.best_metric == 0.9 and out.best_edits == {"train.py": "v1"}
    assert out.improved is True
    assert len(propose.calls) == 2 and len(rescore.seen) == 2   # 每轮都真重跑评分


def test_no_improvement_is_honest_not_faked():
    out = optimize_in_sandbox(BENCH, baseline_metric=1.0,
                              rescore=_rescore_seq([1.2]), propose=_propose_seq([{"train.py": "v1"}]),
                              initial_editable=INIT, max_rounds=1)
    assert out.improved is False and out.best_edits is None


def test_loop_is_bounded_by_max_rounds():
    propose = _propose_seq([{"train.py": f"v{i}"} for i in range(10)])
    optimize_in_sandbox(BENCH, baseline_metric=1.0, rescore=_rescore_seq([1.0]*10),
                        propose=propose, initial_editable=INIT, max_rounds=3)
    assert len(propose.calls) == 3   # 到此为止,不无限跑


def test_second_round_sees_current_best_and_the_new_editable():
    propose = _propose_seq([{"train.py": "v1"}, {"train.py": "v2"}])
    optimize_in_sandbox(BENCH, baseline_metric=1.0, rescore=_rescore_seq([0.9, 0.8]),
                        propose=propose, initial_editable=INIT, max_rounds=2)
    # 第2轮 propose 拿到"当前最好=0.9"和"当前可改内容=v1"(从最好版本继续改)
    _, _, best_at_round2 = propose.calls[1]
    assert best_at_round2 == 0.9
    assert propose.calls[1][0]["train.py"] == "v1"


def test_invalid_rescore_does_not_become_best():
    # 改后跑挂了(valid=False)-> 不算改进
    out = optimize_in_sandbox(BENCH, baseline_metric=1.0,
                              rescore=_rescore_seq([None]), propose=_propose_seq([{"train.py": "v1"}]),
                              initial_editable=INIT, max_rounds=1)
    assert out.improved is False


# --- build_llm_propose:默认 LLM 提议器 ---

class _FakeLLM:
    def __init__(self, replies):
        self._r = list(replies)
    def chat(self, **kw):
        return type("R", (), {"content": self._r.pop(0) if self._r else ""})()


def test_llm_propose_parses_json_and_filters_to_editable():
    llm = _FakeLLM([json.dumps({"train.py": "new content", "evaluate.py": "HACK"})])
    edits = build_llm_propose(llm, BENCH)({"train.py": "old"}, 1.0, None)
    assert edits == {"train.py": "new content"}     # 受保护文件 evaluate.py 被过滤掉


def test_llm_propose_malformed_reply_yields_no_edit():
    assert build_llm_propose(_FakeLLM(["not json"]), BENCH)({"train.py": "old"}, 1.0, None) == {}


def test_llm_propose_emits_reason_on_llm_error():
    # a streaming timeout (or any chat error) must be diagnosable, not a silent {} -- this is the
    # exact failure that produced "0 edits" on SOL/cifar (glm couldn't finish a one-shot whole-file
    # rewrite within the 600s stream deadline).
    class _Boom:
        def chat(self, **kw):
            raise TimeoutError("streaming exceeded 600s wall-clock")
    events = []
    out = build_llm_propose(_Boom(), BENCH, lambda e, p: events.append((e, p)))(
        {"train.py": "old"}, 1.0, None)
    assert out == {}
    assert events[-1] == ("propose_result", {"ok": False, "reason": "llm_error:TimeoutError"})


def test_llm_propose_emits_no_editable_keys_reason():
    # a well-formed reply naming only non-editable files -> diagnosable (path/format mismatch),
    # distinct from an LLM error.
    llm = _FakeLLM([json.dumps({"evaluate.py": "HACK", "README.md": "x"})])
    events = []
    out = build_llm_propose(llm, BENCH, lambda e, p: events.append((e, p)))(
        {"train.py": "old"}, 1.0, None)
    assert out == {}
    assert events[-1][1]["reason"] == "no_editable_keys"
    assert "evaluate.py" in events[-1][1]["got_keys"]


def test_llm_propose_emits_ok_on_success():
    llm = _FakeLLM([json.dumps({"train.py": "new"})])
    events = []
    build_llm_propose(llm, BENCH, lambda e, p: events.append((e, p)))({"train.py": "old"}, 1.0, None)
    assert events[-1] == ("propose_result", {"ok": True, "files": ["train.py"]})


def test_surfaces_best_attempt_metric_when_nothing_beats_baseline():
    # an edit that RUNS + produces a valid metric but LOSES to baseline must still expose its real
    # number via best_attempt_metric -- so the pipeline reports "tried, lost to X", not a bare None.
    propose = _propose_seq([{"train.py": "worse"}])
    rescore = _rescore_seq([2.0])   # ran, valid, 2.0 > 1.0 baseline (worse; lower is better)
    opt = optimize_in_sandbox(BENCH, baseline_metric=1.0, rescore=rescore, propose=propose,
                              initial_editable=INIT, max_rounds=1)
    assert opt.improved is False and opt.best_metric is None        # nothing beat the baseline
    assert opt.best_attempt_metric == 2.0                           # ...but the real number is kept
    assert opt.best_attempt_edits == {"train.py": "worse"}
