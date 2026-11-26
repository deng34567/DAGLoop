"""3-agent implementation discussion (meeting #3 / C row): 3 angle calls + synthesis -> guidance,
best-effort (any failure -> "")."""
from __future__ import annotations

from autoresearch.experiments.repo_improve.benchmark import Benchmark
from autoresearch.experiments.repo_improve.implementation_discussion import discuss_implementation


def _bench():
    return Benchmark(name="b", description="d", repo_url="u",
                     editable_files=("model.py",), protected_files=("eval.py",),
                     prepare_cmd="true", run_cmd="python train.py",
                     metric="val_loss", lower_is_better=True, budget_s=600, keywords=())


class _FakeLLM:
    """Records each call's system prompt; returns queued replies (one per chat call)."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.systems = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.systems.append(messages[0]["content"])
        return type("R", (), {"content": self._replies.pop(0) if self._replies else ""})()


def test_three_angles_plus_synthesis_returns_guidance():
    llm = _FakeLLM(["raise the LR", "tune dropout", "don't touch the evaluator", "PLAN: try LR + dropout"])
    guidance = discuss_implementation(llm, _bench(), "improve the model", 1.0)
    assert guidance == "PLAN: try LR + dropout"          # the synthesis is returned
    assert len(llm.systems) == 4                          # 3 angle calls + 1 synthesis
    # the three distinct lenses were actually used (impact / simplicity / risk)
    joined = " ".join(llm.systems[:3]).upper()
    assert "IMPACT" in joined and "SIMPLICITY" in joined and "RISK" in joined


def test_no_llm_returns_empty():
    assert discuss_implementation(None, _bench(), "goal", 1.0) == ""


def test_all_angles_failing_returns_empty():
    class _BoomLLM:
        def chat(self, **kw):
            raise RuntimeError("boom")
    assert discuss_implementation(_BoomLLM(), _bench(), "goal", 1.0) == ""


def test_synthesis_failure_falls_back_to_raw_opinions():
    # 3 angles succeed, the synthesis call raises -> the raw opinions are still returned (not "").
    class _LLM:
        def __init__(self):
            self.n = 0
        def chat(self, *, messages, response_format=None, max_tokens=None):
            self.n += 1
            if self.n <= 3:
                return type("R", (), {"content": f"opinion{self.n}"})()
            raise RuntimeError("synthesis boom")
    guidance = discuss_implementation(_LLM(), _bench(), "goal", 1.0)
    assert "opinion1" in guidance and "opinion3" in guidance   # fell back to the angle opinions
