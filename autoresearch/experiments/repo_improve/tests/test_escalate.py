"""层3 升级:宿主侧安装自愈耗尽后,开 GPU 沙箱让 agent 摸出可用安装,并把后续执行搬进容器。

纯假件,不碰 docker / 不调真 LLM。
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from autoresearch.experiments.repo_improve.benchmark import Benchmark
from autoresearch.experiments.repo_improve.escalate import escalate_to_sandbox

BENCH = Benchmark(
    name="nanochat", description="", repo_url="https://github.com/karpathy/nanochat",
    editable_files=("scripts/base_train.py",), protected_files=("nanochat/loss_eval.py",),
    prepare_cmd="pip install -e .", run_cmd="python train.py",
    metric="loss", lower_is_better=True, budget_s=300,
)


@pytest.mark.parametrize("configured", [False, True])
def test_sandbox_cache_is_portable_and_honors_explicit_configuration(tmp_path, configured):
    env = dict(os.environ)
    env.pop("AUTORESEARCH_SANDBOX_CACHE", None)
    override = str(tmp_path / "allocated-cache")
    if configured:
        env["AUTORESEARCH_SANDBOX_CACHE"] = override
    result = subprocess.run([
        sys.executable, "-B", "-c",
        "from autoresearch.experiments.repo_improve.escalate import SHARED_SANDBOX_CACHE; "
        "print(SHARED_SANDBOX_CACHE)",
    ], env=env, text=True, capture_output=True, check=True)
    cache = Path(result.stdout.strip())
    assert cache.is_absolute()
    if configured:
        assert str(cache) == override
    else:
        assert cache.is_relative_to(Path.home())


class _Setup:
    def __init__(self, verified=True, **kw):
        self.verified = verified
        self.prepare_cmd = kw.get("prepare_cmd", "cd /workspace/repo && uv sync")
        self.run_cmd = kw.get("run_cmd", "cd /workspace/repo && uv run python -m scripts.base_train")
        self.metric = kw.get("metric", "Validation bpb")
        self.lower_is_better = kw.get("lower_is_better", True)
        self.metric_value = kw.get("metric_value", 2.83)
        self.notes = kw.get("notes", "")


class _FakeBackend:
    def __init__(self):
        self.opened = []
        self.closed = []

    def open_sandbox(self, *, cuda_device=None, mounts=None, env=None):
        self.opened.append({"cuda_device": cuda_device, "mounts": list(mounts or []), "env": env})
        return object()

    def close_sandbox(self, handle):
        self.closed.append(handle)


def test_opens_gpu_sandbox_with_workroot_mounted_and_returns_runner():
    be = _FakeBackend()
    events = []
    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="improve it",
        cuda_device=3, llm=None, emit=lambda e, p: events.append((e, p)),
        backend=be, discover_fn=lambda *a, **k: _Setup(),
    )
    assert out is not None
    new_bench, run_variant_fn, close, backend, handle = out

    # 沙箱拿到了指定的卡,且 workroot 被挂进容器(宿主路径=容器路径)
    assert be.opened[0]["cuda_device"] == 3
    assert "/work" in be.opened[0]["mounts"]
    # 发现的安装写回 benchmark;绝对路径 cd 被剥掉(工作区不是 /workspace/repo)
    assert new_bench.prepare_cmd == "uv sync"
    assert new_bench.run_cmd == "uv run python -m scripts.base_train"
    assert new_bench.metric == "Validation bpb"
    # 边界(可改/受保护/预算)不因升级而改变 —— 反作弊边界与安装无关
    assert new_bench.editable_files == BENCH.editable_files
    assert new_bench.protected_files == BENCH.protected_files
    assert new_bench.budget_s == BENCH.budget_s
    assert callable(run_variant_fn) and callable(close)
    assert any(e == "sandbox_escalation" for e, _ in events)

    close()
    assert be.closed, "close 必须真的拆掉沙箱"


def test_unverified_setup_is_rejected_and_sandbox_closed():
    # agent 没能给出经我们硬判的安装 -> 不升级(不能拿没验证的东西继续跑)
    be = _FakeBackend()
    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=None, emit=lambda e, p: None,
        backend=be, discover_fn=lambda *a, **k: _Setup(verified=False, notes="run failed"),
    )
    assert out is None
    assert be.closed, "失败也必须拆沙箱,不能泄漏容器"


def test_no_setup_at_all_is_rejected_and_sandbox_closed():
    be = _FakeBackend()
    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=None, emit=lambda e, p: None,
        backend=be, discover_fn=lambda *a, **k: None,
    )
    assert out is None and be.closed


def test_infrastructure_failure_degrades_gracefully():
    # 没有 docker / openhands 时不能把整条 pipeline 带崩,升级失败就是失败
    class _Boom:
        def open_sandbox(self, **kw):
            raise RuntimeError("docker not available")
        def close_sandbox(self, h):
            pass

    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=None, emit=lambda e, p: None,
        backend=_Boom(), discover_fn=lambda *a, **k: _Setup(),
    )
    assert out is None


def test_transient_transport_failure_reopens_a_fresh_sandbox_and_retries():
    # 发现容器链路瞬时挂掉(reset/timeout)不是对仓库的判决 —— 换个新容器再试一次
    from autoresearch.agents._reuse.openhands_backend import InfraError

    be = _FakeBackend()
    calls = {"n": 0}

    def _flaky_discover(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InfraError("execute_command transport error: [Errno 104] Connection reset by peer")
        return _Setup()

    events = []
    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=None, emit=lambda e, p: events.append((e, p)),
        backend=be, discover_fn=_flaky_discover,
    )
    assert out is not None, "第二次应成功"
    assert calls["n"] == 2 and len(be.opened) == 2, "应换新容器重试(重新 open)"
    assert len(be.closed) == 1, "死掉的第一个容器要被拆掉"
    assert any(p.get("stage") == "transient_retry" for e, p in events if e == "sandbox_escalation")


def test_retries_are_bounded_then_gives_up():
    from autoresearch.agents._reuse.openhands_backend import InfraError

    be = _FakeBackend()

    def _always_reset(*a, **k):
        raise InfraError("connection reset by peer")

    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=None, emit=lambda e, p: None,
        backend=be, discover_fn=_always_reset,
    )
    assert out is None
    assert len(be.opened) == 2, "重试有界(不能无限重开容器)"
    assert len(be.closed) == 2, "每次死掉的容器都要拆"


def test_a_real_rejection_is_not_retried():
    # agent 真的跑了、判定没有可用安装 —— 这是对仓库的判决,重试只会再烧 20 分钟
    be = _FakeBackend()
    calls = {"n": 0}

    def _rejects(*a, **k):
        calls["n"] += 1
        return _Setup(verified=False, notes="no usable setup")

    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=None, emit=lambda e, p: None,
        backend=be, discover_fn=_rejects,
    )
    assert out is None
    assert calls["n"] == 1 and len(be.opened) == 1, "真判决不重试"


def test_pipeline_llm_is_not_handed_to_the_sandbox_agent():
    """真机 e2e 抓到的 bug:pipeline 的 LLM 客户端 != OpenHands SDK 的 LLM。

    把前者传给 agent 会在 get_default_agent(llm=...) 处直接抛异常,升级 20 秒就
    "no usable setup" 假失败。沙箱 agent 必须自己用 build_sandbox_llm() 构造。
    """
    seen = {}

    def _spy_discover(backend, handle, **kw):
        seen.update(kw)
        return _Setup()

    class _PipelineLLM:      # 冒充 pipeline 那个客户端(不是 OpenHands LLM)
        def chat(self, **kw): ...

    out = escalate_to_sandbox(
        workroot="/work", pristine="/work/pristine", benchmark=BENCH, request="r",
        cuda_device=3, llm=_PipelineLLM(), emit=lambda e, p: None,
        backend=_FakeBackend(), discover_fn=_spy_discover,
    )
    assert out is not None
    assert not isinstance(seen.get("llm"), _PipelineLLM), \
        "绝不能把 pipeline 的 LLM 当作沙箱 agent 的 LLM"
