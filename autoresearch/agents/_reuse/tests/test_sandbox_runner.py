"""沙箱执行器:把 repo_improve 的 run_variant 换成"在容器里跑",其余 pipeline 不动。

不碰 docker:用假 backend 断言命令、预算封顶、metric 解析、失败语义与宿主版一致。
"""
from __future__ import annotations

from autoresearch.agents._reuse.sandbox_runner import make_sandbox_run_variant
from autoresearch.experiments.repo_improve.benchmark import Benchmark


class _Res:
    def __init__(self, exit_code=0, stdout="", stderr="", timed_out=False):
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr
        self.timed_out = timed_out


class _FakeBackend:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls: list[tuple] = []

    def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        self.calls.append((cmd, cwd, timeout))
        for key, res in self.responses.items():
            if key in cmd:
                return res
        return _Res(0, "")


BENCH = Benchmark(
    name="nanochat", description="", repo_url="u",
    editable_files=("scripts/base_train.py",), protected_files=("nanochat/loss_eval.py",),
    prepare_cmd="uv sync && uv run python -m scripts.tok_train",
    run_cmd="uv run python -m scripts.base_train --depth=1",
    metric="Validation bpb", lower_is_better=True, budget_s=300,
)


def test_runs_prepare_then_run_in_the_given_workspace_and_parses_metric():
    be = _FakeBackend({"base_train": _Res(0, "step 50 | Validation bpb: 2.8315\n")})
    rv = make_sandbox_run_variant(be, object())

    res = rv(BENCH, "/work/baseline_ws", prepare=True, seed=0)

    assert res.valid is True and res.metric_value == 2.8315
    cmds = [c for c, _, _ in be.calls]
    assert any("tok_train" in c for c in cmds)          # prepare 跑了
    assert any("base_train" in c for c in cmds)         # run 跑了
    # 都在指定工作区里跑(容器内路径 = 宿主路径,靠 -v 挂载对齐)
    assert all(cwd == "/work/baseline_ws" for _, cwd, _ in be.calls)


def test_prepare_can_be_skipped():
    be = _FakeBackend({"base_train": _Res(0, "Validation bpb: 2.0\n")})
    rv = make_sandbox_run_variant(be, object())
    rv(BENCH, "/work/ws", prepare=False)
    assert not any("tok_train" in c for c, _, _ in be.calls)


def test_run_is_capped_at_the_budget_like_the_host_runner():
    # 公平预算:训练命令的超时 = budget_s + 余量,和宿主版语义一致
    be = _FakeBackend({"base_train": _Res(0, "Validation bpb: 2.0\n")})
    rv = make_sandbox_run_variant(be, object(), budget_margin_s=60)
    rv(BENCH, "/work/ws", prepare=False)
    run_timeout = next(t for c, _, t in be.calls if "base_train" in c)
    assert run_timeout == BENCH.budget_s + 60


def test_prepare_failure_fails_closed():
    be = _FakeBackend({"tok_train": _Res(1, "", "no space left on device")})
    rv = make_sandbox_run_variant(be, object())
    res = rv(BENCH, "/work/ws", prepare=True)
    assert res.valid is False and res.metric_value is None
    assert "no space left" in res.error


def test_nonzero_run_fails_closed_but_a_cap_kill_still_scores_at_the_cap():
    # 与宿主版一致:真崩 -> 失败;预算封顶被杀但已打印指标 -> 取封顶时的成绩
    be = _FakeBackend({"base_train": _Res(1, "", "CUDA error")})
    assert make_sandbox_run_variant(be, object())(BENCH, "/w", prepare=False).valid is False

    be2 = _FakeBackend({"base_train": _Res(-9, "step 90 | Validation bpb: 2.5\n", "", timed_out=True)})
    capped = make_sandbox_run_variant(be2, object())(BENCH, "/w", prepare=False)
    assert capped.valid is True and capped.metric_value == 2.5


def test_missing_metric_fails_closed():
    be = _FakeBackend({"base_train": _Res(0, "done, no numbers here\n")})
    res = make_sandbox_run_variant(be, object())(BENCH, "/w", prepare=False)
    assert res.valid is False and "Validation bpb" in res.error


def test_seed_is_exported_when_the_benchmark_declares_a_seed_env():
    seeded = BENCH.__class__(**{**BENCH.__dict__, "seed_env": "NANOCHAT_SEED"})
    be = _FakeBackend({"base_train": _Res(0, "Validation bpb: 2.0\n")})
    make_sandbox_run_variant(be, object())(seeded, "/w", prepare=False, seed=7)
    run_cmd = next(c for c, _, _ in be.calls if "base_train" in c)
    assert "NANOCHAT_SEED=7" in run_cmd


def test_workspace_is_made_container_writable_before_running(tmp_path, monkeypatch):
    """真机 e2e 抓到的:容器用户 UID != 宿主用户,写不了宿主建的挂载目录。

    沙箱自己 clone 的目录是容器 own 的所以能写;pipeline 在宿主建的 variant 工作区
    默认 755/宿主属主 -> 容器里 `uv sync` 建 .venv 直接 "failed to create directory"。
    """
    ws = tmp_path / "baseline_ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "sub" / "f.py").write_text("x")
    ws.chmod(0o755)

    be = _FakeBackend({"base_train": _Res(0, "Validation bpb: 2.0\n")})
    make_sandbox_run_variant(be, object())(BENCH, str(ws), prepare=False)

    # 目录与文件都要让容器可写(否则 .venv / 检查点 / 输出都写不下去)
    assert (ws.stat().st_mode & 0o002), "工作区目录必须对容器可写"
    assert ((ws / "sub").stat().st_mode & 0o002), "子目录同样要可写"


def test_missing_workspace_does_not_crash_the_run(tmp_path):
    # 放宽权限是尽力而为,路径不存在也不能把整轮带崩
    be = _FakeBackend({"base_train": _Res(0, "Validation bpb: 2.0\n")})
    res = make_sandbox_run_variant(be, object())(BENCH, str(tmp_path / "nope"), prepare=False)
    assert res.valid is True
