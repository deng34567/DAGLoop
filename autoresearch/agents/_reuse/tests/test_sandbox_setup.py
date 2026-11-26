"""沙箱内迭代 setup-agent 的测试:纯假件,不碰 docker / 不调真 LLM。

核心不变量:agent 说"跑通了"不算数 —— 我们自己把它给出的命令重跑一遍,
退出码 + 能否解析出 metric 才是判据(hard judge)。
"""
from __future__ import annotations

import json

from autoresearch.agents._reuse.sandbox_setup import (
    DiscoveredSetup,
    SETUP_REPORT_PATH,
    _setup_task_prompt,
    discover_setup_in_sandbox,
)


class _Res:
    def __init__(self, exit_code=0, stdout="", stderr=""):
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr
        self.timed_out = False


class _FakeBackend:
    """按命令前缀返回预设结果;记录所有执行过的命令。"""

    def __init__(self, responses: dict, agent_status="FINISHED"):
        self.responses = responses
        self.agent_status = agent_status
        self.commands: list[str] = []
        self.agent_tasks: list[str] = []

    def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        self.commands.append(cmd)
        for key, res in self.responses.items():
            if key in cmd:
                return res
        return _Res(0, "")

    def run_agent_task(self, handle, task, *, llm=None, cli_mode=True, timeout=None):
        self.agent_tasks.append(task)
        return type("R", (), {"status": self.agent_status, "cost": "0.01"})()


_GOOD_REPORT = json.dumps({
    "prepare_cmd": "uv sync && python -m nanochat.dataset -n 2 && python -m scripts.tok_train",
    "run_cmd": "python -m scripts.base_train --depth=1 --eval-every=10",
    "metric": "Validation bpb",
    "lower_is_better": True,
})


def test_prompt_carries_repo_goal_budget_and_demands_a_real_run():
    p = _setup_task_prompt("https://github.com/karpathy/nanochat", "lower validation bpb", 300, "/workspace/repo")
    assert "nanochat" in p and "lower validation bpb" in p
    assert SETUP_REPORT_PATH in p          # 必须告诉 agent 把结果写哪
    assert "300" in p                       # 预算传达给 agent
    # 关键:必须真的跑起来并打印 metric,不能只写脚本
    low = p.lower()
    assert "actually run" in low or "must run" in low
    assert "metric" in low


def test_happy_path_verifies_by_rerunning_and_returns_setup():
    # agent 写了报告;我们重跑 prepare/run,run 打印出 metric -> verified
    be = _FakeBackend({
        SETUP_REPORT_PATH: _Res(0, _GOOD_REPORT),
        "scripts.base_train": _Res(0, "step 10 | Validation bpb: 3.3304\n"),
    })
    out = discover_setup_in_sandbox(be, object(), repo_url="https://github.com/karpathy/nanochat",
                                    goal="lower validation bpb", budget_s=300)
    assert isinstance(out, DiscoveredSetup)
    assert out.verified is True
    assert out.metric_value == 3.3304
    assert "tok_train" in out.prepare_cmd
    # 克隆过 repo,且确实把 agent 派过工
    assert any("git clone" in c for c in be.commands)
    assert be.agent_tasks and "nanochat" in be.agent_tasks[0]


def test_agent_claims_success_but_rerun_fails_is_not_verified():
    # HARD JUDGE:agent 自述成功,但我们重跑 run_cmd 非零退出 -> 不认
    be = _FakeBackend({
        SETUP_REPORT_PATH: _Res(0, _GOOD_REPORT),
        "scripts.base_train": _Res(1, "", "ModuleNotFoundError: nanochat"),
    })
    out = discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300)
    assert out is not None and out.verified is False
    assert out.metric_value is None


def test_rerun_succeeds_but_no_metric_is_not_verified():
    # 跑通了却打印不出 metric,同样不算数(否则后续没法比较)
    be = _FakeBackend({
        SETUP_REPORT_PATH: _Res(0, _GOOD_REPORT),
        "scripts.base_train": _Res(0, "training finished, bye\n"),
    })
    out = discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300)
    assert out is not None and out.verified is False


def test_missing_or_malformed_report_returns_none():
    for bad in (_Res(1, ""), _Res(0, "not json"), _Res(0, json.dumps({"prepare_cmd": ""}))):
        be = _FakeBackend({SETUP_REPORT_PATH: bad})
        assert discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300) is None


# ---------------------------------------------------------------------------
# 真机第一次跑暴露的问题:agent 在自己 shell 里 `source .venv/activate` 后用裸
# `python`,报告的命令却依赖那个已激活状态 -> 干净 shell 重跑 ModuleNotFoundError。
# 修:提示词明确要求自包含 + 失败时把错误喂回去让它改(有界一轮)。
# ---------------------------------------------------------------------------


def test_prompt_demands_self_contained_commands():
    p = _setup_task_prompt("u", "g", 300, "/workspace/repo")
    low = p.lower()
    # 必须明说:干净 shell、不能依赖已激活的 venv
    assert "fresh shell" in low or "clean shell" in low
    assert "activate" in low          # 点名 source .venv/bin/activate 这个坑
    assert "uv run" in low or ".venv/bin/python" in low   # 给出正确写法


def test_notes_carry_the_actual_error_tail():
    # 光说 "failed on re-run" 没法诊断,必须带上真实报错
    be = _FakeBackend({
        SETUP_REPORT_PATH: _Res(0, _GOOD_REPORT),
        "scripts.base_train": _Res(1, "", "ModuleNotFoundError: No module named 'requests'"),
    })
    out = discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300,
                                    max_fix_rounds=0)
    assert out.verified is False
    assert "requests" in out.notes


class _FixRoundBackend(_FakeBackend):
    """第一轮报告用裸 python(重跑失败),被喂回错误后第二轮改成 .venv/bin/python(通过)。"""

    def __init__(self):
        super().__init__({})
        self.reads = 0

    def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        self.commands.append(cmd)
        if SETUP_REPORT_PATH in cmd:
            self.reads += 1
            body = {"prepare_cmd": "uv sync", "metric": "Validation bpb", "lower_is_better": True}
            # 第一次报告用裸 python(会失败);被喂回错误后第二次改成 .venv/bin/python
            body["run_cmd"] = ("python -m scripts.base_train" if self.reads == 1
                               else ".venv/bin/python -m scripts.base_train")
            return _Res(0, json.dumps(body))
        if "base_train" in cmd:
            if cmd.startswith(".venv/bin/python"):
                return _Res(0, "Validation bpb: 3.21\n")
            return _Res(1, "", "ModuleNotFoundError: No module named 'requests'")
        return _Res(0, "")


def test_failed_verification_is_fed_back_and_a_fix_round_can_succeed():
    be = _FixRoundBackend()
    out = discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300,
                                    max_fix_rounds=1)
    assert out.verified is True
    assert out.metric_value == 3.21
    assert ".venv/bin/python" in out.run_cmd
    # 第二次派工必须把真实报错带给 agent,否则它无从改起
    assert len(be.agent_tasks) == 2
    assert "requests" in be.agent_tasks[1]


def test_fix_rounds_are_bounded():
    # 永远失败的情况下不能无限派工
    be = _FakeBackend({
        SETUP_REPORT_PATH: _Res(0, _GOOD_REPORT),
        "scripts.base_train": _Res(1, "", "boom"),
    })
    out = discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300,
                                    max_fix_rounds=2)
    assert out.verified is False
    assert len(be.agent_tasks) == 3          # 1 次初始 + 2 次修复,到此为止


# ---------------------------------------------------------------------------
# 反作弊:setup agent 在沙箱里能改仓库源码(例如让 base_train.py 直接打印假指标)。
# 硬判必须在"仓库源码已还原"的前提下重跑,否则验证的是被它改过的代码。
# ---------------------------------------------------------------------------


def test_verification_restores_pristine_sources_before_rerunning():
    be = _FakeBackend({
        SETUP_REPORT_PATH: _Res(0, _GOOD_REPORT),
        "scripts.base_train": _Res(0, "Validation bpb: 3.3304\n"),
    })
    out = discover_setup_in_sandbox(be, object(), repo_url="u", goal="g", budget_s=300)
    assert out.verified is True

    # 必须在跑 prepare/run 之前执行过源码还原
    restore_idx = next(i for i, c in enumerate(be.commands) if "checkout" in c)
    run_idx = next(i for i, c in enumerate(be.commands) if "base_train" in c)
    assert restore_idx < run_idx, "还原必须发生在重跑之前"

    # 提示词也必须把这条规则告诉 agent(别白费力气改源码,也断了作弊念头)
    prompt = be.agent_tasks[0].lower()
    assert "do not modify" in prompt and "restored" in prompt

    restore = be.commands[restore_idx]
    assert "git" in restore and "checkout" in restore
    # 只还原被跟踪文件;绝不能删掉未跟踪的 .venv / 数据缓存(否则每轮都要重装重下)
    assert "-fdx" not in restore and "clean" not in restore
