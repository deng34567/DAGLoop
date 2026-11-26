"""钉住:真实入口必须把 setup_memory 传进 pipeline,否则记忆层就是死代码。

这条曾经真的漏了 —— 记忆写完了但两个入口都没传,线上等于没有。
用源码断言(而非跑真 pipeline),因为两个入口都要拉起 LLM/GPU 才能跑。
"""
from __future__ import annotations

import inspect
import re


def _src(fn) -> str:
    return inspect.getsource(fn)


def test_web_entrypoint_passes_setup_memory_and_isolates_workroot():
    from autoresearch.web import server

    src = _src(server._run_repo_improve_pipeline)
    assert "setup_memory=" in src, "web 入口必须传 setup_memory,否则记忆层是死代码"
    # 每次运行独立目录:两个提交同时落地不能互相覆盖
    assert "uuid" in src, "workroot 必须按运行隔离"
    # 选卡必须是探测出来的,不能写死卡号
    assert "pick_free_card" in src
    assert not re.search(r'cuda_device\s*=\s*["\']\d', src), "不许写死卡号"


def test_runner_agent_entrypoint_passes_setup_memory():
    from autoresearch.agents.runner_agent import handler

    src = inspect.getsource(handler)
    # 找到本地 repo_improve 调用处,确认带了记忆
    call = src[src.index("res = run_repo_improve("):]
    assert "setup_memory=" in call[:600], "runner_agent 入口必须传 setup_memory"


def test_both_entrypoints_share_one_store_by_default():
    # 两条入口用同一个默认路径 -> 一条路径学到的,另一条也能用
    from autoresearch.web import server
    from autoresearch.agents.runner_agent import handler

    assert "DEFAULT_PATH" in _src(server._run_repo_improve_pipeline)
    assert "DEFAULT_PATH" in inspect.getsource(handler)
