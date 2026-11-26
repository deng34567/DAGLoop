"""``_parse_agentic_result``:远程 agentic run 的 RESULT_JSON 标记优先于退出码。

根因回归:agent SDK / docker / httpx 常留一个非 daemon 线程,活干完后卡住解释器退出,
进程被 os._exit / kill → 非零退出,但 RESULT_JSON 早已 flush 到 stdout。旧逻辑先看退出码
就把真实的 baseline+method 值丢了(远程 GPU run 永卡 ImplementingExperiment,"出了值不上报")。
现在:有 RESULT_JSON 就用它,哪怕退出码非零;没有标记才按退出码报错。纯函数,无 ssh/网络。
"""
from __future__ import annotations

import json
import base64

import pytest

from autoresearch.agents.runner_agent import core
from autoresearch.agents.runner_agent.handler import _parse_agentic_result


def test_dgx_driver_decodes_baseline_and_method_contracts():
    """Catch the remote driver silently falling back to a combined baseline+method run."""
    from scripts.dgx.agentic_run import _decode_execution_contract

    assert _decode_execution_contract({"AR_EXECUTION_PHASE": "baseline"}) == (
        "baseline", None, None, None)

    frozen = {
        "metric_value": 1.8,
        "metric_name": "val_loss",
        "lower_is_better": True,
        "fixed_config": "steps=1000",
        "repo_commit": "0123456789abcdef",
    }
    encoded = base64.b64encode(json.dumps(frozen).encode()).decode()
    assert _decode_execution_contract({
        "AR_EXECUTION_PHASE": "method",
        "AR_FROZEN_BASELINE_B64": encoded,
    }) == ("method", frozen, None, None)


def test_dgx_driver_rejects_method_without_frozen_contract():
    from scripts.dgx.agentic_run import _decode_execution_contract

    with pytest.raises(ValueError, match="frozen baseline"):
        _decode_execution_contract({"AR_EXECUTION_PHASE": "method"})


def _stdout_with_result(payload: dict, *, trailing_noise: str = "") -> str:
    return (
        "…agent working…\n"
        "EMIT agentic_run {\"stage\": \"agent_done\"}\n"
        "RESULT_JSON " + json.dumps(payload) + "\n" + trailing_noise
    )


def test_result_json_used_on_clean_exit():
    rr = _parse_agentic_result(_stdout_with_result({"baseline": 1.4674, "candidate": 1.4576}), 0)
    assert rr == {"baseline": 1.4674, "candidate": 1.4576}


def test_result_json_used_despite_nonzero_exit():
    # THE regression: process produced the result, then hung at shutdown and was killed (exit 137).
    # The value must still come through, not be discarded on the post-result exit code.
    rr = _parse_agentic_result(_stdout_with_result({"baseline": 3.2798, "candidate": 3.2492}), 137)
    assert rr == {"baseline": 3.2798, "candidate": 3.2492}


def test_last_result_json_wins_when_multiple():
    # a retrying agent may print more than one marker; the last one is the final answer
    stdout = ("RESULT_JSON " + json.dumps({"candidate": 1.0}) + "\n"
              "RESULT_JSON " + json.dumps({"candidate": 0.9}) + "\n")
    assert _parse_agentic_result(stdout, 0) == {"candidate": 0.9}


def test_no_marker_raises_with_exit_code():
    with pytest.raises(core.RunnerParseError) as ei:
        _parse_agentic_result("agent crashed before any result\n", 1)
    assert "no RESULT_JSON" in str(ei.value)
    assert "exit 1" in str(ei.value)


def test_unparseable_marker_raises():
    with pytest.raises(core.RunnerParseError) as ei:
        _parse_agentic_result("RESULT_JSON {not valid json\n", 0)
    assert "unparseable" in str(ei.value)


def test_empty_stdout_raises():
    with pytest.raises(core.RunnerParseError):
        _parse_agentic_result("", 0)


def test_empty_stdout_surfaces_remote_stderr():
    with pytest.raises(core.RunnerParseError, match="remote import failed"):
        _parse_agentic_result("", 1, "remote import failed\n")
