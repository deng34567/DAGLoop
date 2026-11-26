"""Tests for OpenHandsBackend (Docker execution backend) with all openhands /
docker interactions MOCKED -- the default test suite NEVER starts a real
container or makes a real LLM/network call. Real connectivity is exercised
manually / opt-in only (scripts/smoke_openhands_backend.py).

单测约定：全 mock,无 docker、无网络。openhands 导入在被测模块内惰性发生;
本测试文件顶层 import openhands.sdk 仅用于 monkeypatch 目标(已安装,不需 docker)。
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import openhands.sdk  # monkeypatch target; installed, does not need docker
import pytest

from autoresearch.agents._reuse import openhands_backend as ob


def test_module_imports_without_docker_or_openhands():
    # 顶层 import 不得触发 openhands/docker(惰性导入)
    assert ob.DEFAULT_SERVER_IMAGE.startswith("ghcr.io/openhands/agent-server")
    h = ob.SandboxHandle(workspace=object())
    assert h.workspace is not None
    assert ob.CommandResult(exit_code=0).exit_code == 0
    assert ob.RunResult(status="FINISHED").status == "FINISHED"
    b = ob.OpenHandsBackend()
    assert b.server_image == ob.DEFAULT_SERVER_IMAGE


def test_pause_active_conversations_is_scoped_and_retryable():
    calls = []

    class FakeConversation:
        def pause(self):
            calls.append(self)

    first = FakeConversation()
    second = FakeConversation()
    ob._track_active_conversation(first)
    ob._track_active_conversation(second)
    try:
        assert ob.pause_active_conversations() == 2
        assert calls == [first, second]
        assert ob.pause_active_conversations() == 2
        assert calls == [first, second, first, second]
    finally:
        ob._untrack_active_conversation(first)
        ob._untrack_active_conversation(second)

    assert ob.pause_active_conversations() == 0


def test_sandbox_extra_args_enforces_manifest_resources():
    args = ob._sandbox_extra_args(env={
        "AUTORESEARCH_TASK_CPUS": "2",
        "AUTORESEARCH_TASK_MEMORY_MB": "4096",
    })

    assert args[:6] == [
        "--cpus", "2.0", "--memory", "4096m", "--memory-swap", "4096m",
    ]
    assert "OMP_NUM_THREADS=2" in args
    assert "MKL_NUM_THREADS=2" in args
    assert "OPENBLAS_NUM_THREADS=2" in args


def test_sandbox_extra_args_adds_control_memory_without_changing_task_limit():
    args = ob._sandbox_extra_args(env={
        "AUTORESEARCH_TASK_MEMORY_MB": "512",
        "AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB": "512",
    })

    assert args[:4] == [
        "--memory", "1024m", "--memory-swap", "1024m",
    ]
    assert "AUTORESEARCH_TASK_MEMORY_MB=512" in args


@pytest.mark.parametrize("explicit,expected", [(None, "1024m"), ("256", "768m")])
def test_sandbox_control_memory_uses_process_config_with_explicit_override(monkeypatch, explicit, expected):
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB", "512")
    env = {"AUTORESEARCH_TASK_MEMORY_MB": "512"}
    if explicit is not None:
        env["AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB"] = explicit
    args = ob._sandbox_extra_args(env=env)
    assert args[args.index("--memory") + 1] == expected
    assert "AUTORESEARCH_TASK_MEMORY_MB=512" in args


def test_sandbox_extra_args_does_not_guess_invalid_resources():
    args = ob._sandbox_extra_args(env={
        "AUTORESEARCH_TASK_CPUS": "unknown",
        "AUTORESEARCH_TASK_MEMORY_MB": "0",
    })

    assert "--cpus" not in args
    assert "--memory" not in args


# -- Task 2: build_sandbox_llm --

def test_build_sandbox_llm_openrouter(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_MODEL", raising=False)
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_LLM_RETRIES", raising=False)
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_LLM_STREAM", raising=False)
    ob.build_sandbox_llm()
    assert captured["model"] == "openai/moonshotai/kimi-k2"  # provider 前缀
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["disable_vision"] is True
    assert captured["num_retries"] == 2
    assert "stream" not in captured  # 绝不 stream
    assert "litellm_extra_body" not in captured


@pytest.mark.parametrize(
    "reasoning,stream_setting,expected",
    [("none", None, None), ("max", None, True), ("max", "0", False), ("none", "1", True)],
)
def test_build_sandbox_llm_streams_max_reasoning_unless_overridden(
    monkeypatch, reasoning, stream_setting, expected,
):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_REASONING_EFFORT", reasoning)
    if stream_setting is None:
        monkeypatch.delenv("AUTORESEARCH_SANDBOX_LLM_STREAM", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_STREAM", stream_setting)

    ob.build_sandbox_llm()

    assert captured.get("stream") is expected


def test_build_agent_keeps_streaming_main_llm_and_nonstreaming_condenser():
    llm = ob.build_sandbox_llm(
        api_key="offline-placeholder", base_url="http://127.0.0.1:1", model="offline-model",
    ).model_copy(update={"stream": True})

    agent = ob._build_agent(llm, cli_mode=True)

    assert agent.llm.stream is True
    assert agent.condenser.llm.stream is False


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_explicit_reasoning_effort_survives_unknown_model_capabilities(monkeypatch, effort):
    from openhands.sdk.llm.options import chat_options

    monkeypatch.setenv("AUTORESEARCH_SANDBOX_REASONING_EFFORT", effort)
    monkeypatch.setattr(chat_options, "get_features", lambda _: SimpleNamespace(
        supports_reasoning_effort=False,
        supports_extended_thinking=False,
        supports_prompt_cache_retention=False,
    ))
    llm = ob.build_sandbox_llm(
        api_key="offline-placeholder", base_url="http://127.0.0.1:1",
        model="unlisted-deployment-model",
    )
    options = chat_options.select_chat_options(llm, {}, has_tools=False)
    assert "reasoning_effort" not in options  # SDK capability gate still applies.
    assert options["extra_body"]["reasoning_effort"] == effort


@pytest.mark.parametrize("setting", ["none", "default", "max"])
@pytest.mark.parametrize("temperature", [None, "0.7", "0.0"])
@pytest.mark.parametrize("model", ["glm-5.2", "unlisted-deployment-model"])
@pytest.mark.parametrize("extra_body", [{}, {"thinking": {"type": "disabled"}}])
def test_explicit_reasoning_effort_reaches_http_body_without_network(
    monkeypatch, setting, temperature, model, extra_body,
):
    import httpx
    import litellm
    from openai import OpenAI
    from openhands.sdk.llm.options.chat_options import select_chat_options

    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 0,
            "model": "unlisted-deployment-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.setenv("AUTORESEARCH_SANDBOX_REASONING_EFFORT", setting)
    monkeypatch.setenv("AUTORESEARCH_LLM_EXTRA_BODY", json.dumps(extra_body))
    if temperature is None:
        monkeypatch.delenv("AUTORESEARCH_LLM_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_LLM_TEMPERATURE", temperature)
    llm = ob.build_sandbox_llm(api_key="offline-placeholder",
        base_url="http://offline.invalid/v1", model=model)
    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as http:
        client = OpenAI(api_key="offline-placeholder", base_url="http://offline.invalid/v1",
                        http_client=http)
        litellm.completion(model=llm.model, messages=[{"role": "user", "content": "offline"}],
            client=client, drop_params=True, **select_chat_options(llm, {}, has_tools=False))
    assert len(seen) == 1
    if temperature is None:
        assert "temperature" not in seen[0]
    else:
        assert seen[0]["temperature"] == float(temperature)
    assert "top_p" not in seen[0]
    for key, value in extra_body.items():
        assert seen[0][key] == value
    if setting == "default":
        assert llm.reasoning_effort is None
        assert "reasoning_effort" not in seen[0]
    else:
        assert seen[0]["reasoning_effort"] == setting


@pytest.mark.parametrize("setting,expected", [(None, True), ("1", True), ("0", False)])
def test_build_sandbox_llm_native_tool_calling_is_configurable(monkeypatch, setting, expected):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    if setting is None:
        monkeypatch.delenv("AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", setting)
    ob.build_sandbox_llm()
    assert captured["native_tool_calling"] is expected


def test_build_sandbox_llm_retry_count_is_configurable(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_RETRIES", "3")
    monkeypatch.setenv("HKGAI_STREAM_RETRY_BASE_S", "300")
    monkeypatch.setenv("HKGAI_STREAM_RETRY_CAP_S", "300")

    ob.build_sandbox_llm()

    assert captured["num_retries"] == 4
    assert captured["retry_min_wait"] == 300
    assert captured["retry_max_wait"] == 300


@pytest.mark.parametrize("setting,retries", [(None, 1), ("0", 0), ("1", 1), ("3", 3)])
def test_sandbox_retry_setting_waits_before_each_sdk_retry(monkeypatch, setting, retries):
    from openhands.sdk.llm.utils.retry_mixin import RetryMixin

    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    monkeypatch.setenv("HKGAI_STREAM_RETRY_BASE_S", "300")
    monkeypatch.setenv("HKGAI_STREAM_RETRY_CAP_S", "300")
    if setting is None:
        monkeypatch.delenv("AUTORESEARCH_SANDBOX_LLM_RETRIES", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_RETRIES", setting)
    ob.build_sandbox_llm()
    calls, sleeps = [], []

    @RetryMixin().retry_decorator(
        num_retries=captured["num_retries"],
        retry_exceptions=(TimeoutError,),
        retry_min_wait=captured["retry_min_wait"],
        retry_max_wait=captured["retry_max_wait"],
    )
    def request():
        calls.append(None)
        if len(calls) <= retries:
            raise TimeoutError("synthetic provider timeout")
        return "completed"

    request.retry.sleep = sleeps.append
    assert request() == "completed"
    assert len(calls) == retries + 1
    assert sleeps == [300.0] * retries


def test_build_sandbox_llm_uses_shared_max_output_tokens(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("AUTORESEARCH_LLM_MAX_TOKENS", "262144")

    ob.build_sandbox_llm()

    assert captured["max_output_tokens"] == 262144


def test_build_sandbox_llm_reuses_main_configured_base_url(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openhands.sdk, "LLM", FakeLLM)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://llm.example/v1")
    monkeypatch.delenv("AUTORESEARCH_SANDBOX_BASE_URL", raising=False)

    ob.build_sandbox_llm()

    assert captured["base_url"] == "https://llm.example/v1"


def test_build_sandbox_llm_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    with pytest.raises(ob.InfraError):
        ob.build_sandbox_llm()


# -- Task 3: open_sandbox / close_sandbox --

def test_open_sandbox_uses_factory_and_enters():
    entered = {"n": 0}

    class FakeWS:
        def __enter__(self):
            entered["n"] += 1
            return self

        def __exit__(self, *a):
            pass

    b = ob.OpenHandsBackend()
    h = b.open_sandbox(workspace_factory=FakeWS)
    assert entered["n"] == 1 and isinstance(h, ob.SandboxHandle)


def test_close_sandbox_best_effort():
    class FakeWS:
        def __exit__(self, *a):
            raise RuntimeError("boom")

    b = ob.OpenHandsBackend()
    assert b.close_sandbox(ob.SandboxHandle(workspace=FakeWS())) is False  # 不得抛


# -- commit_sandbox: 冻结容器成镜像 --

def test_commit_sandbox_runs_docker_commit_with_container_id(monkeypatch):
    class WS:  # DockerWorkspace 在 _container_id 里放容器 id
        _container_id = "abc123def"
    seen = {}

    def _fake_run(argv, **kw):
        seen["argv"] = argv
        class _R: returncode, stderr = 0, ""
        return _R()

    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)
    b = ob.OpenHandsBackend()
    ok = b.commit_sandbox(ob.SandboxHandle(workspace=WS()), "autoresearch-setup-xyz")
    assert ok is True
    assert seen["argv"] == ["docker", "commit", "abc123def", "autoresearch-setup-xyz"]


def test_commit_sandbox_no_container_id_returns_false():
    class WS:
        _container_id = None
    b = ob.OpenHandsBackend()
    assert b.commit_sandbox(ob.SandboxHandle(workspace=WS()), "img") is False


def test_commit_sandbox_never_raises_on_docker_failure(monkeypatch):
    class WS:
        _container_id = "cid"
    import subprocess

    def _boom(*a, **k):
        raise OSError("docker gone")
    monkeypatch.setattr(subprocess, "run", _boom)
    b = ob.OpenHandsBackend()
    assert b.commit_sandbox(ob.SandboxHandle(workspace=WS()), "img") is False   # 冻结是 best-effort


# -- Task 4: execute_command --

def test_execute_command_returns_nonzero_not_raised():
    class Raw:
        exit_code = 3
        stdout = "out"
        stderr = "err"

    class FakeWS:
        def execute_command(self, cmd, cwd=None, timeout=None):
            return Raw()

    b = ob.OpenHandsBackend()
    r = b.execute_command(ob.SandboxHandle(workspace=FakeWS()), "python x.py")
    assert r.exit_code == 3 and r.stdout == "out"  # 非零照常返回


def test_execute_command_as_root_targets_only_the_live_sandbox(monkeypatch):
    class WS:
        _container_id = "sandbox-cid"

    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="restored\n", stderr="")

    import subprocess
    monkeypatch.setattr(subprocess, "run", run)
    result = ob.OpenHandsBackend().execute_command_as_root(
        ob.SandboxHandle(workspace=WS()), "cp -- /sealed/file /app/output",
        cwd="/workspace/repo", timeout=37)

    assert result.exit_code == 0 and result.stdout == "restored\n"
    assert seen["argv"] == [
        "docker", "exec", "--user", "0", "--workdir", "/workspace/repo",
        "sandbox-cid", "bash", "-lc", "cp -- /sealed/file /app/output",
    ]
    assert seen["kwargs"]["timeout"] == 37


def test_execute_command_transport_error_is_infra():
    class FakeWS:
        def execute_command(self, *a, **k):
            raise ConnectionError("docker gone")

    b = ob.OpenHandsBackend()
    with pytest.raises(ob.InfraError):
        b.execute_command(ob.SandboxHandle(workspace=FakeWS()), "x")


def test_execute_command_retries_transient_blip_then_succeeds(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_a: None)   # 别真睡

    class Raw:
        exit_code, stdout, stderr, timeout_occurred = 0, "ok", "", False

    class FakeWS:
        n = 0
        def execute_command(self, *a, **k):
            FakeWS.n += 1
            if FakeWS.n == 1:
                raise RuntimeError("[Errno 104] Connection reset by peer")   # 瞬时闪断
            return Raw()

    b = ob.OpenHandsBackend()
    r = b.execute_command(ob.SandboxHandle(workspace=FakeWS()), "cat report.json")
    assert r.exit_code == 0 and r.stdout == "ok"
    assert FakeWS.n == 2, "闪断后应换新连接重试成功"


def test_execute_command_bounded_retries_then_gives_up(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_a: None)
    calls = {"n": 0}

    class FakeWS:
        def execute_command(self, *a, **k):
            calls["n"] += 1
            raise RuntimeError("Server disconnected without sending a response.")

    b = ob.OpenHandsBackend()
    with pytest.raises(ob.InfraError):
        b.execute_command(ob.SandboxHandle(workspace=FakeWS()), "x")
    assert calls["n"] == ob._EXEC_TRANSIENT_RETRIES + 1, "重试有界(不能无限)"


def test_execute_command_timeout_is_not_retried(monkeypatch):
    # 真超时不该重试(否则又白跑一个完整预算);只有掉线才重试
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_a: None)
    calls = {"n": 0}

    class FakeWS:
        def execute_command(self, *a, **k):
            calls["n"] += 1
            raise RuntimeError("read timeout after 1800s")

    b = ob.OpenHandsBackend()
    with pytest.raises(ob.InfraError):
        b.execute_command(ob.SandboxHandle(workspace=FakeWS()), "x")
    assert calls["n"] == 1, "超时立即失败,不重试"


# -- Task 5: run_agent_task --

def test_run_agent_task_runs_conversation_and_collects_status(monkeypatch):
    calls = {"run": 0, "conversation_kwargs": None}

    class FakeStatus:
        name = "FINISHED"

    class FakeMetrics:
        accumulated_cost = 0.42

    class FakeStats:
        def get_combined_metrics(self):
            return FakeMetrics()

    class FakeConvo:
        def __init__(self, agent=None, workspace=None, **kw):
            calls["conversation_kwargs"] = kw
            self.workspace = workspace
            self.state = type("S", (), {"execution_status": FakeStatus()})()
            self.state.events = [SimpleNamespace(
                action=SimpleNamespace(kind="FinishAction", message="Evaluator not run."),
                thought="Do not retain private deliberation",
            )]
            self.conversation_stats = FakeStats()

        def send_message(self, t):
            self.msg = t

        def run(self):
            calls["run"] += 1

        def close(self):
            pass

    monkeypatch.setattr(openhands.sdk, "Conversation", FakeConvo, raising=False)
    # ★ get_default_agent 在 openhands.tools.preset.default,不在 sdk —— 必须 patch 那里
    monkeypatch.setattr(
        "openhands.tools.preset.default.get_default_agent",
        lambda llm=None, cli_mode=False: object(),
        raising=False,
    )
    b = ob.OpenHandsBackend()
    r = b.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "write hello.py", llm=object()
    )
    assert calls["run"] == 1
    assert calls["conversation_kwargs"]["stuck_detection"] is False
    assert calls["conversation_kwargs"]["visualizer"] is None
    assert calls["conversation_kwargs"]["token_callbacks"] == [ob._discard_stream_token]
    assert r.status == "FINISHED" and r.cost == "0.42"  # 状态/成本确实取自 convo
    assert r.event_summary == "Agent finish message (unverified): Evaluator not run."


def test_terminal_summary_is_bounded_and_redacts_configured_credentials(monkeypatch):
    monkeypatch.setenv("SANDBOX_API_KEY", "test-secret-value")
    event = SimpleNamespace(action=SimpleNamespace(
        kind="FinishAction", message="test-secret-value " + "x" * 4000))
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert len(summary) == 2000
    assert "test-secret-value" not in summary and "[REDACTED]" in summary


def test_terminal_summary_keeps_error_code_not_raw_error_details():
    event = SimpleNamespace(kind="ConversationErrorEvent", code="AuthenticationError",
                            detail="private request headers")
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    assert ob._terminal_event_summary(convo) == "Conversation error code: AuthenticationError"


@pytest.mark.parametrize("secret_name", ["SANDBOX_API_KEY", "OPENROUTER_KEY"])
def test_protocol_failure_logs_redacted_reply_without_feeding_markup_back(
        monkeypatch, caplog, secret_name):
    monkeypatch.setenv(secret_name, "test-secret-value")
    reply = "<tool_call>terminal><parameter=command>test-secret-value</parameter>"
    event = SimpleNamespace(kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
        role="assistant", content=[SimpleNamespace(text=reply)]))
    convo = SimpleNamespace(id="diagnostic-conversation", state=SimpleNamespace(
        execution_status=SimpleNamespace(name="FINISHED"), events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert "<tool_call>" not in summary
    assert "<tool_call>" in caplog.text and "diagnostic-conversation" in caplog.text
    assert "test-secret-value" not in caplog.text and "[REDACTED]" in caplog.text


@pytest.mark.parametrize("status,source,role,expected", [
    ("FINISHED", "agent", "assistant", True),
    ("RUNNING", "agent", "assistant", False),
    ("FINISHED", "user", "user", False),
])
def test_terminal_summary_keeps_only_finished_public_assistant_reply(
        monkeypatch, status, source, role, expected):
    monkeypatch.setenv("SANDBOX_API_KEY", "test-secret-value")
    event = SimpleNamespace(
        kind="MessageEvent", source=source, thought="private-thought",
        llm_message=SimpleNamespace(
            role=role, reasoning_content="private-reasoning",
            content=[SimpleNamespace(text="Download pending; test-secret-value")]))
    convo = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(name=status), events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert summary == (
        "Agent final reply (unverified): Download pending; [REDACTED]" if expected else "")


def test_terminal_summary_does_not_use_reply_preceding_tool_execution():
    reply = SimpleNamespace(kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
        role="assistant", content=[SimpleNamespace(text="Old public reply")]))
    tool = SimpleNamespace(kind="ObservationEvent", observation=SimpleNamespace(
        kind="TerminalObservation", exit_code=0))
    convo = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(name="FINISHED"), events=[reply, tool]))
    assert ob._terminal_event_summary(convo) == ""


@pytest.mark.parametrize("reply,expected", [
    ("<think>private reasoning</think>Download is incomplete.",
     "Agent final reply (unverified): Download is incomplete."),
    ("<think>unfinished private reasoning", ""),
    ("x" * 4000, ("Agent final reply (unverified): " + "x" * 4000)[:2000]),
])
def test_terminal_public_reply_excludes_thinking_and_is_bounded(reply, expected):
    event = SimpleNamespace(kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
        role="assistant", content=[SimpleNamespace(text=reply)]))
    convo = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(name="FINISHED"), events=[event]))
    assert ob._terminal_event_summary(convo) == expected


@pytest.mark.parametrize("reply,protocol_error", [
    ("<tool_call>terminal<arg_key>command</arg_key></tool_call>", True),
    ("<tool_call>function=terminal>\n<parameter=command>reset_terminal</parameter>", True),
    ("No measurement yet.<tool_call>function=terminal>\n<parameter=command>pwd</parameter>", True),
    ("Let me read the task.<tool_call>terminal>\n<parameter=command>pwd</parameter>", True),
    ("Let me check.<tool_call>terminal>(\n- command: pwd\n</parameter>\n<parameter=security_risk>LOW</parameter>", True),
    ("Plan next.<tool_call>task_tracker><parameter=command>plan</parameter>", True),
    ("A documentation example contains <tool_call><arg_key>.", False),
])
def test_terminal_summary_identifies_unexecuted_tool_markup(reply, protocol_error):
    event = SimpleNamespace(kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
        role="assistant", content=[SimpleNamespace(text=reply)]))
    convo = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(name="FINISHED"), events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert summary.startswith("Agent tool protocol failure") is protocol_error
    if protocol_error:
        assert "no tool was executed" in summary
        assert "<tool_call>" not in summary
        assert "<parameter=" not in summary
        assert "<arg_key>" not in summary
    else:
        assert summary.endswith(reply)


def test_terminal_protocol_failure_does_not_erase_earlier_execution():
    success = SimpleNamespace(kind="ObservationEvent", observation=SimpleNamespace(
        kind="TerminalObservation", exit_code=0))
    reply = SimpleNamespace(kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
        role="assistant", content=[SimpleNamespace(
            text="<tool_call>terminal><parameter=command>pwd</parameter>")]))
    convo = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(name="FINISHED"), events=[success, reply]))
    summary = ob._terminal_event_summary(convo)
    assert "from this last reply" in summary
    assert "earlier tool executions" in summary
    assert "existing artifacts and host attempt records" in summary


def test_terminal_summary_preserves_quota_classification_without_request_details():
    event = SimpleNamespace(kind="ConversationErrorEvent", code="InternalServerError",
                            detail="pre_consume_token_quota_failed private-request-secret")
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert "insufficient_quota" in summary
    assert "private-request-secret" not in summary


@pytest.mark.parametrize("detail", [
    "LLMTimeoutError: private-request-secret",
    "litellm.Timeout: OpenAIException private-request-secret",
    "<title>504 Gateway Time-out</title> private-request-secret",
])
def test_terminal_summary_preserves_provider_timeout_without_request_details(detail):
    event = SimpleNamespace(kind="ConversationErrorEvent", code="InternalServerError",
                            detail=detail)
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert "llm_provider_timeout" in summary
    assert "private-request-secret" not in summary


def test_terminal_summary_preserves_provider_content_filter_without_request_details():
    event = SimpleNamespace(
        kind="ConversationErrorEvent", code="APIError",
        detail="Output text May contain sensitive information, please try again. private-request-secret")
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    summary = ob._terminal_event_summary(convo)
    assert "llm_provider_content_filter" in summary
    assert "private-request-secret" not in summary


@pytest.mark.parametrize("cached", [False, True])
def test_terminal_error_fetch_is_bounded_when_event_cache_lags(cached):
    calls = []
    event = {"kind": "ConversationErrorEvent", "code": "InternalServerError",
             "detail": "pre_consume_token_quota_failed private-request-secret"}
    class Client:
        def get(self, path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"items": [event]})
    convo = SimpleNamespace(id="conversation-id", state=SimpleNamespace(
        events=[SimpleNamespace(**event)] if cached else []))
    summary = ob._terminal_event_summary(convo, error_workspace=SimpleNamespace(client=Client()))
    assert "insufficient_quota" in summary
    assert "private-request-secret" not in summary
    assert len(calls) == (0 if cached else 1)
    if not cached:
        assert calls[0] == ("/api/conversations/conversation-id/events/search", {
            "params": {"kind": "openhands.sdk.event.conversation_error.ConversationErrorEvent",
                       "sort_order": "TIMESTAMP_DESC", "limit": 1},
            "timeout": 5.0})


@pytest.mark.parametrize("event_arrives", [True, False])
def test_terminal_error_fetch_waits_boundedly_for_delayed_event(monkeypatch, event_arrives):
    calls, waits = [], []
    monkeypatch.setattr(ob.time, "sleep", waits.append)
    event = {"kind": "ConversationErrorEvent", "code": "InternalServerError",
             "detail": "LLMTimeoutError: private-request-secret"}

    class Client:
        def get(self, path, **kwargs):
            calls.append((path, kwargs))
            items = [event] if event_arrives and len(calls) == 2 else []
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"items": items})

    convo = SimpleNamespace(id="conversation-id", state=SimpleNamespace(events=[]))
    summary = ob._terminal_event_summary(convo, error_workspace=SimpleNamespace(client=Client()))
    assert len(calls) == (2 if event_arrives else 5)
    assert waits == [0.5] * (1 if event_arrives else 4)
    assert "private-request-secret" not in summary
    if event_arrives:
        assert "llm_provider_timeout" in summary
    else:
        assert summary == ""


def test_terminal_error_fetch_uses_sdk_server_qualified_kind(monkeypatch):
    from openhands.sdk.event.conversation_error import ConversationErrorEvent

    monkeypatch.setattr(ob.time, "sleep", lambda _: None)
    expected_kind = ConversationErrorEvent.__module__ + "." + ConversationErrorEvent.__name__

    class Client:
        def get(self, path, **kwargs):
            items = ([{"kind": "ConversationErrorEvent", "code": "APIError",
                       "detail": "insufficient_user_quota"}]
                     if kwargs["params"]["kind"] == expected_kind else [])
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"items": items})

    convo = SimpleNamespace(id="conversation-id", state=SimpleNamespace(events=[]))
    summary = ob._terminal_event_summary(convo, error_workspace=SimpleNamespace(client=Client()))
    assert "insufficient_quota" in summary


def test_terminal_error_fetch_preserves_error_published_after_second_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(ob.time, "sleep", lambda _: None)

    class Client:
        def get(self, path, **kwargs):
            calls.append(path)
            event = {"kind": "ConversationErrorEvent", "code": "APIError",
                     "detail": "insufficient_user_quota private-request-secret"}
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"items": [event] if len(calls) == 3 else []})

    convo = SimpleNamespace(id="conversation-id", state=SimpleNamespace(events=[]))
    summary = ob._terminal_event_summary(convo, error_workspace=SimpleNamespace(client=Client()))
    assert "insufficient_quota" in summary
    assert "private-request-secret" not in summary
    assert len(calls) == 3


@pytest.mark.parametrize("timeout", [None, 5])
@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("delayed", [True, False])
def test_remote_conversation_error_preserves_quota_before_close(monkeypatch, timeout, cached, delayed):
    closed, fetched, runs = [], [], []
    event = {"kind": "ConversationErrorEvent", "code": "InternalServerError",
             "detail": "用户额度不足 private-request-secret"}
    class Client:
        def get(self, path, **kwargs):
            assert not closed
            fetched.append(path)
            items = [] if delayed and len(fetched) == 1 else [event]
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"items": items})
    class FakeConvo(_fake_convo_cls(lambda: None)):
        id = "conversation-id"
        def run(self):
            runs.append(True)
            self.state.events = [SimpleNamespace(**event)] if cached else []
            raise RuntimeError("Remote conversation encountered an error")
        def close(self):
            self.state.events = []
            closed.append(True)
    monkeypatch.setattr(openhands.sdk, "Conversation", FakeConvo, raising=False)
    monkeypatch.setattr("openhands.tools.preset.default.get_default_agent",
                        lambda **kwargs: object())
    with pytest.raises(RuntimeError, match="insufficient_quota") as caught:
        ob.OpenHandsBackend().run_agent_task(
            ob.SandboxHandle(workspace=SimpleNamespace(client=Client())),
            "t", llm=object(), timeout=timeout)
    assert "private-request-secret" not in str(caught.value)
    assert closed == [True]
    assert runs == [True]


def test_terminal_summary_does_not_guess_from_nonterminal_events():
    event = SimpleNamespace(action=SimpleNamespace(kind="TerminalAction", command="false"),
                            thought="a guessed diagnosis")
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    assert ob._terminal_event_summary(convo) == ""
    assert ob._terminal_event_summary(SimpleNamespace(state=SimpleNamespace())) == ""


def test_terminal_summary_unavailable_does_not_replace_execution_failure():
    class BrokenState:
        @property
        def events(self):
            raise OSError("closed transport")
    assert ob._terminal_event_summary(SimpleNamespace(state=BrokenState())) == ""


# -- timeout: optional wall-clock backstop on the in-container agent loop --

def _fake_convo_cls(run_body):
    class FakeConvo:
        def __init__(self, agent=None, workspace=None, **kw):
            self.state = type("S", (), {"execution_status": type("St", (), {"name": "FINISHED"})()})()
            self.conversation_stats = type(
                "Stats", (), {"get_combined_metrics": lambda self: type("M", (), {"accumulated_cost": 0})()}
            )()

        def send_message(self, t):
            pass

        def run(self):
            run_body()

        def close(self):
            pass

    return FakeConvo


def test_run_agent_task_returns_timeout_status_when_run_exceeds_budget(monkeypatch):
    import threading

    release = threading.Event()

    monkeypatch.setattr(openhands.sdk, "Conversation",
                        _fake_convo_cls(lambda: release.wait(5)), raising=False)
    monkeypatch.setattr(
        "openhands.tools.preset.default.get_default_agent",
        lambda llm=None, cli_mode=False: object(), raising=False,
    )
    b = ob.OpenHandsBackend()
    # a run() that would block ~5s, but we only allow 0.05s -> TIMEOUT
    r = b.run_agent_task(ob.SandboxHandle(workspace="WS"), "t", llm=object(), timeout=0.05)
    assert r.status == "TIMEOUT"
    release.set()  # let the daemon thread finish


def test_run_agent_task_pauses_when_stop_predicate_fires(monkeypatch):
    """The caller can enforce an in-sandbox experiment cap without trusting the LLM prompt."""
    import threading

    started = threading.Event()
    release = threading.Event()
    calls = {"pause": 0}

    class FakeConvo(_fake_convo_cls(lambda: (started.set(), release.wait(5)))):
        def pause(self):
            calls["pause"] += 1
            release.set()

    monkeypatch.setattr(openhands.sdk, "Conversation", FakeConvo, raising=False)
    monkeypatch.setattr(
        "openhands.tools.preset.default.get_default_agent",
        lambda llm=None, cli_mode=False: object(), raising=False,
    )
    b = ob.OpenHandsBackend()

    r = b.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "t", llm=object(), timeout=5,
        stop_predicate=lambda: started.is_set(),
    )

    assert calls["pause"] == 1
    assert r.status == "PAUSED_BY_PREDICATE"


def test_run_agent_task_no_timeout_runs_inline(monkeypatch):
    calls = {"run": 0}
    monkeypatch.setattr(openhands.sdk, "Conversation",
                        _fake_convo_cls(lambda: calls.__setitem__("run", calls["run"] + 1)),
                        raising=False)
    monkeypatch.setattr(
        "openhands.tools.preset.default.get_default_agent",
        lambda llm=None, cli_mode=False: object(), raising=False,
    )
    b = ob.OpenHandsBackend()
    r = b.run_agent_task(ob.SandboxHandle(workspace="WS"), "t", llm=object())
    assert calls["run"] == 1 and r.status == "FINISHED"


@pytest.mark.parametrize("returncode,stdout,stderr,lost", [
    (0, "true\n", "", False),
    (0, "false\n", "", True),
    (1, "", "Error: No such object: dead-container", True),
    (1, "", "Cannot connect to the Docker daemon", False),
])
def test_container_check_requires_authoritative_loss(monkeypatch, returncode, stdout, stderr, lost):
    import subprocess

    calls = []
    def inspect(args, **kwargs):
        calls.append(args)
        assert not any(k.lower().endswith("_proxy") for k in kwargs["env"])
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(subprocess, "run", inspect)
    ws = SimpleNamespace(_container_id="dead-container")
    if lost:
        with pytest.raises(ob.InfraError, match="container.*not running"):
            ob._check_sandbox_container(ws)
    else:
        ob._check_sandbox_container(ws)
    assert calls == [["docker", "inspect", "--format", "{{.State.Running}}", "dead-container"]]


def test_dead_container_interrupts_sdk_wait(monkeypatch):
    import threading

    release = threading.Event()
    monkeypatch.setattr(openhands.sdk, "Conversation",
                        _fake_convo_cls(lambda: release.wait(5)), raising=False)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    def lost(workspace):
        assert workspace == "WS"
        raise ob.InfraError("sandbox container is not running")
    monkeypatch.setattr(ob, "_check_sandbox_container", lost, raising=False)
    try:
        with pytest.raises(ob.InfraError, match="not running"):
            ob.OpenHandsBackend().run_agent_task(
                ob.SandboxHandle(workspace="WS"), "t", llm=object(), timeout=1)
    finally:
        release.set()


def test_dead_container_stops_remote_polling_thread(monkeypatch):
    import threading

    release = threading.Event()
    finished = threading.Event()

    class RemoteWait:
        def _poll_status_once(self):
            return "running"

        def run(self, **kwargs):
            try:
                while not release.wait(0.01):
                    self._poll_status_once()
            finally:
                finished.set()

    def lost(workspace):
        raise ob.InfraError("sandbox container is not running")

    monkeypatch.setattr(ob, "_check_sandbox_container", lost)
    try:
        with pytest.raises(ob.InfraError, match="not running"):
            ob.OpenHandsBackend._run_convo(RemoteWait(), 5, workspace="WS")
        assert finished.wait(0.2), "dead sandbox left an SDK polling thread running"
    finally:
        release.set()


@pytest.mark.parametrize("terminal_timeout", [None, 20])
def test_valid_empty_terminal_wait_can_exceed_five_seconds(monkeypatch, terminal_timeout):
    import threading

    release = threading.Event()
    completed = threading.Event()
    paused = []

    class FakeConvo:
        def __init__(self):
            self.state = SimpleNamespace(
                execution_status=SimpleNamespace(name="RUNNING"),
                events=[SimpleNamespace(id="soft-wait", action=SimpleNamespace(
                    kind="TerminalAction", command="", is_input=False,
                    timeout=terminal_timeout))],
            )

        def run(self, **kwargs):
            if release.wait(6.2):
                return
            self.state.events.append(SimpleNamespace(observation="command finished"))
            self.state.execution_status = SimpleNamespace(name="FINISHED")
            completed.set()

        def pause(self):
            paused.append(True)
            release.set()

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT", "1")
    try:
        assert ob.OpenHandsBackend._run_convo(FakeConvo(), 10) == (False, False)
        assert completed.is_set()
        assert not paused
    finally:
        release.set()


def test_container_check_does_not_treat_missing_telemetry_as_loss(monkeypatch):
    import subprocess

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker inspect", 10)
    monkeypatch.setattr(subprocess, "run", unavailable)
    ob._check_sandbox_container(SimpleNamespace(_container_id="known-container"))
    ob._check_sandbox_container(object())


def test_run_agent_task_reraises_crash_on_timeout_path(monkeypatch):
    """A crash inside convo.run() on the timeout daemon thread must PROPAGATE, not be
    swallowed into a stale-status success (else a real agent crash masquerades as
    'no changes' and defeats sandbox_propose's retry+fallback). Seen live on SOL-ExecBench."""
    def _boom():
        raise RuntimeError("TmuxPanePool is not initialized or already closed")

    monkeypatch.setattr(openhands.sdk, "Conversation", _fake_convo_cls(_boom), raising=False)
    monkeypatch.setattr(
        "openhands.tools.preset.default.get_default_agent",
        lambda llm=None, cli_mode=False: object(), raising=False,
    )
    b = ob.OpenHandsBackend()
    # timeout is SET (as the sandbox always does) -> crash runs on the daemon thread
    with pytest.raises(RuntimeError, match="TmuxPanePool"):
        b.run_agent_task(ob.SandboxHandle(workspace="WS"), "t", llm=object(), timeout=5)


# -- Task 6: export_workspace --

def test_export_workspace_tars_downloads_and_extracts(tmp_path):
    import tarfile

    order = []

    class FakeWS:
        def execute_command(self, cmd, cwd=None, timeout=None):
            order.append(("exec", cmd))
            return type("R", (), {"exit_code": 0, "stdout": "", "stderr": ""})()

        def file_download(self, src, dst):
            order.append(("download", src, dst))
            # 往 dst 写一个真 tarball,内含一个文件 hello.txt
            payload = tmp_path / "_payload.txt"
            payload.write_text("hi", encoding="utf-8")
            with tarfile.open(dst, "w:gz") as tf:
                tf.add(payload, arcname="hello.txt")

    b = ob.OpenHandsBackend()
    dest = tmp_path / "out"
    result = b.export_workspace(ob.SandboxHandle(workspace=FakeWS()), dest)
    assert result == dest
    # 调用序:先 tar 命令、再 file_download
    assert order[0][0] == "exec" and "tar czf" in order[0][1]
    assert order[1][0] == "download"
    # 解包后 dest 下有 hello.txt
    assert (dest / "hello.txt").read_text(encoding="utf-8") == "hi"


# -- Review fixes: tar exit-code guard + safe extract --

def test_export_workspace_raises_on_tar_failure_and_skips_download(tmp_path):
    order = []

    class FakeWS:
        def execute_command(self, cmd, cwd=None, timeout=None):
            order.append(("exec", cmd))
            return type("R", (), {"exit_code": 2, "stdout": "", "stderr": "no space"})()

        def file_download(self, src, dst):
            order.append(("download", src, dst))

    b = ob.OpenHandsBackend()
    with pytest.raises(ob.InfraError):
        b.export_workspace(ob.SandboxHandle(workspace=FakeWS()), tmp_path / "out")
    # tar 失败后绝不下载
    assert not any(o[0] == "download" for o in order)


def test_export_workspace_safe_extract_blocks_path_traversal(tmp_path):
    import tarfile

    class FakeWS:
        def execute_command(self, cmd, cwd=None, timeout=None):
            return type("R", (), {"exit_code": 0, "stdout": "", "stderr": ""})()

        def file_download(self, src, dst):
            # 构造含 ../escape 恶意成员 + 正常成员 的 tarball
            good = tmp_path / "_good.txt"
            good.write_text("ok", encoding="utf-8")
            with tarfile.open(dst, "w:gz") as tf:
                tf.add(good, arcname="good.txt")
                tf.add(good, arcname="../escape.txt")

    b = ob.OpenHandsBackend()
    dest = tmp_path / "out"
    # filter="data" 对 ../ 逃逸的成员会抛(阻断);抛与否都可,核心安全属性是:
    # 恶意成员绝不落到 dest 之外。
    with pytest.raises(tarfile.TarError):
        b.export_workspace(ob.SandboxHandle(workspace=FakeWS()), dest)
    # 恶意 ../escape 绝不落到 dest 之外
    assert not (tmp_path / "escape.txt").exists()
    # 正常成员在恶意成员之前照常落地(证明是被 filter 阻断,而非整体没解包)
    assert (dest / "good.txt").read_text(encoding="utf-8") == "ok"


# ---------------------------------------------------------------------------
# GPU 沙箱:手动 device 透传(本机没装 NVIDIA Container Toolkit,--gpus all 不可用)
# 纯函数,不碰 docker/openhands。
# ---------------------------------------------------------------------------


def test_gpu_passthrough_args_builds_devices_and_existing_lib_mounts(tmp_path):
    from autoresearch.agents._reuse.openhands_backend import gpu_passthrough_args

    smi = tmp_path / "nvidia-smi"; smi.write_text("")
    libdir = tmp_path / "lib"; libdir.mkdir()
    (libdir / "libcuda.so").write_text("")       # Triton links with -lcuda
    (libdir / "libcuda.so.1").write_text("")
    (libdir / "libnvidia-ml.so.1").write_text("")   # 存在 -> 应挂载
    # libnvidia-ptxjitcompiler.so.1 故意不建 -> 应跳过(不产生无效挂载)

    args = gpu_passthrough_args(3, lib_dir=str(libdir), smi_path=str(smi))

    # 目标卡 + 控制/UVM 节点都透传
    assert args.count("--device") >= 3
    assert "/dev/nvidia3" in args
    assert "/dev/nvidiactl" in args
    assert "/dev/nvidia-uvm" in args
    # 存在的驱动库以只读挂载
    assert any(str(libdir / "libcuda.so.1") in a and a.endswith(":ro") for a in args)
    cuda_link = libdir / "libcuda.so"
    assert f"{cuda_link}:{cuda_link}:ro" in args
    assert any(str(libdir / "libnvidia-ml.so.1") in a for a in args)
    # 不存在的库被跳过
    assert not any("ptxjitcompiler" in a for a in args)
    # nvidia-smi 挂进去(容器里排查用)
    assert any(str(smi) in a for a in args)


def test_gpu_passthrough_args_multi_card_adds_shm_for_nccl(tmp_path):
    # speedrun 这类 8 卡分布式任务: 传 "0,1,..,7" -> 每张卡各一个 --device, 且补 --shm-size 供 NCCL。
    from autoresearch.agents._reuse.openhands_backend import gpu_passthrough_args

    multi = gpu_passthrough_args("0,1,2,3,4,5,6,7",
                                 lib_dir=str(tmp_path), smi_path=str(tmp_path / "nope"))
    for c in range(8):
        assert f"/dev/nvidia{c}" in multi          # 8 张卡各自透传
    assert "--shm-size" in multi and "16g" in multi  # 多卡补 shm 供单节点 NCCL
    assert "--gpus" not in multi                    # 仍绝不 --gpus all

    single = gpu_passthrough_args(3, lib_dir=str(tmp_path), smi_path=str(tmp_path / "nope"))
    assert "/dev/nvidia3" in single
    assert "--shm-size" not in single               # 单卡保持原行为, 不补 shm


def test_gpu_passthrough_args_never_uses_gpus_all(tmp_path):
    # 关键不变量:本机缺 NVIDIA Container Toolkit,--gpus all 会直接失败,
    # 所以这条路径必须永远只发 --device/-v,绝不出现 --gpus。
    from autoresearch.agents._reuse.openhands_backend import gpu_passthrough_args

    args = gpu_passthrough_args(0, lib_dir=str(tmp_path), smi_path=str(tmp_path / "nope"))
    assert "--gpus" not in args
    assert "all" not in args


def test_inject_docker_run_args_inserts_right_after_run():
    from autoresearch.agents._reuse.openhands_backend import _inject_docker_run_args

    cmd = ["docker", "run", "-d", "--rm", "myimage", "--host", "0.0.0.0"]
    out = _inject_docker_run_args(cmd, ["--device", "/dev/nvidia3"])

    # 插在 `docker run` 之后、其余参数之前(docker 允许选项任意序,但必须在 image 前)
    assert out[:4] == ["docker", "run", "--device", "/dev/nvidia3"]
    assert out[4:] == ["-d", "--rm", "myimage", "--host", "0.0.0.0"]
    assert cmd == ["docker", "run", "-d", "--rm", "myimage", "--host", "0.0.0.0"]  # 不原地改


def test_inject_docker_run_args_leaves_other_commands_alone():
    from autoresearch.agents._reuse.openhands_backend import _inject_docker_run_args

    for cmd in (["docker", "version"], ["docker", "stop", "x"], ["ls"], []):
        assert _inject_docker_run_args(cmd, ["--device", "/dev/nvidia3"]) == cmd


def test_sandbox_start_health_check_temporarily_bypasses_proxy(monkeypatch):
    from openhands.workspace import DockerWorkspace
    from openhands.workspace.docker import workspace as ws_mod

    original_urlopen = ws_mod.urlopen
    seen = {}

    def _fake_start(self, image, context):
        seen["urlopen"] = ws_mod.urlopen

    monkeypatch.setattr(DockerWorkspace, "_start_container", _fake_start)
    ob._make_sandbox_workspace("test-image", "linux/amd64", [])

    assert seen["urlopen"] is not original_urlopen
    assert ws_mod.urlopen is original_urlopen


def test_sandbox_workspace_http_client_bypasses_proxy(monkeypatch):
    import httpx
    from openhands.workspace import DockerWorkspace

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(DockerWorkspace, "_start_container", lambda *args: None)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    workspace = ob._make_sandbox_workspace("test-image", "linux/amd64", [])
    assert workspace.client is not None
    assert captured["trust_env"] is False


def test_sandbox_workspace_websocket_bypasses_proxy_only_for_loopback(monkeypatch):
    from openhands.sdk.conversation.impl import remote_conversation
    from openhands.workspace import DockerWorkspace

    captured = []

    def fake_connect(uri, *args, **kwargs):
        captured.append((uri, kwargs))
        return object()

    monkeypatch.setattr(remote_conversation.websockets, "connect", fake_connect)
    monkeypatch.setattr(DockerWorkspace, "_start_container", lambda *args: None)
    ob._make_sandbox_workspace("test-image", "linux/amd64", [])
    patched = remote_conversation.websockets.connect

    patched("ws://127.0.0.1:8000/events")
    patched("wss://events.example.com/socket")

    assert captured[0][1]["proxy"] is None
    assert "proxy" not in captured[1][1]


def test_bind_mount_args_map_host_paths_to_the_same_container_path(tmp_path):
    # pipeline 在宿主建工作区、容器里执行,两边必须是同一个路径才能对上
    from autoresearch.agents._reuse.openhands_backend import bind_mount_args

    d = tmp_path / "workroot"; d.mkdir()
    args = bind_mount_args([str(d), "/nope/missing"])
    assert args == ["-v", f"{d}:{d}"]          # 存在的才挂;同路径映射;可写(pipeline 要写)


def test_bind_mount_args_accepts_explicit_container_alias(tmp_path):
    """A fixed harness path such as /data can reuse an existing host dataset directly."""
    from autoresearch.agents._reuse.openhands_backend import bind_mount_args

    d = tmp_path / "nanochat"; d.mkdir()
    args = bind_mount_args([(str(d), "/data")])
    assert args == ["-v", f"{d}:/data"]


def test_sandbox_extra_args_combine_gpu_and_mounts(tmp_path):
    from autoresearch.agents._reuse.openhands_backend import _sandbox_extra_args

    d = tmp_path / "w"; d.mkdir()
    args = _sandbox_extra_args(cuda_device=3, mounts=[str(d)])
    assert "--device" in args and "/dev/nvidia3" in args
    assert "-v" in args and f"{d}:{d}" in args
    assert "--gpus" not in args

    # 只挂载不要 GPU 也要能用(CPU 仓库)
    cpu_only = _sandbox_extra_args(cuda_device=None, mounts=[str(d)])
    assert "--device" not in cpu_only and f"{d}:{d}" in cpu_only


@pytest.mark.parametrize("options", [{}, {"env": {"TEST_SETTING": "1"}}, {"cuda_device": 3}])
def test_every_production_sandbox_starts_with_an_orphan_reaper(monkeypatch, options):
    seen = []

    class Workspace:
        def __enter__(self):
            return self

    def make_workspace(image, platform, args):
        seen.append(args)
        return Workspace()

    monkeypatch.setattr(ob, "_make_sandbox_workspace", make_workspace)
    monkeypatch.setattr("openhands.workspace.DockerWorkspace", lambda **kwargs: Workspace())
    ob.OpenHandsBackend().open_sandbox(**options)

    assert len(seen) == 1
    assert seen[0].count("--init") == 1
    assert ("--device" in seen[0]) is (options.get("cuda_device") is not None)


@pytest.mark.parametrize("text,source,role,has_action,expected", [
    ("<tool_call>terminal<arg_key>command", "agent", "assistant", False, True),
    ("<tool_call>function=terminal>\n<parameter=command>reset_terminal</parameter>", "agent", "assistant", False, True),
    ("No measurement yet.<tool_call>function=terminal>\n<parameter=command>pwd</parameter>", "agent", "assistant", False, True),
    ("Environment ready.<tool_call>task_tracker>\n<parameter=command>plan</parameter>", "agent", "assistant", False, True),
    ("Next step.<tool_call>function=file_editor>\n<parameter=command>view</parameter>", "agent", "assistant", False, True),
    ("Let me verify.<tool_call>function=think⟩\n<parameter=thought>Check the implementation</parameter>", "agent", "assistant", False, True),
    ("Documentation: <tool_call>function=think⟩ without arguments.", "agent", "assistant", False, False),
    ("<tool_call>future_tool><parameter=target>workspace</parameter>", "agent", "assistant", False, True),
    ("<task-)</think><tool_call>terminal<arg_key>command", "agent", "assistant", False, True),
    ("Here is an example: <tool_call><arg_key>", "agent", "assistant", False, False),
    ("Tool documentation mentions <tool_call>task_tracker> without arguments.", "agent", "assistant", False, False),
    ("<tool_call><arg_key>", "user", "user", False, False),
    ("Finished successfully", "agent", "assistant", False, False),
    ("<tool_call><arg_key>", "agent", "assistant", True, False),
])
def test_tool_markup_repair_only_recognizes_unexecuted_terminal_reply(
        text, source, role, has_action, expected):
    events = [SimpleNamespace(
        kind="MessageEvent", source=source,
        llm_message=SimpleNamespace(role=role, content=[SimpleNamespace(text=text)]),
        thought="<tool_call><arg_key>private reasoning is not inspected")]
    if has_action:
        events.append(SimpleNamespace(kind="ActionEvent"))
    result = ob._tool_markup_repair_message(SimpleNamespace(state=SimpleNamespace(events=events)))
    assert bool(result) is expected
    if expected:
        assert '{"command":" ","reset":false}' in result
        assert "already-running" in result


def test_emulated_function_with_wrong_parameters_gets_bounded_protocol_repair():
    event = SimpleNamespace(kind="MessageEvent", source="agent",
        llm_message=SimpleNamespace(role="assistant", content=[SimpleNamespace(
            text="<function=think><parameter=security_risk>LOW</parameter></function>")]))
    state = SimpleNamespace(events=[event])
    repair = ob._tool_markup_repair_message(
        SimpleNamespace(state=state), native_tool_calling=False)
    assert "did not execute" in repair
    assert "<function=think>" in repair
    assert "thought and optional summary" in repair
    state.events.append(SimpleNamespace(kind="ActionEvent"))
    assert ob._tool_markup_repair_message(
        SimpleNamespace(state=state), native_tool_calling=False) == ""


@pytest.mark.parametrize("timed_out,stopped", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("native_tool_calling", [True, False])
def test_tool_markup_repair_is_bounded_and_preserves_deadline_and_stop_predicate(
        monkeypatch, timed_out, stopped, native_tool_calling):
    messages, runs = [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = [SimpleNamespace(
                kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
                    role="assistant", content=[SimpleNamespace(
                        text="<tool_call>terminal<arg_key>command")]))]
        def send_message(self, text):
            messages.append(text)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    clock = [100.0]
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    predicate = lambda: False
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        clock[0] = 140.0
        return timed_out, stopped
    monkeypatch.setattr(backend, "_run_convo", run)
    backend.run_agent_task(ob.SandboxHandle(workspace="WS"), "task",
                           llm=SimpleNamespace(native_tool_calling=native_tool_calling),
                           timeout=60, stop_predicate=predicate)
    if timed_out or stopped:
        assert len(runs) == 1
    else:
        assert len(runs) == 4  # Initial run plus at most three protocol repairs.
        assert runs[0][0] is runs[1][0]
        assert runs[1][1] == 20.0
        assert runs[1][2] is predicate
        if native_tool_calling:
            assert "structured tool interface" in messages[-1]
        else:
            assert "<function=terminal>" in messages[-1]


@pytest.mark.parametrize("tool_progress", [False, True])
def test_protocol_failure_gets_one_fresh_context(monkeypatch, tool_progress):
    created, runs = [], []
    clock = [100.0]
    completed = [False]

    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.workspace = kwargs["workspace"]
            self.messages = []
            self.state.events = []
            created.append(self)

        def send_message(self, text):
            self.messages.append(text)

    def malformed():
        return SimpleNamespace(kind="MessageEvent", source="agent",
            llm_message=SimpleNamespace(role="assistant", content=[SimpleNamespace(
                text="<tool_call>terminal<arg_key>command")]))

    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout))
        clock[0] += 10
        if convo is created[0]:
            if len(runs) == 1 and tool_progress:
                convo.state.events.extend([
                    SimpleNamespace(kind="ActionEvent"),
                    SimpleNamespace(kind="ObservationEvent",
                        observation=SimpleNamespace(is_error=False)),
                ])
            convo.state.events.append(malformed())
        else:
            completed[0] = True
        return False, False

    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ob.OpenHandsBackend, "_run_convo", staticmethod(run))
    result = ob.OpenHandsBackend().run_agent_task(
        ob.SandboxHandle(workspace="WS"), "original task",
        llm=SimpleNamespace(native_tool_calling=False), timeout=1200,
        completion_check=lambda: None if completed[0] else "TASK EXECUTION INCOMPLETE: save result")

    assert result.status == "FINISHED"
    assert len(created) == 2
    assert len(runs) == 5
    assert all(convo.workspace == "WS" for convo in created)
    assert runs[-1][1] < runs[1][1]
    assert "RECOVERY:" in created[1].messages[0]
    assert "TASK EXECUTION INCOMPLETE" in created[1].messages[0]


@pytest.mark.parametrize("native_tool_calling", [True, False])
def test_initial_task_states_actual_interface_despite_old_advice(
        monkeypatch, native_tool_calling):
    messages, runs = [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def send_message(self, text):
            messages.append(text)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append(timeout)
        return False, False
    monkeypatch.setattr(backend, "_run_convo", run)
    task = "Keep the accepted checkpoint. Prior advice may describe a different tool interface."
    backend.run_agent_task(ob.SandboxHandle(workspace="WS"), task,
                           llm=SimpleNamespace(native_tool_calling=native_tool_calling),
                           timeout=1200)
    assert runs == [1200]
    assert len(messages) == 1
    assert messages[0].startswith(task)
    if native_tool_calling:
        assert "native_tool_calling=true" in messages[0]
        assert "structured tool_calls" in messages[0]
        assert "not executable" in messages[0]
        assert "prior advice" in messages[0]
        assert "budgets" in messages[0]
    else:
        assert "native_tool_calling=false" in messages[0]
        assert "<function=NAME>" in messages[0]
        assert "<parameter=NAME>" in messages[0]
        assert "prior advice" in messages[0]
        assert "budgets" in messages[0]


@pytest.mark.parametrize("tool_progress", [False, True])
def test_protocol_recovery_preserves_real_tool_progress_and_result_handoff(
        monkeypatch, tool_progress):
    messages, runs, checks = [], [], []
    clock = [100.0]

    def malformed():
        return SimpleNamespace(kind="MessageEvent", source="agent",
            llm_message=SimpleNamespace(role="assistant", content=[SimpleNamespace(
                text="<tool_call>terminal<arg_key>command")]))

    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = []

        def send_message(self, text):
            messages.append(text)

    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    backend = ob.OpenHandsBackend()
    predicate = lambda: False

    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((timeout, stop_predicate))
        if len(runs) == 1:
            clock[0] = 140.0
            convo.state.events.append(malformed())
        elif len(runs) == 2:
            # The protocol correction restored actual execution, not just prose.
            if tool_progress:
                assert timeout >= 360, "recovered execution keeps the original deadline"
                clock[0] = 500.0
                convo.state.events.append(SimpleNamespace(kind="ObservationEvent",
                    observation=SimpleNamespace(is_error=False)))
            convo.state.events.append(malformed())
        else:
            clock[0] = 520.0
            convo.state.events.append(SimpleNamespace(kind="ObservationEvent",
                observation=SimpleNamespace(is_error=False)))
        return False, False

    monkeypatch.setattr(backend, "_run_convo", run)
    backend.run_agent_task(ob.SandboxHandle(workspace="WS"), "task",
        llm=SimpleNamespace(native_tool_calling=False), timeout=1200,
        stop_predicate=predicate, completion_check=lambda: checks.append(True))
    assert len(runs) == 3
    assert all(p is predicate for _, p in runs)
    assert checks == [True]
    if tool_progress:
        assert runs[-1][0] == 800.0


@pytest.mark.parametrize("tool_progress", [False, True])
@pytest.mark.parametrize("recovered", [False, True])
def test_result_repair_can_recover_its_first_tool_protocol_failure(monkeypatch, tool_progress, recovered):
    messages, runs = [], []
    clock = [100.0]

    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = []

        def send_message(self, text):
            messages.append(text)

    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    backend = ob.OpenHandsBackend()
    predicate = lambda: False

    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        clock[0] += 10
        if len(runs) == 2:
            if tool_progress:
                convo.state.events.append(SimpleNamespace(kind="ObservationEvent",
                    observation=SimpleNamespace(is_error=False)))
            convo.state.events.append(SimpleNamespace(kind="MessageEvent", source="agent",
                llm_message=SimpleNamespace(role="assistant", content=[SimpleNamespace(
                    text="<tool_call>terminal<arg_key>command")])))
        elif len(runs) == 3 and recovered:
            convo.state.events.append(SimpleNamespace(kind="ObservationEvent",
                observation=SimpleNamespace(is_error=False)))
        return False, False

    monkeypatch.setattr(backend, "_run_convo", run)
    backend.run_agent_task(ob.SandboxHandle(workspace="WS"), "task",
        llm=SimpleNamespace(native_tool_calling=False), timeout=60,
        stop_predicate=predicate,
        completion_check=lambda: "TASK EXECUTION INCOMPLETE: save result" if len(runs) < 3 else None)
    assert len(runs) == (3 if recovered else 5)
    assert all(convo is runs[0][0] for convo, _, _ in runs)
    assert runs[-1][1] == (40 if recovered else 20)
    assert runs[-1][2] is predicate
    assert "<function=terminal>" in messages[-1]


@pytest.mark.parametrize("tool_name", ["terminal", "task_tracker", "future_tool"])
def test_emulated_tool_repair_uses_sdk_function_format(tool_name):
    event = SimpleNamespace(kind="MessageEvent", source="agent", llm_message=SimpleNamespace(
        role="assistant", content=[SimpleNamespace(
            text=f"Ready.<tool_call>function={tool_name}><parameter=command>pwd</parameter>")]))
    message = ob._tool_markup_repair_message(
        SimpleNamespace(state=SimpleNamespace(events=[event])), native_tool_calling=False)
    assert f"<function={tool_name}>" in message
    assert "</function>" in message
    assert '<parameter=command> </parameter>' in message
    assert '<parameter=is_input>' not in message
    assert "already-running" in message
    assert "structured JSON" not in message
    assert "pwd" not in message  # Never rewrite or execute the model's proposed command.


@pytest.mark.parametrize("finished,needs_result,remaining", [
    (True, True, 30.0), (True, False, 30.0),
    (False, True, 30.0), (True, True, 0.0),
])
def test_completion_repair_reuses_conversation_once_within_deadline(
        monkeypatch, finished, needs_result, remaining):
    messages, runs, checks, closed = [], [], [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.execution_status.name = "FINISHED" if finished else "ERROR"
        def send_message(self, text):
            messages.append(text)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    clock = [100.0]
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout))
        clock[0] = 160.0 - remaining
        return False, False
    monkeypatch.setattr(backend, "_run_convo", run)
    def completion_check():
        checks.append(True)
        return "Write only the missing result from existing evidence." if needs_result else None
    result = backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "original experiment", llm=object(),
        timeout=60, completion_check=completion_check)
    repaired = finished and needs_result and remaining > 0
    assert len(runs) == (2 if repaired else 1)
    if repaired:
        assert runs[0][0] is runs[1][0]
        assert 0 < runs[1][1] <= remaining
        assert messages[-1].startswith("Write only")
        assert checks == [True, True]  # The same error must not trigger another run.
    assert closed == [True]
    assert result.status == ("FINISHED" if finished else "ERROR")


@pytest.mark.parametrize("finishes", [True, False])
def test_background_attempt_wait_preserves_original_deadline(monkeypatch, finishes):
    messages, runs, closed = [], [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def send_message(self, text):
            messages.append(text)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    clock = [100.0]
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ob.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    backend = ob.OpenHandsBackend()
    def run(convo, timeout, *args, **kwargs):
        runs.append(timeout)
        clock[0] += 1
        return False, False
    monkeypatch.setattr(backend, "_run_convo", run)
    def check():
        if clock[0] < 103 or not finishes:
            assert not closed
            return "TASK EXECUTION RUNNING: registered original command"
        return "Serialize existing measurement."
    result = backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "task", llm=object(),
        timeout=5, completion_check=check)
    assert all(not message.startswith("TASK EXECUTION RUNNING:") for message in messages)
    assert runs == ([5, 2] if finishes else [5])
    assert result.status == ("FINISHED" if finishes else "TIMEOUT")
    assert closed == [True]


@pytest.mark.parametrize("repair_elapsed,second_error,expected_runs", [
    (40, "missing evaluator JSON", 3),
    (40, "missing attempt ID", 2),
    (40, None, 2),
    (300, "missing evaluator JSON", 3),
    (990, "missing evaluator JSON", 2),
])
def test_result_repair_rechecks_progress_without_extending_shared_budget(
        monkeypatch, repair_elapsed, second_error, expected_runs):
    messages, runs = [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def send_message(self, text):
            messages.append(text)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    clock = [100.0]
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout))
        clock[0] += 10 if len(runs) == 1 else repair_elapsed
        return False, False
    monkeypatch.setattr(backend, "_run_convo", run)
    errors = iter(["missing attempt ID", second_error])
    backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "original experiment", llm=object(),
        timeout=1000, completion_check=lambda: next(errors))
    assert len(runs) == expected_runs
    assert runs[1][1] == 990
    if expected_runs == 3:
        assert runs[2][0] is runs[1][0] is runs[0][0]
        assert runs[2][1] == 990 - repair_elapsed  # Original deadline, never a fresh budget.
        assert messages[-1] == "missing evaluator JSON"


@pytest.mark.parametrize("stopped", [False, True])
def test_unstarted_task_resume_keeps_attempt_stop_and_original_deadline(monkeypatch, stopped):
    runs, messages = [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def send_message(self, text):
            messages.append(text)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    clock = [100.0]
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    predicate = lambda: False
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        clock[0] += 10
        return False, stopped if len(runs) == 2 else False
    monkeypatch.setattr(backend, "_run_convo", run)
    result = backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "original task", llm=object(),
        timeout=1000, stop_predicate=predicate,
        completion_check=lambda: "TASK EXECUTION INCOMPLETE: continue original task.")
    assert len(runs) == 2  # No third retry when the same missing-execution condition persists.
    assert runs[1][0] is runs[0][0]
    assert runs[1][1] == 990
    assert runs[1][2] is predicate
    assert result.status == ("PAUSED_BY_PREDICATE" if stopped else "FINISHED")


@pytest.mark.parametrize("exit_code,is_error,hard_timeout", [
    (-1, False, False), (0, False, False), (124, False, True),
    (-1, True, False), (-1, False, True),
])
def test_finished_agent_waits_for_original_soft_timed_out_command(
        monkeypatch, exit_code, is_error, hard_timeout):
    runs, messages = [], []
    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = [SimpleNamespace(observation=SimpleNamespace(
                kind="TerminalObservation", exit_code=exit_code,
                is_error=is_error, timeout=hard_timeout))]
        def send_message(self, text):
            messages.append(text)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    clock = [100.0]
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        clock[0] = 200.0
        return False, False
    monkeypatch.setattr(backend, "_run_convo", run)
    result = backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "task", llm=object(), timeout=1000,
        stop_predicate=lambda: True, completion_check=lambda: "Serialize results only.")
    pending = exit_code == -1 and not is_error and not hard_timeout
    assert len(runs) == 2  # Same bounded slot, never an unbounded FINISHED retry.
    assert runs[0][0] is runs[1][0]
    assert runs[1][1] == 900
    assert runs[1][2] is None
    if pending:
        assert "empty command" in messages[-1]
        assert "Do not restart" in messages[-1]
    else:
        assert messages[-1] == "Serialize results only."
    assert result.status == "FINISHED"


@pytest.mark.parametrize("suffix,hard_timeout,expected", [
    ('Your command is NOT executed. The previous command is still running', False, True),
    ('Your command is NOT executed. Permission denied', False, False),
    ('Your command is NOT executed. The previous command is still running', True, False),
])
@pytest.mark.parametrize("metadata_factory", [dict, SimpleNamespace])
def test_busy_terminal_rejection_preserves_pending_original_command(
        suffix, hard_timeout, expected, metadata_factory):
    event = SimpleNamespace(observation=SimpleNamespace(
        kind="TerminalObservation", exit_code=-1, is_error=True,
        timeout=hard_timeout, metadata=metadata_factory(suffix=suffix)))
    convo = SimpleNamespace(state=SimpleNamespace(events=[event]))
    assert (ob._terminal_wait_event(convo) is event) is expected


@pytest.mark.parametrize("busy_rejection", [False, True])
def test_pending_wait_preserves_malformed_tool_correction(monkeypatch, busy_rejection):
    messages, runs = [], []

    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = [
                SimpleNamespace(observation=SimpleNamespace(
                    kind="TerminalObservation", exit_code=-1,
                    is_error=busy_rejection, timeout=False,
                    metadata={"suffix": (
                        "Your command is NOT executed. The previous command is still running"
                        if busy_rejection else "")})),
                SimpleNamespace(
                    kind="MessageEvent", source="agent",
                    llm_message=SimpleNamespace(role="assistant", content=[
                        SimpleNamespace(text="<tool_call>terminal<arg_key>command")]))]

        def send_message(self, text):
            messages.append(text)

    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    backend = ob.OpenHandsBackend()
    monkeypatch.setattr(ob.time, "monotonic", lambda: 100.0)

    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        return False, False

    monkeypatch.setattr(backend, "_run_convo", run)
    backend.run_agent_task(ob.SandboxHandle(workspace="WS"), "task",
                           llm=object(), timeout=1000)

    assert len(runs) == 2  # Unchanged wait evidence cannot authorize another run.
    assert runs[0][0] is runs[1][0]
    assert runs[1][1] == 1000  # Preserve the original task budget.
    assert "structured tool interface" in messages[-1]
    assert "did not execute a tool" in messages[-1]
    assert "is_input=false" not in messages[-1]
    assert "single space character" in messages[-1]
    assert "reset=false" in messages[-1]
    assert "Do not restart" in messages[-1]


def test_later_terminal_completion_supersedes_old_soft_wait():
    def event(code):
        return SimpleNamespace(observation=SimpleNamespace(
            kind="TerminalObservation", exit_code=code, is_error=False, timeout=False))
    convo = SimpleNamespace(state=SimpleNamespace(events=[event(-1), event(0)]))
    assert not ob._terminal_wait_pending(convo)
    convo.state.events = [event(-1), SimpleNamespace(action=SimpleNamespace(kind="TerminalAction"))]
    assert not ob._terminal_wait_pending(convo)  # Unobserved later action is not known live.


@pytest.mark.parametrize("finish_command", [True, False])
def test_fresh_soft_wait_continues_same_conversation_with_original_budget(
        monkeypatch, finish_command):
    messages, runs = [], []
    clock = [100.0]
    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = []
        def send_message(self, text):
            messages.append(text)
    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    backend = ob.OpenHandsBackend()
    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        clock[0] += 10
        code = 0 if finish_command and len(runs) == 3 else -1
        convo.state.events.append(SimpleNamespace(
            id=f"wait-{len(runs)}", observation=SimpleNamespace(
                kind="TerminalObservation", exit_code=code, is_error=False, timeout=False)))
        return False, False
    monkeypatch.setattr(backend, "_run_convo", run)
    result = backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "original experiment", llm=object(), timeout=30)
    assert len(runs) == 3
    assert all(row[0] is runs[0][0] for row in runs)
    assert [row[1] for row in runs] == [30, 20, 10]
    assert all(row[2] is None for row in runs[1:])
    assert "empty command" in messages[-1]
    assert result.status == ("FINISHED" if finish_command else "TIMEOUT")


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("native_tool_calling", [False, True])
@pytest.mark.parametrize("busy_rejection", [False, True])
def test_pending_command_protocol_repair_uses_existing_bounded_attempts(
        monkeypatch, recovered, native_tool_calling, busy_rejection):
    created, runs, messages = [], [], []
    clock = [100.0]
    completed = [False]

    class Convo(_fake_convo_cls(lambda: None)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state.events = []
            created.append(self)

        def send_message(self, text):
            messages.append(text)

    def run(convo, timeout, stop_predicate=None, **kwargs):
        runs.append((convo, timeout, stop_predicate))
        clock[0] += 10
        if len(runs) == 1:
            convo.state.events.append(SimpleNamespace(
                id="original-wait", kind="ObservationEvent",
                observation=SimpleNamespace(
                    kind="TerminalObservation", exit_code=-1, timeout=False,
                    is_error=busy_rejection, metadata={"suffix": (
                        "Your command is NOT executed. The previous command is still running"
                        if busy_rejection else "")})))
        if recovered and len(runs) == 3:
            completed[0] = True
            convo.state.events.append(SimpleNamespace(
                id="original-command-finished", kind="ObservationEvent",
                observation=SimpleNamespace(
                    kind="TerminalObservation", exit_code=0,
                    is_error=False, timeout=False)))
        else:
            convo.state.events.append(SimpleNamespace(
                kind="MessageEvent", source="agent",
                llm_message=SimpleNamespace(role="assistant", content=[
                    SimpleNamespace(text="<tool_call>terminal<arg_key>command")])))
        return False, False

    monkeypatch.setattr(openhands.sdk, "Conversation", Convo)
    monkeypatch.setattr(ob, "_build_agent", lambda *args: object())
    monkeypatch.setattr(ob.time, "monotonic", lambda: clock[0])
    backend = ob.OpenHandsBackend()
    monkeypatch.setattr(backend, "_run_convo", run)
    result = backend.run_agent_task(
        ob.SandboxHandle(workspace="WS"), "original task",
        llm=SimpleNamespace(native_tool_calling=native_tool_calling), timeout=70,
        completion_check=lambda: None if completed[0] else
        "TASK EXECUTION INCOMPLETE: wait for the original command and save its result.")

    assert len(runs) == (3 if recovered else 4)
    assert len(created) == 1  # Never replace the conversation of a live command.
    assert [r[1] for r in runs] == ([70, 60, 50] if recovered else [70, 60, 50, 40])
    assert all(r[2] is None for r in runs[1:])
    assert all("Do not restart" in message for message in messages[1:])
    assert completed[0] is recovered
    assert result.status == "FINISHED"  # Conversation status still does not certify a result.
