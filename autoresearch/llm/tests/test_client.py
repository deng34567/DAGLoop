"""
Unit tests for HKGAIClient / HKGAIClient 单元测试（无网络，mock）。
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from autoresearch.llm import HKGAIClient, LLMResult
from autoresearch.llm import client as client_mod
from autoresearch.llm import pricing

MONEY_RE = re.compile(r"^-?[0-9]+(\.[0-9]{1,6})?$")


class _FakeCompletions:
    """Records create() kwargs and returns a canned response. / 记录入参并回固定响应。"""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return self._response


class _FakeOpenAI:
    """Minimal stand-in for openai.OpenAI. / openai.OpenAI 的最小替身。"""

    def __init__(self, response: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _canned_response() -> Any:
    """A response with content, one tool_call, and usage. / 含内容、tool_call、usage 的响应。"""
    tool_call = SimpleNamespace(
        id="call_abc",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "HK"}'),
    )
    message = SimpleNamespace(content="hello", tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=11, completion_tokens=7, total_tokens=18
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _FlakyCompletions:
    """Raises a transient error the first ``fail_times`` calls, then returns ``response``."""

    def __init__(self, response: Any, fail_times: int, exc: BaseException) -> None:
        self._response = response
        self._fail_times = fail_times
        self._exc = exc
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return self._response


def _flaky_client(fail_times: int, exc: BaseException) -> tuple[HKGAIClient, _FlakyCompletions]:
    fake = _FakeOpenAI(_canned_response())
    flaky = _FlakyCompletions(_canned_response(), fail_times, exc)
    fake.chat = SimpleNamespace(completions=flaky)
    c = HKGAIClient(api_key="k", base_url="https://api.hkgai.net/v1", model="DS3.2",
                    openai_client=fake, stream=False)   # non-streaming path (老师: 别改流式)
    return c, flaky


def test_nonstream_retries_transient_then_succeeds(monkeypatch) -> None:
    # 老师 2026-08-05: a lone transient on a non-streaming call must NOT sink the run -- retry.
    monkeypatch.setattr(client_mod, "_STREAM_RETRY_BASE_S", 0.0)   # no real backoff sleep in test
    monkeypatch.setattr(client_mod, "_STREAM_RETRIES", 3)
    c, flaky = _flaky_client(fail_times=2, exc=httpx.TimeoutException("gateway slow"))
    res = c.chat(messages=[{"role": "user", "content": "hi"}])
    assert res.content == "hello"
    assert flaky.calls == 3            # 2 transient failures + 1 success


def test_nonstream_reraises_after_exhausting_retries(monkeypatch) -> None:
    monkeypatch.setattr(client_mod, "_STREAM_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(client_mod, "_STREAM_RETRIES", 2)
    c, flaky = _flaky_client(fail_times=99, exc=httpx.TimeoutException("dead endpoint"))
    with pytest.raises(httpx.TimeoutException):
        c.chat(messages=[{"role": "user", "content": "hi"}])
    assert flaky.calls == 2            # fails fast after exhausting the bounded attempts


def test_nonstream_non_transient_not_retried(monkeypatch) -> None:
    # a non-transient error (e.g. bad request) should surface immediately, not retry.
    monkeypatch.setattr(client_mod, "_STREAM_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(client_mod, "_STREAM_RETRIES", 3)
    c, flaky = _flaky_client(fail_times=99, exc=ValueError("bad kwargs"))
    with pytest.raises(ValueError):
        c.chat(messages=[{"role": "user", "content": "hi"}])
    assert flaky.calls == 1            # not retried


def _make_client(response: Any, response_format_mode: str | None = None) -> tuple[HKGAIClient, _FakeOpenAI]:
    fake = _FakeOpenAI(response)
    c = HKGAIClient(
        api_key="k",
        base_url="https://api.hkgai.net/v1",
        model="DS3.2",
        cookie="HWWAFSESID=xyz",
        openai_client=fake,
        response_format_mode=response_format_mode,
    )
    return c, fake


def test_parses_content_tool_calls_usage_and_cost() -> None:
    c, _ = _make_client(_canned_response())
    result = c.chat(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResult)
    assert result.content == "hello"

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["id"] == "call_abc"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == '{"city": "HK"}'

    assert result.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }

    # Self-deployed model has no real $ price -> nominal token rate (1 per 1M):
    # 11/1_000_000 + 7/1_000_000 = 0.000018. Must still be a valid money string.
    assert result.cost_delta == "0.000018"
    assert MONEY_RE.match(result.cost_delta)


def test_cost_delta_matches_money_regex_with_nonzero_rate(monkeypatch) -> None:
    # Plug a rate to prove the structure supports per-1M pricing later.
    monkeypatch.setitem(pricing.RATES, "DS3.2", pricing.Rate("0.5", "1.5"))
    c, _ = _make_client(_canned_response())
    result = c.chat(messages=[{"role": "user", "content": "hi"}])
    # 11/1_000_000*0.5 + 7/1_000_000*1.5 = 0.0000055 + 0.0000105 = 0.000016
    assert result.cost_delta == "0.000016"
    assert MONEY_RE.match(result.cost_delta)


def test_response_format_json_object_kwarg() -> None:
    # json_object 现为 opt-in:实例 mode 显式开启才发(env OPENAI_RESPONSE_FORMAT_MODE=json_object
    # 或构造传 response_format_mode)。
    c, fake = _make_client(_canned_response(), response_format_mode="json_object")
    c.chat(
        messages=[{"role": "user", "content": "hi"}],
        response_format="json_object",
    )
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs is not None
    assert kwargs["response_format"] == {"type": "json_object"}


def test_json_object_suppressed_when_mode_off() -> None:
    # mode 未开(默认)→ 即便 caller 传 response_format="json_object" 也不发,
    # 给不支持 json_object 的端点用(如第三方 external-api-platform 的 deepseek)。
    c, fake = _make_client(_canned_response())  # response_format_mode=None
    c.chat(
        messages=[{"role": "user", "content": "hi"}],
        response_format="json_object",
    )
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs is not None
    assert "response_format" not in kwargs


def test_required_json_object_is_sent_even_when_optional_mode_is_off() -> None:
    c, fake = _make_client(_canned_response())

    c.chat(
        messages=[{"role": "user", "content": "hi"}],
        response_format="json_object_required",
    )

    kwargs = fake.chat.completions.last_kwargs
    assert kwargs is not None
    assert kwargs["response_format"] == {"type": "json_object"}


def test_no_response_format_sends_no_kwarg() -> None:
    c, fake = _make_client(_canned_response())
    c.chat(messages=[{"role": "user", "content": "hi"}])
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs is not None
    assert "response_format" not in kwargs


def test_tools_and_max_tokens_forwarded() -> None:
    c, fake = _make_client(_canned_response())
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    c.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        max_tokens=128,
    )
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs["tools"] == tools
    assert kwargs["max_tokens"] == 128
    assert kwargs["model"] == "DS3.2"


def test_client_default_max_tokens_applies_only_when_call_omits_limit() -> None:
    fake = _FakeOpenAI(_canned_response())
    client = HKGAIClient(
        api_key="k",
        base_url="https://api.hkgai.net/v1",
        model="DS3.2",
        openai_client=fake,
        stream=False,
        default_max_tokens=2048,
    )

    client.chat(messages=[{"role": "user", "content": "bounded"}])
    assert fake.chat.completions.last_kwargs["max_tokens"] == 2048

    client.chat(messages=[{"role": "user", "content": "explicit"}], max_tokens=128)
    assert fake.chat.completions.last_kwargs["max_tokens"] == 128


def test_no_tool_calls_yields_none() -> None:
    message = SimpleNamespace(content="plain", tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    resp = SimpleNamespace(choices=[choice], usage=usage)
    c, _ = _make_client(resp)
    result = c.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.tool_calls is None
    assert result.content == "plain"


def test_missing_usage_defaults_to_zero() -> None:
    message = SimpleNamespace(content="x", tool_calls=None)
    choice = SimpleNamespace(message=message)
    resp = SimpleNamespace(choices=[choice], usage=None)
    c, _ = _make_client(resp)
    result = c.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_http_client_built_with_trust_env_false(monkeypatch) -> None:
    """
    Proxy bypass: the OpenAI client must be built with an httpx client
    constructed with trust_env=False. / 必须用 trust_env=False 绕过 clash。
    """
    captured: dict[str, Any] = {}

    real_httpx_client = httpx.Client

    def _spy_client(*args: Any, **kwargs: Any) -> Any:
        captured["trust_env"] = kwargs.get("trust_env")
        captured["timeout"] = kwargs.get("timeout")
        return real_httpx_client(*args, **kwargs)

    captured_openai: dict[str, Any] = {}

    def _spy_openai(**kwargs: Any) -> Any:
        captured_openai.update(kwargs)
        return _FakeOpenAI(_canned_response())

    monkeypatch.setattr(client_mod.httpx, "Client", _spy_client)
    monkeypatch.setattr(client_mod, "OpenAI", _spy_openai)

    HKGAIClient(
        api_key="k",
        base_url="https://api.hkgai.net/v1",
        model="DS3.2",
        cookie="HWWAFSESID=xyz",
    )

    # The proxy-bypassing http client was created with trust_env=False.
    assert captured["trust_env"] is False
    # And handed to the OpenAI client.
    assert "http_client" in captured_openai
    # Cookie header is passed through.
    assert captured_openai["default_headers"]["Cookie"] == "HWWAFSESID=xyz"
    assert captured_openai["base_url"] == "https://api.hkgai.net/v1"


def test_explicit_timeout_overrides_default(monkeypatch) -> None:
    """D (#5): an explicit timeout= builds the httpx client with that timeout
    (so the ScenarioAgent's small chunked calls fail fast)."""
    captured: dict[str, Any] = {}
    real_httpx_client = httpx.Client

    def _spy_client(*args: Any, **kwargs: Any) -> Any:
        captured["timeout"] = kwargs.get("timeout")
        return real_httpx_client(*args, **kwargs)

    monkeypatch.setattr(client_mod.httpx, "Client", _spy_client)
    monkeypatch.setattr(client_mod, "OpenAI", lambda **kw: _FakeOpenAI(_canned_response()))

    HKGAIClient(api_key="k", base_url="https://api.hkgai.net/v1", model="DS3.2", timeout=180.0)
    assert captured["timeout"] == 180.0


def test_pricing_unknown_model_uses_nominal_token_rate() -> None:
    # Unknown / self-deployed model -> nominal rate (1 per 1M tokens):
    # 150/1_000_000 = 0.00015 (cost == total_tokens/1_000_000, the token budget unit).
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    assert pricing.cost_for(usage, "unknown-model") == "0.00015"


# --------------------------------------------------------------------
# selective reasoning — direct client (动脑型 agent) knob (#16 spec)
# --------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["high", "max"])
def test_reasoning_effort_sent_when_set(monkeypatch, effort) -> None:  # noqa: ANN001
    """Direct client (动脑型 agent: requirement/architect/scenario/coder分析/moss/doc):
    VIBE_REASONING_EFFORT=high → chat() sends reasoning_effort=high. Decoupled from the
    sandbox path (VIBE_SANDBOX_REASONING_EFFORT)."""
    monkeypatch.setenv("VIBE_REASONING_EFFORT", effort)
    c, fake = _make_client(_canned_response())
    c.chat(messages=[{"role": "user", "content": "hi"}])
    assert fake.chat.completions.last_kwargs.get("reasoning_effort") == effort


def test_reasoning_effort_not_sent_when_empty(monkeypatch) -> None:  # noqa: ANN001
    """Empty/unset → not sent (env-gated: ``"" or None`` → None). Safe for endpoints that
    don't accept the param (deepseek)."""
    monkeypatch.setenv("VIBE_REASONING_EFFORT", "")
    c, fake = _make_client(_canned_response())
    c.chat(messages=[{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in fake.chat.completions.last_kwargs


def test_provider_default_does_not_enable_glm_reasoning_override(monkeypatch) -> None:
    monkeypatch.setenv("VIBE_REASONING_EFFORT", "default")
    fake = _FakeOpenAI(_canned_response())
    c = HKGAIClient(api_key="k", base_url="http://offline.invalid/v1", model="glm-5.2",
                    openai_client=fake, stream=False)
    c.chat(messages=[{"role": "user", "content": "hi"}])
    assert c.reasoning_effort is None
    assert "reasoning_effort" not in fake.chat.completions.last_kwargs


@pytest.mark.parametrize("temperature", [None, "0.7", "0.0"])
@pytest.mark.parametrize("stream", [False, True])
def test_configured_temperature_reaches_both_call_modes(monkeypatch, temperature, stream):
    monkeypatch.setattr(client_mod, "_load_env", lambda: None)
    if temperature is None:
        monkeypatch.delenv("AUTORESEARCH_LLM_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_LLM_TEMPERATURE", temperature)
    response = (_FakeStream([_stream_chunk("ok", finish_reason="stop")])
                if stream else _canned_response())
    fake = _FakeOpenAI(response)
    c = HKGAIClient(model="glm-5.2", openai_client=fake, stream=stream)
    c.chat(messages=[{"role": "user", "content": "offline"}])
    kwargs = fake.chat.completions.last_kwargs
    if temperature is None:
        assert "temperature" not in kwargs
    else:
        assert kwargs["temperature"] == float(temperature)
    assert "top_p" not in kwargs
    assert bool(kwargs.get("stream")) is stream


def test_strip_think_cleans_reasoning_leak():
    # deepseek-v4-flash & other reasoning models occasionally leak <think>...</think> (or a lone
    # tag) into message.content, which would poison JSON extraction / report text. _strip_think
    # removes it; clean content is untouched.
    from autoresearch.llm.client import _strip_think
    assert _strip_think("OK</think>OK") == "OKOK"
    assert _strip_think("<think>chain of thought</think>{\"a\": 1}") == '{"a": 1}'
    assert _strip_think("plain answer, no tags") == "plain answer, no tags"
    assert _strip_think(None) is None
    assert _strip_think("<think>only reasoning</think>") is None


# --- streaming wall-clock deadline ---------------------------------------------------------------

def _stream_chunk(content=None, usage=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.parametrize("progress_kind", ["reasoning_content", "tool_arguments"])
def test_streaming_generation_before_final_content_is_not_a_stall(monkeypatch, progress_kind):
    import time

    monkeypatch.setattr(client_mod, "_STREAM_MAX_S", 10.0)
    monkeypatch.setattr(client_mod, "_STREAM_NO_CONTENT_S", 0.5)

    def chunks():
        for _ in range(25):
            time.sleep(0.1)
            chunk = _stream_chunk()
            delta = chunk.choices[0].delta
            if progress_kind == "reasoning_content":
                delta.reasoning_content = "thinking"
            else:
                delta.tool_calls = [SimpleNamespace(
                    index=0, id=None, function=SimpleNamespace(name=None, arguments=" "))]
            yield chunk
        yield _stream_chunk("answer", finish_reason="stop")

    c, _ = _make_client(_FakeStream(chunks()))
    result = c._chat_streaming({"model": c.model, "messages": [{"role": "user", "content": "x"}]})
    assert result.content == "answer"
    assert "thinking" not in result.content


class _FakeStream:
    """A streaming response stand-in: iterable of chunks, records close(). / 流式响应替身。"""
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


def test_streaming_deadline_aborts_and_closes(monkeypatch):
    # a gateway that trickles tokens without finishing must not hang forever: once the wall-clock
    # deadline passes, the stream is closed and a TimeoutError is raised (caller treats it as a
    # normal failed call). Deadline in the past -> fires on the first chunk, deterministic.
    monkeypatch.setattr(client_mod, "_STREAM_MAX_S", -1.0)
    stream = _FakeStream([_stream_chunk("hi"), _stream_chunk(" there")])
    c, _ = _make_client(stream)
    with pytest.raises(TimeoutError):
        c.chat(messages=[{"role": "user", "content": "x"}], stream=True)
    assert stream.closed   # the stalled stream was actively closed


def test_streaming_watchdog_aborts_a_stalled_first_chunk(monkeypatch):
    # THE live hang: the gateway holds the connection open sending only SSE keep-alives, so the
    # stream NEVER yields a chunk -- ``__next__`` blocks forever and the in-loop deadline check never
    # runs. The watchdog timer must close the stream at the wall-clock deadline and surface a clean
    # TimeoutError, promptly (not after the socket's own long read timeout).
    import threading as _t
    import time as _time
    monkeypatch.setattr(client_mod, "_STREAM_MAX_S", 0.3)   # short deadline so the test is fast

    class _BlockingStream:
        def __init__(self):
            self.closed = False
            self._gate = _t.Event()

        def __iter__(self):
            return self

        def __next__(self):
            self._gate.wait(timeout=5)      # blocks until the watchdog close()s us (keep-alive stall)
            raise RuntimeError("stream closed by watchdog")

        def close(self):
            self.closed = True
            self._gate.set()

    stream = _BlockingStream()
    c, _ = _make_client(stream)
    t0 = _time.monotonic()
    # test the watchdog at its own level (_chat_streaming); chat()'s stall-retry wrapper is covered
    # separately in test_chat_retries_once_on_stream_stall.
    with pytest.raises(TimeoutError):
        c._chat_streaming({"model": c.model, "messages": [{"role": "user", "content": "x"}]})
    assert stream.closed                       # the watchdog actively closed the stalled stream
    assert _time.monotonic() - t0 < 3          # aborted promptly at the deadline, not hung on read


def test_chat_retries_once_on_stream_stall(monkeypatch):
    # a watchdog-aborted stream (keep-alive stall) is TRANSIENT: chat() must retry _chat_streaming
    # ONCE so a single stall doesn't crash a whole pipeline (seen live: one stalled glm call in
    # repo_improve's discover phase failed the whole run). The retry then succeeds.
    calls = {"n": 0}
    good = LLMResult(content="ok", tool_calls=None, usage={}, cost_delta=0.0)

    def _fake_stream(_kwargs, _stream_max_s=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("streaming stalled past 0s with no content")
        return good

    c, _ = _make_client(_FakeStream([]))
    monkeypatch.setattr(c, "_chat_streaming", _fake_stream)
    r = c.chat(messages=[{"role": "user", "content": "x"}], stream=True)
    assert calls["n"] == 2 and r.content == "ok"


def test_streaming_normal_completes_under_deadline():
    # a normal (fast) stream accumulates content + usage and returns without tripping the deadline.
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    stream = _FakeStream([_stream_chunk("hel"), _stream_chunk("lo", finish_reason="stop"),
                          _stream_chunk(None, usage=usage)])
    c, _ = _make_client(stream)
    r = c.chat(messages=[{"role": "user", "content": "x"}], stream=True)
    assert r.content == "hello"
    assert r.usage["total_tokens"] == 5
    assert not stream.closed   # completed naturally, no forced close


def test_stream_truncated_no_finish_reason_raises():
    # Truncation guard (borrowed from vibe ★2026-07-22): a stream that yields content but ENDS
    # without a finish_reason = the gateway cut the SSE mid-generation. Must raise (a transient
    # transport failure) so the caller retries on a fresh connection, NOT return the half-answer
    # that downstream json.loads chokes on and marks permanent.
    stream = _FakeStream([_stream_chunk("half"), _stream_chunk(" answer")])   # no finish_reason
    c, _ = _make_client(stream)
    with pytest.raises(TimeoutError):
        c._chat_streaming({"model": c.model, "messages": [{"role": "user", "content": "x"}]})


def test_stream_retry_uses_exponential_backoff(monkeypatch):
    # borrowed from vibe's backoff-retry: retries WAIT (growing backoff) before re-hitting a stuck
    # gateway instead of immediately re-hammering it (immediate retry under load re-hits the same
    # wall). Verify the backoff grows and the call still succeeds once the endpoint recovers.
    monkeypatch.setattr(client_mod, "_STREAM_RETRIES", 3)
    monkeypatch.setattr(client_mod, "_STREAM_RETRY_BASE_S", 2.0)
    monkeypatch.setattr(client_mod, "_STREAM_RETRY_CAP_S", 30.0)
    sleeps: list[float] = []
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    good = LLMResult(content="ok", tool_calls=None, usage={}, cost_delta=0.0)

    def _fake_stream(_kwargs, _stream_max_s=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("stall")
        return good

    c, _ = _make_client(_FakeStream([]))
    monkeypatch.setattr(c, "_chat_streaming", _fake_stream)
    r = c.chat(messages=[{"role": "user", "content": "x"}], stream=True)
    assert calls["n"] == 3 and r.content == "ok"
    assert sleeps == [2.0, 4.0]   # base*2**0, base*2**1 -- two backoffs before the 3rd succeeds


def test_stream_abandons_on_no_content_stall(monkeypatch):
    # THE 30-min freeze: a connection wedged in SSL recv that close() from another thread does NOT
    # unblock. The no-content detector must ABANDON the worker thread and raise promptly instead of
    # hanging until the gateway acts. _chat_streaming (not chat) so we test the wrapper directly.
    import threading as _t
    import time as _time
    monkeypatch.setattr(client_mod, "_STREAM_MAX_S", 30.0)
    monkeypatch.setattr(client_mod, "_STREAM_NO_CONTENT_S", 0.5)

    class _WedgedStream:
        def __init__(self):
            self.closed = False
            self._never = _t.Event()

        def __iter__(self):
            return self

        def __next__(self):
            self._never.wait()          # blocks FOREVER -- close() below does NOT set it
            raise StopIteration

        def close(self):
            self.closed = True          # does NOT unblock __next__ (models the un-closeable recv)

    stream = _WedgedStream()
    c, _ = _make_client(stream)
    t0 = _time.monotonic()
    with pytest.raises(TimeoutError):
        c._chat_streaming({"model": c.model, "messages": [{"role": "user", "content": "x"}]})
    assert _time.monotonic() - t0 < 5      # abandoned promptly (~no_content+poll), not after 30s
    try:
        deadline = _time.monotonic() + 0.5
        while not stream.closed and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert stream.closed, 'Abandoned requests must be closed before a fresh retry.'
    finally:
        stream._never.set()


def test_stream_created_after_abandonment_is_closed(monkeypatch):
    import threading as _t
    monkeypatch.setattr(client_mod, '_STREAM_MAX_S', 30.0)
    monkeypatch.setattr(client_mod, '_STREAM_NO_CONTENT_S', 0.5)
    allow_create = _t.Event()
    closed = _t.Event()

    class LateStream:
        def __iter__(self):
            closed.wait(5)
            return iter(())

        def close(self):
            closed.set()

    c, _ = _make_client(None)

    def create(**kwargs):
        allow_create.wait(5)
        return LateStream()

    monkeypatch.setattr(c._client.chat.completions, 'create', create)
    try:
        with pytest.raises(TimeoutError):
            c._chat_streaming({'model': c.model, 'messages': []})
        allow_create.set()
        assert closed.wait(0.5), 'A late stream must notice caller abandonment.'
    finally:
        allow_create.set()
        closed.set()


def test_watchdog_stream_close_cannot_return_success():
    import threading as _t
    closed = _t.Event()

    class ClosingStream:
        def __iter__(self):
            closed.wait(5)
            return iter(())

        def close(self):
            closed.set()

    c, _ = _make_client(ClosingStream())
    try:
        with pytest.raises(TimeoutError):
            c._chat_streaming({'model': c.model, 'messages': []}, stream_max_s=0.05)
    finally:
        closed.set()



def test_reasoning_token_accounting_keeps_total_usage():
    raw = SimpleNamespace(prompt_tokens=10, completion_tokens=30, total_tokens=40,
                          completion_tokens_details=SimpleNamespace(reasoning_tokens=20))
    usage = HKGAIClient._extract_usage(raw)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40, "reasoning_tokens": 20}
