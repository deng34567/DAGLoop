from types import SimpleNamespace
import httpx
import openai
import pytest
from autoresearch.llm import client as module
from autoresearch.llm.errors import is_resumable_provider_error


def test_explicit_quota_has_one_http_attempt(monkeypatch):
    calls = []
    def exhausted(request):
        calls.append(request)
        return httpx.Response(429, request=request,
                              json={"error": {"code": "insufficient_quota", "message": "insufficient_quota"}})
    monkeypatch.setattr(module, "_STREAM_RETRY_BASE_S", 0)
    with httpx.Client(transport=httpx.MockTransport(exhausted), trust_env=False) as transport:
        sdk = openai.OpenAI(api_key="offline", base_url="http://offline.invalid", http_client=transport)
        client = module.HKGAIClient(openai_client=sdk, stream=False)
        with pytest.raises(openai.RateLimitError):
            client.chat(messages=[])
    assert len(calls) == 1


def test_midstream_transport_failure_restarts_whole_response(monkeypatch):
    client = module.HKGAIClient(openai_client=SimpleNamespace(), stream=True)
    monkeypatch.setattr(module, "_STREAM_RETRY_BASE_S", 0)
    calls = []
    def stream(*args):
        calls.append(1)
        if len(calls) == 1:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://offline.invalid"))
        return module.LLMResult(content='{"complete":true}', tool_calls=[], usage={}, cost_delta="0")
    monkeypatch.setattr(client, "_chat_streaming", stream)
    assert client.chat(messages=[]).content == '{"complete":true}'
    assert len(calls) == 2


def test_exhausted_stream_connection_reset_is_resumable_provider_error(monkeypatch):
    client = module.HKGAIClient(openai_client=SimpleNamespace(), stream=True)
    monkeypatch.setattr(module, "_STREAM_RETRIES", 3)
    monkeypatch.setattr(module, "_STREAM_RETRY_BASE_S", 0)
    calls = []

    def reset(*args):
        calls.append(1)
        raise httpx.ReadError(
            "[Errno 104] Connection reset by peer",
            request=httpx.Request("POST", "http://offline.invalid"),
        )

    monkeypatch.setattr(client, "_chat_streaming", reset)
    with pytest.raises(httpx.ReadError) as caught:
        client.chat(messages=[])

    assert len(calls) == 3
    assert is_resumable_provider_error(str(caught.value))


def test_raw_task_connection_reset_is_not_globally_misclassified():
    assert not is_resumable_provider_error(
        "Connection reset by peer while downloading a benchmark dataset"
    )


@pytest.mark.parametrize("stream", [False, True])
def test_provider_502_retains_origin_when_callers_serialize_the_error(monkeypatch, stream):
    calls = []
    def unavailable(request):
        calls.append(request)
        return httpx.Response(502, request=request, text="<html>502 Bad Gateway</html>")
    monkeypatch.setattr(module, "_STREAM_RETRIES", 1)
    monkeypatch.setattr(module, "_STREAM_RETRY_BASE_S", 0)
    with httpx.Client(transport=httpx.MockTransport(unavailable), trust_env=False) as transport:
        sdk = openai.OpenAI(api_key="offline", base_url="http://offline.invalid",
                            http_client=transport, max_retries=0)
        client = module.HKGAIClient(openai_client=sdk, stream=stream)
        with pytest.raises(openai.InternalServerError) as caught:
            client.chat(messages=[])
    assert caught.value.status_code == 502
    assert is_resumable_provider_error(str(caught.value))
    assert len(calls) == (2 if stream else 1)
