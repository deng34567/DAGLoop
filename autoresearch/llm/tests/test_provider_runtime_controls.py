"""Configured provider controls reach both real HTTP serializers without network access."""
import json
import math
import threading

import httpx
import pytest
from openai import OpenAI

from autoresearch.llm.client import HKGAIClient
from autoresearch.agents._reuse import openhands_backend as ob
from autoresearch.agents.runner_agent import handler


@pytest.mark.parametrize("route", ["direct", "sandbox"])
def test_configured_provider_body_reaches_http(monkeypatch, route):
    monkeypatch.setenv("AUTORESEARCH_LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    monkeypatch.setenv("AUTORESEARCH_LLM_MAX_TOKENS", "262144")
    monkeypatch.setenv("VIBE_REASONING_EFFORT", "default")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_REASONING_EFFORT", "default")
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 0,
            "model": "unlisted-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as http:
        client = OpenAI(api_key="offline-placeholder", base_url="http://offline.invalid/v1",
                        http_client=http)
        messages = [{"role": "user", "content": "offline"}]
        if route == "direct":
            HKGAIClient(model="unlisted-model", openai_client=client, stream=False).chat(messages=messages)
        else:
            import litellm
            from openhands.sdk.llm.options.chat_options import select_chat_options
            llm = ob.build_sandbox_llm(api_key="offline-placeholder", model="unlisted-model")
            litellm.completion(model=llm.model, messages=messages, client=client,
                drop_params=True, **select_chat_options(llm, {}, has_tools=False))
    assert len(seen) == 1
    assert seen[0]["thinking"] == {"type": "disabled"}
    assert seen[0][
        "max_tokens" if route == "direct" else "max_completion_tokens"
    ] == 262144
    assert "reasoning_effort" not in seen[0]


def test_unlimited_wait_keeps_container_health_checks(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT", "1")
    checked = threading.Event()
    seen = []
    class Conversation:
        def run(self, timeout):
            seen.append(timeout)
            assert checked.wait(3), "unlimited mode must still monitor its workspace"
    monkeypatch.setattr(ob, "_check_sandbox_container", lambda _: checked.set())
    assert ob.OpenHandsBackend._run_convo(Conversation(), None, workspace="offline") == (False, False)
    assert math.isinf(seen[0])
    assert handler._cleanup_cap_s() is None
    forwarded = handler._remote_agentic_env()
    assert forwarded["AUTORESEARCH_AGENTIC_NO_TIME_LIMIT"] == "1"


def test_provider_configuration_forwarded_without_secrets(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    monkeypatch.setenv("AUTORESEARCH_LLM_MAX_TOKENS", "262144")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS", "131072")
    monkeypatch.setenv("OPENAI_MODEL", "unlisted-model")
    monkeypatch.setenv("OPENAI_API_KEY", "private-placeholder")
    forwarded = handler._remote_agentic_env()
    assert forwarded["AUTORESEARCH_LLM_EXTRA_BODY"] == '{"thinking":{"type":"disabled"}}'
    assert forwarded["AUTORESEARCH_LLM_MAX_TOKENS"] == "262144"
    assert forwarded["AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS"] == "131072"
    assert forwarded["OPENAI_MODEL"] == "unlisted-model"
    assert "OPENAI_API_KEY" not in forwarded
