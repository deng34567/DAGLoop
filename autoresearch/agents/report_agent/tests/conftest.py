"""
Shared fixtures for ReportAgent tests. / ReportAgent 测试共享 fixture。

No Redis / network / EventBus / SpecStore. A FakeLLM returns a queue of canned
JSON responses (mirrors memory_agent/tests/conftest.py's FakeLLM).
不涉及 Redis / 网络 / EventBus / SpecStore。FakeLLM 返回预设 JSON 响应队列
（镜像 memory_agent 的 FakeLLM）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeLLMResult:
    content: str | None
    usage: dict
    cost_delta: str
    tool_calls: None = None
    finish_reason: str | None = None


class FakeLLM:
    """Returns a queue of canned responses; records calls. / 返回预设响应队列并记录调用。"""

    def __init__(self, responses: list[str | None | FakeLLMResult]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []
        self.call_kwargs: list[dict] = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        self.call_kwargs.append({
            "response_format": response_format,
            "max_tokens": max_tokens,
        })
        response = self._responses.pop(0) if self._responses else ""
        if isinstance(response, FakeLLMResult):
            return response
        return FakeLLMResult(
            content=response,
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            cost_delta="0.0012",
        )
