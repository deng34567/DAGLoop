"""
Shared fixtures for ProblemModelingAgent tests. / ProblemModelingAgent 测试共享 fixture。

No Redis / network / EventBus / SpecStore. A FakeLLM returns a queue of canned
JSON responses (mirrors vibe/agents/doc_agent/tests/conftest.py's FakeLLM).
无 Redis / 网络 / EventBus / SpecStore。FakeLLM 返回预设 JSON 响应队列
（镜像 doc_agent 的 FakeLLM）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeLLMResult:
    content: str | None
    usage: dict
    cost_delta: str
    tool_calls: None = None


class FakeLLM:
    """Returns a queue of canned responses; records calls. / 返回预设响应队列并记录调用。"""

    def __init__(self, responses: list[str | None]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        content = self._responses.pop(0) if self._responses else ""
        return FakeLLMResult(
            content=content,
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            cost_delta="0.0012",
        )
