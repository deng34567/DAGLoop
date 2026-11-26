"""
Shared LLM layer for the Vibe Programming platform.
Vibe Programming 平台的共享 LLM 层。

Thin client over the HKGAI endpoint (OpenAI-compatible, model DS3.2) used by
all business agents. Returns token usage so callers can compute cost_delta.
对 HKGAI 端点的薄客户端，业务 agent 共用；返回 token usage 供算 cost_delta。
"""

from __future__ import annotations

from autoresearch.llm.client import HKGAIClient, LLMResult

__all__ = ["HKGAIClient", "LLMResult"]
