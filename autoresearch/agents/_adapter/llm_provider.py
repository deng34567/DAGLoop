"""
MeteredProvider / 计量型 LLM provider。

把共享 LLM 层 autoresearch.llm.HKGAIClient 适配成上游 agent 期望的 LLMProvider 接口：

    call(*, messages, tools, response_schema) -> <agent 的 LLMResponse>

并把每次调用的 cost 记入 CostMeter（供 BrokerBus 附到事件 cost_delta）。

要点：
- HKGAI 不接受 OpenAI 严格 json_schema；当 agent 传了 response_schema（想要结构化输出）
  时，改用 response_format="json_object"，由 agent 自己的 schema 校验 + 内部 retry 兜底。
- 不吞异常：HKGAIClient 让 openai 异常原样抛出，agent 的 _is_transient 据此分类
  transient/permanent。
"""

from __future__ import annotations

from typing import Any

from autoresearch.llm import HKGAIClient

from .cost_meter import CostMeter

# HTTP statuses that indicate a TRANSIENT infrastructure failure (retry-worthy).
# Mirrors vibe.agents.scenario_agent.llm._TRANSIENT_STATUS / architect-agent
# kernel/internal-retry.md 的失败分流表。
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _is_transient(exc: Exception) -> bool:
    """Is `exc` a TRANSIENT LLM infra failure (timeout / 5xx / rate-limit /
    connection)? Prefer openai SDK exception types; fall back to a duck-typed
    `status_code`. / 判断异常是否属基础设施 transient。"""
    try:
        import openai

        transient_types = tuple(
            t
            for t in (
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "RateLimitError", None),
                getattr(openai, "InternalServerError", None),
            )
            if isinstance(t, type)
        )
        if transient_types and isinstance(exc, transient_types):
            return True
    except Exception:  # pragma: no cover - openai always importable here
        pass

    status = getattr(exc, "status_code", None)
    if isinstance(status, bool):
        return False
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    # Timeouts sometimes surface as a bare message without status_code.
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg or "connection" in msg:
        return True
    return False


class MeteredProvider:
    """Adapt the shared HKGAIClient to an agent's LLMProvider, metering cost.

    Transient LLM infra failures (timeout / 5xx / rate-limit / connection) are
    re-raised as ``infra_error_cls`` so the consuming agent's handler classifies
    them as ``transient`` (Orchestrator G-counter退避重试) instead of letting a
    raw openai exception fall through to ``permanent``. Mirrors the architect's
    ``_call_llm`` wrapping (core.py). If ``infra_error_cls`` is None, transient
    exceptions propagate unchanged (back-compat for callers that classify
    themselves).
    把共享 HKGAIClient 适配成 agent 的 LLMProvider 并计 cost；transient 失败按
    传入的 infra_error_cls 重抛，使下游归类为 transient。
    """

    def __init__(
        self,
        client: HKGAIClient,
        meter: CostMeter,
        response_cls: type,
        *,
        infra_error_cls: type[Exception] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._meter = meter
        self._response_cls = response_cls  # agent 自己的 LLMResponse dataclass
        self._infra_error_cls = infra_error_cls
        # Optional per-call completion cap. Bounds generation time so a slow
        # DS3.2 deployment does not exceed the HKGAI gateway timeout (~360s) and
        # 504 on large structured outputs. / 可选 completion 上限,限制生成时长。
        self._max_tokens = max_tokens

    def call(
        self,
        *,
        messages: list[dict],
        tools: list[dict] | None,
        response_schema: dict | None,
    ) -> Any:
        response_format = "json_object" if response_schema is not None else None
        try:
            res = self._client.chat(
                messages=messages,
                tools=tools,
                response_format=response_format,
                max_tokens=self._max_tokens,
            )
        except Exception as e:  # network / 5xx / timeout from the SDK
            if self._infra_error_cls is not None and _is_transient(e):
                raise self._infra_error_cls(f"transient LLM failure: {e}") from e
            raise
        self._meter.add(res.cost_delta)
        return self._response_cls(
            content=res.content,
            tool_calls=self._flatten_tool_calls(res.tool_calls),
            raw=res,
            reasoning_content=None,
        )

    @staticmethod
    def _flatten_tool_calls(tool_calls: list[dict] | None) -> list[dict] | None:
        """Flatten HKGAIClient's nested tool_calls into the FLAT shape agents consume.

        HKGAIClient._extract_tool_calls emits ``{id,type,function:{name,arguments}}``,
        but the agent tool loops (e.g. requirement_agent/agent.py:156-159 and the
        OpenAIProvider CLI path) index ``tc["id"]/tc["name"]/tc["arguments"]``. Passing
        the nested dict through raises ``KeyError: 'name'`` on the first tool call —
        the bug that kept RequirementAgent web search dormant. Flatten here (the single
        provider seam) so both shapes converge on flat. Already-flat dicts pass through.
        """
        if not tool_calls:
            return None
        out: list[dict] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                out.append(tc)  # unknown shape: leave as-is, don't mask it
                continue
            fn = tc.get("function")
            if isinstance(fn, dict):  # nested -> flatten
                out.append({
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })
            else:  # already flat (or close enough)
                out.append(tc)
        return out or None
