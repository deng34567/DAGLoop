"""
HKGAI LLM client / HKGAI 大模型客户端。

A thin, proxy-bypassing wrapper over the HKGAI endpoint (OpenAI-compatible,
model DS3.2). Returns token ``usage`` so callers can compute ``cost_delta``.

对 HKGAI 端点（OpenAI 兼容，模型 DS3.2）的薄封装，绕过 clash 代理。
返回 token usage，调用方据此计算 cost_delta。

Self-contained: no imports from vibe.eventbus / vibe.orchestrator.
自包含：不依赖 eventbus / orchestrator。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import openai
from openai import OpenAI

from autoresearch.llm import pricing
from autoresearch.llm.errors import is_quota_exhaustion

logger = logging.getLogger("vibe.llm")

# Per-call token usage is logged here (one INFO line per LLM call + running total).
_USAGE_LOG = logging.getLogger("vibe.llm.usage")

# Global cap on SIMULTANEOUS real-LLM calls -- the overheating guard for concurrent sessions
# (Fix C). Serial / single-session callers never contend (the slot is free); concurrent sessions
# queue here so no more than N real completions are ever in flight at once.
_LLM_MAX_CONCURRENCY = max(1, int(os.environ.get("AUTORESEARCH_LLM_MAX_CONCURRENCY", "4")))
_LLM_SEMAPHORE = threading.BoundedSemaphore(_LLM_MAX_CONCURRENCY)

# Max wall-clock for ONE streaming completion. Bounds a gateway that trickles tokens without ever
# finishing (which the per-read httpx timeout can't catch) so a single call can't freeze the
# state-machine thread for minutes. Generous by default; env-tunable.
_STREAM_MAX_S = float(os.environ.get("HKGAI_STREAM_MAX_S", "1800"))
# STALL detector: if a streaming call receives NO generation progress for this long, the connection is
# treated as dead (glm gateway keep-alive stall) and abandoned so the caller can retry on a FRESH
# connection -- which almost always flows (a stalled connection is a per-connection transient). This
# is independent of total call length: a call still emitting content keeps resetting it, so slow
# heavy reasoning is unaffected; only a "keep-alive but no content" stall trips it.
_STREAM_NO_CONTENT_S = float(os.environ.get("HKGAI_STREAM_NO_CONTENT_S", "1800"))

# Streaming retry policy (borrowed from vibe's upper-layer backoff-retry). A stalled/truncated
# stream is a per-connection transient, so retry on a FRESH connection with EXPONENTIAL BACKOFF --
# giving a stuck gateway time to recover instead of immediately re-hammering it (immediate retry
# under sustained load just re-hits the same wall; live: glm stalled 24x in 1.5h at concurrency=8).
# _STREAM_RETRIES = total attempts (initial + retries). Env-tunable so a stall-prone endpoint (glm)
# can retry more without touching flash/v4-pro.
_STREAM_RETRIES = max(1, int(os.environ.get("HKGAI_STREAM_RETRIES", "3")))
_STREAM_RETRY_BASE_S = float(os.environ.get("HKGAI_STREAM_RETRY_BASE_S", "2"))
_STREAM_RETRY_CAP_S = float(os.environ.get("HKGAI_STREAM_RETRY_CAP_S", "30"))

# 老师 2026-08-05: 非流式调用也要有兜底 —— 一次 timeout/连接抖动/5xx/限流不该让整条 run 挂。TRANSIENT
# errors on a NON-streaming completion are retried (fresh request + exponential backoff), reusing the
# streaming retry budget/backoff; a genuinely dead endpoint still fails after the last attempt.
_TRANSIENT_LLM_ERRORS = (
    openai.APITimeoutError, openai.APIConnectionError,
    openai.InternalServerError, openai.RateLimitError,
    httpx.TimeoutException, httpx.TransportError,
)


def _mark_provider_unavailable(exc: BaseException) -> None:
    """Preserve that an exhausted transport error came from the LLM provider."""
    message = f"LLMServiceUnavailableError: {type(exc).__name__}: {exc}"
    if hasattr(exc, "message"):
        exc.message = message
    exc.args = (message,)


@contextlib.contextmanager
def llm_concurrency_slot():
    """Acquire one of the global LLM concurrency slots for the duration of a completion."""
    _LLM_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _LLM_SEMAPHORE.release()

# Repo-root .env, resolved relative to this file. / 仓库根 .env。
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


@dataclass
class LLMResult:
    """One chat completion result. / 单次对话补全的结果。"""

    content: str | None
    tool_calls: list[dict] | None
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    cost_delta: str  # decimal string from pricing.cost_for
    finish_reason: str | None = None
    request_id: str | None = None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str | None) -> str | None:
    """Remove reasoning-model think blocks / stray tags from message content. Reasoning models
    (e.g. deepseek-v4-flash) put their chain-of-thought in a separate ``reasoning_content`` field,
    but occasionally leak ``<think>...</think>`` (or a lone tag) into ``content`` -- which would
    poison JSON extraction and report text. Normal content has no such tags, so this is a no-op
    for non-reasoning models."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip() or None


def _load_env() -> None:
    """
    Load repo-root .env into os.environ (without clobbering set vars).
    把仓库根 .env 读进 os.environ（已存在的变量不覆盖）。

    Uses python-dotenv if importable, else a minimal manual parser.
    优先用 python-dotenv，不可用则手动解析。
    """
    if not _ENV_PATH.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(_ENV_PATH, override=False)
        return
    except Exception:
        pass

    # Manual fallback parser. / 手动兜底解析。
    for raw in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class HKGAIClient:
    """
    Thin OpenAI-compatible client for HKGAI / DS3.2.
    针对 HKGAI / DS3.2 的薄 OpenAI 兼容客户端。

    Defaults are pulled from env (repo-root .env). The underlying OpenAI
    client is built with a proxy-bypassing httpx client (trust_env=False) so
    that the local clash proxy (127.0.0.1:7890) does not break HKGAI's SSL.
    默认值来自 .env；底层 OpenAI 客户端用 trust_env=False 的 httpx 绕过 clash。
    """

    # process-wide cumulative token usage across ALL HKGAIClient instances,
    # logged per-call so a run's total token spend is visible in the log.
    _cum = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        cookie: str | None = None,
        response_format_mode: str | None = None,
        http_client: httpx.Client | None = None,
        openai_client: Any | None = None,
        stream: bool | None = None,
        timeout: float | None = None,
        default_max_tokens: int | None = None,
    ) -> None:
        _load_env()
        # Explicit per-client HTTP timeout (overrides HKGAI_HTTP_TIMEOUT env). Used
        # e.g. by the ScenarioAgent, whose chunked per-category calls are small and
        # should fail fast rather than wait the generous default — bounds futile
        # retries on a hung call. / 显式超时,绕过 env 默认;给"小调用快失败"用。
        self._timeout_override = timeout
        if default_max_tokens is None:
            raw_max_tokens = os.environ.get("AUTORESEARCH_LLM_MAX_TOKENS")
            default_max_tokens = int(raw_max_tokens) if raw_max_tokens else None
        if default_max_tokens is not None and default_max_tokens < 1:
            raise ValueError("AUTORESEARCH_LLM_MAX_TOKENS must be a positive integer")
        self.default_max_tokens = default_max_tokens
        temperature = os.environ.get("AUTORESEARCH_LLM_TEMPERATURE")
        self.temperature = float(temperature) if temperature else None
        self.extra_body = json.loads(os.environ.get("AUTORESEARCH_LLM_EXTRA_BODY") or "{}")
        if not isinstance(self.extra_body, dict):
            raise ValueError("AUTORESEARCH_LLM_EXTRA_BODY must be a JSON object")

        # Streaming default. The HKGAI gateway enforces a ~360s hard timeout on a
        # NON-streaming response body: large single-shot JSON generations (scenario
        # full-set, architect Mode B/D, Doc) 504 even though the model itself is
        # fast, because the whole body must arrive within that window. Streaming
        # keeps the connection alive with incremental chunks and dodges the 504
        # (verified: a Mode B prompt that 504s at 362s non-streamed completes in
        # ~110s streamed). Callers that produce big JSON build the client with
        # stream=True. / 流式绕过网关对非流式响应体的 360s 硬超时。
        if stream is None:
            stream = os.environ.get("HKGAI_STREAM", "0") == "1"
        self.stream = stream

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.hkgai.net/v1"
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "DS3.2")
        self.cookie = cookie or os.environ.get("HKGAI_COOKIE")
        # json_object 开关:仅当显式开启(构造传 response_format_mode,或 env
        # OPENAI_RESPONSE_FORMAT_MODE=json_object)才发 response_format;不设 → None → 不发。
        # 给不支持 json_object 的端点用(如第三方 external-api-platform 的 deepseek)。
        self.response_format_mode = response_format_mode or os.environ.get(
            "OPENAI_RESPONSE_FORMAT_MODE"
        )
        # reasoning effort 控制 —— 这是【动脑型分析 agent】(走 direct client:requirement /
        # architect / scenario / coder 分析 / moss / doc / sentinel clustering)的档位。沙箱内
        # agent loop(coder 编码 + sentinel 验收)走另一条路(adapter),读 VIBE_SANDBOX_REASONING_EFFORT,
        # 两者解耦。仅当 VIBE_REASONING_EFFORT 显式设置(如 "high")才发 reasoning_effort。
        # ★kimi 语义:参数缺省 = 思考开(慢,kimi 默认);显式发 `none` 才关(注:GLM 是默认 max,
        # 与 kimi 不同)。env-gated:不设即不发,对不认该参数的端点(deepseek)安全。
        reasoning_setting = os.environ.get("VIBE_REASONING_EFFORT")
        self.reasoning_effort = None if reasoning_setting == "default" else reasoning_setting or None
        # ★GLM fix (2026-07-29, 治本): GLM defaults to reasoning_effort=MAX (maximum thinking) -- the
        # SLOWEST possible setting (kimi's default is merely "on"; GLM's is "max"). Under concurrent
        # load its first token then takes MINUTES, tripping every no-content stall/timeout -> the glm
        # runs crawled and 24x-stalled while flash/v4-pro (which ignore the param) were fine. GLM must
        # run with thinking OFF for this optimize-loop workload. So for a glm-* model, DEFAULT
        # reasoning_effort to "none" (explicitly disable thinking) unless VIBE_REASONING_EFFORT was
        # set. Verified semantics (vibe/llm/client.py): sending `none` is what turns GLM thinking off;
        # harmless on deepseek endpoints (they ignore an unrecognized reasoning_effort).
        if reasoning_setting != "default" and self.reasoning_effort is None and str(self.model or "").lower().startswith("glm"):
            self.reasoning_effort = "none"

        # Allow full client injection for tests. / 测试可整客户端注入。
        if openai_client is not None:
            self._client = (openai_client.with_options(max_retries=0)
                            if isinstance(openai_client, OpenAI) else openai_client)
            return

        # Build a proxy-bypassing http client. trust_env=False ignores
        # HTTP(S)_PROXY / clash. / 构造绕过代理的 http 客户端。
        # Timeout is generous and env-tunable: heavy analysis prompts (architect
        # Mode A, scenario full-set generation) on a shared/saturated DS3.2
        # endpoint can legitimately take >60s. A too-short timeout turns normal
        # latency into spurious agent.failed events. / 超时可经 env 调,默认放宽。
        if http_client is None:
            if self._timeout_override is not None:
                timeout_s = self._timeout_override
            else:
                try:
                    timeout_s = float(os.environ.get("HKGAI_HTTP_TIMEOUT", "300"))
                except (TypeError, ValueError):
                    timeout_s = 300.0
            http_client = httpx.Client(trust_env=False, timeout=timeout_s)

        default_headers: dict[str, str] = {}
        if self.cookie:
            default_headers["Cookie"] = self.cookie

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
            default_headers=default_headers or None,
            max_retries=0,  # chat owns the single bounded transport retry budget.
        )

    def chat(
        self,
        *,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: str | None = None,
        max_tokens: int | None = None,
        stream: bool | None = None,
        stream_max_s: float | None = None,
    ) -> LLMResult:
        """
        Run one chat completion. / 跑一次对话补全。

        Args:
            messages: OpenAI-style message dicts.
            tools: optional OpenAI tool/function definitions.
            response_format: "json_object" sends ``{"type": "json_object"}``
                only when the client opt-in mode is enabled;
                "json_object_required" always sends it for a caller that cannot
                safely consume unstructured output; None sends no response_format.
                HKGAI does NOT accept strict json_schema.
            max_tokens: optional cap on completion tokens.
            stream: override the client's streaming default for this call.

        Returns:
            LLMResult with content, tool_calls, usage, and cost_delta.
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens
        # 仅当 caller 要 json_object 且实例 mode 显式开启才真发(一键关:不设该 env 即不发,
        # 给不支持 json_object 的端点用)。★mode=json_schema:发通用 strict object schema 而非
        # json_object —— kimi/sglang 上 reasoning(思考)+ json_object 会返回空 content(其
        # reasoning 解析器与 json_object 的 guided-decoding 冲突,整段 JSON 落进 reasoning_content);
        # 换成 {type:object, additionalProperties:true, strict:true} 的 json_schema 即两全
        # (thinking 开 + content 干净 JSON),且无需每个 agent 单独写 properties(2026-07-01 实测)。
        if response_format == "json_object_required":
            kwargs["response_format"] = {"type": "json_object"}
        elif response_format == "json_object" and self.response_format_mode:
            if self.response_format_mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_output",
                        "strict": True,
                        "schema": {"type": "object", "additionalProperties": True},
                    },
                }
            elif self.response_format_mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        # reasoning 模型的思考力度(env-gated;见 __init__)。加在 streaming 分支前,故流式/非流式
        # 两条路都带上。
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        use_stream = self.stream if stream is None else stream
        # Hold a global concurrency slot for the whole network call (overheating guard, Fix C).
        with llm_concurrency_slot():
            if use_stream:
                # A watchdog-aborted stream (keep-alive-only stall, or trickle past the wall-clock
                # cap) is a TRANSIENT gateway condition, not a permanent failure -- the very next
                # call almost always flows. The watchdog only ABORTS the stall; the RETRY is the
                # other half, and it belongs here so EVERY caller benefits (a live glm run stalled
                # one call in repo_improve's discover phase, the TimeoutError bubbled up through
                # run_repo_improve -> RunnerParseError, and the whole run failed). Retry ONCE only,
                # so a genuinely dead endpoint still fails fast instead of hanging indefinitely.
                # A stalled connection is a per-connection transient -- a FRESH call almost always
                # flows (proven live: same call 6.6s on a new connection). So retry on a fresh
                # connection up to twice to get the REAL answer, rather than fail open. A genuinely
                # dead endpoint still fails fast (each attempt is bounded by the stall detector).
                _attempts = _STREAM_RETRIES
                for _i in range(_attempts):
                    try:
                        started = time.monotonic()
                        logger.info("llm.attempt model=%s stream=true outer=%s/%s sdk_retries=0", self.model, _i + 1, _attempts)
                        response = self._chat_streaming(kwargs, stream_max_s)
                        logger.info("llm.attempt_complete model=%s outer=%s elapsed=%.3f", self.model, _i + 1, time.monotonic() - started)
                        return response
                    except (TimeoutError, *_TRANSIENT_LLM_ERRORS) as exc:
                        if is_quota_exhaustion(exc):
                            raise
                        if _i + 1 >= _attempts:
                            _mark_provider_unavailable(exc)
                            raise
                        logger.warning("llm.stream.stall_retry %d/%d model=%s: %s",
                                       _i + 1, _attempts - 1, self.model, exc)
                        # Exponential backoff before the fresh-connection retry: let a stuck gateway
                        # recover instead of re-hammering it (vibe's backoff-retry). Bounded by cap.
                        time.sleep(min(_STREAM_RETRY_BASE_S * (2 ** _i), _STREAM_RETRY_CAP_S))

            # NON-streaming (default for the short web-agent calls). 老师 2026-08-05: don't let a lone
            # transient (timeout / connection reset / 504 / rate-limit) sink the whole run -- retry on a
            # fresh request with exponential backoff, same budget as the streaming path. A truly dead
            # endpoint still fails fast after the final attempt (no infinite hang).
            _ns_attempts = _STREAM_RETRIES
            for _i in range(_ns_attempts):
                try:
                    started = time.monotonic()
                    logger.info("llm.attempt model=%s stream=false outer=%s/%s sdk_retries=0", self.model, _i + 1, _ns_attempts)
                    resp = self._client.chat.completions.create(**kwargs)
                    logger.info("llm.attempt_complete model=%s outer=%s elapsed=%.3f", self.model, _i + 1, time.monotonic() - started)
                    break
                except _TRANSIENT_LLM_ERRORS as exc:
                    if is_quota_exhaustion(exc):
                        raise
                    if _i + 1 >= _ns_attempts:
                        _mark_provider_unavailable(exc)
                        raise
                    logger.warning("llm.nonstream.retry %d/%d model=%s: %s",
                                   _i + 1, _ns_attempts - 1, self.model, exc)
                    time.sleep(min(_STREAM_RETRY_BASE_S * (2 ** _i), _STREAM_RETRY_CAP_S))

            choice = resp.choices[0]
            message = choice.message

            content = _strip_think(getattr(message, "content", None))

            tool_calls = self._extract_tool_calls(getattr(message, "tool_calls", None))

            usage = self._extract_usage(getattr(resp, "usage", None))
            cost_delta = pricing.cost_for(usage, self.model)
            self._record_usage(usage)
            _USAGE_LOG.info("[LLM completion] model=%s reasoning_setting=%s finish_reason=%s request_id=%s reasoning_tokens=%s",
                            self.model, self.reasoning_effort or "provider_default",
                            getattr(choice, "finish_reason", None), getattr(resp, "id", None), usage.get("reasoning_tokens"))

            return LLMResult(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                cost_delta=cost_delta,
                finish_reason=getattr(choice, "finish_reason", None),
                request_id=getattr(resp, "id", None),
            )

    def _record_usage(self, usage: dict) -> None:
        """Log this call's token usage + the running cumulative total."""
        cls = HKGAIClient
        p = int(usage.get("prompt_tokens", 0) or 0)
        c = int(usage.get("completion_tokens", 0) or 0)
        t = int(usage.get("total_tokens", 0) or (p + c))
        cls._cum["calls"] += 1
        cls._cum["prompt"] += p
        cls._cum["completion"] += c
        cls._cum["total"] += t
        _USAGE_LOG.info(
            "[LLM usage] model=%s call: prompt=%d completion=%d total=%d | "
            "cumulative calls=%d prompt=%d completion=%d total=%d",
            self.model, p, c, t,
            cls._cum["calls"], cls._cum["prompt"],
            cls._cum["completion"], cls._cum["total"],
        )

    def _chat_streaming(self, kwargs: dict, stream_max_s: float | None = None) -> LLMResult:
        """Stream the completion with a HARD wall-clock guarantee.

        ``stream_max_s`` overrides the global cap FOR THIS CALL (best-effort calls pass a short
        one so a stall degrades in seconds instead of minutes).

        Why a worker thread: the inner watchdog closes the stream at the deadline, but a thread
        WEDGED in a blocking SSL ``recv`` (the gateway trickles keep-alive bytes but no content)
        does NOT wake on ``stream.close()`` from another thread -- seen live via py-spy: a
        ``design_experiment`` chat stuck ~30 min in ``ssl.py`` recv despite the 600s watchdog,
        freezing the whole session (ReproducingBaseline showed stale). So run the consume loop on a
        daemon thread and ABANDON it if it overruns: the caller gets a clean ``TimeoutError`` on
        time and the zombie thread dies whenever the gateway finally acts. The +10s grace lets the
        inner watchdog win first (clean close + usage accounting) in the common trickle case.
        """
        _max_s = stream_max_s if stream_max_s is not None else _STREAM_MAX_S
        _no_content_s = min(_max_s, _STREAM_NO_CONTENT_S)
        result: dict = {}
        error: dict = {}
        # shared progress: the inner worker stamps ``last`` for answer, reasoning or tool tokens, so the
        # outer poller can tell "stalled connection" (no content, abandon fast -> retry fresh) from
        # "slow but producing" (keep waiting up to _max_s).
        progress: dict = {"last": time.monotonic(), "start": time.monotonic(),
                          "abandoned": False}

        def _abandon():
            progress["abandoned"] = True
            abort = progress.get("abort")
            if abort is not None:
                # Closing a wedged transport can itself block; never delay the caller's timeout.
                threading.Thread(target=abort, daemon=True).start()

        def _work():
            try:
                result["v"] = self._chat_streaming_inner(kwargs, _max_s, progress)
            except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller thread below
                error["e"] = exc

        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(2)
            if not worker.is_alive():
                break
            now = time.monotonic()
            if now - progress["last"] > _no_content_s:   # STALL: no content -> dead connection
                _abandon()
                raise TimeoutError(
                    f"streaming stalled: no content for {_no_content_s:.0f}s "
                    f"(abandoned, retry on fresh connection; model={self.model})")
            if now - progress["start"] > _max_s + 10:     # hard ceiling even if content trickles
                _abandon()
                raise TimeoutError(
                    f"streaming hard-timeout past {_max_s:.0f}s (abandoned; model={self.model})")
        if "e" in error:
            raise error["e"]
        return result["v"]

    def _chat_streaming_inner(self, kwargs: dict, _max_s: float, progress: dict | None = None) -> LLMResult:
        """Consume the stream + accumulate one LLMResult. Runs on a (possibly-abandoned) worker
        thread; requests a usage chunk via ``stream_options`` (falls back to a token estimate).
        See ``_chat_streaming`` for the hard-timeout wrapper. / 流式累积成一个 LLMResult。"""
        kwargs = dict(kwargs)
        kwargs["stream"] = True
        # Ask the gateway to emit a usage chunk at the end (OpenAI-compatible).
        kwargs.setdefault("stream_options", {"include_usage": True})

        content_parts: list[str] = []
        tool_buf: dict[int, dict] = {}
        usage_obj: Any = None
        finish_reason: str | None = None
        request_id: str | None = None
        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception:
            # some gateways reject stream_options; retry without it.
            if progress is not None and progress.get("abandoned"):
                raise
            kwargs.pop("stream_options", None)
            stream = self._client.chat.completions.create(**kwargs)

        # WALL-CLOCK DEADLINE on the whole stream. The httpx read timeout only bounds the IDLE gap
        # between chunks; a gateway that TRICKLES reasoning tokens (deepseek reasoning_content) keeps
        # resetting that read timeout and can drag a single call to 15+ min -- and because this call
        # runs on the state-machine's own thread, the whole session freezes for that long (seen live
        # on a repo_improve self-heal call). Bound the total stream time so a runaway/trickle aborts
        # instead of hanging; the caller treats the raised error as a normal failed call (retry / no
        # correction), never an indefinite stall. Env-tunable; generous enough for heavy reasoning.
        #
        # A per-chunk check ALONE is not enough: the gateway can hold the connection open sending only
        # SSE keep-alive comments (which the SDK swallows WITHOUT yielding a chunk), so ``for chunk in
        # stream`` blocks inside ``__next__`` forever -- the read timeout keeps resetting on the
        # keep-alives and the loop body (where the per-chunk check lives) never runs. Seen live: a
        # session frozen 13+ min at ReproducingBaseline with the socket ESTABLISHED but no content.
        # So ALSO arm a watchdog timer that closes the stream at the deadline, unblocking that read.
        _deadline = time.monotonic() + _max_s
        _timed_out = {"v": False}

        def _abort_stream():
            _timed_out["v"] = True
            with contextlib.suppress(Exception):
                stream.close()          # unblocks a read stuck waiting for the first/next chunk

        if progress is not None:
            progress["abort"] = _abort_stream
            if progress.get("abandoned"):
                _abort_stream()
                raise TimeoutError(f"streaming request already abandoned (model={self.model})")
        _watchdog = threading.Timer(_max_s, _abort_stream)
        _watchdog.daemon = True
        _watchdog.start()
        try:
            for chunk in stream:
                if (_timed_out["v"] or (progress is not None and progress.get("abandoned"))
                        or time.monotonic() > _deadline):
                    raise TimeoutError(
                        f"streaming exceeded {_max_s:.0f}s wall-clock (model={self.model})")
                cu = getattr(chunk, "usage", None)
                request_id = getattr(chunk, "id", None) or request_id
                if cu is not None:
                    usage_obj = cu
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                fr = getattr(choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    content_parts.append(piece)
                if progress is not None and (piece or getattr(delta, "reasoning_content", None)):
                    progress["last"] = time.monotonic()
                dtcs = getattr(delta, "tool_calls", None)
                if dtcs:
                    for tc in dtcs:
                        idx = getattr(tc, "index", 0) or 0
                        slot = tool_buf.setdefault(
                            idx, {"id": None, "type": "function",
                                   "function": {"name": None, "arguments": ""}}
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if progress is not None and (getattr(tc, "id", None) or (
                                fn is not None and (getattr(fn, "name", None)
                                                    or getattr(fn, "arguments", None)))):
                            progress["last"] = time.monotonic()
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["function"]["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["function"]["arguments"] += fn.arguments
        except Exception:
            # the watchdog closed the stream mid-read (keep-alive-only stall that never yields a
            # chunk) -> surface as a clean wall-clock timeout; any OTHER transport error propagates
            # unchanged for the caller's normal retry/fail-open handling.
            if _timed_out["v"]:
                raise TimeoutError(
                    f"streaming stalled past {_max_s:.0f}s with no content (model={self.model})")
            raise
        finally:
            _watchdog.cancel()

        if _timed_out["v"] or (progress is not None and progress.get("abandoned")):
            raise TimeoutError(f"streaming request timed out (model={self.model})")
        # Truncation guard (borrowed from vibe ★2026-07-22): a stream that ends WITHOUT a
        # finish_reason but WITH partial content = the gateway cut the SSE mid-generation (server
        # budget / ~360s gateway wall). Returning the half-message makes downstream json.loads choke
        # and the caller mark it PERMANENT (retry re-hits the same wall, session dies). Truncation is
        # a TRANSPORT failure: raise so the outer loop retries on a FRESH connection (with backoff),
        # not a silent half-answer. A clean end always carries finish_reason (stop/length/tool_calls/
        # content_filter); the trailing usage-only chunk has empty choices and never clears it.
        if finish_reason is None and (content_parts or tool_buf):
            raise TimeoutError(
                f"streaming truncated: ended with no finish_reason (gateway cut mid-stream; "
                f"model={self.model})")

        content = _strip_think("".join(content_parts) or None)
        tool_calls = list(tool_buf.values()) if tool_buf else None

        if usage_obj is not None:
            usage = self._extract_usage(usage_obj)
        else:
            # No usage chunk: estimate so cost accounting stays non-fatal.
            approx = max(1, len(content or "") // 4)
            usage = {"prompt_tokens": 0, "completion_tokens": approx,
                     "total_tokens": approx}
        cost_delta = pricing.cost_for(usage, self.model)
        self._record_usage(usage)
        _USAGE_LOG.info("[LLM completion] model=%s reasoning_setting=%s finish_reason=%s request_id=%s reasoning_tokens=%s",
                        self.model, self.reasoning_effort or "provider_default", finish_reason, request_id, usage.get("reasoning_tokens"))

        return LLMResult(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            cost_delta=cost_delta,
            finish_reason=finish_reason,
            request_id=request_id,
        )

    @staticmethod
    def _extract_tool_calls(raw: Any) -> list[dict] | None:
        """Normalize SDK tool_calls into plain dicts. / 把 tool_calls 规范成 dict 列表。"""
        if not raw:
            return None
        out: list[dict] = []
        for tc in raw:
            fn = getattr(tc, "function", None)
            out.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    },
                }
            )
        return out or None

    @staticmethod
    def _extract_usage(raw: Any) -> dict:
        """Normalize SDK usage into a plain dict. / 把 usage 规范成 dict。"""
        if raw is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage = {
            "prompt_tokens": int(getattr(raw, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(raw, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(raw, "total_tokens", 0) or 0),
        }
        details = getattr(raw, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if type(reasoning_tokens) is int and reasoning_tokens >= 0:
            usage["reasoning_tokens"] = reasoning_tokens
        return usage


__all__ = ["HKGAIClient", "LLMResult"]
