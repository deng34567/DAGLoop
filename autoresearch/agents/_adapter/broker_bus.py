"""
BrokerBus / 桥接上游 agent 的同步 EventBus Protocol 到本平台异步 broker。

上游 agent（requirement/scenario）的 handler 是**同步**的，通过
`bus.subscribe(event_name, handler)` 订阅、`bus.publish(Event(...))` 发布。
本平台 broker 是**异步**的 EventBusClient（`async publish(*, event_type, session_id,
payload, cost_delta, causation_id)` / `subscribe(pattern, async_handler)`）。

桥接策略（spec §5.7）：
- subscribe：把 agent 的同步 handler 存起来，并在 broker 上注册一个 async 派发器。
- inbound：async 派发器把 Envelope 翻译成 agent 的 Event，用 asyncio.to_thread 在
  worker 线程跑同步 handler（避免阻塞事件循环）。
- publish（在 worker 线程被同步调用）：用 run_coroutine_threadsafe 把 async broker.publish
  marshal 回事件循环并阻塞取结果。
- 注入 cost_delta（CostMeter 累计）与 causation_id（当前 inbound 事件 id，contextvar 传递）。
  event_type 直接用 agent 的 Event.name —— 上游 skill 的 event 名已与 registry 对齐（identity）。
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Callable

from .cost_meter import CostMeter

# 当前正在处理的 inbound 事件 id，供 publish 时作为 causation_id。
# asyncio.to_thread 会复制当前 context，故 worker 线程内可见。
causation_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vibe_adapter_causation_id", default=None
)


class BrokerBus:
    """实现上游 agent 的 EventBus Protocol（publish/subscribe），路由到异步 broker。"""

    def __init__(
        self,
        *,
        client: Any,            # autoresearch.eventbus.broker.client.EventBusClient (duck-typed)
        emitter: str,
        meter: CostMeter,
        event_cls: type,        # 该 agent 的 eventbus.Event 类
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._client = client
        self._emitter = emitter
        self._meter = meter
        self._event_cls = event_cls
        self._loop = loop
        self._handlers: dict[str, list[Callable]] = {}

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    # --- EventBus Protocol ---------------------------------------------------

    def subscribe(self, event_name: str, handler: Callable) -> None:
        first = event_name not in self._handlers
        self._handlers.setdefault(event_name, []).append(handler)
        if first:
            # 每个 event_name 在 broker 上注册一次 async 派发器（exact pattern）。
            self._client.subscribe(event_name, self._make_dispatch(event_name))

    def publish(self, event: Any) -> None:
        """同步发布（被 worker 线程内的 agent handler 调用）。"""
        loop = self._loop or asyncio.get_event_loop()
        cost = self._meter.snapshot_reset()
        causation = causation_var.get()
        coro = self._client.publish(
            event_type=event.name,
            session_id=event.session_id,
            payload=event.payload,
            cost_delta=cost or "0",
            causation_id=causation,
        )
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        fut.result(timeout=60)

    # --- inbound dispatch ----------------------------------------------------

    def _make_dispatch(self, event_name: str) -> Callable:
        async def _dispatch(env: Any) -> None:
            self._ensure_loop()
            ev = self._event_cls(
                name=env.event_type,
                session_id=env.session_id,
                payload=dict(env.payload or {}),
                event_id=env.event_id,
            )
            token = causation_var.set(env.event_id)
            self._meter.snapshot_reset()  # 清零，开始累计本次 handler 的 LLM 成本
            try:
                for handler in list(self._handlers.get(event_name, [])):
                    await asyncio.to_thread(handler, ev)
            finally:
                causation_var.reset(token)

        return _dispatch
