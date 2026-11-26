"""
CostMeter / LLM 成本计量器。

LLM 调用发生在 agent handler 内部；handler 不把成本传给 bus.publish。
MeteredProvider 每次 LLM 调用把 cost 累加进 CostMeter，BrokerBus 在 emit 回执时
snapshot_reset 取出累计成本，作为该事件 envelope 的 cost_delta（spec D2）。

线程安全：agent handler 跑在 worker 线程（asyncio.to_thread），故加锁。
"""

from __future__ import annotations

import threading
from decimal import Decimal, InvalidOperation


class CostMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = Decimal("0")

    def add(self, cost_delta: str | None) -> None:
        """累加一次 LLM 调用的成本（decimal 字符串）。空/非法值忽略。"""
        if not cost_delta:
            return
        with self._lock:
            try:
                self._total += Decimal(str(cost_delta))
            except (InvalidOperation, ValueError):
                pass

    def snapshot_reset(self) -> str:
        """取出累计成本并清零，返回符合 `^-?[0-9]+(\\.[0-9]{1,6})?$` 的 decimal 字符串。"""
        with self._lock:
            value, self._total = self._total, Decimal("0")
        # 量化到 6 位小数，去掉多余尾零但保留整数形态
        q = value.quantize(Decimal("0.000001")) if value != value.to_integral_value() else value
        s = format(q.normalize(), "f")
        return s if s not in ("", "-0") else "0"
