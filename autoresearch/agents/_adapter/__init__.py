"""
EventBus transport adapter / 事件总线传输适配层。

把上游 agent（requirement/scenario）自带的同步 InMemoryEventBus + 自定义 LLMProvider
桥接到本平台的：
- 异步 Redis-Streams broker（EventBusClient）—— 见 BrokerBus
- 共享 LLM 层（autoresearch.llm.HKGAIClient）+ cost 计量 —— 见 MeteredProvider / CostMeter

设计依据：specs/2026-05-24-vibe-agents-round1.md §5.7。
"""

from .cost_meter import CostMeter
from .llm_provider import MeteredProvider
from .broker_bus import BrokerBus, causation_var

__all__ = ["CostMeter", "MeteredProvider", "BrokerBus", "causation_var"]
