"""
AutoResearch EventBus broker library.

V1 MVP: library-embedded broker that wraps Redis Streams + SQLite for outbox
and idempotency. Subscriber processes import this package and instantiate
EventBusClient once.
"""

from .client import EventBusClient
from .envelope import Envelope, EnvelopeBuilder
from .registry import Registry, EventEntry

__all__ = ["EventBusClient", "Envelope", "EnvelopeBuilder", "Registry", "EventEntry"]
