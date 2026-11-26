"""
Dispatcher: routes events to registered handlers based on event_type pattern matching.

Patterns use fnmatch syntax:
- "coder.*_requested" matches "coder.module_implementation_requested", etc.
- "moss.intervention_requested" matches only that exact event_type
- "*" matches everything

A single event may match multiple patterns; all matching handlers are invoked.
Handler errors propagate so a failed event cannot be acknowledged as processed.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import traceback
from typing import Callable, Awaitable

from .envelope import Envelope

logger = logging.getLogger(__name__)

# Handler signature: async (envelope) -> None
HandlerType = Callable[[Envelope], Awaitable[None]]


class Dispatcher:
    def __init__(self):
        self._handlers: list[tuple[str, HandlerType]] = []

    def register(self, pattern: str, handler: HandlerType) -> None:
        self._handlers.append((pattern, handler))

    def matching_handlers(self, event_type: str) -> list[HandlerType]:
        return [h for (p, h) in self._handlers if fnmatch.fnmatchcase(event_type, p)]

    async def dispatch(self, env: Envelope) -> None:
        """Invoke matching handlers sequentially; stop on a failed delivery."""
        handlers = self.matching_handlers(env.event_type)
        if not handlers:
            logger.debug("No handler matched event_type=%s", env.event_type)
            return
        # Run sequentially so transactional handlers (e.g. state machine tick)
        # don't race with each other on the same SQLite connection.
        for handler in handlers:
            try:
                await handler(env)
            except Exception:
                logger.error(
                    "Handler %s raised on event %s:\n%s",
                    handler.__qualname__,
                    env.event_id,
                    traceback.format_exc(),
                )
                raise
