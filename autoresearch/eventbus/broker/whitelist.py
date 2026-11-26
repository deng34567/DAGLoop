"""
Whitelist check: ensures the calling emitter is in the registry's allowed list
for the event_type being published.

A violation produces a PolicyViolation exception, which the client converts
into a synthetic agent.failed(policy_violation) event emitted under the 'system'
emitter identity.
"""

from __future__ import annotations

from .registry import Registry


class PolicyViolation(Exception):
    def __init__(
        self,
        offending_emitter: str,
        event_type: str,
        violation_type: str,
        detail: str = "",
    ):
        self.offending_emitter = offending_emitter
        self.event_type = event_type
        self.violation_type = violation_type
        self.detail = detail
        super().__init__(
            f"{offending_emitter} attempted to emit {event_type}: {violation_type} ({detail})"
        )


def check_emitter(registry: Registry, event_type: str, emitter: str) -> None:
    """Raises PolicyViolation if the emitter is not allowed to emit this event_type.

    Cases:
    - event_type unknown -> PolicyViolation(unknown_event_type)
    - emitter not in whitelist -> PolicyViolation(unauthorized_emit)
    """
    if not registry.has(event_type):
        raise PolicyViolation(
            offending_emitter=emitter,
            event_type=event_type,
            violation_type="unknown_event_type",
            detail=f"event_type {event_type} is not in the registry",
        )
    if not registry.is_legitimate_emitter(event_type, emitter):
        allowed = sorted(registry.get(event_type).emitters)
        raise PolicyViolation(
            offending_emitter=emitter,
            event_type=event_type,
            violation_type="unauthorized_emit",
            detail=f"emitter {emitter} not in allowed list {allowed}",
        )
