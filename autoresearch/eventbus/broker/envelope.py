"""
Envelope: the outer wrapper of every event published to the EventBus.

Schema defined in schemas/envelope.schema.json. This module provides:
- Envelope dataclass
- EnvelopeBuilder: convenient construction with auto-filled fields (event_id, timestamp)
- envelope schema validation
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from jsonschema import Draft202012Validator, FormatChecker


@dataclass
class Envelope:
    """An EventBus envelope. Created by EnvelopeBuilder, then serialized to JSON for XADD."""

    event_id: str
    session_id: str
    event_type: str
    emitter: str
    timestamp: str
    schema_version: int
    cost_delta: str
    payload: dict
    causation_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON encoding. causation_id is omitted when None."""
        d = {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "emitter": self.emitter,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "cost_delta": self.cost_delta,
            "payload": self.payload,
        }
        if self.causation_id is not None:
            d["causation_id"] = self.causation_id
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def to_redis_fields(self) -> dict:
        """Serialize for XADD: Redis Streams requires flat field=value pairs."""
        # We pack the whole envelope into one 'data' field as JSON so the consumer
        # can deserialize cleanly. Alternative would be flattening every field.
        return {"data": self.to_json()}

    @classmethod
    def from_redis_fields(cls, fields: dict) -> "Envelope":
        """Parse from XREAD result. Assumes Envelope was stored under 'data' key."""
        if "data" not in fields:
            raise ValueError(f"Redis fields missing 'data' key: {fields}")
        d = json.loads(fields["data"])
        return cls(
            event_id=d["event_id"],
            session_id=d["session_id"],
            event_type=d["event_type"],
            emitter=d["emitter"],
            timestamp=d["timestamp"],
            schema_version=d["schema_version"],
            cost_delta=d["cost_delta"],
            payload=d["payload"],
            causation_id=d.get("causation_id"),
        )


class EnvelopeBuilder:
    """Constructs Envelope objects with sensible defaults. emitter is bound at init."""

    def __init__(self, emitter: str, schema_version: int = 1):
        self.emitter = emitter
        self.schema_version = schema_version

    def build(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
    ) -> Envelope:
        return Envelope(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            emitter=self.emitter,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            schema_version=self.schema_version,
            cost_delta=cost_delta,
            payload=payload,
            causation_id=causation_id,
        )


class EnvelopeValidator:
    """Wraps the JSON Schema validator for the envelope itself."""

    _validator: Optional[Draft202012Validator] = None

    @classmethod
    def _load(cls, schema_path: Path) -> Draft202012Validator:
        if cls._validator is None:
            schema = json.loads(schema_path.read_text())
            cls._validator = Draft202012Validator(schema, format_checker=FormatChecker())
        return cls._validator

    @classmethod
    def validate(cls, envelope_dict: dict, schema_path: Path) -> list[str]:
        """Returns list of error messages (empty if valid)."""
        v = cls._load(schema_path)
        return [e.message for e in v.iter_errors(envelope_dict)]
