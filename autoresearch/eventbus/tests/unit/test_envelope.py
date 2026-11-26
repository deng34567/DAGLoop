"""Unit tests for autoresearch.eventbus.broker.envelope."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

import pytest

from autoresearch.eventbus.broker.envelope import Envelope, EnvelopeBuilder


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_ISO_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def test_builder_autofills_event_id_uuid_and_timestamp_iso_z():
    b = EnvelopeBuilder(emitter="CoderAgent")
    env = b.build(
        event_type="coder.module_implemented",
        session_id="01890abc-def0-7000-8000-123456789abc",
        payload={"x": 1},
    )

    # event_id must be a valid UUID v4 string
    assert _UUID_RE.match(env.event_id), env.event_id
    # also: uuid.UUID() must accept it
    uuid.UUID(env.event_id)

    # timestamp must be ISO-8601 UTC with Z suffix (NOT +00:00)
    assert _ISO_Z_RE.match(env.timestamp), env.timestamp
    assert env.timestamp.endswith("Z")
    assert "+00:00" not in env.timestamp

    # schema_version default = 1
    assert env.schema_version == 1
    # emitter bound at init
    assert env.emitter == "CoderAgent"
    # causation_id default None
    assert env.causation_id is None
    # cost_delta default '0'
    assert env.cost_delta == "0"


def test_builder_custom_schema_version_and_causation_id():
    b = EnvelopeBuilder(emitter="ResearchOrchestrator", schema_version=2)
    env = b.build(
        event_type="hypothesis.clarification_requested",
        session_id="01890abc-def0-7000-8000-123456789abc",
        payload={"foo": "bar"},
        cost_delta="0.001",
        causation_id="11111111-2222-3333-4444-555555555555",
    )
    assert env.schema_version == 2
    assert env.cost_delta == "0.001"
    assert env.causation_id == "11111111-2222-3333-4444-555555555555"


def test_to_dict_omits_causation_id_when_none():
    env = Envelope(
        event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        event_type="coder.module_implemented",
        emitter="CoderAgent",
        timestamp="2026-05-23T12:00:00Z",
        schema_version=1,
        cost_delta="0",
        payload={"k": "v"},
        causation_id=None,
    )
    d = env.to_dict()
    assert "causation_id" not in d, (
        "causation_id must be omitted when None (storage optimization)"
    )
    # but all the required keys are present
    for k in (
        "event_id",
        "session_id",
        "event_type",
        "emitter",
        "timestamp",
        "schema_version",
        "cost_delta",
        "payload",
    ):
        assert k in d


def test_to_dict_includes_causation_id_when_set():
    env = Envelope(
        event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        event_type="coder.module_implemented",
        emitter="CoderAgent",
        timestamp="2026-05-23T12:00:00Z",
        schema_version=1,
        cost_delta="0",
        payload={"k": "v"},
        causation_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
    )
    d = env.to_dict()
    assert d["causation_id"] == "cccccccc-cccc-cccc-cccc-cccccccccccc"


def test_to_json_is_compact_no_whitespace():
    env = Envelope(
        event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        event_type="x.y",
        emitter="CoderAgent",
        timestamp="2026-05-23T12:00:00Z",
        schema_version=1,
        cost_delta="0",
        payload={"k": "v"},
    )
    s = env.to_json()
    # compact separators -> no ", " or ": "
    assert ", " not in s
    assert ": " not in s
    # parseable
    parsed = json.loads(s)
    assert parsed["event_type"] == "x.y"


def test_redis_fields_roundtrip_with_causation_id():
    original = Envelope(
        event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        event_type="coder.module_implemented",
        emitter="CoderAgent",
        timestamp="2026-05-23T12:00:00Z",
        schema_version=1,
        cost_delta="0.012450",
        payload={"module_id": "m1", "code_ref": "spec://x"},
        causation_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
    )
    fields = original.to_redis_fields()
    assert set(fields.keys()) == {"data"}
    restored = Envelope.from_redis_fields(fields)
    assert restored == original


def test_redis_fields_roundtrip_without_causation_id():
    original = Envelope(
        event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        event_type="coder.module_implemented",
        emitter="CoderAgent",
        timestamp="2026-05-23T12:00:00Z",
        schema_version=1,
        cost_delta="0",
        payload={"module_id": "m1", "code_ref": "spec://x"},
        causation_id=None,
    )
    restored = Envelope.from_redis_fields(original.to_redis_fields())
    assert restored.causation_id is None
    assert restored == original


def test_from_redis_fields_missing_data_key_raises():
    with pytest.raises(ValueError, match="missing 'data' key"):
        Envelope.from_redis_fields({"other": "stuff"})


def test_builder_timestamp_is_recent_utc():
    b = EnvelopeBuilder(emitter="CoderAgent")
    env = b.build(
        event_type="coder.module_implemented",
        session_id="01890abc-def0-7000-8000-123456789abc",
        payload={},
    )
    # Drop trailing Z, parse as UTC, must be within ~5 seconds of now
    parsed = datetime.fromisoformat(env.timestamp.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    delta = (now - parsed).total_seconds()
    assert -1.0 <= delta <= 5.0, f"timestamp {env.timestamp} not within tolerance"
