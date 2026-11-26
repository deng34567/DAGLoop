"""
Registry: loads eventbus-registry.yaml and provides lookup tables.

Used by broker for:
- Looking up legitimate emitters for an event_type (whitelist)
- Looking up the payload JSON Schema path for an event_type
- Knowing the direction of an event (outbound / inbound / cross-cutting)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from jsonschema import Draft202012Validator, FormatChecker


@dataclass
class EventEntry:
    type: str
    direction: str          # outbound | inbound | cross-cutting
    schema_path: str        # relative path under eventbus root
    schema_versions: list[int]
    emitters: set[str]
    skill_status: str = "yaml"

    # Lazily-loaded compiled validator
    _validators: dict[int, Draft202012Validator] = field(default_factory=dict)

    def get_validator(self, schema_version: int, eventbus_root: Path) -> Draft202012Validator:
        if schema_version not in self._validators:
            if schema_version not in self.schema_versions:
                raise ValueError(
                    f"schema_version {schema_version} not supported for {self.type}; "
                    f"supported: {self.schema_versions}"
                )
            schema_path = eventbus_root / self.schema_path
            # For now we only have v1 file; future v2 would require a different file name
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            schema = json.loads(schema_path.read_text())
            self._validators[schema_version] = Draft202012Validator(
                schema, format_checker=FormatChecker()
            )
        return self._validators[schema_version]


class Registry:
    """Loaded from eventbus-registry.yaml at broker startup."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.eventbus_root = registry_path.parent
        self._entries: dict[str, EventEntry] = {}
        self._envelope_validator: Optional[Draft202012Validator] = None
        self.state_machine_skill_version: str = ""
        self._load()

    def _load(self):
        data = yaml.safe_load(self.registry_path.read_text())
        self.state_machine_skill_version = data.get("state_machine_skill_version", "")
        for ev in data.get("events", []):
            entry = EventEntry(
                type=ev["type"],
                direction=ev["direction"],
                schema_path=ev["schema"],
                schema_versions=ev["schema_versions"],
                emitters=set(ev["emitters"]),
                skill_status=ev.get("skill_status", "yaml"),
            )
            self._entries[entry.type] = entry

    @property
    def envelope_validator(self) -> Draft202012Validator:
        if self._envelope_validator is None:
            schema_path = self.eventbus_root / "schemas" / "envelope.schema.json"
            schema = json.loads(schema_path.read_text())
            self._envelope_validator = Draft202012Validator(
                schema, format_checker=FormatChecker()
            )
        return self._envelope_validator

    def get(self, event_type: str) -> EventEntry:
        if event_type not in self._entries:
            raise KeyError(f"Unknown event_type: {event_type}")
        return self._entries[event_type]

    def has(self, event_type: str) -> bool:
        return event_type in self._entries

    def all_types(self) -> list[str]:
        return list(self._entries.keys())

    def is_legitimate_emitter(self, event_type: str, emitter: str) -> bool:
        if event_type not in self._entries:
            return False
        return emitter in self._entries[event_type].emitters

    def get_payload_validator(self, event_type: str, schema_version: int) -> Draft202012Validator:
        return self._entries[event_type].get_validator(schema_version, self.eventbus_root)
