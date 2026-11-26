"""Contracts for events injected by the Web human/watchdog bridge."""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.eventbus.broker.registry import Registry


@pytest.fixture
def registry() -> Registry:
    root = Path(__file__).resolve().parents[2]
    return Registry(root / "eventbus-registry.yaml")


def test_watchdog_failure_is_accepted_by_real_eventbus(registry: Registry) -> None:
    payload = {
        "agent_name": "MossWatchdog",
        "reason_classification": "transient",
        "underlying_reason": "no state progress before timeout",
        "underlying_reason_code": "watchdog_stall_timeout",
    }

    assert registry.is_legitimate_emitter("agent.failed", "MossWatchdog")
    assert list(registry.get_payload_validator("agent.failed", 1).iter_errors(payload)) == []


def test_human_plan_review_is_accepted_by_real_eventbus(registry: Registry) -> None:
    payload = {
        "verdict": "pass",
        "review_report_ref": "spec://critique_report/s/experiment_plan-v1",
    }

    assert registry.is_legitimate_emitter("experiment.plan_reviewed", "Human")
    assert list(
        registry.get_payload_validator("experiment.plan_reviewed", 1).iter_errors(payload)
    ) == []
