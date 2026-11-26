"""Unit tests for the RankingJudgeAgent contract slice (Task 1).

Covers: ranking.requested, ranking.ready (v1); the RankingReport
data-contract schema; the SpecStore whitelist extension for
ranking_report; and the addition of RankingJudgeAgent as a NEW legal
agent.failed / envelope emitter. Also covers the three new SessionState
fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.eventbus.broker.registry import Registry
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.spec_store import EmitterNotAllowed, SpecStore

EVENTBUS_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = EVENTBUS_ROOT / "eventbus-registry.yaml"
EXAMPLES_DIR = EVENTBUS_ROOT / "examples"

AUTORESEARCH_ROOT = EVENTBUS_ROOT.parent
RANKING_JUDGE_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "ranking_judge_agent"
RANKING_REPORT_SCHEMA_PATH = (
    RANKING_JUDGE_AGENT_ROOT / "data-contracts" / "ranking_report.schema.yaml"
)

RANKING_EVENTS = [
    "ranking.requested",
    "ranking.ready",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# New events
# ---------------------------------------------------------------------------


def test_registry_loads_ranking_validators(registry: Registry):
    for event_type in RANKING_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", RANKING_EVENTS)
def test_examples_validate_against_schema(registry: Registry, event_type: str):
    validator = registry.get_payload_validator(event_type, 1)

    happy_payload = json.loads(
        (EXAMPLES_DIR / f"{event_type}.happy.json").read_text()
    )
    errors = list(validator.iter_errors(happy_payload))
    assert errors == [], (
        f"happy example for {event_type} should validate, got: "
        f"{[e.message for e in errors]}"
    )

    invalid_payload = json.loads(
        (EXAMPLES_DIR / f"{event_type}.invalid.json").read_text()
    )
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, f"invalid example for {event_type} should NOT validate"


def test_ranking_requested_registry_entry(registry: Registry):
    entry = registry.get("ranking.requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_ranking_ready_registry_entry(registry: Registry):
    entry = registry.get("ranking.ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"RankingJudgeAgent"}
    assert 1 in entry.schema_versions


def test_ranking_requested_without_candidate_ids_validates(registry: Registry):
    validator = registry.get_payload_validator("ranking.requested", 1)
    payload = {
        "candidate_kind": "hypothesis",
        "source_ref": "ref://hypothesis_bank/s1/v1",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"ranking.requested without candidate_ids should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_ranking_requested_with_candidate_ids_validates(registry: Registry):
    validator = registry.get_payload_validator("ranking.requested", 1)
    payload = {
        "candidate_kind": "hypothesis",
        "candidate_ids": ["h1", "h2"],
        "source_ref": "ref://hypothesis_bank/s1/v1",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"ranking.requested with candidate_ids should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_ranking_requested_bad_candidate_kind_fails(registry: Registry):
    validator = registry.get_payload_validator("ranking.requested", 1)
    payload = {
        "candidate_kind": "benchmark",
        "source_ref": "ref://hypothesis_bank/s1/v1",
    }
    errors = list(validator.iter_errors(payload))
    assert errors, "ranking.requested with bad candidate_kind should NOT validate"


# ---------------------------------------------------------------------------
# RankingReport data-contract schema
# ---------------------------------------------------------------------------


def test_ranking_report_fixture_validates():
    schema = yaml.safe_load(RANKING_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "candidate_kind": "hypothesis",
        "pairwise_results": [
            {"a": "h1", "b": "h2", "winner": "h1", "rationale": "h1 is more novel."}
        ],
        "btl_scores": [
            {"id": "h1", "score": 1.5},
            {"id": "h2", "score": 0.3},
        ],
        "ranked_shortlist": ["h1", "h2"],
        "diversity_status": "diverse",
        "rationale": "Clear separation between candidates.",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"RankingReport fixture should validate, got: {[e.message for e in errors]}"
    )


def test_ranking_report_with_null_rationale_validates():
    schema = yaml.safe_load(RANKING_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "candidate_kind": "hypothesis",
        "pairwise_results": [],
        "btl_scores": [],
        "ranked_shortlist": [],
        "diversity_status": "collapsed",
        "rationale": None,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"RankingReport with rationale=null should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_ranking_report_missing_required_fails():
    schema = yaml.safe_load(RANKING_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "pairwise_results": [],
        "btl_scores": [],
        "ranked_shortlist": [],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "RankingReport missing `candidate_kind`/`diversity_status` should NOT validate"


def test_ranking_report_bad_diversity_status_enum_fails():
    schema = yaml.safe_load(RANKING_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "candidate_kind": "hypothesis",
        "pairwise_results": [],
        "btl_scores": [],
        "ranked_shortlist": [],
        "diversity_status": "unknown",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "RankingReport with bad diversity_status enum should NOT validate"


def test_ranking_report_extra_field_fails():
    schema = yaml.safe_load(RANKING_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "candidate_kind": "hypothesis",
        "pairwise_results": [],
        "btl_scores": [],
        "ranked_shortlist": [],
        "diversity_status": "diverse",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "RankingReport with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# SpecStore: ranking_report kind
# ---------------------------------------------------------------------------


def test_store_accepts_ranking_report_for_ranking_judge_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="ranking_report",
            session_id="s1",
            key="ranking-1",
            obj={
                "candidate_kind": "hypothesis",
                "pairwise_results": [],
                "btl_scores": [],
                "ranked_shortlist": [],
                "diversity_status": "diverse",
            },
            emitter="RankingJudgeAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_ranking_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="ranking_report",
                session_id="s1",
                key="ranking-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# RankingJudgeAgent as a NEW legal agent.failed / envelope emitter.
# ---------------------------------------------------------------------------


def test_agent_failed_includes_ranking_judge_agent(registry: Registry):
    entry = registry.get("agent.failed")
    assert "RankingJudgeAgent" in entry.emitters


def test_envelope_schema_allows_ranking_judge_agent_emitter(registry: Registry):
    validator = registry.envelope_validator
    emitter_enum = validator.schema["properties"]["emitter"]["enum"]
    assert "RankingJudgeAgent" in emitter_enum


def test_agent_failed_payload_schema_allows_ranking_judge_agent():
    schema_path = EVENTBUS_ROOT / "schemas" / "agent.failed.v1.schema.json"
    schema = json.loads(schema_path.read_text())
    agent_name_enum = schema["properties"]["agent_name"]["enum"]
    assert "RankingJudgeAgent" in agent_name_enum


# ---------------------------------------------------------------------------
# SessionState: hypothesis_ranking_report_ref, benchmark_ranking_report_ref,
# passing_hypothesis_ids
# ---------------------------------------------------------------------------


def test_session_state_has_ranking_fields_defaulting_to_none_and_empty():
    state = SessionState(session_id="s1")
    assert state.hypothesis_ranking_report_ref is None
    assert state.benchmark_ranking_report_ref is None
    assert state.passing_hypothesis_ids == []
