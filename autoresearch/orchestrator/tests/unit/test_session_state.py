"""Unit tests for session_state.py: JSON roundtrip + SQLite persistence.

SessionState fields follow AutoResearch V1.10 §5 "SessionState 完整结构"
(NOT vibe's V8.1 schema). Ported persistence API (init_session_state_table /
load_session_state / save_session_state / to_json / from_json) is preserved
verbatim from vibe/orchestrator/session_state.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import aiosqlite
import pytest

from autoresearch.orchestrator.session_state import (
    BranchState,
    SessionState,
    bump_map_counter,
    init_session_state_table,
    load_session_state,
    reset_map_counter,
    save_session_state,
)


def _make_state(session_id: str = "sess-1") -> SessionState:
    now = time.time()
    return SessionState(
        session_id=session_id,
        user_id="user-x",
        created_at=now,
        last_updated_at=now,
        current_state="ReviewingTheory",
        previous_state="DevelopingTheory",
        state_entered_at=now,
        mode="idea_theory",
        direction_critique_round=2,
        baseline_reproduce_round={"bench-A": 1},
        hypothesis_reject_round=1,
        theory_review_round={"hyp-1": 2},
        experiment_plan_review_round={"hyp-1": 1},
        implementation_fix_round={"exp-1": 1},
        result_verification_fail_round={"res-1": 1},
        research_rework_count=1,
        inconsistency_bounce_count=1,
        agent_retry_count={"ReviewingTheory": 2},
        literature_audit_round=1,
        search_diversity_collapse_round=1,
        data_analysis_consensus_round={"ext-1": 1},
        research_spec_ref="ref://research_spec/1",
        current_hypothesis_id="hyp-1",
        active_branches=[
            BranchState(branch_id="b1", branch_score=0.5, budget_share=1.0)
        ],
        theory_draft_refs=["ref://theory_draft/1"],
        moss_interventions=[{"reason": "theory_thrashing", "action": "add_prompt"}],
        cumulative_llm_cost="1.5",
        cumulative_compute_cost="0.2",
        last_agent_failure={
            "failed_in_state": "ReviewingTheory",
            "agent_name": "CriticAgent",
            "reason_classification": "transient",
            "underlying_reason": "timeout",
            "retry_attempts": 1,
            "failed_at": now,
        },
        last_human_checkpoint={
            "requested_at": now,
            "requested_in_state": "HumanIdeaCheckpoint",
            "summary": "need seed idea",
        },
        experiment_status="running",
        last_processed_event_id="evt-1",
    )


def test_to_json_from_json_roundtrip():
    s = _make_state()
    j = s.to_json()
    obj = json.loads(j)
    assert obj["session_id"] == "sess-1"
    assert obj["baseline_reproduce_round"]["bench-A"] == 1

    s2 = SessionState.from_json(j)
    assert s2 == s


def test_to_json_handles_empty_state():
    s = SessionState(session_id="sess-empty")
    j = s.to_json()
    s2 = SessionState.from_json(j)
    assert s2.session_id == "sess-empty"
    assert s2.current_state == "ClarifyingResearchGoal"
    assert s2.baseline_reproduce_round == {}
    assert s2.agent_retry_count == {}
    assert s2.active_branches == []


def test_legacy_pending_targeted_search_gets_a_stable_identity():
    raw = json.dumps({
        "session_id": "legacy-targeted",
        "current_state": "DeepResearching",
        "method_ref": "spec://method/legacy/v1",
        "novelty_audit_report_ref": "spec://novelty/legacy/v1",
        "novelty_targeted_search_pending": True,
    })

    first = SessionState.from_json(raw)
    second = SessionState.from_json(raw)
    assert first.novelty_targeted_search_id
    assert first.novelty_targeted_search_id == second.novelty_targeted_search_id
    assert first.novelty_targeted_search_round == 1


def test_map_counters_are_dicts_not_sets():
    """B/D/E/F/G/J/M must serialize as JSON objects (dict), never a set

    (json.dumps on a set raises TypeError -- vibe's original warning).
    """
    s = SessionState(
        session_id="sess-maps",
        baseline_reproduce_round={"bench-A": 1, "bench-B": 2},
        theory_review_round={"hyp-1": 1},
        experiment_plan_review_round={"hyp-1": 1},
        implementation_fix_round={"exp-1": 1},
        result_verification_fail_round={"res-1": 1},
        agent_retry_count={"ReviewingTheory": 1},
        data_analysis_consensus_round={"ext-1": 1},
    )
    assert isinstance(s.baseline_reproduce_round, dict)
    assert not isinstance(s.baseline_reproduce_round, set)

    j = s.to_json()  # must not raise
    raw = json.loads(j)
    assert isinstance(raw["baseline_reproduce_round"], dict)
    assert raw["baseline_reproduce_round"] == {"bench-A": 1, "bench-B": 2}
    assert isinstance(raw["agent_retry_count"], dict)


def test_bump_map_counter_creates_key():
    counter: dict[str, int] = {}
    bump_map_counter(counter, "bench-A")
    assert counter == {"bench-A": 1}
    bump_map_counter(counter, "bench-A")
    assert counter == {"bench-A": 2}


def test_reset_map_counter_deletes_key_not_sets_zero():
    """V1.10 §4 global rule: Map-counter reset is delete_key, never set_zero.

    Invariant: key-present <=> scope currently in that counter's phase.
    """
    counter = {"bench-A": 2, "bench-B": 1}
    reset_map_counter(counter, "bench-A")
    assert "bench-A" not in counter
    assert counter == {"bench-B": 1}
    # 0 would still mean "present" and violate the invariant.
    assert counter.get("bench-A") is None


def test_reset_map_counter_missing_key_is_noop():
    counter: dict[str, int] = {}
    reset_map_counter(counter, "bench-A")
    assert counter == {}


def test_branch_state_roundtrip_via_json():
    s = SessionState(
        session_id="sess-branches",
        active_branches=[
            BranchState(branch_id="b1", status="active", origin="seed", budget_share=0.6),
            BranchState(
                branch_id="b2",
                parent_branch_id="b1",
                status="pruned",
                origin="diversity_expansion",
                branch_score=0.2,
                budget_share=0.4,
            ),
        ],
        max_concurrent_branches=2,
        branch_schedule_policy="best_first_tree",
    )
    j = s.to_json()
    raw = json.loads(j)
    assert isinstance(raw["active_branches"], list)
    assert raw["active_branches"][0]["branch_id"] == "b1"

    s2 = SessionState.from_json(j)
    assert s2.active_branches == s.active_branches
    assert all(isinstance(b, BranchState) for b in s2.active_branches)
    assert s2.max_concurrent_branches == 2
    assert s2.branch_schedule_policy == "best_first_tree"


@pytest.mark.asyncio
async def test_save_load_roundtrip_aiosqlite(tmp_path: Path):
    db = await aiosqlite.connect(str(tmp_path / "t.db"))
    try:
        await init_session_state_table(db)
        s = _make_state("sess-roundtrip")
        await save_session_state(db, s, auto_commit=True)

        loaded = await load_session_state(db, "sess-roundtrip")
        assert loaded is not None
        assert loaded.session_id == "sess-roundtrip"
        assert loaded.current_state == "ReviewingTheory"
        assert loaded.baseline_reproduce_round == {"bench-A": 1}
        assert loaded.active_branches == s.active_branches
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_load_returns_none_for_unknown_session(tmp_path: Path):
    db = await aiosqlite.connect(str(tmp_path / "t.db"))
    try:
        await init_session_state_table(db)
        result = await load_session_state(db, "no-such-session")
        assert result is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_save_second_time_uses_on_conflict_update(tmp_path: Path):
    """Saving the same session_id twice must UPDATE, not INSERT a duplicate."""
    db = await aiosqlite.connect(str(tmp_path / "t.db"))
    try:
        await init_session_state_table(db)
        s = _make_state("sess-dup")
        await save_session_state(db, s, auto_commit=True)

        s.current_state = "PlanningExperiment"
        s.baseline_reproduce_round["bench-A"] = 3
        await save_session_state(db, s, auto_commit=True)

        cur = await db.execute(
            "SELECT COUNT(*) FROM session_state WHERE session_id=?", ("sess-dup",)
        )
        (cnt,) = await cur.fetchone()
        assert cnt == 1

        loaded = await load_session_state(db, "sess-dup")
        assert loaded.current_state == "PlanningExperiment"
        assert loaded.baseline_reproduce_round["bench-A"] == 3
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_current_state_index_matches_state_data_json(tmp_path: Path):
    """The current_state column (indexed) must match the JSON blob's current_state."""
    db = await aiosqlite.connect(str(tmp_path / "t.db"))
    try:
        await init_session_state_table(db)
        s = SessionState(session_id="sess-idx", current_state="ConsolidatingMemory")
        await save_session_state(db, s, auto_commit=True)

        cur = await db.execute(
            "SELECT current_state, state_data FROM session_state WHERE session_id=?",
            ("sess-idx",),
        )
        row = await cur.fetchone()
        idx_col, blob = row
        assert idx_col == "ConsolidatingMemory"
        assert json.loads(blob)["current_state"] == "ConsolidatingMemory"
    finally:
        await db.close()
