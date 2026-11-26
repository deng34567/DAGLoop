"""Unit tests for Task 4: ReinterpretingSpec disposition routing (scope gate).

Covers: the ``_on_ReinterpretingSpec_spec_reinterpreted`` handler routes on
``event.payload["disposition"]``, updates SessionState (``spec_interpretation_ref``,
and ``research_spec_ref`` <- the revised ref), and transitions:
  - in_scope       -> ModelingProblem (entry action emits
                      ``problem_modeling.model_requested``; 3.3 moved the novelty audit
                      to after GeneratingMethod)
  - scope_changed  -> HumanIdeaCheckpoint (sets human_checkpoint_return_to = ModelingProblem,
                      no entry action -> pauses)
  - out_of_scope   -> AwaitingUserDecision (no entry action -> pauses)

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


async def _advance_to_reinterpreting_spec(eng: OrchestratorEngine, sid: str) -> tuple[str, str, str]:
    """Drive a fresh session (ClarifyingResearchGoal) into ReinterpretingSpec.

    Returns (research_spec_ref, literature_brief_ref, field_trajectory_brief_ref).
    """
    research_spec_ref = f"spec://research_spec/{sid}/final"
    await eng.tick(
        _Evt(
            "research_spec.ready",
            sid,
            "ProblemModelingAgent",
            payload={"research_spec_ref": research_spec_ref},
        )
    )
    literature_brief_ref = f"spec://literature_brief/{sid}/final"
    field_trajectory_brief_ref = f"spec://field_trajectory_brief/{sid}/final"
    await eng.tick(
        _Evt(
            "literature.brief_ready",
            sid,
            "DeepResearchAgent",
            payload={
                "literature_brief_ref": literature_brief_ref,
                "field_trajectory_brief_ref": field_trajectory_brief_ref,
            },
        )
    )
    return research_spec_ref, literature_brief_ref, field_trajectory_brief_ref


async def test_in_scope_advances_to_modeling_problem(fake_bus):
    # 3.3: in_scope now goes straight to ModelingProblem -- the literature-novelty audit was moved
    # to after GeneratingMethod (it audits the METHOD's novelty, not the bare task).
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    _research_spec_ref, literature_brief_ref, _ftb_ref = await _advance_to_reinterpreting_spec(eng, sid)

    spec_interpretation_ref = f"spec_interpretation://{sid}/final"
    revised_research_spec_ref = f"research_spec://{sid}/revised"
    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "in_scope",
                "spec_interpretation_ref": spec_interpretation_ref,
                "revised_research_spec_ref": revised_research_spec_ref,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ModelingProblem"
    assert st.research_spec_ref == revised_research_spec_ref
    assert st.spec_interpretation_ref == spec_interpretation_ref

    emitted = [p for p in fake_bus.published if p.event_type == "problem_modeling.model_requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {
        "research_spec_ref": revised_research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "novelty_audit_report_ref": None,   # 3.3: novelty audit hasn't run yet on the first pass
        "human_idea_seed_ref": None,
    }


async def test_in_scope_transition_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_reinterpreting_spec(eng, sid)

    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "in_scope",
                "spec_interpretation_ref": f"spec_interpretation://{sid}/final",
                "revised_research_spec_ref": f"research_spec://{sid}/revised",
            },
        )
    )

    assert fake_bus.db.commit_count - commits_before == 1


async def test_scope_changed_routes_to_human_idea_checkpoint(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_reinterpreting_spec(eng, sid)

    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "scope_changed",
                "spec_interpretation_ref": f"spec_interpretation://{sid}/final",
                "revised_research_spec_ref": f"research_spec://{sid}/revised",
            },
        )
    )

    # Meeting 2026-07-29: ONE checkpoint only (Layer A). scope_changed no longer pauses at its own
    # checkpoint here -- it flows to ModelingProblem, whose exit is the single HumanIdeaCheckpoint.
    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ModelingProblem"

    emitted = [p for p in fake_bus.published if p.event_type == "evidence.audit_requested"]
    assert len(emitted) == 0


async def test_out_of_scope_now_proceeds_like_in_scope(fake_bus):
    # Meeting 2026-07-25 item 4: the pre-experiment out_of_scope gate was removed. out_of_scope no
    # longer pauses at AwaitingUserDecision -- it proceeds to ModelingProblem exactly like in_scope
    # (judge by real results, not a premature scope verdict).
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_reinterpreting_spec(eng, sid)

    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "out_of_scope",
                "spec_interpretation_ref": f"spec_interpretation://{sid}/final",
                "revised_research_spec_ref": f"research_spec://{sid}/revised",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ModelingProblem"
