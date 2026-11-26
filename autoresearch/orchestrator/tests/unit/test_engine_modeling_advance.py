"""Unit tests for Task 4: ModelingProblem -> HumanIdeaCheckpoint -> {
CritiquingDirection | return_to target} orchestrator wiring.

Covers:
  - ``_on_ModelingProblem_problem_modeled``: stores ``problem_model_ref``, sets
    ``human_checkpoint_return_to = "CritiquingDirection"`` (mainline's return
    target), and transitions to HumanIdeaCheckpoint, which has no entry
    action -> pauses awaiting the human.
  - ``_on_HumanIdeaCheckpoint_human_idea_provided``: optionally stores
    ``human_idea_seed_ref`` (when ``inject: true``), then routes to
    ``state.human_checkpoint_return_to`` (falling back to
    "CritiquingDirection" only if unset) -- NOT a hardcoded
    "CritiquingDirection". This must honor BOTH entry paths into
    HumanIdeaCheckpoint:
      (1) the mainline from ModelingProblem (return_to="CritiquingDirection"),
      (2) the ReinterpretingSpec ``scope_changed`` branch (return_to="ModelingProblem"
          post-3.3, tested in test_engine_reinterpret_advance.py / mirrored here as the c2 case).

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)``), mirroring
test_engine_novelty_advance.py's pattern.
"""

# Imported pytest fixtures intentionally share their parameter names below.
# ruff: noqa: F811

from __future__ import annotations

import pytest

from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state, save_session_state
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


async def _advance_to_modeling_problem(eng: OrchestratorEngine, sid: str) -> tuple[str, str]:
    """Drive a fresh session (ClarifyingResearchGoal) into ModelingProblem.

    Returns (research_spec_ref, literature_brief_ref) -- both read off
    SessionState by ModelingProblem's entry action (already wired; not under
    test here).
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
    revised_research_spec_ref = f"research_spec://{sid}/revised"
    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "in_scope",
                "spec_interpretation_ref": f"spec_interpretation://{sid}/final",
                "revised_research_spec_ref": revised_research_spec_ref,
            },
        )
    )
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "pass",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/final",
            },
        )
    )
    return revised_research_spec_ref, literature_brief_ref


async def test_problem_modeled_advances_to_human_idea_checkpoint_and_pauses(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_modeling_problem(eng, sid)

    problem_model_ref = f"problem_model://{sid}/final"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "problem.modeled",
            sid,
            "ProblemModelingAgent",
            payload={"problem_model_ref": problem_model_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "HumanIdeaCheckpoint"
    assert st.problem_model_ref == problem_model_ref
    # Benchmark design starts only after the real repository baseline is frozen.
    assert st.human_checkpoint_return_to == "SelectingBenchmark"

    # No entry action wired for HumanIdeaCheckpoint -> pauses, no new event.
    assert fake_bus.published[n_before:] == []


# meeting 2026-07-25 item 7 (全删): the autonomous direction-critique loop is RETIRED. The machine
# no longer re-critiques the direction after the human confirmed it, so the old
# test_critique_loop_reentry_skips_the_checkpoint (round>0 -> CritiquingDirection) no longer applies
# and was removed. Direction gatekeeping now happens at the METHOD level (novelty + feasibility).


async def test_problem_modeled_transition_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_modeling_problem(eng, sid)

    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "problem.modeled",
            sid,
            "ProblemModelingAgent",
            payload={"problem_model_ref": f"problem_model://{sid}/final"},
        )
    )

    assert fake_bus.db.commit_count - commits_before == 1


async def test_human_idea_provided_no_inject_routes_to_baseline_selection(fake_bus):
    # A benchmark chooses and reproduces its baseline before initial Method design.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_modeling_problem(eng, sid)
    await eng.tick(
        _Evt(
            "problem.modeled",
            sid,
            "ProblemModelingAgent",
            payload={"problem_model_ref": f"problem_model://{sid}/final"},
        )
    )

    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "human.idea_provided",
            sid,
            "Human",
            payload={"inject": False},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "SelectingBenchmark"
    assert st.human_idea_seed_ref is None

    new_events = fake_bus.published[n_before:]
    benchmark_events = [p for p in new_events if p.event_type == "benchmark.selection_requested"]
    assert len(benchmark_events) == 1
    assert benchmark_events[0].session_id == sid

    assert fake_bus.db.commit_count - commits_before == 1


async def test_human_papers_checkpoint_without_evidence_replaces_exhausted_method(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = await load_session_state(fake_bus.db, sid)
    state.current_state = "HumanIdeaCheckpoint"
    state.human_checkpoint_return_to = "GeneratingMethod"
    state.method_ref = f"method://{sid}/v2"
    state.novelty_audit_report_ref = f"novelty_audit_report://{sid}/round6"
    state.novelty_targeted_search_round = 2
    state.method_swap_round = 1
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(
        _Evt("human.idea_provided", sid, "Human", payload={"inject": False})
    )

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "GeneratingMethod"
    assert state.method_not_working is True
    assert state.method_swap_round == 2


@pytest.mark.parametrize("execution_contract", [None, {"evaluation": {"required_values": {"/split": "dev"}}}])
async def test_human_idea_provided_stores_confirmed_repo_url(fake_bus, execution_contract):
    # the human picked one of the auto-discovered candidate repos at the checkpoint -> it is
    # remembered so the experiment clones+improves it (see runner routing).
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1", confirmed_boundary=(
        {"execution_contract": execution_contract} if execution_contract else {}))
    await _advance_to_modeling_problem(eng, sid)
    await eng.tick(_Evt("problem.modeled", sid, "ProblemModelingAgent",
                        payload={"problem_model_ref": f"problem_model://{sid}/final"}))

    await eng.tick(_Evt("human.idea_provided", sid, "Human",
                        payload={"inject": False,
                                 "confirmed_repo_url": "https://github.com/owner/baseline-repo",
                                 "confirmed_boundary": {"run_cmd": "python run.py -i data",
                                                        "budget_s": 1200}}))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.confirmed_repo_url == "https://github.com/owner/baseline-repo"
    assert st.confirmed_boundary == {"run_cmd": "python run.py -i data", "budget_s": 1200, **(
        {"execution_contract": execution_contract} if execution_contract else {})}
    assert st.current_state == "SelectingBenchmark"


async def test_non_benchmark_problem_still_generates_method_before_experiment(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("theory goal", user_id="u1")
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "idea_theory"
    state.current_state = "ModelingProblem"
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "problem.modeled", sid, "ProblemModelingAgent",
        payload={"problem_model_ref": f"problem_model://{sid}/v1"},
    ))
    updated = await load_session_state(fake_bus.db, sid)
    assert updated.human_checkpoint_return_to == "GeneratingMethod"


async def test_human_idea_provided_with_inject_sets_seed_ref(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_modeling_problem(eng, sid)
    await eng.tick(
        _Evt(
            "problem.modeled",
            sid,
            "ProblemModelingAgent",
            payload={"problem_model_ref": f"problem_model://{sid}/final"},
        )
    )

    human_idea_seed_ref = f"spec://human_idea_seed/{sid}/final"
    await eng.tick(
        _Evt(
            "human.idea_provided",
            sid,
            "Human",
            payload={"inject": True, "human_idea_seed_ref": human_idea_seed_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.human_idea_seed_ref == human_idea_seed_ref
    # meeting 3.4: an INJECTED idea re-derives the direction -> route back to ModelingProblem
    # (re-model WITH the injected idea), not straight forward to CritiquingDirection.
    assert st.current_state == "ModelingProblem"


async def test_human_idea_provided_honors_scope_changed_return_to(fake_bus):
    """The scope_changed entry path into HumanIdeaCheckpoint (from ReinterpretingSpec) sets
    return_to="ModelingProblem" (3.3: novelty audit moved to after the method); the
    human_idea_provided handler MUST honor that stored return_to -- not hardcode CritiquingDirection."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    research_spec_ref = f"spec://research_spec/{sid}/final"
    await eng.tick(
        _Evt(
            "research_spec.ready",
            sid,
            "ProblemModelingAgent",
            payload={"research_spec_ref": research_spec_ref},
        )
    )
    await eng.tick(
        _Evt(
            "literature.brief_ready",
            sid,
            "DeepResearchAgent",
            payload={
                "literature_brief_ref": f"spec://literature_brief/{sid}/final",
                "field_trajectory_brief_ref": f"spec://field_trajectory_brief/{sid}/final",
            },
        )
    )
    revised_research_spec_ref = f"research_spec://{sid}/revised"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "scope_changed",
                "spec_interpretation_ref": f"spec_interpretation://{sid}/final",
                "revised_research_spec_ref": revised_research_spec_ref,
            },
        )
    )

    # Meeting 2026-07-29: ONE checkpoint only (Layer A). scope_changed no longer pauses at its own
    # HumanIdeaCheckpoint here -- it flows straight to ModelingProblem, whose entry action re-requests
    # problem modeling; the single Layer-A checkpoint is ModelingProblem's own exit.
    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ModelingProblem"

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "problem_modeling.model_requested" for p in new_events)
    assert not any(p.event_type == "direction.critique_requested" for p in new_events)
