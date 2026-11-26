"""PlanningExperiment / ReviewingExperimentPlan isolated integration smoke
(M8 Task 6): single session, no Redis, no real LLM.

``PlanningExperiment`` is reached from ``ManagingHypothesisPortfolio``, which
is NOT wired yet (portfolio-selection / counter L is deferred) -- mirrors the
isolation-construction pattern in ``test_m5_hypothesis_smoke.py``. This smoke
seeds a HypothesisBank + VerifierSpec + BenchmarkSpec + ResearchSpec directly
into the Spec Store, constructs + saves a ``SessionState`` directly in
``PlanningExperiment``, then drives it through a REAL ``DeliveringBus``
(genuine subscribe/dispatch routing, not a manual handler call) so the full
cascade -- entry action -> ExperimentPlannerAgent -> orchestrator tick ->
transition -> entry action -> CriticAgent -> orchestrator tick -> transition
-- is exercised for real across three independently-built components:
``OrchestratorEngine``, the ExperimentPlannerAgent service, and the
CriticAgent service.

Drives, in one linear session:
  1. Seed a HypothesisBank (id "h1"), a VerifierSpec (whose editable_region
     contains the file(s) the planner's canned plan will touch), a
     BenchmarkSpec, and a ResearchSpec directly into the Spec Store, as
     upstream agents would have.
  2. Construct + save a ``SessionState`` directly in ``PlanningExperiment``
     with ``current_hypothesis_id="h1"``, ``experiment_plan_refs=[]``, and
     all four refs set (isolated construction -- no prior states are driven).
  3. ``eng.resume_to_current_state(session_id)`` re-emits the
     ``PlanningExperiment`` entry action, ``experiment.planning_requested``
     -> (delivered) -> ExperimentPlannerAgent resolves all four inputs from
     the store, selects hypothesis "h1", runs its (canned) ``plan_experiment``
     (changes[].file all within the VerifierSpec's editable_region),
     persists the ExperimentPlan, and publishes ``experiment.plan_ready`` ->
     (delivered) -> orchestrator appends the ref to ``experiment_plan_refs``
     and transitions PlanningExperiment -> ReviewingExperimentPlan, whose
     entry action (``experiment.review_requested``) -> (delivered) ->
     CriticAgent resolves the ExperimentPlan + VerifierSpec from the store,
     runs its (canned, verdict "pass") ``critique_experiment_plan``, persists
     the CritiqueReport, and publishes ``experiment.plan_reviewed`` (pass) ->
     (delivered) -> orchestrator resets MAP counter E
     (``experiment_plan_review_round["h1"]``, delete-key) and transitions
     ReviewingExperimentPlan -> ImplementingExperiment, which has no entry
     action (M9 CoderAgent not built), so the loop pauses there -- the
     mainline has now reached ImplementingExperiment with a frozen,
     in-scope ExperimentPlan on the books.

No concurrency, no Redis, no real LLM call anywhere in this file (two FakeLLM
canned queues, one per agent). ``resume_to_current_state`` is called exactly
once -- a second call would re-emit the entry action and break the "exactly
one" event counts.

The ``DeliveringBus``/``BusClientView`` pattern is copied from
``test_m7_verifier_smoke.py`` (itself copied from ``test_m6_critic_smoke.py``)
to keep this file self-contained.
"""

from __future__ import annotations

import fnmatch
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM as CriticFakeLLM
from autoresearch.agents.experiment_planner_agent.service import build_experiment_planner_service
from autoresearch.agents.experiment_planner_agent.tests.conftest import FakeLLM as PlannerFakeLLM
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
PLANNER_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "experiment_planner_agent" / "data-contracts"
)
EXPERIMENT_PLAN_SCHEMA_PATH = PLANNER_CONTRACTS_DIR / "experiment_plan.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes (mirrors M7).
# Copied from test_m7_verifier_smoke.py to keep this file self-contained.
# ---------------------------------------------------------------------------


class DeliveringBus:
    """Shared broker state: subscriptions, the real sqlite handle, and every
    published ``Envelope`` -- delivered synchronously to matching subscribers
    (mirrors ``Dispatcher.dispatch``'s sequential fnmatch routing)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        self.subscriptions: list[tuple[str, Any]] = []
        self.published: list[Envelope] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        emitter: str,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        env = Envelope(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            emitter=emitter,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            schema_version=1,
            cost_delta=cost_delta,
            payload=payload,
            causation_id=causation_id,
        )
        self.published.append(env)
        if auto_commit:
            await self.db.commit()
        for pattern, handler in list(self.subscriptions):
            if fnmatch.fnmatchcase(event_type, pattern):
                await handler(env)
        return env.event_id


class BusClientView:
    """One emitter's handle onto a shared ``DeliveringBus`` -- mirrors how
    each real process holds its own ``EventBusClient`` (its own
    ``emitter_name``) while every client shares one broker/outbox."""

    def __init__(self, bus: DeliveringBus, emitter_name: str) -> None:
        self._bus = bus
        self.emitter_name = emitter_name

    @property
    def db(self) -> aiosqlite.Connection:
        return self._bus.db

    def subscribe(self, pattern: str, handler: Any) -> None:
        self._bus.subscribe(pattern, handler)

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        return await self._bus.publish(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            emitter=self.emitter_name,
            cost_delta=cost_delta,
            causation_id=causation_id,
            auto_commit=auto_commit,
        )


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m8_planner_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Fixture data. The SAME fixed file list ("src/model.py") is used in the
# VerifierSpec's editable_region, the planner's canned ExperimentPlan
# changes[], and the critic's canned (pass) plan review -- a mismatch would
# make either the planner's scope guard or the critic's editable_region gate
# force a "fail", diverting the session away from ImplementingExperiment.
# ---------------------------------------------------------------------------

HYP_1 = {
    "id": "h1",
    "statement": "Adding a learned gating mechanism to the retriever improves long-tail recall.",
    "rationale": "Dense retrievers underperform on long-tail entities.",
    "kind": "training",
    "verifiable": True,
    "expected_effect": "exact_match improves on the long-tail subset.",
}

VALID_HYPOTHESIS_BANK = {"hypotheses": [HYP_1]}

VALID_VERIFIER_SPEC = {
    "evaluator_hash": "sha256:abc123",
    "metric": "exact_match",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py"],
    "forbidden_files": ["eval/harness.py"],
    "result_parse_schema": {"type": "object"},
}

VALID_BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.42,
    },
    "resource_requirements": "1x A100 GPU",
    "forbidden_files": ["eval/harness.py"],
}

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

# ExperimentPlannerAgent's canned response: a schema-valid ExperimentPlan
# whose sole changes[].file ("src/model.py") is within VALID_VERIFIER_SPEC's
# editable_region and NOT in its forbidden_files.
VALID_EXPERIMENT_PLAN = {
    "hypothesis_id": "h1",
    "change_summary": "Add a learned gating layer before retrieval fusion.",
    "changes": [
        {"file": "src/model.py", "description": "Add a gating module class."},
    ],
    "ablations": ["no_gating_baseline"],
    "expected_outcome": "Improved exact_match on the long-tail subset.",
    "resource_budget": "1x A100 GPU, 4h wall-clock",
    "success_criteria": "exact_match improves by >=2 points over baseline on the held-out split.",
}

# CriticAgent's canned response: a schema-valid CritiqueReport (target=
# experiment_plan, verdict pass) -- consistent with the plan above being
# fully within editable_region, so the editable_region gate does not
# override the LLM's verdict.
VALID_EXPERIMENT_PLAN_REVIEW_PASS = {
    "target": "experiment_plan",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}


def _plan_json() -> str:
    return json.dumps(VALID_EXPERIMENT_PLAN)


def _review_json() -> str:
    return json.dumps(VALID_EXPERIMENT_PLAN_REVIEW_PASS)


# ===========================================================================
# Smoke: isolated PlanningExperiment -> ExperimentPlannerAgent ->
# experiment.plan_ready -> ReviewingExperimentPlan -> CriticAgent ->
# experiment.plan_reviewed (pass) -> ImplementingExperiment (pauses; M9
# CoderAgent not built).
# ===========================================================================


async def test_planning_experiment_reaches_implementing_experiment_single_session(bus, store):
    # --- wire OrchestratorEngine + ExperimentPlannerAgent + CriticAgent --
    # all on the SAME bus, each with its own emitter identity.
    eng = OrchestratorEngine(client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT)
    await eng.start()

    planner_llm = PlannerFakeLLM([_plan_json()])
    build_experiment_planner_service(
        client=BusClientView(bus, "ExperimentPlannerAgent"),
        spec_store=store,
        llm_client=planner_llm,
    )

    critic_llm = CriticFakeLLM([_review_json()])
    build_critic_service(
        client=BusClientView(bus, "CriticAgent"),
        spec_store=store,
        llm_client=critic_llm,
    )

    # --- Isolated construction: PlanningExperiment is NOT reachable from
    # start_session yet (ManagingHypothesisPortfolio -> PlanningExperiment
    # portfolio-selection / counter L is deferred), so seed its four upstream
    # inputs directly into the store, as HypothesisAgent / EvaluatorAgent /
    # ProblemModelingAgent would have, then construct + save a SessionState
    # directly in PlanningExperiment with all four refs set.
    sid = str(uuid.uuid4())
    hypothesis_bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=sid,
        key="final",
        obj=VALID_HYPOTHESIS_BANK,
        emitter="HypothesisAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=sid,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=sid,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=sid,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="PlanningExperiment",
        current_hypothesis_id="h1",
        experiment_plan_refs=[],
        hypothesis_bank_ref=hypothesis_bank_ref,
        verifier_spec_ref=verifier_spec_ref,
        benchmark_spec_ref=benchmark_spec_ref,
        research_spec_ref=research_spec_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits the PlanningExperiment
    # entry action (experiment.planning_requested) -> (delivered) ->
    # ExperimentPlannerAgent resolves all four inputs from the store, selects
    # hypothesis "h1", runs its (canned) plan_experiment, persists the
    # ExperimentPlan, and publishes experiment.plan_ready -> (delivered) ->
    # orchestrator's tick appends the ref to experiment_plan_refs and
    # transitions PlanningExperiment -> ReviewingExperimentPlan, whose entry
    # action (experiment.review_requested) -> (delivered) -> CriticAgent
    # resolves the ExperimentPlan + VerifierSpec from the store, runs its
    # (canned, verdict "pass") critique_experiment_plan, persists the
    # CritiqueReport, and publishes experiment.plan_reviewed (pass) ->
    # (delivered) -> orchestrator resets MAP counter E
    # (experiment_plan_review_round["h1"], delete-key) and transitions
    # ReviewingExperimentPlan -> ImplementingExperiment (no entry action --
    # M9 CoderAgent not built -- so the loop pauses there). Called exactly
    # once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "ImplementingExperiment"

    # experiment_plan_refs has exactly the one ref appended by the
    # PlanningExperiment -> ReviewingExperimentPlan transition, and it
    # resolves to a schema-valid ExperimentPlan whose hypothesis_id == "h1"
    # and all changes[].file are within the VerifierSpec's editable_region.
    assert len(final_state.experiment_plan_refs) == 1
    experiment_plan = store.get(final_state.experiment_plan_refs[-1])
    plan_errors = list(_validator(EXPERIMENT_PLAN_SCHEMA_PATH).iter_errors(experiment_plan))
    assert plan_errors == [], [e.message for e in plan_errors]
    assert experiment_plan["hypothesis_id"] == "h1"

    editable_region = set(VALID_VERIFIER_SPEC["editable_region"])
    changed_files = {c["file"] for c in experiment_plan["changes"]}
    assert changed_files
    assert changed_files <= editable_region

    # MAP counter E (experiment_plan_review_round) reset = delete-key.
    assert "h1" not in final_state.experiment_plan_review_round

    # Exactly one experiment.plan_ready and one experiment.plan_reviewed(pass).
    plan_ready_events = [e for e in bus.published if e.event_type == "experiment.plan_ready"]
    assert len(plan_ready_events) == 1
    assert plan_ready_events[0].session_id == sid
    assert plan_ready_events[0].payload["experiment_plan_ref"] == final_state.experiment_plan_refs[-1]

    plan_reviewed_events = [e for e in bus.published if e.event_type == "experiment.plan_reviewed"]
    assert len(plan_reviewed_events) == 1
    assert plan_reviewed_events[0].session_id == sid
    assert plan_reviewed_events[0].payload["verdict"] == "pass"

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
