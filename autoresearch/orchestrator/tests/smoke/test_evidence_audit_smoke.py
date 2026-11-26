"""EvidenceAuditAgent milestone Task 6: two focused integration smokes,
each isolating ONE of the two sequential co-audit insertions the milestone
adds to the state machine -- single session, no Redis, no real LLM, no real
execution.

The grand end-to-end smoke (``test_m15_end_to_end_smoke.py``) already wires
EvidenceAuditAgent at both points (Tasks 4+5) and reaches ``Done`` in one
13-agent session; THIS file isolates each insertion on its own, mirroring the
``resume_to_current_state`` isolated-construction pattern used by
``test_m14_reporting_smoke.py`` (seed upstream refs directly into the store,
construct + save a ``SessionState`` already sitting in the target state, then
resume once).

1. ``test_auditing_literature_novelty_co_audit_reaches_confirming_method``:
   Mode A (``audit_literature_support``) co-audits alongside NoveltyAuditAgent
   at ``AuditingLiteratureNovelty`` -- EvidenceAuditAgent runs FIRST (entry
   action), then the orchestrator republishes ``novelty.audit_requested``
   WITH ``evidence_audit_report_ref`` attached so NoveltyAuditAgent's verdict
   is evidence-informed. Verdict pass -> ``ConfirmingMethod`` (3.3: the audit
   now judges the METHOD's novelty and a pass reaches the human method gate,
   which has no entry action so the session pauses there).
2. ``test_reporting_honest_report_gate_drops_unsupported_claim``: Mode C
   (``audit_citation_regression``) gates ``Reporting`` before ReportAgent
   runs -- the CitationAuditReport flags one of ResearchMemory's two
   "verified" findings as citation-unsupported; ReportAgent's canned response
   claims BOTH findings (both grounded in ResearchMemory, so the fabrication
   guard alone would accept them), but ``core.generate_report``'s
   post-processing drops the citation-unsupported one -- the final Report's
   ``verified_findings`` is the honest subset, not the LLM's raw claim.

No concurrency, no Redis, no real LLM call, no real execution backend
anywhere in this file (every LLM is a canned FakeLLM queue).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM as EvidenceAuditFakeLLM
from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM as NoveltyAuditFakeLLM
from autoresearch.agents.report_agent.service import build_report_service
from autoresearch.agents.report_agent.tests.conftest import FakeLLM as ReportFakeLLM
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"

EVIDENCE_AUDIT_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "evidence_audit_agent" / "data-contracts"
)
EVIDENCE_AUDIT_REPORT_SCHEMA_PATH = EVIDENCE_AUDIT_CONTRACTS_DIR / "evidence_audit_report.schema.yaml"
CITATION_AUDIT_REPORT_SCHEMA_PATH = EVIDENCE_AUDIT_CONTRACTS_DIR / "citation_audit_report.schema.yaml"

NOVELTY_AUDIT_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "novelty_audit_agent" / "data-contracts"
)
NOVELTY_AUDIT_REPORT_SCHEMA_PATH = NOVELTY_AUDIT_CONTRACTS_DIR / "novelty_audit_report.schema.yaml"

REPORT_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "report_agent" / "data-contracts"
REPORT_SCHEMA_PATH = REPORT_CONTRACTS_DIR / "report.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _json(obj: dict) -> str:
    return json.dumps(obj)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "evidence_audit_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


@pytest.fixture
def config_root(tmp_path):
    # A shared tmp config_root: no prompt_patch.md is written under it, so
    # ConfigLoader.prompt_patch() returns None -- proves EvidenceAuditAgent's
    # config_root plumbing (MOSS-patchable) is threaded through without
    # requiring an actual patch file to exist.
    return tmp_path / "config_root"


# ===========================================================================
# (1) AuditingLiteratureNovelty co-audit flow: EvidenceAuditAgent Mode A runs
# FIRST, then NoveltyAuditAgent (evidence-informed) -> ModelingProblem.
# ===========================================================================

RESEARCH_SPEC = {
    "topic": "modded-nanogpt speedrun optimization",
    "research_goal": (
        "Reduce modded-nanogpt training wall-clock time to a new speedrun "
        "record while preserving the target validation loss."
    ),
}

SEMINAL_PAPER = {
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "id_or_url": "arxiv:2005.11401",
    "year": 2020,
    "note": "Seminal RAG paper.",
}

LITERATURE_BRIEF = {
    "summary": "RAG combines dense retrieval with generation.",
    "key_papers": [SEMINAL_PAPER],
}

SPEC_INTERPRETATION = {
    "disposition": "in_scope",
    "real_open_problem": (
        "Speedrunning modded-nanogpt remains open given the RAG literature "
        "surfaced is orthogonal to the training setup."
    ),
    "scope_reason": "The original ResearchSpec still holds; no scope change needed.",
}

VALID_EVIDENCE_AUDIT_REPORT = {
    "claim_citation_matrix": [
        {
            "claim": "Reducing modded-nanogpt wall-clock time is achievable via optimizer config changes.",
            "citations": ["arxiv:2005.11401"],
            "support": "strong",
        }
    ],
    "missing_papers": [],
    "expired_citations": [],
    "evidence_strength": "strong",
    "notes": "Citations sufficiently support the ResearchSpec's claims.",
}

VALID_NOVELTY_AUDIT_REPORT_PASS = {
    "verdict": "pass",
    "citation_sufficiency": "Citations are sufficient and current for the proposed direction.",
    "stale_citations": [],
    "unsupported_claims": [],
    "missing_work": [],
    "novelty_status": "novel",
    "method_signature": {
        "primary_core": {
            "core_id": "core-1",
            "intervention": "Change the optimizer update.",
            "causal_mechanism": "Improve optimization efficiency.",
            "pipeline_role": "training update",
            "expected_effect": "Reach the target metric more efficiently.",
        },
        "supporting_components": [],
        "composition_claim": {"material": False, "mechanism": ""},
    },
    "prior_work_comparisons": [
        {
            "prior_work_id": "arxiv:2005.11401",
            "primary_core_relation": "distinct",
            "difference_scope": "material_core",
            "composition_relation": "not_applicable",
            "supporting_overlap": [],
            "evidence": ["The cited work studies retrieval augmentation, not optimizer updates."],
        }
    ],
    "retrieval_coverage": {
        "covered_core_ids": ["core-1"],
        "compared_prior_work_ids": ["arxiv:2005.11401"],
        "unresolved_candidates": [],
    },
    "harness_decision": {
        "novelty_status": "novel",
        "verdict": "pass",
        "decisive_prior_work_ids": [],
        "rationale": "All grounded closest-work comparisons are distinct at the primary causal mechanism.",
    },
    "notes": "No critical gaps found.",
}


async def test_auditing_literature_novelty_co_audit_reaches_confirming_method(bus, store, config_root):
    # --- wire OrchestratorEngine (WITH spec_store -- ModelingProblem's entry
    # action, when it fires, would read frozen specs independently) +
    # EvidenceAuditAgent + NoveltyAuditAgent. Deliberately NO
    # ProblemModelingAgent -- ModelingProblem's entry action fires and
    # nothing responds, so the session pauses there (proving the co-audit
    # insertion in isolation, without pulling in the rest of the loop).
    eng = OrchestratorEngine(client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store)
    await eng.start()

    evidence_audit_llm = EvidenceAuditFakeLLM([_json(VALID_EVIDENCE_AUDIT_REPORT)])
    build_evidence_audit_service(
        client=bus.view("EvidenceAuditAgent"),
        spec_store=store,
        llm_client=evidence_audit_llm,
        config_root=config_root,
    )

    novelty_audit_llm = NoveltyAuditFakeLLM([_json(VALID_NOVELTY_AUDIT_REPORT_PASS)])
    build_novelty_audit_service(
        client=bus.view("NoveltyAuditAgent"),
        spec_store=store,
        llm_client=novelty_audit_llm,
    )

    # --- Isolated construction: seed the three refs AuditingLiteratureNovelty's
    # entry action reads directly into the store, as ProblemModelingAgent /
    # DeepResearchAgent would have upstream, then construct + save a
    # SessionState already sitting in AuditingLiteratureNovelty.
    sid = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec", session_id=sid, key="v1", obj=RESEARCH_SPEC, emitter="ProblemModelingAgent"
    )
    literature_brief_ref = store.put(
        kind="literature_brief", session_id=sid, key="v1", obj=LITERATURE_BRIEF, emitter="DeepResearchAgent"
    )
    spec_interpretation_ref = store.put(
        kind="spec_interpretation",
        session_id=sid,
        key="v1",
        obj=SPEC_INTERPRETATION,
        emitter="ProblemModelingAgent",
    )
    # 3.3: the audit now judges the METHOD's novelty, so seed a frozen method too.
    method_ref = store.put(
        kind="method", session_id=sid, key="v1",
        obj={"approach": "a", "changes_outline": ["c"], "why_it_should_win": "w", "assumptions": []},
        emitter="MethodAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AuditingLiteratureNovelty",
        research_spec_ref=research_spec_ref,
        literature_brief_ref=literature_brief_ref,
        spec_interpretation_ref=spec_interpretation_ref,
        method_ref=method_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits AuditingLiteratureNovelty's
    # entry action (evidence.audit_requested) and cascades: EvidenceAuditAgent
    # (Mode A) -> evidence.audited -> orchestrator republishes
    # novelty.audit_requested (now WITH evidence_audit_report_ref) ->
    # NoveltyAuditAgent (evidence-informed, canned pass) -> novelty.audit_ready
    # -> orchestrator transitions AuditingLiteratureNovelty -> ModelingProblem,
    # emits problem_modeling.model_requested -- no subscriber, loop pauses.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None

    # THE HEADLINE PROOF: the co-audit cascade transitioned all the way to ConfirmingMethod (3.3:
    # a passed method-novelty audit reaches the human method gate), then paused.
    assert final_state.current_state == "ConfirmingMethod"

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # evidence_audit_report_ref set + resolves to a schema-valid
    # EvidenceAuditReport.
    assert final_state.evidence_audit_report_ref is not None
    evidence_report = store.get(final_state.evidence_audit_report_ref)
    assert evidence_report == VALID_EVIDENCE_AUDIT_REPORT
    evidence_report_errors = list(_validator(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH).iter_errors(evidence_report))
    assert evidence_report_errors == [], [e.message for e in evidence_report_errors]

    # novelty_audit_report_ref set + resolves to a schema-valid
    # NoveltyAuditReport with verdict pass.
    assert final_state.novelty_audit_report_ref is not None
    novelty_report = store.get(final_state.novelty_audit_report_ref)
    assert novelty_report == VALID_NOVELTY_AUDIT_REPORT_PASS
    novelty_report_errors = list(_validator(NOVELTY_AUDIT_REPORT_SCHEMA_PATH).iter_errors(novelty_report))
    assert novelty_report_errors == [], [e.message for e in novelty_report_errors]

    # --- Event-order proof: exactly one of each, evidence.audit_requested ->
    # evidence.audited -> novelty.audit_requested (carrying
    # evidence_audit_report_ref) -> novelty.audit_ready(pass).
    for event_type, expected_count in (
        ("evidence.audit_requested", 1),
        ("evidence.audited", 1),
        ("novelty.audit_requested", 1),
        ("novelty.audit_ready", 1),
    ):
        matching = [e for e in bus.published if e.event_type == event_type]
        assert len(matching) == expected_count, (event_type, [e.payload for e in matching])
    # 3.3: a pass now routes to ConfirmingMethod (a human gate, no entry action), so ModelingProblem's
    # entry event is NOT emitted here anymore.
    assert not [e for e in bus.published if e.event_type == "problem_modeling.model_requested"]

    novelty_requested = [e for e in bus.published if e.event_type == "novelty.audit_requested"][0]
    assert novelty_requested.payload["evidence_audit_report_ref"] == final_state.evidence_audit_report_ref

    audit_ready = [e for e in bus.published if e.event_type == "novelty.audit_ready"][0]
    assert audit_ready.payload["verdict"] == "pass"


# ===========================================================================
# (2) Reporting honest-report gate flow: EvidenceAuditAgent Mode C gates
# Reporting before ReportAgent runs -- an unsupported claim it flags is
# dropped from the final Report even though ReportAgent's own LLM claimed it
# and it was grounded in ResearchMemory.
# ===========================================================================

VALID_RESEARCH_SPEC = {
    "topic": "long-tail retrieval fusion",
    "research_goal": "Does a learned gating layer improve long-tail retrieval fusion?",
}

# Two "verified" findings in ResearchMemory -- both would pass ReportAgent's
# fabrication guard (core.generate_report checks report claims are a subset
# of ResearchMemory's), but only ONE survives Mode C's citation check.
SUPPORTED_FINDING = "finding-X"
UNSUPPORTED_FINDING = "finding-Y"

RESEARCH_MEMORY = {
    "verified_findings": [SUPPORTED_FINDING, UNSUPPORTED_FINDING],
    "negative_results": [],
    "consolidated_insights": ["Learned gating before fusion helps long-tail retrieval."],
    "divergence_seeds": ["Try gating conditioned on retrieval confidence."],
}

# EvidenceAuditAgent's Mode C canned response: flags UNSUPPORTED_FINDING as
# lacking citation support -- the honest-report gate's headline input.
VALID_CITATION_AUDIT_REPORT_WITH_UNSUPPORTED = {
    "unsupported_claims": [
        {"claim": UNSUPPORTED_FINDING, "why": "No citation in the frozen ResearchSpec supports this claim."}
    ],
    "checked_count": 2,
}

# ReportAgent's canned response claims BOTH findings as verified -- both are
# grounded in ResearchMemory (the fabrication guard alone would accept them),
# so the citation-audit drop is the ONLY thing that removes UNSUPPORTED_FINDING.
REPORT_LLM_RESPONSE = {
    "summary": "Gating layer improves long-tail retrieval fusion over the baseline.",
    "verified_findings": [SUPPORTED_FINDING, UNSUPPORTED_FINDING],
    "negative_results": [],
    "unverified_ideas": ["Confidence-conditioned gating might help further."],
    "experiment_not_run": ["Ablation on gating placement."],
    "replication_failed": [],
    "external_data_incomplete": [],
    "analysis_caveats": ["Only one seed was tried."],
    "known_risks": ["Overfitting to the held-out split."],
    "experiment_ledger": [
        {
            "experiment_id": "exp1",
            "metric_value": 0.90,
            "verdict": "new_best",
            "notes": "Gating layer applied.",
        }
    ],
}


async def test_reporting_honest_report_gate_drops_unsupported_claim(bus, store, config_root):
    # --- wire OrchestratorEngine (WITH spec_store) + EvidenceAuditAgent +
    # ReportAgent -- all on the SAME bus, each with its own emitter identity.
    eng = OrchestratorEngine(client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store)
    await eng.start()

    evidence_audit_llm = EvidenceAuditFakeLLM([_json(VALID_CITATION_AUDIT_REPORT_WITH_UNSUPPORTED)])
    build_evidence_audit_service(
        client=bus.view("EvidenceAuditAgent"),
        spec_store=store,
        llm_client=evidence_audit_llm,
        config_root=config_root,
    )

    report_llm = ReportFakeLLM([
        _json(REPORT_LLM_RESPONSE),
        _json({"verdict": "CLEAN", "findings": []}),
        "# Research report\n\n" + ("This paragraph reports only evidence grounded in the frozen research memory. " * 20),
    ])
    build_report_service(
        client=bus.view("ReportAgent"),
        spec_store=store,
        llm_client=report_llm,
    )

    # --- Isolated construction: seed research_spec_ref AND a real
    # research_spec artifact in the store (the Task-5 gap: Reporting's entry
    # action / Mode C's audit_citation_regression handler requires it, not
    # just a truthy ref) + research_memory_ref, then construct + save a
    # SessionState already sitting in Reporting.
    sid = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec", session_id=sid, key="v1", obj=VALID_RESEARCH_SPEC, emitter="ProblemModelingAgent"
    )
    research_memory_ref = store.put(
        kind="research_memory", session_id=sid, key="v1", obj=RESEARCH_MEMORY, emitter="MemoryAgent"
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="Reporting",
        research_spec_ref=research_spec_ref,
        research_memory_ref=research_memory_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits Reporting's entry action
    # (evidence.citation_audit_requested) and cascades: EvidenceAuditAgent
    # (Mode C) -> evidence.citation_audited -> orchestrator republishes
    # report.requested (now WITH citation_audit_report_ref) -> ReportAgent ->
    # report.ready -> Done.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None

    # THE HEADLINE PROOF: the citation-gate cascade reached the terminal Done.
    assert final_state.current_state == "Done"

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # citation_audit_report_ref set + resolves to a schema-valid
    # CitationAuditReport with >=1 unsupported claim.
    assert final_state.citation_audit_report_ref is not None
    citation_audit_report = store.get(final_state.citation_audit_report_ref)
    assert citation_audit_report == VALID_CITATION_AUDIT_REPORT_WITH_UNSUPPORTED
    citation_audit_errors = list(
        _validator(CITATION_AUDIT_REPORT_SCHEMA_PATH).iter_errors(citation_audit_report)
    )
    assert citation_audit_errors == [], [e.message for e in citation_audit_errors]
    assert len(citation_audit_report["unsupported_claims"]) >= 1

    # final_report_ref set + resolves to a schema-valid Report.
    assert final_state.final_report_ref is not None
    report = store.get(final_state.final_report_ref)
    report_errors = list(_validator(REPORT_SCHEMA_PATH).iter_errors(report))
    assert report_errors == [], [e.message for e in report_errors]

    # THE HONEST-REPORT PROOF: ReportAgent's own LLM claimed BOTH findings as
    # verified (grounded in ResearchMemory, so the fabrication guard alone
    # would have accepted them both), but the Mode C citation gate drops the
    # unsupported one -- the final Report keeps only the supported finding.
    assert report["verified_findings"] == [SUPPORTED_FINDING]
    assert UNSUPPORTED_FINDING not in report["verified_findings"]

    # --- Event-order proof: exactly one of each,
    # evidence.citation_audit_requested -> evidence.citation_audited ->
    # report.requested (carrying citation_audit_report_ref) -> report.ready.
    for event_type, expected_count in (
        ("evidence.citation_audit_requested", 1),
        ("evidence.citation_audited", 1),
        ("report.requested", 1),
        ("report.ready", 1),
    ):
        matching = [e for e in bus.published if e.event_type == event_type]
        assert len(matching) == expected_count, (event_type, [e.payload for e in matching])

    report_requested = [e for e in bus.published if e.event_type == "report.requested"][0]
    assert report_requested.payload["citation_audit_report_ref"] == final_state.citation_audit_report_ref

    report_ready = [e for e in bus.published if e.event_type == "report.ready"][0]
    assert report_ready.payload["report_ref"] == final_state.final_report_ref
