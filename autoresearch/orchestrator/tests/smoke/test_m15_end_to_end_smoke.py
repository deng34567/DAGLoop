"""M15 Task 2 integration smoke: THE grand end-to-end smoke -- ONE session,
ALL 13 agents, from ``start_session`` to the terminal ``Done``.

Stitches the M10 front half (``test_m10_mainline_to_portfolio_smoke.py``:
start_session -> ... -> ManagingHypothesisPortfolio) through the
RankingJudgeAgent milestone's ranking-driven connector
(``ManagingHypothesisPortfolio -> PlanningExperiment``, Task 5 of that
milestone, replacing the M15 Task 1 ``_advance_managing_hypothesis_portfolio``)
and the M14 back half (``test_m14_reporting_smoke.py``: ImplementingExperiment
-> ... -> Done), adding ExperimentPlannerAgent (M8) and the CriticAgent's
third (experiment-plan review) canned response so the ONE shared CriticAgent
FakeLLM now serves THREE calls in order: direction critique, hypotheses
critique, experiment-plan review.

THE HEADLINE: this is the FIRST single-session drive of the FULL research
loop -- spec -> literature -> evidence audit -> novelty -> problem model ->
direction critique -> benchmark -> verifier -> baseline -> hypothesis
generation -> hypothesis critique -> RankingJudgeAgent portfolio ranking
(selects the top-BTL-ranked hypothesis, "h1") -> experiment planning -> plan
review -> code -> run -> evaluate -> verify -> memory consolidation ->
citation-regression audit -> report -- reaching the terminal ``Done``, in ONE
session, with ALL 14 agents wired (the 13 from before this milestone plus
RankingJudgeAgent).

No concurrency, no Redis, no real LLM call, no real search backend and no
real execution backend anywhere in this file (every LLM is a canned FakeLLM
queue; FakeBackend canned papers; FakeRunnerBackend canned metrics; human
answers/idea-checkpoint response are simulated, not a real UI/human).
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.deep_research_backend import FakeBackend
from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.coder_agent.service import build_coder_service
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM as CoderFakeLLM
from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM as CriticFakeLLM
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deliberation_agent.service import build_deliberation_service
from autoresearch.agents.deliberation_agent.tests.conftest import FakeLLM as DeliberationFakeLLM
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM as DeepResearchFakeLLM
from autoresearch.agents.evaluator_agent.service import build_evaluator_service
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM as EvaluatorFakeLLM
from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM as EvidenceAuditFakeLLM
from autoresearch.agents.experiment_planner_agent.service import build_experiment_planner_service
from autoresearch.agents.experiment_planner_agent.tests.conftest import FakeLLM as ExperimentPlannerFakeLLM
from autoresearch.agents.hypothesis_agent.service import build_hypothesis_service
from autoresearch.agents.hypothesis_agent.tests.conftest import FakeLLM as HypothesisFakeLLM
from autoresearch.agents.knowledge_index_agent.service import build_knowledge_index_service
from autoresearch.agents.memory_agent.service import build_memory_service
from autoresearch.agents.memory_agent.tests.conftest import FakeLLM as MemoryFakeLLM
from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM as NoveltyAuditFakeLLM
from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM as ProblemModelingFakeLLM
from autoresearch.agents.ranking_judge_agent.service import build_ranking_judge_service
from autoresearch.agents.ranking_judge_agent.tests.conftest import FakeLLM as RankingJudgeFakeLLM
from autoresearch.agents.report_agent.service import build_report_service
from autoresearch.agents.report_agent.tests.conftest import FakeLLM as ReportFakeLLM
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"

EVALUATOR_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "evaluator_agent" / "data-contracts"
)
BENCHMARK_SPEC_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "benchmark_spec.schema.yaml"
VERIFIER_SPEC_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "verifier_spec.schema.yaml"
EVALUATION_REPORT_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "evaluation_report.schema.yaml"
VERIFICATION_REPORT_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "verification_report.schema.yaml"

RUNNER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "runner_agent" / "data-contracts"
BASELINE_RESULT_SCHEMA_PATH = RUNNER_CONTRACTS_DIR / "baseline_result.schema.yaml"
RUN_RECORD_SCHEMA_PATH = RUNNER_CONTRACTS_DIR / "run_record.schema.yaml"

HYPOTHESIS_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "hypothesis_agent" / "data-contracts"
)
HYPOTHESIS_BANK_SCHEMA_PATH = HYPOTHESIS_CONTRACTS_DIR / "hypothesis_bank.schema.yaml"

EXPERIMENT_PLANNER_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "experiment_planner_agent" / "data-contracts"
)
EXPERIMENT_PLAN_SCHEMA_PATH = EXPERIMENT_PLANNER_CONTRACTS_DIR / "experiment_plan.schema.yaml"

CODER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "coder_agent" / "data-contracts"
EXPERIMENT_PATCH_SCHEMA_PATH = CODER_CONTRACTS_DIR / "experiment_patch.schema.yaml"

MEMORY_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "memory_agent" / "data-contracts"
RESEARCH_MEMORY_SCHEMA_PATH = MEMORY_CONTRACTS_DIR / "research_memory.schema.yaml"

REPORT_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "report_agent" / "data-contracts"
REPORT_SCHEMA_PATH = REPORT_CONTRACTS_DIR / "report.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m15_end_to_end_smoke.db"))
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
# Fixture data. Front half (spec -> ... -> baseline reproduced -> hypothesis
# generated/critiqued) mirrors test_m10_mainline_to_portfolio_smoke.py
# verbatim; back half (plan -> code -> run -> evaluate -> verify -> memory ->
# report) mirrors test_m14_reporting_smoke.py verbatim, with the
# VerifierSpec's editable_region ("src/model.py", "src/train.py") -- set once
# by EvaluatorAgent's prepare_verifier, in PreparingVerifier -- reused
# downstream by ExperimentPlannerAgent's changes and CoderAgent's patch
# (both restricted to "src/model.py", a subset).
# ---------------------------------------------------------------------------

FINAL_SPEC = {
    "topic": "modded-nanogpt speedrun optimization",
    "research_goal": (
        "Reduce modded-nanogpt training wall-clock time to a new speedrun "
        "record while preserving the target validation loss."
    ),
    "task_mode": "benchmark",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["training script", "optimizer config"],
    "forbidden_items": ["changing the model architecture"],
    "open_questions": [],
}

SEMINAL_PAPER = {
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "id_or_url": "arxiv:2005.11401",
    "year": 2020,
    "note": "Seminal RAG paper.",
}
FRONTIER_PAPER = {
    "title": "Long-Tail Scientific QA via Retrieval-Augmented Fine-Tuning",
    "id_or_url": "arxiv:2401.00000",
    "year": 2024,
    "note": "Recent frontier work on long-tail scientific QA.",
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

# EvidenceAuditAgent's SECOND canned response, consumed by Mode C (the
# citation-regression audit Reporting's entry action gates on -- EvidenceAuditAgent
# milestone Task 5): no unsupported claims, so ReportAgent's honest report
# (below, KNOWN_VERIFIED_FINDING) is unaffected.
VALID_CITATION_AUDIT_REPORT = {
    "unsupported_claims": [],
    "checked_count": 1,
}

VALID_PROBLEM_MODEL = {
    "key_problems": ["Reducing modded-nanogpt wall-clock time without regressing validation loss."],
    "opportunities": ["Reuse RAG-adjacent retrieval tricks are out of scope; focus on optimizer config."],
    "risks": ["Overfitting the speedrun to the benchmark's specific hardware."],
    "verifiable_paths": ["Re-run the benchmark suite and compare wall-clock + validation loss to baseline."],
}

VALID_DIRECTION_REPORT_PASS = {
    "target": "direction",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}

VALID_BENCHMARK_SPEC = {
    "benchmark_id": "modded-nanogpt-speedrun-v1",
    "name": "modded-nanogpt speedrun",
    "metric": "wall_clock_seconds",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.85,
    },
    "resource_requirements": "8x A100 GPU",
    "forbidden_files": ["eval/scorer.py", "data/test/"],
}

VALID_VERIFIER_SPEC = {
    "evaluator_hash": "sha256:def456",
    "metric": "wall_clock_seconds",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py", "src/train.py"],
    "forbidden_files": ["eval/scorer.py", "data/test/"],
    "result_parse_schema": {"type": "object"},
}

VALID_HYPOTHESIS_BANK = {
    "hypotheses": [
        {
            "id": "h1",
            "statement": "Fusing the optimizer's momentum buffers reduces wall-clock time.",
            "rationale": "Momentum buffer fusion cuts kernel launch overhead on this hardware.",
            "kind": "code_optimization",
            "verifiable": True,
            "expected_effect": "wall_clock_seconds improves by an estimated 5-10%.",
        },
        {
            "id": "h2",
            "statement": "Switching to a larger batch size alone reduces wall-clock time.",
            "rationale": "Larger batches amortize fixed per-step overhead.",
            "kind": "training",
            "verifiable": True,
            "expected_effect": "wall_clock_seconds improves by an estimated 2-5%.",
        },
    ]
}

VALID_HYPOTHESES_REPORT_SOME_PASS = {
    "target": "hypotheses",
    "verdict": "some_pass",
    "issues": ["h2 is too narrow / trivially confounded with batch-size-only changes."],
    "passing_hypothesis_ids": ["h1"],
}

# RankingJudgeAgent's canned pairwise-judgment response for the two-hypothesis
# bank above (h1 beats h2); needed to even reach CriticAgent now that
# CritiquingHypotheses' entry action is ranking.requested (RankingJudgeAgent
# milestone Task 4).
VALID_RANKING_PAIRWISE_RESPONSE = {
    "pairwise_results": [
        {"a": "h1", "b": "h2", "winner": "h1", "rationale": "h1 is more falsifiable."},
    ],
    "diversity_status": "diverse",
}

# ExperimentPlannerAgent's canned response: changes[].file is a SUBSET of
# VALID_VERIFIER_SPEC's editable_region, restricted to "src/model.py" only so
# the same file set threads consistently through CoderAgent's patch below.
VALID_EXPERIMENT_PLAN = {
    "hypothesis_id": "h1",
    "change_summary": "standardize-inputs: fuse the optimizer's momentum buffers to cut kernel launch overhead.",
    "changes": [
        {"file": "src/model.py", "description": "Fuse momentum buffers in the optimizer step."},
    ],
    "ablations": ["no_fusion_baseline"],
    "expected_outcome": "wall_clock_seconds improves by 5-10% over baseline.",
    "resource_budget": "8x A100 GPU, 2h wall-clock",
    "success_criteria": "wall_clock_seconds improves by >=5% over baseline on the held-out split.",
}

# CriticAgent's THIRD canned response (experiment-plan review): verdict pass.
VALID_EXPERIMENT_PLAN_REPORT_PASS = {
    "target": "experiment_plan",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}

# CoderAgent's canned response: touches only "src/model.py" -- within
# VALID_VERIFIER_SPEC's editable_region and disjoint from forbidden_files.
VALID_EXPERIMENT_PATCH = {
    "hypothesis_id": "h1",
    "summary": "Fuse the optimizer's momentum buffers to cut kernel launch overhead.",
    "changes": [
        {"file": "src/model.py", "patch": "+def fused_momentum_step(...):\n+    ..."},
    ],
    "touched_files": ["src/model.py"],
}

# EvaluatorAgent's THIRD/FOURTH canned responses: evaluate_experiment
# (new_best) then verify_best_result (verified). The RunnerAgent's
# run_experiment metric (0.90, below) beats VALID_BENCHMARK_SPEC's
# baseline.reported_metric (0.85), so core.evaluate_experiment's
# deterministic anti-reward-hacking guard computes is_improvement=True,
# making the canned "new_best" verdict consistent.
EVALUATE_EXPERIMENT_LLM_RESPONSE = {
    "verdict": "new_best",
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": "0.90 beats the baseline 0.85.",
}

COMPARISON_GUARDIAN_PASS = {
    "verdict": "pass",
    "issues": [],
    "remediation": None,
    "reasoning": "The canonical benchmark comparison was preserved.",
}

VERIFY_BEST_RESULT_LLM_RESPONSE = {
    "verdict": "verified",
    "checks": ["cross-seed: fixed seed=42", "held-out: metric on held-out split"],
    "anti_cheat_flags": [],
    "reasoning": "Result reproduces under the frozen protocol.",
}

# The finding that BOTH MemoryAgent's canned ResearchMemory AND ReportAgent's
# canned Report claim as verified -- SAME STRING on both sides.
KNOWN_VERIFIED_FINDING = "finding-X"

# MemoryAgent's single canned LLM response: routing_verdict MUST be "report"
# for ConsolidatingMemory to route to Reporting (not back to
# GeneratingHypotheses / PlanningExperiment).
CONSOLIDATE_LLM_RESPONSE = {
    "research_memory": {
        "verified_findings": [KNOWN_VERIFIED_FINDING],
        "negative_results": [],
        "excluded_hypotheses": [],
        "consolidated_insights": ["Momentum buffer fusion cuts kernel launch overhead."],
        "divergence_seeds": ["Try fusion combined with a larger batch size."],
        "next_steps": [],
    },
    "routing_verdict": "report",
}

# ReportAgent's single canned LLM response: verified_findings ==
# [KNOWN_VERIFIED_FINDING], a SUBSET of (here, identical to) MemoryAgent's
# ResearchMemory.verified_findings -- the grounding guard holds.
REPORT_LLM_RESPONSE = {
    "summary": "Momentum buffer fusion improves wall_clock_seconds by 0.05 over the baseline.",
    "verified_findings": [KNOWN_VERIFIED_FINDING],
    "negative_results": [],
    "unverified_ideas": ["Fusion combined with a larger batch size might help further."],
    "experiment_not_run": ["Ablation on fusion granularity."],
    "replication_failed": [],
    "external_data_incomplete": [],
    "analysis_caveats": ["Only one seed was tried."],
    "known_risks": ["Overfitting the speedrun to this hardware."],
    "experiment_ledger": [
        {
            "experiment_id": "h1",
            "metric_value": 0.90,
            "verdict": "new_best",
            "notes": "Momentum buffer fusion applied; verified.",
        }
    ],
}


def _ask_json() -> str:
    return json.dumps(
        {
            "action": "ask",
            "question": {
                "question_id": "q-1",
                "question_text": "What is the primary deliverable form you expect?",
                "options": ["paper draft", "code repo", "benchmark report"],
                "allow_free_text": True,
            },
        }
    )


def _finalize_json(spec: dict) -> str:
    return json.dumps({"action": "finalize", "spec": spec})


def _reinterpret_json(revised_spec: dict) -> str:
    return json.dumps(
        {
            "spec_interpretation": {
                "naive_assumptions": [],
                "real_open_problem": (
                    "Speedrunning modded-nanogpt remains open given the RAG "
                    "literature surfaced is orthogonal to the training setup."
                ),
                "topic_position": "Unaffected by the retrieved RAG literature.",
                "disposition": "in_scope",
                "scope_reason": "The original ResearchSpec still holds; no scope change needed.",
                "evaluation_semantics": {
                    "comparison_statement": "Under the same canonical benchmark data, evaluator and resources, compare exact_match while allowing method implementation changes.",
                    "research_question": "Does the method improve exact_match under the canonical benchmark?",
                    "primary_metric": {"name": "exact_match", "direction": "maximize", "measurement": "Read exact_match from the canonical evaluator."},
                    "comparison_mode": "repository_defined",
                    "comparison_basis": "Use the selected repository benchmark.",
                    "budget_interpretation": "Use the benchmark's documented budget.",
                    "conditions_to_hold_constant": ["benchmark data, evaluator and resources"],
                    "method_freedom": ["method implementation"],
                    "data_source": "Canonical benchmark data.",
                    "evaluator_source": "Canonical benchmark evaluator.",
                    "observed_outcomes": ["runtime"],
                    "evaluation_notes": [],
                    "outcome_criteria": {"success": "Valid method exact_match improves.", "failure": "Execution fails or exact_match does not improve.", "inconclusive": "No valid comparable exact_match is produced."},
                    "evidence": [{"source_type": "research_spec", "source_id": "research_spec", "supports": "The research goal asks for benchmark improvement."}],
                    "uncertainties": [],
                },
            },
            "revised_research_spec": revised_spec,
        }
    )


def _synthesis_json() -> str:
    return json.dumps(
        {
            "literature_brief": {
                "summary": "RAG combines dense retrieval with generation.",
                "related_work": [],
                "key_papers": [SEMINAL_PAPER, FRONTIER_PAPER],
                "existing_benchmarks": [],
            },
            "field_trajectory_brief": {
                "origin": "RAG originated from combining retrieval with seq2seq generation.",
                "current_state": "Frontier work extends RAG to long-tail scientific QA.",
                "main_method_families": ["dense retrieval", "hybrid retrieval"],
                "open_problems": ["long-tail factual accuracy"],
            },
        }
    )


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ===========================================================================
# THE grand smoke: start_session -> ClarifyingResearchGoal -> ... ->
# ReproducingBaseline -> GeneratingHypotheses -> CritiquingHypotheses ->
# ManagingHypothesisPortfolio -> (M15 connector) -> PlanningExperiment ->
# ReviewingExperimentPlan -> ImplementingExperiment -> RunningExperiment ->
# EvaluatingExperiment -> VerifyingBestResult -> ConsolidatingMemory ->
# Reporting -> Done.
# ===========================================================================


async def test_start_session_reaches_done_single_session_all_agents(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- the M9 gate + M12/M13/M14
    # entry actions read frozen specs independently of the agents' payloads)
    # + all 13 agents -- all on the SAME bus, each with its own emitter identity.
    eng = OrchestratorEngine(client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store)
    await eng.start()

    # ONE FakeLLM handles all four ProblemModelingAgent LLM calls in order:
    # clarify-ask, clarify-finalize, reinterpret-in_scope, model_step.
    problem_modeling_llm = ProblemModelingFakeLLM(
        [_ask_json(), _finalize_json(FINAL_SPEC), _reinterpret_json(FINAL_SPEC), _json(VALID_PROBLEM_MODEL)]
    )
    build_problem_modeling_service(
        client=bus.view("ProblemModelingAgent"),
        spec_store=store,
        llm_client=problem_modeling_llm,
    )

    deep_research_backend = FakeBackend()
    deep_research_llm = DeepResearchFakeLLM([
        json.dumps({"query": "literature search keywords"}),
        _synthesis_json(),
        json.dumps({"query": "closest prior method keywords", "method_primary_core":
                    VALID_NOVELTY_AUDIT_REPORT_PASS["method_signature"]["primary_core"]}),
        _synthesis_json(),
    ])
    build_deliberation_service(
        client=bus.view("DeliberationAgent"),
        spec_store=store,
        llm_client=DeliberationFakeLLM(),
    )
    build_deep_research_service(
        client=bus.view("DeepResearchAgent"),
        spec_store=store,
        backend=deep_research_backend,
        llm_client=deep_research_llm,
    )

    # Optional: KnowledgeIndexAgent auto-ingests on literature.brief_ready with
    # NO LLM (real KeywordBackend -- local, deterministic lexical ranking, no
    # network); wired here since it's convenient and doesn't block the flow.
    build_knowledge_index_service(
        client=bus.view("KnowledgeIndexAgent"),
        spec_store=store,
    )

    # ONE shared EvidenceAuditAgent FakeLLM serves TWO calls in order: Mode A
    # co-audit at AuditingLiteratureNovelty (first, chronologically), then
    # Mode C citation-regression audit at Reporting's entry action (second,
    # much later in the session -- EvidenceAuditAgent milestone Task 5).
    evidence_audit_llm = EvidenceAuditFakeLLM(
        [_json(VALID_EVIDENCE_AUDIT_REPORT), _json(VALID_CITATION_AUDIT_REPORT)]
    )
    build_evidence_audit_service(
        client=bus.view("EvidenceAuditAgent"),
        spec_store=store,
        llm_client=evidence_audit_llm,
    )

    novelty_audit_llm = NoveltyAuditFakeLLM([_json(VALID_NOVELTY_AUDIT_REPORT_PASS)])
    build_novelty_audit_service(
        client=bus.view("NoveltyAuditAgent"),
        spec_store=store,
        llm_client=novelty_audit_llm,
    )

    # ONE FakeLLM handles CriticAgent's THREE calls IN ORDER: direction
    # critique (CritiquingDirection), hypotheses critique
    # (CritiquingHypotheses), experiment-plan review (ReviewingExperimentPlan).
    critic_llm = CriticFakeLLM(
        [
            _json(VALID_DIRECTION_REPORT_PASS),
            _json(VALID_HYPOTHESES_REPORT_SOME_PASS),
            _json(VALID_EXPERIMENT_PLAN_REPORT_PASS),
        ]
    )
    build_critic_service(
        client=bus.view("CriticAgent"),
        spec_store=store,
        llm_client=critic_llm,
    )
    # phase 3.1: MethodAgent (GeneratingMethod consumer, canned method).
    from autoresearch.agents.method_agent.service import build_method_service
    from autoresearch.orchestrator.tests.smoke._method_smoke_helper import MethodFakeLLM
    build_method_service(client=bus.view("MethodAgent"), spec_store=store, llm_client=MethodFakeLLM())

    # RankingJudgeAgent: consumes CritiquingHypotheses' entry action
    # (ranking.requested), needed to even reach CriticAgent's hypotheses
    # critique now that the rank -> rank-informed critique co-audit is wired
    # (RankingJudgeAgent milestone Task 4).
    ranking_llm = RankingJudgeFakeLLM([_json(VALID_RANKING_PAIRWISE_RESPONSE)])
    build_ranking_judge_service(
        client=bus.view("RankingJudgeAgent"),
        spec_store=store,
        llm_client=ranking_llm,
    )

    # ONE FakeLLM handles EvaluatorAgent + ComparisonGuardianAgent calls in order:
    # select_benchmark, comparison-integrity pass, evaluate_experiment (new_best),
    # verify_best_result (verified).
    # meeting #1: the pre-baseline prepare_verifier call is GONE (PreparingVerifier removed), so no
    # verifier_spec response is queued -- the flow goes SelectingBenchmark -> ReproducingBaseline.
    evaluator_llm = EvaluatorFakeLLM(
        [
            _json(VALID_BENCHMARK_SPEC),
            _json(COMPARISON_GUARDIAN_PASS),
            _json(EVALUATE_EXPERIMENT_LLM_RESPONSE),
            _json(VERIFY_BEST_RESULT_LLM_RESPONSE),
        ]
    )
    build_evaluator_service(
        client=bus.view("EvaluatorAgent"),
        spec_store=store,
        llm_client=evaluator_llm,
    )

    # RunnerAgent: ONE FakeRunnerBackend serves BOTH calls -- run_baseline
    # (ReproducingBaseline, metric 0.85 matching the benchmark's
    # baseline.reported_metric 0.85 within tolerance -> reproduced) AND
    # run_experiment (RunningExperiment, metric 0.90 > 0.85 -> new_best
    # downstream). NEVER a real backend.
    runner_backend = FakeRunnerBackend(
        raw_result={
            "metric_value": 0.85,
            "environment": "py3.11/cpu",
            "command": "python run_benchmark.py --model baseline",
            "seed": "0",
            "runtime_seconds": 1.0,
            "exit_code": 0,
            "log_excerpt": "baseline run completed successfully.",
        },
        metric_value=0.90,
        exit_code=0,
        stdout="ok",
    )
    build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=runner_backend,
    )

    # HypothesisAgent: canned HypothesisBank with ids h1/h2, "h1" of which is
    # the id CriticAgent's hypotheses-critique passes.
    hypothesis_llm = HypothesisFakeLLM([_json(VALID_HYPOTHESIS_BANK)])
    build_hypothesis_service(
        client=bus.view("HypothesisAgent"),
        spec_store=store,
        llm_client=hypothesis_llm,
    )

    # ExperimentPlannerAgent: canned ExperimentPlan whose changes[].file is a
    # subset of the VerifierSpec's editable_region.
    experiment_planner_llm = ExperimentPlannerFakeLLM([_json(VALID_EXPERIMENT_PLAN)])
    build_experiment_planner_service(
        client=bus.view("ExperimentPlannerAgent"),
        spec_store=store,
        llm_client=experiment_planner_llm,
    )

    # CoderAgent: canned ExperimentPatch touching only "src/model.py".
    coder_llm = CoderFakeLLM([_json(VALID_EXPERIMENT_PATCH)])
    build_coder_service(
        client=bus.view("CoderAgent"),
        spec_store=store,
        llm_client=coder_llm,
    )

    memory_llm = MemoryFakeLLM([_json(CONSOLIDATE_LLM_RESPONSE)])
    build_memory_service(
        client=bus.view("MemoryAgent"),
        spec_store=store,
        llm_client=memory_llm,
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

    # --- 1. start_session -> problem_modeling.requested -> (delivered) -----
    # -> ProblemModelingAgent asks its first clarification question.
    sid = await eng.start_session("optimize modded-nanogpt speedrun", user_id="u1")

    questions = [e for e in bus.published if e.event_type == "problem_modeling.clarification_question"]
    assert len(questions) == 1
    question_id = questions[0].payload["question_id"]

    state = await load_session_state(bus.db, sid)
    assert state is not None
    assert state.current_state == "ClarifyingResearchGoal"

    # --- 2. simulate the human answering the clarification question --------
    human = bus.view("Human")
    await human.publish(
        event_type="human.clarification_answer",
        session_id=sid,
        payload={"question_id": question_id, "answer_text": "benchmark report"},
    )

    checkpoint_state = await load_session_state(bus.db, sid)
    assert checkpoint_state is not None
    assert checkpoint_state.current_state == "HumanIdeaCheckpoint"
    assert checkpoint_state.human_checkpoint_return_to == "SelectingBenchmark"

    # --- 3. simulate the human at HumanIdeaCheckpoint {inject: false} ------
    # -> cascades through CritiquingDirection -> RoutingResearchMode ->
    # SelectingBenchmark -> PreparingVerifier -> ReproducingBaseline ->
    # GeneratingHypotheses -> CritiquingHypotheses (rank -> rank-informed
    # critique co-audit, RankingJudgeAgent milestone Task 4) ->
    # ManagingHypothesisPortfolio, whose entry action re-ranks the single
    # passing hypothesis ("h1", the 0/1-candidate trivial short-circuit, no
    # LLM call) and auto-advances (RankingJudgeAgent milestone Task 5,
    # selecting ranked_shortlist[0]) -> PlanningExperiment ->
    # ExperimentPlannerAgent -> ReviewingExperimentPlan -> CriticAgent (3rd
    # queued response, pass) -> ImplementingExperiment -> CoderAgent ->
    # RunningExperiment -> RunnerAgent -> EvaluatingExperiment -> EvaluatorAgent
    # (new_best) -> VerifyingBestResult -> EvaluatorAgent (verified) ->
    # ConsolidatingMemory -> MemoryAgent (routing_verdict "report") ->
    # Reporting -> EvidenceAuditAgent (Mode C, no unsupported claims) ->
    # ReportAgent -> Done -- ALL IN ONE publish call, synchronously, on this
    # one shared DeliveringBus.
    await human.publish(
        event_type="human.idea_provided",
        session_id=sid,
        payload={"inject": False},
    )
    # phase 3.1: pass -> GeneratingMethod -> ConfirmingMethod (pauses); confirm the method to proceed.
    await human.publish(event_type="method.confirmed", session_id=sid, payload={"inject": False})

    # --- Assertions ----------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None

    # THE HEADLINE PROOF: the session reached Done, the terminal state of the
    # state machine (V1.10 SS1), driven in ONE session with ALL 14 agents.
    assert final_state.current_state == "Done"
    event_types = [item.event_type for item in bus.published]
    assert event_types.index("baseline.reproduced") < event_types.index("method.generation_requested")
    initial_method_request = next(
        item for item in bus.published if item.event_type == "method.generation_requested")
    assert initial_method_request.payload["baseline_result_ref"] == final_state.baseline_result_ref
    # meeting 2026-07-25 甲: the benchmark mainline no longer runs the hypothesis-generation /
    # portfolio outer loop, so there are no confirmed hypotheses / ranking. The confirmed METHOD is
    # implemented directly; current_experiment_id is anchored from the method (not a hypothesis id).
    assert final_state.current_experiment_id is not None
    assert final_state.search_diversity_collapse_round == 0

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # --- Every key ref reachable along the way is set on the final SessionState.
    assert final_state.research_spec_ref is not None
    assert final_state.literature_brief_ref is not None
    assert final_state.benchmark_spec_ref is not None
    # meeting #1: the pre-baseline verifier is gone -> no verifier_spec is ever frozen.
    assert final_state.verifier_spec_ref is None
    # Meeting 2026-07-29: the bounded deep-deliberation step (Deliberating / 深思) is RETIRED from
    # the main path -- the confirmed method flows straight into the experiment, so no deliberation
    # report is produced (deliberation_report_ref stays None).
    assert final_state.deliberation_report_ref is None
    assert final_state.baseline_result_ref is not None
    assert final_state.experiment_plan_refs
    assert final_state.experiment_patch_refs
    assert final_state.run_record_refs
    assert final_state.evaluation_report_refs
    assert final_state.verification_report_refs
    assert final_state.research_memory_ref is not None
    assert final_state.citation_audit_report_ref is not None
    assert final_state.final_report_ref is not None

    # --- Schema-valid spot checks along the chain. ---------------------------
    benchmark_spec = store.get(final_state.benchmark_spec_ref)
    assert list(_validator(BENCHMARK_SPEC_SCHEMA_PATH).iter_errors(benchmark_spec)) == []

    baseline_result = store.get(final_state.baseline_result_ref)
    baseline_errors = list(_validator(BASELINE_RESULT_SCHEMA_PATH).iter_errors(baseline_result))
    assert baseline_errors == [], [e.message for e in baseline_errors]
    assert baseline_result["reproduced"] is True

    # 甲: no hypothesis bank on the benchmark mainline (the hypothesis-generation/portfolio loop is
    # retired); the plan below implements the confirmed method directly.
    experiment_plan = store.get(final_state.experiment_plan_refs[-1])
    plan_errors = list(_validator(EXPERIMENT_PLAN_SCHEMA_PATH).iter_errors(experiment_plan))
    assert plan_errors == [], [e.message for e in plan_errors]

    experiment_patch = store.get(final_state.experiment_patch_refs[-1])
    patch_errors = list(_validator(EXPERIMENT_PATCH_SCHEMA_PATH).iter_errors(experiment_patch))
    assert patch_errors == [], [e.message for e in patch_errors]

    run_record = store.get(final_state.run_record_refs[-1])
    run_record_errors = list(_validator(RUN_RECORD_SCHEMA_PATH).iter_errors(run_record))
    assert run_record_errors == [], [e.message for e in run_record_errors]
    assert run_record["succeeded"] is True
    assert run_record["metric_value"] == 0.90

    evaluation_report = store.get(final_state.evaluation_report_refs[-1])
    evaluation_report_errors = list(
        _validator(EVALUATION_REPORT_SCHEMA_PATH).iter_errors(evaluation_report)
    )
    assert evaluation_report_errors == [], [e.message for e in evaluation_report_errors]
    assert evaluation_report["verdict"] == "new_best"

    verification_report = store.get(final_state.verification_report_refs[-1])
    verification_report_errors = list(
        _validator(VERIFICATION_REPORT_SCHEMA_PATH).iter_errors(verification_report)
    )
    assert verification_report_errors == [], [e.message for e in verification_report_errors]
    assert verification_report["verdict"] == "verified"

    research_memory = store.get(final_state.research_memory_ref)
    research_memory_errors = list(
        _validator(RESEARCH_MEMORY_SCHEMA_PATH).iter_errors(research_memory)
    )
    assert research_memory_errors == [], [e.message for e in research_memory_errors]
    assert KNOWN_VERIFIED_FINDING in research_memory["verified_findings"]

    citation_audit_report = store.get(final_state.citation_audit_report_ref)
    assert citation_audit_report["unsupported_claims"] == []

    report = store.get(final_state.final_report_ref)
    report_errors = list(_validator(REPORT_SCHEMA_PATH).iter_errors(report))
    assert report_errors == [], [e.message for e in report_errors]
    assert report["verified_findings"] == [KNOWN_VERIFIED_FINDING]

    # --- A handful of key events fired exactly once, end to end.
    for event_type, expected_count in (
        ("baseline.reproduced", 1),
        # 甲: hypothesis.generated / hypothesis.critiqued / ranking.ready no longer fire on the
        # benchmark mainline (the hypothesis-generation/portfolio loop is retired).
        ("experiment.plan_ready", 1),
        ("code.implemented", 1),
        ("experiment.run_completed", 1),
        ("experiment.evaluated", 1),
        ("result.verified", 1),
        ("memory.consolidated", 1),
        ("report.ready", 1),
    ):
        matching = [e for e in bus.published if e.event_type == event_type]
        assert len(matching) == expected_count, (event_type, [e.payload for e in matching])
