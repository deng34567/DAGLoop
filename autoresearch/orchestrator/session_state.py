"""
SessionState dataclass + SQLite persistence.

Fields mirror AutoResearch state-machine spec V1.10 §5 "SessionState 完整结构"
(`docs/architecture/AutoResearch完整状态机V1.md`). Ported from vibe/orchestrator/session_state.py;
persistence API (init_session_state_table / load_session_state /
save_session_state / to_json / from_json) is preserved verbatim.
Persistence: JSON-blob in TEXT column for simplicity (single-row read/write,
no complex queries needed for V1 MVP).

Map counters (B/D/E/F/G/J/M, V1.10 §4) are plain Python ``dict``, never
``set`` -- ``json.dumps`` on a set raises ``TypeError``. Per §4's global rule
``map_reset_delete_key``, resetting a map counter deletes the key rather than
setting it to 0 (the invariant is "key in Map <=> scope currently in that
counter's phase"); use ``bump_map_counter`` / ``reset_map_counter`` below.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

import aiosqlite


_DDL = """
CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    state_data TEXT NOT NULL,
    current_state TEXT NOT NULL,
    updated_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_current_state ON session_state(current_state);
"""


def bump_map_counter(counter: dict[str, int], key: str, *, initial: int = 1, amount: int = 1) -> None:
    """Increment a Map-type counter (V1.10 §4: B/D/E/F/G/J/M).

    Per counters.yaml's ``on_missing_key: "create key with value 1"``: a
    missing key is created at ``initial`` rather than incremented from 0.
    """
    if key not in counter:
        counter[key] = initial
    else:
        counter[key] += amount


def reset_map_counter(counter: dict[str, int], key: str) -> None:
    """Reset a Map-type counter key by deleting it (V1.10 §4 ``delete_key``, never ``set_zero``)."""
    counter.pop(key, None)


@dataclass
class BranchState:
    """One concurrent research branch (V1.10 §5), used in PortfolioTreeSearch /
    CollectiveArena / multi-trajectory Lab modes."""

    branch_id: str
    parent_branch_id: Optional[str] = None
    status: str = "active"  # active | pruned | merged | done
    origin: str = "seed"  # seed | diversity_expansion | human_seed | arena_agent
    # Per-branch pointers; SessionState's top-level current_* mirror the
    # branch currently being advanced.
    current_hypothesis_id: Optional[str] = None
    current_experiment_id: Optional[str] = None
    current_result_id: Optional[str] = None
    branch_score: Optional[float] = None  # best-first selection / pruning
    budget_share: float = 0.0  # this branch's share of the session budget


@dataclass
class SessionState:
    # 身份 (identity)
    session_id: str
    user_id: Optional[str] = None
    # Persist the user's original goal so a restart while ClarifyingResearchGoal
    # can re-emit the real task instead of an internal recovery marker.
    research_goal_text: str = ""
    created_at: float = 0.0
    last_updated_at: float = 0.0

    # 状态机 (state machine)
    current_state: str = "ClarifyingResearchGoal"
    previous_state: Optional[str] = None
    state_entered_at: float = 0.0
    mode: str = "benchmark"
    # enum: benchmark | idea_theory | experiment_validation
    #       | portfolio_tree_search | collective_arena | lab_in_the_loop

    # 计数器 (counters A-M, V1.10 §4)
    direction_critique_round: int = 0                                          # A scalar
    baseline_reproduce_round: dict[str, int] = field(default_factory=dict)     # B[benchmark_id]
    hypothesis_reject_round: int = 0                                           # C scalar
    theory_review_round: dict[str, int] = field(default_factory=dict)         # D[hypothesis_id]
    experiment_plan_review_round: dict[str, int] = field(default_factory=dict)  # E[hypothesis_id]
    implementation_fix_round: dict[str, int] = field(default_factory=dict)    # F[experiment_id]
    result_verification_fail_round: dict[str, int] = field(default_factory=dict)  # G[result_id]
    research_rework_count: int = 0                                           # H scalar
    inconsistency_bounce_count: int = 0                                      # I scalar
    agent_retry_count: dict[str, int] = field(default_factory=dict)          # J[state]
    literature_audit_round: int = 0                                         # K scalar
    search_diversity_collapse_round: int = 0                                # L scalar
    data_analysis_consensus_round: dict[str, int] = field(default_factory=dict)  # M[external_experiment_id]
    # bounded idea<->experiment exploration loop: rounds taken so far. NOT a counters.yaml
    # counter (no MOSS threshold_action); the engine caps it at AUTORESEARCH_LOOP_MAX_ROUNDS.
    exploration_loop_round: int = 0
    # safety cap: how many times an agent has failed -> AwaitingUserDecision(agent_failure). Bounds
    # the report-phase crash-loop (a report agent that keeps failing would bounce forever).
    agent_failure_round: int = 0

    # phase 3.1: the explicit METHOD derived from the confirmed direction (MethodAgent output).
    method_ref: Optional[str] = None
    # After a real baseline is frozen, reuse DeliberatingMethod once to audit only the executable
    # ComponentPlan.  The handler preserves the already-confirmed core method in this pass.
    component_plan_audit_pending: bool = False
    # One-shot marker for a MOSS novelty targeted-search.  Its literature must re-audit the same
    # frozen method instead of returning through method generation and moving the search target.
    novelty_targeted_search_pending: bool = False
    # Correlation identity for that targeted search.  It stays live through literature,
    # evidence, and novelty audit so delayed events cannot consume a newer request.
    novelty_targeted_search_id: Optional[str] = None
    # Per-method cap counter.  A fundamentally new method resets it; the method-swap cap bounds the
    # total number of targeted searches across the session.
    novelty_targeted_search_round: int = 0
    # phase 3.2 loop counters: inner experiment-design/setting rework, outer method revision.
    setting_rework_round: int = 0
    method_revision_round: int = 0
    # meeting ② (2026-07-31): THIRD (outermost) layer -- when settings AND method-revisions are both
    # exhausted without an advantage, the method is falsified (it doesn't work), so MOSS may SWAP it
    # for a fundamentally DIFFERENT method (back to GeneratingMethod). Bounded by _METHOD_SWAP_CAP.
    method_swap_round: int = 0
    # one-shot flag: set when a swap is triggered so GeneratingMethod's entry feeds MethodAgent the
    # "prior method didn't work -> propose a DIFFERENT approach" instruction; cleared once regenerated.
    method_not_working: bool = False
    # 会议 2026-08-03 (乙): a declared component's ablation checkpoint failed and MOSS routed back to
    # the method layer to fix THAT component. Bounded by _COMPONENT_REVISION_CAP. one-shot flag +
    # the implicated component name, fed to the method layer's entry action; cleared once regenerated.
    component_revision_round: int = 0
    component_ablation_failed: bool = False
    ablation_failed_component: str = ""

    # 研究上下文 (research context refs)
    research_spec_ref: Optional[str] = None
    literature_brief_ref: Optional[str] = None
    field_trajectory_brief_ref: Optional[str] = None
    spec_interpretation_ref: Optional[str] = None
    novelty_audit_report_ref: Optional[str] = None
    evidence_audit_report_ref: Optional[str] = None
    citation_audit_report_ref: Optional[str] = None
    problem_model_ref: Optional[str] = None
    deliberation_report_ref: Optional[str] = None
    human_idea_seed_ref: Optional[str] = None
    # repo-baseline confirmation: the repo the user confirmed (at HumanIdeaCheckpoint) for the
    # experiment to clone+improve. When set, the runner routes to repo-improve with THIS url
    # instead of self-writing -- so a question with no explicit repo can still use an existing
    # baseline the system found + the user picked. None -> current behavior (explicit/self-write).
    confirmed_repo_url: Optional[str] = None
    # user-confirmed trusted-eval BOUNDARY overrides (run_cmd/metric/budget_s/editable_files/...)
    # that win over the LLM's inference -- fixes an unreliable auto-inferred command/metric/budget
    # (e.g. makemore's missing `-i names.txt`, a max-steps budget). Empty -> pure inference.
    confirmed_boundary: dict = field(default_factory=dict)
    benchmark_spec_ref: Optional[str] = None
    verifier_spec_ref: Optional[str] = None
    baseline_result_ref: Optional[str] = None
    # Latest attempt is diagnostic evidence, not necessarily a valid frozen comparator.
    last_baseline_attempt_ref: Optional[str] = None
    external_experiment_package_ref: Optional[str] = None
    hypothesis_bank_ref: Optional[str] = None
    hypothesis_portfolio_ref: Optional[str] = None
    research_tree_ref: Optional[str] = None
    method_feedback_round: int = 0
    method_feedback_ref: Optional[str] = None
    method_feedback_in_progress: bool = False
    research_memory_ref: Optional[str] = None
    research_knowledge_index_ref: Optional[str] = None
    hypothesis_ranking_report_ref: Optional[str] = None
    benchmark_ranking_report_ref: Optional[str] = None
    external_data_analysis_report_ref: Optional[str] = None

    # 当前指针 (points at the branch currently being advanced; the only
    # branch in linear mode)
    current_benchmark_id: Optional[str] = None
    current_branch_id: Optional[str] = None
    current_hypothesis_id: Optional[str] = None
    current_experiment_id: Optional[str] = None
    current_external_experiment_id: Optional[str] = None
    current_result_id: Optional[str] = None
    best_result_id: Optional[str] = None

    # 并发分支 (PortfolioTreeSearch / CollectiveArena / multi-trajectory Lab)
    active_branches: list[BranchState] = field(default_factory=list)  # len <= 1 in single-branch mode
    max_concurrent_branches: int = 1  # set by ResearchOrchestrator per budget
    branch_schedule_policy: str = "linear"  # linear | best_first_tree | collective_arena

    # 产物索引 (artifact ref indexes)
    theory_draft_refs: list[str] = field(default_factory=list)
    experiment_plan_refs: list[str] = field(default_factory=list)
    experiment_patch_refs: list[str] = field(default_factory=list)
    run_record_refs: list[str] = field(default_factory=list)
    evaluation_report_refs: list[str] = field(default_factory=list)
    # Legacy/no-store fallback only. Production binding uses EvaluationReport.run_record_digest,
    # because deterministic SpecStore refs are overwritten and cannot identify a run attempt.
    last_evaluated_run_record_ref: Optional[str] = None
    verification_report_refs: list[str] = field(default_factory=list)
    critique_report_refs: list[str] = field(default_factory=list)
    novelty_audit_refs: list[str] = field(default_factory=list)
    verifier_audit_refs: list[str] = field(default_factory=list)
    external_experiment_package_refs: list[str] = field(default_factory=list)
    external_data_upload_refs: list[str] = field(default_factory=list)
    external_data_analysis_report_refs: list[str] = field(default_factory=list)
    analysis_notebook_refs: list[str] = field(default_factory=list)
    consolidated_insight_refs: list[str] = field(default_factory=list)
    divergence_seed_refs: list[str] = field(default_factory=list)
    passing_hypothesis_ids: list[str] = field(default_factory=list)

    # MOSS
    current_moss_reason: Optional[str] = None
    moss_interventions: list[dict] = field(default_factory=list)  # [MossInterventionRecord]

    # AwaitingUserDecision (user-decision reason-matrix, V1.10 §3)
    awaiting_user_decision_reason: Optional[str] = None
    last_user_decision: Optional[dict] = None
    decision_caveats: list[str] = field(default_factory=list)

    # 人类介入 (human-in-the-loop)
    checkpoint_pending: bool = False
    human_checkpoint_return_to: Optional[str] = None  # state to return to after HumanIdeaCheckpoint
    last_human_checkpoint: Optional[dict] = None
    # {requested_at: timestamp, requested_in_state: enum, summary: string} or None

    # 成本 (cost)
    cumulative_llm_cost: str = "0"
    cumulative_compute_cost: str = "0"
    llm_cost_warning_emitted: bool = False
    compute_cost_warning_emitted: bool = False

    # Agent Failure 上下文
    last_agent_failure: Optional[dict] = None
    # {failed_in_state, agent_name, reason_classification, underlying_reason,
    #  retry_attempts, failed_at} or None

    # 报告 (report)
    final_report_ref: Optional[str] = None
    report_failed: bool = False
    experiment_status: str = "not_started"
    # enum: not_started | not_run | running | verified | failed | inconclusive

    # 幂等性 (idempotency)
    last_processed_event_id: Optional[str] = None
    # New sessions bind automatic replies to one durable request and frozen inputs.
    # Legacy sessions are not silently migrated into the new completion protocol.
    request_binding_required: bool = False
    active_request: Optional[dict] = None
    completion_verified: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> "SessionState":
        d = json.loads(data)
        if (
            d.get("novelty_targeted_search_pending")
            and not d.get("novelty_targeted_search_id")
            and d.get("method_ref")
            and d.get("novelty_audit_report_ref")
        ):
            identity = ":".join((
                str(d.get("session_id") or ""),
                str(d["method_ref"]),
                str(d["novelty_audit_report_ref"]),
            ))
            d["novelty_targeted_search_id"] = (
                "legacy-" + uuid.uuid5(uuid.NAMESPACE_URL, identity).hex)
            d["novelty_targeted_search_round"] = max(
                1, int(d.get("novelty_targeted_search_round") or 0))
        branches = d.get("active_branches")
        if branches is not None:
            d["active_branches"] = [BranchState(**b) for b in branches]
        return cls(**d)


async def init_session_state_table(db: aiosqlite.Connection) -> None:
    for stmt in _DDL.split(";"):
        s = stmt.strip()
        if s:
            await db.execute(s)
    await db.commit()


async def load_session_state(
    db: aiosqlite.Connection, session_id: str
) -> Optional[SessionState]:
    cursor = await db.execute(
        "SELECT state_data FROM session_state WHERE session_id=?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return SessionState.from_json(row[0])


async def save_session_state(
    db: aiosqlite.Connection, state: SessionState, *, auto_commit: bool = False
) -> None:
    """Persist state. Caller must commit unless auto_commit=True.

    For the Orchestrator engine's atomic tick, default is False -- engine
    wraps multiple writes in one transaction.
    """
    now = time.time()
    state.last_updated_at = now
    state_json = state.to_json()
    await db.execute(
        """
        INSERT INTO session_state (session_id, state_data, current_state, updated_at, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            state_data = excluded.state_data,
            current_state = excluded.current_state,
            updated_at = excluded.updated_at
        """,
        (state.session_id, state_json, state.current_state, now,
         state.created_at if state.created_at > 0 else now),
    )
    if auto_commit:
        await db.commit()
