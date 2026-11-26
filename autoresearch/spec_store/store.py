"""
Spec Store: the shared artifact store agents use to exchange products.
共享产物存储：各 agent 之间交换产物的统一存储。

An agent writes an artifact and gets back an opaque ``ref`` string; it puts the
ref into an event; downstream agents read by ref. Structured JSON lives in
SQLite, binary blobs (e.g. screenshots) live as local files.
agent 写入产物拿到不透明 ref，把 ref 塞进事件；下游 agent 凭 ref 读取。
结构化 JSON 存 SQLite，二进制 blob（如截图）以本地文件形式存储。

SYNCHRONOUS by design — agents call it synchronously, so we use stdlib sqlite3.
设计为同步——agent 同步调用，因此用标准库 sqlite3。

Self-contained: no imports from vibe.eventbus / vibe.orchestrator.
自包含：不依赖 vibe.eventbus / vibe.orchestrator。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import refs


class EmitterNotAllowed(Exception):
    """Raised when an emitter is not authorized to write the given kind.
    当某 emitter 无权写入指定 kind 时抛出。"""


# kind -> allowed emitter. The ANY sentinel means any non-empty emitter is ok.
# kind -> 允许的 emitter。ANY 哨兵表示任意非空 emitter 均可。
_ANY = "*"
_EMITTER_WHITELIST: dict[str, object] = {  # value: emitter str, frozenset of strs, or _ANY
    "requirement": "RequirementAgent",
    "conversation": "RequirementAgent",  # 澄清多轮对话历史(transcript);clarification_context ref 指向它
    "architecture": "ArchitectAgent",
    "scenario_set": "ScenarioAgent",
    # CA1: 可执行行为契约核(python3 脚本打 booted app)。★只 ScenarioAgent 能写(独立于
    # builder,防 coder 自评假绿);get 不校验 emitter → coder gate / Sentinel 可读跑(从权威 ref,
    # 非 coder workdir)。
    "behavioral_contract": "ScenarioAgent",
    "impl_summary": "CoderAgent",
    "code": "CoderAgent",  # 可构建的整仓源码快照(code_repo_ref);Sentinel 从此重建被测 app
    # Task#13 交付包验证循环:已验收源码 + 修后 docker 配方的独立产物车道。CoderAgent 的
    # bundle-fix 从此车道 rehydrate/export,★绝不覆盖 manifest.current_code_repo_ref(SSOT)。
    "deliverable_source": "CoderAgent",
    "acceptance_report": "AcceptanceAgent",
    "recording": "AcceptanceAgent",  # 成功验收的 rrweb 录屏(Acceptance Theater)
    "evidence": "AcceptanceAgent",  # 失败验收的原始结构化诊断(给 Coder)
    "bug_history": "Orchestrator",
    "documentation": "DocAgent",
    "screenshot": "AcceptanceAgent",
    # Task#4: Docker 一键交付包(源码+生成 Dockerfile/compose/启动器的 zip)。由 web 网关
    # on-demand 生成并缓存(emitter=frontend,与 gateway 写 conversation/audit 的身份一致)。
    # Task#13: post-Done 打包 worker(DeliverablePackager)验证循环收敛后,把"验证过的 build
    # context + launchers" zip 也写进 deliverable_bundle 车道(按 code_ref 缓存),故放行
    # DeliverablePackager 写本 kind;沿用 frontend 兜底(网关 on-demand 路径)→ _ANY。
    "deliverable_bundle": _ANY,
    # Task#13: 交付包验证循环的状态车道(verifying/ready/degraded + reason/progress/round)。
    # 仅 post-Done 打包 worker(DeliverablePackager)写;前端读。绝不污染源码/manifest SSOT。
    "deliverable_status": "DeliverablePackager",
    "audit": _ANY,
    # AutoResearch M1: frozen ClarifyingResearchGoal output (research_spec) and
    # its in-progress accumulation (research_spec_draft), both written only by
    # ProblemModelingAgent; downstream reads by ref (research_spec.ready.payload).
    "research_spec": "ProblemModelingAgent",
    "research_spec_draft": "ProblemModelingAgent",
    # AutoResearch phase 3.1: the explicit METHOD (concrete approach whose advantage the experiments
    # demonstrate) derived from the confirmed direction. Written only by MethodAgent.
    "method": "MethodAgent",
    # meeting ① (2026-07-31): the bounded proposer<->critic method-debate report (rounds/converged/
    # too_simple + transcript) from DeliberatingMethod -- observability of the "注入灵魂" step.
    "method_debate_report": "MethodAgent",
    # AutoResearch: an idea/direction (or a baseline repo URL) the HUMAN injects at the
    # HumanIdeaCheckpoint; persisted so it actually reaches the engine. Written by the web
    # gateway on the human's behalf (emitter="Human"). Missing this made injection crash.
    "human_idea_seed": "Human",
    # AutoResearch M2: DeepResearchAgent's two-pass literature search output
    # (literature_brief) and field-trajectory synthesis (field_trajectory_brief),
    # both written only by DeepResearchAgent; downstream reads by ref
    # (literature.brief_ready.payload).
    "literature_brief": "DeepResearchAgent",
    "field_trajectory_brief": "DeepResearchAgent",
    # AutoResearch M3: KnowledgeIndexAgent's IndexEntry index (see
    # agents/knowledge_index_agent/data-contracts/index_entry.schema.yaml),
    # written only by KnowledgeIndexAgent.
    "knowledge_index": "KnowledgeIndexAgent",
    # AutoResearch ReinterpretingSpec: ProblemModelingAgent's scope-gate verdict
    # (see agents/problem_modeling_agent/data-contracts/spec_interpretation.schema.yaml),
    # written only by ProblemModelingAgent; downstream reads by ref
    # (spec.reinterpreted.payload).
    "spec_interpretation": "ProblemModelingAgent",
    # AutoResearch M4: NoveltyAuditAgent's audit verdict (see
    # agents/novelty_audit_agent/data-contracts/novelty_audit_report.schema.yaml),
    # written only by NoveltyAuditAgent; downstream reads by ref
    # (novelty.audit_ready.payload).
    "novelty_audit_report": "NoveltyAuditAgent",
    # AutoResearch M5: HypothesisAgent's generated hypothesis bank (see
    # agents/hypothesis_agent/data-contracts/hypothesis_bank.schema.yaml),
    # written only by HypothesisAgent; downstream reads by ref
    # (hypothesis.generated.payload).
    "hypothesis_bank": "HypothesisAgent",
    # AutoResearch ModelingProblem: ProblemModelingAgent's ProblemModel (see
    # agents/problem_modeling_agent/data-contracts/problem_model.schema.yaml),
    # written only by ProblemModelingAgent; downstream reads by ref
    # (problem.modeled.payload).
    "problem_model": "ProblemModelingAgent",
    # AutoResearch M6: CriticAgent's critique verdict (see
    # agents/critic_agent/data-contracts/critique_report.schema.yaml),
    # written only by CriticAgent; downstream reads by ref
    # (direction.critiqued.payload / hypothesis.critiqued.payload).
    # normally the CriticAgent; also "Human" for the opt-in design-review gate, where a
    # person approves/rejects the P1 design at ReviewingExperimentPlan (web / driver).
    "critique_report": frozenset({"CriticAgent", "Human"}),
    # Deep-deliberation step (bounded MOSS-style cross-reference): DeliberationAgent
    # spawns N=2-3 angle takes, cross-references them for one round, and synthesizes a
    # DeliberationReport (see agents/deliberation_agent/data-contracts/
    # deliberation_report.schema.yaml). Written only by DeliberationAgent; downstream
    # (SelectingBenchmark / PlanningExperiment) reads it by ref.
    "deliberation_report": "DeliberationAgent",
    # AutoResearch M7: EvaluatorAgent's selected/prepared BenchmarkSpec (see
    # agents/evaluator_agent/data-contracts/benchmark_spec.schema.yaml) and
    # frozen verifier-first VerifierSpec (see
    # agents/evaluator_agent/data-contracts/verifier_spec.schema.yaml), both
    # written only by EvaluatorAgent; downstream reads by ref
    # (benchmark.spec_ready.payload / verifier.spec_frozen.payload).
    "benchmark_spec": "EvaluatorAgent",
    "verifier_spec": "EvaluatorAgent",
    # AutoResearch M8: ExperimentPlannerAgent's frozen ExperimentPlan (see
    # agents/experiment_planner_agent/data-contracts/experiment_plan.schema.yaml),
    # written only by ExperimentPlannerAgent; downstream reads by ref
    # (experiment.plan_ready.payload).
    "experiment_plan": "ExperimentPlannerAgent",
    # P1 design-review gate: the concrete P1 design generated at PlanningExperiment for
    # human review at ReviewingExperimentPlan (see experiments/general).
    "gen_design": "ExperimentPlannerAgent",
    # AutoResearch M9: CoderAgent's frozen ExperimentPatch (see
    # agents/coder_agent/data-contracts/experiment_patch.schema.yaml),
    # written only by CoderAgent; downstream reads by ref
    # (code.implemented.payload).
    "experiment_patch": "CoderAgent",
    # AutoResearch M10: RunnerAgent's baseline reproduction result (see
    # agents/runner_agent/data-contracts/baseline_result.schema.yaml),
    # written only by RunnerAgent; downstream reads by ref
    # (baseline.reproduced.payload).
    "baseline_result": "RunnerAgent",
    # AutoResearch M11: RunnerAgent's RunningExperiment result (see
    # agents/runner_agent/data-contracts/run_record.schema.yaml),
    # written only by RunnerAgent; downstream reads by ref
    # (experiment.run_completed.payload).
    "run_record": "RunnerAgent",
    "experiment_completion": "RunnerAgent",
    # RunnerAgent's independent, diff-grounded check that the measured method
    # actually implements the frozen Method.  The next method attempt reads this
    # artifact to correct partial or drifted implementations.
    "method_implementation_review": "RunnerAgent",
    "implementation_review_input": "RunnerAgent",
    "pending_implementation_review": "RunnerAgent",
    # Runner's task-identity/comparison audit, separate from method fidelity.
    "repository_comparison_review": "RunnerAgent",
    # AutoResearch general honest-experiment pipeline (P1): a small sidecar the
    # RunnerAgent's gen branch writes alongside the RunRecord, carrying the
    # structured P1 verdict + baseline/method metrics so the MemoryAgent's gen
    # branch can recover {verdict, baseline_metric, method_metric, reasons}
    # (the RunRecord schema is additionalProperties:false, so it can't hold them,
    # and research_memory is MemoryAgent-only). Written only by RunnerAgent.
    "gen_experiment_result": "RunnerAgent",
    "execution_provenance": "RunnerAgent",
    # AutoResearch P1: the DOWNLOADABLE, Docker-runnable experiment bundle -- the
    # agent-written method.py + harness.py + the trusted driver + Dockerfile/README
    # -- so the user can see and run the actual deliverable. Written only by
    # RunnerAgent; served by the web viewer's /api/experiment_bundle (zip).
    "gen_experiment_bundle": "RunnerAgent",
    # Lightweight theory-validation artifacts: structured result plus the
    # downloadable script/raw-data/result bundle, both written by RunnerAgent.
    "validation_study_result": "RunnerAgent",
    "validation_study_bundle": "RunnerAgent",
    # AutoResearch M12: EvaluatorAgent's EvaluatingExperiment result (see
    # agents/evaluator_agent/data-contracts/evaluation_report.schema.yaml) and
    # VerifyingBestResult result (see
    # agents/evaluator_agent/data-contracts/verification_report.schema.yaml),
    # both written only by EvaluatorAgent; downstream reads by ref
    # (experiment.evaluated.payload / result.verified.payload).
    "evaluation_report": "EvaluatorAgent",
    "verification_report": "EvaluatorAgent",
    # AutoResearch M13: MemoryAgent's ConsolidatingMemory result (see
    # agents/memory_agent/data-contracts/research_memory.schema.yaml),
    # written only by MemoryAgent; downstream reads by ref
    # (memory.consolidated.payload).
    "method_improvement_review": "MemoryAgent",
    "research_memory": "MemoryAgent",
    # AutoResearch M14: ReportAgent's final honest Report (see
    # agents/report_agent/data-contracts/report.schema.yaml),
    # written only by ReportAgent; downstream reads by ref
    # (report.ready.payload).
    "report": "ReportAgent",
    # MOSS Step 1: MOSS's intervention decision (see
    # agents/moss_agent/data-contracts/moss_decision.schema.yaml),
    # written only by MOSS; downstream reads by ref
    # (moss.intervention_completed.payload).
    "moss_intervention": "MOSS",
    # EvidenceAuditAgent Task 1: EvidenceAuditAgent's claim-citation audit
    # (see agents/evidence_audit_agent/data-contracts/evidence_audit_report.schema.yaml)
    # and citation audit of consolidated ResearchMemory (see
    # agents/evidence_audit_agent/data-contracts/citation_audit_report.schema.yaml),
    # both written only by EvidenceAuditAgent; downstream reads by ref
    # (evidence.audited.payload / evidence.citation_audited.payload).
    "evidence_audit_report": "EvidenceAuditAgent",
    "citation_audit_report": "EvidenceAuditAgent",
    # RankingJudgeAgent Task 1: RankingJudgeAgent's pairwise/BTL ranking
    # verdict over hypothesis (or, later, benchmark) candidates (see
    # agents/ranking_judge_agent/data-contracts/ranking_report.schema.yaml),
    # written only by RankingJudgeAgent; downstream reads by ref
    # (ranking.ready.payload).
    "ranking_report": "RankingJudgeAgent",
    # DataAnalysisAgent Task 1: DataAnalysisAgent's QC + trajectory-consensus
    # verdict over an uploaded external dataset (see
    # agents/data_analysis_agent/data-contracts/external_data_analysis_report.schema.yaml),
    # written only by DataAnalysisAgent; downstream reads by ref
    # (data_analysis.ready.payload).
    "external_data_analysis_report": "DataAnalysisAgent",
}

# Emitters allowed to mutate the project manifest. / 允许改写 project manifest 的 emitter。
_MANIFEST_EMITTERS: frozenset[str] = frozenset(
    {
        "RequirementAgent",
        "ScenarioAgent",
        "ArchitectAgent",
        "CoderAgent",
        "AcceptanceAgent",
        "Orchestrator",
    }
)

# Chars unsafe for a filesystem path segment. / 文件系统路径段中不安全的字符。
_UNSAFE_PATH_RE = re.compile(r"[^A-Za-z0-9._/-]")

_DDL = """
CREATE TABLE IF NOT EXISTS artifacts (
    ref            TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    key            TEXT NOT NULL,
    emitter        TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    json           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS manifests (
    session_id TEXT PRIMARY KEY,
    json       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    """ISO8601 UTC timestamp. / ISO8601 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _check_emitter(kind: str, emitter: str) -> None:
    """Enforce the kind->emitter whitelist. / 执行 kind->emitter 白名单。"""
    if kind not in _EMITTER_WHITELIST:
        raise EmitterNotAllowed(f"unknown kind: {kind!r}")
    allowed = _EMITTER_WHITELIST[kind]
    if allowed == _ANY:
        if not emitter:
            raise EmitterNotAllowed(f"kind {kind!r} requires a non-empty emitter")
        return
    allowed_set = allowed if isinstance(allowed, (set, frozenset)) else {allowed}
    if emitter not in allowed_set:
        raise EmitterNotAllowed(
            f"emitter {emitter!r} not allowed for kind {kind!r} (expected {allowed!r})"
        )


def _sanitize_segment(seg: str) -> str:
    """Make a path segment filesystem-safe; preserves '/' subdirs.
    把路径段处理为文件系统安全；保留 '/' 作为子目录分隔。"""
    return _UNSAFE_PATH_RE.sub("_", seg)


class SpecStore:
    """SQLite + local-file artifact store. / SQLite + 本地文件的产物存储。"""

    def __init__(self, root: str | Path) -> None:
        """Open/create the store under ``root``.
        在 root 下打开/创建存储（生成 <root>/spec_store.db 与 <root>/blobs/）。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs_dir = self.root / "blobs"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "spec_store.db"
        # check_same_thread=False: BrokerBus runs agent handlers in worker
        # threads (asyncio.to_thread), AND the SentinelAgent runs scenarios
        # concurrently (ThreadPoolExecutor) sharing ONE store instance — so this
        # single connection IS touched concurrently from multiple threads. A
        # single sqlite3 connection is not safe for concurrent cursor use (results
        # cross-contaminate → json.loads(None) crashes), and WAL/busy_timeout only
        # help across SEPARATE connections, not within one. So every _db access is
        # serialized by self._lock below (a reentrant lock: upsert_manifest's
        # read-modify-write nests fetch_project_manifest under the same lock).
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        for stmt in _DDL.split(";"):
            s = stmt.strip()
            if s:
                self._db.execute(s)
        self._db.commit()

    # ---- structured artifacts / 结构化产物 ----

    def put(self, *, kind: str, session_id: str, key: str, obj: dict, emitter: str) -> str:
        """Write a structured artifact; return its ref. / 写结构化产物并返回 ref。

        Idempotent over (kind, session_id, key): the ref is deterministic and a
        repeat call upserts (overwrites) the row, so at-least-once redelivery
        does not create duplicates.
        对 (kind, session_id, key) 幂等：ref 确定，重复调用 upsert 覆盖，
        因此至少一次重投不会产生重复。
        """
        _check_emitter(kind, emitter)
        ref = refs.encode(kind, session_id, key)
        if re.fullmatch(r"snapshot-[0-9a-f]{64}", key):
            from .identity import artifact_digest
            if key != "snapshot-" + artifact_digest(obj).split(":", 1)[1]:
                raise ValueError("snapshot content must match its immutable reference")
        with self._lock:
            self._db.execute(
                """
                INSERT INTO artifacts (ref, kind, session_id, key, emitter, schema_version, created_at, json)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(ref) DO UPDATE SET
                    kind = excluded.kind,
                    session_id = excluded.session_id,
                    key = excluded.key,
                    emitter = excluded.emitter,
                    created_at = excluded.created_at,
                    json = excluded.json
                """,
                (ref, kind, session_id, key, emitter, _now_iso(), json.dumps(obj)),
            )
            self._db.commit()
        return ref

    def snapshot(self, ref: str) -> str:
        """Freeze an existing artifact without changing its producer attribution."""
        from .identity import artifact_digest
        with self._lock:
            row = self._db.execute("SELECT json, emitter FROM artifacts WHERE ref=?", (ref,)).fetchone()
        if row is None:
            raise KeyError(ref)
        obj = json.loads(row[0])
        parts = refs.decode(ref)
        return self.put(kind=parts.kind, session_id=parts.session_id,
                        key="snapshot-" + artifact_digest(obj).split(":", 1)[1],
                        obj=obj, emitter=row[1])

    def get(self, ref: str) -> dict:
        """Read a structured artifact by ref. / 凭 ref 读结构化产物。

        Raises KeyError if absent. / 不存在则抛 KeyError。
        """
        with self._lock:
            row = self._db.execute(
                "SELECT json FROM artifacts WHERE ref=?", (ref,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no artifact for ref: {ref!r}")
        return json.loads(row[0])

    # ---- blobs / 二进制产物 ----

    def put_blob(
        self, *, kind: str, session_id: str, key: str, data: bytes, emitter: str, ext: str
    ) -> str:
        """Write a blob to disk + a metadata row; return its ref.
        把 blob 写到磁盘并记录元数据行，返回 ref。

        Idempotent over (kind, session_id, key), same as put.
        与 put 一样对 (kind, session_id, key) 幂等。
        """
        _check_emitter(kind, emitter)
        ref = refs.encode(kind, session_id, key)
        ext = ext.lstrip(".")
        # ref stays canonical; only the on-disk path is sanitized.
        # ref 保持规范，仅磁盘路径做安全化处理。
        safe_session = _sanitize_segment(session_id)
        safe_kind = _sanitize_segment(kind)
        safe_key = _sanitize_segment(key)
        rel_path = f"{safe_session}/{safe_kind}/{safe_key}.{ext}"
        abs_path = self.blobs_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)
        meta = {"blob_path": rel_path, "ext": ext}
        with self._lock:
            self._db.execute(
                """
                INSERT INTO artifacts (ref, kind, session_id, key, emitter, schema_version, created_at, json)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(ref) DO UPDATE SET
                    kind = excluded.kind,
                    session_id = excluded.session_id,
                    key = excluded.key,
                    emitter = excluded.emitter,
                    created_at = excluded.created_at,
                    json = excluded.json
                """,
                (ref, kind, session_id, key, emitter, _now_iso(), json.dumps(meta)),
            )
            self._db.commit()
        return ref

    def get_blob(self, ref: str) -> bytes:
        """Read a blob by ref. / 凭 ref 读 blob。

        Raises KeyError if no row. / 无元数据行则抛 KeyError。
        """
        meta = self.get(ref)
        rel_path = meta["blob_path"]
        return (self.blobs_dir / rel_path).read_bytes()

    def get_blob_by_keys(
        self, *, session_id: str, scenario_id: str, screenshot_id: str
    ) -> bytes:
        """Read a screenshot blob by its scenario/screenshot keys.
        凭 scenario/screenshot 键读取截图 blob。

        Screenshots use kind="screenshot", key=f"{scenario_id}/{screenshot_id}".
        截图约定 kind="screenshot"，key=f"{scenario_id}/{screenshot_id}"。
        """
        ref = refs.encode("screenshot", session_id, f"{scenario_id}/{screenshot_id}")
        return self.get_blob(ref)

    # ---- project manifest / 项目清单 ----

    def fetch_project_manifest(self, session_id: str) -> dict:
        """Return the session's manifest, or {} if none. / 返回该 session 的 manifest，无则 {}。"""
        with self._lock:
            row = self._db.execute(
                "SELECT json FROM manifests WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return {}
        return json.loads(row[0])

    def upsert_manifest(self, session_id: str, patch: dict, emitter: str) -> None:
        """Shallow-merge ``patch`` into the session's manifest.
        把 patch 浅合并进该 session 的 manifest。

        Raises EmitterNotAllowed for unauthorized emitters.
        emitter 无权时抛 EmitterNotAllowed。
        """
        if emitter not in _MANIFEST_EMITTERS:
            raise EmitterNotAllowed(
                f"emitter {emitter!r} not allowed to upsert manifest"
            )
        # Hold the lock across the read-modify-write so a concurrent upsert can't
        # clobber the shallow merge (RLock: fetch_project_manifest re-enters it).
        with self._lock:
            current = self.fetch_project_manifest(session_id)
            current.update(patch)  # shallow merge / 浅合并
            self._db.execute(
                """
                INSERT INTO manifests (session_id, json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    json = excluded.json,
                    updated_at = excluded.updated_at
                """,
                (session_id, json.dumps(current), _now_iso()),
            )
            self._db.commit()

    # ---- queries / 查询 ----

    def latest_acceptance(self, session_id: str) -> str | None:
        """Ref of the newest passing acceptance_report, else None.
        最近一次通过的 acceptance_report 的 ref，否则 None。

        "Passing" = json.get("all_passed") is True. Ordered by created_at DESC.
        "通过" = json.get("all_passed") 为 True；按 created_at 倒序取最新。
        """
        with self._lock:
            rows = self._db.execute(
                """
                SELECT ref, json FROM artifacts
                WHERE kind='acceptance_report' AND session_id=?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        for ref, raw in rows:
            obj = json.loads(raw)
            if obj.get("all_passed") is True:
                return ref
        return None

    def list_refs(self, session_id: str, kind: str, *, limit: int | None = None) -> list[str]:
        """Refs of all artifacts of `kind` for `session_id`, newest-first.
        某 session 下某 kind 的全部 ref，按 created_at 倒序（最新在前）。

        Read-only helper (additive): lets a consumer (e.g. MOSS reading deep for a
        diagnosis) pull the latest architecture / failing acceptance_report /
        evidence by (session, kind) without holding the producer's opaque ref.
        Unlike `latest_acceptance` this does NOT filter on all_passed, so it can
        surface the FAILING report. Empty list if none."""
        sql = (
            "SELECT ref FROM artifacts WHERE session_id=? AND kind=? "
            "ORDER BY created_at DESC"
        )
        params: tuple = (session_id, kind)
        if limit is not None:
            sql += " LIMIT ?"
            params = (session_id, kind, int(limit))
        with self._lock:
            return [r[0] for r in self._db.execute(sql, params).fetchall()]

    def close(self) -> None:
        """Close the underlying connection. / 关闭底层连接。"""
        self._db.close()
