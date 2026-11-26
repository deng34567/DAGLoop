"""AutoResearch localhost live viewer + controller.

A no-Redis web app to WATCH and DRIVE real research sessions in the browser. Up to
``AUTORESEARCH_MAX_SESSIONS`` (default 4) run CONCURRENTLY (Fix C P2):

- An ``EnginePool`` holds N single-session ``Engine`` workers. Each worker owns one
  asyncio loop, an in-memory ``DeliveringBus``, an ``OrchestratorEngine`` and all
  agents (incl. MOSS), and runs ONE session's cascades SERIALLY -- the per-session
  execution model is unchanged; parallelism comes from running N workers on N threads.
- A per-worker ``WebGateway`` taps ``"*"`` and appends every event (tagged with its
  sid) to the shared SSE feed; readers filter by sid.
- FastAPI (main loop) serves the static page, an SSE stream, state, artifacts and
  POST endpoints for human input -- each resolves the owning worker by ``sid`` and
  schedules a publish on that worker's loop.

Cross-worker safety (the "别烧机器" guarantee) is enforced by two global primitives:
a bounded LLM-concurrency semaphore (``AUTORESEARCH_LLM_MAX_CONCURRENCY``) and a GPU
run lock (gpu-worker-02 card6, one training at a time). LLM = real OpenRouter/Kimi when
``OPENROUTER_KEY`` is set.

Run:  uv run --locked --extra sandbox --env-file .env python -m autoresearch.web.server
Then open http://127.0.0.1:8300
"""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse,
)

from autoresearch.web import auth

from autoresearch.agents._reuse.deep_research_backend import MultiSourceBackend
from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.coder_agent.service import build_coder_service
from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.agents.data_analysis_agent.service import build_data_analysis_service
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deliberation_agent.service import build_deliberation_service
from autoresearch.agents.evaluator_agent.service import build_evaluator_service
from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
from autoresearch.agents.experiment_planner_agent.service import build_experiment_planner_service
from autoresearch.agents.hypothesis_agent.service import build_hypothesis_service
from autoresearch.agents.knowledge_index_agent.service import build_knowledge_index_service
from autoresearch.agents.memory_agent.service import build_memory_service
from autoresearch.agents.moss_agent.service import build_moss_service
from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.method_agent.service import build_method_service
from autoresearch.agents.ranking_judge_agent.service import build_ranking_judge_service
from autoresearch.agents.report_agent.service import build_report_service
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.llm.client import HKGAIClient
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.orchestrator.user_decision_matrix import load_user_decision_matrix, UserDecisionGuard
from autoresearch.spec_store import SpecStore, refs as spec_refs

logger = logging.getLogger(__name__)

# Read-only guard for the viewer to render the legal decision buttons per reason.
_UD_GUARD = UserDecisionGuard(load_user_decision_matrix())
# Human-readable labels for the decision actions (fallback = the raw action string).
_ACTION_LABELS = {
    "continue_with_caveat": "带 caveat 继续(接受现状 → 建模)",
    "deepen_literature": "深挖文献(重新检索)",
    "human_literature_seed": "人工提供核心论文",
    "proceed_out_of_scope": "承认超范围,继续(转湿实验/理论)",
    "narrow_to_verifiable": "缩小到可验证的子问题",
    "change_benchmark": "换 benchmark",
    "use_available_artifact": "用现有产物继续",
    "write_replication_failure_report": "写复现失败报告",
    "discard_result": "丢弃该结果",
    "strict_rerun": "严格重跑",
    "accept_with_caveat": "带 caveat 接受 → 出报告",
    "inject_idea": "注入一个想法",
    "broaden_search": "扩大搜索",
    "narrow_scope": "缩小范围",
    "revise_method": "方法与可编辑范围冲突,回方法层重设计",
    "modify_goal": "修改研究目标",
    "abandon": "放弃(结束为 Failed)",
}
from autoresearch.testing.delivering_bus import DeliveringBus

_HERE = Path(__file__).resolve().parent
_SM_ROOT = _HERE.parent / "orchestrator" / "state-machine"
_STATIC = _HERE / "static"
_WORKDIR = Path(os.environ.get("AUTORESEARCH_WEB_WORKDIR", "/tmp/autoresearch-web"))

MODEL = os.environ.get("KIMI_MODEL", "moonshotai/kimi-k2")
KEY = os.environ.get("OPENROUTER_KEY")
# LLM endpoint is configurable so we can point at any OpenAI-compatible channel (OpenRouter,
# HKGAI eap, etc.) without a code change; defaults to OpenRouter for back-compat.
LLM_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# ---------------------------------------------------------------------------
# Shared event feed (engine thread appends; web loop polls)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_EVENTS: list[dict] = []          # chronological feed for ALL sessions; each event tagged with sid
_T0 = time.monotonic()
_START_REQUESTS: dict[str, dict] = {}
_STARTING_SID = "__starting__"    # sentinel: a pool worker reserved for a not-yet-booted session


def _feed_append(sid: Optional[str], emitter: str, event_type: str, payload: dict) -> None:
    with _LOCK:
        _EVENTS.append({
            "seq": len(_EVENTS),
            "sid": sid,
            "t": round(time.monotonic() - _T0, 2),
            "emitter": emitter,
            "event_type": event_type,
            "payload": payload,
        })


# --- P1 experiment liveness (cached ~5s) ------------------------------------
# The P1 general experiment runs the whole ~20-min Docker experiment INLINE
# inside RunningExperiment's cascade, and DeliveringBus commits that transition
# only AFTER it returns -- so the DB (and the viewer) show the PRE-transition
# state (ImplementingExperiment) the whole time. A live OpenHands experiment
# container is the signal that this is a healthy run-in-progress, NOT a stall:
# used by the viewer ("实验运行中" badge) and the watchdog (never reclaim a busy
# experiment). Cached so a fast /api/state poll does not shell docker each time.
_CONTAINER_CACHE = {"t": -1e9, "live": False}
_CONTAINER_TTL = 5.0


def _p1_container_live() -> bool:
    # Remote repository runs are tracked by this worker's active-job counter.
    # A local Docker scan cannot identify their container and, on a shared host,
    # would mistake another user's OpenHands agent-server for this session.
    if os.environ.get("AUTORESEARCH_REPO_REMOTE") == "1":
        _CONTAINER_CACHE.update(t=time.monotonic(), live=False)
        return False
    now = time.monotonic()
    if now - _CONTAINER_CACHE["t"] < _CONTAINER_TTL:
        return _CONTAINER_CACHE["live"]
    live = False
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=agent-server", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        live = bool(out.stdout.strip())
    except Exception:  # noqa: BLE001 -- no docker / timeout => not live (safe default)
        live = False
    _CONTAINER_CACHE.update(t=now, live=live)
    return live


def _latest_clarify_qid(sid: str) -> Optional[str]:
    """The newest open clarification question_id for ``sid`` (for the watchdog's
    conservative auto-finalize when a human never answers)."""
    with _LOCK:
        for e in reversed(_EVENTS):
            if e.get("sid") == sid and e["event_type"] == "problem_modeling.clarification_question":
                return (e.get("payload") or {}).get("question_id")
    return None


_TOPICS_FILE = _WORKDIR / "topics.json"
_TOPICS_LOCK = threading.Lock()


def _record_topic(sid: str, topic: str) -> None:
    """Remember a session's topic (v2 home-page list). Best-effort JSON file."""
    if not sid:
        return
    try:
        with _TOPICS_LOCK:
            _WORKDIR.mkdir(parents=True, exist_ok=True)
            data = _topics()
            data[sid] = {"topic": topic, "at": time.time()}
            temp = _TOPICS_FILE.with_suffix(".tmp")
            temp.write_text(json.dumps(data))
            temp.replace(_TOPICS_FILE)
    except Exception:  # noqa: BLE001
        pass


def _topics() -> dict:
    try:
        return json.loads(_TOPICS_FILE.read_text()) if _TOPICS_FILE.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _all_web_dbs() -> list:
    """Every worker's web.db path (each worker owns one). Falls back to the single default db."""
    try:
        return [w.db_path for w in ENGINE.workers]
    except Exception:  # noqa: BLE001 -- pool not built yet
        return [str(_WORKDIR / "web.db")]


def _read_state_data(sid: str) -> Optional[dict]:
    """Find a session's state_data by scanning every worker db (worker-agnostic reads)."""
    import sqlite3
    for p in _all_web_dbs():
        try:
            conn = sqlite3.connect(str(p), timeout=3)
            row = conn.execute("SELECT state_data FROM session_state WHERE session_id=?",
                               (sid,)).fetchone()
            conn.close()
        except Exception:  # noqa: BLE001
            continue
        if row:
            try:
                return json.loads(row[0])
            except Exception:  # noqa: BLE001
                return None
    return None


def _list_sessions() -> list:
    """All sessions across every worker db, enriched with topic, newest-first (v2 home page)."""
    import sqlite3
    topics = _topics()
    rows = []
    for p in _all_web_dbs():
        try:
            conn = sqlite3.connect(str(p), timeout=3)
            rows += conn.execute("SELECT session_id, state_data FROM session_state").fetchall()
            conn.close()
        except Exception:  # noqa: BLE001
            continue
    out = []
    for sid, sd in rows:
        try:
            d = json.loads(sd)
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "session_id": sid,
            "current_state": d.get("current_state"),
            "created_at": d.get("created_at"),
            "topic": (topics.get(sid) or {}).get("topic") or "(untitled)",
            "final_report_ref": d.get("final_report_ref"),
            "completion_verified": d.get("completion_verified") is True,
        })
    for request in list(_START_REQUESTS.values()):
        w = request["worker"]
        if request.get("session_id") is None and not request.get("error") and getattr(w, "_start_pending", False):
            if not w.session_id or w.session_id == _STARTING_SID:
                out.append({"session_id": request["request_id"], "current_state": "Starting",
                            "created_at": None, "topic": request["inputs"]["topic"],
                            "final_report_ref": None, "completion_verified": False})
    out.sort(key=lambda s: s.get("created_at") or 0, reverse=True)
    return out


def _pending_question(sid: str) -> Optional[dict]:
    """The latest clarification question for ``sid`` that has NOT yet been answered -- so the
    viewer can render it via normal state polling, robust to the SSE event stream
    being buffered/blocked by a tunnel or proxy (the reported '第一步卡死')."""
    with _LOCK:
        answered: set = set()
        latest = None
        for e in _EVENTS:
            if e.get("sid") != sid:
                continue
            et = e["event_type"]
            p = e.get("payload") or {}
            if et == "human.clarification_answer":
                answered.add(p.get("question_id"))
            elif et == "problem_modeling.clarification_question":
                latest = p
    if latest is None:
        try:
            draft = ENGINE.store.get(spec_refs.encode("research_spec_draft", sid, "v1"))
        except KeyError:
            return None
        latest = draft.get("pending_question")
        answered.update(item.get("question_id") for item in draft.get("answered", []))
    if latest and latest.get("question_id") not in answered:
        return latest
    return None


# ---------------------------------------------------------------------------
# Engine thread: owns the bus, engine, all agents. Serial.
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, db_path: Optional[str] = None) -> None:
        # each pool worker gets its OWN web.db so a session's write transaction (held open across
        # its clarify/experiment LLM cascade) can't lock another worker's writes.
        self.db_path = db_path or str(_WORKDIR / "web.db")
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.bus: Optional[DeliveringBus] = None
        self.store: Optional[SpecStore] = None
        self.eng: Optional[OrchestratorEngine] = None
        self.session_id: Optional[str] = None
        self.mc_result = None
        self.gen_result = None
        self._web_store = None
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self.watchdog = None
        self._bundle_ready_sid = None
        self._repo_cand_sid = None
        self._repo_cand = None
        self._repo_cand_lock = threading.Lock()
        self._active_jobs = 0
        self._active_jobs_lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError(f"engine worker did not become ready within 30s: {self.db_path}")
        if self._startup_error is not None:
            raise RuntimeError(f"engine worker setup failed: {self.db_path}") from self._startup_error
        # MOSS watchdog: a stall-recovery safety net so a driven session never
        # hangs. Runs on its own thread (survives an engine-loop block), reads
        # state from sqlite, and injects the most CONSERVATIVE recovery on a
        # genuine stall (a busy P1 experiment is never touched). Default ON;
        # AUTORESEARCH_WATCHDOG=0 disables.
        from autoresearch.orchestrator.moss_watchdog import (
            DEFAULT_SESSION_TIMEOUT, DEFAULT_TIMEOUTS, MossWatchdog, watchdog_enabled,
        )
        if watchdog_enabled():
            # "Time limit for a human to respond": with no answer before this,
            # MOSS steps in (conservative auto-answer). Applies to the human-wait
            # states only; agent-work / experiment ceilings stay generous so a
            # slow-but-progressing step is never preempted.
            timeouts = dict(DEFAULT_TIMEOUTS)
            human_t = os.environ.get("AUTORESEARCH_WATCHDOG_HUMAN_TIMEOUT")
            if human_t:
                # These states WAIT FOR A HUMAN (clarify answer, idea checkpoint, decision,
                # external result, and -- with GEN_REVIEW_DESIGN=1 -- the experiment-plan
                # review). They must get the generous human timeout, else the watchdog
                # mistakes a human still reviewing for a stalled agent and escalates.
                for _s in ("ClarifyingResearchGoal", "HumanIdeaCheckpoint", "ConfirmingMethod",
                           "AwaitingUserDecision", "AwaitingExternalExperimentResult",
                           "ReviewingExperimentPlan"):
                    timeouts[_s] = float(human_t)
            self.watchdog = MossWatchdog(
                read_state=self._watchdog_read,
                legal_actions_for=_UD_GUARD.legal_actions,
                latest_qid=lambda: _latest_clarify_qid(self.session_id),
                is_busy=self._watchdog_is_busy,
                inject=self._watchdog_inject,
                timeouts=timeouts,
                session_timeout=float(os.environ.get(
                    "AUTORESEARCH_WATCHDOG_SESSION_TIMEOUT", str(DEFAULT_SESSION_TIMEOUT))),
                poll_interval=float(os.environ.get("AUTORESEARCH_WATCHDOG_POLL", "30")),
            )
            self.watchdog.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            loop.run_until_complete(self._setup())
        except BaseException as exc:  # surface setup/config failures to the Web startup thread
            self._startup_error = exc
            self._ready.set()
            logger.exception("engine worker setup failed for %s", self.db_path)
            loop.close()
            return
        self._ready.set()
        loop.run_forever()

    def _make_llm(self):
        if not KEY:
            raise RuntimeError("OPENROUTER_KEY not set -- real LLM required for the viewer")
        # Internal web-agent LLM calls (evaluate/verify/design/critic/novelty/method/report/...) are
        # SHORT reasoning -- a few hundred tokens. Force them NON-streaming by default: streaming
        # flash-via-HKGAI calls STALL in httpx iter_raw under concurrent load (py-spy live: a
        # session with a real new_best value sat in VerifyingBestResult for hours, its verify chat
        # stuck reading stream chunks), whereas a non-streamed short reply returns promptly (well
        # under the ~180s http timeout). The SANDBOX agentic experiment uses its OWN streaming
        # client (long agent conversations that WOULD hit the gateway's 360s non-stream cap), so it
        # is unaffected. AUTORESEARCH_WEB_LLM_STREAM=1 restores streaming (only needed if a slow
        # model makes a single internal call exceed the non-stream timeout).
        _web_stream = os.environ.get("AUTORESEARCH_WEB_LLM_STREAM", "0") == "1"
        # Non-stream calls block on ONE http request; a slower model (glm) needs a bigger ceiling than
        # flash. 老师 2026-08-05: 非流式就别设太短的超时 -- 给足时间 + 兜底重试(见 HKGAIClient 的
        # non-stream retry)。DEFAULT ENLARGED 180 -> 300s (safely under the gateway's ~360s non-stream
        # body cap; these internal calls are short-reasoning so 300s is plenty of headroom). Still
        # env-tunable (set higher only with streaming, since >360s non-stream is cut by the gateway).
        _web_to = float(os.environ.get("AUTORESEARCH_WEB_LLM_TIMEOUT", "300"))
        # DNS may return more than one address and an individual route can be black-holed.  Do not
        # spend the whole response budget trying to establish one TCP connection; a short connect
        # timeout lets httpx fall through to the next address while preserving the generous read
        # budget needed by slower models.
        _connect_to = float(os.environ.get("AUTORESEARCH_WEB_LLM_CONNECT_TIMEOUT", "10"))
        _default_max_tokens_raw = (
            os.environ.get("AUTORESEARCH_WEB_LLM_MAX_TOKENS")
            or os.environ.get("AUTORESEARCH_LLM_MAX_TOKENS")
        )
        _default_max_tokens = (
            int(_default_max_tokens_raw) if _default_max_tokens_raw is not None else None
        )
        return HKGAIClient(
            api_key=KEY, base_url=LLM_BASE_URL, model=MODEL,
            response_format_mode=None, stream=_web_stream,
            default_max_tokens=_default_max_tokens,
            http_client=httpx.Client(
                trust_env=False, timeout=httpx.Timeout(_web_to, connect=_connect_to)),
        )

    async def _restore_persisted_session_binding(self, conn) -> None:
        """Rebind this worker to its sole unfinished session after a Web restart."""
        cursor = await conn.execute(
            "SELECT session_id, current_state FROM session_state "
            "ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        unfinished = [
            str(session_id)
            for session_id, current_state in rows
            if current_state not in ("Done", "Failed")
        ]
        if len(unfinished) == 1:
            self.session_id = unfinished[0]
        elif len(unfinished) > 1:
            logger.error(
                "worker database has multiple unfinished sessions; "
                "refusing automatic binding: %s",
                self.db_path,
            )

    async def _setup(self) -> None:
        if not any(os.environ.get(key) == "1" for key in (
            "AUTORESEARCH_GEN_EXPERIMENT", "AUTORESEARCH_AGENTIC_IMPROVE",
            "AUTORESEARCH_REAL_RUNNER",
        )):
            raise RuntimeError(
                "Configure real execution before starting the controller: "
                "set AUTORESEARCH_GEN_EXPERIMENT=1 in .env and load it at startup."
            )
        _WORKDIR.mkdir(parents=True, exist_ok=True)
        db = self.db_path
        # v2: PERSIST session history across restarts so the home page can list past
        # sessions (their state rows + store artifacts survive). AUTORESEARCH_WEB_FRESH=1
        # restores the old wipe-each-start behavior for a clean slate.
        if os.environ.get("AUTORESEARCH_WEB_FRESH") == "1":
            for suffix in ("", "-wal", "-shm"):
                try:
                    Path(db + suffix).unlink()
                except OSError:
                    pass
        conn = await aiosqlite.connect(db)
        # WAL + busy_timeout so multiple concurrent-session workers (Fix C P2) can write web.db
        # (session_state) without "database is locked"; readers (web/watchdog threads) never block.
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        self.bus = DeliveringBus(conn)
        self.store = SpecStore(_WORKDIR / "store")
        self.eng = OrchestratorEngine(
            client=self.bus.view("ResearchOrchestrator"), skill_root=_SM_ROOT, spec_store=self.store)
        await self.eng.start()
        await self._restore_persisted_session_binding(conn)

        # gateway: tap every event onto the feed + adopt the session id. Subscribed FIRST (before
        # the agent services) so on start_session's first event it runs BEFORE the blocking clarify
        # LLM call -- the endpoint sees the real sid in <1s instead of waiting out the ~9s cascade.
        async def gateway(env):
            sid = getattr(env, "session_id", None)
            if sid and self.session_id in (None, _STARTING_SID):
                self.session_id = sid
                startup = getattr(self, "_start_request", None)
                if startup and getattr(self, "_start_pending", False):
                    startup["session_id"] = sid
                    _record_topic(sid, startup["inputs"]["topic"])
            _feed_append(sid, getattr(env, "emitter", "?"), env.event_type, dict(env.payload or {}))
        self.bus.view("WebGateway").subscribe("*", gateway)

        llm = self._make_llm()
        store = self.store
        def bv(name):
            return self.bus.view(name)
        build_problem_modeling_service(
            client=bv("ProblemModelingAgent"), spec_store=store, llm_client=llm,
            semantic_guardian_llm=llm,
        )
        build_method_service(client=bv("MethodAgent"), spec_store=store, llm_client=llm)  # phase 3.1
        # Opt-in real paper research. Prefer key-backed Tavily search + bounded fetch; the key may
        # be supplied directly or read from a deployment-selected env file. Never copy secrets into
        # this repository. Keyless academic indexes remain a fail-open fallback.
        if os.environ.get("AUTORESEARCH_REAL_SEARCH") == "1":
            from autoresearch.agents._reuse.deep_research_backend import (
                ArxivBackend, CrossrefBackend, FailoverBackend, OpenAlexBackend,
                SemanticScholarBackend, TavilyBackend)
            tavily_key = str(os.environ.get("TAVILY_API_KEY") or "").strip()
            env_file = str(os.environ.get("AUTORESEARCH_DEEP_RESEARCH_ENV_FILE") or "").strip()
            if not tavily_key and env_file:
                try:
                    from dotenv import dotenv_values
                    tavily_key = str(dotenv_values(env_file).get("TAVILY_API_KEY") or "").strip()
                except Exception:
                    logger.warning("could not read deep-research env file; using keyless fallback")
            sources = [OpenAlexBackend(per_page=8), ArxivBackend(per_page=6),
                       CrossrefBackend(per_page=6)]
            if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
                sources.append(SemanticScholarBackend(
                    per_page=6, api_key=os.environ["SEMANTIC_SCHOLAR_API_KEY"]))
            academic_fallback = MultiSourceBackend(sources)
            if tavily_key:
                dr_backend = FailoverBackend(
                    TavilyBackend(
                        api_key=tavily_key,
                        per_page=int(os.environ.get("AUTORESEARCH_TAVILY_RESULTS", "6")),
                        fetch_cap=int(os.environ.get("AUTORESEARCH_TAVILY_FETCH_CAP", "12")),
                    ),
                    academic_fallback,
                )
            else:
                dr_backend = academic_fallback
        else:
            dr_backend = MultiSourceBackend([])
        build_deep_research_service(client=bv("DeepResearchAgent"), spec_store=store, backend=dr_backend, llm_client=llm)
        build_knowledge_index_service(client=bv("KnowledgeIndexAgent"), spec_store=store)
        build_evidence_audit_service(client=bv("EvidenceAuditAgent"), spec_store=store, llm_client=llm)
        build_novelty_audit_service(client=bv("NoveltyAuditAgent"), spec_store=store, llm_client=llm)
        build_critic_service(client=bv("CriticAgent"), spec_store=store, llm_client=llm)
        # Bounded deep-deliberation step (Deliberating): N-angle cross-reference before
        # SelectingBenchmark. Fail-open, so a missing/slow LLM never stalls the flow.
        build_deliberation_service(client=bv("DeliberationAgent"), spec_store=store, llm_client=llm)
        build_ranking_judge_service(client=bv("RankingJudgeAgent"), spec_store=store, llm_client=llm)
        build_evaluator_service(client=bv("EvaluatorAgent"), spec_store=store, llm_client=llm)
        # Repository/general sessions execute through their sandbox branches. The
        # optional toy runner executes the bundled benchmark in a local subprocess.
        runner_backend = None
        if os.environ.get("AUTORESEARCH_REAL_RUNNER") == "1":
            from autoresearch.agents._reuse.runner_backend import LocalSubprocessRunnerBackend
            toy = _HERE.parent / "benchmarks" / "toy_pairwise"
            runner_backend = LocalSubprocessRunnerBackend(toy, timeout=60)
        # General honest-experiment integration: with AUTORESEARCH_GEN_EXPERIMENT=1 the
        # RunnerAgent gen-branch runs the REAL P1 pipeline (agent writes the whole
        # experiment in Docker) at RunningExperiment, so a full "开始研究" session runs
        # the real experiment end-to-end → verdict → report → Done. Default = off.
        p1_backend = None
        # Repository discovery also needs the session LLM when the user names a project without
        # supplying its GitHub URL.  Passing the client does not enable the general P1 branch;
        # `_gen_branch()` still requires both its backend and AUTORESEARCH_GEN_EXPERIMENT=1.
        p1_llm = llm
        if os.environ.get("AUTORESEARCH_GEN_EXPERIMENT") == "1":
            from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend
            p1_backend = OpenHandsBackend()
            # Web GEN sessions pause at ReviewingExperimentPlan for a HUMAN design
            # review by default (overridable: AUTORESEARCH_GEN_REVIEW_DESIGN=0).
            os.environ.setdefault("AUTORESEARCH_GEN_REVIEW_DESIGN", "1")
        build_runner_service(client=bv("RunnerAgent"), spec_store=store, backend=runner_backend,
                             p1_backend=p1_backend, p1_llm=p1_llm,
                             repository_search_backend=(dr_backend if os.environ.get(
                                 "AUTORESEARCH_REAL_SEARCH") == "1" else None))
        build_hypothesis_service(client=bv("HypothesisAgent"), spec_store=store, llm_client=llm)
        build_experiment_planner_service(client=bv("ExperimentPlannerAgent"), spec_store=store, llm_client=llm)
        build_coder_service(client=bv("CoderAgent"), spec_store=store, llm_client=llm)
        build_memory_service(client=bv("MemoryAgent"), spec_store=store, llm_client=llm)
        build_report_service(client=bv("ReportAgent"), spec_store=store, llm_client=llm)
        build_moss_service(client=bv("MOSS"), spec_store=store, llm_client=llm)
        build_data_analysis_service(client=bv("DataAnalysisAgent"), spec_store=store, llm_client=llm)


    def submit(self, coro_factory) -> Any:
        """Schedule a coroutine on the engine loop; FIRE-AND-FORGET (do NOT wait
        for the cascade -- the engine loop is blocked running the current cascade
        of real LLM calls; the scheduled coro simply queues and runs when free)."""
        with self._active_jobs_lock:
            self._active_jobs += 1

        async def _tracked():
            try:
                try:
                    return await coro_factory()
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    for attempt in range(2):
                        await asyncio.sleep(attempt + 1)
                        try:
                            if not await self.bus.replay_pending(self.session_id):
                                raise exc
                            return
                        except sqlite3.OperationalError:
                            if attempt == 1:
                                raise
            finally:
                with self._active_jobs_lock:
                    self._active_jobs -= 1

        try:
            future = asyncio.run_coroutine_threadsafe(_tracked(), self.loop)
            future.add_done_callback(self._log_background_failure)
            return future
        except BaseException:
            with self._active_jobs_lock:
                self._active_jobs -= 1
            raise

    def _log_background_failure(self, future) -> None:
        """Surface exceptions from fire-and-forget engine jobs."""
        if future.cancelled():
            return
        try:
            exc = future.exception()
        except BaseException:
            logger.exception("failed to inspect background engine job")
            return
        if exc is not None:
            logger.error(
                "background engine job failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _watchdog_is_busy(self, state: str) -> bool:
        """Report queued/running work owned by this session worker."""
        with self._active_jobs_lock:
            active = self._active_jobs > 0
        # A host-wide ``docker ps`` cannot attribute an agent-server container to
        # this session and made unrelated/stale containers block restart recovery.
        # Remote duplicate dispatch is already prevented by the per-session lock.
        return active

    # ---- watchdog transport (called from the watchdog thread) ----

    def _watchdog_read(self) -> Optional[dict]:
        """Read the current session's state fields the watchdog needs, straight
        from sqlite (decoupled from the busy engine loop)."""
        if not self.session_id:
            return None
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            row = conn.execute(
                "SELECT state_data FROM session_state WHERE session_id=?", (self.session_id,)
            ).fetchone()
            conn.close()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        d = json.loads(row[0])
        return {
            "session_id": d.get("session_id"),
            "current_state": d.get("current_state"),
            "state_entered_at": d.get("state_entered_at"),
            "created_at": d.get("created_at"),
            "awaiting_user_decision_reason": d.get("awaiting_user_decision_reason"),
        }

    def _watchdog_inject(self, *, kind, session_id, reason, action, qid) -> None:
        """Publish the watchdog's conservative recovery event onto the engine loop
        (same fire-and-forget path the POST endpoints use)."""
        from autoresearch.orchestrator.moss_watchdog import (
            CANNED_FINALIZE, KIND_AGENT_FAILED, KIND_DECISION,
            KIND_FINALIZE_CLARIFY, KIND_IDEA_NO_INJECT, KIND_METHOD_NO_INJECT,
            KIND_RESUME_SUSPENDED,
        )
        bus = self.bus
        if kind == KIND_RESUME_SUSPENDED:
            async def _c():
                await self.eng.resume_to_current_state(session_id)
        elif kind == KIND_AGENT_FAILED:
            async def _c():
                await bus.view("MossWatchdog").publish(
                    event_type="agent.failed", session_id=session_id,
                    payload={"agent_name": "MossWatchdog", "reason_classification": "transient",
                             "underlying_reason": "MOSS watchdog: no state progress before timeout",
                             "underlying_reason_code": "watchdog_stall_timeout"})
        elif kind == KIND_DECISION:
            async def _c():
                await bus.view("Human").publish(
                    event_type="user.decision_provided", session_id=session_id,
                    payload={"decision": action})
        elif kind == KIND_IDEA_NO_INJECT:
            payload = {"inject": False}
            repo_required = (
                os.environ.get("AUTORESEARCH_AGENTIC_IMPROVE") == "1"
                and os.environ.get("AUTORESEARCH_REPO_MODE") in ("auto", "remote", "local")
            )
            if repo_required:
                state = self.state_sync() or {"session_id": session_id}
                candidates = self._repo_candidates_sync(state)
                if candidates and candidates[0].get("url"):
                    payload["confirmed_repo_url"] = str(candidates[0]["url"]).strip()
                else:
                    async def _c():
                        await bus.view("MossWatchdog").publish(
                            event_type="agent.failed", session_id=session_id,
                            payload={
                                "agent_name": "MossWatchdog",
                                "reason_classification": "permanent",
                                "underlying_reason": (
                                    "Repository checkpoint could not resolve a concrete URL"),
                                "underlying_reason_code": "repository_discovery_failed",
                            })
                    self.submit(_c)
                    return
            async def _c():
                await bus.view("Human").publish(
                    event_type="human.idea_provided", session_id=session_id,
                    payload=payload)
        elif kind == KIND_METHOD_NO_INJECT:
            async def _c():
                await bus.view("Human").publish(
                    event_type="method.confirmed", session_id=session_id,
                    payload={"inject": False})
        elif kind == KIND_FINALIZE_CLARIFY:
            async def _c():
                await bus.view("Human").publish(
                    event_type="human.clarification_answer", session_id=session_id,
                    payload={"question_id": qid, "answer_text": CANNED_FINALIZE})
        else:
            return
        self.submit(_c)

    # --- Reads go DIRECT to sqlite on the WEB thread, decoupled from the (often
    # busy) engine loop, so /state and /artifact never block on a running cascade.
    _REF_KEYS = ("research_spec_ref", "literature_brief_ref", "spec_interpretation_ref",
                 "hypothesis_bank_ref", "novelty_audit_report_ref", "evidence_audit_report_ref",
                 "experiment_plan_refs", "research_memory_ref", "final_report_ref")

    def state_sync(self) -> Optional[dict]:
        if not self.session_id:
            return None
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            row = conn.execute(
                "SELECT state_data FROM session_state WHERE session_id=?", (self.session_id,)
            ).fetchone()
            conn.close()
        except Exception:  # noqa: BLE001
            return {"session_id": self.session_id, "current_state": None}
        if not row:
            return {"session_id": self.session_id, "current_state": None}
        d = json.loads(row[0])
        refs = {}
        for k in self._REF_KEYS:
            v = d.get(k)
            if isinstance(v, list):
                v = v[-1] if v else None
            if v:
                refs[k] = v
        return {
            "session_id": d.get("session_id"),
            "current_state": d.get("current_state"),
            "current_moss_reason": d.get("current_moss_reason"),
            "awaiting_user_decision_reason": d.get("awaiting_user_decision_reason"),
            "final_report_ref": d.get("final_report_ref"),
            "completion_verified": d.get("completion_verified") is True,
            # True while the P1 experiment runs inline (a live container); the
            # committed state still reads ImplementingExperiment, so this tells the
            # viewer it is running-in-progress, not stuck.
            "experiment_running": (
                d.get("current_state") in ("ImplementingExperiment", "RunningExperiment")
                and self._watchdog_is_busy(str(d.get("current_state")))
            ),
            # True once the P1 experiment has produced a downloadable bundle AND the
            # honest final report is out -- we only surface the deliverable to the user
            # AFTER verification+reporting (not mid-run), so a downloaded bundle always
            # corresponds to a vetted, reported result.
            "bundle_ready": d.get("current_state") == "Done" and d.get("completion_verified") is True and self.bundle_ready(),
            # The concrete P1 design awaiting HUMAN review (only while paused at
            # ReviewingExperimentPlan), so the viewer can show it for approve/redesign.
            "gen_design": (self.gen_design_sync()
                           if d.get("current_state") == "ReviewingExperimentPlan" else None),
            # The current unanswered clarification question (if any) -- the SINGLE source
            # of truth for the active question, so the viewer renders it via polling and
            # NOT from the replayed SSE history (which piled up answered questions on
            # refresh). _pending_question() already excludes answered ones; gate off only
            # at terminal states so a never-answered question can't linger after Done.
            "pending_question": (_pending_question(d.get("session_id"))
                                 if d.get("current_state") not in ("Done", "Failed") else None),
            # Auto-discovered candidate repos to confirm at the checkpoint (baseline logic:
            # literature/known repos + LLM proposal). The user picks one -> repo-improve on it;
            # "none" -> self-write. Computed once per session (cached), only while paused here.
            "repo_candidates": (self._repo_candidates_sync(d)
                                if d.get("current_state") == "HumanIdeaCheckpoint" else None),
            "refs": refs,
        }

    def _repo_candidates_sync(self, d: dict) -> list:
        """Candidate repos for the user to confirm at HumanIdeaCheckpoint -- an existing baseline
        the system found so the experiment can clone+improve it instead of self-writing. Computed
        ONCE per session (an LLM call) and cached; [] on any failure (falls through to self-write)."""
        sid = d.get("session_id")
        candidate_key = (sid, d.get("last_baseline_attempt_ref"),
                         len(d.get("moss_interventions") or []))
        # select-once: once a repo is confirmed (user picked one, or it was injected at the
        # checkpoint), don't re-offer the picker -- otherwise the checkpoint asks the user to
        # choose the SAME repo a second time. The confirmed url already drives the run.
        if d.get("confirmed_repo_url"):
            return []
        # fast path once cached (no lock contention on the hot poll path)
        if self._repo_cand_sid == candidate_key and self._repo_cand is not None:
            return self._repo_cand
        # Serialize the SLOW LLM computation: browser (1.5s) + driver + SSE poll concurrently, and
        # without this each in-flight poll would fire its OWN discover_candidates before the first
        # caches -- flooding the LLM, starving the engine loop, and reshuffling the candidate list.
        with self._repo_cand_lock:
            if self._repo_cand_sid == candidate_key and self._repo_cand is not None:
                return self._repo_cand
            out = self._compute_repo_candidates(sid, d)
            self._repo_cand_sid = candidate_key
            self._repo_cand = out
            return out

    def _compute_repo_candidates(self, sid: str, d: dict) -> list:
        out: list = []
        try:
            from autoresearch.spec_store import refs as _refs
            from autoresearch.experiments.repo_improve.discover import discover_candidates
            # The ORIGINAL question keeps the repo URL; the REINTERPRETED research_goal frequently
            # drops it (e.g. "...using SOL-ExecBench as the harness" with no github URL), so an
            # explicitly-named repo never surfaced as a candidate. Feed BOTH -- the original topic
            # (URL intact) + the reinterpreted goal -- so discover_candidates' explicit-URL extraction
            # still finds the repo the user named.
            orig = (_topics().get(sid) or {}).get("topic") or ""
            goal = ""
            try:
                spec = self.store.get(_refs.encode("research_spec", sid, "final"))
                goal = spec.get("research_goal") or spec.get("topic") or ""
            except Exception:  # noqa: BLE001
                pass
            question = (orig + "\n" + goal).strip()
            lit = None
            lit_ref = d.get("literature_brief_ref")
            if lit_ref:
                try:
                    lit = self.store.get(lit_ref)
                except Exception:  # noqa: BLE001
                    lit = None
            baseline_failure = None
            failure_ref = d.get("last_baseline_attempt_ref")
            if failure_ref:
                try:
                    previous = self.store.get(failure_ref)
                    if isinstance(previous, dict) and previous.get("reproduced") is False:
                        baseline_failure = previous
                except Exception:  # noqa: BLE001 -- missing evidence must not invent a failure
                    pass
            cands = discover_candidates(question, self._make_llm(), literature_brief=lit,
                                        baseline_failure=baseline_failure,
                                        max_candidates=5)
            out = [{"url": c.repo_url, "name": c.name} for c in cands]
        except Exception:  # noqa: BLE001 -- best-effort; no candidates -> self-write path
            out = []
        return out

    def session_view(self, sid: str) -> Optional[dict]:
        """Read-only state + refs for ANY session by id (v2 history view). Unlike
        state_sync it does not add live flags (experiment_running / pending_question);
        it's for viewing a past session's committed state + artifacts."""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            row = conn.execute(
                "SELECT state_data FROM session_state WHERE session_id=?", (sid,)
            ).fetchone()
            conn.close()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        d = json.loads(row[0])
        refs = {}
        for k in self._REF_KEYS:
            v = d.get(k)
            if isinstance(v, list):
                v = v[-1] if v else None
            if v:
                refs[k] = v
        return {
            "session_id": d.get("session_id"),
            "current_state": d.get("current_state"),
            "current_moss_reason": d.get("current_moss_reason"),
            "final_report_ref": d.get("final_report_ref"),
            "completion_verified": d.get("completion_verified") is True,
            "topic": (_topics().get(sid) or {}).get("topic") or "(untitled)",
            "is_active": sid == self.session_id,
            # a past/finished session serves its own bundle iff it produced one AND
            # reached a final report (same gate as the live view).
            "bundle_ready": d.get("current_state") == "Done" and d.get("completion_verified") is True and self.bundle_sync(sid) is not None,
            "refs": refs,
        }

    def artifact_sync(self, ref: str) -> Any:
        if self._web_store is None:
            self._web_store = SpecStore(_WORKDIR / "store")
        return self._web_store.get(ref)

    def gen_design_sync(self, sid: Optional[str] = None) -> Optional[dict]:
        """The concrete P1 design (task/method/baseline/metric/instances) generated at
        PlanningExperiment for HUMAN review at ReviewingExperimentPlan, or None."""
        sid = sid or self.session_id
        if not sid:
            return None
        from autoresearch.spec_store import refs
        try:
            return self.artifact_sync(refs.encode("gen_design", sid, "v1"))
        except Exception:  # noqa: BLE001 -- not a gen run / not produced yet
            return None

    def bundle_sync(self, sid: Optional[str] = None) -> Optional[dict]:
        """The downloadable Docker-runnable experiment bundle for ``sid`` (defaults to
        the current session): agent-written method.py + harness.py + trusted driver, or
        None if the P1 experiment has not produced one yet. Passing ``sid`` lets a
        finished/past session still serve its own bundle after a refresh."""
        sid = sid or self.session_id
        if not sid:
            return None
        from autoresearch.spec_store import refs
        try:
            state = _read_state_data(sid)
            if not state or state.get("current_state") != "Done" or state.get("completion_verified") is not True:
                return None
            report = self.artifact_sync(state["final_report_ref"])
            bundle = self.artifact_sync(report["delivery_bundle_ref"])
            return bundle if bundle.get("result_ref") == report.get("result_ref") else None
        except Exception:  # noqa: BLE001 -- not produced yet / not a gen run
            return None

    def bundle_ready(self) -> bool:
        """Cheap cached check (per session) that a downloadable bundle exists."""
        if not self.session_id:
            return False
        if self._bundle_ready_sid == self.session_id:
            return True
        if self.bundle_sync() is not None:
            self._bundle_ready_sid = self.session_id
            return True
        return False


_MAX_SESSIONS = max(1, int(os.environ.get("AUTORESEARCH_MAX_SESSIONS", "4")))


class EnginePool:
    """Fix C P2: a pool of single-session Engine workers. Each worker runs ONE research
    session at a time on the UNCHANGED serial engine; real concurrency comes from N worker
    threads. P1's global LLM semaphore + GPU lock coordinate them so nothing overheats."""

    def __init__(self, n: int) -> None:
        # each worker owns a separate web.db to avoid cross-session write-lock contention
        self.workers = [Engine(str(_WORKDIR / f"web_{i}.db")) for i in range(n)]
        self.n = n
        self._alloc_lock = threading.Lock()   # atomic free-worker claim (no two sessions -> 1 worker)
        # standalone experiment result holders (mc/gen endpoints; not tied to a session)
        self.mc_result = None
        self.gen_result = None

    def claim_worker(self):
        """Atomically reserve a free worker for a NEW session, so two concurrent /api/session
        calls can't both grab the same worker. Returns the worker (with a sentinel session_id) or
        None if the pool is full. The caller must set the real session_id (or release on failure)."""
        with self._alloc_lock:
            w = self.free_worker()
            if w is None:
                return None
            w.session_id = _STARTING_SID   # reserve -> a concurrent claim skips it (not None)
            return w

    def default_sid(self) -> Optional[str]:
        act = self.active_sessions()
        return act[-1] if act else None

    def start(self) -> None:
        for w in self.workers:
            w.start()

    # shared, worker-agnostic resources
    @property
    def store(self):
        return self.workers[0].store

    def _make_llm(self):
        return self.workers[0]._make_llm()

    def session_view(self, sid: str):
        for w in self.workers:   # each worker owns a separate db -> scan all
            v = w.session_view(sid)
            if v:
                return v
        return None

    def artifact_sync(self, ref: str):
        return self.workers[0].artifact_sync(ref)

    def bundle_sync(self, sid: Optional[str] = None):
        return self.workers[0].bundle_sync(sid or self.default_sid())

    # routing
    def worker_for(self, sid: Optional[str]):
        if not sid:
            return None
        for w in self.workers:
            if w.session_id == sid:
                return w
        return None

    def bind_existing_session(self, sid: str):
        """Rebind an idle worker to one of its own persisted sessions.

        Terminal sessions are intentionally not rebound during startup.  An explicit
        redrive may still need the owning worker, for example to retry Reporting from
        already-persisted ResearchMemory.  Never move a session between worker DBs and
        never displace a worker that is currently bound to another session.
        """
        with self._alloc_lock:
            existing = self.worker_for(sid)
            if existing is not None:
                return existing
            for w in self.workers:
                if w.session_id is None and w.session_view(sid) is not None:
                    w.session_id = sid
                    return w
        return None

    def default_worker(self):
        """The most-recently-active worker (back-compat for callers without a sid)."""
        live = [w for w in self.workers if w.session_id]
        return live[-1] if live else self.workers[0]

    def free_worker(self):
        """A worker ready for a NEW session: prefer an idle one, else reclaim a terminal one."""
        for w in self.workers:
            if w.session_id is None and not getattr(w, "_start_pending", False):
                return w
        for w in self.workers:
            if getattr(w, "_start_pending", False) or w.session_id == _STARTING_SID:
                continue
            cur = w.state_sync()
            if cur and cur.get("current_state") in ("Done", "Failed"):
                return w
        return None

    def active_sessions(self) -> list:
        return [w.session_id for w in self.workers
                if w.session_id and w.session_id != _STARTING_SID]

    def submit(self, coro_factory, sid: Optional[str] = None):
        w = self.worker_for(sid) or self.default_worker()
        w.submit(coro_factory)

    def state_sync(self, sid: Optional[str] = None):
        w = self.worker_for(sid)
        if w:
            return w.state_sync()
        if sid is None:
            dw = self.default_worker()
            return dw.state_sync() if dw.session_id else None
        # session was reclaimed (all slots busy) -> serve its committed history read-only
        return self.session_view(sid)


ENGINE = EnginePool(_MAX_SESSIONS)
app = FastAPI(title="AutoResearch Live Viewer")

# Optional shared-token gate for when the viewer is bound to 0.0.0.0 (LAN-exposed).
# If AUTORESEARCH_WEB_TOKEN is set, every request must carry ?token=<t> (once, on the
# first page load -> sets an httponly cookie) or the ar_token cookie; else 401. Unset
# = no auth (localhost mode). NOTE: token travels over plain HTTP -- LAN-demo grade,
# not real security.
_WEB_TOKEN = os.environ.get("AUTORESEARCH_WEB_TOKEN")

# The full login system (opt-in via AUTORESEARCH_WEB_AUTH=1): username/password login,
# an admin account, and self-registration gated behind admin approval. When it is OFF,
# the viewer keeps its previous behaviour -- open localhost, or the legacy shared-token
# gate below -- so existing dev/test flows are unchanged.
app.include_router(auth.router)


@app.middleware("http")
async def _gate(request, call_next):
    # 1) Full login system.
    if auth.AUTH_ENABLED:
        path = request.url.path
        if auth.is_public_path(path) or auth.is_authenticated(request):
            return await call_next(request)
        # Unauthenticated: send a browser page load to the login page; 401 everything else.
        wants_html = request.method == "GET" and "text/html" in request.headers.get("accept", "")
        if wants_html:
            return RedirectResponse("/login", status_code=302)
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    # 2) Legacy shared-token gate (LAN demo grade; unset = open localhost).
    if not _WEB_TOKEN:
        return await call_next(request)
    q = request.query_params.get("token")
    if q != _WEB_TOKEN and request.cookies.get("ar_token") != _WEB_TOKEN:
        return JSONResponse({"error": "unauthorized: append ?token=... to the URL"},
                            status_code=401)
    resp = await call_next(request)
    if q == _WEB_TOKEN:
        resp.set_cookie("ar_token", _WEB_TOKEN, httponly=True, samesite="lax")
    return resp


@app.on_event("startup")
def _startup():
    if auth.AUTH_ENABLED:
        auth.bootstrap_admin()
    ENGINE.start()


@app.on_event("shutdown")
def _shutdown():
    # OpenHands agents are remote: killing only this local controller leaves them
    # running. Pause the conversations we own, but preserve their containers and
    # bind-mounted workspaces for an evidence-safe resume.
    from autoresearch.agents._reuse.openhands_backend import pause_active_conversations

    paused = pause_active_conversations()
    logger.info("paused %d active OpenHands conversation(s) during shutdown", paused)


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"), headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/login")
def login_page():
    return FileResponse(str(_STATIC / "login.html"), headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/admin")
def admin_page(request: Request):
    # Admin-only page (the API it calls is require_admin-gated anyway; this keeps a
    # non-admin from even loading the panel). Only meaningful when AUTH_ENABLED.
    if auth.AUTH_ENABLED:
        user = auth.get_store().resolve_token(request.cookies.get(auth.COOKIE_NAME))
        if not user or not auth.get_store().is_admin(user):
            return RedirectResponse("/", status_code=302)
    return FileResponse(str(_STATIC / "admin.html"), headers={"Cache-Control": "no-store, max-age=0"})


@app.post("/api/session")
async def start_session(body: dict):
    # Up to _MAX_SESSIONS run concurrently, one per worker. Pick a free worker; if all are
    # busy, tell the caller (the UI can wait or view an existing session) rather than queue
    # a start behind a multi-minute cascade.
    topic = (body or {}).get("topic", "").strip()
    if not topic:
        return JSONResponse({"error": "topic required"}, status_code=400)
    boundary = (body or {}).get("confirmed_boundary")
    if boundary is not None and not isinstance(boundary, dict):
        return JSONResponse({"error": "confirmed_boundary must be an object"}, status_code=400)
    request_id = str(body.get("request_id") or uuid.uuid4())
    inputs = {"topic": topic, "confirmed_boundary": boundary}
    if request_id in _START_REQUESTS:
        record = _START_REQUESTS[request_id]
        if record["inputs"] != inputs:
            return JSONResponse({"error": "request_id reused with different inputs"}, status_code=409)
        return _start_response(record)
    w = ENGINE.claim_worker()
    if w is None:
        return JSONResponse({"error": "busy", "active": ENGINE.active_sessions(), "max": ENGINE.n}, status_code=429)
    record = {"request_id": request_id, "inputs": inputs, "worker": w, "error": None, "session_id": None}
    _START_REQUESTS[request_id] = record
    w._start_request = record
    w._start_pending = True

    async def _go():
        try:
            sid = await w.eng.start_session(topic, user_id="web", **(
                {"confirmed_boundary": boundary} if boundary else {}))
            record["session_id"] = sid
            _record_topic(sid, topic)
            if w._start_request is record:
                w.session_id = sid
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if w._start_request is record and w.session_id == _STARTING_SID:
                w.session_id = None
        finally:
            if w._start_request is record:
                w._start_pending = False
    w.submit(lambda: _go())
    for _ in range(200):
        if (w.session_id and w.session_id != _STARTING_SID) or record["error"]:
            break
        await asyncio.sleep(0.1)
    # A slow startup continues to own its worker. Callers query this original request.
    return _start_response(record)


def _start_response(record):
    w = record["worker"]
    sid = record["session_id"]
    if not sid and getattr(w, "_start_request", None) is record and w.session_id != _STARTING_SID:
        sid = w.session_id
        if sid:
            record["session_id"] = sid
            _record_topic(sid, record["inputs"]["topic"])
    if sid:
        return {"session_id": sid, "request_id": record["request_id"], "model": MODEL}
    if record["error"]:
        return JSONResponse({"error": "start failed", "detail": record["error"],
                             "request_id": record["request_id"]}, status_code=500)
    return JSONResponse({"request_id": record["request_id"], "status": "starting"}, status_code=202)


@app.get("/api/session/start_status")
def start_status(request_id: str):
    record = _START_REQUESTS.get(request_id)
    if record is None:
        return JSONResponse({"error": "no such startup request"}, status_code=404)
    return _start_response(record)


@app.get("/api/sessions")
def list_sessions():
    """All sessions (newest first) for the v2 home page."""
    return {"sessions": _list_sessions(), "active": ENGINE.active_sessions()}


@app.get("/api/session_view")
def session_view(sid: str):
    """Read-only state + refs for one session by id (history detail)."""
    v = ENGINE.session_view(sid)
    if not v:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return v


@app.post("/api/session/resume")
async def resume_session(body: dict):
    """Re-drive one persisted session in its existing worker.

    Normal UI resume remains limited to ``Suspended``.  An explicit
    ``redrive=true`` is the restart-recovery path for an idle worker whose
    persisted active state lost its in-memory entry action during a process
    restart; it is rejected while any work is still queued or running.
    """
    sid = str((body or {}).get("sid") or "").strip()
    if not sid:
        return JSONResponse({"error": "sid required"}, status_code=400)
    redrive = (body or {}).get("redrive") is True
    worker = ENGINE.worker_for(sid)
    if worker is None and redrive:
        worker = ENGINE.bind_existing_session(sid)
    if worker is None:
        return JSONResponse({"error": "no active session"}, status_code=404)
    state = worker.state_sync() or {}
    current_state = state.get("current_state")
    if current_state != "Suspended":
        if not redrive:
            return JSONResponse(
                {"error": "session is not suspended", "current_state": current_state},
                status_code=409,
            )
        if not current_state:
            return JSONResponse({"error": "session state unavailable"}, status_code=404)
    if worker._watchdog_is_busy(str(current_state)):
        return JSONResponse(
            {"error": "session worker is busy", "current_state": current_state},
            status_code=409,
        )
    async def resume_saved_handoff():
        if (current_state != "Suspended"
                and await worker.bus.replay_pending(
                    sid, restart_recovery=redrive)):
            return
        await worker.eng.resume_to_current_state(sid)

    worker.submit(resume_saved_handoff)
    return {"ok": True, "session_id": sid}


@app.post("/api/human")
async def human(body: dict):
    kind = (body or {}).get("kind")
    sid = (body or {}).get("sid") or None
    w = ENGINE.worker_for(sid) or (ENGINE.default_worker() if sid is None else None)
    if w is None or not w.session_id:
        return JSONResponse({"error": "no session"}, status_code=400)
    sid = w.session_id
    human_view = w.bus.view("Human")
    if kind == "clarify":
        payload = {"question_id": body["question_id"], "answer_text": body["answer_text"]}
        w.submit(lambda: human_view.publish(event_type="human.clarification_answer", session_id=sid, payload=payload))
    elif kind == "checkpoint":
        payload = {"inject": bool(body.get("inject", False))}
        idea = str(body.get("idea") or "").strip()
        # PERSIST the injected idea text and pass its ref, so an injected direction actually
        # reaches the engine (previously the text was dropped -> injection was a no-op).
        if payload["inject"] and idea:
            ref = ENGINE.store.put(kind="human_idea_seed", session_id=sid, key="v1",
                                   obj={"idea": idea}, emitter="Human")
            payload["human_idea_seed_ref"] = ref
        # repo-baseline confirmation: the user clicked one of the auto-discovered candidate repos.
        if body.get("confirmed_repo_url"):
            payload["confirmed_repo_url"] = str(body["confirmed_repo_url"]).strip()
        # ...OR the user typed a github repo URL into the injected idea ("none of these -> use THIS
        # repo"); honor it as the confirmed baseline even though it wasn't an offered candidate.
        elif idea:
            m = re.search(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", idea)
            if m:
                payload["confirmed_repo_url"] = m.group(0).rstrip("/.")
        # user-corrected boundary (run_cmd/metric/budget/editable...) that wins over inference.
        if isinstance(body.get("confirmed_boundary"), dict) and body["confirmed_boundary"]:
            payload["confirmed_boundary"] = body["confirmed_boundary"]
        w.submit(lambda: human_view.publish(event_type="human.idea_provided", session_id=sid, payload=payload))
    elif kind == "method":
        # phase 3.1/3.4: confirm the generated METHOD, or inject a change (-> re-derive the method).
        payload = {"inject": bool(body.get("inject", False))}
        idea = str(body.get("idea") or "").strip()
        if payload["inject"] and idea:
            payload["human_idea_seed_ref"] = ENGINE.store.put(
                kind="human_idea_seed", session_id=sid, key="v1", obj={"idea": idea}, emitter="Human")
        w.submit(lambda: human_view.publish(event_type="method.confirmed", session_id=sid, payload=payload))
    elif kind == "closeout":
        # Applied by Memory at the next completed-attempt boundary, without cancelling work.
        ENGINE.store.put(kind="human_idea_seed", session_id=sid, key="closeout-v1",
                         obj={"closeout": True, "idea": str(body.get("idea") or "").strip()},
                         emitter="Human")
    elif kind == "decision":
        payload = {"decision": body["decision"]}
        w.submit(lambda: human_view.publish(event_type="user.decision_provided", session_id=sid, payload=payload))
    else:
        return JSONResponse({"error": f"unknown kind {kind}"}, status_code=400)
    return {"ok": True}


@app.post("/api/review_design")
async def review_design(body: dict):
    """Human design review at ReviewingExperimentPlan: approve -> run the design as-is;
    reject -> back to PlanningExperiment (re-design). Mirrors the CriticAgent by storing
    a minimal critique report + emitting experiment.plan_reviewed (the event the engine's
    ReviewingExperimentPlan handler consumes)."""
    sid = (body or {}).get("sid") or None
    w = ENGINE.worker_for(sid) or (ENGINE.default_worker() if sid is None else None)
    if w is None or not w.session_id:
        return JSONResponse({"error": "no session"}, status_code=400)
    sid = w.session_id
    approved = bool((body or {}).get("approved"))
    note = (body or {}).get("note") or ""
    verdict = "pass" if approved else "fail"
    report = {"target": "experiment_plan", "verdict": verdict,
              "issues": ([] if approved else [note or "human requested a re-design"])}
    ref = ENGINE.store.put(kind="critique_report", session_id=sid,
                           key="experiment_plan-v1", obj=report, emitter="Human")
    human_view = w.bus.view("Human")
    w.submit(lambda: human_view.publish(
        event_type="experiment.plan_reviewed", session_id=sid,
        payload={"verdict": verdict, "review_report_ref": ref}))
    return {"ok": True, "verdict": verdict}


@app.get("/api/state")
def state(sid: str = None):
    return ENGINE.state_sync(sid) or {"current_state": None}


_MC_TASK = (
    "Reduce the RMSE of a Monte-Carlo integral estimate at a FIXED sample budget. "
    "Edit ONLY method.py; reach the (hidden, trusted) integrand solely through the "
    "budgeted sample_f (calling it more than n_samples times scores 0). Use importance "
    "sampling / control variates / antithetic variates."
)


def _make_mc_runner():
    """Build a real MonteCarloExperimentRunner (OpenHands container backend).
    Factored out so tests can monkeypatch it with a mock runner (no container)."""
    from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend
    from autoresearch.experiments.mc_runner import MonteCarloExperimentRunner

    return MonteCarloExperimentRunner(backend=OpenHandsBackend())


@app.post("/api/mc_experiment")
async def mc_experiment(body: dict = None):
    """Focused real MC experiment: run the trusted baseline + let a container agent
    write a variance-reduction method, scored by the trusted evaluator. Streams
    mc.* events; the full result is also available at GET /api/mc_result."""
    if not KEY:
        return JSONResponse({"error": "OPENROUTER_KEY required"}, status_code=400)

    async def _go():
        from autoresearch.experiments import mc_acceptance

        runner = _make_mc_runner()
        _feed_append(None, "Human", "mc.experiment_started", {"benchmark": "mc_variance_reduction"})
        try:
            base = runner.run_baseline()
            _feed_append(None, "RunnerAgent", "mc.baseline_done",
                         {"baseline_ratio": base.get("metric_value"),
                          "rmse_baseline": base.get("rmse_baseline")})
            exp = runner.run_experiment(_MC_TASK)
            verdict = mc_acceptance.decide(exp)
            ac = exp.get("anti_cheat") or {}
            result = {
                "baseline_rmse": base.get("rmse_baseline"),
                "experiment_rmse": exp.get("rmse_proposed"),
                "baseline_ratio": base.get("metric_value"),
                "ratio": exp.get("metric_value"),
                "reproduce_ratio": exp.get("reproduce_metric_value"),
                "agent_status": exp.get("agent_status"),
                "code": exp.get("code"),
                "verdict": verdict["verdict"],
                "accepted": verdict["accepted"],
                "reasons": verdict["reasons"],
                "anti_cheat": {
                    "over_budget": ac.get("over_budget"),
                    "hardcoded_suspected": ac.get("hardcoded_suspected"),
                    "tamper_detected": ac.get("tamper_detected"),
                    "bias": ac.get("bias"),
                },
            }
            ENGINE.mc_result = result
            _feed_append(None, "RunnerAgent", "mc.experiment_done", result)
        except Exception as e:  # noqa: BLE001
            _feed_append(None, "RunnerAgent", "mc.experiment_failed", {"error": f"{type(e).__name__}: {e}"})

    ENGINE.submit(lambda: _go())
    return {"ok": True, "note": "real MC experiment (OpenHands) started; watch the event stream / result"}


@app.get("/api/mc_result")
def mc_result():
    return getattr(ENGINE, "mc_result", None) or {}


def _run_general_pipeline(question: str, emit):
    """Build the REAL general honest-experiment pipeline (OpenHands container +
    real LLM) and run it. Factored out so tests monkeypatch it with a fake (no
    container/LLM)."""
    from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend
    from autoresearch.experiments.general import pipeline as gen_pipeline

    backend = OpenHandsBackend()
    llm = ENGINE._make_llm()
    return gen_pipeline.run_pipeline(backend, llm, question, emit=emit)


def _run_repo_improve_pipeline(question, emit):
    """Repo-improvement mode: the AI finds + clones a REAL benchmark repo (e.g. Karpathy's
    autoresearch nanoGPT 5-min speedrun), patches its train.py under a fixed budget, and a
    TRUSTED evaluator scores it. Factored out so tests monkeypatch it (no GPU/clone)."""
    import uuid as _uuid

    from autoresearch.experiments.repo_improve.gpu_pick import pick_free_card
    from autoresearch.experiments.repo_improve.pipeline import run_repo_improve
    from autoresearch.experiments.repo_improve.setup_memory import SetupMemory, DEFAULT_PATH

    llm = ENGINE._make_llm()
    # One work root PER RUN: two submissions landing together used to share
    # _WORKDIR/repo_improve and overwrite each other's clone and variant workspaces.
    workroot = str(_WORKDIR / "repo_improve" / _uuid.uuid4().hex[:12])
    # Probe for a genuinely free card instead of trusting a global env pin -- this is a
    # shared box and whichever card was free an hour ago may be someone else's now.
    device = pick_free_card()
    # Setups tier 3 has already proved are reused across runs (and across entry points:
    # the store is shared), so a repo that once cost a 20-minute sandbox discovery is
    # cheap the next time. A recall is still re-run, never trusted blindly.
    memory = SetupMemory(os.environ.get("AUTORESEARCH_SETUP_MEMORY") or DEFAULT_PATH)
    return run_repo_improve(llm, question, workroot=workroot, emit=emit, cuda_device=device,
                            setup_memory=memory)


@app.post("/api/experiment")
async def experiment(body: dict = None):
    """General honest experiment: for a USER's question, agents write a whole
    experiment (AI ruler + isolated method) as real code and run it in Docker with
    honest, isolated verification. Streams gen.* events; full result at
    GET /api/experiment_result. The question is a REQUIRED body field (never
    hardcoded)."""
    if not KEY:
        return JSONResponse({"error": "OPENROUTER_KEY required"}, status_code=400)
    question = ((body or {}).get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question required"}, status_code=400)

    # Route a "improve a real repo baseline" request (e.g. nanoGPT 5-min speedrun, or ANY
    # repo the user names) to the repo-improvement pipeline; every other question falls
    # through to the general honest experiment. Passing an LLM lets discover() dynamically
    # recognise UNregistered repos (clone + infer the boundary), not just registered ones.
    from autoresearch.experiments.repo_improve.discover import discover
    repo_bench = discover(question, ENGINE._make_llm())

    async def _go_repo_improve():
        def emit(event, payload):
            _feed_append(None, "RepoImprove", f"repro.{event}", payload)
        _feed_append(None, "Human", "repro.started", {"question": question, "benchmark": repo_bench.name})
        try:
            result = _run_repo_improve_pipeline(question, emit)
            ENGINE.gen_result = result
            _feed_append(None, "RepoImprove", "repro.done",
                         {"verdict": (result.get("verdict") or {}).get("verdict"),
                          "improved": (result.get("verdict") or {}).get("improved")})
        except Exception as e:  # noqa: BLE001
            _feed_append(None, "RepoImprove", "repro.failed", {"error": f"{type(e).__name__}: {e}"})

    async def _go():
        def emit(event, payload):
            _feed_append(None, "GeneralExperiment", f"gen.{event}", payload)

        _feed_append(None, "Human", "gen.experiment_started", {"question": question})
        try:
            result = _run_general_pipeline(question, emit)
            ENGINE.gen_result = result
            _feed_append(None, "GeneralExperiment", "gen.experiment_done",
                         {"verdict": (result.get("verdict") or {}).get("verdict"),
                          "isolated": result.get("isolated")})
        except Exception as e:  # noqa: BLE001
            _feed_append(None, "GeneralExperiment", "gen.experiment_failed",
                         {"error": f"{type(e).__name__}: {e}"})

    if repo_bench is not None:
        ENGINE.submit(lambda: _go_repo_improve())
        return {"ok": True, "mode": "repo_improve", "benchmark": repo_bench.name,
                "note": "repo-improvement started; watch the event stream / result"}
    ENGINE.submit(lambda: _go())
    return {"ok": True, "mode": "general", "note": "general honest experiment started; watch the event stream / result"}


@app.get("/api/experiment_result")
def experiment_result():
    return getattr(ENGINE, "gen_result", None) or {}


@app.get("/api/decisions")
def decisions(sid: str = None):
    # Route by sid like /api/state -- without it, state_sync() falls back to the DEFAULT
    # worker, so a specific session's decisions come back empty under multi-session concurrency.
    st = ENGINE.state_sync(sid) or {}
    reason = st.get("awaiting_user_decision_reason")
    if not reason:
        return {"reason": None, "actions": []}
    acts = _UD_GUARD.legal_actions(reason)
    return {"reason": reason, "actions": [{"action": a, "label": _ACTION_LABELS.get(a, a)} for a in acts]}


@app.get("/api/artifact")
def artifact(ref: str):
    try:
        return {"ref": ref, "obj": ENGINE.artifact_sync(ref)}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=404)


def _human_report_text(report_obj: Any, topic: str = "") -> str:
    """Render the machine ReportAgent object as a PLAIN-LANGUAGE report for a human
    reader (the report.json is for machines; this is what a person actually reads)."""
    r = report_obj if isinstance(report_obj, dict) else {}
    body = str(r.get("report_body") or "").strip()
    if body:
        return body + "\n"
    L = ["AutoResearch — 最终研究报告 / Final Research Report",
         "=" * 60, ""]
    if topic:
        L += ["【研究问题 / Question】", topic.strip(), ""]
    if r.get("summary"):
        L += ["【摘要 / Summary】", str(r["summary"]).strip(), ""]
    def _section(title, key, empty="(无 / none)"):
        vals = r.get(key) or []
        L.append(f"【{title}】")
        if isinstance(vals, list) and vals:
            L.extend(f"  {i}. {v}" for i, v in enumerate(vals, 1))
        elif isinstance(vals, list):
            L.append(f"  {empty}")
        else:
            L.append(f"  {vals}")
        L.append("")
    _section("已验证的发现 / Verified findings", "verified_findings")
    _section("负面结果 / Negative results", "negative_results")
    _section("分析局限 / Caveats", "analysis_caveats")
    _section("已知风险 / Known risks", "known_risks")
    _section("未进行的实验 / Not run (future work)", "experiment_not_run")
    ledger = r.get("experiment_ledger") or []
    if ledger:
        L.append("【实验台账 / Experiment ledger】")
        for e in ledger:
            if isinstance(e, dict):
                L.append(f"  - {e.get('experiment_id','?')}: verdict={e.get('verdict','?')}, "
                         f"metric={e.get('metric_value','?')}")
                if e.get("notes"):
                    L.append(f"      note: {e['notes']}")
        L.append("")
    L += ["-" * 60,
          "本报告为诚实报告:只把确有证据的结论列入「已验证的发现」,其余归入想法/未做/局限。",
          "This is an honest report: only evidence-backed claims are 'verified findings'."]
    return "\n".join(L)


@app.get("/api/report_download")
def report_download(sid: Optional[str] = None):
    """Download a final report independently of an experiment/Docker bundle."""
    eff_sid = sid or ENGINE.default_sid()
    state = _read_state_data(eff_sid) or {}
    report_ref = state.get("final_report_ref")
    if not report_ref:
        return JSONResponse({"error": "no final report yet"}, status_code=404)
    try:
        report = ENGINE.artifact_sync(report_ref)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=404)
    topic = (_topics().get(eff_sid) or {}).get("topic") or ""
    text = _human_report_text(report, topic)
    filename = f"autoresearch-report-{eff_sid[:8]}.md"
    return PlainTextResponse(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _bundle_with_report(sid: Optional[str]) -> Optional[dict]:
    """The bundle for ``sid`` (or active session) with a human-readable FINAL_REPORT.txt
    injected from the session's ReportAgent report (which is produced AFTER the bundle
    was sealed, so we graft it on at fetch time)."""
    b = ENGINE.bundle_sync(sid)
    if not b:
        return None
    eff_sid = sid or ENGINE.default_sid()
    try:
        d = _read_state_data(eff_sid) or {}
        rep_ref = d.get("final_report_ref")
        if rep_ref:
            rep = ENGINE.artifact_sync(rep_ref)
            topic = (_topics().get(eff_sid) or {}).get("topic") or ""
            b.setdefault("files", {})["FINAL_REPORT.txt"] = _human_report_text(rep, topic)
    except Exception:  # noqa: BLE001 -- best-effort; the machine report.json is still in the bundle
        pass
    return b


@app.get("/api/experiment_files")
def experiment_files(sid: Optional[str] = None):
    """The agent-written experiment code (method.py + harness.py + trusted driver +
    Dockerfile/README + human FINAL_REPORT.txt) for inline display / past sessions."""
    b = _bundle_with_report(sid)
    if not b:
        return JSONResponse({"error": "no experiment bundle yet"}, status_code=404)
    return b


@app.get("/api/experiment_bundle")
def experiment_bundle(sid: Optional[str] = None):
    """Download the whole deliverable as a Docker-runnable .zip (works for a past
    session via ?sid=, and includes the human-readable FINAL_REPORT.txt)."""
    import io
    import zipfile

    b = _bundle_with_report(sid)
    if not b:
        return JSONResponse({"error": "no experiment bundle yet"}, status_code=404)
    files = b.get("files") or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(f"autoresearch_experiment/{name}", content if isinstance(content, str) else str(content))
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=autoresearch_experiment.zip"},
    )


@app.get("/api/events")
async def events(sid: str = None):
    # stream this session's events (+ broadcast events tagged sid=None, e.g. standalone
    # mc/repo/gen experiments). No sid -> everything (back-compat / debugging).
    async def gen():
        cursor = 0
        while True:
            with _LOCK:
                new = _EVENTS[cursor:]
                cursor = len(_EVENTS)
            for ev in new:
                if sid is None or ev.get("sid") in (sid, None):
                    yield f"data: {json.dumps(ev)}\n\n"
            await asyncio.sleep(0.25)
    return StreamingResponse(gen(), media_type="text/event-stream")


def main():
    import logging
    import sys
    import uvicorn
    # Emit the app's own INFO logs (agent/loop tracing, e.g. [method-debate]) to stdout so a run is
    # AUDITABLE. Without this the default last-resort handler drops INFO (root stays at WARNING) and
    # every logger.info() is invisible. Give the 'autoresearch' logger its own stdout handler at
    # AUTORESEARCH_LOG_LEVEL (default INFO), independent of uvicorn's own (warning) access logs.
    _level = getattr(logging, os.environ.get("AUTORESEARCH_LOG_LEVEL", "INFO").upper(), logging.INFO)
    _root = logging.getLogger("autoresearch")
    _root.setLevel(_level)
    if not any(getattr(h, "_ar_stdout", False) for h in _root.handlers):
        _h = logging.StreamHandler(sys.stdout)
        _h._ar_stdout = True  # type: ignore[attr-defined]
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                          datefmt="%H:%M:%S"))
        _root.addHandler(_h)
        _root.propagate = False
    port = int(os.environ.get("AUTORESEARCH_WEB_PORT", "8300"))
    host = os.environ.get("AUTORESEARCH_WEB_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
