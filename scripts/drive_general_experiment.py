#!/usr/bin/env python
"""HEADLESS end-to-end driver for the AutoResearch 34-state machine.

Drives ONE research question from ``start_session`` to the terminal ``Done``
(or ``Failed``) with EVERY human-input point auto-answered, over the now-wired
P1 general honest-experiment branch (``AUTORESEARCH_GEN_EXPERIMENT=1``): the
RunnerAgent's ``experiment.run_requested`` runs a REAL experiment (an AI "ruler"
harness + isolated method) in a REAL Docker container via ``OpenHandsBackend``,
scored honestly, and the baseline-vs-method numbers flow forward into the honest
report.

This is a MANUAL, coordinator-only run -- NOT part of the pytest suite. It is
DOUBLE-gated (``AUTORESEARCH_REAL_SANDBOX=1`` + ``OPENROUTER_KEY``), single-
session, SERIAL, no Redis, no concurrency: one in-process ``DeliveringBus`` owns
the engine + all 16 agents, and the DeliveringBus delivers every publish
SYNCHRONOUSLY, so each human answer runs the whole cascade of real LLM/Docker
work to the NEXT human-wait (or terminal) before returning.

Run (from the repo root):

    AUTORESEARCH_REAL_SANDBOX=1 OPENROUTER_KEY=sk-... \
        python scripts/drive_general_experiment.py \
        "Can a simple heuristic beat First-Fit on synthetic 1D bin packing?"

Optional env:
    AUTORESEARCH_MODEL           OpenRouter model (default moonshotai/kimi-k2)
    AUTORESEARCH_DRIVE_TIMEOUT   overall wall-clock budget, seconds (default 3600)
    AUTORESEARCH_MAX_CLARIFY     max clarification answers before we stop (default 6)

Prints a readable morning log: one line per auto-answer (state/reason -> action),
then the final ``current_state`` and, if a report was produced, the full honest
report (the P1 baseline-vs-method numbers appear in verified_findings /
negative_results). Any exception prints a diagnostic (state + last events) and
exits non-zero -- it never hangs silently.

NOTE (version compat): if the container handshake fails, bump
AUTORESEARCH_SANDBOX_IMAGE to a tag matching the host openhands SDK and rerun.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

# Clear proxy env BEFORE any httpx/docker client is built: a local clash proxy
# breaks host<->docker and in-container OpenRouter calls (same reason vibe's
# clear_proxy_env exists).
for _var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_var, None)

# Allow running directly from the repo root (scripts/ parent).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DEFAULT_QUESTION = (
    "Can a simple heuristic beat First-Fit on synthetic 1D bin packing "
    "(capacity 1.0, item sizes ~ U(0,0.7)) measured by number of bins used?"
)

# A firm canned answer to EVERY clarification question: stop asking, finalize
# now, use best judgment. Keeps the front-half clarify loop bounded.
CANNED_FINALIZE = (
    "Use your best judgment. Do NOT ask any more questions. Finalize the "
    "research spec NOW with reasonable defaults and proceed."
)

MODEL = os.environ.get("AUTORESEARCH_MODEL") or os.environ.get("KIMI_MODEL", "moonshotai/kimi-k2")
DRIVE_TIMEOUT = float(os.environ.get("AUTORESEARCH_DRIVE_TIMEOUT", "3600"))
MAX_CLARIFY = int(os.environ.get("AUTORESEARCH_MAX_CLARIFY", "6"))
# How many consecutive no-progress polls (same state, nothing published) before
# we treat the session as stuck and stop.
NO_PROGRESS_LIMIT = 5
POLL_INTERVAL = 1.0


class _EchoRunnerBackend:
    """Demo benchmark RunnerBackend (NO execution) -- COPIED from
    web/server.py so we don't import the FastAPI app. Echoes the benchmark's
    reported baseline metric so baseline reproduction passes (instead of looping
    on a fixed-vs-real mismatch) and returns a small improvement for the
    benchmark experiment. The REAL work in this driver is the P1 general
    honest-experiment branch (OpenHands), routed via p1_backend."""

    def __init__(self) -> None:
        self._baseline = 0.85

    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        rep = ((benchmark_spec or {}).get("baseline") or {}).get("reported_metric")
        m = float(rep) if rep is not None else 0.85
        self._baseline = m
        return {"metric_value": m, "environment": "py3.11/cpu",
                "command": "python baseline.py", "seed": "0", "runtime_seconds": 1.0,
                "exit_code": 0, "log_excerpt": "baseline reproduced (echo demo backend)."}

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        return {"metric_value": self._baseline * 1.1 + 0.01, "environment": "py3.11/cpu",
                "seed": "0", "runtime_seconds": 1.0, "exit_code": 0,
                "stdout": "experiment improved (echo demo backend).", "stderr": None}


def _make_llm():
    # Match web/server.py::_make_llm: read the endpoint/model from env (default openrouter.ai/kimi)
    # so the driver uses the SAME orchestration LLM as the web flow (e.g. flash via the HKGAI
    # endpoint). Non-streaming short-reasoning calls (streaming flash-via-HKGAI stalls in httpx
    # under load); 300s ceiling (safely under the gateway ~360s non-stream cap) + client retry.
    import httpx
    from autoresearch.llm.client import HKGAIClient

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    _stream = os.environ.get("AUTORESEARCH_WEB_LLM_STREAM", "0") == "1"
    _to = float(os.environ.get("AUTORESEARCH_WEB_LLM_TIMEOUT", "300"))
    return HKGAIClient(
        api_key=os.environ["OPENROUTER_KEY"],
        base_url=base_url,
        model=MODEL,
        response_format_mode=None,
        stream=_stream,
        http_client=httpx.Client(trust_env=False, timeout=_to),
    )


async def _build(bus, store, sm_root):
    """Assemble the engine + all 16 agents on the shared bus (adapted from
    web/server.py::Engine._setup)."""
    from autoresearch.agents._reuse.deep_research_backend import OpenAlexBackend
    from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend
    from autoresearch.agents.coder_agent.service import build_coder_service
    from autoresearch.agents.critic_agent.service import build_critic_service
    from autoresearch.agents.data_analysis_agent.service import build_data_analysis_service
    from autoresearch.agents.deep_research_agent.service import build_deep_research_service
    from autoresearch.agents.evaluator_agent.service import build_evaluator_service
    from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
    from autoresearch.agents.experiment_planner_agent.service import build_experiment_planner_service
    from autoresearch.agents.hypothesis_agent.service import build_hypothesis_service
    from autoresearch.agents.knowledge_index_agent.service import build_knowledge_index_service
    from autoresearch.agents.memory_agent.service import build_memory_service
    from autoresearch.agents.moss_agent.service import build_moss_service
    from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
    from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
    from autoresearch.agents.ranking_judge_agent.service import build_ranking_judge_service
    from autoresearch.agents.report_agent.service import build_report_service
    from autoresearch.agents.runner_agent.service import build_runner_service
    from autoresearch.orchestrator.engine import OrchestratorEngine

    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=sm_root, spec_store=store
    )
    await eng.start()

    llm = _make_llm()

    def bv(name):
        return bus.view(name)

    build_problem_modeling_service(client=bv("ProblemModelingAgent"), spec_store=store, llm_client=llm)
    build_deep_research_service(
        client=bv("DeepResearchAgent"), spec_store=store,
        backend=OpenAlexBackend(per_page=8), llm_client=llm,
    )
    build_knowledge_index_service(client=bv("KnowledgeIndexAgent"), spec_store=store)
    build_evidence_audit_service(client=bv("EvidenceAuditAgent"), spec_store=store, llm_client=llm)
    build_novelty_audit_service(client=bv("NoveltyAuditAgent"), spec_store=store, llm_client=llm)
    build_critic_service(client=bv("CriticAgent"), spec_store=store, llm_client=llm)
    build_ranking_judge_service(client=bv("RankingJudgeAgent"), spec_store=store, llm_client=llm)
    build_evaluator_service(client=bv("EvaluatorAgent"), spec_store=store, llm_client=llm)
    # RunnerAgent: benchmark backend stays the echo demo; the P1 general
    # honest-experiment branch is wired via p1_backend (real Docker) + p1_llm.
    build_runner_service(
        client=bv("RunnerAgent"), spec_store=store,
        backend=_EchoRunnerBackend(),
        p1_backend=OpenHandsBackend(), p1_llm=llm,
    )
    build_hypothesis_service(client=bv("HypothesisAgent"), spec_store=store, llm_client=llm)
    build_experiment_planner_service(client=bv("ExperimentPlannerAgent"), spec_store=store, llm_client=llm)
    build_coder_service(client=bv("CoderAgent"), spec_store=store, llm_client=llm)
    build_memory_service(client=bv("MemoryAgent"), spec_store=store, llm_client=llm)
    build_report_service(client=bv("ReportAgent"), spec_store=store, llm_client=llm)
    build_moss_service(client=bv("MOSS"), spec_store=store, llm_client=llm)
    build_data_analysis_service(client=bv("DataAnalysisAgent"), spec_store=store, llm_client=llm)

    return eng


class _Recorder:
    """Bus listener: remembers the latest clarification question_id and the last
    report.ready ref, and keeps a rolling tail of events for diagnostics."""

    def __init__(self) -> None:
        self.latest_question_id = None
        self.report_ref = None
        self.tail: list[tuple[str, str]] = []  # (emitter, event_type)

    async def __call__(self, env) -> None:
        et = getattr(env, "event_type", "?")
        self.tail.append((getattr(env, "emitter", "?"), et))
        self.tail = self.tail[-40:]
        if et == "problem_modeling.clarification_question":
            self.latest_question_id = (env.payload or {}).get("question_id")
        elif et == "report.ready":
            self.report_ref = (env.payload or {}).get("report_ref")


def _log(msg: str) -> None:
    print(f"[drive {time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _drive(bus, eng, store, sid, recorder) -> str:
    """The auto-answer loop. Returns the final current_state.

    Because DeliveringBus delivers publishes SYNCHRONOUSLY, each auto-answer
    below runs the entire downstream cascade (real LLM + Docker) to the next
    human-wait or terminal state before ``publish`` returns; the loop then
    re-reads state and answers the next pause.
    """
    from autoresearch.orchestrator.session_state import load_session_state
    from autoresearch.orchestrator.user_decision_matrix import (
        UserDecisionGuard, load_user_decision_matrix,
    )
    from autoresearch.experiments.general.driver_logic import pick_action

    guard = UserDecisionGuard(load_user_decision_matrix())
    human = bus.view("Human")

    start = time.monotonic()
    last_state = None
    same_state_polls = 0
    clarify_answers = 0
    answered_qids: set = set()
    agent_failure_retries = 0  # bounded retry_session count (mirrors the watchdog)

    while True:
        if time.monotonic() - start > DRIVE_TIMEOUT:
            _log(f"WALL-CLOCK TIMEOUT ({DRIVE_TIMEOUT}s) exceeded; stopping.")
            break

        state = await load_session_state(bus.db, sid)
        if state is None:
            _log("session state vanished; stopping.")
            break
        cur = state.current_state
        if cur in ("Done", "Failed"):
            break

        progressed = False

        if cur == "ClarifyingResearchGoal":
            qid = recorder.latest_question_id
            if qid is not None and qid not in answered_qids and clarify_answers < MAX_CLARIFY:
                _log(f"ClarifyingResearchGoal: answering clarification {qid} -> CANNED_FINALIZE "
                     f"({clarify_answers + 1}/{MAX_CLARIFY})")
                await human.publish(
                    event_type="human.clarification_answer",
                    session_id=sid,
                    payload={"question_id": qid, "answer_text": CANNED_FINALIZE},
                )
                answered_qids.add(qid)
                clarify_answers += 1
                progressed = True
            elif clarify_answers >= MAX_CLARIFY:
                _log(f"ClarifyingResearchGoal: clarify cap ({MAX_CLARIFY}) reached; "
                     "no longer answering, letting it proceed / stall.")

        elif cur == "HumanIdeaCheckpoint":
            _log("HumanIdeaCheckpoint: replying idea_provided {inject: False}")
            await human.publish(
                event_type="human.idea_provided",
                session_id=sid,
                payload={"inject": False},
            )
            progressed = True

        elif cur == "ReviewingExperimentPlan" and os.environ.get("AUTORESEARCH_GEN_REVIEW_DESIGN") == "1":
            # Design-review gate (opt-in): the critic defers to a human, so the driver
            # supplies the conservative human answer -- APPROVE the design as-is (don't
            # inject changes, don't loop back to re-design). Mirrors the web endpoint.
            _log("ReviewingExperimentPlan: approving P1 design as-is {inject: False}")
            ref = store.put(kind="critique_report", session_id=sid, key="experiment_plan-v1",
                            obj={"target": "experiment_plan", "verdict": "pass", "issues": []},
                            emitter="Human")
            await human.publish(
                event_type="experiment.plan_reviewed", session_id=sid,
                payload={"verdict": "pass", "review_report_ref": ref},
            )
            progressed = True

        elif cur == "AwaitingUserDecision":
            reason = state.awaiting_user_decision_reason
            legal = guard.legal_actions(reason)
            action = pick_action(reason, legal, agent_failure_retries=agent_failure_retries)
            if action == "retry_session":
                agent_failure_retries += 1  # bound the retry so a hard failure can't loop
            if action is not None:
                _log(f"AwaitingUserDecision(reason={reason}): decision -> {action} "
                     f"(legal={legal})")
                await human.publish(
                    event_type="user.decision_provided",
                    session_id=sid,
                    payload={"decision": action},
                )
                progressed = True
            else:
                _log(f"AwaitingUserDecision(reason={reason}): NO safe action in "
                     f"legal={legal}; stopping.")
                break

        else:
            # A working state -- the synchronous cascade should have already
            # driven past it. If we ever observe one and it does not advance,
            # the no-progress guard below treats it as stuck.
            _log(f"(working state {cur}; awaiting synchronous cascade)")

        if cur == last_state and not progressed:
            same_state_polls += 1
            if same_state_polls >= NO_PROGRESS_LIMIT:
                _log(f"NO PROGRESS: stuck in {cur} for {same_state_polls} polls; stopping.")
                break
            await asyncio.sleep(POLL_INTERVAL)
        else:
            same_state_polls = 0
        last_state = cur

    final = await load_session_state(bus.db, sid)
    return final.current_state if final else "?"


def _print_report(store, state, recorder) -> None:
    ref = (state.final_report_ref if state else None) or recorder.report_ref
    if not ref:
        _log("No report was produced (final_report_ref / report.ready both empty).")
        return
    try:
        report = store.get(ref)
    except Exception as exc:  # noqa: BLE001
        _log(f"Could not fetch report {ref}: {type(exc).__name__}: {exc}")
        return
    print("\n========== HONEST FINAL REPORT ==========", flush=True)
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("=========================================", flush=True)


async def _main_async(question: str) -> int:
    import aiosqlite
    from autoresearch.orchestrator.session_state import load_session_state
    from autoresearch.spec_store import SpecStore
    from autoresearch.testing.delivering_bus import DeliveringBus

    # Turn the real P1 general honest-experiment branch ON, plus real paper search.
    os.environ["AUTORESEARCH_GEN_EXPERIMENT"] = "1"
    os.environ["AUTORESEARCH_REAL_SEARCH"] = "1"

    sm_root = Path(__file__).resolve().parents[1] / "autoresearch" / "orchestrator" / "state-machine"
    workdir = Path(os.environ.get(
        "AUTORESEARCH_DRIVE_WORKDIR",
        f"/tmp/autoresearch-drive-{os.getpid()}",
    ))
    workdir.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(str(workdir / "drive.db"))
    store = SpecStore(workdir / "store")
    bus = DeliveringBus(conn)
    recorder = _Recorder()
    bus.view("DriveListener").subscribe("*", recorder)

    sid = None
    final_state = None
    try:
        eng = await _build(bus, store, sm_root)
        _log(f"engine + 16 agents assembled; workdir={workdir}")
        _log(f"question: {question}")
        sid = await eng.start_session(question, user_id="drive")
        _log(f"session started: {sid}")

        final_current = await _drive(bus, eng, store, sid, recorder)
        final_state = await load_session_state(bus.db, sid)
        _log(f"FINISHED. current_state={final_current}")
        _print_report(store, final_state, recorder)
        return 0 if final_current == "Done" else 1
    except Exception:  # noqa: BLE001 -- never hang silently
        print("\n!!!!! DRIVER EXCEPTION !!!!!", flush=True)
        traceback.print_exc()
        try:
            if sid is not None:
                st = await load_session_state(bus.db, sid)
                print(f"state at failure: {st.current_state if st else None}", flush=True)
        except Exception:
            pass
        print("last events:", flush=True)
        for emitter, et in recorder.tail[-20:]:
            print(f"  {emitter} -> {et}", flush=True)
        return 2
    finally:
        try:
            store.close()
        except Exception:
            pass
        await conn.close()


def main() -> int:
    if not os.environ.get("OPENROUTER_KEY"):
        print("OPENROUTER_KEY not set; export it and rerun. Aborting.")
        return 2
    if os.environ.get("AUTORESEARCH_REAL_SANDBOX") != "1":
        print("Set AUTORESEARCH_REAL_SANDBOX=1 to run the REAL headless drive. Aborting.")
        return 2
    question = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_QUESTION
    return asyncio.run(_main_async(question))


if __name__ == "__main__":
    raise SystemExit(main())
