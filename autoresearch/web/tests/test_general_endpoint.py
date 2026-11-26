"""Task 7: POST /api/experiment general honest-experiment entry (pipeline mocked
-- no container/LLM) + the viewer input box/rendering.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from autoresearch.web import server


def _fake_pipeline(question, emit):
    emit("design_ready", {"design": {"question": question}})
    emit("harness_ready", {"harness": "# harness"})
    emit("method_ready", {"method": "def solve(i): return []"})
    emit("run_done", {"primary": {"baseline_metric": 10, "method_metric": 8,
                                  "valid": True, "seed": 0, "isolated": True},
                      "reproduce": {"baseline_metric": 11, "method_metric": 9,
                                    "valid": True, "seed": 1, "isolated": True}})
    verdict = {"verdict": "pass", "accepted": True, "reasons": ["ok"]}
    emit("verdict", {"verdict": verdict, "report": "# report"})
    return {
        "design": {"question": question}, "harness": "# harness",
        "method": "def solve(i): return []",
        "primary": {"baseline_metric": 10, "method_metric": 8, "valid": True,
                    "seed": 0, "isolated": True},
        "reproduce": {"baseline_metric": 11, "method_metric": 9, "valid": True,
                      "seed": 1, "isolated": True},
        "isolated": True, "verdict": verdict, "report": "# report",
    }


def test_experiment_endpoint_runs_pipeline_and_surfaces_result(monkeypatch):
    monkeypatch.setattr(server, "KEY", "sk-test")
    monkeypatch.setattr(server, "_run_general_pipeline", _fake_pipeline)
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    captured = {}
    monkeypatch.setattr(server.ENGINE, "submit", lambda cf: captured.__setitem__("cf", cf))
    client = TestClient(server.app)

    r = client.post("/api/experiment", json={"question": "beat First-Fit on bin packing?"})
    assert r.status_code == 200 and r.json()["ok"] is True

    asyncio.run(captured["cf"]())

    res = client.get("/api/experiment_result").json()
    assert res["verdict"]["verdict"] == "pass"
    assert res["isolated"] is True
    assert res["primary"]["method_metric"] == 8
    assert res["report"] == "# report"


def test_experiment_endpoint_requires_key(monkeypatch):
    monkeypatch.setattr(server, "KEY", None)
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    c = TestClient(server.app)
    r = c.post("/api/experiment", json={"question": "q"})
    assert r.status_code == 400


def test_experiment_endpoint_requires_question(monkeypatch):
    monkeypatch.setattr(server, "KEY", "sk-test")
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    c = TestClient(server.app)
    r = c.post("/api/experiment", json={})
    assert r.status_code == 400


def test_experiment_route_registered():
    routes = {getattr(r, "path", None) for r in server.app.routes}
    assert "/api/experiment" in routes
    assert "/api/experiment_result" in routes


def test_viewer_single_start_entry_and_experiment_rendering():
    html = (Path(server.__file__).resolve().parent / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    # a SINGLE "开始" entry drives the full pipeline (incl. the real experiment);
    # the standalone mc/exp shortcut buttons were removed.
    assert 'id="startBtn"' in html
    assert "expBtn" not in html and "expQuestion" not in html
    # the general experiment events are still rendered (they fire during the real run)
    assert "gen.verdict" in html
    assert "gen.run_done" in html


# --- Viewer v2: home page / session list / read-only view ----------------------

def test_topic_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_WORKDIR", tmp_path)
    monkeypatch.setattr(server, "_TOPICS_FILE", tmp_path / "topics.json")
    server._record_topic("s1", "beat First-Fit")
    assert server._topics()["s1"]["topic"] == "beat First-Fit"


def test_sessions_endpoint_lists_and_marks_active(monkeypatch):
    monkeypatch.setattr(server, "_list_sessions",
                        lambda: [{"session_id": "s1", "topic": "T", "current_state": "Done"}])
    # multi-session pool: `active` is the list of live worker sessions
    monkeypatch.setattr(server.ENGINE.workers[0], "session_id", "s1")
    r = TestClient(server.app).get("/api/sessions").json()
    assert "s1" in r["active"]
    assert r["sessions"][0]["topic"] == "T"


def test_session_view_endpoint(monkeypatch):
    monkeypatch.setattr(server.ENGINE, "session_view",
                        lambda sid: {"session_id": sid, "current_state": "Done", "topic": "T"} if sid == "s1" else None)
    c = TestClient(server.app)
    assert c.get("/api/session_view?sid=s1").json()["topic"] == "T"
    assert c.get("/api/session_view?sid=nope").status_code == 404


def test_checkpoint_injected_url_becomes_confirmed_repo(monkeypatch):
    # a user who injects "none of these -> use https://github.com/NVIDIA/SOL-ExecBench" at the
    # checkpoint must have that URL honored as the baseline repo (previously the idea text was
    # dropped entirely, so injection was a no-op).
    import asyncio
    from starlette.testclient import TestClient

    cap = {}

    class _V:
        async def publish(self, *, event_type, session_id, payload):
            cap["event"] = event_type
            cap["payload"] = payload

    class _Bus:
        def view(self, name):
            return _V()

    class _W:
        session_id = "s-inj"
        bus = _Bus()

        def submit(self, cf):
            import threading

            def _run():
                loop = asyncio.new_event_loop()
                loop.run_until_complete(cf())
                loop.close()

            t = threading.Thread(target=_run)
            t.start()
            t.join()

    class _Store:
        def put(self, *, kind, session_id, key, obj, emitter):
            return f"ref://{kind}/{session_id}/{key}"

    w = _W()
    monkeypatch.setattr(server.ENGINE, "worker_for", lambda sid: w)
    monkeypatch.setattr(server.ENGINE.workers[0], "store", _Store())
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)

    r = TestClient(server.app).post("/api/human", json={
        "sid": "s-inj", "kind": "checkpoint", "inject": True,
        "idea": "none of these, should be https://github.com/NVIDIA/SOL-ExecBench."})
    assert r.status_code == 200
    p = cap["payload"]
    assert p["confirmed_repo_url"] == "https://github.com/NVIDIA/SOL-ExecBench"  # extracted from idea
    assert p["human_idea_seed_ref"]                                             # idea persisted + ref sent
    assert p["inject"] is True


def test_explicit_closeout_persists_without_interrupting_current_attempt(tmp_path, monkeypatch):
    from autoresearch.spec_store import SpecStore, refs

    store = SpecStore(tmp_path / 'closeout-store')
    worker = SimpleNamespace(session_id='closeout-session', bus=SimpleNamespace(view=lambda _: None))
    monkeypatch.setattr(server.ENGINE, 'worker_for', lambda sid: worker)
    monkeypatch.setattr(server.ENGINE.workers[0], 'store', store)
    monkeypatch.setattr(server.ENGINE, 'start', lambda: None)
    try:
        response = TestClient(server.app).post('/api/human', json={
            'sid': worker.session_id, 'kind': 'closeout',
            'idea': 'Finish the active attempt, retain best and formally validate.'})
        assert response.status_code == 200
        saved = store.get(refs.encode('human_idea_seed', worker.session_id, 'closeout-v1'))
        assert saved == {'closeout': True,
                         'idea': 'Finish the active attempt, retain best and formally validate.'}
    finally:
        store.close()


def test_confirmed_repo_suppresses_candidate_picker():
    # select-once (#2): once a repo is confirmed (picked or injected), the checkpoint must not
    # recompute + re-offer the candidate list, or the user is asked to choose the SAME repo twice.
    w = server.ENGINE.workers[0]
    # a discover call here would be a bug -- if reached, blow up loudly instead of silently returning []
    def _boom(*a, **k):  # noqa: ANN001
        raise AssertionError("candidates recomputed for an already-confirmed repo")
    orig = w._compute_repo_candidates
    w._compute_repo_candidates = _boom
    try:
        d = {"session_id": "s-conf", "current_state": "HumanIdeaCheckpoint",
             "confirmed_repo_url": "https://github.com/NVIDIA/SOL-ExecBench"}
        assert w._repo_candidates_sync(d) == []          # picker suppressed, no recompute
    finally:
        w._compute_repo_candidates = orig


def test_pool_claim_worker_atomic_no_double_grab():
    # two concurrent /api/session starts must NOT both grab the same worker (the race that gave
    # two sessions the same sid). claim_worker is lock-guarded; distinct workers, then full.
    import threading
    from autoresearch.web.server import EnginePool
    pool = EnginePool(2)
    got = []

    def claim():
        got.append(pool.claim_worker())

    ts = [threading.Thread(target=claim) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert None not in got            # both got a worker (pool size 2)
    assert got[0] is not got[1]       # DIFFERENT workers -- no double-grab
    assert pool.claim_worker() is None  # pool now full


def test_reselecting_repository_invalidates_candidate_cache(monkeypatch):
    w = server.Engine()
    calls = []
    def compute(sid, state):
        calls.append(sid)
        return [{"url": f"https://github.com/example/proposal-{len(calls)}"}]
    monkeypatch.setattr(w, "_compute_repo_candidates", compute)
    state = {"session_id": "selection", "confirmed_repo_url": None}
    first = w._repo_candidates_sync(state)
    assert w._repo_candidates_sync(state) == first
    state["last_baseline_attempt_ref"] = "spec://baseline_result/selection/v1"
    state["moss_interventions"] = [{"reason": "baseline_reproduction_thrashing"}]
    second = w._repo_candidates_sync(state)
    assert second != first
    # Attempt artifacts may reuse a v1 key; another actual intervention must still refresh.
    state["moss_interventions"].append({"reason": "baseline_reproduction_thrashing"})
    assert w._repo_candidates_sync(state) != second
    assert len(calls) == 3


def test_v2_routes_registered():
    paths = {getattr(r, "path", None) for r in server.app.routes}
    assert "/api/sessions" in paths and "/api/session_view" in paths
    assert "/api/session/resume" in paths


def test_repository_reselection_passes_failed_attempt_separately_from_user_request(monkeypatch):
    from autoresearch.experiments.repo_improve import discover
    failure_ref = "spec://baseline_result/reselection/v1"
    failure = {"repo_url": "https://github.com/example/unsuitable", "reproduced": False,
               "log_excerpt": "Repository does not implement the requested workload."}

    class Store:
        def get(self, ref):
            return failure if ref == failure_ref else {"research_goal": "Improve scheduling"}

    calls = []
    def candidates(request, llm, **kwargs):
        calls.append((request, kwargs))
        return []

    w = server.Engine()
    monkeypatch.setattr(w, "store", Store())
    monkeypatch.setattr(w, "_make_llm", lambda: object())
    monkeypatch.setattr(server, "_topics", lambda: {})
    monkeypatch.setattr(discover, "discover_candidates", candidates)
    w._compute_repo_candidates("reselection", {"last_baseline_attempt_ref": failure_ref})
    assert calls[-1][1]["baseline_failure"] == failure
    assert failure["repo_url"] not in calls[-1][0]
    failure["reproduced"] = True
    w._compute_repo_candidates("reselection", {"last_baseline_attempt_ref": failure_ref})
    assert calls[-1][1]["baseline_failure"] is None


def test_resume_endpoint_redrives_suspended_worker(monkeypatch):
    captured = {}

    class _Engine:
        async def resume_to_current_state(self, sid):
            captured["resumed_sid"] = sid

    class _Worker:
        session_id = "s-suspended"
        eng = _Engine()

        def state_sync(self):
            return {"session_id": self.session_id, "current_state": "Suspended"}

        def _watchdog_is_busy(self, state):
            return False

        def submit(self, coro_factory):
            captured["coro_factory"] = coro_factory

    worker = _Worker()
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    monkeypatch.setattr(server.ENGINE, "worker_for", lambda sid: worker if sid == worker.session_id else None)

    client = TestClient(server.app)
    response = client.post("/api/session/resume", json={"sid": worker.session_id})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "session_id": worker.session_id}
    asyncio.run(captured["coro_factory"]())
    assert captured["resumed_sid"] == worker.session_id


def test_resume_endpoint_rejects_non_suspended_session(monkeypatch):
    class _Worker:
        session_id = "s-running"

        def state_sync(self):
            return {"session_id": self.session_id, "current_state": "RunningExperiment"}

    worker = _Worker()
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    monkeypatch.setattr(server.ENGINE, "worker_for", lambda sid: worker)

    response = TestClient(server.app).post(
        "/api/session/resume", json={"sid": worker.session_id}
    )
    assert response.status_code == 409


def test_resume_endpoint_explicit_redrive_accepts_idle_persisted_state(monkeypatch):
    captured = {}

    class _Engine:
        async def resume_to_current_state(self, sid):
            captured["resumed_sid"] = sid

    class _Worker:
        bus = SimpleNamespace(replay_pending=AsyncMock(return_value=False))
        session_id = "s-persisted"
        eng = _Engine()

        def state_sync(self):
            return {"session_id": self.session_id, "current_state": "PlanningExperiment"}

        def _watchdog_is_busy(self, state):
            return False

        def submit(self, coro_factory):
            captured["coro_factory"] = coro_factory

    worker = _Worker()
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    monkeypatch.setattr(server.ENGINE, "worker_for", lambda sid: worker)

    response = TestClient(server.app).post(
        "/api/session/resume", json={"sid": worker.session_id, "redrive": True}
    )

    assert response.status_code == 200
    asyncio.run(captured["coro_factory"]())
    assert captured["resumed_sid"] == worker.session_id


def test_resume_endpoint_redrive_rebinds_archived_session(monkeypatch):
    """A restart leaves terminal sessions unbound, but report recovery must still be reachable."""
    captured = {}

    class _Engine:
        async def resume_to_current_state(self, sid):
            captured["resumed_sid"] = sid

    class _Worker:
        bus = SimpleNamespace(replay_pending=AsyncMock(return_value=False))
        session_id = "s-failed-report"
        eng = _Engine()

        def state_sync(self):
            return {"session_id": self.session_id, "current_state": "Failed"}

        def _watchdog_is_busy(self, state):
            return False

        def submit(self, coro_factory):
            captured["coro_factory"] = coro_factory

    worker = _Worker()
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    monkeypatch.setattr(server.ENGINE, "worker_for", lambda sid: None)
    monkeypatch.setattr(
        server.ENGINE, "bind_existing_session",
        lambda sid: worker if sid == worker.session_id else None,
    )

    response = TestClient(server.app).post(
        "/api/session/resume", json={"sid": worker.session_id, "redrive": True}
    )

    assert response.status_code == 200
    asyncio.run(captured["coro_factory"]())
    assert captured["resumed_sid"] == worker.session_id


@pytest.mark.parametrize("current_state", ["RunningExperiment", "Suspended"])
def test_resume_endpoint_explicit_redrive_rejects_busy_worker(monkeypatch, current_state):
    class _Worker:
        session_id = "s-busy"

        def state_sync(self):
            return {"session_id": self.session_id, "current_state": current_state}

        def _watchdog_is_busy(self, state):
            return True

    worker = _Worker()
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    monkeypatch.setattr(server.ENGINE, "worker_for", lambda sid: worker)

    response = TestClient(server.app).post(
        "/api/session/resume", json={"sid": worker.session_id, "redrive": True}
    )

    assert response.status_code == 409


def test_remote_repo_liveness_does_not_scan_unrelated_local_agent_containers(monkeypatch):
    calls = []
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE", "1")
    monkeypatch.setattr(server, "_CONTAINER_CACHE", {"t": -1e9, "live": False})
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert server._p1_container_live() is False
    assert calls == []


def test_human_report_text_is_readable_prose():
    """#4: the machine report.json is rendered into a plain-language report for people."""
    from autoresearch.web.server import _human_report_text
    rep = {
        "summary": "价值密度贪心显著优于顺序基线。",
        "verified_findings": ["Method beats baseline: baseline=1100 vs method=1615."],
        "negative_results": [],
        "analysis_caveats": ["报告的是绝对总价值,非近似比。"],
        "known_risks": ["小规模结论未必外推。"],
        "experiment_not_run": ["近似比标准差"],
        "experiment_ledger": [{"experiment_id": "knap1", "verdict": "pass", "metric_value": 1615}],
    }
    txt = _human_report_text(rep, topic="背包:价值密度贪心 vs 顺序")
    assert "最终研究报告" in txt or "Final Research Report" in txt
    assert "背包:价值密度贪心 vs 顺序" in txt          # the question
    assert "价值密度贪心显著优于顺序基线。" in txt        # the summary prose
    assert "Method beats baseline" in txt               # verified finding listed
    assert "近似比标准差" in txt                          # future-work item
    assert "knap1" in txt and "pass" in txt              # ledger
    assert "{" not in txt                                # NOT raw JSON


def test_human_report_text_prefers_complete_markdown_body():
    from autoresearch.web.server import _human_report_text
    body = "# 理论报告\n\n## 定理\n在公理 A1 下，命题 P 成立。"
    txt = _human_report_text({"report_body": body, "summary": "不应取代正文"})
    assert body in txt
    assert "不应取代正文" not in txt


def test_viewer_has_trilingual_i18n():
    """i18n: page language selectable (简体/繁體/English), with a t()/applyLang engine
    and an output-language directive (_outdir) threaded into new sessions."""
    html = (Path(server.__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
    # the three languages + selector
    for code in ('"zh-CN"', '"zh-TW"', '"en"'):
        assert code in html
    assert 'id="langSel"' in html and "setLang(this.value)" in html
    # the i18n engine
    for fn in ("function t(", "function applyLang", "data-i18n", 'localStorage.getItem("ar_lang")'):
        assert fn in html
    # sample translations present for all three (Start button)
    assert "开始" in html and "開始" in html and '"Start"' in html
    # output-language directive is seeded into the pipeline
    assert "_outdir" in html and "language:_lang" in html


def test_review_design_endpoint_registered_and_guards_no_session(monkeypatch):
    """Design-review gate: /api/review_design exists; without an active session -> 400."""
    from starlette.testclient import TestClient
    paths = {getattr(r, "path", None) for r in server.app.routes}
    assert "/api/review_design" in paths
    # no live session on any worker -> 400
    for w in server.ENGINE.workers:
        monkeypatch.setattr(w, "session_id", None)
    r = TestClient(server.app).post("/api/review_design", json={"approved": True})
    assert r.status_code == 400


def test_viewer_has_design_review_ui():
    html = (Path(server.__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "renderDesignReview" in html and "/api/review_design" in html
    assert "ReviewingExperimentPlan" in html
    assert "rp_approve" in html and "review_plan_title" in html


def test_viewer_submits_initial_download_authorization():
    html = (Path(server.__file__).resolve().parent / "static" / "index.html").read_text(
        encoding="utf-8")
    assert 'id="allowDownloads"' in html
    assert "confirmed_boundary:confirmedBoundary" in html
    assert "download_authorization" in html
    assert "repository-required dependencies, datasets, model artifacts" in html


async def test_session_entry_passes_explicit_contract_and_rejects_invalid_boundary(monkeypatch):
    from types import SimpleNamespace
    calls = []
    async def start(topic, **kwargs):
        calls.append((topic, kwargs))
        return "contract-session"
    worker = SimpleNamespace(eng=SimpleNamespace(start_session=start), session_id=None)
    worker.submit = lambda factory: asyncio.create_task(factory())
    def claim():
        calls.append("claim")
        return worker
    monkeypatch.setattr(server.ENGINE, "claim_worker", claim)
    monkeypatch.setattr(server, "_record_topic", lambda *args: None)
    rejected = await server.start_session({"topic": "q", "confirmed_boundary": "not-an-object"})
    assert rejected.status_code == 400 and calls == []
    boundary = {"execution_contract": {"evaluation": {"required_values": {"/split": "dev"}}}}
    result = await server.start_session({"topic": "q", "confirmed_boundary": boundary})
    assert result["session_id"] == "contract-session"
    assert calls == ["claim", ("q", {"user_id": "web", "confirmed_boundary": boundary})]
