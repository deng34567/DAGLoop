import asyncio
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from autoresearch.web import server
from autoresearch.agents._reuse import openhands_backend
from autoresearch.orchestrator.moss_watchdog import (
    KIND_IDEA_NO_INJECT, KIND_METHOD_NO_INJECT, KIND_RESUME_SUSPENDED)
from autoresearch.spec_store.store import SpecStore


@pytest.mark.asyncio
@pytest.mark.parametrize("search_enabled", [True, False])
async def test_example_configuration_selects_real_search_and_sandbox(monkeypatch, tmp_path,
                                                                    search_enabled):
    from dotenv import dotenv_values
    from autoresearch.agents._reuse.deep_research_backend import MultiSourceBackend

    for key, value in dotenv_values(Path(__file__).parents[3] / ".env.example").items():
        if value is not None:
            monkeypatch.setenv(key, value)
    if not search_enabled:
        monkeypatch.setenv("AUTORESEARCH_REAL_SEARCH", "0")
    for key in ("TAVILY_API_KEY", "AUTORESEARCH_DEEP_RESEARCH_ENV_FILE",
                "SEMANTIC_SCHOLAR_API_KEY", "AUTORESEARCH_REAL_RUNNER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(server, "_WORKDIR", tmp_path)
    monkeypatch.setattr(server.Engine, "_make_llm", lambda self: object())
    sandbox = object()
    monkeypatch.setattr(openhands_backend, "OpenHandsBackend", lambda: sandbox)
    captured = {}
    for name in vars(server):
        if name.startswith("build_") and name.endswith("_service"):
            monkeypatch.setattr(server, name, lambda **kwargs: None)
    monkeypatch.setattr(server, "build_deep_research_service",
                        lambda **kwargs: captured.update(search=kwargs["backend"]))
    monkeypatch.setattr(server, "build_runner_service",
                        lambda **kwargs: captured.update(runner=kwargs))
    engine = server.Engine(str(tmp_path / "web.db"))
    try:
        await engine._setup()
        assert isinstance(captured["search"], MultiSourceBackend)
        if search_enabled:
            assert captured["search"].backends
            assert captured["runner"]["repository_search_backend"] is captured["search"]
        else:
            assert captured["search"].search_seminal("query", {}) == []
            assert captured["search"].search_frontier("query", {}) == []
            assert captured["runner"]["repository_search_backend"] is None
        assert captured["runner"]["backend"] is None
        assert captured["runner"]["p1_backend"] is sandbox
    finally:
        if engine.bus is not None:
            await engine.bus.db.close()
        if engine.store is not None:
            engine.store.close()


@pytest.mark.asyncio
async def test_engine_requires_execution_configuration_before_startup(monkeypatch, tmp_path):
    for key in ("AUTORESEARCH_GEN_EXPERIMENT", "AUTORESEARCH_AGENTIC_IMPROVE",
                "AUTORESEARCH_REAL_RUNNER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(server, "_WORKDIR", tmp_path)
    monkeypatch.setattr(server.aiosqlite, "connect",
                        lambda *args, **kwargs: pytest.fail("configuration must precede DB setup"))
    with pytest.raises(RuntimeError, match="AUTORESEARCH_GEN_EXPERIMENT=1"):
        await server.Engine(str(tmp_path / "web.db"))._setup()


def test_pending_question_survives_empty_event_feed_after_restart(monkeypatch, tmp_path):
    store = SpecStore(tmp_path / "store")
    question = {"question_id": "q-001", "question_text": "Research direction?"}
    store.put(kind="research_spec_draft", session_id="sid", key="v1",
              obj={"pending_question": question, "answered": []}, emitter="ProblemModelingAgent")
    monkeypatch.setattr(server, "_EVENTS", [])
    monkeypatch.setattr(server, "ENGINE", SimpleNamespace(store=store))
    assert server._pending_question("sid") == question
    monkeypatch.setattr(server, "_EVENTS", [{"sid": "sid", "event_type": "human.clarification_answer",
                                           "payload": {"question_id": "q-001"}}])
    assert server._pending_question("sid") is None
    store.put(kind="research_spec_draft", session_id="sid", key="v1",
              obj={"pending_question": question, "answered": [{"question_id": "q-001"}]},
              emitter="ProblemModelingAgent")
    monkeypatch.setattr(server, "_EVENTS", [])
    assert server._pending_question("sid") is None
    store.close()


@pytest.mark.asyncio
async def test_engine_restores_the_only_nonterminal_session_binding(tmp_path):
    db_path = tmp_path / "web.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "CREATE TABLE session_state ("
            "session_id TEXT PRIMARY KEY, state_data TEXT NOT NULL, "
            "current_state TEXT NOT NULL, updated_at REAL NOT NULL, created_at REAL NOT NULL)"
        )
        await conn.executemany(
            "INSERT INTO session_state VALUES (?, '{}', ?, ?, ?)",
            [
                ("old-terminal", "Failed", 1.0, 1.0),
                ("resume-me", "Suspended", 2.0, 2.0),
            ],
        )
        await conn.commit()

        engine = server.Engine(str(db_path))
        await engine._restore_persisted_session_binding(conn)

    assert engine.session_id == "resume-me"


@pytest.mark.asyncio
async def test_engine_does_not_guess_between_multiple_nonterminal_sessions(tmp_path):
    db_path = tmp_path / "web.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "CREATE TABLE session_state ("
            "session_id TEXT PRIMARY KEY, state_data TEXT NOT NULL, "
            "current_state TEXT NOT NULL, updated_at REAL NOT NULL, created_at REAL NOT NULL)"
        )
        await conn.executemany(
            "INSERT INTO session_state VALUES (?, '{}', ?, ?, ?)",
            [
                ("first", "Suspended", 1.0, 1.0),
                ("second", "PlanningExperiment", 2.0, 2.0),
            ],
        )
        await conn.commit()

        engine = server.Engine(str(db_path))
        await engine._restore_persisted_session_binding(conn)

    assert engine.session_id is None


def test_engine_start_fails_if_worker_never_becomes_ready(monkeypatch, tmp_path):
    """A dead setup thread must fail Web startup instead of exposing a broken endpoint."""
    engine = server.Engine(str(tmp_path / "web.db"))

    class NeverReady:
        def wait(self, timeout=None):
            return False

        def set(self):
            pass

    engine._ready = NeverReady()

    class ThreadThatNeverRuns:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", ThreadThatNeverRuns)
    monkeypatch.setenv("AUTORESEARCH_WATCHDOG", "0")

    with pytest.raises(RuntimeError, match="did not become ready"):
        engine.start()


def test_engine_start_surfaces_setup_exception(monkeypatch, tmp_path):
    engine = server.Engine(str(tmp_path / "web.db"))

    async def broken_setup():
        raise ValueError("bad deployment config")

    class InlineThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(engine, "_setup", broken_setup)
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setenv("AUTORESEARCH_WATCHDOG", "0")

    with pytest.raises(RuntimeError, match="worker setup failed") as exc:
        engine.start()
    assert isinstance(exc.value.__cause__, ValueError)


def test_web_shutdown_pauses_remote_conversations(monkeypatch):
    calls = []
    monkeypatch.setattr(
        openhands_backend, "pause_active_conversations", lambda: calls.append(True) or 3)

    server._shutdown()

    assert calls == [True]


def test_web_llm_uses_short_connect_timeout_with_long_response_timeout(monkeypatch, tmp_path):
    """A dead DNS address must not consume the full LLM response budget."""
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(server, "KEY", "test-key")
    monkeypatch.setattr(server, "HKGAIClient", FakeClient)
    monkeypatch.setenv("AUTORESEARCH_WEB_LLM_TIMEOUT", "300")
    monkeypatch.setenv("AUTORESEARCH_WEB_LLM_CONNECT_TIMEOUT", "10")
    monkeypatch.setenv("AUTORESEARCH_WEB_LLM_MAX_TOKENS", "2048")

    server.Engine(str(tmp_path / "web.db"))._make_llm()

    timeout = captured["http_client"].timeout
    assert timeout.connect == 10.0
    assert timeout.read == 300.0
    assert captured["default_max_tokens"] == 2048


def test_web_llm_uses_shared_max_tokens_when_no_web_override(monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(server, "KEY", "test-key")
    monkeypatch.setattr(server, "HKGAIClient", FakeClient)
    monkeypatch.delenv("AUTORESEARCH_WEB_LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("AUTORESEARCH_LLM_MAX_TOKENS", "262144")

    server.Engine(str(tmp_path / "web.db"))._make_llm()

    assert captured["default_max_tokens"] == 262144


@pytest.mark.asyncio
async def test_submitted_agent_cascade_is_watchdog_busy_until_it_finishes(monkeypatch, tmp_path):
    """A live remote/LLM cascade must not be reclaimed solely because its DB state is old."""
    engine = server.Engine(str(tmp_path / "web.db"))
    engine.loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = asyncio.Event()
    scheduled = []

    async def long_agent_cascade():
        started.set()
        await release.wait()

    def schedule(coro, loop):
        assert loop is engine.loop
        task = asyncio.create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(server.asyncio, "run_coroutine_threadsafe", schedule)

    engine.submit(long_agent_cascade)
    await started.wait()
    assert engine._watchdog_is_busy("SelectingBenchmark") is True

    release.set()
    await scheduled[0]
    assert engine._watchdog_is_busy("SelectingBenchmark") is False


@pytest.mark.asyncio
async def test_watchdog_auto_confirms_method_without_injection(tmp_path):
    published = []

    class View:
        async def publish(self, **kwargs):
            published.append(kwargs)

    class Bus:
        def view(self, emitter):
            assert emitter == "Human"
            return View()

    engine = server.Engine(str(tmp_path / "web.db"))
    engine.bus = Bus()
    engine.submit = lambda factory: asyncio.create_task(factory())

    engine._watchdog_inject(
        kind=KIND_METHOD_NO_INJECT, session_id="session-1", reason=None,
        action=None, qid=None,
    )
    await asyncio.sleep(0)

    assert published == [{
        "event_type": "method.confirmed", "session_id": "session-1",
        "payload": {"inject": False},
    }]


@pytest.mark.asyncio
async def test_watchdog_resumes_suspended_session(tmp_path):
    resumed = []

    class Orchestrator:
        async def resume_to_current_state(self, session_id):
            resumed.append(session_id)

    engine = server.Engine(str(tmp_path / "web.db"))
    engine.eng = Orchestrator()
    engine.submit = lambda factory: asyncio.create_task(factory())

    engine._watchdog_inject(
        kind=KIND_RESUME_SUSPENDED, session_id="session-1", reason=None,
        action=None, qid=None,
    )
    await asyncio.sleep(0)

    assert resumed == ["session-1"]


@pytest.mark.asyncio
async def test_watchdog_auto_checkpoint_selects_discovered_repository(
    monkeypatch, tmp_path,
):
    published = []

    class View:
        async def publish(self, **kwargs):
            published.append(kwargs)

    class Bus:
        def view(self, emitter):
            assert emitter == "Human"
            return View()

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    engine = server.Engine(str(tmp_path / "web.db"))
    engine.bus = Bus()
    engine.submit = lambda factory: asyncio.create_task(factory())
    engine.state_sync = lambda: {"session_id": "session-1", "current_state": "HumanIdeaCheckpoint"}
    engine._repo_candidates_sync = lambda state: [
        {"url": "https://github.com/example/autoresearch", "name": "autoresearch"},
        {"url": "https://github.com/example/other", "name": "other"},
    ]

    engine._watchdog_inject(
        kind=KIND_IDEA_NO_INJECT, session_id="session-1", reason=None,
        action=None, qid=None,
    )
    await asyncio.sleep(0)

    assert published == [{
        "event_type": "human.idea_provided", "session_id": "session-1",
        "payload": {
            "inject": False,
            "confirmed_repo_url": "https://github.com/example/autoresearch",
        },
    }]
