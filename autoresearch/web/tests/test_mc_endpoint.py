"""Task 9: POST /api/mc_experiment focused entry (runner mocked -- no container)
and confirmation the dead toy button is fully gone from the viewer.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from autoresearch.web import server


class _FakeMCRunner:
    def run_baseline(self, handle=None):
        return {"metric_value": 1.0, "rmse_baseline": 0.0091, "rmse_proposed": 0.0091}

    def run_experiment(self, research_task, handle=None):
        return {
            "metric_value": 4.6, "reproduce_metric_value": 4.1,
            "reproduce_seeds": [100, 101], "rmse_baseline": 0.0091,
            "rmse_proposed": 0.00198,
            "over_budget": False, "hardcoded_suspected": False,
            "agent_status": "FINISHED", "code": "def proposed_estimate(): return 0.58",
            "anti_cheat": {"over_budget": False, "hardcoded_suspected": False,
                           "tamper_detected": False, "bias": 1e-4},
        }


def test_mc_experiment_returns_metrics_code_and_anti_cheat(monkeypatch):
    monkeypatch.setattr(server, "KEY", "sk-test")  # pass the opt-in guard
    monkeypatch.setattr(server, "_make_mc_runner", lambda: _FakeMCRunner())
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)  # no engine thread
    # capture the fire-and-forget coro factory; run it in this thread (no running loop)
    captured = {}
    monkeypatch.setattr(server.ENGINE, "submit", lambda cf: captured.__setitem__("cf", cf))
    client = TestClient(server.app)

    r = client.post("/api/mc_experiment", json={})
    assert r.status_code == 200 and r.json()["ok"] is True

    asyncio.run(captured["cf"]())  # execute the scheduled MC coro to completion

    res = client.get("/api/mc_result").json()
    assert res["baseline_rmse"] == 0.0091
    assert res["experiment_rmse"] == 0.00198
    assert res["ratio"] == 4.6
    assert res["code"] == "def proposed_estimate(): return 0.58"
    assert res["anti_cheat"] == {
        "over_budget": False, "hardcoded_suspected": False,
        "tamper_detected": False, "bias": 1e-4,
    }


def test_mc_experiment_surfaces_acceptance_verdict(monkeypatch):
    monkeypatch.setattr(server, "KEY", "sk-test")
    monkeypatch.setattr(server, "_make_mc_runner", lambda: _FakeMCRunner())
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    captured = {}
    monkeypatch.setattr(server.ENGINE, "submit", lambda cf: captured.__setitem__("cf", cf))
    client = TestClient(server.app)

    client.post("/api/mc_experiment", json={})
    asyncio.run(captured["cf"]())

    res = client.get("/api/mc_result").json()
    # deterministic acceptance verdict surfaced (4.6x, reproduced 4.1x, no flags)
    assert res["verdict"] == "pass"
    assert res["accepted"] is True
    assert isinstance(res["reasons"], list) and res["reasons"]
    assert res["reproduce_ratio"] == 4.1


def test_viewer_renders_acceptance_verdict():
    html = (Path(server.__file__).resolve().parent / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "验收" in html
    assert "p.verdict" in html
    assert "reasons" in html


def test_mc_experiment_requires_key(monkeypatch):
    monkeypatch.setattr(server, "KEY", None)
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    c = TestClient(server.app)
    r = c.post("/api/mc_experiment", json={})
    assert r.status_code == 400


def test_toy_and_mc_buttons_removed_from_viewer():
    html = (Path(server.__file__).resolve().parent / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "toy" not in html.lower()
    assert "/api/toy_experiment" not in html
    # the standalone mc + "提问并真跑" shortcut buttons were removed -- one 开始 entry only
    assert "mcBtn" not in html and "/api/mc_experiment" not in html
    assert "expBtn" not in html


def test_toy_endpoints_gone_from_server():
    routes = {getattr(r, "path", None) for r in server.app.routes}
    assert "/api/toy_experiment" not in routes
    assert "/api/toy_result" not in routes
    assert "/api/mc_experiment" in routes
    assert "/api/mc_result" in routes
