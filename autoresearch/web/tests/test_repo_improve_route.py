"""The /api/experiment endpoint routes a repo-improvement request to the repo_improve
pipeline, and every other question still goes to the general pipeline (unchanged)."""
from __future__ import annotations

import json

from starlette.testclient import TestClient

from autoresearch.web import server


class _FakeLLM:
    """Answers discover()'s is_repo question; never touches the network."""
    def __init__(self, is_repo=False, repo_url="", name=""):
        self._obj = {"is_repo": is_repo, "repo_url": repo_url, "name": name}

    def chat(self, *, messages, **_):
        return type("R", (), {"content": json.dumps(self._obj)})()


def _setup(monkeypatch, llm=None):
    monkeypatch.setattr(server, "KEY", "sk-test")
    monkeypatch.setattr(server.ENGINE, "_make_llm", lambda: llm or _FakeLLM())
    monkeypatch.setattr(server.ENGINE, "start", lambda: None)
    called = {}
    monkeypatch.setattr(server.ENGINE, "submit", lambda cf: called.__setitem__("cf", cf))
    monkeypatch.setattr(server, "_run_repo_improve_pipeline",
                        lambda q, emit: called.__setitem__("repo", q) or {"verdict": {"verdict": "pass"}})
    monkeypatch.setattr(server, "_run_general_pipeline",
                        lambda q, emit: called.__setitem__("gen", q) or {"verdict": {"verdict": "pass"}})
    return TestClient(server.app), called


def test_nanogpt_request_routes_to_repo_improve_dynamically(monkeypatch):
    # no registered benchmark anymore: a nanoGPT request routes to repo_improve only because
    # the LLM recognises it as a real repo (dynamic discovery) and names its URL.
    client, called = _setup(monkeypatch, llm=_FakeLLM(
        is_repo=True, repo_url="https://github.com/karpathy/nanoGPT", name="nanoGPT"))
    r = client.post("/api/experiment",
                    json={"question": "improve nanoGPT single-GPU 5-minute training"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "repo_improve" and body["benchmark"] == "nanoGPT"
    assert "cf" in called                            # a task was submitted to the engine


def test_ordinary_request_still_routes_to_general(monkeypatch):
    # LLM says is_repo=false -> general path (default fake)
    client, called = _setup(monkeypatch)
    r = client.post("/api/experiment",
                    json={"question": "can first-fit-decreasing beat first-fit on bin packing?"})
    assert r.status_code == 200 and r.json()["mode"] == "general"


def test_unregistered_repo_request_routes_to_repo_improve(monkeypatch):
    # not in the registry, but the LLM recognises a real repo -> dynamic repo_improve path
    client, called = _setup(monkeypatch, llm=_FakeLLM(
        is_repo=True, repo_url="https://github.com/some-owner/tiny-classifier", name="tiny-classifier"))
    r = client.post("/api/experiment",
                    json={"question": "improve the some-owner/tiny-classifier accuracy"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "repo_improve" and body["benchmark"] == "tiny-classifier"
