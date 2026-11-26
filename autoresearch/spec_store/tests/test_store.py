"""
Spec Store tests / Spec Store 单测 (pytest, synchronous).
"""

from __future__ import annotations

import re

import pytest

from autoresearch.spec_store import SpecStore, EmitterNotAllowed
from autoresearch.spec_store import refs

REF_RE = re.compile(r"^[a-z]+://.+")


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def test_put_get_roundtrip(store):
    obj = {"goal": "build a CLI", "n": 3, "nested": {"a": [1, 2]}}
    ref = store.put(
        kind="requirement", session_id="s1", key="main", obj=obj, emitter="RequirementAgent"
    )
    assert store.get(ref) == obj


def test_emitter_whitelist(store):
    # wrong emitter for a kind -> raises
    with pytest.raises(EmitterNotAllowed):
        store.put(
            kind="requirement", session_id="s1", key="main", obj={}, emitter="CoderAgent"
        )
    # correct emitter -> ok
    store.put(
        kind="architecture", session_id="s1", key="main", obj={}, emitter="ArchitectAgent"
    )
    # audit kind accepts any non-empty emitter
    store.put(kind="audit", session_id="s1", key="a", obj={"x": 1}, emitter="WhoeverAgent")
    store.put(kind="audit", session_id="s1", key="b", obj={"x": 2}, emitter="Orchestrator")
    # audit with empty emitter still rejected
    with pytest.raises(EmitterNotAllowed):
        store.put(kind="audit", session_id="s1", key="c", obj={}, emitter="")
    # new kinds: code (CoderAgent), recording/evidence (AcceptanceAgent)
    store.put(kind="code", session_id="s1", key="module-1", obj={"ref": "x"}, emitter="CoderAgent")
    store.put(kind="recording", session_id="s1", key="S1", obj={"rrweb": "x"}, emitter="AcceptanceAgent")
    store.put(kind="evidence", session_id="s1", key="S1", obj={"dom": "x"}, emitter="AcceptanceAgent")
    # wrong emitter for the new kinds rejected
    with pytest.raises(EmitterNotAllowed):
        store.put(kind="code", session_id="s1", key="m", obj={}, emitter="AcceptanceAgent")
    with pytest.raises(EmitterNotAllowed):
        store.put(kind="evidence", session_id="s1", key="m", obj={}, emitter="CoderAgent")
    # conversation kind: RequirementAgent only (Clarifying multi-turn transcript)
    store.put(kind="conversation", session_id="s1", key="v1",
              obj={"messages": []}, emitter="RequirementAgent")
    with pytest.raises(EmitterNotAllowed):
        store.put(kind="conversation", session_id="s1", key="v1", obj={}, emitter="Orchestrator")
    # human_idea_seed: the HUMAN's injected idea/repo at the checkpoint (emitter="Human").
    # Regression: this kind was unregistered, so injecting an idea crashed with EmitterNotAllowed
    # and froze the session at HumanIdeaCheckpoint.
    store.put(kind="human_idea_seed", session_id="s1", key="v1",
              obj={"idea": "none of these, use https://github.com/NVIDIA/SOL-ExecBench"}, emitter="Human")
    with pytest.raises(EmitterNotAllowed):
        store.put(kind="human_idea_seed", session_id="s1", key="v1", obj={}, emitter="SomeAgent")


def test_method_implementation_review_is_runner_owned(store):
    review = {"verdict": "partial", "issues": ["component missing"]}
    ref = store.put(
        kind="method_implementation_review",
        session_id="s1",
        key="v1",
        obj=review,
        emitter="RunnerAgent",
    )

    assert store.get(ref) == review
    with pytest.raises(EmitterNotAllowed):
        store.put(
            kind="method_implementation_review",
            session_id="s1",
            key="v2",
            obj=review,
            emitter="CoderAgent",
        )


def test_idempotent_put_overwrites(store):
    ref1 = store.put(
        kind="impl_summary", session_id="s1", key="mod1", obj={"v": 1}, emitter="CoderAgent"
    )
    ref2 = store.put(
        kind="impl_summary", session_id="s1", key="mod1", obj={"v": 2}, emitter="CoderAgent"
    )
    assert ref1 == ref2
    assert store.get(ref1) == {"v": 2}  # second overwrites


def test_blob_roundtrip_and_by_keys(store):
    data = b"\x89PNG\r\n\x1a\n fake image bytes"
    ref = store.put_blob(
        kind="screenshot",
        session_id="s1",
        key="scenA/shot1",
        data=data,
        emitter="AcceptanceAgent",
        ext="png",
    )
    assert store.get_blob(ref) == data
    got = store.get_blob_by_keys(
        session_id="s1", scenario_id="scenA", screenshot_id="shot1"
    )
    assert got == data


def test_blob_wrong_emitter(store):
    with pytest.raises(EmitterNotAllowed):
        store.put_blob(
            kind="screenshot",
            session_id="s1",
            key="scenA/shot1",
            data=b"x",
            emitter="CoderAgent",
            ext="png",
        )


def test_manifest_shallow_merge(store):
    assert store.fetch_project_manifest("unknown") == {}
    store.upsert_manifest("s1", {"a": 1, "b": 2}, emitter="RequirementAgent")
    store.upsert_manifest("s1", {"b": 3, "c": 4}, emitter="ArchitectAgent")
    assert store.fetch_project_manifest("s1") == {"a": 1, "b": 3, "c": 4}


def test_manifest_emitter_rejected(store):
    with pytest.raises(EmitterNotAllowed):
        store.upsert_manifest("s1", {"a": 1}, emitter="DocAgent")


def test_latest_acceptance(store):
    assert store.latest_acceptance("s1") is None
    store.put(
        kind="acceptance_report",
        session_id="s1",
        key="r1",
        obj={"all_passed": False},
        emitter="AcceptanceAgent",
    )
    store.put(
        kind="acceptance_report",
        session_id="s1",
        key="r2",
        obj={"all_passed": True},
        emitter="AcceptanceAgent",
    )
    # newest passing one (created_at DESC); r3 written last
    ref3 = store.put(
        kind="acceptance_report",
        session_id="s1",
        key="r3",
        obj={"all_passed": True},
        emitter="AcceptanceAgent",
    )
    assert store.latest_acceptance("s1") == ref3
    # different session has none
    assert store.latest_acceptance("s2") is None


def test_latest_acceptance_production_key_roundtrip(store):
    """Mirror the production write (core._persist_acceptance_report): an
    acceptance_report with key='latest' and all_passed at top level must be
    resolvable by latest_acceptance. Load-bearing for L3 (DocAgent reads the
    passing report via latest_acceptance). 复刻生产写法的 round-trip。"""
    sess = "sess-prod"
    assert store.latest_acceptance(sess) is None
    ref = store.put(
        kind="acceptance_report",
        session_id=sess,
        key="latest",  # exact production key
        obj={
            "mode": "initial",
            "all_passed": True,
            "successful_scenarios": [{"scenario_id": "S1"}],
        },
        emitter="AcceptanceAgent",
    )
    resolved = store.latest_acceptance(sess)
    assert resolved == ref, "passing report at key='latest' must resolve"
    report = store.get(resolved)
    assert report["all_passed"] is True
    assert report["successful_scenarios"][0]["scenario_id"] == "S1"


def test_list_refs(store):
    assert store.list_refs("s1", "audit") == []
    ref_a = store.put(kind="audit", session_id="s1", key="a", obj={"x": 1}, emitter="Orchestrator")
    ref_b = store.put(kind="audit", session_id="s1", key="b", obj={"x": 2}, emitter="Orchestrator")
    ref_c = store.put(kind="audit", session_id="s1", key="c", obj={"x": 3}, emitter="Orchestrator")
    # newest-first (created_at DESC)
    assert store.list_refs("s1", "audit") == [ref_c, ref_b, ref_a]
    # limit truncates to the newest N
    assert store.list_refs("s1", "audit", limit=2) == [ref_c, ref_b]
    # different session / kind see nothing
    assert store.list_refs("s2", "audit") == []
    assert store.list_refs("s1", "requirement") == []


def test_ref_encode_decode_and_regex():
    ref = refs.encode("screenshot", "sess-1", "scenA/shot1")
    assert REF_RE.match(ref)
    parts = refs.decode(ref)
    assert parts.kind == "screenshot"
    assert parts.session_id == "sess-1"
    assert parts.key == "scenA/shot1"  # key with '/' survives round-trip
    with pytest.raises(ValueError):
        refs.decode("not-a-ref")
