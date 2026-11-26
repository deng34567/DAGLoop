"""DeliberationAgent handler: persists a report + publishes deliberation.ready; FAIL-OPEN."""
from __future__ import annotations

import json
from dataclasses import dataclass

from autoresearch.agents.deliberation_agent.handler import DeliberationAgentHandler

_UNIVERSAL = {"approach": "A", "supporting": "S", "expected_baseline": "B=1",
              "chosen_direction": "D", "chosen_baseline_hint": "B=1",
              "convergence": [], "divergence": [], "synthesis": "synth"}


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, raise_always=False):
        self._raise = raise_always
        self.calls = 0

    def chat(self, *, messages, **_):
        self.calls += 1
        if self._raise:
            raise RuntimeError("LLM down")
        return _Resp(json.dumps(_UNIVERSAL))


class _FakeStore:
    def __init__(self):
        self.objs = {}
        self.puts = []

    def get(self, ref):
        return self.objs.get(ref)

    def put(self, *, kind, session_id, key, obj, emitter):
        ref = f"spec://{kind}/{session_id}/{key}"
        self.objs[ref] = obj
        self.puts.append((kind, emitter, obj))
        return ref


class _FakeClient:
    def __init__(self):
        self.published = []

    async def publish(self, *, event_type, session_id, payload, causation_id=None, **_):
        self.published.append((event_type, payload))


@dataclass
class _Env:
    session_id: str
    event_id: str
    payload: dict


def _seed(store, sid="s1"):
    sr = store.put(kind="research_spec", session_id=sid, key="final",
                   obj={"research_goal": "does X help Y?"}, emitter="ProblemModelingAgent")
    lr = store.put(kind="literature_brief", session_id=sid, key="final",
                   obj={"summary": "context"}, emitter="DeepResearchAgent")
    return sr, lr


async def test_handler_persists_report_and_publishes_ready():
    store, client = _FakeStore(), _FakeClient()
    sr, lr = _seed(store)
    h = DeliberationAgentHandler(spec_store=store, llm_client=_FakeLLM(), config_root="/nonexistent")
    await h.handle_requested(client, _Env("s1", "e1",
                             {"research_spec_ref": sr, "literature_brief_ref": lr}))

    kinds = [k for k, _, _ in store.puts]
    assert "deliberation_report" in kinds
    rep = next(o for k, _, o in store.puts if k == "deliberation_report")
    assert len(rep["angles"]) == 3 and rep["synthesis"] == "synth"

    assert len(client.published) == 1
    ev, payload = client.published[0]
    assert ev == "deliberation.ready" and payload["deliberation_report_ref"].startswith("spec://deliberation_report/")


async def test_handler_is_fail_open_when_llm_dies():
    # LLM down -> a MINIMAL report is still persisted and deliberation.ready still fires;
    # the flow must never stall or emit agent.failed on a deliberation failure.
    store, client = _FakeStore(), _FakeClient()
    sr, lr = _seed(store)
    h = DeliberationAgentHandler(spec_store=store, llm_client=_FakeLLM(raise_always=True),
                                 config_root="/nonexistent")
    await h.handle_requested(client, _Env("s1", "e1",
                             {"research_spec_ref": sr, "literature_brief_ref": lr}))

    rep = next(o for k, _, o in store.puts if k == "deliberation_report")
    assert rep["angles"] == []                      # minimal report
    events = [ev for ev, _ in client.published]
    assert events == ["deliberation.ready"]         # NOT agent.failed
