"""
Tests for ReportAgent's EventBus handler.
ReportAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as memory_agent's FakeBusClient) +
FakeLLM (from this package's tests/conftest.py) + a real tmp SpecStore. The
FakeBusClient does NOT auto-deliver events to subscribed handlers -- each test
calls the handler's event method directly with a constructed Envelope.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.report_agent import core
from autoresearch.agents.report_agent.handler import ReportAgentHandler
from autoresearch.agents.report_agent.tests.conftest import FakeLLM
from autoresearch.eventbus.broker.envelope import EnvelopeValidator
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
_EVENTBUS_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "eventbus" / "schemas"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


REPORT_VALIDATOR = _validator("report.schema.yaml")

RESEARCH_MEMORY = {
    "best_metric": 0.91,
    "best_result_ref": "research_store://run_record/best",
    "verified_findings": [
        "Technique X improves metric over baseline.",
        "Batch size 64 is stable.",
    ],
    "negative_results": ["Technique Z degrades the metric."],
    "excluded_hypotheses": ["Hypothesis about learning rate warmup."],
    "consolidated_insights": ["Technique X is promising."],
    "divergence_seeds": ["Try technique Y next."],
    "next_steps": ["Run technique Y."],
}

RESEARCH_SPEC = {
    "research_question": "Does technique X improve the metric?",
}


@pytest.mark.asyncio
async def test_report_does_not_append_legacy_external_score(store):
    sid = str(uuid.uuid4())
    payload = _seed_report_inputs(store, sid)
    store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
        "verdict": "new_best", "accepted": True,
        "baseline_metric": 1.0, "method_metric": 0.9, "metric_name": "loss",
        "lower_is_better": True,
        "measurement_evidence": {"official_score": 0.83,
                                 "score_pointer": "/official_reward"}}, emitter="RunnerAgent")
    handler = ReportAgentHandler(spec_store=store, llm_client=FakeLLM([_json(_report())]))
    client = FakeBusClient()
    await handler.handle_report_requested(
        client, _make_envelope("report.requested", sid, payload, "ResearchOrchestrator"))
    ready = [event for event in client.published if event.event_type == "report.ready"]
    report = store.get(ready[0].payload["report_ref"])
    assert "Native evaluator score" not in json.dumps(report)
    assert "official_score" not in json.dumps(report)
    assert "official_reward" not in json.dumps(report)


def test_finite_unaccepted_latest_does_not_hide_verified_component_observation():
    from autoresearch.agents.report_agent.handler import _apply_latest_verified_structured_result
    finding = ("Verified component observation: component=compress, baseline=100.0, "
               "metric=50.0, role=direct_gain, implementation=faithful.")
    memory = {"verified_findings": [finding], "best_metric": None,
              "diagnosis": "method_description_reconciliation"}
    latest = {"baseline_metric": 100., "method_metric": 70., "accepted": False,
              "verdict": "valid_not_best", "lower_is_better": True}
    report = _apply_latest_verified_structured_result({}, memory, latest)
    assert "method=70.0" in report["summary"]
    assert finding in report["summary"]
    assert "not complete-Method results" in report["summary"]
    assert report["experiment_ledger"][1]["verdict"] == "inconclusive"
    assert memory["best_metric"] is None


def _report(**overrides: object) -> dict:
    report = {
        "summary": "Technique X improves the metric; technique Z does not.",
        "verified_findings": ["Technique X improves metric over baseline."],
        "negative_results": ["Technique Z degrades the metric."],
        "unverified_ideas": ["Technique W might help."],
        "experiment_not_run": ["Ablation on dataset size."],
        "replication_failed": [],
        "external_data_incomplete": [],
        "analysis_caveats": ["Only one seed was tried."],
        "known_risks": ["Overfitting to validation split."],
        "experiment_ledger": [
            {
                "experiment_id": "exp-1",
                "metric_value": 0.91,
                "verdict": "new_best",
                "notes": "Technique X applied.",
            }
        ],
    }
    report.update(overrides)
    return report


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    """Captures publishes; mirrors the real EventBusClient.publish/subscribe
    signature. Does NOT auto-deliver events to subscribed handlers."""

    def __init__(self) -> None:
        self.emitter_name = "ReportAgent"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        self.published.append(
            _Published(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                causation_id=causation_id,
            )
        )
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    """The subset of the real broker Envelope the handler reads."""

    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


def _make_envelope(event_type: str, session_id: str, payload: dict, emitter: str) -> _Envelope:
    return _Envelope(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type=event_type,
        emitter=emitter,
        payload=payload,
    )


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_report_inputs(
    store: SpecStore,
    session_id: str,
    *,
    research_memory: dict = RESEARCH_MEMORY,
    research_spec: dict | None = None,
) -> dict:
    """Seed the store with a ResearchMemory (as MemoryAgent would), plus an
    optional ResearchSpec (as ProblemModelingAgent would); return the
    report.requested payload (refs)."""
    research_memory_ref = store.put(
        kind="research_memory",
        session_id=session_id,
        key="v1",
        obj=research_memory,
        emitter="MemoryAgent",
    )
    payload: dict[str, Any] = {"research_memory_ref": research_memory_ref}
    if research_spec is not None:
        payload["research_spec_ref"] = store.put(
            kind="research_spec",
            session_id=session_id,
            key="v1",
            obj=research_spec,
            emitter="ProblemModelingAgent",
        )
    return payload


# ---------------------------------------------------------------------------
# (a) report.requested -> report.ready, ref resolves to a schema-valid Report.
# ---------------------------------------------------------------------------


async def test_report_requested_publishes_ready_with_resolvable_ref(store):
    body = "# Detailed report\n\n" + ("Evidence-grounded analysis. " * 60)
    llm = FakeLLM([
        _json(_report()),
        _json({"verdict": "CLEAN", "findings": []}),
        body,
    ])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id, research_spec=RESEARCH_SPEC)

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    ready = [p for p in client.published if p.event_type == "report.ready"]
    assert len(ready) == 1
    result_payload = ready[0].payload
    assert set(result_payload.keys()) == {"report_ref"}

    report = store.get(result_payload["report_ref"])
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == ["Technique X improves metric over baseline."]


async def test_report_requested_works_without_research_spec_ref(store):
    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id)
    assert "research_spec_ref" not in payload

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    ready = [p for p in client.published if p.event_type == "report.ready"]
    assert len(ready) == 1
    report = store.get(ready[0].payload["report_ref"])
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# (b) provenance: research_memory always read from store; research_spec read
# when its ref is given in the payload, and NOT read (None passed to
# core.generate_report) when absent.
# ---------------------------------------------------------------------------


async def test_report_requested_reads_research_spec_when_given(store, monkeypatch):
    captured: dict[str, Any] = {}
    real_generate_report = core.generate_report

    def _spy_generate_report(
        research_memory, research_spec, llm_client, *, prompt_patch=None,
        citation_audit_report=None, reviewer_llm=None, research_context=None,
    ):
        captured["research_memory"] = research_memory
        captured["research_spec"] = research_spec
        captured["research_context"] = research_context
        return real_generate_report(
            research_memory, research_spec, llm_client,
            prompt_patch=prompt_patch, citation_audit_report=citation_audit_report,
            reviewer_llm=reviewer_llm, research_context=research_context,
        )

    monkeypatch.setattr(core, "generate_report", _spy_generate_report)

    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id, research_spec=RESEARCH_SPEC)
    payload["literature_brief_ref"] = store.put(
        kind="literature_brief", session_id=session_id, key="v1",
        obj={"key_papers": [{"title": "Rate Distortion"}]}, emitter="DeepResearchAgent")
    payload["method_ref"] = store.put(
        kind="method", session_id=session_id, key="v1",
        obj={"approach": "derive theorem"}, emitter="MethodAgent")

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    assert captured["research_memory"] == RESEARCH_MEMORY
    assert captured["research_spec"] == RESEARCH_SPEC
    assert captured["research_context"]["literature_brief"]["key_papers"][0]["title"] == "Rate Distortion"
    assert captured["research_context"]["confirmed_method"]["approach"] == "derive theorem"


async def test_report_requested_passes_none_when_research_spec_ref_absent(store, monkeypatch):
    captured: dict[str, Any] = {}
    real_generate_report = core.generate_report

    def _spy_generate_report(
        research_memory, research_spec, llm_client, *, prompt_patch=None,
        citation_audit_report=None, reviewer_llm=None, research_context=None,
    ):
        captured["research_spec"] = research_spec
        return real_generate_report(
            research_memory, research_spec, llm_client,
            prompt_patch=prompt_patch, citation_audit_report=citation_audit_report,
            reviewer_llm=reviewer_llm, research_context=research_context,
        )

    monkeypatch.setattr(core, "generate_report", _spy_generate_report)

    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id)
    assert "research_spec_ref" not in payload

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    assert captured["research_spec"] is None


async def test_report_requested_treats_null_research_spec_ref_as_absent(store, monkeypatch):
    """payload may carry an explicit null research_spec_ref (per the schema's
    nullable optional field) -- this must also be treated as absent."""
    captured: dict[str, Any] = {}
    real_generate_report = core.generate_report

    def _spy_generate_report(
        research_memory, research_spec, llm_client, *, prompt_patch=None,
        citation_audit_report=None, reviewer_llm=None, research_context=None,
    ):
        captured["research_spec"] = research_spec
        return real_generate_report(
            research_memory, research_spec, llm_client,
            prompt_patch=prompt_patch, citation_audit_report=citation_audit_report,
            reviewer_llm=reviewer_llm, research_context=research_context,
        )

    monkeypatch.setattr(core, "generate_report", _spy_generate_report)

    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id)
    payload["research_spec_ref"] = None

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    assert captured["research_spec"] is None


async def test_report_requested_with_none_research_memory_ref_produces_honest_report(store):
    """A session that reached Reporting via a decision/MOSS path that skipped
    ConsolidatingMemory has research_memory_ref=None. The handler must NOT
    call spec_store.get(None) -- it substitutes an empty ResearchMemory so
    core.generate_report still produces an honest (no verified findings)
    Report, and reaches report.ready (never agent.failed)."""
    honest_report = _report(
        summary="Research did not complete: no consolidated findings are available.",
        verified_findings=[],
        negative_results=[],
    )
    llm = FakeLLM([_json(honest_report)])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="v1",
        obj=RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    payload = {"research_memory_ref": None, "research_spec_ref": research_spec_ref}

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    assert not [p for p in client.published if p.event_type == "agent.failed"]

    ready = [p for p in client.published if p.event_type == "report.ready"]
    assert len(ready) == 1
    report = store.get(ready[0].payload["report_ref"])
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == []
    assert report["negative_results"] == []


async def test_report_requested_with_missing_research_memory_ref_produces_honest_report(store):
    """Same as above, but research_memory_ref is absent from the payload
    entirely (not just None)."""
    honest_report = _report(verified_findings=[], negative_results=[])
    llm = FakeLLM([_json(honest_report)])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload: dict = {}

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    assert not [p for p in client.published if p.event_type == "agent.failed"]
    ready = [p for p in client.published if p.event_type == "report.ready"]
    assert len(ready) == 1


async def test_agentic_report_preserves_grounded_memory_without_legacy_gen_flag(
        store, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    memory = {**RESEARCH_MEMORY, "verified_findings": [],
              "negative_results": ["Measured method=1.1 did not beat baseline=1.0."]}
    llm = FakeLLM([_json(_report(verified_findings=[], negative_results=[]))])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id, research_memory=memory)
    store.put(
        kind="gen_experiment_result", session_id=session_id, key="current",
        obj={"verdict": "valid_not_best", "accepted": False,
             "baseline_metric": 1.0, "method_metric": 1.1},
        emitter="RunnerAgent")

    await handler.handle_report_requested(
        client, _make_envelope("report.requested", session_id, payload,
                               "ResearchOrchestrator"))

    ready = [p for p in client.published if p.event_type == "report.ready"]
    report = store.get(ready[0].payload["report_ref"])
    assert report["negative_results"] == memory["negative_results"]


@pytest.mark.parametrize("retain_best", [False, True])
async def test_report_synthesis_receives_selected_attempt_evidence(store, monkeypatch, retain_best):
    from autoresearch.spec_store.identity import artifact_digest
    sid = str(uuid.uuid4())
    selected_method = {"approach": "actually measured mechanism"}
    selected_method_ref = store.put(
        kind="method", session_id=sid,
        key="snapshot-" + artifact_digest(selected_method).split(":", 1)[1],
        obj=selected_method, emitter="MethodAgent")
    latest_method = {"approach": "later unimplemented extension"}
    latest_method_ref = store.put(
        kind="method", session_id=sid, key="current", obj=latest_method, emitter="MethodAgent")
    runtime_trace = [{"attempt_id": "corrected-attempt", "rc": 0,
                      "verified_output_fields": {"/method": 60., "/diagnostics/enabled": True}}]
    selected = {
        "accepted": True, "verdict": "new_best", "baseline_metric": 100.,
        "method_metric": 60., "lower_is_better": True, "metric_name": "size",
        "measurement_evidence": {"valid": True, "attempt_id": "corrected-attempt",
                                 "gates": {"/candidate_valid": True}},
        "method_implementation_review": {"verdict": "faithful", "method_ref": selected_method_ref},
        "attempt_review_evidence": runtime_trace,
        "component_execution": {"mode": "integration"}, "baseline_relative_diff": "",
    }
    selected_ref = store.put(
        kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(selected).split(":", 1)[1],
        obj=selected, emitter="RunnerAgent")
    latest = {**selected, "accepted": False, "method_metric": 90.}
    latest_ref = store.put(
        kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(latest).split(":", 1)[1],
        obj=latest, emitter="RunnerAgent")
    memory = {**RESEARCH_MEMORY, "best_metric": 60., "diagnosis": "valid_improvement",
              "best_result_ref": selected_ref if retain_best else None,
              "consolidated_insights": ["Earlier attempt failed its correctness gate."]}
    payload = _seed_report_inputs(store, sid, research_memory=memory)
    payload["method_ref"] = latest_method_ref
    payload["result_ref"] = latest_ref if retain_best else selected_ref
    store.put(kind="gen_experiment_result", session_id=sid, key="current",
              obj=latest, emitter="RunnerAgent")
    captured = {}
    def synthesize(research_memory, *args, **kwargs):
        captured.update(research_memory)
        captured["context"] = kwargs["research_context"]
        return _report()
    monkeypatch.setattr(core, "generate_report", synthesize)
    handler = ReportAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    await handler.handle_report_requested(
        FakeBusClient(), _make_envelope("report.requested", sid, payload, "ResearchOrchestrator"))
    evidence = captured["selected_result"]
    assert evidence["result_ref"] == selected_ref
    assert evidence["method_metric"] == 60.
    assert evidence["measurement_evidence"]["attempt_id"] == "corrected-attempt"
    assert evidence["measurement_evidence"]["gates"] == {"/candidate_valid": True}
    assert evidence["method_implementation_review"]["verdict"] == "faithful"
    assert evidence["attempt_review_evidence"] == runtime_trace
    assert captured["context"]["confirmed_method"] == selected_method
    assert store.get(payload["research_memory_ref"]) == memory


async def test_latest_verified_structured_win_supersedes_stale_report_headline(store):
    """A corrected winning attempt must not be summarized as its earlier failed attempt."""
    finding = (
        "Method beats baseline: baseline=1.163671 vs "
        "method=1.1586406767640685 (metric value), reproduced, run-time isolated."
    )
    stale_negative = (
        "Earlier implementation drifted from the comparison protocol and was rejected."
    )
    memory = {
        **RESEARCH_MEMORY,
        "best_metric": 1.1586406767640685,
        "verified_findings": [finding],
        "negative_results": [stale_negative],
        "diagnosis": "valid_improvement",
        "diagnosis_evidence": [finding],
    }
    stale_report = _report(
        summary="The old attempt improved by 0.77%, but the result is unreliable.",
        verified_findings=[finding],
        negative_results=[stale_negative],
        analysis_caveats=[],
        experiment_ledger=[{
            "experiment_id": "method",
            "metric_value": 1.1586406767640685,
            "verdict": "inconclusive",
            "notes": "Copied the old implementation review.",
        }],
    )
    llm = FakeLLM([_json(stale_report)])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id, research_memory=memory)
    store.put(
        kind="gen_experiment_result", session_id=session_id, key="current",
        obj={
            "verdict": "new_best",
            "accepted": True,
            "baseline_metric": 1.163671,
            "method_metric": 1.1586406767640685,
            "metric_name": "val_bpb",
            "lower_is_better": True,
            "method_implementation_review": {"verdict": "partial"},
        },
        emitter="RunnerAgent",
    )

    await handler.handle_report_requested(
        client, _make_envelope("report.requested", session_id, payload,
                               "ResearchOrchestrator"))

    ready = [p for p in client.published if p.event_type == "report.ready"]
    report = store.get(ready[0].payload["report_ref"])
    assert "method=1.1586406767640685 beats baseline=1.163671" in report["summary"]
    assert "accepted measured improvement from a partial implementation" in report["summary"]
    assert "0.77" not in report["summary"]
    assert "unreliable" not in report["summary"]
    assert report["negative_results"] == [stale_negative]
    assert any("historical" in item.lower() for item in report["analysis_caveats"])
    assert report["experiment_ledger"] == [
        {
            "experiment_id": "baseline",
            "metric_value": 1.163671,
            "verdict": "pass",
            "notes": "Baseline from the latest verified structured comparison.",
        },
        {
            "experiment_id": "method",
            "metric_value": 1.1586406767640685,
            "verdict": "pass",
            "notes": (
                "Best accepted partial implementation retained; only implemented frozen "
                "components are credited."
            ),
        },
    ]


async def test_best_verified_result_is_not_replaced_by_later_worse_attempt(store):
    """A valid later attempt must not overwrite the best accepted metric in the report."""
    memory = {
        **RESEARCH_MEMORY,
        "best_metric": 0.025558,
        "verified_findings": ["Official evaluator verified BM25 runtime=0.025558 s."],
        "diagnosis": "valid_improvement",
    }
    llm = FakeLLM([_json(_report(summary="Stale latest-only summary."))])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    from autoresearch.spec_store.identity import artifact_digest
    saved = {
        "accepted": True, "verdict": "new_best", "baseline_metric": 10.241726,
        "method_metric": 0.025558, "metric_name": "runtime_s", "lower_is_better": True,
    }
    memory["best_result_ref"] = store.put(
        kind="gen_experiment_result", session_id=session_id,
        key="snapshot-" + artifact_digest(saved).split(":", 1)[1],
        obj=saved, emitter="RunnerAgent")
    payload = _seed_report_inputs(store, session_id, research_memory=memory)
    store.put(
        kind="gen_experiment_result", session_id=session_id, key="current",
        obj={
            "verdict": "new_best",
            "accepted": True,
            "baseline_metric": 10.241726,
            "method_metric": 0.059015,
            "metric_name": "runtime_s",
            "lower_is_better": True,
        },
        emitter="RunnerAgent",
    )

    await handler.handle_report_requested(
        client, _make_envelope("report.requested", session_id, payload,
                               "ResearchOrchestrator"))

    ready = [p for p in client.published if p.event_type == "report.ready"]
    report = store.get(ready[0].payload["report_ref"])
    assert "method=0.025558 beats baseline=10.241726" in report["summary"]
    assert "latest accepted attempt measured 0.059015" in report["summary"].lower()
    assert report["experiment_ledger"] == [
        {
            "experiment_id": "baseline",
            "metric_value": 10.241726,
            "verdict": "pass",
            "notes": "Baseline from the selected verified candidate snapshot.",
        },
        {
            "experiment_id": "method",
            "metric_value": 0.025558,
            "verdict": "pass",
            "notes": "Best accepted method retained from verified ResearchMemory.",
        },
        {
            "experiment_id": "method_latest_accepted",
            "metric_value": 0.059015,
            "verdict": "pass",
            "notes": "Latest accepted attempt, recorded separately from the selected candidate.",
        },
    ]


async def test_latest_partial_structured_result_supersedes_stale_method_headline(store):
    """A current measured intermediate must not be reported as an older method."""
    stale_negative = "An earlier optimizer experiment failed badly."
    memory = {
        **RESEARCH_MEMORY,
        "verified_findings": [],
        "negative_results": [stale_negative],
        "diagnosis": "inconclusive",
    }
    stale_report = _report(
        summary="The primary method was the earlier optimizer experiment.",
        verified_findings=[],
        negative_results=[stale_negative],
        experiment_ledger=[{
            "experiment_id": "old method",
            "metric_value": 1.49,
            "verdict": "fail",
            "notes": "Historical result.",
        }],
    )
    llm = FakeLLM([_json(stale_report)])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id, research_memory=memory)
    store.put(
        kind="gen_experiment_result", session_id=session_id, key="current",
        obj={
            "verdict": "valid_not_best",
            "accepted": False,
            "baseline_metric": 1.047458,
            "method_metric": 1.044472,
            "metric_name": "val_bpb",
            "lower_is_better": True,
            "reasons": ["method implementation review=partial"],
        },
        emitter="RunnerAgent",
    )

    await handler.handle_report_requested(
        client, _make_envelope("report.requested", session_id, payload,
                               "ResearchOrchestrator"))

    ready = [p for p in client.published if p.event_type == "report.ready"]
    report = store.get(ready[0].payload["report_ref"])
    assert "method=1.044472 is numerically better than baseline=1.047458" in report["summary"]
    assert "not accepted" in report["summary"]
    assert "method implementation review=partial" in report["summary"]
    assert "earlier optimizer experiment" not in report["summary"]
    assert report["experiment_ledger"][-1] == {
        "experiment_id": "method",
        "metric_value": 1.044472,
        "verdict": "inconclusive",
        "notes": "Latest measured method was not accepted: method implementation review=partial",
    }


# ---------------------------------------------------------------------------
# (c) fabricated grounded claims are sanitized; malformed output -> agent.failed.
# ---------------------------------------------------------------------------


async def test_report_requested_sanitizes_fabricated_finding(store):
    bad_response = _report(
        verified_findings=["A claim that never appeared in ResearchMemory."]
    )
    llm = FakeLLM([_json(bad_response)])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id)

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    assert not [p for p in client.published if p.event_type == "agent.failed"]
    ready = [p for p in client.published if p.event_type == "report.ready"]
    assert len(ready) == 1
    report = store.get(ready[0].payload["report_ref"])
    assert "A claim that never appeared in ResearchMemory." not in report["verified_findings"]
    assert report["verified_findings"] == []


async def test_report_requested_malformed_llm_response_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id)

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].payload["agent_name"] == "ReportAgent"
    assert failed[0].payload["reason_classification"] == "permanent"

    assert not [p for p in client.published if p.event_type == "report.ready"]


async def test_report_requested_unexpected_error_publishes_agent_failed(store):
    """A non-core exception (e.g. a broken ref) must also become agent.failed,
    never a crash."""
    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"research_memory_ref": "research_memory://does-not-exist/v1"}

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].payload["agent_name"] == "ReportAgent"
    assert failed[0].payload["reason_classification"] in ("transient", "permanent")
    assert failed[0].payload["underlying_reason"]


# ---------------------------------------------------------------------------
# (d) causation_id set.
# ---------------------------------------------------------------------------


async def test_report_requested_causation_id_set(store):
    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id)

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    ready = [p for p in client.published if p.event_type == "report.ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (e) enum prerequisite: a published envelope with emitter:"ReportAgent"
# validates against envelope.schema.json.
# ---------------------------------------------------------------------------


def test_report_agent_emitter_validates_against_envelope_schema():
    schema_path = _EVENTBUS_SCHEMAS_DIR / "envelope.schema.json"
    envelope_dict = {
        "event_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "event_type": "report.ready",
        "emitter": "ReportAgent",
        "timestamp": "2026-07-08T00:00:00Z",
        "schema_version": 1,
        "cost_delta": "0",
        "payload": {
            "report_ref": "report://sess-123/v1",
        },
    }
    errors = EnvelopeValidator.validate(envelope_dict, schema_path)
    assert errors == [], errors


def test_report_agent_agent_failed_validates_against_schema():
    _CONTRACTS = json.loads(
        (_EVENTBUS_SCHEMAS_DIR / "agent.failed.v1.schema.json").read_text()
    )
    validator = Draft202012Validator(_CONTRACTS)
    payload = {
        "agent_name": "ReportAgent",
        "reason_classification": "permanent",
        "underlying_reason": "LLM did not produce a valid response after retries",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# register() wires the subscription.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MOSS prompt_patch: config_root/agent_config/ReportAgent/prompt_patch.md gets
# read by self.config.prompt_patch() and injected into the system message.
# ---------------------------------------------------------------------------


async def test_report_requested_prompt_patch_from_config_root_injected_into_system_message(
    store, tmp_path
):
    patch_dir = tmp_path / "agent_config" / "ReportAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_json(_report())])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_report_inputs(store, session_id, research_spec=RESEARCH_SPEC)

    env = _make_envelope("report.requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_report_requested(client, env)

    messages = llm.calls[0]  # report-generation call (calls[1] is the blind review)
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_register_wires_subscription(store):
    llm = FakeLLM([])
    handler = ReportAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "report.requested" in patterns


async def test_finite_failed_baseline_is_reported_without_promoting_it(store, monkeypatch):
    observation = {"ref": "spec://baseline_result/s/snapshot-observation", "metric_name": "bits",
                   "metric_value": 7.0, "reproduced": False,
                   "measurement_evidence": {"quality_observations": {"/accuracy": 0.9}}}
    memory = {**RESEARCH_MEMORY, "best_metric": None, "best_result_ref": None,
              "verified_findings": [], "negative_results": [],
              "diagnosis": "baseline_reproduction_failed", "baseline_observations": [observation]}
    monkeypatch.setattr(core, "generate_report", lambda *args, **kwargs: _report(
        summary="No measured result was ever produced.", verified_findings=[], negative_results=[],
        experiment_ledger=[], review={"verdict": "CLEAN", "findings": []}))
    client = FakeBusClient()
    handler = ReportAgentHandler(spec_store=store, llm_client=object())
    await handler.handle_report_requested(client, _make_envelope(
        "report.requested", "s", _seed_report_inputs(store, "s", research_memory=memory), "ResearchOrchestrator"))
    assert client.published[-1].event_type == "report.ready"
    report = store.get(client.published[-1].payload["report_ref"])
    assert not list(REPORT_VALIDATOR.iter_errors(report))
    assert "No measured result" not in report["summary"]
    assert "7.0" in report["summary"]
    assert report["experiment_ledger"][0]["verdict"] == "inconclusive"
    assert '"/accuracy": 0.9' in report["experiment_ledger"][0]["notes"]
    assert report["verified_findings"] == []
    assert "review" not in report
