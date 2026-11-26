"""Incremental Method revisions must retain indirect-component type audits."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.orchestrator.tests.unit.test_engine_min import fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.mark.parametrize("role,changed,expected", [
    ("indirect_gain", True, "DeliberatingMethod"),
    ("direct_gain", True, "DeliberatingMethod"),
    ("indirect_gain", False, "DeliberatingMethod"),
])
async def test_incremental_revision_audits_new_indirect_contract_before_execution(
        fake_bus, monkeypatch, role, changed, expected):
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    method = {"approach": "Preserve the accepted model and validate its mechanism.",
              "components": [{
                  "component_id": "check", "name": "Check", "type": role,
                  "depends_on": [], "acceptance_requires": [],
                  "purpose": "Establish the declared intermediate property.",
                  "inputs": ["frozen model"], "outputs": ["validated model"],
                  "mechanism": "Measure the declared intermediate property.",
                  "implementation_check": "Run the specified mechanism and bind its output.",
                  "optimization_space": "Retain the frozen comparison boundary.",
                  "expected_effect": "Establish the intermediate property.",
                  "downstream_contract": "Preserve the validated model.",
                  "functional_check": "failed_checks == 0" if role == "indirect_gain" else "",
                  "ablation_test": "signed_metric_improvement > 0" if role == "direct_gain" else "",
              }]}
    engine.spec_store = SimpleNamespace(get=lambda _ref: method)
    from autoresearch.experiments.repo_improve.component_plan import build_component_plan
    build_component_plan(method)
    monkeypatch.setattr(engine, "_incremental_method_revision", lambda _state: True)

    async def transition(state, target, **_kwargs):
        state.current_state = target

    monkeypatch.setattr(engine, "_transition", transition)
    state = SessionState(
        session_id="incremental-kind", current_state="RevisingMethod", mode="benchmark",
        method_ref="method-old", baseline_result_ref="baseline-frozen",
        method_feedback_in_progress=True, method_feedback_round=1, method_revision_round=1,
        run_record_refs=["run-accepted"],
    )
    new_ref = "method-new" if changed else state.method_ref

    await engine._on_RevisingMethod_method_generated(
        state, SimpleNamespace(payload={"method_ref": new_ref}, event_id="revision"))

    assert state.current_state == expected
    assert state.component_plan_audit_pending is (expected == "DeliberatingMethod")
    assert state.method_ref == new_ref
    assert state.baseline_result_ref == "baseline-frozen"
    assert state.run_record_refs == ["run-accepted"]
    assert state.method_feedback_round == 1
    assert state.method_revision_round == 1
