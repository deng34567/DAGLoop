from __future__ import annotations

import json

import pytest

from autoresearch.experiments.repo_improve.component_plan import (
    ComponentLedger,
    ComponentPlanError,
    build_component_plan,
    component_memory_packet,
    evaluate_component_contract,
    explicit_root_checkpoint,
    ledger_for_workroot,
    validate_component_acceptance,
)


def _component(
    component_id: str, role: str = "direct_gain", depends_on=None, acceptance_requires=None,
) -> dict:
    item = {
        "component_id": component_id,
        "name": component_id,
        "type": role,
        "depends_on": list(depends_on or []),
        "purpose": f"purpose for {component_id}",
        "inputs": ["parent checkpoint"],
        "outputs": [f"{component_id} output"],
        "acceptance_requires": list(
            (depends_on or []) if acceptance_requires is None else (acceptance_requires or [])),
        "mechanism": f"mechanism for {component_id}",
        "expected_effect": f"effect for {component_id}",
        "ablation_test": "metric delta > 0",
        "functional_check": "signal >= 0.9",
        "implementation_check": f"host diff and runtime evidence match {component_id}",
        "optimization_space": f"improve the mechanism of {component_id}",
        "downstream_contract": f"provide {component_id} output to declared children",
    }
    return item


@pytest.mark.parametrize("damage", ["json", "list", "empty", "missing-entry", "negative-count", "missing-checkpoint"])
def test_existing_damaged_ledger_fails_closed_without_rewriting(tmp_path, damage):
    plan = build_component_plan({"components": [_component("a"), _component("b")]})
    path = tmp_path / "ledger.json"
    ledger = ComponentLedger(str(path), plan)
    ledger.record_integration_attempt()
    data = ledger.snapshot()
    if damage == "missing-entry":
        del data["entries"]["b"]
    elif damage == "negative-count":
        data["entries"]["a"]["attempts"] = -1
    elif damage == "missing-checkpoint":
        data["entries"]["a"]["status"] = "accepted"
    content = {"json": "{", "list": "[]", "empty": "{}"}.get(damage, json.dumps(data))
    path.write_text(content)
    with pytest.raises(ComponentPlanError):
        ComponentLedger(str(path), plan)
    assert path.read_text() == content


def test_ledger_io_error_preserves_original(tmp_path, monkeypatch):
    import builtins
    path = tmp_path / "ledger.json"
    plan = build_component_plan({"components": [_component("a")]})
    ComponentLedger(str(path), plan)
    original = path.read_bytes()
    real_open = builtins.open
    def denied(name, *args, **kwargs):
        if str(name) == str(path):
            raise PermissionError("test denial")
        return real_open(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", denied)
    with pytest.raises(ComponentPlanError, match="cannot read"):
        ComponentLedger(str(path), plan)
    assert path.read_bytes() == original


def test_task_improvement_cannot_be_proved_by_positive_runtime_scalar():
    component = _component("a", "indirect_gain")
    component["functional_check"] = "signed_metric_improvement > 0"
    plan = build_component_plan({"components": [component]})
    with pytest.raises(ComponentPlanError, match="parent/candidate"):
        validate_component_acceptance(plan)
    result = evaluate_component_contract(
        plan.node("a"), parent_metric=10, candidate_metric=11, lower_is_better=True,
        ablation_report=[{"component": "a", "actual": 11, "_host_bound": True,
                         "evidence": {"source": "runtime", "ref": "attempt", "json_pointer": "/runtime_seconds"}}])
    assert result["passed"] is False
    assert result["plan_invalid"] is True


@pytest.mark.parametrize("instruction", [
    "Run git add solve.py && git commit so HEAD advances.",
    "实现后运行 git commit，使 HEAD 真正前进。",
])
def test_component_plan_rejects_repository_history_mutation_contract(instruction):
    component = _component("change")
    component["implementation_check"] = instruction
    plan = build_component_plan({"components": [component]})

    with pytest.raises(ComponentPlanError, match="git commit or HEAD mutation"):
        validate_component_acceptance(plan)


@pytest.mark.parametrize("instruction", [
    "Do not create a git commit or change HEAD; leave a working-tree diff.",
    "禁止 git commit，不要修改 HEAD，保留工作区 diff。",
])
def test_component_plan_allows_frozen_head_contract(instruction):
    component = _component("change")
    component["implementation_check"] = instruction

    validate_component_acceptance(build_component_plan({"components": [component]}))


@pytest.mark.parametrize("content", ["{", "{}", "[]"])
def test_legacy_ledger_damage_cannot_create_a_fresh_budget(tmp_path, content):
    legacy = tmp_path / "component_ledger.json"
    legacy.write_text(content)
    plan = build_component_plan({"components": [_component("a")]})
    with pytest.raises(ComponentPlanError):
        ledger_for_workroot(str(tmp_path), plan)
    assert legacy.read_text() == content
    assert list(tmp_path.glob("component_ledger-*.json")) == []


def test_pure_composition_evaluation_join_belongs_to_final_verification():
    parent = _component("parent", "indirect_gain")
    join = _component("join", "indirect_gain", ["parent"])
    join.update(purpose="Composition/evaluation join; no new target-specific code is produced",
                mechanism="Materialize the accepted dependency and run the unchanged evaluator",
                optimization_space="none at this join")
    plan = build_component_plan({"components": [parent, join]})
    with pytest.raises(ComponentPlanError, match="preserve its required check"):
        validate_component_acceptance(plan)


@pytest.mark.parametrize("role", ["direct_gain", "indirect_gain"])
@pytest.mark.parametrize("reported_role", ["direct_gain", "indirect_gain"])
@pytest.mark.parametrize("parent_metric", [None, 148.0])
def test_failed_control_gate_uses_frozen_role_not_reported_role(role, reported_role, parent_metric):
    node = build_component_plan({"components": [_component("change", role)]}).node("change")
    report = [{"component": "change", "type": reported_role, "actual": 1.0,
               "_host_bound": True, "evidence": {"source": "runtime_attempt", "ref": "candidate", "json_pointer": "/signal"},
               "parent_metric": 148.0, "parent_evidence": {
                   "source": "runtime_attempt", "ref": "parent", "host_bound": True,
                   "failed_gates": ["candidate_valid"]}}]
    result = evaluate_component_contract(node, parent_metric=parent_metric,
                                        candidate_metric=100.0, lower_is_better=True,
                                        ablation_report=report)
    assert result["passed"] is (role == "indirect_gain")
    if role == "direct_gain":
        assert "parent/control" in result["reason"]
        assert "correctness" in result["reason"]


def test_direct_gain_accepts_host_verified_ineligible_frozen_baseline_control():
    node = build_component_plan({"components": [_component("change")]}).node("change")
    evidence = {
        "source": "runtime_attempt", "ref": "parent", "host_bound": True,
        "failed_gates": ["candidate_valid"], "reproduced_ineligible_baseline": True,
    }
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=100.0, lower_is_better=True,
        ablation_report=[{
            "component": "change", "actual": 100.0, "parent_metric": 148.0,
            "evidence": {"source": "runtime_attempt", "ref": "candidate"},
            "parent_evidence": evidence,
        }],
    )
    assert result["passed"] is True
    assert result["parent_evidence"] == evidence


def test_invalid_direct_checks_report_every_component_and_expression():
    first, second, valid = _component("first"), _component("second"), _component("valid")
    first["ablation_test"] = "custom_score - parent.score > 0.005"
    second["ablation_test"] = "abs(custom_score - 0.08) <= 0.002"
    plan = build_component_plan({"components": [first, second, valid]})
    with pytest.raises(ComponentPlanError) as caught:
        validate_component_acceptance(plan)
    message = str(caught.value)
    assert "first.ablation_test='custom_score - parent.score > 0.005'" in message
    assert "second.ablation_test='abs(custom_score - 0.08) <= 0.002'" in message
    assert "valid.ablation_test" not in message
    assert "signed_metric_improvement" in message


def test_terminal_official_evaluator_is_not_a_component():
    implement = _component("implement", "indirect_gain")
    final_check = _component("final-check", "indirect_gain", ["implement"])
    final_check.update(
        purpose="Measurement-only official evaluation of the unchanged integrated candidate",
        mechanism="Run the official evaluator and report its task score; no source change",
        optimization_space="None — no optimization knobs",
    )
    plan = build_component_plan({"components": [implement, final_check]})
    with pytest.raises(ComponentPlanError, match="only a terminal official evaluation"):
        validate_component_acceptance(plan)

    final_check["purpose"] = "Measure a new intermediate property with added instrumentation"
    validate_component_acceptance(build_component_plan({"components": [implement, final_check]}))


def test_plan_validates_and_orders_dependencies():
    plan = build_component_plan({"components": [
        _component("feature", "indirect_gain"),
        _component("processor", "indirect_gain", ["feature"]),
        _component("loss", "direct_gain", ["processor"], ["feature", "processor"]),
    ]})
    assert plan.topological_order == ("feature", "processor", "loss")
    assert plan.node("feature").acceptance_check == "signal >= 0.9"
    assert plan.node("loss").acceptance_check == "metric delta > 0"


def test_named_verified_prior_checkpoint_becomes_root_parent(tmp_path):
    component = _component("follow-on")
    component["inputs"] = ["Prior checkpoint-8af556bb implementation"]
    plan = build_component_plan({"components": [component]})
    checkpoint = str(tmp_path / "checkpoint-8af556bb1234")
    history = [{"selected_checkpoint": checkpoint, "checkpoint": checkpoint,
                "contract_passed": True, "acceptance_result": {"passed": True},
                "mir_verdict": "faithful", "verification_verdict": "verified"}]
    root = explicit_root_checkpoint(plan, history)
    assert root == checkpoint
    ledger = ledger_for_workroot(str(tmp_path), plan, root_checkpoint=root)
    ledger.begin("follow-on")
    assert ledger.entry("follow-on").parent_checkpoint == checkpoint
    assert ledger.entry("follow-on").parent_checkpoints == [checkpoint]


@pytest.mark.parametrize("inputs", [
    ["checkpoint-a11111", "checkpoint-b22222"],
    ["Copy checkpoint-a11111 and checkpoint-b22222 implementations"],
])
def test_plan_rejects_multiple_executable_root_checkpoints_before_review(inputs):
    from autoresearch.agents.deliberation_agent.core import _component_contract_problem

    component = _component("follow-on")
    component["inputs"] = inputs
    method = {"components": [component]}
    with pytest.raises(ComponentPlanError, match="names multiple root checkpoints"):
        build_component_plan(method)
    problem = _component_contract_problem(method, require_design_fields=True)
    assert isinstance(problem, ComponentPlanError)
    assert "names multiple root checkpoints" in str(problem)


def test_code_copy_reference_does_not_replace_verified_root_checkpoint(tmp_path):
    component = _component("follow-on")
    component["inputs"] = ["Prior checkpoint-8af556bb implementation"]
    component["mechanism"] = "Transplant a helper from checkpoint-b22222 into the parent"
    plan = build_component_plan({"components": [component]})
    checkpoint = str(tmp_path / "checkpoint-8af556bb1234")
    history = [{"selected_checkpoint": checkpoint,
                "contract_passed": True, "acceptance_result": {"passed": True},
                "mir_verdict": "faithful", "verification_verdict": "verified"}]
    assert explicit_root_checkpoint(plan, history) == checkpoint


def test_named_prior_checkpoint_requires_a_unique_verified_observation(tmp_path):
    component = _component("follow-on")
    component["inputs"] = ["Prior checkpoint-8af556bb implementation"]
    plan = build_component_plan({"components": [component]})
    history = [{"selected_checkpoint": str(tmp_path / "checkpoint-8af556bb1234"),
                "contract_passed": True, "acceptance_result": {"passed": True},
                "mir_verdict": "drifted", "verification_verdict": "verified"}]
    with pytest.raises(ComponentPlanError, match="no unique verified"):
        explicit_root_checkpoint(plan, history)


def test_excluded_checkpoint_in_inputs_gets_an_actionable_repair():
    component = _component("follow-on")
    excluded = "Do not inherit checkpoint-a11111; start from the frozen baseline"
    component["inputs"] = ["frozen baseline", excluded]
    plan = build_component_plan({"components": [component]})
    with pytest.raises(ComponentPlanError, match="no unique verified") as error:
        explicit_root_checkpoint(plan, [], baseline_checkpoint="/work/checkpoint-b22222")
    assert "Remove excluded checkpoint references from inputs" in str(error.value)
    assert "mechanism or reference_sources" in str(error.value)

    # Preserve the exclusion as documentation without turning it into a dependency.
    component["inputs"] = ["frozen baseline"]
    component["mechanism"] += " " + excluded
    repaired = build_component_plan({"components": [component]})
    assert explicit_root_checkpoint(
        repaired, [], baseline_checkpoint="/work/checkpoint-b22222") == ""
    assert excluded in repaired.node("follow-on").mechanism


def test_named_frozen_baseline_checkpoint_can_seed_a_new_component(tmp_path):
    component = _component("follow-on")
    component["inputs"] = ["Frozen baseline checkpoint-a11111"]
    plan = build_component_plan({"components": [component]})
    checkpoint = str(tmp_path / "checkpoint-a11111")
    root = explicit_root_checkpoint(plan, [], baseline_checkpoint=checkpoint)
    ledger = ledger_for_workroot(str(tmp_path), plan, root_checkpoint=root)
    assert not ledger.all_accepted()
    ledger.begin("follow-on")
    assert ledger.entry("follow-on").parent_checkpoint == checkpoint


def test_named_reconciled_checkpoint_can_seed_fresh_whole_method_validation(tmp_path):
    component = _component("retained-whole-method")
    component["inputs"] = ["Retained source checkpoint-c33333"]
    plan = build_component_plan({"components": [component]})
    checkpoint = str(tmp_path / "checkpoint-c33333abcdef")
    root = explicit_root_checkpoint(plan, [], reconciled_checkpoint=checkpoint)
    ledger = ledger_for_workroot(str(tmp_path), plan, root_checkpoint=root)
    assert root == checkpoint
    assert not ledger.all_accepted()
    ledger.begin("retained-whole-method")
    assert ledger.entry("retained-whole-method").parent_checkpoint == checkpoint


@pytest.mark.parametrize("baseline_name", ["checkpoint-b22222", ""])
def test_unrelated_baseline_does_not_validate_an_unknown_root(tmp_path, baseline_name):
    component = _component("follow-on")
    component["inputs"] = ["Prior checkpoint-a11111"]
    plan = build_component_plan({"components": [component]})
    baseline = str(tmp_path / baseline_name) if baseline_name else ""
    with pytest.raises(ComponentPlanError, match="no unique verified"):
        explicit_root_checkpoint(plan, [], baseline_checkpoint=baseline)


def test_baseline_and_prior_checkpoint_prefix_must_be_unambiguous(tmp_path):
    component = _component("follow-on")
    component["inputs"] = ["Prior checkpoint-a11111"]
    plan = build_component_plan({"components": [component]})
    history = [{"checkpoint": str(tmp_path / "checkpoint-a111112"),
                "contract_passed": True, "acceptance_result": {"passed": True},
                "mir_verdict": "faithful", "verification_verdict": "verified"}]
    with pytest.raises(ComponentPlanError, match="no unique verified"):
        explicit_root_checkpoint(plan, history,
                                 baseline_checkpoint=str(tmp_path / "checkpoint-a111111"))


def test_different_root_checkpoint_does_not_inherit_prior_component(tmp_path):
    plan = build_component_plan({"components": [_component("follow-on")]})
    first = ledger_for_workroot(str(tmp_path), plan, root_checkpoint="/work/checkpoint-a11111")
    first.begin("follow-on")
    first.record_attempt("follow-on", execution_succeeded=True, contract_passed=True,
                         checkpoint="/work/result", max_attempts=1)
    second = ledger_for_workroot(str(tmp_path), plan, root_checkpoint="/work/checkpoint-b22222")
    assert second.path != first.path
    assert second.entry("follow-on").status == "pending"


def test_plan_exposes_dependency_closure_and_terminal_branches():
    plan = build_component_plan({"components": [
        _component("a"),
        _component("b", depends_on=["a"]),
        _component("c", depends_on=["b"], acceptance_requires=["a", "b"]),
        _component("d", depends_on=["a"]),
        _component("e", depends_on=["d"], acceptance_requires=["a", "d"]),
        _component("f", depends_on=["d"], acceptance_requires=["a", "d"]),
        _component("g", depends_on=["d"], acceptance_requires=["a", "d"]),
        _component("h", depends_on=["e", "f", "g"],
                   acceptance_requires=["a", "d", "e", "f", "g"]),
    ]})

    assert plan.dependency_closure("c") == ("a", "b")
    assert plan.dependency_closure("h") == ("a", "d", "e", "f", "g")
    assert "b" not in plan.dependency_closure("h")
    assert "c" not in plan.dependency_closure("h")
    assert plan.terminal_components == ("c", "h")


def test_plan_canonicalizes_transitive_dependency_to_immediate_parents():
    plan = build_component_plan({"components": [
        _component("a"),
        _component("b", depends_on=["a"]),
        _component("c", depends_on=["a", "b"], acceptance_requires=["a", "b"]),
    ]})

    assert plan.node("c").depends_on == ("b",)
    assert plan.dependency_closure("c") == ("a", "b")


@pytest.mark.parametrize("missing", [
    "purpose", "inputs", "outputs", "acceptance_requires",
    "implementation_check", "optimization_space", "downstream_contract",
])
def test_component_design_fields_are_required(missing):
    component = _component("a")
    component.pop(missing)
    with pytest.raises(ComponentPlanError, match=missing):
        build_component_plan({"components": [component]})


def test_acceptance_prerequisites_cannot_omit_parent_or_add_unrelated_branch():
    missing_parent = _component("b", depends_on=["a"])
    missing_parent["acceptance_requires"] = []
    with pytest.raises(ComponentPlanError, match="full dependency closure"):
        build_component_plan({"components": [_component("a"), missing_parent]})

    unrelated = _component("b", depends_on=["a"])
    unrelated["acceptance_requires"] = ["a", "c"]
    with pytest.raises(ComponentPlanError, match="unrelated"):
        build_component_plan({"components": [_component("a"), unrelated, _component("c")]})


def test_missing_transitive_acceptance_dependency_names_required_and_missing_ids():
    with pytest.raises(ComponentPlanError) as error:
        build_component_plan({"components": [
            _component("a"), _component("b", depends_on=["a"]),
            _component("c", depends_on=["b"], acceptance_requires=["b"]),
        ]})
    assert "required IDs: ['a', 'b']" in str(error.value)
    assert "missing IDs: ['a']" in str(error.value)


def test_artifact_acceptance_prerequisite_reports_component_id_contract():
    with pytest.raises(ComponentPlanError) as error:
        build_component_plan({"components": [
            _component("prepare"),
            _component("consumer", depends_on=["prepare"],
                       acceptance_requires=["prepare:weights.npz"]),
        ]})
    message = str(error.value)
    assert "acceptance_requires must contain component IDs" in message
    assert "['consumer', 'prepare']" in message
    assert "artifact/output bindings" in message.lower() and "inputs" in message


@pytest.mark.parametrize("comparison", [
    "Keep the same weight-decay-tune control in both runs",
    "Do not enable weight-decay-tune",
])
def test_mentioning_a_sibling_does_not_prove_it_is_a_dependency(comparison):
    first = _component("beta-tune")
    first["ablation_test"] = (
        f"{comparison}; compare beta-tune on versus off; metric delta > 0")
    plan = build_component_plan({"components": [first, _component("weight-decay-tune")]})
    assert plan.node("beta-tune").depends_on == ()


def test_acceptance_check_may_name_its_declared_dependency_closure():
    child = _component(
        "warmup-ramp", depends_on=["beta-tune"], acceptance_requires=["beta-tune"])
    child["ablation_test"] = (
        "With beta_tune already accepted, compare warmup-ramp on versus off; metric delta > 0")
    plan = build_component_plan({"components": [_component("beta-tune"), child]})
    assert plan.node("warmup-ramp").acceptance_requires == ("beta-tune",)


@pytest.mark.parametrize(
    "components",
    [
        [_component("a", depends_on=["missing"])],
        [_component("a", depends_on=["b"]), _component("b", depends_on=["a"])],
        [_component("a"), _component("a")],
    ],
    ids=["missing-dependency", "cycle", "duplicate"],
)
def test_invalid_dag_is_rejected(components):
    with pytest.raises(ComponentPlanError):
        build_component_plan({"components": components})


def test_missing_parent_feedback_names_declared_nodes_without_inventing_parent():
    child = _component("child", depends_on=["retained_parent"])
    with pytest.raises(ComponentPlanError) as error:
        build_component_plan({"components": [child]})
    assert "unknown dependencies: ['retained-parent']" in str(error.value)
    assert "declared component IDs: ['child']" in str(error.value)
    assert "including retained prerequisite contracts" in str(error.value)
    parent = _component("retained_parent")
    plan = build_component_plan({"components": [parent, child]})
    assert plan.node("child").depends_on == ("retained-parent",)


def test_independent_components_branch_from_baseline(tmp_path):
    plan = build_component_plan({"components": [_component("a"), _component("b")]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)

    assert ledger.next_ready().component_id == "a"
    ledger.begin("a")
    first = ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-a", quality_value=1.0)
    assert first.status == "active"
    ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-a-worse", quality_value=1.1)

    assert ledger.next_ready().component_id == "b"
    ledger.begin("b")
    assert ledger.entry("b").parent_checkpoint == "baseline"


def test_dependent_component_waits_for_parent_checkpoint(tmp_path):
    plan = build_component_plan({"components": [
        _component("feature", "indirect_gain"),
        _component("loss", "direct_gain", ["feature"]),
    ]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("feature")
    ledger.record_attempt(
        "feature", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-feature", quality_value=0.95, quality_lower_is_better=False,
    )
    ledger.record_attempt(
        "feature", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-feature-worse", quality_value=0.94, quality_lower_is_better=False,
    )
    assert ledger.next_ready().component_id == "loss"
    ledger.begin("loss")
    assert ledger.entry("loss").parent_checkpoint == "cp-feature"


def test_indirect_selected_candidate_carries_matching_metric_and_diff(tmp_path):
    plan = build_component_plan({"components": [
        _component("feature", "indirect_gain"),
        _component("loss", "direct_gain", ["feature"]),
    ]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("feature")
    for checkpoint, quality, metric in (("cp-first", 0.95, 3.0),
                                        ("cp-second", 0.99, 2.0)):
        ledger.record_attempt(
            "feature", execution_succeeded=True, contract_passed=True,
            checkpoint=checkpoint, quality_value=quality,
            quality_lower_is_better=False, metric_value=metric,
            diff=f"diff-{checkpoint}")
    assert ledger.entry("feature").checkpoint == "cp-second"
    selected = ledger.decide_qualified(
        "feature", result_ref="result-second", action="advance",
        reason="first candidate has the interface needed downstream",
        selected_checkpoint="cp-first")
    assert (selected.checkpoint, selected.metric_value, selected.diff) == (
        "cp-first", 3.0, "diff-cp-first")
    ledger.begin("loss")
    assert ledger.entry("loss").parent_checkpoints == ["cp-first"]
    with pytest.raises(ComponentPlanError, match="already finalized"):
        ledger.decide_qualified(
            "feature", result_ref="late-result", action="advance",
            reason="too late", selected_checkpoint="cp-second")


def test_attempt_cap_selection_cannot_change_started_child_parent(tmp_path):
    plan = build_component_plan({"components": [
        _component("feature", "indirect_gain"),
        _component("loss", "direct_gain", ["feature"]),
    ]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("feature")
    for checkpoint in ("first", "second"):
        ledger.record_attempt(
            "feature", execution_succeeded=True, contract_passed=True,
            checkpoint=checkpoint, quality_value=1.0, max_attempts=2)
    ledger.begin("loss")
    with pytest.raises(ComponentPlanError, match="downstream work starts"):
        ledger.decide_qualified(
            "feature", result_ref="late-result", action="advance",
            reason="too late", selected_checkpoint="second")


def test_direct_gain_retains_incumbent_within_selection_tolerance(tmp_path):
    plan = build_component_plan({"components": [_component("gain")]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("gain")
    for checkpoint, metric in (("cp-1", 1.0), ("cp-2", 0.99)):
        ledger.record_attempt(
            "gain", execution_succeeded=True, contract_passed=True,
            checkpoint=checkpoint, metric_value=metric, quality_value=metric,
            diff=f"diff-{checkpoint}", minimum_metric_improvement=0.1)
    assert (ledger.entry("gain").checkpoint, ledger.entry("gain").diff) == (
        "cp-1", "diff-cp-1")


def test_join_component_keeps_all_immediate_parent_checkpoints(tmp_path):
    plan = build_component_plan({"components": [
        _component("a"),
        _component("d", depends_on=["a"]),
        _component("e", depends_on=["d"], acceptance_requires=["a", "d"]),
        _component("f", depends_on=["d"], acceptance_requires=["a", "d"]),
        _component("g", depends_on=["d"], acceptance_requires=["a", "d"]),
        _component("h", depends_on=["e", "f", "g"],
                   acceptance_requires=["a", "d", "e", "f", "g"]),
    ]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    for component_id, checkpoint in (
        ("a", "cp-a"), ("d", "cp-d"), ("e", "cp-e"),
        ("f", "cp-f"), ("g", "cp-g"),
    ):
        ledger.begin(component_id)
        ledger.record_attempt(
            component_id, execution_succeeded=True, contract_passed=True,
            checkpoint=checkpoint, quality_value=1.0,
        )
        ledger.record_attempt(
            component_id, execution_succeeded=True, contract_passed=True,
            checkpoint=checkpoint + "-worse", quality_value=2.0,
        )

    ledger.begin("h")
    entry = ledger.entry("h")
    assert entry.parent_checkpoints == ["cp-e", "cp-f", "cp-g"]
    assert entry.parent_checkpoint == "dependency-set:e,f,g"

    recovered = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    assert recovered.next_ready().component_id == "h"
    assert recovered.entry("e").checkpoint == "cp-e"
    assert recovered.entry("f").checkpoint == "cp-f"
    assert recovered.entry("g").checkpoint == "cp-g"


def test_failures_count_and_do_not_advance(tmp_path):
    plan = build_component_plan({"components": [_component("a"), _component("b")]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("a")
    first = ledger.record_attempt("a", execution_succeeded=False, error="compile failed")
    second = ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=False,
        contract_result={"actual": 0.1, "expected": 0.2},
    )
    assert first.attempts == 1
    assert second.attempts == 2
    assert ledger.next_ready().component_id == "a"
    assert ledger.entry("b").status == "pending"


@pytest.mark.parametrize("execution_started", [False, True])
def test_failed_component_resumes_unverified_tree_without_qualifying_it(tmp_path, execution_started, monkeypatch):
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.experiments.repo_improve import checkpoint_artifacts
    snapshots = []
    def snapshot(source, attempt_id):
        snapshots.append((source, attempt_id))
        return "frozen-unqualified-a"
    monkeypatch.setattr(checkpoint_artifacts, "snapshot_checkpoint", snapshot)
    plan = build_component_plan({"components": [_component("a"), _component("b", depends_on=["a"])]})
    path = str(tmp_path / "ledger.json")
    ledger = ComponentLedger(path, plan, root_checkpoint="baseline")
    ledger.begin("a")
    ledger.record_attempt("a", execution_succeeded=False, execution_started=execution_started,
                          checkpoint="unfinished-a", error="provider timeout")
    recovered = ComponentLedger(path, plan, root_checkpoint="baseline")
    assert recovered.entry("a").debug_checkpoint == "unfinished-a"
    assert recovered.entry("a").checkpoint == ""
    assert recovered.entry("a").qualified_attempts == 0
    assert not recovered.all_accepted()
    assert recovered.next_ready().component_id == "a"
    handler = RunnerAgentHandler.__new__(RunnerAgentHandler)
    handler._active_component_context = {"s": {"mode": "component", "ledger": recovered,
                                              "node": plan.node("a"), "operator": "debug"}}
    assert handler._prior_worktree("s") == "frozen-unqualified-a"
    assert snapshots == [("unfinished-a", "debug-s")]
    assert handler._prior_worktree("s", dependencies_only=True) == "baseline"
    assert recovered.entry("b").status == "pending"


def test_failed_debug_tree_does_not_replace_a_qualified_checkpoint(tmp_path):
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    plan = build_component_plan({"components": [_component("a")]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("a")
    ledger.record_attempt("a", execution_succeeded=True, contract_passed=True,
                          checkpoint="qualified-a", quality_value=1.0)
    ledger.record_attempt("a", execution_succeeded=False,
                          checkpoint="unfinished-upgrade", error="provider timeout")
    handler = RunnerAgentHandler.__new__(RunnerAgentHandler)
    handler._active_component_context = {"s": {"mode": "component", "ledger": ledger,
                                              "node": plan.node("a"), "operator": "debug"}}
    assert handler._prior_worktree("s") == "qualified-a"


def test_checkpoint_recovery_resumes_active_component(tmp_path):
    path = str(tmp_path / "ledger.json")
    plan = build_component_plan({"components": [_component("a"), _component("b", depends_on=["a"])]})
    ledger = ComponentLedger(path, plan)
    ledger.begin("a")
    ledger.record_attempt("a", execution_succeeded=False, error="temporary failure")

    recovered = ComponentLedger(path, plan)

    assert recovered.next_ready().component_id == "a"
    assert recovered.entry("a").attempts == 1


def test_integration_attempt_count_survives_restart(tmp_path):
    path = str(tmp_path / "ledger.json")
    plan = build_component_plan({"components": [_component("a")]})
    ledger = ComponentLedger(path, plan)

    assert ledger.record_integration_attempt() == 1
    assert ComponentLedger(path, plan).record_integration_attempt() == 2


def test_revised_method_gets_a_separate_ledger_without_overwriting_prior_evidence(tmp_path):
    first_plan = build_component_plan({"components": [_component("a")]})
    first = ledger_for_workroot(str(tmp_path), first_plan)
    first.begin("a")
    first.record_attempt("a", execution_succeeded=False, error="evidence from prior method")

    revised_plan = build_component_plan({"components": [_component("a"), _component("b")]})
    revised = ledger_for_workroot(str(tmp_path), revised_plan)

    assert revised.path != first.path
    assert first.entry("a").last_error == "evidence from prior method"
    assert revised.next_ready().component_id == "a"


def _recoverable_checkpoint(tmp_path, label):
    import subprocess
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
    repo = tmp_path / ("repo-" + label)
    repo.mkdir()
    (repo / "code.py").write_text("# " + label)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@local",
                    "commit", "-qm", "fixture"], check=True)
    return snapshot_checkpoint(str(repo), "attempt-" + label)


@pytest.mark.parametrize("reviewed", [False, True])
def test_revision_does_not_inherit_unreviewed_indirect_probe(tmp_path, reviewed):
    root = _component("routing", role="indirect_gain")
    first = ledger_for_workroot(str(tmp_path), build_component_plan({
        "components": [root, _component("leaf")]}))
    first.begin("routing")
    contract = {"passed": True}
    if reviewed:
        contract["measurement_review"] = {"verdict": "pass"}
    first.record_attempt("routing", execution_succeeded=True, contract_passed=True,
                         checkpoint=_recoverable_checkpoint(tmp_path, "routing"),
                         contract_result=contract, max_attempts=1)
    revised = ledger_for_workroot(str(tmp_path), build_component_plan({
        "components": [root, _component("leaf", depends_on=["routing"])]}))
    assert first.entry("routing").status == "accepted"
    assert first.entry("routing").attempts == 1
    assert revised.entry("routing").status == ("accepted" if reviewed else "pending")


def test_component_plan_revision_keeps_unchanged_accepted_parent(tmp_path):
    """Repairing a child's dependency must not discard a valid accepted parent checkpoint."""
    first_plan = build_component_plan({"approach": "same core", "components": [
        _component("wsd"), _component("tail"),
    ]})
    first = ledger_for_workroot(str(tmp_path), first_plan)
    checkpoint = _recoverable_checkpoint(tmp_path, "wsd")
    first.begin("wsd")
    first.record_attempt(
        "wsd", execution_succeeded=True, contract_passed=True,
        checkpoint=checkpoint, metric_value=1.50, quality_value=1.50)
    first.record_attempt(
        "wsd", execution_succeeded=True, contract_passed=True,
        checkpoint="checkpoint-wsd-later", metric_value=1.51, quality_value=1.51)
    assert first.entry("wsd").status == "accepted"

    repaired_tail = _component("tail", depends_on=["wsd"])
    repaired_tail["acceptance_requires"] = ["wsd"]
    revised_plan = build_component_plan({"approach": "same core", "components": [
        _component("wsd"), repaired_tail,
    ]})
    revised = ledger_for_workroot(str(tmp_path), revised_plan)

    assert revised.path != first.path
    assert revised.entry("wsd").status == "accepted"
    assert revised.entry("wsd").checkpoint == checkpoint
    assert revised.entry("tail").status == "pending"
    assert revised.next_ready().component_id == "tail"
    assert revised.snapshot()["inherited_accepted_components"][0]["component_id"] == "wsd"


def test_component_plan_revision_keeps_unchanged_remote_parent(tmp_path, monkeypatch):
    """A DGX checkpoint is not mounted on the Web host, but remains recoverable."""
    import subprocess

    first_plan = build_component_plan({"components": [
        _component("root"), _component("leaf"),
    ]})
    first = ledger_for_workroot(str(tmp_path), first_plan)
    checkpoint = "/remote/checkpoint-" + "a" * 64
    first.begin("root")
    first.record_attempt("root", execution_succeeded=True, contract_passed=True,
                         checkpoint=checkpoint, max_attempts=1)
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE", "1")
    monkeypatch.setenv("AUTORESEARCH_GPU_HOST", "test-gpu")
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE_DIR", "/remote/runtime")
    seen = []

    def remote_check(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", remote_check)
    leaf = _component("leaf", depends_on=["root"])
    revised = ledger_for_workroot(str(tmp_path), build_component_plan({"components": [
        _component("root"), leaf,
    ]}))

    assert revised.entry("root").status == "accepted"
    assert revised.entry("root").checkpoint == checkpoint
    assert revised.next_ready().component_id == "leaf"
    assert seen and seen[0][0] == "ssh"


def test_remote_checkpoint_failure_does_not_inherit_parent(tmp_path, monkeypatch):
    import subprocess

    first = ledger_for_workroot(str(tmp_path), build_component_plan({"components": [
        _component("root"), _component("leaf"),
    ]}))
    first.begin("root")
    first.record_attempt("root", execution_succeeded=True, contract_passed=True,
                         checkpoint="/remote/checkpoint-" + "b" * 64, max_attempts=1)
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE", "1")
    monkeypatch.setenv("AUTORESEARCH_GPU_HOST", "test-gpu")
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE_DIR", "/remote/runtime")
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 1, stdout="", stderr="invalid"))

    revised = ledger_for_workroot(str(tmp_path), build_component_plan({"components": [
        _component("root"), _component("leaf", depends_on=["root"]),
    ]}))

    assert revised.entry("root").status == "pending"
    assert revised.next_ready().component_id == "root"


@pytest.mark.parametrize("changed", ["root", "middle", "unrelated"])
def test_revision_inherits_only_dependency_complete_evidence(tmp_path, changed):
    components = [
        _component("root"),
        _component("middle", depends_on=["root"]),
        _component("leaf", depends_on=["middle"], acceptance_requires=["root", "middle"]),
        _component("unrelated"),
    ]
    first_plan = build_component_plan({"components": components})
    first = ledger_for_workroot(str(tmp_path), first_plan)
    for cid in first_plan.topological_order:
        first.begin(cid)
        first.record_attempt(cid, execution_succeeded=True, contract_passed=True,
                             checkpoint=_recoverable_checkpoint(tmp_path, cid), max_attempts=1)
    components = [dict(c, mechanism="revised mechanism") if c["component_id"] == changed
                  else c for c in components]
    revised_plan = build_component_plan({"components": components})
    revised = ledger_for_workroot(str(tmp_path), revised_plan)
    for cid in revised_plan.topological_order:
        needs_retest = cid == changed or changed in revised_plan.dependency_closure(cid)
        assert revised.entry(cid).status == ("pending" if needs_retest else "accepted")
        assert first.entry(cid).status == "accepted"
    resumed = ledger_for_workroot(str(tmp_path), revised_plan)
    assert resumed.snapshot() == revised.snapshot()


def test_revision_does_not_inherit_child_tested_on_different_parent_checkpoint(tmp_path):
    root = _component("root")
    leaf = _component("leaf", depends_on=["root"])
    first = ledger_for_workroot(str(tmp_path), build_component_plan({"components": [root, leaf]}))
    for cid in ("root", "leaf"):
        first.begin(cid)
        first.record_attempt(cid, execution_succeeded=True, contract_passed=True,
                             checkpoint=f"old-{cid}", max_attempts=1)
    plan = build_component_plan({"components": [leaf, root, _component("extra")]})
    suffix = plan.method_fingerprint.removeprefix("sha256:")[:16]
    current = ComponentLedger(str(tmp_path / f"component_ledger-{suffix}.json"), plan)
    current.begin("root")
    current.record_attempt("root", execution_succeeded=True, contract_passed=True,
                           checkpoint="new-root", max_attempts=1)

    resumed = ledger_for_workroot(str(tmp_path), plan)

    assert resumed.entry("root").checkpoint == "new-root"
    assert resumed.entry("leaf").status == "pending"
    assert first.entry("leaf").status == "accepted"
    assert first.entry("leaf").parent_checkpoints == ["old-root"]


def test_passing_component_improves_from_best_until_first_non_improvement(tmp_path):
    plan = build_component_plan({"components": [_component("a")]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("a")

    first = ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-1", metric_value=0.9, quality_value=0.9)
    better = ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-2", metric_value=0.8, quality_value=0.8)
    stopped = ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-3", metric_value=0.85, quality_value=0.85)

    assert first.status == "active"
    assert better.status == "active"
    assert stopped.status == "accepted"
    assert stopped.checkpoint == "cp-2"
    assert stopped.metric_value == pytest.approx(0.8)


def test_indirect_component_uses_official_metric_to_break_equal_contract_quality(tmp_path):
    plan = build_component_plan({"components": [_component("a", "indirect_gain")]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("a")

    ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-1", metric_value=1.16, metric_lower_is_better=True,
        quality_value=0.1, quality_lower_is_better=True,
    )
    better = ledger.record_attempt(
        "a", execution_succeeded=True, contract_passed=True,
        checkpoint="cp-2", metric_value=1.15, metric_lower_is_better=True,
        quality_value=0.1, quality_lower_is_better=True,
    )

    assert better.status == "active"
    assert better.checkpoint == "cp-2"
    assert better.metric_value == pytest.approx(1.15)


def test_unaccepted_component_cannot_unlock_child(tmp_path):
    plan = build_component_plan({"components": [_component("a"), _component("b", depends_on=["a"])]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    with pytest.raises(ComponentPlanError):
        ledger.begin("b")


def test_direct_gain_contract_uses_parent_metric_and_direction():
    node = build_component_plan({"components": [_component("gain")] }).node("gain")
    result = evaluate_component_contract(
        node, parent_metric=1.0, candidate_metric=0.9, lower_is_better=True)
    assert result["passed"] is True
    assert result["actual"] == pytest.approx(0.1)


def test_direct_gain_retains_real_gain_below_cap_expectation():
    component = _component("gain")
    component["ablation_test"] = "signed_metric_improvement >= 0.2"
    node = build_component_plan({"components": [component]}).node("gain")
    result = evaluate_component_contract(
        node, parent_metric=1.0, candidate_metric=0.9, lower_is_better=True)
    assert result["passed"] is True
    assert result["actual"] == pytest.approx(0.1)
    assert result["acceptance_check_result"]["passed"] is False
    assert "expected gain was not reached" in result["reason"]


def test_direct_gain_still_rejects_no_improvement_below_cap_expectation():
    component = _component("gain")
    component["ablation_test"] = "signed_metric_improvement >= 0.2"
    node = build_component_plan({"components": [component]}).node("gain")
    result = evaluate_component_contract(
        node, parent_metric=1.0, candidate_metric=1.0, lower_is_better=True)
    assert result["passed"] is False
    assert result["acceptance_check_result"]["passed"] is False


@pytest.mark.parametrize("parent,candidate", [
    (float("inf"), 0.9), (1.0, float("-inf")),
    (float("nan"), 0.9), (1.0, float("nan")),
    (True, 0.9), (1.0, False),
])
def test_direct_gain_requires_finite_non_boolean_measurements(parent, candidate):
    node = build_component_plan({"components": [_component("gain")]}).node("gain")
    result = evaluate_component_contract(
        node, parent_metric=parent, candidate_metric=candidate, lower_is_better=True)
    assert result["passed"] is False
    assert "measurement" not in result


def test_direct_gain_join_uses_grounded_composite_parent_measurement():
    node = build_component_plan({"components": [
        _component("e"), _component("f"), _component("h", depends_on=["e", "f"]),
    ]}).node("h")
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=0.8, lower_is_better=True,
        ablation_report=[{
            "component": "h", "parent_metric": 1.0,
            "evidence": {"host_bound": True, "source": "official evaluator", "ref": "attempt:parent"},
        }],
    )
    assert result["passed"] is True
    assert result["parent_metric"] == pytest.approx(1.0)


def test_direct_gain_preserves_explicit_control_evidence_not_candidate_evidence():
    node = build_component_plan({"components": [
        _component("e"), _component("f"), _component("h", depends_on=["e", "f"]),
    ]}).node("h")
    control_evidence = {"source": "frozen_baseline_reference", "ref": "baseline-result", "host_bound": True}
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=0.8, lower_is_better=True,
        ablation_report=[{
            "component": "h", "parent_metric": 1.0,
            "evidence": {"source": "runtime_attempt", "ref": "attempt:candidate"},
            "parent_evidence": control_evidence,
        }],
    )
    assert result["passed"] is True
    assert result["parent_evidence"] == control_evidence


@pytest.mark.parametrize("control_evidence", [None, {}, {"source": "measured"}])
def test_direct_gain_does_not_replace_incomplete_explicit_control_evidence(
    control_evidence,
):
    node = build_component_plan({"components": [_component("h")]}).node("h")
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=0.8, lower_is_better=True,
        ablation_report=[{
            "component": "h", "parent_metric": 1.0,
            "evidence": {"source": "runtime_attempt", "ref": "attempt:candidate"},
            "parent_evidence": control_evidence,
        }],
    )
    assert result["passed"] is False
    assert "measurement" not in result


def test_indirect_gain_contract_rejects_conflicting_reported_threshold():
    node = build_component_plan({"components": [_component("feature", "indirect_gain")] }).node(
        "feature")
    result = evaluate_component_contract(
        node,
        parent_metric=1.0,
        candidate_metric=1.1,
        lower_is_better=True,
        ablation_report=[{"_host_bound": True,
            "component": "feature",
            "actual": 0.95,
            "expected": {"operator": ">", "value": 999},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-1", "json_pointer": "/signal"},
        }],
    )
    assert result["passed"] is False
    assert "does not match the active ComponentPlan" in result["reason"]
    assert "measurement" not in result


@pytest.mark.parametrize("reported,actual,passed", [
    ({"operator": ">=", "value": 0.9}, 0.95, True),
    ({"operator": ">=", "value": "0.9"}, 0.8, False),
    (None, 0.95, True),
])
def test_indirect_gain_matching_or_omitted_threshold_uses_frozen_plan(reported, actual, passed):
    node = build_component_plan({"components": [_component("feature", "indirect_gain")]}).node(
        "feature")
    item = {"_host_bound": True, "component": "feature", "actual": actual,
            "evidence": {"source": "runtime_attempt", "ref": "attempt-1", "json_pointer": "/signal"}}
    if reported is not None:
        item["expected"] = reported
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=None, lower_is_better=True,
        ablation_report=[item])
    assert result["passed"] is passed
    assert result["measurement"] == "indirect_gain_check"
    assert result["expected"] == {"operator": ">=", "value": 0.9}


def test_indirect_gain_checks_orthogonality_even_when_final_metric_is_worse():
    component = _component("orthogonal-representation", "indirect_gain")
    component["functional_check"] = "orthogonality_error <= 0.01"
    node = build_component_plan({"components": [component]}).node(
        "orthogonal-representation")

    result = evaluate_component_contract(
        node,
        parent_metric=1.0,
        candidate_metric=1.2,
        lower_is_better=True,
        ablation_report=[{"_host_bound": True,
            "component": "orthogonal-representation",
            "actual": 0.008,
            "evidence": {"source": "runtime_attempt", "ref": "attempt-orthogonal", "json_pointer": "/orthogonality_error"},
        }],
    )

    assert result["passed"] is True
    assert result["measurement"] == "indirect_gain_check"


@pytest.mark.parametrize("encoding", [str, json.dumps, lambda value: value])
@pytest.mark.parametrize("actual,passed", [(0, True), (1, False)])
def test_structured_indirect_check_uses_explicit_comparison(encoding, actual, passed):
    component = _component("check", "indirect_gain")
    check = {
        "test_procedure": "Compute size_ok = (size == 8) and aligned = (address % 64 == 0). "
                          "Return failed_check_count = int(not size_ok) + int(not aligned).",
        "expected_output": "failed_check_count == 0; size_ok == True; aligned == True",
        "numeric_comparison": "failed_check_count <= 0",
        "emit_auxiliary_values": False,
    }
    component["functional_check"] = encoding(check)
    node = build_component_plan({"components": [component]}).node("check")
    assert node.threshold_operator == "<="
    assert node.threshold_value == 0
    assert node.acceptance_check == str(component["functional_check"])
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=None, lower_is_better=True,
        ablation_report=[{"_host_bound": True, "component": "check", "actual": actual,
                          "evidence": {"source": "runtime_attempt", "ref": "attempt-check", "json_pointer": "/failed_check_count"}}])
    assert result["passed"] is passed


@pytest.mark.parametrize("comparison", [None, True, [], "count == 0 AND error < 0.1", "count <= limit"])
def test_structured_indirect_check_rejects_invalid_explicit_comparison(comparison):
    component = _component("check", "indirect_gain")
    component["functional_check"] = str({
        "test_procedure": "Return count = number of failed cases; auxiliary_error < 0.1.",
        "numeric_comparison": comparison,
    })
    with pytest.raises(ComponentPlanError):
        build_component_plan({"components": [component]})


def test_indirect_gain_requires_one_aggregate_scalar_threshold():
    component = _component("scheduler", "indirect_gain")
    component["functional_check"] = "warmup_at_0 == 0 AND warmup_at_20 == 0.5"

    with pytest.raises(ComponentPlanError, match="one aggregate scalar threshold"):
        build_component_plan({"components": [component]})

    component["functional_check"] = "failed_check_count == 0"
    plan = build_component_plan({"components": [component]})
    assert plan.node("scheduler").threshold_operator == "=="
    assert plan.node("scheduler").threshold_value == 0


@pytest.mark.parametrize("comparison", ["observed < reference", "observed <= reference.npy"])
def test_indirect_variable_threshold_error_explains_equivalent_scalar_repair(comparison):
    component = _component("check", "indirect_gain")
    component["functional_check"] = comparison
    with pytest.raises(ComponentPlanError) as exc:
        build_component_plan({"components": [component]})
    message = str(exc.value)
    assert "finite numeric literal" in message
    assert "not a variable or file path" in message
    assert "difference = observed - reference" in message
    assert "preserve the original comparison operator" in message


def test_indirect_arithmetic_rhs_is_not_silently_truncated_to_first_number():
    node = _component("builder", "indirect_gain")
    node["functional_check"] = "bvh_node_count == 2 * N - 1"
    with pytest.raises(ComponentPlanError, match="arithmetic expression"):
        build_component_plan({"components": [node]})


@pytest.mark.parametrize("formula", [
    "sum(v.size for v in model.values()) == 36 * model['W1'].shape[1] "
    "+ model['W1'].shape[1] + model['W1'].shape[1] + 1",
    "input_count + output_count == 36",
])
def test_indirect_formula_lhs_is_not_silently_reduced_to_numeric_literal(formula):
    node = _component("trainer", "indirect_gain")
    node["functional_check"] = formula
    with pytest.raises(ComponentPlanError, match="named measured scalar"):
        build_component_plan({"components": [node]})


@pytest.mark.parametrize("operator,actual,passed", [
    ("<", -1, True), ("<", 0, False), ("<=", 0, True), ("<=", 1, False),
])
def test_indirect_materialized_difference_preserves_threshold_boundary(operator, actual, passed):
    component = _component("check", "indirect_gain")
    component["functional_check"] = f"difference {operator} 0"
    node = build_component_plan({"components": [component]}).node("check")
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=None, lower_is_better=True,
        ablation_report=[{"_host_bound": True, "component": "check", "actual": actual,
                          "evidence": {"source": "runtime_attempt", "ref": "attempt-check", "json_pointer": "/difference"}}])
    assert result["passed"] is passed


@pytest.mark.parametrize("actual,passed", [(0, True), (1, False)])
@pytest.mark.parametrize("definition", [
    "failure_count == 0 where failure_count = "
    "(error_left >= 0.001 ? 1 : 0) + (error_right >= 0.001 ? 1 : 0). "
    "Each output must have error < 0.001.",
    "failure_count == 0 where: available = 1 if samples > 0, else 0. "
    "When available == 0: failure_count = (fallback_valid == 0 ? 1 : 0). "
    "When available == 1: failure_count = (error_left >= 0.001 ? 1 : 0) "
    "+ (error_right >= 0.001 ? 1 : 0).",
    "After each case, verify flag==0 and all auxiliary==0. "
    "Define failure_count = number of cases violating those assertions. "
    "Require failure_count == 0",
    "Verify error_left < 0.001 and error_right < 0.001. "
    "Define failure_count = number of failed assertions. 要求 failure_count == 0。",
    "Acceptance contract: violation_count == 0. Computation recipe: "
    "alignment_count counts addresses where address % 64 == 0; "
    "size_ok = (sizeof(length) == 1) ? 1 : 0. "
    "Return violation_count = (100 - alignment_count) + (1 - size_ok). "
    "The host gate consumes this scalar and asserts violation_count == 0.",
])
def test_explicit_aggregate_definition_does_not_add_independent_gates(actual, passed, definition):
    component = _component("check", "indirect_gain")
    component["functional_check"] = definition
    node = build_component_plan({"components": [component]}).node("check")
    assert node.threshold_operator == "=="
    assert node.threshold_value == 0
    assert node.acceptance_check == component["functional_check"]
    result = evaluate_component_contract(
        node, parent_metric=None, candidate_metric=None, lower_is_better=True,
        ablation_report=[{"_host_bound": True,
            "component": "check", "actual": actual,
            "evidence": {"source": "runtime_attempt", "ref": "attempt-check", "json_pointer": ("/violation_count" if definition.startswith("Acceptance contract:") else "/failure_count")},
        }],
    )
    assert result["passed"] is passed


@pytest.mark.parametrize("check", [
    "failures == 0 AND error < 0.001",
    "failures == 0 where other_count = (error >= 0.001 ? 1 : 0)",
    "failures == 0 where failures == 1 AND error < 0.001",
    "failures == 0 where: other_count = (error >= 0.001 ? 1 : 0)",
    "failures == 0 where: failures == 1 AND error < 0.001",
    "Verify error < 0.001. Define other_count = number of errors. Require failures == 0",
    "Define failures = number of failed cases. Require failures == 0. Require error < 0.001",
    "Define failures = number of failed cases. Require failures == 0 AND error < 0.001",
    "Define failures = average failed cases. Require failures < 0.5 AND error < 0.001",
    "Acceptance contract: failures == 0 AND error < 0.001. "
    "Computation recipe: failures = count of failed cases.",
    "Acceptance contract: failures == 0. Computation recipe: other_count = (error > 1).",
    "Acceptance contract: failures == 0. Computation recipe: failures = count of errors. "
    "Acceptance contract: other_count == 0.",
])
def test_ambiguous_checks_are_not_treated_as_aggregate_definitions(check):
    component = _component("check", "indirect_gain")
    component["functional_check"] = check
    with pytest.raises(ComponentPlanError, match="one aggregate scalar threshold"):
        build_component_plan({"components": [component]})


def test_component_memory_does_not_replay_stale_whole_method_remediation():
    packet = component_memory_packet([{
        "component_id": "beta",
        "metric": 0.9,
        "contract_passed": True,
        "mir_verdict": "partial",
        "failure_reason": (
            "method implementation review=partial. Required correction: implement sibling x"),
        "run_ref": "run:1",
        "diff_ref": "diff:1",
    }], component_id="beta", dependency_closure=[])

    item = packet["current_and_ancestor_rounds"][0]
    assert item["metric"] == 0.9
    assert item["run_ref"] == "run:1"
    assert item["diff_ref"] == "diff:1"
    assert "implement sibling" not in item["failure_reason"]
    assert "currently scheduled component contract" in item["failure_reason"]
@pytest.mark.parametrize("check", [
    "signed_metric_improvement < 0", "signed_metric_improvement <= 0",
    "signed_metric_improvement == 0", "metric delta < -0.1",
])
def test_direct_gain_rejects_predicates_incompatible_with_positive_gain(check):
    from autoresearch.experiments.repo_improve.component_plan import (
        _direct_gain_predicate, ComponentPlanFormatError,
    )
    with pytest.raises(ComponentPlanFormatError, match="strict positive improvement"):
        _direct_gain_predicate(check)


@pytest.mark.parametrize("check", [
    "signed_metric_improvement > 0", "signed_metric_improvement < 0.2",
    "signed_metric_improvement == 0.2", "signed_metric_improvement >= -1",
    "candidate_metric < 0", "candidate_metric == -2",
])
def test_direct_gain_preserves_satisfiable_bounds_and_raw_metric_direction(check):
    from autoresearch.experiments.repo_improve.component_plan import _direct_gain_predicate
    assert _direct_gain_predicate(check)


@pytest.mark.parametrize('integration', [False, True])
def test_result_accounting_recovers_after_committed_save_without_spending_twice(tmp_path, monkeypatch, integration):
    plan = build_component_plan({'components': [_component('a')]})
    ledger = ComponentLedger(str(tmp_path / 'ledger.json'), plan)
    ledger.begin('a')
    real_save = ledger._save
    fault = []
    def after_commit(data):
        real_save(data)
        if not fault:
            fault.append(1)
            raise OSError('interrupted after atomic write')
    monkeypatch.setattr(ledger, '_save', after_commit)
    kwargs = dict(execution_id='original')
    if integration:
        kwargs['best_result'] = {'checkpoint':'saved', 'method_metric':8}
        record = lambda: ledger.record_integration_attempt(**kwargs)
    else:
        kwargs.update(execution_succeeded=True, contract_passed=True, checkpoint='saved', metric_value=8)
        record = lambda: ledger.record_attempt('a', **kwargs)
    with pytest.raises(OSError):
        record()
    record()
    if integration:
        assert ledger.snapshot()['integration_attempts'] == 1
        assert ledger.snapshot()['integration_best']['checkpoint'] == 'saved'
    else:
        assert ledger.entry('a').attempts == 1
        assert ledger.entry('a').checkpoint == 'saved'
