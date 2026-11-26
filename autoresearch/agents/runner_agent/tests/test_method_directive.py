"""老师②核心(2026-08-07): confirmed method 必须作为"要实现的方法"主线注入沙箱任务。

之前深思确认的 method(如"用 KAN 层替换 FFN")根本没进沙箱 agent,agent 只拿原始 topic ->
脱离 method 瞎试 SwiGLU/调参。这些测试锁住修复: _method_directive 读 spec_store 的 method 抽成
强指令, _with_operator 无论 operators 开关都前置它;缺 method / 关开关 -> fail-open 空串。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler


class _Store:
    """最小 spec_store: 只实现 _method_directive 用到的 get(ref)。"""

    def __init__(self, method_obj):
        self._m = method_obj

    def get(self, ref):
        return self._m


def _handler(method_obj):
    return RunnerAgentHandler(spec_store=_Store(method_obj), backend=FakeRunnerBackend())


@pytest.mark.parametrize("reproduced", [True, False])
def test_component_root_uses_only_the_bound_reproduced_baseline(monkeypatch, tmp_path, reproduced):
    from autoresearch.spec_store import SpecStore
    from autoresearch.agents.runner_agent.core import ComponentPlanInvalidError
    store = SpecStore(tmp_path / "root-store")
    checkpoint = str(tmp_path / "checkpoint-a11111")
    store.put(kind="baseline_result", session_id="s1", key="v1",
              obj={"reproduced": reproduced, "checkpoint": checkpoint, "metric_value": 1.0},
              emitter="RunnerAgent")
    method = {"components": [_component_method()["components"][0]]}
    method["components"][0]["inputs"] = ["Frozen baseline checkpoint-a11111"]
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setattr(handler, "_latest_method", lambda _: method)
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path / "research"))
    if not reproduced:
        with pytest.raises(ComponentPlanInvalidError, match="no unique verified"):
            handler._component_directive("s1")
        return
    handler._component_directive("s1")
    assert handler._active_component_context["s1"]["ledger"].root_checkpoint == checkpoint
    assert handler._prior_worktree("s1") == checkpoint


def test_mir_protocol_failure_is_not_an_implementation_correction(monkeypatch):
    method_ref = "spec://method/s1/current"
    review = {
        "verdict": "insufficient_evidence", "method_ref": method_ref,
        "review_error": "implementation review failed: JSONDecodeError",
        "issues": ["implementation review failed: JSONDecodeError"],
        "remediation": "Retry the review on the unchanged diff and run evidence.",
    }
    handler = _handler(review)
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: method_ref)
    directive = handler._method_implementation_directive("s1")
    assert "REVIEW UNAVAILABLE" in directive
    assert "unchanged" in directive
    assert "Correct the implementation itself" not in directive


def test_mir_receives_selected_measured_source_checkpoint(monkeypatch):
    handler = _handler({"approach": "Mix the data", "components": []})
    seen = {}
    def review(*args, **kwargs):
        seen.update(kwargs)
        return {"verdict": "faithful", "component_reviews": [], "issues": []}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", review)
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 0.6}, "diff": "+MIXED = True",
        "checkpoint": "/selected/immutable-checkpoint",
        "measurement_evidence": {"attempt_id": "attempt-selected", "valid": True},
    }, "method")
    assert seen["source_checkpoint"] == "/selected/immutable-checkpoint"
    assert seen["runtime_evidence"]["measurement_evidence"]["attempt_id"] == "attempt-selected"


def test_mir_scopes_named_checkpoint_sources_to_frozen_task_documents(monkeypatch):
    handler = _handler({"approach": "Measure current solve.c", "components": []})
    seen = {}

    def review(*args, **kwargs):
        seen.update(kwargs)
        return {"verdict": "faithful", "component_reviews": [], "issues": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", review)
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 0.6}, "diff": "+probe();",
        "checkpoint": "/selected/immutable-checkpoint",
        "measurement_evidence": {"attempt_id": "attempt-selected", "valid": True},
    }, "method", repository_contract={"sources": [
        {"path": "tasks/example/instruction.md", "content": "Optimize solve.c."},
        {"path": "tasks/example/environment/Dockerfile", "content": "FROM base"},
    ]})

    assert seen["source_roots"] == ("tasks/example",)


@pytest.mark.parametrize('accepted, bound, kind', [
    (True, True, 'measurement'), (False, True, 'measurement'),
    (True, False, 'measurement'), (True, True, 'intervention')])
def test_mir_transports_frozen_cap_kind_and_executed_wrapper(
        tmp_path, monkeypatch, accepted, bound, kind):
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    method = {'approach': 'Inspect the parent.', 'components': [{
        'component_id': 'probe', 'name': 'Probe', 'type': 'indirect_gain',
        'mechanism': 'Measure the existing parent under the frozen configuration.'}]}
    store = SpecStore(tmp_path / 'store')
    method_ref = store.put(kind='method', session_id='s1', key='v1', obj=method,
                           emitter='MethodAgent')
    key = 'method-' + artifact_digest(method if bound else {}).split(':', 1)[1]
    store.put(kind='method_debate_report', session_id='s1', key=key,
        obj={'accepted': accepted, 'scope': 'component_plan', 'transcript': [{
            'component_checks': [{'component_id': 'probe', 'kind': kind}]}]},
        emitter='MethodAgent')
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setattr(handler, '_latest_method_ref', lambda _sid: method_ref)
    handler._active_component_context = {'s1': {'mode': 'component', 'node':
        SimpleNamespace(component_id='probe', role='indirect_gain', depends_on=())}}
    seen = {}

    def review(*args, **kwargs):
        seen.update(kwargs)
        return {'verdict': 'faithful', 'component_reviews': [], 'issues': []}

    monkeypatch.setattr(
        'autoresearch.agents.method_implementation_agent.core.review_implementation', review)
    command_sources = [{'path': '/work/probe.py', 'source': 'measure()\n', 'sha256': 'digest'}]
    handler._review_method_implementation('s1', {
        'candidate': {'valid': True, 'metric': 0.6}, 'diff': '',
        'measurement_evidence': {'valid': True, 'attempt_id': 'attempt-selected'},
        'attempt_review_evidence': [{'attempt_id': 'attempt-selected', 'rc': 0,
            'command_sources': command_sources, 'command_sources_stable': True}],
    }, 'method')

    assert (seen.get('component_kind') == 'measurement') is (accepted and bound and kind == 'measurement')
    assert seen['runtime_evidence']['attempts'][0]['command_sources'] == command_sources


def test_mir_binds_passed_indirect_component_contract_as_selected_attempt(monkeypatch):
    method = {"approach": "Inspect the starter.", "components": [{
        "component_id": "inspect-starter", "name": "Inspect starter",
        "type": "indirect_gain", "mechanism": "Read-only inspection.",
    }]}
    handler = _handler(method)
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(
            component_id="inspect-starter", role="indirect_gain", depends_on=()),
    }}
    seen = {}

    def review(*args, **kwargs):
        seen.update(kwargs)
        return {"verdict": "faithful", "component_reviews": [], "issues": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", review)
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": None},
        "diff": "",
        "checkpoint": "/selected/immutable-checkpoint",
        "component_attempt_id": "attempt-selected",
        "attempt_review_evidence": [{"attempt_id": "attempt-selected", "rc": 0}],
        "component_execution": {"attempt_contract_result": {
            "passed": True,
            "measurement_name": "failed_check_count",
            "actual": 0.0,
            "evidence": {
                "source": "runtime_attempt", "ref": "attempt-selected",
                "json_pointer": "/failed_check_count",
            },
        }},
    }, "method")

    assert seen["source_checkpoint"] == "/selected/immutable-checkpoint"
    assert seen["runtime_evidence"]["measurement_evidence"] == {
        "attempt_id": "attempt-selected",
        "valid": True,
        "contract_passed": True,
        "measurement_name": "failed_check_count",
        "actual": 0.0,
    }


def test_mir_protocol_failure_stays_unverified_without_accusing_candidate(monkeypatch, tmp_path):
    from autoresearch.spec_store import SpecStore
    store = SpecStore(tmp_path / "review-store")
    method_ref = store.put(kind="method", session_id="s1", key="current",
                           obj={"approach": "change an algorithm"}, emitter="MethodAgent")
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: method_ref)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {
            "verdict": "insufficient_evidence", "component_reviews": [],
            "review_error": "implementation review failed: JSONDecodeError",
            "issues": ["implementation review failed: JSONDecodeError"],
            "remediation": "Retry the review on unchanged evidence.",
        })
    out = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 0.9},
        "verdict": {"improved": True}, "diff": "+changed_algorithm()",
    }, "method")
    assert out["candidate"]["valid"] is False
    assert out["candidate"]["metric"] == 0.9
    assert out["verdict"]["improved"] is False
    assert out["diagnosis"] == "implementation_review_unavailable"
    pending = store.get("spec://pending_implementation_review/s1/v1")
    assert pending["status"] == "pending"
    assert store.get(pending["input_ref"])["result"]["candidate"]["valid"] is True
    assert "JSONDecodeError" in out["diagnosis_evidence"][0]


def _component_method():
    return {
        "approach": "add a prerequisite transform, then a gain mechanism",
        "components": [
            {
                "component_id": "split-feature",
                "name": "Split feature",
                "type": "indirect_gain",
                "depends_on": [],
                "purpose": "produce the representation required by the loss",
                "inputs": ["baseline representation"],
                "outputs": ["separated representation"],
                "acceptance_requires": [],
                "mechanism": "separate the coupled representation",
                "expected_effect": "produce two independently consumable tensors",
                "functional_check": "separation_score >= 0.9",
                "implementation_check": "diff and runtime output show separated tensors",
                "optimization_space": "improve separation while preserving the interface",
                "downstream_contract": "provide the representation to new-loss",
            },
            {
                "component_id": "new-loss",
                "name": "New loss",
                "type": "direct_gain",
                "depends_on": ["split-feature"],
                "purpose": "use the separated target to improve the official metric",
                "inputs": ["separated representation"],
                "outputs": ["trained candidate"],
                "acceptance_requires": ["split-feature"],
                "mechanism": "train on the separated target",
                "expected_effect": "improve the official metric",
                "ablation_test": "signed_metric_improvement > 0",
                "implementation_check": "diff shows the new loss consumes the separated target",
                "optimization_space": "improve the loss without changing task semantics",
                "downstream_contract": "provide the candidate to final integration",
            },
        ],
    }


def test_unsupported_acceptance_is_revised_before_creating_an_attempt(monkeypatch):
    from autoresearch.agents.runner_agent.core import ComponentPlanInvalidError
    from autoresearch.agents.deliberation_agent.core import _component_contract_problem

    method = _component_method()
    method["components"][1]["ablation_test"] = "matched ablation > 3 and cache-off > 1"
    handler = _handler(method)
    monkeypatch.setattr(handler, "_latest_method", lambda _: method)

    def unexpected_workroot(_):
        pytest.fail("unsupported acceptance must fail before creating a ledger or attempt")

    monkeypatch.setattr(
        "autoresearch.experiments.repo_improve.search_memory.session_workroot",
        unexpected_workroot)
    assert _component_contract_problem(method, require_design_fields=True) is not None
    with pytest.raises(ComponentPlanInvalidError, match="executable comparison"):
        handler._component_directive("s1")


def test_method_directive_injects_confirmed_method():
    d = _handler({"method": "Replace the transformer FFN with a single KAN layer"})._method_directive("s1")
    assert "CONFIRMED METHOD" in d
    assert "KAN layer" in d
    assert "After you establish the baseline" not in d
    assert "runner-supplied frozen baseline" in d
    assert "not an algorithm that must be retained" in d
    assert "full algorithm replacement is valid" in d
    assert d.endswith("\n\n")   # 自带分隔,可直接拼 question


def test_dependency_checkpoint_conflict_is_component_plan_feedback_only_in_dag_context():
    handler = _handler(_component_method())
    error = (
        "dependency checkpoint conflict for environment/solve.cu: "
        "repo-a, repo-b produced different states"
    )

    assert handler._is_component_checkpoint_conflict("s1", error) is False
    handler._active_component_context = {"s1": {"mode": "integration"}}
    assert handler._is_component_checkpoint_conflict("s1", error) is True
    assert handler._is_component_checkpoint_conflict(
        "s1", "no RESULT_JSON: eckpoint conflict for environment/solve.cu: "
        "repo-a, repo-b produced different states"
    ) is True
    assert handler._is_component_checkpoint_conflict("s1", "training command failed") is False


@pytest.mark.parametrize("mode", ["component", "integration"])
def test_prior_worktree_carries_shared_accepted_ancestor(tmp_path, mode):
    from autoresearch.experiments.repo_improve.component_plan import (
        ComponentLedger, build_component_plan)

    components = []
    for cid, deps, closure in [("origin", [], []), ("root", ["origin"], ["origin"]),
                              ("left", ["root"], ["origin", "root"]),
                              ("right", ["root"], ["origin", "root"])]:
        components.append({**_component_method()["components"][0],
                           "component_id": cid, "name": cid, "depends_on": deps,
                           "acceptance_requires": closure})
    if mode == "component":
        components.append({**components[0], "component_id": "join", "name": "join",
                           "depends_on": ["left", "right"],
                           "acceptance_requires": ["origin", "root", "left", "right"]})
    plan = build_component_plan({"components": components})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    for cid in ("origin", "root", "left", "right"):
        ledger.begin(cid)
        ledger.record_attempt(cid, execution_succeeded=True, contract_passed=True,
                              checkpoint=f"/work/{cid}", max_attempts=1)
    handler = _handler({"components": components})
    handler._active_component_context = {"s1": {
        "mode": mode, "plan": plan, "ledger": ledger,
        "node": plan.node("join") if mode == "component" else None,
    }}
    if mode == "component":
        ledger.begin("join")
    assert handler._prior_worktree("s1") == {
        "checkpoints": ["/work/left", "/work/right"], "merge_bases": ["/work/root"]}
    if mode == "integration":
        # Final method contrasts with the frozen baseline, not with itself or
        # one terminal branch. [] is the existing frozen-baseline marker.
        assert handler._prior_worktree("s1", dependencies_only=True) == []


@pytest.mark.parametrize("mode", ["component", "integration"])
@pytest.mark.parametrize("has_baseline", [False, True])
def test_independent_component_branches_use_frozen_baseline_merge_base(tmp_path, mode, has_baseline):
    from autoresearch.experiments.repo_improve.component_plan import (
        ComponentLedger, build_component_plan)

    components = [{**_component_method()["components"][0], "component_id": cid,
                   "name": cid, "depends_on": [], "acceptance_requires": []}
                  for cid in ("left", "right")]
    if mode == "component":
        components.append({**components[0], "component_id": "join", "name": "join",
                           "depends_on": ["left", "right"],
                           "acceptance_requires": ["left", "right"]})
    plan = build_component_plan({"components": components})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    for cid in ("left", "right"):
        ledger.begin(cid)
        ledger.record_attempt(cid, execution_succeeded=True, contract_passed=True,
                              checkpoint=f"/work/{cid}", max_attempts=1)
    handler = _handler({"components": components})
    original_get = handler.spec_store.get

    def get(ref):
        if ref == "spec://baseline_result/s1/v1":
            if has_baseline:
                return {"checkpoint": "/work/frozen-baseline"}
            raise KeyError(ref)
        return original_get(ref)

    handler.spec_store.get = get
    handler._active_component_context = {"s1": {
        "mode": mode, "plan": plan, "ledger": ledger,
        "node": plan.node("join") if mode == "component" else None,
    }}
    expected = ["/work/left", "/work/right"]
    if has_baseline:
        expected = {"checkpoints": expected, "merge_bases": ["/work/frozen-baseline"]}
    assert handler._prior_worktree("s1") == expected
    assert handler._prior_worktree("s1", dependencies_only=True) == (
        [] if mode == "integration" else expected)


def test_component_review_parent_does_not_move_to_current_component_best():
    handler = _handler(_component_method())
    entries = {
        "current": SimpleNamespace(checkpoint="/work/current-best"),
        "parent": SimpleNamespace(checkpoint="/work/accepted-parent"),
    }
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="current", depends_on=("parent",)),
        "ledger": SimpleNamespace(entry=entries.__getitem__),
    }}
    assert handler._prior_worktree("s1") == "/work/current-best"
    assert handler._prior_worktree("s1", dependencies_only=True) == ["/work/accepted-parent"]


def test_root_component_uses_verified_prior_checkpoint_as_execution_parent(tmp_path):
    from autoresearch.experiments.repo_improve.component_plan import (
        ComponentLedger, build_component_plan)

    component = dict(_component_method()["components"][0])
    component["inputs"] = ["Prior checkpoint-8af556bb implementation"]
    plan = build_component_plan({"components": [component]})
    seed = "/work/checkpoint-8af556bb1234"
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan, root_checkpoint=seed)
    ledger.begin("split-feature")
    handler = _handler({"components": [component]})
    handler._active_component_context = {"s1": {
        "mode": "component", "plan": plan, "ledger": ledger,
        "node": plan.node("split-feature"),
    }}
    assert handler._prior_worktree("s1", dependencies_only=True) == seed


def test_integration_keeps_seeded_component_output_over_older_reconciliation(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.component_plan import (
        ComponentLedger, build_component_plan)

    component = dict(_component_method()["components"][0])
    plan = build_component_plan({"components": [component]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan,
                             root_checkpoint="/work/prior-verified")
    ledger.begin("split-feature")
    ledger.record_attempt("split-feature", execution_succeeded=True, contract_passed=True,
                          checkpoint="/work/new-accepted", max_attempts=1)
    handler = _handler({"components": [component]})
    monkeypatch.setattr(handler, "_latest_method", lambda _: {
        "reconciled_result_ref": "spec://gen_experiment_result/s1/older"})
    handler._active_component_context = {"s1": {
        "mode": "integration", "plan": plan, "ledger": ledger, "node": None,
    }}
    assert handler._prior_worktree("s1") == ["/work/new-accepted"]


def test_integration_uses_accepted_terminal_checkpoint_without_root_seed(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.component_plan import (
        ComponentLedger, build_component_plan)
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest

    component = dict(_component_method()["components"][0])
    plan = build_component_plan({"components": [component]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("split-feature")
    ledger.record_attempt("split-feature", execution_succeeded=True, contract_passed=True,
                          checkpoint="/work/current-accepted", max_attempts=1)
    store = SpecStore(tmp_path / "store")
    older = {"checkpoint": "/work/older-method",
             "off_method_observation": {"valid": True}}
    older_ref = store.put(
        kind="gen_experiment_result", session_id="s1",
        key="snapshot-" + artifact_digest(older).split(":", 1)[1],
        obj=older, emitter="RunnerAgent")
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setattr(handler, "_latest_method", lambda _: {
        "reconciled_result_ref": older_ref})
    handler._active_component_context = {"s1": {
        "mode": "integration", "plan": plan, "ledger": ledger, "node": None,
    }}

    assert handler._prior_worktree("s1") == ["/work/current-accepted"]


def test_integration_keeps_measured_gain_but_rejects_wrong_parent_control(tmp_path):
    from autoresearch.experiments.repo_improve.component_plan import (
        ComponentLedger, build_component_plan)

    component = dict(_component_method()["components"][0])
    plan = build_component_plan({"components": [component]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("split-feature")
    ledger.record_attempt("split-feature", execution_succeeded=True, contract_passed=True,
                          checkpoint="/work/parent", max_attempts=1)
    handler = _handler({"components": [component]})
    handler._active_component_context = {"s1": {
        "mode": "integration", "plan": plan, "ledger": ledger, "node": None,
    }}

    result = handler._finalize_component_attempt("s1", {
        "verdict": "new_best", "accepted": True,
        "baseline_metric": 1.0, "method_metric": 0.8,
        "lower_is_better": True, "checkpoint": "/work/candidate",
        "component_evidence_error": "control attempt is not the exact dependency parent",
    })

    assert result["method_metric"] == 0.8
    assert result["component_evidence_error"]
    assert result["component_execution"]["status"] == "execution_failed"
    assert result["component_execution"]["attempt_contract_result"]["passed"] is False
    assert result["component_execution"]["became_best"] is False
    assert ledger.snapshot().get("integration_best") is None


def test_method_directive_uses_latest_content_addressed_method():
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            assert session_id == "s1" and kind == "method" and limit == 1
            return ["spec://method/s1/method-new"]

        def get(self, ref):
            if ref == "spec://method/s1/method-new":
                return {"method": "current sign update with error feedback"}
            if ref.endswith("/v1"):
                return {"method": "obsolete Newton draft"}
            raise KeyError(ref)

    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    directive = handler._method_directive("s1")

    assert "current sign update" in directive
    assert "obsolete Newton" not in directive


def test_method_directive_empty_when_no_method():
    assert _handler(None)._method_directive("s1") == ""


def test_method_directive_empty_when_text_unavailable():
    # dict 里没有任何可抽字段 -> _method_text 兜底 json.dumps 仍是文本;但空 dict -> "{}" 非空,
    # 真正的"不可用"是 method_obj falsy。这里断言 falsy 分支已由上一个用例覆盖;此用例锁 env 开关。
    import os
    os.environ["AUTORESEARCH_METHOD_DIRECTIVE"] = "0"
    try:
        assert _handler({"method": "x"})._method_directive("s1") == ""
    finally:
        os.environ.pop("AUTORESEARCH_METHOD_DIRECTIVE", None)


def test_with_operator_injects_method_even_when_operators_off(monkeypatch):
    # operators 关闭也必须注入 method(注入独立于算子开关),且原 question 保留。
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    q = _handler({"method": "Replace FFN with a KAN layer"})._with_operator("s1", "improve nanochat lower BPB")
    assert "CONFIRMED METHOD" in q
    assert "KAN layer" in q
    assert "improve nanochat lower BPB" in q


def test_with_operator_no_method_returns_plain_question(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    q = _handler(None)._with_operator("s1", "improve X")
    assert q == "improve X"
    assert "CONFIRMED METHOD" not in q


def test_with_operator_carries_frozen_research_constraints(monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if ref == "spec://method/s1/current":
                return {"approach": "optimize the starter"}
            if ref == "spec://research_spec/s1/reinterpreted":
                return {"forbidden_items": [
                    "do not read or use repository-provided solution artifacts",
                    "do not modify the official evaluator",
                ]}
            if ref == "spec://benchmark_spec/s1/v1":
                return {"forbidden_files": ["private/reference/*"]}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    question = handler._with_operator("s1", "run method")

    assert "FROZEN USER/TASK CONSTRAINTS" in question
    assert "Every item below is a prohibited action" in question
    assert "do not read or use repository-provided solution artifacts" in question
    assert "do not modify the official evaluator" in question
    assert "private/reference/*" not in question
    assert "a ban on modifying a file does not prohibit reading or executing it" in question


def test_forced_search_does_not_report_unchanged_checkpoint_as_improvement(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "1")
    handler = _handler(_component_method())
    for index in (1, 2):
        handler._with_operator("s1", "run")
        result = handler._finalize_component_attempt("s1", {
            "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 1.1,
            "lower_is_better": True, "repo_path": f"/checkpoint/split-{index}",
            "diff": "split diff", "method_implementation_review": {"verdict": "faithful"},
            "ablation_report": [{"_host_bound": True, "component": "split-feature",
                "actual": 0.95, "evidence": {"source": "runtime", "ref": f"attempt:{index}", "json_pointer": "/separation_score"}}],
        })
    execution = result["component_execution"]
    assert execution["status"] == "improving"
    assert execution["continue_improve"] is True
    assert execution["became_best"] is False
    assert execution["checkpoint"] == "/checkpoint/split-1"
    assert execution["continue_reason"] == (
        "configured search continues; best checkpoint retained without improvement")


def test_component_controller_improves_indirect_gain_before_accepting_and_advancing(
        tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())

    first = handler._with_operator("s1", "run")
    assert "CURRENT COMPONENT [split-feature]" in first
    assert "copied exactly from CURRENT COMPONENT" in first
    assert "do not rename or normalize it" in first
    assert "left-hand aggregate scalar" in first
    assert "not an input/output dimension" in first
    assert "host applies the frozen" in first
    result = handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best",
        "baseline_metric": 1.0,
        "method_metric": 1.1,
        "lower_is_better": True,
        "repo_path": "/checkpoint/split",
        "diff": "split diff",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True,
            "component": "split-feature", "actual": 0.95,
            "evidence": {"source": "runtime", "ref": "attempt:1", "json_pointer": "/separation_score"},
        }],
    })

    assert result["component_execution"]["status"] == "qualified"
    assert result["component_execution"]["contract_result"]["passed"] is True
    assert result["component_execution"]["contract_result"]["actual"] == 0.95
    assert result["method_metric"] == 1.1
    assert result["verdict"] == "valid_not_best"
    improve = handler._with_operator("s1", "improve current")
    assert "CURRENT COMPONENT [split-feature]" in improve
    assert "operator=IMPROVE" in improve
    assert handler._prior_worktree("s1") == "/checkpoint/split"
    accepted = handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 1.1,
        "lower_is_better": True, "repo_path": "/checkpoint/split-2", "diff": "split 2",
        "method_implementation_review": {
            "verdict": "partial", "component_reviews": [
                {"name": "Split feature", "status": "implemented"},
                {"name": "New loss", "status": "missing"}],
        },
        "ablation_report": [{"_host_bound": True,
            "component": "split-feature", "actual": 0.95,
            "evidence": {"source": "runtime", "ref": "attempt:2", "json_pointer": "/separation_score"},
        }],
    })
    assert accepted["component_execution"]["status"] == "accepted"
    assert accepted["method_metric"] == 1.1
    second = handler._with_operator("s1", "run next")
    assert "CURRENT COMPONENT [new-loss]" in second
    assert "/checkpoint/split" in second
    assert "measure its exact fully materialized DAG parent checkpoint" in second
    assert "A stored number alone is not parent evidence" in second
    assert "an already matching receipt without rerunning" in second
    assert "do not re-parent the component" in second
    assert "copy dependency code into a different lineage" in second
    assert "Reserve at least one official-evaluator ar-attempt" in second
    assert "ablation_report.actual is the raw candidate metric" in second
    assert "host computes signed_metric_improvement" in second
    assert "do not spend the entire attempt budget on parent diagnostics" in second


@pytest.mark.parametrize("role", ["direct_gain", "indirect_gain"])
@pytest.mark.parametrize("target_status", ["implemented", "partial", "missing", "contradicted"])
def test_partial_review_qualifies_only_the_implemented_target(
        tmp_path, monkeypatch, role, target_status):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    method = _component_method()
    target = method["components"][0]
    target["type"] = role
    if role == "direct_gain":
        target["ablation_test"] = "signed_metric_improvement > 0"
        target["functional_check"] = ""
    handler = _handler(method)
    handler._with_operator("s1", "run")
    base = {
        "verdict": "new_best", "baseline_metric": 1.0, "method_metric": 0.9,
        "lower_is_better": True, "checkpoint": "/checkpoint/prior",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True, "component": "split-feature", "actual": 0.95,
                             "evidence": {"source": "runtime", "ref": "attempt-prior",
                                          "json_pointer": "/separation_score"}}],
    }
    handler._finalize_component_attempt("s1", base)
    result = handler._finalize_component_attempt("s1", {
        **base, "method_metric": 0.8, "checkpoint": "/checkpoint/new",
        "method_implementation_review": {
            "verdict": "partial", "target_component_id": "split-feature",
            "component_reviews": [{"name": "Split feature", "status": target_status}],
            "issues": ["Incomplete frozen treatment"],
        },
    })
    execution = result["component_execution"]
    assert result["method_metric"] == 0.8
    assert execution["execution_succeeded"] is True
    assert execution["qualified_attempts"] == (2 if target_status == "implemented" else 1)
    if target_status != "implemented":
        from autoresearch.agents.runner_agent.handler import _gen_result_quality
        assert result["accepted"] is False
        assert result["verdict"] == "invalid"
        assert _gen_result_quality(result) == 2
        assert execution["status"] == "implementation_failed"
        assert execution["became_best"] is False
        assert execution["checkpoint"] == "/checkpoint/prior"


@pytest.mark.parametrize("probe_change,reason", [
    ({"expected": {"operator": "==", "value": 0.9}}, "threshold does not match"),
    ({"evidence": {}}, "lacks grounded evidence"),
])
@pytest.mark.parametrize("execution_error", [None, "runtime attempt commit mismatch"])
def test_indirect_invalid_probe_reports_contract_cause_not_missing_primary_metric(
        tmp_path, monkeypatch, probe_change, reason, execution_error):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "check component")
    probe = {"_host_bound": True, "component": "split-feature", "actual": 0.95,
             "evidence": {"source": "runtime_attempt", "ref": "attempt-check", "json_pointer": "/separation_score"}}
    probe.update(probe_change)
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "accepted": False, "method_metric": None,
        "baseline_metric": 1.0, "repo_path": "/checkpoint/check",
        "component_attempt_id": "attempt-check", "error": execution_error,
        "reasons": ["method phase produced no finite method value"],
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [probe],
    })
    entry = handler._active_component_context["s1"]["ledger"].entry("split-feature")
    expected = execution_error or reason
    assert expected in result["error"]
    assert expected in entry.last_error
    assert "no finite method value" not in entry.last_error
    assert result["verdict"] == "invalid" and result["accepted"] is False
    assert entry.attempts == 1 and entry.qualified_attempts == 0
    assert not entry.checkpoint


def test_indirect_component_can_qualify_without_final_metric(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())

    directive = handler._with_operator("s1", "run component check")
    assert "Run that check through ar-attempt" in directive
    assert "method=null when no primary metric was measured" in directive
    assert "Do not launch full parent/candidate training solely to fill metric fields" in directive
    assert "unless the frozen acceptance check itself requires it" in directive
    assert "Reserve at least one official-evaluator ar-attempt" not in directive
    assert "include parent_metric" not in directive
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "baseline_metric": 1.0, "method_metric": None,
        "lower_is_better": True, "repo_path": "/checkpoint/split",
        "component_attempt_id": "attempt-123",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True,
            "component": "split-feature", "actual": 0.95,
            "evidence": {"source": "runtime_attempt", "ref": "attempt-123", "json_pointer": "/separation_score"},
        }],
    })

    execution = result["component_execution"]
    assert execution["execution_succeeded"] is True
    assert execution["status"] == "qualified"
    assert execution["contract_result"]["passed"] is True
    assert result["verdict"] == "valid_not_best"
    assert result["method_metric"] is None


@pytest.mark.parametrize("review,actual", [
    ("drifted", 0.95),
    ("insufficient_evidence", 0.95),
    ("faithful", 0.1),
])
def test_unqualified_indirect_attempt_is_not_a_valid_metricless_result(
        tmp_path, monkeypatch, review, actual):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "run component check")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "baseline_metric": 1.0, "method_metric": None,
        "lower_is_better": True, "repo_path": "/checkpoint/split",
        "component_attempt_id": "attempt-123",
        "method_implementation_review": {
            "verdict": review,
            "issues": ["component design contradicts implementation"] if review != "faithful" else [],
        },
        "ablation_report": [{"_host_bound": True,
            "component": "split-feature", "actual": actual,
            "evidence": {"source": "runtime_attempt", "ref": "attempt-123", "json_pointer": "/separation_score"},
        }],
    })

    assert result["verdict"] == "invalid"
    assert result["method_metric"] is None
    assert result["accepted"] is False
    assert result["error"]
    assert not any("contract passed" in r for r in result["reasons"])
    assert handler._active_component_context["s1"]["ledger"].entry("split-feature").attempts == 1


@pytest.mark.parametrize("receipt,actual,status", [
    ("attempt-failed", 0.1, "contract_failed"),
    (None, 0.1, "execution_failed"),
    ("attempt-failed", 0.95, "execution_failed"),
])
def test_failed_indirect_check_is_counted_without_accepting_a_checkpoint(
        tmp_path, monkeypatch, receipt, actual, status):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "run component check")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "baseline_metric": 1.0, "method_metric": None,
        "repo_path": "/checkpoint/check", "component_failed_attempt_id": receipt,
        "error": "component acceptance check reported unmet; exit 1",
        "ablation_report": [{"_host_bound": True, "component": "split-feature", "actual": actual,
                             "evidence": {"source": "runtime_attempt", "ref": "attempt-failed", "json_pointer": "/separation_score"}}],
    })
    entry = handler._active_component_context["s1"]["ledger"].entry("split-feature")
    assert result["component_execution"]["status"] == status
    assert result["component_execution"]["execution_succeeded"] is False
    assert result["verdict"] == "invalid"
    assert entry.attempts == int(receipt is not None) and entry.qualified_attempts == 0
    assert entry.infrastructure_attempts == int(receipt is None)
    assert not entry.checkpoint


@pytest.mark.parametrize("started", [True, False, None, "true"])
def test_started_command_with_invalid_evidence_counts_as_failed_component_attempt(tmp_path, monkeypatch, started):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "run component check")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "method_metric": None,
        "component_execution_started": started, "error": "measurement must be a finite number"})
    entry = handler._active_component_context["s1"]["ledger"].entry("split-feature")
    assert entry.attempts == int(started is True)
    assert entry.infrastructure_attempts == int(started is not True)
    assert entry.qualified_attempts == 0 and not entry.checkpoint
    assert not result.get("accepted")


@pytest.mark.parametrize("task_verdict", ["violation", "insufficient_evidence"])
def test_indirect_component_respects_explicit_task_contract_rejection(
        tmp_path, monkeypatch, task_verdict):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "run component check")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "baseline_metric": 1.0, "method_metric": None,
        "lower_is_better": True, "repo_path": "/checkpoint/split",
        "component_attempt_id": "attempt-123",
        "method_implementation_review": {
            "verdict": "faithful", "task_contract_review": {
                "verdict": task_verdict, "reasoning": "Task rule has not been satisfied",
            },
        },
        "ablation_report": [{"_host_bound": True,
            "component": "split-feature", "actual": 0.95,
            "evidence": {"source": "runtime_attempt", "ref": "attempt-123", "json_pointer": "/separation_score"},
        }],
    })

    assert result["verdict"] == "invalid"
    assert result["component_execution"]["status"] == "contract_failed"
    assert result["component_execution"]["contract_result"]["passed"] is True
    assert "Task rule has not been satisfied" in result["error"]
    entry = handler._active_component_context["s1"]["ledger"].entry("split-feature")
    assert entry.attempts == 1
    assert entry.qualified_attempts == 0
    assert not entry.checkpoint


def test_component_controller_suppresses_whole_method_reflection(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "1")
    handler = _handler(_component_method())
    monkeypatch.setattr(
        handler, "_reflect_preamble",
        lambda session_id, board: "IMPLEMENT THE LATER new-loss COMPONENT\n")

    question = handler._with_operator("s1", "continue")

    assert "CURRENT COMPONENT [split-feature]" in question
    assert "IMPLEMENT THE LATER" not in question


def test_component_failure_stays_on_same_node_and_switches_to_debug(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())

    handler._with_operator("s1", "run")
    failed = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "baseline_metric": 1.0, "method_metric": None,
        "lower_is_better": True, "repo_path": "",
        "method_implementation_review": {"verdict": "insufficient_evidence"},
    })

    assert failed["component_execution"]["status"] == "execution_failed"
    retry = handler._with_operator("s1", "retry")
    assert "CURRENT COMPONENT [split-feature]" in retry
    assert "operator=DEBUG" in retry
    assert "CURRENT COMPONENT [new-loss]" not in retry


def test_component_missing_measurement_does_not_hide_execution_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "run")
    error = "model service returned HTTP 403 before writing the result"

    failed = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "method_metric": None,
        "error": error,
        "method_implementation_review": {"verdict": "insufficient_evidence"},
    })

    assert failed["component_execution"]["status"] == "execution_failed"
    ledger = handler._active_component_context["s1"]["ledger"]
    entry = ledger.entry("split-feature")
    assert entry.last_error == error
    assert entry.attempts == 0
    assert entry.infrastructure_attempts == 1
    retry = handler._with_operator("s1", "retry")
    assert error in retry
    assert "operator=DEBUG" in retry


def test_stale_indirect_check_retries_current_component_not_design(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "run")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid", "baseline_metric": 1.0, "method_metric": None,
        "lower_is_better": True, "repo_path": "/checkpoint/stale-check",
        "component_attempt_id": "attempt-stale",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True,
            "component": "split-feature", "actual": 0.95,
            "expected": {"operator": ">=", "value": 0.7},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-stale", "json_pointer": "/separation_score"},
        }],
    })
    assert result["component_execution"]["status"] == "execution_failed"
    entry = handler._active_component_context["s1"]["ledger"].entry("split-feature")
    assert entry.attempts == 1
    assert entry.qualified_attempts == 0
    assert not entry.checkpoint
    retry = handler._with_operator("s1", "retry")
    assert "CURRENT COMPONENT [split-feature]" in retry
    assert "operator=DEBUG" in retry
    assert "does not match the active ComponentPlan" in retry
    assert "CURRENT COMPONENT [new-loss]" not in retry


def test_runner_exception_becomes_structured_component_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "draft")

    execution = handler._record_component_failure("s1", "dependency install failed")

    assert execution is not None
    assert execution["component_id"] == "split-feature"
    assert execution["attempts"] == 0
    assert execution["infrastructure_attempts"] == 1
    assert execution["execution_succeeded"] is False
    assert execution["attempt_contract_result"]["passed"] is False
    assert "operator=DEBUG" in handler._with_operator("s1", "retry")


def test_structured_component_failure_is_not_recorded_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "draft")

    execution = handler._record_component_failure("s1", "dependency carry failed")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "invalid",
        "component_execution": execution,
    })

    assert result["component_execution"]["attempts"] == 0
    assert result["component_execution"]["infrastructure_attempts"] == 1
    assert handler._active_component_context["s1"]["ledger"].entry(
        "split-feature").attempts == 0


@pytest.mark.parametrize("failure_path", ["exception", "result"])
def test_infrastructure_failure_keeps_qualified_component_available_for_recovery(
    tmp_path, monkeypatch, failure_path,
):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    monkeypatch.setenv("AUTORESEARCH_COMPONENT_MAX_ATTEMPTS", "2")
    handler = _handler(_component_method())

    handler._with_operator("s1", "draft")
    first = handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 1.1,
        "lower_is_better": True, "repo_path": "/checkpoint/split", "diff": "split",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True, "component": "split-feature", "actual": 0.95,
                             "evidence": {"source": "runtime", "ref": "a1", "json_pointer": "/separation_score"}}],
    })
    assert first["component_execution"]["status"] == "qualified"

    handler._with_operator("s1", "improve")
    failed = handler._record_component_failure("s1", "result contract missing")

    assert failed["status"] == "execution_failed"
    entry = handler._active_component_context["s1"]["ledger"].entry("split-feature")
    assert entry.status == "active"
    assert entry.attempts == 1 and entry.infrastructure_attempts == 1
    assert entry.checkpoint == "/checkpoint/split"

    # Once optional improvement can no longer execute, keep the already passed
    # component instead of reporting the whole method blocked.
    if failure_path == "exception":
        exhausted = handler._record_component_failure("s1", "tool protocol failed")
    else:
        exhausted = handler._finalize_component_attempt("s1", {
            "verdict": "invalid", "error": "tool protocol failed",
        })["component_execution"]
    assert exhausted["status"] == "accepted"
    assert exhausted["execution_succeeded"] is False
    assert exhausted["attempt_contract_result"]["passed"] is False
    assert exhausted["contract_result"]["passed"] is True
    assert exhausted["checkpoint"] == "/checkpoint/split"
    assert exhausted["diff_ref"] == "/checkpoint/split#git-diff"
    assert exhausted["attempt_diff_ref"] == ""
    assert exhausted["attempts"] == 1 and exhausted["infrastructure_attempts"] == 2
    from autoresearch.agents.memory_agent.handler import _append_component_history
    history = _append_component_history(
        {}, {"component_execution": exhausted, "verdict": "invalid"}, "failed-run", None)
    assert history[0]["diff_ref"] == ""
    assert history[0]["selected_diff_ref"] == "/checkpoint/split#git-diff"
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence
    diagnosis, _ = _classify_component_evidence({"component_execution": exhausted})
    assert diagnosis == "component_accepted"
    assert "CURRENT COMPONENT [new-loss]" in handler._with_operator("s1", "continue")


def test_drifted_component_returns_to_debug_until_formal_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())

    handler._with_operator("s1", "run")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 0.9,
        "lower_is_better": True, "repo_path": "/checkpoint/off-method",
        "method_implementation_review": {"verdict": "drifted"},
        "off_method_observation": {"valid": True, "improved": True, "metric": 0.9},
    })

    assert result["component_execution"]["status"] == "drifted"
    assert "operator=DEBUG" in handler._with_operator("s1", "realign")


def test_root_component_improve_keeps_baseline_comparison_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    node = {**_component_method()["components"][1],
            "depends_on": [], "acceptance_requires": []}
    handler = _handler({"components": [node]})
    for label, metric in [("draft", 0.8), ("improve", 0.85)]:
        handler._with_operator("s1", label)
        result = handler._finalize_component_attempt("s1", {
            "verdict": "new_best", "baseline_metric": 1.0, "method_metric": metric,
            "lower_is_better": True, "repo_path": f"/checkpoint/{label}",
            "method_implementation_review": {"verdict": "faithful"},
        })
        execution = result["component_execution"]
        assert execution["parent_checkpoint"] == "baseline"
        assert execution["attempt_contract_result"]["parent_metric"] == 1.0
    assert execution["checkpoint"] == "/checkpoint/draft"
    assert execution["became_best"] is False
    assert execution["status"] == "accepted"


def test_six_attempts_carry_best_checkpoint_and_matching_diff_to_child(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    monkeypatch.setenv("AUTORESEARCH_EXECUTION_STRATEGY", "efficient")
    monkeypatch.setenv("AUTORESEARCH_COMPONENT_MAX_ATTEMPTS", "6")
    root = {**_component_method()["components"][1],
            "depends_on": [], "acceptance_requires": []}
    child = {**root, "component_id": "later", "name": "Later",
             "depends_on": ["new-loss"], "acceptance_requires": ["new-loss"]}
    handler = _handler({"components": [root, child]})
    last = None
    for index, metric in enumerate((3.0, 4.0, 2.0, 1.0, 3.0, 3.5), 1):
        handler._with_operator("s1", f"attempt {index}")
        last = handler._finalize_component_attempt("s1", {
            "verdict": "new_best", "baseline_metric": 5.0,
            "method_metric": metric, "lower_is_better": True,
            "repo_path": f"/checkpoint/{index}", "diff": f"diff-{index}",
            "method_implementation_review": {"verdict": "faithful"},
        })
    ledger = handler._active_component_context["s1"]["ledger"]
    best = ledger.entry("new-loss")
    assert best.status == "accepted" and best.attempts == 6
    assert (best.checkpoint, best.metric_value, best.diff) == (
        "/checkpoint/4", 1.0, "diff-4")
    execution = last["component_execution"]
    assert (execution["checkpoint"], execution["diff_ref"],
            execution["metric_value"]) == (
                "/checkpoint/4", "/checkpoint/4#git-diff", 1.0)
    assert (execution["attempt_checkpoint"], execution["attempt_diff_ref"],
            execution["attempt_metric_value"]) == (
                "/checkpoint/6", "/checkpoint/6#git-diff", 3.5)
    from autoresearch.agents.memory_agent.handler import _append_component_history
    history = _append_component_history({}, last, "run-6", {"verdict": "verified"})
    assert (history[0]["checkpoint"], history[0]["diff_ref"],
            history[0]["metric"]) == (
                "/checkpoint/6", "/checkpoint/6#git-diff", 3.5)
    assert (history[0]["selected_checkpoint"], history[0]["selected_diff_ref"]) == (
        "/checkpoint/4", "/checkpoint/4#git-diff")
    directive = handler._with_operator("s1", "next component")
    assert "CURRENT COMPONENT [later]" in directive
    assert "Parent checkpoint: /checkpoint/4" in directive
    assert '"diff_ref": "/checkpoint/4#git-diff"' in directive
    assert handler._prior_worktree("s1", dependencies_only=True) == ["/checkpoint/4"]


def test_component_parent_lookup_failure_does_not_use_search_memory(tmp_path, monkeypatch):
    from autoresearch.agents.runner_agent.core import ComponentPlanInvalidError
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    handler = _handler(_component_method())
    handler._with_operator("s1", "run")
    ledger = handler._active_component_context["s1"]["ledger"]
    monkeypatch.setattr(ledger, "entry", lambda *_: (_ for _ in ()).throw(KeyError("lost")))
    with pytest.raises(ComponentPlanInvalidError, match="component checkpoint lookup failed"):
        handler._prior_worktree("s1")


def test_all_components_must_pass_before_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())
    handler._with_operator("s1", "first")
    handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 1.1,
        "lower_is_better": True, "repo_path": "/checkpoint/split", "diff": "split",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True, "component": "split-feature", "actual": 0.95,
                             "evidence": {"source": "runtime", "ref": "a1", "json_pointer": "/separation_score"}}],
    })
    handler._with_operator("s1", "improve first")
    handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 1.1,
        "lower_is_better": True, "repo_path": "/checkpoint/split-2", "diff": "split-2",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{"_host_bound": True, "component": "split-feature", "actual": 0.95,
                             "evidence": {"source": "runtime", "ref": "a2", "json_pointer": "/separation_score"}}],
    })
    handler._with_operator("s1", "gain draft")
    handler._finalize_component_attempt("s1", {
        "verdict": "new_best", "baseline_metric": 1.0, "method_metric": 0.9,
        "lower_is_better": True, "repo_path": "/checkpoint/loss", "diff": "loss",
        "method_implementation_review": {"verdict": "faithful"},
        "ablation_report": [{
            "component": "new-loss", "parent_metric": 1.1,
            "evidence": {"host_bound": True, "source": "official_evaluator", "ref": "parent-attempt:1"},
        }],
    })
    handler._with_operator("s1", "gain improve")
    improved = handler._finalize_component_attempt("s1", {
        "verdict": "new_best", "baseline_metric": 1.0, "method_metric": 0.8,
        "lower_is_better": True, "repo_path": "/checkpoint/loss-2", "diff": "loss-2",
        "method_implementation_review": {"verdict": "faithful"},
    })
    assert improved["component_execution"]["contract_result"]["parent_metric"] == 1.1
    assert improved["component_execution"]["parent_checkpoint"] == "/checkpoint/split"
    handler._with_operator("s1", "gain improve again")
    not_improved = handler._finalize_component_attempt("s1", {
        "verdict": "valid_not_best", "baseline_metric": 1.0, "method_metric": 0.85,
        "lower_is_better": True, "repo_path": "/checkpoint/loss-3", "diff": "loss-3",
        "method_implementation_review": {"verdict": "faithful"},
    })
    assert not_improved["component_execution"]["contract_result"]["parent_metric"] == 1.1
    assert not_improved["component_execution"]["parent_checkpoint"] == "/checkpoint/split"

    history = [
        {"component_id": "split-feature", "round": 1, "attempt": 1,
         "failure_reason": "acceptance parser depended on the expected output"},
        {"component_id": "split-feature", "round": 2, "attempt": 2,
         "checkpoint": "/checkpoint/split", "verification": "independent observation checked"},
        {"component_id": "new-loss", "round": 1, "attempt": 1,
         "run_ref": "spec://run_record/s1/loss"},
    ]
    original_get = handler.spec_store.get
    monkeypatch.setattr(handler.spec_store, "get", lambda ref: (
        {"component_history": history} if "/research_memory/" in ref else original_get(ref)))
    integration = handler._with_operator("s1", "integrate")
    assert "COMPONENT INTEGRATION" in integration
    assert '"terminal_component": "new-loss"' in integration
    assert '"implementation_check"' in integration
    assert '"optimization_space"' in integration
    assert '"downstream_contract"' in integration
    assert '"diff_ref": "/checkpoint/split#git-diff"' in integration
    assert '"diff_ref": "/checkpoint/loss-2#git-diff"' in integration
    assert "compare the concrete data manifest" in integration
    assert "missing or different inputs" in integration
    assert "Historical component receipts are context, not this run's measurements" in integration
    assert "ablation_report=[]" in integration
    assert "Never echo old component values" in integration
    assert "Prior component memory for integration" in integration
    assert '"history_entry_count": 3' in integration
    assert "acceptance parser depended on the expected output" in integration
    assert "independent observation checked" in integration
    assert "spec://run_record/s1/loss" in integration
    assert "do not replace them with weaker checks" in integration

    failed_integration = handler._record_component_failure("s1", "link step failed")
    assert failed_integration["mode"] == "integration"
    assert failed_integration["round"] == 0
    assert failed_integration["execution_succeeded"] is False
    assert "JOINT FINE-TUNING" in integration
    assert '"contract_result"' in integration
    assert '"depends_on": ["split-feature"]' in integration


@pytest.mark.parametrize("verdict", ["fail", "insufficient_evidence"])
def test_integration_cannot_accept_failed_candidate_admissibility(verdict):
    handler = _handler(_component_method())
    recorded = []
    ledger = SimpleNamespace(
        snapshot=lambda: {}, terminal_components=(),
        record_integration_attempt=lambda **kwargs: recorded.append(kwargs) or 1)
    plan = SimpleNamespace(version=1, method_fingerprint="method", topological_order=(),
                           terminal_components=())
    handler._active_component_context = {"s1": {
        "mode": "integration", "ledger": ledger, "plan": plan}}
    result = handler._finalize_component_attempt("s1", {
        "verdict": "new_best", "accepted": True, "method_metric": 0.8,
        "checkpoint": "/checkpoint/final", "lower_is_better": True,
        "candidate_admissibility": {"verdict": verdict, "reason": "gate unverified"}})
    assert result["accepted"] is False
    assert result["verdict"] == "invalid"
    assert result["component_execution"]["execution_succeeded"] is False
    assert recorded == [{"best_result": None}]


def test_direct_gain_compares_against_measured_indirect_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = _handler(_component_method())

    for suffix in ("one", "two"):
        handler._with_operator("s1", f"indirect {suffix}")
        handler._finalize_component_attempt("s1", {
            "verdict": "valid_not_best", "baseline_metric": 1.0,
            "method_metric": 1.1, "lower_is_better": True,
            "repo_path": f"/checkpoint/split-{suffix}",
            "method_implementation_review": {"verdict": "faithful"},
            "ablation_report": [{"_host_bound": True,
                "component": "split-feature", "actual": 0.95,
                "evidence": {"source": "runtime", "ref": f"attempt-{suffix}", "json_pointer": "/separation_score"},
            }],
        })

    handler._with_operator("s1", "direct draft")
    result = handler._finalize_component_attempt("s1", {
        "verdict": "new_best", "baseline_metric": 1.0,
        "method_metric": 0.9, "lower_is_better": True,
        "repo_path": "/checkpoint/loss",
        "method_implementation_review": {"verdict": "faithful"},
    })

    contract = result["component_execution"]["contract_result"]
    assert contract["passed"] is True
    assert contract["parent_metric"] == 1.1


@pytest.mark.parametrize("parent_layout", ["single", "chain", "join"])
@pytest.mark.parametrize("measured_parent,improved", [(0.9, True), (0.7, False)])
@pytest.mark.parametrize("review_verdict", ["faithful", "partial"])
def test_missing_parent_measurement_retries_measurement_not_component_design(
    tmp_path, monkeypatch, parent_layout, measured_parent, improved, review_verdict,
):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    method = _component_method()
    if parent_layout != "single":
        other = dict(method["components"][0], component_id="other-feature", name="Other feature")
        if parent_layout == "chain":
            other["depends_on"] = ["split-feature"]
            other["acceptance_requires"] = ["split-feature"]
        method["components"].insert(1, other)
        method["components"][-1]["depends_on"] = (
            ["other-feature"] if parent_layout == "chain"
            else ["split-feature", "other-feature"])
        method["components"][-1]["acceptance_requires"] = ["split-feature", "other-feature"]
    handler = _handler(method)
    for component in [item["component_id"] for item in method["components"][:-1]]:
        for index in (1, 2):
            handler._with_operator("s1", "prepare parent")
            handler._finalize_component_attempt("s1", {
                "verdict": "valid_not_best", "baseline_metric": 1.0,
                "method_metric": (1.0 if parent_layout == "chain"
                                  and component == "split-feature" else None),
                "lower_is_better": True,
                "repo_path": f"/checkpoint/{component}-{index}",
                "component_attempt_id": f"attempt-{component}-{index}",
                "method_implementation_review": {"verdict": "faithful"},
                "ablation_report": [{"_host_bound": True, "component": component, "actual": 0.95,
                                     "evidence": {"source": "runtime", "ref": "attempt-parent", "json_pointer": "/separation_score"}}],
            })
    handler._with_operator("s1", "run joined child")
    candidate = {
        "verdict": "new_best", "accepted": True,
        "baseline_metric": 1.0, "method_metric": 0.8,
        "lower_is_better": True, "repo_path": "/checkpoint/join",
        "method_implementation_review": {
            "verdict": review_verdict,
            "component_reviews": [{"name": "New loss", "status": "implemented"}],
            "issues": ["Optional diagnostic has not been implemented"]
            if review_verdict == "partial" else [],
        },
    }
    result = handler._finalize_component_attempt("s1", candidate)
    execution = result["component_execution"]
    assert execution["status"] == "execution_failed"
    assert execution["execution_succeeded"] is True  # Candidate really ran.
    assert execution["qualified_attempts"] == 0
    assert execution["attempts"] == 1
    assert result["method_metric"] == 0.8
    assert execution["became_best"] is False
    entry = handler._active_component_context["s1"]["ledger"].entry("new-loss")
    assert "lacks finite parent/candidate metrics" in entry.last_error
    if review_verdict == "partial":
        assert "Optional diagnostic" in entry.last_error
    directive = handler._with_operator("s1", "retry")
    assert "CURRENT COMPONENT [new-loss]" in directive
    assert "operator=DEBUG" in directive
    assert "measure its exact fully materialized DAG parent checkpoint" in directive
    candidate["ablation_report"] = [{
        "component": "new-loss", "parent_metric": measured_parent,
        "evidence": {"host_bound": True, "source": "official evaluator", "ref": "attempt-joined-parent"},
    }]
    recovered = handler._finalize_component_attempt("s1", candidate)
    execution = recovered["component_execution"]
    assert execution["contract_result"]["parent_metric"] == measured_parent
    assert execution["contract_result"]["passed"] is improved
    assert execution["became_best"] is improved
    assert recovered["method_metric"] == 0.8
    if improved:
        assert execution["status"] == "qualified"
        assert recovered["accepted"] is True
        assert recovered["verdict"] == "new_best"
    else:
        assert recovered["accepted"] is False
        assert recovered["verdict"] == "valid_not_best"


def test_runner_mir_reviews_only_the_active_component(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(kwargs)
        return {
            "verdict": "faithful", "approach_evidence": ["diff"],
            "component_reviews": [], "issues": [], "remediation": "",
        }

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        fake_review)
    handler = _handler(_component_method())
    handler._component_directive("s1")

    reviewed = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": False}, "diff": "implementation",
    }, "method")

    assert captured["target_component_id"] == "split-feature"
    assert reviewed["method_implementation_review"]["target_component_id"] == "split-feature"


@pytest.mark.parametrize("status", ["accepted", "active"])
def test_runner_mir_supplies_bounded_accepted_parent_diff(monkeypatch, status):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(kwargs)
        return {"verdict": "faithful", "issues": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        fake_review)
    handler = _handler(_component_method())
    parent = SimpleNamespace(status=status, checkpoint="/checkpoint/parent",
                             diff="+parent implementation\n" * 1000,
                             contract_result={"passed": True, "actual": 0.8,
                                              "evidence": {"ref": "attempt-parent"}})
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="new-loss", role="direct_gain",
                                depends_on=("split-feature",)),
        "ledger": SimpleNamespace(entry=lambda _: parent),
    }}
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": False}, "diff": "cumulative implementation",
    }, "method")
    if status == "accepted":
        assert captured["accepted_dependencies"] == [{
            "component_id": "split-feature", "checkpoint": "/checkpoint/parent",
            "diff": parent.diff[:16000], "truncated": True,
            "acceptance_evidence": parent.contract_result,
        }]
    else:
        assert captured["accepted_dependencies"] == []


def test_integration_review_keeps_starting_source_delta_separate_from_baseline(monkeypatch):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(diff=diff, **kwargs)
        return {"verdict": "faithful", "issues": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", fake_review)
    handler = _handler(_component_method())
    parent = SimpleNamespace(status="accepted", checkpoint="/checkpoint/qualified",
                             diff="# 新增文件 settings.py:\nWINDOW = 128\n",
                             contract_result={"passed": True, "actual": 0.8,
                                              "evidence": {"ref": "attempt-parent"}})
    handler._active_component_context = {"s1": {
        "mode": "integration", "plan": SimpleNamespace(terminal_components=("new-loss",)),
        "ledger": SimpleNamespace(entry=lambda _: parent),
    }}
    cumulative = "# 新增文件 settings.py:\nWINDOW = 64\n"
    delta = "diff --git a/settings.py b/settings.py\n@@\n-WINDOW = 128\n+WINDOW = 64\n"
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0}, "verdict": {"improved": False},
        "diff": cumulative, "parent_relative_diff": cumulative, "starting_point_diff": delta,
    }, "method")

    assert captured["diff"] == cumulative
    assert captured["starting_point_diff"] == delta
    assert captured["accepted_dependencies"][0]["checkpoint"] == parent.checkpoint
    assert captured["accepted_dependencies"][0]["acceptance_evidence"] == parent.contract_result
    assert "WINDOW = 128" in captured["accepted_dependencies"][0]["diff"]


def test_integration_source_delta_survives_both_result_adapters():
    from autoresearch.agents.runner_agent.handler import _res_to_improve_json, _map_repo_result
    delta = "diff --git a/settings.py b/settings.py\n@@\n-WINDOW = 128\n+WINDOW = 64\n"
    intermediate = _res_to_improve_json({"starting_point_diff": delta,
                                       "starting_point_diff_error": None})
    _, generated = _map_repo_result("integrate", intermediate)
    assert generated["starting_point_diff"] == delta
    assert generated["starting_point_diff_error"] is None


@pytest.mark.parametrize("status", ["accepted", "active"])
@pytest.mark.parametrize("parent_delta", [None, "diff against materialized dependencies"])
def test_runner_scopes_mir_but_preserves_full_diff_for_task_contract(monkeypatch, status, parent_delta):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured["fidelity"] = diff
        return {"verdict": "faithful", "issues": []}

    def fake_contract(contract, diff, llm):
        captured["contract"] = diff
        return {"verdict": "not_available"}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", fake_review)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract", fake_contract)
    inherited = "# 新增文件 parent.py:\n" + "parent_line = 1\n" * 2000
    target = "# 新增文件 target.py:\nloss = new_loss(encode(inputs))\n"
    handler = _handler(_component_method())
    parent = SimpleNamespace(status=status, checkpoint="/checkpoint/parent", diff=inherited)
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="new-loss", role="direct_gain",
                                depends_on=("split-feature",)),
        "ledger": SimpleNamespace(entry=lambda _: parent),
    }}
    full_diff = inherited + target
    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": False}, "diff": full_diff,
        "parent_relative_diff": parent_delta,
    }, "method")
    assert captured["fidelity"] == (
        parent_delta if parent_delta is not None else target if status == "accepted" else full_diff)
    assert captured["contract"] == full_diff
    assert result["diff"] == full_diff


@pytest.mark.parametrize("baseline_error", [None, "baseline checkpoint unavailable"])
def test_reviews_exclude_files_inherited_from_frozen_baseline(monkeypatch, baseline_error):
    captured = {}
    def review_implementation(method, diff, llm, **kwargs):
        captured["fidelity"] = diff
        return {"verdict": "faithful", "issues": []}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        review_implementation)
    def review_contract(contract, diff, llm):
        captured["diff"] = diff
        return {"verdict": "not_available"}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        review_contract)
    handler = _handler(_component_method())
    candidate_delta = "diff --git a/train.py b/train.py\n+use_all_sources = True\n"
    full_diff = candidate_delta + "# 新增文件 generated_cache/Compiled.py:\n# generated\n"
    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": True}, "diff": full_diff,
        "baseline_relative_diff": candidate_delta,
        "baseline_diff_error": baseline_error,
    }, "method")
    expected = full_diff if baseline_error else candidate_delta
    assert captured["fidelity"] == expected
    assert captured["diff"] == expected
    assert result["diff"] == full_diff


def test_task_contract_review_receives_immutable_source_when_baseline_delta_is_empty(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})

    def review_contract(contract, diff, llm, source_context=""):
        captured.update(diff=diff, source_context=source_context)
        return {"verdict": "not_available"}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        review_contract)
    handler = _handler(_component_method())
    immutable_source = (
        "--- a/external:/checkpoint/tasks/example/solve.c\n"
        "+++ b/external:/checkpoint/tasks/example/solve.c\n"
        "@@ -0,0 +1 @@\n+int solve(void) { return 0; }\n"
    )

    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": False}, "diff": immutable_source,
        "baseline_relative_diff": "",
    }, "method")

    assert captured["diff"] == ""
    assert captured["source_context"] == immutable_source
    assert result["diff"] == immutable_source


def test_empty_parent_delta_uses_cumulative_diff_only_as_reconciliation_context(monkeypatch):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(diff=diff, **kwargs)
        return {"verdict": "insufficient_evidence", "issues": ["no target-local change"]}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", fake_review)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        lambda *args: {"verdict": "not_available"})
    handler = _handler(_component_method())
    parent = SimpleNamespace(
        status="accepted", checkpoint="/checkpoint/parent",
        diff="# 新增文件 parent.py:\nparent_implementation = True\n")
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="new-loss", role="direct_gain",
                                depends_on=("split-feature",)),
        "ledger": SimpleNamespace(entry=lambda _: parent),
    }}
    full_diff = parent.diff + "# 新增文件 assembled.py:\nmethod_is_assembled = True\n"

    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": True}, "diff": full_diff,
        "parent_relative_diff": "",
    }, "method")

    assert captured["diff"] == full_diff
    assert captured["target_delta_empty"] is True
    assert "exact target-component delta is empty" in captured["supporting_summary"]
    assert result["method_implementation_review"]["diff_basis"] == \
        "empty_dependency_delta_with_cumulative_context"


def test_qualified_component_remeasurement_reviews_persisted_source(monkeypatch):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(diff=diff, **kwargs)
        return {"verdict": "faithful", "issues": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        fake_review)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        lambda *args: {"verdict": "not_available"})
    handler = _handler(_component_method())
    qualified = SimpleNamespace(status="active", checkpoint="/checkpoint/target",
                                qualified_results=[{"checkpoint": "/checkpoint/target"}])
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="new-loss", role="direct_gain",
                                depends_on=()),
        "ledger": SimpleNamespace(entry=lambda _: qualified),
    }}
    treatment = "diff --git a/model.py b/model.py\n+loss = new_loss(inputs)\n"
    measurement = "diff --git a/artifacts/eval.py b/artifacts/eval.py\n+measure()\n"

    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": True},
        "diff": treatment + measurement,
        "parent_relative_diff": measurement,
    }, "method")

    assert captured["diff"] == treatment + measurement
    assert "already qualified" in captured["supporting_summary"]
    assert result["method_implementation_review"]["diff_basis"] == \
        "qualified_component_cumulative"


def test_qualified_component_remeasurement_bounds_generated_cumulative_diff(monkeypatch):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(diff=diff, **kwargs)
        return {"verdict": "faithful", "issues": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        fake_review)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        lambda *args: {"verdict": "not_available"})
    handler = _handler(_component_method())
    qualified = SimpleNamespace(status="active", checkpoint="/checkpoint/target",
                                qualified_results=[{"checkpoint": "/checkpoint/target"}])
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="new-loss", role="direct_gain",
                                depends_on=()),
        "ledger": SimpleNamespace(entry=lambda _: qualified),
    }}
    treatment = "diff --git a/model.py b/model.py\n@@ -1 +1 @@\n-old()\n+new_loss(inputs)\n"
    generated = (
        "diff --git a/generated/results.txt b/generated/results.txt\n@@ -0,0 +1 @@\n"
        + "+generated-observation\n" * 20_000
    )
    full_diff = treatment + generated

    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": True},
        "diff": full_diff,
        "parent_relative_diff": generated,
    }, "method")

    assert treatment in captured["diff"]
    assert "SOURCE_CAPTURE_INCOMPLETE" in captured["diff"]
    assert len(captured["diff"]) < 120_000
    assert result["diff"] == full_diff
    assert "full cumulative diff remains" in captured["supporting_summary"]


def test_large_parent_delta_keeps_exact_parent_scope_without_cumulative_fallback(monkeypatch):
    captured = {}

    def fake_review(method, diff, llm, **kwargs):
        captured.update(diff=diff, **kwargs)
        return {"verdict": "insufficient_evidence", "issues": ["missing implementation context"]}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation", fake_review)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        lambda *args: {"verdict": "not_available"})
    handler = _handler(_component_method())
    handler._active_component_context = {"s1": {
        "mode": "component",
        "node": SimpleNamespace(component_id="new-loss", role="direct_gain", depends_on=()),
    }}
    full_diff = "# 新增文件 model.py:\nloss = new_loss(inputs)\n"
    source_delta = "diff --git a/model.py b/model.py\n+loss = new_loss(inputs)\n"
    parent_delta = "diff --git a/results.json b/results.json\n" + "+generated_observation = 0\n" * 10000 + source_delta
    result = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0},
        "verdict": {"improved": False}, "diff": full_diff,
        "parent_relative_diff": parent_delta,
    }, "method")

    assert source_delta in captured["diff"]
    assert "SOURCE_CAPTURE_INCOMPLETE" in captured["diff"]
    assert "results.json" in captured["diff"]
    assert full_diff not in captured["diff"]
    assert "exact parent comparison" in captured["supporting_summary"]
    assert "unexpanded" in captured["supporting_summary"]
    assert result["parent_relative_diff"] == parent_delta
    assert result["method_implementation_review"]["diff_basis"] == "dependency_checkpoint"
    assert result["method_implementation_review"]["verdict"] == "insufficient_evidence"


def test_fidelity_review_receives_native_measurements_and_attempt_trace(monkeypatch):
    from autoresearch.agents.runner_agent.handler import _res_to_improve_json, _map_repo_result
    captured = {}
    handler = _handler(_component_method())
    def review(method, diff, llm, **kwargs):
        captured.update(kwargs)
        return {"verdict": "faithful", "issues": []}
    monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_implementation", review)
    monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_task_contract",
                        lambda *a: {"verdict": "not_available"})
    measured = {"value": 0.98, "attempt_id": "attempt-negative", "stdout_sha256": "b" * 64}
    attempts = [{"attempt_id": "attempt-negative", "rc": 0,
                 "verified_output_fields": {"/method": 0.98, "/notes/correctness_gate": "passed"}}]
    rr = _res_to_improve_json({
        "baseline": SimpleNamespace(metric_value=1.0, valid=True, error=None),
        "candidate": SimpleNamespace(metric_value=0.98, valid=True, error=None),
        "diff": "+do_work()", "lower_is_better": False,
        "verdict": {"improved": False}, "measurement_evidence": measured,
        "attempt_review_evidence": attempts})
    reviewed = handler._review_method_implementation("s1", rr, "method")
    assert captured["runtime_evidence"] == {"measurement_evidence": measured, "attempts": attempts}
    _, saved = _map_repo_result("goal", reviewed)
    assert saved["attempt_review_evidence"] == attempts


def test_contract_review_receives_original_task_and_frozen_semantics(monkeypatch):
    captured = {}
    handler = _handler(_component_method())
    monkeypatch.setattr(handler, "_original_question", lambda sid: "No result memoization across requests.")
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})

    def review(contract, diff, llm):
        captured.update(contract)
        return {"verdict": "compliant"}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract", review)
    contract = {"sources": [{"path": "README.md", "content": "A general library."}]}
    baseline = {
        "benchmark_id": "repo-task", "reproduced": True, "source_kind": "live_repo",
        "metric_name": "score", "metric_value": 0.5, "lower_is_better": False,
        "measurement_evidence": {
            "value": 0.5, "valid": True, "attempt_id": "attempt-baseline",
            "stdout_sha256": "a" * 64, "protocol": {"kind": "official_single"},
        },
        "log_excerpt": "official starter and evaluator completed",
    }
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 1.0}, "diff": "+do_work()",
        "verdict": {"improved": True},
        "attempt_review_evidence": [{"attempt_id": "attempt-1", "argv": ["test"]}],
        "rationale": "Chose the second setting after evaluation."},
        "method", repository_contract=contract,
        evaluation_semantics={"evaluation_notes": ["Hold the query stream fixed."]},
        frozen_baseline=baseline)
    text = str(captured["sources"])
    assert "No result memoization" in text
    assert "Hold the query stream fixed" in text
    assert "A general library" in text
    assert captured["runtime_evidence"]["attempts"][0]["attempt_id"] == "attempt-1"
    assert "second setting" in captured["runtime_evidence"]["candidate_selection_report"]
    frozen = captured["runtime_evidence"]["frozen_baseline"]
    assert frozen["reproduced"] is True
    assert frozen["measurement_evidence"]["attempt_id"] == "attempt-baseline"
    assert frozen["log_excerpt"] == "official starter and evaluator completed"
    assert frozen["artifact_digest"].startswith("sha256:")
    assert len(contract["sources"]) == 1  # Do not mutate the frozen baseline artifact.


@pytest.mark.parametrize("legacy_sources", [[], [{"path": "legacy.py", "content": "pass"}]])
def test_contract_review_uses_host_recovered_sources_when_frozen_contract_was_incomplete(
        monkeypatch, legacy_sources):
    handler = _handler(_component_method())
    seen = {}
    monkeypatch.setattr(handler, "_original_question", lambda _: "Optimize the declared program.")
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})
    def review(contract, diff, llm):
        seen.update(contract)
        return {"verdict": "compliant"} if contract.get("sources") else {"verdict": "not_available"}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract", review)
    recovered = {"sources": [{"path": "instruction.md", "content": "Edit only solve.py."}],
                 "recovered_task_sources": ["instruction.md"]}
    reviewed = handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 0.5}, "diff": "+do_work()",
        "verdict": {"improved": True}, "repository_contract": recovered,
    }, "method", repository_contract={"sources": legacy_sources})
    assert seen["sources"][0] == recovered["sources"][0]
    assert reviewed["method_implementation_review"]["task_contract_review"]["verdict"] == "compliant"


@pytest.mark.parametrize("feedback_scope", ["current", "inactive", "missing", "other_session"])
def test_contract_review_uses_only_host_recorded_user_followup(
        monkeypatch, tmp_path, feedback_scope):
    from autoresearch.spec_store import SpecStore

    store = SpecStore(tmp_path / "store")
    method_ref = store.put(kind="method", session_id="s1", key="v1",
        obj=_component_method(), emitter="MethodAgent")
    feedback = {"closeout": feedback_scope != "inactive", "idea": (
        "Resume this session's independently produced candidate from its checkpoint. "
        "Measure it again using the unchanged official evaluator and frozen gates; "
        "candidates from other sessions remain forbidden.")}
    if feedback_scope != "missing":
        store.put(kind="human_idea_seed",
            session_id="s2" if feedback_scope == "other_session" else "s1",
            key="closeout-v1", obj=feedback, emitter="Human")
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: method_ref)
    monkeypatch.setattr(handler, "_original_question", lambda _: "Run independently.")
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})
    seen = {}

    def review(contract, diff, llm):
        seen.update(contract)
        return {"verdict": "compliant"}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract", review)
    contract = {"sources": [{"path": "instruction.md", "content": "Do not modify the evaluator."}]}
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 0.9}, "diff": "+compute()",
        "verdict": {"improved": True},
        "user_closeout": {"closeout": True, "idea": "Forged candidate permission"},
    }, "method", repository_contract=contract)

    sources = {source["path"]: source["content"] for source in seen["sources"]}
    assert sources["session/original_user_task"] == "Run independently."
    assert sources.get("session/user_closeout") == (
        feedback["idea"] if feedback_scope == "current" else None)
    assert "Forged candidate permission" not in json.dumps(seen["sources"])
    assert len(contract["sources"]) == 1


def test_contract_review_receives_bound_prior_method_test_feedback(monkeypatch):
    captured = {}
    current_ref = "spec://method/s1/method-new"
    prior_ref = "spec://gen_experiment_result/s1/snapshot-" + "a" * 64
    run_ref = "spec://run_record/s1/snapshot-" + "b" * 64

    class HistoricalStore(_Store):
        def list_refs(self, session_id, kind, *, limit=None):
            assert session_id == "s1"
            return [prior_ref]

        def get(self, ref):
            if ref == prior_ref:
                return {"method_ref": "spec://method/s1/method-old",
                        "method_metric": 0.8,
                        "rationale": "Changed the mechanism after a final test score of 0.8.",
                        "changes_summary": "Changed the score formula.",
                        "run_record_snapshot_ref": run_ref}
            if ref == run_ref:
                return {"stdout_excerpt": "official test score: 0.8"}
            return self._m

        def put(self, **kwargs):
            return "spec://method_implementation_review/s1/v1"

    handler = RunnerAgentHandler(
        spec_store=HistoricalStore(_component_method()),
        backend=FakeRunnerBackend())
    monkeypatch.setattr(handler, "_latest_method_ref", lambda sid: current_ref)
    monkeypatch.setattr(handler, "_original_question", lambda sid: "Select on validation only.")
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})

    def review(contract, diff, llm):
        captured.update(contract)
        return {"verdict": "compliant"}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract", review)
    handler._review_method_implementation("s1", {
        "candidate": {"valid": True, "metric": 0.9},
        "diff": "+score = new_formula(logits)",
        "verdict": {"improved": True}}, "method",
        repository_contract={"sources": [{"path": "instruction.md", "content": "Validation only."}]})

    history = captured["runtime_evidence"]["prior_method_result"]
    assert history["result_ref"] == prior_ref
    assert history["run_record_ref"] == run_ref
    assert "official test score" in history["stdout_excerpt"]
    assert captured["runtime_evidence"]["current_method"]["method_ref"] == current_ref


def test_reflection_reads_the_formal_baseline_in_method_phase(monkeypatch):
    captured = {}

    class Store:
        def get(self, ref):
            if "baseline_result" in ref:
                return {"metric_name": "val_bpb", "metric_value": 1.0,
                        "lower_is_better": True, "reproduced": True}
            if "method" in ref:
                return {"method": "new optimizer"}
            raise KeyError(ref)

    def fake_reflect(method, cards, direction, llm, *, baseline=None):
        captured["baseline"] = baseline
        from autoresearch.agents.reflection_agent.core import ReflectionResult
        return ReflectionResult()

    monkeypatch.setattr(
        "autoresearch.agents.reflection_agent.core.reflect_on_search", fake_reflect)
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend(), p1_llm=object())
    board = SimpleNamespace(cards=lambda: [SimpleNamespace(lower_is_better=True, operator="")])

    assert handler._reflect_preamble("s1", board) == ""
    assert captured["baseline"]["metric_value"] == 1.0


def test_reflection_revision_is_bound_to_next_valid_nonwin(monkeypatch):
    class Store:
        def get(self, ref):
            if "baseline_result" in ref:
                return {"metric_name": "rmse", "metric_value": 0.5,
                        "lower_is_better": True, "reproduced": True}
            return {"method": "decrease learning rate"}

    def fake_reflect(method, cards, direction, llm, *, baseline=None):
        from autoresearch.agents.reflection_agent.core import ReflectionResult
        return ReflectionResult(
            route="revise_method",
            verdict="faithful trials falsified the learning-rate mechanism",
            method_note="drop the learning-rate decrease and retain the supported component",
        )

    monkeypatch.setattr(
        "autoresearch.agents.reflection_agent.core.reflect_on_search", fake_reflect)
    handler = RunnerAgentHandler(
        spec_store=Store(), backend=FakeRunnerBackend(), p1_llm=object())
    board = SimpleNamespace(cards=lambda: [SimpleNamespace(
        lower_is_better=True, operator="", ablation_report="")])

    handler._reflect_preamble("s1", board)
    result = handler._apply_reflection_route("s1", {
        "candidate": {"valid": True, "metric": 0.51},
        "verdict": {"improved": False},
    })

    assert result["host_reflection"]["route"] == "revise_method"
    assert "drop the learning-rate decrease" in result["host_reflection"]["method_note"]


def test_reflection_revision_does_not_override_a_new_win_or_failed_execution():
    handler = _handler({"method": "x"})
    handler._pending_method_revision = {
        "win": {"route": "revise_method", "method_note": "old evidence"},
        "failed": {"route": "revise_method", "method_note": "old evidence"},
    }

    win = handler._apply_reflection_route("win", {
        "candidate": {"valid": True, "metric": 0.4},
        "verdict": {"improved": True},
    })
    failed = handler._apply_reflection_route("failed", {
        "candidate": {"valid": False, "metric": None},
        "verdict": {"improved": False},
    })

    assert "host_reflection" not in win
    assert "host_reflection" not in failed
    assert "win" not in handler._pending_method_revision
    assert "failed" in handler._pending_method_revision


def test_low_confidence_ablation_is_not_described_as_all_components_met(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    handler = _handler({"method": "new optimizer"})
    last = SimpleNamespace(ablation_report=json.dumps([{"_host_bound": True,
        "component": "scheduler", "type": "gain",
        "actual": {"val_loss": 1.2, "throughput": 100},
        "expected": {"operator": "<", "value": 1.0},
        "evidence": {"source": "run", "ref": "attempt:1", "json_pointer": "/separation_score"},
        "verdict": "component_degraded",
    }]))

    preamble = handler._check_expectation_preamble("s1", last)

    assert "方法组件都兑现了预期" not in preamble
    assert "补成单一有限数字 actual" in preamble


def test_with_operator_injects_trusted_comparison_correction_even_when_operators_off(
        monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if "evaluation_report" in ref:
                return {
                    "verdict": "suspicious", "suspicious": True,
                    "anti_cheat_flags": [
                        "comparison_semantics_changed:method changed the evaluator"],
                    "reasoning": "Required correction: restore the frozen evaluator.",
                }
            if "run_record" in ref:
                return {"experiment_id": "spec://method/s1/current"}
            if "method" in ref:
                return {"method": "new optimizer"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    question = handler._with_operator("s1", "improve the method")

    assert "COMPARISON GUARD CORRECTION" in question
    assert "restore the frozen evaluator" in question
    assert question.index("COMPARISON GUARD CORRECTION") < question.index("improve the method")


def test_with_operator_injects_verified_failure_cause_into_next_attempt(monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if "verification_report" in ref:
                return {
                    "verdict": "verified",
                    "checks": ["failure_confirmed: attempt recorded an invalid outcome and a cause"],
                    "anti_cheat_flags": [],
                    "reasoning": "Recorded failure cause: CUDA out of memory while allocating logits",
                }
            if "run_record" in ref:
                return {"experiment_id": "spec://method/s1/current"}
            if "method" in ref:
                return {"method": "new optimizer"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    question = handler._with_operator("s1", "continue the next attempt")

    assert "VERIFIED FAILURE FROM THE PREVIOUS ATTEMPT" in question
    assert "CUDA out of memory" in question
    assert question.index("CUDA out of memory") < question.index("continue the next attempt")


def test_with_operator_does_not_tell_experiment_agent_to_debug_conversation_infra(
        monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if "verification_report" in ref:
                return {
                    "verdict": "verified",
                    "checks": ["failure_confirmed: invalid outcome"],
                    "anti_cheat_flags": [],
                    "reasoning": (
                        "Recorded failure cause: Conversation run failed for id=<id>: "
                        "APIError: output text may contain sensitive information"),
                }
            if "run_record" in ref:
                return {"experiment_id": "spec://method/s1/current"}
            if "method" in ref:
                return {"method": "new optimizer"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    question = handler._with_operator("s1", "continue the next attempt")

    assert "VERIFIED FAILURE FROM THE PREVIOUS ATTEMPT" not in question
    assert "conversation run failed" not in question.lower()


def test_with_operator_does_not_leak_verified_failure_from_replaced_method(monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/new"] if kind == "method" else []

        def get(self, ref):
            if "verification_report" in ref:
                return {
                    "verdict": "verified",
                    "checks": ["failure_confirmed: old method failed"],
                    "reasoning": "Old GC implementation failed.",
                }
            if "run_record" in ref:
                return {"experiment_id": "spec://method/s1/old"}
            if ref == "spec://method/s1/new":
                return {"method": "short warmup plus clipping"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    question = RunnerAgentHandler(
        spec_store=Store(), backend=FakeRunnerBackend())._with_operator(
            "s1", "run the replacement")

    assert "CONFIRMED METHOD" in question
    assert "short warmup plus clipping" in question
    assert "VERIFIED FAILURE FROM THE PREVIOUS ATTEMPT" not in question
    assert "Old GC" not in question


def test_method_implementation_review_accepts_grounded_partial_measured_win(monkeypatch):
    stored = {}

    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if ref == "spec://method/s1/current":
                return {
                    "approach": "Replace the FFN with KAN and a residual gate.",
                    "components": [
                        {"name": "KAN projection"}, {"name": "Residual gate"}],
                }
            if "method_implementation_review" in ref:
                return stored["review"]
            raise KeyError(ref)

        def put(self, *, kind, session_id, key, obj, emitter):
            assert kind == "method_implementation_review"
            stored["review"] = obj
            return f"spec://{kind}/{session_id}/{key}"

    review = {
        "verdict": "partial",
        "approach_evidence": ["self.kan = KANProjection(config)"],
        "component_reviews": [
            {"name": "KAN projection", "status": "implemented",
             "evidence": ["self.kan = KANProjection(config)"]},
            {"name": "Residual gate", "status": "missing", "evidence": []},
        ],
        "issues": ["Residual gate is missing."],
        "remediation": "Implement the residual gate before claiming the full method.",
        "reasoning": "Only one of two frozen components is present.",
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda method, diff, llm, supporting_summary="", evaluation_semantics=None: review)
    handler = RunnerAgentHandler(
        spec_store=Store(), backend=FakeRunnerBackend(), p1_llm=object())
    rr = {
        "candidate": {"valid": True, "metric": 0.9},
        "verdict": {"improved": True, "reason": "lower loss"},
        "diff": "self.kan = KANProjection(config)",
    }

    semantics = {"comparison_mode": "quality_at_fixed_time"}
    out = handler._review_method_implementation(
        "s1", rr, "method", evaluation_semantics=semantics)

    assert out["candidate"]["valid"] is True
    assert out["verdict"]["improved"] is True
    assert "lower loss" in out["verdict"]["reason"]
    assert "only the implemented frozen components may be credited" \
        in out["verdict"]["reason"]
    assert "diagnosis" not in out
    assert "diagnosis_evidence" not in out
    expected_review = {**review, "method_ref": "spec://method/s1/current",
                       "diff_basis": "cumulative_diff"}
    assert out["method_implementation_review"] == expected_review
    assert stored["review"] == expected_review


def test_method_implementation_review_receives_authoritative_comparison_semantics(monkeypatch):
    handler = _handler({
        "approach": "Use curriculum ordering.",
        "components": [{"name": "Curriculum ordering", "type": "gain"}],
    })
    captured = {}
    review = {
        "verdict": "faithful",
        "approach_evidence": ["curriculum = True"],
        "component_reviews": [{
            "name": "Curriculum ordering", "status": "implemented",
            "evidence": ["curriculum = True"],
        }],
        "issues": [], "remediation": None,
        "reasoning": "The treatment is implemented.",
    }

    def fake_review(method, diff, llm, *, supporting_summary="",
                    evaluation_semantics=None):
        captured["semantics"] = evaluation_semantics
        captured["supporting_summary"] = supporting_summary
        return review

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        fake_review)
    semantics = {
        "primary_metric": {"name": "val_bpb", "direction": "minimize"},
        "comparison_mode": "quality_at_fixed_time",
    }
    rr = {
        "candidate": {"valid": True, "metric": 0.9},
        "verdict": {"improved": True},
        "diff": "curriculum = True",
        "config": {"best_variant": {"curriculum": False}},
    }

    out = handler._review_method_implementation(
        "s1", rr, "method", evaluation_semantics=semantics)

    assert out["candidate"]["valid"] is True
    assert captured["semantics"] == semantics
    assert '"candidate_config": {"best_variant": {"curriculum": false}}' \
        in captured["supporting_summary"]


def test_method_implementation_contract_review_receives_confirmed_boundary(monkeypatch):
    handler = _handler({
        "approach": "Use curriculum ordering.",
        "components": [{"name": "Curriculum ordering", "type": "gain"}],
    })
    captured = {}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {
            "verdict": "faithful", "approach_evidence": ["curriculum = True"],
            "component_reviews": [{"name": "Curriculum ordering",
                                   "status": "implemented",
                                   "evidence": ["curriculum = True"]}],
            "issues": [], "remediation": None, "reasoning": "Implemented.",
        })

    def contract_review(contract, diff, llm):
        captured.update(contract)
        return {"verdict": "compliant", "violations": [], "evidence": [],
                "reasoning": "The local run boundary is confirmed."}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        contract_review)
    boundary = {"accelerator_type": "local-device", "conflicts_with_official": True}

    handler._review_method_implementation(
        "s1", {"candidate": {"valid": True, "metric": 0.9},
               "verdict": {"improved": True}, "diff": "+curriculum = True"},
        "method", repository_contract={"sources": [{"content": "Use official-device."}]},
        confirmed_boundary=boundary)

    assert captured["host_confirmed_boundary"] == boundary


def test_repository_contract_violation_invalidates_a_real_improvement(monkeypatch):
    stored = {}

    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            return {"approach": "Use algorithm Q.",
                    "components": [{"name": "Algorithm Q"}]}

        def put(self, *, kind, session_id, key, obj, emitter):
            stored["review"] = obj
            return f"spec://{kind}/{session_id}/{key}"

    fidelity = {
        "verdict": "faithful", "approach_evidence": ["algorithm_q()"],
        "component_reviews": [{"name": "Algorithm Q", "status": "implemented",
                               "evidence": ["algorithm_q()"]}],
        "issues": [], "remediation": None, "reasoning": "Implemented as proposed.",
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: fidelity)
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        lambda *args, **kwargs: {
            "verdict": "violation", "violations": ["Algorithm Q is prohibited."],
            "evidence": ["algorithm_q()"], "reasoning": "The diff invokes it.",
        })
    handler = RunnerAgentHandler(
        spec_store=Store(), backend=FakeRunnerBackend(), p1_llm=object())

    out = handler._review_method_implementation(
        "s1", {"candidate": {"valid": True, "metric": 0.1},
               "verdict": {"improved": True}, "diff": "+algorithm_q()"},
        "method", repository_contract={"sources": [{"content": "No algorithm Q."}]})

    assert out["candidate"]["valid"] is False
    assert out["verdict"]["improved"] is False
    assert out["diagnosis"] == "candidate_contract_violation"
    assert "Algorithm Q is prohibited" in out["verdict"]["reason"]
    assert stored["review"]["task_contract_review"]["verdict"] == "violation"


def test_contract_review_receives_scheduled_component_not_final_result_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    handler = _handler(_component_method())
    handler._component_directive("s1")
    captured = {}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})

    def contract_review(contract, diff, llm):
        captured.update(contract)
        return {"verdict": "compliant", "violations": [], "evidence": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract",
        contract_review)
    contract = {"sources": [{"content": "Final result must pass the official evaluator."}]}
    handler._review_method_implementation("s1", {
        "candidate": {"valid": False, "metric": None},
        "verdict": {"improved": False}, "diff": "+split_features(x)",
        "component_attempt_id": "attempt-123",
        "ablation_report": [{"component": "split-feature", "actual": 0.95,
                             "evidence": {"ref": "attempt-123"}}],
    }, "method", repository_contract=contract)

    assert captured["review_scope"]["mode"] == "component"
    assert captured["review_scope"]["component_id"] == "split-feature"
    assert captured["review_scope"]["role"] == "indirect_gain"
    assert captured["review_scope"]["acceptance_check"] == "separation_score >= 0.9"
    assert captured["runtime_evidence"]["indirect_component_probe"] == {
        "attempt_id": "attempt-123",
        "ablation_report": [{"component": "split-feature", "actual": 0.95,
                             "evidence": {"ref": "attempt-123"}}],
    }
    assert "review_scope" not in contract


@pytest.mark.parametrize("valid", [True, False])
def test_contract_review_receives_unchanged_host_measurement_receipt(monkeypatch, valid):
    handler = _handler(_component_method())
    captured = {}
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda *args, **kwargs: {"verdict": "faithful", "issues": []})

    def review(contract, diff, llm):
        captured.update(contract)
        return {"verdict": "compliant", "violations": [], "evidence": []}

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_task_contract", review)
    receipt = {"value": 0.8, "valid": valid, "gates": {"/candidate_valid": valid},
               "attempt_id": "attempt-official", "stdout_sha256": "a" * 64}
    handler._review_method_implementation("s1", {
        "candidate": {"valid": valid, "metric": 0.8},
        "verdict": {"improved": False}, "diff": "+candidate(x)",
        "measurement_evidence": receipt,
    }, "method", repository_contract={"sources": [{"content": "Preserve exact outputs."}]})

    if valid:
        assert captured["runtime_evidence"]["measurement_evidence"] == receipt
        assert "review_scope" not in captured
    else:
        assert captured == {}  # Invalid execution still stops before source review.


def test_method_implementation_review_keeps_partial_measurement_without_core_attribution(
        monkeypatch):
    handler = _handler({
        "approach": "Replace the FFN with KAN and a residual gate.",
        "components": [{"name": "KAN projection"}, {"name": "Residual gate"}],
    })
    review = {
        "verdict": "partial",
        "approach_evidence": [],
        "component_reviews": [
            {"name": "KAN projection", "status": "implemented",
             "evidence": ["self.kan = KANProjection(config)"]},
            {"name": "Residual gate", "status": "missing", "evidence": []},
        ],
        "issues": ["The core approach is not evidenced."],
        "remediation": "Implement the frozen core approach.",
        "reasoning": "A helper exists but the core replacement is not evidenced.",
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda method, diff, llm, supporting_summary="", evaluation_semantics=None: review)
    rr = {
        "candidate": {"valid": True, "metric": 0.9},
        "verdict": {"improved": True, "reason": "lower loss"},
        "diff": "self.kan = KANProjection(config)",
    }

    out = handler._review_method_implementation("s1", rr, "method")

    assert out["candidate"]["valid"] is True
    assert out["verdict"]["improved"] is True
    assert "only the implemented frozen components may be credited" \
        in out["verdict"]["reason"]
    assert "diagnosis" not in out


def test_drifted_improvement_is_retained_as_off_method_observation(monkeypatch):
    handler = _handler({
        "approach": "Use a fused optimizer schedule.",
        "components": [{"name": "fused schedule", "type": "gain"}],
    })
    review = {
        "verdict": "drifted",
        "approach_evidence": ["use_gradient_clipping = True"],
        "component_reviews": [{
            "name": "fused schedule", "status": "contradicted",
            "evidence": ["use_gradient_clipping = True"],
        }],
        "issues": ["The gain came from clipping, not the fused schedule."],
        "remediation": "Align the candidate or formally revise the method description.",
        "reasoning": "The measured code uses a different mechanism.",
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda method, diff, llm, supporting_summary="", evaluation_semantics=None: review)
    rr = {
        "candidate": {"valid": True, "metric": 0.9},
        "verdict": {"improved": True, "reason": "lower loss"},
        "diff": "use_gradient_clipping = True",
        "candidate_admissibility": {
            "verdict": "pass", "reason": "host contract and evaluator passed"},
    }

    out = handler._review_method_implementation("s1", rr, "method")

    assert out["candidate"]["valid"] is True
    assert out["verdict"]["improved"] is False
    assert out["off_method_observation"]["valid"] is True
    assert out["off_method_observation"]["improved"] is True
    assert out["off_method_observation"]["metric"] == 0.9
    assert out["off_method_observation"]["admissibility"]["verdict"] == "pass"


def test_drifted_observation_without_contract_review_is_not_retained(monkeypatch):
    handler = _handler({
        "approach": "Use a fused optimizer schedule.",
        "components": [{"name": "fused schedule", "type": "gain"}],
    })
    review = {
        "verdict": "drifted",
        "approach_evidence": ["use_gradient_clipping = True"],
        "component_reviews": [{
            "name": "fused schedule", "status": "contradicted",
            "evidence": ["use_gradient_clipping = True"],
        }],
        "issues": ["Different mechanism."],
        "remediation": "Align the candidate.",
        "reasoning": "The code drifted.",
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        lambda method, diff, llm, supporting_summary="", evaluation_semantics=None: review)
    rr = {
        "candidate": {"valid": True, "metric": 0.9},
        "verdict": {"improved": True, "reason": "lower loss"},
        "diff": "use_gradient_clipping = True",
        "candidate_admissibility": {
            "verdict": "fail", "reason": "changed the official evaluator"},
    }

    out = handler._review_method_implementation("s1", rr, "method")

    assert out["off_method_observation"]["valid"] is False
    assert out["off_method_observation"]["improved"] is False
    assert "not retained" in out["diagnosis_evidence"][0]
    assert out["diagnosis"] == "candidate_contract_violation"


def test_baseline_phase_does_not_run_method_implementation_review():
    handler = _handler({"approach": "new optimizer"})
    rr = {"candidate": {"valid": False}, "verdict": {"improved": False}}

    assert handler._review_method_implementation("s1", rr, "baseline") is rr


def test_check_expectation_diagnostic_is_not_gated_as_whole_method(monkeypatch):
    """A/B component probes intentionally do not each implement the whole frozen method."""
    handler = _handler({
        "approach": "Combine GC with a fixed learning rate.",
        "components": [{"name": "GC"}, {"name": "fixed learning rate"}],
    })
    handler._last_operator = {"s1": "check-expectation"}

    def must_not_review(*args, **kwargs):
        raise AssertionError("diagnostic component probes are not whole-method candidates")

    monkeypatch.setattr(
        "autoresearch.agents.method_implementation_agent.core.review_implementation",
        must_not_review)
    rr = {
        "candidate": {"valid": True, "metric": 1.2},
        "verdict": {"improved": False, "reason": "diagnostic only"},
        "diff": "variant_b = no_gc\nvariant_c = cosine_schedule",
        "ablation_report": [{"_host_bound": True,
            "component": "GC", "type": "gain", "actual": -0.01,
            "expected": {"operator": ">=", "value": 0.0},
            "evidence": {"source": "A/B", "ref": "attempts A and B", "json_pointer": "/separation_score"},
        }],
    }

    out = handler._review_method_implementation("s1", rr, "method")

    assert out is rr
    assert out["candidate"]["valid"] is True
    assert "diagnosis" not in out


def test_with_operator_injects_method_implementation_remediation(monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if "method_implementation_review" in ref:
                return {
                    "method_ref": "spec://method/s1/current",
                    "verdict": "partial",
                    "issues": ["Residual gate is missing."],
                    "remediation": "Implement the residual gate before measuring again.",
                }
            if "method" in ref:
                return {"approach": "KAN with residual gate"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    question = handler._with_operator("s1", "continue implementation")

    assert "IMPLEMENTATION FIDELITY CORRECTION" in question
    assert "Residual gate is missing" in question
    assert "valid measured intermediate" in question
    assert "next missing confirmed component" in question
    assert "Do not run whole-method ablations yet" in question
    assert question.index("Residual gate is missing") < question.index("continue implementation")


@pytest.mark.parametrize("fidelity", ["faithful", "partial", "drifted"])
@pytest.mark.parametrize("contract_verdict", ["violation", "insufficient_evidence"])
@pytest.mark.parametrize("operators", ["0", "1"])
def test_integration_receives_contract_failure_even_when_faithful(
        monkeypatch, fidelity, contract_verdict, operators):
    review = {
        "method_ref": "spec://method/s1/current", "verdict": fidelity,
        "issues": [], "remediation": None,
        "task_contract_review": {
            "verdict": contract_verdict,
            "violations": ["The build does not use the required interface."]
            if contract_verdict == "violation" else [],
            "reasoning": "Check the changed build command against the frozen interface.",
        },
    }
    handler = _handler(review)
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: review["method_ref"])
    monkeypatch.setattr(handler, "_method_directive", lambda _: "")
    monkeypatch.setattr(handler, "_research_constraints_directive", lambda _: "")
    monkeypatch.setattr(handler, "_comparison_correction_directive", lambda _: "")
    monkeypatch.setattr(handler, "_verified_failure_directive", lambda _: "verbose old log " * 300)

    def integration_directive(session_id):
        handler._active_component_context = {session_id: {"mode": "integration", "operator": "improve"}}
        return "COMPONENT INTEGRATION: assemble accepted checkpoints\n"

    monkeypatch.setattr(handler, "_component_directive", integration_directive)
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", operators)
    question = handler._with_operator("s1", "continue integration")
    assert "TASK CONTRACT CORRECTION" in question
    assert "Check the changed build command" in question
    assert question.index("Check the changed build command") < question.index("verbose old log")
    if contract_verdict == "violation":
        assert "The build does not use the required interface" in question


@pytest.mark.parametrize("current,verdict", [(False, "violation"), (True, "compliant")])
def test_contract_feedback_does_not_reject_compliance_or_leak_old_method(monkeypatch, current, verdict):
    handler = _handler({
        "method_ref": "spec://method/s1/current" if current else "spec://method/s1/old",
        "verdict": "faithful", "task_contract_review": {
            "verdict": verdict, "violations": [], "reasoning": "review detail",
        },
    })
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: "spec://method/s1/current")
    assert handler._method_implementation_directive("s1") == ""


def test_component_retry_does_not_inject_whole_method_remediation(monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/current"] if kind == "method" else []

        def get(self, ref):
            if "method_implementation_review" in ref:
                return {
                    "method_ref": "spec://method/s1/current",
                    "verdict": "partial",
                    "issues": ["A later sibling component is missing."],
                    "remediation": "Implement the later sibling too.",
                }
            if "verification_report" in ref:
                return {
                    "verdict": "verified",
                    "checks": ["failure_confirmed: implementation review rejected the run"],
                    "reasoning": "Recorded failure cause: implement the later sibling too.",
                }
            if "run_record" in ref:
                return {"experiment_id": "spec://method/s1/current"}
            if "method" in ref:
                return {"approach": "component DAG method"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    handler = RunnerAgentHandler(spec_store=Store(), backend=FakeRunnerBackend())

    def component_directive(session_id):
        handler._active_component_context = {
            session_id: {"mode": "component", "node": SimpleNamespace(component_id="c1")}}
        return "CURRENT COMPONENT c1 ONLY\n"

    monkeypatch.setattr(handler, "_component_directive", component_directive)
    question = handler._with_operator("s1", "continue implementation")

    assert "CURRENT COMPONENT c1 ONLY" in question
    assert "IMPLEMENTATION FIDELITY CORRECTION" not in question
    assert "VERIFIED FAILURE FROM THE PREVIOUS ATTEMPT" not in question
    assert "later sibling" not in question


def test_with_operator_does_not_leak_implementation_review_from_replaced_method(monkeypatch):
    class Store:
        def list_refs(self, session_id, kind, *, limit=None):
            return ["spec://method/s1/new"] if kind == "method" else []

        def get(self, ref):
            if "method_implementation_review" in ref:
                return {
                    "method_ref": "spec://method/s1/old",
                    "verdict": "drifted",
                    "issues": ["Old GC and fixed LR were split."],
                    "remediation": "Combine old GC with fixed LR.",
                }
            if ref == "spec://method/s1/new":
                return {"approach": "short warmup plus clipping"}
            raise KeyError(ref)

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    question = RunnerAgentHandler(
        spec_store=Store(), backend=FakeRunnerBackend())._with_operator(
            "s1", "run the replacement")

    assert "CONFIRMED METHOD" in question
    assert "short warmup plus clipping" in question
    assert "IMPLEMENTATION FIDELITY CORRECTION" not in question
    assert "Old GC" not in question


def test_failed_debug_worktree_restores_as_unqualified_immutable_snapshot(tmp_path):
    import subprocess
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import restore_external, validate_checkpoint

    source = tmp_path / "repo-failed"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "solve.py").write_text("original = True\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                    "-c", "user.email=test@example.test", "commit", "-qm", "baseline"], check=True)
    target = tmp_path / "repo-next"
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True)
    (source / "solve.py").write_text("candidate = True\n")
    entry = SimpleNamespace(checkpoint="", debug_checkpoint=str(source),
                            last_outcome="execution_failed")
    runner = _handler(_component_method())
    runner._active_component_context = {"s1": {
        "mode": "component", "operator": "debug",
        "node": SimpleNamespace(component_id="current", depends_on=()),
        "ledger": SimpleNamespace(entry=lambda _: entry),
    }}
    snapshot = runner._prior_worktree("s1")
    validate_checkpoint(snapshot)
    _apply_prior_worktree(str(target), snapshot)
    restore_external(snapshot, FakeRunnerBackend(), None, str(target))
    assert (target / "solve.py").read_text() == "candidate = True\n"
    assert entry.checkpoint == ""
    assert entry.debug_checkpoint == str(source)
    assert runner._prior_worktree("s1", dependencies_only=True) == []


@pytest.mark.parametrize("value,expected", [(False, False), ("false", False), (True, True)])
def test_gpu_routing_requires_boolean(monkeypatch, value, expected):
    handler = _handler({})
    handler.p1_llm = SimpleNamespace(chat=lambda **_: SimpleNamespace(content=json.dumps({"needs_gpu": value})))
    assert handler._needs_gpu("sort a small array on CPU", None) is expected


def test_preparation_failure_does_not_freeze_indirect_parent(tmp_path):
    from autoresearch.experiments.repo_improve.component_plan import ComponentLedger, build_component_plan
    plan = build_component_plan({"components": [_component_method()["components"][0]]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("split-feature")
    ledger.record_attempt("split-feature", execution_succeeded=True, contract_passed=True,
                          checkpoint="/saved-parent", max_attempts=1)
    handler = _handler({})
    handler._active_component_context = {"s1": {"mode": "integration", "plan": plan, "ledger": ledger}}
    failure = handler._record_component_failure("s1", "preparation failed")
    assert failure["attempts"] == 0
    ledger.decide_qualified("split-feature", result_ref="saved", action="advance", reason="retain")
    started = handler._record_component_failure("s1", "reply lost", execution_started=True)
    assert started["attempts"] == 1



@pytest.mark.parametrize("stage", ["implementation", "comparison", "comparison_parse", "failed_control"])
def test_pending_review_reuses_saved_measurement_without_executing(monkeypatch, tmp_path, stage):
    import subprocess
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    store = SpecStore(tmp_path / "saved-review")
    method_ref = store.put(kind="method", session_id="s1", key="current",
                           obj={"approach": "change algorithm"}, emitter="MethodAgent")
    baseline = {"source_kind": "live_repo", "metric_value": 1.0,
                "recovery_binding": "rebound-snapshot"}
    stale_baseline = {**baseline, "recovery_binding": "mutable-v1-slot"}
    store.put(kind="baseline_result", session_id="s1", key="v1",
              obj=stale_baseline, emitter="RunnerAgent")
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend(), p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "goal")
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: method_ref)
    monkeypatch.setattr(handler, "_component_directive", lambda _: "")
    replies = [{"verdict": "insufficient_evidence", "review_error": "review transport failed"},
               {"verdict": "faithful", "component_reviews": []}]
    monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_implementation",
                        lambda *a, **k: replies.pop(0))
    monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_task_contract",
                        lambda *a, **k: {"verdict": "not_available"})
    repo = tmp_path / "measured-repo"
    repo.mkdir()
    (repo / "solve.py").write_text("def solve(x): return x\n")
    for args in [("init", "-q"), ("add", "solve.py"),
                 ("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    receipt = dict(attempt_id="attempt-saved", launched=True, finished=True, rc=0)
    receipt.update(save_output(tmp_path, receipt["attempt_id"], '{"method":0.9}', ""))
    original = {"baseline": {"metric": 1.0, "valid": True},
                "candidate": {"metric": 0.9, "valid": True},
                "verdict": {"improved": True}, "diff": "+changed()", "lower_is_better": True,
                "checkpoint": snapshot_checkpoint(str(repo), "saved"), "review_receipts": [receipt]}
    if stage == "failed_control":
        original["review_receipts"].append(dict(
            attempt_id="attempt-failed-control", launched=True, finished=True, rc=126,
            **save_output(tmp_path, "attempt-failed-control", "", "permission denied\n")))
    if stage in {"comparison", "comparison_parse"}:
        from autoresearch.agents.comparison_guardian_agent.core import (
            ComparisonGuardianLLMError, ComparisonGuardianParseError)
        failure = (ComparisonGuardianParseError("invalid response schema")
                   if stage == "comparison_parse" else
                   ComparisonGuardianLLMError("LLM call failed: APITimeoutError: Request timed out."))
        calls = []
        def audit(*args, **kwargs):
            calls.append(args[2])
            if len(calls) == 1:
                raise failure
            return {"verdict": "pass", "issues": [], "remediation": None, "reasoning": "same task"}
        monkeypatch.setattr("autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", audit)
        replies.pop(0)
        with pytest.raises(type(failure)):
            handler._review_repository_comparison("s1", original, "method", frozen_baseline=baseline)
    else:
        unavailable = handler._review_method_implementation(
            "s1", original, "method", frozen_baseline=baseline)
        assert unavailable["candidate"]["valid"] is False
    pending = store.get("spec://pending_implementation_review/s1/v1")
    immutable = store.get(pending["input_ref"])
    assert immutable["result"] == original
    record, result = handler._resume_implementation_review("s1", "goal", baseline)
    assert result["method_metric"] == 0.9
    assert result["method_implementation_review"]["verdict"] == "faithful"
    if stage in {"comparison", "comparison_parse"}:
        assert len(calls) == 2
    assert store.get(pending["input_ref"]) == immutable
    assert pending["baseline_digest"] == artifact_digest(baseline)
    with pytest.raises(Exception, match="different Method or baseline"):
        handler._resume_implementation_review("s1", "goal", {"metric_value": 2})



@pytest.mark.parametrize("source_kind", ["off_method", "partial_integration", "component_local", "accepted_whole", "qualified_component"])
@pytest.mark.parametrize("repair_source", ["none", "same_method", "other_method"])
def test_formal_revision_materializes_selected_source_at_new_root(tmp_path, monkeypatch, source_kind, repair_source):
    import subprocess
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "solve.py").write_text("def solve(x): return x\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    git("init", "-q")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "add", "solve.py")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    (repo / "solve.py").write_text("def solve(x): return x + 1\n")
    checkpoint = snapshot_checkpoint(str(repo), "selected")
    store = SpecStore(tmp_path / "root-store")
    selected = {"checkpoint": checkpoint, "off_method_observation": {"valid": True}}
    if source_kind == "partial_integration":
        selected = {"checkpoint": checkpoint, "accepted": True, "method_metric": 1.0,
            "measurement_evidence": {"value": 1.0, "valid": True,
                "attempt_id": "attempt-retained", "stdout_sha256": "a" * 64,
                "protocol_fingerprint": "b" * 64, "protocol": {"kind": "official_single"}},
            "candidate_admissibility": {"verdict": "pass", "attempt_id": "attempt-retained"},
            "method_implementation_review": {"verdict": "partial",
                "task_contract_review": {"verdict": "compliant"}},
            "repository_comparison_review": {"verdict": "pass"},
            "component_execution": {"mode": "integration", "status": "new_best",
                "execution_succeeded": True, "attempt_contract_result": {"passed": True}}}
    if source_kind in {"component_local", "qualified_component"}:
        from autoresearch.agents.memory_agent.tests.test_incremental_method_updates import component_observation
        selected = component_observation()
        selected["checkpoint"] = checkpoint
        if source_kind == "qualified_component":
            selected["accepted"] = True
            selected["method_implementation_review"]["verdict"] = "faithful"
            selected["component_execution"]["status"] = "qualified"
    if source_kind == "accepted_whole":
        selected = {"checkpoint": checkpoint, "accepted": True, "method_metric": 1.0,
            "method_ref": "spec://method/s1/source",
            "measurement_evidence": {"value": 1.0, "valid": True,
                "attempt_id": "attempt-retained", "stdout_sha256": "a" * 64,
                "protocol_fingerprint": "b" * 64, "protocol": {"kind": "official_single"}},
            "candidate_admissibility": {"verdict": "pass", "attempt_id": "attempt-retained"},
            "repository_comparison_review": {"verdict": "pass"},
            "method_implementation_review": {"verdict": "faithful", "method_ref": "spec://method/s1/source",
                "task_contract_review": {"verdict": "compliant"}}}
    selected_ref = store.put(kind="gen_experiment_result", session_id="s1",
        key="snapshot-" + artifact_digest(selected).split(":", 1)[1], obj=selected, emitter="RunnerAgent")
    method = {"components": [_component_method()["components"][0]], "reconciled_result_ref": selected_ref}
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path / "research"))
    from autoresearch.experiments.repo_improve.component_plan import build_component_plan, ledger_for_workroot
    from autoresearch.experiments.repo_improve.search_memory import session_workroot
    original = ledger_for_workroot(session_workroot('s1'),
        build_component_plan({'components': method['components']}))
    original.begin('split-feature')
    for _ in range(8):
        original.record_attempt('split-feature', execution_succeeded=False, max_attempts=8)
    original_snapshot = original.snapshot()
    monkeypatch.setattr(handler, "_latest_method", lambda _: method)
    directive = handler._component_directive("s1")
    assert handler._active_component_context["s1"]["mode"] == "revalidation"
    assert "whole Method" in directive and "official evaluator" in directive
    retained_ledger = handler._active_component_context["s1"]["ledger"]
    assert not retained_ledger.all_accepted()
    assert all(retained_ledger.entry(cid).attempts == 0
               for cid in retained_ledger.plan.topological_order)
    pending_snapshot = retained_ledger.snapshot()
    full_result = {"accepted": True, "verdict": "new_best", "method_metric": 1.0}
    finalized = handler._finalize_component_attempt("s1", full_result)
    assert "component_execution" not in finalized
    assert retained_ledger.snapshot() == pending_snapshot
    prior = handler._prior_worktree("s1")
    assert prior == checkpoint
    assert original.snapshot() == original_snapshot
    assert original.entry('split-feature').status == 'exhausted'
    assert original.entry('split-feature').attempts == 8
    assert handler._active_component_context["s1"]["ledger"].root_checkpoint == ""
    assert handler._prior_worktree("s1", dependencies_only=True) == []
    handler._active_component_context["s1"]["mode"] = "integration"
    assert handler._prior_worktree("s1") == checkpoint
    target = tmp_path / "next"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    assert _apply_prior_worktree(str(target), prior) == 1
    assert (target / "solve.py").read_text() == "def solve(x): return x + 1\n"
    if repair_source != "none":
        (repo / "solve.py").write_text("def solve(x): return x + 2\n")
        repaired_checkpoint = snapshot_checkpoint(str(repo), "partial-repair")
        method_ref = store.put(kind="method", session_id="s1", key="final", obj=method, emitter="MethodAgent")
        monkeypatch.setattr(handler, "_latest_method_ref", lambda _: method_ref)
        observed_method = method_ref if repair_source == "same_method" else "method://s1/other"
        store.put(kind="gen_experiment_result", session_id="s1", key="current", emitter="RunnerAgent",
            obj={"method_ref": observed_method, "accepted": True, "verdict": "new_best",
                 "method_metric": 1.0, "checkpoint": repaired_checkpoint,
                 "measurement_evidence": {"valid": True},
                 "candidate_admissibility": {"verdict": "pass"},
                 "repository_comparison_review": {"verdict": "pass"},
                 "method_implementation_review": {"verdict": "partial", "method_ref": observed_method,
                     "task_contract_review": {"verdict": "compliant"}}})
        handler._component_directive("s1")
        assert handler._prior_worktree("s1") == (
            repaired_checkpoint if repair_source == "same_method" else checkpoint)
        assert original.snapshot() == original_snapshot


def test_formal_revision_accepts_its_bound_checkpoint_as_named_root(tmp_path, monkeypatch):
    import os
    import subprocess
    from autoresearch.spec_store import SpecStore
    from autoresearch.spec_store.identity import artifact_digest
    from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
    from autoresearch.agents.memory_agent.tests.test_incremental_method_updates import (
        component_observation,
    )

    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "solve.py").write_text("def solve(x): return x + 1\n")
    for args in [("init", "-q"), ("add", "solve.py"),
                 ("-c", "user.name=test", "-c", "user.email=test@example.invalid",
                  "commit", "-qm", "source")]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    checkpoint = snapshot_checkpoint(str(repo), "retained")
    selected = component_observation()
    selected["checkpoint"] = checkpoint
    store = SpecStore(tmp_path / "root-store")
    selected_ref = store.put(kind="gen_experiment_result", session_id="s1",
        key="snapshot-" + artifact_digest(selected).split(":", 1)[1], obj=selected,
        emitter="RunnerAgent")
    component = _component_method()["components"][0]
    component["inputs"] = [f"Retained source {os.path.basename(checkpoint)}"]
    method = {"components": [component], "reconciled_result_ref": selected_ref}
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path / "research"))
    monkeypatch.setattr(handler, "_latest_method", lambda _: method)

    directive = handler._component_directive("s1")

    assert "RETAINED METHOD REVALIDATION" in directive
    assert handler._active_component_context["s1"]["mode"] == "revalidation"
    assert handler._active_component_context["s1"]["ledger"].root_checkpoint == checkpoint
    assert handler._prior_worktree("s1") == checkpoint


def test_revalidation_does_not_promote_candidate_rejected_by_frozen_adoption_gate():
    handler = _handler(_component_method())
    handler._active_component_context = {"s1": {"mode": "revalidation"}}
    attempt_id = "attempt-paired-candidate"
    result = handler._finalize_component_attempt("s1", {
        "accepted": True,
        "verdict": "new_best",
        "method_metric": 0.9,
        "measurement_evidence": {"attempt_id": attempt_id},
        "ablation_report": [{
            "component": "split-feature",
            "type": "direct_gain",
            "actual": 0.9,
            "expected": {"operator": "<", "value": 0.8},
            "evidence": {"source": "runtime_attempt", "ref": attempt_id},
        }],
    })

    assert result["accepted"] is False
    assert result["verdict"] == "valid_not_best"
    assert result["diagnosis"] == "component_expectation_failed"
    assert "split-feature" in result["diagnosis_evidence"]


def test_revalidation_can_accept_retained_source_when_rejected_probe_is_not_method_metric():
    handler = _handler(_component_method())
    handler._active_component_context = {"s1": {"mode": "revalidation"}}
    attempt_id = "attempt-paired-candidate"
    result = handler._finalize_component_attempt("s1", {
        "accepted": True,
        "verdict": "new_best",
        "method_metric": 1.0,
        "measurement_evidence": {"attempt_id": attempt_id},
        "ablation_report": [{
            "component": "split-feature",
            "type": "direct_gain",
            "actual": 0.9,
            "expected": {"operator": "<", "value": 0.8},
            "evidence": {"source": "runtime_attempt", "ref": attempt_id},
        }],
    })

    assert result["accepted"] is True
    assert result["verdict"] == "new_best"


@pytest.mark.parametrize("already_snapshotted", [False, True])
def test_remote_failed_debug_checkpoint_uses_execution_host(tmp_path, monkeypatch, already_snapshotted):
    import shlex
    import subprocess
    import sys
    from pathlib import Path
    from autoresearch.agents.runner_agent import handler as module
    from autoresearch.experiments.repo_improve import checkpoint_artifacts as checkpoints

    source = tmp_path / "remote-repo"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "solve.py").write_text("baseline = True\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                    "-c", "user.email=test@example.test", "commit", "-qm", "baseline"], check=True)
    (source / "solve.py").write_text("failed_candidate = True\n")
    if already_snapshotted:
        source = Path(checkpoints.snapshot_checkpoint(str(source), "failed"))
    entry = SimpleNamespace(checkpoint="", debug_checkpoint=str(source), last_outcome="execution_failed")
    runner = _handler(_component_method())
    runner._active_component_context = {"s1": {
        "mode": "component", "operator": "debug",
        "node": SimpleNamespace(component_id="current", depends_on=()),
        "ledger": SimpleNamespace(entry=lambda _: entry),
    }}
    monkeypatch.setenv("AUTORESEARCH_GPU_HOST", "remote-host")
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE_DIR", str(Path.cwd()))
    monkeypatch.setenv("AUTORESEARCH_REMOTE_PY", sys.executable)
    calls = []
    def remote(host, command, timeout):
        calls.append((host, command, timeout))
        return subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=timeout)
    monkeypatch.setattr(module, "_ssh_dispatch", remote)
    monkeypatch.setattr(checkpoints, "snapshot_checkpoint", lambda *_: pytest.fail("local snapshot attempted"))
    monkeypatch.setattr(checkpoints, "validate_checkpoint", lambda *_: pytest.fail("local validation attempted"))
    saved = runner._prior_worktree("s1", remote=True)
    assert len(calls) == 1 and calls[0][0] == "remote-host"
    assert Path(saved, "solve.py").read_text() == "failed_candidate = True\n"
    assert (saved == str(source)) is already_snapshotted
    assert entry.checkpoint == "" and entry.debug_checkpoint == str(source)
    assert runner._prior_worktree("s1", remote=True, dependencies_only=True) == []


def test_remote_debug_checkpoint_failure_is_not_accepted_or_local_fallback(monkeypatch):
    from autoresearch.agents.runner_agent import handler as module
    from autoresearch.agents.runner_agent.core import ComponentPlanInvalidError
    entry = SimpleNamespace(checkpoint="", debug_checkpoint="/remote/missing", last_outcome="execution_failed")
    runner = _handler(_component_method())
    runner._active_component_context = {"s1": {
        "mode": "component", "operator": "debug",
        "node": SimpleNamespace(component_id="current", depends_on=()),
        "ledger": SimpleNamespace(entry=lambda _: entry),
    }}
    monkeypatch.setenv("AUTORESEARCH_GPU_HOST", "remote-host")
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE_DIR", "/remote/runtime")
    monkeypatch.setenv("AUTORESEARCH_REMOTE_PY", "/remote/python")
    monkeypatch.setattr(module, "_ssh_dispatch", lambda *_args, **_kw: SimpleNamespace(
        returncode=1, stdout="", stderr="checkpoint identity changed"))
    with pytest.raises(ComponentPlanInvalidError, match="checkpoint identity changed"):
        runner._prior_worktree("s1", remote=True)
    assert entry.checkpoint == ""
