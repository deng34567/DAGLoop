"""Method-layer NESTED FOUR-ROLE deliberation (meeting 2026-08-03, option B).

`deliberate_method` runs a NESTED bounded loop:
  OUTER (<= outer_rounds): each outer round re-deliberates.
    INNER (<= inner_rounds): PURE A1<->A2 convergence -- both propose + critique the other; a
      convergence judge checks; NOT converged -> MOSS GUIDES them toward agreement (drives next inner
      exchange, not a judge). The CRITIC does NOT run inside the inner loop (前两个先收敛).
    after inner: capped-without-converging -> stop (keep A1). converged -> the CRITIC reviews the
      agreed method against its TARGET (a defensible, real improvement on the metric); no undermining
      problem -> accept; else MOSS (coordinator, NOT a judge) GUIDINGLY relays the critique -> another
      OUTER round.
"""
from __future__ import annotations

import json

import pytest

from autoresearch.agents.deliberation_agent import core

_PM = {"research_goal": "lower val loss on nanoGPT"}
_LIT = {"summary": "prior work tunes lr schedule"}
_DRAFT = {"approach": "draft", "changes_outline": ["x"], "why_it_should_win": "w", "assumptions": []}


def test_revert_only_gain_ancestor_requires_method_revision():
    method_critic = core._METHOD_CRITIC_SYSTEM
    plan_critic = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "control or revert checkpoint" in method_critic
    assert "implemented dependencies in the final candidate" in method_critic
    assert "Read-only features of the current state" in method_critic
    assert "qualified component evidence from accepted complete results" in method_critic
    assert "gain-bearing ancestor is only a control or revert target" in plan_critic
    assert "changing an implementation_check alone cannot repair" in plan_critic


def test_component_revision_preserves_unaffected_contracts_verbatim():
    context = core._method_ctx(
        _PM, _LIT, None, component_plan_audit=True,
        component_plan_revision_feedback="Repair the final component's acceptance contract.",
    )
    assert "copy every unaffected component object verbatim" in context
    assert "including IDs, acceptance text, and downstream descriptions" in context
    assert "exact contract equality" in context
    assert "state the concrete reason" in context
    assert "file-path collision is not by itself a scientific dependency" in context
    assert "component-scoped helper paths" in context
    assert "Adding a join alone cannot resolve a conflict" in context
    assert "distinguish an acceptance-only diagnostic from treatment behavior" in context
    assert "merge those tightly coupled nodes" in context
    assert "Do not repeatedly keep the same conflicting boundary" in context
    initial = core._method_ctx(_PM, _LIT, None, component_plan_audit=True)
    assert "copy every unaffected component object verbatim" not in initial


def test_quota_is_not_converted_into_ten_fake_nonconvergent_rounds():
    calls = []
    class NoQuota:
        def chat(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("pre_consume_token_quota_failed")
    with pytest.raises(core.DeliberationLLMError, match="pre_consume"):
        core.deliberate_method(_PM, _LIT, None, _DRAFT, NoQuota(),
                               component_plan_audit=True)
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    "LLMServiceUnavailableError: InternalServerError: Error code: 503",
    "Conversation error code: LLMTimeoutError; llm_provider_timeout",
])
def test_provider_outage_is_not_converted_into_cap_protocol_failure(error):
    calls = []

    class ProviderOutage:
        def chat(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError(error)

    with pytest.raises(core.DeliberationLLMError) as caught:
        core.deliberate_method(_PM, _LIT, None, _DRAFT, ProviderOutage(),
                               component_plan_audit=True)
    assert error in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("raw", ['{"approach":', '{}', '[]'])
def test_cap_unusable_proposer_does_not_review_unchanged_fallback(raw):
    calls = []

    class InvalidProposal:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return _Resp(raw)

    with pytest.raises(core.DeliberationLLMError, match="A1.*usable"):
        core.deliberate_method(_PM, _LIT, None, _DRAFT, InvalidProposal(),
                               component_plan_audit=True)
    assert len(calls) == 2
    assert all("PROPOSER" in c["messages"][0]["content"] for c in calls)


def test_cap_proposer_recovers_one_invalid_response_without_new_debate_round():
    responses = iter(['{"approach":', json.dumps({"approach": "repaired", "components": []})])
    calls = []

    class RecoveringProposal:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return _Resp(next(responses))

    method, _ = core._propose(RecoveringProposal(), lambda x: x, "A1", "context",
                              _DRAFT, _DRAFT, "repair the contract",
                              require_valid_response=True)
    assert method["approach"] == "repaired"
    assert len(calls) == 2
    assert "valid JSON" in calls[1]["messages"][1]["content"]


class _Resp:
    def __init__(self, content):
        self.content = content


class _RoleLLM:
    """Routes by role marker. Convergence / critic are scriptable; the two proposers auto-return a
    revised method with the per-component CONTRACT (mechanism + expected_effect + ablation_test).
    MOSS is a GUIDE/RELAY (not a judge): it returns {"guidance": ...} for both the inner-loop nudge
    (not-converged) and the converged-critique relay."""

    def __init__(self, *, conv=None, critic=None, raise_all=False):
        self.conv = list(conv or [{"converged": True, "reason": "ok"}])
        self.critic = list(critic or [{"problems": []}])
        self.raise_all = raise_all
        self.calls = []
        self.call_kwargs = []
        self.roles: list[str] = []
        self._prop_n = 0

    def chat(self, *, messages, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        if self.raise_all:
            raise RuntimeError("simulated LLM failure")
        s = messages[0].get("content", "")
        if "METHOD PROPOSER" in s:
            self._prop_n += 1
            label = "A1" if "PROPOSER A1" in s else "A2"
            self.roles.append(f"proposer:{label}")
            return _Resp(json.dumps({
                "approach": f"m-{label}-{self._prop_n}",
                "changes_outline": ["c"], "why_it_should_win": "w", "assumptions": ["a"],
                "components": [{"name": "comp1", "purpose": "p",
                               "mechanism": "reduces gradient variance so it moves val_loss",
                               "expected_effect": "-0.02 val_loss vs baseline",
                               "ablation_test": "add vs remove comp1; expect the -0.02 to vanish"}],
                "critique_of_other": ["objection to the other proposer"]}))
        if "METHOD CRITIC" in s:   # check BEFORE the converge marker (critic text mentions CONVERGED)
            self.roles.append("critic")
            return _Resp(json.dumps(self.critic.pop(0) if self.critic else {"problems": []}))
        if "You judge whether" in s:
            self.roles.append("converge")
            return _Resp(json.dumps(self.conv.pop(0) if self.conv else {"converged": True}))
        if "You are MOSS, the coordinator" in s:   # MOSS = guide/relay, NOT a judge
            self.roles.append("moss_relay" if "RELAY" in s else "moss_guide")
            return _Resp(json.dumps({"guidance": "adopt the other's stronger idea / strengthen the ablation"}))
        return _Resp("{}")


class _FunctionalContractLLM(_RoleLLM):
    def __init__(self):
        super().__init__(
            conv=[{"converged": True}, {"converged": True}],
            critic=[{"problems": []}, {"problems": []}],
        )

    def chat(self, *, messages, **kwargs):
        system = messages[0].get("content", "")
        if "METHOD PROPOSER" in system:
            self.calls.append(messages)
            self._prop_n += 1
            label = "A1" if "PROPOSER A1" in system else "A2"
            self.roles.append(f"proposer:{label}")
            check = "below the repository threshold" if self._prop_n <= 2 else "mismatches == 0"
            return _Resp(json.dumps({
                "approach": "functional method",
                "changes_outline": ["guard ordering"],
                "why_it_should_win": "preserves correctness",
                "assumptions": [],
                "components": [{
                    "component_id": "order-guard",
                    "name": "order guard",
                    "depends_on": [],
                    "purpose": "preserve order",
                    "type": "functional",
                    "mechanism": "prevents output reordering",
                    "expected_effect": "zero ordering mismatches",
                    "ablation_test": "",
                    "functional_check": check,
                }],
                "critique_of_other": [],
            }))
        return super().chat(messages=messages, **kwargs)


def test_inner_converges_first_exchange_critic_clean_accepts():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    res = core.deliberate_method(_PM, _LIT, "my idea", _DRAFT, llm, inner_rounds=3, outer_rounds=2)
    assert res.report["converged"] is True and res.report["accepted"] is True
    assert res.report["outer_rounds"] == 1 and res.report["total_exchanges"] == 1
    assert res.method["approach"] == "m-A1-1"          # the agreed (A1) method, not the draft
    assert res.method["components"][0]["ablation_test"].startswith("add vs remove")
    assert res.method["components"][0]["mechanism"]        # contract carries a mechanism
    assert res.method["components"][0]["expected_effect"]  # ... and an expected effect (the hook)
    assert res.report["transcript"][-1]["outcome"] == "accepted_converged_clean"
    assert llm.roles.count("critic") == 1              # critic ran ONCE, AFTER convergence


@pytest.mark.parametrize("raw", ['{"problems":', '{}', '{"problems": null}',
                                 '{"problems": "none"}'])
@pytest.mark.parametrize("component_plan_audit", [False, True])
def test_unavailable_critic_never_counts_as_clean_acceptance(raw, component_plan_audit):
    class InvalidCriticLLM(_RoleLLM):
        def chat(self, *, messages, **kwargs):
            if "METHOD CRITIC" in messages[0]["content"]:
                return _Resp(raw)
            response = super().chat(messages=messages, **kwargs)
            if "METHOD PROPOSER" in messages[0]["content"]:
                method = json.loads(response.content)
                method["components"][0].update({
                    "component_id": "order-guard", "type": "indirect_gain",
                    "depends_on": [], "acceptance_requires": [],
                    "inputs": ["parent output"], "outputs": ["ordered output"],
                    "functional_check": "ordering_errors == 0",
                    "implementation_check": "runtime output preserves order",
                    "optimization_space": "reduce guard overhead",
                    "downstream_contract": "preserve the output interface",
                })
                return _Resp(json.dumps(method))
            return response

    if component_plan_audit:
        with pytest.raises(core.DeliberationLLMError, match="Critic.*protocol failure"):
            core.deliberate_method(
                _PM, _LIT, None, _DRAFT, InvalidCriticLLM(), inner_rounds=1, outer_rounds=1,
                component_plan_audit=True)
        return
    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, InvalidCriticLLM(), inner_rounds=1, outer_rounds=1)

    assert result.report["accepted"] is False
    assert result.report["transcript"][-1]["outcome"] == "critic_review_unavailable"


def test_cap_critic_repairs_missing_problems_in_same_review():
    llm = _RoleLLM(critic=[{"review": "incomplete response"}, {"problems": []}])
    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        component_plan_audit=True)
    calls = [(call, kwargs) for call, kwargs in zip(llm.calls, llm.call_kwargs)
             if "METHOD CRITIC" in call[0]["content"]]
    assert len(calls) == 2
    assert [kwargs["max_tokens"] for _, kwargs in calls] == [262144, 262144]
    assert "problems" in calls[1][0][1]["content"]
    assert "protocol" in calls[1][0][1]["content"]
    assert llm.roles.count("proposer:A1") == llm.roles.count("proposer:A2") == 1
    assert not any(t.get("outcome") == "critic_review_unavailable"
                   for t in result.report["transcript"])


def test_method_proposer_and_critic_request_room_for_structured_json():
    llm = _RoleLLM()
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
                           inner_rounds=1, outer_rounds=1)
    proposer_limits = [kwargs.get("max_tokens")
                       for call, kwargs in zip(llm.calls, llm.call_kwargs)
                       if "METHOD PROPOSER" in call[0]["content"]]
    critic_limits = [kwargs.get("max_tokens")
                     for call, kwargs in zip(llm.calls, llm.call_kwargs)
                     if "METHOD CRITIC" in call[0]["content"]]
    assert proposer_limits == [262144, 262144]
    assert critic_limits == [262144]


def test_method_proposer_json_contract_declares_reference_source_strings():
    assert '\"reference_sources\": [\"<exact allowed path or URL>\", ...]' in (
        core._METHOD_PROPOSER_SYSTEM
    )


@pytest.mark.parametrize("initial_limit", [8192, 16384, 32768, 262144])
def test_cap_compact_proposal_preserves_method_and_bounds_retry(initial_limit):
    calls = []
    responses = iter(['{"components":', json.dumps({"components": [
        {"name": "guard", "type": "indirect_gain", "functional_check": "errors == 0"}],
        "critique_of_other": []})])

    class CompactProposal:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return _Resp(next(responses))

    method, _ = core._propose(CompactProposal(), lambda x: x, "A1", "context",
        _DRAFT, _DRAFT, "repair", require_valid_response=True, max_tokens=initial_limit,
        system_template=core._COMPONENT_PLAN_PROPOSER_SYSTEM)
    assert method["approach"] == _DRAFT["approach"]
    assert method["changes_outline"] == _DRAFT["changes_outline"]
    assert method["components"][0]["functional_check"] == "errors == 0"
    assert [call["max_tokens"] for call in calls] == [
        initial_limit, max(initial_limit, min(initial_limit * 2, 65536))
    ]


@pytest.mark.parametrize("substantive_problem", [False, True])
def test_critic_repairs_own_citation_without_reproposing_or_losing_real_problem(
        monkeypatch, substantive_problem):
    problem = {"category": "component_soundness", "issue": "fallback contradicts gain",
               "suggestion": "repair the branch acceptance", "component": "guard"}
    first = {"problems": [problem] if substantive_problem else [], "quote": "wrong"}
    second = {**first, "quote": "exact"}
    llm = _RoleLLM(critic=[first, second])
    monkeypatch.setattr(core, "_component_contract_problem", lambda *a, **k: None)
    monkeypatch.setattr(core, "_repository_review_problems", lambda c, *a: []
        if c.get("quote") == "exact" else [{"response_protocol_error": True,
            "category": "feasibility", "issue": "citation mismatch", "suggestion": "copy quote"}])
    result = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=1, component_plan_audit=True,
        baseline_result={"source_kind": "live_repo", "metric_value": 1.0})
    assert llm.roles.count("critic") == 2
    assert llm.roles.count("proposer:A1") == llm.roles.count("proposer:A2") == 1
    assert result.report["accepted"] is not substantive_problem
    assert any(t.get("outcome") == "critic_protocol_repair" for t in result.report["transcript"])
    critic_calls = [call for call in llm.calls if "METHOD CRITIC" in call[0]["content"]]
    assert "one contiguous source substring" in critic_calls[0][0]["content"]
    assert "are JSON booleans (true/false)" in critic_calls[0][0]["content"]
    assert "never in parent_control_available" in critic_calls[0][0]["content"]
    assert "all bind to a valid frozen baseline has an available parent/control" in critic_calls[0][0]["content"]
    assert "not whether an ancestor component checkpoint exists" in critic_calls[0][0]["content"]
    assert "select one short exact line as the citation anchor" in critic_calls[0][0]["content"]
    assert "Read the full source" in critic_calls[0][0]["content"]
    assert "existing_mechanism describes the CURRENT repository mechanism" in critic_calls[0][0]["content"]
    assert "Its success remains unknown until measured" in critic_calls[0][0]["content"]
    assert "Do not join nonadjacent fields" in critic_calls[1][1]["content"]
    assert "preserve its valid input_bindings verbatim" in critic_calls[1][1]["content"]
    if substantive_problem:
        assert problem in result.report["transcript"][-1]["critic_deep_problems"]


def test_cap_failed_citation_repair_preserves_draft_without_redesign(monkeypatch):
    real_problem = {"category": "component_soundness", "issue": "genuine design issue"}
    llm = _RoleLLM(critic=[{"problems": [real_problem]}, {"problems": []}])
    protocol_error = {"response_protocol_error": True, "category": "feasibility",
                      "issue": "missing repository_checks"}
    monkeypatch.setattr(core, "_repository_review_problems", lambda *a: [protocol_error])
    with pytest.raises(core.DeliberationLLMError, match="Critic.*protocol failure") as caught:
        core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
            inner_rounds=1, outer_rounds=3, component_plan_audit=True)
    assert llm.roles.count("critic") == 3
    assert llm.roles.count("proposer:A1") == llm.roles.count("proposer:A2") == 1
    assert caught.value.draft_method["components"]
    last = caught.value.debate_transcript[-1]
    assert last["outcome"] == "critic_review_unavailable"
    assert last["protocol_errors"] == [protocol_error]
    assert real_problem in last["review"]["problems"]


def test_cap_critic_repairs_citation_after_truncated_first_response(monkeypatch):
    wrong = {"topic": "resource_scale", "source_path": "task.toml",
             "status": "supported", "quote": "gpu = 1"}
    fixed = {**wrong, "quote": "gpus = 1"}
    responses = iter(["{", json.dumps({"problems": [], "repository_checks": [wrong]}),
                      json.dumps({"repository_checks": [fixed]})])

    class RecoveringCritic(_RoleLLM):
        def chat(self, *, messages, **kwargs):
            if "METHOD CRITIC" in messages[0]["content"]:
                self.calls.append(messages)
                self.call_kwargs.append(kwargs)
                self.roles.append("critic")
                return _Resp(next(responses))
            return super().chat(messages=messages, **kwargs)

    error = {"response_protocol_error": True, "repository_topic": "resource_scale",
             "source_path": "task.toml", "verbatim_candidate": "gpus = 1",
             "category": "feasibility", "issue": "citation mismatch"}
    monkeypatch.setattr(core, "_component_contract_problem", lambda *a, **k: None)
    monkeypatch.setattr(core, "_repository_review_problems", lambda review, *a:
        [] if fixed in review.get("repository_checks", []) else [error])
    llm = RecoveringCritic()
    result = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=1, component_plan_audit=True,
        baseline_result={"source_kind": "live_repo", "metric_value": 1.0})
    assert result.report["accepted"] is True
    assert llm.roles.count("critic") == 3
    assert llm.roles.count("proposer:A1") == llm.roles.count("proposer:A2") == 1
    assert [kwargs["max_tokens"] for call, kwargs in zip(llm.calls, llm.call_kwargs)
            if "METHOD CRITIC" in call[0]["content"]] == [262144, 262144, 262144]


def test_partial_critic_protocol_repair_preserves_valid_checks_and_real_problems(monkeypatch):
    problem = {"category": "component_soundness", "component": "guard",
               "issue": "the treatment cannot establish its claimed effect", "suggestion": ""}
    valid = {"topic": "task_direction", "status": "contradicted", "quote": "minimize size"}
    bad = {"topic": "source_boundary", "status": "supported", "quote": "The source"}
    fixed = {**bad, "quote": "the source"}
    first = {"problems": [problem], "repository_checks": [valid, bad],
             "component_checks": [{"component_id": "guard", "kind": "intervention"}]}
    # Only one broken citation is returned; prior semantic findings must survive.
    llm = _RoleLLM(critic=[first, {"repository_checks": [fixed]}])
    monkeypatch.setattr(core, "_component_contract_problem", lambda *a, **k: None)
    error = {"response_protocol_error": True, "repository_topic": "source_boundary",
             "category": "feasibility", "issue": "citation mismatch"}
    def validate(review, *args):
        if fixed in review.get("repository_checks", []):
            assert valid in review["repository_checks"]
            assert review["component_checks"] == first["component_checks"]
            assert problem in review["problems"]
            return []
        return [error]
    monkeypatch.setattr(core, "_repository_review_problems", validate)
    result = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=1, component_plan_audit=True)
    assert llm.roles.count("critic") == 2
    assert llm.roles.count("proposer:A1") == llm.roles.count("proposer:A2") == 1
    assert result.report["accepted"] is False
    assert problem in result.report["transcript"][-1]["critic_deep_problems"]


@pytest.mark.parametrize("replacement", [[], [{"topic": "a"}, {"topic": "a"}]])
def test_critic_repair_cannot_clear_findings_or_overwrite_unreported_checks(replacement):
    previous = {"problems": [{"issue": "real"}],
                "repository_checks": [{"topic": "a", "quote": "bad"},
                                      {"topic": "b", "status": "contradicted"}]}
    repaired = core._merge_critic_protocol_repair(previous,
        {"problems": [], "repository_checks": replacement + [{"topic": "b", "status": "supported"}]},
        [{"repository_topic": "a"}])
    assert repaired == previous
    assert previous["repository_checks"][1]["status"] == "contradicted"


@pytest.mark.parametrize("status", ["supported", "contradicted"])
def test_critic_can_explicitly_select_exact_quote_hint_without_losing_findings(status):
    original = {"topic": "existing_mechanism", "source_path": "README.md",
                "quote": "Segment pool: shared regions", "status": status}
    source = "Segment pool**: shared regions"
    previous = {"problems": [{"issue": "component design still needs repair"}],
                "repository_checks": [original]}
    baseline = {"repository_contract": {"sources": [{"path": "README.md", "content": source}]}}
    errors = core._repository_review_problems(previous, baseline, {"components": []})
    replacement = {k: v for k, v in original.items() if k != "quote"}
    replacement["use_verbatim_candidate"] = True
    merged = core._merge_critic_protocol_repair(previous,
        {"repository_checks": [replacement], "problems": []}, errors)
    assert merged["repository_checks"][0]["quote"] == source
    assert merged["repository_checks"][0]["status"] == status
    assert merged["problems"] == previous["problems"]
    checked = core._repository_review_problems(merged, baseline, {"components": []})
    assert not any(p.get("repository_topic") == "existing_mechanism" for p in checked)
    if status == "contradicted":
        assert any(p.get("grounded") and p.get("status") == status for p in checked)


@pytest.mark.parametrize("override", [
    {"use_verbatim_candidate": "true"}, {"source_path": "other.md"},
    {"quote": "different claimed quotation"},
])
def test_critic_quote_hint_requires_explicit_selection_same_source_and_no_conflicting_quote(override):
    original = {"topic": "existing_mechanism", "source_path": "README.md", "quote": "bad"}
    replacement = {"topic": "existing_mechanism", "source_path": "README.md",
                   "use_verbatim_candidate": True, **override}
    merged = core._merge_critic_protocol_repair(
        {"repository_checks": [original]}, {"repository_checks": [replacement]},
        [{"repository_topic": "existing_mechanism", "source_path": "README.md",
          "verbatim_candidate": "exact source"}])
    assert merged["repository_checks"][0].get("quote") != "exact source"


@pytest.mark.parametrize("indices", [[1, 0], [1], [-1, 0], [2, 0], [True, 0], [1.0, 0]])
def test_critic_indexed_binding_repair_preserves_exact_fields_and_rejects_bad_indices(indices):
    inputs = ["{'key': 'x', 'description': '\"quoted\" input'}", "config: current"]
    sources = ["baseline", "parent:features: complete output description"]
    original = {"component_id": "child", "kind": "intervention",
                "independently_executable": True, "parent_control_available": True,
                "input_bindings": {"wrong spelling": "baseline"}}
    previous = {"problems": [{"issue": "still needs a real control"}],
                "component_checks": [original]}
    replacement = {**original, "input_source_indices": indices}
    errors = [{"component": "child", "response_protocol_error": True,
               "expected_input_keys": inputs, "allowed_sources": sources}]
    merged = core._merge_critic_protocol_repair(previous,
        {"component_checks": [replacement], "problems": []}, errors)
    actual = merged["component_checks"][0]
    if indices == [1, 0] and all(type(i) is int for i in indices):
        assert actual["input_bindings"] == dict(zip(inputs, [sources[1], sources[0]]))
        assert "input_source_indices" not in actual
    else:
        assert actual["input_bindings"] == original["input_bindings"]
    assert merged["problems"] == previous["problems"]
    assert original["input_bindings"] == {"wrong spelling": "baseline"}


@pytest.mark.parametrize("inputs,bindings,valid", [
    ([], {}, True),
    ([], None, False),
    ([], {"extra": "baseline"}, False),
    (["x"], {}, False),
    (["x"], {"x": "baseline"}, True),
])
def test_component_input_bindings_match_declared_inputs_including_empty(inputs, bindings, valid):
    method = {"components": [{"component_id": "source", "type": "indirect_gain",
                              "inputs": inputs, "outputs": ["fixture"]}]}
    critic = {"component_checks": [{"component_id": "source", "kind": "measurement",
        "input_bindings": bindings, "parent_control_available": True,
        "independently_executable": True}]}
    problems = core._repository_review_problems(critic, {"source_kind": "live_repo"}, method)
    errors = [p for p in problems if p.get("component") == "source"]
    assert bool(errors) is not valid


@pytest.mark.parametrize("field", ["independently_executable", "parent_control_available"])
@pytest.mark.parametrize("judgment,protocol_error", [(None, True), ("true", True), ("baseline", True), (False, False)])
def test_missing_or_malformed_component_judgment_is_not_a_negative_judgment(field, judgment, protocol_error):
    method = {"components": [{"component_id": "c", "type": "direct_gain",
                               "inputs": ["x"], "outputs": ["y"]}]}
    critic = {"component_checks": [{"component_id": "c", "kind": "intervention",
        "input_bindings": {"x": "baseline"}, "parent_control_available": True,
        "independently_executable": True, field: judgment}]}
    problems = core._repository_review_problems(critic, {"source_kind": "live_repo"}, method)
    node_problems = [p for p in problems if p.get("component") == "c"]
    assert len(node_problems) == 1
    assert node_problems[0]["response_protocol_error"] is protocol_error
    if protocol_error:
        assert "JSON booleans" in node_problems[0]["suggestion"]


@pytest.mark.parametrize("source,supplied,changed,grounded", [
    ("evaluation_semantics", True, False, True),
    ("baseline_result", True, False, False),
    ("evaluation_semantics", False, False, False),
    ("evaluation_semantics", True, True, False),
])
def test_frozen_semantics_citation_is_bound_to_its_actual_source(source, supplied, changed, grounded):
    semantics = {"primary_metric": {"name": "model_size", "direction": "minimize"}}
    quote = json.dumps(semantics, ensure_ascii=False, sort_keys=True)
    if changed:
        quote = quote.replace("minimize", "maximize")
    critic = {"repository_checks": [{"topic": "task_direction", "status": "supported",
                                    "source_path": source, "quote": quote}]}
    problems = core._repository_review_problems(critic, {"source_kind": "live_repo"},
        {"components": []}, semantics if supplied else None)
    errors = [p for p in problems if "task_direction" in p.get("issue", "")]
    assert bool(errors) is not grounded


def test_initial_draft_error_is_not_repeated_as_current_evidence_after_proposer_repair(monkeypatch):
    checks = iter([ValueError("old_gate_name is invalid"), None])
    monkeypatch.setattr(core, "_component_contract_problem", lambda *a, **kw: next(checks))
    llm = _RoleLLM()
    result = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=1, component_plan_audit=True)
    proposals = [c[1]["content"] for c in llm.calls if "METHOD PROPOSER" in c[0]["content"]]
    reviews = [c[1]["content"] for c in llm.calls if "METHOD CRITIC" in c[0]["content"]]
    assert all("old_gate_name is invalid" in prompt for prompt in proposals)
    assert all("old_gate_name is invalid" not in prompt for prompt in reviews)
    assert result.report["accepted"] is True


def test_terminal_measurement_does_not_invent_a_new_direct_gain_requirement():
    assert "a terminal evaluator/join that only measures" in core._COMPONENT_PLAN_PROPOSER_SYSTEM
    assert "terminal measurement/join remains indirect_gain" in core._COMPONENT_PLAN_CRITIC_SYSTEM


def test_component_planning_checks_control_feasibility_before_scalar_gain():
    for prompt in (core._COMPONENT_PLAN_PROPOSER_SYSTEM, core._COMPONENT_PLAN_CRITIC_SYSTEM):
        assert "infeasible parent/control" in prompt
        assert "Do not change the official objective or fabricate a feasible baseline" in prompt
    assert "test the enabling property" in core._COMPONENT_PLAN_PROPOSER_SYSTEM
    assert "Reject an unattainable direct-gain predicate" in core._COMPONENT_PLAN_CRITIC_SYSTEM


@pytest.mark.parametrize("executable,role,expected", [
    (True, "direct_gain", "measurement-only component is direct_gain"),
    (False, "indirect_gain", "independently_executable=false"),
])
def test_component_review_names_the_actual_failed_condition(executable, role, expected):
    method = {"components": [{"component_id": "join", "inputs": ["data"], "type": role}]}
    critic = {"component_checks": [{"component_id": "join", "kind": "measurement",
        "independently_executable": executable, "parent_control_available": True,
        "input_bindings": {"data": "baseline"}}]}
    errors = core._repository_review_problems(critic, {"source_kind": "live_repo"}, method)
    error = next(p for p in errors if p.get("component") == "join")
    assert expected in error["issue"]
    assert "WITH its declared dependency closure" in error["suggestion"]
    if role == "direct_gain":
        assert "official evaluator or correctness gate on unchanged code" in error["suggestion"]
        assert "Otherwise classify a genuine measurement-only component as indirect_gain" in error["suggestion"]
    assert error["response_protocol_error"] is False


def test_cap_moss_summary_cannot_erase_actual_contract_error():
    llm = _RoleLLM()
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=2, component_plan_audit=True)
    proposals = [c for c in llm.calls if "METHOD PROPOSER" in c[0]["content"]]
    assert "Authoritative current review errors:" in proposals[2][1]["content"]
    assert "ComponentPlan cannot execute" in proposals[2][1]["content"]


def test_citation_repair_hint_returns_actual_source_without_accepting_similar_quote():
    source = "The measured model has 42 parameters; evaluation uses held-out inputs."
    critic = {"repository_checks": [{"topic": "resource_scale", "status": "supported",
        "source_path": "README.md", "quote": source.replace("42", "24")}]}
    baseline = {"repository_contract": {"sources": [{"path": "README.md", "content": source}]}}
    problems = core._repository_review_problems(critic, baseline, {"components": []})
    error = next(p for p in problems if "resource_scale" in p["issue"])
    assert error["grounded"] is False and error["response_protocol_error"] is True
    assert error["evidence_excerpt"] == source
    assert "24 parameters" not in error["evidence_excerpt"]


def test_repository_citation_restores_omitted_line_without_changing_judgment():
    source = ('[environment]\ncpus = 2\nmemory_mb = 2048\nstorage_mb = 1024\n'
              'gpus = 0\nallow_internet = false\n\n[optimization]\n')
    quote = 'cpus = 2\nmemory_mb = 2048\ngpus = 0\nallow_internet = false'
    check = {"topic": "resource_scale", "status": "supported",
             "source_path": "task.toml", "quote": quote}
    substantive = {"issue": "The component has no runnable acceptance test"}
    critic = {"repository_checks": [check], "problems": [substantive]}
    baseline = {"repository_contract": {"sources": [{"path": "task.toml", "content": source}]}}
    errors = core._repository_review_problems(critic, baseline, {"components": []})
    assert not any(p.get("repository_topic") == "resource_scale" for p in errors)
    assert check["quote_before_line_restoration"] == quote
    assert check["quote"] == quote.replace('gpus = 0', 'storage_mb = 1024\ngpus = 0')
    assert check["quote"] in source
    assert critic["problems"] == [substantive]
    check["status"] = "contradicted"
    errors = core._repository_review_problems(critic, baseline, {"components": []})
    assert any(p.get("grounded") and p.get("status") == "contradicted" for p in errors)
    assert check["quote_before_line_restoration"] == quote


@pytest.mark.parametrize("source,quote", [
    ('a = 1\nb = 2\nc = 3', 'a = 1\nc = 4'),  # invented value
    ('a = 1\nb = 2\nc = 3', 'c = 3\na = 1'),  # reordered lines
    ('a = 1\nb = 2\nc = 3', 'a=1\nc = 3'),  # not exact source bytes
    ('a = 1\nb = 2\nc = 3\na = 1', 'a = 1\nc = 3'),  # ambiguous anchor
    ('a = 1\n\nb = 2\nc = 3', 'a = 1\nc = 3'),  # different paragraphs
    ('a = 1\n' + 'b' * 513 + '\nc = 3', 'a = 1\nc = 3'),  # unbounded context
    (json.dumps({'one': 'a = 1', 'two': 'c = 3'}), 'a = 1\nc = 3'),
])
def test_citation_line_restoration_rejects_unverifiable_or_ambiguous_excerpts(source, quote):
    assert core._restore_omitted_quote_lines(source, quote) is None


def test_citation_line_restoration_preserves_intervening_limitations():
    source = 'Use the implementation.\nDo not modify its evaluator.\nRecord timing.'
    quote = 'Use the implementation.\nRecord timing.'
    assert core._restore_omitted_quote_lines(source, quote) == source
    assert core._restore_omitted_quote_lines(json.dumps({'notes': source}), quote) == source


def test_citation_repair_names_missing_quote_key_instead_of_only_mismatch():
    critic = {"repository_checks": [{"topic": "resource_scale", "status": "supported",
        "source_path": "README.md", "verbatim_quote": "four CPU cores"}]}
    baseline = {"repository_contract": {"sources": [
        {"path": "README.md", "content": "four CPU cores"}]}}
    errors = core._repository_review_problems(critic, baseline, {"components": []})
    error = next(p for p in errors if p.get("repository_topic") == "resource_scale")
    assert "missing nonempty 'quote' field" in error["issue"]
    assert "literal JSON key 'quote'" in error["suggestion"]
    assert error["grounded"] is False
    critic["repository_checks"][0]["quote"] = "four CPU cores"
    errors = core._repository_review_problems(critic, baseline, {"components": []})
    assert not any(p.get("repository_topic") == "resource_scale" for p in errors)


def test_citation_hint_preserves_source_delimiters_without_accepting_reformatted_quote():
    source = "## Implementation\n- **Segment pool**: Regions released after use\n## Rules\n"
    critic = {"repository_checks": [{"topic": "existing_mechanism", "status": "supported",
        "source_path": "README.md", "quote": "Segment pool: Regions released after use"}]}
    baseline = {"repository_contract": {"sources": [{"path": "README.md", "content": source}]}}
    error = next(p for p in core._repository_review_problems(critic, baseline, {"components": []})
                 if p.get("repository_topic") == "existing_mechanism")
    assert error["grounded"] is False
    assert error["verbatim_candidate"] == "Segment pool**: Regions released after use"
    assert error["verbatim_candidate"] in source


def test_duplicate_repository_checks_are_reported_as_duplicates_not_missing_quotes():
    check = {"topic": "resource_scale", "status": "supported",
             "source_path": "README.md", "quote": "four cores"}
    baseline = {"repository_contract": {"sources": [{"path": "README.md", "content": "four cores"}]}}
    error = next(p for p in core._repository_review_problems(
        {"repository_checks": [check, check]}, baseline, {"components": []})
        if p.get("repository_topic") == "resource_scale")
    assert "2 entries" in error["issue"]
    assert "exactly one" in error["issue"]
    assert "missing nonempty" not in error["issue"]


@pytest.mark.parametrize("status,quote", [
    ("supported", "four cores"),
    ("contradicted", "four cores"),
    ("requires_measurement", "four cores"),
    ("supported", "invented capability"),
])
def test_repository_check_mapping_keeps_same_grounding_and_judgments(status, quote):
    check = {"topic": "resource_scale", "status": status,
             "source_path": "README.md", "quote": quote}
    baseline = {"repository_contract": {"sources": [{"path": "README.md", "content": "four cores"}]}}
    method = {"components": []}
    expected = core._repository_review_problems({"repository_checks": [dict(check)]}, baseline, method)
    actual = core._repository_review_problems({"repository_checks": {"resource_scale": dict(check)}}, baseline, method)
    assert actual == expected


def test_mapping_protocol_repair_preserves_substantive_problems_and_other_checks():
    valid = {"topic": "artifact_scope", "status": "contradicted", "quote": "read-only"}
    bad = {"topic": "resource_scale", "status": "supported", "quote": "wrong"}
    fixed = {"topic": "resource_scale", "status": "requires_measurement", "quote": "four cores"}
    problem = {"issue": "core method violates the read-only boundary"}
    previous = {"problems": [problem], "repository_checks": {"artifact_scope": valid, "resource_scale": bad}}
    merged = core._merge_critic_protocol_repair(previous,
        {"repository_checks": {"resource_scale": fixed}}, [{"repository_topic": "resource_scale"}])
    assert merged["problems"] == [problem]
    assert merged["repository_checks"] == [valid, fixed]
    assert previous["repository_checks"]["resource_scale"] == bad


@pytest.mark.parametrize("quote,accepted", [
    ('You may modify:\n- optimizer\nDo not change clients.', True),
    ('You may modify:\n- optimizer\nDo change clients.', False),
    ('You may modify:\n- optimizer\nDo not change clients. Second field.', False),
])
def test_repository_quote_matches_decoded_json_value_without_changing_claim(quote, accepted):
    source = json.dumps({"prompt": "You may modify:\n- optimizer\nDo not change clients.",
                         "other": "Second field."})
    baseline = {"repository_contract": {"sources": [{"path": "prompt.json", "content": source}]}}
    critic = {"repository_checks": [{"topic": "source_boundary", "status": "supported",
                                     "source_path": "prompt.json", "quote": quote}]}
    errors = core._repository_review_problems(critic, baseline, {"components": []})
    assert any(p.get("repository_topic") == "source_boundary" for p in errors) is not accepted
    assert not core._source_contains_quote('not JSON: "a\\nb"', 'a\nb')


def test_binding_repair_receives_lossless_keys_and_only_real_ancestor_outputs():
    key = "state called 'x': provide it (do not append comments)"
    method = {"components": [
        {"component_id": "parent", "inputs": ["data"], "outputs": ["state: full description"]},
        {"component_id": "child", "depends_on": ["parent"], "inputs": [key]},
        {"component_id": "unrelated", "outputs": ["not a parent"]}]}
    errors = core._repository_review_problems({}, {"source_kind": "live_repo"}, method)
    error = next(p for p in errors if p.get("component") == "child" and "expected_input_keys" in p)
    assert error["expected_input_keys"] == [key]
    assert error["allowed_sources"] == ["baseline", "parent:state: full description"]


def test_citation_hint_locates_short_quote_inside_long_source():
    sentence = "the measured baseline uses the repository implementation and stores weights."
    source = ("repository configuration and measurement evidence. " * 100 + sentence
              + " additional repository configuration and measurement evidence." * 100)
    critic = {"repository_checks": [{"topic": "source_boundary", "status": "supported",
        "source_path": "README.md", "quote": sentence.replace("the measured", "The measured")}]}
    baseline = {"repository_contract": {"sources": [{"path": "README.md", "content": source}]}}
    error = next(p for p in core._repository_review_problems(critic, baseline, {"components": []})
                 if "source_boundary" in p["issue"])
    assert error["grounded"] is False
    assert sentence in error["evidence_excerpt"]
    assert len(error["evidence_excerpt"]) <= 600
    assert error["verbatim_candidate"] == sentence
    assert error["verbatim_candidate"] in source


def test_citation_repair_hint_uses_exact_decoded_json_string():
    sentence = 'the source files are "src/kernel.cu" and "src/check.cu".\nKeep the evaluator unchanged.'
    source = json.dumps({"notes": sentence, "other": "unrelated evidence"})
    baseline = {"repository_contract": {"sources": [{"path": "inspection.json", "content": source}]}}
    critic = {"repository_checks": [{"topic": "source_boundary", "status": "supported",
        "source_path": "inspection.json", "quote": sentence.replace("the source", "The source")}]}
    error = next(p for p in core._repository_review_problems(critic, baseline, {"components": []})
                 if p.get("repository_topic") == "source_boundary")
    assert error["grounded"] is False
    assert error["verbatim_candidate"] == sentence
    assert core._source_contains_quote(source, error["verbatim_candidate"])


def test_citation_hint_prefers_field_over_serialized_envelope_on_equal_match():
    log = 'INSPECTION: ' + json.dumps({"files": ["src/first_kernel.cu", "src/second_check.cu"]}).replace('"', '\\"')
    baseline = {"source_kind": "live_repo", "log_excerpt": log}
    critic = {"repository_checks": [{"topic": "source_boundary", "status": "supported",
        "source_path": "baseline_result", "quote": 'src/first_kernel.cu", "src/second_check.cu'}]}
    error = next(p for p in core._repository_review_problems(critic, baseline, {"components": []})
                 if p.get("repository_topic") == "source_boundary")
    assert error["grounded"] is False
    assert error["verbatim_candidate"] in log
    assert error["evidence_excerpt"] == log


def test_cap_critic_uses_same_acceptance_fields_and_scope_as_proposer():
    prompt = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "signed_metric_improvement" in prompt and "candidate_metric" in prompt
    assert "additional ablation_test " in prompt and "is not required" in prompt
    assert "implementation-time inspection and failure condition" in prompt
    assert "method_revision_required rather than asking CAP proposers" in prompt


def test_cap_reconciliation_cannot_ignore_legal_measured_mechanism():
    critic = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "reconciled_result_ref" in critic
    assert "host-bound diff" in critic
    assert "Reject an unchanged plan" in critic
    context = core._method_ctx(
        _PM, _LIT, None, component_plan_audit=True,
        component_plan_revision_feedback=(
            "admissible improved off-method observation retained"),
    )
    assert "reconcile the owning component with the measured mechanism" in context
    assert "incompatible old mechanism" in context
    assert "formal Method revision" in context


def test_cap_plan_distinguishes_raw_candidate_from_host_computed_gain():
    for prompt in (core._COMPONENT_PLAN_PROPOSER_SYSTEM, core._COMPONENT_PLAN_CRITIC_SYSTEM):
        assert "raw candidate_metric" in prompt
        assert "separately measured parent_metric" in prompt
        assert "host computes signed_metric_improvement" in prompt
        assert "Do not require a delta in the raw candidate measurement" in prompt


def test_cap_does_not_turn_aspirational_effect_size_into_hard_gate():
    proposer = core._COMPONENT_PLAN_PROPOSER_SYSTEM
    critic = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "signed_metric_improvement > 0 as the ordinary" in proposer
    assert "aspirational target" in proposer
    assert "authoritative repository task contract itself" in proposer
    assert "Ordinarily require signed_metric_improvement > 0" in critic
    assert "reference/leaderboard anchor" in critic
    assert "Do not discard a legal measured improvement" in critic
    assert "Do not move the same aspirational magnitude into implementation_check" in proposer
    assert "smuggled into implementation_check" in critic
    assert "not a non-mandatory performance target" in critic


def test_cap_review_keeps_correctness_gates_outside_scalar_predicate():
    for prompt in (core._COMPONENT_PLAN_PROPOSER_SYSTEM, core._COMPONENT_PLAN_CRITIC_SYSTEM):
        assert "Do not conjoin correctness gates into ablation_test" in prompt
        assert "implementation_check and runtime gate evidence" in prompt
        assert "a scalar gain predicate never waives them" in prompt
        assert "An earlier repair checklist is not authority" in prompt
    assert "not a conjunction with and/or" in core._COMPONENT_PLAN_CRITIC_SYSTEM


def test_component_contract_harness_rejects_prose_threshold_then_accepts_numeric_retry():
    llm = _FunctionalContractLLM()

    res = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=2)

    assert res.report["accepted"] is True
    assert res.report["outer_rounds"] == 2
    first = res.report["transcript"][1]
    assert first["critic_deep_problems"][0]["category"] == "ablation_verifiability"
    assert "numeric threshold" in first["critic_deep_problems"][0]["issue"]
    assert res.method["components"][0]["functional_check"] == "mismatches == 0"


def test_component_contract_prompt_requires_machine_evaluable_numeric_comparison():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1)

    proposer_systems = [call[0]["content"] for call in llm.calls
                        if "METHOD PROPOSER" in call[0].get("content", "")]
    assert proposer_systems
    assert all("mismatch_count == 0" in prompt for prompt in proposer_systems)
    assert all("below the repository threshold" in prompt for prompt in proposer_systems)


def test_component_plan_critic_understands_runtime_measurement_binding():
    prompt = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "measured_error <= 0.01" in prompt
    assert "implementation_check may define the measurement" in prompt
    assert "binds the runtime ablation_report actual" in prompt
    assert "does not execute functional_check as Python" in prompt
    assert "numeric right-hand threshold" in prompt


def test_component_plan_prompts_keep_conditional_acceptance_consistent():
    proposer = core._COMPONENT_PLAN_PROPOSER_SYSTEM
    critic = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "branch-aware failed_check_count == 0" in proposer
    assert "Runner cannot replace the frozen threshold" in proposer
    assert "compare the frozen functional_check with every branch" in critic
    assert "purpose, mechanism, expected_effect, and implementation_check" in critic
    assert "actual upstream observation" in critic
    assert "another valid branch is promised elsewhere" in critic


def test_component_plan_leaves_result_envelope_to_runner():
    prompt = core._COMPONENT_PLAN_PROPOSER_SYSTEM
    assert "Runner-owned result envelope" in prompt
    assert "separate evidence artifacts" in prompt
    assert "do not redefine ar_result.json" in prompt
    assert "omit those as ComponentPlan nodes" in prompt
    assert "Omitting workflow nodes alone does not revise the scientific Method" in prompt
    assert "not candidate components or Method changes" in core._METHOD_PROPOSER_SYSTEM


def test_component_plan_keeps_check_only_gate_in_acceptance():
    proposer = core._COMPONENT_PLAN_PROPOSER_SYSTEM
    critic = core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "official correctness on an unchanged candidate" in proposer
    assert "merged_from for the removed ID" in proposer
    assert "official correctness on an unchanged candidate" in critic
    assert "Do not waive the check" in critic
    assert "not a second run of the task's existing correctness gate" in proposer
    assert "only reruns the task's existing correctness gate" in critic


def test_component_plan_critic_relays_workflow_node_removal_without_method_swap():
    issue = "A node only registers a baseline host attempt; it is not a Method component."
    llm = _RoleLLM(
        conv=[{"converged": True}, {"converged": True}],
        critic=[{"problems": [{
            "category": "component_soundness", "component": "gate",
            "issue": issue,
            "suggestion": "Remove the workflow node and bind consumers to baseline.",
        }]}, {"problems": []}],
    )
    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=2,
        component_plan_audit=True,
    )
    critics = [call for call in llm.calls
               if "METHOD CRITIC" in call[0].get("content", "")]
    assert "Runner-owned workflow operations" in critics[0][0]["content"]
    assert "do not invent a source edit" in critics[0][0]["content"]
    assert "Preserve genuine" in critics[0][0]["content"]
    assert issue in critics[1][1]["content"]
    assert result.report["outer_rounds"] == 2
    # This fixture's proposers still return a prose-only acceptance contract;
    # relaying the critique must not bypass the existing contract checks.
    assert result.report["accepted"] is False
    assert not result.report.get("method_revision_required")


def test_method_debate_and_critic_receive_authoritative_comparison_semantics():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    semantics = {
        "comparison_statement": (
            "Give baseline and method 300 seconds pure training; compare final val_bpb."),
        "comparison_mode": "quality_at_fixed_time",
        "budget_interpretation": "steps are observed outcomes, not controls",
    }

    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        evaluation_semantics=semantics)

    prompts = ["\n".join(message.get("content", "") for message in call)
               for call in llm.calls]
    assert all("FROZEN COMPARISON SEMANTICS" in prompt for prompt in prompts)
    assert all("quality_at_fixed_time" in prompt for prompt in prompts)
    proposer_systems = [call[0]["content"] for call in llm.calls
                        if "METHOD PROPOSER" in call[0].get("content", "")]
    assert all("never promote them into components" in prompt
               for prompt in proposer_systems)
    assert all("no authoritative frozen BaselineResult" in prompt
               for prompt in proposer_systems)
    assert all("literature context does not prove" in prompt
               for prompt in proposer_systems)
    assert "comparison_integrity" in core.METHOD_CRITIC_RUBRIC
    assert result.report["rubric"][0] == "comparison_integrity"


def test_repository_defined_method_context_prioritizes_task_edit_rules():
    context = core._method_ctx(
        _PM, _LIT, None,
        baseline_result={"source_kind": "live_repo", "repository_contract": {
            "sources": [{"path": "task/prompt.json",
                         "content": "You may change the clipping setting."}]}},
        evaluation_semantics={
            "comparison_mode": "repository_defined",
            "conditions_to_hold_constant": ["clipping setting fixed"],
        },
    )
    assert "You may change the clipping setting" in context
    assert "repository_contract task rules override inferred conditions_to_hold_constant" in context
    assert "do not reject a Method on that basis alone" in context


def test_method_debate_all_roles_receive_prior_failure_diagnosis():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        prior_failure_context={"action": "discard_hypothesis",
                               "rationale": "Prior measured mechanism failed under this workload."})
    prompts = ["\n".join(message.get("content", "") for message in call)
               for call in llm.calls]
    assert prompts and all("Prior measured mechanism failed" in prompt for prompt in prompts)
    assert all("identify new task evidence" in prompt for prompt in prompts)


def test_method_debate_improvement_review_does_not_assert_prior_mechanism_failed():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        prior_failure_context={"method_revision_evidence": {
            "trigger": "complete_result",
            "result_ref": "spec://gen_experiment_result/session/snapshot-result",
            "retain": ["Verified tiled kernel"],
            "modify": ["Try a wider vector path with fallback"],
            "candidate": {},
            "qualified_alternative": {"measurement_valid": None},
        }},
    )
    prompts = ["\n".join(message.get("content", "") for message in call)
               for call in llm.calls]
    assert prompts and all("Verified tiled kernel" in prompt for prompt in prompts)
    assert all("An improvement review or revision request does not establish" in prompt
               for prompt in prompts)
    assert all("Preserve supported mechanisms" in prompt for prompt in prompts)
    assert all("Only when cited evidence establishes a mechanism's failure" in prompt
               for prompt in prompts)
    assert all("A null or missing metric is missing measurement evidence, not zero gain" in prompt
               for prompt in prompts)
    assert all("or proof that hardware lacks a capability" in prompt for prompt in prompts)


def test_component_plan_audit_focuses_existing_four_role_deliberation_on_contracts():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])

    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        baseline_result={"metric_name": "val_loss", "metric_value": 1.0},
        component_plan_audit=True,
    )

    assert result.method["approach"] == _DRAFT["approach"]
    assert result.method["changes_outline"] == _DRAFT["changes_outline"]
    assert result.method["why_it_should_win"] == _DRAFT["why_it_should_win"]
    assert result.method["assumptions"] == _DRAFT["assumptions"]

    prompts = ["\n".join(message.get("content", "") for message in call)
               for call in llm.calls]
    assert all("COMPONENT-PLAN AUDIT" in prompt for prompt in prompts)
    assert all("testable WITH its declared transitive prerequisites" in prompt for prompt in prompts)
    assert all("Consuming an accepted ancestor output does not make this flag false" in prompt
               for prompt in prompts)
    assert all("orthogonality_error <= 0.01" in prompt for prompt in prompts)
    proposer_prompts = [call[0]["content"] for call in llm.calls
                        if "METHOD PROPOSER" in call[0].get("content", "")]
    assert len(proposer_prompts) == 2
    assert all("in COMPONENT-PLAN scope" in prompt for prompt in proposer_prompts)
    assert all("preserve its approach" in prompt for prompt in proposer_prompts)
    assert all("Every component must implement or enable that SAME approach" in prompt
               for prompt in proposer_prompts)
    assert all("do not silently substitute a different algorithm" in prompt
               for prompt in proposer_prompts)
    assert all("Do not add optional machinery" in prompt for prompt in proposer_prompts)
    assert all("audit every component's acceptance environment" in prompt
               for prompt in proposer_prompts)
    assert all("Independence is semantic" in prompt for prompt in proposer_prompts)
    assert all("must not branch from an earlier shared ancestor" in prompt
               for prompt in proposer_prompts)
    assert all("may activate only the current component plus acceptance_requires" in prompt
               for prompt in proposer_prompts)
    assert all("distinguish measurement validity from the hypothesis being measured" in prompt
               for prompt in proposer_prompts)
    assert all("not an instrumentation failure to debug until the observation changes" in prompt
               for prompt in proposer_prompts)
    assert all("must not weaken its acceptance test or add diagnostic components" in prompt
               for prompt in proposer_prompts)
    assert all("derive the expected output for the actual test input" in prompt
               for prompt in proposer_prompts)
    converge_prompts = [call[0]["content"] for call in llm.calls
                        if "You judge whether" in call[0].get("content", "")]
    assert len(converge_prompts) == 1
    assert "SAME ComponentPlan, not merely the same scientific Method" in converge_prompts[0]
    assert "parent acceptance environment" in converge_prompts[0]
    critic_prompts = [call[0]["content"] for call in llm.calls
                      if "METHOD CRITIC" in call[0].get("content", "")]
    assert len(critic_prompts) == 1
    assert "contain any DEEP problem" in critic_prompts[0]
    assert "whether a problem could later be located" in critic_prompts[0]
    assert "Reject a vague acceptance test" in critic_prompts[0]
    assert "confuses measurement validity with confirmation of the measured hypothesis" in critic_prompts[0]
    assert "independently derive the expected output for the actual test input" in critic_prompts[0]
    assert "A correct validator must reject an invalid object" in critic_prompts[0]
    assert "CREATE an intermediate property must still demonstrate that property" in critic_prompts[0]
    assert "remaining optimization space" in critic_prompts[0]
    assert "downstream contract" in critic_prompts[0]
    assert "terminal branches together implement the whole confirmed Method" in critic_prompts[0]
    assert "FIRST gate is method-to-component coherence" in critic_prompts[0]
    assert "components implement a different one" in critic_prompts[0]
    assert "omit hypothetical risks and optional improvements" in critic_prompts[0]
    assert "that intermediate-property check is sufficient" in critic_prompts[0]
    assert "do NOT demand an add-vs-remove performance ablation" in critic_prompts[0]
    assert "immediate target-metric gain" in critic_prompts[0]
    assert "re-check every unresolved blocking repair" in critic_prompts[0]
    assert "tested from the baseline parent environment" in critic_prompts[0]
    assert "at most three concrete problems" in critic_prompts[0]
    assert "later ExperimentPlan and Runner choose exact shell commands" in critic_prompts[0]
    assert "Do not require exact CLI strings" in critic_prompts[0]
    assert "repeated seeds" in critic_prompts[0]
    assert "One real repository-owned official metric comparison" in critic_prompts[0]
    assert "parameter, code-axis, input/output, or implementation-check wording" in (
        critic_prompts[0]
    )
    assert "measures, selects, guards, configures, tunes" in critic_prompts[0]
    assert "built and tested on the correct checkpoint" in critic_prompts[0]
    proposer_limits = [
        kwargs.get("max_tokens")
        for call, kwargs in zip(llm.calls, llm.call_kwargs)
        if "METHOD PROPOSER" in call[0].get("content", "")
    ]
    assert proposer_limits == [262144, 262144]


def test_component_plan_critic_feedback_uses_component_scope_moss_relay():
    llm = _RoleLLM(
        conv=[{"converged": True}, {"converged": True}],
        critic=[
            {"problems": [{
                "category": "ablation_verifiability",
                "issue": "component check is not executable",
                "suggestion": "name the numeric check",
                "component": "gate",
            }]},
            {"problems": []},
        ],
    )

    core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=2,
        component_plan_audit=True,
    )

    relay_prompts = [call[0]["content"] for call in llm.calls
                     if "coordinator for COMPONENT-PLAN" in call[0].get("content", "")]
    assert relay_prompts
    assert all("group duplicate findings by component" in prompt
               for prompt in relay_prompts)
    assert all("Do not overrule executable schema/parser errors" in prompt
               for prompt in relay_prompts)
    assert all("with its declared ancestors present" in prompt
               for prompt in relay_prompts)
    convergence_inputs = [call[1]["content"] for call in llm.calls
                          if "SAME ComponentPlan" in call[0].get("content", "")]
    assert "UNRESOLVED COMPONENT-PLAN FEEDBACK" not in convergence_inputs[0]
    assert "UNRESOLVED COMPONENT-PLAN FEEDBACK" not in convergence_inputs[1]
    critic_inputs = [call[1]["content"] for call in llm.calls
                     if "METHOD CRITIC" in call[0].get("content", "")]
    assert "UNRESOLVED COMPONENT-PLAN FEEDBACK" not in critic_inputs[0]
    assert "UNRESOLVED COMPONENT-PLAN FEEDBACK" in critic_inputs[1]

    converge_prompts = [call[0]["content"] for call in llm.calls
                        if "SAME ComponentPlan" in call[0].get("content", "")]
    assert all("the dedicated Critic, not you" in prompt for prompt in converge_prompts)


def test_component_plan_critic_cannot_reject_a_field_value_the_plan_does_not_have():
    method = {"components": [{
        "component_id": "gate",
        "depends_on": ["profile", "implementation"],
        "acceptance_requires": ["profile", "implementation"],
    }]}
    stale = {"component": "gate", "category": "component_soundness",
             "issue": 'gate.depends_on is ["implementation"] but profile is required'}
    current = {"component": "gate", "category": "component_soundness",
               "issue": 'gate.depends_on is ["profile", "implementation"] but the test is invalid'}
    assert core._critic_field_claim_conflicts(stale, method) is True
    assert core._critic_field_claim_conflicts(current, method) is False


def test_component_plan_critic_ignores_empty_optional_check_format_nit():
    method = {"components": [
        {"component_id": "guard", "type": "indirect_gain",
         "functional_check": "failed_check_count == 0", "ablation_test": ""},
        {"component_id": "gain", "type": "direct_gain",
         "ablation_test": "signed_metric_improvement > 0", "functional_check": ""},
    ]}
    for component_id, field in (("guard", "ablation_test"),
                                ("gain", "functional_check")):
        problem = {"component": component_id, "category": "component_soundness",
                   "issue": (f'{component_id}.{field} is still the empty string "" '
                             "instead of JSON null; this is ambiguous for host parsing")}
        assert core._critic_field_claim_conflicts(problem, method) is True
    method["components"][0]["functional_check"] = ""
    assert core._critic_field_claim_conflicts({
        "component": "guard", "issue": 'guard.ablation_test is the empty string "" instead of JSON null'
    }, method) is False


def test_component_plan_critic_cannot_invent_a_token_inside_an_input_description():
    actual_input = "candidate_metric (baseline — Runner-parsed evaluator value)"
    method = {"components": [{"component_id": "gain", "inputs": [actual_input]}]}
    false_claim = {"component": "gain", "category": "component_soundness",
                   "issue": "gain.inputs[0] has binding source token 'runner:candidate_metric' "
                            "in the input string, so its binding is invalid"}
    real_claim = {**false_claim,
                  "issue": "gain.inputs[0] has binding source token 'candidate_metric' "
                           "in the input string, so its binding is invalid"}
    assert core._critic_field_claim_conflicts(false_claim, method) is True
    assert core._critic_field_claim_conflicts(real_claim, method) is False


def test_component_plan_merge_prompts_require_explicit_prior_component_coverage():
    assert "merged_from" in core._COMPONENT_PLAN_PROPOSER_SYSTEM
    assert "merged_from" in core._COMPONENT_PLAN_CRITIC_SYSTEM
    assert "not conceptual submechanisms" in core._COMPONENT_PLAN_PROPOSER_SYSTEM
    assert "exact confirmed Method component ID" in core._COMPONENT_PLAN_CRITIC_SYSTEM


def test_resumed_component_plan_uses_confirmed_ids_not_partial_draft_ids():
    class CaptureLLM:
        calls = []

        def chat(self, *, messages, **kwargs):
            self.calls.append(messages)
            raise RuntimeError("stop after capturing the proposal context")

    llm = CaptureLLM()
    draft = {"components": [{"component_id": "interim-combined"}]}
    with pytest.raises(core.DeliberationLLMError):
        core.deliberate_method(
            _PM, _LIT, None, draft, llm, inner_rounds=1, outer_rounds=1,
            component_plan_audit=True, confirmed_component_ids={"confirmed-a", "confirmed-b"},
        )
    proposal_context = llm.calls[0][1]["content"]
    assert 'Only these confirmed Method component IDs may appear in merged_from: ["confirmed-a", "confirmed-b"]' in proposal_context
    assert 'Only these confirmed Method component IDs may appear in merged_from: ["interim-combined"]' not in proposal_context


def test_cap_rejects_merge_of_unconfirmed_submechanism():
    component = {
        "component_id": "combined", "name": "combined", "type": "direct_gain",
        "depends_on": [], "acceptance_requires": [], "purpose": "reduce latency",
        "inputs": ["baseline"], "outputs": ["candidate"],
        "mechanism": "combine three operations", "expected_effect": "lower latency",
        "ablation_test": "signed_metric_improvement > 0",
        "implementation_check": "host diff and runtime result match",
        "optimization_space": "test legal settings",
        "downstream_contract": "candidate feeds final evaluation",
        "merged_from": ["internal-operation"],
    }
    method = {"components": [component]}
    problem = core._component_contract_problem(
        method, require_design_fields=True, confirmed_component_ids={"combined"})
    assert "internal-operation" in str(problem)
    assert "combined" in str(problem)
    component["merged_from"] = []
    assert core._component_contract_problem(
        method, require_design_fields=True, confirmed_component_ids={"combined"}) is None


@pytest.mark.parametrize("merged_from", ["prior-node", " prior-node ", ["prior-node"], "", []])
def test_cap_merge_source_accepts_existing_scalar_id_compatibility(merged_from):
    component = {
        "component_id": "combined", "name": "combined", "type": "direct_gain",
        "depends_on": [], "acceptance_requires": [], "purpose": "reduce latency",
        "inputs": ["baseline"], "outputs": ["candidate"],
        "mechanism": "combine operations", "expected_effect": "lower latency",
        "ablation_test": "signed_metric_improvement > 0",
        "implementation_check": "host diff and runtime result match",
        "optimization_space": "test legal settings",
        "downstream_contract": "candidate feeds final evaluation",
        "merged_from": merged_from,
    }
    assert core._component_contract_problem(
        {"components": [component]}, require_design_fields=True,
        confirmed_component_ids={"prior-node"}) is None
    assert component["merged_from"] == merged_from


@pytest.mark.parametrize("merged_from", ["unknown-node", ["unknown-node"], 3,
                                        {"prior-node": True}, [True], [""]])
def test_cap_merge_source_rejects_unknown_ids_and_invalid_types(merged_from):
    component = {
        "component_id": "combined", "name": "combined", "type": "direct_gain",
        "depends_on": [], "acceptance_requires": [], "purpose": "reduce latency",
        "inputs": ["baseline"], "outputs": ["candidate"],
        "mechanism": "combine operations", "expected_effect": "lower latency",
        "ablation_test": "signed_metric_improvement > 0",
        "implementation_check": "host diff and runtime result match",
        "optimization_space": "test legal settings",
        "downstream_contract": "candidate feeds final evaluation",
        "merged_from": merged_from,
    }
    assert core._component_contract_problem(
        {"components": [component]}, require_design_fields=True,
        confirmed_component_ids={"prior-node"}) is not None


def test_component_plan_drops_stale_critic_field_claim_but_records_it():
    class PlanLLM(_RoleLLM):
        def chat(self, *, messages, **kwargs):
            response = super().chat(messages=messages, **kwargs)
            if "METHOD PROPOSER" in messages[0]["content"]:
                method = json.loads(response.content)
                method["components"][0].update({
                    "component_id": "order-guard", "type": "indirect_gain",
                    "depends_on": [], "acceptance_requires": [],
                    "inputs": ["parent output"], "outputs": ["ordered output"],
                    "functional_check": "ordering_errors == 0",
                    "implementation_check": "runtime output preserves order",
                    "optimization_space": "reduce guard overhead",
                    "downstream_contract": "preserve the output interface",
                })
                return _Resp(json.dumps(method))
            return response

    llm = PlanLLM(critic=[{"problems": [{
        "category": "component_soundness", "component": "order-guard",
        "issue": 'order-guard.depends_on is ["ghost"] but ghost is missing',
    }]}])
    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        component_plan_audit=True)
    assert result.report["accepted"] is True
    assert len(result.report["transcript"][-1]["contradicted_critic_problems"]) == 1


def test_component_plan_critic_can_route_a_conflicting_confirmed_method_to_revision():
    llm = _RoleLLM(
        conv=[{"converged": True}],
        critic=[{
            "method_revision_required": True,
            "method_revision_reason": "the confirmed algorithm is prohibited by the repository contract",
            "problems": [{
                "category": "component_soundness",
                "issue": "components cannot faithfully implement the prohibited core algorithm",
                "suggestion": "revise the top-level Method before planning components",
                "component": "plan",
            }],
        }],
    )

    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=5, component_plan_audit=True,
    )

    assert result.report["accepted"] is False
    assert result.report["method_revision_required"] is True
    assert "prohibited" in result.report["method_revision_reason"]
    assert result.report["outer_rounds"] == 1
    assert result.report["transcript"][-1]["outcome"] == (
        "confirmed_method_requires_revision"
    )
    critic_prompts = [call[0]["content"] for call in llm.calls
                      if "METHOD CRITIC" in call[0].get("content", "")]
    assert critic_prompts
    assert "mutually inconsistent requirements" in critic_prompts[0]


def test_repository_quote_alone_cannot_force_confirmed_method_revision(monkeypatch):
    monkeypatch.setattr(core, "_component_contract_problem", lambda *a, **k: None)
    monkeypatch.setattr(core, "_repository_review_problems", lambda *a, **k: [{
        "category": "feasibility",
        "grounded": True,
        "status": "contradicted",
        "issue": "Critic labeled an exact repository quote as contradictory",
        "suggestion": "Check the frozen Method claim against the quote",
    }])
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{
        "method_revision_required": False, "problems": [],
    }])

    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm,
        inner_rounds=1, outer_rounds=1, component_plan_audit=True,
    )

    assert result.report["accepted"] is False
    assert result.report["method_revision_required"] is False
    assert result.report["transcript"][-1]["outcome"] != "confirmed_method_requires_revision"


def test_component_plan_critic_routes_an_uneditable_dependency_claim_without_more_cap_rounds():
    draft = {**_DRAFT, "approach": "The terminal must use only the first stage output."}
    reason = (
        "Frozen approach requires only the first stage output, but a second required stage "
        "produces the terminal's input; no component dependency edit satisfies both."
    )
    llm = _RoleLLM(
        conv=[{"converged": True}],
        critic=[{
            "method_revision_required": True,
            "method_revision_reason": reason,
            "problems": [{
                "category": "component_soundness",
                "component": "terminal",
                "issue": reason,
                "suggestion": "Reconcile the frozen dependency statement before CAP refinement.",
            }],
        }],
    )

    result = core.deliberate_method(
        _PM, _LIT, None, draft, llm,
        inner_rounds=1, outer_rounds=5, component_plan_audit=True,
    )

    assert result.method["approach"] == draft["approach"]
    assert result.report["accepted"] is False
    assert result.report["method_revision_required"] is True
    assert result.report["method_revision_reason"] == reason
    assert result.report["outer_rounds"] == 1
    critic_prompt = next(call[0]["content"] for call in llm.calls
                         if "METHOD CRITIC" in call[0].get("content", ""))
    assert "do not repeatedly tell CAP proposers to edit a frozen field" in critic_prompt
    assert "stay within CAP and must not trigger that flag" in critic_prompt
    assert "wording-only difference is not a substantive contradiction" in critic_prompt


def test_critic_does_NOT_run_while_inner_not_converged():
    # B: the inner loop is pure A1<->A2 (+ MOSS guiding the non-converged exchange); the critic only
    # appears AFTER convergence.
    llm = _RoleLLM(conv=[{"converged": False}, {"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=2)
    assert llm.roles.index("critic") > llm.roles.index("moss_guide")   # critic strictly after
    assert llm.roles.count("critic") == 1              # not once per inner exchange
    assert "moss_guide" in llm.roles                   # MOSS GUIDED the non-converged inner exchange


def test_two_proposers_each_run_and_cross_critique():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1)
    assert llm.roles.count("proposer:A1") == 1 and llm.roles.count("proposer:A2") == 1


@pytest.mark.parametrize("critique,expected", [
    ("组件验收需要前置依赖。", ["组件验收需要前置依赖。"]),
    (["检查依赖", "检查验收"], ["检查依赖", "检查验收"]),
    ("  ", []),
    (None, []),
])
def test_proposer_critique_string_is_one_transcript_item(critique, expected):
    class StringCritiqueLLM(_RoleLLM):
        def chat(self, *, messages, **kwargs):
            response = super().chat(messages=messages, **kwargs)
            if "METHOD PROPOSER" in messages[0]["content"]:
                payload = json.loads(response.content)
                payload["critique_of_other"] = critique
                return _Resp(json.dumps(payload))
            return response

    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, StringCritiqueLLM(), inner_rounds=1, outer_rounds=1)
    exchange = result.report["transcript"][0]
    assert exchange["crit_a"] == expected
    assert exchange["crit_b"] == expected
    assert result.report["accepted"] is True


def test_inner_capped_without_convergence_is_never_accepted():
    # #62: inner caps WITHOUT converging -> the critic STILL reviews (its problems seed next round) but
    # a non-converged method is NEVER accepted, even with a clean critic. outer_rounds=1 -> the forced
    # re-round has no budget, so we hit the honest unconverged fallback (keep A1, accepted=False).
    llm = _RoleLLM(conv=[{"converged": False}] * 9, critic=[{"problems": []}])
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=1)
    assert res.report["converged"] is False            # never converged
    assert res.report["total_exchanges"] == 3          # inner ran to its cap
    assert llm.roles.count("critic") == 1              # critic STILL reviewed after the cap
    assert res.report["accepted"] is False             # #62: capped-not-converged is NOT accepted
    assert res.report["unconverged_fallback"] is True
    assert res.report["transcript"][-1]["outcome"] == "capped_unconverged_fallback"
    assert res.method["approach"].startswith("m-A1")   # fail-open: A1 still returned, honestly labelled


def test_not_converged_forces_another_outer_round_then_converges():
    # #62: outer round 1 caps unconverged (critic clean) -> MUST force outer round 2 rather than accept;
    # if round 2's inner converges, THEN accept. Proves non-convergence drives another round, not a pass.
    llm = _RoleLLM(conv=[{"converged": False}] * 3 + [{"converged": True}],
                   critic=[{"problems": []}])
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=2)
    assert res.report["outer_rounds"] == 2             # did NOT stop after the unconverged round 1
    assert res.report["converged"] is True
    assert res.report["accepted"] is True
    assert res.report["unconverged_fallback"] is False
    assert any(r.get("outcome") == "outer_another_round_unconverged"
               for r in res.report["transcript"])


def test_capped_critic_sees_both_methods_and_moss_divergence():
    # 会议 2026-08-04: 内层跑满未收敛时, critic 要看 A1、A2 两个现方法 + MOSS 分歧, 不是只留 A1.
    llm = _RoleLLM(conv=[{"converged": False}] * 9, critic=[{"problems": []}])
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=1)
    critic_msgs = [" ".join(m.get("content", "") for m in msgs)
                   for msgs in llm.calls if "METHOD CRITIC" in msgs[0].get("content", "")]
    assert critic_msgs, "critic must have run after the cap"
    blob = critic_msgs[-1]
    assert "A1 现方法" in blob and "A2 现方法" in blob and "MOSS 分歧" in blob
    assert "m-A1" in blob and "m-A2" in blob        # BOTH proposers' methods, not just A1
    assert "adopt the other" in blob                # MOSS's GUIDING note is carried into the review
    cap = [r for r in res.report["transcript"] if r.get("capped_review")]
    assert cap and cap[-1]["capped_review"]["b_approach"]   # A2's method recorded in the transcript


def test_converged_critic_sees_only_agreed_method():
    # 收敛时 A1 代表两者, critic 看 AGREED 方法即可 (不塞两个).
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=1)
    critic_msgs = [" ".join(m.get("content", "") for m in msgs)
                   for msgs in llm.calls if "METHOD CRITIC" in msgs[0].get("content", "")]
    assert critic_msgs and "agreed" in critic_msgs[-1].lower()
    assert "A2 现方法" not in critic_msgs[-1]


def test_critic_problem_triggers_next_OUTER_round_via_moss_guidance():
    # outer1: inner converges; critic finds a target-undermining problem -> MOSS GUIDINGLY relays it
    #         (NOT a worth-judgment) -> another outer round.
    # outer2: inner converges; critic clean -> accept.
    llm = _RoleLLM(
        conv=[{"converged": True}, {"converged": True}],
        critic=[{"problems": [{"category": "component_soundness", "issue": "comp1 has no mechanism",
                               "suggestion": "give comp1 a real mechanism + ablation"}]},
                {"problems": []}])
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=3)
    assert res.report["outer_rounds"] == 2 and res.report["accepted"] is True
    assert llm.roles.count("critic") == 2 and llm.roles.count("moss_relay") == 1
    assert "moss_worth" not in llm.roles          # MOSS no longer judges 'worth'
    assert res.report["transcript"][-1]["outcome"] == "accepted_converged_clean"
    # the round-1 entry is the guided relay, NOT a move-on acceptance
    assert any(r.get("outcome") == "outer_another_round_guided" for r in res.report["transcript"])


def test_critic_problem_at_outer_cap_is_not_accepted():
    # meeting: MOSS no longer 'moves on' and accepts a still-criticised method. A target-undermining
    # problem at the outer cap ends WITHOUT acceptance (fail-open at the caller keeps the method).
    llm = _RoleLLM(
        conv=[{"converged": True}],
        critic=[{"problems": [{"category": "feasibility", "issue": "no ablation for comp1"}]}])
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=1)
    assert res.report["outer_rounds"] == 1 and res.report["accepted"] is False
    assert res.report["transcript"][-1]["outcome"] == "outer_another_round_guided"
    assert llm.roles.count("moss_relay") == 1 and "moss_worth" not in llm.roles


def test_rubric_drops_shallow_problems_then_accepts():
    llm = _RoleLLM(
        conv=[{"converged": True}],
        critic=[{"problems": [{"category": "wording", "issue": "rename this"},
                              {"category": "formatting", "issue": "use bullets"}]}])
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=2)
    assert res.report["transcript"][-1]["critic_deep_problems"] == []   # rubric filtered them all
    assert res.report["transcript"][-1]["outcome"] == "accepted_converged_clean"
    assert "moss_relay" not in llm.roles     # critic came back clean -> MOSS never had to relay


def test_critic_and_moss_are_separate_roles():
    llm = _RoleLLM(
        conv=[{"converged": True}],
        critic=[{"problems": [{"category": "novelty_conflict", "issue": "subsumed by X"}]}])
    core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1)
    # critic (tactical problem-finder) and MOSS (coordinator that guidingly relays) are distinct roles
    assert "critic" in llm.roles and "moss_relay" in llm.roles


def test_max_rounds_is_backward_compat_alias_for_inner():
    llm = _RoleLLM(conv=[{"converged": False}] * 5)
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, max_rounds=2, outer_rounds=1)
    assert res.report["inner_cap"] == 2 and res.report["total_exchanges"] == 2


def test_user_idea_and_other_method_reach_the_proposer():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    core.deliberate_method(_PM, _LIT, "USE_ADAMW_BETAS", _DRAFT, llm, inner_rounds=1, outer_rounds=1)
    prop_blobs = [" ".join(m.get("content", "") for m in msgs)
                  for msgs in llm.calls if "METHOD PROPOSER" in msgs[0].get("content", "")]
    assert any("USE_ADAMW_BETAS" in b for b in prop_blobs)
    # MOSS-mediated (老师 2026-08-07): the OTHER's method reaches the proposer RELAYED BY MOSS
    # as a colleague reference (not a direct A<->B debate).
    assert any("MOSS relays" in b and "colleague" in b for b in prop_blobs)


def test_frozen_baseline_runtime_evidence_reaches_all_method_roles():
    llm = _RoleLLM(conv=[{"converged": True}], critic=[{"problems": []}])
    baseline = {
        "metric_name": "pure_train_time_seconds",
        "metric_value": 196.682,
        "fixed_config": "Muon + AdamW hybrid; bf16; manual all-reduce",
    }

    core.deliberate_method(
        _PM,
        _LIT,
        None,
        _DRAFT,
        llm,
        inner_rounds=1,
        outer_rounds=1,
        baseline_result=baseline,
    )

    role_blobs = [" ".join(m.get("content", "") for m in messages)
                  for messages in llm.calls]
    assert role_blobs
    assert all("Authoritative frozen BaselineResult" in blob for blob in role_blobs)
    assert all("Muon + AdamW hybrid" in blob for blob in role_blobs)


def test_failsoft_llm_raises_keeps_draft():
    llm = _RoleLLM(raise_all=True)
    res = core.deliberate_method(_PM, _LIT, None, _DRAFT, llm, inner_rounds=3, outer_rounds=2)
    assert res.method["approach"] == "draft"           # no crash; keep the (normalized) draft
    assert res.report["converged"] is False


def test_rubric_categories_exposed():
    assert "component_soundness" in core.METHOD_CRITIC_RUBRIC
    assert "ablation_verifiability" in core.METHOD_CRITIC_RUBRIC



def test_component_plan_audit_checks_formula_identity_and_boundary_consistency():
    # These are prompt/transport assertions, not claims that a real LLM proved the algebra.
    for prompt in (core._COMPONENT_PLAN_PROPOSER_SYSTEM, core._COMPONENT_PLAN_CRITIC_SYSTEM):
        assert "neutral weights" in prompt
        assert "sum versus mean" in prompt
        assert "rounding" in prompt
        assert "same derived value" in prompt


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_component_plan_consensus_retries_non_boolean_without_failing_agent(value):
    llm = _RoleLLM(conv=[{"converged": value}, {"converged": True}])

    result = core.deliberate_method(
        _PM, _LIT, None, _DRAFT, llm, inner_rounds=1, outer_rounds=1,
        component_plan_audit=True,
    )

    assert result.report["converged"] is True
    assert llm.roles.count("converge") == 2
    assert "critic" in llm.roles
