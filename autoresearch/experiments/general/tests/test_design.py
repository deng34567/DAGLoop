"""Task 1: ``design_experiment`` -- a user question -> a structured, code-runnable
experiment design (7 fields). FULLY MOCKED LLM (canned JSON queue); no network.
"""
from __future__ import annotations

import json

import pytest

from autoresearch.experiments.general import design as gen_design


class _FakeLLMResult:
    def __init__(self, content):
        self.content = content
        self.usage = {"total_tokens": 1}
        self.cost_delta = "0"


class _FakeLLM:
    """Returns a queue of canned responses; records the messages it was called with."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        content = self._responses.pop(0) if self._responses else ""
        return _FakeLLMResult(content)


_GOOD_DESIGN = {
    "question": "Can a heuristic beat First-Fit on synthetic 1D bin packing?",
    "task": "1D bin packing, capacity C=1.0, item sizes ~ U(0,0.7).",
    "instance_spec": "For a given seed, draw 200 item sizes with numpy default_rng(seed).",
    "baseline": "First-Fit: place each item in the first bin it fits.",
    "method_interface": "solve(instance) -> list[int] (bin index per item).",
    "metric": "number of bins used (lower is better); valid = all items placed, no bin over capacity.",
    "success_criterion": "method uses strictly fewer bins than First-Fit.",
}


def test_method_set_to_lose_reason_catches_role_inversion():
    from autoresearch.experiments.general.design import _method_set_to_lose_reason
    d = {"success_criterion": "The proposed method (EST greedy) achieves strictly lower count "
         "than the baseline (EFT greedy), establishing that EST is suboptimal."}
    assert _method_set_to_lose_reason(d) is not None
    # Chinese phrasing too
    assert _method_set_to_lose_reason({"success_criterion": "所提方法被证明是次优的"}) is not None


def test_method_set_to_lose_reason_passes_normal_designs():
    from autoresearch.experiments.general.design import _method_set_to_lose_reason
    assert _method_set_to_lose_reason(_GOOD_DESIGN) is None
    assert _method_set_to_lose_reason(
        {"success_criterion": "the method achieves higher accuracy than the raw baseline"}) is None


def test_design_experiment_redesigns_on_role_inversion():
    inverted = {**_GOOD_DESIGN,
                "success_criterion": "the proposed method is shown to be suboptimal / worse "
                "than the baseline"}
    # design1 = inverted; LLM critic passes (faithful); the DETERMINISTIC guard fires -> design2
    llm = _FakeLLM([json.dumps(inverted),
                    json.dumps({"faithful": True, "issues": []}),
                    json.dumps(_GOOD_DESIGN)])
    out = gen_design.design_experiment(llm, "q")
    assert out["success_criterion"] == _GOOD_DESIGN["success_criterion"]
    assert len(llm.calls) == 3  # design + critic + corrective re-design


def test_missing_heldout_reason_fires_for_ml_metric_without_split():
    from autoresearch.experiments.general.design import _missing_heldout_reason
    ml_no_split = {"metric": "classification accuracy (higher is better)",
                   "success_criterion": "method beats baseline",
                   "instance_spec": "draw 200 samples with numpy default_rng(seed)", "task": "kNN"}
    assert _missing_heldout_reason(ml_no_split) is not None
    # with a held-out split named -> OK
    with_split = {**ml_no_split,
                  "instance_spec": "80/20 train/test split; scale on train only, score on held-out test"}
    assert _missing_heldout_reason(with_split) is None


def test_missing_heldout_reason_ignores_combinatorial_metric():
    from autoresearch.experiments.general.design import _missing_heldout_reason
    # _GOOD_DESIGN's metric is 'number of bins used' -> not ML -> held-out not applicable
    assert _missing_heldout_reason(_GOOD_DESIGN) is None


def test_design_experiment_redesigns_when_ml_design_lacks_heldout():
    ml_no_split = {**_GOOD_DESIGN, "metric": "classification accuracy, higher is better",
                   "instance_spec": "draw 300 labeled samples", "success_criterion": "method > baseline"}
    good_split = {**ml_no_split,
                  "instance_spec": "300 samples, 80/20 train/test split; fit scaler on train, score on held-out test"}
    # design1 = ml_no_split; critic passes; role-inversion clean; T6 guard fires -> design2
    llm = _FakeLLM([json.dumps(ml_no_split),
                    json.dumps({"faithful": True, "issues": []}),
                    json.dumps(good_split)])
    out = gen_design.design_experiment(llm, "does scaling help kNN?")
    assert "held-out" in out["instance_spec"] or "held out" in out["instance_spec"]


def test_synthetic_for_real_dataset_reason_fires_on_substitution():
    from autoresearch.experiments.general.design import _synthetic_for_real_dataset_reason
    d = {"task": "kNN on Acme-2026", "instance_spec": "generate 200 samples with make_classification",
         "success_criterion": "method > baseline", "metric": "accuracy"}
    question = "Does scaling help kNN on the previously unseen Acme-2026 dataset?"
    assert _synthetic_for_real_dataset_reason(d, question) is not None
    # real loader named -> OK
    ok = {**d, "instance_spec": "load the official Acme-2026 data, 80/20 held-out split"}
    assert _synthetic_for_real_dataset_reason(ok, question) is None


def test_synthetic_for_real_dataset_reason_ignores_no_named_dataset():
    from autoresearch.experiments.general.design import _synthetic_for_real_dataset_reason
    # no real dataset named -> synthetic is legitimate
    d = {"task": "bin packing", "instance_spec": "make synthetic sizes with default_rng(seed)",
         "success_criterion": "fewer bins", "metric": "bins"}
    assert _synthetic_for_real_dataset_reason(d, "Can a heuristic beat First-Fit?") is None


def test_synthetic_for_real_dataset_reason_allows_explicit_synthetic_request():
    from autoresearch.experiments.general.design import _synthetic_for_real_dataset_reason
    d = {"task": "classification", "instance_spec": "make synthetic samples with default_rng(seed)",
         "success_criterion": "higher accuracy", "metric": "accuracy"}
    question = "Compare the methods on a reproducible synthetic classification dataset"
    assert _synthetic_for_real_dataset_reason(d, question) is None


def test_synthetic_for_real_dataset_reason_ignores_non_data_url():
    from autoresearch.experiments.general.design import _synthetic_for_real_dataset_reason
    d = {"task": "simulation", "instance_spec": "generate synthetic instances with default_rng(seed)",
         "success_criterion": "lower error", "metric": "error"}
    question = "Explore this method from https://example.org/paper/1234"
    assert _synthetic_for_real_dataset_reason(d, question) is None


def test_design_experiment_redesigns_on_synthetic_substitution():
    bad = {**_GOOD_DESIGN, "task": "kNN on the Wine dataset", "metric": "accuracy",
           "instance_spec": "generate 200 synthetic samples via make_classification",
           "success_criterion": "method beats baseline on a held-out test split"}
    good = {**bad, "instance_spec": "load_wine() from sklearn; 80/20 held-out test split"}
    llm = _FakeLLM([json.dumps(bad),
                    json.dumps({"faithful": True, "issues": []}),
                    json.dumps(good)])
    out = gen_design.design_experiment(llm, "Does standardization help kNN on the UCI Wine dataset?")
    assert "load_wine" in out["instance_spec"]


def test_design_experiment_returns_seven_field_dict():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN)])

    out = gen_design.design_experiment(llm, "Can a heuristic beat First-Fit on bin packing?")

    assert set(out) >= set(gen_design.REQUIRED_FIELDS)
    assert out["task"].startswith("1D bin packing")
    assert out["method_interface"].startswith("solve(instance)")
    # prompt carried the user's question
    sent = " ".join(m["content"] for m in llm.calls[0])
    assert "First-Fit" in sent


def test_design_experiment_prompt_asks_for_seed_and_computable_metric():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN)])
    gen_design.design_experiment(llm, "some question")
    sent = " ".join(m["content"] for m in llm.calls[0]).lower()
    assert "seed" in sent
    assert "solve(instance)" in sent


def test_design_experiment_missing_field_raises_clear_error():
    bad = dict(_GOOD_DESIGN)
    del bad["metric"]
    llm = _FakeLLM([json.dumps(bad)])
    with pytest.raises(gen_design.DesignError) as ei:
        gen_design.design_experiment(llm, "q")
    assert "metric" in str(ei.value)


def test_design_experiment_bad_json_raises_clear_error():
    llm = _FakeLLM(["not json at all"])
    with pytest.raises(gen_design.DesignError):
        gen_design.design_experiment(llm, "q")


def test_design_with_dependencies_exposes_them():
    d = dict(_GOOD_DESIGN, dependencies=["numpy", "scipy"])
    out = gen_design.design_experiment(_FakeLLM([json.dumps(d)]), "q")
    assert out["dependencies"] == ["numpy", "scipy"]


def test_design_without_dependencies_defaults_to_empty_list():
    # _GOOD_DESIGN has no dependencies key -> [] (pure stdlib), not a hard failure
    out = gen_design.design_experiment(_FakeLLM([json.dumps(_GOOD_DESIGN)]), "q")
    assert out["dependencies"] == []


def test_design_non_list_dependencies_coerces_to_empty_list():
    d = dict(_GOOD_DESIGN, dependencies="numpy")  # a string, not a list
    out = gen_design.design_experiment(_FakeLLM([json.dumps(d)]), "q")
    assert out["dependencies"] == []


def test_design_prompt_asks_for_dependencies():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN)])
    gen_design.design_experiment(llm, "q")
    sent = " ".join(m["content"] for m in llm.calls[0]).lower()
    assert "dependencies" in sent
    assert "pip" in sent


def test_sanitize_dependencies_drops_unsafe_and_non_strings():
    deps = ["numpy", "scipy", "numpy; rm -rf /", "$(evil)", "a/b", 123, "", "pandas"]
    assert gen_design.sanitize_dependencies(deps) == ["numpy", "scipy", "pandas"]
    assert gen_design.sanitize_dependencies("numpy") == []   # non-list
    assert gen_design.sanitize_dependencies(None) == []


def test_pip_install_command_builds_safe_command_or_none():
    assert gen_design.pip_install_command([]) is None
    assert gen_design.pip_install_command(["numpy; rm -rf /"]) is None  # all dropped
    cmd = gen_design.pip_install_command(["numpy", "scipy"])
    assert cmd == "pip install -q numpy scipy"
    # a package with regex-safe but shell-quotable chars is still quoted, not injected
    assert ";" not in gen_design.pip_install_command(["numpy", "scipy"])


# --- P5: design-scope discipline (one method vs one baseline) -----------------

_SURVEY_DESIGN = {
    **_GOOD_DESIGN,
    "task": "Systematically compare multiple classical heuristics on 1D bin packing.",
    "baseline": "First-Fit, Best-Fit, Next-Fit, First-Fit-Decreasing",
    "success_criterion": "characterize how the classical heuristics compare across distributions.",
}


def test_overscoped_reason_flags_survey_passes_single():
    assert gen_design._overscoped_reason(_SURVEY_DESIGN) is not None
    assert gen_design._overscoped_reason(_GOOD_DESIGN) is None


def test_design_narrows_overscoped_then_accepts():
    llm = _FakeLLM([json.dumps(_SURVEY_DESIGN), json.dumps(_GOOD_DESIGN)])
    out = gen_design.design_experiment(llm, "compare bin packing heuristics")
    assert len(llm.calls) == 3                          # design + one narrowing re-design + critic
    assert out["baseline"].startswith("First-Fit:")     # the narrowed one-vs-one design
    assert "REDESIGN" in llm.calls[1][1]["content"]      # 2nd call carried the narrowing instruction


def test_design_one_vs_one_no_redesign():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN)])
    gen_design.design_experiment(llm, "beat First-Fit")
    assert len(llm.calls) == 2                          # design + critic (no re-design on a good design)


def test_design_narrowing_accepts_second_even_if_still_broad():
    # A false positive costs only ONE extra call and never loops: the 2nd design is
    # accepted as-is even if the heuristic would still flag it.
    llm = _FakeLLM([json.dumps(_SURVEY_DESIGN), json.dumps(_SURVEY_DESIGN)])
    out = gen_design.design_experiment(llm, "q")
    assert len(llm.calls) == 3                          # design + narrowing re-design + critic
    assert out["baseline"] == _SURVEY_DESIGN["baseline"]


# --- unbeatable baseline (an optimal/oracle baseline makes the run meaningless) -

_OPTIMAL_BASELINE_DESIGN = dict(
    _GOOD_DESIGN,
    baseline="Optimal-DP: the exact optimum via dynamic programming.",
    success_criterion="method achieves ratio closer to the optimal than the baseline.",
)


def test_unbeatable_baseline_reason_flags_optimal_passes_simple():
    assert gen_design._unbeatable_baseline_reason(_OPTIMAL_BASELINE_DESIGN) is not None
    # a simple heuristic baseline is fine
    assert gen_design._unbeatable_baseline_reason(_GOOD_DESIGN) is None
    # "optimal" appearing only in the METRIC (as a normalization reference) is fine --
    # only the BASELINE field being the optimum is the problem.
    ok_metric = dict(_GOOD_DESIGN, metric="approximation ratio vs the optimal (higher is better)")
    assert gen_design._unbeatable_baseline_reason(ok_metric) is None


def test_unbeatable_baseline_reason_flags_other_oracles():
    for base in ("exact brute-force search", "ILP solver (exact)", "oracle ground-truth",
                 "穷举最优解"):
        assert gen_design._unbeatable_baseline_reason(dict(_GOOD_DESIGN, baseline=base)), base


def test_design_fixes_unbeatable_baseline_then_accepts():
    llm = _FakeLLM([json.dumps(_OPTIMAL_BASELINE_DESIGN), json.dumps(_GOOD_DESIGN)])
    out = gen_design.design_experiment(llm, "can value-density greedy beat index-order?")
    assert len(llm.calls) == 3                       # design + one corrective re-design + critic
    assert out["baseline"].startswith("First-Fit:")  # the simple-baseline redesign
    msg = llm.calls[1][1]["content"].lower()
    assert "baseline" in msg and ("beat" in msg or "unbeatable" in msg or "optimal" in msg)


def test_design_good_baseline_no_redesign():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN)])
    gen_design.design_experiment(llm, "beat First-Fit")
    assert len(llm.calls) == 2                        # design + critic (no re-design on a fine baseline)


# --- LLM design-faithfulness critic (P2): the general net over the regex guards ----

_CRITIC_OK = json.dumps({"faithful": True, "issues": []})
# a WSPT baseline is domain-optimal for weighted completion time -> the regex can't
# catch it (no literal "optimal"), but the critic can flag it.
_WSPT_DESIGN = dict(
    _GOOD_DESIGN,
    baseline="WSPT: schedule jobs by nondecreasing p_j/w_j (Smith's rule).",
    metric="total weighted completion time (lower is better)",
    success_criterion="method achieves lower total weighted completion time than WSPT.",
)


def test_design_critic_prompt_is_leak_safe_and_checks_faithfulness():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN), _CRITIC_OK])
    gen_design.design_experiment(llm, "beat First-Fit on bin packing")
    critic_msgs = llm.calls[1]                      # 2nd call is the critic
    system = critic_msgs[0]["content"].lower()
    user = critic_msgs[1]["content"]
    # it checks the four faithfulness dimensions
    assert "beatable" in system or "optimal" in system
    assert "faithful" in system and "direction" in system
    # leak-safe: the critic never sees harness/scoring internals
    for leaked in ("make_instances", "run_baseline", "def score", "harness"):
        assert leaked not in user


def test_design_critic_flags_domain_optimal_baseline_and_redesigns():
    critic_flag = json.dumps({"faithful": False,
                              "issues": ["baseline WSPT is domain-optimal for this metric"]})
    llm = _FakeLLM([json.dumps(_WSPT_DESIGN), critic_flag, json.dumps(_GOOD_DESIGN)])
    out = gen_design.design_experiment(llm, "can rule X beat FCFS on weighted completion time?")
    assert len(llm.calls) == 3                       # design + critic + one corrective re-design
    assert out["baseline"].startswith("First-Fit:")  # the fixed, beatable-baseline design
    redesign = llm.calls[2][1]["content"].lower()
    assert "wspt" in redesign or "domain-optimal" in redesign or "beatable" in redesign


def test_design_critic_checks_fair_comparison():
    # #3 P2 fix: the critic system prompt must demand a FAIR comparison so it can flag
    # a rigged design (e.g. one shared learning rate that diverges for one optimizer).
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN), _CRITIC_OK])
    gen_design.design_experiment(llm, "beat First-Fit on bin packing")
    system = llm.calls[1][0]["content"].lower()
    assert "fair" in system
    assert "learning rate" in system or "hyperparameter" in system
    assert "budget" in system                          # unequal compute/time budget is unfair too


def test_design_critic_checks_disguised_optimal_baseline_and_role_inversion():
    # baseline=0.0 root cause: an optimal baseline WITHOUT the word "optimal" (earliest-finish
    # greedy), a "gap-from-optimal" metric (baseline=0 by definition), and a success criterion
    # that expects the method to be WORSE. The critic prompt must cover all three + the fix.
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN), _CRITIC_OK])
    gen_design.design_experiment(llm, "beat First-Fit on bin packing")
    system = llm.calls[1][0]["content"].lower()
    assert "earliest-finish" in system                              # disguised-optimal example
    assert "gap" in system and ("regret" in system or "distance" in system)  # gap-metric disguise
    assert "worse" in system                                        # role inversion
    assert "direct metric" in system                                # the fix: best algo as method


def test_refaithful_message_guides_best_algorithm_as_method():
    msg = gen_design._refaithful_message("q", ["baseline is optimal"]).lower()
    assert "proposed method" in msg and "beatable baseline" in msg
    assert "direct metric" in msg and "not a gap" in msg


def test_design_critic_flags_unfair_hyperparams_and_redesigns():
    # AI proposed the SAME lr for Adam and SGD (Adam diverges) -> honest but unfair.
    unfair = dict(_GOOD_DESIGN,
                  task="Adam vs SGD, both at learning rate 0.01, on a small MLP.")
    critic_flag = json.dumps({"faithful": False,
                              "issues": ["same lr=0.01 for both Adam and SGD is unfair; "
                                         "Adam diverges at that rate"]})
    llm = _FakeLLM([json.dumps(unfair), critic_flag, json.dumps(_GOOD_DESIGN)])
    gen_design.design_experiment(llm, "does Adam beat SGD?")
    assert len(llm.calls) == 3                          # design + critic + one corrective re-design
    redesign = llm.calls[2][1]["content"].lower()
    assert "fair" in redesign                            # the redesign message carries fairness guidance


def test_design_critic_pass_no_redesign():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN), _CRITIC_OK])
    out = gen_design.design_experiment(llm, "beat First-Fit")
    assert len(llm.calls) == 2                        # design + critic, no re-design
    assert out["baseline"] == _GOOD_DESIGN["baseline"]


def test_design_critic_fails_open_on_bad_json():
    # a flaky critic (unparseable output) must NOT block a real design -> no re-design.
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN), "not json at all"])
    out = gen_design.design_experiment(llm, "beat First-Fit")
    assert len(llm.calls) == 2
    assert out["baseline"] == _GOOD_DESIGN["baseline"]


def test_design_critic_unit_failopen_and_parse():
    # direct unit test of the critic helper
    ok = gen_design._design_critic(_FakeLLM([_CRITIC_OK]), "q", _GOOD_DESIGN)
    assert ok == {"faithful": True, "issues": []}
    flagged = gen_design._design_critic(
        _FakeLLM([json.dumps({"faithful": False, "issues": ["x"]})]), "q", _GOOD_DESIGN)
    assert flagged["faithful"] is False and flagged["issues"] == ["x"]
    # bad JSON -> fail-open
    assert gen_design._design_critic(_FakeLLM(["garbage"]), "q", _GOOD_DESIGN)["faithful"] is True


# --- metric direction (higher-is-better must not be inverted) ------------------

def test_infer_lower_is_better():
    assert gen_design._infer_lower_is_better({"metric": "number of bins, lower is better"}) is True
    assert gen_design._infer_lower_is_better({"metric": "total value; higher is better"}) is False
    assert gen_design._infer_lower_is_better({"success_criterion": "maximize accuracy"}) is False
    assert gen_design._infer_lower_is_better({"metric": "RMSE"}) is True   # default


def test_design_respects_llm_lower_is_better():
    d = dict(_GOOD_DESIGN, lower_is_better=False)
    out = gen_design.design_experiment(_FakeLLM([json.dumps(d)]), "q")
    assert out["lower_is_better"] is False


def test_design_infers_direction_when_missing():
    d = {**_GOOD_DESIGN, "metric": "total value, higher is better", "success_criterion": "more value"}
    d.pop("lower_is_better", None)
    out = gen_design.design_experiment(_FakeLLM([json.dumps(d)]), "q")
    assert out["lower_is_better"] is False   # inferred from "higher is better"


def test_prompt_asks_for_discriminating_instances():
    # A1: the design prompt must steer toward instances that can reveal a difference
    # (the machine-precision integration case motivated this).
    p = gen_design._SYSTEM_PROMPT.lower()
    assert "discriminat" in p and ("machine precision" in p or "floor" in p or "saturat" in p)


def test_design_coerces_gpu_flag():
    d = dict(_GOOD_DESIGN, gpu=True, dependencies=["torch"])
    out = gen_design.design_experiment(_FakeLLM([json.dumps(d)]), "does warmup help training?")
    assert out["gpu"] is True
    # a CPU design defaults gpu to False
    out2 = gen_design.design_experiment(_FakeLLM([json.dumps(_GOOD_DESIGN)]), "beat First-Fit")
    assert out2["gpu"] is False


def test_design_prompt_describes_gpu_training_teacher_student():
    llm = _FakeLLM([json.dumps(_GOOD_DESIGN)])
    gen_design.design_experiment(llm, "q")
    sys_prompt = llm.calls[0][0]["content"]
    assert "gpu" in sys_prompt.lower()
    assert "teacher" in sys_prompt.lower() and "torch" in sys_prompt.lower()
