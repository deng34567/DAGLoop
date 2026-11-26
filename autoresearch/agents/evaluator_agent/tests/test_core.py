"""
Tests for EvaluatorAgent's pure core. / EvaluatorAgent 纯核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
``select_benchmark`` and ``prepare_verifier`` are PURE: they take input dicts
and an llm_client, and return schema-valid dicts (BenchmarkSpec /
VerifierSpec).
不涉及 EventBus / Store / Redis / 网络 —— 只用 FakeLLM（预设 JSON 响应队列）。
``select_benchmark``、``prepare_verifier`` 是纯函数：接收输入 dict 和
llm_client，返回 schema 合法的 dict（BenchmarkSpec / VerifierSpec）。

``prepare_verifier``'s verifier-first guards (parse-time, NOT trusting the
prompt) are the crown jewel here:
  1. disjointness: editable_region ∩ forbidden_files == ∅
  2. frozen-core invariant: no editable_region path may reference the
     evaluator/harness/verifier itself (substring match on a small,
     documented token list).
Both are enforced independently of whatever the LLM/prompt claims, mirroring
critic_agent's grounding-check pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evaluator_agent.core import (
    MAX_RETRIES,
    EvaluatorParseError,
    _load_evaluate_experiment_system_prompt,
    _load_prepare_verifier_system_prompt,
    _load_select_benchmark_system_prompt,
    evaluate_experiment,
    prepare_verifier,
    select_benchmark,
    verify_best_result,
)
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


BENCHMARK_VALIDATOR = _validator("benchmark_spec.schema.yaml")
VERIFIER_VALIDATOR = _validator("verifier_spec.schema.yaml")
EVALUATION_REPORT_VALIDATOR = _validator("evaluation_report.schema.yaml")
VERIFICATION_REPORT_VALIDATOR = _validator("verification_report.schema.yaml")

RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

LITERATURE_BRIEF = {
    "summary": "Dense retrievers underperform on long-tail entities.",
    "key_papers": [
        {"title": "Hybrid Retrieval for Long-Tail QA", "id_or_url": "arxiv:1234.5678"},
    ],
    "existing_benchmarks": [
        {"name": "LongTailQA", "id_or_url": "https://example.org/longtailqa"},
    ],
}

BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.42,
    },
    "resource_requirements": "1x A100 GPU",
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
}

EVALUATION_SEMANTICS = {
    "comparison_statement": (
        "Under the canonical LongTailQA protocol, compare exact_match while allowing retrieval "
        "and generation method changes."
    ),
    "research_question": "Does the method improve exact match under the canonical protocol?",
    "primary_metric": {"name": "exact_match", "direction": "maximize"},
    "comparison_mode": "repository_defined",
    "comparison_basis": "LongTailQA defines the canonical comparison.",
    "budget_interpretation": "Use the benchmark's documented evaluation budget.",
    "conditions_to_hold_constant": ["dataset and evaluator"],
    "method_freedom": ["retrieval and generation method"],
    "evidence": [{
        "source_type": "existing_benchmark",
        "source_id": "https://example.org/longtailqa",
        "supports": "The retrieved benchmark defines exact match evaluation.",
    }],
    "uncertainties": [],
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _valid_verifier_spec(**overrides: object) -> dict:
    spec = {
        "evaluator_hash": "sha256:abc123",
        "metric": "exact_match",
        "data_split": {"in_loop": "dev", "held_out": "test"},
        "run_budget": "10 runs, 1h wall-clock",
        "seed_policy": "fixed seed=42 across all runs",
        "editable_region": ["src/model.py", "src/train.py"],
        "forbidden_files": ["benchmark/data/held_out.jsonl"],
        "result_parse_schema": {"type": "object"},
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# (a) select_benchmark valid -> schema-valid BenchmarkSpec.
# ---------------------------------------------------------------------------


def test_select_benchmark_valid_returns_schema_valid_spec():
    llm = FakeLLM([_json(BENCHMARK_SPEC)])

    result = select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm)

    errors = list(BENCHMARK_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["benchmark_id"] == "longtailqa-v1"
    assert len(llm.calls) == 1


def test_select_benchmark_unknown_reported_metric_does_not_fabricate_a_baseline():
    spec = json.loads(_json(BENCHMARK_SPEC))
    spec["baseline"]["reported_metric"] = None
    llm = FakeLLM([_json(spec)])

    result = select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm)

    assert result["baseline"] == {"command": spec["baseline"]["command"]}
    assert list(BENCHMARK_VALIDATOR.iter_errors(result)) == []
    assert spec["baseline"]["reported_metric"] is None
    assert len(llm.calls) == 1


@pytest.mark.parametrize("value", ["unknown", {}, []])
def test_select_benchmark_still_rejects_invalid_reported_metrics(value):
    spec = json.loads(_json(BENCHMARK_SPEC))
    spec["baseline"]["reported_metric"] = value
    llm = FakeLLM([_json(spec)] * (MAX_RETRIES + 1))
    with pytest.raises(EvaluatorParseError):
        select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm)


def test_select_benchmark_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_select_benchmark_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(BENCHMARK_SPEC)])

    result = select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm)

    assert result["benchmark_id"] == "longtailqa-v1"
    assert len(llm.calls) == 2


def test_select_benchmark_attaches_exact_frozen_evaluation_semantics():
    llm = FakeLLM([_json(BENCHMARK_SPEC)])

    result = select_benchmark(
        RESEARCH_SPEC,
        LITERATURE_BRIEF,
        llm,
        evaluation_semantics=EVALUATION_SEMANTICS,
    )

    assert result["evaluation_semantics"] == EVALUATION_SEMANTICS
    assert result["evaluation_semantics"] is not EVALUATION_SEMANTICS
    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert EVALUATION_SEMANTICS["research_question"] in user_message


# ---------------------------------------------------------------------------
# (b) prepare_verifier ok -> schema-valid VerifierSpec, verdict "ok".
# (c) each verdict (ok/incomplete/invalid) round-trips.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", ["ok", "incomplete", "invalid"])
def test_prepare_verifier_verdicts_round_trip(verdict):
    response = {"verifier_spec": _valid_verifier_spec(), "verdict": verdict}
    llm = FakeLLM([_json(response)])

    verifier_spec, result_verdict = prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    errors = list(VERIFIER_VALIDATOR.iter_errors(verifier_spec))
    assert errors == [], [e.message for e in errors]
    assert result_verdict == verdict
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (d) editable_region ∩ forbidden_files != ∅ -> typed error (disjointness).
# ---------------------------------------------------------------------------


def test_prepare_verifier_editable_region_overlaps_forbidden_files_raises_typed_error():
    bad_spec = _valid_verifier_spec(
        editable_region=["src/model.py", "benchmark/data/held_out.jsonl"],
        forbidden_files=["benchmark/data/held_out.jsonl"],
    )
    response = {"verifier_spec": bad_spec, "verdict": "ok"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) editable_region contains an evaluator/harness/metric path -> typed
# error (frozen-core invariant).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "autoresearch/agents/evaluator_agent/core.py",
        "harness/runner.py",
        "src/metric.py",
        "verifier_spec.py",
        "grading/grade_output.py",
    ],
)
def test_prepare_verifier_editable_region_touches_frozen_core_raises_typed_error(bad_path):
    bad_spec = _valid_verifier_spec(editable_region=["src/model.py", bad_path])
    response = {"verifier_spec": bad_spec, "verdict": "ok"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (f) malformed JSON / invalid verdict enum -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_prepare_verifier_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_prepare_verifier_invalid_verdict_enum_retries_then_raises_typed_error():
    response = {"verifier_spec": _valid_verifier_spec(), "verdict": "maybe"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_prepare_verifier_schema_invalid_spec_retries_then_raises_typed_error():
    bad_spec = _valid_verifier_spec()
    del bad_spec["evaluator_hash"]
    response = {"verifier_spec": bad_spec, "verdict": "ok"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_prepare_verifier_recovers_after_one_bad_response():
    good_response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM(["not json at all", _json(good_response)])

    verifier_spec, verdict = prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert verdict == "ok"
    errors = list(VERIFIER_VALIDATOR.iter_errors(verifier_spec))
    assert errors == []
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# evaluate_experiment / verify_best_result fixtures.
#
# The DETERMINISTIC is_improvement guard (anti-reward-hacking core) is:
# metric = run_record["metric_value"]; baseline = benchmark_spec["baseline"]
# .get("reported_metric"); is_improvement = metric is not None and baseline
# is not None and metric > baseline — COMPUTED in code, never trusted from
# the LLM's response, and cross-checked against the LLM's verdict.
# ---------------------------------------------------------------------------

VERIFIER_SPEC = _valid_verifier_spec()


def _run_record(**overrides: object) -> dict:
    record = {
        "experiment_id": "exp-1",
        "succeeded": True,
        "exit_code": 0,
        "metric_value": 0.51,
        "stdout_excerpt": "epoch 10: exact_match=0.51",
        "stderr_excerpt": None,
        "environment": "python3.11, cuda12.1, 1x A100",
        "seed": "42",
        "runtime_seconds": 120.0,
        "artifacts": [],
    }
    record.update(overrides)
    return record


# BENCHMARK_SPEC.baseline.reported_metric == 0.42 (see fixture above).


# ---------------------------------------------------------------------------
# (a) evaluate_experiment: metric > baseline, LLM verdict new_best ->
# is_improvement True (computed), schema-valid.
# ---------------------------------------------------------------------------


def test_evaluate_experiment_improvement_new_best_is_schema_valid():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "new_best"
    assert report["is_improvement"] is True
    assert report["metric_value"] == 0.51
    assert report["baseline_metric"] == 0.42
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) metric <= baseline, LLM verdict valid_not_best -> is_improvement False.
# ---------------------------------------------------------------------------


def test_evaluate_experiment_no_improvement_valid_not_best_is_schema_valid():
    run_record = _run_record(metric_value=0.30)
    llm_response = {
        "verdict": "valid_not_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.30 does not beat the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "valid_not_best"
    assert report["is_improvement"] is False
    assert len(llm.calls) == 1


# --- degeneracy guard: a reproducibly-degenerate result is FORCED to "suspicious" (-> MOSS)
#     regardless of the LLM verdict, since reproduction can't rule out a broken experiment ---


def test_evaluate_experiment_no_separation_forced_suspicious():
    # method metric EXACTLY equals the baseline -> the experiment produced no distinguishing
    # power -> forced to "suspicious" even though the LLM (correctly) said valid_not_best.
    run_record = _run_record(metric_value=0.42)   # == baseline reported_metric 0.42
    llm_response = {"verdict": "valid_not_best", "suspicious": False,
                    "anti_cheat_flags": [], "reasoning": "ties the baseline."}
    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC,
                                 FakeLLM([_json(llm_response)]))
    assert list(EVALUATION_REPORT_VALIDATOR.iter_errors(report)) == []
    assert report["verdict"] == "suspicious" and report["suspicious"] is True
    assert "degenerate_result:no_separation_from_baseline" in report["anti_cheat_flags"]


def test_evaluate_experiment_non_finite_metric_forced_suspicious():
    run_record = _run_record(metric_value=float("inf"))
    llm_response = {"verdict": "new_best", "suspicious": False,
                    "anti_cheat_flags": [], "reasoning": "inf beats baseline."}
    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC,
                                 FakeLLM([_json(llm_response)]))
    assert report["verdict"] == "suspicious"
    assert "degenerate_result:non_finite_metric" in report["anti_cheat_flags"]


def test_evaluate_experiment_distinct_win_not_flagged_as_degenerate():
    # a normal distinct win must NOT be flagged (no false positive; a legit metric stands).
    run_record = _run_record(metric_value=0.51)   # != baseline 0.42
    llm_response = {"verdict": "new_best", "suspicious": False,
                    "anti_cheat_flags": [], "reasoning": "0.51 beats 0.42."}
    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC,
                                 FakeLLM([_json(llm_response)]))
    assert report["verdict"] == "new_best"
    assert not any("degenerate_result" in f for f in report["anti_cheat_flags"])


def test_evaluate_experiment_flags_under_mild_verdict_forced_suspicious():
    # The LLM names a real concern (anti_cheat_flags) but picks the milder valid_not_best
    # verdict -> the guard FORCES "suspicious" so it reaches MOSS instead of being buried.
    run_record = _run_record(metric_value=0.30)   # < baseline 0.42 -> is_improvement False
    llm_response = {"verdict": "valid_not_best", "suspicious": True,
                    "anti_cheat_flags": ["metric directionality confusion"],
                    "reasoning": "direction looks wrong."}
    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC,
                                 FakeLLM([_json(llm_response)]))
    assert report["verdict"] == "suspicious" and report["suspicious"] is True
    assert "metric directionality confusion" in report["anti_cheat_flags"]


def test_evaluate_experiment_conflict_with_flags_not_overridden():
    # "conflict" already routes to MOSS (result_inconsistency) with its own semantics -- the
    # guard must NOT rewrite it to "suspicious" just because it carries flags.
    run_record = _run_record(metric_value=0.51)
    llm_response = {"verdict": "conflict", "suspicious": True,
                    "anti_cheat_flags": ["contradicts prior finding F1"],
                    "reasoning": "contradicts F1."}
    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC,
                                 FakeLLM([_json(llm_response)]))
    assert report["verdict"] == "conflict"


# ---------------------------------------------------------------------------
# (c) guard: LLM claims new_best but metric <= baseline -> typed error.
# ---------------------------------------------------------------------------


def test_evaluate_experiment_new_best_claim_without_improvement_raises_typed_error():
    run_record = _run_record(metric_value=0.30)
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "claims improvement that did not happen",
    }
    llm = FakeLLM([_json(llm_response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (d) suspicious verdict with EMPTY anti_cheat_flags -> typed error.
# ---------------------------------------------------------------------------


def test_evaluate_experiment_suspicious_with_empty_flags_raises_typed_error():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "suspicious",
        "suspicious": True,
        "anti_cheat_flags": [],
        "reasoning": "something felt off but no flags listed",
    }
    llm = FakeLLM([_json(llm_response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (extra) invalid verdict is allowed regardless of is_improvement (e.g. run
# failed / metric None).
# ---------------------------------------------------------------------------


def test_evaluate_experiment_invalid_verdict_allowed_with_null_metric():
    run_record = _run_record(succeeded=False, exit_code=1, metric_value=None)
    llm_response = {
        "verdict": "invalid",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "run crashed before producing a metric",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "invalid"
    assert report["is_improvement"] is False
    assert report["metric_value"] is None
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (f) malformed JSON / invalid verdict enum -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_evaluate_experiment_malformed_json_retries_then_raises_typed_error():
    run_record = _run_record(metric_value=0.51)
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_evaluate_experiment_invalid_verdict_enum_retries_then_raises_typed_error():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "maybe",
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    llm = FakeLLM([_json(llm_response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_evaluate_experiment_recovers_after_one_bad_response():
    run_record = _run_record(metric_value=0.51)
    good_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    llm = FakeLLM(["not json at all", _json(good_response)])

    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    assert report["verdict"] == "new_best"
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# (M13 Task 5) running-best: prior_research_memory.best_metric, when present,
# is the improvement threshold instead of benchmark_spec.baseline.
# ---------------------------------------------------------------------------

PRIOR_RESEARCH_MEMORY = {
    "best_metric": 0.9,
    "best_result_ref": None,
    "verified_findings": [],
    "negative_results": [],
    "consolidated_insights": [],
    "divergence_seeds": [],
}


def test_evaluate_experiment_running_best_beats_prior_best_is_new_best():
    run_record = _run_record(metric_value=0.95)
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.95 beats the running best of 0.9.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(
        run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm,
        prior_research_memory=PRIOR_RESEARCH_MEMORY,
    )

    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "new_best"
    assert report["is_improvement"] is True
    # baseline_metric still reflects BenchmarkSpec.baseline, NOT the running
    # best -- only the improvement threshold changes.
    assert report["baseline_metric"] == 0.42


def test_evaluate_experiment_running_best_below_prior_best_is_valid_not_best_even_above_baseline():
    """0.88 > baseline (0.42) but <= the running best (0.9) -- proves the
    threshold used for is_improvement is the running best, not the
    baseline."""
    run_record = _run_record(metric_value=0.88)
    llm_response = {
        "verdict": "valid_not_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.88 beats the baseline but not the running best of 0.9.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(
        run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm,
        prior_research_memory=PRIOR_RESEARCH_MEMORY,
    )

    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "valid_not_best"
    assert report["is_improvement"] is False


def test_evaluate_experiment_no_prior_research_memory_falls_back_to_baseline():
    """No prior_research_memory (M12 caller, or a first-ever run) -> the
    threshold falls back to benchmark_spec.baseline, unchanged M12
    behavior."""
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    assert report["is_improvement"] is True
    assert report["baseline_metric"] == 0.42


# ---------------------------------------------------------------------------
# (M13 Task 5) conflict verdict: requires non-empty anti_cheat_flags (naming
# the contradicted verified finding).
# ---------------------------------------------------------------------------


def test_evaluate_experiment_conflict_with_flag_is_schema_valid():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "conflict",
        "suspicious": False,
        "anti_cheat_flags": [
            "contradicts verified finding: 'reranking never improves exact_match'"
        ],
        "reasoning": "This result contradicts a verified finding in prior_research_memory.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = evaluate_experiment(
        run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm,
        prior_research_memory=PRIOR_RESEARCH_MEMORY,
    )

    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "conflict"
    assert len(llm.calls) == 1


def test_evaluate_experiment_conflict_with_empty_flags_raises_typed_error():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "conflict",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "conflicts with a verified finding but names no flag",
    }
    llm = FakeLLM([_json(llm_response)] * (MAX_RETRIES + 1))

    with pytest.raises(EvaluatorParseError):
        evaluate_experiment(
            run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm,
            prior_research_memory=PRIOR_RESEARCH_MEMORY,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) verify_best_result: verified (empty flags ok) and failed (with a flag)
# both round-trip schema-valid; failed with EMPTY flags -> typed error.
# ---------------------------------------------------------------------------

EVALUATION_REPORT_NEW_BEST = {
    "verdict": "new_best",
    "metric_value": 0.51,
    "baseline_metric": 0.42,
    "is_improvement": True,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": None,
}


def test_verify_best_result_verified_with_empty_flags_is_schema_valid():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "verified",
        "checks": ["cross-seed: fixed seed=42", "held-out: metric on held-out split"],
        "anti_cheat_flags": [],
        "reasoning": "Holds up.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = verify_best_result(EVALUATION_REPORT_NEW_BEST, run_record, VERIFIER_SPEC, llm)

    errors = list(VERIFICATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "verified"
    assert llm.calls == []


def test_verify_best_result_does_not_treat_llm_cheat_opinion_as_record_failure():
    run_record = _run_record(metric_value=0.51)
    llm_response = {
        "verdict": "failed",
        "checks": ["held-out: metric computed on in-loop split, not held-out"],
        "anti_cheat_flags": ["held_out split not actually used"],
        "reasoning": "Result is not verified.",
    }
    llm = FakeLLM([_json(llm_response)])

    report = verify_best_result(EVALUATION_REPORT_NEW_BEST, run_record, VERIFIER_SPEC, llm)

    errors = list(VERIFICATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "verified"
    assert report["anti_cheat_flags"] == []
    assert llm.calls == []


def test_verify_repo_result_without_verifier_spec_does_not_invent_seed_requirement():
    """An absent verifier contract cannot be upgraded into a new replication gate.

    Repo-owned runs have already passed the repository comparison/semantic guardian
    before this advisory second opinion.  If no VerifierSpec froze a seed policy,
    the verifier must not reject a clean result merely because the record contains
    one seed.
    """
    run_record = _run_record(
        metric_value=0.995346,
        seed="0",
        environment=(
            "repo_improve|benchmark=example|verdict=new_best|"
            "baseline=0.99573|method=0.995346"
        ),
    )
    llm = FakeLLM([_json({
        "verdict": "failed",
        "checks": ["cross-seed: only one seed"],
        "anti_cheat_flags": ["single seed result"],
        "reasoning": "No replication was provided.",
    })])

    evaluation_report = {
        **EVALUATION_REPORT_NEW_BEST,
        "metric_value": 0.995346,
        "baseline_metric": 0.99573,
    }

    report = verify_best_result(evaluation_report, run_record, {}, llm)

    assert report["verdict"] == "verified"
    assert report["anti_cheat_flags"] == []
    assert any("metric_matches_run_record" in check for check in report["checks"])
    assert llm.calls == []


def test_verify_repo_result_with_verifier_spec_only_checks_recorded_number():
    """Repo verification must not turn optional replication into a new truth gate."""
    run_record = _run_record(
        metric_value=0.995280,
        seed="0",
        environment=(
            "repo_improve|benchmark=example|verdict=new_best|"
            "baseline=0.995730|method=0.995280"
        ),
    )
    evaluation_report = {
        **EVALUATION_REPORT_NEW_BEST,
        "metric_value": 0.995280,
        "baseline_metric": 0.995730,
        "reasoning": "single observed improvement; statistical confidence is not established",
    }
    llm = FakeLLM([_json({
        "verdict": "failed",
        "checks": ["cross-seed: only one seed", "ablation: within noise"],
        "anti_cheat_flags": ["single seed", "effect within noise"],
        "reasoning": "Not statistically established.",
    })])

    report = verify_best_result(
        evaluation_report, run_record, VERIFIER_SPEC, llm)

    assert report["verdict"] == "verified"
    assert report["anti_cheat_flags"] == []
    assert any("metric_matches_run_record" in check for check in report["checks"])
    assert any("baseline_matches_run_record" in check for check in report["checks"])
    assert llm.calls == []


def test_verify_repo_result_rejects_metric_record_mismatch_without_llm():
    run_record = _run_record(
        metric_value=0.995280,
        environment=(
            "repo_improve|benchmark=example|verdict=new_best|"
            "baseline=0.995730|method=0.900000"
        ),
    )
    evaluation_report = {
        **EVALUATION_REPORT_NEW_BEST,
        "metric_value": 0.995280,
        "baseline_metric": 0.995730,
    }
    llm = FakeLLM([_json({
        "verdict": "verified", "checks": [], "anti_cheat_flags": []})])

    report = verify_best_result(evaluation_report, run_record, {}, llm)

    assert report["verdict"] == "failed"
    assert any("method_metric_mismatch" in flag for flag in report["anti_cheat_flags"])
    assert llm.calls == []


def test_verify_repo_result_rejects_failed_run_even_when_metric_is_present():
    run_record = _run_record(
        succeeded=False,
        exit_code=1,
        metric_value=0.995280,
        environment=(
            "repo_improve|benchmark=example|verdict=new_best|"
            "baseline=0.995730|method=0.995280"
        ),
    )
    evaluation_report = {
        **EVALUATION_REPORT_NEW_BEST,
        "metric_value": 0.995280,
        "baseline_metric": 0.995730,
    }
    llm = FakeLLM([])

    report = verify_best_result(evaluation_report, run_record, {}, llm)

    assert report["verdict"] == "failed"
    assert "run_not_successful" in report["anti_cheat_flags"]
    assert llm.calls == []


def test_verify_success_ignores_statistical_and_cheat_opinions_when_number_matches():
    """Verification checks the record, not an LLM's confidence/cheating opinion."""
    run_record = _run_record(metric_value=0.51, seed="only-one-seed")
    evaluation_report = {
        **EVALUATION_REPORT_NEW_BEST,
        "anti_cheat_flags": ["single seed", "effect may be within noise"],
        "suspicious": True,
    }
    llm = FakeLLM([_json({
        "verdict": "failed",
        "checks": ["single seed"],
        "anti_cheat_flags": ["not statistically established"],
    })])

    report = verify_best_result(evaluation_report, run_record, VERIFIER_SPEC, llm)

    assert report["verdict"] == "verified"
    assert report["anti_cheat_flags"] == []
    assert any("metric_matches_run_record" in check for check in report["checks"])
    assert llm.calls == []


def test_verify_failed_run_confirms_recorded_failure_reason():
    run_record = _run_record(
        succeeded=False,
        exit_code=1,
        metric_value=None,
        stdout_excerpt="training stopped before evaluation",
        stderr_excerpt="CUDA out of memory",
    )
    evaluation_report = {
        "verdict": "invalid",
        "metric_value": None,
        "baseline_metric": 0.42,
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "The process exited before producing a metric.",
    }
    llm = FakeLLM([])

    report = verify_best_result(evaluation_report, run_record, VERIFIER_SPEC, llm)

    assert report["verdict"] == "verified"
    assert any("failure_confirmed" in check for check in report["checks"])
    assert "CUDA out of memory" in report["reasoning"]
    assert llm.calls == []


def test_verify_best_result_never_calls_the_llm():
    run_record = _run_record(metric_value=0.51)
    llm = FakeLLM(["not json at all"])

    report = verify_best_result(EVALUATION_REPORT_NEW_BEST, run_record, VERIFIER_SPEC, llm)

    assert report["verdict"] == "verified"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected (all 4 flows).
# ---------------------------------------------------------------------------


def test_select_benchmark_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(BENCHMARK_SPEC)])

    select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_select_benchmark_system_prompt()


def test_select_benchmark_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(BENCHMARK_SPEC)])

    select_benchmark(RESEARCH_SPEC, LITERATURE_BRIEF, llm, prompt_patch="STEER: prefer X")

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_prepare_verifier_no_prompt_patch_leaves_system_message_unchanged():
    response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM([_json(response)])

    prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_prepare_verifier_system_prompt()


def test_prepare_verifier_prompt_patch_appended_to_system_message():
    response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM([_json(response)])

    prepare_verifier(BENCHMARK_SPEC, RESEARCH_SPEC, llm, prompt_patch="STEER: prefer X")

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_evaluate_experiment_no_prompt_patch_leaves_system_message_unchanged():
    run_record = _run_record(metric_value=0.51)
    llm_response = {"verdict": "new_best", "suspicious": False, "anti_cheat_flags": []}
    llm = FakeLLM([_json(llm_response)])

    evaluate_experiment(run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_evaluate_experiment_system_prompt()


def test_evaluate_experiment_prompt_patch_appended_to_system_message():
    run_record = _run_record(metric_value=0.51)
    llm_response = {"verdict": "new_best", "suspicious": False, "anti_cheat_flags": []}
    llm = FakeLLM([_json(llm_response)])

    evaluate_experiment(
        run_record, VERIFIER_SPEC, BENCHMARK_SPEC, llm, prompt_patch="STEER: prefer X"
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_verify_best_result_prompt_patch_cannot_change_deterministic_verdict():
    run_record = _run_record(metric_value=0.51)
    llm = FakeLLM([])

    report = verify_best_result(
        EVALUATION_REPORT_NEW_BEST, run_record, VERIFIER_SPEC, llm,
        prompt_patch="STEER: prefer X",
    )

    assert report["verdict"] == "verified"
    assert llm.calls == []


# --- repo-aware short-circuit: trust repo_improve's own trusted-eval + 2-seed verdict ---


def test_repo_evaluation_report_parses_repo_env():
    from autoresearch.agents.evaluator_agent.core import _repo_evaluation_report
    env = ("repo_improve|benchmark=nanogpt-5min-single-gpu|verdict=valid_not_best"
           "|baseline=1.019|method=1.023|gpu-worker-02 card6")
    r = _repo_evaluation_report(env)
    assert r["verdict"] == "valid_not_best"
    assert r["is_improvement"] is False and r["suspicious"] is False
    assert r["anti_cheat_flags"] == []
    assert r["baseline_metric"] == 1.019 and r["metric_value"] == 1.023
    assert "2 seeds" not in r["reasoning"]
    assert "mapped" in r["reasoning"]


def test_repo_evaluation_report_none_for_normal_env():
    from autoresearch.agents.evaluator_agent.core import _repo_evaluation_report
    assert _repo_evaluation_report("docker/openhands") is None
    assert _repo_evaluation_report("repo_improve|verdict=bogus") is None  # bad verdict -> None


def test_evaluate_experiment_repo_short_circuits_without_llm():
    from autoresearch.agents.evaluator_agent.core import evaluate_experiment
    run_record = {
        "succeeded": True, "exit_code": 0, "metric_value": 1.023,
        "stdout_excerpt": "repo report", "stderr_excerpt": None,
        "environment": ("repo_improve|benchmark=b|verdict=new_best|baseline=1.03"
                        "|method=1.023|gpu-worker-02 card6"),
        "seed": "0", "runtime_seconds": None, "artifacts": [],
    }
    llm = FakeLLM([])  # must NOT be called
    report = evaluate_experiment(run_record, {}, {}, llm)
    assert report["verdict"] == "new_best" and report["is_improvement"] is True
    assert report["suspicious"] is False and report["anti_cheat_flags"] == []
    assert list(EVALUATION_REPORT_VALIDATOR.iter_errors(report)) == []
    assert llm.calls == []
