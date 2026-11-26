"""
Tests for ReportAgent's pure core. / ReportAgent 纯核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
``generate_report`` is PURE: it takes a ResearchMemory, an optional
ResearchSpec, and an llm_client, and returns a schema-valid Report dict.

The DETERMINISTIC grounding guard (crown jewel here, the honest-report
anti-hallucination core): the LLM MUST NOT retain a ``verified_finding`` or
a ``negative_result`` that is not present verbatim in the ResearchMemory it
was given —
  ``set(report["verified_findings"]) ⊆ set(research_memory["verified_findings"])``
  ``set(report["negative_results"]) ⊆ set(research_memory["negative_results"])``
Any extra (fabricated/paraphrased) claim is dropped deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.report_agent.core import (
    MAX_RETRIES,
    ReportParseError,
    _load_generate_report_system_prompt,
    generate_report,
)
from autoresearch.agents.report_agent.tests.conftest import FakeLLM, FakeLLMResult

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


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

EMPTY_RESEARCH_MEMORY = {
    "best_metric": None,
    "best_result_ref": None,
    "verified_findings": [],
    "negative_results": [],
    "excluded_hypotheses": [],
    "consolidated_insights": [],
    "divergence_seeds": [],
    "next_steps": [],
}

RESEARCH_SPEC = {
    "research_question": "Does technique X improve the metric?",
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


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


# ---------------------------------------------------------------------------
# (a) valid: report.verified_findings/negative_results subsets of memory's ->
# schema-valid Report with all 8 classification arrays present.
# ---------------------------------------------------------------------------


def test_generate_report_valid_returns_schema_valid_report():
    response = _report()
    llm = FakeLLM([_json(response)])

    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == ["Technique X improves metric over baseline."]
    assert report["negative_results"] == ["Technique Z degrades the metric."]
    assert len(llm.calls) == 1


def test_generate_report_works_without_research_spec():
    response = _report()
    llm = FakeLLM([_json(response)])

    report = generate_report(RESEARCH_MEMORY, None, llm)

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert len(llm.calls) == 1


def test_report_prompt_requests_substantive_theory_body_without_benchmark():
    prompt = _load_generate_report_system_prompt()
    assert "actual `LiteratureBrief`" in prompt
    assert "dedicated plain-Markdown writer" in prompt


def test_report_user_message_includes_reviewed_research_context():
    from autoresearch.agents.report_agent.core import _build_generate_report_user_message
    msg = _build_generate_report_user_message(
        RESEARCH_MEMORY, RESEARCH_SPEC,
        {"literature_brief": {"key_papers": [{"title": "Rate Distortion"}]},
         "confirmed_method": {"approach": "derive theorem"}},
    )
    assert "Rate Distortion" in msg
    assert "derive theorem" in msg


def test_report_prompts_compact_unbounded_history_without_mutating_memory():
    from autoresearch.agents.report_agent.core import _build_generate_report_user_message

    secret = "FULL-IMMUTABLE-EVIDENCE-" + ("x" * 5000)
    memory = {
        **RESEARCH_MEMORY,
        "negative_results": [secret],
        "component_history": [
            {
                "sequence": index,
                "component_id": f"component-{index}",
                "status": "qualified",
                "metric": 0.4,
                "checkpoint": secret,
                "changes_summary": secret,
                "acceptance_result": {
                    "passed": True,
                    "candidate_metric": 0.4,
                    "measurement_evidence": secret,
                },
            }
            for index in range(40)
        ],
    }

    msg = _build_generate_report_user_message(memory, RESEARCH_SPEC)

    assert secret not in msg
    assert "FULL-IMMUTABLE-EVIDENCE-" in msg
    assert '"component_history_omitted_from_prompt": 8' in msg
    assert '"sequence": 39' in msg
    assert '"candidate_metric": 0.4' in msg
    assert "measurement_evidence" not in msg
    assert memory["negative_results"] == [secret]
    assert memory["component_history"][0]["checkpoint"] == secret


def test_long_report_body_uses_same_compact_memory_view():
    secret = "FULL-HISTORY-PAYLOAD-" + ("z" * 5000)
    memory = {
        **RESEARCH_MEMORY,
        "component_history": [{
            "sequence": 1,
            "component_id": "candidate",
            "status": "qualified",
            "checkpoint": secret,
            "changes_summary": "bounded summary",
        }],
    }
    body = "# Report\n\n" + ("Grounded result. " * 80)
    llm = FakeLLM([_json(_report()), body])

    report = generate_report(
        memory, RESEARCH_SPEC, llm,
        research_context={"confirmed_method": {"approach": "bounded method"}},
    )

    assert report["report_body"] == body.rstrip()
    assert len(llm.calls) == 2
    assert all(secret not in message["content"] for call in llm.calls for message in call)


def test_generate_report_attaches_long_markdown_from_reviewed_context():
    body = "# 理论报告\n\n" + ("具体推导与文献定位。" * 120)
    llm = FakeLLM([_json(_report()), body])
    report = generate_report(
        RESEARCH_MEMORY, RESEARCH_SPEC, llm,
        research_context={
            "literature_brief": {"key_papers": [{"title": "Rate Distortion"}]},
            "confirmed_method": {"approach": "derive theorem"},
        },
    )
    assert report["report_body"].startswith("# 理论报告")
    assert len(report["report_body"]) >= 800


def test_long_report_prompt_forbids_promoting_plans_to_results():
    from autoresearch.agents.report_agent.core import _REPORT_BODY_SYSTEM
    assert "do not label any theorem or proposition as proved" in _REPORT_BODY_SYSTEM
    assert "do not turn expected/example numbers into measured results" in _REPORT_BODY_SYSTEM


def test_long_report_keeps_attempt_reviews_separate_from_performance():
    from autoresearch.agents.report_agent.core import _REPORT_BODY_SYSTEM

    assert "same attempt/checkpoint" in _REPORT_BODY_SYSTEM
    assert "no improvement does not imply off-method" in _REPORT_BODY_SYSTEM
    assert "faithful implementation can fail its gain acceptance check" in _REPORT_BODY_SYSTEM
    assert "Do not transfer an earlier attempt's contract violation" in _REPORT_BODY_SYSTEM
    assert "do not erase the earlier violation" in _REPORT_BODY_SYSTEM


# --- T4: guaranteed blind-review gate --------------------------------------


def test_blind_review_flags_summary_overclaim():
    from autoresearch.agents.report_agent.core import _blind_review
    llm = FakeLLM([json.dumps(
        {"verdict": "FLAGGED", "findings": ["summary claims an improvement, verified_findings empty"]})])
    rep = {"summary": "The method clearly improves accuracy over the baseline.",
           "verified_findings": [], "negative_results": []}
    rev = _blind_review(llm, rep)
    assert rev["verdict"] == "FLAGGED"
    assert rev["findings"]


def test_blind_review_fails_open_on_bad_llm():
    from autoresearch.agents.report_agent.core import _blind_review
    rev = _blind_review(FakeLLM(["not json at all"]),
                        {"summary": "x", "verified_findings": [], "negative_results": []})
    assert rev["verdict"] == "unreviewed"


def test_generate_report_attaches_blind_review_when_reviewer_given():
    reviewer = FakeLLM([json.dumps({"verdict": "CLEAN", "findings": []})])
    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, FakeLLM([_json(_report())]),
                             reviewer_llm=reviewer)
    assert report["review"]["verdict"] == "CLEAN"
    assert list(REPORT_VALIDATOR.iter_errors(report)) == []  # schema-valid with review
    assert len(reviewer.calls) == 1


def test_generate_report_no_reviewer_leaves_report_unchanged():
    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, FakeLLM([_json(_report())]))
    assert "review" not in report  # default: byte-identical to prior behavior


# --- T8: failure-mode clustering -------------------------------------------


def test_cluster_failure_modes_groups_by_mode():
    from autoresearch.agents.report_agent.core import _cluster_failure_modes
    negs = [
        "Result flagged suspicious; flags: missing held_out validation; seed mismatch with protocol",
        "forbidden_files touched during execution",
        "metric improbably high",
    ]
    clusters = _cluster_failure_modes(negs)
    assert "validation_and_leakage" in clusters
    assert "reproducibility" in clusters
    assert "anti_cheat_violation" in clusters
    assert "degenerate_or_implausible_metric" in clusters


def test_cluster_failure_modes_empty_for_plain_negatives():
    from autoresearch.agents.report_agent.core import _cluster_failure_modes
    assert _cluster_failure_modes(["Technique Z simply degrades the metric."]) == {}


def test_generate_report_attaches_failure_clusters():
    mem = {
        "verified_findings": [],
        "negative_results": ["Result flagged suspicious; flags: missing held_out validation; "
                             "forbidden_files touched"],
    }
    resp = {**_report(), "verified_findings": [],
            "negative_results": mem["negative_results"]}
    report = generate_report(mem, RESEARCH_SPEC, FakeLLM([_json(resp)]))
    assert "validation_and_leakage" in report["failure_clusters"]
    assert "anti_cheat_violation" in report["failure_clusters"]
    assert list(REPORT_VALIDATOR.iter_errors(report)) == []


# --- T5: deterministic number-traceability ---------------------------------


def test_untraceable_numbers_flags_decimal_absent_from_memory():
    from autoresearch.agents.report_agent.core import _untraceable_numbers
    mem = {"verified_findings": [], "best_metric": 0.42}
    assert _untraceable_numbers("accuracy was 0.42 and also 0.99", mem) == ["0.99"]
    assert _untraceable_numbers("k=5 in 2024 with 200 items", mem) == []  # integers ignored


def test_generate_report_flags_untraceable_summary_number():
    resp = {**_report(), "summary": "The method reached 0.999 silhouette, a clear win."}
    reviewer = FakeLLM([json.dumps({"verdict": "CLEAN", "findings": []})])
    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, FakeLLM([_json(resp)]),
                             reviewer_llm=reviewer)
    assert report["review"]["verdict"] == "FLAGGED"
    assert any("0.999" in f for f in report["review"]["findings"])


# ---------------------------------------------------------------------------
# (b) grounding guard: a fabricated verified_finding (or negative_result) NOT
# present in ResearchMemory is dropped without failing the whole report.
# ---------------------------------------------------------------------------


def test_generate_report_drops_fabricated_verified_finding():
    bad_response = _report(
        verified_findings=["A claim that never appeared in ResearchMemory."]
    )
    llm = FakeLLM([_json(bad_response)])

    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert report["verified_findings"] == []
    assert len(llm.calls) == 1


def test_generate_report_drops_fabricated_negative_result():
    bad_response = _report(
        negative_results=["A negative result that never appeared in ResearchMemory."]
    )
    llm = FakeLLM([_json(bad_response)])

    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert report["negative_results"] == []
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (c) empty ResearchMemory: an honest empty-verified_findings Report is valid;
# a NON-empty verified_finding on empty memory is fabrication -> dropped.
# ---------------------------------------------------------------------------


def test_generate_report_empty_memory_with_empty_verified_findings_is_valid():
    response = _report(verified_findings=[], negative_results=[])
    llm = FakeLLM([_json(response)])

    report = generate_report(EMPTY_RESEARCH_MEMORY, None, llm)

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == []
    assert report["negative_results"] == []
    assert len(llm.calls) == 1


def test_generate_report_empty_memory_drops_nonempty_verified_finding():
    bad_response = _report(verified_findings=["Fabricated finding."], negative_results=[])
    llm = FakeLLM([_json(bad_response)])

    report = generate_report(EMPTY_RESEARCH_MEMORY, None, llm)

    assert report["verified_findings"] == []
    assert len(llm.calls) == 1


def test_generate_report_retries_short_long_body():
    body = "# 完整报告\n\n" + ("详细结果与限制。" * 120)
    llm = FakeLLM([_json(_report()), "too short", body])

    report = generate_report(
        RESEARCH_MEMORY,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert report["report_body"] == body.strip()
    assert len(llm.calls) == 3
    long_body_system = llm.calls[-1][0]["content"]
    assert "PROJECT-LOCAL ACADEMIC WRITING SKILL" in long_body_system
    assert "off-method observation" in long_body_system


def test_report_body_retry_expands_token_budget():
    body = "# Complete report\n\n" + ("Grounded result and limitation. " * 80)
    llm = FakeLLM([_json(_report()), "", "too short", body])

    report = generate_report(
        RESEARCH_MEMORY,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert report["report_body"] == body.strip()
    assert [call["max_tokens"] for call in llm.call_kwargs] == [
        None, 262144, 262144, 262144,
    ]


def test_long_report_length_truncation_continues_only_the_missing_suffix():
    body = "# Complete report\n\n" + ("Grounded result and limitation. " * 80)
    split = 900
    llm = FakeLLM([
        _json(_report()),
        FakeLLMResult(body[:split], {}, "0", finish_reason="length"),
        FakeLLMResult(body[split:], {}, "0", finish_reason="stop"),
    ])

    report = generate_report(
        RESEARCH_MEMORY,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert report["report_body"] == body.strip()
    assert llm.calls[2][-2] == {"role": "assistant", "content": body[:split]}
    assert "return only the missing suffix" in llm.calls[2][-1]["content"]


def test_long_report_continuation_accepts_provider_full_restart():
    partial = "# Incomplete report\n\n" + ("Grounded result. " * 20)
    complete = "# Complete report\n\n" + ("Grounded result and limitation. " * 80)
    llm = FakeLLM([
        _json(_report()),
        FakeLLMResult(partial, {}, "0", finish_reason="length"),
        FakeLLMResult(complete, {}, "0", finish_reason="stop"),
    ])

    report = generate_report(
        RESEARCH_MEMORY,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert report["report_body"] == complete.strip()


def test_long_report_retries_speedup_that_uses_the_wrong_baseline():
    memory = {
        **RESEARCH_MEMORY,
        "selected_result": {
            "baseline_metric": 0.131625,
            "method_metric": 0.054757,
            "lower_is_better": True,
            "attempt_review_evidence": [{
                "verified_output_fields": {
                    "/notes/reward_json/speedup": 1.81,
                },
            }],
        },
    }
    wrong = (
        "# Report\n\nThe measured baseline 0.131625 and method 0.054757 produced a "
        "1.81x speedup. " + ("Grounded implementation and limitations. " * 40)
    )
    corrected = (
        "# Report\n\nThe measured baseline 0.131625 and method 0.054757 produced a "
        "2.404x measured speedup. The evaluator separately reports 1.81x relative "
        "to its fixed scoring anchor, not the reproduced baseline. "
        + ("Grounded implementation and limitations. " * 40)
    )
    llm = FakeLLM([_json(_report()), wrong, corrected])

    report = generate_report(
        memory,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert report["report_body"] == corrected.strip()
    assert len(llm.calls) == 3
    retry_message = llm.calls[-1][-1]["content"]
    assert "0.131625" in retry_message
    assert "0.054757" in retry_message
    assert "2.403802" in retry_message
    assert "fixed scoring anchor" in retry_message


def test_long_report_repairs_auditable_comparisons_after_retries_are_exhausted():
    memory = {
        **RESEARCH_MEMORY,
        "selected_result": {
            "baseline_metric": 5.070274,
            "method_metric": 0.139595,
            "lower_is_better": True,
            "attempt_review_evidence": [{
                "verified_output_fields": {
                    "/notes/reward_json/speedup": 5.070274,
                },
            }],
        },
    }
    wrong = (
        "# Report\n\nThe reproduced baseline 5.070274 and method 0.139595 produced a "
        "5.07027x speedup and a 100% improvement. "
        + ("Grounded implementation and limitations. " * 40)
    )
    llm = FakeLLM([_json(_report())] + [wrong] * (MAX_RETRIES + 1))

    report = generate_report(
        memory,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert "36.321315x speedup" in report["report_body"]
    assert "97.246796% improvement" in report["report_body"]
    assert "5.07027x speedup" not in report["report_body"]
    assert "100% improvement" not in report["report_body"]
    assert len(llm.calls) == MAX_RETRIES + 2


def test_generate_report_falls_back_when_machine_audit_never_returns_json():
    body = "# Complete report\n\n" + ("Grounded result and limitation. " * 80)
    llm = FakeLLM(["not json"] * (MAX_RETRIES + 1) + [body])

    report = generate_report(
        RESEARCH_MEMORY,
        RESEARCH_SPEC,
        llm,
        research_context={"confirmed_method": {"approach": "Technique X"}},
    )

    assert report["verified_findings"] == RESEARCH_MEMORY["verified_findings"]
    assert report["negative_results"] == RESEARCH_MEMORY["negative_results"]
    assert report["experiment_ledger"] == []
    assert "conservatively reconstructed" in report["analysis_caveats"][0]
    assert report["report_body"] == body.strip()


# ---------------------------------------------------------------------------
# (d) malformed JSON / missing a required array -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_generate_report_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(ReportParseError):
        generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_generate_report_missing_required_array_retries_then_raises_typed_error():
    bad_report = _report()
    del bad_report["known_risks"]
    llm = FakeLLM([_json(bad_report)] * (MAX_RETRIES + 1))

    with pytest.raises(ReportParseError):
        generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_generate_report_non_dict_response_retries_then_raises_typed_error():
    llm = FakeLLM(["[1, 2, 3]"] * (MAX_RETRIES + 1))

    with pytest.raises(ReportParseError):
        generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_generate_report_recovers_after_one_bad_response():
    good_response = _report()
    llm = FakeLLM(["not json at all", _json(good_response)])

    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == []
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_generate_report_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(_report())])

    generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_generate_report_system_prompt()


def test_generate_report_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(_report())])

    generate_report(
        RESEARCH_MEMORY, RESEARCH_SPEC, llm, prompt_patch="STEER: prefer X"
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


# ---------------------------------------------------------------------------
# (e) citation-audit consumption (EvidenceAuditAgent milestone Task 5): a
# CitationAuditReport flagging a claim as citation-unsupported -> that claim
# is DROPPED from verified_findings/negative_results, reinforcing the
# grounding guard above (honest report). ``citation_audit_report=None`` (the
# default) is backward-safe -- behaves exactly as before this parameter
# existed (proven throughout (a)-(d) above, none of which pass it).
# ---------------------------------------------------------------------------


def test_generate_report_drops_citation_unsupported_verified_finding():
    response = _report(
        verified_findings=[
            "Technique X improves metric over baseline.",
            "Batch size 64 is stable.",
        ]
    )
    llm = FakeLLM([_json(response)])
    citation_audit_report = {
        "unsupported_claims": [
            {"claim": "Batch size 64 is stable.", "why": "No citation found."}
        ],
        "checked_count": 2,
    }

    memory = dict(RESEARCH_MEMORY)
    memory["verified_findings"] = [
        "Technique X improves metric over baseline.",
        "Batch size 64 is stable.",
    ]

    report = generate_report(
        memory, RESEARCH_SPEC, llm, citation_audit_report=citation_audit_report
    )

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == ["Technique X improves metric over baseline."]
    assert "Batch size 64 is stable." not in report["verified_findings"]


def test_generate_report_drops_citation_unsupported_negative_result():
    response = _report(negative_results=["Technique Z degrades the metric."])
    llm = FakeLLM([_json(response)])
    citation_audit_report = {
        "unsupported_claims": [
            {"claim": "Technique Z degrades the metric.", "why": "No citation found."}
        ],
        "checked_count": 1,
    }

    report = generate_report(
        RESEARCH_MEMORY, RESEARCH_SPEC, llm, citation_audit_report=citation_audit_report
    )

    assert report["negative_results"] == []


def test_generate_report_no_citation_audit_report_is_backward_safe():
    llm = FakeLLM([_json(_report())])

    report = generate_report(RESEARCH_MEMORY, RESEARCH_SPEC, llm, citation_audit_report=None)

    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == ["Technique X improves metric over baseline."]
    assert report["negative_results"] == ["Technique Z degrades the metric."]


def test_generate_report_citation_audit_report_with_no_unsupported_claims_is_noop():
    llm = FakeLLM([_json(_report())])
    citation_audit_report = {"unsupported_claims": [], "checked_count": 5}

    report = generate_report(
        RESEARCH_MEMORY, RESEARCH_SPEC, llm, citation_audit_report=citation_audit_report
    )

    assert report["verified_findings"] == ["Technique X improves metric over baseline."]
    assert report["negative_results"] == ["Technique Z degrades the metric."]


def test_report_prompt_constrains_ledger_verdict_to_experiment_enum():
    # A3: the ledger verdict must match the experiment verdict enum, not an invented
    # label like "partial_success" that contradicts the pass/fail/inconclusive result.
    p = _load_generate_report_system_prompt()
    assert '"pass"' in p and '"fail"' in p and '"inconclusive"' in p
    assert "partial_success" in p   # named explicitly as a label to AVOID
