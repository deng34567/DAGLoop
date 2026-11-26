from __future__ import annotations

import copy

import pytest

from autoresearch.agents.novelty_audit_agent.harness import (
    NoveltyHarnessInputError,
    decide_novelty,
)


LITERATURE = {
    "key_papers": [
        {"title": "Prior A", "id_or_url": "paper:a"},
        {"title": "Prior B", "id_or_url": "paper:b"},
    ],
    "related_work": [],
}


def _packet(*relations: tuple[str, str]) -> dict:
    comparisons = [
        {
            "prior_work_id": prior_id,
            "primary_core_relation": core_relation,
            "evidence": [f"{prior_id} states the compared mechanism."],
        }
        for prior_id, core_relation in relations
    ]
    return {
        "method_signature": {
            "primary_core": {
                "core_id": "core-1",
                "intervention": "replace the baseline transition",
                "causal_mechanism": "change the information path",
                "pipeline_role": "primary transformation",
                "expected_effect": "lower the target loss",
            },
        },
        "prior_work_comparisons": comparisons,
        "retrieval_coverage": {
            "covered_core_ids": ["core-1"],
            "compared_prior_work_ids": [item["prior_work_id"] for item in comparisons],
            "unresolved_candidates": [],
        },
    }


@pytest.mark.parametrize(
    ("packet", "expected"),
    [
        (_packet(("paper:a", "same")), "known"),
        (_packet(("paper:a", "different")), "novel"),
    ],
    ids=[
        "same-main-method",
        "different-main-method",
    ],
)
def test_core_first_truth_table(packet, expected):
    assert decide_novelty(packet, LITERATURE).novelty_status == expected


def test_generic_unseen_work_speculation_does_not_veto_distinct_comparisons():
    packet = _packet(("paper:a", "different"))
    packet["retrieval_coverage"]["unresolved_candidates"] = ["possible close paper"]
    decision = decide_novelty(packet, LITERATURE)
    assert decision.novelty_status == "novel"
    assert decision.verdict == "pass"


def test_concrete_grounded_unresolved_paper_is_uncertain():
    packet = _packet(("paper:a", "different"))
    packet["retrieval_coverage"]["unresolved_candidates"] = ["paper:b"]
    decision = decide_novelty(packet, LITERATURE)
    assert decision.novelty_status == "uncertain"
    assert decision.verdict == "evidence_insufficient"


def test_same_core_is_decisive_even_if_generic_search_gaps_remain():
    packet = _packet(("paper:a", "same"))
    packet["retrieval_coverage"]["unresolved_candidates"] = ["possible unseen work"]
    decision = decide_novelty(packet, LITERATURE)
    assert decision.novelty_status == "known"
    assert decision.verdict == "not_novel"


def test_targeted_search_and_comparison_must_reuse_the_same_primary_core():
    literature = copy.deepcopy(LITERATURE)
    literature["searched_method_primary_core"] = copy.deepcopy(
        _packet(("paper:a", "different"))["method_signature"]["primary_core"])
    packet = _packet(("paper:a", "different"))
    assert decide_novelty(packet, literature).novelty_status == "novel"

    packet["method_signature"]["primary_core"]["intervention"] = "a different intervention"
    with pytest.raises(NoveltyHarnessInputError, match="exactly reuse"):
        decide_novelty(packet, literature)


def test_legacy_overlap_is_uncertain_instead_of_assumed_different():
    decision = decide_novelty(_packet(("paper:a", "overlaps")), LITERATURE)
    assert decision.novelty_status == "uncertain"
    assert decision.verdict == "evidence_insufficient"


def test_comparison_order_does_not_change_decision():
    packet = _packet(
        ("paper:a", "different"),
        ("paper:b", "same"),
    )
    reversed_packet = copy.deepcopy(packet)
    reversed_packet["prior_work_comparisons"].reverse()
    reversed_packet["retrieval_coverage"]["compared_prior_work_ids"].reverse()

    first = decide_novelty(packet, LITERATURE)
    second = decide_novelty(reversed_packet, LITERATURE)

    assert first.novelty_status == second.novelty_status == "known"
    assert first.verdict == second.verdict == "not_novel"
