"""Host-side aggregation for literature-grounded, main-method novelty audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CORE_RELATIONS = {
    "same", "different", "unknown",
    # Legacy aliases remain readable for persisted audit artifacts.
    "equivalent", "subsumes", "overlaps", "distinct",
}


class NoveltyHarnessInputError(ValueError):
    """The model did not supply a complete, grounded harness packet."""


@dataclass(frozen=True)
class NoveltyDecision:
    novelty_status: str
    verdict: str
    decisive_prior_work_ids: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "novelty_status": self.novelty_status,
            "verdict": self.verdict,
            "decisive_prior_work_ids": list(self.decisive_prior_work_ids),
            "rationale": self.rationale,
        }


def _paper_ids(literature_brief: dict) -> set[str]:
    ids: set[str] = set()
    for field in ("key_papers", "related_work"):
        for paper in literature_brief.get(field) or []:
            if not isinstance(paper, dict):
                continue
            for value in (paper.get("id_or_url"), paper.get("title")):
                if isinstance(value, str) and value.strip():
                    ids.add(value.strip())
    return ids


def _require_text(obj: dict, key: str, where: str, errors: list[str]) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}.{key} must be non-empty text")
        return ""
    return value.strip()


def validate_harness_packet(report: dict, literature_brief: dict) -> list[str]:
    """Return structural/grounding errors; semantic uncertainty is not an error."""

    errors: list[str] = []
    signature = report.get("method_signature")
    if not isinstance(signature, dict):
        return ["method_signature must be an object"]
    primary = signature.get("primary_core")
    if not isinstance(primary, dict):
        return ["method_signature.primary_core must be an object"]

    core_id = _require_text(primary, "core_id", "primary_core", errors)
    for key in ("intervention", "causal_mechanism", "pipeline_role", "expected_effect"):
        _require_text(primary, key, "primary_core", errors)
    searched_primary = literature_brief.get("searched_method_primary_core")
    if isinstance(searched_primary, dict) and primary != searched_primary:
        errors.append(
            "method_signature.primary_core must exactly reuse the core frozen before retrieval")

    comparisons = report.get("prior_work_comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append("prior_work_comparisons must contain at least one grounded comparison")
        comparisons = []

    known_papers = _paper_ids(literature_brief)
    comparison_ids: list[str] = []
    for index, comparison in enumerate(comparisons):
        where = f"prior_work_comparisons[{index}]"
        if not isinstance(comparison, dict):
            errors.append(f"{where} must be an object")
            continue
        prior_id = _require_text(comparison, "prior_work_id", where, errors)
        comparison_ids.append(prior_id)
        if prior_id and prior_id not in known_papers:
            errors.append(f"{where}.prior_work_id is not grounded in LiteratureBrief: {prior_id}")
        relation = comparison.get("primary_core_relation")
        if relation not in CORE_RELATIONS:
            errors.append(f"{where}.primary_core_relation is invalid")
        evidence = comparison.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            errors.append(f"{where}.evidence must contain at least one source-grounded statement")

    coverage = report.get("retrieval_coverage")
    if not isinstance(coverage, dict):
        errors.append("retrieval_coverage must be an object")
    else:
        covered_core_ids = coverage.get("covered_core_ids")
        if not isinstance(covered_core_ids, list) or core_id not in covered_core_ids:
            errors.append("retrieval_coverage must include the primary core_id")
        compared_ids = coverage.get("compared_prior_work_ids")
        if not isinstance(compared_ids, list) or set(compared_ids) != set(comparison_ids):
            errors.append("retrieval_coverage.compared_prior_work_ids must match comparisons")
        unresolved = coverage.get("unresolved_candidates")
        if not isinstance(unresolved, list) or not all(isinstance(x, str) for x in unresolved):
            errors.append("retrieval_coverage.unresolved_candidates must be an array of strings")

    return errors


def decide_novelty(report: dict, literature_brief: dict) -> NoveltyDecision:
    """Apply the core-first truth table after validating the evidence packet."""

    errors = validate_harness_packet(report, literature_brief)
    if errors:
        raise NoveltyHarnessInputError("; ".join(errors))

    comparisons = report["prior_work_comparisons"]
    unresolved = report["retrieval_coverage"]["unresolved_candidates"]

    same_core = [
        item for item in comparisons
        if item["primary_core_relation"] in {"same", "equivalent", "subsumes"}
    ]
    if same_core:
        return NoveltyDecision(
            "known",
            "not_novel",
            tuple(item["prior_work_id"] for item in same_core),
            "Prior work covers the same main causal method; secondary differences do not change that identity.",
        )

    unresolved_relations = [
        item for item in comparisons
        if item["primary_core_relation"] in {"unknown", "overlaps"}
    ]
    # A finite search can always speculate that an unseen similar paper may exist.  Only a
    # concrete paper already grounded in the LiteratureBrief is an unresolved comparison leaf;
    # generic search-gap prose remains a disclosed limitation, not a permanent novelty veto.
    known_papers = _paper_ids(literature_brief)
    grounded_unresolved = tuple(
        item.strip() for item in unresolved
        if isinstance(item, str) and item.strip() in known_papers
    )
    if not comparisons or unresolved_relations or grounded_unresolved:
        return NoveltyDecision(
            "uncertain",
            "evidence_insufficient",
            tuple(item["prior_work_id"] for item in unresolved_relations)
            + grounded_unresolved,
            "The main-method relation is unresolved.",
        )

    return NoveltyDecision(
        "novel",
        "pass",
        tuple(),
        "All grounded closest-work comparisons are distinct at the primary causal mechanism.",
    )


def apply_novelty_harness(report: dict, literature_brief: dict) -> dict:
    """Return a copy whose status/verdict come only from the host truth table."""

    decision = decide_novelty(report, literature_brief)
    normalized = dict(report)
    normalized["novelty_status"] = decision.novelty_status
    normalized["verdict"] = decision.verdict
    normalized["harness_decision"] = decision.as_dict()
    return normalized


__all__ = [
    "NoveltyDecision",
    "NoveltyHarnessInputError",
    "apply_novelty_harness",
    "decide_novelty",
    "validate_harness_packet",
]
