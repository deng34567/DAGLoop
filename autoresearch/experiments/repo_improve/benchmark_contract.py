"""Canonical fairness contract shared by repository baseline and method phases."""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
from copy import deepcopy
from typing import Any


class BenchmarkContractError(ValueError):
    """Raised when a benchmark contract cannot be frozen safely."""


def reconcile_repository_evaluation_semantics(
    evaluation_semantics: Any, baseline_result: Any,
) -> Any:
    """Resolve repository-defined metric placeholders from a reproduced baseline.

    ReinterpretingSpec runs before the repository sandbox, so repository-defined
    tasks may temporarily carry a guessed metric name or direction. Once the
    official live-repository baseline has run, its measured metric contract is
    the first grounded authority available to downstream agents. User-defined
    comparison modes are intentionally left untouched.
    """
    if (not isinstance(evaluation_semantics, dict)
            or evaluation_semantics.get("comparison_mode") != "repository_defined"
            or not isinstance(baseline_result, dict)
            or baseline_result.get("source_kind") != "live_repo"
            or baseline_result.get("reproduced") is not True
            or not isinstance(baseline_result.get("lower_is_better"), bool)):
        return evaluation_semantics
    metric_name = baseline_result.get("metric_name")
    if not isinstance(metric_name, str) or not metric_name.strip():
        return evaluation_semantics
    reconciled = deepcopy(evaluation_semantics)
    primary = dict(reconciled.get("primary_metric") or {})
    primary["name"] = metric_name.strip()
    primary["direction"] = (
        "minimize" if baseline_result["lower_is_better"] else "maximize")
    primary["measurement"] = (
        "Frozen from the reproduced official repository evaluator and baseline result.")
    reconciled["primary_metric"] = primary
    return reconciled


_REQUIRED = (
    "objective",
    "protocol",
    "controls",
    "allowed_treatment_axes",
    "provenance",
)
_SOURCE_PRIORITY = {
    "confirmed_boundary": 0,
    "user": 0,
    "official_harness": 1,
    "repository": 2,
    "paper": 2,
    "fallback": 3,
}
_PROTOCOL_ALIASES = {
    "fixed_wall_clock": "quality_at_fixed_time",
    "fixed_work": "quality_at_fixed_work",
}
_PROTOCOL_KINDS = {
    "quality_at_fixed_time",
    "quality_at_fixed_work",
    "time_to_target",
}
_MERGE_METADATA = {
    "version",
    "provenance",
    "deviations",
    "selected_source",
    "benchmark_kind",
}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_v2(contract: dict) -> None:
    protocol = contract["protocol"]
    kind = _PROTOCOL_ALIASES.get(protocol.get("kind"), protocol.get("kind"))
    if kind not in _PROTOCOL_KINDS:
        raise BenchmarkContractError(f"unsupported protocol kind: {kind}")
    protocol["kind"] = kind

    controls = contract["controls"]
    treatment = controls.get("treatment")
    allowed_paths = treatment.get("allowed_paths") if isinstance(treatment, dict) else None
    if not isinstance(allowed_paths, list) or not allowed_paths:
        raise BenchmarkContractError(
            "controls treatment allowed_paths must be a non-empty list")
    for path in allowed_paths:
        if (not isinstance(path, str) or not path
                or posixpath.isabs(path)
                or posixpath.normpath(path) in {"", ".", ".."}
                or posixpath.normpath(path).startswith("../")):
            raise BenchmarkContractError(
                "controls treatment allowed_paths must contain repository-relative paths")

    evaluator = controls.get("evaluator")
    source_claims = (
        evaluator.get("official_source_evidence") if isinstance(evaluator, dict) else None)
    required_claim_fields = {
        "path", "line_start", "line_end", "file_sha256", "excerpt_sha256", "command",
    }
    if (not isinstance(source_claims, list) or not source_claims
            or any(not isinstance(claim, dict) or set(claim) != required_claim_fields
                   for claim in source_claims)):
        raise BenchmarkContractError(
            "controls evaluator official_source_evidence must be a non-empty array of claims")

    objective = contract["objective"]
    objective_metric = objective.get("metric")

    outcomes = contract.get("observed_outcomes")
    if not isinstance(outcomes, dict) or not outcomes:
        raise BenchmarkContractError("observed_outcomes must be a non-empty object")
    quality = outcomes.get("quality")
    if not isinstance(quality, dict):
        raise BenchmarkContractError("observed_outcomes quality must be an object")
    quality_value = quality.get("value")
    if not _is_finite_number(quality_value):
        raise BenchmarkContractError(
            "observed_outcomes quality value must be a finite number")

    evidence = contract.get("verified_evidence")
    if not isinstance(evidence, dict):
        raise BenchmarkContractError("verified_evidence must be an object")
    for field in ("source", "artifact"):
        if not isinstance(evidence.get(field), str) or not evidence[field].strip():
            raise BenchmarkContractError(f"verified_evidence missing {field}")
    observations = evidence.get("observations")
    if not isinstance(observations, dict) or not observations:
        raise BenchmarkContractError("verified_evidence missing observations")
    verified_quality = observations.get("quality")
    if not isinstance(verified_quality, dict):
        raise BenchmarkContractError(
            "verified_evidence observations missing structured quality")
    verified_value = verified_quality.get("value")
    if not _is_finite_number(verified_value):
        raise BenchmarkContractError(
            "verified_evidence quality value must be finite")

    if kind == "time_to_target":
        target = protocol.get("target")
        if not isinstance(target, dict):
            raise BenchmarkContractError(
                "time_to_target target must be a structured object")
        target_metric = target.get("metric")
        if not isinstance(target_metric, str) or not target_metric.strip():
            raise BenchmarkContractError(
                "time_to_target target metric must be a non-empty string")
        target_value = target.get("value")
        if not _is_finite_number(target_value):
            raise BenchmarkContractError(
                "time_to_target target value must be finite")
        target_direction = target.get("direction")
        if target_direction not in {"minimize", "maximize"}:
            raise BenchmarkContractError(
                "time_to_target target direction must be minimize or maximize")
        if objective_metric != "elapsed_seconds" or objective.get("direction") != "minimize":
            raise BenchmarkContractError(
                "time_to_target objective must minimize elapsed_seconds")
        if quality.get("metric") != target_metric:
            raise BenchmarkContractError(
                "observed_outcomes quality metric must match target metric")

        elapsed = outcomes.get("objective")
        if (not isinstance(elapsed, dict)
                or elapsed.get("metric") != "elapsed_seconds"
                or not _is_finite_number(elapsed.get("value"))
                or elapsed["value"] <= 0):
            raise BenchmarkContractError(
                "observed_outcomes objective must be positive finite elapsed_seconds")
        if ((target_direction == "minimize" and quality_value > target_value)
                or (target_direction == "maximize" and quality_value < target_value)):
            raise BenchmarkContractError(
                "observed_outcomes quality did not reach target")

        if (verified_quality.get("metric") != target_metric
                or verified_value != quality_value):
            raise BenchmarkContractError(
                "verified_evidence quality must match observed quality")
        verified_elapsed = observations.get("objective")
        if (not isinstance(verified_elapsed, dict)
                or verified_elapsed.get("metric") != "elapsed_seconds"
                or not _is_finite_number(verified_elapsed.get("value"))
                or verified_elapsed["value"] <= 0
                or verified_elapsed["value"] != elapsed["value"]):
            raise BenchmarkContractError(
                "verified_evidence objective must match observed objective")
        return

    if not isinstance(objective_metric, str) or not objective_metric.strip():
        raise BenchmarkContractError("objective metric must be a non-empty string")
    if objective.get("direction") not in {"minimize", "maximize"}:
        raise BenchmarkContractError("objective direction must be minimize or maximize")
    if quality.get("metric") != objective_metric:
        raise BenchmarkContractError(
            "objective metric must match observed_outcomes quality metric")
    if verified_quality.get("metric") != objective_metric:
        raise BenchmarkContractError(
            "verified_evidence quality metric must match objective metric")
    if verified_value != quality_value:
        raise BenchmarkContractError(
            "verified_evidence quality value must match observed quality value")

    if kind == "quality_at_fixed_time":
        workload = contract["controls"].get("workload", {})
        budget = protocol.get("budget", {})
        wall_clock_s = budget.get("wall_clock_s") if isinstance(budget, dict) else None
        if not _is_finite_number(wall_clock_s) or wall_clock_s <= 0:
            raise BenchmarkContractError(
                "wall_clock_s must be a finite positive number")
        timing_scope = protocol.get("timing_scope")
        if timing_scope not in {"process", "training"}:
            raise BenchmarkContractError(
                "timing_scope must be process or training")
        timer_authority = protocol.get("timer_authority")
        if timer_authority not in {"host", "official_harness"}:
            raise BenchmarkContractError(
                "timer_authority must be host or official_harness")
        if (timing_scope, timer_authority) not in {
            ("process", "host"),
            ("training", "official_harness"),
        }:
            raise BenchmarkContractError(
                f"{timing_scope} timing_scope requires "
                f"{'host' if timing_scope == 'process' else 'official_harness'} "
                "timer authority")
        for field in ("steps", "tokens"):
            if field in workload or (isinstance(budget, dict) and field in budget):
                raise BenchmarkContractError(
                    f"fixed-time controls cannot require observed {field}")
    elif kind == "quality_at_fixed_work":
        budget = protocol.get("budget")
        work_units = [
            field for field in ("steps", "tokens")
            if isinstance(budget, dict) and field in budget
        ]
        if (len(work_units) != 1
                or not _is_finite_number(budget[work_units[0]])
                or budget[work_units[0]] <= 0):
            raise BenchmarkContractError(
                "quality_at_fixed_work requires exactly one positive steps or tokens budget")


def _source_for_fragment(contract: dict) -> str:
    provenance = contract.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise BenchmarkContractError("provenance must be a non-empty array")
    if any(not isinstance(item, dict) or item.get("source") not in _SOURCE_PRIORITY
           for item in provenance):
        raise BenchmarkContractError("provenance contains an unknown source")
    return min(provenance, key=lambda item: _SOURCE_PRIORITY[item["source"]])["source"]


def _record_leaf_owners(value: Any, path: tuple[str, ...], source: str,
                        owners: dict[tuple[str, ...], str]) -> None:
    if isinstance(value, dict) and value:
        for key, child in value.items():
            _record_leaf_owners(child, path + (str(key),), source, owners)
    else:
        owners[path] = source


def _owner_at_or_below(path: tuple[str, ...],
                       owners: dict[tuple[str, ...], str]) -> str | None:
    sources = {
        source for owned_path, source in owners.items()
        if owned_path == path or owned_path[:len(path)] == path
    }
    if not sources:
        return None
    return min(sources, key=lambda source: _SOURCE_PRIORITY[source])


def _replace_owner_tree(path: tuple[str, ...], value: Any, source: str,
                        owners: dict[tuple[str, ...], str]) -> None:
    for owned_path in list(owners):
        if owned_path == path or owned_path[:len(path)] == path:
            owners.pop(owned_path)
    _record_leaf_owners(value, path, source, owners)


def _drop_lower_priority_fixed_work_fields(
    merged: dict, owners: dict[tuple[str, ...], str], deviations: list[dict],
) -> None:
    """Remove inherited work-unit controls made invalid by an authoritative fixed-time kind."""
    protocol = merged.get("protocol")
    if not isinstance(protocol, dict):
        return
    kind = _PROTOCOL_ALIASES.get(protocol.get("kind"), protocol.get("kind"))
    if kind != "quality_at_fixed_time":
        return
    kind_source = _owner_at_or_below(("protocol", "kind"), owners)
    if kind_source is None:
        return

    for parent_path in (("protocol", "budget"), ("controls", "workload")):
        parent: Any = merged
        for key in parent_path:
            if not isinstance(parent, dict):
                parent = None
                break
            parent = parent.get(key)
        if not isinstance(parent, dict):
            continue
        for field in ("steps", "tokens"):
            if field not in parent:
                continue
            path = parent_path + (field,)
            field_source = _owner_at_or_below(path, owners)
            if (field_source is None
                    or _SOURCE_PRIORITY[kind_source] >= _SOURCE_PRIORITY[field_source]):
                continue
            rejected_value = parent.pop(field)
            owners.pop(path, None)
            deviations.append({
                "path": ".".join(path),
                "selected_source": kind_source,
                "selected_value": kind,
                "rejected_source": field_source,
                "rejected_value": rejected_value,
                "reason": "incompatible_with_protocol_kind",
            })


def _merge_value(current: Any, incoming: Any, path: tuple[str, ...], source: str,
                 owners: dict[tuple[str, ...], str], deviations: list[dict]) -> Any:
    if isinstance(current, dict) and isinstance(incoming, dict):
        for key, value in incoming.items():
            child_path = path + (str(key),)
            if key not in current:
                current[key] = value
                _record_leaf_owners(value, child_path, source, owners)
            else:
                current[key] = _merge_value(
                    current[key], value, child_path, source, owners, deviations)
        return current
    selected_source = _owner_at_or_below(path, owners)
    if current == incoming:
        if (selected_source is not None
                and _SOURCE_PRIORITY[source] < _SOURCE_PRIORITY[selected_source]):
            _replace_owner_tree(path, current, source, owners)
        return current

    if selected_source is None:
        raise BenchmarkContractError(f"missing merge ownership for {'.'.join(path)}")
    selected_priority = _SOURCE_PRIORITY[selected_source]
    incoming_priority = _SOURCE_PRIORITY[source]
    dotted_path = ".".join(path)
    if selected_priority == incoming_priority:
        raise BenchmarkContractError(f"same-priority conflict at {dotted_path}")
    if incoming_priority < selected_priority:
        deviations.append({
            "path": dotted_path,
            "selected_source": source,
            "selected_value": incoming,
            "rejected_source": selected_source,
            "rejected_value": current,
        })
        _replace_owner_tree(path, incoming, source, owners)
        return incoming
    deviations.append({
        "path": dotted_path,
        "selected_source": selected_source,
        "selected_value": current,
        "rejected_source": source,
        "rejected_value": incoming,
    })
    return current


def normalize_contract(contract: Any) -> dict:
    """Validate and return a JSON-normalized contract with its selected source."""
    if not isinstance(contract, dict):
        raise BenchmarkContractError("contract must be an object")
    for field in _REQUIRED:
        if field not in contract:
            raise BenchmarkContractError(f"missing {field}")
    if not isinstance(contract["objective"], dict) or not contract["objective"]:
        raise BenchmarkContractError("objective must be a non-empty object")
    if not isinstance(contract["protocol"], dict) or not contract["protocol"]:
        raise BenchmarkContractError("protocol must be a non-empty object")
    if not isinstance(contract["controls"], dict) or not contract["controls"]:
        raise BenchmarkContractError("controls must be a non-empty object")
    for field in ("resources", "data", "evaluator", "workload"):
        if field not in contract["controls"]:
            raise BenchmarkContractError(f"controls missing {field}")
    if not isinstance(contract["allowed_treatment_axes"], list):
        raise BenchmarkContractError("allowed_treatment_axes must be an array")
    provenance = contract["provenance"]
    if not isinstance(provenance, list) or not provenance:
        raise BenchmarkContractError("provenance must be a non-empty array")
    if any(not isinstance(item, dict) or item.get("source") not in _SOURCE_PRIORITY
           for item in provenance):
        raise BenchmarkContractError("provenance contains an unknown source")

    try:
        normalized = json.loads(json.dumps(contract, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(f"contract must contain finite JSON values: {exc}") from exc
    selected = min(provenance, key=lambda item: _SOURCE_PRIORITY[item["source"]])
    normalized["selected_source"] = selected["source"]
    if (_SOURCE_PRIORITY[selected["source"]] == 0
            and selected.get("conflicts_with_official") is True):
        normalized["benchmark_kind"] = "user_defined"
    normalized.setdefault("version", 1)
    normalized.setdefault("deviations", [])
    if normalized["version"] >= 2:
        _validate_v2(normalized)
    return normalized


def merge_benchmark_contracts(*contracts: Any) -> dict:
    """Merge source fragments by field authority and validate the resulting contract.

    Each input is one source contribution.  Higher-priority sources replace conflicting lower-
    priority leaves; lower-priority sources only fill fields the higher source omitted.  Equal-
    priority disagreement is ambiguous and therefore rejected.
    """
    if not contracts:
        raise BenchmarkContractError("at least one contract fragment is required")

    fragments: list[tuple[dict, str]] = []
    for contract in contracts:
        if not isinstance(contract, dict):
            raise BenchmarkContractError("contract fragment must be an object")
        try:
            fragment = json.loads(json.dumps(contract, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError(
                f"contract must contain finite JSON values: {exc}") from exc
        fragments.append((fragment, _source_for_fragment(fragment)))

    merged: dict = {}
    owners: dict[tuple[str, ...], str] = {}
    deviations: list[dict] = []
    provenance: list[dict] = []
    versions: list[int] = []
    benchmark_kinds: list[tuple[int, str]] = []
    for fragment, source in fragments:
        provenance.extend(fragment.get("provenance", []))
        deviations.extend(fragment.get("deviations", []))
        versions.append(int(fragment.get("version", 1)))
        if fragment.get("benchmark_kind"):
            benchmark_kinds.append((_SOURCE_PRIORITY[source], fragment["benchmark_kind"]))
        for key, value in fragment.items():
            if key in _MERGE_METADATA:
                continue
            path = (str(key),)
            if key not in merged:
                merged[key] = value
                _record_leaf_owners(value, path, source, owners)
            else:
                merged[key] = _merge_value(
                    merged[key], value, path, source, owners, deviations)

    if max(versions) >= 2:
        _drop_lower_priority_fixed_work_fields(merged, owners, deviations)

    merged["version"] = max(versions)
    merged["provenance"] = sorted(
        provenance, key=lambda item: (
            _SOURCE_PRIORITY[item["source"]],
            json.dumps(item, sort_keys=True, separators=(",", ":")),
        ),
    )
    merged["deviations"] = sorted(
        deviations, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":")),
    )
    priority_zero_overrode_official = any(
        isinstance(item, dict)
        and item.get("selected_source") in {"confirmed_boundary", "user"}
        and item.get("rejected_source") == "official_harness"
        for item in deviations
    )
    if priority_zero_overrode_official:
        merged["benchmark_kind"] = "user_defined"
    elif benchmark_kinds:
        merged["benchmark_kind"] = min(benchmark_kinds)[1]
    return normalize_contract(merged)


def canonical_contract_json(contract: Any) -> str:
    """Return the stable serialized form used as the frozen contract identity."""
    return json.dumps(
        normalize_contract(contract), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def contract_fingerprint(contract: Any) -> str:
    """Return a content fingerprint for a validated benchmark contract."""
    canonical = canonical_contract_json(contract).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def contract_from_confirmed_boundary(
    boundary: Any, *, metric: str, lower_is_better: bool, fixed_config: str,
) -> dict:
    """Build the frozen contract deterministically from the trusted human boundary.

    The sandbox supplies measurements and its actual config, but it is not the authority for the
    already-confirmed fairness policy.  This also makes a missing copied contract fail-safe without
    weakening runs that have no confirmed boundary.
    """
    if not isinstance(boundary, dict) or not boundary:
        raise BenchmarkContractError("confirmed boundary must be a non-empty object")
    boundary_metric = boundary.get("metric")
    if boundary_metric is not None and str(boundary_metric) != metric:
        raise BenchmarkContractError("confirmed boundary metric disagrees with baseline result")
    expected_direction = "minimize" if lower_is_better else "maximize"
    direction = str(boundary.get("direction") or expected_direction)
    if direction not in {"minimize", "maximize"}:
        raise BenchmarkContractError("confirmed boundary direction must be minimize or maximize")
    if direction != expected_direction:
        raise BenchmarkContractError("confirmed boundary direction disagrees with baseline result")
    protocol_kind = str(boundary.get("protocol_kind") or "confirmed_protocol")
    protocol = {"kind": protocol_kind}
    for key in ("training_budget_s", "tokens", "steps", "wall_clock_s"):
        if boundary.get(key) is not None:
            protocol[key] = boundary[key]
    resources = {
        key: boundary[key] for key in (
            "accelerator_count", "accelerator_type", "world_size", "precision")
        if boundary.get(key) is not None
    }
    resources.setdefault("accelerator_count", 1)
    controls = {
        "resources": resources,
        "data": {"requirement": boundary.get("data", "repository canonical data")},
        "evaluator": {
            "requirement": boundary.get("evaluator", "repository canonical evaluator")},
        "workload": {
            "accounting": boundary.get("workload_accounting", protocol_kind),
            "fixed_config": fixed_config,
        },
    }
    return normalize_contract({
        "version": 1,
        "benchmark_kind": (
            "user_defined" if boundary.get("conflicts_with_official") is True else "confirmed"),
        "objective": {
            "metric": str(boundary.get("metric") or metric),
            "direction": direction,
            "mode": str(boundary.get("objective_mode") or "quality"),
        },
        "protocol": protocol,
        "controls": controls,
        "allowed_treatment_axes": list(
            boundary.get("allowed_treatment_axes") or ["method implementation"]),
        "provenance": [{
            "source": "confirmed_boundary",
            "detail": boundary,
            "conflicts_with_official": boundary.get("conflicts_with_official") is True,
        }],
        "deviations": list(boundary.get("deviations") or []),
    })
