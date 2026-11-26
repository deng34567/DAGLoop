from __future__ import annotations

import pytest

from autoresearch.experiments.repo_improve.benchmark_contract import (
    BenchmarkContractError,
    canonical_contract_json,
    contract_fingerprint,
    merge_benchmark_contracts,
    normalize_contract,
    reconcile_repository_evaluation_semantics,
)


def _contract() -> dict:
    return {
        "version": 1,
        "benchmark_kind": "official",
        "objective": {
            "metric": "validation_loss",
            "direction": "minimize",
            "mode": "quality",
        },
        "protocol": {"kind": "fixed_work", "budget": {"steps": 1000}},
        "controls": {
            "resources": {"accelerator_count": 1},
            "data": {"split": "validation"},
            "evaluator": {"name": "repository canonical evaluator"},
            "workload": {"steps": 1000},
        },
        "allowed_treatment_axes": ["implementation"],
        "provenance": [
            {"source": "official_harness", "detail": "repository benchmark script"}
        ],
        "deviations": [],
    }


def _v2_contract(protocol_kind: str = "quality_at_fixed_time") -> dict:
    contract = _contract()
    contract["version"] = 2
    canonical_kind = {
        "fixed_wall_clock": "quality_at_fixed_time",
        "fixed_work": "quality_at_fixed_work",
    }.get(protocol_kind, protocol_kind)
    if canonical_kind == "quality_at_fixed_time":
        contract["protocol"] = {
            "kind": protocol_kind,
            "budget": {"wall_clock_s": 300},
            "timing_scope": "process",
            "timer_authority": "host",
        }
        contract["controls"]["workload"] = {"training_time_s": 300}
    elif canonical_kind == "quality_at_fixed_work":
        contract["protocol"] = {
            "kind": protocol_kind,
            "budget": {"steps": 1000},
        }
        contract["controls"]["workload"] = {"steps": 1000}
    else:
        contract["protocol"] = {
            "kind": protocol_kind,
            "target": {
                "metric": "validation_loss",
                "value": 1.75,
                "direction": "minimize",
            },
        }
        contract["objective"] = {
            "metric": "elapsed_seconds",
            "direction": "minimize",
            "mode": "speed",
        }
        contract["controls"]["workload"] = {"training_time_s": 300}
    contract["controls"]["treatment"] = {"allowed_paths": ["train.py"]}
    contract["controls"]["evaluator"]["official_source_evidence"] = [{
        "path": "README.md",
        "line_start": 1,
        "line_end": 1,
        "file_sha256": "sha256:file",
        "excerpt_sha256": "sha256:excerpt",
        "command": "python evaluate.py",
    }]
    contract["observed_outcomes"] = {
        "quality": {"metric": "validation_loss", "value": 1.75},
        "steps": 812,
        "tokens": 1662976,
    }
    contract["verified_evidence"] = {
        "source": "repository evaluator output",
        "artifact": "runs/baseline/metrics.json",
        "observations": {
            "quality": {"metric": "validation_loss", "value": 1.75},
        },
    }
    if canonical_kind == "time_to_target":
        contract["observed_outcomes"]["quality"]["value"] = 1.5
        contract["observed_outcomes"]["objective"] = {
            "metric": "elapsed_seconds",
            "value": 42.0,
        }
        contract["verified_evidence"]["observations"] = {
            "quality": {"metric": "validation_loss", "value": 1.5},
            "objective": {"metric": "elapsed_seconds", "value": 42.0},
        }
    return contract


def test_reconcile_repository_semantics_uses_reproduced_official_metric() -> None:
    semantics = {
        "comparison_mode": "repository_defined",
        "primary_metric": {
            "name": "repository_defined_primary_metric",
            "direction": "minimize",
            "measurement": "to be confirmed from the repository",
        },
        "conditions_to_hold_constant": ["same evaluator"],
    }
    baseline = {
        "source_kind": "live_repo",
        "reproduced": True,
        "metric_name": "combined_score",
        "lower_is_better": False,
    }

    reconciled = reconcile_repository_evaluation_semantics(semantics, baseline)

    assert reconciled["primary_metric"] == {
        "name": "combined_score",
        "direction": "maximize",
        "measurement": (
            "Frozen from the reproduced official repository evaluator and baseline result."),
    }
    assert reconciled["conditions_to_hold_constant"] == ["same evaluator"]
    assert semantics["primary_metric"]["direction"] == "minimize"


@pytest.mark.parametrize("baseline", [
    {"source_kind": "live_repo", "reproduced": False,
     "metric_name": "score", "lower_is_better": False},
    {"source_kind": "paper", "reproduced": True,
     "metric_name": "score", "lower_is_better": False},
])
def test_reconcile_repository_semantics_requires_reproduced_live_baseline(
    baseline: dict,
) -> None:
    semantics = {
        "comparison_mode": "repository_defined",
        "primary_metric": {"name": "loss", "direction": "minimize"},
    }
    assert reconcile_repository_evaluation_semantics(semantics, baseline) is semantics


def test_reconcile_repository_semantics_does_not_override_user_mode() -> None:
    semantics = {
        "comparison_mode": "user_defined",
        "primary_metric": {"name": "loss", "direction": "minimize"},
    }
    baseline = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_name": "score", "lower_is_better": False,
    }
    assert reconcile_repository_evaluation_semantics(semantics, baseline) is semantics


def test_contract_fingerprint_is_stable_across_mapping_order():
    """Catch semantically identical contracts producing different frozen identities."""
    first = _contract()
    second = dict(reversed(list(first.items())))
    second["controls"] = dict(reversed(list(first["controls"].items())))

    assert canonical_contract_json(first) == canonical_contract_json(second)
    assert contract_fingerprint(first) == contract_fingerprint(second)
    assert contract_fingerprint(first).startswith("sha256:")


@pytest.mark.parametrize(
    "missing",
    ["objective", "protocol", "controls", "allowed_treatment_axes", "provenance"],
)
def test_contract_rejects_missing_fairness_sections(missing):
    """Catch an incomplete contract being frozen as a trustworthy comparator."""
    contract = _contract()
    contract.pop(missing)

    with pytest.raises(BenchmarkContractError, match=missing):
        normalize_contract(contract)


@pytest.mark.parametrize("missing", ["resources", "data", "evaluator", "workload"])
def test_contract_rejects_missing_frozen_control_group(missing):
    """Catch a contract that cannot audit hardware, data, evaluator, or training work."""
    contract = _contract()
    contract["controls"].pop(missing)

    with pytest.raises(BenchmarkContractError, match=missing):
        normalize_contract(contract)


def test_contract_selects_highest_priority_provenance():
    """Catch repository defaults overriding an explicit confirmed user requirement."""
    contract = _contract()
    contract["provenance"] = [
        {"source": "repository", "detail": "README default"},
        {"source": "confirmed_boundary", "detail": "user fixed five-minute budget"},
        {"source": "official_harness", "detail": "official script default"},
    ]

    normalized = normalize_contract(contract)

    assert normalized["selected_source"] == "confirmed_boundary"


def test_explicit_user_conflict_is_a_user_defined_benchmark():
    """Catch an explicit user override being mislabeled as the official benchmark."""
    contract = _contract()
    contract["provenance"] = [
        {
            "source": "confirmed_boundary",
            "detail": "user requests ten minutes",
            "conflicts_with_official": True,
        },
        {"source": "official_harness", "detail": "official five-minute protocol"},
    ]

    normalized = normalize_contract(contract)

    assert normalized["benchmark_kind"] == "user_defined"
    assert normalized["selected_source"] == "confirmed_boundary"


@pytest.mark.parametrize(
    ("given", "canonical"),
    [
        ("quality_at_fixed_time", "quality_at_fixed_time"),
        ("fixed_wall_clock", "quality_at_fixed_time"),
        ("quality_at_fixed_work", "quality_at_fixed_work"),
        ("time_to_target", "time_to_target"),
    ],
)
def test_v2_contract_accepts_supported_protocols_and_normalizes_legacy_alias(
    given, canonical,
):
    """Catch supported experiment questions being rejected or the legacy alias drifting."""
    contract = _v2_contract(given)

    normalized = normalize_contract(contract)

    assert normalized["protocol"]["kind"] == canonical


@pytest.mark.parametrize("work_unit", ["steps", "tokens"])
def test_fixed_time_keeps_variable_work_units_out_of_controls(work_unit):
    """Catch observed work completed in a time budget being frozen as a fairness control."""
    contract = _v2_contract()
    contract["controls"]["workload"][work_unit] = 812

    with pytest.raises(BenchmarkContractError, match=f"fixed-time.*{work_unit}"):
        normalize_contract(contract)


@pytest.mark.parametrize("work_unit", ["steps", "tokens"])
def test_fixed_time_keeps_variable_work_units_out_of_protocol_budget(work_unit):
    """Catch a fixed-time protocol silently requiring equal realized steps or tokens."""
    contract = _v2_contract()
    contract["protocol"]["budget"][work_unit] = 812

    with pytest.raises(BenchmarkContractError, match=f"fixed-time.*{work_unit}"):
        normalize_contract(contract)


def test_v2_contract_requires_minimal_verified_evidence():
    """Catch an unauditable metric artifact being accepted as verified evidence."""
    contract = _v2_contract()
    contract["verified_evidence"].pop("observations")

    with pytest.raises(BenchmarkContractError, match="verified_evidence.*observations"):
        normalize_contract(contract)


def test_v2_objective_metric_must_match_observed_quality_metric():
    """Catch a claimed objective being validated against a different measured metric."""
    contract = _v2_contract()
    contract["observed_outcomes"]["quality"]["metric"] = "accuracy"

    with pytest.raises(BenchmarkContractError, match="objective.*metric.*quality"):
        normalize_contract(contract)


@pytest.mark.parametrize("direction", ["ascending", "", None])
def test_v2_objective_direction_must_be_minimize_or_maximize(direction):
    """Catch an ambiguous comparison direction being frozen into a v2 contract."""
    contract = _v2_contract()
    contract["objective"]["direction"] = direction

    with pytest.raises(BenchmarkContractError, match="objective.*direction"):
        normalize_contract(contract)


@pytest.mark.parametrize("value", [None, "1.75", True])
def test_v2_observed_quality_value_must_be_a_finite_number(value):
    """Catch missing or nonnumeric quality measurements being treated as outcomes."""
    contract = _v2_contract()
    contract["observed_outcomes"]["quality"]["value"] = value

    with pytest.raises(BenchmarkContractError, match="quality.*value.*finite"):
        normalize_contract(contract)


def test_v2_verified_evidence_requires_structured_quality_observation():
    """Catch evidence that cannot bind its value to the objective metric."""
    contract = _v2_contract()
    contract["verified_evidence"]["observations"] = {"validation_loss": 1.75}

    with pytest.raises(BenchmarkContractError, match="observations.*quality"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("metric", "accuracy", "metric"),
        ("value", 1.5, "value"),
        ("value", "1.75", "finite"),
    ],
)
def test_v2_verified_quality_must_match_observed_quality(field, value, message):
    """Catch evidence for a different or nonnumeric result validating the outcome."""
    contract = _v2_contract()
    contract["verified_evidence"]["observations"]["quality"][field] = value

    with pytest.raises(BenchmarkContractError, match=f"verified_evidence.*{message}"):
        normalize_contract(contract)


@pytest.mark.parametrize("wall_clock_s", [None, 0, -1, "300"])
def test_fixed_time_requires_a_positive_finite_wall_clock_budget(wall_clock_s):
    """Catch fixed-time comparisons without a usable equal time budget."""
    contract = _v2_contract()
    if wall_clock_s is None:
        contract["protocol"]["budget"].pop("wall_clock_s")
    else:
        contract["protocol"]["budget"]["wall_clock_s"] = wall_clock_s

    with pytest.raises(BenchmarkContractError, match="wall_clock_s.*finite.*positive"):
        normalize_contract(contract)


@pytest.mark.parametrize("timing_scope", [None, "gpu", ""])
def test_fixed_time_rejects_unknown_timing_scope(timing_scope):
    """Catch an undefined timed region making baseline and method incomparable."""
    contract = _v2_contract()
    if timing_scope is None:
        contract["protocol"].pop("timing_scope")
    else:
        contract["protocol"]["timing_scope"] = timing_scope

    with pytest.raises(BenchmarkContractError, match="timing_scope"):
        normalize_contract(contract)


@pytest.mark.parametrize("timer_authority", [None, "agent", ""])
def test_fixed_time_rejects_unknown_timer_authority(timer_authority):
    """Catch an untrusted or undefined clock deciding a fixed-time result."""
    contract = _v2_contract()
    if timer_authority is None:
        contract["protocol"].pop("timer_authority")
    else:
        contract["protocol"]["timer_authority"] = timer_authority

    with pytest.raises(BenchmarkContractError, match="timer_authority"):
        normalize_contract(contract)


def test_training_timing_scope_requires_official_harness_authority():
    """Catch an agent-controlled training timer excluding arbitrary setup or evaluation work."""
    contract = _v2_contract()
    contract["protocol"].update({"timing_scope": "training", "timer_authority": "host"})

    with pytest.raises(BenchmarkContractError, match="training.*official_harness"):
        normalize_contract(contract)


def test_process_timing_scope_requires_host_authority():
    """Catch an official harness clock being mislabeled as whole-process timing."""
    contract = _v2_contract()
    contract["protocol"].update({
        "timing_scope": "process", "timer_authority": "official_harness",
    })

    with pytest.raises(BenchmarkContractError, match="process.*host"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("timing_scope", "timer_authority"),
    [("process", "host"), ("training", "official_harness")],
)
def test_fixed_time_accepts_auditable_timer_combinations(timing_scope, timer_authority):
    """Catch supported host and official timing arrangements being rejected."""
    contract = _v2_contract()
    contract["protocol"].update({
        "timing_scope": timing_scope,
        "timer_authority": timer_authority,
    })

    assert normalize_contract(contract)["protocol"]["timing_scope"] == timing_scope


def test_time_to_target_requires_a_target():
    """Catch time-to-target runs that never define the stopping threshold."""
    contract = _v2_contract("time_to_target")
    contract["protocol"].pop("target")

    with pytest.raises(BenchmarkContractError, match="time_to_target.*target"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [("metric", ""), ("value", "1.75"), ("value", True),
     ("direction", "ascending")],
)
def test_time_to_target_requires_a_structured_finite_target(field, value):
    """Catch an ambiguous threshold that cannot prove the target was reached."""
    contract = _v2_contract("time_to_target")
    contract["protocol"]["target"][field] = value

    with pytest.raises(BenchmarkContractError, match="time_to_target target"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("metric", "direction"),
    [("validation_loss", "minimize"), ("elapsed_seconds", "maximize")],
)
def test_time_to_target_objective_is_elapsed_seconds_minimized(metric, direction):
    """Catch target quality being confused with the runtime objective."""
    contract = _v2_contract("time_to_target")
    contract["objective"].update({"metric": metric, "direction": direction})

    with pytest.raises(BenchmarkContractError, match="time_to_target objective"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [("metric", "runtime"), ("value", 0), ("value", "42")],
)
def test_time_to_target_requires_a_positive_elapsed_outcome(field, value):
    """Catch missing or invalid runtime being accepted as a time-to-target result."""
    contract = _v2_contract("time_to_target")
    contract["observed_outcomes"]["objective"][field] = value

    with pytest.raises(BenchmarkContractError, match="observed_outcomes objective"):
        normalize_contract(contract)


def test_time_to_target_requires_an_objective_outcome():
    """Catch a reached quality threshold being reported without elapsed time."""
    contract = _v2_contract("time_to_target")
    contract["observed_outcomes"].pop("objective")

    with pytest.raises(BenchmarkContractError, match="observed_outcomes objective"):
        normalize_contract(contract)


def test_time_to_target_quality_must_use_target_metric():
    """Catch reaching an unrelated quality metric being counted against the target."""
    contract = _v2_contract("time_to_target")
    contract["observed_outcomes"]["quality"]["metric"] = "accuracy"

    with pytest.raises(BenchmarkContractError, match="quality metric.*target"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("direction", "target_value", "quality_value"),
    [("minimize", 1.5, 9.0), ("maximize", 0.9, 0.8)],
)
def test_time_to_target_rejects_quality_that_did_not_reach_target(
    direction, target_value, quality_value,
):
    """Catch elapsed time being reported before the declared target was reached."""
    contract = _v2_contract("time_to_target")
    contract["protocol"]["target"].update({
        "direction": direction, "value": target_value,
    })
    contract["observed_outcomes"]["quality"]["value"] = quality_value
    contract["verified_evidence"]["observations"]["quality"]["value"] = quality_value

    with pytest.raises(BenchmarkContractError, match="quality did not reach target"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("quality", "metric", "accuracy"),
        ("quality", "value", 1.6),
        ("objective", "metric", "runtime"),
        ("objective", "value", 43.0),
    ],
)
def test_time_to_target_evidence_must_match_both_outcomes(section, field, value):
    """Catch evidence that attests to different quality or runtime observations."""
    contract = _v2_contract("time_to_target")
    contract["verified_evidence"]["observations"][section][field] = value

    with pytest.raises(BenchmarkContractError, match=f"verified_evidence {section}"):
        normalize_contract(contract)


def test_time_to_target_evidence_requires_objective_observation():
    """Catch quality-only evidence being used to attest elapsed time."""
    contract = _v2_contract("time_to_target")
    contract["verified_evidence"]["observations"].pop("objective")

    with pytest.raises(BenchmarkContractError, match="verified_evidence objective"):
        normalize_contract(contract)


def test_quality_at_fixed_work_requires_steps_or_tokens_budget():
    """Catch fixed-work comparisons with no frozen amount of work."""
    contract = _v2_contract("quality_at_fixed_work")
    contract["protocol"]["budget"] = {"wall_clock_s": 300}

    with pytest.raises(BenchmarkContractError, match="fixed_work.*steps.*tokens"):
        normalize_contract(contract)


def test_quality_at_fixed_work_rejects_both_steps_and_tokens_budget():
    """Catch a fixed-work protocol freezing two potentially conflicting work units."""
    contract = _v2_contract("quality_at_fixed_work")
    contract["protocol"]["budget"] = {"steps": 1000, "tokens": 2048000}

    with pytest.raises(BenchmarkContractError, match="exactly one.*steps.*tokens"):
        normalize_contract(contract)


@pytest.mark.parametrize(
    "allowed_paths",
    [None, [], [""], ["/tmp/train.py"], ["../train.py"], ["src/../../train.py"],
     ["."], ["train.py", 7]],
)
def test_v2_treatment_requires_nonempty_repo_relative_allowed_paths(allowed_paths):
    """Catch an absent, empty, escaping, or nonstring treatment boundary."""
    contract = _v2_contract()
    if allowed_paths is None:
        contract["controls"].pop("treatment")
    else:
        contract["controls"]["treatment"]["allowed_paths"] = allowed_paths

    with pytest.raises(BenchmarkContractError, match="treatment.*allowed_paths"):
        normalize_contract(contract)


def test_v2_keeps_legacy_allowed_treatment_axes_alongside_path_boundary():
    """Catch path-scoped treatment validation removing the legacy descriptive field."""
    contract = _v2_contract()

    normalized = normalize_contract(contract)

    assert normalized["controls"]["treatment"]["allowed_paths"] == ["train.py"]
    assert normalized["allowed_treatment_axes"] == ["implementation"]


@pytest.mark.parametrize(
    "evidence",
    [None, {}, [], ["README.md"], [{}], [{"path": "README.md"}]],
)
def test_v2_requires_nonempty_structured_official_source_claims(evidence):
    """Catch missing, legacy singleton, or incomplete official-source evidence."""
    contract = _v2_contract()
    if evidence is None:
        contract["controls"]["evaluator"].pop("official_source_evidence")
    else:
        contract["controls"]["evaluator"]["official_source_evidence"] = evidence

    with pytest.raises(BenchmarkContractError, match="official_source_evidence"):
        normalize_contract(contract)


def test_v1_contract_remains_compatible_with_v2_only_semantic_gates():
    """Catch semantic hardening accidentally invalidating persisted v1 contracts."""
    contract = _contract()
    contract["objective"]["direction"] = "legacy_direction"
    contract["observed_outcomes"] = {
        "quality": {"metric": "different_metric", "value": "legacy-value"},
    }

    assert normalize_contract(contract)["version"] == 1


def test_merge_drops_lower_priority_work_units_when_protocol_becomes_fixed_time():
    """Catch stale official fixed-work controls invalidating a confirmed fixed-time override."""
    official = _v2_contract("quality_at_fixed_work")
    confirmed = {
        "version": 2,
        "protocol": {
            "kind": "quality_at_fixed_time",
            "budget": {"wall_clock_s": 300},
            "timing_scope": "process",
            "timer_authority": "host",
        },
        "provenance": [{"source": "confirmed_boundary", "detail": "five minutes"}],
    }

    merged = merge_benchmark_contracts(official, confirmed)

    assert merged["protocol"]["budget"] == {"wall_clock_s": 300}
    assert "steps" not in merged["controls"]["workload"]
    assert merged["observed_outcomes"]["steps"] == 812
    removed_paths = {
        item["path"] for item in merged["deviations"]
        if isinstance(item, dict) and item.get("reason") == "incompatible_with_protocol_kind"
    }
    assert removed_paths == {"protocol.budget.steps", "controls.workload.steps"}


def test_merge_uses_high_priority_fields_and_supplements_only_missing_fields():
    """Catch an agent or official default overriding the confirmed fairness boundary."""
    confirmed = {
        "version": 2,
        "objective": {"metric": "validation_loss", "direction": "minimize"},
        "protocol": {
            "kind": "quality_at_fixed_time",
            "budget": {"wall_clock_s": 300},
        },
        "controls": {"resources": {"accelerator_count": 1}},
        "allowed_treatment_axes": ["implementation"],
        "provenance": [{"source": "confirmed_boundary", "detail": "confirmed by user"}],
    }
    official = _v2_contract()
    official["protocol"]["budget"]["wall_clock_s"] = 600
    official["controls"]["resources"] = {
        "accelerator_count": 8,
        "accelerator_type": "L20",
    }

    merged = merge_benchmark_contracts(official, confirmed)

    assert merged["protocol"]["budget"]["wall_clock_s"] == 300
    assert merged["controls"]["resources"] == {
        "accelerator_count": 1,
        "accelerator_type": "L20",
    }
    assert merged["controls"]["data"] == {"split": "validation"}
    assert merged["verified_evidence"] == official["verified_evidence"]
    assert {
        (item["path"], item["selected_source"], item["rejected_source"])
        for item in merged["deviations"]
    } >= {
        ("protocol.budget.wall_clock_s", "confirmed_boundary", "official_harness"),
        ("controls.resources.accelerator_count", "confirmed_boundary", "official_harness"),
    }


def test_merge_fails_closed_on_same_priority_field_conflict():
    """Catch equally authoritative user statements being resolved by input ordering."""
    first = _v2_contract()
    first["provenance"] = [{"source": "user", "detail": "first request"}]
    second = _v2_contract()
    second["provenance"] = [{"source": "confirmed_boundary", "detail": "confirmation"}]
    second["controls"]["resources"]["accelerator_count"] = 2

    with pytest.raises(
        BenchmarkContractError,
        match="same-priority conflict.*controls.resources.accelerator_count",
    ):
        merge_benchmark_contracts(first, second)


def test_merge_can_replace_a_lower_priority_field_group_with_a_confirmed_scalar():
    """Catch field ownership being lost when authoritative input changes a value's shape."""
    official = _v2_contract()
    confirmed = {
        "version": 2,
        "controls": {"data": "confirmed canonical data"},
        "provenance": [{"source": "confirmed_boundary", "detail": "confirmed split"}],
    }

    merged = merge_benchmark_contracts(official, confirmed)

    assert merged["controls"]["data"] == "confirmed canonical data"
    assert any(item["path"] == "controls.data" for item in merged["deviations"])


def test_merge_preserves_legacy_free_text_deviations():
    """Catch field-level merging breaking v1 contracts with legacy deviation notes."""
    contract = _contract()
    contract["deviations"] = ["repository evaluator was unavailable"]

    merged = merge_benchmark_contracts(contract)

    assert "repository evaluator was unavailable" in merged["deviations"]
