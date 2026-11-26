"""Tests for P4 failure classification (leak-safe repair decisions)."""
from __future__ import annotations

from autoresearch.experiments.general.classify import classify


def _res(valid, kind=None, detail="", timed_out=False):
    diag = {"first_failure": ({"kind": kind, "detail": detail} if kind else None),
            "timed_out": timed_out}
    return {"valid": valid, "diagnostics": diag}


def test_valid_run_not_repairable():
    assert classify(_res(True))["repairable"] is False


def test_method_error_repairable_with_own_detail():
    v = classify(_res(False, "method_error", "method solve error: ValueError: boom"))
    assert v["repairable"] is True and v["kind"] == "method_error"
    assert "boom" in v["feedback"]


def test_serialize_error_gives_concrete_conversion_hint():
    # A non-JSON-serializable return (torch tensor / numpy array) is a very common
    # GPU/ML mistake; the feedback must tell the model HOW to fix it, not just "error".
    v = classify(_res(False, "method_error",
                      "method serialize error: TypeError: Object of type Tensor is not JSON serializable"))
    assert v["repairable"] is True and v["kind"] == "method_error"
    fb = v["feedback"].lower()
    assert "tolist" in fb                       # concrete conversion recipe
    assert "serializ" in fb


def test_shape_mismatch_repairable_leaksafe():
    v = classify(_res(False, "shape_mismatch", "expected int, got float"))
    assert v["repairable"] is True
    assert "expected int, got float" in v["feedback"]


def test_invalid_solution_repairable_generic_only():
    v = classify(_res(False, "invalid_solution", "solution is well-formed but the ruler rejected it"))
    assert v["repairable"] is True and v["kind"] == "invalid_solution"
    # GENERIC hint only -- must NOT surface any harness/score internals
    fb = v["feedback"].lower()
    assert "constraint" in fb or "legal" in fb
    for leaked in ("score", "baseline", "harness.validate"):
        assert leaked not in v["feedback"]


def test_timeout_repairable():
    v = classify(_res(False, "method_error", "x", timed_out=True))
    assert v["repairable"] is True and v["kind"] == "timeout"


def test_validate_error_not_repairable():
    # harness's own validate() crashed -> not the method's fault
    assert classify(_res(False, "validate_error", "TypeError"))["repairable"] is False


def test_unknown_or_isolation_broken_not_repairable():
    assert classify({"valid": False, "isolated": False})["repairable"] is False
    assert classify(_res(False, None))["repairable"] is False
