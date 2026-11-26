"""The shared, EXECUTABLE, LEAK-SAFE I/O format contract (P3+P6).

The anti-cheat design keeps the harness-writer and the method-writer isolated, so
they independently guess the solution's SHAPE and often disagree -> the harness's
``validate`` rejects a perfectly-reasonable method as "invalid". This module is the
public API contract that closes that gap WITHOUT weakening isolation: it checks
only the STRUCTURE of a solution against a concrete EXAMPLE solution from the
design (``io_contract``), and says nothing about scoring, the baseline, or the
harness internals. It is FIXED, non-AI-written code (like ``_orchestrator.py``),
shipped to both the authoring container (so the method self-tests to the exact
shape) and the trusted scored run (so a shape mismatch yields a leak-safe repair
reason, separate from the harness's feasibility check).

``check_shape(solution, example) -> (ok, reason)`` structurally compares
``solution`` to the design's ``example`` solution. Number tolerance: an ``int`` is
accepted where the example is a ``float`` (a whole-number value), but a ``float``
is NOT accepted where the example is an ``int`` (indices must stay ints -- the
exact "returned values instead of indices" bug). Stdlib only.
"""
from __future__ import annotations

from typing import Any, Tuple


def _kind(x: Any) -> str:
    if x is None:
        return "none"
    if isinstance(x, bool):  # bool before int (bool is an int subclass)
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, (list, tuple)):
        return "list"
    if isinstance(x, dict):
        return "dict"
    return type(x).__name__


def _match(got: Any, exp: Any, path: str) -> Tuple[bool, str]:
    kg, ke = _kind(got), _kind(exp)
    # number tolerance: int is OK where a float is expected (whole value); a float
    # is NOT OK where an int is expected (indices must be ints).
    if ke == "float" and kg == "int":
        return True, ""
    if kg != ke:
        return False, f"{path}: expected {ke}, got {kg}"
    if kg == "list":
        # An empty list on either side gives no element type to check -> accept.
        if not got or not exp:
            return True, ""
        # Homogeneous-list assumption: check the first element's shape.
        return _match(got[0], exp[0], path + "[0]")
    if kg == "dict":
        for k in exp:
            if k not in got:
                return False, f"{path}: missing key {k!r}"
            ok, reason = _match(got[k], exp[k], f"{path}[{k!r}]")
            if not ok:
                return False, reason
        return True, ""
    return True, ""


def shape_signature(x: Any) -> str:
    """A short human structural signature (e.g. ``list[list[int]]``) for messages."""
    k = _kind(x)
    if k == "list":
        if not x:
            return "list[]"
        return f"list[{shape_signature(x[0])}]"
    if k == "dict":
        inner = ", ".join(f"{key!r}: {shape_signature(x[key])}" for key in x)
        return "dict{" + inner + "}"
    return k


def check_shape(solution: Any, example: Any) -> Tuple[bool, str]:
    """Return ``(ok, reason)``: does ``solution`` structurally match the design's
    ``example`` solution? ``reason`` (on failure) is a LEAK-SAFE format message
    naming the expected vs got shape -- never any scoring/harness internals."""
    if example is None:
        # No example provided -> cannot format-check; accept (fall back to validate).
        return True, ""
    ok, reason = _match(solution, example, "solution")
    if ok:
        return True, ""
    return False, (
        f"solution shape mismatch: expected like {shape_signature(example)}, "
        f"got {shape_signature(solution)} ({reason})"
    )


def strip_to_contract(instance: Any, example_instance: Any) -> Any:
    """Anti-cheat b: drop any top-level instance field NOT declared in the shape
    example, so a colluding harness cannot smuggle an answer key to the method
    through an EXTRA instance field. Best-effort (top-level dict keys); a
    value-encoded leak inside a declared field needs the full FS sandbox. A
    non-dict instance or a missing/non-dict example is returned unchanged."""
    if isinstance(instance, dict) and isinstance(example_instance, dict):
        return {k: v for k, v in instance.items() if k in example_instance}
    return instance


def example_instance(design_or_contract) -> Any:
    """The concrete example instance from a design's ``io_contract`` (or None)."""
    ioc = (design_or_contract or {}).get("io_contract") or {}
    return ioc.get("instance_example")


def example_solution(design_or_contract) -> Any:
    """The concrete example solution from a design's ``io_contract`` (or None)."""
    ioc = (design_or_contract or {}).get("io_contract") or {}
    return ioc.get("solution_example")


__all__ = ["check_shape", "shape_signature", "strip_to_contract",
           "example_instance", "example_solution"]
