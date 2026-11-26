"""P4: classify an invalid run into a LEAK-SAFE repair decision.

The format-repair loop (P2) must only re-prompt the method-writer for failures that
are (a) the METHOD's fault and (b) describable WITHOUT revealing ruler internals.
``classify(run_result)`` reads the P1 diagnostics and returns
``{repairable, kind, feedback}``:

  - method_error / shape_mismatch / timeout -> repairable, with the method's OWN
    error or the public shape reason as feedback (leak-safe).
  - invalid_solution (well-formed but the ruler rejected it as infeasible) ->
    repairable with a GENERIC constraint hint only (the harness's specific reason
    is never surfaced).
  - validate_error (the harness's validate() itself crashed) -> NOT the method's
    fault -> not repairable (escalate, don't loop on it).
  - unknown / no diagnostics / isolation broken -> conservative: not repairable
    (never blame the method blindly -- vibe's cure for the AI-fixes-AI death loop).
"""
from __future__ import annotations


def classify(run_result: dict) -> dict:
    """Return ``{repairable: bool, kind: str, feedback: str|None}`` for a run."""
    if not run_result or run_result.get("valid"):
        return {"repairable": False, "kind": "ok", "feedback": None}

    diag = run_result.get("diagnostics") or {}
    ff = diag.get("first_failure") or {}
    kind = ff.get("kind")
    detail = (ff.get("detail") or "").strip()

    if diag.get("timed_out"):
        return {
            "repairable": True, "kind": "timeout",
            "feedback": ("Your solve() was too slow and timed out. Make it fast enough "
                         "to return a solution for each instance well within the time limit."),
        }
    if kind == "method_error":
        low = detail.lower()
        # a very common GPU/ML failure: solve() returned a torch tensor / numpy array,
        # which is not JSON-serializable. The generic hint doesn't tell the model HOW
        # to fix it, so give the concrete conversion.
        if "serializ" in low or "not json" in low or "json serializable" in low:
            return {
                "repairable": True, "kind": kind,
                "feedback": ("Your solve() returned a value that is NOT JSON-serializable "
                             "(e.g. a torch tensor or a numpy array). Return PLAIN Python "
                             "numbers / lists instead -- convert before returning with "
                             "`.cpu().tolist()` (torch) or `.tolist()` (numpy), and use "
                             "`float(x)` for a single scalar."),
            }
        return {
            "repairable": True, "kind": kind,
            "feedback": (f"Your method raised an error on an instance: {detail}. "
                         "Fix solve() so it runs without error on every instance."),
        }
    if kind == "shape_mismatch":
        return {
            "repairable": True, "kind": kind,
            "feedback": (f"Your solve() returned the WRONG SHAPE: {detail}. Return the "
                         "solution in EXACTLY the declared shape (e.g. integer indices, "
                         "not values; the right nesting)."),
        }
    if kind == "invalid_solution":
        return {
            "repairable": True, "kind": kind,
            "feedback": ("Your solution was well-formed but rejected as INVALID/INFEASIBLE "
                         "-- it violated a problem constraint. Re-read the task's rules and "
                         "make solve() ALWAYS return a LEGAL solution."),
        }
    # harness bug, isolation broken, or unknown -> not the method's fault; don't loop.
    return {"repairable": False, "kind": kind or "unknown", "feedback": None}


__all__ = ["classify"]
