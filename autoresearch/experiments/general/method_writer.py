"""Task 3: the method-writer agent (role B), ISOLATED FROM THE RULER at write time.

``write_method(backend, handle, design, *, timeout=None) -> str`` runs a container
agent -- in a SEPARATE sandbox from the harness writer -- that only ever sees the
``method_interface`` + ``task`` + ``baseline`` DESCRIPTION. It never receives the
harness source or the scoring internals, so it cannot tailor the method to the
(AI-written) ruler. It writes ``/workspace/method.py`` implementing
``solve(instance) -> solution``; the source is returned via ``cat method.py``.

This is the write-time half of the anti-cheat design; the run-time half is that
the method is later executed only inside the trusted isolator (Task 3.5/4).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from autoresearch.experiments.general.design import pip_install_command

_GEN_DIR = Path(__file__).resolve().parent
_AGENT_TIMEOUT_ENV = "AUTORESEARCH_GEN_AGENT_TIMEOUT"
_DEFAULT_AGENT_TIMEOUT = 7200.0


def _agent_timeout(explicit) -> float:
    if explicit is not None:
        return float(explicit)
    return float(os.environ.get(_AGENT_TIMEOUT_ENV, _DEFAULT_AGENT_TIMEOUT))


def write_method(backend, handle, design: dict, *, timeout=None, feedback=None) -> str:
    """Let an ISOLATED container agent write ``method.py``; return its source.

    ``feedback`` (P2 repair loop) is a LEAK-SAFE message about why a PREVIOUS attempt
    was invalid (its own error / wrong shape / infeasible) -- appended so the agent
    fixes that specific mistake. It never carries ruler/scoring internals."""
    # Pre-install the DECLARED dependencies (whatever this experiment needs, not a
    # fixed numpy/scipy) so the agent can self-test the method during authoring.
    # Names are sanitized + shell-quoted before they reach the shell.
    pip_cmd = pip_install_command(design.get("dependencies"))
    if pip_cmd:
        backend.execute_command(handle, pip_cmd)
    # Ship the LEAK-SAFE shape contract so the agent can self-test that
    # solve(example_instance) matches the exact declared solution shape (closes the
    # harness<->method format-drift gap without revealing any ruler internals).
    handle.workspace.file_upload(str(_GEN_DIR / "contract.py"), "/workspace/contract.py")
    backend.run_agent_task(handle, _method_prompt(design, feedback=feedback),
                           timeout=_agent_timeout(timeout))
    _enforce_shape(backend, handle, design, timeout)  # D11: auto-run the shape self-test
    return backend.execute_command(handle, "cat method.py").stdout


def _enforce_shape(backend, handle, design: dict, timeout) -> None:
    """D11: automatically run the shape self-test in the authoring container and, if
    solve()'s output has the wrong shape, do ONE corrective agent turn -- so the
    contract is enforced even when the agent skipped its own self-test. Best-effort:
    a missing io_contract or any harness error just skips it (the scored run's
    check_shape + the repair loop remain the backstop)."""
    ioc = design.get("io_contract") or {}
    inst, sol = ioc.get("instance_example"), ioc.get("solution_example")
    if inst is None or sol is None:
        return
    try:
        handle.workspace.file_upload(str(_GEN_DIR / "_shapecheck.py"), "/workspace/_shapecheck.py")
        with tempfile.TemporaryDirectory(prefix="gen-shape-") as td:
            ex = Path(td) / "_shape_example.json"
            ex.write_text(json.dumps({"instance": inst, "solution": sol}), encoding="utf-8")
            handle.workspace.file_upload(str(ex), "/workspace/_shape_example.json")
        out = (backend.execute_command(handle, "python _shapecheck.py").stdout or "")
        if "SHAPE_FAIL" in out:
            reason = out.split("SHAPE_FAIL:", 1)[1].strip()[:400]
            backend.run_agent_task(handle, _shape_fix_prompt(reason, sol),
                                   timeout=_agent_timeout(timeout))
    except Exception:  # noqa: BLE001 -- never block method writing on the self-test harness
        pass


def _shape_fix_prompt(reason: str, sol_example) -> str:
    return (
        "Your method.py's solve() output has the WRONG SHAPE (auto-checked):\n"
        f"  {reason}\n"
        "It must have the SAME STRUCTURE and TYPES as this example -- the specific "
        "NUMBERS below are ILLUSTRATIVE only; your solve() must COMPUTE its own values "
        "for each instance and must NOT hardcode/return these example numbers:\n"
        f"  {json.dumps(sol_example, ensure_ascii=False)}\n"
        "Fix method.py so solve(instance) returns a value of that structure (e.g. the "
        "right keys, integer INDICES vs floats, the right nesting), then stop. Do not "
        "change anything else."
    )


def _example_block(design: dict) -> str:
    ioc = design.get("io_contract") or {}
    inst, sol = ioc.get("instance_example"), ioc.get("solution_example")
    if inst is None or sol is None:
        return ""
    return f"""
CONCRETE SHAPE CONTRACT (obey it EXACTLY -- the harness that scores you was written
to the SAME shape, so this is how you two agree):
  - one example instance, exactly as you will receive it:
      {json.dumps(inst, ensure_ascii=False)}
  - a VALID solution for THAT instance has this SHAPE -- match the STRUCTURE and TYPES
    (e.g. integer INDICES not values, the right keys/nesting). The specific NUMBERS shown
    are ILLUSTRATIVE ONLY: your solve() MUST COMPUTE its own values for each instance and
    must NOT hardcode or return these example numbers:
      {json.dumps(sol, ensure_ascii=False)}

MANDATORY SELF-TEST before you finish (this is the #1 cause of failure): a fixed
file `contract.py` is provided (do NOT edit it). Write a small script that imports
your method + contract, calls solve() on the example instance above, and asserts
`contract.check_shape(solution, <the example solution above>)` returns ok=True. Run
it and FIX method.py until check_shape passes. Getting the shape right matters more
than the algorithm.
"""


def _repair_block(feedback) -> str:
    if not feedback:
        return ""
    return f"""
YOUR PREVIOUS ATTEMPT WAS REJECTED. Fix EXACTLY this and do not repeat it:
  {feedback}
Re-run the mandatory shape self-test after fixing.
"""


def _gpu_block(design: dict) -> str:
    if not design.get("gpu"):
        return ""
    return """
GPU EXPERIMENT: this is scored on a GPU. Use the declared GPU framework (e.g. `torch`) and
put tensors/models on the device (`dev = "cuda" if torch.cuda.is_available() else "cpu"`,
then `.to(dev)`). Keep the work SMALL so a run finishes in seconds, and return plain
CPU-side Python values (e.g. `.cpu().tolist()`). Follow method_interface EXACTLY for what
solve() receives and must return -- for a TRAINING task that is usually: train on the
instance's training data, then return predictions on the instance's test inputs.
"""


def _method_prompt(design: dict, feedback=None) -> str:
    return f"""You are implementing a METHOD for this research task. Write a single
file /workspace/method.py.
{_repair_block(feedback)}{_gpu_block(design)}

Task: {design.get('task')}
Baseline to beat: {design.get('baseline')}
Goal (what "better" means): {design.get('success_criterion')}

Implement EXACTLY this interface:

    {design.get('method_interface')}
{_example_block(design)}

That is, define:

    def solve(instance):
        # return a solution for ONE instance
        ...

Rules:
- CRITICAL -- FORMAT: `instance` and `solution` have EXACTLY the structure defined
  in method_interface above (the harness that scores you was written separately and
  obeys the SAME contract). Read each instance's fields exactly as specified, and
  return the solution in EXACTLY that structure (e.g. if a bin is a list of item
  INDICES, return indices, NOT item values). Any format mismatch makes every
  solution INVALID and scores 0 -- getting the format right matters more than the
  algorithm.
- Implement ONLY solve(instance) in /workspace/method.py. It is called once per
  instance and must return that instance's solution.
- You do NOT get the evaluation harness, the instances, or the scoring code. Work
  only from the task/interface above. Make solve robust to reasonable instances.
- Your GOAL is to actually BEAT the baseline on the metric ({design.get('success_criterion')}).
  Propose a GENUINELY BETTER, SUBSTANTIVELY DIFFERENT method (a real algorithmic
  improvement) using a simple but effective, well-motivated technique for this task
  -- do NOT just re-implement the baseline, rename it, or make a cosmetic tweak that
  produces near-identical results. Prefer a strong, well-known heuristic/algorithm over a naive one,
  but keep it CORRECT: always return a VALID solution (that is more important than
  a risky over-optimisation that produces illegal solutions). A simple, powerful,
  widely-applicable idea: PREPROCESS the input before applying the greedy rule
  (e.g. sort items largest-first) instead of processing them in the given order --
  this alone often beats a naive online/greedy baseline.
- Keep it self-contained (stdlib plus the experiment's declared dependencies) and
  finish promptly.

Run the MANDATORY shape self-test above with the provided `contract.py` before
finishing. You have a shell: if a self-test fails because a third-party module is
missing, you MAY `pip install <package>` to add it, then re-run the self-test.
"""


__all__ = ["write_method"]
