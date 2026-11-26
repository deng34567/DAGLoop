"""Task 2: the harness-writer agent (role A -- the AI "ruler").

``write_harness(backend, handle, design, *, timeout=None) -> str`` runs a
container agent (M9 ``run_agent_task``) that writes ``/workspace/harness.py`` for
THIS design as a PURE LIBRARY, then returns its source (via ``cat harness.py``).

Runtime-isolation contract (this is the anti-cheat point): the harness is a
LIBRARY with only ``make_instances`` / ``run_baseline`` / ``validate`` / ``score``
-- NO ``__main__``, and it NEVER imports or runs the method. A FIXED, trusted
``_orchestrator.py`` (not AI-written) drives the run: it imports the harness,
DELETES harness.py from disk, runs the method inside the trusted isolator, and
calls the harness only to validate/score the returned solutions. So the method
can never reach the ruler, by construction.

Default test suite never touches Docker/LLM: tests pass a mock ``handle`` +
mock ``run_agent_task`` / ``file_upload`` / ``execute_command``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from autoresearch.experiments.general.design import pip_install_command

# ML/statistical metrics that can OVERFIT -> the harness must measure them on a held-out split.
_ML_METRIC_RE = re.compile(
    r"accuracy|error|\bloss\b|\bf1\b|\bauc\b|precision|recall|\brmse\b|\bmae\b|perplexity"
    r"|\bbpb\b|silhouette|r2|准确率|误差|损失",
    re.I,
)

_GEN_DIR = Path(__file__).resolve().parent

# Generous wall-clock SAFETY BACKSTOP on the in-container agent loop (env-config).
_AGENT_TIMEOUT_ENV = "AUTORESEARCH_GEN_AGENT_TIMEOUT"
_DEFAULT_AGENT_TIMEOUT = 7200.0

# FIXED trusted modules pre-placed so the agent can SELF-TEST its library harness
# (``python _selftest.py 0`` with its own stub method.py) during authoring. The
# self-test wrapper drives the SAME trusted orchestrator but WITHOUT deleting
# harness.py, so the agent can re-run it freely (scoring still deletes).
_TRUSTED_MODULES = ("_orchestrator.py", "_selftest.py", "isolated_method.py", "_method_driver.py", "contract.py")


def _agent_timeout(explicit) -> float:
    if explicit is not None:
        return float(explicit)
    return float(os.environ.get(_AGENT_TIMEOUT_ENV, _DEFAULT_AGENT_TIMEOUT))


def write_harness(backend, handle, design: dict, *, timeout=None) -> str:
    """Let the container agent write ``harness.py`` (a pure library) for ``design``;
    return its source."""
    # pre-place the trusted orchestrator + self-test wrapper + isolator so the
    # agent's self-test can run ``python _selftest.py 0`` against a stub method.
    for name in _TRUSTED_MODULES:
        handle.workspace.file_upload(str(_GEN_DIR / name), f"/workspace/{name}")

    # Pre-install the DECLARED dependencies (whatever this experiment needs, not a
    # fixed numpy/scipy) so the agent's in-container self-test (`python
    # _selftest.py 0`, which imports the harness) doesn't fail on a missing
    # module. Names are sanitized + shell-quoted before they reach the shell.
    pip_cmd = pip_install_command(design.get("dependencies"))
    if pip_cmd:
        backend.execute_command(handle, pip_cmd)
    backend.run_agent_task(handle, _harness_prompt(design), timeout=_agent_timeout(timeout))
    return backend.execute_command(handle, "cat harness.py").stdout


def _harness_prompt(design: dict) -> str:
    # METRIC DIRECTION: the ruler's returned number MUST point the same way as
    # design.lower_is_better (which accept.decide also honors). Hardcoding "lower is
    # better" made a higher-is-better metric (e.g. knapsack total value) get turned
    # into a gap/negated -> the more optimal solution scored "worse" -> inverted
    # verdict. Now the flag drives the wording so harness <-> design <-> decide agree.
    lower = bool(design.get("lower_is_better", True))
    better = "LOWER is better" if lower else "HIGHER is better"
    if lower:
        raw_note = (
            "The metric is an error/cost/gap-style quantity where LOWER is better; "
            "return it directly (do not flip its sign)."
        )
    else:
        raw_note = (
            "The metric is a value/reward/accuracy-style quantity where HIGHER is "
            "better: return the RAW metric directly. Do NOT negate it, invert it, or "
            "convert it into an error/gap/regret -- the verdict compares the two "
            "numbers assuming HIGHER is better, so any sign flip inverts the result."
        )
    gpu_hint = ("""
GPU EXPERIMENT: this is scored on a GPU. Use the declared GPU framework (e.g. `torch`)
and put tensors/models on the device (`dev = "cuda" if torch.cuda.is_available() else
"cpu"`, then `.to(dev)`). Keep the work SMALL so a run finishes in seconds. run_baseline()
and score() must compute the GROUND TRUTH themselves (it is NEVER inside the instance)
and compare against it. IF this is a TRAINING task, use a teacher-student setup: hold a
FIXED teacher (a fixed seed) as the answer key, put ONLY the training data + test INPUTS in
each instance (never the teacher or the clean test targets), and score predictions against
the teacher's clean outputs.
""" if design.get("gpu") else "")
    # HELD-OUT enforcement for ML/statistical metrics (accuracy/error/loss/...): the harness
    # must measure the metric on data the method never trained on, so the metric is a genuine
    # generalization measure and the evaluator's "missing held-out validation" flag is satisfied
    # STRUCTURALLY (the method cannot see the test labels). Skipped for combinatorial metrics
    # (bin count, tour length) and for the GPU teacher-student path (handled above).
    heldout_hint = ("""
HELD-OUT VALIDATION (this metric can OVERFIT -- enforce it in the harness): inside
make_instances build a TRAIN / HELD-OUT-TEST split of the real data. Put in each instance ONLY
(a) the TRAINING data (features + labels) and (b) the TEST INPUTS (features only, NO labels);
keep the test LABELS inside the harness. In run_baseline and score, fit the model AND all
preprocessing (scaling, feature selection, imputation) on the TRAINING partition ONLY, predict
the held-out test inputs, and compute the metric against the test LABELS the method never saw.
Never fit or standardize on the test data. This makes the metric a true held-out generalization
score (no train-on-test leakage).
""" if (_ML_METRIC_RE.search(str(design.get("metric") or "")) and not design.get("gpu")) else "")
    return f"""You are writing the TRUSTED EVALUATION HARNESS (the "ruler") for this experiment,
AS A PURE LIBRARY.
{gpu_hint}{heldout_hint}

Experiment design (JSON):
  question: {design.get('question')}
  task: {design.get('task')}
  instance_spec: {design.get('instance_spec')}
  baseline: {design.get('baseline')}
  method_interface: {design.get('method_interface')}
  io_contract (the CONCRETE shape BOTH sides must obey -- make_instances must produce
    each instance like instance_example, and validate/score must accept a solution
    shaped like solution_example): {design.get('io_contract')}
  metric: {design.get('metric')}
  success_criterion: {design.get('success_criterion')}

Write a single file /workspace/harness.py that is a PURE LIBRARY -- it defines
EXACTLY these four functions and NOTHING that runs on its own:

  def make_instances(seed):
      # build a list of instances reproducibly from the integer seed
      # (per instance_spec above). Same seed -> same instances.
      ...

  def run_baseline(instances) -> float:
      # run the baseline ({design.get('baseline')}); return ONE number (the
      # aggregate primary metric, {better}) -- e.g. the AVERAGE, NOT a dict.
      ...

  def validate(instance, solution) -> bool:
      # True iff the solution is a LEGAL solution for the instance
      # (per the metric's validity criterion).
      ...

  def score(instances, solutions) -> float:
      # aggregate the metric over the (already-validated) solutions; return ONE
      # number (same aggregate as run_baseline, {better}), NOT a dict/list.
      ...

HARD RULES (enforced -- violating them breaks the experiment):
- make_instances MUST accept ANY non-negative integer seed and NEVER cap/validate
  the seed range (e.g. do NOT `raise` for "seed out of range"). The run is
  reproduced on a fresh, arbitrary seed -- a seed-range check breaks reproduction.
- FORMAT CONTRACT: make_instances MUST produce each `instance` in EXACTLY the
  structure given in method_interface above, and validate/score MUST expect the
  `solution` in EXACTLY the `solution` structure in method_interface. The method
  (solve) is written SEPARATELY and obeys the SAME contract -- if your instance or
  solution format differs from method_interface, every solution is rejected as
  invalid and the whole experiment fails. Follow method_interface literally.
- Do NOT write an `if __name__ == "__main__":` block. Do NOT read argv, do NOT
  write result.json, do NOT run anything at import time.
- The harness is a LIBRARY. You MUST NOT `import method` and MUST NOT
  `from method import ...`, and you MUST NOT run the method in any way. You only
  ever receive `solution` objects (already produced elsewhere) to validate/score.
- A FIXED, trusted orchestrator (which you do not write and cannot change) will
  import your library, run the method in isolation, and call your validate/score
  on the results. Never depend on how the method is produced.
- `validate` and `score` must be robust: if given a malformed/None solution,
  `validate` returns False (do not crash).
- `run_baseline` and `score` MUST each return ONE plain number (a Python
  int/float -- the aggregate primary metric, {better}). NEVER return a
  dict, list, tuple or numpy array. If you want the average bins, return the
  average as a single float, not `{{"average_bins": ...}}`.
- METRIC DIRECTION (critical -- {better}): {raw_note} `run_baseline` and `score`
  MUST report the metric in this SAME direction so the two numbers are directly
  comparable; the verdict is decided by comparing them under "{better}".

To self-test, write your OWN small stub method.py implementing
`{design.get('method_interface')}` (a trivial legal solver), then run
`python _selftest.py 0` and confirm it writes result.json with valid=true.
Fix harness.py until that runs cleanly, then finish. This self-test does NOT
delete harness.py, so you can re-run `python _selftest.py 0` as many times as you
need. (Your stub method.py will be replaced by the real one at scoring time; do
not rely on its quality.)

You have a shell: if a self-test fails because a third-party module is missing,
you MAY `pip install <package>` to add it, then re-run the self-test.
"""


__all__ = ["write_harness"]
