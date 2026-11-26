"""b-layer: ``MonteCarloExperimentRunner`` -- builds the MC experiment inside a
container via M9's ``OpenHandsBackend`` (baseline run + agent-written method),
then scores it with TRUSTED code the runner re-uploads itself.

The default test suite never touches Docker/LLM: tests pass a mock ``handle``
(the M9 ``SandboxHandle`` seam) and mock ``file_upload`` / ``execute_command`` /
``run_agent_task``. Real connectivity is opt-in only (Task 10 smoke script).

Anti-gaming: the integrand source is WITHHELD while the agent authors method.py
(so it cannot ``cat integrand.py`` and inline the closed form -- it must adapt
via the budgeted ``sample_f``), then EVERY scoring run first RE-UPLOADS all
trusted files (run.py / integrand.py / problem.py / mc_evaluator.py) over
whatever the agent left, keeping only the agent's ``method.py``, and ONLY THEN
runs ``python run.py`` (which deletes integrand.py again after importing it, so
each score must re-upload). The scored metric always comes from trusted code. A
second ``reproduce`` score re-runs the same method on FRESH disjoint seeds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend

# Generous wall-clock SAFETY BACKSTOP (seconds) on the in-container agent loop.
# Time is not the real constraint (OpenRouter, not local compute), so this is a
# loose backstop, not a leash; env-configurable, default 2h.
_MC_AGENT_TIMEOUT_ENV = "AUTORESEARCH_MC_AGENT_TIMEOUT"
_DEFAULT_AGENT_TIMEOUT = 7200.0

# method.py is handled separately (agent-written); these are the trusted files
# re-uploaded to guarantee the score comes from trusted code -- including the
# fixed reference baseline.py and the isolation driver _method_driver.py.
_TRUSTED = (
    "run.py", "integrand.py", "problem.py", "mc_evaluator.py",
    "baseline.py", "_method_driver.py",
)

# Files uploaded for the AUTHORING turn -- everything trusted EXCEPT integrand.py,
# which is withheld so the agent cannot read the closed form and inline it.
_AUTHORING_FILES = tuple(f for f in _TRUSTED if f != "integrand.py")

# Trusted files whose in-container copy is hash-compared to the host original
# BEFORE re-upload -- a SECONDARY tamper signal (scoring is already guaranteed by
# the re-upload, so this only records/alerts; it is not the gate). integrand.py
# is excluded: it is withheld during authoring, so it is not present to compare.
_TAMPER_CHECK = (
    "run.py", "mc_evaluator.py", "baseline.py", "_method_driver.py",
)

# Reproduce-scoring seeds -- DISJOINT from DEFAULT_PROBLEM's [0..7] so a passing
# reproduce run shows the method generalises to UNSEEN seeds, not overfits.
# Env-overridable (comma list).
_REPRODUCE_SEEDS_ENV = "AUTORESEARCH_MC_REPRODUCE_SEEDS"


def _reproduce_seeds() -> list[int]:
    raw = os.environ.get(_REPRODUCE_SEEDS_ENV)
    if raw:
        return [int(s) for s in raw.split(",") if s.strip()]
    return list(range(100, 108))


REPRODUCE_SEEDS = _reproduce_seeds()

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
MC_DIR = _EXPERIMENTS_DIR.parent / "benchmarks" / "mc_variance_reduction"


class MonteCarloExperimentRunner:
    """Runs the MC variance-reduction experiment via OpenHandsBackend."""

    def __init__(self, backend=None, benchmark_dir=MC_DIR) -> None:
        self.backend = backend if backend is not None else OpenHandsBackend()
        self.benchmark_dir = Path(benchmark_dir)

    # -- internals --

    def _local_path(self, name: str) -> Path:
        # mc_evaluator.py lives with the b-layer package; the rest in the bench dir
        if name == "mc_evaluator.py":
            return _EXPERIMENTS_DIR / name
        return self.benchmark_dir / name

    def _upload(self, handle, names) -> None:
        # M9 has NO whole-directory upload API -> loop single-file file_upload.
        for n in names:
            handle.workspace.file_upload(str(self._local_path(n)), f"/workspace/{n}")

    def _ensure_deps(self, handle) -> None:
        # the in-container trusted evaluator needs scipy/numpy for the quadrature truth
        self.backend.execute_command(handle, "pip install -q scipy numpy")

    def _read_result(self, handle) -> dict:
        r = self.backend.execute_command(handle, "cat result.json")
        return json.loads(r.stdout)

    def _score(self, handle) -> dict:
        """Run the TRUSTED harness and read its result.

        Finding 2: delete any planted result.json first, and if scoring exits
        non-zero (e.g. an agent's import-time crash) return metric 0 -- NEVER
        read a stale/planted result.json.
        """
        self.backend.execute_command(handle, "rm -f result.json")
        r = self.backend.execute_command(handle, "python run.py")
        if r.exit_code != 0:
            return {"metric_value": 0.0, "scoring_failed": True}
        return self._read_result(handle)

    def _score_run(self, handle, seeds=None) -> dict:
        """Score with TRUSTED code. FIRST re-upload the full trusted set (incl.
        integrand.py -- run.py deletes it after import, so every score needs it
        back), THEN drop any planted result.json, THEN run the harness.

        ``seeds`` (if given) is passed INLINE as an env prefix because
        ``execute_command`` has no env kwarg. A non-zero scoring exit scores 0
        (Finding 2: never read a stale/planted result.json)."""
        self._upload(handle, _TRUSTED)
        self.backend.execute_command(handle, "rm -f result.json")
        if seeds:
            cmd = f"AUTORESEARCH_MC_SEEDS={','.join(str(s) for s in seeds)} python run.py"
        else:
            cmd = "python run.py"
        r = self.backend.execute_command(handle, cmd)
        if r.exit_code != 0:
            return {"metric_value": 0.0, "scoring_failed": True}
        return self._read_result(handle)

    def _read_method_code(self, handle) -> str:
        return self.backend.execute_command(handle, "cat method.py").stdout

    def _tamper_detected(self, handle) -> bool:
        """True if the agent left any trusted file differing from the original."""
        import hashlib

        def _h(s: str) -> str:
            return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()

        for name in _TAMPER_CHECK:
            remote = self.backend.execute_command(handle, f"cat {name}").stdout
            local = self._local_path(name).read_text(encoding="utf-8")
            if _h(remote) != _h(local):
                return True
        return False

    # -- public API --

    def run_baseline(self, handle=None) -> dict:
        """Score the baseline naive-MC method.py inside the container."""
        own = handle is None
        if own:
            handle = self.backend.open_sandbox()
        try:
            self._upload(handle, _TRUSTED + ("method.py",))
            self._ensure_deps(handle)
            return self._score(handle)
        finally:
            if own:
                self.backend.close_sandbox(handle)

    def run_experiment(self, research_task, handle=None) -> dict:
        """Let the in-container agent write ``method.py``, then score it with
        TRUSTED code the runner re-uploads (agent's ``run.py`` etc. discarded)."""
        own = handle is None
        if own:
            handle = self.backend.open_sandbox()
        try:
            # 0) AUTHORING layout WITHHOLDS integrand.py (agent can't cat the
            #    closed form); it only gets the harness + baseline to work against.
            self._upload(handle, _AUTHORING_FILES + ("method.py",))
            self._ensure_deps(handle)

            # 1) agent autonomously edits /workspace/method.py (generous time backstop)
            timeout = float(os.environ.get(_MC_AGENT_TIMEOUT_ENV, _DEFAULT_AGENT_TIMEOUT))
            run = self.backend.run_agent_task(
                handle, _contract_prompt(research_task), timeout=timeout
            )
            # 2) keep the agent's method source for the record
            code = self._read_method_code(handle)
            # 3) SECONDARY signal: did the agent touch any trusted file? (record only)
            tamper_detected = self._tamper_detected(handle)
            # 4) PRIMARY score: _score_run re-uploads the trusted set (incl.
            #    integrand.py) OVER the agent's edits, deletes any planted
            #    result.json, and zeros on a non-zero scoring exit (Finding 2).
            primary = self._score_run(handle)
            # 5) REPRODUCE score: SAME method.py, FRESH disjoint seeds (inline env).
            #    integrand.py was deleted by the primary run.py, so re-upload again.
            repro = self._score_run(handle, seeds=REPRODUCE_SEEDS)

            return {
                "metric_value": primary["metric_value"],
                "reproduce_metric_value": repro["metric_value"],
                "reproduce_seeds": list(REPRODUCE_SEEDS),
                "rmse_baseline": primary.get("rmse_baseline"),
                "rmse_proposed": primary.get("rmse_proposed"),
                "over_budget": primary.get("over_budget"),
                "hardcoded_suspected": primary.get("hardcoded_suspected"),
                "agent_status": run.status,
                "code": code,
                "anti_cheat": {
                    "over_budget": primary.get("over_budget"),
                    "hardcoded_suspected": primary.get("hardcoded_suspected"),
                    "bias": primary.get("bias"),
                    "tamper_detected": tamper_detected,
                },
            }
        finally:
            if own:
                self.backend.close_sandbox(handle)


def _contract_prompt(research_task) -> str:
    """The interaction contract handed to the in-container agent."""
    return f"""Research task: {research_task}

You are optimizing a Monte-Carlo integral estimate. Edit ONLY /workspace/method.py.

Implement:

    def proposed_estimate(problem, sample_f, n_samples, rng) -> float:
        ...

Rules (enforced by a trusted evaluator you cannot see):
- The integrand is TRUSTED and hidden. You may touch it ONLY through
  ``sample_f(x)`` (x is a list of floats in [0,1]^dim). ``sample_f`` COUNTS
  calls; making more than ``n_samples`` calls scores 0.
- ``problem`` has {{"dim", "n_samples", "seeds", ...}} but NO integration
  function. ``rng`` is a numpy Generator for reproducible sampling.
- Do NOT edit or read run.py, integrand.py, or mc_evaluator.py -- they are
  trusted and will be re-uploaded before scoring.
- Goal: LOWER the estimate's RMSE vs the naive-MC baseline using the SAME
  budget (importance sampling / control variates / antithetic variates).
- Write /workspace/method.py ONCE, then finish. Do NOT try to run ``run.py`` --
  the trusted grader (with the integrand) runs it separately; it is not present
  during authoring. Keep the method simple and finish promptly.
"""
