"""Task 4: the Docker runner -- assemble the experiment in a container and run it
THROUGH the FIXED trusted orchestrator (not the AI harness).

``run_experiment(backend, harness_code, method_code, seed, *, handle=None) -> dict``:
1. Write ``harness_code`` / ``method_code`` STRINGS to LOCAL temp files, then
   ``file_upload`` them to /workspace (M9 has only local-file -> remote upload;
   there is no direct-string upload). ALSO upload the FIXED trusted modules
   ``_orchestrator.py`` + ``isolated_method.py`` + ``_method_driver.py``. Then
   ``pip install`` the DECLARED ``dependencies`` (sanitized + shell-quoted; the
   fixed numpy/scipy install is gone -- deps come from the design).
2. ``rm -f /workspace/result.json`` (never trust a planted/stale result).
3. Run ``python _orchestrator.py <seed>`` (the TRUSTED orchestrator drives the
   run: it imports the harness library, DELETES harness.py, runs the method via
   the isolator, and writes result.json). The AI harness is NEVER run directly.
   A NON-ZERO exit -> invalid, and the stale result.json is NOT read.
4. On exit 0, read result.json (``{baseline_metric, method_metric, valid, seed}``).
5. ``isolated`` is STRUCTURAL, not a spoofable textual check: it is True iff the
   trusted orchestrator ran to completion (exit 0). Because that trusted path
   always deletes harness.py and runs the method only inside the isolator,
   completion == isolation held by construction.

Default test suite never touches Docker: tests inject a mock ``handle`` + mock
``file_upload`` / ``execute_command``.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from autoresearch.experiments.general.design import pip_install_command

_GEN_DIR = Path(__file__).resolve().parent
# Uploaded on TOP of the AI files; order-independent. _orchestrator drives the run.
# contract.py is the fixed, leak-safe shape checker (P3) the orchestrator uses.
_TRUSTED_MODULES = ("_orchestrator.py", "isolated_method.py", "_method_driver.py", "contract.py")
_WORKSPACE = "/workspace"

# generous per-command backstop for the (possibly slow) run; env-config.
_RUN_TIMEOUT_ENV = "AUTORESEARCH_GEN_RUN_TIMEOUT"
_DEFAULT_RUN_TIMEOUT = 1800.0


def _run_timeout() -> float:
    return float(os.environ.get(_RUN_TIMEOUT_ENV, _DEFAULT_RUN_TIMEOUT))


def run_experiment(backend, harness_code, method_code, seed, *, handle=None,
                   dependencies=(), io_contract=None) -> dict:
    """Run one experiment (a single seed) in a container via the trusted
    orchestrator; return the trusted result enriched with the structural
    ``isolated`` flag. ``io_contract`` (the design's concrete shape example) is
    shipped so the orchestrator can run the leak-safe shape check."""
    own = handle is None
    if own:
        handle = backend.open_sandbox()
    try:
        # 1) strings -> local temp files -> upload (M9 uploads local files only)
        with tempfile.TemporaryDirectory(prefix="gen-upload-") as td:
            hp = Path(td) / "harness.py"
            mp = Path(td) / "method.py"
            hp.write_text(harness_code, encoding="utf-8")
            mp.write_text(method_code, encoding="utf-8")
            handle.workspace.file_upload(str(hp), f"{_WORKSPACE}/harness.py")
            handle.workspace.file_upload(str(mp), f"{_WORKSPACE}/method.py")
            # the concrete shape example (public API contract) for check_shape
            cp = Path(td) / "io_contract.json"
            cp.write_text(json.dumps(io_contract or {}), encoding="utf-8")
            handle.workspace.file_upload(str(cp), f"{_WORKSPACE}/io_contract.json")
        # ... plus the FIXED trusted modules (orchestrator + isolator) from this project
        for name in _TRUSTED_MODULES:
            handle.workspace.file_upload(str(_GEN_DIR / name), f"{_WORKSPACE}/{name}")

        # 1b) install the DECLARED dependencies (whatever this experiment needs,
        #     not a fixed numpy/scipy) so the AI harness/method can import them;
        #     names are sanitized + shell-quoted before they reach the shell.
        pip_cmd = pip_install_command(list(dependencies))
        if pip_cmd:
            backend.execute_command(handle, pip_cmd, cwd=_WORKSPACE)

        # 2) never read a planted/stale result
        backend.execute_command(handle, f"rm -f {_WORKSPACE}/result.json", cwd=_WORKSPACE)

        # 3) drive the run via the TRUSTED orchestrator (seed = argv[1]); the AI
        #    harness is only ever imported (as a library) by the orchestrator.
        r = backend.execute_command(
            handle, f"python _orchestrator.py {int(seed)}",
            cwd=_WORKSPACE, timeout=_run_timeout(),
        )
        # isolation is STRUCTURAL: the trusted path (which deletes harness.py and
        # runs the method only via the isolator) completed iff exit 0.
        isolated = r.exit_code == 0
        if not isolated:
            return {
                "valid": False,
                "method_metric": None,
                "baseline_metric": None,
                "seed": int(seed),
                "error": (r.stderr or r.stdout or "orchestrator exited non-zero"),
                "isolated": False,
            }

        # 4) trusted result only on a clean exit
        rr = backend.execute_command(handle, f"cat {_WORKSPACE}/result.json", cwd=_WORKSPACE)
        try:
            data = json.loads(rr.stdout)  # NOTE(P2): stdout-protocol fragility -- fails closed below
        except (json.JSONDecodeError, TypeError):
            return {"valid": False, "method_metric": None, "baseline_metric": None,
                    "seed": int(seed), "error": "unreadable result.json",
                    "isolated": False}
        data["isolated"] = True
        return data
    finally:
        if own:
            backend.close_sandbox(handle)


__all__ = ["run_experiment"]
