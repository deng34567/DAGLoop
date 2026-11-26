"""FIXED, TRUSTED isolated method runner -- half of the run-time anti-cheat core.

``run_method_isolated(instances, method_path="method.py")`` scores nothing; it
just runs an untrusted ``method.py``'s ``solve(instance) -> solution`` in a
SEPARATE subprocess whose temp dir / cwd / ``PYTHONPATH`` contain ONLY the method
+ the driver. This gives PROCESS isolation: the method runs in its own process
(never in the scorer's), cannot share memory with the ruler, cannot ``import
harness`` (empty PYTHONPATH, not on the temp cwd), and cannot open the ruler by a
RELATIVE path (it isn't in the temp cwd).

It does NOT by itself stop an ABSOLUTE-path read (``open('/workspace/harness.py')``).
That hole is closed by the trusted ``_orchestrator.py``, which DELETES
``harness.py`` from disk BEFORE calling this -- so by the time the method runs the
ruler file is gone. Together (orchestrator deletion + this process isolation) the
method cannot read the ruler and cannot plant a fake ``result.json`` (only the
orchestrator writes it).

This is the generalisation of ``mc_evaluator.run_method_isolated`` (sample_f ->
generic solve(instance)). It is FIXED code shipped into the container next to the
harness; it is never written by an agent. Stdlib only; no Docker / LLM / network.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _find_driver() -> str:
    """Locate ``_method_driver.py`` (flat next to the orchestrator in the
    container, or next to this module on the host)."""
    candidates = [
        Path.cwd() / "_method_driver.py",
        Path(__file__).resolve().parent / "_method_driver.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError("_method_driver.py not found on any known path")


def run_method_isolated(instances, method_path="method.py", *, driver_path=None,
                        timeout=30.0, return_diagnostics=False):
    """Run ``method.py``'s ``solve(instance)`` on each instance in an ISOLATED
    subprocess. Returns a list of solutions, one per instance, with ``None`` for
    any instance the method failed / could not reach / timed out on.

    With ``return_diagnostics=True`` returns ``(solutions, diag)`` where ``diag``
    carries LEAK-SAFE signal about the method's OWN failures (for P1 diagnostics /
    the P2 repair loop): ``{"errors": [<per-instance err|None>], "stderr_tail":
    <str>, "timed_out": <bool>}``. Each error is the method's own exception
    (phase/type/msg) -- never any ruler internals. Default (no diagnostics) keeps
    the historical ``list[solution]`` contract.
    """
    instances = list(instances)
    driver_path = driver_path or _find_driver()
    tmp = Path(tempfile.mkdtemp(prefix="gen-iso-"))
    proc = None
    # stderr -> a real FILE (not a PIPE) so a chatty method can't fill a pipe
    # buffer and deadlock, yet its crash output is still captured for diagnostics.
    stderr_path = tmp / "_method_stderr.log"
    solutions: list = []
    errors: list = []
    timed_out = False
    try:
        shutil.copy2(method_path, tmp / "method.py")
        shutil.copy2(driver_path, tmp / "_method_driver.py")
        # inherit env so python/numpy import, but EMPTY PYTHONPATH so nothing extra
        # (e.g. /workspace) is importable -> `import harness` fails here. (Absolute
        # reads of harness.py are stopped by the orchestrator deleting it first.)
        env = {**os.environ, "PYTHONPATH": ""}
        stderr_f = open(stderr_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "_method_driver.py"],
            cwd=str(tmp),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            text=True,
        )
        stderr_f.close()  # the child holds its own fd; we read the file at the end
        deadline = time.monotonic() + timeout
        for inst in instances:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break
            try:
                proc.stdin.write(json.dumps(inst) + "\n")
                proc.stdin.flush()
            except Exception:
                break  # child died -> remaining instances stay unanswered
            rlist, _, _ = select.select([proc.stdout], [], [], remaining)
            if not rlist:
                timed_out = True
                proc.kill()
                break
            line = proc.stdout.readline()
            if line == "":
                break  # child ended / EOF
            try:
                obj = json.loads(line)
                solutions.append(obj.get("solution"))
                errors.append(obj.get("error"))
            except Exception:
                solutions.append(None)
                errors.append({"phase": "protocol", "type": "JSONDecodeError",
                               "msg": "driver emitted a non-JSON line"})
        # pad any unanswered instances (child died / timed out) with None
        while len(solutions) < len(instances):
            solutions.append(None)
            errors.append({"phase": "unreached", "type": "ChildEnded",
                           "msg": "method process ended/timed out before this instance"})
        # finish the child BEFORE reading its stderr file, so its crash output is
        # flushed (a hard crash the per-instance protocol couldn't report).
        _finish(proc)
        proc = None
        if not return_diagnostics:
            return solutions
        return solutions, {
            "errors": errors,
            "stderr_tail": _read_tail(stderr_path),
            "timed_out": timed_out,
        }
    finally:
        _finish(proc)
        shutil.rmtree(tmp, ignore_errors=True)


def _finish(proc):
    if proc is None:
        return
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def _read_tail(path, cap=1500):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return text[-cap:] if len(text) > cap else text


__all__ = ["run_method_isolated"]
