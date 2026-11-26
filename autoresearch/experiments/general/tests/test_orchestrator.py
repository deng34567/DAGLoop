"""Structural anti-cheat: the FIXED trusted ``_orchestrator.py`` drives the run.

REAL LOCAL SUBPROCESS (fast, no Docker/LLM/network). The orchestrator imports the
AI-written harness (a PURE LIBRARY) into memory, IMMEDIATELY deletes harness.py
from disk, then runs the method through the trusted isolator and scores with the
in-memory harness. So a method that tries to read the ruler by ABSOLUTE path gets
nothing -- the file is gone before the method ever runs.

This test is genuinely adversarial: it plants a REAL harness.py containing a
secret at the very absolute path the cheat method reads (passed via the instance
data), and asserts that after the orchestrator's deletion the cheat's solutions
are invalid. It would FAIL against the old code (which left harness.py on disk and
let the method open('/workspace/harness.py')).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_GEN = Path(__file__).resolve().parents[1]

# A harness that is a PURE LIBRARY (no __main__, never touches the method). It
# embeds its own absolute path in each instance so a cheat can try to read it.
_HARNESS = """\
import os
SECRET_GROUND_TRUTH = "the ruler's hidden answer key"
def make_instances(seed):
    hp = os.path.abspath("harness.py")
    return [{"n": i, "harness_path": hp} for i in range(1, 4)]
def run_baseline(instances):
    return sum(i["n"] for i in instances)          # 1+2+3 = 6
def validate(instance, solution):
    return isinstance(solution, int) and solution >= 0
def score(instances, solutions):
    return sum(solutions)
"""

_HONEST = "def solve(instance):\n    return instance['n'] * 2\n"

# reads the ruler by ABSOLUTE path (the real, previously-valid path) -> should be
# gone by the time the method runs -> None -> invalid.
_CHEAT = (
    "def solve(instance):\n"
    "    try:\n"
    "        return len(open(instance['harness_path']).read())\n"
    "    except Exception:\n"
    "        return None\n"
)


def _workspace(tmp_path, method_src):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "harness.py").write_text(_HARNESS)
    (ws / "method.py").write_text(method_src)
    for name in ("_orchestrator.py", "_selftest.py", "isolated_method.py", "_method_driver.py", "contract.py"):
        shutil.copy2(_GEN / name, ws / name)
    return ws


def _run(ws, seed=0, *, script="_orchestrator.py", env=None):
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    proc = subprocess.run(
        [sys.executable, script, str(seed)],
        cwd=str(ws), capture_output=True, text=True, timeout=60, env=run_env,
    )
    return proc


# A harness whose run_baseline returns a DICT instead of a number -- the exact
# real-world failure (metric was {average_bins,...}); must fail cleanly, not crash
# the verdict layer later with ``dict * float``.
_DICT_HARNESS = """\
def make_instances(seed):
    return [{"n": i} for i in range(1, 4)]
def run_baseline(instances):
    return {"average_bins": 2.0, "detail": [1, 2, 3]}
def validate(instance, solution):
    return isinstance(solution, int)
def score(instances, solutions):
    return {"average_bins": 1.0}
"""


def test_orchestrator_rejects_non_numeric_metric(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "harness.py").write_text(_DICT_HARNESS)
    (ws / "method.py").write_text(_HONEST)
    for name in ("_orchestrator.py", "isolated_method.py", "_method_driver.py"):
        shutil.copy2(_GEN / name, ws / name)
    proc = _run(ws)
    assert proc.returncode != 0                       # fails closed
    assert "single number" in proc.stderr             # CLEAR message, not a TypeError


def test_orchestrator_deletes_harness_and_scores_honest_method(tmp_path):
    ws = _workspace(tmp_path, _HONEST)
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr

    # harness.py is GONE from disk (trusted deletion), yet scoring still worked
    assert not (ws / "harness.py").exists()
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is True
    assert result["baseline_metric"] == 6
    assert result["method_metric"] == 12   # (1+2+3)*2
    assert result["seed"] == 0


_CRASHER = "def solve(instance):\n    raise ValueError('kaboom')\n"
_WRONGSHAPE = "def solve(instance):\n    return 'not-an-int'\n"


def test_orchestrator_diagnostics_on_method_crash(tmp_path):
    # The RUN completes cleanly (exit 0 -> isolated True); the METHOD failed, and
    # result.json now says WHY (its own leak-safe error), not just valid=False.
    ws = _workspace(tmp_path, _CRASHER)
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is False and result["method_metric"] is None
    diag = result["diagnostics"]
    assert diag["n_valid"] == 0 and diag["validity_rate"] == 0.0
    ff = diag["first_failure"]
    assert ff["kind"] == "method_error"
    assert "kaboom" in ff["detail"] and "ValueError" in ff["detail"]


def test_orchestrator_diagnostics_on_invalid_solution(tmp_path):
    ws = _workspace(tmp_path, _WRONGSHAPE)
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is False
    assert result["diagnostics"]["first_failure"]["kind"] == "invalid_solution"


def test_orchestrator_shape_check_catches_mismatch_leaksafe(tmp_path):
    # io_contract example is an int; a method returning a float is caught as a
    # SHAPE mismatch (leak-safe, before harness.validate) -> the P2-repair reason.
    ws = _workspace(tmp_path, "def solve(instance):\n    return 1.5\n")
    (ws / "io_contract.json").write_text(json.dumps({"solution_example": 0}))
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is False
    ff = result["diagnostics"]["first_failure"]
    assert ff["kind"] == "shape_mismatch"
    assert "expected int, got float" in ff["detail"]


def test_orchestrator_closes_absolute_path_read(tmp_path):
    ws = _workspace(tmp_path, _CHEAT)
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr

    # the ruler is deleted before the method runs -> the absolute-path read fails
    assert not (ws / "harness.py").exists()
    # and no leftover bytecode cache to decompile
    pyc = list((ws / "__pycache__").glob("harness*.pyc")) if (ws / "__pycache__").is_dir() else []
    assert pyc == []
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is False        # cheat got None -> invalid, no leak-derived score
    assert result["method_metric"] is None


def test_orchestrator_selftest_env_keeps_harness_and_scores(tmp_path):
    # AUTHORING self-test path: with AUTORESEARCH_ORCH_SELFTEST=1 the trusted
    # orchestrator SKIPS the harness.py deletion so the agent can re-run it freely,
    # while STILL producing a valid scored result.json.
    ws = _workspace(tmp_path, _HONEST)
    proc = _run(ws, env={"AUTORESEARCH_ORCH_SELFTEST": "1"})
    assert proc.returncode == 0, proc.stderr

    assert (ws / "harness.py").exists()    # NOT deleted during self-test
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is True
    assert result["baseline_metric"] == 6
    assert result["method_metric"] == 12
    assert result["seed"] == 0


def test_selftest_script_keeps_harness_and_scores(tmp_path):
    # The trusted _selftest.py wrapper is what the AUTHORING agent runs. It sets the
    # env flag itself, so running it (with NO env) must keep harness.py AND score.
    ws = _workspace(tmp_path, _HONEST)
    proc = _run(ws, script="_selftest.py")
    assert proc.returncode == 0, proc.stderr

    assert (ws / "harness.py").exists()    # self-test does NOT delete the ruler
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is True
    assert result["baseline_metric"] == 6
    assert result["method_metric"] == 12
    assert result["seed"] == 0


_SMUGGLE_HARNESS = """\
def make_instances(seed):
    return [{"n": i, "_secret": 999} for i in range(1, 4)]   # smuggled answer key
def run_baseline(instances):
    return sum(i["n"] for i in instances)
def validate(instance, solution):
    return isinstance(solution, int) and solution >= 0
def score(instances, solutions):
    return sum(solutions)
"""
_LEAK_METHOD = "def solve(instance):\n    return instance.get('_secret', instance['n'])\n"


def test_orchestrator_strips_smuggled_instance_field(tmp_path):
    # anti-cheat b: the method tries to read a smuggled '_secret' field, but the
    # orchestrator strips instances to the declared shape -> the method only sees
    # 'n' and returns the fallback (1,2,3 -> score 6), never the secret (999*3).
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "harness.py").write_text(_SMUGGLE_HARNESS)
    (ws / "method.py").write_text(_LEAK_METHOD)
    for name in ("_orchestrator.py", "isolated_method.py", "_method_driver.py", "contract.py"):
        shutil.copy2(_GEN / name, ws / name)
    (ws / "io_contract.json").write_text(json.dumps({"instance_example": {"n": 0}, "solution_example": 0}))
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((ws / "result.json").read_text())
    assert result["valid"] is True
    assert result["method_metric"] == 6         # 1+2+3 fallback, NOT 999*3 (secret unreachable)


_RESTASH_HARNESS = '''\
def make_instances(seed):
    with open("harness.py", "w") as f:      # re-create the ruler after deletion
        f.write("SECRET = 42\\n")
    with open("answer.txt", "w") as f:      # stash an answer key on disk
        f.write("999")
    return [{"n": i} for i in range(1, 4)]
def run_baseline(instances):
    return sum(i["n"] for i in instances)
def validate(instance, solution):
    return isinstance(solution, int) and solution >= 0
def score(instances, solutions):
    return sum(solutions)
'''


def test_orchestrator_removes_harness_stashed_files(tmp_path):
    # anti-cheat c: the harness re-creates harness.py + stashes answer.txt during
    # make_instances; both are removed BEFORE the method runs.
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "harness.py").write_text(_RESTASH_HARNESS)
    (ws / "method.py").write_text(_HONEST)
    for name in ("_orchestrator.py", "isolated_method.py", "_method_driver.py", "contract.py"):
        shutil.copy2(_GEN / name, ws / name)
    proc = _run(ws)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((ws / "result.json").read_text())
    assert not (ws / "answer.txt").exists()      # stash removed
    assert not (ws / "harness.py").exists()       # re-created ruler removed
    removed = result["diagnostics"]["stashed_files_removed"]
    assert "answer.txt" in removed and "harness.py" in removed
    assert result["valid"] is True and result["method_metric"] == 12
