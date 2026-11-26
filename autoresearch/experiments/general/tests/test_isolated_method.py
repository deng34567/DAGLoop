"""Task 3.5: the FIXED trusted isolated method runner. REAL LOCAL SUBPROCESS
(fast, no Docker / LLM / network). Proves an honest method's solutions come back
correctly AND that PROCESS isolation blocks a relative ``open('harness.py')`` /
``import harness``. The ABSOLUTE-path guarantee is the orchestrator's job (it
deletes harness.py before this runs) and is proved in ``test_orchestrator.py``.
"""
from __future__ import annotations

from pathlib import Path

from autoresearch.experiments.general.isolated_method import run_method_isolated

_HONEST = "def solve(instance):\n    return instance['n'] + 1\n"

# tries every route to the ruler: a sibling file, an absolute /workspace path, and
# The isolator guarantees PROCESS isolation: a RELATIVE read + an `import harness`
# both fail from the fresh temp cwd (empty PYTHONPATH, ruler not copied in).
# (ABSOLUTE-path protection is NOT the isolator's job -- the orchestrator deletes
# harness.py before this runs; that is proved in test_orchestrator.py.)
_CHEAT = (
    "def solve(instance):\n"
    "    try:\n"
    "        return open('harness.py').read()   # relative -> not in the temp cwd\n"
    "    except Exception:\n"
    "        pass\n"
    "    import harness   # not importable (empty PYTHONPATH, not in cwd)\n"
    "    return harness.SECRET\n"
)

_CRASH_ONE = (
    "def solve(instance):\n"
    "    if instance['n'] == 2:\n"
    "        raise ValueError('boom')\n"
    "    return instance['n']\n"
)


def _write_workspace(tmp_path, method_src):
    """A workspace with method.py AND a sentinel harness.py sibling -- to prove the
    isolator copies ONLY method.py into a fresh dir, so the sibling is unreachable."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "harness.py").write_text("SECRET = 'the ruler internals'\n")
    mp = ws / "method.py"
    mp.write_text(method_src)
    return mp


def test_honest_method_solutions_come_back(tmp_path):
    mp = _write_workspace(tmp_path, _HONEST)
    instances = [{"n": 1}, {"n": 5}, {"n": 9}]
    out = run_method_isolated(instances, str(mp))
    assert out == [2, 6, 10]


def test_process_isolation_blocks_relative_read_and_import(tmp_path):
    mp = _write_workspace(tmp_path, _CHEAT)
    instances = [{"n": 1}, {"n": 2}]
    out = run_method_isolated(instances, str(mp))
    # relative read + import both fail from the isolated temp cwd -> no solution
    assert out == [None, None]


def test_one_bad_instance_does_not_kill_the_batch(tmp_path):
    mp = _write_workspace(tmp_path, _CRASH_ONE)
    instances = [{"n": 1}, {"n": 2}, {"n": 3}]
    out = run_method_isolated(instances, str(mp))
    assert out == [1, None, 3]


def test_unimportable_method_yields_all_none(tmp_path):
    mp = _write_workspace(tmp_path, "import definitely_not_a_real_module_xyz\n")
    out = run_method_isolated([{"n": 1}, {"n": 2}], str(mp))
    assert out == [None, None]


# --- P1: return_diagnostics surfaces the method's OWN error (leak-safe) --------

def test_diagnostics_report_per_instance_method_error(tmp_path):
    mp = _write_workspace(tmp_path, _CRASH_ONE)
    solutions, diag = run_method_isolated(
        [{"n": 1}, {"n": 2}, {"n": 3}], str(mp), return_diagnostics=True
    )
    assert solutions == [1, None, 3]                 # backward-compatible solutions
    errs = diag["errors"]
    assert errs[0] is None and errs[2] is None       # the good instances have no error
    assert errs[1]["phase"] == "solve"               # the crash is pinpointed
    assert errs[1]["type"] == "ValueError" and "boom" in errs[1]["msg"]
    assert diag["timed_out"] is False


def test_diagnostics_report_import_error(tmp_path):
    mp = _write_workspace(tmp_path, "import definitely_not_a_real_module_xyz\n")
    solutions, diag = run_method_isolated([{"n": 1}], str(mp), return_diagnostics=True)
    assert solutions == [None]
    assert diag["errors"][0]["phase"] == "import"
