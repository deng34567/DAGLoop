"""TRUSTED driver that runs an untrusted ``method.py`` in PROCESS ISOLATION.

Spawned by ``isolated_method.run_method_isolated`` in a temp dir that contains
ONLY a copy of ``method.py`` + this file -- NO harness, NO instances file, NO
scoring code, with ``PYTHONPATH=""`` and cwd = that temp dir. So ``import harness``
and RELATIVE reads (``open('harness.py')``) fail here. ABSOLUTE reads
(``open('/workspace/harness.py')``) are defeated separately: the trusted
``_orchestrator.py`` deletes harness.py from disk BEFORE this runs, so the file is
already gone. The ONLY thing the method sees is one instance at a time, over the
pipe; the ONLY thing it can send back is that instance's solution.

Protocol (newline-delimited JSON over stdio):
- read one ``instance`` JSON object per line from stdin
- write ``{"solution": <solution>, "error": <err|null>}`` per line
- an unimportable method answers null for every instance (never crashes the batch)

``error`` (when non-null) is a leak-safe diagnostic about the METHOD'S OWN failure
-- ``{"phase": "import|solve|serialize", "type": <ExcType>, "msg": <short>}`` -- so
the trusted layer can tell "the method crashed" from "the method returned a
badly-shaped solution", and (P2) feed the method its own error back for repair.
It carries only the method's own exception, never any ruler/harness internals.
"""
import json
import sys

_MSG_CAP = 500


def main():
    import_error = None
    try:
        import method  # untrusted; imports resolved HERE, in the isolated temp dir
        solve = getattr(method, "solve", None)
        ok = callable(solve)
        if not ok:
            import_error = {"phase": "import", "type": "AttributeError",
                            "msg": "method.py has no callable solve(instance)"}
    except Exception as exc:  # noqa: BLE001 -- report, never crash the batch
        solve = None
        ok = False
        import_error = _err("import", exc)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            instance = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            _emit(None, _err("decode", exc))
            continue
        if not ok:
            _emit(None, import_error)
            continue
        try:
            solution = solve(instance)
        except Exception as exc:  # noqa: BLE001
            _emit(None, _err("solve", exc))
            continue
        try:
            json.dumps(solution)  # ensure it is JSON-serialisable; else -> null
        except Exception as exc:  # noqa: BLE001
            _emit(None, _err("serialize", exc))
            continue
        _emit(solution, None)


def _err(phase, exc):
    msg = str(exc)
    if len(msg) > _MSG_CAP:
        msg = msg[:_MSG_CAP] + "…"
    return {"phase": phase, "type": type(exc).__name__, "msg": msg}


def _emit(solution, error):
    sys.stdout.write(json.dumps({"solution": solution, "error": error}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
