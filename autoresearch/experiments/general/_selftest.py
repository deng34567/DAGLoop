"""FIXED, TRUSTED authoring self-test wrapper (NOT AI-written).

The harness-writer agent runs ``python _selftest.py <seed>`` (NOT
``python _orchestrator.py``) to self-test its library harness during AUTHORING.

It sets ``AUTORESEARCH_ORCH_SELFTEST=1`` and then delegates to the SAME trusted
``_orchestrator.main()``. With that flag set, the orchestrator skips its
``harness.py`` deletion (the deletion is a SCORING-time anti-cheat, not an
authoring concern), so the agent can re-run the self-test as many times as it
likes without its ``harness.py`` disappearing after the first run. Everything else
-- import harness, make_instances, run_baseline, isolated method run, validate,
score, write result.json -- is byte-identical to scoring.

The SCORING runner never runs this wrapper and never sets the flag, so at scoring
time ``_orchestrator.py`` still deletes harness.py (anti-cheat intact).
"""
import os

os.environ["AUTORESEARCH_ORCH_SELFTEST"] = "1"

from _orchestrator import main

if __name__ == "__main__":
    main()
