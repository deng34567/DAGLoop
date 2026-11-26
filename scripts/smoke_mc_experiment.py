#!/usr/bin/env python
"""Opt-in REAL end-to-end smoke for the Monte-Carlo variance-reduction experiment.
NOT part of the pytest suite.

This starts a REAL Docker container and makes REAL OpenRouter LLM calls (Kimi).
Run MANUALLY, SERIAL, single-session only:

    AUTORESEARCH_REAL_SANDBOX=1 OPENROUTER_KEY=sk-... \
        python scripts/smoke_mc_experiment.py

Flow (single shared sandbox):
  1. run_baseline  -> trusted naive-MC scored by the trusted evaluator (ratio ~= 1).
  2. run_experiment -> the container agent writes a variance-reduction method.py;
     the runner RE-UPLOADS the trusted files over it and scores with trusted code.
Prints baseline/experiment RMSE + RMSE-reduction ratio + the agent's method.py +
the anti_cheat summary (over_budget / hardcoded_suspected / tamper_detected / bias).

NOTE (version compat): if the container handshake fails, bump
AUTORESEARCH_SANDBOX_IMAGE to a tag matching the host openhands SDK and rerun.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Clear proxy env: a local clash proxy breaks host<->docker and in-container
# OpenRouter calls (same reason vibe's clear_proxy_env exists).
for _var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_var, None)

# Allow running directly from the repo root (scripts/ parent).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend  # noqa: E402
from autoresearch.experiments import mc_acceptance  # noqa: E402
from autoresearch.experiments.mc_runner import MonteCarloExperimentRunner  # noqa: E402

RESEARCH_TASK = (
    "Reduce the RMSE of a Monte-Carlo integral estimate at a FIXED sample budget. "
    "Edit ONLY method.py; reach the (hidden, trusted) integrand solely through the "
    "budgeted sample_f (calling it more than n_samples times scores 0). Use importance "
    "sampling / control variates / antithetic variates."
)


def main() -> int:
    if not os.environ.get("OPENROUTER_KEY"):
        print("OPENROUTER_KEY not set; export it and rerun. Aborting.")
        return 2
    if os.environ.get("AUTORESEARCH_REAL_SANDBOX") != "1":
        print("Set AUTORESEARCH_REAL_SANDBOX=1 to run the REAL smoke. Aborting.")
        return 2

    backend = OpenHandsBackend()
    runner = MonteCarloExperimentRunner(backend=backend)
    handle = backend.open_sandbox()  # one shared container, serial
    try:
        print("[smoke] running trusted baseline (naive MC) ...")
        base = runner.run_baseline(handle=handle)
        print(f"[smoke] baseline: ratio={base.get('metric_value')} "
              f"rmse_baseline={base.get('rmse_baseline')} "
              f"over_budget={base.get('over_budget')}")

        print("[smoke] running experiment (container agent writes method.py) ...")
        exp = runner.run_experiment(RESEARCH_TASK, handle=handle)

        verdict = mc_acceptance.decide(exp)

        print("\n========== MC EXPERIMENT RESULT ==========")
        print(f"agent_status      : {exp.get('agent_status')}")
        print(f"rmse_baseline     : {exp.get('rmse_baseline')}")
        print(f"rmse_proposed     : {exp.get('rmse_proposed')}")
        print(f"RMSE reduction    : {exp.get('metric_value')}x")
        print(f"reproduce (fresh) : {exp.get('reproduce_metric_value')}x "
              f"(seeds {exp.get('reproduce_seeds')})")
        print(f"anti_cheat        : {json.dumps(exp.get('anti_cheat'), ensure_ascii=False)}")
        print(f"ACCEPTANCE        : {verdict['verdict']} (accepted={verdict['accepted']})")
        print(f"  reasons         : {'; '.join(verdict['reasons'])}")
        print("\n---------- agent's method.py ----------")
        print(exp.get("code") or "(none captured)")
        print("=========================================")
        return 0
    finally:
        backend.close_sandbox(handle)


if __name__ == "__main__":
    raise SystemExit(main())
