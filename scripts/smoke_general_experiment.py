#!/usr/bin/env python
"""Opt-in REAL end-to-end smoke for the GENERAL honest experiment. NOT part of the
pytest suite.

For a USER's research question (a COMMAND-LINE ARG -- never hardcoded), agents
write a whole experiment as real code (an AI "ruler" harness + an isolated method)
and run it in a REAL Docker container with honest, isolated verification.

Starts a REAL Docker container and makes REAL OpenRouter LLM calls (Kimi). Run
MANUALLY, SERIAL, single-session only (double-gated opt-in):

    AUTORESEARCH_REAL_SANDBOX=1 OPENROUTER_KEY=sk-... \
        python scripts/smoke_general_experiment.py \
        "Can a heuristic beat First-Fit on synthetic 1D bin packing?"

If no question arg is given, the default bin-packing question is used (still
passed AS AN ARGUMENT into the pipeline, not baked into the code path).

Prints every stage: design, the agent-written harness.py + method.py, the real
baseline-vs-method numbers, the reproduce run, the isolation evidence, the
verdict, and the honest report.

NOTE (version compat): if the container handshake fails, bump
AUTORESEARCH_SANDBOX_IMAGE to a tag matching the host openhands SDK and rerun.
"""
from __future__ import annotations

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
from autoresearch.experiments.general import pipeline as gen_pipeline  # noqa: E402
from autoresearch.llm import HKGAIClient  # noqa: E402

_DEFAULT_QUESTION = (
    "Can a simple heuristic beat First-Fit on synthetic 1D bin packing "
    "(capacity 1.0, item sizes ~ U(0,0.7)) measured by number of bins used?"
)


def _emit(event, payload):
    print(f"\n===== [{event}] =====")
    if event == "design_ready":
        for k, v in (payload.get("design") or {}).items():
            print(f"  {k}: {v}")
    elif event == "harness_ready":
        print(payload.get("harness"))
    elif event == "method_ready":
        print(payload.get("method"))
    elif event == "run_done":
        print("  primary  :", payload.get("primary"))
        print("  reproduce:", payload.get("reproduce"))
    elif event == "verdict":
        print("  verdict:", payload.get("verdict"))


def main() -> int:
    if not os.environ.get("OPENROUTER_KEY"):
        print("OPENROUTER_KEY not set; export it and rerun. Aborting.")
        return 2
    if os.environ.get("AUTORESEARCH_REAL_SANDBOX") != "1":
        print("Set AUTORESEARCH_REAL_SANDBOX=1 to run the REAL smoke. Aborting.")
        return 2

    question = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_QUESTION
    print(f"[smoke] question: {question}")

    import httpx

    llm = HKGAIClient(
        api_key=os.environ["OPENROUTER_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=os.environ.get("AUTORESEARCH_MODEL", "moonshotai/kimi-k2"),
        response_format_mode=None,
        http_client=httpx.Client(trust_env=False, timeout=180),
    )
    backend = OpenHandsBackend()

    result = gen_pipeline.run_pipeline(backend, llm, question, emit=_emit)

    print("\n========== HONEST EXPERIMENT REPORT ==========")
    print(result.get("report"))
    print("==============================================")
    print(f"verdict={result['verdict']['verdict']} accepted={result['verdict']['accepted']} "
          f"isolated={result['isolated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
