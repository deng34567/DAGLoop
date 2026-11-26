"""Benchmark shape for the repo-improvement capability.

A ``Benchmark`` describes a REAL external repo whose baseline our agents try to improve
under a fixed budget, scored by a TRUSTED evaluator the agent cannot touch: agent edits
``editable_files``; ``protected_files`` (data prep + the evaluator) are restored from the
pristine clone before scoring, and the official metric is computed by trusted code -- the
agent's self-reported number is never trusted.

There is NO hand-registered benchmark: ``REGISTRY`` is empty on purpose. Every request is
resolved DYNAMICALLY (``discover`` -> LLM names the repo -> clone -> ``infer_benchmark``
derives this boundary from the cloned files). A static registry entry was removed because
its keywords greedily hijacked unrelated requests (any "single-GPU training speedrun" ->
that one repo). ``EXAMPLE_BENCHMARK`` below is a NEUTRAL fixture for the unit tests only --
it is not registered and never matched.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Benchmark:
    name: str
    description: str
    repo_url: str
    editable_files: tuple[str, ...]      # the only files the agent may modify
    protected_files: tuple[str, ...]     # trusted prep/eval; restored before scoring
    prepare_cmd: str                     # one-time data + tokenizer prep (trusted)
    run_cmd: str                         # trains under the budget, saves a checkpoint
    metric: str                          # e.g. "val_bpb"
    lower_is_better: bool
    budget_s: int                        # wall-clock training budget (seconds)
    keywords: tuple[str, ...] = field(default_factory=tuple)  # for request -> benchmark
    # Optional TRUSTED re-eval: files we write into the workspace AFTER the agent's patch
    # (so the agent can't have altered them) + a command that recomputes the metric from
    # the run's saved artifacts using the pristine evaluator. When set, its output -- not
    # the number train.py printed -- is authoritative (closes the self-report residual).
    trusted_eval_files: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trusted_eval_cmd: str | None = None
    # Env var the trusted runner sets to pin the training seed (for reproduce). The
    # benchmark's training must read it for reproduce to be meaningful.
    seed_env: str | None = None
    # Docker image this repo needs to run in when the host/base sandbox can't provide its
    # environment (e.g. a CUDA/CUTLASS kernel benchmark shipping its own Dockerfile). None =
    # run on the host / in the default sandbox. A value can be an existing image tag OR a
    # ``build:<Dockerfile-relpath>`` spec the pipeline builds from the cloned repo. Inferred
    # by infer_benchmark from the repo's own files -- never a hard-coded per-repo pin.
    image: str | None = None


# Neutral fixture for unit tests (propose_patch / run_variant / boundary shape). NOT
# registered and NOT matched by discover -- keywords are empty on purpose.
EXAMPLE_BENCHMARK = Benchmark(
    name="example-train-speedrun",
    description="A generic single-file training benchmark: agent edits train.py; prepare.py "
                "(data + trusted evaluator) is protected; metric = val_bpb (lower better).",
    repo_url="https://github.com/example/train-speedrun",
    editable_files=("train.py",),
    protected_files=("prepare.py",),
    prepare_cmd="uv run prepare.py",
    run_cmd="uv run train.py",
    metric="val_bpb",
    lower_is_better=True,
    budget_s=300,
    keywords=(),
)

# Intentionally EMPTY: all discovery is dynamic (see module docstring / discover.py).
REGISTRY: dict[str, Benchmark] = {}


def get(name: str) -> Benchmark | None:
    return REGISTRY.get(name)


__all__ = ["Benchmark", "REGISTRY", "EXAMPLE_BENCHMARK", "get"]
