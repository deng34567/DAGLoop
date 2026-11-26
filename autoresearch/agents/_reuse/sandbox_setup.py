"""In-sandbox setup discovery: let an OpenHands agent ITERATE inside a GPU sandbox
until a repo's baseline actually runs, then HARD-VERIFY what it claims.
在 GPU 沙箱里让 OpenHands agent 反复试错,直到仓库的 baseline 真的跑起来;
它自述的结果一律不信,由我们重跑硬判。

Why this exists
---------------
``repo_improve.discover.correct_execution_from_run`` heals a broken setup/metric by READING the repo's
files and guessing corrected commands, bounded at 2 rounds. That is enough for a
"just download the data" repo (minGPT) but not for one whose setup is a multi-step
chain nobody can guess from a file listing -- nanochat needs
``uv sync --extra gpu`` -> download HF parquet shards -> **train a tokenizer**
(``scripts.tok_train``, because ``get_tokenizer()`` asserts on token bytes that
only that script writes) -> only then can ``base_train`` run. A human found that
by running things and reading errors; so should the agent.

The hard-judge invariant
------------------------
The agent's own "I got it working" is NEVER trusted (it is exactly the thing an
LLM is most likely to be wrong or over-optimistic about). We take only the
COMMANDS it reports, re-run them ourselves in the same sandbox, and accept the
setup only if the run exits cleanly AND prints a parseable metric. Anything else
comes back ``verified=False`` so the caller can fall back or escalate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Where the agent must write its machine-readable result inside the sandbox.
SETUP_REPORT_PATH = "/workspace/autoresearch_setup.json"

#: Where we clone the repo inside the sandbox.
REPO_DIR = "/workspace/repo"


@dataclass
class DiscoveredSetup:
    """A setup the agent found, plus OUR verdict on it (``verified`` is ours, not the agent's).
    agent 摸出来的安装方式 + 我们自己的判定(verified 由我们重跑得出,不是 agent 说的)。"""

    prepare_cmd: str
    run_cmd: str
    metric: str
    lower_is_better: bool = True
    metric_value: float | None = None
    verified: bool = False
    notes: str = ""


def _setup_task_prompt(repo_url: str, goal: str, budget_s: int, repo_dir: str) -> str:
    """The instruction handed to the in-sandbox agent.

    Deliberately demands a REAL run (not a plausible-looking script) and a
    machine-readable report, and warns about the two things that actually bite:
    a training script that never terminates, and a metric that is never printed.
    """
    return f"""You are setting up a research repository so its BASELINE can be benchmarked.

Repository (already cloned for you): {repo_dir}
Origin: {repo_url}
Goal of the eventual experiment: {goal}
Wall-clock budget for ONE training/eval run: about {budget_s} seconds.

Your job is NOT to improve anything. It is to make the baseline ACTUALLY RUN,
end to end, and print a metric — then report exactly how.

Do NOT modify the repository's own source files to get there. Every tracked file
is restored to its committed state before your report is checked, so any edit you
make to the code will be reverted and a setup that depended on it will simply fail.
Configuration belongs on the COMMAND LINE (flags/env), not in edited source.

You MUST:
1. Explore the repo (README, scripts, entry points, config) to learn its real setup chain.
2. Actually run things. Install dependencies, fetch/prepare any dataset, build or
   train whatever prerequisite artifacts the code asserts on (some repos require
   training a tokenizer before training a model). Read each error and fix it.
3. Keep shrinking the training configuration (smallest model/steps/batch/sequence
   length the repo supports) until ONE run finishes well inside {budget_s} seconds.
   A script that trains forever is NOT acceptable — cap its iterations/steps.
4. Make sure the run PRINTS a metric number on stdout (e.g. a loss / bpb /
   accuracy). If evaluation only happens every N steps, lower N so it evaluates.
5. Your two commands MUST be SELF-CONTAINED: they will be executed in a FRESH
   shell that has NOTHING carried over from your session. In particular an
   `activate`d virtualenv does NOT survive -- if you ran
   `source .venv/bin/activate` and then bare `python`, that bare `python` will be
   the SYSTEM python in the fresh shell and the run will die with
   ModuleNotFoundError. Always spell the interpreter out: use `uv run python ...`
   or `.venv/bin/python ...`. Same for any other tool that needs the venv.
6. Verify this yourself: open a NEW shell (`bash -lc '<your command>'`), run both
   commands there, and confirm they still work before reporting.

Then write your result to {SETUP_REPORT_PATH} as a single JSON object:
{{
  "prepare_cmd": "<one shell command that does ALL one-time setup: deps, data, tokenizer...>",
  "run_cmd":     "<one shell command that runs ONE short baseline and prints the metric>",
  "metric":      "<the exact metric name as PRINTED by run_cmd, e.g. 'Validation bpb'>",
  "lower_is_better": true
}}

Both commands must work when run from {repo_dir}. Do not invent a metric name —
copy it exactly as the program prints it. Your report will be checked by re-running
these commands, so do not claim success you have not observed."""


def _fix_task_prompt(setup: "DiscoveredSetup", failure: str) -> str:
    """Hand the agent the EXACT failure our clean-shell re-run hit, so it can fix it.

    Without the real error text the agent has nothing to work from and just
    re-reports the same commands.
    """
    return f"""Your reported setup did NOT reproduce. I re-ran your own commands in a FRESH
shell (nothing carried over from your session) and it failed:

  prepare_cmd: {setup.prepare_cmd}
  run_cmd:     {setup.run_cmd}

  FAILURE: {failure}

The most common cause is depending on shell state that does not survive: an
`activate`d virtualenv, an exported variable, or a working directory. Bare
`python` in a fresh shell is the SYSTEM python, not your venv.

Fix it: make both commands self-contained (spell out `uv run python ...` or
`.venv/bin/python ...`), verify them yourself in a NEW shell via
`bash -lc '<command>'`, then OVERWRITE {SETUP_REPORT_PATH} with the corrected JSON."""


def _parse_report(raw: str) -> dict | None:
    """Parse the agent's JSON report; require the fields we actually need."""
    try:
        obj = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(obj, dict):
        return None
    if not (str(obj.get("run_cmd") or "").strip() and str(obj.get("metric") or "").strip()):
        return None
    return obj


def _err_tail(res, limit: int = 400) -> str:
    """The most informative tail of a failed command (stderr first, else stdout)."""
    text = (getattr(res, "stderr", "") or "").strip() or (getattr(res, "stdout", "") or "").strip()
    return text[-limit:]


def _verify_setup(backend, handle, setup: DiscoveredSetup, budget_s: int, repo_dir: str):
    """Re-run the agent's reported commands in a FRESH shell. Returns
    ``(ok, failure_note, metric_value)``.

    This is the hard judge: the agent's own claim of success counts for nothing.
    Failures carry the REAL error tail so it can be fed back for a fix round.

    ANTI-CHEAT: the agent had free rein in this sandbox and may have EDITED the
    repo's own sources -- nothing stops it from making the training script print a
    flattering number. Before re-running we restore every TRACKED file to its
    pristine committed state, so what we verify is "the repo's ORIGINAL code, plus
    your setup commands". Untracked paths are deliberately left alone: that is
    where the venv, the downloaded dataset and the trained tokenizer live, and
    wiping them would force a full reinstall on every round (and they are inputs,
    not the code under test).
    """
    from autoresearch.experiments.repo_improve.runner import _parse_metric  # reuse

    backend.execute_command(handle, "git checkout -- .", cwd=repo_dir, timeout=120)

    if setup.prepare_cmd and setup.prepare_cmd != "true":
        prep = backend.execute_command(handle, setup.prepare_cmd, cwd=repo_dir, timeout=1800)
        if getattr(prep, "exit_code", 1) != 0:
            return False, f"prepare_cmd failed on re-run: {_err_tail(prep)}", None

    run = backend.execute_command(handle, setup.run_cmd, cwd=repo_dir, timeout=budget_s + 120)
    out = (getattr(run, "stdout", "") or "") + "\n" + (getattr(run, "stderr", "") or "")
    if getattr(run, "exit_code", 1) != 0:
        return False, f"run_cmd failed on re-run: {_err_tail(run)}", None

    value = _parse_metric(setup.metric, out)
    if value is None:
        return False, f"run_cmd succeeded but printed no parseable {setup.metric!r}", None
    return True, "", value


def discover_setup_in_sandbox(
    backend,
    handle,
    *,
    repo_url: str,
    goal: str,
    budget_s: int = 300,
    llm=None,
    agent_timeout: float | None = 1800.0,
    repo_dir: str = REPO_DIR,
    max_fix_rounds: int = 1,
) -> DiscoveredSetup | None:
    """Clone ``repo_url`` in the sandbox, let the agent iterate until the baseline runs,
    then re-run its reported commands OURSELVES and return the verdict.

    When our re-run fails we feed the REAL error back and give the agent a bounded
    number of fix rounds (``max_fix_rounds``) -- the first live run showed why this
    matters: the agent had ``source .venv/bin/activate``d in its own shell and then
    reported bare ``python ...`` commands, which die with ModuleNotFoundError in a
    fresh shell. Telling it exactly that lets it correct the commands.

    Returns ``None`` when the agent produced no usable report at all. Otherwise a
    ``DiscoveredSetup`` whose ``verified`` says whether OUR re-run reproduced a
    metric -- callers must check it rather than assuming success.
    """
    backend.execute_command(
        handle, f"rm -rf {repo_dir} && git clone --depth 1 {repo_url} {repo_dir}", timeout=600
    )

    setup: DiscoveredSetup | None = None
    for attempt in range(max_fix_rounds + 1):
        task = (
            _setup_task_prompt(repo_url, goal, budget_s, repo_dir)
            if setup is None
            else _fix_task_prompt(setup, setup.notes)
        )
        try:
            res = backend.run_agent_task(handle, task, llm=llm, timeout=agent_timeout)
            logger.info("sandbox setup agent round %d: status=%s", attempt,
                        getattr(res, "status", "?"))
        except Exception:  # noqa: BLE001 -- a crash still lets us read a partial report
            logger.warning("sandbox setup agent raised; reading its report anyway", exc_info=True)

        got = backend.execute_command(handle, f"cat {SETUP_REPORT_PATH}", timeout=60)
        report = (
            _parse_report(getattr(got, "stdout", "") or "")
            if getattr(got, "exit_code", 1) == 0
            else None
        )
        if report is None:
            # No usable report at all on the first pass -> nothing to work with.
            # On a later pass keep the previous (unverified) setup as the answer.
            if setup is None:
                return None
            break

        setup = DiscoveredSetup(
            prepare_cmd=str(report.get("prepare_cmd") or "true").strip(),
            run_cmd=str(report["run_cmd"]).strip(),
            metric=str(report["metric"]).strip(),
            lower_is_better=bool(report.get("lower_is_better", True)),
        )

        ok, note, value = _verify_setup(backend, handle, setup, budget_s, repo_dir)
        if ok:
            setup.metric_value = value
            setup.verified = True
            return setup
        setup.notes = note
        logger.info("setup verification failed (round %d): %s", attempt, note)

    return setup


__all__ = ["DiscoveredSetup", "SETUP_REPORT_PATH", "REPO_DIR", "discover_setup_in_sandbox"]
