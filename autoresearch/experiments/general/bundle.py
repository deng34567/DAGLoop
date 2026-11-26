"""Package the P1 experiment as a self-contained, DOWNLOADABLE, Docker-runnable
deliverable -- so the user can SEE and RUN the actual code the agents wrote
(mirrors vibe's exported code repo, but for a from-scratch honest experiment).

``build_bundle(question, design, harness_code, method_code, primary, reproduce,
verdict, report) -> dict`` returns ``{"files": {name: content}, "meta": {...}}``:
a directory that, unzipped, runs the SAME trusted scored evaluation via
``docker build . && docker run`` (or ``python _orchestrator.py 0`` locally):

  method.py        the AI-written method (solve(instance) -> solution)
  harness.py       the AI-written evaluation harness / "ruler" (a pure library)
  _orchestrator.py trusted driver (imports harness -> DELETES it -> runs the
  isolated_method.py / _method_driver.py / _selftest.py   method in an isolated
                   subprocess -> validate/score) -- the anti-cheat core, NOT
                   AI-written and byte-identical to what scored the real run
  requirements.txt the AI-declared pip deps
  Dockerfile       python:3.11-slim + deps + `python _orchestrator.py 0`
  run.sh           docker build + run one-liner
  design.json      the structured experiment design
  report.json      the honest final report
  README.md        question, verdict, metrics, how to run, the honesty note

The trusted modules are copied VERBATIM from this package so the downloaded run
reproduces the real scoring exactly (same deletion + isolation).
"""
from __future__ import annotations

import json
from pathlib import Path

_GEN_DIR = Path(__file__).resolve().parent

# The trusted, non-AI-written modules that drive the scored run (same set the
# runner uploads). Copied verbatim into the bundle.
_TRUSTED_MODULES = ("_orchestrator.py", "_selftest.py", "isolated_method.py", "_method_driver.py", "contract.py")


def _requirements(design: dict) -> str:
    from autoresearch.experiments.general.design import sanitize_dependencies

    deps = sanitize_dependencies(design.get("dependencies"))
    return ("\n".join(deps) + "\n") if deps else "# (pure stdlib -- no third-party deps)\n"


def _dockerfile(design: dict) -> str:
    # GPU (training) experiments ship a CUDA+torch base so the deliverable reproduces on
    # a GPU (run with --gpus all). The harness/method use `cuda if available else cpu`,
    # so it still runs on CPU if no GPU is present, just slower.
    base = ("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
            if design.get("gpu") else "python:3.11-slim")
    return (
        f"FROM {base}\n"
        "WORKDIR /workspace\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt || true\n"
        "COPY . .\n"
        "# The trusted orchestrator imports harness.py, DELETES it from disk, then\n"
        "# runs method.py in an isolated subprocess and scores it -- the method can\n"
        "# never reach the ruler (anti-cheat by construction). Prints result.json.\n"
        'CMD ["sh", "-c", "python _orchestrator.py 0 && echo \'=== result.json ===\' && cat result.json"]\n'
    )


def _run_sh(design: dict) -> str:
    gpus = " --gpus all" if design.get("gpu") else ""
    note = ("# GPU experiment: needs an NVIDIA GPU + the nvidia-container-toolkit.\n"
            if design.get("gpu") else "")
    return (
        "#!/usr/bin/env bash\n"
        "# Mac/Linux: build + run the honest scored evaluation in Docker (prints result.json).\n"
        + note +
        "set -e\n"
        'cd "$(dirname "$0")"\n'
        "docker build -t autoresearch-experiment .\n"
        f"docker run --rm{gpus} autoresearch-experiment\n"
    )


def _run_bat(design: dict) -> str:
    """Windows one-click launcher (``run.bat``): same build+run as run.sh, with a
    Docker-running check + UTF-8 console, so the bundle is usable on Windows too."""
    gpus = " --gpus all" if design.get("gpu") else ""
    return (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "REM Windows: build + run the honest scored evaluation in Docker (prints result.json).\r\n"
        'cd /d "%~dp0"\r\n'
        "\r\n"
        "docker info >nul 2>&1\r\n"
        "if errorlevel 1 (\r\n"
        "    echo [ERROR] Docker is not running. Start Docker Desktop, wait until it is ready, then double-click run.bat again.\r\n"
        "    pause\r\n"
        "    exit /b 1\r\n"
        ")\r\n"
        "\r\n"
        "echo Building the image... first run may take a few minutes; later runs are fast.\r\n"
        "docker build -t autoresearch-experiment .\r\n"
        "if errorlevel 1 (\r\n"
        "    echo [ERROR] docker build failed.\r\n"
        "    pause\r\n"
        "    exit /b 1\r\n"
        ")\r\n"
        "\r\n"
        f"docker run --rm{gpus} autoresearch-experiment\r\n"
        "echo.\r\n"
        "echo Done. The scored result.json is printed above.\r\n"
        "pause\r\n"
    )


def _readme(question, design, primary, reproduce, verdict) -> str:
    v = (verdict or {}).get("verdict")
    reasons = "; ".join((verdict or {}).get("reasons") or [])
    base = (primary or {}).get("baseline_metric")
    meth = (primary or {}).get("method_metric")
    rep = (reproduce or {}).get("method_metric")
    isolated = (primary or {}).get("isolated") and (reproduce or {}).get("isolated")
    return f"""# AutoResearch — experiment deliverable

**Research question:** {question}

**Task:** {design.get('task')}
**Baseline:** {design.get('baseline')}
**Metric:** {design.get('metric')}

## Result (honest, machine-computed)
- verdict: **{v}**
- baseline metric: **{base}**
- method metric (seed 0): **{meth}**   ·   reproduce (seed 1): **{rep}**
- run-time isolated: **{isolated}**
- reasons: {reasons}

## What this is
The `method.py` and `harness.py` here were WRITTEN BY THE AGENT for this
question, in separate isolated Docker sandboxes. `harness.py` is the evaluation
"ruler" (generates instances, runs the baseline, validates + scores); `method.py`
implements `solve(instance) -> solution`. The two were written independently and
never saw each other.

## How to run it yourself
- **Windows:** double-click **`run.bat`** (needs Docker Desktop running).
- **Mac/Linux:** `bash run.sh`
- **Any OS, locally** (after `pip install -r requirements.txt`): `python _orchestrator.py 0`

Each builds the Docker image and prints the honest scored `result.json`.

## Anti-cheat (why the number is trustworthy)
`_orchestrator.py` is a FIXED, trusted driver (NOT written by the agent). It
imports `harness.py`, **immediately deletes it from disk**, then runs `method.py`
in an ISOLATED subprocess (fresh cwd, `PYTHONPATH=""`, piped I/O) and only calls
the harness to validate/score the returned solutions. So the method can never
read or tamper with the ruler. `isolated=true` means the trusted orchestrator
drove the run to completion.

## Honest reading of a `pass`
A `pass` means: the AI method beat the AI baseline on the AI-defined metric,
reproducibly and run-time-isolated. It does NOT certify that the metric is the
right one to care about, or that the ruler is fair — only that, given this ruler,
the method genuinely won without cheating.
"""


def build_bundle(question, design, harness_code, method_code,
                 primary=None, reproduce=None, verdict=None, report=None) -> dict:
    """Assemble the downloadable, Docker-runnable experiment bundle."""
    files: dict[str, str] = {}
    files["method.py"] = method_code or "# (method source unavailable)\n"
    files["harness.py"] = harness_code or "# (harness source unavailable)\n"
    for name in _TRUSTED_MODULES:
        try:
            files[name] = (_GEN_DIR / name).read_text()
        except OSError:
            pass
    # the leak-safe shape example so the orchestrator's check_shape runs in the
    # downloaded bundle exactly as it did during scoring.
    files["io_contract.json"] = json.dumps((design or {}).get("io_contract") or {}, indent=2, ensure_ascii=False)
    files["requirements.txt"] = _requirements(design or {})
    files["Dockerfile"] = _dockerfile(design or {})
    files["run.sh"] = _run_sh(design or {})      # Mac/Linux
    files["run.bat"] = _run_bat(design or {})    # Windows (double-click)
    files["design.json"] = json.dumps(design or {}, indent=2, ensure_ascii=False)
    if isinstance(report, str):
        files["report.txt"] = report  # build_report returns the honest report TEXT
    elif report is not None:
        files["report.json"] = json.dumps(report, indent=2, ensure_ascii=False)
    files["README.md"] = _readme(question, design or {}, primary, reproduce, verdict)

    return {
        "question": question,
        "verdict": (verdict or {}).get("verdict"),
        "baseline_metric": (primary or {}).get("baseline_metric"),
        "method_metric": (primary or {}).get("method_metric"),
        "isolated": bool((primary or {}).get("isolated") and (reproduce or {}).get("isolated")),
        "files": files,
    }


__all__ = ["build_bundle"]
