"""Tests for the downloadable, Docker-runnable experiment bundle."""
from __future__ import annotations

from autoresearch.experiments.general.bundle import build_bundle

_DESIGN = {
    "task": "1D bin packing",
    "baseline": "First-Fit",
    "metric": "number of bins",
    "dependencies": ["numpy"],
}
_PRIMARY = {"baseline_metric": 37.97, "method_metric": 35.89, "isolated": True}
_REPRODUCE = {"method_metric": 35.9, "isolated": True}
_VERDICT = {"verdict": "pass", "reasons": ["method beats baseline"]}


def _bundle():
    return build_bundle(
        "beat First-Fit?", _DESIGN, "def make_instances(seed): ...\n",
        "def solve(instance): ...\n", primary=_PRIMARY, reproduce=_REPRODUCE,
        verdict=_VERDICT, report={"summary": "ok"},
    )


def test_bundle_has_agent_code_and_trusted_driver():
    files = _bundle()["files"]
    assert "def solve" in files["method.py"]
    assert "def make_instances" in files["harness.py"]
    # trusted, non-AI driver copied verbatim from the package
    assert "_orchestrator.py" in files
    assert "isolated_method.py" in files


def test_bundle_is_docker_runnable():
    files = _bundle()["files"]
    assert "FROM python:3.11-slim" in files["Dockerfile"]
    assert "_orchestrator.py" in files["Dockerfile"]
    assert "docker build" in files["run.sh"]
    assert files["requirements.txt"].strip() == "numpy"


def test_bundle_ships_windows_launcher():
    """The bundle is usable on Windows too: a run.bat (double-click) mirrors run.sh."""
    files = _bundle()["files"]
    bat = files["run.bat"]
    assert "@echo off" in bat
    assert "docker build" in bat and "docker run" in bat
    assert "\r\n" in bat                       # Windows CRLF line endings
    assert "pause" in bat                      # keeps the console window open
    assert "run.bat" in files["README.md"] and "Windows" in files["README.md"]


def test_bundle_readme_carries_the_honest_result():
    files = _bundle()["files"]
    rd = files["README.md"]
    assert "37.97" in rd and "35.89" in rd
    assert "pass" in rd
    assert "isolated" in rd.lower()


def test_bundle_meta_summary():
    b = _bundle()
    assert b["verdict"] == "pass"
    assert b["baseline_metric"] == 37.97 and b["method_metric"] == 35.89
    assert b["isolated"] is True


def test_bundle_survives_missing_deps_and_report():
    b = build_bundle("q", {"task": "t"}, "h\n", "m\n")
    files = b["files"]
    assert "report.json" not in files  # omitted when report is None
    assert "stdlib" in files["requirements.txt"]  # no deps -> stdlib note


def test_bundle_string_report_stored_as_txt():
    # build_report returns TEXT; it must be stored as report.txt (readable), not a
    # double-JSON-encoded report.json.
    b = build_bundle("q", _DESIGN, "h\n", "m\n", report="Honest report: method beat baseline.")
    files = b["files"]
    assert "report.json" not in files
    assert files["report.txt"] == "Honest report: method beat baseline."


def test_bundle_gpu_experiment_uses_cuda_image_and_gpus_flag():
    """A GPU experiment ships a CUDA/torch Docker base + `--gpus all` in run.sh/run.bat."""
    from autoresearch.experiments.general.bundle import build_bundle
    b = build_bundle("q", {"gpu": True, "dependencies": ["torch"], "io_contract": {}},
                     "# harness", "def solve(i): return []")
    files = b["files"]
    assert "cuda" in files["Dockerfile"].lower() and "pytorch" in files["Dockerfile"].lower()
    assert "--gpus all" in files["run.sh"]
    assert "--gpus all" in files["run.bat"]
    # a CPU experiment does NOT
    b2 = build_bundle("q", {"gpu": False}, "# h", "def solve(i): return []")
    assert "python:3.11-slim" in b2["files"]["Dockerfile"]
    assert "--gpus" not in b2["files"]["run.sh"]
