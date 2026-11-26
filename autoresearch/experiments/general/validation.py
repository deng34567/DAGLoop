"""Lightweight public-data validation for theory-oriented research.

This is an execution seam, not a new agent or state-machine layer.  The coding
agent writes one study script in an isolated workspace; the host then runs that
script and persists the inputs, script, raw data, and structured result together.
"""
from __future__ import annotations

import base64
import json
import math
import os
import posixpath
import tempfile
from pathlib import Path
from typing import Any


_RESULT_PATH = "/workspace/validation_result.json"
_SCRIPT_PATH = "/workspace/validation_study.py"
_CONCLUSIONS = {"supports", "contradicts", "inconclusive"}


def _task(study: dict) -> str:
    return f"""Implement the following frozen lightweight validation study.

ValidationStudy (JSON):
{json.dumps(study, ensure_ascii=False, indent=2)}

This is a survey/statistical validation, not a benchmark and not a
baseline-vs-method contest.  Do not change the question, sampling frame,
variables, falsification condition, or finite budget.

Write /workspace/validation_study.py and run it once while developing.  The
script must be self-contained and, when the host runs `python
validation_study.py`, create /workspace/validation_result.json with exactly:
{{
  "conclusion": "supports|contradicts|inconclusive",
  "metric_name": "primary descriptive or test statistic",
  "metric_value": <finite number>,
  "sample_ids": ["stable public identifiers"],
  "retrieved_at": "ISO-8601 retrieval date/time",
  "data_sources": ["sources actually used"],
  "analysis_summary": "what was observed, including null results",
  "artifacts": ["relative paths such as raw_data.json and figure.png"]
}}

Save the raw sampled records (including sample identifiers) in at least one
declared artifact.  Save any analysis table or figure you produce and declare
it too.  Keep downloads small and within the frozen budget.  Every network
call must be direct: unset inherited HTTP_PROXY/HTTPS_PROXY/ALL_PROXY and make
the client ignore proxy environment settings.  A null result is acceptable;
never fabricate records or significance.  Finish only after the script and
result files exist.
"""


def _relative_artifact(path: Any) -> str | None:
    if not isinstance(path, str) or not path or path.startswith("/"):
        return None
    normalized = posixpath.normpath(path)
    if normalized in {".", ".."} or normalized.startswith("../"):
        return None
    return normalized


def _validated_result(value: Any) -> dict | None:
    if not isinstance(value, dict) or set(value) != {
        "conclusion", "metric_name", "metric_value", "sample_ids",
        "retrieved_at", "data_sources", "analysis_summary", "artifacts",
    }:
        return None
    metric = value.get("metric_value")
    artifacts = value.get("artifacts")
    if (value.get("conclusion") not in _CONCLUSIONS
            or not isinstance(value.get("metric_name"), str)
            or not value["metric_name"].strip()
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or not isinstance(value.get("sample_ids"), list)
            or not value["sample_ids"]
            or any(not isinstance(item, str) or not item.strip()
                   for item in value["sample_ids"])
            or not isinstance(value.get("retrieved_at"), str)
            or not value["retrieved_at"].strip()
            or not isinstance(value.get("data_sources"), list)
            or not value["data_sources"]
            or any(not isinstance(item, str) or not item.strip()
                   for item in value["data_sources"])
            or not isinstance(value.get("analysis_summary"), str)
            or not value["analysis_summary"].strip()
            or not isinstance(artifacts, list)
            or not artifacts):
        return None
    normalized = [_relative_artifact(item) for item in artifacts]
    if any(item is None for item in normalized) or len(set(normalized)) != len(normalized):
        return None
    result = dict(value)
    result["metric_value"] = float(metric)
    result["artifacts"] = normalized
    return result


def _download(workspace: Any, source: str, destination: Path) -> bytes:
    workspace.file_download(source, str(destination))
    data = destination.read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError(f"validation artifact exceeds 2 MiB: {source}")
    return data


def _bundle_file(data: bytes) -> dict:
    try:
        return {"encoding": "utf-8", "content": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"encoding": "base64", "content": base64.b64encode(data).decode("ascii")}


def _failure_record(message: str, *, exit_code: int = 1, stdout: str = "") -> dict:
    return {
        "succeeded": False,
        "exit_code": int(exit_code),
        "metric_value": None,
        "stdout_excerpt": stdout[-4000:],
        "stderr_excerpt": message[-4000:],
        "environment": "survey_validation|host_rerun",
        "seed": None,
        "runtime_seconds": None,
        "artifacts": [],
    }


def run_validation_study(backend: Any, llm: Any, study: dict) -> tuple[dict, dict | None, dict | None]:
    """Write and host-rerun one survey validation in an isolated sandbox."""
    handle = backend.open_sandbox()
    try:
        agent_timeout = float(os.environ.get("AUTORESEARCH_VALIDATION_AGENT_TIMEOUT_S", "1200"))
        run_timeout = float(os.environ.get("AUTORESEARCH_VALIDATION_RUN_TIMEOUT_S", "1200"))
        # The web service's shared HKGAIClient is not an OpenHands SDK LLM.  Let the sandbox
        # backend construct its configured SDK client instead of passing an incompatible object
        # (which fails while building the condenser with ``model_copy``).
        backend.run_agent_task(handle, _task(study), timeout=agent_timeout)
        backend.execute_command(
            handle, "rm -f validation_result.json", cwd="/workspace", timeout=30)
        executed = backend.execute_command(
            handle, "python validation_study.py", cwd="/workspace", timeout=run_timeout)
        if executed.exit_code != 0 or executed.timed_out:
            return _failure_record(
                executed.stderr or "validation script did not complete",
                exit_code=executed.exit_code or 1,
                stdout=executed.stdout,
            ), None, None

        with tempfile.TemporaryDirectory(prefix="ar-validation-") as tmp:
            root = Path(tmp)
            result_bytes = _download(
                handle.workspace, _RESULT_PATH, root / "validation_result.json")
            result = _validated_result(json.loads(result_bytes.decode("utf-8")))
            if result is None:
                return _failure_record(
                    "validation_result.json failed the survey-validation contract",
                    stdout=executed.stdout,
                ), None, None
            files = {
                "validation_study.py": _bundle_file(_download(
                    handle.workspace, _SCRIPT_PATH, root / "validation_study.py")),
                "validation_result.json": _bundle_file(result_bytes),
            }
            for artifact in result["artifacts"]:
                data = _download(
                    handle.workspace, "/workspace/" + artifact, root / Path(artifact).name)
                files[artifact] = _bundle_file(data)

        bundle = {"study": study, "result": result, "files": files}
        record = {
            "succeeded": True,
            "exit_code": 0,
            "metric_value": result["metric_value"],
            "stdout_excerpt": result["analysis_summary"],
            "stderr_excerpt": None,
            "environment": "survey_validation|host_rerun",
            "seed": None,
            "runtime_seconds": None,
            "artifacts": [],
        }
        return record, result, bundle
    except Exception as exc:  # execution evidence becomes an honest failed RunRecord
        return _failure_record(f"{type(exc).__name__}: {exc}"), None, None
    finally:
        backend.close_sandbox(handle)


__all__ = ["run_validation_study"]
