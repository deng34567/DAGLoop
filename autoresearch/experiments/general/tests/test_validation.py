from __future__ import annotations

import json
from dataclasses import dataclass

from autoresearch.experiments.general.validation import run_validation_study


@dataclass
class _Result:
    exit_code: int = 0
    stdout: str = "analysis complete"
    stderr: str = ""
    timed_out: bool = False


class _Workspace:
    def __init__(self, files):
        self.files = files

    def file_download(self, source, destination):
        with open(destination, "wb") as handle:
            handle.write(self.files[source])


class _Backend:
    def __init__(self, result):
        self.handle = type("Handle", (), {"workspace": _Workspace({})})()
        self.result = result
        self.commands = []
        self.task = ""
        self.llm = "unset"
        self.closed = False

    def open_sandbox(self):
        return self.handle

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.task = task
        self.llm = llm
        files = {
            "/workspace/validation_study.py": b"print('analysis complete')\n",
            "/workspace/raw_data.json": b'[{"id":"W1","value":1}]',
            "/workspace/validation_result.json": json.dumps(self.result).encode(),
        }
        handle.workspace.files.update(files)
        return type("AgentResult", (), {"status": "FINISHED"})()

    def execute_command(self, handle, command, *, cwd="/workspace", timeout=None):
        self.commands.append((command, cwd, timeout))
        return _Result()

    def close_sandbox(self, handle):
        self.closed = True


STUDY = {
    "kind": "survey_validation",
    "question": "Does the observable trend support the claim?",
    "sampling_frame": "A dated public-metadata sample.",
    "data_sources": ["OpenAlex"],
    "variables": ["year", "citation_count"],
    "analysis": "Descriptive trend and permutation test.",
    "falsification_condition": "No trend or an opposite trend.",
    "budget": "100 records and 10 minutes.",
}


def _valid_result():
    return {
        "conclusion": "inconclusive",
        "metric_name": "spearman_r",
        "metric_value": 0.03,
        "sample_ids": ["W1"],
        "retrieved_at": "2026-08-20T10:00:00Z",
        "data_sources": ["OpenAlex"],
        "analysis_summary": "No stable association was observed.",
        "artifacts": ["raw_data.json"],
    }


def test_validation_study_is_host_run_and_bundled():
    backend = _Backend(_valid_result())

    record, result, bundle = run_validation_study(backend, object(), STUDY)

    assert backend.closed is True
    assert backend.llm is None
    assert "python validation_study.py" in backend.commands[-1][0]
    assert "sample identifiers" in backend.task.lower()
    assert record["succeeded"] is True
    assert record["metric_value"] == 0.03
    assert result["conclusion"] == "inconclusive"
    assert bundle["study"] == STUDY
    assert "validation_study.py" in bundle["files"]
    assert "raw_data.json" in bundle["files"]


def test_validation_study_without_sample_ids_fails_closed():
    bad = _valid_result()
    bad["sample_ids"] = []
    backend = _Backend(bad)

    record, result, bundle = run_validation_study(backend, object(), STUDY)

    assert record["succeeded"] is False
    assert record["metric_value"] is None
    assert result is None
    assert bundle is None
