"""meeting ③: offline test of the Paper2Code adapter -- switch-gating + our-gateway env + pipeline
construction, with an injected runner (no real Paper2Code / LLM needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.experiments.paper_repro import paper2code_adapter as adapter
from autoresearch.experiments.paper_repro.paper2code_adapter import PaperReproError


class _Res:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


class _Runner:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def __call__(self, cmd, *, env, cwd, capture_output, text):
        self.calls.append({"cmd": cmd, "env": env, "cwd": cwd})
        return _Res(self.returncode)


def _fake_p2c(tmp_path) -> Path:
    d = tmp_path / "Paper2Code"
    (d / "codes").mkdir(parents=True)
    (d / "codes" / "1_planning.py").write_text("# stub", encoding="utf-8")
    return d


def _paper(tmp_path) -> Path:
    p = tmp_path / "paper.json"
    p.write_text("{}", encoding="utf-8")
    return p


def test_backend_off_by_default_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_PAPER_REPRO_BACKEND", raising=False)
    assert adapter.paper_repro_backend() is None
    with pytest.raises(PaperReproError, match="OFF"):
        adapter.build_repo_from_paper(_paper(tmp_path), tmp_path, paper_name="X")


def test_missing_p2c_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_PAPER_REPRO_BACKEND", "paper2code")
    monkeypatch.delenv("AUTORESEARCH_PAPER2CODE_DIR", raising=False)
    with pytest.raises(PaperReproError, match="not located"):
        adapter.build_repo_from_paper(_paper(tmp_path), tmp_path, paper_name="X")


def test_builds_pipeline_pointed_at_our_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_PAPER_REPRO_BACKEND", "paper2code")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("KIMI_MODEL", "deepseek-v4-flash-260425")
    runner = _Runner()
    repo = adapter.build_repo_from_paper(_paper(tmp_path), tmp_path, paper_name="Transformer",
                                         p2c_dir=_fake_p2c(tmp_path), runner=runner)
    assert repo.endswith("Transformer_repo")
    # the 5 pipeline stages ran, in order
    stage_scripts = [Path(c["cmd"][1]).name for c in runner.calls]
    assert stage_scripts == ["0_pdf_process.py", "1_planning.py", "1.1_extract_config.py",
                             "2_analyzing.py", "3_coding.py"]
    # every stage points the OpenAI client at OUR gateway
    for c in runner.calls:
        assert c["env"]["OPENAI_BASE_URL"] == "https://gw.example/v1"
        assert c["env"]["OPENAI_API_KEY"] == "sk-test"
    # the LLM stages (planning/analyzing/coding) pass our model via Paper2Code's real --gpt_version flag
    for c in runner.calls:
        if Path(c["cmd"][1]).name in ("1_planning.py", "2_analyzing.py", "3_coding.py"):
            assert "--gpt_version" in c["cmd"]
            assert c["cmd"][c["cmd"].index("--gpt_version") + 1] == "deepseek-v4-flash-260425"


def test_gateway_key_falls_back_to_openrouter(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_PAPER_REPRO_BACKEND", "paper2code")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-or")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_BASE_URL", "https://gw.example/v1")
    runner = _Runner()
    adapter.build_repo_from_paper(_paper(tmp_path), tmp_path, paper_name="X",
                                  p2c_dir=_fake_p2c(tmp_path), runner=runner)
    assert runner.calls[0]["env"]["OPENAI_API_KEY"] == "sk-or"


def test_stage_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_PAPER_REPRO_BACKEND", "paper2code")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_BASE_URL", "https://gw.example/v1")
    runner = _Runner(returncode=1)
    with pytest.raises(PaperReproError, match="stage failed"):
        adapter.build_repo_from_paper(_paper(tmp_path), tmp_path, paper_name="X",
                                      p2c_dir=_fake_p2c(tmp_path), runner=runner)
