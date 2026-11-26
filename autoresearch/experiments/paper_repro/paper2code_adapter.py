"""Adapter that reuses a checked-out Paper2Code (going-doer/Paper2Code) to synthesize a code repo
from a paper, with its OpenAI client pointed at OUR HKGAI gateway (not OpenAI, not a local vLLM).

Switch-gated (``AUTORESEARCH_PAPER_REPRO_BACKEND=paper2code``, default OFF) and NOT wired into the
engine routing -- callable/offline-testable only, per the meeting ③ decision.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


class PaperReproError(RuntimeError):
    """Paper-repro backend disabled, misconfigured, or a pipeline stage failed."""


def paper_repro_backend() -> str | None:
    """Configured backend name (lower-cased), or None when the line is OFF (the default)."""
    v = os.environ.get("AUTORESEARCH_PAPER_REPRO_BACKEND", "").strip().lower()
    return v or None


def _resolve_p2c_dir(p2c_dir: str | Path | None) -> Path:
    d = p2c_dir or os.environ.get("AUTORESEARCH_PAPER2CODE_DIR")
    if not d:
        raise PaperReproError(
            "Paper2Code not located: set AUTORESEARCH_PAPER2CODE_DIR to a checked-out "
            "going-doer/Paper2Code (or pass p2c_dir).")
    p = Path(d)
    if not (p / "codes" / "1_planning.py").exists():
        raise PaperReproError(f"AUTORESEARCH_PAPER2CODE_DIR={p} is not a Paper2Code checkout "
                              "(missing codes/1_planning.py).")
    return p


def _gateway_env(model: str | None, base_url: str | None, api_key: str | None) -> dict:
    """Point Paper2Code's OpenAI client at OUR gateway: OpenAI() reads OPENAI_API_KEY and honours
    OPENAI_BASE_URL, so a flash/glm model on the HKGAI endpoint is a drop-in for o3-mini."""
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = (api_key or os.environ.get("OPENAI_API_KEY")
                             or os.environ.get("OPENROUTER_KEY") or "")
    env["OPENAI_BASE_URL"] = (base_url or os.environ.get("OPENAI_BASE_URL")
                              or os.environ.get("AUTORESEARCH_SANDBOX_BASE_URL")
                              or os.environ.get("OPENROUTER_BASE_URL") or "")
    if not env["OPENAI_API_KEY"] or not env["OPENAI_BASE_URL"]:
        raise PaperReproError("gateway not configured: need OPENAI_API_KEY + OPENAI_BASE_URL "
                              "(or OPENROUTER_KEY / AUTORESEARCH_SANDBOX_BASE_URL).")
    return env


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("KIMI_MODEL") or "deepseek-v4-flash-260425"


def build_repo_from_paper(paper_json_path: str | Path, workdir: str | Path, *, paper_name: str,
                          model: str | None = None, base_url: str | None = None,
                          api_key: str | None = None, p2c_dir: str | Path | None = None,
                          runner=subprocess.run) -> str:
    """Synthesize a runnable repo from ``paper_json_path`` via Paper2Code's OpenAI pipeline pointed at
    our gateway. Returns the output repo dir. Raises PaperReproError if OFF / misconfigured / a stage
    fails. ``runner`` is injectable (subprocess.run) so the wiring is offline-testable.

    Pipeline (mirrors scripts/run.sh): 0_pdf_process -> 1_planning -> 1.1_extract_config -> (copy
    planning_config.yaml -> repo/config.yaml) -> 2_analyzing -> 3_coding. Each stage runs with our
    gateway env; the model is passed to Paper2Code as ``--gpt_version`` (its real CLI flag)."""
    if paper_repro_backend() != "paper2code":
        raise PaperReproError("paper-repro line is OFF (set AUTORESEARCH_PAPER_REPRO_BACKEND=paper2code).")
    p2c = _resolve_p2c_dir(p2c_dir)
    env = _gateway_env(model, base_url, api_key)
    gpt = _resolve_model(model)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / f"{paper_name}_plan"
    repo_dir = work / f"{paper_name}_repo"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    cleaned = work / f"{paper_name}_cleaned.json"
    codes = p2c / "codes"
    py = os.environ.get("AUTORESEARCH_PAPER2CODE_PY", "python")

    stages = [
        [py, str(codes / "0_pdf_process.py"),
         "--input_json_path", str(paper_json_path), "--output_json_path", str(cleaned)],
        [py, str(codes / "1_planning.py"), "--paper_name", paper_name, "--gpt_version", gpt,
         "--pdf_json_path", str(cleaned), "--output_dir", str(out_dir)],
        [py, str(codes / "1.1_extract_config.py"),
         "--paper_name", paper_name, "--output_dir", str(out_dir)],
        [py, str(codes / "2_analyzing.py"), "--paper_name", paper_name, "--gpt_version", gpt,
         "--pdf_json_path", str(cleaned), "--output_dir", str(out_dir)],
        [py, str(codes / "3_coding.py"), "--paper_name", paper_name, "--gpt_version", gpt,
         "--pdf_json_path", str(cleaned), "--output_dir", str(out_dir),
         "--output_repo_dir", str(repo_dir)],
    ]
    for i, stage in enumerate(stages):
        res = runner(stage, env=env, cwd=str(codes), capture_output=True, text=True)
        if getattr(res, "returncode", 1) != 0:
            tail = (getattr(res, "stderr", "") or "")[-500:]
            raise PaperReproError(f"Paper2Code stage failed: {Path(stage[1]).name}: {tail}")
        # after 1.1_extract_config, run.sh copies the planning config into the repo (3_coding reads it)
        if Path(stage[1]).name == "1.1_extract_config.py":
            cfg = out_dir / "planning_config.yaml"
            if cfg.exists():
                (repo_dir / "config.yaml").write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
    return str(repo_dir)


__all__ = ["build_repo_from_paper", "paper_repro_backend", "PaperReproError"]
