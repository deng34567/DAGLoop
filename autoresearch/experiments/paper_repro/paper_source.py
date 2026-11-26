"""Bridge the paper-reproduction line into the agentic improve pipeline.

``run_agentic_improve`` takes a ``clone_fn(repo_url, repo_path)`` that materializes the
repo contents on disk; everything after it is repo-agnostic (the agent explores, builds,
runs, fixes and measures whatever is at ``repo_path``). For a paper-reproduction task we
replace "git clone a repo" with "synthesize a repo FROM the paper via Paper2Code":

    acquire_paper_json(question)  ->  build_repo_from_paper(...)  ->  copy into repo_path

The agentic loop then runs the generated code, and -- crucially -- its natural
run->error->fix cycle IS the "debug round" the generated code needs (generated code
routinely has trivial bugs like a missing import). So no separate debug stage is required.
"""
from __future__ import annotations

import os
import shutil

from autoresearch.experiments.paper_repro.paper2code_adapter import build_repo_from_paper
from autoresearch.experiments.paper_repro.paper_acquire import acquire_paper_json


def make_paper_clone_fn(question: str, llm, *, model=None, base_url=None, api_key=None,
                        p2c_dir=None, build_fn=build_repo_from_paper,
                        acquire_fn=acquire_paper_json, emit=None):
    """Return a ``clone_fn(repo_url, repo_path)`` that builds a repo from the paper named in
    ``question`` and materializes its contents AT ``repo_path`` (where the agentic loop expects
    the repo). ``build_fn``/``acquire_fn`` are injectable seams for offline tests.

    ``emit`` (optional) mirrors run_agentic_improve's event sink. Exceptions propagate: a failed
    acquire/build is an honest hard error (there is no repo to improve)."""
    emit = emit or (lambda event, payload: None)

    def _clone(repo_url, repo_path):  # repo_url is ignored -- the paper IS the source
        workdir = os.path.dirname(repo_path.rstrip("/")) or "."
        paper_json, meta = acquire_fn(question, llm, workdir=workdir)
        emit("paper_acquire", {"paper_json": paper_json, "title": meta.get("title"),
                               "metric": meta.get("metric"),
                               "baseline_value": meta.get("baseline_value")})
        gen_repo = build_fn(paper_json, workdir, paper_name=meta["paper_name"],
                            model=model, base_url=base_url, api_key=api_key, p2c_dir=p2c_dir)
        emit("paper_build", {"generated_repo": gen_repo})
        # Materialize the generated repo's contents AT repo_path (build_fn wrote them to its own
        # {paper_name}_repo dir; the agentic loop expects them at the caller-chosen repo_path).
        os.makedirs(repo_path, exist_ok=True)
        for entry in os.listdir(gen_repo):
            src = os.path.join(gen_repo, entry)
            dst = os.path.join(repo_path, entry)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        # Drop the paper text in too, so the agent can consult it while fixing/reproducing.
        try:
            shutil.copy2(paper_json, os.path.join(repo_path, "PAPER_source.json"))
        except OSError:
            pass

    return _clone


__all__ = ["make_paper_clone_fn"]
