"""Offline test for the paper->agentic bridge: make_paper_clone_fn must acquire the paper,
build a repo from it, and materialize the generated repo's contents AT the caller's repo_path
(the seam where run_agentic_improve expects the repo)."""
from __future__ import annotations

import os

from autoresearch.experiments.paper_repro.paper_source import make_paper_clone_fn


def test_paper_clone_fn_materializes_generated_repo(tmp_path):
    # fake acquire: writes a paper json, returns (path, meta)
    def _acquire(question, llm, *, workdir, paper_name=None):
        p = os.path.join(workdir, "P_cleaned.json")
        with open(p, "w") as f:
            f.write("{}")
        return p, {"paper_name": "P", "metric": "acc", "lower_is_better": False,
                   "baseline_value": 99.0, "title": "P"}

    # fake build: writes a generated repo dir with two files, returns its path
    def _build(paper_json, workdir, *, paper_name, model=None, base_url=None, api_key=None,
               p2c_dir=None):
        gen = os.path.join(workdir, f"{paper_name}_repo")
        os.makedirs(gen, exist_ok=True)
        open(os.path.join(gen, "main.py"), "w").write("print('hi')")
        open(os.path.join(gen, "model.py"), "w").write("# model")
        return gen

    events = []
    clone = make_paper_clone_fn("Reproduce paper P", llm=object(),
                                acquire_fn=_acquire, build_fn=_build,
                                emit=lambda e, p: events.append((e, p)))
    repo_path = os.path.join(tmp_path, "repo")
    clone("ignored-url", repo_path)

    # generated files landed AT repo_path (not in a nested {name}_repo dir)
    assert os.path.exists(os.path.join(repo_path, "main.py"))
    assert os.path.exists(os.path.join(repo_path, "model.py"))
    # the paper source is dropped in for the agent to consult
    assert os.path.exists(os.path.join(repo_path, "PAPER_source.json"))
    # events surfaced
    kinds = [e for e, _ in events]
    assert "paper_acquire" in kinds and "paper_build" in kinds
