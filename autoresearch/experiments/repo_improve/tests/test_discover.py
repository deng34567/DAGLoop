"""request -> benchmark discovery. The registry is EMPTY: every request resolves DYNAMICALLY
(LLM decides is_repo + names the URL, then infer_benchmark derives the boundary)."""
from __future__ import annotations

import json
import os
import importlib

import pytest

from autoresearch.experiments.repo_improve.benchmark import EXAMPLE_BENCHMARK
from autoresearch.experiments.repo_improve.discover import discover, infer_benchmark, _public_repo_status


@pytest.fixture(autouse=True)
def no_live_repository_probes(monkeypatch):
    module = importlib.import_module("autoresearch.experiments.repo_improve.discover")
    monkeypatch.setattr(module, "_public_repo_status", lambda url: 200)


@pytest.mark.parametrize("api_status,git_status,git_output,expected", [
    (403, 0, "abc\tHEAD", 200), (429, 0, "abc\tHEAD", 200),
    (403, 128, "", None), (403, 0, "", None), (404, 0, "abc\tHEAD", 404),
])
def test_public_repository_probe_handles_api_quota(monkeypatch, api_status, git_status, git_output, expected):
    import subprocess
    import urllib.error
    import urllib.request
    from types import SimpleNamespace
    url = "https://github.com/example/public-repo"
    calls = []

    def open_request(request, timeout):
        raise urllib.error.HTTPError(request.full_url, api_status, "test", {}, None)

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 8
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert not any(k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
                       for k in kwargs["env"])
        return SimpleNamespace(returncode=git_status, stdout=git_output)

    def build_opener(proxy_handler):
        assert proxy_handler.proxies == {}
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-use.invalid")
    assert _public_repo_status(url) == expected
    if api_status in (403, 429):
        assert calls[0][-3:] == ["--exit-code", url + ".git", "HEAD"]
    else:
        assert not calls


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Returns queued JSON responses in order (one per .chat call)."""
    def __init__(self, responses):
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.calls = []

    def chat(self, *, messages, **_):
        self.calls.append(messages)
        return _Resp(self._responses.pop(0))


@pytest.mark.parametrize("direction", [False, "false", 0])
def test_direction_only_correction_preserves_boundary_and_requires_bool(tmp_path, direction):
    from autoresearch.experiments.repo_improve.discover import correct_execution_from_run
    from dataclasses import replace
    bench = replace(EXAMPLE_BENCHMARK, lower_is_better=True)
    corrected = correct_execution_from_run(
        str(tmp_path), bench, "metric direction was wrong", "", "goal",
        _FakeLLM([{"lower_is_better": direction}]))
    if type(direction) is not bool:
        assert corrected is None
    else:
        assert corrected == replace(bench, lower_is_better=False)


def test_search_hits_are_not_automatically_selected_as_baselines():
    from autoresearch.experiments.repo_improve.discover import discover_candidates
    llm = _FakeLLM([{"repos": []}])
    hits = [{"title": "Unrelated repository", "id_or_url": "https://github.com/example/unrelated"}]
    assert discover_candidates("Find an implementation for the task", llm,
                               literature_brief={"repository_search_results": hits}) == []
    assert "Unrelated repository" in llm.calls[0][1]["content"]


def test_no_registered_match_without_llm_returns_none():
    # empty registry: a repo-shaped request with NO llm cannot be resolved -> None (no static
    # benchmark to keyword-match; dynamic discovery needs the llm).
    assert discover("improve nanoGPT single-GPU 5-minute training performance") is None
    assert discover("提升 nanogpt 在单卡上的五分钟训练性能") is None


def test_repo_request_resolves_dynamically_with_llm():
    # WITH an llm, a repo-shaped request is resolved dynamically (LLM names the URL).
    llm = _FakeLLM([{"is_repo": True, "repo_url": "https://github.com/karpathy/nanoGPT",
                     "name": "nanoGPT"}])
    b = discover("improve nanoGPT single-GPU training", llm)
    assert b is not None and b.repo_url == "https://github.com/karpathy/nanoGPT"
    assert b.editable_files == ()   # dynamic stub -> boundary inferred after clone


def test_unrelated_request_returns_none():
    # a general (non-repo) question is not a repo task -> None (no llm, nothing to match)
    assert discover("can first-fit-decreasing beat first-fit on bin packing?") is None


def test_autonomous_repo_request_is_not_rejected_for_missing_named_project():
    llm = _FakeLLM([{"is_repo": True,
                     "repo_url": "https://github.com/example/reference-code"}])
    request = "Compare tree access strategies; autonomously choose a public runnable implementation."
    result = discover(request, llm)
    assert result.repo_url == "https://github.com/example/reference-code"
    assert result.editable_files == ()  # Still scan and establish the actual boundary after clone.
    prompt = llm.calls[0][0]["content"]
    assert "absence of a supplied URL is not a rejection reason" in prompt
    assert "Do not invent a repository" in prompt


def test_example_benchmark_encodes_anticheat_invariant():
    # the neutral test fixture keeps the anti-cheat shape: agent edits train.py; prepare.py
    # (data + evaluator) is protected/trusted.
    assert EXAMPLE_BENCHMARK.editable_files == ("train.py",)
    assert "prepare.py" in EXAMPLE_BENCHMARK.protected_files
    assert EXAMPLE_BENCHMARK.metric == "val_bpb" and EXAMPLE_BENCHMARK.lower_is_better
    assert EXAMPLE_BENCHMARK.budget_s == 300


# --- dynamic discovery: an UNREGISTERED repo the request names -> clone stub -------------

def test_dynamic_stub_for_unregistered_repo():
    # not in the registry, but clearly about a specific real repo -> LLM names its URL
    llm = _FakeLLM([{"is_repo": True,
                     "repo_url": "https://github.com/some-owner/tiny-diffusion",
                     "name": "tiny-diffusion"}])
    b = discover("improve the some-owner/tiny-diffusion image sampler quality", llm)
    assert b is not None
    assert b.repo_url == "https://github.com/some-owner/tiny-diffusion"
    assert b.editable_files == ()   # empty = 'infer the trusted-eval boundary after clone'
    assert b.name == "tiny-diffusion"


def test_explicit_repo_routes_deterministically_without_llm():
    # an explicitly-named repo must route reliably (no dependence on the LLM's is_repo whim):
    # owner/repo slug + a repo hint (仓库/clone/github), or a full URL, resolves with NO llm.
    b = discover("克隆并改进 KellerJordan/cifar10-airbench 这个仓库", None)
    assert b is not None and b.repo_url == "https://github.com/KellerJordan/cifar10-airbench"
    assert b.name == "cifar10-airbench" and b.editable_files == ()
    b2 = discover("please improve https://github.com/owner/proj on its benchmark", None)
    assert b2 is not None and b2.repo_url == "https://github.com/owner/proj"
    # a stray slash in prose (no repo hint) must NOT be treated as a repo -> needs the llm
    assert discover("compare TP/FP tradeoffs on the dataset", None) is None


def test_explicit_repo_url_strips_sentence_punctuation():
    from autoresearch.experiments.repo_improve.discover import discover_candidates

    request = "Clone and improve https://github.com/karpathy/nanochat. Then report the result."
    direct = discover(request, None)
    candidates = discover_candidates(request, None)

    assert direct is not None
    assert direct.repo_url == "https://github.com/karpathy/nanochat"
    assert [candidate.repo_url for candidate in candidates] == [
        "https://github.com/karpathy/nanochat"]


def test_discover_candidates_merges_explicit_literature_and_llm():
    from autoresearch.experiments.repo_improve.discover import discover_candidates
    # explicit repo in the request + repos from the literature brief + an LLM proposal, deduped
    lit = {"existing_benchmarks": [{"name": "airbench", "id_or_url": "https://github.com/KellerJordan/cifar10-airbench"}],
           "key_papers": [{"title": "p", "id_or_url": "https://github.com/some/paper-code"}]}
    llm = _FakeLLM([{"repos": [{"url": "https://github.com/KellerJordan/cifar10-airbench", "why": "dup"},
                               {"url": "https://github.com/tysam-code/hlb-CIFAR10", "why": "fast"}]}])
    cands = discover_candidates("improve https://github.com/owner/proj CIFAR training", llm,
                                literature_brief=lit, max_candidates=5)
    urls = [c.repo_url for c in cands]
    assert "https://github.com/owner/proj" in urls          # explicit, first
    assert "https://github.com/KellerJordan/cifar10-airbench" in urls   # from literature
    assert "https://github.com/some/paper-code" in urls
    assert "https://github.com/tysam-code/hlb-CIFAR10" in urls          # from llm
    assert len(urls) == len(set(u.lower() for u in urls))   # airbench dup collapsed
    assert all(c.editable_files == () for c in cands)       # dynamic stubs


def test_discover_candidates_ranks_llm_target_before_reference_url():
    from autoresearch.experiments.repo_improve.discover import discover_candidates

    llm = _FakeLLM([{"repos": [
        {"url": "https://github.com/upstream/benchmark", "why": "canonical runnable baseline"},
    ]}])
    cands = discover_candidates(
        "Use https://github.com/example/results as a reference and discover the authoritative "
        "upstream benchmark repository to improve.",
        llm,
    )

    assert [candidate.repo_url for candidate in cands[:2]] == [
        "https://github.com/upstream/benchmark",
        "https://github.com/example/results",
    ]


def test_discovery_failure_evidence_is_not_an_explicit_repository(monkeypatch):
    module = importlib.import_module("autoresearch.experiments.repo_improve.discover")
    llm = _FakeLLM([{"repos": [{"url": "https://github.com/example/missing"},
                               {"url": "https://github.com/example/runnable"}]}])
    monkeypatch.setattr(module, "_public_repo_status",
                        lambda url: 200 if url.endswith('/runnable') else 404)
    failure = {"reproduced": False, "repo_url": "https://github.com/example/missing",
               "log_excerpt": "repository not found"}
    candidates = module.discover_candidates("Autonomously choose a runnable repository", llm,
                                            baseline_failure=failure)
    assert [c.repo_url for c in candidates] == ["https://github.com/example/runnable"]
    assert "repository not found" in llm.calls[0][1]["content"]


def test_discover_candidates_empty_for_general_question():
    from autoresearch.experiments.repo_improve.discover import discover_candidates
    llm = _FakeLLM([{"repos": []}])   # LLM: no canonical repo
    assert discover_candidates("does z-score standardization help k-NN?", llm) == []


def test_candidates_filter_unavailable_and_unverified_before_applying_limit(monkeypatch):
    module = importlib.import_module("autoresearch.experiments.repo_improve.discover")
    urls = ["https://github.com/example/" + name for name in
            ("missing", "rate-limited", "timeout", "reachable")]
    status = dict(zip(urls, [404, 429, None, 200]))
    seen = []
    monkeypatch.setattr(module, "_public_repo_status", lambda url: seen.append(url) or status[url])
    llm = _FakeLLM([{"repos": [{"url": url} for url in urls[:3]]}])
    literature = {"existing_benchmarks": [{"id_or_url": urls[3], "name": "from research"}]}
    candidates = module.discover_candidates("improve retrieval", llm,
                                            literature_brief=literature, max_candidates=1)
    assert [c.repo_url for c in candidates] == [urls[3]]
    assert seen == urls
    assert urls[3] in llm.calls[0][1]["content"]


def test_dynamic_returns_none_for_general_question():
    # a general algorithmic question is NOT a repo task -> None (fail-closed)
    llm = _FakeLLM([{"is_repo": False, "repo_url": "", "name": ""}])
    assert discover("does z-score standardization improve k-NN accuracy?", llm) is None


def test_dynamic_rejects_non_github_url():
    llm = _FakeLLM([{"is_repo": True, "repo_url": "not-a-url", "name": "x"}])
    assert discover("improve the mystery repo", llm) is None


def test_dynamic_needs_llm():
    # no LLM available -> cannot decide -> None
    assert discover("improve some-owner/some-repo", None) is None


# --- infer_benchmark: derive the trusted-eval boundary from a cloned repo ----------------

def _stub():
    from autoresearch.experiments.repo_improve.benchmark import Benchmark
    return Benchmark(name="some-speedrun", description="(dynamic) improve it",
                     repo_url="https://github.com/some-owner/some-speedrun",
                     editable_files=(), protected_files=(), prepare_cmd="", run_cmd="",
                     metric="", lower_is_better=True, budget_s=300, keywords=())


def test_infer_benchmark_from_repo_files(tmp_path):
    (tmp_path / "train.py").write_text("# training script\n")
    (tmp_path / "prepare.py").write_text("# builds data + scores metric\n")
    (tmp_path / "README.md").write_text("Run `python train.py`; it prints val_loss.\n")
    llm = _FakeLLM([{"editable_files": ["train.py"], "protected_files": ["prepare.py"],
                     "metric": "val_loss", "lower_is_better": True,
                     "prepare_cmd": "python prepare.py", "run_cmd": "python train.py",
                     "budget_s": 600}])
    b = infer_benchmark(str(tmp_path), "improve it", _stub(), llm)
    assert b is not None
    assert b.editable_files == ("train.py",) and b.protected_files == ("prepare.py",)
    assert b.metric == "val_loss" and b.lower_is_better and b.budget_s == 600
    assert b.run_cmd == "python train.py"
    # identity carried over from the stub
    assert b.name == "some-speedrun"
    assert b.repo_url == "https://github.com/some-owner/some-speedrun"


def test_infer_benchmark_correction_fixes_entrypoint_and_budget(tmp_path):
    # the validation/correction pass reads the ACTUAL scripts and fixes a guessed entrypoint +
    # a too-small budget (the airbench failure mode: 10 runs + compile + download >> 5 min).
    (tmp_path / "airbench94.py").write_text(
        "# trains 10 times with torch.compile, downloads CIFAR-10 on first run\n"
        "for run in range(10):\n    ...\nprint('mean accuracy', acc)\n")
    (tmp_path / "data.py").write_text("# downloads + provides CIFAR-10 (protected)\n")
    (tmp_path / "README.md").write_text("Run `python airbench94.py`; prints mean accuracy over 10 runs.\n")
    pass1 = {"editable_files": ["train.py"], "protected_files": ["prepare.py"],   # guessed, absent
             "metric": "loss", "lower_is_better": True, "prepare_cmd": "true",
             "run_cmd": "python train.py", "budget_s": 300}
    correction = {"editable_files": ["airbench94.py"], "protected_files": ["data.py"],
                  "metric": "accuracy", "lower_is_better": False, "prepare_cmd": "true",
                  "run_cmd": "python airbench94.py", "budget_s": 1800}
    b = infer_benchmark(str(tmp_path), "improve airbench", _stub(), _FakeLLM([pass1, correction]))
    assert b is not None
    assert b.editable_files == ("airbench94.py",)     # corrected to the REAL entrypoint (exists)
    assert b.run_cmd == "python airbench94.py"
    assert b.budget_s == 1800                          # corrected budget covers compile + 10 runs
    assert b.metric == "accuracy" and b.lower_is_better is False
    assert b.protected_files == ("data.py",)


def test_cmd_script_tokens():
    from autoresearch.experiments.repo_improve.discover import _cmd_script_tokens
    assert _cmd_script_tokens("python train.py --config config/train_fineweb.py --max_steps 1000") == [
        "train.py", "config/train_fineweb.py"]
    assert _cmd_script_tokens("torchrun --standalone --nproc_per_node=1 train_gpt.py") == ["train_gpt.py"]
    assert _cmd_script_tokens("bash scripts/run.sh") == ["scripts/run.sh"]
    assert _cmd_script_tokens("python -m package.module") == []   # no .py/.sh file token


def test_infer_benchmark_entrypoint_guard_fixes_hallucinated_run_cmd(tmp_path):
    # modded-nanogpt failure mode: the LLM copies nanoGPT's `python train.py config/..` onto a repo
    # that has ONLY train_gpt.py + run.sh (no train.py, no config/). The deterministic entrypoint
    # guard detects the run_cmd names non-existent script(s) and re-derives it from the repo's REAL
    # entrypoint (run.sh) BEFORE the run is ever attempted -- so the baseline isn't a guaranteed fail.
    (tmp_path / "train_gpt.py").write_text("print('val_loss', 1.0)\n")
    (tmp_path / "run.sh").write_text("torchrun --standalone --nproc_per_node=8 train_gpt.py\n")
    (tmp_path / "README.md").write_text("Run `sh run.sh`.\n")
    pass1 = {"editable_files": ["train.py"], "metric": "val_loss", "lower_is_better": True,
             "prepare_cmd": "true", "budget_s": 600,
             "run_cmd": "python train.py --config config/train_fineweb.py --max_steps 1000"}
    # the generic correction pass fixes the editable file but (weak LLM) leaves the bad run_cmd
    correction = {"editable_files": ["train_gpt.py"],
                  "run_cmd": "python train.py --config config/train_fineweb.py --max_steps 1000"}
    # the entrypoint guard's targeted fix, grounded in run.sh
    entry_fix = {"run_cmd": "torchrun --standalone --nproc_per_node=1 train_gpt.py", "prepare_cmd": "true"}
    b = infer_benchmark(str(tmp_path), "improve modded-nanogpt", _stub(),
                        _FakeLLM([pass1, correction, entry_fix]), permissive=True)
    assert b is not None
    assert b.run_cmd == "torchrun --standalone --nproc_per_node=1 train_gpt.py"  # real entrypoint (exists)
    assert b.editable_files == ("train_gpt.py",)


def test_infer_benchmark_infers_seed_env_from_script(tmp_path):
    # the correction pass sees the script reads a seed env var -> reproduce can really vary the
    # seed. When the script has NO seed hook, seed_env stays None (reproduce = honest rerun).
    (tmp_path / "run.py").write_text(
        "import os\nseed = int(os.environ.get('SEED', 0))\nprint('acc', 1)\n")
    (tmp_path / "data.py").write_text("# provides data (protected)\n")
    pass1 = {"editable_files": ["run.py"], "protected_files": ["data.py"], "metric": "acc",
             "lower_is_better": False, "prepare_cmd": "true", "run_cmd": "python run.py",
             "budget_s": 300}
    correction = {"run_cmd": "python run.py", "seed_env": "SEED"}
    b = infer_benchmark(str(tmp_path), "improve run", _stub(), _FakeLLM([pass1, correction]))
    assert b is not None and b.seed_env == "SEED"        # real seed hook -> reproduce varies it

    # a repo with no seed hook -> the model returns "" -> seed_env is None (not a faked swap)
    correction_noseed = {"run_cmd": "python run.py", "seed_env": ""}
    b2 = infer_benchmark(str(tmp_path), "improve run", _stub(), _FakeLLM([pass1, correction_noseed]))
    assert b2 is not None and b2.seed_env is None


def test_infer_benchmark_drops_nonexistent_editable_after_correction(tmp_path):
    # deterministic guard: if the (post-correction) editable file does not exist in the clone,
    # fail-closed (None) rather than run a broken boundary that can only crash/time out.
    (tmp_path / "real_train.py").write_text("print('acc', 1)\n")
    pass1 = {"editable_files": ["ghost.py"], "protected_files": [], "metric": "acc",
             "lower_is_better": False, "prepare_cmd": "true", "run_cmd": "python ghost.py",
             "budget_s": 300}
    # correction also names a non-existent file -> nothing editable exists -> None
    correction = {"editable_files": ["also_ghost.py"]}
    assert infer_benchmark(str(tmp_path), "x", _stub(), _FakeLLM([pass1, correction])) is None


def test_infer_benchmark_user_overrides_win(tmp_path):
    # the user CONFIRMED the boundary (they know the repo): their run_cmd / metric / budget win
    # over the LLM's inference (fixes e.g. makemore's missing `-i names.txt`, a max-steps budget).
    (tmp_path / "makemore.py").write_text("import argparse\nprint('val loss', 1.0)\n")
    (tmp_path / "names.txt").write_text("emma\n")
    infer = {"editable_files": ["makemore.py"], "protected_files": ["names.txt"],
             "metric": "loss", "lower_is_better": True, "prepare_cmd": "true",
             "run_cmd": "python makemore.py", "budget_s": 300}   # LLM: missing -i names.txt
    overrides = {"run_cmd": "python makemore.py -i names.txt --max-steps 2000",
                 "metric": "val loss", "budget_s": 1200}
    b = infer_benchmark(str(tmp_path), "improve makemore", _stub(),
                        _FakeLLM([infer, infer]), overrides=overrides)
    assert b is not None
    assert b.run_cmd == "python makemore.py -i names.txt --max-steps 2000"   # user's run_cmd wins
    assert b.metric == "val loss" and b.budget_s == 1200
    assert b.editable_files == ("makemore.py",)   # inference kept (not overridden), exists


def test_infer_benchmark_complete_override_works_without_llm(tmp_path):
    # a COMPLETE user-confirmed boundary needs no LLM at all (they told us everything).
    (tmp_path / "run.py").write_text("print('acc', 0.9)\n")
    (tmp_path / "data.py").write_text("# protected\n")
    overrides = {"editable_files": ["run.py"], "protected_files": ["data.py"],
                 "metric": "acc", "lower_is_better": False, "prepare_cmd": "true",
                 "run_cmd": "python run.py", "budget_s": 600}
    b = infer_benchmark(str(tmp_path), "x", _stub(), None, overrides=overrides)
    assert b is not None
    assert b.editable_files == ("run.py",) and b.run_cmd == "python run.py"
    assert b.metric == "acc" and b.lower_is_better is False and b.budget_s == 600


def test_infer_benchmark_no_llm_no_overrides_is_none(tmp_path):
    (tmp_path / "train.py").write_text("x\n")
    assert infer_benchmark(str(tmp_path), "x", _stub(), None) is None   # unchanged


def test_infer_benchmark_fail_closed_without_editable(tmp_path):
    (tmp_path / "train.py").write_text("x\n")
    llm = _FakeLLM([{"editable_files": [], "protected_files": ["train.py"],
                     "metric": "m", "lower_is_better": True,
                     "prepare_cmd": "true", "run_cmd": "python train.py", "budget_s": 300}])
    # no editable file named -> cannot form a safe boundary -> None
    assert infer_benchmark(str(tmp_path), "improve it", _stub(), llm) is None


def test_infer_benchmark_skips_git_dir(tmp_path):
    (tmp_path / "train.py").write_text("x\n")
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]\n")
    llm = _FakeLLM([{"editable_files": ["train.py"], "protected_files": [],
                     "metric": "m", "lower_is_better": True,
                     "prepare_cmd": "true", "run_cmd": "python train.py", "budget_s": 300}])
    b = infer_benchmark(str(tmp_path), "improve it", _stub(), llm)
    assert b is not None
    # the .git internals must not be surfaced to the model as candidate files
    user_msg = llm.calls[0][1]["content"]
    assert ".git" not in user_msg
    assert os.path.join(".git", "config") not in user_msg


def test_read_setup_files_picks_up_manifests(tmp_path):
    # the install/build manifests get read (whole-repo style) so prepare_cmd can be derived per
    # project, no hardcoded rule -- covers python packaging, containers, make, install scripts.
    from autoresearch.experiments.repo_improve.discover import _read_setup_files
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "Dockerfile").write_text("RUN uv sync\n")
    (tmp_path / "train.py").write_text("print('hi')\n")
    files = ["pyproject.toml", "docker/Dockerfile", "train.py"]
    out = _read_setup_files(str(tmp_path), files)
    assert "pyproject.toml" in out and "docker/Dockerfile" in out and "uv sync" in out
    assert "train.py" not in out            # non-manifest source is not swept in


def test_infer_strips_leading_cd_from_commands(tmp_path):
    # LLMs copy `cd /repo && ...` from Dockerfiles; the runner already cwd's into the workspace,
    # so a leading absolute cd would break the run -> it must be stripped.
    from autoresearch.experiments.repo_improve.discover import infer_benchmark
    (tmp_path / "run.py").write_text("print('acc', 1)\n")
    (tmp_path / "data.py").write_text("# data\n")
    pass1 = {"editable_files": ["run.py"], "protected_files": ["data.py"], "metric": "acc",
             "lower_is_better": False, "prepare_cmd": "true", "run_cmd": "python run.py", "budget_s": 300}
    correction = {"prepare_cmd": "cd /repo && uv sync --frozen",
                  "run_cmd": "cd /repo && uv run python run.py"}
    b = infer_benchmark(str(tmp_path), "improve run", _stub(), _FakeLLM([pass1, correction]))
    assert b is not None
    assert b.prepare_cmd == "uv sync --frozen"       # leading cd stripped
    assert b.run_cmd == "uv run python run.py"


def test_infer_preserves_relative_subdir_cd(tmp_path):
    # A RELATIVE subdir cd must be KEPT: repos like minGPT's chargpt example must run from their own
    # directory (the script + its data are resolved relative to CWD). Only ABSOLUTE cds are stripped.
    from autoresearch.experiments.repo_improve.discover import infer_benchmark
    (tmp_path / "run.py").write_text("print('acc', 1)\n")
    (tmp_path / "data.py").write_text("# data\n")
    pass1 = {"editable_files": ["run.py"], "protected_files": ["data.py"], "metric": "acc",
             "lower_is_better": False, "prepare_cmd": "true", "run_cmd": "python run.py", "budget_s": 300}
    correction = {"prepare_cmd": "cd projects/chargpt && wget -q http://x/input.txt",
                  "run_cmd": "cd projects/chargpt && python chargpt.py"}
    b = infer_benchmark(str(tmp_path), "improve run", _stub(), _FakeLLM([pass1, correction]))
    assert b is not None
    assert b.prepare_cmd == "cd projects/chargpt && wget -q http://x/input.txt"   # relative cd kept
    assert b.run_cmd == "cd projects/chargpt && python chargpt.py"


def test_correct_run_heals_setup_from_error(tmp_path):
    # ONE general self-heal: given a run failure + the repo, correct_execution_from_run asks the LLM
    # for a corrected prepare_cmd/run_cmd (from the ERROR) and returns a NEW benchmark with those
    # swapped -- the trusted boundary (editable/protected) is unchanged. Interactive docker flags stripped.
    from autoresearch.experiments.repo_improve.discover import correct_execution_from_run
    from autoresearch.experiments.repo_improve.benchmark import Benchmark
    (tmp_path / "chargpt.py").write_text("open('input.txt').read()\n")
    (tmp_path / "README.md").write_text("needs tiny-shakespeare as input.txt\n")
    bench = Benchmark(name="minGPT", description="", repo_url="u", editable_files=("chargpt.py",),
                      protected_files=("README.md",), prepare_cmd="true", run_cmd="python chargpt.py",
                      metric="loss", lower_is_better=True, budget_s=300)
    llm = _FakeLLM([{"prepare_cmd": "docker run -it foo && curl -L http://x/input.txt -o input.txt",
                     "run_cmd": "python chargpt.py", "metric": "loss", "lower_is_better": True,
                     "budget_s": 300, "diagnosis": "missing input.txt"}])
    r = correct_execution_from_run(str(tmp_path), bench, "FileNotFoundError: 'input.txt'", "", "improve", llm)
    assert r is not None
    assert "curl" in r.prepare_cmd and "input.txt" in r.prepare_cmd   # data download proposed
    assert "-it" not in r.prepare_cmd                                 # interactive docker flag stripped
    assert r.editable_files == ("chargpt.py",) and r.protected_files == ("README.md",)  # boundary fixed
    assert r.metric == "loss" and r.budget_s == 300


def test_correct_run_heals_metric_from_actual_output(tmp_path):
    # the SAME self-heal fixes a wrong metric NAME/PATH from the run's ACTUAL output -- no separate
    # repair_metric, no hard-coded metric string. The commands + boundary stay unchanged.
    from autoresearch.experiments.repo_improve.discover import correct_execution_from_run
    from autoresearch.experiments.repo_improve.benchmark import Benchmark
    bench = Benchmark(name="sol", description="", repo_url="u", editable_files=("solution.json",),
                      protected_files=("harness.py",), prepare_cmd="uv sync",
                      run_cmd="uv run bench --json", metric="latency_ms", lower_is_better=True, budget_s=300)
    output = '{"evaluation": {"performance": {"latency_ms": 0.115}}}'
    llm = _FakeLLM([{"prepare_cmd": "uv sync", "run_cmd": "uv run bench --json",
                     "metric": "evaluation.performance.latency_ms", "lower_is_better": True,
                     "budget_s": 300, "diagnosis": "metric is nested"}])
    r = correct_execution_from_run(str(tmp_path), bench, "could not find metric 'latency_ms'", output,
                                   "lower latency", llm)
    assert r is not None
    assert r.metric == "evaluation.performance.latency_ms"            # metric healed from real output
    assert r.run_cmd == "uv run bench --json"                         # commands unchanged
    assert r.editable_files == ("solution.json",) and r.protected_files == ("harness.py",)  # boundary fixed


def test_correct_run_none_when_no_change_or_no_llm(tmp_path):
    from autoresearch.experiments.repo_improve.discover import correct_execution_from_run
    from autoresearch.experiments.repo_improve.benchmark import Benchmark
    bench = Benchmark(name="x", description="", repo_url="u", editable_files=("a.py",),
                      protected_files=(), prepare_cmd="true", run_cmd="python a.py",
                      metric="m", lower_is_better=True, budget_s=300)
    assert correct_execution_from_run(str(tmp_path), bench, "err", "", "goal", None) is None   # no llm
    # proposal identical to current -> no-op guard returns None (loop stops, no spin)
    llm = _FakeLLM([{"prepare_cmd": "true", "run_cmd": "python a.py", "metric": "m",
                     "lower_is_better": True, "budget_s": 300}])
    assert correct_execution_from_run(str(tmp_path), bench, "err", "", "goal", llm) is None
