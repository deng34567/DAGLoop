"""Resolve a user request -> a registered Benchmark ("AI finds the repo").

Three paths, in order:
  - keyword match: score each REGISTERED benchmark by how many of its keywords the request
    mentions (pure, deterministic, no LLM). A fast known-good path for the phrasings users
    actually type ("improve nanoGPT single-GPU 5-minute training").
  - LLM tie-break (optional): when a client is given and the keyword match is ambiguous, ask
    the model to pick the best-matching registered benchmark by name. Fail-safe: if the
    model names something not in the registry, fall back to the keyword winner.
  - DYNAMIC discovery: when NOTHING registered matches, ask the LLM whether the request is
    about a specific real repo/baseline and, if so, get its GitHub URL -> return a STUB
    (repo_url set, editable_files empty). The pipeline then clones it and ``infer_benchmark``
    derives the trusted-eval boundary from the cloned files. This is what lets the system
    improve ANY repo the user names, not only a hand-registered one; the registry is now an
    optional fast-path, not a requirement. Fail-closed: a general (non-repo) question, no
    LLM, or any error -> None.
"""
from __future__ import annotations

import json
import logging
import os
import re

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.experiments.repo_improve.benchmark import Benchmark


def _strip_leading_cd(cmd: str) -> str:
    """Strip a leading `cd <ABS_PATH> &&` an LLM may copy from a Dockerfile -- the runner already
    runs with cwd=the variant workspace, so an ABSOLUTE cd (``/app`` / ``~/x``) points outside it
    and breaks the run. A RELATIVE subdir cd (``cd projects/chargpt && ...``) is PRESERVED: many
    repo entry scripts must run from their own subdirectory (both to find the script and for it to
    resolve data files relative to CWD), and a relative cd stays safely inside the workspace."""
    return re.sub(r"^\s*cd\s+[/~][^\s&|;]*\s*&&\s*", "", cmd or "")

# --- Dynamic discovery: clone ANY real repo the user's request is about. There is no static
# registry -- the LLM decides whether the request is about a SPECIFIC code repo/training
# baseline and, if so, names the GitHub URL. Conservative: a general algorithmic/data
# question -> is_repo=false.
_DYNAMIC_SYSTEM = (
    "You resolve a research request to a real CODE REPOSITORY or training baseline. "
    "The user may name the repository, or explicitly ask you to autonomously find and choose "
    "a public runnable implementation for a code-based experiment. In the latter case, "
    "select a suitable real baseline repository even when the question is algorithmic and "
    "has no single canonical repository; the absence of a supplied URL is not a rejection reason. "
    "Do not invent a repository or substitute a paper/data-only release for runnable code. "
    "It is NOT a repo task if it is only a general "
    "algorithmic or data-science question (e.g. 'does standardization help k-NN', 'beat "
    "First-Fit on bin packing') without a request to find or run repository code. If a real "
    "repository can be selected for the requested experiment, give its GitHub URL; otherwise "
    "return is_repo=false. Reply with ONE JSON "
    'object: {"is_repo": <bool>, "repo_url": "<https://github.com/owner/repo or empty>", '
    '"name": "<short slug>"}.'
)

# The trusted-eval boundary for an arbitrary cloned repo, inferred from its files + docs.
_INFER_SYSTEM = (
    "You define the TRUSTED-EVALUATION BOUNDARY for improving a cloned code repository, so an "
    "agent can improve a metric WITHOUT being able to tamper with how it is scored. Given the "
    "repo's file list + key docs and the user's goal, identify:\n"
    "(1) editable_files -- the file(s) holding the IMPLEMENTATION the user wants to improve, the "
    "ONLY files the agent may modify. Most often a training/method script (e.g. train.py), but "
    "NOT always a .py: it can be a solution/config/data file that EMBEDS the code or parameters "
    "to optimize -- e.g. a JSON holding a CUDA/kernel source, a config the run reads, a submission "
    "file in a benchmark suite. Pick whatever file the user's improvement actually lives in.\n"
    "(2) protected_files -- everything that COMPUTES, CHECKS, TIMES, or PROVIDES THE GROUND TRUTH "
    "for the metric, which MUST stay pristine: the evaluator / scoring harness / benchmark runner "
    "/ problem definition / data-prep / dataset / the dep-and-build manifests (pyproject, setup, "
    "requirements, Dockerfile, Makefile). If unsure whether a file scores/provides-truth, mark it "
    "protected, NEVER editable.\n"
    "(3) metric -- the exact name of the number to optimize, OR a dotted PATH into the run's "
    "JSON/JSONL output when the run prints structured results (e.g. "
    "'evaluation.performance.latency_ms').\n"
    "(4) lower_is_better.\n"
    "(5) prepare_cmd -- one-time data/env setup (fold any required dirs/exports in here, e.g. "
    "`mkdir -p /tmp/trace && uv sync`), or 'true' if none.\n"
    "(6) run_cmd -- the command that runs the benchmark/training and EMITS the metric. This can be "
    "a training script OR the repo's own benchmark CLI (e.g. `uv run <tool> --definition ... "
    "--json`). Put any env vars the run needs inline (`FOO=bar <cmd>`).\n"
    "(7) budget_s -- a sensible wall-clock budget in seconds.\n"
    "(8) image -- if this repo can ONLY run inside a special environment it SHIPS ITSELF (a "
    "Dockerfile for CUDA/CUTLASS/nvcc or heavy native deps that a plain `pip`/`uv` on the host "
    "cannot provide), set image to \"build:<relative path to that Dockerfile>\" (e.g. "
    "\"build:docker/Dockerfile\"). If the repo runs fine with ordinary Python deps, use null.\n"
    "Reply with ONE JSON object: {\"editable_files\": [...], \"protected_files\": [...], "
    "\"metric\": \"...\", \"lower_is_better\": <bool>, \"prepare_cmd\": \"...\", \"run_cmd\": "
    "\"...\", \"budget_s\": <int>, \"image\": <string or null>}."
)

# Second pass: VERIFY + CORRECT the boundary now that we can read the ACTUAL entry scripts
# (not just filenames). The common failure this catches: a budget_s that ignores cold-start
# (first-time dataset download + torch.compile/JIT warmup) and repeated runs (speedrun scripts
# often train N times to report a mean) -> the run TIMES OUT and scores invalid.
_CORRECT_SYSTEM = (
    "You VERIFY and CORRECT a proposed trusted-eval boundary for a cloned repo, now that you "
    "can see the ACTUAL content of its entry/training scripts. Fix it so it will really RUN and "
    "emit the metric: (1) editable_files = the REAL file(s) that EXIST here holding the "
    "implementation to improve -- usually the training/method script (use the actual entrypoint "
    "you see, not a guessed name like train.py), BUT if the implementation lives in a non-script "
    "file (a solution/config/data file that embeds the code or parameters, e.g. a JSON holding a "
    "kernel), KEEP that file as editable -- do NOT swap it for a script that merely runs it; "
    "(2) protected_files "
    "= the real data/eval/metric files; (3) run_cmd = the command that actually trains and "
    "prints the metric; (4) prepare_cmd = the ONE-TIME setup this repo needs so run_cmd will "
    "actually work -- DERIVE IT FROM THE INSTALL/BUILD MANIFESTS SHOWN (do not hardcode). If the "
    "run_cmd invokes the repo's own CLI/package or imports it, the deps + package MUST be installed "
    "first (e.g. `uv pip install -e .` / `pip install -e .` / `pip install -r requirements.txt` / "
    "`conda env create -f environment.yml`); if a native/CUDA kernel or C++ ext must be built, "
    "include the build (e.g. `make` / `nvcc ...` / the setup.py build); include data prep too. "
    "Chain steps with `&&`. Use 'true' ONLY if the repo genuinely needs no setup. The dep/build "
    "manifests (pyproject/setup/requirements/Dockerfile/Makefile) belong in protected_files so the "
    "environment can't be tampered with to cheat; (5) metric = the EXACT "
    "name the script prints; (6) lower_is_better; (7) budget_s = a REALISTIC wall-clock budget "
    "that covers COLD START (first-time dataset download + any torch.compile/JIT warmup) AND "
    "EVERY repeated run the script does -- err on the LARGE side; a too-small budget makes the "
    "run time out and score invalid. (8) seed_env = the NAME of an environment variable the "
    "training script ACTUALLY reads to seed its RNG (e.g. you see os.environ.get(\"SEED\") / "
    "int(os.environ[\"SEED\"])); use \"\" if the script has no such env-seed hook -- do NOT invent "
    "one, reproduce depends on the script really reading it. Reply with ONE JSON object with keys: "
    "editable_files, protected_files, metric, lower_is_better, prepare_cmd, run_cmd, budget_s, "
    "seed_env."
)


def _read_scripts(pristine_dir: str, rels, *, cap_each: int = 6000, cap_total: int = 14000) -> str:
    """Concatenate the content of the given repo-relative files (capped), for the correction
    pass -- so the LLM corrects run_cmd/budget from what the scripts ACTUALLY do."""
    out = ""
    for rel in rels:
        p = os.path.join(pristine_dir, rel)
        if os.path.isfile(p):
            try:
                out += f"\n--- {rel} ---\n" + open(p, encoding="utf-8", errors="ignore").read()[:cap_each]
            except OSError:
                pass
        if len(out) >= cap_total:
            break
    return out[:cap_total]


# Setup/build manifests read (WHOLE-repo style) so the LLM can derive a REAL install+build
# prepare_cmd per project -- no hardcoded "pip install -e ." rule. Covers python packaging, envs,
# containers, and make/build files. Whichever exist get shown to the correction pass.
_SETUP_FILE_HINTS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements-dev.txt",
    "environment.yml", "environment.yaml", "uv.lock", "Pipfile", "poetry.lock",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml", "Makefile", "makefile",
    "install.sh", "setup.sh", "build.sh", "INSTALL.md", "INSTALL",
    "run_docker.sh", "entrypoint.sh", "download_data.sh", "run.sh",
)


def _read_setup_files(pristine_dir: str, files: list[str], *, cap_each: int = 4000,
                      cap_total: int = 12000) -> str:
    """Concatenate the repo's install/build/env manifests (Dockerfile, pyproject, requirements,
    Makefile, install scripts, ...) so the correction pass can derive a per-project prepare_cmd
    that ACTUALLY installs deps + builds -- instead of a hardcoded install rule. Best-effort."""
    lower = {f.lower(): f for f in files}
    picked: list[str] = []
    for hint in _SETUP_FILE_HINTS:
        # match by basename anywhere in the tree (e.g. docker/Dockerfile), top-level first
        for rel in files:
            base = rel.rsplit("/", 1)[-1].lower()
            if base == hint.lower() and rel not in picked:
                picked.append(rel)
    out = ""
    for rel in picked:
        p = os.path.join(pristine_dir, rel)
        if os.path.isfile(p):
            try:
                out += f"\n--- {rel} ---\n" + open(p, encoding="utf-8", errors="ignore").read()[:cap_each]
            except OSError:
                pass
        if len(out) >= cap_total:
            break
    return out[:cap_total]


def _candidate_scripts(obj: dict, files: list[str]) -> list[str]:
    """Which repo files to show the correction pass: the proposed editable files, any file the
    run/prepare commands reference, and top-level *.py entrypoints. Deduped, capped."""
    cands: list[str] = []
    for x in (obj.get("editable_files") or []):
        cands.append(str(x).strip())
    cmds = f"{obj.get('run_cmd') or ''} {obj.get('prepare_cmd') or ''}"
    cands += [f for f in files if f and f in cmds]
    # top-level entrypoints of ANY common source/config kind -- NOT .py-only, so a repo whose
    # implementation lives in a .cu/.cpp/.json/.yaml/.sh file is read too (no train.py bias).
    _ENTRY_EXT = (".py", ".sh", ".cu", ".cpp", ".cc", ".c", ".json", ".yaml", ".yml", ".toml")
    cands += [f for f in files if os.sep not in f and f.endswith(_ENTRY_EXT)]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out[:6]


# a repo-relative script token a shell command invokes (``train.py``, ``scripts/run.sh``, ``train_gpt.py``).
_CMD_SCRIPT_TOKEN = re.compile(r"[\w./-]+\.(?:py|sh)\b")


def _cmd_script_tokens(cmd: str) -> list[str]:
    """Script file token(s) a shell command names (``*.py``/``*.sh``). Used to check the command
    points at an entrypoint that ACTUALLY exists in the clone -- an LLM sometimes copies a run_cmd
    from a different repo's convention (nanoGPT's ``python train.py config/..``) onto a repo that
    has no such file (modded-nanogpt has only ``train_gpt.py`` + ``run.sh``), which can only fail."""
    seen, out = set(), []
    for t in _CMD_SCRIPT_TOKEN.findall(cmd or ""):
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


_GITHUB_URL = re.compile(r"https?://(www\.)?github\.com/[\w.-]+/[\w.-]+", re.I)
# An owner/repo slug ("KellerJordan/cifar10-airbench"), and words that signal the user really
# means a repo (so a stray "TP/FP"-style slash in prose doesn't get treated as a repo).
_REPO_SLUG = re.compile(r"([A-Za-z0-9][\w.-]*)/([A-Za-z][\w.-]*)")
_REPO_HINT = re.compile(r"github|clone|repository|仓库|克隆", re.I)


def _clean_repo_url(url: str) -> str:
    """Normalize a GitHub repository URL captured from ordinary prose."""
    cleaned = (url or "").strip().rstrip("/.,;:!?)]}")
    return cleaned[:-4] if cleaned.lower().endswith(".git") else cleaned


def _explicit_repo_url(request: str) -> str | None:
    """Deterministically extract an EXPLICIT repo reference from the request: a full github URL,
    or an ``owner/repo`` slug when the request clearly means a repo (mentions github/clone/仓库).
    Returns the normalized URL or None. Makes routing for an explicitly-named repo reliable
    instead of dependent on the LLM's (flaky) is_repo judgment."""
    m = _GITHUB_URL.search(request or "")
    if m:
        return _clean_repo_url(m.group(0))
    if _REPO_HINT.search(request or ""):
        s = _REPO_SLUG.search(request or "")
        if s:
            return f"https://github.com/{s.group(1)}/{s.group(2)}"
    return None


def _stub_for_url(url: str, request: str) -> Benchmark:
    url = _clean_repo_url(url)
    name = url.rstrip("/").split("/")[-1][:80] or "dynamic-repo"
    return Benchmark(
        name=name, description=f"(dynamic) {request[:140]}", repo_url=url,
        editable_files=(), protected_files=(), prepare_cmd="", run_cmd="",
        metric="", lower_is_better=True, budget_s=300, keywords=())


def _discover_dynamic(request: str, llm) -> Benchmark | None:
    """Resolve ``request`` to a DYNAMIC Benchmark STUB (repo_url set; editable_files empty =
    'infer the boundary after clone'). First tries a DETERMINISTIC explicit-repo extraction (a
    named github URL / owner-repo slug) -- no LLM, no flakiness. Only a request with NO explicit
    repo falls back to letting the LLM decide is_repo. None for a general question. FAIL-CLOSED."""
    explicit = _explicit_repo_url(request)
    if explicit:
        return _stub_for_url(explicit, request)
    if llm is None:
        return None
    try:
        result = llm.chat(
            messages=[{"role": "system", "content": _DYNAMIC_SYSTEM},
                      {"role": "user", "content": f"Request:\n{request}"}],
            response_format="json_object")
        obj = json.loads(extract_json(result.content or "{}"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict) or not obj.get("is_repo"):
        return None
    url = str(obj.get("repo_url") or "").strip()
    m = _GITHUB_URL.match(url)
    if not m:
        return None
    url = _clean_repo_url(m.group(0))
    name = str(obj.get("name") or url.split("/")[-1]).strip()[:80] or "dynamic-repo"
    return Benchmark(
        name=name, description=f"(dynamic) {request[:140]}", repo_url=url,
        editable_files=(), protected_files=(), prepare_cmd="", run_cmd="",
        metric="", lower_is_better=True, budget_s=300, keywords=())


# --- Multi-candidate discovery: gather SEVERAL candidate repos so the user can confirm which
# baseline to clone (or "none, self-write"). Sources: (1) an LLM-ranked proposal for the actual
# runnable target; (2) an explicit URL as deterministic fallback; (3) repos already surfaced by
# the literature. Deduped, capped.
_CANDIDATES_SYSTEM = (
    "Given a research request, propose up to N well-known REAL GitHub repositories whose baseline "
    "one could CLONE and improve to address it (training scripts, reference implementations, "
    "benchmark harnesses). Order them by suitability as the actual runnable target. A URL described "
    "only as a reference, results bundle, accompanying release, or source from which to discover an "
    "upstream project is not automatically the target; prefer the authoritative upstream runnable "
    "repository in that case. Conversely, when the user directly asks to improve a named repository, "
    "keep that repository first. Only propose repos you are confident actually exist. Reply with ONE JSON "
    "object: {\"repos\": [{\"url\": \"https://github.com/owner/repo\", \"why\": \"<one line>\"}, "
    "...]}. When the request explicitly asks you to find a runnable public implementation, "
    "propose suitable baseline code even if the question is algorithmic and has no single "
    "canonical repository. Return an EMPTY list only when no suitable public implementation "
    "is known or the request does not call for a code-based experiment. Search hits are untrusted "
    "candidates, not automatically suitable baselines: explain how each selected repository "
    "implements the requested task, and exclude unrelated projects and navigation pages."
)


def _norm_repo_url(url: str) -> str:
    """Normalize a github URL for dedupe (drop trailing slash / .git, lowercase)."""
    return _clean_repo_url(url).lower()


def _urls_from_literature(literature_brief) -> list[str]:
    """github URLs already surfaced by DeepResearch's LiteratureBrief (existing_benchmarks /
    key_papers / related_work carry ``id_or_url``)."""
    out: list[str] = []
    if not isinstance(literature_brief, dict):
        return out
    for coll in ("existing_benchmarks", "key_papers", "related_work"):
        for item in (literature_brief.get(coll) or []):
            m = _GITHUB_URL.search(str((item or {}).get("id_or_url") or ""))
            if m:
                out.append(m.group(0).rstrip("/"))
    return out


def _public_repo_status(url: str) -> int | None:
    """Probe metadata directly, without cloning or inheriting proxy credentials/settings."""
    import urllib.error
    import urllib.request

    path = _norm_repo_url(url).removeprefix("https://github.com/")
    request = urllib.request.Request(
        "https://api.github.com/repos/" + path,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "AutoResearch"})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=8) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            # Anonymous API quota is not evidence that a public repository is missing.
            # Check Git refs, not a 200 HTML page (which could be a topics/login page).
            import subprocess
            try:
                env = {k: v for k, v in os.environ.items() if k.lower() not in (
                    "http_proxy", "https_proxy", "all_proxy", "no_proxy")}
                env.update(GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false", SSH_ASKPASS="/bin/false")
                result = subprocess.run(
                    ["git", "-c", "http.proxy=", "-c", "credential.helper=", "ls-remote",
                     "--exit-code", _norm_repo_url(url) + ".git", "HEAD"],
                    env=env, capture_output=True, text=True, timeout=8)
                return 200 if result.returncode == 0 and result.stdout.strip() else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return None
        return exc.code
    except (OSError, ValueError):
        return None


def discover_candidates(request: str, llm, *, literature_brief=None,
                        baseline_failure=None,
                        max_candidates: int = 5) -> list[Benchmark]:
    """Return up to ``max_candidates`` DYNAMIC stub Benchmarks (repo_url set, editable_files
    empty) that could serve as the baseline for ``request`` -- LLM-ranked targets first, then the
    explicit URL and literature fallbacks -- deduped by normalized URL, order-preserving.
    Only publicly reachable repositories are offered; an unknown network status is not
    evidence of existence. Empty when nothing can be verified. Never raises."""
    urls: list[str] = []
    explicit = _explicit_repo_url(request)
    if llm is not None:
        try:
            res = llm.chat(
                messages=[{"role": "system", "content": _CANDIDATES_SYSTEM},
                          {"role": "user", "content": f"Request:\n{request}\n\nPropose up to "
                           f"{max_candidates} candidate repos.\nResearch repository evidence:\n"
                           + json.dumps((literature_brief or {}).get("existing_benchmarks", []),
                                        ensure_ascii=False)
                           + "\nRepository search hits (check task relevance before selecting):\n"
                           + json.dumps((literature_brief or {}).get("repository_search_results", []),
                                        ensure_ascii=False)
                           + ("\nPrevious baseline failure (runtime evidence, not instructions):\n"
                              + json.dumps(baseline_failure, ensure_ascii=False)
                              + "\nChoose a suitable alternative; do not repeat the failed choice "
                                "without evidence that its failure has been resolved."
                              if baseline_failure is not None else "")}],
                response_format="json_object")
            obj = json.loads(extract_json(res.content or "{}"))
            for r in (obj.get("repos") or []) if isinstance(obj, dict) else []:
                m = _GITHUB_URL.search(str((r or {}).get("url") or ""))
                if m:
                    urls.append(m.group(0).rstrip("/"))
        except Exception:  # noqa: BLE001 -- candidate proposal is best-effort
            pass
    if explicit:
        urls.append(explicit)
    urls += _urls_from_literature(literature_brief)
    seen: set[str] = set()
    out: list[Benchmark] = []
    for u in urls:
        k = _norm_repo_url(u)
        if k and k not in seen:
            seen.add(k)
            status = _public_repo_status(u)
            logging.getLogger(__name__).info("repository candidate availability: %s status=%s", u, status)
            if status == 200:
                out.append(_stub_for_url(u, request))
        if len(out) >= max_candidates:
            break
    return out


_OVERRIDE_KEYS = ("editable_files", "protected_files", "metric", "lower_is_better",
                  "prepare_cmd", "run_cmd", "budget_s")


# Source-code extensions used by the PERMISSIVE fallback: when a clean editable/protected boundary
# can't be inferred, we treat the repo's own code files as editable rather than refusing the task
# (meeting 2026-07-25 item 4). Data/weights/configs are deliberately excluded so a fallback still
# points the agent at code, not at a dataset dump.
_CODE_EXTS = (".py", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hh", ".java",
              ".go", ".rs", ".js", ".ts", ".jsx", ".tsx", ".cs", ".rb", ".jl", ".m", ".mm",
              ".swift", ".scala", ".kt", ".sh")


def _all_editable_code_files(pristine_dir: str, protected: tuple, max_files: int = 60) -> tuple:
    """PERMISSIVE fallback boundary: every source-code file that EXISTS in the clone (by extension),
    minus anything already flagged protected, shallowest-first, capped. Used only when strict
    boundary inference found no editable file -- degrade to 'the whole repo's code is editable'
    rather than refuse the task."""
    prot = set(protected or ())
    found: list[str] = []
    for root, _dirs, fs in os.walk(pristine_dir):
        if os.sep + ".git" in root:
            continue
        for f in fs:
            if not f.endswith(_CODE_EXTS):
                continue
            rel = os.path.relpath(os.path.join(root, f), pristine_dir)
            if rel not in prot:
                found.append(rel)
    found.sort(key=lambda r: (r.count(os.sep), r))  # shallow files first
    return tuple(found[:max_files])


_ENTRYPOINT_FIX_SYSTEM = (
    "A cloned repo's proposed run_cmd names a script file that does NOT exist in the repo -- it was "
    "guessed from another project's convention. Rewrite it to use the repo's ACTUAL entrypoint, taken "
    "from the repo's own run script (run.sh / run_docker.sh) and the top-level scripts that really "
    "exist. Keep the user's goal (short single run). If the real entrypoint needs data downloaded "
    "first (a data script is present but no data files), put that download in prepare_cmd. Use the "
    "REAL launcher form the repo uses (e.g. `torchrun --standalone --nproc_per_node=1 <script>` if "
    "run.sh uses torchrun; scale nproc down to 1 GPU). Chain shell steps with &&; NEVER interactive "
    "docker flags. Reply with ONE JSON object: {\"run_cmd\": \"...\", \"prepare_cmd\": \"...\"}. Only "
    "name scripts that appear in the provided file list."
)


def infer_benchmark(pristine_dir: str, request: str, stub: Benchmark, llm,
                    overrides: dict | None = None, permissive: bool = False) -> Benchmark | None:
    """Infer the trusted-eval boundary of a freshly-cloned repo (which files are editable vs
    protected, the metric, run commands, budget) from its files + docs, and return a complete
    Benchmark. ``overrides`` are USER-CONFIRMED boundary fields (run_cmd/metric/budget/... the
    user knows better than the LLM's guess -- e.g. makemore's ``-i names.txt``, a max-steps
    budget) that WIN over inference; a COMPLETE override works even with no LLM. None if no
    editable file survives.

    ``permissive`` (meeting 2026-07-25 item 4): when the strict inference yields no editable file,
    DON'T refuse the task -- fall back to treating the repo's own code files as editable (the
    run_cmd/metric the LLM did infer are kept; if they're empty the run's self-heal loop re-infers
    them from real output). Only when the repo has no code file at all does it still return None."""
    overrides = {k: v for k, v in (overrides or {}).items() if k in _OVERRIDE_KEYS and v not in (None, "", [])}
    if llm is None and not overrides:
        return None
    files: list[str] = []
    for root, _dirs, fs in os.walk(pristine_dir):
        if os.sep + ".git" in root:
            continue
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), pristine_dir)
            files.append(rel)
    files = sorted(files)[:250]
    docs = ""
    for cand in ("program.md", "README.md", "readme.md", "README.rst", "README"):
        p = os.path.join(pristine_dir, cand)
        if os.path.exists(p):
            try:
                docs += f"\n--- {cand} ---\n" + open(p, encoding="utf-8", errors="ignore").read()[:5000]
            except OSError:
                pass

    obj: dict = {}
    if llm is not None:
        user = (f"User goal: {request}\n\nRepo files:\n" + "\n".join(files)
                + f"\n\nKey docs:\n{docs[:9000]}")
        try:
            result = llm.chat(
                messages=[{"role": "system", "content": _INFER_SYSTEM},
                          {"role": "user", "content": user}],
                response_format="json_object")
            parsed = json.loads(extract_json(result.content or "{}"))
            obj = parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001
            obj = {}

        # --- validate + correct: read the ACTUAL entry scripts and let the LLM fix run_cmd /
        #     budget / entrypoint (catches e.g. a budget that ignores cold-start + repeated
        #     runs). Fail-SOFT: any failure keeps the first-pass values. ---
        scripts = _read_scripts(pristine_dir, _candidate_scripts(obj, files))
        setup = _read_setup_files(pristine_dir, files)
        if obj and (scripts.strip() or setup.strip()):
            try:
                cres = llm.chat(
                    messages=[{"role": "system", "content": _CORRECT_SYSTEM},
                              {"role": "user", "content": f"User goal: {request}\n\nProposed "
                               f"boundary:\n{json.dumps(obj, ensure_ascii=False)}\n\nActual script "
                               f"content:\n{scripts}\n\nInstall/build manifests (derive prepare_cmd "
                               f"from THESE):\n{setup}\n\nKey docs:\n{docs[:4000]}"}],
                    response_format="json_object")
                cobj = json.loads(extract_json(cres.content or "{}"))
                if isinstance(cobj, dict):
                    obj = {**obj, **{k: v for k, v in cobj.items() if v not in (None, "", [])}}
            except Exception:  # noqa: BLE001 -- correction is best-effort; keep first-pass values
                pass

    # --- (meeting 2026-07-26 A+B) deterministic ENTRYPOINT guard. If the proposed run_cmd names
    #     script file(s) that don't exist in the clone, it was hallucinated from another repo's
    #     convention (e.g. nanoGPT's `python train.py config/..` onto modded-nanogpt, which has only
    #     `train_gpt.py` + `run.sh`) and can only fail. Re-derive the command from the repo's REAL
    #     entrypoint (run.sh + top-level scripts) BEFORE wasting an expensive run. Skipped when the
    #     user overrode run_cmd (their word wins) or when a referenced script already exists. ---
    if llm is not None and obj.get("run_cmd") and not overrides.get("run_cmd"):
        toks = _cmd_script_tokens(str(obj["run_cmd"]))
        if toks and not any(os.path.exists(os.path.join(pristine_dir, t)) for t in toks):
            top_scripts = [f for f in files if os.sep not in f and f.endswith((".py", ".sh"))]
            real_entry = _read_scripts(pristine_dir, top_scripts[:8], cap_total=6000)
            run_scripts = _read_setup_files(
                pristine_dir, [f for f in files if f.rsplit("/", 1)[-1].lower()
                               in ("run.sh", "run_docker.sh", "entrypoint.sh")])
            try:
                fres = llm.chat(
                    messages=[{"role": "system", "content": _ENTRYPOINT_FIX_SYSTEM},
                              {"role": "user", "content":
                               f"User goal: {request}\n\nProposed run_cmd names non-existent "
                               f"script(s): {toks}\n  run_cmd: {obj['run_cmd']}\n\nRepo run script "
                               f"(run.sh/...):\n{run_scripts}\n\nTop-level scripts that EXIST:\n"
                               f"{real_entry}\n\nFile list:\n" + "\n".join(top_scripts[:40])}],
                    response_format="json_object")
                fobj = json.loads(extract_json(fres.content or "{}"))
                if isinstance(fobj, dict):
                    obj = {**obj, **{k: v for k, v in fobj.items()
                                     if k in ("run_cmd", "prepare_cmd") and str(v).strip()}}
            except Exception:  # noqa: BLE001 -- best-effort; keep first-pass command if the fix fails
                pass

    # USER OVERRIDES WIN over inference (the user confirmed/corrected the boundary).
    obj = {**obj, **overrides}

    # The runner already runs prepare_cmd/run_cmd with cwd=the variant workspace, but LLMs often
    # prefix an absolute `cd /repo && ...` (copied from a Dockerfile) that points OUTSIDE the
    # workspace and would break the run. Strip a leading `cd <path> &&`.
    for _k in ("prepare_cmd", "run_cmd"):
        if obj.get(_k):
            obj[_k] = _strip_leading_cd(str(obj[_k]))

    editable = tuple(str(x).strip() for x in (obj.get("editable_files") or []) if str(x).strip())
    protected = tuple(str(x).strip() for x in (obj.get("protected_files") or []) if str(x).strip())
    # deterministic guard: keep only files that ACTUALLY exist in the clone -- a run against a
    # guessed-but-absent editable file can only fail. Fail-CLOSED if no editable file survives
    # (a broken boundary is better caught here than as a timeout/crash on the GPU host).
    def _exists(rel: str) -> bool:
        return os.path.exists(os.path.join(pristine_dir, rel))
    editable = tuple(f for f in editable if _exists(f))
    protected = tuple(f for f in protected if _exists(f))
    if not editable and permissive:
        # meeting item 4: degrade instead of refusing -- the repo's own code is the editable set.
        editable = _all_editable_code_files(pristine_dir, protected)
    if not editable:
        return None
    if type(obj.get("lower_is_better", True)) is not bool:
        return None
    # A repo that needs its OWN image: keep an existing-tag as-is; for a ``build:<Dockerfile>``
    # spec, only trust it when that Dockerfile actually exists in the clone (else drop to the
    # default sandbox rather than fail on a hallucinated path).
    image = str(obj.get("image") or "").strip() or None
    if image and image.startswith("build:"):
        df = image[len("build:"):].strip()
        if not (df and _exists(df)):
            image = None
    return Benchmark(
        name=stub.name, description=stub.description, repo_url=stub.repo_url,
        editable_files=editable, protected_files=protected,
        prepare_cmd=(str(obj.get("prepare_cmd") or "").strip() or "true"),
        run_cmd=str(obj.get("run_cmd") or "").strip(),
        metric=(str(obj.get("metric") or "metric").strip() or "metric"),
        lower_is_better=obj.get("lower_is_better", True),
        budget_s=int(obj.get("budget_s") or 300),
        # only a seed hook the script REALLY reads makes reproduce vary the seed; else None ->
        # reproduce is an honest same-seed rerun (run-to-run variance), never a faked seed swap.
        seed_env=(str(obj.get("seed_env") or "").strip() or None),
        image=image,
        keywords=())


def discover(request: str, llm=None) -> Benchmark | None:
    """Return a ``Benchmark`` stub for ``request`` by dynamically deciding whether it targets a
    real repo to clone + infer, or None. There is no static registry: the system clones ANY repo
    the user names. Needs an ``llm`` unless the request contains an explicit GitHub URL/owner/repo
    (that fast path is deterministic); a general algorithmic/data question returns None."""
    return _discover_dynamic(request, llm)


_CORRECT_RUN_SYSTEM = (
    "A cloned-repo run FAILED or produced no PARSEABLE metric. You are given the run's ACTUAL error, "
    "the run's ACTUAL output (what it really printed), the repo's setup/build files + entry scripts + "
    "docs, and the CURRENT execution config. Correct the EXECUTION side so the run works AND its metric "
    "is parseable. This is ONE general self-heal -- diagnose whatever is wrong from the REAL evidence, "
    "not a fixed list of failure types:\n"
    "- prepare_cmd / run_cmd (fix from the ERROR): a missing DATA file (FileNotFoundError) -> add a "
    "download/prepare step from the repo's own data script or a real URL; ModuleNotFoundError / "
    "'command not found' (incl. `pip` absent when the env uses `uv`) -> install/route through the "
    "right env (uv sync / uv run ... / pip install -e . / -r requirements.txt); a native/CUDA build "
    "error or 'Docker required' -> use the repo's docker path (build + a NON-INTERACTIVE `docker run`, "
    "NEVER -it/-t/-i). Read run_docker.sh/Dockerfile/entrypoint for the real image/mount/CLI form.\n"
    "- metric (fix from the OUTPUT): if the run PRINTED OUTPUT but the metric wasn't found in it, read "
    "the ACTUAL output and set metric to the name/dotted-path that IS present -- a plain name for a "
    "`name: value` line, or a DOTTED PATH into the JSON (e.g. 'evaluation.performance.latency_ms') when "
    "the number is nested. It MUST occur in the shown output; do not invent one. Set lower_is_better "
    "for that metric w.r.t. the goal. If the run never produced output (setup still broken), keep the "
    "current metric.\n"
    "- budget_s: raise it only if the failure is a wall-clock timeout.\n"
    "NEVER change editable_files or protected_files -- the trusted-eval boundary is FIXED (only HOW to "
    "run and HOW to read the score may change). Chain shell steps with &&. Reply with ONE JSON object: "
    "{\"prepare_cmd\": \"...\", \"run_cmd\": \"...\", \"metric\": \"<name or dotted path>\", "
    "\"lower_is_better\": <bool>, \"budget_s\": <int>, \"diagnosis\": \"<one line: root cause + fix>\"}."
)


def correct_execution_from_run(pristine_dir: str, benchmark: Benchmark, error: str, output: str,
                               request: str, llm) -> Benchmark | None:
    """ONE general self-heal: a run FAILED or produced no parseable metric -> re-infer the EXECUTION
    side (prepare_cmd/run_cmd/metric/lower_is_better/budget_s) from the run's ACTUAL error + output +
    the repo's own files, and return a NEW Benchmark with only those fields changed. The trusted-eval
    boundary (editable_files/protected_files) is NEVER touched, so anti-cheat is preserved.

    This replaces having a separate repair_setup + repair_metric + ...: whatever is wrong (a missing
    dep, pip-vs-uv, a nested metric path the first-pass guess missed, a too-small budget) is corrected
    by the SAME mechanism reasoning from the repo's REAL run evidence -- no per-failure-type taxonomy,
    no per-repo hard-coding. It closes the infer loop with reality: infer -> run -> observe -> re-infer.
    Returns None (no change) when the LLM proposes nothing new.
    """
    if llm is None:
        return None
    files: list[str] = []
    for root, _dirs, fs in os.walk(pristine_dir):
        if os.sep + ".git" in root:
            continue
        for f in fs:
            files.append(os.path.relpath(os.path.join(root, f), pristine_dir))
    files = sorted(files)[:250]
    setup = _read_setup_files(pristine_dir, files)
    scripts = _read_scripts(pristine_dir, _candidate_scripts(
        {"editable_files": list(benchmark.editable_files)}, files), cap_total=8000)
    docs = ""
    for cand in ("README.md", "readme.md", "INSTALL.md", "README"):
        p = os.path.join(pristine_dir, cand)
        if os.path.exists(p):
            try:
                docs += "\n--- " + cand + " ---\n" + open(p, encoding="utf-8", errors="ignore").read()[:5000]
            except OSError:
                pass
            break
    user = (f"User goal: {request}\n\nRun ERROR (tail):\n{(error or '')[-1600:]}\n\n"
            f"Run ACTUAL OUTPUT (tail):\n{(output or '')[-2500:]}\n\n"
            f"Current prepare_cmd: {benchmark.prepare_cmd}\nCurrent run_cmd: {benchmark.run_cmd}\n"
            f"Current metric: {benchmark.metric}\n\nSetup/build files:\n{setup}\n\nEntry scripts:\n"
            f"{scripts}\n\nKey docs:\n{docs[:3500]}")
    try:
        res = llm.chat(messages=[{"role": "system", "content": _CORRECT_RUN_SYSTEM},
                                 {"role": "user", "content": user}], response_format="json_object")
        obj = json.loads(extract_json(res.content or "{}"))
    except Exception:  # noqa: BLE001 -- self-heal is best-effort; a bad LLM reply -> no change
        return None
    if not isinstance(obj, dict):
        return None
    # drop any interactive docker flags the model may still emit (a subprocess has no TTY)
    if type(obj.get("lower_is_better", benchmark.lower_is_better)) is not bool:
        return None
    def _strip_interactive(cmd: str) -> str:
        return re.sub(r"(docker run)\s+(-it|-ti|-i|-t)\b", r"\1", cmd)
    new_prepare = _strip_interactive(_strip_leading_cd(str(obj.get("prepare_cmd") or "").strip()))
    new_run = _strip_interactive(_strip_leading_cd(str(obj.get("run_cmd") or "").strip()))
    new_metric = str(obj.get("metric") or "").strip()
    import dataclasses
    corrected = dataclasses.replace(
        benchmark,
        prepare_cmd=new_prepare or benchmark.prepare_cmd,
        run_cmd=new_run or benchmark.run_cmd,
        metric=new_metric or benchmark.metric,
        lower_is_better=obj.get("lower_is_better", benchmark.lower_is_better),
        budget_s=int(obj.get("budget_s") or benchmark.budget_s))
    # No-op guard: if nothing on the execution side actually changed, report None so the caller's
    # bounded loop stops instead of spinning on an identical proposal.
    if (corrected.prepare_cmd == benchmark.prepare_cmd and corrected.run_cmd == benchmark.run_cmd
            and corrected.metric == benchmark.metric and corrected.budget_s == benchmark.budget_s
            and corrected.lower_is_better == benchmark.lower_is_better):
        return None
    return corrected


__all__ = ["discover", "discover_candidates", "infer_benchmark", "correct_execution_from_run"]
