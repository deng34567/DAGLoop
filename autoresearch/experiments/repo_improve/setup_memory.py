"""Remember a setup that tier 3 discovered, so the next run doesn't re-discover it.
把层3 摸出来的安装固化下来,下次同仓库不必再花二十分钟重摸。

Discovering nanochat's setup costs a GPU sandbox, an agent loop and ~20 minutes. The
answer is stable -- the repo still needs its tokenizer trained tomorrow -- so paying
that price on every run is pure waste.

Three disciplines, from what memory gets WRONG when done naively:

* **Stale.** A repo moves and the remembered commands rot. So a recall is a HINT, not
  a truth: the caller re-runs it (cheap, one baseline) and falls back to the ladder if
  it no longer works. Nothing here asserts a recalled setup still runs.
* **Wrong entries poison every later run.** Only setups that passed the hard judge are
  admitted -- an entry must carry the metric value actually observed, so a later reader
  can see what it is trusting rather than a bare command string.
* **Over-generalisation.** The micro-config tier 3 settles on is tuned to one GPU and
  one wall-clock budget; ``--depth=4`` inside 300s on an L20 says nothing about 600s on
  an H100. Recall therefore matches on (repo, budget, gpu), not on the repo alone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# #41: the setup memory is keyed by (repo_url, budget_s, gpu) and lives in a SHARED ~/.cache, so an
# entry recorded by one session/deployment is recalled by another. But an agent-discovered command
# often embeds the ABSOLUTE path of the session workspace it ran in (e.g.
# ``cd /data/.../ar-web-public/repo_improve_local/<uuid>/_sandbox/repo`` or ``REPO=/data/.../
# _sandbox/repo && cd "$REPO/..."``). Recalled in a DIFFERENT session (different uuid/deployment),
# that path doesn't exist -> the run crashes -> baseline=None. The runner always runs with cwd = the
# current repo dir, so any absolute path to a ``.../_sandbox/repo`` is the (stale) repo dir: rewrite
# it to ``$PWD`` so the setup is portable across sessions/deployments. (``_strip_leading_cd`` in
# discover.py only handled a *leading* ``cd /abs &&`` on the inference path, not these forms.)
_WORKSPACE_REPO_RE = re.compile(r'/[^\s"\'`;|&()]*/_sandbox/repo(?=[/"\'`\s;|&)]|$)')


def _portablize(cmd):
    """Rewrite absolute session-workspace repo paths (.../_sandbox/repo) to ``$PWD``. Idempotent."""
    if not cmd:
        return cmd
    return _WORKSPACE_REPO_RE.sub('"$PWD"', str(cmd))

DEFAULT_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch", "setup_memory.json"
)


class SetupMemory:
    """A tiny JSON-backed store keyed by (repo_url, budget_s, gpu).

    Deliberately a plain file: this is a cache of reproducible facts, and a corrupt or
    missing one must degrade to "no memory" rather than take a research run down.
    """

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}   # missing or corrupt -> behave as if we never knew anything

    def _save(self, data: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)   # atomic: a crash mid-write can't corrupt the store
        except OSError:
            logger.warning("could not persist setup memory to %s", self.path, exc_info=True)


def _key(repo_url: str, budget_s, gpu) -> str:
    """Context-scoped key -- a setup is only reusable under the conditions it was tuned for."""
    return f"{(repo_url or '').strip()}|{budget_s}|{(gpu or '').strip()}"


def setup_image_tag(repo_url: str, budget_s, gpu) -> str:
    """Deterministic docker image tag for a frozen setup: same context -> same tag, so a
    re-discovery overwrites its own image rather than leaking a new one each run."""
    digest = hashlib.sha1(_key(repo_url, budget_s, gpu).encode("utf-8")).hexdigest()[:12]
    return f"autoresearch-setup-{digest}"


def remember_setup(memory: SetupMemory, *, repo_url: str, prepare_cmd: str, run_cmd: str,
                   metric: str, lower_is_better: bool, metric_value, budget_s, gpu,
                   verified_in: str = "sandbox", image: str | None = None) -> bool:
    """Record a VERIFIED setup. Returns False (and stores nothing) when it isn't verified.

    ``metric_value`` is the number our hard judge actually observed; without it there is
    no evidence the setup works, so the entry is refused rather than saved as a guess.

    ``verified_in`` records the ENVIRONMENT the setup was proved in ("sandbox" or "host").
    It matters on recall: a setup verified in a sandbox uses container-only paths (a
    ``.venv`` the host lacks, a JIT the host can't run), so re-trying it on the host first
    is guaranteed to fail and waste the repair ladder -- the caller must reopen the sandbox
    and run it there instead.

    ``image`` is the frozen (``docker commit``-ed) image of the verified container. With it,
    a recall reruns in the SAME environment and is reliable; without it, the two commands
    alone often can't rebuild what the agent installed at the container level, so the recall
    goes stale and re-discovers. ``None`` = no frozen image (commit failed / host setup).
    """
    if metric_value is None or not str(run_cmd or "").strip():
        return False
    data = memory._load()
    data[_key(repo_url, budget_s, gpu)] = {
        "repo_url": repo_url,
        # #41: portablize absolute session-workspace paths so this entry is reusable in ANOTHER
        # session/deployment (the store is a shared ~/.cache keyed without the workspace).
        "prepare_cmd": _portablize(prepare_cmd),
        "run_cmd": _portablize(run_cmd),
        "metric": metric,
        "lower_is_better": bool(lower_is_better),
        "verified_in": verified_in,
        "image": image,
        "context": {"budget_s": budget_s, "gpu": gpu},
        "evidence": {"metric_value": metric_value},
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    memory._save(data)
    return True


def recall_setup(memory: SetupMemory, *, repo_url: str, budget_s, gpu):
    """Return a previously verified setup for this exact context, or ``None``.

    A HINT only: the caller must still run it and fall back to the healing ladder if the
    repo has moved on since. Never raises.
    """
    hit = memory._load().get(_key(repo_url, budget_s, gpu))
    if isinstance(hit, dict):
        # #41: clean legacy entries written before the fix (absolute session-workspace paths on
        # disk) at read time, so an old shared ~/.cache doesn't keep crashing every recall.
        hit = {**hit, "prepare_cmd": _portablize(hit.get("prepare_cmd")),
               "run_cmd": _portablize(hit.get("run_cmd"))}
    return hit


__all__ = ["SetupMemory", "DEFAULT_PATH", "remember_setup", "recall_setup", "setup_image_tag"]
