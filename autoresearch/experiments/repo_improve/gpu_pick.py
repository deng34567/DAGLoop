"""Pick a GPU to run on by PROBING the box, never by hard-coded card numbers.
靠探测挑一张真正空闲的卡,绝不写死卡号。

This is a shared machine: other people's jobs come and go, so "use card N" written
anywhere in the code is wrong within the hour. We ask the driver what is actually free
right now and take the roomiest idle card, and fall back to CPU when nothing qualifies
rather than fighting a neighbour's training for memory.
"""
from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

#: Operators can still pin a card explicitly; automatic selection is only the default.
ENV_OVERRIDE = "AUTORESEARCH_REPO_IMPROVE_CUDA"

#: Below this much free VRAM a training run is more likely to OOM than to finish.
DEFAULT_MIN_FREE_MIB = 8000

#: A card already computing is "free memory, busy silicon" -- taking it would just make
#: both runs slower and muddy any wall-clock-budgeted comparison.
BUSY_UTIL_PCT = 50


def _probe() -> list[dict]:
    """Ask the driver for per-card memory/utilisation. Raises if unavailable."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append({"index": parts[0], "used": int(parts[1]),
                         "total": int(parts[2]), "util": int(parts[3])})
        except ValueError:
            continue
    return rows


def _choose(rows, min_free_mib: int = DEFAULT_MIN_FREE_MIB):
    """Roomiest card that is both idle enough and has enough free VRAM; ``None`` -> CPU."""
    usable = [r for r in rows
              if r["util"] < BUSY_UTIL_PCT and (r["total"] - r["used"]) >= min_free_mib]
    if not usable:
        return None
    best = max(usable, key=lambda r: r["total"] - r["used"])
    return str(best["index"])


def pick_free_card(min_free_mib: int = DEFAULT_MIN_FREE_MIB):
    """Return a card index (str) to run on, or ``None`` to run on CPU.

    Never raises: a box with no driver, no nvidia-smi, or nothing free simply means CPU.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return override
    try:
        rows = _probe()
    except Exception as exc:  # noqa: BLE001 -- no GPU is a normal condition, not an error
        logger.info("GPU probe unavailable (%s); running on CPU", exc)
        return None
    choice = _choose(rows, min_free_mib=min_free_mib)
    if choice is None:
        logger.info("no card with >=%dMiB free and idle; running on CPU", min_free_mib)
    return choice


__all__ = ["pick_free_card", "ENV_OVERRIDE", "DEFAULT_MIN_FREE_MIB"]
