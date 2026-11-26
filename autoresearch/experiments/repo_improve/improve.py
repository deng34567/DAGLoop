"""The improve step: the agent proposes a CONSTRAINED, MULTI-FILE patch to the repo.

This is AutoResearch, not a training-script tuner: the agent decides WHAT to change across the
repo's editable files to better address the research goal -- it is NOT limited to a single
`train.py`, nor to "reduce validation loss". It returns a small list of targeted find/replace
edits, each naming a file; each ``find`` must match that file EXACTLY ONCE or the edit is
rejected. This bounds the blast radius (you see precisely what changed across which files) and
makes an invalid/hallucinated edit fail loudly instead of silently corrupting unrelated code.

Anti-cheat is enforced structurally elsewhere (``prepare_variant_workspace`` restores the
protected data + evaluator from the pristine clone before scoring), so the agent may edit any
file it is shown, but can never affect the trusted measurement.
"""
from __future__ import annotations

import json

from autoresearch.agents._reuse.json_extract import extract_json

_SYSTEM = (
    "You are an ML/research engineer improving an EXISTING code repository to better achieve a "
    "stated research goal, within a FIXED wall-clock budget. You are given the repo's own "
    "instructions, the goal + metric, and the current source of the files you MAY edit.\n\n"
    "Decide yourself WHAT to change (any of the shown files, one or several): hyperparameters, "
    "algorithm, model/architecture, data ordering, config -- whatever the goal calls for. Do NOT "
    "rewrite whole files; make PRINCIPLED, TARGETED edits. Do NOT edit the protected files "
    "(data + evaluator) -- they are restored from the original before scoring, so touching them "
    "is pointless. Every edit must be a genuine improvement, not a no-op.\n\n"
    "Respond with ONE JSON object: {\"rationale\": \"<why these help the goal>\", \"edits\": "
    "[{\"file\": \"<one of the shown files>\", \"find\": \"<exact unique substring of THAT "
    "file>\", \"replace\": \"<new text>\"}, ...]}. Each \"find\" MUST appear EXACTLY ONCE in its "
    "file (copy it verbatim, including whitespace). Keep the number of edits small."
)


class PatchError(Exception):
    """A proposed edit did not apply cleanly (unknown file, find missing or not unique)."""


def apply_edits(sources: dict[str, str], edits: list[dict]) -> dict[str, str]:
    """Apply file-scoped find/replace edits to ``sources`` (rel_path -> text). Returns the new
    sources dict (only changed files differ). Each ``find`` must occur exactly once IN ITS FILE;
    an edit naming a file not in ``sources`` (e.g. a protected file) is rejected. Raises
    PatchError."""
    out = dict(sources)
    for i, e in enumerate(edits or []):
        f = e.get("file")
        if f not in out:
            raise PatchError(f"edit #{i} targets an unknown or protected file: {f!r}")
        find = e.get("find", "")
        if not find:
            raise PatchError(f"edit #{i} has an empty 'find'")
        n = out[f].count(find)
        if n == 0:
            raise PatchError(f"edit #{i} 'find' not found in {f}: {find[:80]!r}")
        if n > 1:
            raise PatchError(f"edit #{i} 'find' is not unique ({n}x) in {f}: {find[:80]!r}")
        out[f] = out[f].replace(find, e.get("replace", ""))
    return out


def propose_patch(llm, benchmark, program_md: str, baseline_sources: dict[str, str],
                  baseline_metric: float | None, *, goal: str = "", advice: str = "",
                  max_attempts: int = 2) -> dict:
    """Ask the agent for a constrained MULTI-FILE patch and apply it. ``baseline_sources`` is the
    current source of every editable file (rel_path -> text). ``advice`` is an optional MOSS
    suggestion (what to change / how to fix a prior failure). Returns ``{"rationale", "edits",
    "patched_sources", "error"}``; ``patched_sources`` is rel_path -> new text for the changed
    files, or None if no patch applied cleanly after retries (the pipeline then treats the
    candidate as invalid -- honest, no silent fallback)."""
    files_blob = "\n\n".join(
        f"--- {rel} (editable) ---\n```\n{src}\n```" for rel, src in baseline_sources.items())
    protected = ", ".join(benchmark.protected_files) or "(none)"
    user = (f"Research goal: {goal or '(improve this repo)'}\n"
            f"Repo instructions ({benchmark.repo_url}):\n{program_md[:5000]}\n\n"
            f"Metric: {benchmark.metric} (lower is better={benchmark.lower_is_better}), "
            f"budget {benchmark.budget_s}s. Baseline {benchmark.metric} = {baseline_metric}.\n"
            f"Protected (do NOT edit, restored before scoring): {protected}\n"
            + (f"\nSupervisor (MOSS) advice: {advice}\n" if advice else "")
            + f"\nEditable files you may change:\n{files_blob}")
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
    last_err = "no attempt"
    for _ in range(max(1, max_attempts)):
        result = llm.chat(messages=messages, response_format="json_object")
        content = result.content or ""
        try:
            obj = json.loads(extract_json(content))
            edits = obj.get("edits") or []
            patched = apply_edits(baseline_sources, edits)
            # return only the files that actually changed
            changed = {rel: src for rel, src in patched.items()
                       if src != baseline_sources.get(rel)}
            if not changed:
                raise PatchError("no file was changed (all edits were no-ops)")
            return {"rationale": str(obj.get("rationale") or ""), "edits": edits,
                    "patched_sources": changed, "error": None}
        except (PatchError, json.JSONDecodeError, TypeError, KeyError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             f"That patch did not apply: {last_err}. Return corrected JSON; each "
                             "edit needs a valid 'file' and a 'find' copied verbatim that occurs "
                             "exactly once in that file."})
    return {"rationale": "", "edits": [], "patched_sources": None, "error": last_err}


__all__ = ["propose_patch", "apply_edits", "PatchError"]
