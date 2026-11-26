"""Meeting 2026-07-25 #3 (C row): a lightweight 3-agent discussion of HOW to implement the
confirmed method, BEFORE the in-container coding agent starts editing.

The professor's C row is "confirm method -> DISCUSS how to implement -> implement -> review". The
implementation (OpenHands agent) and the boundary removal (T1) are already done; this fills the
missing "discuss how to implement" step. Three lightweight LLM calls each look at the SAME
improvement task from a DISTINCT angle -- highest-impact lever, simplest robust change, pitfalls to
avoid -- and a short synthesis fuses them into concise, actionable guidance. That guidance is handed
to the coding agent as ADVICE (appended to its task), so the agent starts from a considered plan
instead of a cold prompt.

Bounded + best-effort by construction: at most 4 short LLM calls, and ANY failure (no llm, a bad
reply, a timeout) returns "" -- the caller then proceeds exactly as before, so this can only help,
never block or crash a run.
"""
from __future__ import annotations

#: The three discussion angles. Each is a distinct lens on the SAME task so the perspectives don't
#: collapse into one -- impact (what moves the metric most), simplicity (what is safe + likely to
#: work), and risk (what must not break). Kept to three to stay lightweight (the professor's "轻量").
_ANGLES = (
    ("impact",
     "You are the IMPACT reviewer. Name the ONE or TWO changes most likely to move the metric the "
     "most for this repo/method, and say briefly why. Be concrete about where in the code they go."),
    ("simplicity",
     "You are the SIMPLICITY reviewer. Name the simplest change that is very likely to run correctly "
     "and help at least a little -- prefer a small, low-risk edit over an ambitious rewrite."),
    ("pitfalls",
     "You are the RISK reviewer. Name the top pitfalls for this task: what would silently break the "
     "run, invalidate the metric, or fail to reproduce, and how to avoid each."),
)

_SYNTH_SYSTEM = (
    "You synthesize three short implementation reviews (impact / simplicity / risk) into ONE concise, "
    "actionable implementation plan for a coding agent about to edit the repo. Output 3-6 short "
    "bullet points: the concrete change(s) to try first, kept simple, and the pitfalls to avoid. No "
    "preamble, no code -- just the bullets the agent should follow."
)


def _task_brief(benchmark, goal: str, baseline_metric) -> str:
    direction = "lower" if benchmark.lower_is_better else "raise"
    return (
        f"Goal: {goal}\nRepo: {benchmark.repo_url}\n"
        f"Metric to {direction}: {benchmark.metric} (baseline = {baseline_metric}).\n"
        f"The repo is run with: {benchmark.run_cmd}\n"
        f"The implementation most likely lives in: {', '.join(benchmark.editable_files) or '(unknown)'}."
    )


def discuss_implementation(llm, benchmark, goal: str, baseline_metric, *, emit=None) -> str:
    """Run the 3-angle implementation discussion + synthesis and return concise guidance for the
    coding agent. Returns "" on any failure (no llm / bad reply / timeout) so the caller degrades to
    its prior cold-start behavior. Never raises."""
    emit = emit or (lambda event, payload: None)
    if llm is None:
        return ""
    brief = _task_brief(benchmark, goal, baseline_metric)
    opinions: list[str] = []
    for name, lens in _ANGLES:
        try:
            res = llm.chat(messages=[
                {"role": "system", "content": lens + " Answer in 2-4 short sentences, no code."},
                {"role": "user", "content": brief},
            ])
            text = (getattr(res, "content", "") or "").strip()
            if text:
                opinions.append(f"[{name}] {text}")
        except Exception:  # noqa: BLE001 -- one angle failing must not sink the discussion
            emit("impl_discussion", {"ok": False, "angle": name})
            continue
    if not opinions:
        return ""
    try:
        syn = llm.chat(messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": brief + "\n\nThree reviews:\n" + "\n\n".join(opinions)},
        ])
        guidance = (getattr(syn, "content", "") or "").strip()
    except Exception:  # noqa: BLE001 -- synthesis failed; fall back to the raw opinions (still useful)
        guidance = "\n".join(opinions)
    emit("impl_discussion", {"ok": True, "angles": len(opinions), "chars": len(guidance)})
    return guidance


__all__ = ["discuss_implementation"]
