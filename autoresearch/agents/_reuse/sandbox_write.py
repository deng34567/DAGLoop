"""Write a from-scratch experiment inside an OpenHands sandbox (GEN path, meeting 2026-07-25 item 6).

Mirrors ``sandbox_setup.discover_setup_in_sandbox``, but instead of discovering how to RUN an
existing repo, it has the in-container agent WRITE the experiment implementation for a GEN
(no-repo) task -- the paper-reproduction path. The files the agent writes are exported and
assembled into an ``experiment_patch`` (the same contract ``CoderAgent`` produces from a single
LLM call today), so the rest of the pipeline (RunnerAgent + frozen VerifierSpec) is unchanged.

STRUCTURAL ANTI-CHEAT: the benchmark's protected/forbidden files (evaluator + data) are NEVER
included in the exported patch -- so an agent that writes its own scoring code can't reach the
trusted measurement, exactly as in the repo path. Best-effort: any infra failure returns None so
the caller degrades to an honest failure (or the existing single-LLM path), never a fake result.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _write_task_prompt(plan: dict, verifier_spec: dict, budget_s: int) -> str:
    """Task handed to the in-container agent: implement the plan as runnable code that prints the
    metric. editable_region is a LOCATION HINT, not a hard whitelist (item 4d)."""
    editable = verifier_spec.get("editable_region", []) or []
    metric = (verifier_spec.get("metric") or verifier_spec.get("metric_name")
              or "the target metric")
    summary = plan.get("change_summary") or plan.get("summary") or plan.get("approach") or ""
    changes = plan.get("changes", []) or []
    change_lines = "\n".join(
        f"- {c.get('file', '?')}: {c.get('description', '')}" for c in changes) or "(none listed)"
    return (
        "Implement the following experiment as runnable code under /workspace.\n\n"
        f"Goal / plan:\n{summary}\n\nIntended changes:\n{change_lines}\n\n"
        f"HARD REQUIREMENTS:\n"
        f"- FAIR WALL-CLOCK BUDGET: the code is scored under a HARD {budget_s}s wall-clock cap and "
        f"is KILLED at {budget_s}s -- training LONGER does not help, it is cut off. The code MUST "
        f"run to completion and PRINT the metric named '{metric}' WITHIN {budget_s}s, on its own "
        f"line, parseable (e.g. `{metric}=<number>`). When you test it yourself, wrap it as "
        f"`timeout {budget_s}s <cmd>`; a run exceeding {budget_s}s is killed and wasted.\n"
        f"- Likely implementation location(s) (a starting HINT, not a hard boundary): {editable}.\n"
        "- Do NOT write or modify any evaluation / scoring / data-preparation file: those are "
        "supplied by a trusted harness and ANY change you make to them is discarded before "
        "scoring. Write only the method/training/model code.\n"
        "- After writing, run the code once yourself to confirm it executes and prints the metric."
    )


def _protected_rels(verifier_spec: dict) -> set[str]:
    """Files the exported patch must never carry (the trusted evaluator + data)."""
    out: set[str] = set()
    for key in ("protected_files", "forbidden_files"):
        for f in (verifier_spec.get(key) or []):
            rel = str(f).strip().lstrip("./")
            if rel:
                out.add(rel)
    return out


def _collect_written_files(root: str, protected: set[str], *, max_bytes: int = 400_000) -> list[dict]:
    """Read the agent-written text files under ``root`` into experiment_patch ``changes`` entries,
    skipping protected/evaluator files (anti-cheat), version-control/cache noise, and binaries."""
    changes: list[dict] = []
    rootp = Path(root)
    for p in sorted(rootp.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(rootp))
        parts = set(rel.split(os.sep))
        if rel in protected or ".git" in parts or "__pycache__" in parts:
            continue
        if p.name == "ws_export.tgz":
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
            content = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary / unreadable -> not a code change we carry
        changes.append({"file": rel, "patch": content})
    return changes


def write_experiment_in_sandbox(
    *,
    plan: dict,
    verifier_spec: dict,
    workroot: str,
    request: str = "",
    backend: Any = None,
    cuda_device: Any = None,
    sandbox_llm: Any = None,
    budget_s: int = 1800,
    emit: Any = None,
    server_image: str | None = None,
) -> dict | None:
    """Open a sandbox, have the in-container agent write the experiment, export what it wrote, and
    return an ``experiment_patch`` dict ({summary, changes:[{file,patch}], touched_files}). Returns
    None on any infra failure or when the agent wrote nothing usable (honest degrade).

    ``sandbox_llm`` MUST be an OpenHands-SDK LLM or None (None lets ``run_agent_task`` build the
    right one from env) -- never our own client wrapper, which blows up inside the SDK agent.
    """
    emit = emit or (lambda event, payload: None)
    from autoresearch.agents._reuse.openhands_backend import InfraError
    if backend is None:
        from autoresearch.agents._reuse.openhands_backend import (
            DEFAULT_SERVER_IMAGE, OpenHandsBackend)
        backend = OpenHandsBackend(server_image=server_image or DEFAULT_SERVER_IMAGE)

    handle = None
    dest = os.path.join(workroot, "gen_written")
    try:
        handle = backend.open_sandbox(cuda_device=cuda_device, mounts=[workroot])
        task = _write_task_prompt(plan, verifier_spec, budget_s)
        emit("gen_sandbox_write", {"stage": "running", "budget_s": budget_s})
        res = backend.run_agent_task(handle, task, llm=sandbox_llm, timeout=budget_s)
        status = getattr(res, "status", "") or ""
        # export even on a non-FINISHED status: the agent may have written usable files before
        # timing out; scoring the partial work honestly beats discarding it.
        backend.export_workspace(handle, dest, src="/workspace")
        emit("gen_sandbox_write", {"stage": "exported", "status": status})
    except InfraError as exc:
        emit("gen_sandbox_write", {"stage": "infra_error", "error": str(exc)})
        return None
    except Exception as exc:  # noqa: BLE001 -- sandbox write is best-effort; never take the run down
        emit("gen_sandbox_write", {"stage": "error", "error": f"{type(exc).__name__}: {exc}"})
        return None
    finally:
        if handle is not None:
            try:
                backend.close_sandbox(handle)
            except Exception:  # noqa: BLE001
                pass

    changes = _collect_written_files(dest, _protected_rels(verifier_spec))
    if not changes:
        emit("gen_sandbox_write", {"stage": "empty"})
        return None
    summary = (plan.get("change_summary") or plan.get("summary")
               or plan.get("approach") or "sandbox-written experiment")
    return {
        "summary": str(summary)[:2000],
        "changes": changes,
        "touched_files": [c["file"] for c in changes],
    }


__all__ = ["write_experiment_in_sandbox"]
