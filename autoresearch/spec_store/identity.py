"""Stable content identities for mutable deterministic SpecStore refs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def runtime_identity(modules=None):
    """Separate on-disk source identity from loaded Python function code identity.

    This inventory is opt-in evidence, not proof of unobserved external workers,
    dependency versions, model identity, or all mutable interpreter state.
    """
    import inspect
    import marshal
    import os
    from pathlib import Path
    import sys
    inventory = {}
    for name, module in sorted((modules if modules is not None else dict(sys.modules)).items()):
        if (not name.startswith("autoresearch.") or ".tests" in name
                or name.startswith("autoresearch.testing") or module is None):
            continue
        source = getattr(module, "__file__", None)
        if not source or not source.endswith(".py"):
            continue
        codes = {}
        for member_name, member in vars(module).copy().items():
            if getattr(member, "__module__", None) != name:
                continue
            members = vars(member).items() if inspect.isclass(member) else [(member_name, member)]
            for child_name, child in members:
                child = getattr(child, "__func__", child)
                if inspect.isfunction(child):
                    codes[member_name + "." + child_name] = hashlib.sha256(marshal.dumps(child.__code__)).hexdigest()
        try:
            source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except OSError:
            source_hash = None
        inventory[name] = {"source_path": source, "source_sha256": source_hash,
                           "loaded_functions_sha256": artifact_digest(codes), "function_count": len(codes)}
    return {"python": sys.version, "process_id": os.getpid(), "modules": inventory,
            "scope": "loaded Python functions and separately observed on-disk source; external processes not inferred"}


def artifact_digest(obj: Any) -> str:
    """Return a deterministic SHA-256 identity for one JSON artifact value."""
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def result_for_run(store, run_record_ref):
    """Resolve a run's own result; a later mutable 'current' is not its evidence."""
    from . import refs
    parts = refs.decode(run_record_ref)
    if not parts.key.startswith("snapshot-"):
        try:
            return store.get(refs.encode("gen_experiment_result", parts.session_id, "current"))
        except KeyError:
            return store.get(refs.encode("gen_experiment_result", parts.session_id, parts.key))
    for ref in store.list_refs(parts.session_id, "gen_experiment_result"):
        item = store.get(ref)
        if (item.get("run_record_snapshot_ref") == run_record_ref
                and refs.decode(ref).key == "snapshot-" + artifact_digest(item).split(":", 1)[1]):
            return {**item, "candidate_snapshot_ref": ref}
    raise KeyError("no immutable result is bound to this run snapshot")


__all__ = ["artifact_digest", "result_for_run"]
