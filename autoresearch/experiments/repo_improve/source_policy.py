"""One declared source boundary for repository documents, plans and attribution."""
from pathlib import PurePosixPath
import re


def permitted(path, policy=None, *, directory=False):
    policy = policy or {}
    normalized = str(path).replace("\\", "/").strip("/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        return False
    def matches(pattern):
        return (normalized == pattern.rstrip("/")
                or normalized.startswith(pattern.rstrip("/") + "/")
                or PurePosixPath(normalized).match(pattern))
    if any(matches(p) for p in policy.get("forbidden", [])):
        return False
    if any(matches(p) for p in policy.get("allowed", [])):
        return True
    if directory and any(p.startswith(normalized + "/") for p in policy.get("allowed", [])):
        return True
    return True


def source_findings(method, contract):
    policy = ((contract or {}).get("execution_contract") or {}).get("sources") or {}
    sources = method.get("reference_sources") or []
    if isinstance(sources, list):
        # Method provenance can be stored as explicit source records. Read its
        # identifier without losing the original metadata or bypassing policy.
        normalized = []
        for source in sources:
            if isinstance(source, dict):
                key = "source_id" if "source_id" in source else "path"
                if (set(source) <= {key, "source_type", "usage", "relevance", "used_for", "supports", "role", "evidence"}
                        and isinstance(source.get(key), str) and source[key].strip()):
                    source = source[key]
            normalized.append(source)
        sources = normalized
    if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
        raise ValueError('reference_sources must be a JSON array of strings or explicit source_id/path records (optional source_type/usage/relevance/used_for/supports/role/evidence), e.g. ["path/to/source.py", "https://example.org/paper"]. List every path/URL actually used; do not use ambiguous source objects.')
    findings = [{"path": path, "allowed": permitted(path, policy),
                 "role": "method_reference"} for path in sources]
    # A declared component input is a source even if the proposer omits its bibliography.
    for component in method.get("components") or []:
        if not isinstance(component, dict):
            continue
        for declared in component.get("inputs") or []:
            for path in re.findall(r"(?:[\w.-]+/)+[\w.-]+", str(declared)):
                if not permitted(path, policy):
                    finding = {"path": path, "allowed": False, "role": "method_input"}
                    if finding not in findings:
                        findings.append(finding)
    return findings


def merge_source_history(*histories):
    result = []
    for history in histories:
        for item in history or []:
            if isinstance(item, dict) and item not in result:
                result.append(item)
    return result


def attribution(history):
    if any(item.get("allowed") is False for item in history):
        return "Prohibited reference exposure was recorded; reverting code does not establish an independent run."
    notes = []
    if any(item.get("role") == "human_feedback" for item in history):
        notes.append("Human-assisted run: preserve the cited feedback when assessing autonomous recovery.")
    if any(item.get("role") == "method_reference" for item in history):
        notes.append("Reference-assisted work; preserve the listed sources when attributing the result.")
    if any(item.get("attribution") and item.get("role") in {"implementation", "reference", "data", "evaluator"}
           for item in history):
        notes.append("Declared execution inputs have file identities and separate roles; the bootstrap repository URL alone does not establish native implementation performance.")
    if any(item.get("role") == "runtime_version" for item in history):
        notes.append("Runtime version snapshots are recorded separately from experiment outcomes; cross-version results do not isolate repository-selection effects.")
    return " ".join(notes) or "Source records do not by themselves certify that the entire run was independent."
