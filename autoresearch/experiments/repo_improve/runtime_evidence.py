"""Read numeric evidence from host-saved execution output, never from its retelling."""
from __future__ import annotations

import hashlib
import json
import math
import operator
from pathlib import Path
import re
import statistics


def can_revalidate_partial_integration(result):
    """A measured legal integration may seed revalidation; it grants no acceptance."""
    if not isinstance(result, dict):
        return False
    evidence = result.get("measurement_evidence") or {}
    admissibility = result.get("candidate_admissibility") or {}
    review = result.get("method_implementation_review") or {}
    execution = result.get("component_execution") or {}
    contract = execution.get("attempt_contract_result") or execution.get("contract_result") or {}
    try:
        measured = finite(evidence.get("value")) == finite(result.get("method_metric"))
    except (TypeError, ValueError):
        return False
    return bool(result.get("checkpoint") and measured
        and evidence.get("valid") is True and evidence.get("stdout_sha256")
        and evidence.get("protocol_fingerprint")
        and (evidence.get("protocol") or {}).get("kind") in {
            "official_single", "deterministic", "repeated"}
        and evidence.get("attempt_id")
        and admissibility.get("attempt_id") == evidence["attempt_id"]
        and admissibility.get("verdict") == "pass"
        and result.get("accepted") is True
        and review.get("verdict") == "partial"
        # Older/lean MIR receipts omit this optional sub-review when the only
        # issue is incomplete Method coverage.  A comparison review that passed
        # the unchanged official repository contract is enough to retain the
        # checkpoint as *source* for fresh revalidation; an explicit violation
        # still fails closed and the historical score is never promoted here.
        and (review.get("task_contract_review") or {}).get("verdict") in {
            None, "", "compliant"}
        and (result.get("repository_comparison_review") or {}).get("verdict") == "pass"
        and execution.get("mode") == "integration"
        and execution.get("status") in {"new_best", "valid_not_best"}
        and execution.get("execution_succeeded") is True and contract.get("passed") is True)


def can_revalidate_accepted_whole_result(result):
    """Reuse a verified whole implementation as source; its next measurement is still fresh."""
    if not isinstance(result, dict):
        return False
    evidence = result.get("measurement_evidence") or {}
    admissibility = result.get("candidate_admissibility") or {}
    review = result.get("method_implementation_review") or {}
    try:
        measured = finite(evidence.get("value")) == finite(result.get("method_metric"))
    except (TypeError, ValueError):
        return False
    return bool(result.get("accepted") is True and result.get("checkpoint") and measured
        and (result.get("component_execution") or {}).get("mode") in {None, "integration"}
        and evidence.get("valid") is True and evidence.get("stdout_sha256")
        and evidence.get("protocol_fingerprint") and isinstance(evidence.get("attempt_id"), str)
        and evidence["attempt_id"].startswith("attempt-")
        and (evidence.get("protocol") or {}).get("kind") in {
            "official_single", "deterministic", "repeated"}
        and admissibility.get("attempt_id") == evidence["attempt_id"]
        and admissibility.get("verdict") == "pass"
        and (result.get("repository_comparison_review") or {}).get("verdict") == "pass"
        and review.get("verdict") == "faithful" and isinstance(review.get("method_ref"), str)
        and review["method_ref"]
        and review["method_ref"] == result.get("method_ref")
        and (review.get("task_contract_review") or {}).get("verdict") == "compliant")


def can_revalidate_reconciliation_candidate(result, *, allow_qualified=False):
    """Retain measured source without qualifying its score under the revised Method."""
    if can_revalidate_partial_integration(result):
        return True
    if not isinstance(result, dict):
        return False
    evidence = result.get("measurement_evidence") or {}
    admissibility = result.get("candidate_admissibility") or {}
    review = result.get("method_implementation_review") or {}
    execution = result.get("component_execution") or {}
    contract = execution.get("attempt_contract_result") or execution.get("contract_result") or {}

    # A qualified indirect component has no primary task metric by design, so it
    # cannot pass the direct-gain comparison below.  Closeout may nevertheless
    # restore its exact implementation as *source* for a fresh whole-Method
    # evaluation.  This grants no score or acceptance: require the bound runtime
    # attempt, component contract, implementation review and task review to have
    # all passed, and leave the official metric to the subsequent full run.
    component_evidence = contract.get("evidence") or {}
    observed_metric_bound = False
    if result.get("method_metric") is not None:
        try:
            observed_metric_bound = bool(
                finite(evidence.get("value")) == finite(result.get("method_metric"))
                and evidence.get("valid") is True
                and evidence.get("stdout_sha256")
                and (evidence.get("protocol") or {}).get("kind") == "component_local"
                and evidence.get("attempt_id") == component_evidence.get("ref")
                and contract.get("measurement_evidence") == evidence
                and (result.get("repository_comparison_review") or {}).get("verdict") == "pass"
            )
        except (TypeError, ValueError):
            observed_metric_bound = False
    qualified_indirect_source = bool(
        allow_qualified
        and result.get("checkpoint")
        and not result.get("component_evidence_error")
        and (result.get("method_metric") is None or observed_metric_bound)
        and execution.get("mode") == "component"
        and execution.get("role") == "indirect_gain"
        and execution.get("status") in {"accepted", "qualified", "improving"}
        and execution.get("execution_succeeded") is True
        and execution.get("checkpoint") == result.get("checkpoint")
        and execution.get("attempt_checkpoint") == result.get("checkpoint")
        and execution.get("became_best") is True
        and contract.get("passed") is True
        and component_evidence.get("source") == "runtime_attempt"
        and isinstance(component_evidence.get("ref"), str)
        and component_evidence["ref"].startswith("attempt-")
        and bool(component_evidence.get("sha256"))
        and (contract.get("measurement_review") or {}).get("verdict") == "pass"
        and admissibility.get("attempt_id") == component_evidence["ref"]
        and admissibility.get("verdict") == "insufficient_evidence"
        and review.get("verdict") == "faithful"
        and isinstance(review.get("method_ref"), str)
        and bool(review["method_ref"])
        and review.get("method_ref") == result.get("method_ref")
        and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
    )
    if qualified_indirect_source:
        return True

    try:
        metric = finite(result.get("method_metric"))
        measured = finite(evidence.get("value")) == metric
    except (TypeError, ValueError):
        return False
    try:
        parent = finite(contract.get("parent_metric"))
        minimum = comparison_epsilon(result)
        qualified_comparison = True
    except (TypeError, ValueError):
        parent = minimum = 0.0
        qualified_comparison = False
    qualified_component = bool(qualified_comparison
        and result.get("checkpoint") and not result.get("component_evidence_error")
        and measured and evidence.get("valid") is True and evidence.get("stdout_sha256")
        and (evidence.get("protocol") or {}).get("kind") == "component_local"
        and isinstance(evidence.get("attempt_id"), str)
        and evidence["attempt_id"].startswith("attempt-")
        and admissibility.get("attempt_id") == evidence["attempt_id"]
        and admissibility.get("verdict") in {"pass", "insufficient_evidence"}
        and (result.get("repository_comparison_review") or {}).get("verdict") == "pass"
        and (review.get("verdict") in {"drifted", "partial"}
             or (allow_qualified and review.get("verdict") == "faithful"
                 and result.get("accepted") is True
                 and execution.get("status") in {"accepted", "qualified", "improving"}))
        and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
        and execution.get("mode") == "component" and execution.get("role") == "direct_gain"
        and execution.get("execution_succeeded") is True and contract.get("passed") is True
        and contract.get("candidate_metric") == metric
        and (contract.get("parent_evidence") or {}).get("host_bound") is True
        and contract.get("measurement_evidence") == evidence
        and type(result.get("lower_is_better")) is bool
        and (parent - metric > minimum if result["lower_is_better"] else metric - parent > minimum))
    if qualified_component:
        return True

    # A component attempt may legally discover a better whole implementation while
    # failing only the frozen component boundary or its parent-comparison binding.
    # Keep that exact checkpoint as input to a fresh whole-Method revalidation; none
    # of its historical score, component qualification, or Method attribution is
    # accepted here.  The revalidation path reruns the official evaluator and all
    # correctness gates before the candidate can become a formal result.
    gates = evidence.get("gates")
    attempt_id = evidence.get("attempt_id")
    return bool(result.get("checkpoint")
        and measured and evidence.get("valid") is True and evidence.get("stdout_sha256")
        and (evidence.get("protocol") or {}).get("kind") == "component_local"
        and isinstance(attempt_id, str) and attempt_id.startswith("attempt-")
        and result.get("component_attempt_id") == attempt_id
        and admissibility.get("attempt_id") == attempt_id
        and admissibility.get("verdict") in {"pass", "insufficient_evidence"}
        and isinstance(gates, dict) and gates and all(value is True for value in gates.values())
        and (result.get("repository_comparison_review") or {}).get("verdict") == "pass"
        and review.get("verdict") in {"drifted", "partial"}
        and isinstance(review.get("method_ref"), str) and review.get("method_ref")
        and review.get("method_ref") == result.get("method_ref")
        and (review.get("task_contract_review") or {}).get("verdict") == "compliant"
        and execution.get("mode") == "component"
        and execution.get("attempt_checkpoint") == result.get("checkpoint")
        and bool(execution.get("attempt_diff_ref"))
        and contract.get("measurement_evidence") == evidence
        and type(result.get("lower_is_better")) is bool)


def comparison_epsilon(new, previous=None):
    """A conservative selection tolerance, not a statistical significance claim."""
    import os
    try:
        relative = max(0, finite(float(os.environ.get("AUTORESEARCH_BESTSOFAR_EPS_REL", "0"))))
    except (ValueError, TypeError):
        relative = 0.0
    tolerance = 0.0
    for result in (new or {}, previous or {}):
        evidence = result.get("measurement_evidence") or {}
        minimum = evidence.get("minimum_improvement", 0)
        try:
            tolerance = max(tolerance, finite(minimum))
            value = finite(result.get("method_metric"))
            if (evidence.get("protocol") or {}).get("kind") != "deterministic":
                tolerance = max(tolerance, abs(value) * relative)
            # Only host-bound observations affect selection. Model-written std alone cannot.
            samples = evidence.get("samples")
            if evidence.get("stdout_sha256") and isinstance(samples, list) and len(samples) >= 2:
                tolerance = max(tolerance, statistics.stdev([finite(v) for v in samples]))
        except (ValueError, TypeError):
            continue
    return tolerance


class MissingGateEvidence(ValueError):
    """A runtime result omitted a required native quality gate."""


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("measurement must be a finite number")
    return float(value)


def pointer(document, path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("measurement requires an explicit JSON pointer")
    if any(re.sub(r"[^a-z0-9]", "", key.casefold()) in {
            "officialreward", "officialscore"} for key in path[1:].split("/")):
        raise ValueError("external benchmark reward cannot be used as research feedback")
    try:
        for key in path[1:].split("/"):
            key = key.replace("~1", "/").replace("~0", "~")
            document = document[int(key)] if isinstance(document, list) else document[key]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError(
            f"execution output is missing {path} (JSON field pointer, not a filesystem path); "
            "emit the required field in the measured command's structured stdout result"
        ) from exc
    return document


def save_output(directory, attempt_id, stdout, stderr):
    result = {}
    for stream, value in (("stdout", stdout), ("stderr", stderr)):
        path = Path(directory) / f"{attempt_id}.{stream}"
        raw = value.encode("utf-8")
        with path.open("xb") as handle:
            handle.write(raw)
        result[stream + "_path"] = str(path)
        result[stream + "_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def execution_file_evidence(repo_path, configuration):
    """Hash only explicitly declared local evaluator/input files at launch/finish."""
    config = configuration or {}
    immutable = config.get("immutable_files", {})
    sources = config.get("source_files", {})
    if not isinstance(immutable, dict) or not isinstance(sources, dict):
        raise ValueError("immutable_files and source_files must be path mappings")
    root = Path(repo_path).resolve()
    evidence = {}
    from .checkpoint_artifacts import digest
    for name in sorted(set(immutable) | set(sources)):
        path = root / name
        if (Path(name).is_absolute() or not path.resolve().is_relative_to(root)
                or path.is_symlink() or not path.is_file()):
            raise ValueError(f"declared execution file is not a regular checkout file: {name}")
        role = sources.get(name, "evaluator")
        if role not in {"implementation", "evaluator", "reference", "data"}:
            raise ValueError(f"invalid execution source role for {name}: {role}")
        value = digest(path)
        if name in immutable and immutable[name] != value:
            raise ValueError(f"frozen evaluator/input file changed: {name}")
        evidence[name] = {"sha256": value, "role": role}
    return evidence


def read_output(record, *, diagnostic=False):
    if not record or not record.get("launched") or not record.get("finished") or record.get("rc") not in ((0, 1) if diagnostic else (0,)):
        raise ValueError("measurement requires a successfully finished host attempt")
    try:
        raw = Path(record["stdout_path"]).read_bytes()
    except (KeyError, OSError) as exc:
        raise ValueError("attempt has no saved runtime output; rerun measurement") from exc
    if hashlib.sha256(raw).hexdigest() != record.get("stdout_sha256"):
        raise ValueError("saved execution output content changed")
    marker = (record.get("execution_protocol") or {}).get("result_marker")
    if marker is not None:
        if (not isinstance(marker, str) or not marker.strip() or len(marker) > 128
                or "\n" in marker or "\r" in marker):
            raise ValueError("result_marker must be a nonempty single-line prefix of at most 128 characters")
        marked = [line[len(marker):].strip() for line in raw.decode("utf-8").splitlines()
                  if line.startswith(marker)]
        if len(marked) != 1:
            raise ValueError("execution must emit exactly one explicitly marked JSON result")
        payload = json.loads(marked[0])
        if not isinstance(payload, dict):
            raise ValueError("execution result must be a JSON object")
        return payload
    try:
        payload = json.loads(raw)
    except ValueError:
        # Logging may surround one JSON result, including a pretty-printed one.
        # Consume the whole object so its nested dictionaries are not extra results.
        payloads = []
        text = raw.decode("utf-8")
        decoder = json.JSONDecoder()
        consumed = 0
        for match in re.finditer(r"(?m)^[ \t]*\{", text):
            start = match.end() - 1
            if start < consumed:
                continue
            try:
                value, consumed = decoder.raw_decode(text, start)
            except ValueError as exc:
                raise ValueError(
                    f"execution must emit one unambiguous JSON result; JSON decoding failed: {exc}. "
                    "Use a JSON serializer in the producing command; do not edit saved output."
                ) from exc
            line_end = text.find("\n", consumed)
            if text[consumed:line_end if line_end >= 0 else len(text)].strip():
                raise ValueError("execution must emit one unambiguous JSON result")
            payloads.append(value)
        # A command and its wrapper may print the same result. Deduplicate only
        # identical full payloads; conflicting measurements or gates stay invalid.
        unique = {json.dumps(value, sort_keys=True): value for value in payloads}
        if len(unique) != 1:
            raise ValueError("execution must emit one unambiguous JSON result")
        payload = next(iter(unique.values()))
    if not isinstance(payload, dict):
        raise ValueError("execution result must be a JSON object")
    return payload


def _receipt_id(reference):
    text, _, fragment = str(reference or "").partition("#")
    ids = set(re.findall(r"\battempt-[A-Za-z0-9-]+\b", text))
    if len(ids) > 1:
        raise ValueError("evidence ref names multiple attempts; cite exactly one host attempt")
    return next(iter(ids)) if ids else text, fragment


def _preserved_snapshot(record, target_record=None, target_directory=None):
    """Prove byte preservation, not just equal commit/diff or a relocated cwd.

    New components may add files, but may not silently change any file covered
    by the inherited measurement. Changed dependencies need a fresh check.
    """
    from .checkpoint_artifacts import validate_checkpoint, digest
    if not record.get("checkpoint") or record.get("checkpoint_error"):
        return False
    try:
        previous = validate_checkpoint(record["checkpoint"])
        if not previous.get("files"):
            return False
        if target_record is not None:
            if not target_record.get("checkpoint") or target_record.get("checkpoint_error"):
                return False
            current = validate_checkpoint(target_record["checkpoint"])
            return all(current["files"].get(name) == value
                       for name, value in previous["files"].items()) and (
                all(current.get("external", {}).get(name) == relative
                    for name, relative in previous.get("external", {}).items())) and all(
                        current.get("directory_links", {}).get(name) == value
                        for name, value in previous.get("directory_links", {}).items())
        if not target_directory:
            return False
        for name, value in previous.get("directory_links", {}).items():
            link = Path(target_directory) / name
            if (not link.is_symlink() or str(link.readlink()) != value
                    or not link.resolve().is_relative_to(Path(target_directory).resolve())):
                return False
        external = {relative: original for original, relative in previous.get("external", {}).items()}
        return all(digest(Path(external[name]) if name in external else
                          Path(target_directory) / name) == value
                   for name, value in previous["files"].items())
    except (OSError, ValueError, KeyError):
        return False


def bind_component_measurements(report, candidate_record, records, *, diagnostic=False):
    payload = read_output(candidate_record, diagnostic=diagnostic)
    bound = []
    for raw in report:
        item = dict(raw)
        item.pop("_host_bound", None)
        item.pop("parent_evidence", None)
        if "actual" in item:
            ref = raw.get("evidence") or {}
            evidence_id, fragment = _receipt_id(ref.get("ref"))
            cited_ids = set(re.findall(r"\battempt-[A-Za-z0-9-]+\b", evidence_id))
            measurement_record, measurement_payload = candidate_record, payload
            if (evidence_id != candidate_record["attempt_id"]
                    and (cited_ids or evidence_id in records)):
                # A lightweight intermediate-property check can run separately
                # from the final evaluator, but must measure the identical code.
                probe = records.get(evidence_id)
                identity_keys = ("cwd", "phase", "repo_commit", "repo_commit_end",
                                 "tracked_diff_hash_start", "tracked_diff_hash_end",
                                 "execution_protocol")
                declared_files_match = bool(probe and
                    probe.get("execution_files_start", {}) == probe.get("execution_files_end", {})
                    == candidate_record.get("execution_files_start", {})
                    == candidate_record.get("execution_files_end", {}))
                same_state = bool(probe and item.get("type") == "indirect_gain"
                    and "parent_metric" not in item and declared_files_match and all(
                        key in probe and key in candidate_record and probe[key] == candidate_record[key]
                        for key in identity_keys))
                inherited = bool(probe and probe.get("phase") == candidate_record.get("phase")
                    and "execution_protocol" in probe and "execution_protocol" in candidate_record
                    and probe["execution_protocol"] == candidate_record["execution_protocol"]
                    and declared_files_match
                    and _preserved_snapshot(probe, target_record=candidate_record))
                if (ref.get("source") != "runtime_attempt" or not probe
                        or not (same_state or inherited)
                        or not probe.get("repo_commit")
                        or probe.get("repo_commit") != probe.get("repo_commit_end")
                        or not probe.get("tracked_diff_hash_start")
                        or probe.get("tracked_diff_hash_start") != probe.get("tracked_diff_hash_end")):
                    raise ValueError(
                        f"component {item.get('component')} cites {evidence_id}, "
                        f"but its evidence cannot bind to candidate {candidate_record['attempt_id']}: "
                        "separate component evidence must be an explicit indirect-gain host "
                        "attempt with the same cwd, phase, unchanged code state and execution "
                        "protocol as the candidate, or a cited successful checkpoint whose "
                        "measured files are preserved in the candidate snapshot. Changed or "
                        "unverifiable dependencies require a fresh component check. "
                        "This is not proof that the candidate command never ran. Inspect the "
                        "candidate receipt separately and repair the named component's evidence "
                        "without discarding successful execution records."
                    )
                measurement_payload = read_output(probe)
                measurement_record = probe
            path = (ref.get("json_pointer") or (fragment if fragment.startswith("/") else "")
                    or "/measurements/" + str(item.get("component")))
            try:
                measured = pointer(measurement_payload, path)
            except ValueError as exc:
                raise ValueError(
                    f"execution output is missing {path} (JSON field pointer, not a filesystem path); "
                    f"component {item.get('component')} evidence.json_pointer cannot resolve "
                    "a field in saved execution output; "
                    f"available top-level fields: {list(measurement_payload)[:20]}. "
                    "Bind evidence.json_pointer to the existing measured scalar for the "
                    "declared acceptance check, with the exact attempt ID in evidence.ref; "
                    "Do not guess a different metric; do not rewrite saved output. "
                    "If that measurement was not emitted, rerun the measurement with "
                    "structured output. A binding repair alone does not require retraining."
                ) from exc
            try:
                actual = finite(measured)
            except ValueError as exc:
                keys = list(measured)[:12] if isinstance(measured, dict) else []
                raise ValueError(
                    f"component {item.get('component')} evidence.json_pointer {path!r} "
                    f"selects {type(measured).__name__}, not a finite numeric scalar; "
                    f"available object keys: {keys}. Bind the measured scalar field in "
                    "this saved execution output; do not guess or rewrite its value."
                ) from exc
            # Permit floating-point roundoff only; retain the exact host value
            # below, and never let an absolute tolerance erase tiny metrics.
            if not math.isclose(finite(item["actual"]), actual, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    f"component scalar disagrees with saved execution output: "
                    f"component={item.get('component')}, evidence.json_pointer={path!r}, "
                    f"saved={actual}, reported={item['actual']}. Bind the measured field "
                    "for the declared acceptance check; do not substitute a correctness "
                    "probe for the performance metric or rewrite saved output. "
                    "The reported value above is ablation_report.actual, not necessarily "
                    "the top-level method field or the wrapper's measurements field. "
                    "For direct gain, bind actual to the raw candidate metric and bind "
                    "parent_metric separately; the host computes signed_metric_improvement. "
                    "Do not copy the derived contract_result.actual back into the report."
                )
            item["actual"] = actual
            item["evidence"] = {"source": "runtime_attempt", "ref": measurement_record["attempt_id"],
                                "json_pointer": path, "sha256": measurement_record["stdout_sha256"]}
            item["_host_bound"] = True
        if "parent_metric" in item:
            ref = raw.get("parent_evidence") or {}
            parent_id, fragment = _receipt_id(ref.get("ref"))
            parent = records.get(parent_id)
            try:
                control = read_output(parent)
            except ValueError as exc:
                detail = ("reference is not registered in the current sandbox ledger; "
                          "this does not establish that the historical run failed"
                          if parent is None else str(exc))
                control_cwd = candidate_record.get("dependency_parent_cwd")
                available_controls = [
                    attempt_id for attempt_id, record in records.items()
                    if control_cwd and record.get("cwd") == control_cwd
                    and record.get("launched") and record.get("finished")
                    and record.get("rc") == 0
                ][-3:]
                raise ValueError(
                    "parent/control evidence is invalid: parent_evidence.ref must identify "
                    "a successfully finished host control attempt, not a file description. "
                    "Bind the exact dependency-parent measurement; the candidate execution "
                    f"was read successfully. Received ref={ref.get('ref')!r}; "
                    f"registered successful attempts in expected control cwd={control_cwd!r}: "
                    f"{available_controls!r}. These are diagnostic hints, not automatically accepted "
                    "evidence: verify their saved output and exact parent state before binding. "
                    f"Detail: {detail}"
                ) from exc
            failed_gates = [name for name in ("candidate_valid", "baseline_valid")
                            if control.get(name) is False]
            # A preserved historical component keeps its own parent contrast;
            # it must not be rebound against the final combination's parent.
            contrast = measurement_record if "actual" in item else candidate_record
            expected = contrast.get("dependency_parent_state")
            evaluation = contrast.get("execution_protocol") or {}
            declared_checks, _ = _declared_result_checks(control, evaluation)
            if "gate_pointers" in evaluation:
                declared_checks.update(_boolean_gate_checks(control, evaluation, parent.get("phase", "method")))
            failed_gates.extend(name for name, passed in declared_checks.items() if not passed)
            if evaluation.get("argv"):
                from .agentic_improve import _normalized_attempt_argv
                frozen_argv = _normalized_attempt_argv(evaluation["argv"])
                if any(_normalized_attempt_argv(run.get("argv")) != frozen_argv
                       for run in (parent, contrast)):
                    raise ValueError("parent and candidate must use the frozen evaluator entry point")
            if (contrast.get("execution_protocol") or {}).get("source_files") or (contrast.get("execution_protocol") or {}).get("immutable_files"):
                files = parent.get("execution_files_start")
                if (not files or files != parent.get("execution_files_end")
                        or files != (expected or {}).get("execution_files_start")):
                    raise ValueError("control attempt did not measure the declared dependency-parent source files")
            control_cwd = str(Path(contrast.get("dependency_parent_cwd")
                                   or contrast.get("cwd") or ""))
            same_location = parent.get("cwd") == control_cwd
            relocated = not same_location and _preserved_snapshot(
                parent, target_directory=control_cwd)
            if (not expected or any(parent.get(key) != expected.get(key) for key in (
                    "repo_commit", "tracked_diff_hash_start"))
                    or parent.get("repo_commit_end") != parent.get("repo_commit")
                    or parent.get("tracked_diff_hash_end") != parent.get("tracked_diff_hash_start")
                    or not (same_location or relocated)
                    or parent.get("execution_protocol") != contrast.get("execution_protocol")):
                raise ValueError(
                    "control attempt is not the exact dependency parent: "
                    f"expected cwd={control_cwd!r}, "
                    f"observed cwd={parent.get('cwd')!r}; "
                    f"expected code state={expected!r}, observed start="
                    f"{ {key: parent.get(key) for key in ('repo_commit', 'tracked_diff_hash_start')}!r}, "
                    "observed end="
                    f"{ {key: parent.get(key) for key in ('repo_commit_end', 'tracked_diff_hash_end')}!r}; "
                    "protocol_matches="
                    f"{parent.get('execution_protocol') == contrast.get('execution_protocol')}. "
                    "Run the same evaluator through ar-attempt in the exact parent-control checkout, "
                    "without changing its tracked code or protocol; cite that successful attempt, "
                    "not a control run inside the candidate checkout."
                )
            parent_pointer = (ref.get("json_pointer")
                              or (fragment if fragment.startswith("/") else "") or "/method")
            try:
                value = finite(pointer(control, parent_pointer))
            except ValueError as exc:
                raise ValueError(
                    f"parent/control attempt {parent['attempt_id']} cannot resolve "
                    f"parent_evidence.json_pointer={parent_pointer!r} to a finite scalar; "
                    f"available top-level fields: {list(control)[:20]}. Bind the existing "
                    "parent measurement field for the declared comparison, not the "
                    f"candidate output; do not rewrite saved output. Detail: {exc}"
                ) from exc
            if not math.isclose(finite(item["parent_metric"]), value, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    "parent scalar disagrees with its saved output: "
                    f"attempt={parent['attempt_id']!r}, pointer={parent_pointer!r}, "
                    f"saved={value}, reported={item['parent_metric']!r}. "
                    "Bind parent_metric to the exact dependency-parent measurement; "
                    "do not substitute the frozen baseline or the candidate metric. "
                    "If the reference is wrong, cite the correct parent attempt and field; "
                    "do not rewrite saved output."
                )
            metric_pointer = evaluation.get("metric_pointer")
            performance_pointers = {metric_pointer} if metric_pointer else {"/method", "/baseline"}
            if parent_pointer not in performance_pointers:
                raise ValueError(
                    "parent/control evidence must bind the primary performance quantity, "
                    f"not an intermediate check: received {parent_pointer!r}; "
                    f"expected one of {sorted(performance_pointers)!r}. "
                    "Measure the exact dependency parent with the frozen task evaluator; "
                    "if it cannot provide a valid performance control, revise the comparison "
                    "design rather than substitute a check count or the original baseline. "
                    "Preserve the candidate measurement and saved outputs."
                )
            primary_pointer = metric_pointer or (
                "/baseline" if parent.get("phase") == "baseline" else
                "/method" if parent.get("phase") == "method" else parent_pointer)
            if not metric_pointer and control.get(primary_pointer[1:]) is None:
                # Legacy controls may leave the unused canonical scalar absent or null.
                primary_pointer = parent_pointer
            primary_value = pointer(control, primary_pointer)
            if isinstance(primary_value, dict):
                primary_value = primary_value.get("value", primary_value.get("metric"))
            if not math.isclose(value, finite(primary_value), rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    "parent/control primary performance quantity differs from its measured "
                    f"{primary_pointer!r} field; do not substitute another treatment's "
                    "baseline or candidate scalar. Preserve the saved execution output.")
            item["parent_metric"] = value
            item["parent_evidence"] = {"source": "runtime_attempt", "ref": parent["attempt_id"],
                                       "json_pointer": parent_pointer,
                                       "sha256": parent["stdout_sha256"], "host_bound": True}
            if failed_gates:
                # Preserve failed controls for diagnostics. The frozen component role,
                # not a sandbox-supplied type, decides whether performance is comparable.
                item["parent_evidence"]["failed_gates"] = failed_gates
                outcome = parent.get("frozen_baseline_outcome")
                eligibility_only = all(
                    gate in {"candidate_valid", "baseline_valid", "/candidate_valid",
                             "/baseline_valid"} or str(gate).startswith("quality:")
                    for gate in failed_gates)
                if (parent.get("frozen_baseline_control") is True
                        and isinstance(outcome, dict)
                        and (outcome.get("feasible") is False
                             # Baseline reproduction may pass while the unchanged starter
                             # fails the candidate-only eligibility gate in this comparison.
                             or (outcome.get("feasible") is True
                                 and set(failed_gates) <= {"candidate_valid", "/candidate_valid"}))
                        and eligibility_only):
                    item["parent_evidence"]["reproduced_ineligible_baseline"] = True
        bound.append(item)
    return bound


def _declared_result_checks(payload, config):
    checks = {}
    quality = {}
    required = config.get("required_values", {})
    if not isinstance(required, dict):
        raise ValueError("evaluation required_values must map JSON pointers to frozen values")
    for path, expected in required.items():
        actual = pointer(payload, path)
        checks["identity:" + path] = (type(actual) is type(expected) and actual == expected)
    comparisons = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
                   ">=": operator.ge, "==": operator.eq, "!=": operator.ne}
    quality_gates = config.get("quality_gates", [])
    if not isinstance(quality_gates, list):
        raise ValueError("evaluation quality_gates must be an array")
    for gate in quality_gates:
        if (not isinstance(gate, dict) or set(gate) != {"pointer", "operator", "threshold"}
                or gate.get("operator") not in comparisons):
            raise ValueError("quality gate requires pointer, numeric threshold and comparison operator")
        path = gate["pointer"]
        actual, threshold = finite(pointer(payload, path)), finite(gate["threshold"])
        quality[path] = actual
        checks["quality:" + path + gate["operator"] + str(threshold)] = comparisons[gate["operator"]](actual, threshold)
    return checks, quality


def _boolean_gate_checks(payload, config, phase):
    gates = config.get("gate_pointers", ["/baseline_valid"] if phase == "baseline" else ["/candidate_valid"])
    if not isinstance(gates, list) or not gates or not all(isinstance(g, str) for g in gates):
        raise ValueError("gate_pointers must be a nonempty list")
    try:
        outcomes = [pointer(payload, gate) for gate in gates]
    except ValueError as exc:
        raise MissingGateEvidence(str(exc)) from exc
    if any(type(outcome) is not bool for outcome in outcomes):
        raise ValueError("every declared evaluator gate must return a boolean")
    return dict(zip(gates, outcomes))


def measured_result(record, configuration, phase, *, expected_metric_name=None):
    """The repository may declare JSON pointers and a repeated-measurement protocol."""
    if isinstance(expected_metric_name, str) and re.sub(
            r"[^a-z0-9]", "", expected_metric_name.casefold()) in {
                "officialreward", "officialscore"}:
        raise ValueError("external benchmark reward cannot be the research objective")
    payload = read_output(record)
    config = configuration or {}
    if config.get("immutable_files") or config.get("source_files"):
        start, end = record.get("execution_files_start"), record.get("execution_files_end")
        if not isinstance(start, dict) or not start or start != end:
            raise ValueError("declared execution sources lack unchanged host file evidence")
        expected_files = set(config.get("immutable_files", {})) | set(config.get("source_files", {}))
        if set(start) != expected_files or any(
                start[name].get("sha256") != expected
                for name, expected in config.get("immutable_files", {}).items()) or any(
                    start[name].get("role") != config.get("source_files", {}).get(name, "evaluator")
                    for name in expected_files):
            raise ValueError("execution source evidence differs from frozen file contract")
    if "argv" in config:
        from .agentic_improve import _normalized_attempt_argv
        if _normalized_attempt_argv(record.get("argv")) != _normalized_attempt_argv(config["argv"]):
            raise ValueError("runtime command differs from the frozen evaluator entry point")
    metric_path = config.get("metric_pointer") or ("/baseline" if phase == "baseline" else "/method")
    value = pointer(payload, metric_path)
    if isinstance(value, dict):
        value = value.get("value", value.get("metric"))
    value = finite(value)
    if phase == "method" and isinstance(expected_metric_name, str):
        metric_name = expected_metric_name.casefold()
        if metric_name.startswith("test_"):
            validation_name = "val_" + metric_name[len("test_"):]
            validation_value = payload.get(validation_name)
            if (metric_name not in payload and payload.get("val_exit_code") == 0
                    and isinstance(validation_value, (int, float))
                    and not isinstance(validation_value, bool)
                    and math.isclose(value, finite(validation_value), rel_tol=1e-15, abs_tol=0)):
                raise ValueError(
                    f"host output binds /method to validation-only {validation_name}, "
                    f"not the declared {expected_metric_name}; preserve this validation "
                    "observation as a proxy. Do not run held-out test for component selection; "
                    "claim the primary metric only after its declared final evaluator runs")
    checks = _boolean_gate_checks(payload, config, phase)
    declared_checks, quality = _declared_result_checks(payload, config)
    checks.update(declared_checks)
    policy = config.get("measurement") or {"kind": "official_single"}
    kind = policy.get("kind")
    minimum = finite(policy.get("minimum_improvement", 0))
    if minimum < 0 or kind not in {"official_single", "deterministic", "repeated"}:
        raise ValueError("invalid measurement policy")
    sample_evidence = {}
    if kind == "repeated":
        count = policy.get("repeats")
        aggregation = policy.get("aggregation")
        if type(count) is not int or count < 2 or aggregation not in {"mean", "median"}:
            raise ValueError("repeated protocol needs repeats >= 2 and mean/median aggregation")
        samples = pointer(payload, policy.get("samples_pointer") or "/samples")
        if not isinstance(samples, list) or len(samples) != count:
            raise ValueError("measurement sample count differs from the frozen protocol")
        samples = [finite(v) for v in samples]
        aggregate = statistics.mean if aggregation == "mean" else statistics.median
        if not math.isclose(value, aggregate(samples), rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("reported scalar is not the frozen sample aggregate")
        value = aggregate(samples)
        sample_evidence = {"samples": samples, "aggregation": aggregation,
                           "method_runs": samples, "method_std": statistics.stdev(samples)}
        if phase == "method":
            controls = pointer(payload, policy.get("control_samples_pointer") or "/control_samples")
            if not isinstance(controls, list) or len(controls) != count:
                raise ValueError("paired control sample count differs from the frozen protocol")
            controls = [finite(v) for v in controls]
            sample_evidence.update(control_samples=controls, control_metric=aggregate(controls))
    reproduction_checks = checks
    if phase == "baseline" and "baseline_outcome" in payload:
        # External benchmark scores are not research feedback. Only declared raw
        # checks determine feasibility; identity and implementation still bind.
        outcome = payload["baseline_outcome"]
        if (not isinstance(outcome, dict) or type(outcome.get("feasible")) is not bool
                or not isinstance(outcome.get("quality_observations"), dict)
                or not isinstance(outcome.get("reason"), str) or not outcome["reason"].strip()):
            raise ValueError("baseline_outcome requires feasible, quality_observations and reason")
        metric = payload.get("metric")
        metric_name = metric.get("name") if isinstance(metric, dict) else metric
        if isinstance(metric_name, str) and metric_name.strip():
            measured_key = "measured_" + re.sub(r"[^a-z0-9]+", "_", metric_name.casefold()).strip("_")
            if measured_key in outcome["quality_observations"] and not math.isclose(
                    value, finite(outcome["quality_observations"][measured_key]),
                    rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"baseline objective disagrees with {measured_key}")
        quality_passed = all(ok for name, ok in declared_checks.items() if name.startswith("quality:"))
        if outcome["feasible"] and not quality_passed:
            raise ValueError("baseline claims feasibility despite failed declared quality checks")
        if not quality_passed:
            reproduction_checks = {name: ok for name, ok in checks.items()
                                   if not name.startswith("quality:")}
        sample_evidence["baseline_outcome"] = {
            "feasible": all(checks.values()),
            "quality_observations": quality,
            "reason": ("Declared research checks passed." if all(checks.values())
                       else "Declared research checks were not met."),
        }
    return {"value": value, "valid": all(reproduction_checks.values()), "gates": checks,
            **({"quality_observations": quality} if quality else {}),
            **({"execution_sources": record["execution_files_start"]}
               if record.get("execution_files_start") else {}),
            "attempt_id": record["attempt_id"], "stdout_sha256": record["stdout_sha256"],
            "protocol_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "minimum_improvement": minimum, "protocol": policy, **sample_evidence}
