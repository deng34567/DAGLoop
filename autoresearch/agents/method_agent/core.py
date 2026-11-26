"""MethodAgent core: turn a confirmed research DIRECTION into an explicit, concrete METHOD.

The "method" is the thing whose ADVANTAGE the experiments must demonstrate -- distinct from the
experiment SPEC (interfaces/metric/baseline, which are objective). It is the concrete approach:
what specifically to do/derive/change, and WHY it should beat the baseline. Pure (no transport)."""
from __future__ import annotations

import json

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

_SYSTEM = (
    "You are the MethodAgent. Given a research DIRECTION (problem model + literature + the critique "
    "of the direction), propose ONE concrete METHOD -- the specific approach whose advantage the "
    "experiments will then demonstrate. This is NOT the experiment design (metric/baseline/interface "
    "are objective constraints from supplied task and baseline evidence); it is WHAT you will actually do differently. "
    "The measured starter baseline is a comparison point, not a mandatory algorithmic template. "
    "Design the candidate from the task objective and permitted solution space; you are not required "
    "to start from the starter algorithm. A different mechanism, complete replacement, or local "
    "improvement is allowed; choose among them based on evidence. "
    "若仓库与任务规则未禁止，可在保持评测约束的前提下，考虑复用已有计算或中间产物，以提升效率和效果。 "
    "Keep the official evaluation and correctness contract unchanged. Reply with ONE "
    "JSON object: {\"approach\": \"<the core idea/derivation, 1-3 sentences>\", \"changes_outline\": "
    "[\"<concrete change/step>\", ...], \"why_it_should_win\": \"<the mechanism by which this beats a "
    "reasonable baseline>\", \"assumptions\": [\"<key assumption>\", ...]}. Be concrete and honest -- "
    "the method must be falsifiable, not a restatement of the goal. If a human redirection is given, "
    "you MUST build the method around it. Every named component must either be supported by the "
    "provided repository/literature evidence or include a complete implementable mechanism rather "
    "than a new label. Do not invent literature support, prior experimental validation, repository "
    "files, or formulas. Any schedule or staged change must fit the observed run horizon in the "
    "frozen baseline evidence; a component already present or already absent is not a method delta. "
    "Attribute a measured outcome only to the exact evaluated implementation and its bound "
    "task conditions. A null or missing metric is missing measurement evidence, not zero gain "
    "or proof that hardware lacks a capability. A scientific no-gain conclusion requires "
    "bound measurements for the exact evaluated mechanism; planned changes remain untested "
    "until evaluated. Independent explicit failed gates or execution errors remain negative evidence. "
    "If a revision removes or changes an operation from a prior candidate, "
    "treat that candidate's score as comparative evidence, not a measured score for the revision; "
    "state the revised outcome as unverified until it is evaluated. "
    "If feedback supplies an admissible off-Method result, include optional reconciled_result_ref "
    "only when the proposed Method explicitly adopts that exact measured intervention. Otherwise "
    "omit the field; the result remains comparative evidence. "
    "For benchmarks, distinguish the measured starter baseline from any official reward anchor. "
    "When the supplied target repository contract or evaluator gives a score band, use its exact "
    "gate to judge whether a proposed improvement can earn positive reward; do not invent one. "
    "The approach and changes_outline contain only candidate treatment changes. Do not put the "
    "metric, time/step budget, baseline reruns, control groups, replications, or ablation procedure "
    "into the Method as if they were treatment changes; those belong to experiment design. "
    "Freeze the scientific mechanism and task-required invariants, not incidental choices in a "
    "first implementation. A choice is an adjustable experiment setting unless the task contract "
    "fixes it or evidence shows it is necessary for the mechanism. Describe adjustable dimensions "
    "in the ComponentPlan; specify an initial value only when needed for a runnable experiment. "
    "Do not invent narrow numeric limits or turn an "
    "initial value into a Method-fidelity requirement. "
    "Repository inspection, environment preparation, and final evaluator submission are workflow "
    "steps, not treatment changes; do not list them as Method components or changes_outline steps. "
    "Match declared output types literally: do not add metadata values of another type or use "
    "global state as a workaround for an interface constraint. Keep execution time limits in "
    "the experiment runner unless the task explicitly requires timing logic in submitted code. "
    "When no authoritative frozen BaselineResult is supplied, repository-specific baseline facts "
    "are unresolved: do not assert its optimizer, precision, architecture, schedule, or numeric "
    "hyperparameters from literature context alone. Describe the candidate change against the "
    "component actually found during repository inspection, and label any needed precondition as "
    "an assumption. State unresolved facts as assumptions instead of presenting them as evidence. Reply in the "
    "SAME language as the research direction."
)

MAX_RETRIES = 2


class MethodError(Exception):
    def __init__(self, classification: str, code: str, msg: str):
        super().__init__(msg)
        self.classification = classification
        self.code = code


def _user_message(problem_model, literature_brief, critique_report, feedback, human_idea) -> str:
    lines = []
    if human_idea:
        lines += ["The human injected this redirection you MUST build the method around:",
                  str(human_idea).strip(), ""]
    if feedback:
        lines += ["Evidence and design guidance (baseline observations are not prior Method experiments; "
                  "follow explicit revision instructions only when supplied):", str(feedback).strip(), ""]
    lines += ["Problem model (JSON):", json.dumps(problem_model, ensure_ascii=False, indent=2), "",
              "Literature brief (JSON):", json.dumps(literature_brief or {}, ensure_ascii=False)[:4000], "",
              "Direction critique (JSON):", json.dumps(critique_report or {}, ensure_ascii=False)[:3000]]
    return "\n".join(lines)


def method_step(problem_model, literature_brief, critique_report, llm_client, *,
                prompt_patch=None, feedback=None, human_idea=None,
                incremental_revision=False, prior_method=None, retained_treatment_revision=False,
                reconciliation_result_ref=None, closeout_revision=False) -> dict:
    """Produce ONE concrete method dict from the direction. Raises MethodError on failure."""
    from autoresearch.experiments.repo_improve.execution_policy import (
        efficient_execution, EFFICIENT_PLANNING)
    if efficient_execution():
        feedback = str(feedback or "") + "\n" + EFFICIENT_PLANNING
    system = _SYSTEM
    if incremental_revision:
        system += (
            " INCREMENTAL UPDATE after a completed evidence-based Method reflection: apply "
            "its selected local change or extension to the existing Method, preserving the "
            "effective core and unchanged component contracts verbatim. "
            "A changed treatment changes_outline must update the affected executable component "
            "contracts; changing only top-level prose leaves old checkpoints reusable and does "
            "not schedule the new implementation. Preserve unaffected contracts verbatim. "
            "For a description or evaluation-only clarification, preserve the prior "
            "changes_outline verbatim; do not restate unchanged treatment operations. "
            "When repairing a contradicted mechanism, synchronize its formulas, executable steps "
            "and top-level descriptions; do not retain known-failed steps or claim verification "
            "without matching measured artifacts. New mechanisms and "
            "dependent components are allowed; do not reconstruct the whole Method or repeat "
            "its debate or literature review. Return the complete updated Method INCLUDING "
            "executable components with all existing ComponentPlan fields, numeric role-specific "
            "acceptance checks and valid dependencies. Each component must retain component_id, "
            "name, type, depends_on, purpose, inputs, outputs, acceptance_requires, mechanism, "
            "expected_effect, ablation_test, functional_check, implementation_check, optimization_space "
            "and downstream_contract. depends_on and acceptance_requires contain only component_id "
            "values from this Method; numeric conditions belong in ablation_test or functional_check. "
            "direct_gain uses a numeric ablation_test; indirect_gain uses "
            "a numeric functional_check. Do not invent thresholds or omit unchanged contracts. "
            "For direct_gain, ablation_test must be exactly one comparison: "
            "'signed_metric_improvement <operator> <finite number>' or "
            "'candidate_metric <operator> <finite number>', using <, <=, ==, >=, or >. "
            "signed_metric_improvement is parent - candidate for minimization, candidate - parent "
            "for maximization, relative to the exact dependency checkpoint; gain is strictly positive. "
            "Do not put arithmetic over task-specific variables, units, explanations, and/or, "
            "or no-gain/fallback branches in ablation_test. Keep mandatory correctness and control "
            "checks explicit in implementation_check and runtime evidence; preserve search, stop "
            "and fallback requirements in optimization_space and downstream_contract. The scalar "
            "predicate never waives these requirements. Preserve frozen thresholds; unsupported "
            "mandatory comparisons need formal plan revision, not deletion. "
            "Preserve task rules, comparison, baseline "
            "and measured artifacts; extensions must still be tested before claiming a gain.")
        if retained_treatment_revision:
            system += (
                " RETAINED TREATMENT EXCEPTION: local implementation repair or formal reconciliation "
                "may keep the intended mechanism and its component contracts unchanged. Do not "
                "invent new component work merely to change a contract, or claim an untested "
                "component is measured. Implementation and complete-result validation still apply.")
            if closeout_revision:
                system += (" Restore the retained measured treatment and remove "
                           "unimplemented extensions from the prose.")
            if closeout_revision and reconciliation_result_ref:
                system += (" Bind the selected retained implementation by setting reconciled_result_ref to "
                           + str(reconciliation_result_ref)
                           + ". This schedules its source restoration and fresh validation; "
                             "returning the old unbound Method would rerun its exhausted plan.")
        elif reconciliation_result_ref:
            system += (
                " Exact adoption of the admissible measured result may retain matching component "
                "contracts when reconciled_result_ref equals " + str(reconciliation_result_ref)
                + "; a new treatment still requires updated component contracts.")
    base_messages = [
        {"role": "system", "content": apply_prompt_patch(system, prompt_patch)},
        {"role": "user", "content": _user_message(problem_model, literature_brief, critique_report,
                                                   feedback, human_idea)},
    ]
    messages = list(base_messages)
    last = ""
    truncated_prefix = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            res = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # noqa: BLE001 -- transient LLM/infra failure
            raise MethodError("transient", "method_llm_error", f"{type(exc).__name__}: {exc}")
        try:
            raw_content = res.content or "{}"
            combined_content = truncated_prefix + raw_content if truncated_prefix else raw_content
            try:
                obj = json.loads(extract_json(combined_content))
            except (ValueError, TypeError):
                # A provider may ignore a continuation request and restart the complete
                # JSON object. Accept that response on its own when it is valid.
                if not truncated_prefix:
                    raise
                obj = json.loads(extract_json(raw_content))
        except (ValueError, TypeError) as exc:
            content_length = len(combined_content)
            usage = getattr(res, "usage", None) or {}
            tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            finish_reason = getattr(res, "finish_reason", None)
            parse_position = getattr(exc, "pos", None)
            last = (f"unparseable JSON ({type(exc).__name__}; chars={content_length}; "
                    f"completion_tokens={tokens}; finish_reason={finish_reason}; "
                    f"parse_position={parse_position})")
            if finish_reason == "length" and raw_content:
                # A complete incremental Method can exceed the provider's per-response
                # output ceiling. Asking for the whole object again repeats the same
                # truncation; continue only the missing suffix and validate the joined
                # object through the unchanged schema/ComponentPlan checks below.
                if (truncated_prefix and raw_content.lstrip().startswith("{")):
                    truncated_prefix = raw_content
                else:
                    truncated_prefix = combined_content
                messages = list(base_messages)
                messages.append({"role": "assistant", "content": truncated_prefix})
                messages.append({"role": "user", "content": (
                    "The JSON response was cut off by the output-length limit. Continue exactly "
                    "from its final character and return only the missing suffix needed to close "
                    "the same JSON object. Do not restart, repeat, summarize, or wrap it in markdown."
                )})
                continue
            # Keep retry context bounded. Retaining every prior malformed full Method
            # can consume the provider's remaining context window and truncate the
            # repair itself; the latest draft is sufficient for a syntax repair.
            truncated_prefix = ""
            messages = list(base_messages)
            if raw_content:
                messages.append({"role": "assistant", "content": raw_content})
            messages.append({"role": "user", "content": (
                             "Repair the preceding malformed response into valid JSON with the complete "
                             "updated Method and executable components. Preserve its intended method and "
                             "contracts; fix JSON syntax/completeness only. No markdown or surrounding text."
                             if incremental_revision else "Return a compact valid JSON object "
                             "with only approach, changes_outline, why_it_should_win, and "
                             "assumptions. Omit components and long explanations; the next "
                             "stage designs the components. No markdown or surrounding text.")})
            continue
        if isinstance(obj, dict) and str(obj.get("approach") or "").strip():
            obj.setdefault("changes_outline", [])
            obj.setdefault("why_it_should_win", "")
            obj.setdefault("assumptions", [])
            if incremental_revision:
                from autoresearch.experiments.repo_improve.component_plan import (
                    ComponentPlanError, build_component_plan, validate_component_acceptance)
                try:
                    validate_component_acceptance(build_component_plan(obj))
                    retained_treatment_unchanged = (
                        retained_treatment_revision
                        and (closeout_revision or (
                            isinstance(prior_method, dict)
                            and all(obj.get(key) == prior_method.get(key) for key in (
                                "approach", "changes_outline", "why_it_should_win",
                                "assumptions", "components"))))
                    )
                    if (closeout_revision and reconciliation_result_ref
                            and obj.get("reconciled_result_ref") != reconciliation_result_ref):
                        raise ComponentPlanError(
                            "Closeout must bind the selected retained implementation with "
                            "reconciled_result_ref=" + str(reconciliation_result_ref))
                    if (not retained_treatment_unchanged
                            and not (reconciliation_result_ref and obj.get("reconciled_result_ref")
                                     == reconciliation_result_ref)
                            and isinstance(prior_method, dict)
                            and prior_method.get("components")
                            and obj["changes_outline"] != prior_method.get("changes_outline", [])
                            and obj.get("components") == prior_method.get("components")):
                        raise ComponentPlanError(
                            "Treatment changes_outline changed but kept every executable "
                            "component contract unchanged. Update the affected component's "
                            "mechanism, implementation_check and required outputs to schedule "
                            "the new work; preserve unaffected contracts verbatim. If this is "
                            "only a description or experiment-design clarification, preserve "
                            "the treatment changes_outline instead.")
                except ComponentPlanError as exc:
                    last = str(exc)
                    messages = list(base_messages)
                    messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
                    repair = "Repair only this component contract error in the complete JSON: " + last
                    repair += ("\nRe-express an invalid acceptance string using the executable format above "
                               "without changing its scientific thresholds. Keep required correctness "
                               "and control checks in implementation_check, and search/stop/fallback "
                               "requirements in optimization_space and downstream_contract; do not "
                               "discard them or treat a no-gain stopping branch as a passing gain.")
                    if reconciliation_result_ref:
                        repair += ("\nIf this draft adopts the approved retained implementation, set "
                                   "reconciled_result_ref=" + str(reconciliation_result_ref)
                                   + ". This restores its source for fresh validation; it does not "
                                   "accept its score. Do not invent component work just to change a contract.")
                    if isinstance(prior_method, dict):
                        repair += ("\nFrozen prior treatment changes_outline:\n"
                                   + json.dumps(prior_method.get("changes_outline") or [], ensure_ascii=False)
                                   + "\nFor a description or evaluation-only clarification, copy this "
                                     "outline verbatim. If the treatment really changes, update its "
                                     "affected executable component contracts instead.")
                        repair += ("\nFrozen prior component contracts for this repair. Include retained "
                                   "prerequisites in the returned graph; preserve unchanged contracts verbatim:\n"
                                   + json.dumps(prior_method.get("components") or [], ensure_ascii=False))
                    messages.append({"role": "user", "content": repair})
                    continue
            return obj
        last = "missing non-empty 'approach'"
        messages = list(base_messages)
        messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Invalid: {last}. Reply again with the full JSON."})
    raise MethodError("permanent", "method_parse_error", f"no valid method after retries: {last}")


__all__ = ["method_step", "MethodError"]
