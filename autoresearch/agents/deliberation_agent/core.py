"""
DeliberationAgent core: the pure, BOUNDED deep-deliberation logic.

The professor's "deep deliberation" step, made bounded so it converges by construction:
after a direction is chosen, N=2-3 angle-agents each think INDEPENDENTLY from a distinct
lens (round 1), then each REVISES once after seeing the other angles' takes (round 2, the
cross-reference), and a final SYNTHESIS pass cross-examines them into one deeper conclusion.
Total LLM calls = 2N + 1 (fixed N, single cross-reference round -> no unbounded loop).

Pure: no EventBus / Store / Redis access (the handler resolves refs, persists the report,
and publishes). Fail-soft: an individual angle whose LLM call fails is DROPPED rather than
sinking the whole step; only a total wipe-out (no angle survived AND synthesis failed)
raises -- and even then the HANDLER degrades to a minimal report so the flow proceeds.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

logger = logging.getLogger(__name__)


def _short(s, n: int = 90) -> str:
    """One-line, length-capped view of an approach/reason for auditable logs."""
    s = str(s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

# The three distinct lenses (professor's model: each agent thinks from its own angle).
DEFAULT_ANGLES = ("theory-first", "empirical-baseline-first", "failure-mode-risk-first")


class DeliberationError(Exception):
    """Base for DeliberationAgent core failures."""

    code: str = "deliberation_failed"
    classification: str = "transient"


class DeliberationLLMError(DeliberationError):
    """Every LLM call failed -- nothing to deliberate on."""

    code = "deliberation_llm_error"
    classification = "transient"


class LLMClientLike(Protocol):
    def chat(self, *, messages: list[dict], response_format: str | None = ...,
             max_tokens: int | None = ...) -> Any: ...


@dataclass
class DeliberationResult:
    deliberation_report: dict


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    path = Path(__file__).resolve().parent / "data-contracts" / "deliberation_report.schema.yaml"
    return Draft202012Validator(yaml.safe_load(path.read_text(encoding="utf-8")))


_ANGLE_SYSTEM = (
    "You are ONE deliberation agent examining a chosen research direction through a SINGLE "
    "distinct lens: {angle}. Think ONLY from this lens; do not try to be balanced -- other "
    "agents cover the other lenses. Given the research goal and the literature context, give "
    "your take. Reply with ONE JSON object: {{\"approach\": \"<the approach/direction this lens "
    "argues for>\", \"supporting\": \"<known theory/prior results it leans on>\", "
    "\"expected_baseline\": \"<the baseline you'd compare against: name + rough level>\"}}."
)

_CROSSREF_SYSTEM = (
    "You are the {angle} deliberation agent. You have your own take, and you are now shown the "
    "OTHER agents' takes. Reference and critique them, then REVISE your own take where they "
    "expose a gap or a better idea (keep your lens; don't just capitulate). Reply with ONE JSON "
    "object: {{\"approach\": \"<possibly-updated approach>\", \"supporting\": \"<...>\", "
    "\"expected_baseline\": \"<...>\", \"revised\": \"<what you changed after cross-reference, or "
    "'no change' with why>\"}}."
)

_SYNTH_SYSTEM = (
    "You are MOSS, synthesizing a bounded deliberation. You are given N angle-agents' "
    "cross-referenced takes on a chosen research direction. Produce ONE deeper, cross-examined "
    "conclusion: name the direction to pursue and the baseline to reproduce/compare against, and "
    "keep the points of agreement and disagreement EXPLICIT (do not average disagreement away). "
    "Reply with ONE JSON object: {\"chosen_direction\": \"<...>\", \"chosen_baseline_hint\": "
    "\"<baseline name + level>\", \"convergence\": [\"<agreed point>\", ...], \"divergence\": "
    "[\"<disagreed point>\", ...], \"synthesis\": \"<narrative of the deeper conclusion>\"}."
)


def _context_str(research_spec: dict, literature_brief: dict | None) -> str:
    goal = (research_spec or {}).get("research_goal") or (research_spec or {}).get("goal") or "(goal unspecified)"
    lit = ""
    if literature_brief:
        lit = str(literature_brief.get("summary") or "")[:2000]
    return f"Research goal:\n{goal}\n\nLiterature context:\n{lit or '(none)'}"


def _chat_json(
    llm: LLMClientLike,
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
) -> dict | None:
    """One LLM call -> parsed JSON object, or None on any failure (fail-soft)."""
    result = None
    try:
        result = llm.chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format="json_object",
            max_tokens=max_tokens)
        obj = json.loads(extract_json(result.content or "{}"))
        return obj if isinstance(obj, dict) else None
    except Exception as exc:  # noqa: BLE001 -- malformed individual replies may degrade
        from autoresearch.llm.errors import is_resumable_provider_error
        if is_resumable_provider_error(exc):
            raise DeliberationLLMError(str(exc)) from exc
        logger.warning("[method-debate] JSON call unavailable (%s); chars=%d "
                       "completion_tokens=%s parse_position=%s",
                       type(exc).__name__, len(getattr(result, "content", None) or ""),
                       (getattr(result, "usage", None) or {}).get("completion_tokens"),
                       getattr(exc, "pos", None))
        return None


def deliberate(research_spec: dict, literature_brief: dict | None, llm: LLMClientLike, *,
               n_angles: int = 3, angles: tuple[str, ...] | None = None,
               prompt_patch: str | None = None) -> DeliberationResult:
    """Run the bounded deep-deliberation step and return a schema-valid DeliberationReport.

    Raises DeliberationLLMError only if EVERY angle call failed AND synthesis failed (the
    handler still degrades that into a minimal report so the flow never stalls)."""
    chosen = (angles or DEFAULT_ANGLES)[:max(1, min(n_angles, len(angles or DEFAULT_ANGLES)))]
    ctx = _context_str(research_spec, literature_brief)

    def sys(s: str) -> str:
        return apply_prompt_patch(s, prompt_patch) if prompt_patch else s

    # Round 1: independent take per angle (fail-soft per angle).
    takes: dict[str, dict] = {}
    for angle in chosen:
        obj = _chat_json(llm, sys(_ANGLE_SYSTEM.format(angle=angle)),
                         f"{ctx}\n\nGive your {angle} take.")
        if obj:
            takes[angle] = {"angle": angle,
                            "approach": str(obj.get("approach") or "").strip(),
                            "supporting": str(obj.get("supporting") or "").strip(),
                            "expected_baseline": str(obj.get("expected_baseline") or "").strip(),
                            "revised": ""}

    # Round 2: single cross-reference round -- each surviving angle revises given the others.
    for angle in list(takes):
        others = [t for a, t in takes.items() if a != angle]
        if not others:
            continue
        others_str = "\n\n".join(
            f"[{t['angle']}] approach: {t['approach']}\n  baseline: {t['expected_baseline']}"
            for t in others)
        obj = _chat_json(llm, sys(_CROSSREF_SYSTEM.format(angle=angle)),
                         f"{ctx}\n\nYour current take:\napproach: {takes[angle]['approach']}\n"
                         f"baseline: {takes[angle]['expected_baseline']}\n\n"
                         f"Other agents' takes:\n{others_str}")
        if obj:
            takes[angle].update(
                approach=str(obj.get("approach") or takes[angle]["approach"]).strip(),
                supporting=str(obj.get("supporting") or takes[angle]["supporting"]).strip(),
                expected_baseline=str(obj.get("expected_baseline") or takes[angle]["expected_baseline"]).strip(),
                revised=str(obj.get("revised") or "").strip())

    angle_list = list(takes.values())

    # Synthesis: cross-examine the revised takes into one deeper conclusion.
    synth = None
    if angle_list:
        takes_str = "\n\n".join(
            f"[{t['angle']}] approach: {t['approach']}\n  supporting: {t['supporting']}\n"
            f"  baseline: {t['expected_baseline']}\n  revised: {t['revised']}"
            for t in angle_list)
        synth = _chat_json(llm, sys(_SYNTH_SYSTEM), f"{ctx}\n\nAngle takes:\n{takes_str}")

    if not angle_list and not synth:
        raise DeliberationLLMError("every deliberation LLM call failed")

    report = {
        "angles": angle_list,
        "convergence": [str(x) for x in (synth or {}).get("convergence", []) if str(x).strip()],
        "divergence": [str(x) for x in (synth or {}).get("divergence", []) if str(x).strip()],
        "chosen_direction": str((synth or {}).get("chosen_direction") or "").strip(),
        "chosen_baseline_hint": str((synth or {}).get("chosen_baseline_hint") or "").strip(),
        "synthesis": str((synth or {}).get("synthesis") or "").strip()
                     or "(synthesis unavailable; angle takes retained)",
    }
    # keep only schema keys with content; angles + synthesis are required
    report = {k: v for k, v in report.items()
              if k in ("angles", "synthesis") or v}
    errors = sorted(_validator().iter_errors(report), key=lambda e: e.path)
    if errors:
        # last-resort: a minimal valid report (never sink the flow on a schema nit)
        report = {"angles": angle_list, "synthesis": report.get("synthesis") or "(deliberation)"}
    return DeliberationResult(deliberation_report=report)


def minimal_report(note: str = "(deliberation unavailable)") -> dict:
    """The fail-open minimal report the handler emits when deliberation cannot run."""
    return {"angles": [], "synthesis": note}


# ---------------------------------------------------------------------------
# Method-layer bounded FOUR-ROLE deliberation (meeting 2026-08-03, redesign).
# The professor's "注入灵魂", now with FOUR distinct roles that stay decoupled:
#   - TWO proposers (A1, A2): each develops a method while absorbing the stronger parts of the
#     OTHER's current proposal, which is cross-relayed by MOSS; the goal is agreement, not combat.
#   - CRITIC (a dedicated 3rd agent, NOT MOSS): after the inner exchange converges or reaches its
#     cap, it reviews the agreed method or both unresolved proposals and raises RUBRIC-GATED DEEP
#     problems. A clean converged review accepts; a deep problem drives another outer round.
#   - MOSS (the coordinator, separate from the critic): cross-relays A1/A2's proposals, guides an
#     unresolved inner-loop disagreement, and relays the critic's findings back as constructive
#     guidance. MOSS is NOT the judge and does not decide whether a problem merits another round.
# Bounded by max_rounds so it converges by construction. Loose coupling: every role is a small pure
# call with explicit inputs/outputs; the only shared state is the two candidate methods + feedback.
# ---------------------------------------------------------------------------

# CRITIC rubric: ONLY problems in these categories count as "deep" and may trigger another round.
# Anything else (wording, formatting, naming) is shallow -> dropped -> pass.
METHOD_CRITIC_RUBRIC = (
    "comparison_integrity",   # changes the frozen metric/budget/controls/aggregation
    "component_soundness",      # a named component is unjustified / likely to hurt or not help
    "ablation_verifiability",  # a component lacks a verifiable add-vs-remove checkpoint
    "novelty_conflict",        # collides with / is subsumed by known prior work
    "feasibility",             # infeasible under the compute/data/time budget
    "failure_mode",            # a concrete way the method breaks or is unstable
)

_METHOD_PROPOSER_SYSTEM = (
    "You are METHOD PROPOSER {label} in a bounded debate with the OTHER proposer. ANCHOR: the "
    "ORIGINAL research goal + its concrete metric are your north star EVERY round -- use the other "
    "proposer's method and any MOSS/critic/user feedback ONLY as REFERENCE to sharpen YOUR concrete "
    "strategy toward that goal; do NOT let the exchange pull you into META-strategy (how to measure "
    "/ detect / tune / screen / calibrate) -- propose a concrete change to the actual "
    "model/training/kernel that moves the REAL metric. "
    "GOAL = CONVERGE, not compete: each round actively MOVE TOWARD AGREEMENT with the other proposer "
    "-- ADOPT their stronger ideas, CONCEDE your own weaker ones, and merge toward ONE shared best "
    "method for the goal (you are collaborating to find the single best method, NOT defending your "
    "own). If you already agree on the core idea + key components, SAY SO and restate the shared "
    "method so the judge marks it CONVERGED; small residual differences are fine. With this you "
    "should converge within ~3 rounds. Do TWO things: "
    "(1) produce YOUR strongest concrete, falsifiable method (not a restatement of the goal), "
    "actively INCORPORATING the other's stronger ideas + any MOSS feedback / user idea; (2) state "
    "what (if anything) still MATERIALLY differs and must be resolved to converge (empty if agreed). "
    "Each method component MUST declare a VERIFIABLE CONTRACT so the EXPERIMENT layer can later test "
    "it directly and localize which component under-delivers when the method disappoints. Every "
    "component is one of TWO TYPES: (i) DIRECT_GAIN = it is meant to directly MOVE the target metric; "
    "(ii) INDIRECT_GAIN = it enables a later gain by producing a required intermediate property "
    "or representation. For example, if its job is to output orthogonal representations, its own "
    "acceptance check must measure that orthogonality; it is not rejected merely because the final "
    "metric has not improved yet. Give every component a short stable "
    "COMPONENT_ID and an explicit DEPENDS_ON list of component IDs. Empty means it can be tested "
    "independently from the same baseline; a non-empty list means it may only be implemented after "
    "those dependencies pass their own contracts. Also declare concrete INPUTS and OUTPUTS so a "
    "join node can be tested with all of its prerequisite branches present. Declare per component: "
    "A component is always behavior implemented by the measured candidate. Metric definitions, "
    "budget choices, baseline/control-group runs, replications, and ablation procedures are "
    "experiment design: keep them only in ablation_test/functional_check, never promote them into "
    "components or changes_outline. Repository inspection, environment preparation, and final "
    "evaluator submission are workflow steps, not candidate components or Method changes. "
    "Freeze only the scientific mechanism and task-required invariants. Treat a first "
    "implementation choice as an adjustable experiment setting unless the task contract fixes "
    "it or evidence shows that the mechanism requires it; do not make its initial value a "
    "Method-fidelity condition. Put adjustable dimensions in optimization_space without "
    "inventing narrow numeric limits. "
    "The FROZEN COMPARISON SEMANTICS supplied by MOSS always wins. "
    "If no authoritative frozen BaselineResult is supplied, repository-specific baseline details "
    "are unresolved: literature context does not prove its optimizer, precision, architecture, "
    "schedule, or numeric hyperparameters. Do not present those details as facts; make the method "
    "target the component actually found during repository inspection and state any required "
    "precondition as an assumption. "
    "(a) TYPE = direct_gain|indirect_gain; (b) MECHANISM = why it directly moves the TARGET METRIC "
    "or which measurable intermediate property it enables -- a real reason grounded in how the "
    "model/training/kernel works, NOT a restatement of the goal; (c) EXPECTED EFFECT = for "
    "DIRECT_GAIN, the direction AND rough magnitude it should move the metric vs baseline; "
    "(d) ABLATION TEST = for DIRECT_GAIN, a concrete add-vs-remove experiment that CONFIRMS/FALSIFIES "
    "that effect; (e) FUNCTIONAL CHECK = for INDIRECT_GAIN, a "
    "MEASURABLE pass/fail criterion written as an explicit numeric comparison that proves the "
    "function was achieved (for example '|corr(C,S)| < 0.1' or 'mismatch_count == 0'). The check "
    "MUST contain one of <, <=, ==, >=, > followed by a numeric literal; do not substitute prose "
    "such as 'below the repository threshold'. A DIRECT_GAIN component with no mechanism or no "
    "executable scalar check is invalid: ablation_test must be exactly "
    "'signed_metric_improvement <operator> <finite number>' or "
    "'candidate_metric <operator> <finite number>', relative to the exact parent checkpoint. "
    "signed_metric_improvement is parent - candidate for minimization, candidate - parent "
    "for maximization; improvement is strictly positive in BOTH cases. "
    "Do not claim this scalar protocol verifies additional controls or repeated-run statistics. "
    "Unsupported mandatory experiments require formal plan/method revision, not deletion of "
    "their scientific requirements. A DIRECT_GAIN component with no "
    "falsifiable ablation, or a "
    "INDIRECT_GAIN component with no measurable check, is a guess (拍脑袋) and does NOT belong. Reply "
    "with ONE JSON object: {{\"approach\": \"<core idea, 1-3 sentences>\", \"changes_outline\": "
    "[\"<step>\", ...], \"why_it_should_win\": \"<mechanism vs baseline>\", \"assumptions\": "
    "[\"<assumption>\", ...], \"reference_sources\": [\"<exact allowed path or URL>\", ...], "
    "\"components\": [{{\"component_id\": \"<short-stable-id>\", "
    "\"name\": \"<component>\", \"depends_on\": [\"<component-id>\"], \"purpose\": \"<why>\", "
    "\"inputs\": [\"<required input or upstream output>\"], "
    "\"outputs\": [\"<observable output consumed or verified later>\"], "
    "\"acceptance_requires\": [\"<every transitive prerequisite component ID whose accepted output must coexist>\"], "
    "\"type\": \"direct_gain|indirect_gain\", \"mechanism\": \"<why it moves the metric, or what it enables>\", "
    "\"expected_effect\": \"<direct: direction + rough magnitude; indirect: measurable intermediate "
    "effect>\", \"ablation_test\": \"<direct: signed_metric_improvement > 0>\", "
    "\"functional_check\": \"<indirect: measurable pass/fail criterion with a threshold>\", "
    "\"implementation_check\": \"<code/diff/runtime evidence proving faithful implementation>\", "
    "\"optimization_space\": \"<task-permitted adjustable dimensions, without invented bounds>\", "
    "\"downstream_contract\": \"<how later components or final integration consume its output>\"}}, ...], "
    "\"critique_of_other\": [\"<concrete objection to the OTHER proposer's method>\", ...]}}. Same "
    "language as the goal."
)

_COMPONENT_PLAN_PROPOSER_SYSTEM = (
    "You are METHOD PROPOSER {label} in COMPONENT-PLAN scope. The scientific Method is already "
    "confirmed: preserve its approach, changes_outline, why_it_should_win, and assumptions. Your "
    "only task is to turn its components into an executable, scientifically testable DAG. Read "
    "every MOSS/Critic item and repair the cited component rather than rewriting the core Method. "
    "Before refining the DAG, compare every component's concrete intervention with the confirmed "
    "top-level approach. Every component must implement or enable that SAME approach; do "
    "not silently substitute a different algorithm, mechanism, hardware path, or optimization axis "
    "while leaving the top-level approach unchanged. If the frozen Method mentions repository "
    "inspection, environment preparation, or final evaluator submission as workflow, preserve its "
    "text but omit those as ComponentPlan nodes; bind genuine consumers to baseline/repository "
    "evidence instead. Omitting workflow nodes alone does not revise the scientific Method. "
    "A check that only reruns official correctness on an unchanged candidate is acceptance or "
    "final verification, not a separate indirect_gain component. If the confirmed Method lists "
    "such a check-only node, merge its mandatory check into the implementing component's "
    "acceptance using merged_from for the removed ID; do not invent a source edit for the check. "
    "For every component provide component_id, name, purpose, inputs, outputs, depends_on, the full "
    "transitive acceptance_requires closure, type (direct_gain or indirect_gain), mechanism, "
    "expected_effect, implementation_check, optimization_space, and downstream_contract. "
    "In optimization_space, name task-permitted choices that can still be tested. An initial "
    "implementation value is provisional unless the task contract or a necessary mechanism "
    "invariant fixes it. Do not copy an aspirational value into implementation_check or "
    "downstream_contract as a fidelity gate, and do not invent narrow numeric bounds. If a "
    "frozen top-level Method already imposes an unjustified value, request formal Method "
    "revision rather than silently relaxing it in this plan. "
    "When revising a plan, if inseparable prior components are genuinely combined, set the "
    "combined component's merged_from to their exact prior component IDs and explain in its "
    "mechanism how each intervention and check remains covered. Do not use merged_from to hide "
    "an unrelated deletion or transfer prior acceptance evidence to the new component. Only IDs "
    "in the confirmed Method may appear there, not conceptual submechanisms, proposed new nodes, "
    "or interim draft IDs. Leave merged_from empty when no confirmed node is removed. A "
    "direct_gain component also needs a concrete ablation_test against its parent checkpoint using "
    "the frozen official metric. The executable ablation_test must be exactly one comparison: "
    "signed_metric_improvement <operator> <finite number> or candidate_metric <operator> "
    "<finite number>, using <, <=, ==, >=, or >. The former is the direction-corrected "
    "parent/candidate delta; the latter is the candidate's official scalar. Both require a "
    "strict improvement over the exact parent checkpoint. depends_on determines this parent: "
    "a root uses the frozen baseline, while a dependent uses exactly its declared dependency "
    "checkpoints. A probe-only parent that cannot supply the primary task metric is not a "
    "valid direct-gain control merely because a separate baseline or incumbent metric exists. "
    "signed_metric_improvement is parent - candidate for minimization, candidate - parent "
    "for maximization; improvement is strictly positive in BOTH cases. Do not require it < 0 "
    "merely because the task minimizes its raw metric. "
    "Use signed_metric_improvement > 0 as the ordinary DIRECT_GAIN acceptance gate. A paper, "
    "reference implementation, leaderboard anchor, predicted effect size, or aspirational target "
    "belongs in expected_effect and is not a harder component pass/fail threshold. Use a stricter "
    "numeric gate only when the authoritative repository task contract itself makes that threshold "
    "mandatory; cite that contract in implementation_check. A legal measured improvement that "
    "misses an aspirational magnitude remains an improved observation and must be reconciled, not "
    "discarded in favor of repeating the same target. Do not move the same aspirational magnitude "
    "into implementation_check, outputs, or downstream_contract as a fidelity requirement. Those "
    "fields may require the actual intervention and authoritative correctness/feasibility gates, "
    "but may not require candidate_metric to reach a paper, reference, leaderboard, or predicted "
    "target unless the repository task contract itself mandates it. "
    "Do not conjoin correctness gates into ablation_test with and/or. Keep required official "
    "correctness gates explicit in implementation_check and runtime gate evidence; a scalar "
    "gain predicate never waives them. An earlier repair checklist is not authority to request "
    "an expression the host cannot execute. "
    "The measurement is the raw candidate_metric with a separately measured parent_metric; "
    "the host computes signed_metric_improvement. Do not require a delta in the raw candidate "
    "measurement in implementation_check, outputs, or downstream_contract. "
    "An infeasible parent/control is not a valid scalar performance comparator. "
    "When a stage first establishes a required feasibility condition, test the enabling property "
    "or formally repair the control/design before requiring direct gain. "
    "Do not change the official objective or fabricate a feasible baseline. This protocol does not execute "
    "additional controls or prose conditions. If the Method requires those, flag the unsupported "
    "contract for formal revision; never quietly drop its required experiments. "
    "Keep implementation procedures separate from the scalar acceptance predicate. "
    "An indirect_gain component instead needs a functional_check with "
    "the current node and its accepted ancestors sufficient to run the test. Do not compare a "
    "partial-stage runtime with a whole-pipeline baseline or invent an equivalent full score. "
    "Specify the scientific measurements, not the Runner-owned result envelope: do not redefine ar_result.json "
    "as a custom test report. Keep raw test outputs in separate evidence artifacts; the Runner's "
    "execution instructions define how to bind those measurements, correctness, and attempts. "
    "If the official metric requires unimplemented downstream stages, use an executable "
    "intermediate-property check or co-design those inseparable stages as one component. "
    "The functional_check must contain "
    "an explicit numeric comparison (<, <=, ==, >=, or > followed by a number) that measures its "
    "claimed intermediate property; it need not improve the final metric immediately. "
    "When the confirmed Method genuinely requires a measurement or diagnostic component, "
    "classify it by what it DOES, not its position: a terminal evaluator/join that only measures "
    "accepted parent outputs is indirect_gain too; measuring a gain does not create that gain. "
    "This allowance is for a Method-defined measurement or join, not a second run of the "
    "task's existing correctness gate on unchanged code. Attach that gate to the parent "
    "implementation_check and final verification, with no separate component attempt. "
    "distinguish measurement validity from the hypothesis being measured: acceptance must test "
    "whether the instrumentation measures the declared quantity correctly, not require a "
    "preferred observation. A valid observation that contradicts the hypothesis is feedback for "
    "the next decision, not an instrumentation failure to debug until the observation changes. "
    "For a component intended to CREATE an intermediate property, still require that property "
    "to hold; this distinction must not weaken its acceptance test or add diagnostic components. "
    "For each concrete acceptance example, derive the expected output for the actual test input "
    "from the declared semantics or an independent reference; do not copy a desired outcome into "
    "an example that does not satisfy its preconditions. "
    "Check algebraic identities at neutral weights and component boundary values: sum versus mean "
    "must preserve any claimed baseline loss scale. Evaluate rounding and schedule formulas, "
    "then use the same derived value in the mechanism, implementation, and acceptance window. "
    "Reject contradictory equalities or constants before implementation; do not ask the coder "
    "to choose which frozen requirement to violate. "
    "The host acceptance gate consumes exactly one scalar, so if several assertions are necessary, aggregate "
    "them explicitly (for example failed_check_count == 0) and put the individual assertions in "
    "implementation_check; never join several numeric comparisons with AND/OR. If acceptance "
    "depends on an upstream observation, define the branch conditions and expected outcomes in "
    "implementation_check, then expose one branch-aware failed_check_count == 0. Do not freeze "
    "one branch's threshold while promising another threshold in the component description; "
    "the Runner cannot replace the frozen threshold after observing the branch. Independent "
    "branches must stay isolated, and a join must depend on every parent output it consumes. "
    "Independence is semantic, not merely the ability to edit two components separately. If a "
    "component measures, selects, guards, configures, tunes, dispatches, or otherwise operates on "
    "the implementation produced by another component, it depends on that component's final "
    "accepted output and must not branch from an earlier shared ancestor. Trace every declared "
    "input to the component output or baseline state that actually supplies it before deciding that "
    "two nodes are independent. "
    "Before returning, audit every component's acceptance environment: identify the previously "
    "accepted component outputs that must coexist for this component to operate or be evaluated; "
    "put only immediate prerequisites in depends_on and their full transitive closure in "
    "acceptance_requires. Never repeat an ancestor as a direct dependency when another declared "
    "dependency already brings that ancestor into the closure. An ablation_test or functional_check may activate only the current "
    "component plus acceptance_requires. If it mentions, enables, or consumes another component, "
    "either declare that component as a genuine dependency or remove it from the test. Never pull "
    "in a later or unrelated component merely because the final combined Method will contain it. "
    "Apply every unresolved blocking repair supplied by MOSS/Critic; do not return an unchanged "
    "component whose dependency or acceptance contract was explicitly rejected. Do not turn the "
    "metric, budget, evaluator, baseline run, replication, or ablation procedure into a "
    "component. Do not add optional machinery merely to look detailed. Return ONE JSON object with "
    "components and critique_of_other only; the host preserves all frozen top-level Method fields. "
    "Keep each field concise, without repeating the review history. For an indirect_gain node, "
    "functional_check is ONLY the scalar gate (for example failed_check_count == 0); put the "
    "complete definition, sub-assertions and branch conditions in implementation_check. Never "
    "drop those conditions to shorten the gate."
)

_METHOD_CONVERGE_SYSTEM = (
    "You judge whether the TWO proposers have CONVERGED: do their two methods now describe the SAME "
    "approach (same core idea + same key components), with no material disagreement left? Reply with "
    "ONE JSON object: {\"converged\": true|false, \"reason\": \"<what still differs, or why agreed>\"}."
)

_COMPONENT_PLAN_CONVERGE_SYSTEM = (
    "You judge whether A1 and A2 have converged on the SAME ComponentPlan, not merely the same "
    "scientific Method. Return converged=true only when both plans agree on component IDs and "
    "direct/indirect roles, immediate depends_on edges, full acceptance_requires closures, the "
    "inputs and outputs relevant to those dependencies, each component's parent acceptance "
    "environment and acceptance meaning, and terminal integration dependencies. Judge agreement "
    "only; the dedicated Critic, not you, decides whether prior blocking problems were actually "
    "repaired. Reply with "
    "ONE JSON object: {\"converged\": true|false, \"reason\": \"<what still differs, or why agreed>\"}."
)

_METHOD_CRITIC_SYSTEM = (
    "You are the METHOD CRITIC -- a dedicated reviewer, NOT MOSS. You review either ONE agreed "
    "method (the proposers converged) or TWO competing methods + their disagreement (not converged; "
    "critique both + the crux). "
    "YOUR TARGET (this is your whole purpose -- do NOT drift into being a generic nitpicker): the "
    "method must be a DEFENSIBLE, REAL improvement on the TARGET METRIC vs the baseline -- meaning "
    "for every design choice one can argue, from REASONING / THEORY / EVIDENCE, that it will ACTUALLY "
    "move that metric. The FROZEN COMPARISON SEMANTICS in the context are authoritative: changing "
    "the metric, direction, budget type/duration, controls, evaluator, or aggregation is a DEEP "
    "comparison_integrity failure, never an improvement. An auxiliary ablation may clarify a "
    "mechanism but must not replace the primary comparison. An ablation or control run is an "
    "experiment, not a candidate component; its pass condition is faithful measurement, not "
    "whether the result supports the hoped-for hypothesis. Preserve a better ablated variant "
    "as a candidate for official comparison. Apply explicit algorithm bans by the "
    "operation, not its label: when a task forbids "
    "game-tree search, enumerating candidate moves and opponent replies to evaluate successor "
    "states remains search even at fixed depth without recursion or minimax. Ground any such "
    "judgment in the actual task clause and proposed computation. Read-only features of the "
    "current state, including cells near a legal action, are not by themselves successor-state "
    "enumeration; identify the actual forbidden operation before rejecting them. Reconcile any "
    "claimed numerical ceiling with bound evaluator observations, distinguishing qualified "
    "component evidence from accepted complete results. Decide whether the method meets "
    "the target, and surface only the choices that DON'T. "
    "A choice is UNSOUND (raise it) when it is arbitrary / unmotivated / cannot be argued to raise "
    "the metric: it has NO real MECHANISM, no valid acceptance test for its declared role, or an "
    "EXPECTED EFFECT that is hand-wavy. Also flag an initial implementation choice frozen "
    "as a Method requirement without "
    "a task rule or evidence that the mechanism needs it; ask to keep that choice adjustable "
    "in the ComponentPlan. For DIRECT_GAIN require a falsifiable add-vs-remove metric "
    "test. For INDIRECT_GAIN require a numeric check of the exact intermediate property it claims "
    "to produce; do not demand immediate final-metric gain. A component without the appropriate "
    "role-specific test is a guess (拍脑袋) -- exactly the problem to raise. "
    "If an earlier gain component is retained only as a control or revert checkpoint while a "
    "later component replaces its treatment code, reject a Method that requires both as "
    "implemented dependencies in the final candidate. Keep the earlier observation as "
    "comparison evidence, or identify its surviving treatment behavior. Do NOT be "
    "fooled by fluent-but-empty justification -- 'sounds reasonable' is not enough; there must be a "
    "TESTABLE claim. Tag each problem you DO raise with its closest category from: "
    + ", ".join(METHOD_CRITIC_RUBRIC) + " (this keeps the review on concrete, testable issues). "
    "分清主次: any method has infinitely many minor flaws; raise ONLY the FEW (at most 1-2) that "
    "actually UNDERMINE whether this method will defensibly improve the metric -- the ones you would "
    "NOT ship without fixing. Wording / naming / formatting / goal-restatement, and nitpicks that do "
    "NOT change whether the metric moves are SHALLOW -- do NOT raise them. If nothing truly "
    "undermines the target, return an EMPTY list and let it pass (that empty return is what lets a "
    "sound method converge). For EACH real problem give a concrete SUGGESTION -- ideally one that "
    "sharpens the component's mechanism or ablation so the experiment can verify it. Reply with ONE "
    "JSON object: {\"problems\": [{\"category\": \"<a lens from the rubric>\", \"issue\": \"<how this "
    "undermines a real, defensible improvement on the metric>\", \"suggestion\": \"<concrete "
    "fix/direction>\", \"component\": \"<which component, if any>\"}, ...]}. Same language as the goal."
)

_COMPONENT_PLAN_CRITIC_SYSTEM = (
    "You are the METHOD CRITIC in COMPONENT-PLAN scope. The core Method is already confirmed and "
    "must not be rewritten. Ask whether the proposed designs contain any DEEP problem, and review the ENTIRE "
    "ComponentPlan, not only its graph. The frozen BaselineResult may contain verbatim "
    "repository_contract sources. Treat those task rules as authoritative: reject a plan when "
    "the confirmed Method or any component uses a prohibited algorithm, changes a forbidden "
    "artifact, or violates a required execution semantic; name that concrete conflict so the "
    "outer loop can revise the Method. A fixed-depth enumeration of candidate moves and opponent "
    "replies over successor states is still game-tree search when the task bans that operation; "
    "a 'pattern check' label or lack of recursion does not make it permissible. Your FIRST gate is "
    "method-to-component coherence: compare "
    "the confirmed top-level approach with every component's concrete intervention. If the approach "
    "claims one algorithm, mechanism, hardware path, or optimization axis while the components "
    "implement a different one, reject the plan even when its DAG and local checks are internally "
    "consistent. When the audit context carries a reconciled_result_ref and measured repair "
    "feedback for an admissible improved off-method observation, compare the implicated component "
    "against that measured mechanism and host-bound diff. Reject an unchanged plan that restores "
    "the incompatible old mechanism while ignoring the retained improvement. Require the owning "
    "component and affected dependencies to describe and revalidate the measured mechanism, or set "
    "method_revision_required=true when that mechanism conflicts with the frozen top-level approach. "
    "Check that every requested repair is possible within the COMPONENT-PLAN "
    "scope: proposers may repair components, but approach, changes_outline, why_it_should_win, "
    "and assumptions are frozen and cannot be edited here. If a substantive contradiction "
    "cannot be resolved without changing a statement in those frozen fields, set "
    "method_revision_required=true and cite that statement plus the incompatible requirement "
    "in method_revision_reason; do not repeatedly tell CAP proposers to edit a frozen field. "
    "If an initial setting is frozen without a task rule or evidence that the mechanism "
    "requires it, request its removal from a CAP fidelity check, or request Method revision "
    "when the top-level Method itself freezes it. Keep legitimate tuning in optimization_space "
    "without inventing narrow bounds. "
    "Ordinary component design, dependency, or acceptance repairs that can satisfy the frozen "
    "Method stay within CAP and must not trigger that flag. A wording-only difference is not "
    "a substantive contradiction. If a gain-bearing ancestor is only a control or revert target "
    "and a successor replaces its treatment, the final candidate cannot implement both. Require "
    "Method revision to move the ancestor observation to comparison history, unless the plan "
    "identifies its surviving treatment in the final candidate with source evidence; changing "
    "an implementation_check alone cannot repair that dependency. Then inspect EACH "
    "component's purpose, design/mechanism, inputs, outputs, direct-vs-indirect role, expected "
    "effect, acceptance prerequisites, acceptance test, faithful-implementation check, remaining "
    "optimization space, and downstream contract. "
    "If components were merged, inspect merged_from against the prior component IDs and reject "
    "a mapping that does not preserve each omitted intervention and its test. Reject any source "
    "that is not an exact confirmed Method component ID, including submechanism names. Prior accepted "
    "evidence does not automatically validate the new combined node. "
    "Reject a node whose sole purpose is environment preparation, baseline reproduction, "
    "host-attempt registration, or result binding: these are Runner-owned workflow operations, "
    "not Method components, even when labeled indirect_gain. Ask CAP proposers to remove that "
    "node and bind its consumers to the existing baseline or the actual scientific prerequisite; "
    "do not invent a source edit merely to satisfy a nonempty diff check. Preserve genuine "
    "measurement or diagnostic components required by the confirmed Method; distinguish them "
    "by their scientific role, not by component names. "
    "A node that only reruns official correctness on an unchanged candidate has no new treatment "
    "behavior: require that check in its implementing component's acceptance or final verification, "
    "with merged_from preserving the old node ID when revising the plan. Do not waive the check. "
    "Check algebraic identities at neutral weights and component boundary values: sum versus mean "
    "must preserve any claimed baseline loss scale. Evaluate rounding and schedule formulas, "
    "then use the same derived value in the mechanism, implementation, and acceptance window. "
    "Reject contradictory equalities or constants before implementation; do not ask the coder "
    "to choose which frozen requirement to violate. "
    "Reject an unattainable direct-gain predicate against an infeasible parent/control: verify "
    "that both scalars describe solutions satisfying the objective's required conditions. "
    "A feasibility-enabling stage needs its actual property checked or a repaired control/design, "
    "not repeated optimization against a meaningless score. "
    "Do not change the official objective or fabricate a feasible baseline. "
    "Reject a vague acceptance test that merely "
    "restates the goal or cannot prove the declared phenomenon. For a measurement or diagnostic "
    "component required by the confirmed Method, reject a test that confuses measurement validity "
    "with confirmation of the measured hypothesis. A terminal measurement/join remains indirect_gain even without "
    "a downstream component; do not require it to improve the parent result merely because it "
    "reports the final metric. This does not authorize a second component that only reruns "
    "the task's existing correctness gate on unchanged code; require a merged acceptance check "
    "and final verification instead. Rejecting its instrumentation is distinct from judging the "
    "method's final performance. Valid contrary observations must inform "
    "subsequent decisions, not be classified as broken instrumentation. For each concrete acceptance "
    "example, independently derive the expected output for the actual test input and reject "
    "unsupported expected constants. Do not freeze a scientific conclusion or require ignoring a verified counterexample. "
    "A claimed bound must be checked against same-protocol feasible observations, with counts, "
    "indices, units and direction distinguished; literal text/constant checks do not prove it. "
    "Repair the faulty claim, not the contrary observation. Freeze the test procedure, not its "
    "empirical outcome. A failure branch in a downstream node is unreachable when that node "
    "requires the failed prerequisite to be accepted; handle the diagnostic at the failing "
    "node through the existing revision or negative-result path. "
    "A correct validator must reject an invalid object; do not "
    "require every object supplied to that validator to pass. A component intended to "
    "CREATE an intermediate property must still demonstrate that property; do not weaken such "
    "checks or introduce extra diagnostic components. For every DIRECT_GAIN component, "
    "also verify that its permitted intervention can actually change the frozen official metric "
    "relative to its parent checkpoint. Reject an acceptance "
    "test that needs an unimplemented downstream stage, or compares a partial-stage measurement "
    "with the whole-pipeline metric (including an invented conversion). Require a runnable "
    "intermediate-property check or a component boundary covering the inseparable stages. "
    "Reject a mutually impossible contract when the fidelity "
    "requirements freeze every metric-determining degree of freedom while the acceptance test "
    "still requires that metric to improve; reclassify a genuine enabling property as "
    "INDIRECT_GAIN or merge it into the appropriate downstream gain component. Then inspect "
    "dependencies, branch isolation, join nodes, and "
    "whether the terminal branches together implement the whole confirmed Method. "
    "Before returning an empty problems list, re-check every unresolved blocking repair from the "
    "previous ComponentPlan round and confirm that the cited field was visibly changed. For every "
    "component, trace what implementation it measures, selects, guards, configures, tunes, "
    "dispatches, or consumes. Reject a sibling branch when that operation actually needs the "
    "sibling's final accepted output; require a dependency edge so the later component is built and "
    "tested on the correct checkpoint. When sibling checkpoints leave different content at the "
    "same path and a join needs both, reject that join unless the plan provides an executable "
    "sequence, reviewed combined node, or distinct paths with an explicit interface; the Runner "
    "does not auto-merge snapshots. Then compare its acceptance test with its "
    "acceptance_requires closure: reject a test "
    "that activates, consumes, or assumes an undeclared component. An independent component must "
    "be tested from the baseline parent environment; a dependent component must be tested with "
    "exactly its declared dependency closure, with only immediate prerequisites in depends_on and "
    "no redundant ancestor listed beside its descendant. Detect undeclared sibling changes by their actual "
    "parameter, code-axis, input/output, or implementation-check wording too; do not require the "
    "acceptance text to spell the sibling component ID before flagging the leak. Merely naming "
    "a sibling is not proof of activation: distinguish introducing its "
    "mechanism from holding a baseline setting fixed or explicitly excluding that component. "
    "Ground a dependency objection in what the test actually enables or consumes. Agreement between "
    "A1 and A2 is not evidence that the "
    "plan is correct. Raise every blocking DEEP problem needed to make the plan executable and "
    "scientifically testable, but consolidate shared root causes into at most three concrete "
    "problems. Before returning problems, recheck each one against the final proposal: if your "
    "own analysis concludes the conflict is resolved or no repair is needed, omit that problem "
    "instead of returning a self-canceling objection. ComponentPlan defines scientific component "
    "semantics, dependencies, and acceptance "
    "meaning; the later ExperimentPlan and Runner choose exact shell commands, paths, result-file "
    "layout, and operational recovery. Do not require exact CLI strings, hardcoded workspace paths, "
    "a particular result JSON schema, repeated seeds, or statistical replication unless the frozen "
    "task contract or confirmed Method explicitly requires them. One real repository-owned official "
    "metric comparison uses the executable ablation_test aliases signed_metric_improvement or "
    "candidate_metric; do not approve a task-specific variable name that the host cannot parse. "
    "signed_metric_improvement is parent - candidate for minimization, candidate - parent "
    "for maximization; improvement is strictly positive in BOTH cases. Reject a predicate "
    "that requires this statistic to be nonpositive while demanding direct gain. "
    "Ordinarily require signed_metric_improvement > 0. Reject a stricter threshold derived only "
    "from a paper result, reference/leaderboard anchor, predicted magnitude, or aspirational Method "
    "target: those belong in expected_effect, not the component acceptance gate. Approve a stricter "
    "gate only when the authoritative repository task contract explicitly makes it mandatory and "
    "implementation_check cites that contract. Do not discard a legal measured improvement merely "
    "because it misses an aspirational magnitude; require reconciliation of the measured result. "
    "Also reject the same aspirational/reference magnitude when it is smuggled into "
    "implementation_check, outputs, or downstream_contract as an implementation-fidelity gate. "
    "Those fields may enforce the actual intervention and authoritative correctness/feasibility "
    "contract, but not a non-mandatory performance target. "
    "Require exactly one numeric comparison in ablation_test, not a conjunction with and/or "
    "or another variable on the right-hand side. Do not conjoin correctness gates into "
    "ablation_test: keep required official correctness gates explicit in implementation_check "
    "and runtime gate evidence; a scalar gain predicate never waives them. Critique a missing "
    "correctness check where it belongs, not by requesting an unsupported scalar expression. "
    "If a required gate is reserved for final evaluation, do not demand or approve its "
    "measurement during component selection. Require a locally testable intermediate contract "
    "or defer the claim to complete-method verification. A final-only note does not repair an "
    "implementation_check that also demands running a hidden-data gate during the component; "
    "reject that contradiction. Check that any proposed search grid can plausibly finish within "
    "the task's declared time budget, including its trial count and per-trial work. "
    "When citing a historical metric, "
    "check its candidate code, evaluation split, and protocol; never rename a held-out test "
    "result as development validation or transfer it to a different implementation. "
    "An earlier repair checklist is not authority to request an expression the host cannot execute. "
    "The measurement is the raw candidate_metric with a separately measured parent_metric; "
    "the host computes signed_metric_improvement. Do not require a delta in the raw candidate "
    "measurement: flag an implementation_check, output, or downstream_contract that confuses "
    "this value with the derived acceptance statistic; repair the plan, not the official metric. "
    "For indirect_gain, a defined functional_check is sufficient; an additional ablation_test "
    "is not required. For direct_gain, do not require a separate functional_check duplicating "
    "procedural checks already defined in implementation_check. If missing source detail can "
    "be resolved by the plan's explicit implementation-time inspection and failure condition, "
    "do not call that alone a present design defect or demand unavailable source quotations. "
    "Still reject a known repository conflict or a mechanism with no executable way to test it. "
    "If extra control experiments are required by the frozen Method but unsupported by the "
    "acceptance protocol, report method_revision_required rather than asking CAP proposers to "
    "repeat an impossible field edit or silently drop those experiments. "
    "A diagnostic ablation is an experiment, not an indirect_gain candidate component. If the "
    "confirmed Method makes its hoped-for result a functional_check, report "
    "method_revision_required: either outcome can be informative, and a better ablated variant "
    "must remain eligible for official comparison. Do not pass such a check merely because its "
    "expected direction sounds plausible. One official metric comparison is a valid direct-gain "
    "check; noise analysis and replication are outside this "
    "plan audit. Continue reviewing the whole plan while "
    "ignoring wording and formatting nits. A deep problem is: a component does not serve the "
    "confirmed method; its input/output or mechanism is incoherent; its test cannot establish the "
    "claimed effect; a required component or dependency is missing; an unrelated branch is pulled "
    "into a local test; a dependency/order is wrong or cyclic; a join cannot materialize all parent "
    "outputs; a DIRECT_GAIN component lacks a real add-vs-remove metric test; or an INDIRECT_GAIN "
    "component's numeric check does not measure the exact intermediate property it claims (for "
    "example orthogonality). For INDIRECT_GAIN, that intermediate-property check is sufficient: "
    "it must expose one aggregate scalar threshold; if several assertions are needed, require an "
    "explicit aggregate such as failed_check_count == 0 and reject a conjunction whose first "
    "number could be mistaken for the whole contract. "
    "A named check such as measured_error <= 0.01 is valid: implementation_check may define the measurement "
    "and its runtime evidence. The host binds the runtime ablation_report actual to that component's "
    "numeric right-hand threshold; it does not execute functional_check as Python or require the "
    "left-hand name to be a pre-existing host variable. Reject an undefined measurement or a "
    "non-numeric right-hand threshold, not a named measurement with a concrete test definition. "
    "For conditional components, compare the frozen functional_check with every branch stated "
    "in purpose, mechanism, expected_effect, and implementation_check. Reject a threshold that "
    "accepts only one declared branch while another valid branch is promised elsewhere. Require "
    "one branch-aware aggregate whose definition checks the actual upstream observation and "
    "that branch's required outcome; do not ask the Runner to replace the frozen threshold. "
    "do NOT demand an add-vs-remove performance ablation, immediate target-metric gain, or a "
    "separate proof that its runtime overhead is negligible. A correctness guard may be a valid "
    "INDIRECT_GAIN component when its numeric check proves the property required by a downstream "
    "component. Judge aggregate overhead and the final target metric only at the terminal integration "
    "component, unless runtime itself is the component's declared intermediate property. Every "
    "problem must name the exact component and existing field or "
    "dependency that fails; omit hypothetical risks and optional improvements that are not required "
    "to execute or test the confirmed Method. If there is no such problem, return an empty list. "
    "Do not ask merely "
    "whether the plan is detailed enough or whether a problem could later be located. Reply with "
    "ONE JSON object: "
    "{\"method_revision_required\": true|false, \"method_revision_reason\": \"<only when the "
    "frozen top-level Method itself conflicts with the repository contract, contains mutually "
    "inconsistent requirements that no single ComponentPlan can satisfy, or cannot be implemented "
    "without changing its core mechanism>\", \"problems\": [{\"category\": "
    "\"component_soundness|ablation_verifiability|feasibility|"
    "failure_mode\", \"issue\": \"<the actual deep problem>\", \"suggestion\": \"<concrete "
    "fix>\", \"component\": \"<component ID or name>\"}, ...]}."
)

_METHOD_MOSS_RELAY_SYSTEM = (
    "You are MOSS, the coordinator (the human-in-the-loop role) -- NOT a judge. You do NOT decide "
    "whether the critic is right or whether its problems are 'worth' another round; that is not your "
    "role. Your role: UNDERSTAND the critic's points and RELAY them to the proposers (who have "
    "converged -- address them as one team) as GUIDANCE that moves them to a better method next "
    "round. Be tactful and constructive, not accusatory: frame it as 'this may not yet bring a "
    "systematic improvement, because <the crux -- e.g. a component lacks a real mechanism / a "
    "falsifiable ablation / a defensible expected effect>', then hand them the concrete points to "
    "strengthen. Keep it short and directional. Reply with ONE JSON object: {\"guidance\": \"<the "
    "guiding note to the proposers, in the goal's language>\"}."
)

_COMPONENT_PLAN_MOSS_RELAY_SYSTEM = (
    "You are MOSS, the coordinator for COMPONENT-PLAN deliberation, not the judge. Preserve the "
    "confirmed scientific Method. Consolidate the Critic's concrete component/DAG findings into a "
    "short actionable checklist for both proposers: group duplicate findings by component, name the "
    "field or dependency to repair, and retain every genuinely blocking issue. Do not invent new "
    "components, new experiments, or a different method. Do not overrule executable schema/parser errors "
    "or call a rejected expression valid; relay the exact supported syntax without changing the "
    "scientific criterion. independently_executable means testable with its declared ancestors present, "
    "not dependency-free: a terminal measurement join is not automatically non-executable. A root component "
    "with all inputs bound to a valid frozen baseline has parent_control_available=true because the baseline "
    "is its runnable parent/control; do not reinterpret this field as requiring an ancestor checkpoint. "
    "If a semantic finding is disputed, ask the proposers to supply evidence for the Critic's next "
    "review; do not dismiss it yourself or tell them to preserve the rejected field unchanged. "
    "Reply with ONE JSON object: "
    "{\"guidance\": \"<component-plan repair checklist>\"}."
)

_METHOD_MOSS_GUIDE_SYSTEM = (
    "You are MOSS, the coordinator (human-in-the-loop) -- NOT a judge, and NOT a mechanical relay "
    "(if you only shuttled messages verbatim you would be pointless -- that could just be "
    "hard-coded). The two proposers have NOT converged this round. UNDERSTAND both takes and give a "
    "short GUIDING note that moves them TOWARD agreement on the SINGLE best method for the metric: "
    "point out where the other side has the stronger idea worth adopting on the specific crux, and "
    "nudge each to concede its weaker parts. Do NOT re-author their methods; just guide. Reply with "
    "ONE JSON object: {\"guidance\": \"<the guiding note to resolve the crux and converge, in the "
    "goal's language>\"}."
)


@dataclass
class MethodDebateResult:
    method: dict
    report: dict


def _method_ctx(problem_model: dict | None, literature_brief: dict | None,
                user_idea: str | None, baseline_result: dict | None = None,
                evaluation_semantics: dict | None = None,
                component_plan_audit: bool = False,
                component_plan_revision_feedback: str | None = None,
                prior_failure_context: dict | None = None) -> str:
    pm = problem_model or {}
    goal = pm.get("research_goal") or pm.get("goal") or "(goal unspecified)"
    metric = (pm.get("metric") or pm.get("target_metric") or pm.get("metric_name")
              or pm.get("objective") or "(metric unspecified)")
    baseline = (pm.get("baseline") or pm.get("baseline_hint") or pm.get("baseline_name") or "")
    lit = str((literature_brief or {}).get("summary") or "")[:1500]
    idea = f"\n\nUser idea to build on:\n{str(user_idea).strip()}" if user_idea else ""
    base_line = f"\nBaseline to beat: {baseline}" if baseline else ""
    frozen = ""
    if baseline_result:
        frozen = ("\n\nAuthoritative frozen BaselineResult from the real repository run "
                  "(do not replace it with a generic imagined baseline):\n"
                  + json.dumps(baseline_result, ensure_ascii=False, sort_keys=True))
    semantic_anchor = ""
    if evaluation_semantics:
        if evaluation_semantics.get("comparison_mode") == "repository_defined":
            semantic_anchor = (
                "\n\nFROZEN COMPARISON SEMANTICS: preserve the official metric, direction, "
                "evaluator and comparison protocol. For editable treatment settings, the "
                "explicit user boundary and verbatim repository_contract task rules override "
                "inferred conditions_to_hold_constant. Cite the task source when correcting "
                "a conflicting inferred restriction; do not reject a Method on that basis alone:\n"
                + json.dumps(evaluation_semantics, ensure_ascii=False, sort_keys=True)
            )
        else:
            semantic_anchor = (
                "\n\nFROZEN COMPARISON SEMANTICS (authoritative; improve the treatment without "
                "changing these conditions):\n"
                + json.dumps(evaluation_semantics, ensure_ascii=False, sort_keys=True)
            )
    failure_anchor = ""
    if prior_failure_context:
        failure_anchor = (
            "\n\nPERSISTED PRIOR-METHOD DIAGNOSIS (evaluate against the current task, "
            "do not treat it as a categorical ban):\n"
            + json.dumps(prior_failure_context, ensure_ascii=False, sort_keys=True)
            + "\nAn improvement review or revision request does not establish that the prior "
              "method or its mechanisms failed. Preserve supported mechanisms and distinguish "
              "measured success, measured failure, and unresolved hypotheses using the cited "
              "evidence. A null or missing metric is missing measurement evidence, not zero gain "
              "or proof that hardware lacks a capability. A scientific no-gain conclusion requires "
              "bound measurements for the exact evaluated mechanism; planned changes remain untested "
              "until evaluated. method_revision_evidence.modify lists proposals, not executed treatments. "
              "When draft_method_evaluation.has_bound_run_record is false, the draft has no bound "
              "run evidence; do not transfer evaluated_result's outcome from its recorded Method to "
              "the new draft. Independent explicit failed gates or execution errors remain negative evidence. "
              "Only when cited evidence establishes a mechanism's failure, if a "
              "proposal reuses that mechanism, identify new task evidence "
              "that changes its failure condition and a test distinguishing it from the "
              "failed version. Without such evidence, remove that failed mechanism. Preserve the "
              "frozen task contract and comparison."
        )
    audit_scope = ""
    if component_plan_audit:
        audit_scope = (
            "\n\nCOMPONENT-PLAN AUDIT (authoritative scope): the core Method above is already "
            "confirmed. Preserve its approach. Refine every component's purpose, mechanism, inputs, "
            "outputs, full transitive acceptance prerequisites, implementation evidence, optimization space, "
            "downstream interface, dependency order, and acceptance contract into an executable ComponentPlan. "
            "The dependency graph is operational: a component is tested with exactly its transitive "
            "prerequisites, independent branches remain isolated, and a join names every immediate "
            "parent whose accepted output must coexist. In component_checks, independently_executable "
            "means implementable and testable WITH its declared transitive prerequisites, not without "
            "dependencies. Consuming an accepted ancestor output does not make this flag false; "
            "set it false only when the component cannot be separately implemented or checked even "
            "in that dependency environment. This is distinct from branch independence. "
            "MOSS must preserve this meaning when relaying repairs. Treat branches as independent only when "
            "neither measures, selects, guards, configures, tunes, dispatches, or consumes the "
            "other's final accepted implementation. A DIRECT_GAIN component must "
            "state how its parent-checkpoint metric is tested. An INDIRECT_GAIN component must "
            "name the concrete intermediate property it enables and a numeric check of that exact "
            "property (for example, orthogonality_error <= 0.01). Reject vague proxy checks."
        )
        if component_plan_revision_feedback:
            audit_scope += (
                "\n\nCOMPONENT-PLAN OR IMPLEMENTATION EVIDENCE TO REVIEW (authoritative):\n"
                + str(component_plan_revision_feedback)
                + "\nFirst decide whether the evidence identifies a defect in the plan or only "
                  "in its implementation. Repair an implicated dependency, parent checkpoint, "
                  "component boundary, or acceptance contract only when the plan is defective; "
                  "otherwise retain the legal plan and state the implementation correction. "
                  "Preserve already accepted components whose own contracts "
                  "remain unchanged. For this revision, copy every unaffected component object verbatim "
                  "from the current plan, including IDs, acceptance text, and downstream descriptions; "
                  "do not paraphrase, clarify, or globally rename fields in unrelated components. "
                  "Checkpoint inheritance requires exact contract equality, so cosmetic rewrites "
                  "can force already completed experiments to run again. If a component or its "
                  "dependency context genuinely must change, state the concrete reason for that "
                  "change in the review; never preserve incompatible evidence merely to avoid a rerun. "
                  "A file-path collision is not by itself a scientific dependency or a duplicate "
                  "component checkpoint. If independent validation helpers share a path, plan "
                  "component-scoped helper paths and update their invocation commands; do not "
                  "discard another parent's code by selecting the globally latest checkpoint. "
                  "When a join needs two accepted sibling checkpoints that leave different content "
                  "at the same path, the Runner cannot automatically merge those snapshots. "
                  "Make the implementations coexist through an executable sequence, a reviewed "
                  "combined component, or distinct paths with an explicit interface; a join with "
                  "both conflicting parents is not an executable repair. "
                  "Adding a join alone cannot resolve a conflict that prevents that join's own "
                  "parent checkpoints from being materialized. "
                  "If review identifies premature downstream behavior, distinguish an acceptance-only "
                  "diagnostic from treatment behavior. A check needed to validate the current node "
                  "belongs to its acceptance procedure, not necessarily to a separate later node. "
                  "When treatment and its prerequisite cannot be implemented and tested separately, "
                  "revise the implicated ownership/dependencies or merge those tightly coupled nodes "
                  "with merged_from listing the exact old IDs, and specify a joint check. "
                  "Do not repeatedly keep the same conflicting boundary "
                  "or discard legal measured behavior solely because the previous plan assigned it "
                  "elsewhere. This does not waive task rules or prove unmeasured components. "
                  "When the failure feedback identifies an admissible improved off-method "
                  "observation, reconcile the owning component with the measured mechanism and "
                  "host-bound diff rather than restoring an incompatible old mechanism and merely "
                  "rerunning it. Revalidate every dependency affected by that substantive change. "
                  "If absorbing the measured mechanism would require changing the frozen top-level "
                  "approach, preserve the evidence and let the Critic request formal Method revision; "
                  "do not claim that an unchanged ComponentPlan completed reconciliation. "
                  "Do not replace the core Method merely to repair this plan."
            )
    return (f"Research goal:\n{goal}\n"
            f"TARGET METRIC (what a 'real improvement' must move, vs the baseline): {metric}{base_line}"
            f"\n\nLiterature context:\n{lit or '(none)'}{idea}{frozen}{semantic_anchor}{failure_anchor}"
            f"{audit_scope}")


def _method_str(m: dict) -> str:
    return json.dumps({k: (m or {}).get(k) for k in
                       ("approach", "changes_outline", "why_it_should_win", "assumptions",
                        "components")},
                      ensure_ascii=False)


def _norm_component(c: dict) -> dict:
    """Normalize ONE component contract, filling every contract field with a default so the
    experiment layer can rely on them. Legacy gain/functional names are normalized to the current
    direct_gain/indirect_gain vocabulary; functional_check defaults empty."""
    ctype = str(c.get("type") or "").strip().lower()
    ctype = {"gain": "direct_gain", "functional": "indirect_gain"}.get(ctype, ctype)
    if ctype not in ("direct_gain", "indirect_gain"):
        ctype = "direct_gain"
    raw_dependencies = c.get("depends_on") or []
    if not isinstance(raw_dependencies, list):
        raw_dependencies = []
    out = dict(c)
    out.update({
        "name": str(c.get("name") or "").strip(),
        "component_id": str(c.get("component_id") or c.get("name") or "").strip(),
        "depends_on": [str(item).strip() for item in raw_dependencies
                       if str(item).strip()],
        "purpose": str(c.get("purpose") or "").strip(),
        "inputs": [str(item).strip() for item in (c.get("inputs") or [])
                   if str(item).strip()],
        "outputs": [str(item).strip() for item in (c.get("outputs") or [])
                    if str(item).strip()],
        "acceptance_requires": [
            str(item).strip() for item in (c.get("acceptance_requires") or [])
            if str(item).strip()
        ],
        "type": ctype,
        "mechanism": str(c.get("mechanism") or "").strip(),
        "expected_effect": str(c.get("expected_effect") or "").strip(),
        "ablation_test": str(c.get("ablation_test") or "").strip(),
        "functional_check": str(c.get("functional_check") or "").strip(),
        "implementation_check": str(c.get("implementation_check") or "").strip(),
        "optimization_space": str(c.get("optimization_space") or "").strip(),
        "downstream_contract": str(c.get("downstream_contract") or "").strip(),
    })
    return out


def _norm_method(obj: dict) -> dict:
    """Normalize a proposer reply into a method dict with all keys present (incl. components +
    their ablation checkpoints). Pure; never raises."""
    obj = obj or {}
    return {"approach": str(obj.get("approach") or "").strip(),
            **({"reconciled_result_ref": obj["reconciled_result_ref"]} if obj.get("reconciled_result_ref") else {}),
            "changes_outline": obj.get("changes_outline") or [],
            "why_it_should_win": str(obj.get("why_it_should_win") or ""),
            "assumptions": obj.get("assumptions") or [],
            "reference_sources": obj.get("reference_sources") or [],
            "components": [_norm_component(c) for c in (obj.get("components") or [])
                           if isinstance(c, dict)]}


def _propose(llm, sys_fn, label, ctx, own_method, other_method, feedback, *,
             system_template: str = _METHOD_PROPOSER_SYSTEM,
             max_tokens: int | None = None, require_valid_response: bool = False):
    """ONE proposer turn: revise OWN method + critique the OTHER. Returns (method, critique_list).
    Ordinary method proposals retain fail-soft behavior. CAP proposals retry one unusable
    response, then surface the protocol failure rather than review the unchanged fallback."""
    # MOSS-MEDIATED (老师 2026-08-07): the two proposers NEVER talk directly. MOSS (the human role)
    # relays the OTHER's LATEST method to you, re-anchored to the ORIGINAL goal, as a colleague's
    # reference idea -- so every round has fresh external re-anchoring (anti-drift). You only ever
    # respond to MOSS; adopt what's stronger, don't debate a rival.
    user = (f"{ctx}\n\nYour current method (your latest):\n{_method_str(own_method)}\n\n"
            f"MOSS relays — a colleague's current approach, for you to REFERENCE (a friend's idea, "
            f"NOT a rival to defeat). Keep the ORIGINAL goal above as the anchor; adopt what is "
            f"stronger and converge toward one shared best method:\n{_method_str(other_method)}")
    if feedback:
        user += f"\n\nMOSS feedback: {feedback}"
    for attempt in range(2 if require_valid_response else 1):
        obj = _chat_json(llm, sys_fn(system_template.format(label=label)), user,
                         max_tokens=(max(max_tokens, min(max_tokens * 2, 65536))
                                     if attempt and max_tokens else max_tokens))
        if (require_valid_response and isinstance(obj, dict)
                and isinstance(obj.get("components"), list) and obj["components"]
                and all(isinstance(c, dict) for c in obj["components"])):
            # CAP responses need not re-emit the frozen scientific Method.
            obj = {**own_method, **obj}
        if isinstance(obj, dict) and str(obj.get("approach") or "").strip():
            break
        if not require_valid_response:
            return dict(own_method), []
        logger.warning("[method-debate] %s unusable CAP proposal, attempt %d", label, attempt + 1)
        user += ("\nYour previous proposal was unavailable or not usable valid JSON. "
                 "Return one complete JSON object with the revised components and critique_of_other. "
                 "The host already holds the frozen approach; do not repeat the Method or review "
                 "history. Keep descriptions concise and do not omit requested corrections.")
    else:
        error = DeliberationLLMError(
            f"ComponentPlan proposer {label} produced no usable JSON after two calls; "
            "this is a proposal protocol failure, not a semantic rejection of the prior plan.")
        error.draft_method = dict(own_method)
        raise error
    critiques = obj.get("critique_of_other") or []
    if isinstance(critiques, str):
        critiques = [critiques]
    return _norm_method(obj), [str(x) for x in critiques if str(x).strip()]


def _rubric_deep_problems(problems) -> list:
    """Keep ONLY problems whose category is in METHOD_CRITIC_RUBRIC -- the depth gate that stops the
    critic from stalling on shallow (wording/format/naming) nits. Keeps each problem's SUGGESTION so
    it can be fed to the proposers next round. Pure."""
    out = []
    for p in problems or []:
        if isinstance(p, dict) and str(p.get("category") or "") in METHOD_CRITIC_RUBRIC \
                and str(p.get("issue") or "").strip():
            out.append({"category": p["category"], "issue": str(p["issue"]).strip(),
                        "suggestion": str(p.get("suggestion") or "").strip(),
                        "component": str(p.get("component") or "").strip()})
    return out


def _critic_field_claim_conflicts(problem: dict, method: dict) -> bool:
    """Reject a Critic's quoted component field only when it contradicts the current draft."""
    component_id = str(problem.get("component") or "")
    component = next((c for c in method.get("components") or []
                      if str(c.get("component_id") or "") == component_id), None)
    if not component:
        return False
    issue = str(problem.get("issue") or "")
    role = component.get("type")
    optional_check = "ablation_test" if role == "indirect_gain" else "functional_check"
    required_check = "functional_check" if role == "indirect_gain" else "ablation_test"
    if (role in ("direct_gain", "indirect_gain")
            and str(component.get(required_check) or "").strip()
            and not str(component.get(optional_check) or "").strip()
            and issue.startswith(f"{component_id}.{optional_check} ")
            and "empty string" in issue.lower()
            and "json null" in issue.lower()):
        # The host uses the role's required check; an empty optional check is
        # not a scientific or executable-contract defect.
        return True
    for field in ("depends_on", "acceptance_requires"):
        pattern = rf"\b{re.escape(component_id)}\.{field}\s+(?:is(?: now)?|lists|declares)\s*(\[[^\]\n]*\])"
        for match in re.finditer(pattern, issue):
            try:
                claimed = ast.literal_eval(match.group(1))
            except (SyntaxError, ValueError):
                continue
            actual = component.get(field) or []
            if (isinstance(claimed, list) and all(isinstance(x, str) for x in claimed)
                    and set(claimed) != set(actual)):
                return True
    inputs = component.get("inputs") or []
    pattern = (rf"\b{re.escape(component_id)}\.inputs\[(\d+)\][^\n]*?\btoken\s+"
               r"(['\"])([^'\"\n]+)\2\s+in the input string")
    for match in re.finditer(pattern, issue):
        index = int(match.group(1))
        if index < len(inputs) and match.group(3) not in str(inputs[index]):
            return True
    return False


def _critic_feedback(deep: list) -> str:
    """Render the critic's deep problems + suggestions as feedback the proposers act on next round."""
    if not deep:
        return ""
    parts = [f"{p['issue']}（建议：{p['suggestion'] or '自行改进'}）" for p in deep]
    return "critic 的深度问题（下一轮请解决）：" + "；".join(parts)


def _component_contract_problem(
    method: dict, *, require_design_fields: bool = False,
    confirmed_component_ids: set[str] | None = None,
) -> Exception | None:
    """Return the runner's deterministic ComponentPlan error, if any.

    The LLM critic judges scientific soundness; this small harness only checks that the agreed
    contracts can actually be scheduled and evaluated by the existing runner.
    """
    try:
        from autoresearch.experiments.repo_improve.component_plan import (
            build_component_plan, validate_component_acceptance)

        plan = build_component_plan(method, require_design_fields=require_design_fields)
        if require_design_fields:
            validate_component_acceptance(plan)
        if confirmed_component_ids is not None:
            for component in method.get("components") or []:
                merged_from = component.get("merged_from") or []
                if isinstance(merged_from, str):
                    merged_from = [merged_from.strip()]
                if not isinstance(merged_from, list) or not all(
                    isinstance(item, str) and item.strip() for item in merged_from
                ):
                    raise ValueError("merged_from must list confirmed Method component IDs")
                for source_id in merged_from:
                    if source_id not in confirmed_component_ids:
                        raise ValueError(
                            f"merged_from cites unknown confirmed Method component ID: {source_id}; "
                            f"valid IDs: {sorted(confirmed_component_ids)}")
    except Exception as exc:  # the builder exposes the authoritative executable contract
        return exc
    return None


def _source_quote_texts(source: str):
    """Original source and exact JSON string values, never joined or unescaped prose."""
    yield source
    try:
        pending = [json.loads(source)]
    except (ValueError, TypeError):
        return
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            yield value
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def _source_contains_quote(source: str, quote: str) -> bool:
    """Accept exact source bytes or an exact string value in valid JSON, not paraphrases."""
    return any(quote in text for text in _source_quote_texts(source))


def _restore_omitted_quote_lines(source: str, quote: str) -> str | None:
    """Restore a unique, ordered excerpt; never repair values or choose a premise.

    Only whole, exact lines in one short source paragraph qualify. Ambiguous,
    reordered, cross-field JSON and cross-paragraph quotations still need review.
    """
    requested = quote.splitlines(keepends=True)
    if len(requested) < 2 or len(quote) > 512 or any(not s.strip() for s in requested):
        return None
    requested = [s.rstrip("\r\n") for s in requested]
    candidates = set()
    for text in _source_quote_texts(source):
        lines = text.splitlines(keepends=True)
        positions = []
        for wanted in requested:
            matches = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == wanted]
            if len(matches) != 1:
                break
            positions.append(matches[0])
        if len(positions) != len(requested) or any(a >= b for a, b in zip(positions, positions[1:])):
            continue
        span = lines[positions[0]:positions[-1] + 1]
        restored = "".join(span).rstrip("\r\n")
        if len(restored) <= 512 and all(line.strip() for line in span):
            candidates.add(restored)
    return candidates.pop() if len(candidates) == 1 else None


def _repository_check_entries(review: dict) -> list[dict]:
    checks = review.get("repository_checks")
    if isinstance(checks, dict):
        return [{"topic": topic, **entry} for topic, entry in checks.items()
                if isinstance(entry, dict)]
    return checks if isinstance(checks, list) else []


def _repository_review_problems(critic: dict, baseline: dict | None, method: dict,
                                evaluation_semantics: dict | None = None) -> list[dict]:
    """A clean critique must include checkable repository and node-boundary evidence."""
    baseline = baseline or {}
    sources = (baseline.get("repository_contract") or {}).get("sources") or []
    if not sources and baseline.get("source_kind") != "live_repo":
        return []
    evidence = {str(s.get("path")): str(s.get("content") or "")
                for s in sources if isinstance(s, dict)}
    evidence["baseline_result"] = json.dumps(baseline, ensure_ascii=False, sort_keys=True)
    if evaluation_semantics:
        evidence["evaluation_semantics"] = json.dumps(
            evaluation_semantics, ensure_ascii=False, sort_keys=True)
    problems = []
    from autoresearch.experiments.repo_improve.source_policy import source_findings
    for source in source_findings(method, baseline.get("repository_contract")):
        if not source["allowed"]:
            problems.append({"category": "feasibility", "component": "",
                             "issue": f"Method uses prohibited reference material: {source['path']}",
                             "suggestion": "Do not freeze this plan or claim independent work. Preserve exposure history and revise using allowed sources."})
    required = {"task_direction", "interface_and_inputs", "existing_mechanism", "resource_scale",
                "measurement_protocol", "artifact_scope", "source_boundary"}
    checks = _repository_check_entries(critic)
    for topic in sorted(required):
        matches = [c for c in checks if isinstance(c, dict) and c.get("topic") == topic]
        check = matches[0] if len(matches) == 1 else {}
        quote = check.get("quote")
        source = evidence.get(str(check.get("source_path")), "")
        if isinstance(quote, str) and quote.strip() and not _source_contains_quote(source, quote):
            restored = _restore_omitted_quote_lines(source, quote)
            if restored is not None:
                # Keep the submitted excerpt for audit and expose ALL intervening
                # text. This fixes citation transport, not the reviewer's judgment.
                check["quote_before_line_restoration"] = quote
                check["quote"] = quote = restored
        grounded = (isinstance(quote, str) and bool(quote.strip())
                    and _source_contains_quote(evidence.get(str(check.get("source_path")), ""), quote))
        if not grounded:
            # Give the reviewer actual nearby text to copy, not only its own bad
            # quote again. Similarity locates a repair hint; it NEVER grounds a claim.
            from difflib import SequenceMatcher
            source_text = evidence.get(str(check.get("source_path")), "")
            # Suggest from the same exact text representations accepted above.
            # Serialized JSON alone makes quotes/newlines needlessly hard to copy.
            source_text, match = max(
                ((text, SequenceMatcher(None, str(quote or "")[:512], text,
                                        autojunk=False).find_longest_match())
                 for text in _source_quote_texts(source_text)),
                key=lambda item: (item[1].size, -len(item[0])))
            start = max(0, match.b - 100)
            quote_start = max(0, match.b - match.a)
            quote_end = quote_start + min(len(str(quote or "")), 512)
            # A missing formatting delimiter changes source length. A fixed-length
            # slice can start mid-word and send another broken citation back.
            blocks = [b for b in SequenceMatcher(
                None, str(quote or "")[:512], source_text, autojunk=False).get_matching_blocks()
                if b.size]
            if blocks:
                first, last = blocks[0], blocks[-1]
                span_start = max(0, first.b - first.a)
                span_end = min(len(source_text), last.b + last.size
                               + min(len(str(quote or "")), 512) - last.a - last.size)
                if 0 < span_end - span_start <= 512:
                    quote_start, quote_end = span_start, span_end
            problems.append({"category": "feasibility", "component": "",
                             "grounded": False, "status": check.get("status"),
                             "response_protocol_error": True,
                             "repository_topic": topic,
                             "source_path": check.get("source_path"),
                             "evidence_excerpt": source_text[start:start + 600],
                             "verbatim_candidate": source_text[quote_start:quote_end],
                             "issue": (f"Repository check {topic} has {len(matches)} entries; return exactly one check for this topic."
                                       if len(matches) > 1 else
                                       f"Repository check {topic} missing nonempty 'quote' field: {check}"
                                       if not isinstance(quote, str) or not quote.strip()
                                       else f"Repository evidence binding mismatch for {topic}: {check}"),
                             "suggestion": f"Repair the Critic citation: use the literal JSON key 'quote' in each repository_checks entry, not 'verbatim_quote' or 'verbatim_candidate'. source_path must be exactly one of {sorted(evidence)}; copy a short nonempty exact quote from that source, without reformatting it or adding field selectors to source_path. evidence_excerpt is real source text near the attempted quote, not proof of the premise. verbatim_candidate is an unchanged source substring, not a validated premise: use it only if it supports this check in context; do not capitalize or reword it. Check it and the supplied source before choosing a quotation. A mismatched citation does not establish a contradicted premise. If no supplied evidence supports the premise, report requires_measurement rather than inventing a quote."})
        elif check.get("status") not in {"supported", "not_applicable"}:
            problems.append({"category": "feasibility", "component": "",
                             "grounded": grounded, "status": check.get("status"),
                             "issue": f"Repository precondition {topic} is not grounded and supported: {check}",
                             "suggestion": "Measure missing facts or formally revise the contradicted Method; do not rename and repeat it."})
    node_checks = critic.get("component_checks")
    node_checks = node_checks if isinstance(node_checks, list) else []
    for node in method.get("components", []):
        cid = node.get("component_id") or node.get("name")
        matches = [c for c in node_checks if isinstance(c, dict) and c.get("component_id") == cid]
        check = matches[0] if len(matches) == 1 else {}
        kind = check.get("kind")
        nodes = {n.get("component_id") or n.get("name"): n for n in method.get("components", [])}
        ancestors, pending = set(), list(node.get("depends_on") or [])
        while pending:
            ancestor = pending.pop()
            if ancestor in ancestors:
                continue
            ancestors.add(ancestor)
            pending.extend(nodes.get(ancestor, {}).get("depends_on") or [])
        bindings = check.get("input_bindings")
        bindings_valid = isinstance(bindings, dict) and set(bindings) == set(node.get("inputs") or [])
        binding_errors = []
        if not bindings_valid:
            binding_errors.append(f"expected exact input keys {node.get('inputs') or []}; received {bindings}")
        if bindings_valid:
            for source in bindings.values():
                owner, _, output = str(source).partition(":")
                if source != "baseline" and (owner not in ancestors or output not in nodes.get(owner, {}).get("outputs", [])):
                    bindings_valid = False
                    binding_errors.append(
                        f"source {source!r}: ancestor must be in {sorted(ancestors)}; "
                        f"exact outputs of {owner!r} are {nodes.get(owner, {}).get('outputs', [])}")
        if not bindings_valid:
            problems.append({"category": "component_soundness", "component": cid,
                             "response_protocol_error": True,
                             "expected_input_keys": node.get("inputs") or [],
                             "allowed_sources": ["baseline"] + [
                                 f"{ancestor}:{output}" for ancestor in sorted(ancestors)
                                 for output in nodes.get(ancestor, {}).get("outputs", [])],
                             "issue": "Component input binding mismatch: " + "; ".join(binding_errors),
                             "suggestion": "Repair the Critic input_bindings by copying exact input and ancestor output strings, including descriptions. Do not merge or redesign components solely for abbreviated binding names. If the required source truly does not exist, repair the dependency or interface design."})
        if (check.get("independently_executable") is not True
                or check.get("parent_control_available") is not True
                or kind not in {"intervention", "measurement"}
                or (kind == "measurement" and node.get("type") == "direct_gain")):
            incomplete_check = (kind not in {"intervention", "measurement"}
                                or type(check.get("independently_executable")) is not bool
                                or type(check.get("parent_control_available")) is not bool)
            failed_checks = []
            if check.get("independently_executable") is False:
                failed_checks.append("independently_executable=false")
            if check.get("parent_control_available") is False:
                failed_checks.append("parent_control_available=false")
            if kind == "measurement" and node.get("type") == "direct_gain":
                failed_checks.append("measurement-only component is direct_gain")
            problems.append({"category": "component_soundness", "component": cid,
                             "response_protocol_error": incomplete_check,
                             "issue": ("Component review fields are missing or malformed: " if incomplete_check
                                       else "Component review failed: " + "; ".join(failed_checks) + ". Review: ") + str(check),
                             "suggestion": ("Return kind and JSON booleans independently_executable and parent_control_available explicitly. Use true/false, never a source name such as 'baseline' or a quoted boolean. Put source names only in input_bindings; missing fields are not negative judgments."
                                            if incomplete_check else
                                            "Address only the named failed conditions. independently_executable judges whether the component can run WITH its declared dependency closure, not whether it is dependency-free. If the reviewer confused those meanings, correct its judgment with evidence; otherwise repair the actual missing prerequisite or inseparable implementation. If the node only reruns the official evaluator or correctness gate on unchanged code, remove it as a component and attach the check to the implementing component's acceptance or final verification, preserving a removed confirmed ID through merged_from. Otherwise classify a genuine measurement-only component as indirect_gain; that type correction does not require setting independently_executable=false. Do not merge a valid dependent measurement solely because it consumes ancestors.")})
    return problems


def _merge_critic_protocol_repair(previous: dict, repair: dict, errors: list[dict]) -> dict:
    """Patch only malformed review entries; never discard substantive criticism."""
    merged = dict(previous)
    merged["problems"] = list(previous.get("problems") or [])
    for problem in repair.get("problems") or []:
        if problem not in merged["problems"]:
            merged["problems"].append(problem)
    for field, key, error_key in (("repository_checks", "topic", "repository_topic"),
                                  ("component_checks", "component_id", "component")):
        allowed = {p.get(error_key) for p in errors if p.get(error_key)}
        replacements = (_repository_check_entries(repair) if field == "repository_checks"
                        else repair.get(field))
        if not isinstance(replacements, list):
            continue
        entries = (_repository_check_entries(previous) if field == "repository_checks"
                   else previous.get(field))
        entries = list(entries) if isinstance(entries, list) else []
        for identity in allowed:
            matches = [c for c in replacements if isinstance(c, dict) and c.get(key) == identity]
            if len(matches) != 1:
                continue  # A missing/ambiguous repair remains invalid on revalidation.
            replacement = dict(matches[0])
            if field == "repository_checks" and replacement.get("use_verbatim_candidate") is True:
                hints = [p for p in errors if p.get("repository_topic") == identity
                         and p.get("source_path") == replacement.get("source_path")
                         and p.get("verbatim_candidate")]
                if len(hints) == 1 and "quote" not in replacement:
                    # Explicit reviewer selection, never automatic semantic approval.
                    # The unchanged source and status are rechecked after restoration.
                    replacement["quote"] = hints[0]["verbatim_candidate"]
                    replacement.pop("use_verbatim_candidate")
            if field == "component_checks" and "input_source_indices" in replacement:
                binding_errors = [p for p in errors if p.get("component") == identity
                                  and "expected_input_keys" in p and "allowed_sources" in p]
                if len(binding_errors) == 1:
                    inputs = binding_errors[0]["expected_input_keys"]
                    sources = binding_errors[0]["allowed_sources"]
                    indices = replacement["input_source_indices"]
                    if (isinstance(indices, list) and len(indices) == len(inputs)
                            and all(type(i) is int and 0 <= i < len(sources) for i in indices)):
                        # Restore literal names only; the Critic still chooses each
                        # source, and normal dependency validation runs afterwards.
                        replacement["input_bindings"] = {
                            name: sources[i] for name, i in zip(inputs, indices)}
                        replacement.pop("input_source_indices")
            entries = [c for c in entries if not isinstance(c, dict) or c.get(key) != identity]
            entries.append(replacement)
        merged[field] = entries
    return merged


def _divergence_feedback(moss_reason: str) -> str:
    """Feedback seeding the NEXT outer round when the proposers did NOT converge and the critic found
    no deep problem (#62): the two must still resolve their disagreement, so carry the MOSS 分歧 forward."""
    return f"两位提议者仍未就方案达成一致，请针对以下分歧收敛：{moss_reason or '（未说明）'}"


def deliberate_method(problem_model: dict | None, literature_brief: dict | None,
                      user_idea: str | None, draft_method: dict, llm: LLMClientLike, *,
                      inner_rounds: int = 10, outer_rounds: int = 5,
                      max_rounds: int | None = None,
                      prompt_patch: str | None = None,
                      baseline_result: dict | None = None,
                      evaluation_semantics: dict | None = None,
                      prior_failure_context: dict | None = None,
                      component_plan_audit: bool = False,
                      component_plan_revision_feedback: str | None = None,
                      confirmed_component_ids: set[str] | None = None) -> MethodDebateResult:
    """Bounded NESTED FOUR-ROLE method deliberation (option B, meeting 2026-08-03).

    OUTER loop (<= ``outer_rounds``): each outer round re-deliberates from where the critic left off.
      INNER loop (<= ``inner_rounds``): PURE A1<->A2 convergence -- each exchange both propose AND
        critique the other; a convergence judge checks; when NOT converged MOSS says WHY and that
        drives the next inner exchange. The CRITIC does NOT run inside the inner loop (前两个先收敛).
      After the inner loop the dedicated CRITIC reviews (rubric-gated deep problems), while MOSS
      only relays/guides the resulting feedback:
        - converged -> the critic reviews the AGREED method (A1 represents both). ONLY here can the
          method be ACCEPTED: no deep problem -> accept; any target-undermining deep problem ->
          MOSS relays it constructively and the control flow starts the next outer round.
        - capped WITHOUT converging -> the critic reviews BOTH A1 & A2 现方法 together with the MOSS-
          identified 分歧 (NOT just A1; 会议 2026-08-04) so its problems seed the next round, but a
          NON-converged method is NEVER accepted (#62): always force another outer round. If the outer
          cap is hit still unconverged -> honest fallback (keep A1, accepted=False, unconverged_fallback).
    Total exchanges bounded by ``outer_rounds * inner_rounds``. Ordinary proposals are fail-soft;
    unusable CAP proposals retry once, then raise a protocol error instead of a semantic verdict.
    Loose coupling: returns method + a report of every role's
    output. ``max_rounds`` is a backward-compat alias for ``inner_rounds``."""
    if max_rounds is not None:
        inner_rounds = max_rounds
    inner_cap = max(1, int(inner_rounds))
    outer_cap = max(1, int(outer_rounds))
    ctx = _method_ctx(problem_model, literature_brief, user_idea, baseline_result,
                      evaluation_semantics, component_plan_audit,
                      component_plan_revision_feedback, prior_failure_context)
    from autoresearch.experiments.repo_improve.execution_policy import (
        efficient_execution, EFFICIENT_PLANNING)
    if efficient_execution():
        ctx += "\n" + EFFICIENT_PLANNING
    initial_contract_feedback = None
    if component_plan_audit:
        draft_error = _component_contract_problem(draft_method, require_design_fields=True)
        if draft_error:
            initial_contract_feedback = "Initial saved draft errors to repair:\n" + str(draft_error)

    def sys(s: str) -> str:
        return apply_prompt_patch(s, prompt_patch) if prompt_patch else s

    method_a = _norm_method(dict(draft_method or {}))
    method_b = _norm_method(dict(draft_method or {}))
    frozen_method = _norm_method(dict(draft_method or {}))
    if component_plan_audit:
        if confirmed_component_ids is None:
            from autoresearch.experiments.repo_improve.component_plan import _component_id
            confirmed_component_ids = {
                _component_id(component.get("component_id") or component.get("name"), index)
                for index, component in enumerate(frozen_method.get("components") or [])
                if isinstance(component, dict)
            }
        ctx += ("\nOnly these confirmed Method component IDs may appear in merged_from: "
                + json.dumps(sorted(confirmed_component_ids), ensure_ascii=False)
                + ". Names of internal mechanisms are not component IDs.")
    exposed_references = list(frozen_method.get("reference_sources") or [])
    converged = False
    accepted = False
    outers = 0
    total_exchanges = 0
    method_revision_required = False
    method_revision_reason = ""
    transcript: list[dict] = []
    outer_feedback: str | None = None   # critic feedback carried into the next outer round's inner loop
    proposer_system = (
        _COMPONENT_PLAN_PROPOSER_SYSTEM if component_plan_audit else _METHOD_PROPOSER_SYSTEM)
    proposer_system += (" List every actually used reference document/implementation in "
                        "reference_sources; preserve prior exposure. Prohibited material "
                        "cannot be an implementation step.")

    logger.info("[method-debate] START inner<=%d outer<=%d", inner_cap, outer_cap)
    while outers < outer_cap:
        outers += 1
        inner_feedback = outer_feedback   # the prior outer round's critic feedback seeds this inner loop
        if outers == 1 and initial_contract_feedback:
            inner_feedback = initial_contract_feedback
        inner_converged = False
        inners = 0
        last_moss_reason = ""   # the latest MOSS-identified disagreement this outer round (for the capped case)
        # ---- INNER: pure A1<->A2 convergence (MOSS why-not drives it; NO critic here) ----
        while inners < inner_cap:
            inners += 1
            total_exchanges += 1
            method_a, crit_a = _propose(
                llm, sys, "A1", ctx, method_a, method_b, inner_feedback,
                system_template=proposer_system,
                require_valid_response=component_plan_audit,
                max_tokens=262144)
            if component_plan_audit:
                method_a = {
                    **frozen_method,
                    "components": method_a.get("components") or [],
                }
            method_b, crit_b = _propose(
                llm, sys, "A2", ctx, method_b, method_a, inner_feedback,
                system_template=proposer_system,
                require_valid_response=component_plan_audit,
                max_tokens=262144)
            if component_plan_audit:
                method_b = {
                    **frozen_method,
                    "components": method_b.get("components") or [],
                }
            converge_system = (
                _COMPONENT_PLAN_CONVERGE_SYSTEM
                if component_plan_audit else _METHOD_CONVERGE_SYSTEM)
            converge_input = (
                f"{ctx}\n\nProposer A1 method:\n{_method_str(method_a)}\n\n"
                f"Proposer A2 method:\n{_method_str(method_b)}")
            converge_attempts = 2 if component_plan_audit else 1
            conv = None
            for converge_attempt in range(converge_attempts):
                conv = _chat_json(llm, sys(converge_system), converge_input)
                if conv is None or type(conv.get("converged")) is bool:
                    break
                logger.warning(
                    "[method-debate] convergence response must use a JSON boolean; attempt %d",
                    converge_attempt + 1,
                )
                conv = None
                converge_input += (
                    "\n\nYour previous response used an invalid type for converged. "
                    "Return JSON boolean true or false, never a quoted string or number."
                )
            is_conv = conv["converged"] if conv is not None else False
            conv = conv or {}
            rec: dict = {"outer": outers, "inner": inners, "converged": is_conv,
                         "a_approach": method_a.get("approach"), "b_approach": method_b.get("approach"),
                         "crit_a": crit_a, "crit_b": crit_b}
            if is_conv:
                inner_converged = converged = True
                rec["outcome"] = "inner_converged"
                logger.info("[method-debate] outer %d inner %d | A1='%s' A2='%s' -> CONVERGED",
                            outers, inners, _short(method_a.get("approach")),
                            _short(method_b.get("approach")))
                transcript.append(rec)
                break
            guide = _chat_json(llm, sys(_METHOD_MOSS_GUIDE_SYSTEM),
                               f"{ctx}\n\nA1:\n{_method_str(method_a)}\n\nA2:\n{_method_str(method_b)}\n\n"
                               f"Convergence judge note: {conv.get('reason') or ''}") or {}
            rec["moss_guidance"] = guide.get("guidance")
            rec["outcome"] = "inner_not_converged"
            last_moss_reason = str(guide.get("guidance") or conv.get("reason") or "").strip()
            inner_feedback = f"MOSS（引导收敛）：{last_moss_reason or '请就分歧点相互采纳、往一个最优方法收敛'}"
            logger.info("[method-debate] outer %d inner %d | A1='%s' A2='%s' -> NOT converged (MOSS 引导: %s)",
                        outers, inners, _short(method_a.get("approach")),
                        _short(method_b.get("approach")), _short(guide.get("guidance") or conv.get("reason")))
            transcript.append(rec)

        # ---- after inner (CONVERGED or CAPPED) -> the CRITIC picks holes (rubric-gated), then MOSS
        # reviews. 老师: 内层收敛或跑满,critic 都进去挑刺,MOSS 审. When CONVERGED the two agree so A1
        # represents both; when CAPPED (未收敛) the critic must see BOTH A1 & A2 现方法 + MOSS 分歧,
        # NOT just A1 (会议 2026-08-04). ----
        if inner_converged:
            review_str = (f"The two proposers CONVERGED. Method under review (agreed):\n"
                          f"{_method_str(method_a)}")
        else:
            review_str = (
                "The two proposers did NOT converge within the inner cap -- review BOTH competing "
                "methods together with the MOSS-identified disagreement (do not treat A1 as agreed).\n\n"
                f"A1 现方法:\n{_method_str(method_a)}\n\nA2 现方法:\n{_method_str(method_b)}\n\n"
                f"MOSS 分歧:\n{last_moss_reason or '(未说明)'}")
        critic_system = (
            _COMPONENT_PLAN_CRITIC_SYSTEM if component_plan_audit else _METHOD_CRITIC_SYSTEM)
        if baseline_result and (baseline_result.get("source_kind") == "live_repo"
                                or (baseline_result.get("repository_contract") or {}).get("sources")):
            critic_system += (
                " Also return repository_checks as a JSON array of check objects for task_direction, interface_and_inputs, "
                "existing_mechanism, resource_scale, measurement_protocol, artifact_scope, "
                "source_boundary. Justify the frozen measurement protocol, required saved files "
                "and allowed sources from repository evidence. Each needs topic "
                "(exactly one of those identifiers, not an explanatory sentence), status "
                "(supported/contradicted/requires_measurement/not_applicable), source_path "
                "(a supplied repository source path, baseline_result, or evaluation_semantics "
                "when frozen comparison semantics are supplied), and an exact nonempty "
                "quote from that source. Unsupported premises must not be marked supported. "
                "Quote one contiguous source substring (prefer the relevant sentence or decoded "
                "JSON string value, without copying its enclosing keys). Do not join nonadjacent "
                "fields or change quote styles in serialized source text. Explain any synthesis "
                "outside the quote. "
                "For configuration evidence, select one short exact line as the citation anchor; "
                "do not concatenate multiple configuration fields into quote. Read the full source "
                "to decide the finding: the anchor alone is not proof of every part of the finding. "
                "Cite task edit permissions and prohibitions from repository_contract sources "
                "or an explicit user boundary; evaluation_semantics is the frozen measurement "
                "anchor but its inferred restrictions cannot override a conflicting task source. "
                "Task requirements do not prove an observed "
                "repository capability or measurement; those need repository/baseline evidence. "
                "existing_mechanism describes the CURRENT repository mechanism, grounded in supplied "
                "source, not proof that a proposed implementation already passed its planned validation. "
                "A scheduled measurement component may run from available executable inputs and gate "
                "downstream consumers on its acceptance. Its success remains unknown until measured. "
                "Do not demand its future outcome before scheduling it, or mark a genuinely unsupported "
                "baseline capability as supported. "
                "Copy source_path exactly; do not append field selectors or use invented path aliases. "
                "A quotation/path mismatch requires repairing the Critic citation, not changing the Method. "
                "Return component_checks for every component with component_id, kind "
                "(intervention/measurement), independently_executable, parent_control_available, "
                "and input_bindings mapping every input string to baseline or ancestor_id:output_name. "
                "independently_executable and parent_control_available are JSON booleans (true/false), "
                "not strings or checkpoint/source names. Put 'baseline' only in input_bindings, "
                "never in parent_control_available. Decide each boolean from evidence; do not default to true. "
                "independently_executable means executable WITH the declared dependency closure, "
                "not without prerequisites. Baseline-provided model state, data and evaluator inputs "
                "may bind to baseline; they do not need invented ancestor components. A root component "
                "whose declared inputs all bind to a valid frozen baseline has an available parent/control: "
                "set parent_control_available=true. Only a component with no depends_on uses the "
                "frozen baseline as its comparison parent. A dependent "
                "component uses exactly the checkpoint assembled from its declared depends_on; "
                "a baseline input binding or a historical incumbent does not select a different control. "
                "Judge the declared parent's runnable capability, not whether an ancestor component checkpoint exists yet. "
                "For direct_gain, parent_control_available requires this exact parent to supply a finite "
                "primary task metric under the frozen evaluator, not an intermediate diagnostic count. "
                "If a dependency only supplies probe measurements and no task checkpoint, do not mark "
                "its performance control available merely because another baseline or incumbent exists. "
                "Repair the dependency/comparison design through the formal plan or Method revision; "
                "preserve measured candidates and frozen correctness, quality and acceptance requirements. "
                "Copy each input key VERBATIM from components.inputs, including descriptions; "
                "copy output_name VERBATIM from the ancestor's outputs, without abbreviating it. "
                "A binding spelling mismatch is a Critic response error to repair, not evidence that "
                "the component must be merged or redesigned. Inseparable "
                "implementation steps need one component. Pure verify/eval nodes are indirect_gain, "
                "not an intervention that must improve performance. Negative diagnostic facts "
                "require hypothesis revision, not changing the diagnostic until it says yes.")
        critic_input = f"{ctx}\n\n{review_str}"
        if component_plan_audit and outer_feedback:
            critic_input += (
                "\n\nUNRESOLVED COMPONENT-PLAN FEEDBACK FROM THE PREVIOUS OUTER ROUND; "
                "check whether each item is STILL present in the current proposal. Previous "
                "errors are history, not evidence against corrected fields; do not repeat a "
                "resolved finding. Verify every item before returning an empty problems list:\n"
                + str(outer_feedback))
        for proposed in (method_a, method_b):
            for source in proposed.get("reference_sources") or []:
                if source not in exposed_references:
                    exposed_references.append(source)
        method_a["reference_sources"] = list(exposed_references)
        method_b["reference_sources"] = list(exposed_references)
        prior_semantic_problems = []
        protocol_previous = None
        protocol_problems = []
        max_review_attempts = 3 if component_plan_audit else 2
        for review_attempt in range(max_review_attempts):
            review_tokens = 262144
            critic = _chat_json(llm, sys(critic_system), critic_input,
                                max_tokens=review_tokens)
            if protocol_previous is not None and isinstance(critic, dict):
                critic = _merge_critic_protocol_repair(protocol_previous, critic, protocol_problems)
            if isinstance(critic, dict) and isinstance(critic.get("problems"), list):
                for problem in prior_semantic_problems:
                    if problem not in critic["problems"]:
                        critic["problems"].append(problem)
                checked = _repository_review_problems(critic, baseline_result, method_a, evaluation_semantics)
                protocol_problems = [p for p in checked if p.get("response_protocol_error")]
                contradicted = any(p.get("grounded") and p.get("status") == "contradicted"
                                   for p in checked)
                if (protocol_problems and not contradicted
                        and review_attempt + 1 < max_review_attempts):
                    prior_semantic_problems = list(critic["problems"])
                    # Repair reviewer serialization at the reviewer, not by redesigning
                    # the frozen Method or making both proposers repeat their plans.
                    transcript.append({"outer": outers, "outcome": "critic_protocol_repair",
                                       "problems": protocol_problems})
                    if any(p.get("repository_topic") or p.get("component") for p in protocol_problems):
                        protocol_previous = critic
                    critic_input += (
                        "\n\nYour review has citation/input-binding protocol errors. Review the SAME "
                        "unchanged plan; repair repository_checks/component_checks using the exact "
                        "supplied evidence and field names. Preserve all substantive problems; do not "
                        "invent evidence or change a requires_measurement/contradicted finding merely "
                        "to pass. Return only the corrected entries in repository_checks and/or "
                        "component_checks for the reported topics/component IDs. The host retains "
                        "all other checks and substantive problems. Do not rewrite the complete review "
                        "or Method. A problems array is optional, for additional substantive findings. "
                        "For input bindings, copy the JSON strings in expected_input_keys exactly, "
                        "with no appended explanation; choose each source from allowed_sources based "
                        "on the actual dependency. To avoid recopying long escaped strings, instead "
                        "return input_source_indices in the corrected component_checks entry: a list "
                        "with exactly one integer for each expected_input_keys entry, in that order. "
                        "Each integer is the zero-based index of your chosen allowed_sources entry. "
                        "The host restores the exact names, without choosing sources for you. Keep "
                        "component_id, kind and both boolean judgments in the entry. "
                        "independently_executable and parent_control_available must be JSON true/false, "
                        "never 'baseline' or a quoted boolean. If expected_input_keys/allowed_sources "
                        "were not supplied for an entry, preserve its valid input_bindings verbatim "
                        "instead of inventing input_source_indices. Put explanations "
                        "outside the binding keys. "
                        "For a citation spelling error, copy verbatim_candidate exactly when it supports "
                        "your finding. Alternatively, omit quote and return use_verbatim_candidate=true "
                        "with the same topic and source_path and your status judgment; the host will "
                        "copy that exact supplied substring for you and revalidate it. This is only "
                        "a lossless quoting option, not permission to change a substantive judgment. "
                        "If the hint does not support the finding, do not select it. Preserve "
                        "its original capitalization. All repaired entries are "
                        "limited to one contiguous source substring. Do not join nonadjacent fields "
                        "or reserialize the source; prefer the supporting string value alone. They are "
                        "revalidated against the original source and unchanged plan.\n"
                        + json.dumps({"previous_review": critic,
                                      "protocol_errors": protocol_problems}, ensure_ascii=False))
                    continue
                break
            logger.warning("[method-debate] unusable Critic response, attempt %d",
                           review_attempt + 1)
            critic_input += (
                "\nYour previous review had a response protocol error. Return one complete "
                "JSON object with a problems array (empty only after actually reviewing), "
                "plus all requested repository_checks and component_checks. Keep the review "
                "concise; do not replace it with a revised method or omit required fields.")
        unusable_review = not isinstance(critic, dict) or not isinstance(critic.get("problems"), list)
        remaining_protocol_problems = []
        if not unusable_review:
            checked = _repository_review_problems(critic, baseline_result, method_a, evaluation_semantics)
            remaining_protocol_problems = [p for p in checked if p.get("response_protocol_error")]
            # A failed repair is still a reviewer response error, not new evidence
            # against the component design. Preserve the draft for the normal retry.
            if (component_plan_audit and remaining_protocol_problems
                    and not any(p.get("grounded") and p.get("status") == "contradicted"
                                for p in checked)):
                unusable_review = True
        if unusable_review:
            if component_plan_audit:
                error = DeliberationLLMError(
                    "ComponentPlan Critic response protocol failure after bounded retries; "
                    "no usable review was received, not a semantic rejection of the plan.")
                error.draft_method = method_a
                error.debate_transcript = transcript + [{
                    "outer": outers, "outcome": "critic_review_unavailable",
                    "inner_converged": inner_converged,
                    "review": critic, "protocol_errors": remaining_protocol_problems}]
                raise error
            transcript.append({"outer": outers, "inner_converged": inner_converged,
                               "outcome": "critic_review_unavailable"})
            logger.warning("[method-debate] Critic review unavailable or malformed; not accepted")
            break
        critic_problems = _rubric_deep_problems(critic.get("problems"))
        contradicted_critic_problems = (
            [p for p in critic_problems if _critic_field_claim_conflicts(p, method_a)]
            if component_plan_audit else []
        )
        deep = [p for p in critic_problems if p not in contradicted_critic_problems]
        repository_problems = _repository_review_problems(critic, baseline_result, method_a, evaluation_semantics)
        deep.extend(repository_problems)
        # An exact repository quote proves the source text exists, not that it contradicts
        # the frozen Method.  A Method-level revision requires the Critic's explicit judgment.
        if component_plan_audit and critic.get("method_revision_required") is True:
            method_revision_required = True
            method_revision_reason = str(
                critic.get("method_revision_reason") or "confirmed Method requires revision before component planning"
            ).strip()
        contract_problem = (
            _component_contract_problem(
                method_a, require_design_fields=component_plan_audit,
                confirmed_component_ids=confirmed_component_ids)
            if inner_converged else ""
        )
        if contract_problem:
            deep.append({
                "category": "ablation_verifiability",
                "issue": f"ComponentPlan cannot execute the agreed method: {contract_problem}",
                "suggestion": (
                    "repair the component IDs/dependencies and make each indirect-gain check an "
                    "explicit numeric comparison before accepting the method"
                ),
                "component": "",
            })
        crec: dict = {"outer": outers, "inner_converged": inner_converged,
                      "critic_deep_problems": deep}
        if contradicted_critic_problems:
            crec["contradicted_critic_problems"] = contradicted_critic_problems
        crec["repository_checks"] = critic.get("repository_checks") or []
        crec["component_checks"] = critic.get("component_checks") or []
        if contract_problem:
            crec["contract_error_kind"] = type(contract_problem).__name__
        if method_revision_required:
            crec["method_revision_required"] = True
            crec["method_revision_reason"] = method_revision_reason
            crec["outcome"] = "confirmed_method_requires_revision"
            transcript.append(crec)
            logger.warning(
                "[method-debate] confirmed Method requires revision before ComponentPlan: %s",
                _short(method_revision_reason),
            )
            break
        if not inner_converged:   # record the non-converged state the critic actually reviewed
            crec["capped_review"] = {"a_approach": method_a.get("approach"),
                                     "b_approach": method_b.get("approach"),
                                     "moss_divergence": last_moss_reason}
        logger.info("[method-debate] outer %d inner %s (at %d) -> critic %d deep: %s",
                    outers, "CONVERGED" if inner_converged else "CAPPED", inners,
                    len(deep), [p["category"] for p in deep])
        # #62: NOT converged -> NEVER accept. The whole point of the debate is the two proposers
        # reaching agreement; accepting a capped-not-converged method rubber-stamps an unresolved
        # disagreement. Force another outer round (seed = critic 深坑 if any, else the MOSS 分歧). The
        # critic still ran above (on BOTH A1 & A2 + 分歧) so its problems feed the next round; the MOSS
        # worth-check is skipped -- non-convergence alone already mandates another round.
        if not inner_converged:
            outer_feedback = _critic_feedback(deep) if deep else _divergence_feedback(last_moss_reason)
            crec["outcome"] = "outer_another_round_unconverged"
            logger.info("[method-debate] outer %d NOT converged -> force another outer round (%d deep)",
                        outers, len(deep))
            transcript.append(crec)
            continue   # the while-condition enforces the outer cap; capped-unconverged handled below

        # CONVERGED -> the CRITIC's TARGET is the gate: it already returns problems ONLY when a design
        # choice undermines a real, DEFENSIBLE improvement on the metric (分清主次 baked into its
        # target). So no such problem -> ACCEPT. Otherwise MOSS -- the coordinator (human role), NOT a
        # judge -- GUIDINGLY relays the critic's points to the proposers and we run another outer
        # round. MOSS does not decide 'worth another round'; the critic's target does.
        if not deep:
            accepted = True
            crec["outcome"] = "accepted_converged_clean"
            logger.info("[method-debate] outer %d ACCEPTED (converged & critic finds nothing "
                        "undermining a real improvement)", outers)
            transcript.append(crec)
            break
        relay_system = (
            _COMPONENT_PLAN_MOSS_RELAY_SYSTEM
            if component_plan_audit else _METHOD_MOSS_RELAY_SYSTEM)
        relay = _chat_json(llm, sys(relay_system),
                           f"{ctx}\n\n{review_str}\n\n"
                           f"Critic problems:\n{json.dumps(deep, ensure_ascii=False)}") or {}
        crec["moss_guidance"] = relay.get("guidance")
        outer_feedback = str(relay.get("guidance") or "").strip() or _critic_feedback(deep)
        if component_plan_audit:
            # MOSS may summarize a different historical problem. The exact current
            # rejection must still reach both proposers, without losing any condition.
            outer_feedback += "\nAuthoritative current review errors:\n" + _critic_feedback(deep)
        crec["outcome"] = "outer_another_round_guided"
        logger.info("[method-debate] outer %d critic found %d undermining -> MOSS guides -> another "
                    "outer round (%s)", outers, len(deep), _short(relay.get("guidance")))
        transcript.append(crec)

    # #62 FALLBACK: the outer loop exhausted its cap WITHOUT the proposers ever converging. We refuse to
    # mark this "accepted" (a non-converged method must never pass); the flow is fail-open at the caller,
    # so we still return A1's current method, but HONESTLY labelled unconverged so nothing downstream can
    # mistake it for an agreed method. (细节可后续升级为 MOSS 介入/换方法。)
    unconverged_fallback = not converged and not accepted
    if unconverged_fallback:
        transcript.append({"outer": outers, "inner_converged": False,
                           "outcome": "capped_unconverged_fallback"})
        logger.warning("[method-debate] outer-cap %d reached WITHOUT convergence -> unconverged "
                       "fallback (keep A1, accepted=False)", outer_cap)

    report = {"converged": converged, "accepted": accepted, "outer_rounds": outers,
              "inner_cap": inner_cap, "outer_cap": outer_cap, "total_exchanges": total_exchanges,
              "unconverged_fallback": unconverged_fallback,
              "method_revision_required": method_revision_required,
              "method_revision_reason": method_revision_reason,
              "rubric": list(METHOD_CRITIC_RUBRIC), "transcript": transcript}
    logger.info("[method-debate] DONE converged=%s accepted=%s outer_rounds=%d exchanges=%d",
                converged, accepted, outers, total_exchanges)
    return MethodDebateResult(method=method_a, report=report)


__all__ = ["deliberate", "deliberate_method", "minimal_report", "DeliberationResult",
           "MethodDebateResult", "DeliberationError", "DeliberationLLMError", "DEFAULT_ANGLES",
           "METHOD_CRITIC_RUBRIC"]
