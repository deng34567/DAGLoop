"""Explicit, default-off efficient research policy; no whole-task time limit."""
from __future__ import annotations

import os


def efficient_execution() -> bool:
    return os.environ.get("AUTORESEARCH_EXECUTION_STRATEGY") == "efficient"


def method_feedback_limit() -> int:
    if os.environ.get("AUTORESEARCH_METHOD_FEEDBACK") != "1":
        return 0
    try:
        return max(0, int(os.environ.get("AUTORESEARCH_METHOD_FEEDBACK_ROUNDS", "5")))
    except ValueError:
        return 5


EFFICIENT_PLANNING = (
    "EFFICIENT EXECUTION: form the earliest credible COMPLETE candidate. Group mechanisms that "
    "require joint training into an executable component rather than demanding an isolated gain "
    "for each sub-mechanism. Do not turn every save/load/format step into a separate optimization "
    "component. A check that only validates an unchanged candidate belongs to "
    "candidate acceptance or final verification, not a separate indirect-gain "
    "treatment component; still run and bind every mandatory correctness check. "
    "Preserve all mandatory correctness, quality and real-artifact requirements. "
    "Use independent component experiments only to establish an expensive downstream prerequisite, "
    "a required quality/correctness gate, stability that affects selection, or to distinguish "
    "a concrete diagnosis. Otherwise attach engineering checks and mechanism observations to "
    "the joint candidate run; do not manufacture separate gain claims for its internals. "
    "Unmeasured individual contribution remains unmeasured; do not label it proven or silently "
    "change direct_gain to indirect_gain. Preserve unchanged component contracts and dependencies "
    "verbatim when revising a plan, so their accepted checkpoints remain reusable. Discuss only "
    "unresolved issues; repair reviewer format errors locally. A method change still needs a "
    "formal Method and ComponentPlan revision."
)

EFFICIENT_EXPERIMENT = (
    "EFFICIENT EXECUTION: before an additional expensive run state its changed code/configuration "
    "or its new validation question. First inspect existing host-bound evidence and saved artifacts. "
    "Reuse a compatible control instead of rerunning it; record its reference. Resume only from "
    "a recoverable checkpoint with compatible code/data/protocol and sufficient native training "
    "state; weights alone do not prove equivalent training continuation. If training completed "
    "but scoring failed, preserve weights/predictions and repair scoring. If only the attempt "
    "reference is wrong, reconcile the existing successful host receipt before retraining. Never "
    "repair a failed execution by claiming success. Check actual entry points, required artifacts, "
    "allowed edits and selected Method configuration BEFORE long training. Small checks cannot "
    "replace mandatory full-scale correctness or final quality gates. Passing a component contract "
    "does not by itself establish optimization completion. Follow its recorded advance-or-improve "
    "decision. For improve, execute the specific recorded next_experiment within the frozen Method "
    "and task-permitted scope, and validate it with the corresponding trusted evaluator or frozen check. "
    "Every additional run needs an explicit change or new validation question, not merely remaining "
    "attempts. If a follow-up fails or does not improve, retain the earlier qualified checkpoint and "
    "return the result to the decision step to reassess value. For advance, do not add optional "
    "tuning; mandatory correctness checks and final acceptance still apply. "
    "A repeated identical observation is not an independent replication. There is no new "
    "whole-task wall-clock limit in this policy; existing command and failure protections remain."
)
