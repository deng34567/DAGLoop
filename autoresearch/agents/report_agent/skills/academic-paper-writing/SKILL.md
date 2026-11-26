---
name: academic-paper-writing
description: Turn frozen AutoResearch evidence into a publication-style scientific manuscript without upgrading observations, sketches, or unsupported claims into verified results.
---

# Academic Paper Writing

Write a self-contained manuscript from the supplied research artifacts. Preserve the user's
language and adapt the section structure to the research mode instead of forcing benchmark
terminology onto theoretical or survey work.

When the evidence supports it, cover the scientific question, related work, problem formulation,
method and component contracts, experimental setup, results, ablations or component checks,
limitations, conclusion, and reproducibility details. For theoretical work, replace experimental
sections with assumptions, definitions, propositions, derivations, counterexamples, and validation
studies actually performed.

Keep these evidence distinctions explicit:

- A locally reproduced baseline is not a published or cross-hardware result.
- A measured off-method observation is useful empirical evidence and a comparison anchor, but is
  not an accepted result of the frozen method until alignment or formal method revision completes.
- Partial, drifted, failed, and unverified attempts belong in the empirical narrative and failure
  analysis; do not silently discard them or present them as accepted wins.
- Cite only identifiers, titles, authors, venues, and URLs present in the supplied artifacts. Mark
  missing literature instead of inventing references.
- Separate measured facts, supported interpretation, hypotheses, and future work.

Before drafting, build a compact claim-evidence map. Mark every major claim as `VERIFIED`,
`UNVERIFIED`, or `AUTHOR_INPUT_NEEDED`, and carry that status into the manuscript wording. A
metric can be `VERIFIED` even when it is an off-method observation; in that case the number is
reportable, but the claim that it validates the confirmed method remains `UNVERIFIED`.

Prefer concrete tables or compact prose for baseline/method numbers, hardware, data, evaluator,
attempt lineage, and component outcomes. Explain the mechanism and the evidence, not merely the
final score. If the evidence is insufficient for a paper-level claim, produce an honest
publication-style research report and state exactly what remains unverified.
