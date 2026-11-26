# EvaluatorAgent — attempt record verification

This file is retained as state-machine documentation. Verification is now a
deterministic code path and does not invoke an LLM.

For every evaluated attempt:

- If it completed with a metric, confirm that RunRecord, EvaluationReport and
  any repository result mapping contain the same finite value.
- If it was invalid or failed, confirm the failure outcome and preserve the
  observable reason from the evaluation/stdout/stderr for the next attempt.
- Do not judge cheating, seed count, statistical significance, effect size,
  ablation strength or whether the method is scientifically convincing.

`verified` means only that the attempt record is internally consistent. It
does not mean that the method won or that the observed improvement replicated.
