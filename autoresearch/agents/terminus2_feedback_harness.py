"""A lightweight AutoResearch feedback layer for Harbor's Terminus-2 agent.

This module intentionally keeps Terminus-2's single terminal loop.  It adds a
small, benchmark-agnostic planning and feedback protocol without importing the
full AutoResearch state machine into Harbor.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2


_RESEARCH_PROTOCOL = """
AutoResearch feedback protocol (apply this in addition to the task contract):

1. Read the task instruction, repository, editable boundary, metric direction,
   correctness gates, and available evaluator commands before changing code.
2. Maintain a compact component DAG for the proposed method.  A node is a
   testable design decision, not merely a file.  Add A -> B only when B consumes
   A's output or requires a property established by A.  Record for every node:
   hypothesis, dependencies, validation command, expected observation, and the
   best valid result.  Keep this compact; do not spend the task budget writing a
   report.
3. Use the task's real evaluator feedback.  Preserve the best contract-valid
   implementation and its exact metric.  A failed or slower trial must not
   overwrite the best candidate.  Restore the best candidate before final
   submission.
4. Debug execution failures locally.  When a trial runs but disappoints, use
   the measured evidence to choose one applicable diagnosis: wrong mechanism,
   missing mechanism, shifted optimization target, imbalance/tradeoff, weak
   mechanism understanding, unmet precondition, artificially restricted
   solution space, or overly narrow problem framing.  Turn that diagnosis into
   one concrete next experiment.
5. Indirect components are accepted when their declared intermediate property
   is demonstrated in the full dependency context; they need not immediately
   improve the final metric.  Direct-gain components must be compared with the
   appropriate parent candidate.  Use end-to-end evaluation to decide the final
   combination.
6. Iterate while there is a concrete evidence-based improvement direction and
   time remains.  Do not mark complete merely because the first valid result
   passes.  Before completion, run the official correctness/evaluation command
   on the best candidate and confirm that protected evaluator artifacts were not
   modified.
""".strip()


_REFLECTION_CHECKPOINT = """
AutoResearch supervisor checkpoint:
- Re-read the latest measured results and identify the best contract-valid
  candidate so far.
- Check the component DAG: completed nodes, failed assumptions, and the next
  dependency-ready experiment.
- Decide whether the current limitation is implementation/debugging or one of
  the eight method diagnoses in the initial protocol.
- Choose one concrete action with an explicit expected observation.  Preserve
  or restore the best candidate before trying a risky alternative.
- If the official evaluator has not been run recently, prefer obtaining real
  feedback over further speculative planning.

Continue from the terminal state below.
""".strip()


class AutoResearchTerminus2(Terminus2):
    """Terminus-2 with a compact DAG and evidence-feedback discipline."""

    def __init__(self, *args: Any, reflection_interval: int = 8, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reflection_interval = max(1, int(reflection_interval))

    @staticmethod
    def name() -> str:
        return "autoresearch-terminus-2"

    async def run(self, instruction, environment, context) -> None:
        augmented = f"{instruction}\n\n{_RESEARCH_PROTOCOL}"
        await super().run(augmented, environment, context)

    async def _handle_llm_interaction(
        self,
        chat,
        prompt,
        logging_paths,
        original_instruction="",
        session=None,
    ):
        episode = int(getattr(self, "_n_episodes", 0) or 0)
        if episode > 1 and episode % self._reflection_interval == 0:
            prompt = f"{_REFLECTION_CHECKPOINT}\n\n{prompt}"
        return await super()._handle_llm_interaction(
            chat,
            prompt,
            logging_paths,
            original_instruction,
            session,
        )
