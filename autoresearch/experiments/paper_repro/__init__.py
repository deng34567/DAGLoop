"""meeting ③ (2026-07-31): paper-reproduction line via Paper2Code (going-doer/Paper2Code).

WHAT: given a PAPER (no code repo), synthesize a runnable code repository that reproduces it, so the
existing agentic clone -> baseline -> improve pipeline can then run it. This fills the SEL routing
branch "无仓库有论文 -> 论文线" that was previously empty.

REUSE ASSESSMENT of Paper2Code:
- It is a script pipeline: 0_pdf_process -> 1_planning -> 1.1_extract_config -> 2_analyzing ->
  3_coding, orchestrated by scripts/run*.sh, outputting a repo dir.
- The ``*_llm.py`` variants load a LOCAL model via vLLM (``from vllm import LLM``) -- NOT reusable
  against our hosted gateway.
- The OpenAI variants (``1_planning.py`` etc.) use ``OpenAI(api_key=os.environ["OPENAI_API_KEY"])``,
  which ALSO honours ``OPENAI_BASE_URL`` -> **reusable by pointing the OpenAI client at OUR HKGAI
  gateway** (set OPENAI_BASE_URL + OPENAI_API_KEY + a flash/glm model). This adapter does exactly
  that: shell out to a checked-out Paper2Code with our gateway env.

STATUS: switch-gated (``AUTORESEARCH_PAPER_REPRO_BACKEND``, default off) and NOT wired into the main
routing yet (per the meeting decision -- offline only). The engine's SelectingBenchmark routing is
left untouched; enabling this is a deliberate future step.
"""
from autoresearch.experiments.paper_repro.paper2code_adapter import (
    build_repo_from_paper,
    paper_repro_backend,
)

__all__ = ["build_repo_from_paper", "paper_repro_backend"]
