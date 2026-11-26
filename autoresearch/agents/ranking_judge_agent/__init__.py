"""
RankingJudgeAgent: ranks a set of candidates (currently only
candidate_kind="hypothesis") via LLM pairwise judgments aggregated by a
deterministic Bradley-Terry-Luce (BTL) score, yielding a ranked shortlist and
a diversity verdict.
RankingJudgeAgent：通过 LLM 成对判断 + 确定性 Bradley-Terry-Luce（BTL）分数
聚合对一组候选（当前仅 candidate_kind="hypothesis"）排序，产出排序列表与
多样性判定。

Public surface / 公开接口:
  - ``core.rank_candidates``: pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.RankingJudgeAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_ranking_judge_service`` / ``run_ranking_judge_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.ranking_judge_agent import core, kernel
from autoresearch.agents.ranking_judge_agent.core import rank_candidates
from autoresearch.agents.ranking_judge_agent.handler import RankingJudgeAgentHandler
from autoresearch.agents.ranking_judge_agent.service import (
    build_ranking_judge_service,
    run_ranking_judge_agent,
)

__all__ = [
    "core",
    "kernel",
    "rank_candidates",
    "RankingJudgeAgentHandler",
    "build_ranking_judge_service",
    "run_ranking_judge_agent",
]
