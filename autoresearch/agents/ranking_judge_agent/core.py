"""
RankingJudgeAgent core: the pure pairwise-ranking logic.
RankingJudgeAgent 核心：与传输层无关的纯"成对排序"逻辑。

``rank_candidates`` calls an injected LLM client to judge each unordered pair
of candidates (which of the two is stronger), but the LLM is used ONLY for
those pairwise judgments. The aggregate ranking -- a Bradley-Terry-Luce (BTL)
score per candidate, and the resulting best-to-worst shortlist -- is computed
DETERMINISTICALLY by ``_btl_scores`` (pure, no LLM involvement), so the final
ranking cannot be gamed by a misbehaving or adversarial LLM response beyond
the individual pairwise verdicts it renders.

``rank_candidates`` validates its assembled result against
``data-contracts/ranking_report.schema.yaml``, with NO EventBus / Redis /
Store access. The handler (a later milestone) is responsible for resolving
refs, persisting the report, and publishing events.

``rank_candidates`` 调用注入的 LLM 客户端对每一对候选（无序对）做判断（两者
中哪个更强），但 LLM 仅用于这些成对判断。聚合排序 —— 每个候选的
Bradley-Terry-Luce（BTL）分数，以及由此得到的从优到劣排序列表 —— 由
``_btl_scores`` 确定性计算（纯函数，不涉及 LLM），因此最终排序不会被
LLM 的单次成对判断之外的行为操纵。

``rank_candidates`` 对照 ``data-contracts/ranking_report.schema.yaml``
校验其组装结果 —— 不含任何 EventBus/Redis/Store 访问。handler（后续
里程碑）负责解析 ref、持久化报告、发布事件。
"""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class RankingError(Exception):
    """Base for RankingJudgeAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "ranking_failed"
    #: transient | permanent
    classification: str = "transient"


class RankingParseError(RankingError):
    """LLM never returned a well-formed, schema-valid report after retries.
    重试用尽后 LLM 仍未返回合法、schema 校验通过的报告。"""

    code = "ranking_parse_error"
    classification = "permanent"


class RankingLLMError(RankingError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "ranking_llm_error"
    classification = "transient"


# ---------------------------------------------------------------------------
# LLM caller protocol / LLM 调用协议
# ---------------------------------------------------------------------------


class LLMResultLike(Protocol):
    content: str | None
    usage: dict
    cost_delta: str


class LLMClientLike(Protocol):
    """Subset of autoresearch.llm.HKGAIClient used here. / 这里用到的客户端子集。"""

    def chat(
        self,
        *,
        messages: list[dict],
        response_format: str | None = ...,
        max_tokens: int | None = ...,
    ) -> LLMResultLike: ...


# ---------------------------------------------------------------------------
# Static resources: system prompt + schema validator
# 静态资源：system prompt + schema 校验器
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_pairwise_ranking_system_prompt() -> str:
    return (_PROMPTS_DIR / "pairwise_ranking_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _ranking_report_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "ranking_report.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_ranking_report(report: dict) -> list[str]:
    return [e.message for e in _ranking_report_validator().iter_errors(report)]


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------


def _all_pairs(ids: list[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(ids, 2))


def _build_user_message(candidates: list[dict], pairs: list[tuple[str, str]]) -> str:
    return "\n".join(
        [
            "Candidates (JSON):",
            json.dumps(candidates, ensure_ascii=False, indent=2),
            "",
            "Pairs to judge (JSON, each [a, b]):",
            json.dumps([list(pair) for pair in pairs], ensure_ascii=False, indent=2),
            "",
            "Judge each pair and respond with a single JSON object matching the "
            "shape described in the system prompt.",
        ]
    )


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object matching the "
            "shape described in the system prompt."
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic BTL aggregation (the anti-gaming core)
# 确定性 BTL 聚合（反操纵核心）
# ---------------------------------------------------------------------------

ITERS = 50
# Pseudocount (1 win + 1 loss vs a phantom "average" opponent of strength
# 1.0) applied to every candidate. Keeps an undefeated candidate's score
# finite (rather than diverging over fixed iterations) without changing the
# RELATIVE ordering among real candidates, since it is applied uniformly.
_PHANTOM_PSEUDOCOUNT = 1.0
_PHANTOM_STRENGTH = 1.0


def _btl_scores(pairwise_results: list[dict], ids: list[str]) -> list[dict]:
    """Compute per-candidate Bradley-Terry-Luce scores from pairwise results,
    via Zermelo's iterative MLE algorithm. PURE and DETERMINISTIC: no
    ``random``, no time, uniform init, a fixed number of iterations -- the
    same inputs always produce identical outputs.
    用 Zermelo 迭代 MLE 算法从成对结果计算每个候选的 Bradley-Terry-Luce
    分数。纯函数且确定性：无 random、无时间依赖，均匀初始化、固定迭代次数
    —— 相同输入总是产出相同输出。

    Args:
        pairwise_results: list of ``{"a": id, "b": id, "winner": id, ...}``.
            Entries referencing an id not in ``ids``, or whose ``winner`` is
            neither ``a`` nor ``b``, are ignored.
        ids: the full set of candidate ids (defines the output domain and
            the tie-break order used by ``rank_candidates``).

    Returns:
        ``[{"id": id, "score": float}, ...]`` for every id in ``ids``.
    """
    n = len(ids)
    if n == 0:
        return []
    if n == 1:
        return [{"id": ids[0], "score": 1.0}]

    index = {id_: i for i, id_ in enumerate(ids)}
    wins = [[0.0] * n for _ in range(n)]
    for result in pairwise_results:
        a, b, winner = result.get("a"), result.get("b"), result.get("winner")
        if a not in index or b not in index:
            continue
        if winner not in (a, b):
            continue
        i, j = index[a], index[b]
        if winner == a:
            wins[i][j] += 1
        else:
            wins[j][i] += 1

    p = [1.0] * n
    for _sweep in range(ITERS):
        new_p = list(p)
        for i in range(n):
            total_wins_i = sum(wins[i][j] for j in range(n) if j != i) + _PHANTOM_PSEUDOCOUNT
            denom = (2 * _PHANTOM_PSEUDOCOUNT) / (p[i] + _PHANTOM_STRENGTH)
            for j in range(n):
                if j == i:
                    continue
                games_ij = wins[i][j] + wins[j][i]
                if games_ij <= 0:
                    continue  # no games between i and j: no contribution
                denom += games_ij / (p[i] + p[j])
            if denom > 0:
                new_p[i] = total_wins_i / denom
        total = sum(new_p)
        if total > 0:
            new_p = [x / total for x in new_p]
        p = new_p

    return [{"id": ids[i], "score": p[i]} for i in range(n)]


def _ranked_shortlist(btl_scores: list[dict], ids: list[str]) -> list[str]:
    """Sort candidate ids by BTL score descending, tie-broken by original
    candidate order (both deterministic, given deterministic ``btl_scores``).
    按 BTL 分数降序排列候选 id，同分按原始候选顺序打破平局。"""
    order = {id_: i for i, id_ in enumerate(ids)}
    ordered = sorted(btl_scores, key=lambda s: (-s["score"], order[s["id"]]))
    return [s["id"] for s in ordered]


# ---------------------------------------------------------------------------
# LLM pairwise-response parsing/validation
# LLM 成对判断响应的解析/校验
# ---------------------------------------------------------------------------


def _parse_pairwise_response(content: str) -> tuple[dict | None, str | None]:
    """Parse+structurally-validate the LLM's raw pairwise-judgment JSON.
    Returns ``(obj, None)`` on success, or ``(None, reason)`` on failure.
    解析并结构校验 LLM 原始成对判断 JSON。成功返回 ``(obj, None)``，失败
    返回 ``(None, reason)``。"""
    try:
        obj = json.loads(extract_json(content))
    except (json.JSONDecodeError, TypeError):
        return None, "response was not valid JSON"

    if not isinstance(obj, dict):
        return None, "response JSON is not an object"

    pairwise_results = obj.get("pairwise_results")
    if not isinstance(pairwise_results, list):
        return None, "pairwise_results is missing or not a list"

    for item in pairwise_results:
        if not isinstance(item, dict):
            return None, "pairwise_results item is not an object"
        a, b, winner = item.get("a"), item.get("b"), item.get("winner")
        if not isinstance(a, str) or not isinstance(b, str) or not isinstance(winner, str):
            return None, "pairwise_results item missing string a/b/winner"
        if winner not in (a, b):
            return None, "pairwise_results item's winner must equal a or b"

    if obj.get("diversity_status") not in ("diverse", "collapsed"):
        return None, "diversity_status must be 'diverse' or 'collapsed'"

    return obj, None


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # 1 initial + up to 2 retries = 3 LLM calls max


def _trivial_report(candidates: list[dict], candidate_kind: str) -> dict:
    """0-or-1-candidate short circuit: no pairs exist to judge, so no LLM
    call is needed. 0 或 1 个候选：没有需要判断的对，无需调用 LLM。"""
    ids = [c["id"] for c in candidates]
    btl_scores = [{"id": ids[0], "score": 1.0}] if ids else []
    return {
        "candidate_kind": candidate_kind,
        "pairwise_results": [],
        "btl_scores": btl_scores,
        "ranked_shortlist": ids,
        "diversity_status": "diverse",
    }


def rank_candidates(
    candidates: list[dict],
    candidate_kind: str,
    llm_client: LLMClientLike,
    *,
    prompt_patch: str | None = None,
) -> dict:
    """Rank ``candidates`` via LLM pairwise judgments + deterministic BTL
    aggregation (pure, no transport).
    通过 LLM 成对判断 + 确定性 BTL 聚合对候选排序（纯，无传输层）。

    Args:
        candidates: each a dict with at least an ``"id"`` key.
        candidate_kind: the ``RankingReport.candidate_kind`` value.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        A schema-valid ``RankingReport`` dict.

    Raises:
        RankingParseError (permanent): malformed/unparseable LLM JSON, or a
            schema-invalid assembled report, after bounded retries.
        RankingLLMError (transient): the LLM client itself raised.
    """
    if len(candidates) <= 1:
        return _trivial_report(candidates, candidate_kind)

    ids = [c["id"] for c in candidates]
    pairs = _all_pairs(ids)

    messages = [
        {
            "role": "system",
            "content": apply_prompt_patch(_load_pairwise_ranking_system_prompt(), prompt_patch),
        },
        {
            "role": "user",
            "content": _build_user_message(candidates, pairs),
        },
    ]

    last_error = "no attempts made"

    for _attempt in range(MAX_RETRIES + 1):
        try:
            result = llm_client.chat(messages=messages, response_format="json_object")
        except Exception as exc:  # any client exception is infra/transient
            raise RankingLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

        content = result.content or ""

        obj, error = _parse_pairwise_response(content)
        if error:
            last_error = error
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        btl_scores = _btl_scores(obj["pairwise_results"], ids)
        report = {
            "candidate_kind": candidate_kind,
            "pairwise_results": obj["pairwise_results"],
            "btl_scores": btl_scores,
            "ranked_shortlist": _ranked_shortlist(btl_scores, ids),
            "diversity_status": obj["diversity_status"],
        }

        errors = _validate_ranking_report(report)
        if errors:
            last_error = f"assembled report failed schema validation: {errors}"
            messages.append({"role": "assistant", "content": content})
            messages.append(_retry_feedback_message(last_error))
            continue

        return report

    raise RankingParseError(f"LLM did not produce a valid response after retries: {last_error}")


__all__ = [
    "RankingError",
    "RankingParseError",
    "RankingLLMError",
    "rank_candidates",
    "MAX_RETRIES",
    "ITERS",
]
