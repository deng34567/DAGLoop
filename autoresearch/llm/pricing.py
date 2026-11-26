"""
Token-usage pricing / 按 token 用量计价。

``cost_for`` converts an OpenAI-style ``usage`` dict into a decimal string
that matches the cross-system money regex ``^-?[0-9]+(\\.[0-9]{1,6})?$``.

把 OpenAI 风格的 usage 字典换算成符合跨系统金额正则的十进制字符串。

V1 计量(R3 先按 token 算):自部署 / 未知模型没有真实美元单价,走 ``_nominal_rate()``——
默认每 **1M(百万)** token = 1 单位,于是累计 cost == 总 token 数 / 1_000_000(即一个以
"百万 token"为单位的 token 预算,对齐业界 $/1M-token 计价惯例)。这样 Cost Budget 在自部署
模型下也非 0、能触发。接入有公开单价的计费模型时,把真实 $/1M 填进 ``RATES`` 即得真实美元
成本。名义费率可经 env ``VIBE_TOKEN_COST_PER_1M`` 覆盖。
"""

from __future__ import annotations

import os
from decimal import ROUND_HALF_UP, Decimal

# Cross-system contract: every cost string must match this. / 跨系统契约。
_MONEY_QUANT = Decimal("0.000001")  # 6 dp, matches ^-?[0-9]+(\.[0-9]{1,6})?$


class Rate:
    """Per-1M-token rate for one model. / 单模型的每百万 token 费率。"""

    __slots__ = ("prompt_per_1m", "completion_per_1m")

    def __init__(self, prompt_per_1m: str = "0", completion_per_1m: str = "0") -> None:
        self.prompt_per_1m = Decimal(prompt_per_1m)
        self.completion_per_1m = Decimal(completion_per_1m)


# Keyed by model name. Put models with a REAL published per-1M-token $ price here
# to get true dollar cost. Self-deployed / unknown models are NOT listed and fall
# through to _nominal_rate() (token-based).
# 以模型名为键。只有有真实公开单价的计费模型放这里以得真实美元成本;自部署/未知模型不列,
# 走 _nominal_rate()(按 token)。
RATES: dict[str, Rate] = {}


def _nominal_rate() -> Rate:
    """Nominal per-1M-token rate for self-deployed / unknown models.

    Default "1" => cumulative cost == total_tokens / 1_000_000 (a token budget in
    M-token units, matching the industry $/1M-token convention; R3 'count by tokens').
    Env ``VIBE_TOKEN_COST_PER_1M`` overrides.
    自部署/未知模型的名义每百万 token 费率,默认 1(累计 cost == 总 token/1_000_000)。
    """
    r = os.environ.get("VIBE_TOKEN_COST_PER_1M", "1")
    return Rate(r, r)


def _normalize(amount: Decimal) -> str:
    """Quantize to <=6 dp and strip trailing zeros. / 量化到 <=6 位并去尾零。"""
    q = amount.quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)
    # Strip trailing zeros but keep at least an integer part. / 去尾零，保留整数部分。
    text = format(q.normalize(), "f")
    if text in ("-0", "-0.0"):
        text = "0"
    return text


def cost_for(usage: dict, model: str) -> str:
    """
    Compute cost for a usage dict under ``model``'s rate.
    按 model 费率计算一次调用的成本。

    Args:
        usage: ``{prompt_tokens, completion_tokens, total_tokens}``.
        model: model name, e.g. ``"DS3.2"``.

    Returns:
        Decimal string matching ``^-?[0-9]+(\\.[0-9]{1,6})?$``. For self-deployed
        models this is total_tokens/1_000_000 under the nominal rate (a token budget).
    """
    rate = RATES.get(model) or _nominal_rate()
    prompt_tokens = Decimal(int(usage.get("prompt_tokens", 0) or 0))
    completion_tokens = Decimal(int(usage.get("completion_tokens", 0) or 0))

    cost = (
        prompt_tokens / Decimal(1_000_000) * rate.prompt_per_1m
        + completion_tokens / Decimal(1_000_000) * rate.completion_per_1m
    )
    return _normalize(cost)
