"""Recognize explicit provider outages, not ordinary experiment failures."""
from __future__ import annotations


def is_quota_exhaustion(error: object) -> bool:
    if isinstance(error, dict):
        text = " ".join(str(error.get(key) or "") for key in (
            "underlying_reason_code", "underlying_reason", "error", "reason",
            "stderr_excerpt", "stdout_excerpt", "report", "reasons",
        ))
    else:
        text = str(error)
    text = text.lower()
    return any(marker in text for marker in (
        "insufficient_user_quota", "insufficient_quota",
        "pre_consume_token_quota_failed", "token quota is not enough",
        "allocated quota exceeded",
        "account_overdue", "access denied due to overdue account",
        "用户额度不足", "额度不足", "余额不足", "预扣费额度失败",
        "token:invalid",
    )) or ("invalid token" in text and any(marker in text for marker in (
        "authenticationerror", "unauthorized",
    )))


def is_resumable_provider_error(error: object) -> bool:
    """Keep SDK-reported provider outages out of scientific failure counters."""
    return is_quota_exhaustion(error) or any(marker in str(error).lower() for marker in (
        "llmratelimiterror", "llmserviceunavailableerror",
        "conversation error code: badgatewayerror",
        "conversation error code: llmtimeouterror", "llm_provider_timeout",
        "apitimeouterror",
    ))
