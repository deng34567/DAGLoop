import pytest

from autoresearch.llm.errors import is_quota_exhaustion, is_resumable_provider_error


@pytest.mark.parametrize("error", [
    "SyntaxError: invalid token at line 401",
    {"stderr_excerpt": "ParserError: invalid token in input data"},
])
def test_invalid_program_token_is_not_a_provider_outage(error):
    assert not is_quota_exhaustion(error)


@pytest.mark.parametrize("error", [
    "AuthenticationError: token:invalid - Invalid token",
    "AuthenticationError: Invalid token",
    {"underlying_reason": "Unauthorized: invalid token"},
    "insufficient_user_quota",
    "pre_consume_token_quota_failed",
    "RateLimitError: OpenAIException - Allocated quota exceeded, please increase your quota limit.",
    {"stderr_excerpt": "Conversation error code: RateLimitError; Allocated quota exceeded"},
    "PermissionDeniedError: Error code: 403 - {'error': {'code': 'account_overdue'}}",
    {"underlying_reason": "Access denied due to overdue account"},
])
def test_explicit_provider_outage_remains_resumable(error):
    assert is_quota_exhaustion(error)
    assert is_resumable_provider_error(error)


def test_typed_provider_api_timeout_is_resumable():
    error = "ComparisonGuardianLLMError: LLM call failed: APITimeoutError: Request timed out."
    assert is_resumable_provider_error(error)
    assert not is_quota_exhaustion(error)


def test_sdk_model_rate_limit_is_resumable_without_claiming_quota_exhaustion():
    error = {"stderr_excerpt": "Remote conversation ended with error: Conversation error code: LLMRateLimitError"}
    assert is_resumable_provider_error(error)
    assert not is_quota_exhaustion(error)


@pytest.mark.parametrize("code", ["LLMServiceUnavailableError", "BadGatewayError",
                                  "LLMTimeoutError", "LLMError; llm_provider_timeout"])
def test_sdk_service_unavailable_is_resumable_without_claiming_quota_exhaustion(code):
    error = {"stderr_excerpt": (
        "Conversation run failed: Remote conversation ended with error: "
        f"Conversation error code: {code}")}
    assert is_resumable_provider_error(error)
    assert not is_quota_exhaustion(error)


@pytest.mark.parametrize("error", ["HTTP 429", "algorithm imposes a rate limit", "candidate regression",
                                  "Disk quota exceeded", "HTTP 502 downloading a dataset",
                                  "BadGatewayError while downloading a dataset",
                                  "timeout training a model", "HTTP 504 downloading a dataset",
                                  "LLMTimeoutError from a training function"])
def test_other_failures_are_not_misclassified_as_sdk_model_throttling(error):
    assert not is_resumable_provider_error(error)
