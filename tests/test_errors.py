"""Tests for error classification."""

from __future__ import annotations

from src.core.errors import ErrorCategory, RecoveryStrategy, classify_error


class TestClassifyError:
    def test_network_error(self):
        result = classify_error("Connection timed out to server")
        assert result.category == ErrorCategory.NETWORK
        assert result.retryable is True
        assert result.recovery == RecoveryStrategy.RETRY_WITH_BACKOFF

    def test_auth_error(self):
        result = classify_error("401 Unauthorized: invalid API key")
        assert result.category == ErrorCategory.AUTH
        assert result.retryable is False
        assert result.recovery == RecoveryStrategy.ASK_USER

    def test_rate_limit_error(self):
        result = classify_error("429 Too Many Requests: rate limit exceeded")
        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.retryable is True

    def test_context_length_error(self):
        result = classify_error("context_length exceeded: max tokens is 4096")
        assert result.category == ErrorCategory.CONTEXT_LENGTH
        assert result.recovery == RecoveryStrategy.REDUCE_CONTEXT

    def test_model_error(self):
        result = classify_error("model not found: llama-99B")
        assert result.category == ErrorCategory.MODEL

    def test_permission_error(self):
        result = classify_error("403 Forbidden: access denied")
        assert result.category == ErrorCategory.PERMISSION

    def test_parse_error(self):
        result = classify_error("JSON decode error at position 42")
        assert result.category == ErrorCategory.PARSE
        assert result.retryable is True

    def test_unknown_error(self):
        result = classify_error("Something completely unexpected happened")
        assert result.category == ErrorCategory.UNKNOWN
        assert result.recovery == RecoveryStrategy.ASK_USER

    def test_classify_exception(self):
        result = classify_error(ConnectionError("Network unreachable"))
        assert result.category == ErrorCategory.NETWORK
