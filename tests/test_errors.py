"""Tests for error classification system."""

import pytest
from src.core.errors import (
    classify_error,
    ErrorCategory,
    ClassifiedError,
    RecoveryStrategy,
    AnimusError,
    ContextOverflowError,
    AuthenticationError,
    RateLimitError,
    ToolExecutionError,
)


class TestErrorClassification:
    """Tests for classify_error function."""

    def test_classify_context_overflow(self):
        """Test classification of context overflow errors."""
        errors = [
            Exception("context length exceeded"),
            Exception("Error: maximum context window reached"),
            Exception("too many tokens in request"),
            Exception("context_length_exceeded: max 4096"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.CONTEXT_OVERFLOW
            assert classified.strategy.should_compact

    def test_classify_auth_failure(self):
        """Test classification of authentication errors."""
        errors = [
            Exception("401 Unauthorized"),
            Exception("Invalid API key provided"),
            Exception("authentication failed"),
            Exception("403 Forbidden"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.AUTH_FAILURE
            assert classified.strategy.should_fallback

    def test_classify_rate_limit(self):
        """Test classification of rate limit errors."""
        errors = [
            Exception("429 Too Many Requests"),
            Exception("rate limit exceeded"),
            Exception("quota exceeded, retry after 60 seconds"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.RATE_LIMIT
            assert classified.strategy.should_retry

    def test_classify_rate_limit_extracts_retry_after(self):
        """Test that retry-after is extracted from error message."""
        error = Exception("rate limited, retry-after: 30")
        classified = classify_error(error)
        assert classified.category == ErrorCategory.RATE_LIMIT
        assert classified.metadata.get("retry_after") == 30.0

    def test_classify_timeout(self):
        """Test classification of timeout errors."""
        errors = [
            Exception("request timed out"),
            Exception("connection timeout"),
            Exception("deadline exceeded"),
            TimeoutError("operation timed out"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.TIMEOUT
            assert classified.strategy.should_retry

    def test_classify_network_error(self):
        """Test classification of network errors."""
        errors = [
            Exception("connection refused"),
            Exception("network unreachable"),
            Exception("socket error"),
            ConnectionError("failed to connect"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.NETWORK_ERROR
            assert classified.strategy.should_retry

    def test_classify_permission_denied(self):
        """Test classification of permission errors."""
        errors = [
            Exception("permission denied"),
            Exception("access denied"),
            PermissionError("operation not permitted"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.PERMISSION_DENIED
            assert not classified.strategy.should_retry

    def test_classify_model_not_found(self):
        """Test classification of model not found errors."""
        errors = [
            Exception("model not found: llama-7b"),
            Exception("no such model exists"),
            Exception("unknown model: gpt-5"),
        ]
        for error in errors:
            classified = classify_error(error)
            assert classified.category == ErrorCategory.MODEL_NOT_FOUND
            assert classified.strategy.should_fallback

    def test_classify_unknown(self):
        """Test classification of unknown errors."""
        error = Exception("some random error that doesn't match any pattern")
        classified = classify_error(error)
        assert classified.category == ErrorCategory.UNKNOWN
        assert not classified.strategy.should_retry


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy."""

    def test_default_strategy(self):
        """Test default recovery strategy values."""
        strategy = RecoveryStrategy()
        assert not strategy.should_retry
        assert strategy.max_retries == 3
        assert strategy.backoff_seconds == 1.0

    def test_custom_strategy(self):
        """Test custom recovery strategy."""
        strategy = RecoveryStrategy(
            should_retry=True,
            max_retries=5,
            backoff_seconds=2.0,
            backoff_multiplier=3.0,
            should_compact=True,
        )
        assert strategy.should_retry
        assert strategy.max_retries == 5
        assert strategy.backoff_seconds == 2.0
        assert strategy.backoff_multiplier == 3.0
        assert strategy.should_compact


class TestClassifiedError:
    """Tests for ClassifiedError dataclass."""

    def test_classified_error_properties(self):
        """Test ClassifiedError convenience properties."""
        strategy = RecoveryStrategy(
            should_retry=True,
            user_message="Custom message",
        )
        classified = ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            original_error=Exception("rate limit"),
            message="Rate limited",
            strategy=strategy,
        )
        assert classified.should_retry
        assert classified.user_message == "Custom message"

    def test_classified_error_fallback_message(self):
        """Test that user_message falls back to original error."""
        strategy = RecoveryStrategy()  # No user_message
        error = Exception("original error text")
        classified = ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            original_error=error,
            message="classified message",
            strategy=strategy,
        )
        assert classified.user_message == "original error text"


class TestAnimusErrors:
    """Tests for custom Animus exception classes."""

    def test_context_overflow_error(self):
        """Test ContextOverflowError."""
        error = ContextOverflowError()
        assert error.category == ErrorCategory.CONTEXT_OVERFLOW
        assert error.classified.strategy.should_compact

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid token")
        assert error.category == ErrorCategory.AUTH_FAILURE
        assert "Invalid token" in str(error)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(retry_after=60.0)
        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.retry_after == 60.0

    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        error = ToolExecutionError("read_file", "file not found")
        assert error.category == ErrorCategory.TOOL_FAILURE
        assert "read_file" in str(error)
        assert "file not found" in str(error)
        assert error.tool_name == "read_file"
