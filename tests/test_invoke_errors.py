"""Tests for the Unified Invoke Error Translation Layer."""

import asyncio
import json
import pytest
from unittest.mock import Mock

from src.core.errors import (
    InvokeError,
    InvokeConnectionError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeServerUnavailableError,
    InvokeBadRequestError,
    translate_invoke_error,
    invoke_error_to_classified,
    ErrorCategory,
    INVOKE_TO_CATEGORY,
)


# =============================================================================
# InvokeError Base Class Tests
# =============================================================================

class TestInvokeErrorBase:
    """Test InvokeError base class functionality."""

    def test_invoke_error_stores_message_and_original(self):
        """InvokeError should store message and original exception."""
        original = ValueError("original error")
        error = InvokeError("test message", original)

        assert error.message == "test message"
        assert error.original is original
        assert str(error) == "test message"

    def test_invoke_error_without_original(self):
        """InvokeError should work without original exception."""
        error = InvokeError("test message")

        assert error.message == "test message"
        assert error.original is None

    def test_to_dict_serialization(self):
        """InvokeError should serialize to dict."""
        original = ValueError("original error")
        error = InvokeConnectionError("connection failed", original)

        result = error.to_dict()

        assert result["type"] == "InvokeConnectionError"
        assert result["message"] == "connection failed"
        assert result["original_type"] == "ValueError"
        assert result["original_message"] == "original error"

    def test_to_dict_without_original(self):
        """to_dict should handle missing original."""
        error = InvokeConnectionError("connection failed")

        result = error.to_dict()

        assert result["original_type"] is None
        assert result["original_message"] is None

    def test_to_json_serialization(self):
        """InvokeError should serialize to JSON string."""
        error = InvokeConnectionError("connection failed")

        json_str = error.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "InvokeConnectionError"
        assert parsed["message"] == "connection failed"


# =============================================================================
# InvokeError Subclass Tests
# =============================================================================

class TestInvokeErrorSubclasses:
    """Test the 5 canonical InvokeError subclasses."""

    def test_invoke_connection_error(self):
        """InvokeConnectionError default message."""
        error = InvokeConnectionError()
        assert error.message == "Connection failed"
        assert isinstance(error, InvokeError)

    def test_invoke_rate_limit_error(self):
        """InvokeRateLimitError with retry_after."""
        error = InvokeRateLimitError("rate limited", retry_after=30.0)
        assert error.message == "rate limited"
        assert error.retry_after == 30.0
        assert isinstance(error, InvokeError)

    def test_invoke_rate_limit_error_default(self):
        """InvokeRateLimitError default retry_after is None."""
        error = InvokeRateLimitError()
        assert error.retry_after is None

    def test_invoke_authorization_error(self):
        """InvokeAuthorizationError default message."""
        error = InvokeAuthorizationError()
        assert error.message == "Authorization failed"
        assert isinstance(error, InvokeError)

    def test_invoke_server_unavailable_error(self):
        """InvokeServerUnavailableError default message."""
        error = InvokeServerUnavailableError()
        assert error.message == "Server unavailable"
        assert isinstance(error, InvokeError)

    def test_invoke_bad_request_error(self):
        """InvokeBadRequestError default message."""
        error = InvokeBadRequestError()
        assert error.message == "Bad request"
        assert isinstance(error, InvokeError)


# =============================================================================
# Error Translation Tests
# =============================================================================

class TestTranslateInvokeError:
    """Test translate_invoke_error function."""

    # --- Connection Errors ---

    def test_translate_connection_refused(self):
        """ConnectionRefusedError -> InvokeConnectionError."""
        error = ConnectionRefusedError("Connection refused")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)
        assert result.original is error

    def test_translate_connection_reset(self):
        """ConnectionResetError -> InvokeConnectionError."""
        error = ConnectionResetError("Connection reset by peer")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    def test_translate_connection_error(self):
        """Generic ConnectionError -> InvokeConnectionError."""
        error = ConnectionError("Network unreachable")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    def test_translate_timeout_error(self):
        """TimeoutError -> InvokeConnectionError."""
        error = TimeoutError("Request timed out")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    def test_translate_asyncio_timeout(self):
        """asyncio.TimeoutError -> InvokeConnectionError."""
        error = asyncio.TimeoutError()
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    def test_translate_socket_error_message(self):
        """Error with 'socket' in message -> InvokeConnectionError."""
        error = RuntimeError("Socket connection failed")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    def test_translate_dns_error_message(self):
        """Error with 'dns' in message -> InvokeConnectionError."""
        error = RuntimeError("DNS resolution failed")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    def test_translate_ssl_error_message(self):
        """Error with 'ssl' in message -> InvokeConnectionError."""
        error = RuntimeError("SSL handshake failed")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeConnectionError)

    # --- Rate Limit Errors ---

    def test_translate_rate_limit_message(self):
        """Error with 'rate limit' in message -> InvokeRateLimitError."""
        error = RuntimeError("Rate limit exceeded")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)

    def test_translate_429_message(self):
        """Error with '429' in message -> InvokeRateLimitError."""
        error = RuntimeError("HTTP 429 Too Many Requests")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)

    def test_translate_quota_exceeded_message(self):
        """Error with 'quota exceeded' in message -> InvokeRateLimitError."""
        error = RuntimeError("Quota exceeded for this month")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)

    def test_translate_rate_limit_extracts_retry_after(self):
        """Rate limit error should extract retry-after value."""
        error = RuntimeError("Rate limited, retry-after: 30 seconds")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)
        assert result.retry_after == 30.0

    def test_translate_rate_limit_retry_after_alternate_format(self):
        """Rate limit error should extract retry_after with underscore."""
        error = RuntimeError("Rate limited, retry_after: 60")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)
        assert result.retry_after == 60.0

    def test_translate_billing_error(self):
        """Error with 'billing' in message -> InvokeRateLimitError."""
        error = RuntimeError("Billing issue: payment required")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)

    # --- Authorization Errors ---

    def test_translate_unauthorized_message(self):
        """Error with 'unauthorized' in message -> InvokeAuthorizationError."""
        error = RuntimeError("Unauthorized access")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_401_message(self):
        """Error with '401' in message -> InvokeAuthorizationError."""
        error = RuntimeError("HTTP 401 Unauthorized")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_403_message(self):
        """Error with '403' in message -> InvokeAuthorizationError."""
        error = RuntimeError("HTTP 403 Forbidden")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_invalid_api_key_message(self):
        """Error with 'invalid api key' in message -> InvokeAuthorizationError."""
        error = RuntimeError("Invalid API key provided")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_permission_error(self):
        """PermissionError -> InvokeAuthorizationError."""
        error = PermissionError("Access denied")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_expired_token_message(self):
        """Error with 'expired' in message -> InvokeAuthorizationError."""
        error = RuntimeError("Token expired")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    # --- Server Unavailable Errors ---

    def test_translate_500_message(self):
        """Error with '500' in message -> InvokeServerUnavailableError."""
        error = RuntimeError("HTTP 500 Internal Server Error")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)

    def test_translate_502_message(self):
        """Error with '502' in message -> InvokeServerUnavailableError."""
        error = RuntimeError("HTTP 502 Bad Gateway")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)

    def test_translate_503_message(self):
        """Error with '503' in message -> InvokeServerUnavailableError."""
        error = RuntimeError("HTTP 503 Service Unavailable")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)

    def test_translate_model_not_found_message(self):
        """Error with 'model not found' in message -> InvokeServerUnavailableError."""
        error = RuntimeError("Model not found: gpt-5")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)

    def test_translate_file_not_found_error(self):
        """FileNotFoundError -> InvokeServerUnavailableError."""
        error = FileNotFoundError("model.gguf not found")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)

    def test_translate_server_overloaded_message(self):
        """Error with 'overload' in message -> InvokeServerUnavailableError."""
        error = RuntimeError("Server overloaded")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)

    # --- Bad Request Errors ---

    def test_translate_400_message(self):
        """Error with '400' in message -> InvokeBadRequestError."""
        error = RuntimeError("HTTP 400 Bad Request")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)

    def test_translate_invalid_json_message(self):
        """Error with 'json' in message -> InvokeBadRequestError."""
        error = RuntimeError("Invalid JSON in request")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)

    def test_translate_context_length_message(self):
        """Error with 'context length' in message -> InvokeBadRequestError."""
        error = RuntimeError("Context length exceeded")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)

    def test_translate_malformed_message(self):
        """Error with 'malformed' in message -> InvokeBadRequestError."""
        error = RuntimeError("Malformed request body")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)

    def test_translate_max_tokens_message(self):
        """Error with 'max_tokens' in message -> InvokeBadRequestError."""
        error = RuntimeError("max_tokens exceeds limit")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)

    # --- Special Cases ---

    def test_translate_invoke_error_passthrough(self):
        """InvokeError should pass through unchanged."""
        error = InvokeConnectionError("already translated")
        result = translate_invoke_error(error)
        assert result is error

    def test_translate_unknown_error(self):
        """Unknown error -> InvokeBadRequestError (safest default)."""
        error = RuntimeError("Something completely unknown happened")
        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)
        assert "Unknown error" in result.message

    # --- HTTP Response Errors ---

    def test_translate_http_response_401(self):
        """Error with response.status_code=401 -> InvokeAuthorizationError."""
        response = Mock()
        response.status_code = 401
        error = RuntimeError("HTTP error")
        error.response = response

        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_http_response_403(self):
        """Error with response.status_code=403 -> InvokeAuthorizationError."""
        response = Mock()
        response.status_code = 403
        error = RuntimeError("HTTP error")
        error.response = response

        result = translate_invoke_error(error)
        assert isinstance(result, InvokeAuthorizationError)

    def test_translate_http_response_429(self):
        """Error with response.status_code=429 -> InvokeRateLimitError."""
        response = Mock()
        response.status_code = 429
        error = RuntimeError("HTTP error")
        error.response = response

        result = translate_invoke_error(error)
        assert isinstance(result, InvokeRateLimitError)

    def test_translate_http_response_400(self):
        """Error with response.status_code=400 -> InvokeBadRequestError."""
        response = Mock()
        response.status_code = 400
        error = RuntimeError("HTTP error")
        error.response = response

        result = translate_invoke_error(error)
        assert isinstance(result, InvokeBadRequestError)

    def test_translate_http_response_500(self):
        """Error with response.status_code=500 -> InvokeServerUnavailableError."""
        response = Mock()
        response.status_code = 500
        error = RuntimeError("HTTP error")
        error.response = response

        result = translate_invoke_error(error)
        assert isinstance(result, InvokeServerUnavailableError)


# =============================================================================
# Integration with ClassifiedError System
# =============================================================================

class TestInvokeToClassified:
    """Test integration with existing ClassifiedError system."""

    def test_invoke_to_category_mapping_exists(self):
        """All InvokeError subclasses should map to ErrorCategory."""
        assert InvokeConnectionError in INVOKE_TO_CATEGORY
        assert InvokeRateLimitError in INVOKE_TO_CATEGORY
        assert InvokeAuthorizationError in INVOKE_TO_CATEGORY
        assert InvokeServerUnavailableError in INVOKE_TO_CATEGORY
        assert InvokeBadRequestError in INVOKE_TO_CATEGORY

    def test_invoke_to_category_values(self):
        """InvokeError types should map to correct ErrorCategory."""
        assert INVOKE_TO_CATEGORY[InvokeConnectionError] == ErrorCategory.NETWORK_ERROR
        assert INVOKE_TO_CATEGORY[InvokeRateLimitError] == ErrorCategory.RATE_LIMIT
        assert INVOKE_TO_CATEGORY[InvokeAuthorizationError] == ErrorCategory.AUTH_FAILURE
        assert INVOKE_TO_CATEGORY[InvokeServerUnavailableError] == ErrorCategory.MODEL_NOT_FOUND
        assert INVOKE_TO_CATEGORY[InvokeBadRequestError] == ErrorCategory.INVALID_INPUT

    def test_invoke_error_to_classified_connection(self):
        """InvokeConnectionError should convert to ClassifiedError."""
        error = InvokeConnectionError("connection failed")
        result = invoke_error_to_classified(error)

        assert result.category == ErrorCategory.NETWORK_ERROR
        assert result.message == "connection failed"
        assert result.strategy.should_retry is True

    def test_invoke_error_to_classified_rate_limit(self):
        """InvokeRateLimitError should convert to ClassifiedError."""
        error = InvokeRateLimitError("rate limited")
        result = invoke_error_to_classified(error)

        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.strategy.should_retry is True
        assert result.strategy.max_retries >= 1

    def test_invoke_error_to_classified_authorization(self):
        """InvokeAuthorizationError should convert to ClassifiedError."""
        error = InvokeAuthorizationError("invalid api key")
        result = invoke_error_to_classified(error)

        assert result.category == ErrorCategory.AUTH_FAILURE
        assert result.strategy.should_fallback is True

    def test_invoke_error_to_classified_server_unavailable(self):
        """InvokeServerUnavailableError should convert to ClassifiedError."""
        error = InvokeServerUnavailableError("model not found")
        result = invoke_error_to_classified(error)

        assert result.category == ErrorCategory.MODEL_NOT_FOUND
        assert result.strategy.should_fallback is True

    def test_invoke_error_to_classified_bad_request(self):
        """InvokeBadRequestError should convert to ClassifiedError."""
        error = InvokeBadRequestError("invalid parameter")
        result = invoke_error_to_classified(error)

        assert result.category == ErrorCategory.INVALID_INPUT
        assert result.strategy.should_retry is False

    def test_invoke_error_to_classified_preserves_original(self):
        """invoke_error_to_classified should preserve original exception."""
        original = ValueError("original")
        error = InvokeConnectionError("connection failed", original)
        result = invoke_error_to_classified(error)

        assert result.original_error is original
