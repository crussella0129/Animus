"""Error classification and recovery strategies for Animus agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import re


class ErrorCategory(Enum):
    """Categories of errors with different recovery strategies."""
    CONTEXT_OVERFLOW = "context_overflow"      # Context window exceeded
    AUTH_FAILURE = "auth_failure"              # API key invalid or expired
    RATE_LIMIT = "rate_limit"                  # Rate limited by provider
    TIMEOUT = "timeout"                        # Request timed out
    TOOL_FAILURE = "tool_failure"              # Tool execution failed
    MODEL_NOT_FOUND = "model_not_found"        # Requested model unavailable
    NETWORK_ERROR = "network_error"            # Connection/network issues
    PERMISSION_DENIED = "permission_denied"    # File/resource access denied
    INVALID_INPUT = "invalid_input"            # Bad input from user/LLM
    UNKNOWN = "unknown"                        # Unclassified error


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error."""
    should_retry: bool = False
    max_retries: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    should_compact: bool = False          # Compact context before retry
    should_fallback: bool = False         # Try fallback provider/model
    should_notify_user: bool = True       # Alert user about the error
    user_message: Optional[str] = None    # Custom message for user


# Default recovery strategies per error category
DEFAULT_STRATEGIES: dict[ErrorCategory, RecoveryStrategy] = {
    ErrorCategory.CONTEXT_OVERFLOW: RecoveryStrategy(
        should_retry=True,
        max_retries=1,
        should_compact=True,
        user_message="Context too long, compacting conversation history..."
    ),
    ErrorCategory.AUTH_FAILURE: RecoveryStrategy(
        should_retry=False,
        should_fallback=True,
        user_message="Authentication failed. Trying alternative provider..."
    ),
    ErrorCategory.RATE_LIMIT: RecoveryStrategy(
        should_retry=True,
        max_retries=5,
        backoff_seconds=5.0,
        backoff_multiplier=2.0,
        user_message="Rate limited, waiting before retry..."
    ),
    ErrorCategory.TIMEOUT: RecoveryStrategy(
        should_retry=True,
        max_retries=2,
        backoff_seconds=2.0,
        user_message="Request timed out, retrying..."
    ),
    ErrorCategory.TOOL_FAILURE: RecoveryStrategy(
        should_retry=False,
        user_message="Tool execution failed."
    ),
    ErrorCategory.MODEL_NOT_FOUND: RecoveryStrategy(
        should_retry=False,
        should_fallback=True,
        user_message="Model not found. Trying fallback model..."
    ),
    ErrorCategory.NETWORK_ERROR: RecoveryStrategy(
        should_retry=True,
        max_retries=3,
        backoff_seconds=2.0,
        user_message="Network error, retrying..."
    ),
    ErrorCategory.PERMISSION_DENIED: RecoveryStrategy(
        should_retry=False,
        user_message="Permission denied for this operation."
    ),
    ErrorCategory.INVALID_INPUT: RecoveryStrategy(
        should_retry=False,
        user_message="Invalid input provided."
    ),
    ErrorCategory.UNKNOWN: RecoveryStrategy(
        should_retry=False,
        user_message="An unexpected error occurred."
    ),
}


@dataclass
class ClassifiedError:
    """An error with its classification and recovery strategy."""
    category: ErrorCategory
    original_error: Exception
    message: str
    strategy: RecoveryStrategy
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_retry(self) -> bool:
        return self.strategy.should_retry

    @property
    def user_message(self) -> str:
        return self.strategy.user_message or str(self.original_error)


def classify_error(error: Exception) -> ClassifiedError:
    """
    Classify an error and determine recovery strategy.

    Args:
        error: The exception to classify

    Returns:
        ClassifiedError with category, message, and recovery strategy
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Context overflow patterns
    context_patterns = [
        "context length", "context window", "token limit",
        "maximum context", "too many tokens", "context_length_exceeded",
        "max_tokens", "context size"
    ]
    if any(p in error_str for p in context_patterns):
        return ClassifiedError(
            category=ErrorCategory.CONTEXT_OVERFLOW,
            original_error=error,
            message="Context window exceeded",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.CONTEXT_OVERFLOW],
        )

    # Auth failure patterns
    auth_patterns = [
        "unauthorized", "authentication", "invalid api key",
        "api key", "401", "403", "forbidden", "invalid_api_key",
        "invalid token", "expired"
    ]
    if any(p in error_str for p in auth_patterns):
        return ClassifiedError(
            category=ErrorCategory.AUTH_FAILURE,
            original_error=error,
            message="Authentication failed",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.AUTH_FAILURE],
        )

    # Rate limit patterns (including billing/quota issues)
    rate_patterns = [
        "rate limit", "rate_limit", "too many requests",
        "429", "throttle", "quota exceeded", "capacity",
        "insufficient_quota", "billing", "payment required",
        "quota_exceeded", "usage limit"
    ]
    if any(p in error_str for p in rate_patterns):
        # Try to extract retry-after if present
        retry_after = 5.0
        match = re.search(r'retry.?after[:\s]*(\d+)', error_str)
        if match:
            retry_after = float(match.group(1))

        strategy = RecoveryStrategy(
            should_retry=True,
            max_retries=5,
            backoff_seconds=retry_after,
            backoff_multiplier=2.0,
            user_message=f"Rate limited, waiting {retry_after}s before retry..."
        )
        return ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            original_error=error,
            message="Rate limit exceeded",
            strategy=strategy,
            metadata={"retry_after": retry_after}
        )

    # Timeout patterns
    timeout_patterns = [
        "timeout", "timed out", "deadline exceeded",
        "connection timed out", "read timeout"
    ]
    if any(p in error_str for p in timeout_patterns):
        return ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            original_error=error,
            message="Request timed out",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.TIMEOUT],
        )

    # Model not found patterns
    model_patterns = [
        "model not found", "model_not_found", "no such model",
        "unknown model", "model does not exist"
    ]
    if any(p in error_str for p in model_patterns):
        return ClassifiedError(
            category=ErrorCategory.MODEL_NOT_FOUND,
            original_error=error,
            message="Model not available",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.MODEL_NOT_FOUND],
        )

    # Network error patterns
    network_patterns = [
        "connection", "network", "socket", "dns",
        "unreachable", "refused", "reset by peer"
    ]
    if any(p in error_str for p in network_patterns):
        return ClassifiedError(
            category=ErrorCategory.NETWORK_ERROR,
            original_error=error,
            message="Network error",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.NETWORK_ERROR],
        )

    # Permission patterns
    permission_patterns = [
        "permission denied", "access denied", "not permitted",
        "operation not permitted", "eacces"
    ]
    if any(p in error_str for p in permission_patterns):
        return ClassifiedError(
            category=ErrorCategory.PERMISSION_DENIED,
            original_error=error,
            message="Permission denied",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.PERMISSION_DENIED],
        )

    # Check for specific exception types using isinstance for reliability
    if isinstance(error, PermissionError):
        return ClassifiedError(
            category=ErrorCategory.PERMISSION_DENIED,
            original_error=error,
            message="Permission denied",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.PERMISSION_DENIED],
        )

    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            original_error=error,
            message="Operation timed out",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.TIMEOUT],
        )

    if isinstance(error, (ConnectionError, ConnectionRefusedError)):
        return ClassifiedError(
            category=ErrorCategory.NETWORK_ERROR,
            original_error=error,
            message="Connection failed",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.NETWORK_ERROR],
        )

    if isinstance(error, OSError) and "permission" in error_str:
        return ClassifiedError(
            category=ErrorCategory.PERMISSION_DENIED,
            original_error=error,
            message="Permission denied",
            strategy=DEFAULT_STRATEGIES[ErrorCategory.PERMISSION_DENIED],
        )

    # Unknown/unclassified
    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        original_error=error,
        message=str(error),
        strategy=DEFAULT_STRATEGIES[ErrorCategory.UNKNOWN],
    )


class AnimusError(Exception):
    """Base exception for Animus-specific errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN):
        super().__init__(message)
        self.category = category
        self.classified = ClassifiedError(
            category=category,
            original_error=self,
            message=message,
            strategy=DEFAULT_STRATEGIES[category],
        )


class ContextOverflowError(AnimusError):
    """Context window exceeded."""

    def __init__(self, message: str = "Context window exceeded"):
        super().__init__(message, ErrorCategory.CONTEXT_OVERFLOW)


class AuthenticationError(AnimusError):
    """Authentication/authorization failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, ErrorCategory.AUTH_FAILURE)


class RateLimitError(AnimusError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 5.0):
        super().__init__(message, ErrorCategory.RATE_LIMIT)
        self.retry_after = retry_after


class ToolExecutionError(AnimusError):
    """Tool execution failed."""

    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool '{tool_name}' failed: {message}", ErrorCategory.TOOL_FAILURE)
        self.tool_name = tool_name


# =============================================================================
# Unified Invoke Error Types (Dify-inspired)
# =============================================================================
# These 5 canonical error types normalize all backend-specific errors so that
# consumer code handles only these types regardless of the underlying provider.


class InvokeError(Exception):
    """
    Base class for unified invoke errors.

    All LLM provider operations should raise one of the 5 InvokeError subclasses,
    enabling consistent error handling across llama-cpp-python, Ollama, LiteLLM,
    and external APIs.
    """

    def __init__(self, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original = original

    def to_dict(self) -> dict[str, Any]:
        """Serialize error for LLM self-correction or logging."""
        return {
            "type": type(self).__name__,
            "message": self.message,
            "original_type": type(self.original).__name__ if self.original else None,
            "original_message": str(self.original) if self.original else None,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict())


class InvokeConnectionError(InvokeError):
    """
    Network or connection failure.

    Covers: DNS resolution, socket errors, connection refused, connection reset,
    network unreachable, SSL/TLS failures.

    Recovery: Retry with backoff, check network connectivity.
    """

    def __init__(self, message: str = "Connection failed", original: Optional[Exception] = None):
        super().__init__(message, original)


class InvokeRateLimitError(InvokeError):
    """
    Rate limit or quota exceeded.

    Covers: HTTP 429, quota exhausted, capacity limits, billing issues,
    concurrent request limits.

    Recovery: Wait for retry_after seconds, switch to backup API key.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        original: Optional[Exception] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, original)
        self.retry_after = retry_after


class InvokeAuthorizationError(InvokeError):
    """
    Authentication or authorization failure.

    Covers: Invalid API key, expired token, missing credentials, forbidden access,
    HTTP 401/403.

    Recovery: Rotate to backup credentials, prompt user for new API key.
    """

    def __init__(self, message: str = "Authorization failed", original: Optional[Exception] = None):
        super().__init__(message, original)


class InvokeServerUnavailableError(InvokeError):
    """
    Server-side error or unavailability.

    Covers: HTTP 5xx errors, model not found, server overloaded, maintenance mode,
    internal server errors, service temporarily unavailable.

    Recovery: Retry with backoff, fall back to alternative model/provider.
    """

    def __init__(self, message: str = "Server unavailable", original: Optional[Exception] = None):
        super().__init__(message, original)


class InvokeBadRequestError(InvokeError):
    """
    Invalid request or input.

    Covers: Malformed JSON, invalid parameters, context length exceeded,
    unsupported model features, HTTP 400.

    Recovery: Fix input, reduce context size, adjust parameters.
    """

    def __init__(self, message: str = "Bad request", original: Optional[Exception] = None):
        super().__init__(message, original)


def translate_invoke_error(error: Exception) -> InvokeError:
    """
    Translate a backend-specific error into a unified InvokeError.

    This function maps errors from various sources (httpx, litellm, llama-cpp,
    standard library) into one of 5 canonical InvokeError types.

    Args:
        error: Any exception from an LLM provider operation.

    Returns:
        An InvokeError subclass appropriate for the error type.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Already an InvokeError — return as-is
    if isinstance(error, InvokeError):
        return error

    # --- Connection Errors ---
    connection_patterns = [
        "connection", "network", "socket", "dns", "unreachable", "refused",
        "reset by peer", "ssl", "tls", "certificate", "handshake", "econnreset",
        "econnrefused", "etimedout", "enetunreach", "ehostunreach",
    ]
    if any(p in error_str for p in connection_patterns):
        return InvokeConnectionError(str(error), error)

    # Python standard connection errors
    if isinstance(error, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
        return InvokeConnectionError(str(error), error)

    # httpx connection errors
    if error_type in ("ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout"):
        return InvokeConnectionError(str(error), error)

    # --- Rate Limit Errors ---
    # Note: "overloaded" removed - server_patterns handles "overload" for server issues
    rate_patterns = [
        "rate limit", "rate_limit", "too many requests", "429", "throttle",
        "quota exceeded", "capacity", "insufficient_quota", "billing",
        "payment required", "quota_exceeded", "usage limit",
    ]
    if any(p in error_str for p in rate_patterns):
        # Try to extract retry-after
        retry_after = None
        retry_match = re.search(r'retry.?after[:\s]*(\d+)', error_str)
        if retry_match:
            retry_after = float(retry_match.group(1))
        return InvokeRateLimitError(str(error), error, retry_after)

    # --- Authorization Errors ---
    auth_patterns = [
        "unauthorized", "authentication", "invalid api key", "api key",
        "401", "403", "forbidden", "invalid_api_key", "invalid token",
        "expired", "credentials", "access denied", "permission denied",
    ]
    if any(p in error_str for p in auth_patterns):
        return InvokeAuthorizationError(str(error), error)

    if isinstance(error, PermissionError):
        return InvokeAuthorizationError(str(error), error)

    # --- Server Unavailable Errors ---
    server_patterns = [
        "500", "502", "503", "504", "internal server error", "server error",
        "service unavailable", "bad gateway", "gateway timeout", "overload",
        "maintenance", "model not found", "model_not_found", "no such model",
        "server unavailable", "temporarily unavailable",
    ]
    if any(p in error_str for p in server_patterns):
        return InvokeServerUnavailableError(str(error), error)

    # httpx HTTP status errors (5xx)
    if hasattr(error, "response"):
        status = getattr(error.response, "status_code", None)
        if status and 500 <= status < 600:
            return InvokeServerUnavailableError(str(error), error)

    # --- Bad Request Errors ---
    bad_request_patterns = [
        "400", "bad request", "invalid", "malformed", "parse error",
        "json", "context length", "context window", "token limit",
        "max_tokens", "context_length_exceeded", "unsupported", "invalid parameter",
    ]
    if any(p in error_str for p in bad_request_patterns):
        return InvokeBadRequestError(str(error), error)

    # httpx HTTP status errors (4xx except 401, 403, 429)
    if hasattr(error, "response"):
        status = getattr(error.response, "status_code", None)
        if status:
            if status == 401 or status == 403:
                return InvokeAuthorizationError(str(error), error)
            if status == 429:
                return InvokeRateLimitError(str(error), error)
            if 400 <= status < 500:
                return InvokeBadRequestError(str(error), error)

    # Timeouts typically indicate connection issues
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return InvokeConnectionError("Request timed out", error)

    # FileNotFoundError for model loading
    if isinstance(error, FileNotFoundError):
        return InvokeServerUnavailableError(f"Model not found: {error}", error)

    # Default to BadRequest for unknown errors (safest default for retrying)
    return InvokeBadRequestError(f"Unknown error: {error}", error)


# Mapping from InvokeError types to ErrorCategory for integration with existing system
INVOKE_TO_CATEGORY: dict[type, ErrorCategory] = {
    InvokeConnectionError: ErrorCategory.NETWORK_ERROR,
    InvokeRateLimitError: ErrorCategory.RATE_LIMIT,
    InvokeAuthorizationError: ErrorCategory.AUTH_FAILURE,
    InvokeServerUnavailableError: ErrorCategory.MODEL_NOT_FOUND,
    InvokeBadRequestError: ErrorCategory.INVALID_INPUT,
}


def invoke_error_to_classified(error: InvokeError) -> ClassifiedError:
    """Convert an InvokeError to a ClassifiedError for the existing recovery system."""
    category = INVOKE_TO_CATEGORY.get(type(error), ErrorCategory.UNKNOWN)
    return ClassifiedError(
        category=category,
        original_error=error.original or error,
        message=error.message,
        strategy=DEFAULT_STRATEGIES[category],
    )
