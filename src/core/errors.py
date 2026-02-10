"""Error classification and recovery strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    NETWORK = "network"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    MODEL = "model"
    TOOL = "tool"
    PERMISSION = "permission"
    PARSE = "parse"
    CONTEXT_LENGTH = "context_length"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    REDUCE_CONTEXT = "reduce_context"
    SWITCH_PROVIDER = "switch_provider"
    ASK_USER = "ask_user"
    ABORT = "abort"


@dataclass
class ClassifiedError:
    category: ErrorCategory
    message: str
    recovery: RecoveryStrategy
    retryable: bool = False


# Patterns for classifying errors
_PATTERNS: list[tuple[str, ErrorCategory, RecoveryStrategy, bool]] = [
    (r"connection|timeout|connect|network|dns", ErrorCategory.NETWORK, RecoveryStrategy.RETRY_WITH_BACKOFF, True),
    (r"401|unauthorized|auth|api.key|invalid.key", ErrorCategory.AUTH, RecoveryStrategy.ASK_USER, False),
    (r"429|rate.limit|too.many.requests|quota", ErrorCategory.RATE_LIMIT, RecoveryStrategy.RETRY_WITH_BACKOFF, True),
    (r"context.length|too.long|max.tokens|context.window", ErrorCategory.CONTEXT_LENGTH, RecoveryStrategy.REDUCE_CONTEXT, True),
    (r"model.not.found|no.such.model|invalid.model", ErrorCategory.MODEL, RecoveryStrategy.ASK_USER, False),
    (r"permission|denied|forbidden|403", ErrorCategory.PERMISSION, RecoveryStrategy.ASK_USER, False),
    (r"json|parse|decode|syntax|invalid.format", ErrorCategory.PARSE, RecoveryStrategy.RETRY, True),
    (r"tool|function|execute|command", ErrorCategory.TOOL, RecoveryStrategy.RETRY, True),
]


def classify_error(error: str | Exception) -> ClassifiedError:
    """Classify an error and suggest a recovery strategy."""
    msg = str(error).lower()
    for pattern, category, recovery, retryable in _PATTERNS:
        if re.search(pattern, msg):
            return ClassifiedError(
                category=category,
                message=str(error),
                recovery=recovery,
                retryable=retryable,
            )
    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        message=str(error),
        recovery=RecoveryStrategy.ASK_USER,
        retryable=False,
    )
