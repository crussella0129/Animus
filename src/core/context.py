"""Context window management for agents.

This module provides token tracking and context limit management
to prevent context overflow during long conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum
import re


class ContextStatus(Enum):
    """Status of the context window."""
    OK = "ok"                    # Under soft limit
    WARNING = "warning"          # Approaching limit (soft threshold)
    CRITICAL = "critical"        # At limit, compaction needed
    OVERFLOW = "overflow"        # Over limit, must compact


@dataclass
class TokenUsage:
    """Token usage for a single turn."""
    turn_number: int
    role: str                    # "user", "assistant", "system", "tool"
    content_tokens: int
    total_tokens: int = 0        # Running total at this point
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """Configuration for context window management."""
    max_tokens: int = 8192               # Hard limit (model's context window)
    soft_limit_ratio: float = 0.85       # Warn at 85% of max
    critical_limit_ratio: float = 0.95   # Force compact at 95% of max
    reserve_tokens: int = 512            # Reserve for response generation
    chars_per_token: float = 4.0         # Rough estimate for tokenization

    @property
    def soft_limit(self) -> int:
        """Get soft token limit."""
        return int(self.max_tokens * self.soft_limit_ratio)

    @property
    def critical_limit(self) -> int:
        """Get critical token limit."""
        return int(self.max_tokens * self.critical_limit_ratio)

    @property
    def effective_limit(self) -> int:
        """Get effective limit after reserving tokens for response."""
        return self.max_tokens - self.reserve_tokens


class TokenEstimator:
    """Estimates token count from text.

    Uses character-based estimation by default.
    Can be extended to use actual tokenizers.
    """

    def __init__(self, chars_per_token: float = 4.0):
        """Initialize with chars per token ratio.

        Args:
            chars_per_token: Average characters per token.
                            4.0 is typical for English text.
                            3.0 might be better for code.
        """
        self.chars_per_token = chars_per_token

    def estimate(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based heuristic.
        Accounts for:
        - Whitespace compression
        - Special tokens for code
        - Common subword patterns

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        # Count different character types for better estimation
        # Code typically has more tokens per character due to symbols
        code_chars = len(re.findall(r'[{}\[\]();:,.<>!=+\-*/&|^~%]', text))
        whitespace = len(re.findall(r'\s+', text))
        total_chars = len(text)

        # Adjust ratio based on content type
        adjusted_ratio = self.chars_per_token
        if code_chars > total_chars * 0.1:  # Lots of code symbols
            adjusted_ratio = 3.0  # Code is denser
        elif whitespace > total_chars * 0.3:  # Lots of whitespace
            adjusted_ratio = 5.0  # Whitespace-heavy is sparser

        # Base estimation
        base_tokens = int(total_chars / adjusted_ratio)

        # Add tokens for newlines (often separate tokens)
        newlines = text.count('\n')
        base_tokens += newlines

        return max(1, base_tokens)

    def estimate_messages(self, messages: list[dict]) -> int:
        """Estimate tokens for a list of messages.

        Args:
            messages: List of message dicts with 'content' key.

        Returns:
            Total estimated tokens.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.estimate(content)
            # Add overhead for message structure (~4 tokens per message)
            total += 4
        return total


@dataclass
class ContextWindow:
    """Manages the context window for an agent session.

    Tracks token usage, warns on approaching limits,
    and signals when compaction is needed.
    """
    config: ContextConfig = field(default_factory=ContextConfig)
    estimator: TokenEstimator = field(default_factory=TokenEstimator)
    usage_history: list[TokenUsage] = field(default_factory=list)
    _total_tokens: int = 0
    _system_tokens: int = 0

    def set_system_prompt(self, prompt: str) -> int:
        """Set and track the system prompt tokens.

        Args:
            prompt: The system prompt text.

        Returns:
            Token count for system prompt.
        """
        self._system_tokens = self.estimator.estimate(prompt)
        self._total_tokens = self._system_tokens
        return self._system_tokens

    def add_turn(
        self,
        turn_number: int,
        role: str,
        content: str,
        **metadata: Any,
    ) -> TokenUsage:
        """Add a turn and track its token usage.

        Args:
            turn_number: The turn number.
            role: Message role (user, assistant, tool).
            content: Message content.
            **metadata: Additional metadata.

        Returns:
            TokenUsage for this turn.
        """
        content_tokens = self.estimator.estimate(content)
        self._total_tokens += content_tokens

        usage = TokenUsage(
            turn_number=turn_number,
            role=role,
            content_tokens=content_tokens,
            total_tokens=self._total_tokens,
            metadata=metadata,
        )
        self.usage_history.append(usage)
        return usage

    def remove_turns(self, count: int) -> int:
        """Remove oldest turns (for compaction).

        Args:
            count: Number of turns to remove.

        Returns:
            Tokens freed.
        """
        if count <= 0 or not self.usage_history:
            return 0

        freed = 0
        for _ in range(min(count, len(self.usage_history))):
            if self.usage_history:
                usage = self.usage_history.pop(0)
                freed += usage.content_tokens

        self._total_tokens -= freed
        return freed

    @property
    def total_tokens(self) -> int:
        """Get current total token count."""
        return self._total_tokens

    @property
    def available_tokens(self) -> int:
        """Get tokens available for new content."""
        return max(0, self.config.effective_limit - self._total_tokens)

    @property
    def usage_ratio(self) -> float:
        """Get current usage as ratio of max."""
        if self.config.max_tokens == 0:
            return 1.0
        return self._total_tokens / self.config.max_tokens

    @property
    def status(self) -> ContextStatus:
        """Get current context status."""
        if self._total_tokens >= self.config.max_tokens:
            return ContextStatus.OVERFLOW
        elif self._total_tokens >= self.config.critical_limit:
            return ContextStatus.CRITICAL
        elif self._total_tokens >= self.config.soft_limit:
            return ContextStatus.WARNING
        return ContextStatus.OK

    def needs_compaction(self) -> bool:
        """Check if compaction is needed."""
        return self.status in (ContextStatus.CRITICAL, ContextStatus.OVERFLOW)

    def should_warn(self) -> bool:
        """Check if a warning should be shown."""
        return self.status == ContextStatus.WARNING

    def get_compaction_target(self) -> int:
        """Get how many tokens need to be freed.

        Returns:
            Number of tokens that should be freed to get back to safe level.
        """
        if self.status == ContextStatus.OK:
            return 0

        # Target getting back to 70% of limit
        target = int(self.config.max_tokens * 0.7)
        return max(0, self._total_tokens - target)

    def get_turns_to_compact(self) -> int:
        """Calculate how many turns to remove for compaction.

        Returns:
            Number of turns to remove.
        """
        target_tokens = self.get_compaction_target()
        if target_tokens <= 0:
            return 0

        tokens_counted = 0
        turns_to_remove = 0

        for usage in self.usage_history:
            if tokens_counted >= target_tokens:
                break
            tokens_counted += usage.content_tokens
            turns_to_remove += 1

        return turns_to_remove

    def get_stats(self) -> dict[str, Any]:
        """Get context window statistics."""
        return {
            "total_tokens": self._total_tokens,
            "system_tokens": self._system_tokens,
            "max_tokens": self.config.max_tokens,
            "available_tokens": self.available_tokens,
            "usage_ratio": round(self.usage_ratio, 3),
            "status": self.status.value,
            "turn_count": len(self.usage_history),
            "needs_compaction": self.needs_compaction(),
        }

    def reset(self) -> None:
        """Reset the context window."""
        self.usage_history.clear()
        self._total_tokens = self._system_tokens  # Keep system prompt


# Common context window sizes for different models
CONTEXT_PRESETS = {
    "small": ContextConfig(max_tokens=4096),      # Smaller models
    "medium": ContextConfig(max_tokens=8192),     # 7B-13B models
    "large": ContextConfig(max_tokens=16384),     # Larger models
    "xlarge": ContextConfig(max_tokens=32768),    # 32K context
    "xxlarge": ContextConfig(max_tokens=131072),  # 128K context (Claude, etc.)
}


def get_context_config(
    max_tokens: Optional[int] = None,
    preset: Optional[str] = None,
) -> ContextConfig:
    """Get a context configuration.

    Args:
        max_tokens: Override max tokens directly.
        preset: Use a named preset (small, medium, large, xlarge, xxlarge).

    Returns:
        ContextConfig instance.
    """
    if max_tokens is not None:
        return ContextConfig(max_tokens=max_tokens)
    if preset and preset in CONTEXT_PRESETS:
        return CONTEXT_PRESETS[preset]
    return ContextConfig()  # Default
