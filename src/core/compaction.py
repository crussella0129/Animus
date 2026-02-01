"""Session compaction for long conversations.

This module provides conversation summarization to prevent
context overflow during long agent sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable, Awaitable, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.llm.base import ModelProvider
    from src.core.agent import Turn


class CompactionStrategy(Enum):
    """Strategies for compacting conversation history."""
    SUMMARIZE = "summarize"          # LLM-based summarization
    TRUNCATE = "truncate"            # Simple oldest-first removal
    SLIDING_WINDOW = "sliding"       # Keep last N turns only
    HYBRID = "hybrid"                # Summarize old, keep recent


@dataclass
class CompactionConfig:
    """Configuration for session compaction."""
    strategy: CompactionStrategy = CompactionStrategy.HYBRID
    keep_recent_turns: int = 5           # Always keep this many recent turns
    summary_max_tokens: int = 500        # Max tokens for summary
    trigger_ratio: float = 0.85          # Compact when context reaches this ratio
    min_turns_to_compact: int = 10       # Don't compact if fewer turns than this
    summary_prompt: str = """Summarize the following conversation, preserving:
1. The user's original goal/request
2. Key information discovered or shared
3. Important decisions made
4. Current state of the task

Be concise but include essential context for continuing the conversation.

Conversation:
{conversation}

Summary:"""


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    success: bool
    turns_removed: int
    tokens_freed: int
    summary: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "turns_removed": self.turns_removed,
            "tokens_freed": self.tokens_freed,
            "summary": self.summary,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class SessionCompactor:
    """Handles session compaction to prevent context overflow.

    Compacts conversation history by summarizing old turns
    while preserving recent context.
    """

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        provider: Optional["ModelProvider"] = None,
    ):
        """Initialize the compactor.

        Args:
            config: Compaction configuration.
            provider: LLM provider for summarization.
        """
        self.config = config or CompactionConfig()
        self.provider = provider
        self._compaction_history: list[CompactionResult] = []

    def set_provider(self, provider: "ModelProvider") -> None:
        """Set the LLM provider for summarization."""
        self.provider = provider

    def should_compact(
        self,
        total_tokens: int,
        max_tokens: int,
        turn_count: int,
    ) -> bool:
        """Check if compaction should be triggered.

        Args:
            total_tokens: Current total tokens used.
            max_tokens: Maximum context window size.
            turn_count: Number of conversation turns.

        Returns:
            True if compaction should be triggered.
        """
        if turn_count < self.config.min_turns_to_compact:
            return False

        usage_ratio = total_tokens / max_tokens if max_tokens > 0 else 1.0
        return usage_ratio >= self.config.trigger_ratio

    def _format_turns_for_summary(self, turns: list["Turn"]) -> str:
        """Format turns into a string for summarization."""
        formatted = []
        for turn in turns:
            role = turn.role.upper()
            content = turn.content[:1000]  # Limit individual turn length
            if len(turn.content) > 1000:
                content += "..."
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)

    async def _generate_summary(self, conversation: str) -> str:
        """Generate a summary using the LLM provider."""
        if self.provider is None:
            raise RuntimeError("No LLM provider set for summarization")

        from src.llm.base import Message, GenerationConfig

        prompt = self.config.summary_prompt.format(conversation=conversation)

        result = await self.provider.generate(
            messages=[Message(role="user", content=prompt)],
            config=GenerationConfig(
                max_tokens=self.config.summary_max_tokens,
                temperature=0.3,  # Lower temperature for factual summary
            ),
        )

        return result.content.strip()

    async def compact_turns(
        self,
        turns: list["Turn"],
        target_tokens_to_free: int = 0,
    ) -> tuple[list["Turn"], CompactionResult]:
        """Compact conversation turns.

        Args:
            turns: List of conversation turns.
            target_tokens_to_free: Optional target for tokens to free.

        Returns:
            Tuple of (new_turns, result).
        """
        if len(turns) <= self.config.keep_recent_turns:
            return turns, CompactionResult(
                success=False,
                turns_removed=0,
                tokens_freed=0,
                error="Not enough turns to compact",
            )

        strategy = self.config.strategy

        if strategy == CompactionStrategy.TRUNCATE:
            return await self._compact_truncate(turns)
        elif strategy == CompactionStrategy.SLIDING_WINDOW:
            return await self._compact_sliding(turns)
        elif strategy == CompactionStrategy.SUMMARIZE:
            return await self._compact_summarize(turns)
        elif strategy == CompactionStrategy.HYBRID:
            return await self._compact_hybrid(turns)
        else:
            return turns, CompactionResult(
                success=False,
                turns_removed=0,
                tokens_freed=0,
                error=f"Unknown strategy: {strategy}",
            )

    async def _compact_truncate(
        self,
        turns: list["Turn"],
    ) -> tuple[list["Turn"], CompactionResult]:
        """Simple truncation - remove oldest turns."""
        keep = self.config.keep_recent_turns
        if len(turns) <= keep:
            return turns, CompactionResult(
                success=False,
                turns_removed=0,
                tokens_freed=0,
            )

        removed = turns[:-keep]
        new_turns = turns[-keep:]

        # Estimate tokens freed
        tokens_freed = sum(len(t.content) // 4 for t in removed)

        result = CompactionResult(
            success=True,
            turns_removed=len(removed),
            tokens_freed=tokens_freed,
        )
        self._compaction_history.append(result)

        return new_turns, result

    async def _compact_sliding(
        self,
        turns: list["Turn"],
    ) -> tuple[list["Turn"], CompactionResult]:
        """Sliding window - keep only last N turns."""
        return await self._compact_truncate(turns)

    async def _compact_summarize(
        self,
        turns: list["Turn"],
    ) -> tuple[list["Turn"], CompactionResult]:
        """Summarize all turns into a single context message."""
        try:
            conversation = self._format_turns_for_summary(turns)
            summary = await self._generate_summary(conversation)

            # Create a summary turn
            from src.core.agent import Turn
            summary_turn = Turn(
                role="system",
                content=f"[Previous conversation summary]\n{summary}",
                metadata={"is_summary": True, "summarized_turns": len(turns)},
            )

            tokens_freed = sum(len(t.content) // 4 for t in turns)
            tokens_used = len(summary) // 4

            result = CompactionResult(
                success=True,
                turns_removed=len(turns),
                tokens_freed=tokens_freed - tokens_used,
                summary=summary,
            )
            self._compaction_history.append(result)

            return [summary_turn], result

        except Exception as e:
            return turns, CompactionResult(
                success=False,
                turns_removed=0,
                tokens_freed=0,
                error=str(e),
            )

    async def _compact_hybrid(
        self,
        turns: list["Turn"],
    ) -> tuple[list["Turn"], CompactionResult]:
        """Hybrid approach - summarize old turns, keep recent."""
        keep = self.config.keep_recent_turns

        if len(turns) <= keep:
            return turns, CompactionResult(
                success=False,
                turns_removed=0,
                tokens_freed=0,
            )

        old_turns = turns[:-keep]
        recent_turns = turns[-keep:]

        try:
            # Summarize old turns
            conversation = self._format_turns_for_summary(old_turns)
            summary = await self._generate_summary(conversation)

            # Create a summary turn
            from src.core.agent import Turn
            summary_turn = Turn(
                role="system",
                content=f"[Previous conversation summary]\n{summary}",
                metadata={"is_summary": True, "summarized_turns": len(old_turns)},
            )

            # Combine summary with recent turns
            new_turns = [summary_turn] + recent_turns

            tokens_freed = sum(len(t.content) // 4 for t in old_turns)
            tokens_used = len(summary) // 4

            result = CompactionResult(
                success=True,
                turns_removed=len(old_turns),
                tokens_freed=tokens_freed - tokens_used,
                summary=summary,
            )
            self._compaction_history.append(result)

            return new_turns, result

        except Exception as e:
            # Fallback to truncation if summarization fails
            return await self._compact_truncate(turns)

    def get_compaction_history(self) -> list[CompactionResult]:
        """Get history of compaction operations."""
        return self._compaction_history.copy()

    def clear_history(self) -> None:
        """Clear compaction history."""
        self._compaction_history.clear()


# Convenience function for quick compaction
async def compact_conversation(
    turns: list["Turn"],
    provider: "ModelProvider",
    keep_recent: int = 5,
    strategy: CompactionStrategy = CompactionStrategy.HYBRID,
) -> tuple[list["Turn"], CompactionResult]:
    """Compact a conversation with sensible defaults.

    Args:
        turns: Conversation turns to compact.
        provider: LLM provider for summarization.
        keep_recent: Number of recent turns to keep.
        strategy: Compaction strategy to use.

    Returns:
        Tuple of (compacted_turns, result).
    """
    config = CompactionConfig(
        strategy=strategy,
        keep_recent_turns=keep_recent,
    )
    compactor = SessionCompactor(config=config, provider=provider)
    return await compactor.compact_turns(turns)
