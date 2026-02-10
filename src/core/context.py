"""Token estimation and context window management with model-size-aware chunking."""

from __future__ import annotations

from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # overhead per message (role, formatting)
    return total


@dataclass
class ContextBudget:
    """Token budget breakdown for a generation call."""
    total: int
    system_tokens: int
    history_tokens: int
    available_for_input: int
    available_for_output: int


# Tier-specific limits: what fraction of context to use for history vs. output
_TIER_PROFILES = {
    "small": {
        "max_history_ratio": 0.3,   # Small models thrash with too much history
        "max_output_tokens": 512,
        "instruction_chunk_size": 256,  # Break large instructions into smaller pieces
    },
    "medium": {
        "max_history_ratio": 0.5,
        "max_output_tokens": 1024,
        "instruction_chunk_size": 512,
    },
    "large": {
        "max_history_ratio": 0.7,
        "max_output_tokens": 2048,
        "instruction_chunk_size": 0,  # No chunking needed
    },
}


class ContextWindow:
    """Manage context window with model-size-aware budgeting."""

    def __init__(
        self,
        context_length: int = 4096,
        size_tier: str = "medium",
    ) -> None:
        self.context_length = context_length
        self.size_tier = size_tier if size_tier in _TIER_PROFILES else "medium"

    @property
    def profile(self) -> dict:
        return _TIER_PROFILES[self.size_tier]

    def compute_budget(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> ContextBudget:
        """Compute token budget given current system prompt and history."""
        system_tokens = estimate_tokens(system_prompt)
        history_tokens = estimate_messages_tokens(messages)
        max_output = self.profile["max_output_tokens"]

        # Reserve tokens for output
        available_context = self.context_length - system_tokens - max_output
        max_history = int(available_context * self.profile["max_history_ratio"])

        return ContextBudget(
            total=self.context_length,
            system_tokens=system_tokens,
            history_tokens=min(history_tokens, max_history),
            available_for_input=max(0, available_context - min(history_tokens, max_history)),
            available_for_output=max_output,
        )

    def trim_messages(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """Trim message history to fit within context budget for this model tier."""
        budget = self.compute_budget(system_prompt, messages)
        if estimate_messages_tokens(messages) <= budget.history_tokens:
            return messages

        # Keep most recent messages that fit
        trimmed = []
        running_tokens = 0
        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg.get("content", "")) + 4
            if running_tokens + msg_tokens > budget.history_tokens:
                break
            trimmed.insert(0, msg)
            running_tokens += msg_tokens
        return trimmed

    def chunk_instruction(self, instruction: str) -> list[str]:
        """Break a large instruction into smaller chunks for small models.

        Large models get the instruction as-is. Small models get it broken
        into digestible pieces to prevent context thrashing.
        """
        chunk_size = self.profile["instruction_chunk_size"]
        if chunk_size == 0:
            return [instruction]

        tokens = estimate_tokens(instruction)
        if tokens <= chunk_size:
            return [instruction]

        # Split by paragraphs first, then by sentences
        paragraphs = instruction.split("\n\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if estimate_tokens(current + "\n\n" + para) > chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current = (current + "\n\n" + para).strip() if current else para
        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [instruction]
