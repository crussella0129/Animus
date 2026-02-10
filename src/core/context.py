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


@dataclass
class TokenUsage:
    """Cumulative token usage for a session."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_input(self, tokens: int) -> None:
        self.input_tokens += tokens

    def add_output(self, tokens: int) -> None:
        self.output_tokens += tokens


# ---------------------------------------------------------------------------
# Ratio-based tier profiles: absolute values computed from context_length
# ---------------------------------------------------------------------------

_TIER_RATIOS = {
    "small": {
        "max_history_ratio": 0.3,   # Small models thrash with too much history
        "output_ratio": 0.25,
        "chunk_ratio": 0.125,       # Break large instructions into smaller pieces
    },
    "medium": {
        "max_history_ratio": 0.5,
        "output_ratio": 0.25,
        "chunk_ratio": 0.125,
    },
    "large": {
        "max_history_ratio": 0.7,
        "output_ratio": 0.25,
        "chunk_ratio": 0.0,         # No chunking needed
    },
}

# Deprecated: computed alias for backward compatibility with external references.
# Values match the original fixed profiles at context_length=2048/4096/8192.
_TIER_PROFILES = {
    "small": {
        "max_history_ratio": 0.3,
        "max_output_tokens": 512,          # 2048 * 0.25
        "instruction_chunk_size": 256,     # 2048 * 0.125
    },
    "medium": {
        "max_history_ratio": 0.5,
        "max_output_tokens": 1024,         # 4096 * 0.25
        "instruction_chunk_size": 512,     # 4096 * 0.125
    },
    "large": {
        "max_history_ratio": 0.7,
        "max_output_tokens": 2048,         # 8192 * 0.25
        "instruction_chunk_size": 0,       # No chunking
    },
}


class ContextWindow:
    """Manage context window with model-size-aware budgeting.

    Token budgets are computed dynamically from context_length * tier ratios,
    scaling proportionally with the model's actual context window.
    """

    def __init__(
        self,
        context_length: int = 4096,
        size_tier: str = "medium",
    ) -> None:
        self.context_length = context_length
        self.size_tier = size_tier if size_tier in _TIER_RATIOS else "medium"

        # Pre-compute absolute values from ratios
        ratios = _TIER_RATIOS[self.size_tier]
        self._max_history_ratio = ratios["max_history_ratio"]
        self._max_output_tokens = int(context_length * ratios["output_ratio"])
        self._instruction_chunk_size = int(context_length * ratios["chunk_ratio"])

    @property
    def profile(self) -> dict:
        """Return profile dict for backward compatibility."""
        return {
            "max_history_ratio": self._max_history_ratio,
            "max_output_tokens": self._max_output_tokens,
            "instruction_chunk_size": self._instruction_chunk_size,
        }

    def compute_budget(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> ContextBudget:
        """Compute token budget given current system prompt and history."""
        system_tokens = estimate_tokens(system_prompt)
        history_tokens = estimate_messages_tokens(messages)
        max_output = self._max_output_tokens

        # Reserve tokens for output
        available_context = self.context_length - system_tokens - max_output
        max_history = int(available_context * self._max_history_ratio)

        return ContextBudget(
            total=self.context_length,
            system_tokens=system_tokens,
            history_tokens=min(history_tokens, max_history),
            available_for_input=max(0, available_context - min(history_tokens, max_history)),
            available_for_output=max_output,
        )

    def compute_frame_budget(self, reserved_system_tokens: int = 200) -> dict:
        """Compute the 'flicker fusion' frame budget for a single generation call.

        Returns a dict with:
            frame_total: total tokens available in the frame
            system: tokens reserved for system prompt
            history: max tokens for conversation history
            input: tokens available for new input
            output: tokens reserved for model output
        """
        frame_total = self.context_length
        system = reserved_system_tokens
        output = self._max_output_tokens
        remaining = frame_total - system - output
        history = int(remaining * self._max_history_ratio)
        input_budget = max(0, remaining - history)

        return {
            "frame_total": frame_total,
            "system": system,
            "history": history,
            "input": input_budget,
            "output": output,
        }

    def compute_step_context(self) -> dict:
        """Compute context budget for a planner step (no history, all context for input).

        Returns a dict with:
            step_input: tokens available for input (system prompt + step description)
            step_output: tokens reserved for output
            step_total: total context available
        """
        step_output = self._max_output_tokens
        step_input = max(0, self.context_length - step_output)

        return {
            "step_input": step_input,
            "step_output": step_output,
            "step_total": self.context_length,
        }

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
        chunk_size = self._instruction_chunk_size
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
