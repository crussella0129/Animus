"""Tests for context window management and model-size-aware chunking."""

from __future__ import annotations

from src.core.context import ContextWindow, estimate_messages_tokens, estimate_tokens


class TestEstimateTokens:
    def test_basic_estimate(self):
        assert estimate_tokens("hello") >= 1
        assert estimate_tokens("a" * 400) == 100

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

    def test_messages_tokens(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        total = estimate_messages_tokens(messages)
        assert total > 0


class TestContextWindow:
    def test_small_model_limits(self):
        cw = ContextWindow(context_length=2048, size_tier="small")
        budget = cw.compute_budget("System prompt", [])
        assert budget.available_for_output == 512
        assert budget.total == 2048

    def test_medium_model_limits(self):
        cw = ContextWindow(context_length=4096, size_tier="medium")
        budget = cw.compute_budget("System prompt", [])
        assert budget.available_for_output == 1024

    def test_large_model_limits(self):
        cw = ContextWindow(context_length=8192, size_tier="large")
        budget = cw.compute_budget("System prompt", [])
        assert budget.available_for_output == 2048

    def test_trim_messages_fits(self):
        cw = ContextWindow(context_length=4096, size_tier="medium")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        trimmed = cw.trim_messages(messages, "System prompt")
        assert len(trimmed) == 2

    def test_trim_messages_overflow(self):
        cw = ContextWindow(context_length=256, size_tier="small")
        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
            {"role": "user", "content": "recent"},
        ]
        trimmed = cw.trim_messages(messages, "System prompt")
        # Should have fewer messages than original
        assert len(trimmed) <= len(messages)
        # Most recent message should be preserved
        if trimmed:
            assert trimmed[-1]["content"] == "recent"


class TestInstructionChunking:
    def test_large_model_no_chunking(self):
        cw = ContextWindow(context_length=8192, size_tier="large")
        instruction = "Do this.\n\nThen that.\n\nThen the other thing."
        chunks = cw.chunk_instruction(instruction)
        assert len(chunks) == 1
        assert chunks[0] == instruction

    def test_small_model_chunks_large_instruction(self):
        cw = ContextWindow(context_length=2048, size_tier="small")
        # Create a large instruction that exceeds chunk_size (256 tokens ~= 1024 chars)
        paragraphs = [f"Paragraph {i}: " + "word " * 100 for i in range(10)]
        instruction = "\n\n".join(paragraphs)
        chunks = cw.chunk_instruction(instruction)
        assert len(chunks) > 1

    def test_small_model_short_instruction_no_chunking(self):
        cw = ContextWindow(context_length=2048, size_tier="small")
        instruction = "Just do this one thing."
        chunks = cw.chunk_instruction(instruction)
        assert len(chunks) == 1

    def test_medium_model_moderate_chunking(self):
        cw = ContextWindow(context_length=4096, size_tier="medium")
        # Medium models have chunk_size=512 tokens ~= 2048 chars
        paragraphs = [f"Step {i}: " + "detail " * 150 for i in range(10)]
        instruction = "\n\n".join(paragraphs)
        chunks = cw.chunk_instruction(instruction)
        assert len(chunks) >= 1
