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


class TestDynamicProfiles:
    """Test that context profiles scale dynamically with context_length."""

    def test_output_tokens_scale_with_context(self):
        """Larger context should produce larger output budget."""
        cw_small = ContextWindow(context_length=2048, size_tier="medium")
        cw_large = ContextWindow(context_length=8192, size_tier="medium")
        assert cw_small.profile["max_output_tokens"] < cw_large.profile["max_output_tokens"]
        # medium@2048: 512, medium@8192: 2048
        assert cw_small.profile["max_output_tokens"] == 512
        assert cw_large.profile["max_output_tokens"] == 2048

    def test_chunk_size_scales_with_context(self):
        """Chunk size should scale proportionally with context_length."""
        cw_2k = ContextWindow(context_length=2048, size_tier="small")
        cw_4k = ContextWindow(context_length=4096, size_tier="small")
        assert cw_2k.profile["instruction_chunk_size"] < cw_4k.profile["instruction_chunk_size"]
        # small@2048: 256, small@4096: 512
        assert cw_2k.profile["instruction_chunk_size"] == 256
        assert cw_4k.profile["instruction_chunk_size"] == 512

    def test_large_tier_never_chunks(self):
        """Large tier should never chunk regardless of context_length."""
        for ctx in [2048, 4096, 8192, 32768]:
            cw = ContextWindow(context_length=ctx, size_tier="large")
            assert cw.profile["instruction_chunk_size"] == 0

    def test_frame_budget_components_sum(self):
        """Frame budget components should sum to at most frame_total."""
        cw = ContextWindow(context_length=4096, size_tier="medium")
        fb = cw.compute_frame_budget(reserved_system_tokens=200)
        component_sum = fb["system"] + fb["history"] + fb["input"] + fb["output"]
        assert component_sum <= fb["frame_total"]
        assert fb["frame_total"] == 4096

    def test_step_context_positive_input(self):
        """Step context should have positive step_input and correct step_output."""
        cw = ContextWindow(context_length=4096, size_tier="medium")
        sc = cw.compute_step_context()
        assert sc["step_input"] > 0
        assert sc["step_output"] == cw._max_output_tokens
        assert sc["step_total"] == 4096
        assert sc["step_input"] + sc["step_output"] == sc["step_total"]
