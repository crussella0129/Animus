"""Tests for the tokenizer module."""

import pytest
from src.core.tokenizer import (
    count_tokens,
    count_tokens_messages,
    count_tokens_cached,
    truncate_to_tokens,
    split_by_tokens,
    is_tiktoken_available,
    get_model_context_limit,
    estimate_tokens_rough,
)


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        assert count_tokens("") == 0

    def test_single_word(self):
        """Single word should return reasonable token count."""
        tokens = count_tokens("hello")
        assert tokens >= 1
        assert tokens <= 2

    def test_sentence(self):
        """Sentence should return reasonable token count."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = count_tokens(text)
        # Should be roughly 10-12 tokens
        assert tokens >= 8
        assert tokens <= 15

    def test_code_snippet(self):
        """Code should tokenize reasonably."""
        code = "def hello_world():\n    print('Hello, World!')"
        tokens = count_tokens(code)
        # Code tends to have more tokens per character
        assert tokens >= 10
        assert tokens <= 25

    def test_long_text(self):
        """Long text should scale appropriately."""
        text = "Hello world. " * 100
        tokens = count_tokens(text)
        # Should be roughly 300 tokens (3 per "Hello world. ")
        assert tokens >= 200
        assert tokens <= 400

    def test_non_ascii(self):
        """Non-ASCII characters should be handled."""
        text = "你好世界"  # Chinese for "Hello World"
        tokens = count_tokens(text)
        assert tokens >= 1

    def test_special_characters(self):
        """Special characters should be handled."""
        text = "Hello! @#$%^&*() World"
        tokens = count_tokens(text)
        assert tokens >= 5


class TestCountTokensMessages:
    """Tests for count_tokens_messages function."""

    def test_single_message(self):
        """Single message should include overhead."""
        messages = [{"role": "user", "content": "Hello"}]
        tokens = count_tokens_messages(messages)
        # Should include message overhead
        assert tokens >= 5

    def test_multiple_messages(self):
        """Multiple messages should accumulate."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = count_tokens_messages(messages)
        # Should be sum of content tokens + overhead per message + priming
        assert tokens >= 15

    def test_empty_messages(self):
        """Empty message list should return minimal tokens."""
        tokens = count_tokens_messages([])
        # Just priming tokens
        assert tokens == 2


class TestCountTokensCached:
    """Tests for count_tokens_cached function."""

    def test_caching_returns_same_result(self):
        """Same text should return same result from cache."""
        text = "This is a test sentence for caching."
        result1 = count_tokens_cached(text)
        result2 = count_tokens_cached(text)
        assert result1 == result2

    def test_different_text_different_cache(self):
        """Different text should return different results."""
        result1 = count_tokens_cached("Hello")
        result2 = count_tokens_cached("Goodbye")
        # Results may or may not be equal, but both should work
        assert isinstance(result1, int)
        assert isinstance(result2, int)


class TestTruncateToTokens:
    """Tests for truncate_to_tokens function."""

    def test_short_text_unchanged(self):
        """Short text should not be truncated."""
        text = "Hello world"
        result = truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_long_text_truncated(self):
        """Long text should be truncated."""
        text = "Hello world. " * 100
        result = truncate_to_tokens(text, max_tokens=10)
        assert len(result) < len(text)
        assert result.endswith("...")

    def test_custom_suffix(self):
        """Custom suffix should be used."""
        text = "Hello world. " * 100
        result = truncate_to_tokens(text, max_tokens=10, suffix="[truncated]")
        assert result.endswith("[truncated]")

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert truncate_to_tokens("", max_tokens=10) == ""

    def test_zero_max_tokens(self):
        """Zero max tokens should return empty string."""
        assert truncate_to_tokens("Hello world", max_tokens=0) == ""


class TestSplitByTokens:
    """Tests for split_by_tokens function."""

    def test_short_text_single_chunk(self):
        """Short text should be single chunk."""
        text = "Hello world"
        chunks = split_by_tokens(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        """Long text should be split into chunks."""
        text = "Hello world. " * 100
        chunks = split_by_tokens(text, chunk_size=50)
        assert len(chunks) > 1
        # All chunks together should cover the text
        combined = "".join(chunks)
        # May have some overlap, so combined could be longer
        assert len(combined) >= len(text)

    def test_with_overlap(self):
        """Overlap should cause overlapping chunks."""
        text = "Hello world. " * 100
        chunks = split_by_tokens(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        # With overlap, combined length should be longer
        combined = "".join(chunks)
        assert len(combined) >= len(text)

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert split_by_tokens("", chunk_size=10) == []

    def test_zero_chunk_size(self):
        """Zero chunk size should return empty list."""
        assert split_by_tokens("Hello", chunk_size=0) == []


class TestIsTiktokenAvailable:
    """Tests for is_tiktoken_available function."""

    def test_returns_bool(self):
        """Should return a boolean."""
        result = is_tiktoken_available()
        assert isinstance(result, bool)

    def test_consistent_result(self):
        """Should return same result on multiple calls."""
        result1 = is_tiktoken_available()
        result2 = is_tiktoken_available()
        assert result1 == result2


class TestGetModelContextLimit:
    """Tests for get_model_context_limit function."""

    def test_qwen_model(self):
        """Qwen models should have 32K context."""
        limit = get_model_context_limit("qwen2.5-coder-7b")
        assert limit == 32768

    def test_llama_model(self):
        """LLaMA models should have appropriate context."""
        limit = get_model_context_limit("llama-2-7b")
        assert limit == 4096

    def test_llama3_model(self):
        """LLaMA 3 models should have 8K context."""
        limit = get_model_context_limit("llama-3-8b")
        assert limit == 8192

    def test_mistral_model(self):
        """Mistral models should have 8K context."""
        limit = get_model_context_limit("mistral-7b")
        assert limit == 8192

    def test_unknown_model(self):
        """Unknown models should get default context."""
        limit = get_model_context_limit("some-unknown-model")
        assert limit == 4096  # Default

    def test_case_insensitive(self):
        """Model name matching should be case insensitive."""
        limit1 = get_model_context_limit("Qwen2.5-Coder")
        limit2 = get_model_context_limit("QWEN2.5-CODER")
        assert limit1 == limit2


class TestEstimateTokensRough:
    """Tests for estimate_tokens_rough function."""

    def test_english_text(self):
        """English text should estimate ~4 chars per token."""
        text = "Hello world test"  # 16 chars
        tokens = estimate_tokens_rough(text)
        assert tokens == 4  # 16 // 4

    def test_code(self):
        """Code should estimate ~3 chars per token."""
        code = "def foo(): pass"  # Has code indicators
        tokens = estimate_tokens_rough(code)
        assert tokens == 5  # 15 // 3

    def test_empty_string(self):
        """Empty string should return 0."""
        assert estimate_tokens_rough("") == 0


class TestIntegration:
    """Integration tests for tokenizer module."""

    def test_count_and_truncate(self):
        """count_tokens and truncate_to_tokens should be consistent."""
        text = "Hello world. " * 50
        max_tokens = 20

        truncated = truncate_to_tokens(text, max_tokens)
        token_count = count_tokens(truncated)

        # Truncated text should be within limit
        # (may be slightly over due to suffix)
        assert token_count <= max_tokens + 5

    def test_split_coverage(self):
        """split_by_tokens should cover all content."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        original_tokens = count_tokens(text)

        chunks = split_by_tokens(text, chunk_size=50, overlap=0)
        chunk_tokens = sum(count_tokens(chunk) for chunk in chunks)

        # Chunks should have at least as many tokens as original
        assert chunk_tokens >= original_tokens * 0.9  # Allow small variance
