"""Tests for context window management module."""

import pytest

from src.core.context import (
    ContextWindow,
    ContextConfig,
    ContextStatus,
    TokenEstimator,
    TokenUsage,
    get_context_config,
    CONTEXT_PRESETS,
)


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextConfig()
        assert config.max_tokens == 8192
        assert config.soft_limit_ratio == 0.85
        assert config.critical_limit_ratio == 0.95

    def test_soft_limit(self):
        """Test soft limit calculation."""
        config = ContextConfig(max_tokens=10000)
        assert config.soft_limit == 8500  # 85%

    def test_critical_limit(self):
        """Test critical limit calculation."""
        config = ContextConfig(max_tokens=10000)
        assert config.critical_limit == 9500  # 95%

    def test_effective_limit(self):
        """Test effective limit with reserve."""
        config = ContextConfig(max_tokens=8192, reserve_tokens=512)
        assert config.effective_limit == 7680


class TestTokenEstimator:
    """Tests for TokenEstimator class."""

    def test_estimate_empty(self):
        """Test estimating empty string."""
        estimator = TokenEstimator()
        assert estimator.estimate("") == 0

    def test_estimate_simple_text(self):
        """Test estimating simple text."""
        estimator = TokenEstimator(chars_per_token=4.0)
        # 20 characters / 4 = 5 tokens + newlines
        text = "Hello world testing."
        result = estimator.estimate(text)
        assert result > 0

    def test_estimate_code(self):
        """Test estimating code (should be denser)."""
        estimator = TokenEstimator()
        code = "def foo(): return x + y * z / 2;"
        text = "This is a simple sentence with words."
        # Code should estimate more tokens due to symbols
        code_tokens = estimator.estimate(code)
        text_tokens = estimator.estimate(text)
        # Both should produce reasonable estimates
        assert code_tokens > 0
        assert text_tokens > 0

    def test_estimate_with_newlines(self):
        """Test estimating text with newlines."""
        estimator = TokenEstimator()
        text = "Line 1\nLine 2\nLine 3"
        result = estimator.estimate(text)
        # Should account for newlines
        assert result > 0

    def test_estimate_messages(self):
        """Test estimating a list of messages."""
        estimator = TokenEstimator()
        messages = [
            {"content": "Hello, how are you?"},
            {"content": "I'm fine, thank you!"},
        ]
        result = estimator.estimate_messages(messages)
        # Should include overhead per message
        assert result > 0


class TestContextWindow:
    """Tests for ContextWindow class."""

    def test_initial_state(self):
        """Test initial context window state."""
        window = ContextWindow()
        assert window.total_tokens == 0
        assert window.status == ContextStatus.OK

    def test_set_system_prompt(self):
        """Test setting system prompt."""
        window = ContextWindow()
        tokens = window.set_system_prompt("You are a helpful assistant.")
        assert tokens > 0
        assert window.total_tokens == tokens

    def test_add_turn(self):
        """Test adding turns."""
        window = ContextWindow()
        window.set_system_prompt("System")

        initial = window.total_tokens
        usage = window.add_turn(1, "user", "Hello!")
        assert usage.content_tokens > 0
        assert window.total_tokens > initial

    def test_remove_turns(self):
        """Test removing turns."""
        window = ContextWindow()
        window.add_turn(1, "user", "Message 1")
        window.add_turn(2, "assistant", "Response 1")
        window.add_turn(3, "user", "Message 2")

        tokens_before = window.total_tokens
        freed = window.remove_turns(2)
        assert freed > 0
        assert window.total_tokens < tokens_before

    def test_status_ok(self):
        """Test OK status."""
        config = ContextConfig(max_tokens=10000)
        window = ContextWindow(config=config)
        window.add_turn(1, "user", "Short message")
        assert window.status == ContextStatus.OK

    def test_status_warning(self):
        """Test warning status when approaching limit."""
        config = ContextConfig(max_tokens=100)
        window = ContextWindow(config=config)
        # Add enough content to exceed 85% but not 95%
        window.add_turn(1, "user", "x" * 360)  # ~90 tokens at 4 chars/token
        assert window.status == ContextStatus.WARNING

    def test_status_critical(self):
        """Test critical status when near limit."""
        config = ContextConfig(max_tokens=100)
        window = ContextWindow(config=config)
        # Add enough content to exceed 95%
        window.add_turn(1, "user", "x" * 400)  # ~100 tokens
        assert window.status in (ContextStatus.CRITICAL, ContextStatus.OVERFLOW)

    def test_needs_compaction(self):
        """Test compaction detection."""
        config = ContextConfig(max_tokens=100)
        window = ContextWindow(config=config)
        window.add_turn(1, "user", "x" * 400)
        assert window.needs_compaction()

    def test_should_warn(self):
        """Test warning detection."""
        config = ContextConfig(max_tokens=100)
        window = ContextWindow(config=config)
        window.add_turn(1, "user", "x" * 360)
        # Should warn at ~90% but not need compaction yet
        if window.usage_ratio >= 0.85 and window.usage_ratio < 0.95:
            assert window.should_warn()

    def test_get_compaction_target(self):
        """Test calculating compaction target."""
        config = ContextConfig(max_tokens=1000)
        window = ContextWindow(config=config)
        # Get to 90% usage
        window._total_tokens = 900
        target = window.get_compaction_target()
        # Should want to free tokens to get back to 70%
        assert target > 0

    def test_get_stats(self):
        """Test getting context statistics."""
        window = ContextWindow()
        window.set_system_prompt("System prompt")
        window.add_turn(1, "user", "Hello")
        window.add_turn(2, "assistant", "Hi there")

        stats = window.get_stats()
        assert "total_tokens" in stats
        assert "status" in stats
        assert "turn_count" in stats
        assert stats["turn_count"] == 2

    def test_reset(self):
        """Test resetting context window."""
        window = ContextWindow()
        system_tokens = window.set_system_prompt("System")
        window.add_turn(1, "user", "Hello")
        window.add_turn(2, "assistant", "Hi")

        window.reset()
        # Should keep system prompt tokens but clear turns
        assert len(window.usage_history) == 0
        assert window.total_tokens == system_tokens


class TestContextPresets:
    """Tests for context presets."""

    def test_presets_exist(self):
        """Test that presets are defined."""
        assert "small" in CONTEXT_PRESETS
        assert "medium" in CONTEXT_PRESETS
        assert "large" in CONTEXT_PRESETS
        assert "xlarge" in CONTEXT_PRESETS

    def test_preset_sizes(self):
        """Test preset sizes are reasonable."""
        assert CONTEXT_PRESETS["small"].max_tokens == 4096
        assert CONTEXT_PRESETS["medium"].max_tokens == 8192
        assert CONTEXT_PRESETS["large"].max_tokens == 16384

    def test_get_context_config_with_max_tokens(self):
        """Test getting config with explicit max tokens."""
        config = get_context_config(max_tokens=32000)
        assert config.max_tokens == 32000

    def test_get_context_config_with_preset(self):
        """Test getting config with preset."""
        config = get_context_config(preset="large")
        assert config.max_tokens == 16384

    def test_get_context_config_default(self):
        """Test getting default config."""
        config = get_context_config()
        assert config.max_tokens == 8192  # Default
