"""Tests for action loop detection."""

import pytest
from src.core.loop_detector import (
    LoopDetector,
    LoopDetectorConfig,
    InterventionLevel,
    INTERVENTION_MESSAGES,
)
from src.core.agent import Agent, AgentConfig


class TestLoopDetectorConfig:
    """Tests for LoopDetectorConfig."""

    def test_defaults(self):
        config = LoopDetectorConfig()
        assert config.window_size == 10
        assert config.nudge_threshold == 3
        assert config.force_threshold == 5
        assert config.break_threshold == 7

    def test_custom(self):
        config = LoopDetectorConfig(nudge_threshold=2, break_threshold=4)
        assert config.nudge_threshold == 2
        assert config.break_threshold == 4


class TestLoopDetector:
    """Tests for LoopDetector."""

    def make_detector(self, **kwargs):
        config = LoopDetectorConfig(**kwargs)
        return LoopDetector(config)

    def test_no_loop_few_actions(self):
        detector = self.make_detector(nudge_threshold=3)
        detector.record("read_file", {"path": "/x"})
        detector.record("read_file", {"path": "/x"})
        assert detector.check() == InterventionLevel.NONE

    def test_nudge_on_threshold(self):
        detector = self.make_detector(nudge_threshold=3)
        for _ in range(3):
            detector.record("read_file", {"path": "/x"})
        assert detector.check() == InterventionLevel.NUDGE

    def test_force_on_threshold(self):
        detector = self.make_detector(nudge_threshold=3, force_threshold=5)
        for _ in range(5):
            detector.record("read_file", {"path": "/x"})
        assert detector.check() == InterventionLevel.FORCE

    def test_break_on_threshold(self):
        detector = self.make_detector(nudge_threshold=3, force_threshold=5, break_threshold=7)
        for _ in range(7):
            detector.record("read_file", {"path": "/x"})
        assert detector.check() == InterventionLevel.BREAK

    def test_different_actions_no_loop(self):
        detector = self.make_detector(nudge_threshold=3)
        detector.record("read_file", {"path": "/a"})
        detector.record("write_file", {"path": "/b"})
        detector.record("read_file", {"path": "/c"})
        assert detector.check() == InterventionLevel.NONE

    def test_same_tool_different_args_no_loop(self):
        detector = self.make_detector(nudge_threshold=3)
        detector.record("read_file", {"path": "/a"})
        detector.record("read_file", {"path": "/b"})
        detector.record("read_file", {"path": "/c"})
        assert detector.check() == InterventionLevel.NONE

    def test_args_ignored_when_disabled(self):
        detector = self.make_detector(nudge_threshold=3, include_args=False)
        detector.record("read_file", {"path": "/a"})
        detector.record("read_file", {"path": "/b"})
        detector.record("read_file", {"path": "/c"})
        assert detector.check() == InterventionLevel.NUDGE

    def test_alternating_pattern_detected(self):
        detector = self.make_detector(nudge_threshold=3, force_threshold=5)
        for _ in range(3):
            detector.record("read_file", {"path": "/x"})
            detector.record("write_file", {"path": "/y"})
        assert detector.check() in (InterventionLevel.NUDGE, InterventionLevel.FORCE)

    def test_reset_clears_state(self):
        detector = self.make_detector(nudge_threshold=3)
        for _ in range(3):
            detector.record("read_file", {"path": "/x"})
        assert detector.check() == InterventionLevel.NUDGE

        detector.reset()
        assert detector.check() == InterventionLevel.NONE
        assert detector.intervention_count == 0

    def test_intervention_count(self):
        detector = self.make_detector(nudge_threshold=3)
        for _ in range(3):
            detector.record("read_file", {"path": "/x"})
        detector.check()
        detector.check()
        assert detector.intervention_count == 2

    def test_stats(self):
        detector = self.make_detector(nudge_threshold=3)
        detector.record("read_file", {"path": "/x"})
        detector.record("read_file", {"path": "/x"})
        stats = detector.stats()
        assert stats["history_length"] == 2
        assert stats["current_consecutive"] == 2

    def test_get_message(self):
        detector = self.make_detector()
        assert detector.get_message(InterventionLevel.NONE) is None
        assert "different approach" in detector.get_message(InterventionLevel.NUDGE)
        assert "MUST" in detector.get_message(InterventionLevel.FORCE)
        assert "LOOP DETECTED" in detector.get_message(InterventionLevel.BREAK)

    def test_window_size_limits_history(self):
        detector = self.make_detector(window_size=5, nudge_threshold=3)
        # Fill with varied actions
        for i in range(5):
            detector.record("tool", {"arg": str(i)})
        # Now add 3 identical — window should only have 5 items
        for _ in range(3):
            detector.record("loop_tool", {"arg": "same"})
        assert detector.check() == InterventionLevel.NUDGE

    def test_no_args(self):
        detector = self.make_detector(nudge_threshold=3)
        for _ in range(3):
            detector.record("simple_tool")
        assert detector.check() == InterventionLevel.NUDGE


class TestInterventionMessages:
    """Tests for intervention message content."""

    def test_all_levels_have_messages(self):
        assert InterventionLevel.NUDGE in INTERVENTION_MESSAGES
        assert InterventionLevel.FORCE in INTERVENTION_MESSAGES
        assert InterventionLevel.BREAK in INTERVENTION_MESSAGES

    def test_messages_are_nonempty(self):
        for msg in INTERVENTION_MESSAGES.values():
            assert len(msg) > 0


class TestAgentLoopDetectorIntegration:
    """Tests for Agent integration with loop detector."""

    @pytest.fixture
    def mock_provider(self):
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = ""
                    tool_calls = None
                return Result()
        return MockProvider()

    def test_loop_detector_created_by_default(self, mock_provider):
        agent = Agent(provider=mock_provider)
        assert agent._loop_detector is not None

    def test_loop_detector_disabled(self, mock_provider):
        config = AgentConfig(enable_loop_detection=False)
        agent = Agent(provider=mock_provider, config=config)
        assert agent._loop_detector is None

    def test_custom_thresholds(self, mock_provider):
        config = AgentConfig(
            loop_nudge_threshold=2,
            loop_force_threshold=4,
            loop_break_threshold=6,
        )
        agent = Agent(provider=mock_provider, config=config)
        assert agent._loop_detector.config.nudge_threshold == 2
        assert agent._loop_detector.config.force_threshold == 4
        assert agent._loop_detector.config.break_threshold == 6

    def test_reset_resets_loop_detector(self, mock_provider):
        agent = Agent(provider=mock_provider)
        agent._loop_detector.record("test", {"x": "1"})
        agent._loop_detector.record("test", {"x": "1"})
        agent.reset()
        assert agent._loop_detector.stats()["history_length"] == 0

    @pytest.mark.asyncio
    async def test_step_records_tool_calls(self):
        """Tool calls made during step should be recorded in loop detector."""
        class ToolCallProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = '{"tool": "read_file", "arguments": {"path": "/test"}}'
                    tool_calls = None
                return Result()

        config = AgentConfig(
            enable_loop_detection=True,
            use_memory=False,
            enable_compaction=False,
        )
        agent = Agent(provider=ToolCallProvider(), config=config)
        await agent.step("read /test")

        assert agent._loop_detector.stats()["history_length"] == 1
