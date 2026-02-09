"""Tests for model fallback chain."""

import time
import pytest
from src.core.fallback import ModelFallbackChain, FallbackModel, FallbackEvent
from src.core.agent import Agent, AgentConfig


class TestFallbackModel:
    """Tests for FallbackModel dataclass."""

    def test_defaults(self):
        m = FallbackModel(model="local/test-7b")
        assert m.model == "local/test-7b"
        assert m.max_failures == 3
        assert m.cooldown_seconds == 60.0
        assert m.consecutive_failures == 0
        assert m.total_failures == 0
        assert m.total_successes == 0

    def test_custom_config(self):
        m = FallbackModel(model="gpt-4o", max_failures=1, cooldown_seconds=30.0)
        assert m.max_failures == 1
        assert m.cooldown_seconds == 30.0


class TestModelFallbackChain:
    """Tests for ModelFallbackChain."""

    def make_chain(self, models=None, auto_deescalate=True):
        if models is None:
            models = [
                FallbackModel("local/small-7b", max_failures=2),
                FallbackModel("local/large-32b", max_failures=2),
                FallbackModel("gpt-4o", max_failures=1),
            ]
        return ModelFallbackChain(models=models, auto_deescalate=auto_deescalate)

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ModelFallbackChain(models=[])

    def test_initial_model(self):
        chain = self.make_chain()
        assert chain.current_model == "local/small-7b"
        assert chain.current_index == 0
        assert not chain.is_escalated

    def test_single_failure_no_escalation(self):
        chain = self.make_chain()
        escalated = chain.record_failure()
        assert not escalated
        assert chain.current_model == "local/small-7b"

    def test_escalation_after_max_failures(self):
        chain = self.make_chain()
        chain.record_failure()
        escalated = chain.record_failure()
        assert escalated
        assert chain.current_model == "local/large-32b"
        assert chain.current_index == 1
        assert chain.is_escalated

    def test_double_escalation(self):
        chain = self.make_chain()
        # Exhaust first model
        chain.record_failure()
        chain.record_failure()
        assert chain.current_model == "local/large-32b"

        # Exhaust second model
        chain.record_failure()
        chain.record_failure()
        assert chain.current_model == "gpt-4o"
        assert chain.current_index == 2

    def test_last_model_no_further_escalation(self):
        chain = self.make_chain()
        # Exhaust all models
        for _ in range(2):
            chain.record_failure()
        for _ in range(2):
            chain.record_failure()
        assert chain.current_model == "gpt-4o"

        # One more failure at last model
        escalated = chain.record_failure()
        assert not escalated
        assert chain.current_model == "gpt-4o"

    def test_success_resets_failure_count(self):
        chain = self.make_chain()
        chain.record_failure()
        assert chain._models[0].consecutive_failures == 1

        chain.record_success()
        assert chain._models[0].consecutive_failures == 0

    def test_success_tracks_total(self):
        chain = self.make_chain()
        chain.record_success()
        chain.record_success()
        assert chain._models[0].total_successes == 2

    def test_deescalation_after_cooldown(self):
        chain = self.make_chain()
        # Set very short cooldown for testing
        chain._models[0].cooldown_seconds = 0.0

        # Escalate
        chain.record_failure()
        chain.record_failure()
        assert chain.current_model == "local/large-32b"

        # Success should de-escalate since cooldown is 0
        deescalated = chain.record_success()
        assert deescalated
        assert chain.current_model == "local/small-7b"

    def test_no_deescalation_during_cooldown(self):
        chain = self.make_chain()
        # Long cooldown
        chain._models[0].cooldown_seconds = 9999.0

        # Escalate
        chain.record_failure()
        chain.record_failure()
        assert chain.current_model == "local/large-32b"

        # Success should NOT de-escalate (cooldown not expired)
        deescalated = chain.record_success()
        assert not deescalated
        assert chain.current_model == "local/large-32b"

    def test_no_deescalation_when_disabled(self):
        chain = self.make_chain(auto_deescalate=False)
        chain._models[0].cooldown_seconds = 0.0

        chain.record_failure()
        chain.record_failure()
        assert chain.current_model == "local/large-32b"

        deescalated = chain.record_success()
        assert not deescalated
        assert chain.current_model == "local/large-32b"

    def test_events_recorded(self):
        chain = self.make_chain()
        assert len(chain.events) == 0

        chain.record_failure()
        chain.record_failure()
        assert len(chain.events) == 1
        assert chain.events[0].reason == "escalation"
        assert chain.events[0].from_model == "local/small-7b"
        assert chain.events[0].to_model == "local/large-32b"

    def test_deescalation_events(self):
        chain = self.make_chain()
        chain._models[0].cooldown_seconds = 0.0

        chain.record_failure()
        chain.record_failure()
        chain.record_success()

        assert len(chain.events) == 2
        assert chain.events[1].reason == "de-escalation"

    def test_reset_clears_all_state(self):
        chain = self.make_chain()
        chain.record_failure()
        chain.record_failure()
        chain.record_success()

        chain.reset()
        assert chain.current_model == "local/small-7b"
        assert chain.current_index == 0
        assert len(chain.events) == 0
        assert chain._models[0].consecutive_failures == 0
        assert chain._models[0].total_failures == 0
        assert chain._models[0].total_successes == 0

    def test_stats(self):
        chain = self.make_chain()
        chain.record_success()
        chain.record_failure()

        stats = chain.stats()
        assert stats["current_model"] == "local/small-7b"
        assert stats["is_escalated"] is False
        assert stats["models"][0]["total_successes"] == 1
        assert stats["models"][0]["total_failures"] == 1

    def test_single_model_chain(self):
        chain = self.make_chain(
            models=[FallbackModel("local/only-model", max_failures=2)]
        )
        assert chain.current_model == "local/only-model"
        chain.record_failure()
        chain.record_failure()
        # Can't escalate, stays on same model
        assert chain.current_model == "local/only-model"

    def test_models_property_returns_copy(self):
        chain = self.make_chain()
        models = chain.models
        assert len(models) == 3
        models.pop()
        # Original should be unmodified
        assert len(chain.models) == 3


class TestAgentFallbackIntegration:
    """Tests for Agent integration with fallback chain."""

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

    def test_no_fallback_by_default(self, mock_provider):
        agent = Agent(provider=mock_provider)
        assert agent._fallback_chain is None
        assert agent.active_model == ""  # Default empty model

    def test_fallback_chain_created(self, mock_provider):
        config = AgentConfig(
            fallback_models=[
                ("local/small-7b", 3),
                ("gpt-4o", 1),
            ],
        )
        agent = Agent(provider=mock_provider, config=config)
        assert agent._fallback_chain is not None
        assert agent.active_model == "local/small-7b"

    def test_active_model_with_fallback(self, mock_provider):
        config = AgentConfig(
            model="default-model",
            fallback_models=[
                ("local/small-7b", 3),
                ("gpt-4o", 1),
            ],
        )
        agent = Agent(provider=mock_provider, config=config)
        # Fallback chain takes priority over config.model
        assert agent.active_model == "local/small-7b"

    def test_active_model_without_fallback(self, mock_provider):
        config = AgentConfig(model="my-model")
        agent = Agent(provider=mock_provider, config=config)
        assert agent.active_model == "my-model"

    def test_reset_resets_fallback(self, mock_provider):
        config = AgentConfig(
            fallback_models=[
                ("local/small-7b", 1),
                ("gpt-4o", 1),
            ],
        )
        agent = Agent(provider=mock_provider, config=config)
        agent._fallback_chain.record_failure()
        assert agent.active_model == "gpt-4o"

        agent.reset()
        assert agent.active_model == "local/small-7b"

    @pytest.mark.asyncio
    async def test_step_records_success(self, mock_provider):
        config = AgentConfig(
            fallback_models=[("local/test", 3), ("gpt-4o", 1)],
            use_memory=False,
            enable_compaction=False,
        )
        agent = Agent(provider=mock_provider, config=config)
        await agent.step("test")
        assert agent._fallback_chain._models[0].total_successes == 1

    @pytest.mark.asyncio
    async def test_step_escalates_on_failure(self):
        """When generation fails, fallback chain should escalate."""
        call_count = 0

        class FailThenSucceedProvider:
            is_available = True

            async def generate(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RuntimeError("Model unavailable")

                class Result:
                    content = "Hello"
                    tool_calls = None
                return Result()

        config = AgentConfig(
            fallback_models=[("local/small", 2), ("gpt-4o", 1)],
            max_retries=5,
            use_memory=False,
            enable_compaction=False,
        )
        agent = Agent(provider=FailThenSucceedProvider(), config=config)
        turn = await agent.step("test")

        # Should have escalated to gpt-4o after 2 failures
        assert agent.active_model == "gpt-4o"
        assert turn.content == "Hello"
