"""Tests for auth profile rotation."""

import time

import pytest
from src.core.auth_rotation import AuthProfile, AuthRotator, RotationEvent
from src.core.agent import Agent, AgentConfig


class TestAuthProfile:
    """Tests for AuthProfile dataclass."""

    def test_defaults(self):
        p = AuthProfile(name="main", api_key="sk-test")
        assert p.name == "main"
        assert p.api_key == "sk-test"
        assert p.api_base is None
        assert p.cooldown_seconds == 300.0
        assert p.consecutive_failures == 0
        assert p.total_requests == 0
        assert not p.is_on_cooldown

    def test_custom_api_base(self):
        p = AuthProfile(name="alt", api_key="sk-alt", api_base="https://alt.api/v1")
        assert p.api_base == "https://alt.api/v1"

    def test_cooldown_not_active_initially(self):
        p = AuthProfile(name="x", api_key="k")
        assert not p.is_on_cooldown

    def test_cooldown_active_after_failure(self):
        p = AuthProfile(name="x", api_key="k", cooldown_seconds=60.0)
        p.last_failure_time = time.monotonic()
        assert p.is_on_cooldown

    def test_cooldown_expired(self):
        p = AuthProfile(name="x", api_key="k", cooldown_seconds=0.01)
        p.last_failure_time = time.monotonic() - 1.0  # 1s ago, cooldown is 0.01s
        assert not p.is_on_cooldown

    def test_models_filter(self):
        p = AuthProfile(name="x", api_key="k", models=["gpt-4o", "gpt-4o-mini"])
        assert p.models == ["gpt-4o", "gpt-4o-mini"]


class TestAuthRotator:
    """Tests for AuthRotator."""

    def make_rotator(self, n=2, **kwargs):
        profiles = [
            AuthProfile(name=f"profile_{i}", api_key=f"sk-{i}", **kwargs)
            for i in range(n)
        ]
        return AuthRotator(profiles)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            AuthRotator([])

    def test_initial_state(self):
        rotator = self.make_rotator(3)
        assert rotator.current.name == "profile_0"
        assert len(rotator.profiles) == 3
        assert len(rotator.events) == 0

    def test_get_profile_by_name(self):
        rotator = self.make_rotator(2)
        p = rotator.get_profile("profile_1")
        assert p is not None
        assert p.api_key == "sk-1"

    def test_get_profile_unknown(self):
        rotator = self.make_rotator(1)
        assert rotator.get_profile("nonexistent") is None

    def test_failure_rotates(self):
        rotator = self.make_rotator(3, cooldown_seconds=0.01)
        rotated = rotator.record_failure()
        assert rotated
        assert rotator.current.name == "profile_1"
        assert len(rotator.events) == 1
        assert rotator.events[0].reason == "auth_failure"

    def test_double_failure_rotates_twice(self):
        rotator = self.make_rotator(3, cooldown_seconds=0.01)
        rotator.record_failure()  # 0 → 1
        rotator.record_failure()  # 1 → 2
        assert rotator.current.name == "profile_2"

    def test_failure_wraps_around(self):
        rotator = self.make_rotator(3, cooldown_seconds=0.01)
        rotator.record_failure()  # 0 → 1
        rotator.record_failure()  # 1 → 2
        rotator.record_failure()  # 2 → wraps to find 0 (cooldown 0.01s may have expired)
        # Since cooldown is very short, it should wrap to 0
        time.sleep(0.02)
        rotator.record_failure()  # try again after cooldown
        assert rotator.current.name == "profile_0"

    def test_all_on_cooldown(self):
        rotator = self.make_rotator(2, cooldown_seconds=300.0)
        rotated = rotator.record_failure()  # 0 → 1
        assert rotated
        rotated = rotator.record_failure()  # 1 → can't rotate (0 on cooldown)
        assert not rotated
        assert rotator.current.name == "profile_1"  # stuck

    def test_success_resets_failures(self):
        rotator = self.make_rotator(2)
        rotator.current.consecutive_failures = 5
        rotator.record_success()
        assert rotator.current.consecutive_failures == 0
        assert rotator.current.total_successes == 1

    def test_record_request(self):
        rotator = self.make_rotator(1)
        rotator.record_request()
        assert rotator.current.total_requests == 1

    def test_try_recover_preferred_already_preferred(self):
        rotator = self.make_rotator(2)
        assert not rotator.try_recover_preferred()

    def test_try_recover_preferred_cooldown_expired(self):
        rotator = self.make_rotator(2, cooldown_seconds=0.01)
        rotator.record_failure()  # 0 → 1
        assert rotator.current.name == "profile_1"
        time.sleep(0.02)
        recovered = rotator.try_recover_preferred()
        assert recovered
        assert rotator.current.name == "profile_0"
        assert rotator.events[-1].reason == "cooldown_expired"

    def test_try_recover_preferred_still_cooling(self):
        rotator = self.make_rotator(2, cooldown_seconds=300.0)
        rotator.record_failure()  # 0 → 1
        assert not rotator.try_recover_preferred()

    def test_reset_clears_state(self):
        rotator = self.make_rotator(2, cooldown_seconds=0.01)
        rotator.record_failure()
        rotator.record_success()
        rotator.reset()
        assert rotator.current.name == "profile_0"
        assert len(rotator.events) == 0
        for p in rotator.profiles:
            assert p.consecutive_failures == 0
            assert p.total_failures == 0
            assert p.total_requests == 0

    def test_stats(self):
        rotator = self.make_rotator(2, cooldown_seconds=0.01)
        rotator.record_success()
        rotator.record_failure()
        stats = rotator.stats()
        assert stats["current_profile"] == "profile_1"
        assert stats["rotation_count"] == 1
        assert len(stats["profiles"]) == 2
        assert stats["profiles"][0]["total_successes"] == 1
        assert stats["profiles"][0]["total_failures"] == 1

    def test_single_profile_no_rotation(self):
        rotator = self.make_rotator(1)
        rotated = rotator.record_failure()
        assert not rotated  # only one profile, can't rotate


class TestAgentAuthRotatorIntegration:
    """Tests for Agent integration with auth rotator."""

    @pytest.fixture
    def mock_provider(self):
        class MockProvider:
            is_available = True
            api_key = "initial-key"
            _api_key = "initial-key"
            _client = None

            async def generate(self, **kwargs):
                class Result:
                    content = ""
                    tool_calls = None
                return Result()
        return MockProvider()

    def test_no_rotator_by_default(self, mock_provider):
        agent = Agent(provider=mock_provider)
        assert agent._auth_rotator is None

    def test_rotator_created_from_config(self, mock_provider):
        config = AgentConfig(auth_profiles=[
            ("primary", "sk-aaa"),
            ("backup", "sk-bbb"),
        ])
        agent = Agent(provider=mock_provider, config=config)
        assert agent._auth_rotator is not None
        assert agent._auth_rotator.current.name == "primary"
        assert agent._auth_rotator.current.api_key == "sk-aaa"

    def test_rotator_with_api_base(self, mock_provider):
        config = AgentConfig(auth_profiles=[
            ("primary", "sk-aaa", "https://api.example.com/v1"),
        ])
        agent = Agent(provider=mock_provider, config=config)
        assert agent._auth_rotator.current.api_base == "https://api.example.com/v1"

    def test_apply_auth_profile(self, mock_provider):
        config = AgentConfig(auth_profiles=[
            ("primary", "sk-aaa"),
            ("backup", "sk-bbb"),
        ])
        agent = Agent(provider=mock_provider, config=config)
        backup = agent._auth_rotator.get_profile("backup")
        agent._apply_auth_profile(backup)
        assert mock_provider.api_key == "sk-bbb"
        assert mock_provider._api_key == "sk-bbb"
        assert mock_provider._client is None  # Reset

    def test_reset_resets_rotator(self, mock_provider):
        config = AgentConfig(auth_profiles=[
            ("primary", "sk-aaa"),
            ("backup", "sk-bbb"),
        ])
        agent = Agent(provider=mock_provider, config=config)
        agent._auth_rotator.record_failure()
        assert agent._auth_rotator.current.name == "backup"
        agent.reset()
        assert agent._auth_rotator.current.name == "primary"

    @pytest.mark.asyncio
    async def test_step_records_success(self):
        """Successful step should record success on auth rotator."""
        class SuccessProvider:
            is_available = True
            api_key = "k"
            async def generate(self, **kwargs):
                class Result:
                    content = "done"
                    tool_calls = None
                return Result()

        config = AgentConfig(
            auth_profiles=[("main", "sk-1")],
            use_memory=False,
            enable_compaction=False,
        )
        agent = Agent(provider=SuccessProvider(), config=config)
        await agent.step("test")
        assert agent._auth_rotator.current.total_successes == 1

    @pytest.mark.asyncio
    async def test_step_rotates_on_auth_failure(self):
        """Auth failure should trigger rotation to next profile."""
        call_count = 0

        class FailThenSucceedProvider:
            is_available = True
            api_key = "sk-initial"
            _api_key = "sk-initial"
            _client = None

            async def generate(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("401 Unauthorized: invalid api key")
                class Result:
                    content = "ok"
                    tool_calls = None
                return Result()

        config = AgentConfig(
            auth_profiles=[
                ("primary", "sk-aaa"),
                ("backup", "sk-bbb"),
            ],
            use_memory=False,
            enable_compaction=False,
            max_retries=3,
        )
        provider = FailThenSucceedProvider()
        agent = Agent(provider=provider, config=config)
        turn = await agent.step("test")
        assert turn.content == "ok"
        # Provider should have rotated key
        assert provider._api_key == "sk-bbb"
        assert agent._auth_rotator.current.name == "backup"
