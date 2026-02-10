"""Tests for specialist sub-agent presets."""

from pathlib import Path

import pytest
from src.core.specialists import (
    SpecialistType,
    SpecialistConfig,
    get_specialist,
    spawn_specialist,
    list_specialists,
    _SPECIALIST_CONFIGS,
    EXPLORE_PROMPT,
    PLAN_PROMPT,
    DEBUG_PROMPT,
    TEST_PROMPT,
)
from src.core.subagent import SubAgentRole, SubAgentScope, SubAgentOrchestrator
from src.tools.base import ToolRegistry


class TestSpecialistType:
    """Tests for SpecialistType enum."""

    def test_all_types_exist(self):
        assert SpecialistType.EXPLORE == "explore"
        assert SpecialistType.PLAN == "plan"
        assert SpecialistType.DEBUG == "debug"
        assert SpecialistType.TEST == "test"

    def test_all_types_have_configs(self):
        for stype in SpecialistType:
            assert stype in _SPECIALIST_CONFIGS


class TestSpecialistConfig:
    """Tests for SpecialistConfig dataclass."""

    def test_name_property(self):
        spec = get_specialist(SpecialistType.EXPLORE)
        assert spec.name == "explore"

    def test_config_has_required_fields(self):
        spec = get_specialist(SpecialistType.DEBUG)
        assert spec.type == SpecialistType.DEBUG
        assert spec.role is not None
        assert spec.scope is not None
        assert spec.prompt != ""


class TestGetSpecialist:
    """Tests for get_specialist factory."""

    def test_explore_config(self):
        spec = get_specialist(SpecialistType.EXPLORE)
        assert spec.role == SubAgentRole.RESEARCHER
        assert "read_file" in spec.scope.allowed_tools
        assert "list_dir" in spec.scope.allowed_tools
        assert not spec.scope.can_write
        assert not spec.scope.can_execute

    def test_plan_config(self):
        spec = get_specialist(SpecialistType.PLAN)
        assert spec.role == SubAgentRole.RESEARCHER
        assert not spec.scope.can_write
        assert not spec.scope.can_execute

    def test_debug_config(self):
        spec = get_specialist(SpecialistType.DEBUG)
        assert spec.role == SubAgentRole.DEBUGGER
        assert "run_shell" in spec.scope.allowed_tools
        assert not spec.scope.can_write
        assert spec.scope.can_execute

    def test_test_config(self):
        spec = get_specialist(SpecialistType.TEST)
        assert spec.role == SubAgentRole.TESTER
        assert "write_file" in spec.scope.allowed_tools
        assert "run_shell" in spec.scope.allowed_tools
        assert spec.scope.can_write
        assert spec.scope.can_execute

    def test_allowed_paths(self):
        paths = [Path("/src"), Path("/tests")]
        spec = get_specialist(SpecialistType.EXPLORE, allowed_paths=paths)
        assert spec.scope.allowed_paths == paths

    def test_context_passed(self):
        spec = get_specialist(SpecialistType.PLAN, context="some parent context")
        assert spec.scope.context == "some parent context"
        assert spec.scope.template_vars["previous"] == "some parent context"

    def test_scope_overrides(self):
        spec = get_specialist(SpecialistType.EXPLORE, max_turns=5)
        assert spec.scope.max_turns == 5

    def test_explore_read_only(self):
        spec = get_specialist(SpecialistType.EXPLORE)
        assert "write_file" not in spec.scope.allowed_tools
        assert not spec.scope.can_write

    def test_plan_read_only(self):
        spec = get_specialist(SpecialistType.PLAN)
        assert "write_file" not in spec.scope.allowed_tools
        assert not spec.scope.can_write


class TestSpecialistPrompts:
    """Tests for specialist prompt templates."""

    def test_explore_prompt_has_placeholders(self):
        assert "{task}" in EXPLORE_PROMPT
        assert "{tools}" in EXPLORE_PROMPT
        assert "{scope}" in EXPLORE_PROMPT

    def test_plan_prompt_has_placeholders(self):
        assert "{task}" in PLAN_PROMPT
        assert "{previous}" in PLAN_PROMPT

    def test_debug_prompt_has_placeholders(self):
        assert "{task}" in DEBUG_PROMPT
        assert "{previous}" in DEBUG_PROMPT

    def test_test_prompt_has_placeholders(self):
        assert "{task}" in TEST_PROMPT
        assert "{previous}" in TEST_PROMPT

    def test_explore_prompt_read_only_instructions(self):
        assert "Do NOT modify" in EXPLORE_PROMPT
        assert "Do NOT run shell" in EXPLORE_PROMPT

    def test_plan_prompt_no_modify(self):
        assert "Do NOT modify" in PLAN_PROMPT

    def test_debug_prompt_no_modify(self):
        assert "Do NOT modify" in DEBUG_PROMPT

    def test_test_prompt_allows_writing(self):
        assert "write_file" in TEST_PROMPT


class TestListSpecialists:
    """Tests for list_specialists."""

    def test_lists_all_types(self):
        specs = list_specialists()
        types = {s["type"] for s in specs}
        assert "explore" in types
        assert "plan" in types
        assert "debug" in types
        assert "test" in types

    def test_entry_has_required_fields(self):
        specs = list_specialists()
        for spec in specs:
            assert "type" in spec
            assert "role" in spec
            assert "tools" in spec
            assert "can_write" in spec
            assert "can_execute" in spec
            assert "max_turns" in spec

    def test_explore_is_read_only(self):
        specs = list_specialists()
        explore = next(s for s in specs if s["type"] == "explore")
        assert not explore["can_write"]
        assert not explore["can_execute"]

    def test_test_can_write(self):
        specs = list_specialists()
        test = next(s for s in specs if s["type"] == "test")
        assert test["can_write"]
        assert test["can_execute"]


class TestSpawnSpecialist:
    """Tests for spawn_specialist convenience function."""

    @pytest.fixture
    def mock_provider(self):
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = "analysis complete"
                    tool_calls = None
                return Result()
        return MockProvider()

    @pytest.mark.asyncio
    async def test_spawn_explore(self, mock_provider):
        registry = ToolRegistry()
        # Register minimal tools for the test
        from src.tools import ReadFileTool, ListDirectoryTool
        registry.register(ReadFileTool())
        registry.register(ListDirectoryTool())

        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=registry,
        )

        result = await spawn_specialist(
            orchestrator,
            SpecialistType.EXPLORE,
            task="Find all Python files",
        )

        assert result.status in ("completed", "max_turns_reached")
        assert result.role == SubAgentRole.RESEARCHER

    @pytest.mark.asyncio
    async def test_spawn_with_context(self, mock_provider):
        registry = ToolRegistry()
        from src.tools import ReadFileTool, ListDirectoryTool
        registry.register(ReadFileTool())
        registry.register(ListDirectoryTool())

        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=registry,
        )

        result = await spawn_specialist(
            orchestrator,
            SpecialistType.PLAN,
            task="Plan the refactoring",
            context="We need to split agent.py into smaller modules",
        )

        assert result.status in ("completed", "max_turns_reached")
