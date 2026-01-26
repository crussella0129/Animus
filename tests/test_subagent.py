"""Tests for sub-agent orchestration."""

import pytest
from pathlib import Path
import tempfile

from src.core.subagent import (
    SubAgentRole,
    SubAgentScope,
    ScopedToolRegistry,
    ROLE_PROMPTS,
)
from src.tools import create_default_registry


class TestSubAgentScope:
    def test_default_scope(self):
        scope = SubAgentScope()
        assert scope.can_write is False
        assert scope.can_execute is False
        assert scope.max_turns == 10

    def test_scope_with_restrictions(self):
        scope = SubAgentScope(
            allowed_paths=[Path("/tmp")],
            allowed_tools=["read_file"],
            can_write=True,
        )
        assert scope.can_write is True
        assert "read_file" in scope.allowed_tools

    def test_validate_path_no_restrictions(self):
        scope = SubAgentScope()
        assert scope.validate_path(Path("/any/path"))

    def test_validate_path_with_restrictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scope = SubAgentScope(allowed_paths=[Path(tmpdir)])

            # Path within scope
            assert scope.validate_path(Path(tmpdir) / "file.txt")

            # Path outside scope
            assert not scope.validate_path(Path("/other/path"))


class TestSubAgentRole:
    def test_role_prompts_exist(self):
        for role in SubAgentRole:
            if role != SubAgentRole.CUSTOM:
                assert role in ROLE_PROMPTS

    def test_coder_role(self):
        assert SubAgentRole.CODER.value == "coder"
        assert "coding" in ROLE_PROMPTS[SubAgentRole.CODER].lower()

    def test_reviewer_role(self):
        assert SubAgentRole.REVIEWER.value == "reviewer"
        assert "review" in ROLE_PROMPTS[SubAgentRole.REVIEWER].lower()

    def test_tester_role(self):
        assert SubAgentRole.TESTER.value == "tester"
        assert "test" in ROLE_PROMPTS[SubAgentRole.TESTER].lower()


class TestScopedToolRegistry:
    def test_filters_tools(self):
        base_registry = create_default_registry()
        scope = SubAgentScope(allowed_tools=["read_file", "list_dir"])

        scoped = ScopedToolRegistry(base_registry, scope)
        tools = scoped.list_tools()
        tool_names = [t.name for t in tools]

        assert "read_file" in tool_names
        assert "list_dir" in tool_names
        assert "write_file" not in tool_names
        assert "run_shell" not in tool_names

    def test_write_tool_requires_can_write(self):
        base_registry = create_default_registry()

        # Without can_write
        scope1 = SubAgentScope(allowed_tools=["read_file", "write_file"], can_write=False)
        scoped1 = ScopedToolRegistry(base_registry, scope1)
        assert scoped1.get("write_file") is None

        # With can_write
        scope2 = SubAgentScope(allowed_tools=["read_file", "write_file"], can_write=True)
        scoped2 = ScopedToolRegistry(base_registry, scope2)
        assert scoped2.get("write_file") is not None

    def test_shell_tool_requires_can_execute(self):
        base_registry = create_default_registry()

        # Without can_execute
        scope1 = SubAgentScope(allowed_tools=["run_shell"], can_execute=False)
        scoped1 = ScopedToolRegistry(base_registry, scope1)
        assert scoped1.get("run_shell") is None

        # With can_execute
        scope2 = SubAgentScope(allowed_tools=["run_shell"], can_execute=True)
        scoped2 = ScopedToolRegistry(base_registry, scope2)
        assert scoped2.get("run_shell") is not None


class TestSubAgentResult:
    def test_result_structure(self):
        from src.core.subagent import SubAgentResult

        result = SubAgentResult(
            id="test-123",
            role=SubAgentRole.CODER,
            task="Write a function",
            status="completed",
            output="Done!",
            turns=[],
        )

        assert result.id == "test-123"
        assert result.role == SubAgentRole.CODER
        assert result.status == "completed"
        assert result.error is None
        assert result.files_read == []
        assert result.files_written == []
