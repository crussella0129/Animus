"""Tests for multi-agent delegation system."""

import pytest
import tempfile
from pathlib import Path

from src.core.subagent import (
    SubAgentOrchestrator,
    SubAgentRole,
    SubAgentScope,
    SubAgentResult,
    ScopedToolRegistry,
)
from src.tools.delegate import (
    DelegateTaskTool,
    DelegateParallelTool,
    create_delegation_tools,
)
from src.tools.base import ToolRegistry
from src.core.agent import Agent, AgentConfig


class TestSubAgentRole:
    """Tests for SubAgentRole enum."""

    def test_role_values(self):
        """Test all role values exist."""
        assert SubAgentRole.CODER.value == "coder"
        assert SubAgentRole.REVIEWER.value == "reviewer"
        assert SubAgentRole.TESTER.value == "tester"
        assert SubAgentRole.DOCUMENTER.value == "documenter"
        assert SubAgentRole.REFACTORER.value == "refactorer"
        assert SubAgentRole.DEBUGGER.value == "debugger"
        assert SubAgentRole.RESEARCHER.value == "researcher"
        assert SubAgentRole.CUSTOM.value == "custom"


class TestSubAgentScope:
    """Tests for SubAgentScope."""

    def test_default_scope(self):
        """Test default scope values."""
        scope = SubAgentScope()
        assert scope.allowed_paths == []
        assert "read_file" in scope.allowed_tools
        assert "list_dir" in scope.allowed_tools
        assert scope.max_turns == 10
        assert scope.can_write is False
        assert scope.can_execute is False

    def test_validate_path_no_restrictions(self):
        """Test path validation with no restrictions."""
        scope = SubAgentScope()
        # No restrictions means all paths allowed
        assert scope.validate_path(Path("/any/path")) is True
        assert scope.validate_path(Path("relative/path")) is True

    def test_validate_path_with_restrictions(self):
        """Test path validation with restrictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scope = SubAgentScope(allowed_paths=[Path(tmpdir)])

            # Path within allowed directory
            allowed = Path(tmpdir) / "subdir" / "file.txt"
            assert scope.validate_path(allowed) is True

            # Path outside allowed directory
            outside = Path("/other/path")
            assert scope.validate_path(outside) is False

    def test_custom_scope(self):
        """Test custom scope configuration."""
        scope = SubAgentScope(
            allowed_tools=["read_file", "write_file", "run_shell"],
            max_turns=5,
            can_write=True,
            can_execute=True,
            context="Custom context",
        )

        assert "write_file" in scope.allowed_tools
        assert scope.max_turns == 5
        assert scope.can_write is True
        assert scope.can_execute is True
        assert scope.context == "Custom context"


class TestScopedToolRegistry:
    """Tests for ScopedToolRegistry."""

    def test_scoped_registry_filters_tools(self):
        """Test that scoped registry only includes allowed tools."""
        from src.tools import ReadFileTool, WriteFileTool, ShellTool

        base_registry = ToolRegistry()
        base_registry.register(ReadFileTool())
        base_registry.register(WriteFileTool())
        base_registry.register(ShellTool())

        # Scope with read only
        scope = SubAgentScope(
            allowed_tools=["read_file"],
            can_write=False,
            can_execute=False,
        )

        scoped = ScopedToolRegistry(base_registry, scope)
        tools = scoped.list_tools()
        tool_names = [t.name for t in tools]

        assert "read_file" in tool_names
        assert "write_file" not in tool_names
        assert "run_shell" not in tool_names

    def test_scoped_registry_respects_can_write(self):
        """Test that write tools are included when can_write is True."""
        from src.tools import ReadFileTool, WriteFileTool

        base_registry = ToolRegistry()
        base_registry.register(ReadFileTool())
        base_registry.register(WriteFileTool())

        scope = SubAgentScope(
            allowed_tools=["read_file", "write_file"],
            can_write=True,
        )

        scoped = ScopedToolRegistry(base_registry, scope)
        tool_names = [t.name for t in scoped.list_tools()]

        assert "write_file" in tool_names


class TestSubAgentOrchestrator:
    """Tests for SubAgentOrchestrator."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = "Task completed successfully."
                return Result()
        return MockProvider()

    @pytest.fixture
    def base_registry(self):
        """Create a base tool registry."""
        from src.tools import ReadFileTool, WriteFileTool, ListDirectoryTool

        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(ListDirectoryTool())
        return registry

    def test_orchestrator_init(self, mock_provider, base_registry):
        """Test orchestrator initialization."""
        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=base_registry,
        )

        assert orchestrator.provider == mock_provider
        assert orchestrator.tool_registry == base_registry
        assert orchestrator._active_agents == {}
        assert orchestrator._results == {}

    def test_build_system_prompt(self, mock_provider, base_registry):
        """Test system prompt building."""
        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=base_registry,
        )

        scope = SubAgentScope(
            allowed_tools=["read_file", "list_dir"],
            context="Test context",
            template_vars={"previous": "Previous conversation"},
        )

        prompt = orchestrator._build_system_prompt(
            role=SubAgentRole.CODER,
            scope=scope,
            task="Write a function",
        )

        assert "Write a function" in prompt
        assert "read_file" in prompt
        assert "list_dir" in prompt
        assert "Previous conversation" in prompt

    @pytest.mark.asyncio
    async def test_spawn_subagent(self, mock_provider, base_registry):
        """Test spawning a sub-agent."""
        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=base_registry,
        )

        result = await orchestrator.spawn_subagent(
            task="Analyze the code",
            role=SubAgentRole.RESEARCHER,
        )

        assert isinstance(result, SubAgentResult)
        assert result.role == SubAgentRole.RESEARCHER
        assert result.task == "Analyze the code"
        assert result.status in ("completed", "max_turns_reached", "failed")

    @pytest.mark.asyncio
    async def test_spawn_parallel(self, mock_provider, base_registry):
        """Test spawning multiple sub-agents in parallel."""
        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=base_registry,
        )

        tasks = [
            ("Task 1", SubAgentRole.CODER, None),
            ("Task 2", SubAgentRole.REVIEWER, None),
            ("Task 3", SubAgentRole.TESTER, None),
        ]

        results = await orchestrator.spawn_parallel(tasks)

        assert len(results) == 3
        assert all(isinstance(r, SubAgentResult) for r in results)

    def test_list_results(self, mock_provider, base_registry):
        """Test listing results."""
        orchestrator = SubAgentOrchestrator(
            provider=mock_provider,
            tool_registry=base_registry,
        )

        # Initially empty
        assert orchestrator.list_results() == []


class TestDelegateTaskTool:
    """Tests for DelegateTaskTool."""

    def test_tool_properties(self):
        """Test tool properties."""
        tool = DelegateTaskTool()

        assert tool.name == "delegate_task"
        assert "delegate" in tool.description.lower()
        assert tool.requires_confirmation is True

    def test_tool_parameters(self):
        """Test tool parameters."""
        tool = DelegateTaskTool()
        params = tool.parameters
        param_names = [p.name for p in params]

        assert "task" in param_names
        assert "role" in param_names
        assert "scope_paths" in param_names
        assert "can_write" in param_names
        assert "can_execute" in param_names

    @pytest.mark.asyncio
    async def test_execute_without_orchestrator(self):
        """Test execution fails gracefully without orchestrator."""
        tool = DelegateTaskTool(orchestrator=None)

        result = await tool.execute(task="Test task")

        assert result.success is False
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_orchestrator(self):
        """Test execution with orchestrator."""
        from src.tools import ReadFileTool, ListDirectoryTool

        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = "Analysis complete."
                return Result()

        base_registry = ToolRegistry()
        base_registry.register(ReadFileTool())
        base_registry.register(ListDirectoryTool())

        orchestrator = SubAgentOrchestrator(
            provider=MockProvider(),
            tool_registry=base_registry,
        )

        tool = DelegateTaskTool(orchestrator)

        result = await tool.execute(
            task="Analyze the code",
            role="researcher",
        )

        assert result.success is True
        assert "completed" in result.output.lower()


class TestDelegateParallelTool:
    """Tests for DelegateParallelTool."""

    def test_tool_properties(self):
        """Test tool properties."""
        tool = DelegateParallelTool()

        assert tool.name == "delegate_parallel"
        assert "parallel" in tool.description.lower()
        assert tool.requires_confirmation is True

    @pytest.mark.asyncio
    async def test_execute_without_orchestrator(self):
        """Test execution fails gracefully without orchestrator."""
        tool = DelegateParallelTool(orchestrator=None)

        result = await tool.execute(tasks='[{"task": "Test"}]')

        assert result.success is False
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_json(self):
        """Test execution with invalid JSON."""
        class MockProvider:
            is_available = True

        from src.tools import ReadFileTool

        base_registry = ToolRegistry()
        base_registry.register(ReadFileTool())

        orchestrator = SubAgentOrchestrator(
            provider=MockProvider(),
            tool_registry=base_registry,
        )

        tool = DelegateParallelTool(orchestrator)

        result = await tool.execute(tasks="invalid json")

        assert result.success is False
        assert "invalid" in result.error.lower()


class TestCreateDelegationTools:
    """Tests for create_delegation_tools function."""

    def test_creates_both_tools(self):
        """Test that both delegation tools are created."""
        tools = create_delegation_tools()

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "delegate_task" in tool_names
        assert "delegate_parallel" in tool_names

    def test_tools_share_orchestrator(self):
        """Test that tools receive the same orchestrator."""
        class MockProvider:
            is_available = True

        from src.tools import ReadFileTool

        base_registry = ToolRegistry()
        base_registry.register(ReadFileTool())

        orchestrator = SubAgentOrchestrator(
            provider=MockProvider(),
            tool_registry=base_registry,
        )

        tools = create_delegation_tools(orchestrator)

        assert all(t._orchestrator is orchestrator for t in tools)


class TestAgentDelegationIntegration:
    """Tests for agent delegation integration."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = "Response"
                return Result()
        return MockProvider()

    def test_delegation_disabled_by_default(self):
        """Test that delegation is disabled by default."""
        config = AgentConfig()
        assert config.enable_delegation is False

    def test_delegation_can_be_enabled(self):
        """Test that delegation can be enabled."""
        config = AgentConfig(enable_delegation=True)
        assert config.enable_delegation is True

    def test_agent_is_delegation_enabled(self, mock_provider):
        """Test is_delegation_enabled accessor."""
        config = AgentConfig(enable_delegation=False)
        agent = Agent(provider=mock_provider, config=config)
        assert agent.is_delegation_enabled() is False

        config2 = AgentConfig(enable_delegation=True)
        agent2 = Agent(provider=mock_provider, config=config2)
        assert agent2.is_delegation_enabled() is True

    def test_agent_orchestrator_initialized(self, mock_provider):
        """Test that orchestrator is initialized when delegation is enabled."""
        config = AgentConfig(enable_delegation=True)
        agent = Agent(provider=mock_provider, config=config)

        assert agent._orchestrator is not None

    def test_agent_no_orchestrator_when_disabled(self, mock_provider):
        """Test that orchestrator is not initialized when disabled."""
        config = AgentConfig(enable_delegation=False)
        agent = Agent(provider=mock_provider, config=config)

        assert agent._orchestrator is None

    def test_delegation_tools_registered(self, mock_provider):
        """Test that delegation tools are registered when enabled."""
        config = AgentConfig(enable_delegation=True)
        agent = Agent(provider=mock_provider, config=config)

        assert agent.tool_registry.get("delegate_task") is not None
        assert agent.tool_registry.get("delegate_parallel") is not None

    def test_get_delegation_results_empty(self, mock_provider):
        """Test get_delegation_results with no delegations."""
        config = AgentConfig(enable_delegation=True)
        agent = Agent(provider=mock_provider, config=config)

        assert agent.get_delegation_results() == []

    def test_get_delegation_results_disabled(self, mock_provider):
        """Test get_delegation_results when delegation is disabled."""
        config = AgentConfig(enable_delegation=False)
        agent = Agent(provider=mock_provider, config=config)

        assert agent.get_delegation_results() == []
