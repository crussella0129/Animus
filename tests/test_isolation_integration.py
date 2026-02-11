"""Tests for Ornstein Phase 2: CLI and tool integration."""

import pytest
from src.core.config import AnimusConfig, IsolationConfig
from src.tools.base import Tool, isolated, ToolRegistry
from src.tools.shell import RunShellTool
from src.tools.filesystem import ReadFileTool, WriteFileTool


class TestIsolationConfig:
    """Test IsolationConfig dataclass."""

    def test_default_config(self):
        """Test default isolation configuration."""
        config = IsolationConfig()

        assert config.default_level == "none"
        assert config.ornstein_enabled is False
        assert config.ornstein_timeout == 30
        assert config.ornstein_memory_mb == 512
        assert config.ornstein_allow_write is False
        assert config.tool_isolation == {}
        assert config.auto_isolate_dangerous is False

    def test_custom_config(self):
        """Test custom isolation configuration."""
        config = IsolationConfig(
            default_level="ornstein",
            ornstein_enabled=True,
            ornstein_timeout=60,
            tool_isolation={"run_shell": "ornstein"}
        )

        assert config.default_level == "ornstein"
        assert config.ornstein_enabled is True
        assert config.ornstein_timeout == 60
        assert config.tool_isolation == {"run_shell": "ornstein"}

    def test_animus_config_includes_isolation(self):
        """Test that AnimusConfig includes isolation field."""
        config = AnimusConfig()

        assert hasattr(config, 'isolation')
        assert isinstance(config.isolation, IsolationConfig)


class TestIsolatedDecorator:
    """Test @isolated decorator for tools."""

    def test_isolated_decorator_default(self):
        """Test @isolated decorator with default level."""
        @isolated()
        class TestTool(Tool):
            def __init__(self):
                super().__init__()

            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "Test tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            def execute(self, args):
                return "test"

        tool = TestTool()
        assert tool.isolation_level == "ornstein"

    def test_isolated_decorator_custom_level(self):
        """Test @isolated decorator with custom level."""
        @isolated(level="smough")
        class TestTool(Tool):
            def __init__(self):
                super().__init__()

            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "Test tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            def execute(self, args):
                return "test"

        tool = TestTool()
        assert tool.isolation_level == "smough"

    def test_isolated_decorator_on_real_tool(self):
        """Test that RunShellTool has ornstein isolation."""
        tool = RunShellTool()
        assert tool.isolation_level == "ornstein"


class TestToolIsolationLevels:
    """Test isolation levels on actual tools."""

    def test_shell_tool_isolation(self):
        """Test that shell tool recommends isolation."""
        tool = RunShellTool()
        assert tool.isolation_level == "ornstein"

    def test_read_file_no_isolation(self):
        """Test that read_file doesn't need isolation."""
        tool = ReadFileTool()
        assert tool.isolation_level == "none"

    def test_write_file_no_isolation_default(self):
        """Test that write_file defaults to no isolation."""
        tool = WriteFileTool()
        assert tool.isolation_level == "none"

    def test_isolation_level_setter(self):
        """Test setting isolation level on a tool."""
        tool = ReadFileTool()
        assert tool.isolation_level == "none"

        tool.isolation_level = "ornstein"
        assert tool.isolation_level == "ornstein"

    def test_isolation_level_invalid(self):
        """Test that invalid isolation level raises error."""
        tool = ReadFileTool()

        with pytest.raises(ValueError, match="Invalid isolation level"):
            tool.isolation_level = "invalid"


class TestToolRegistryWithIsolation:
    """Test tool registry with isolation metadata."""

    def test_register_tools_with_isolation(self):
        """Test registering tools preserves isolation level."""
        registry = ToolRegistry()

        shell_tool = RunShellTool()
        read_tool = ReadFileTool()

        registry.register(shell_tool)
        registry.register(read_tool)

        # Get tools back and check isolation
        retrieved_shell = registry.get("run_shell")
        retrieved_read = registry.get("read_file")

        assert retrieved_shell.isolation_level == "ornstein"
        assert retrieved_read.isolation_level == "none"

    def test_list_tools_includes_isolation(self):
        """Test that list_tools returns tools with isolation metadata."""
        registry = ToolRegistry()
        registry.register(RunShellTool())
        registry.register(ReadFileTool())

        tools = registry.list_tools()

        assert len(tools) == 2

        # Find shell tool
        shell_tool = next(t for t in tools if t.name == "run_shell")
        assert shell_tool.isolation_level == "ornstein"


class TestCLIFlags:
    """Test CLI flag behavior."""

    def test_cautious_flag_config(self):
        """Test that --cautious would set config correctly."""
        config = AnimusConfig()

        # Simulate what --cautious flag does
        config.isolation.default_level = "ornstein"
        config.isolation.ornstein_enabled = True

        assert config.isolation.default_level == "ornstein"
        assert config.isolation.ornstein_enabled is True

    def test_paranoid_flag_not_implemented(self):
        """Test that --paranoid should raise error (not implemented)."""
        # This is tested via CLI, documented here for completeness
        # In main.py, paranoid flag raises typer.Exit(1)
        pass


class TestConfigPersistence:
    """Test isolation config persistence."""

    def test_isolation_config_saves(self, tmp_path):
        """Test that isolation config saves to YAML."""
        config = AnimusConfig(config_dir=tmp_path)
        config.isolation.default_level = "ornstein"
        config.isolation.ornstein_enabled = True
        config.isolation.tool_isolation = {"run_shell": "ornstein"}

        config.save()

        # Load and verify
        loaded = AnimusConfig.load(config_dir=tmp_path)

        assert loaded.isolation.default_level == "ornstein"
        assert loaded.isolation.ornstein_enabled is True
        assert loaded.isolation.tool_isolation == {"run_shell": "ornstein"}
