"""Tests for agent behavior improvements."""

import pytest
import json
from src.core.agent import Agent, AgentConfig
from src.core.config import AgentBehaviorConfig, AnimusConfig


class TestAgentConfig:
    """Tests for AgentConfig with new behavior settings."""

    def test_default_auto_execute_tools(self):
        """Test default auto-execute tools."""
        config = AgentConfig()
        assert "read_file" in config.auto_execute_tools
        assert "list_dir" in config.auto_execute_tools

    def test_default_safe_commands(self):
        """Test default safe shell commands."""
        config = AgentConfig()
        assert "ls" in config.safe_shell_commands
        assert "git status" in config.safe_shell_commands
        assert "pwd" in config.safe_shell_commands

    def test_default_blocked_commands(self):
        """Test default blocked commands."""
        config = AgentConfig()
        assert "rm -rf /" in config.blocked_commands


class TestAgentBehaviorConfig:
    """Tests for AgentBehaviorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentBehaviorConfig()
        assert "read_file" in config.auto_execute_tools
        assert "list_dir" in config.auto_execute_tools
        assert len(config.safe_shell_commands) > 10
        assert len(config.blocked_commands) > 5
        assert config.track_working_directory is True
        assert config.max_autonomous_turns == 10

    def test_require_confirmation_defaults(self):
        """Test require_confirmation default actions."""
        config = AgentBehaviorConfig()
        assert "create_file" in config.require_confirmation
        assert "modify_file" in config.require_confirmation
        assert "delete_file" in config.require_confirmation
        assert "git_push" in config.require_confirmation
        assert "security_warning" in config.require_confirmation

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgentBehaviorConfig(
            auto_execute_tools=["read_file"],
            safe_shell_commands=["ls"],
            max_autonomous_turns=5,
        )
        assert config.auto_execute_tools == ["read_file"]
        assert config.safe_shell_commands == ["ls"]
        assert config.max_autonomous_turns == 5


class TestAnimusConfigAgent:
    """Tests for AnimusConfig with agent behavior."""

    def test_animus_config_includes_agent(self):
        """Test that AnimusConfig includes agent behavior."""
        config = AnimusConfig()
        assert hasattr(config, "agent")
        assert isinstance(config.agent, AgentBehaviorConfig)

    def test_animus_config_agent_defaults(self):
        """Test agent behavior defaults in AnimusConfig."""
        config = AnimusConfig()
        assert "read_file" in config.agent.auto_execute_tools
        assert config.agent.track_working_directory is True


class TestAgentJSONParsing:
    """Tests for improved JSON tool call parsing."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = ""
                return Result()
        return MockProvider()

    @pytest.fixture
    def agent(self, mock_provider):
        """Create an agent for testing."""
        return Agent(provider=mock_provider)

    @pytest.mark.asyncio
    async def test_parse_simple_json(self, agent):
        """Test parsing simple JSON tool calls."""
        content = '{"tool": "read_file", "arguments": {"path": "/test/file.txt"}}'
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "/test/file.txt"

    @pytest.mark.asyncio
    async def test_parse_multiline_json(self, agent):
        """Test parsing multiline JSON tool calls."""
        content = '''{
            "tool": "write_file",
            "arguments": {
                "path": "/test/file.py",
                "content": "print('hello')"
            }
        }'''
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "write_file"
        assert calls[0]["arguments"]["path"] == "/test/file.py"

    @pytest.mark.asyncio
    async def test_parse_json_in_markdown(self, agent):
        """Test parsing JSON in markdown code blocks."""
        content = '''Here's what I'll do:
```json
{"tool": "list_dir", "arguments": {"path": "/home"}}
```
This will list the directory.'''
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_dir"

    @pytest.mark.asyncio
    async def test_parse_multiple_json_calls(self, agent):
        """Test parsing multiple JSON tool calls."""
        content = '''I'll read two files:
{"tool": "read_file", "arguments": {"path": "/file1.txt"}}
And then:
{"tool": "read_file", "arguments": {"path": "/file2.txt"}}'''
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_parse_function_style(self, agent):
        """Test parsing function-style tool calls."""
        content = 'Let me read the file: read_file("/test/file.txt")'
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "/test/file.txt"

    @pytest.mark.asyncio
    async def test_parse_command_style(self, agent):
        """Test parsing command-style tool calls."""
        content = 'Running: run_shell "git status"'
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "run_shell"
        assert calls[0]["arguments"]["command"] == "git status"

    @pytest.mark.asyncio
    async def test_no_duplicate_calls(self, agent):
        """Test that duplicate tool calls are not parsed twice."""
        content = '''read_file("/test.txt")
Now using read_file("/test.txt") again in text.'''
        calls = await agent._parse_tool_calls(content)
        # Should only find one unique call
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_extract_json_with_nested_braces(self, agent):
        """Test extracting JSON with nested content."""
        # Test the _extract_json_objects helper
        content = '{"tool": "write_file", "arguments": {"path": "/t.json", "content": "{\\"key\\": \\"value\\"}"}}'
        objects = agent._extract_json_objects(content)
        assert len(objects) == 1
        assert objects[0]["tool"] == "write_file"


class TestAgentDirectoryTracking:
    """Tests for working directory tracking."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = ""
                return Result()
        return MockProvider()

    @pytest.fixture
    def agent(self, mock_provider):
        """Create an agent for testing."""
        config = AgentConfig(initial_working_directory="/home/user/project")
        return Agent(provider=mock_provider, config=config)

    def test_is_directory_change_cd(self, agent):
        """Test detecting cd commands."""
        is_change, new_dir = agent._is_directory_change("cd /home/user/other")
        assert is_change
        assert new_dir == "/home/user/other"

    def test_is_directory_change_cd_quoted(self, agent):
        """Test detecting cd with quoted path."""
        is_change, new_dir = agent._is_directory_change('cd "/path/with spaces"')
        assert is_change
        assert new_dir == "/path/with spaces"

    def test_is_directory_change_pushd(self, agent):
        """Test detecting pushd commands."""
        is_change, new_dir = agent._is_directory_change("pushd /tmp")
        assert is_change
        assert new_dir == "/tmp"

    def test_is_directory_change_other_commands(self, agent):
        """Test that non-cd commands are not detected."""
        is_change, new_dir = agent._is_directory_change("ls -la")
        assert not is_change
        assert new_dir is None

        is_change, new_dir = agent._is_directory_change("cat file.txt")
        assert not is_change

    def test_is_blocked_command(self, agent):
        """Test blocked command detection."""
        assert agent._is_blocked_command("rm -rf /")
        assert agent._is_blocked_command("sudo rm -rf /home")
        assert agent._is_blocked_command("format c:")
        assert not agent._is_blocked_command("rm file.txt")
        assert not agent._is_blocked_command("ls -la")

    def test_is_safe_shell_command(self, agent):
        """Test safe command detection."""
        assert agent._is_safe_shell_command("ls -la")
        assert agent._is_safe_shell_command("git status")
        assert agent._is_safe_shell_command("python --version")
        assert agent._is_safe_shell_command("pwd")
        assert not agent._is_safe_shell_command("rm file.txt")
        assert not agent._is_safe_shell_command("git push")
