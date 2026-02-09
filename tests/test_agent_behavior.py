"""Tests for agent behavior improvements."""

import pytest
import json
from src.core.agent import (
    Agent, AgentConfig,
    SYSTEM_PROMPT_FULL, SYSTEM_PROMPT_COMPACT, SYSTEM_PROMPT_MINIMAL,
    SYSTEM_PROMPT_TIERS,
)
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

    @pytest.mark.asyncio
    async def test_parse_python_string_concat(self, agent):
        """Test parsing JSON with Python-style string concatenation.

        Some LLMs output invalid JSON like:
            "content": "line1\\n"
                       "line2\\n"
        This should be fixed and parsed correctly.
        """
        # This is the malformed JSON that Qwen models sometimes produce
        content = '''{
    "tool": "write_file",
    "arguments": {
        "path": "C:/Users/test/file.py",
        "content": "# Comment\\n"
                   "def main():\\n"
                   "    pass"
    }
}'''
        calls = await agent._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "write_file"
        assert calls[0]["arguments"]["path"] == "C:/Users/test/file.py"
        # The content should be joined
        assert "def main" in calls[0]["arguments"]["content"]

    def test_fix_python_string_concat(self, agent):
        """Test the _fix_python_string_concat helper directly."""
        # Test simple case
        input_str = '"line1"\n"line2"'
        result = agent._fix_python_string_concat(input_str)
        assert result == '"line1line2"'

        # Test with indentation
        input_str = '"line1"\n                    "line2"'
        result = agent._fix_python_string_concat(input_str)
        assert result == '"line1line2"'

        # Test multiple concatenations
        input_str = '"a"\n"b"\n"c"'
        result = agent._fix_python_string_concat(input_str)
        assert result == '"abc"'

        # Test that normal JSON is unchanged
        input_str = '{"key": "value"}'
        result = agent._fix_python_string_concat(input_str)
        assert result == '{"key": "value"}'


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


class TestStreamChunk:
    """Tests for StreamChunk class."""

    def test_from_token(self):
        """Test creating a token chunk."""
        from src.core.agent import StreamChunk
        chunk = StreamChunk.from_token("Hello")
        assert chunk.type == "token"
        assert chunk.token == "Hello"
        assert chunk.turn is None

    def test_from_turn(self):
        """Test creating a turn chunk."""
        from src.core.agent import StreamChunk, Turn
        turn = Turn(role="assistant", content="Hello, world!")
        chunk = StreamChunk.from_turn(turn)
        assert chunk.type == "turn"
        assert chunk.turn == turn
        assert chunk.token is None

    def test_token_chunk_str(self):
        """Test that token chunks hold string tokens."""
        from src.core.agent import StreamChunk
        chunk = StreamChunk.from_token("")
        assert chunk.token == ""
        chunk = StreamChunk.from_token(" ")
        assert chunk.token == " "


class TestAgentStreaming:
    """Tests for agent streaming functionality."""

    @pytest.fixture
    def streaming_provider(self):
        """Create a mock provider that supports streaming."""
        class MockStreamingProvider:
            is_available = True

            async def generate(self, **kwargs):
                class Result:
                    content = "Hello, I am Animus!"
                return Result()

            async def generate_stream(self, **kwargs):
                tokens = ["Hello", ", ", "I ", "am ", "Animus", "!"]
                for token in tokens:
                    yield token

        return MockStreamingProvider()

    @pytest.fixture
    def streaming_agent(self, streaming_provider):
        """Create an agent with streaming provider."""
        from src.core.agent import Agent, AgentConfig
        config = AgentConfig(max_turns=1)
        return Agent(provider=streaming_provider, config=config)

    @pytest.mark.asyncio
    async def test_step_stream_yields_tokens(self, streaming_agent):
        """Test that step_stream yields tokens."""
        from src.core.agent import StreamChunk

        chunks = []
        async for chunk in streaming_agent.step_stream("Hi"):
            chunks.append(chunk)

        # Should have token chunks and a final turn chunk
        token_chunks = [c for c in chunks if c.type == "token"]
        turn_chunks = [c for c in chunks if c.type == "turn"]

        assert len(token_chunks) == 6  # "Hello", ", ", "I ", "am ", "Animus", "!"
        assert len(turn_chunks) == 1
        assert turn_chunks[0].turn.role == "assistant"

    @pytest.mark.asyncio
    async def test_step_stream_content_matches(self, streaming_agent):
        """Test that streamed content matches final turn."""
        from src.core.agent import StreamChunk

        tokens = []
        final_turn = None
        async for chunk in streaming_agent.step_stream("Hi"):
            if chunk.type == "token":
                tokens.append(chunk.token)
            elif chunk.type == "turn":
                final_turn = chunk.turn

        streamed_content = "".join(tokens)
        assert final_turn is not None
        assert final_turn.content == streamed_content

    @pytest.mark.asyncio
    async def test_run_stream_yields_chunks(self, streaming_agent):
        """Test that run_stream yields StreamChunks."""
        from src.core.agent import StreamChunk

        chunks = []
        async for chunk in streaming_agent.run_stream("Hello"):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        # Should have at least one turn
        turn_chunks = [c for c in chunks if c.type == "turn"]
        assert len(turn_chunks) >= 1


class TestParallelToolExecution:
    """Tests for parallel tool execution."""

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
    def parallel_agent(self, mock_provider):
        """Create an agent with parallel execution enabled."""
        config = AgentConfig(parallel_tool_execution=True)
        return Agent(provider=mock_provider, config=config)

    @pytest.fixture
    def sequential_agent(self, mock_provider):
        """Create an agent with parallel execution disabled."""
        config = AgentConfig(parallel_tool_execution=False)
        return Agent(provider=mock_provider, config=config)

    def test_parallel_execution_enabled_by_default(self):
        """Test that parallel execution is enabled by default."""
        config = AgentConfig()
        assert config.parallel_tool_execution is True

    def test_parallel_execution_can_be_disabled(self):
        """Test that parallel execution can be disabled."""
        config = AgentConfig(parallel_tool_execution=False)
        assert config.parallel_tool_execution is False

    @pytest.mark.asyncio
    async def test_parallel_call_tools_empty_list(self, parallel_agent):
        """Test parallel execution with empty tool list."""
        results = await parallel_agent._call_tools_parallel([])
        assert results == []

    @pytest.mark.asyncio
    async def test_parallel_call_tools_single_tool(self, parallel_agent):
        """Test parallel execution with single tool (should still work)."""
        results = await parallel_agent._call_tools_parallel([
            {"name": "read_file", "arguments": {"path": "/nonexistent"}}
        ])
        assert len(results) == 1
        # File doesn't exist, so should fail
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_parallel_call_tools_multiple_tools(self, parallel_agent):
        """Test parallel execution with multiple tools."""
        results = await parallel_agent._call_tools_parallel([
            {"name": "read_file", "arguments": {"path": "/nonexistent1"}},
            {"name": "read_file", "arguments": {"path": "/nonexistent2"}},
        ])
        assert len(results) == 2
        # Both should fail but execute
        assert all(not r.success for r in results)

    @pytest.mark.asyncio
    async def test_sequential_execution_works(self, sequential_agent):
        """Test that sequential execution still works when parallel is disabled."""
        results = await sequential_agent._call_tools_parallel([
            {"name": "read_file", "arguments": {"path": "/nonexistent1"}},
            {"name": "read_file", "arguments": {"path": "/nonexistent2"}},
        ])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self, parallel_agent):
        """Test that parallel execution preserves result order."""
        import tempfile
        import os

        # Create temp files with different content
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(tmpdir, "file2.txt")
            with open(file1, "w") as f:
                f.write("content1")
            with open(file2, "w") as f:
                f.write("content2")

            results = await parallel_agent._call_tools_parallel([
                {"name": "read_file", "arguments": {"path": file1}},
                {"name": "read_file", "arguments": {"path": file2}},
            ])

            assert len(results) == 2
            assert results[0].success is True
            assert "content1" in results[0].output
            assert results[1].success is True
            assert "content2" in results[1].output


class TestPlanningIntegration:
    """Tests for agent planning integration."""

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

    def test_planning_disabled_by_default(self):
        """Test that planning is disabled by default."""
        config = AgentConfig()
        assert config.enable_planning is False

    def test_planning_can_be_enabled(self):
        """Test that planning can be enabled."""
        config = AgentConfig(enable_planning=True)
        assert config.enable_planning is True

    def test_planning_config_options(self):
        """Test planning configuration options."""
        config = AgentConfig(
            enable_planning=True,
            planning_threshold=3,
            auto_revise_plan=False,
        )
        assert config.enable_planning is True
        assert config.planning_threshold == 3
        assert config.auto_revise_plan is False

    def test_agent_is_planning_enabled(self, mock_provider):
        """Test is_planning_enabled accessor."""
        config = AgentConfig(enable_planning=False)
        agent = Agent(provider=mock_provider, config=config)
        assert agent.is_planning_enabled() is False

        config2 = AgentConfig(enable_planning=True)
        agent2 = Agent(provider=mock_provider, config=config2)
        assert agent2.is_planning_enabled() is True

    def test_agent_planner_initialization(self, mock_provider):
        """Test that planner is initialized when enabled."""
        config = AgentConfig(enable_planning=True)
        agent = Agent(provider=mock_provider, config=config)
        assert agent._planner is not None

    def test_agent_no_planner_when_disabled(self, mock_provider):
        """Test that planner is not initialized when disabled."""
        config = AgentConfig(enable_planning=False)
        agent = Agent(provider=mock_provider, config=config)
        assert agent._planner is None

    def test_get_plan_progress_no_plan(self, mock_provider):
        """Test get_plan_progress with no plan."""
        config = AgentConfig(enable_planning=True)
        agent = Agent(provider=mock_provider, config=config)
        # Planner exists but no plan yet
        assert agent.get_plan_progress() == (0, 0)

    def test_get_current_plan_no_plan(self, mock_provider):
        """Test get_current_plan with no plan."""
        config = AgentConfig(enable_planning=True)
        agent = Agent(provider=mock_provider, config=config)
        assert agent.get_current_plan() is None

    @pytest.mark.asyncio
    async def test_create_plan_returns_plan(self, mock_provider):
        """Test create_plan returns an ExecutionPlan."""
        # Mock provider that returns a valid plan JSON
        class PlanningProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = '''```json
{
    "summary": "Test plan",
    "steps": [
        {"description": "Step 1", "reasoning": "First", "dependencies": [], "tool_hints": [], "estimated_complexity": "low"}
    ]
}
```'''
                return Result()

        config = AgentConfig(enable_planning=True)
        agent = Agent(provider=PlanningProvider(), config=config)

        plan = await agent.create_plan("Test request")

        assert plan is not None
        assert plan.goal == "Test request"
        assert len(plan.steps) == 1

    def test_reset_clears_plan(self, mock_provider):
        """Test that reset clears the plan."""
        from src.core.planner import ExecutionPlan, PlanStep

        config = AgentConfig(enable_planning=True)
        agent = Agent(provider=mock_provider, config=config)

        # Set a plan manually
        plan = ExecutionPlan.create(goal="Test")
        plan.add_step(PlanStep.create(description="Test step"))
        agent._planner._current_plan = plan

        assert agent.get_current_plan() is not None

        # Reset should clear it
        agent.reset()
        assert agent.get_current_plan() is None


class TestLooksLikeToolAttempt:
    """Tests for _looks_like_tool_attempt detection."""

    def test_empty_content(self):
        assert Agent._looks_like_tool_attempt("") is False
        assert Agent._looks_like_tool_attempt("short") is False

    def test_no_braces(self):
        assert Agent._looks_like_tool_attempt("I will read the file now.") is False

    def test_valid_tool_call_detected(self):
        content = '{"tool": "read_file", "arguments": {"path": "/tmp"}}'
        assert Agent._looks_like_tool_attempt(content) is True

    def test_malformed_tool_call_detected(self):
        content = 'Sure, I\'ll do that: {"tool": "read_file", "arguments": {path: /tmp}}'
        assert Agent._looks_like_tool_attempt(content) is True

    def test_plain_json_no_tool_keywords(self):
        content = '{"color": "blue", "size": 42}'
        assert Agent._looks_like_tool_attempt(content) is False

    def test_single_keyword_not_enough(self):
        content = 'The "tool" was used: {"data": 1}'
        assert Agent._looks_like_tool_attempt(content) is False

    def test_two_keywords_sufficient(self):
        content = 'I tried to call "tool" with "arguments": {broken json'
        assert Agent._looks_like_tool_attempt(content) is True

    def test_name_arguments_format(self):
        content = '{"name": "write_file", "arguments": {"path": "/x"'  # truncated
        assert Agent._looks_like_tool_attempt(content) is True


class TestParseRetryCorrectLoop:
    """Tests for the parse-retry-correct loop in step()."""

    @pytest.fixture
    def make_agent(self):
        """Factory to create agents with configurable provider responses."""
        def _make(responses, parse_retry_max=3):
            call_count = 0

            class MockProvider:
                is_available = True

                async def generate(self, **kwargs):
                    nonlocal call_count
                    idx = min(call_count, len(responses) - 1)
                    resp = responses[idx]
                    call_count += 1

                    class Result:
                        content = resp.get("content", "")
                        tool_calls = resp.get("tool_calls", None)
                    return Result()

            provider = MockProvider()
            config = AgentConfig(
                parse_retry_max=parse_retry_max,
                use_memory=False,
                enable_compaction=False,
            )
            agent = Agent(provider=provider, config=config)
            return agent, provider

        return _make

    @pytest.mark.asyncio
    async def test_no_retry_when_valid_tool_call(self, make_agent):
        """Valid JSON tool call should not trigger retry."""
        agent, provider = make_agent([
            {"content": '{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'},
        ])
        turn = await agent.step("read /tmp/x")
        assert turn.tool_calls is not None
        assert turn.tool_calls[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_no_retry_for_plain_text(self, make_agent):
        """Plain text response should not trigger retry."""
        agent, provider = make_agent([
            {"content": "Sure, I can help with that. The file contains data."},
        ])
        turn = await agent.step("what's in /tmp?")
        assert turn.tool_calls is None

    @pytest.mark.asyncio
    async def test_retry_on_malformed_tool_call(self, make_agent):
        """Malformed tool call should trigger retry, then succeed."""
        agent, provider = make_agent([
            # First: malformed (missing quotes around keys)
            {"content": '{"tool": "read_file", "arguments": {path: "/tmp/x"}}'},
            # Second: corrected
            {"content": '{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'},
        ])
        turn = await agent.step("read /tmp/x")
        assert turn.tool_calls is not None
        assert turn.tool_calls[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_retry_respects_max(self, make_agent):
        """Should stop retrying after parse_retry_max attempts."""
        malformed = {"content": '{"tool": "read_file", "arguments": {bad json!!!}'}
        agent, provider = make_agent(
            [malformed] * 5,  # All malformed
            parse_retry_max=2,
        )
        turn = await agent.step("read /tmp/x")
        # After 2 retries (3 total attempts), should give up
        assert turn.tool_calls is None

    @pytest.mark.asyncio
    async def test_retry_zero_disables(self, make_agent):
        """parse_retry_max=0 should disable retry entirely."""
        agent, provider = make_agent(
            [
                {"content": '{"tool": "read_file", "arguments": {bad: json}}'},
                {"content": '{"tool": "read_file", "arguments": {"path": "/ok"}}'},
            ],
            parse_retry_max=0,
        )
        turn = await agent.step("read file")
        # No retry, malformed response accepted as-is (no tool calls)
        assert turn.tool_calls is None

    @pytest.mark.asyncio
    async def test_no_retry_for_native_tool_calls(self, make_agent):
        """Provider-level tool_calls bypass the retry loop entirely."""
        agent, provider = make_agent([
            {
                "content": "",
                "tool_calls": [{"name": "read_file", "arguments": {"path": "/x"}}],
            },
        ])
        turn = await agent.step("read /x")
        assert turn.tool_calls is not None
        assert turn.tool_calls[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_retry_correction_message_format(self, make_agent):
        """The correction message should include the malformed output."""
        captured_messages = []

        class CapturingProvider:
            is_available = True

            async def generate(self, **kwargs):
                captured_messages.append(kwargs.get("messages", []))

                class Result:
                    content = '{"tool": "read_file", "arguments": {"path": "/ok"}}'
                    tool_calls = None

                # First call returns malformed, second returns valid
                if len(captured_messages) == 1:
                    class BadResult:
                        content = '{"tool": "read_file", "arguments": {bad}}'
                        tool_calls = None
                    return BadResult()
                return Result()

        config = AgentConfig(
            parse_retry_max=3,
            use_memory=False,
            enable_compaction=False,
        )
        agent = Agent(provider=CapturingProvider(), config=config)
        await agent.step("read a file")

        # Second call should have correction in messages
        assert len(captured_messages) >= 2
        retry_msgs = captured_messages[1]
        # Last message should be the correction prompt
        last_msg = retry_msgs[-1]
        assert "malformed" in last_msg.content.lower()
        assert '{"tool"' in last_msg.content


class TestCapabilityTieredPrompts:
    """Tests for capability-tiered system prompts."""

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

    def test_three_tiers_exist(self):
        """All three tiers should be defined."""
        assert "full" in SYSTEM_PROMPT_TIERS
        assert "compact" in SYSTEM_PROMPT_TIERS
        assert "minimal" in SYSTEM_PROMPT_TIERS

    def test_full_is_longest(self):
        """Full prompt should be the longest, minimal the shortest."""
        assert len(SYSTEM_PROMPT_FULL) > len(SYSTEM_PROMPT_COMPACT)
        assert len(SYSTEM_PROMPT_COMPACT) > len(SYSTEM_PROMPT_MINIMAL)

    def test_all_tiers_mention_tool_format(self):
        """All tiers should include the JSON tool call format."""
        for name, prompt in SYSTEM_PROMPT_TIERS.items():
            assert '"tool"' in prompt, f"{name} tier missing tool call format"

    def test_explicit_tier_full(self, mock_provider):
        config = AgentConfig(prompt_tier="full")
        agent = Agent(provider=mock_provider, config=config)
        assert agent.system_prompt == SYSTEM_PROMPT_FULL

    def test_explicit_tier_compact(self, mock_provider):
        config = AgentConfig(prompt_tier="compact")
        agent = Agent(provider=mock_provider, config=config)
        assert agent.system_prompt == SYSTEM_PROMPT_COMPACT

    def test_explicit_tier_minimal(self, mock_provider):
        config = AgentConfig(prompt_tier="minimal")
        agent = Agent(provider=mock_provider, config=config)
        assert agent.system_prompt == SYSTEM_PROMPT_MINIMAL

    def test_custom_system_prompt_overrides_tier(self, mock_provider):
        """Explicit system_prompt should override tier selection."""
        config = AgentConfig(prompt_tier="minimal", system_prompt="Custom prompt.")
        agent = Agent(provider=mock_provider, config=config)
        assert agent.system_prompt == "Custom prompt."

    def test_auto_tier_gpt4(self, mock_provider):
        config = AgentConfig(model="gpt-4-turbo", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "full"

    def test_auto_tier_claude(self, mock_provider):
        config = AgentConfig(model="claude-3-opus-20240229", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "full"

    def test_auto_tier_70b(self, mock_provider):
        config = AgentConfig(model="llama-3.1-70b-instruct", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "full"

    def test_auto_tier_7b(self, mock_provider):
        config = AgentConfig(model="local/mistral-7b-instruct", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "compact"

    def test_auto_tier_phi(self, mock_provider):
        config = AgentConfig(model="local/phi-3-mini", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "compact"

    def test_auto_tier_local_gguf_unknown(self, mock_provider):
        """Unknown local .gguf model defaults to minimal."""
        config = AgentConfig(model="local/some-model.gguf", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "minimal"

    def test_auto_tier_unknown_api(self, mock_provider):
        """Unknown API model defaults to full."""
        config = AgentConfig(model="some-new-api-model", prompt_tier="auto")
        agent = Agent(provider=mock_provider, config=config)
        assert agent._resolve_prompt_tier() == "full"

    def test_compact_limits_actions(self):
        """Compact prompt should instruct ONE tool call per response."""
        assert "ONE tool call" in SYSTEM_PROMPT_COMPACT

    def test_minimal_is_terse(self):
        """Minimal prompt should be under 200 characters."""
        assert len(SYSTEM_PROMPT_MINIMAL) < 200


class TestProgressiveDisclosureRAG:
    """Tests for progressive disclosure memory retrieval."""

    SAMPLE_RESULTS = [
        ("Full content of first result about Python decorators and how they work "
         "in detail with many examples and explanations spanning multiple lines.",
         0.92, {"source": "docs/decorators.md"}),
        ("Second result about async/await patterns in Python 3.10+ with examples "
         "of coroutines and event loops and task scheduling.",
         0.85, {"source": "docs/async.md"}),
        ("Third result with lower relevance about general Python tips.",
         0.60, {"source": "tips/python.md"}),
    ]

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

    @pytest.fixture
    def progressive_agent(self, mock_provider):
        """Agent with progressive disclosure enabled."""
        config = AgentConfig(
            use_memory=True,
            memory_progressive=True,
            memory_snippet_length=80,
            memory_token_budget=500,
        )
        return Agent(provider=mock_provider, config=config)

    @pytest.fixture
    def legacy_agent(self, mock_provider):
        """Agent with progressive disclosure disabled."""
        config = AgentConfig(
            use_memory=True,
            memory_progressive=False,
        )
        return Agent(provider=mock_provider, config=config)

    def test_progressive_format_is_compact(self, progressive_agent):
        """Progressive format should be shorter than full format."""
        progressive = progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        full = progressive_agent._format_full_context(self.SAMPLE_RESULTS)
        assert len(progressive) < len(full)

    def test_progressive_format_has_indices(self, progressive_agent):
        """Progressive format should include numbered indices."""
        result = progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert "[0]" in result
        assert "[1]" in result
        assert "[2]" in result

    def test_progressive_format_has_scores(self, progressive_agent):
        """Progressive format should include relevance scores."""
        result = progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert "0.92" in result
        assert "0.85" in result

    def test_progressive_format_has_snippets(self, progressive_agent):
        """Progressive format should include truncated snippets."""
        result = progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        # Snippet should be truncated with ...
        assert "..." in result

    def test_progressive_stores_pending_context(self, progressive_agent):
        """Progressive format should store full results for expansion."""
        progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert len(progressive_agent._pending_context) == 3
        assert 0 in progressive_agent._pending_context
        assert 1 in progressive_agent._pending_context
        assert 2 in progressive_agent._pending_context

    def test_expand_context_returns_full(self, progressive_agent):
        """expand_context should return full content for a given index."""
        progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        expanded = progressive_agent.expand_context(0)
        assert expanded is not None
        assert "decorators" in expanded
        assert "docs/decorators.md" in expanded
        assert "0.92" in expanded

    def test_expand_context_invalid_id(self, progressive_agent):
        """expand_context should return None for invalid index."""
        progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert progressive_agent.expand_context(99) is None

    def test_expand_context_empty(self, progressive_agent):
        """expand_context with no pending results should return None."""
        assert progressive_agent.expand_context(0) is None

    def test_legacy_format_includes_full_content(self, legacy_agent):
        """Legacy format should include full content inline."""
        result = legacy_agent._format_full_context(self.SAMPLE_RESULTS)
        assert "decorators and how they work" in result
        assert "async/await patterns" in result

    def test_progressive_cleared_on_new_search(self, progressive_agent):
        """New search should clear previous pending context."""
        progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert len(progressive_agent._pending_context) == 3

        # New search with fewer results
        progressive_agent._format_progressive_context(self.SAMPLE_RESULTS[:1])
        assert len(progressive_agent._pending_context) == 1

    def test_progressive_cleared_on_reset(self, progressive_agent):
        """Reset should clear pending context."""
        progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert len(progressive_agent._pending_context) == 3
        progressive_agent.reset()
        assert len(progressive_agent._pending_context) == 0

    def test_token_budget_limits_results(self, mock_provider):
        """Token budget should limit how many results appear in compact index."""
        config = AgentConfig(
            use_memory=True,
            memory_progressive=True,
            memory_snippet_length=80,
            memory_token_budget=50,  # Very tight budget
        )
        agent = Agent(provider=mock_provider, config=config)

        # Many results
        many_results = [
            (f"Content for result {i} with enough text to fill the snippet area.",
             0.90 - i * 0.05, {"source": f"file_{i}.md"})
            for i in range(10)
        ]

        result = agent._format_progressive_context(many_results)
        # Should have a "more results available" line
        assert "more results available" in result
        # But should still have all 10 stored for expansion
        assert len(agent._pending_context) == 10

    def test_progressive_expand_context_header(self, progressive_agent):
        """Compact index header should mention expand_context."""
        result = progressive_agent._format_progressive_context(self.SAMPLE_RESULTS)
        assert "expand_context" in result

    @pytest.mark.asyncio
    async def test_retrieve_context_progressive_mode(self, mock_provider):
        """_retrieve_context should use progressive format when enabled."""
        class MockMemory:
            async def search(self, query, k=10, filter=None):
                return [
                    ("Content about testing.", 0.88, {"source": "test.md"}),
                ]

        config = AgentConfig(
            use_memory=True,
            memory_progressive=True,
        )
        agent = Agent(provider=mock_provider, config=config)
        agent.memory = MockMemory()

        ctx = await agent._retrieve_context("testing")
        assert ctx is not None
        assert "[0]" in ctx  # Progressive index format
        assert "expand_context" in ctx

    @pytest.mark.asyncio
    async def test_retrieve_context_legacy_mode(self, mock_provider):
        """_retrieve_context should use full format when progressive is off."""
        class MockMemory:
            async def search(self, query, k=10, filter=None):
                return [
                    ("Content about testing.", 0.88, {"source": "test.md"}),
                ]

        config = AgentConfig(
            use_memory=True,
            memory_progressive=False,
        )
        agent = Agent(provider=mock_provider, config=config)
        agent.memory = MockMemory()

        ctx = await agent._retrieve_context("testing")
        assert ctx is not None
        assert "[0]" not in ctx  # Not progressive format
        assert "Content about testing." in ctx  # Full content inline
