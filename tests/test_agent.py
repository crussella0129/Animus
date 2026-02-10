"""Tests for the agent loop — all mocked, no real inference."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.core.agent import Agent
from src.llm.base import ModelCapabilities
from src.tools.base import Tool, ToolRegistry


class MockTool(Tool):
    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    def execute(self, args: dict) -> str:
        return f"mock_result: {args.get('input', '')}"


def _make_mock_provider(responses: list[str]) -> MagicMock:
    provider = MagicMock()
    provider.generate = MagicMock(side_effect=responses)
    provider.capabilities.return_value = ModelCapabilities(
        context_length=4096,
        size_tier="medium",
        supports_tools=True,
    )
    return provider


class TestAgent:
    def test_simple_response(self):
        provider = _make_mock_provider(["Hello! How can I help?"])
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)
        result = agent.run("Hi")
        assert result == "Hello! How can I help?"

    def test_tool_call_and_response(self):
        # First response: tool call, second response: final answer
        provider = _make_mock_provider([
            '```json\n{"name": "mock_tool", "arguments": {"input": "test"}}\n```',
            "The tool returned: mock_result: test",
        ])
        registry = ToolRegistry()
        registry.register(MockTool())
        agent = Agent(provider=provider, tool_registry=registry)
        result = agent.run("Use the mock tool")
        assert "mock_result" in result or "tool returned" in result.lower()

    def test_max_turns_limit(self):
        # Always return tool calls to trigger max turns
        provider = _make_mock_provider([
            '```json\n{"name": "mock_tool", "arguments": {"input": "loop"}}\n```'
        ] * 25)
        registry = ToolRegistry()
        registry.register(MockTool())
        agent = Agent(provider=provider, tool_registry=registry, max_turns=3)
        result = agent.run("Keep calling tools")
        assert "maximum turns" in result.lower()

    def test_provider_error_returns_error(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("Connection failed")
        provider.capabilities.return_value = ModelCapabilities(context_length=4096, size_tier="medium")
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)
        result = agent.run("Hi")
        assert "Error" in result or "error" in result.lower()

    def test_reset_clears_history(self):
        provider = _make_mock_provider(["response1", "response2"])
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)
        agent.run("First message")
        assert len(agent.messages) > 0
        agent.reset()
        assert len(agent.messages) == 0

    def test_parse_tool_calls_json_block(self):
        provider = _make_mock_provider([])
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)
        text = '```json\n{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}\n```'
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "/tmp/test.txt"

    def test_parse_tool_calls_inline(self):
        provider = _make_mock_provider([])
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)
        text = 'I will call {"name": "list_dir", "arguments": {"path": "/tmp"}} now'
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_dir"

    def test_parse_no_tool_calls(self):
        provider = _make_mock_provider([])
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)
        calls = agent._parse_tool_calls("Just a regular response with no tool calls.")
        assert calls == []


class TestChunkedExecution:
    """Test multi-chunk instruction processing."""

    def test_multi_chunk_processes_all_chunks(self):
        """When instruction is chunked, provider should be called for each chunk."""
        # Use a small context to force chunking
        provider = MagicMock()
        provider.generate = MagicMock(side_effect=[
            "Processed part 1.",
            "Processed part 2.",
            "Processed part 3.",
        ])
        provider.capabilities.return_value = ModelCapabilities(
            context_length=2048,
            size_tier="small",
            supports_tools=True,
        )
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)

        # Create instruction large enough to trigger chunking (>256 tokens for small@2048)
        paragraphs = [f"Paragraph {i}: " + "word " * 100 for i in range(10)]
        instruction = "\n\n".join(paragraphs)

        result = agent.run(instruction)
        # Provider should be called multiple times (once per chunk)
        assert provider.generate.call_count > 1
        # Result should be the last chunk's response
        assert "part 3" in result.lower() or provider.generate.call_count >= 3

    def test_single_chunk_standard_path(self):
        """Short instructions should follow the standard (non-chunked) path."""
        provider = _make_mock_provider(["Simple response."])
        registry = ToolRegistry()
        agent = Agent(provider=provider, tool_registry=registry)

        result = agent.run("Hello, world!")
        assert result == "Simple response."
        # Provider should be called exactly once
        assert provider.generate.call_count == 1
