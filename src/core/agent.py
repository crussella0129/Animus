"""Agent class - The agentic reasoning loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Awaitable
from pathlib import Path

from src.llm.base import ModelProvider, Message, GenerationConfig, GenerationResult
from src.tools.base import Tool, ToolRegistry, ToolResult
from src.tools import create_default_registry
from src.memory import Ingester


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model: str = "qwen2.5-coder:7b"
    max_turns: int = 10
    max_context_messages: int = 20
    temperature: float = 0.7
    system_prompt: Optional[str] = None

    # Tool settings
    require_tool_confirmation: bool = True
    auto_confirm_safe_tools: bool = True

    # Memory settings
    use_memory: bool = True
    memory_search_k: int = 5


@dataclass
class Turn:
    """A single turn in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_results: Optional[list[ToolResult]] = None
    metadata: dict = field(default_factory=dict)


DEFAULT_SYSTEM_PROMPT = """You are Animus, an intelligent coding assistant. You help users with software development tasks by:

1. Understanding their requirements
2. Reading and analyzing code
3. Writing and modifying files
4. Running commands and tests
5. Explaining concepts and solutions

You have access to tools for:
- Reading files (read_file)
- Writing files (write_file)
- Listing directories (list_dir)
- Running shell commands (run_shell)

When using tools:
- Always read files before modifying them
- Use list_dir to understand project structure
- Run tests after making changes
- Explain what you're doing and why

Be concise but thorough. Ask clarifying questions when needed.
"""


class Agent:
    """
    The Animus agent - implements the Think -> Act -> Observe loop.

    The agent:
    1. Receives user input
    2. Retrieves relevant context from memory (if enabled)
    3. Generates a response using the LLM
    4. Executes any tool calls
    5. Incorporates tool results
    6. Continues until done or max turns reached
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[Ingester] = None,
        confirm_callback: Optional[Callable[[str, str], Awaitable[bool]]] = None,
    ):
        """
        Initialize the agent.

        Args:
            provider: LLM provider for generation.
            config: Agent configuration.
            tool_registry: Registry of available tools.
            memory: Memory/RAG system for context retrieval.
            confirm_callback: Callback for tool confirmation.
                             Receives (tool_name, description), returns bool.
        """
        self.provider = provider
        self.config = config or AgentConfig()
        self.tool_registry = tool_registry or create_default_registry()
        self.memory = memory
        self.confirm_callback = confirm_callback

        self.history: list[Turn] = []
        self._current_turn = 0

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

    def _build_messages(self, include_tools: bool = True) -> list[Message]:
        """Build message list for the LLM."""
        messages = [Message(role="system", content=self.system_prompt)]

        # Add conversation history (respecting max context)
        start_idx = max(0, len(self.history) - self.config.max_context_messages)
        for turn in self.history[start_idx:]:
            if turn.role == "tool":
                # Tool results get added as assistant continuation
                content = f"Tool result:\n{turn.content}"
                messages.append(Message(role="assistant", content=content))
            else:
                messages.append(Message(role=turn.role, content=turn.content))

        return messages

    async def _retrieve_context(self, query: str) -> Optional[str]:
        """Retrieve relevant context from memory."""
        if not self.memory or not self.config.use_memory:
            return None

        try:
            results = await self.memory.search(
                query,
                k=self.config.memory_search_k,
            )

            if not results:
                return None

            context_parts = ["Relevant context from knowledge base:"]
            for content, score, metadata in results:
                source = metadata.get("source", "unknown")
                context_parts.append(f"\n[{source}] (relevance: {score:.2f})\n{content[:500]}")

            return "\n".join(context_parts)

        except Exception:
            return None

    async def _call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Call a tool with confirmation if needed."""
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        # Check if confirmation is needed
        needs_confirm = (
            self.config.require_tool_confirmation
            and tool.requires_confirmation
            and not (self.config.auto_confirm_safe_tools and not tool.requires_confirmation)
        )

        if needs_confirm and self.confirm_callback:
            description = f"{tool_name}({json.dumps(arguments, indent=2)})"
            confirmed = await self.confirm_callback(tool_name, description)
            if not confirmed:
                return ToolResult(
                    success=False,
                    output="",
                    error="User declined to execute this tool.",
                )

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution error: {e}",
            )

    async def _parse_tool_calls(self, content: str) -> list[dict]:
        """
        Parse tool calls from assistant response.

        Supports both JSON function call format and natural language patterns.
        """
        tool_calls = []

        # Try to find JSON tool calls in the content
        # Format: {"tool": "tool_name", "arguments": {...}}
        import re
        json_pattern = r'\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\}'

        for match in re.finditer(json_pattern, content, re.DOTALL):
            try:
                data = json.loads(match.group())
                if "tool" in data and "arguments" in data:
                    tool_calls.append({
                        "name": data["tool"],
                        "arguments": data["arguments"],
                    })
            except json.JSONDecodeError:
                continue

        return tool_calls

    async def step(self, user_input: Optional[str] = None) -> Turn:
        """
        Execute one step of the agent loop.

        Args:
            user_input: User message to process. None to continue from last state.

        Returns:
            The turn produced by this step.
        """
        self._current_turn += 1

        # Add user input to history
        if user_input:
            # Retrieve context if memory is enabled
            context = await self._retrieve_context(user_input)
            if context:
                enhanced_input = f"{context}\n\nUser query: {user_input}"
            else:
                enhanced_input = user_input

            self.history.append(Turn(role="user", content=enhanced_input))

        # Build messages for LLM
        messages = self._build_messages()

        # Generate response
        gen_config = GenerationConfig(
            temperature=self.config.temperature,
            max_tokens=4096,
        )

        result = await self.provider.generate(
            messages=messages,
            model=self.config.model,
            config=gen_config,
        )

        # Parse tool calls from response
        tool_calls = await self._parse_tool_calls(result.content)

        # Create assistant turn
        assistant_turn = Turn(
            role="assistant",
            content=result.content,
            tool_calls=tool_calls if tool_calls else None,
        )
        self.history.append(assistant_turn)

        # Execute tool calls if any
        if tool_calls:
            tool_results = []
            for call in tool_calls:
                tool_result = await self._call_tool(call["name"], call["arguments"])
                tool_results.append(tool_result)

            # Add tool results to history
            results_text = "\n\n".join([
                f"Tool: {call['name']}\nResult: {result.output if result.success else result.error}"
                for call, result in zip(tool_calls, tool_results)
            ])
            tool_turn = Turn(
                role="tool",
                content=results_text,
                tool_results=tool_results,
            )
            self.history.append(tool_turn)

            assistant_turn.tool_results = tool_results

        return assistant_turn

    async def run(
        self,
        user_input: str,
        stream: bool = False,
    ) -> AsyncIterator[Turn]:
        """
        Run the agent loop until completion.

        Args:
            user_input: Initial user message.
            stream: Whether to stream responses.

        Yields:
            Turns as they are produced.
        """
        self._current_turn = 0

        # First step with user input
        turn = await self.step(user_input)
        yield turn

        # Continue if there were tool calls
        while turn.tool_calls and self._current_turn < self.config.max_turns:
            turn = await self.step()
            yield turn

            # Break if no more tool calls
            if not turn.tool_calls:
                break

    def reset(self) -> None:
        """Reset the agent state."""
        self.history = []
        self._current_turn = 0

    def get_history(self) -> list[Turn]:
        """Get conversation history."""
        return self.history.copy()
