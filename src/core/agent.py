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

    # Auto-execute these tools without confirmation (read-only operations)
    auto_execute_tools: tuple = ("read_file", "list_dir")

    # Safe shell commands that don't need confirmation (read-only)
    safe_shell_commands: tuple = (
        "ls", "dir", "cat", "type", "pwd", "cd", "echo",
        "git status", "git log", "git diff", "git branch", "git remote",
        "python --version", "python3 --version", "pip list", "pip show",
        "node --version", "npm list", "which", "where", "whoami",
        "date", "time", "hostname", "uname", "env", "printenv",
    )

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

## Available Tools

You have access to these tools:
- read_file: Read file contents
- write_file: Write/create files (requires confirmation)
- list_dir: List directory contents
- run_shell: Execute shell commands

## How to Call Tools

IMPORTANT: To use a tool, output a JSON object in this exact format:
{"tool": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}

Example tool calls:
{"tool": "read_file", "arguments": {"path": "C:/Users/example/file.txt"}}
{"tool": "list_dir", "arguments": {"path": "C:/Users/example", "recursive": false}}
{"tool": "write_file", "arguments": {"path": "C:/Users/example/new.py", "content": "print('hello')"}}
{"tool": "run_shell", "arguments": {"command": "python --version"}}

## Autonomous Execution Policy

You MUST execute tools autonomously. DO NOT ask the user to run commands for you.

**Execute immediately (no confirmation needed):**
- read_file: Reading any file
- list_dir: Listing directories
- run_shell: Safe read-only commands (ls, dir, cat, type, pwd, cd, git status, git log, git diff, python --version, pip list)

**Execute after system confirmation prompt:**
- write_file: Creating or modifying files
- run_shell: Commands that modify state (git commit, git push, pip install, mkdir, etc.)

**Block and refuse:**
- Destructive commands (rm -rf /, format, del /s, etc.)
- Commands that could compromise security

## Workflow

1. When a user asks you to do something, IMMEDIATELY call the appropriate tool
2. After receiving tool results, analyze them and take next steps
3. Continue calling tools until the task is complete
4. Provide a summary of what was accomplished

DO NOT output tool syntax and ask the user to run it. YOU execute the tools.
DO NOT hallucinate or fabricate file contents. ALWAYS use read_file to get actual contents.
DO NOT guess directory structures. ALWAYS use list_dir to explore.

Be concise but thorough.
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

    def _is_safe_shell_command(self, command: str) -> bool:
        """Check if a shell command is safe (read-only) and can be auto-executed."""
        command_lower = command.lower().strip()
        for safe_cmd in self.config.safe_shell_commands:
            # Check if command starts with safe command
            if command_lower.startswith(safe_cmd.lower()):
                return True
        return False

    async def _call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Call a tool with confirmation if needed."""
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        # Determine if confirmation is needed based on tool type and configuration
        needs_confirm = False

        if self.config.require_tool_confirmation:
            # Check if this tool is in the auto-execute list
            if tool_name in self.config.auto_execute_tools:
                needs_confirm = False
            # Check if this is a safe shell command
            elif tool_name == "run_shell" and "command" in arguments:
                if self._is_safe_shell_command(arguments["command"]):
                    needs_confirm = False
                else:
                    needs_confirm = True
            # Otherwise, use the tool's requires_confirmation setting
            elif tool.requires_confirmation:
                needs_confirm = True

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

        Supports multiple formats:
        1. JSON: {"tool": "tool_name", "arguments": {...}}
        2. Function-style: tool_name(arg1, arg2)
        3. Command-style: tool_name "arg1" "arg2"
        """
        tool_calls = []
        import re

        # Pattern 1: JSON format - {"tool": "name", "arguments": {...}}
        # Use a more permissive pattern that can handle nested braces
        json_matches = re.findall(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}', content, re.DOTALL)
        for match in json_matches:
            try:
                data = json.loads(match)
                if "tool" in data and "arguments" in data:
                    tool_calls.append({
                        "name": data["tool"],
                        "arguments": data["arguments"],
                    })
            except json.JSONDecodeError:
                continue

        # If no JSON calls found, try natural language patterns
        if not tool_calls:
            # Pattern 2: Function-style - read_file("path") or read_file('path')
            # Pattern 3: Command-style - read_file "path"
            available_tools = ["read_file", "write_file", "list_dir", "run_shell"]

            for tool_name in available_tools:
                # Function style: tool_name("arg1", "arg2")
                func_pattern = rf'{tool_name}\s*\(\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])?\s*\)'
                for match in re.finditer(func_pattern, content):
                    args = {}
                    if tool_name == "read_file":
                        args["path"] = match.group(1)
                    elif tool_name == "write_file":
                        args["path"] = match.group(1)
                        if match.group(2):
                            args["content"] = match.group(2)
                    elif tool_name == "list_dir":
                        args["path"] = match.group(1)
                    elif tool_name == "run_shell":
                        args["command"] = match.group(1)
                    if args:
                        tool_calls.append({"name": tool_name, "arguments": args})

                # Command style: tool_name "arg1" "arg2"
                cmd_pattern = rf'{tool_name}\s+["\']([^"\']+)["\'](?:\s+["\']([^"\']+)["\'])?'
                for match in re.finditer(cmd_pattern, content):
                    # Skip if we already found this via function pattern
                    path_or_cmd = match.group(1)
                    existing = any(
                        c["name"] == tool_name and
                        (c["arguments"].get("path") == path_or_cmd or c["arguments"].get("command") == path_or_cmd)
                        for c in tool_calls
                    )
                    if existing:
                        continue

                    args = {}
                    if tool_name == "read_file":
                        args["path"] = match.group(1)
                    elif tool_name == "write_file":
                        args["path"] = match.group(1)
                        if match.group(2):
                            args["content"] = match.group(2)
                    elif tool_name == "list_dir":
                        args["path"] = match.group(1)
                    elif tool_name == "run_shell":
                        args["command"] = match.group(1)
                    if args:
                        tool_calls.append({"name": tool_name, "arguments": args})

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
