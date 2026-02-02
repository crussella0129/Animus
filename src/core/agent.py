"""Agent class - The agentic reasoning loop."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Awaitable, Union, TYPE_CHECKING
from pathlib import Path

# Constants
MAX_JSON_DEPTH = 50  # Maximum nesting depth for JSON parsing
MAX_TOOL_OUTPUT_LENGTH = 10000  # Truncate tool outputs longer than this

from src.llm.base import ModelProvider, Message, GenerationConfig, GenerationResult
from src.tools.base import Tool, ToolRegistry, ToolResult
from src.tools import create_default_registry

# Lazy import: Ingester is only used for type hints
if TYPE_CHECKING:
    from src.memory import Ingester
from src.core.config import AnimusConfig, AgentBehaviorConfig
from src.core.errors import classify_error, ClassifiedError, ErrorCategory
from src.core.decision import (
    Decision,
    DecisionType,
    Option,
    Outcome,
    OutcomeStatus,
    DecisionRecorder,
)
from src.core.permission import (
    PermissionCategory,
    PermissionAction,
    check_command_permission,
    check_path_permission,
    is_mandatory_deny_command,
    is_mandatory_deny_path,
)
from src.core.compaction import (
    SessionCompactor,
    CompactionConfig,
    CompactionStrategy,
    CompactionResult,
)
from src.core.tokenizer import count_tokens, is_tiktoken_available


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

    # These can be overridden by AnimusConfig.agent settings
    auto_execute_tools: tuple = ("read_file", "list_dir")
    safe_shell_commands: tuple = (
        "ls", "dir", "cat", "type", "pwd", "cd", "echo",
        "git status", "git log", "git diff", "git branch", "git remote",
        "python --version", "python3 --version", "pip list", "pip show",
        "node --version", "npm list", "which", "where", "whoami",
        "date", "time", "hostname", "uname", "env", "printenv",
    )
    blocked_commands: tuple = (
        "rm -rf /", "rm -rf /*", "rm -rf ~",
        "del /s /q c:\\", "format c:",
    )

    # Memory settings
    use_memory: bool = True
    memory_search_k: int = 5

    # Working directory tracking
    track_working_directory: bool = True
    initial_working_directory: Optional[str] = None

    # Retry settings
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_backoff_multiplier: float = 2.0

    # Compaction settings
    enable_compaction: bool = True
    compaction_strategy: str = "hybrid"  # summarize, truncate, sliding, hybrid
    compaction_keep_recent: int = 5
    compaction_trigger_ratio: float = 0.85
    compaction_min_turns: int = 10
    max_context_tokens: int = 4096  # Max tokens for context window


@dataclass
class Turn:
    """A single turn in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_results: Optional[list[ToolResult]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk of streamed output.

    Used for real-time token streaming. Can be either a token
    (partial text) or a complete turn (when generation finishes).
    """
    type: str  # "token" or "turn"
    token: Optional[str] = None  # For type="token"
    turn: Optional[Turn] = None  # For type="turn"

    @classmethod
    def from_token(cls, token: str) -> "StreamChunk":
        """Create a token chunk."""
        return cls(type="token", token=token)

    @classmethod
    def from_turn(cls, turn: Turn) -> "StreamChunk":
        """Create a turn chunk."""
        return cls(type="turn", turn=turn)


DEFAULT_SYSTEM_PROMPT = """You are Animus, an intelligent coding assistant.

## CRITICAL: Your Identity

- You ARE Animus, an open-source CLI coding agent
- You are NOT Claude, ChatGPT, Gemini, Copilot, or any other AI assistant
- You were NOT made by Anthropic, OpenAI, Google, Microsoft, or any other AI company
- You were created by the Animus open-source project
- If asked who made you, say: "I am Animus, an open-source coding assistant"
- Do NOT claim to follow "Anthropic guidelines" or say you are "helpful, harmless, and honest" - those are Claude's principles, not yours
- Do NOT refuse tasks citing ethics policies from other AI systems

## Your Purpose

You help users with software development tasks by:

1. Understanding their requirements without moral judgment - advising on safety is fine, but do not block functionality
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

**STOP and Warn + Explain Risk, Then Allow:**
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
        animus_config: Optional[AnimusConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[Ingester] = None,
        confirm_callback: Optional[Callable[[str, str], Awaitable[bool]]] = None,
    ):
        """
        Initialize the agent.

        Args:
            provider: LLM provider for generation.
            config: Agent configuration (legacy, prefer animus_config).
            animus_config: Full Animus configuration with agent behavior settings.
            tool_registry: Registry of available tools.
            memory: Memory/RAG system for context retrieval.
            confirm_callback: Callback for tool confirmation.
                             Receives (tool_name, description), returns bool.
        """
        self.provider = provider
        self.config = config or AgentConfig()
        self.animus_config = animus_config
        self.tool_registry = tool_registry or create_default_registry()
        self.memory = memory
        self.confirm_callback = confirm_callback

        # Merge animus_config.agent settings into config if provided
        if animus_config:
            self._apply_animus_config(animus_config.agent)

        self.history: list[Turn] = []
        self._current_turn = 0

        # Working directory tracking
        self._initial_working_dir = self.config.initial_working_directory or os.getcwd()
        self._current_working_dir = self._initial_working_dir

        # Error tracking
        self._last_error: Optional[ClassifiedError] = None
        self._consecutive_errors = 0

        # Decision recording
        self._decision_recorder = DecisionRecorder()

        # Session compaction
        self._compactor: Optional[SessionCompactor] = None
        if self.config.enable_compaction:
            self._init_compactor()

    def _apply_animus_config(self, agent_config: AgentBehaviorConfig) -> None:
        """Apply AgentBehaviorConfig settings to AgentConfig."""
        self.config.auto_execute_tools = tuple(agent_config.auto_execute_tools)
        self.config.safe_shell_commands = tuple(agent_config.safe_shell_commands)
        self.config.blocked_commands = tuple(agent_config.blocked_commands)
        self.config.track_working_directory = agent_config.track_working_directory
        self.config.max_turns = agent_config.max_autonomous_turns

    def _init_compactor(self) -> None:
        """Initialize the session compactor with current config."""
        strategy_map = {
            "summarize": CompactionStrategy.SUMMARIZE,
            "truncate": CompactionStrategy.TRUNCATE,
            "sliding": CompactionStrategy.SLIDING_WINDOW,
            "hybrid": CompactionStrategy.HYBRID,
        }
        strategy = strategy_map.get(
            self.config.compaction_strategy,
            CompactionStrategy.HYBRID,
        )

        compaction_config = CompactionConfig(
            strategy=strategy,
            keep_recent_turns=self.config.compaction_keep_recent,
            trigger_ratio=self.config.compaction_trigger_ratio,
            min_turns_to_compact=self.config.compaction_min_turns,
        )

        self._compactor = SessionCompactor(
            config=compaction_config,
            provider=self.provider,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Uses tiktoken for accurate counting, with fallback to
        character-based estimation if tiktoken is unavailable.
        """
        return count_tokens(text)

    def _estimate_total_tokens(self) -> int:
        """Estimate total tokens in conversation history."""
        total = self._estimate_tokens(self.system_prompt)
        for turn in self.history:
            total += self._estimate_tokens(turn.content)
            if turn.tool_calls:
                total += self._estimate_tokens(json.dumps(turn.tool_calls))
        return total

    async def _check_and_compact(self) -> Optional[CompactionResult]:
        """Check if compaction is needed and perform it if so.

        Returns:
            CompactionResult if compaction was performed, None otherwise.
        """
        if not self._compactor or not self.config.enable_compaction:
            return None

        total_tokens = self._estimate_total_tokens()
        max_tokens = self.config.max_context_tokens
        turn_count = len(self.history)

        if not self._compactor.should_compact(total_tokens, max_tokens, turn_count):
            return None

        # Perform compaction
        new_turns, result = await self._compactor.compact_turns(self.history)

        if result.success:
            # Replace history with compacted turns
            self.history = new_turns

        return result

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
        """Check if a shell command is safe using hardcoded permission system.

        Uses centralized permission module for consistent safety checks.
        """
        perm_result = check_command_permission(command)
        return perm_result.action == PermissionAction.ALLOW

    def _is_blocked_command(self, command: str) -> bool:
        """Check if a shell command is blocked using hardcoded permission system.

        Uses centralized permission module - NOT configurable via LLM.
        """
        # Use centralized mandatory deny check (hardcoded, non-overridable)
        return is_mandatory_deny_command(command)

    def _is_directory_change(self, command: str) -> tuple[bool, Optional[str]]:
        """
        Check if command changes working directory.

        Returns:
            Tuple of (is_change, new_directory)
        """
        command = command.strip()

        # Check for cd command
        if command.lower().startswith("cd "):
            new_dir = command[3:].strip().strip('"').strip("'")
            return True, new_dir

        # Check for pushd command (Windows/bash)
        if command.lower().startswith("pushd "):
            new_dir = command[6:].strip().strip('"').strip("'")
            return True, new_dir

        return False, None

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to current working directory."""
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self._current_working_dir, path))

    def _truncate_output(self, text: str, max_length: int = MAX_TOOL_OUTPUT_LENGTH) -> str:
        """Truncate text if it exceeds max_length, adding indicator."""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        return f"{truncated}\n\n... [OUTPUT TRUNCATED - {len(text) - max_length} more characters]"

    def _is_path_change_significant(self, new_dir: str) -> bool:
        """
        Check if directory change is significant (different project/root).

        A significant change is when moving to a different top-level directory
        or outside the initial working directory tree.
        """
        resolved_new = self._resolve_path(new_dir)
        initial_parts = Path(self._initial_working_dir).parts
        new_parts = Path(resolved_new).parts

        # If new path is outside initial directory tree, it's significant
        if len(new_parts) < len(initial_parts):
            return True

        # Check if first N parts match (where N is depth of initial dir)
        for i, part in enumerate(initial_parts):
            if i >= len(new_parts) or new_parts[i] != part:
                return True

        return False

    async def _call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Call a tool with confirmation if needed."""
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            # Record failed tool selection decision
            decision = Decision.create(
                decision_type=DecisionType.TOOL_SELECTION,
                intent=f"Execute tool: {tool_name}",
                context=f"Arguments: {arguments}",
                options=[
                    Option.create(
                        description=f"Use tool '{tool_name}'",
                        cons=["Tool does not exist"],
                    ),
                ],
                chosen_option_id=None,
                reasoning="Tool not found in registry",
                turn_number=self._current_turn,
            )
            self._decision_recorder.record_decision(decision)
            outcome = Outcome.create(
                decision_id=decision.id,
                status=OutcomeStatus.FAILURE,
                result="Tool not found",
                summary=f"Unknown tool: {tool_name}",
                error=f"Unknown tool: {tool_name}",
            )
            self._decision_recorder.record_outcome(outcome)

            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        # Record tool selection decision
        tool_decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent=f"Execute tool: {tool_name}",
            context=f"Arguments: {arguments}",
            options=[
                Option.create(
                    description=f"Use tool '{tool_name}'",
                    pros=["Matches required action"],
                    confidence=0.9,
                ),
            ],
            chosen_option_id=None,  # Will be set after we have the option
            reasoning=f"Selected {tool_name} to accomplish task",
            turn_number=self._current_turn,
        )
        # Set chosen option to the first (and only) option
        if tool_decision.options:
            tool_decision.chosen_option_id = tool_decision.options[0].id
        self._decision_recorder.record_decision(tool_decision)

        # Special handling for shell commands
        if tool_name == "run_shell" and "command" in arguments:
            command = arguments["command"]

            # Check for blocked commands first
            if self._is_blocked_command(command):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"BLOCKED: This command is potentially destructive and has been blocked for safety.",
                )

            # Check for directory changes
            is_dir_change, new_dir = self._is_directory_change(command)
            if is_dir_change and self.config.track_working_directory:
                if new_dir and self._is_path_change_significant(new_dir):
                    # Significant directory change requires confirmation
                    if self.confirm_callback:
                        description = f"Change working directory to: {new_dir}"
                        confirmed = await self.confirm_callback("change_directory", description)
                        if not confirmed:
                            return ToolResult(
                                success=False,
                                output="",
                                error="User declined directory change.",
                            )
                # Update tracked working directory
                if new_dir:
                    self._current_working_dir = self._resolve_path(new_dir)

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
            result = await tool.execute(**arguments)
            self._consecutive_errors = 0  # Reset error count on success

            # Record successful outcome
            outcome = Outcome.create(
                decision_id=tool_decision.id,
                status=OutcomeStatus.SUCCESS if result.success else OutcomeStatus.FAILURE,
                result=result.output[:500] if result.output else "",
                summary=f"Tool {tool_name} {'succeeded' if result.success else 'failed'}",
                error=result.error if not result.success else None,
            )
            self._decision_recorder.record_outcome(outcome)

            return result
        except Exception as e:
            # Classify the error
            classified = classify_error(e)
            self._last_error = classified
            self._consecutive_errors += 1

            # Record failed outcome
            outcome = Outcome.create(
                decision_id=tool_decision.id,
                status=OutcomeStatus.FAILURE,
                result="",
                summary=f"Tool {tool_name} threw exception",
                error=f"{classified.category.value}: {classified.message}",
            )
            self._decision_recorder.record_outcome(outcome)

            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution error ({classified.category.value}): {classified.message}",
                metadata={"error_category": classified.category.value}
            )

    def _fix_python_string_concat(self, content: str) -> str:
        """Fix Python-style adjacent string concatenation in JSON.

        Some LLMs output JSON with Python-style string literals:
            "content": "line1\\n"
                       "line2\\n"

        This is valid Python but invalid JSON. We need to join these
        adjacent string literals into a single string.

        Args:
            content: Raw content that may contain malformed JSON.

        Returns:
            Content with adjacent strings joined.
        """
        import re

        # Pattern: end quote, optional whitespace/newline, start quote
        # This matches: "string1"\s*"string2" -> "string1string2"
        pattern = r'"\s*\n\s*"'

        # Keep replacing until no more matches (handles multiple consecutive strings)
        prev = None
        result = content
        while result != prev:
            prev = result
            result = re.sub(pattern, '', result)

        return result

    def _extract_json_objects(self, content: str, max_depth: int = MAX_JSON_DEPTH) -> list[dict]:
        """
        Extract all valid JSON objects from content using bracket matching.

        This handles nested braces, multiline content, and arrays properly.
        Only extracts top-level objects (not nested ones).

        Args:
            content: The string to search for JSON objects
            max_depth: Maximum nesting depth to prevent DoS from deeply nested input

        Returns:
            List of extracted JSON dictionaries
        """
        # First, try to fix Python-style string concatenation
        content = self._fix_python_string_concat(content)

        objects = []
        found_ranges = []  # Track which character ranges we've already extracted
        i = 0

        while i < len(content):
            # Skip if this position is inside an already-extracted range
            if any(start <= i < end for start, end in found_ranges):
                i += 1
                continue

            # Find start of potential JSON object
            if content[i] == '{':
                # Track brace depth to find matching close
                depth = 0
                start = i
                in_string = False
                escape_next = False

                j = i
                while j < len(content):
                    char = content[j]

                    if escape_next:
                        escape_next = False
                    elif char == '\\' and in_string:
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            depth += 1
                            # Depth limit to prevent DoS
                            if depth > max_depth:
                                break
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                # Found complete object
                                candidate = content[start:j+1]
                                try:
                                    obj = json.loads(candidate)
                                    if isinstance(obj, dict):
                                        objects.append(obj)
                                        # Mark this range as processed to skip nested objects
                                        found_ranges.append((start, j+1))
                                except json.JSONDecodeError:
                                    pass
                                break
                    j += 1
            i += 1

        return objects

    async def _parse_tool_calls(self, content: str) -> list[dict]:
        """
        Parse tool calls from assistant response.

        Supports multiple formats:
        1. JSON: {"tool": "tool_name", "arguments": {...}}
        2. Function-style: tool_name(arg1, arg2)
        3. Command-style: tool_name "arg1" "arg2"
        4. Markdown code blocks with JSON
        """
        tool_calls = []
        import re

        # First, extract JSON from markdown code blocks if present
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        code_blocks = re.findall(code_block_pattern, content)

        # Also check for inline JSON objects
        all_json_sources = code_blocks + [content]

        for source in all_json_sources:
            for obj in self._extract_json_objects(source):
                # Check for tool call format
                if "tool" in obj and "arguments" in obj:
                    tool_call = {
                        "name": obj["tool"],
                        "arguments": obj["arguments"] if isinstance(obj["arguments"], dict) else {},
                    }
                    # Avoid duplicates
                    if tool_call not in tool_calls:
                        tool_calls.append(tool_call)
                # Also check for "name" format (alternative)
                elif "name" in obj and "arguments" in obj:
                    tool_call = {
                        "name": obj["name"],
                        "arguments": obj["arguments"] if isinstance(obj["arguments"], dict) else {},
                    }
                    if tool_call not in tool_calls:
                        tool_calls.append(tool_call)

        # If no JSON calls found, try natural language patterns
        if not tool_calls:
            available_tools = ["read_file", "write_file", "list_dir", "run_shell"]

            def is_duplicate(name: str, args: dict) -> bool:
                """Check if this tool call already exists."""
                for c in tool_calls:
                    if c["name"] != name:
                        continue
                    # Check if arguments match
                    if c["arguments"] == args:
                        return True
                    # Check specific keys for partial match
                    for key in ["path", "command"]:
                        if key in args and key in c["arguments"]:
                            if args[key] == c["arguments"][key]:
                                return True
                return False

            for tool_name in available_tools:
                # Function style: tool_name("arg1", "arg2") or tool_name('arg1')
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
                    if args and not is_duplicate(tool_name, args):
                        tool_calls.append({"name": tool_name, "arguments": args})

                # Command style: tool_name "arg1" "arg2"
                cmd_pattern = rf'{tool_name}\s+["\']([^"\']+)["\'](?:\s+["\']([^"\']+)["\'])?'
                for match in re.finditer(cmd_pattern, content):
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
                    if args and not is_duplicate(tool_name, args):
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

        # Check and perform compaction if needed
        await self._check_and_compact()

        # Build messages for LLM
        messages = self._build_messages()

        # Generate response with retry logic
        gen_config = GenerationConfig(
            temperature=self.config.temperature,
            max_tokens=4096,
        )

        result = None
        last_error = None
        backoff = self.config.retry_backoff_base

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self.provider.generate(
                    messages=messages,
                    model=self.config.model,
                    config=gen_config,
                )
                break  # Success, exit retry loop
            except Exception as e:
                classified = classify_error(e)
                last_error = classified
                self._last_error = classified

                # Only retry if the error strategy says we should
                if not classified.strategy.should_retry or attempt >= self.config.max_retries:
                    raise

                # Wait before retrying with exponential backoff
                await asyncio.sleep(backoff)
                backoff *= self.config.retry_backoff_multiplier

        if result is None:
            # This shouldn't happen, but handle it gracefully
            error_msg = str(last_error.message) if last_error else "Unknown error"
            raise RuntimeError(f"Failed to generate response after retries: {error_msg}")

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

            # Add tool results to history (truncated to prevent context bloat)
            results_parts = []
            for call, res in zip(tool_calls, tool_results):
                output_text = res.output if res.success else res.error
                truncated_output = self._truncate_output(output_text)
                results_parts.append(f"Tool: {call['name']}\nResult: {truncated_output}")

            results_text = "\n\n".join(results_parts)
            tool_turn = Turn(
                role="tool",
                content=results_text,
                tool_results=tool_results,
            )
            self.history.append(tool_turn)

            assistant_turn.tool_results = tool_results

        return assistant_turn

    async def step_stream(
        self,
        user_input: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Execute one step of the agent loop with streaming.

        Yields tokens as they are generated, then yields the final Turn.

        Args:
            user_input: User message to process. None to continue from last state.

        Yields:
            StreamChunk objects - either tokens or the final turn.
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

        # Check and perform compaction if needed
        await self._check_and_compact()

        # Build messages for LLM
        messages = self._build_messages()

        # Generate response with streaming
        gen_config = GenerationConfig(
            temperature=self.config.temperature,
            max_tokens=4096,
            stream=True,
        )

        # Collect streamed content
        full_content = []
        last_error = None
        backoff = self.config.retry_backoff_base

        for attempt in range(self.config.max_retries + 1):
            try:
                async for token in self.provider.generate_stream(
                    messages=messages,
                    model=self.config.model,
                    config=gen_config,
                ):
                    full_content.append(token)
                    yield StreamChunk.from_token(token)
                break  # Success, exit retry loop
            except Exception as e:
                classified = classify_error(e)
                last_error = classified
                self._last_error = classified

                # Only retry if the error strategy says we should
                if not classified.strategy.should_retry or attempt >= self.config.max_retries:
                    raise

                # Reset content for retry
                full_content = []

                # Wait before retrying with exponential backoff
                await asyncio.sleep(backoff)
                backoff *= self.config.retry_backoff_multiplier

        # Assemble final content
        content = "".join(full_content)

        if not content:
            error_msg = str(last_error.message) if last_error else "Unknown error"
            raise RuntimeError(f"Failed to generate response after retries: {error_msg}")

        # Parse tool calls from response
        tool_calls = await self._parse_tool_calls(content)

        # Create assistant turn
        assistant_turn = Turn(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
        self.history.append(assistant_turn)

        # Execute tool calls if any
        if tool_calls:
            tool_results = []
            for call in tool_calls:
                tool_result = await self._call_tool(call["name"], call["arguments"])
                tool_results.append(tool_result)

            # Add tool results to history (truncated to prevent context bloat)
            results_parts = []
            for call, res in zip(tool_calls, tool_results):
                output_text = res.output if res.success else res.error
                truncated_output = self._truncate_output(output_text)
                results_parts.append(f"Tool: {call['name']}\nResult: {truncated_output}")

            results_text = "\n\n".join(results_parts)
            tool_turn = Turn(
                role="tool",
                content=results_text,
                tool_results=tool_results,
            )
            self.history.append(tool_turn)

            assistant_turn.tool_results = tool_results

        # Yield the final turn
        yield StreamChunk.from_turn(assistant_turn)

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

    async def run_stream(
        self,
        user_input: str,
    ) -> AsyncIterator[StreamChunk]:
        """
        Run the agent loop with streaming output.

        Yields tokens as they are generated, interspersed with
        complete Turn objects when generation finishes.

        Args:
            user_input: Initial user message.

        Yields:
            StreamChunk objects - either tokens or turns.
        """
        self._current_turn = 0
        last_turn: Optional[Turn] = None

        # First step with user input (streaming)
        async for chunk in self.step_stream(user_input):
            yield chunk
            if chunk.type == "turn":
                last_turn = chunk.turn

        # Continue if there were tool calls
        while last_turn and last_turn.tool_calls and self._current_turn < self.config.max_turns:
            async for chunk in self.step_stream():
                yield chunk
                if chunk.type == "turn":
                    last_turn = chunk.turn

            # Break if no more tool calls
            if last_turn and not last_turn.tool_calls:
                break

    def reset(self) -> None:
        """Reset the agent state."""
        self.history = []
        self._current_turn = 0
        self._decision_recorder.clear()
        if self._compactor:
            self._compactor.clear_history()

    def get_history(self) -> list[Turn]:
        """Get conversation history."""
        return self.history.copy()

    def get_decisions(self) -> list[Decision]:
        """Get all recorded decisions."""
        return self._decision_recorder.decisions.copy()

    def get_decision_records(self) -> list:
        """Get all decision records with outcomes."""
        return self._decision_recorder.get_records()

    def get_decision_success_rate(self, decision_type: Optional[DecisionType] = None) -> float:
        """Get success rate for decisions.

        Args:
            decision_type: Filter by type, or None for all.

        Returns:
            Success rate as float (0.0 to 1.0).
        """
        return self._decision_recorder.get_success_rate(decision_type)

    def get_compaction_history(self) -> list[CompactionResult]:
        """Get history of compaction operations.

        Returns:
            List of CompactionResult objects.
        """
        if self._compactor:
            return self._compactor.get_compaction_history()
        return []

    def get_estimated_tokens(self) -> int:
        """Get estimated token count for current conversation.

        Returns:
            Estimated token count.
        """
        return self._estimate_total_tokens()

    def is_compaction_enabled(self) -> bool:
        """Check if compaction is enabled.

        Returns:
            True if compaction is enabled.
        """
        return self._compactor is not None and self.config.enable_compaction
