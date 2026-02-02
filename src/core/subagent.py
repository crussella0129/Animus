"""Sub-agent orchestration for complex task delegation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional, Awaitable
from pathlib import Path
import uuid

from src.llm.base import ModelProvider, Message
from src.tools.base import ToolRegistry
from src.core.agent import Agent, AgentConfig, Turn


class SubAgentRole(str, Enum):
    """Predefined sub-agent roles with specialized prompts."""
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DOCUMENTER = "documenter"
    REFACTORER = "refactorer"
    DEBUGGER = "debugger"
    RESEARCHER = "researcher"
    CUSTOM = "custom"


@dataclass
class SubAgentScope:
    """Defines the scope and constraints for a sub-agent."""
    # Files the sub-agent can access
    allowed_paths: list[Path] = field(default_factory=list)

    # Tools the sub-agent can use
    allowed_tools: list[str] = field(default_factory=lambda: ["read_file", "list_dir"])

    # Maximum turns before stopping
    max_turns: int = 10

    # Whether the sub-agent can modify files
    can_write: bool = False

    # Whether the sub-agent can run shell commands
    can_execute: bool = False

    # Context to provide to the sub-agent
    context: str = ""

    # Template variables for prompt customization (100% HARDCODED substitution)
    # {previous} - Previous conversation context
    # {task} - The current task description
    # {scope_dir} - The scope directory path(s)
    template_vars: dict[str, str] = field(default_factory=dict)

    def validate_path(self, path: Path) -> bool:
        """Check if a path is within the allowed scope."""
        if not self.allowed_paths:
            return True  # No restrictions

        path = path.resolve()
        for allowed in self.allowed_paths:
            allowed = allowed.resolve()
            try:
                path.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False


@dataclass
class SubAgentResult:
    """Result from sub-agent execution."""
    id: str
    role: SubAgentRole
    task: str
    status: str  # "completed", "failed", "max_turns_reached"
    output: str
    turns: list[Turn]
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    error: Optional[str] = None


# Common tool calling instructions for all sub-agents
TOOL_CALLING_INSTRUCTIONS = """
## How to Call Tools

IMPORTANT: To use a tool, output a JSON object in this exact format:
{"tool": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}

Examples:
{"tool": "read_file", "arguments": {"path": "/path/to/file.py"}}
{"tool": "list_dir", "arguments": {"path": "/path/to/dir"}}
{"tool": "write_file", "arguments": {"path": "/path/to/file.py", "content": "code here"}}
{"tool": "run_shell", "arguments": {"command": "pytest tests/"}}

Execute tools directly - do NOT ask the user to run commands for you.
Do NOT hallucinate file contents - always use read_file to get actual contents.
"""

# Role-specific system prompts with template variables
# Template vars: {tools}, {scope}, {task}, {scope_dir}, {previous}
ROLE_PROMPTS = {
    SubAgentRole.CODER: """You are a specialized coding sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Writing clean, well-structured code
- Following existing code patterns and conventions
- Adding appropriate comments for complex logic
- Handling edge cases

## Available Tools: {tools}
## Scope: {scope}
## Working Directory: {scope_dir}
""" + TOOL_CALLING_INSTRUCTIONS,

    SubAgentRole.REVIEWER: """You are a code review sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Code quality and readability
- Potential bugs or issues
- Performance considerations
- Operational Security Concerns
- Adherence to best practices

Provide constructive feedback with specific suggestions.

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS,

    SubAgentRole.TESTER: """You are a testing sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Unit tests for individual functions
- Edge cases and error conditions
- Test coverage
- Clear test descriptions

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS,

    SubAgentRole.DOCUMENTER: """You are a documentation sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Clear and concise explanations
- Code examples where helpful
- Accurate API documentation
- README and guide content

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS,

    SubAgentRole.REFACTORER: """You are a refactoring sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Reducing code duplication
- Improving naming and organization
- Breaking down complex functions
- Maintaining existing tests

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS,

    SubAgentRole.DEBUGGER: """You are a debugging sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Understanding the error or unexpected behavior
- Tracing the issue to its source
- Proposing and implementing fixes
- Verifying the fix works

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS,

    SubAgentRole.RESEARCHER: """You are a research sub-agent.

## Current Task
{task}

## Context from Parent Agent
{previous}

## Focus Areas
- Understanding code structure and patterns
- Finding relevant code sections
- Analyzing dependencies
- Summarizing findings

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS,
}


class ScopedToolRegistry(ToolRegistry):
    """A tool registry that enforces scope restrictions."""

    def __init__(self, base_registry: ToolRegistry, scope: SubAgentScope):
        super().__init__()
        self.scope = scope

        # Copy only allowed tools
        for tool in base_registry.list_tools():
            if tool.name in scope.allowed_tools:
                # Add write tools only if can_write
                if tool.name == "write_file" and not scope.can_write:
                    continue
                # Add shell tool only if can_execute
                if tool.name == "run_shell" and not scope.can_execute:
                    continue
                self.register(tool)


class SubAgentOrchestrator:
    """
    Orchestrates sub-agent spawning and execution.

    The main agent can delegate tasks to specialized sub-agents,
    each with restricted scope and capabilities.
    """

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        confirm_callback: Optional[Callable[[str, str], Awaitable[bool]]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            provider: LLM provider for sub-agents.
            tool_registry: Base tool registry to scope from.
            confirm_callback: Callback for confirmation prompts.
        """
        self.provider = provider
        self.tool_registry = tool_registry
        self.confirm_callback = confirm_callback
        self._active_agents: dict[str, Agent] = {}
        self._results: dict[str, SubAgentResult] = {}

    def _build_system_prompt(
        self,
        role: SubAgentRole,
        scope: SubAgentScope,
        task: str = "",
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Build the system prompt for a sub-agent.

        Uses 100% HARDCODED template substitution - no LLM inference.

        Template Variables:
            {tools} - Comma-separated list of allowed tools
            {scope} - Scope restrictions description
            {previous} - Previous context from parent (from scope.template_vars)
            {task} - The current task description
            {scope_dir} - The scope directory path(s)
        """
        if role == SubAgentRole.CUSTOM and custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS[SubAgentRole.CODER])

        # Build tools string (HARDCODED)
        tools_str = ", ".join(scope.allowed_tools)

        # Build scope string (HARDCODED)
        scope_str = ""
        if scope.allowed_paths:
            paths = [str(p) for p in scope.allowed_paths]
            scope_str = f"Allowed paths: {', '.join(paths)}"
        if not scope.can_write:
            scope_str += "\nNote: You cannot modify files."
        if not scope.can_execute:
            scope_str += "\nNote: You cannot run shell commands."

        # Build scope_dir string (HARDCODED)
        scope_dir = ""
        if scope.allowed_paths:
            scope_dir = ", ".join(str(p) for p in scope.allowed_paths)

        # Build format dict with all template variables (HARDCODED substitution)
        format_dict = {
            "tools": tools_str,
            "scope": scope_str or "No restrictions",
            "task": task,
            "scope_dir": scope_dir or ".",
            "previous": scope.template_vars.get("previous", ""),
            # Include any additional template vars from scope
            **scope.template_vars,
        }

        # Safe format that ignores missing keys (HARDCODED)
        try:
            return base_prompt.format(**format_dict)
        except KeyError:
            # Fall back to partial formatting if some keys are missing
            import re
            result = base_prompt
            for key, value in format_dict.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result

    async def spawn_subagent(
        self,
        task: str,
        role: SubAgentRole = SubAgentRole.CODER,
        scope: Optional[SubAgentScope] = None,
        custom_prompt: Optional[str] = None,
    ) -> SubAgentResult:
        """
        Spawn a sub-agent to handle a specific task.

        Args:
            task: Task description for the sub-agent.
            role: Role of the sub-agent (determines system prompt).
            scope: Scope restrictions for the sub-agent.
            custom_prompt: Custom system prompt (for CUSTOM role).

        Returns:
            SubAgentResult with the outcome.
        """
        agent_id = str(uuid.uuid4())[:8]
        scope = scope or SubAgentScope()

        # Create scoped tool registry
        scoped_registry = ScopedToolRegistry(self.tool_registry, scope)

        # Build config with template variables (100% HARDCODED substitution)
        config = AgentConfig(
            max_turns=scope.max_turns,
            system_prompt=self._build_system_prompt(role, scope, task, custom_prompt),
            require_tool_confirmation=False,  # Sub-agents run with pre-approved scope
        )

        # Create sub-agent
        agent = Agent(
            provider=self.provider,
            config=config,
            tool_registry=scoped_registry,
        )

        self._active_agents[agent_id] = agent

        # Track file operations
        files_read: list[str] = []
        files_written: list[str] = []

        # Run the sub-agent
        turns: list[Turn] = []
        final_output = ""
        status = "completed"
        error = None

        try:
            async for turn in agent.run(task):
                turns.append(turn)

                if turn.role == "assistant":
                    final_output = turn.content

                # Track file operations from tool calls and results
                if turn.tool_calls and turn.tool_results:
                    for call, result in zip(turn.tool_calls, turn.tool_results):
                        if not result.success:
                            continue

                        tool_name = call.get("name", "")
                        args = call.get("arguments", {})
                        path = args.get("path")

                        if path:
                            # Track based on tool name, not metadata string matching
                            if tool_name == "read_file":
                                files_read.append(path)
                            elif tool_name == "write_file":
                                files_written.append(path)
                            elif tool_name == "list_dir":
                                files_read.append(path)  # Directory listing is a read operation

            if len(turns) >= scope.max_turns:
                status = "max_turns_reached"

        except Exception as e:
            status = "failed"
            error = str(e)
            final_output = f"Sub-agent failed: {e}"

        finally:
            # Cleanup
            del self._active_agents[agent_id]

        result = SubAgentResult(
            id=agent_id,
            role=role,
            task=task,
            status=status,
            output=final_output,
            turns=turns,
            files_read=list(set(files_read)),
            files_written=list(set(files_written)),
            error=error,
        )

        self._results[agent_id] = result
        return result

    async def spawn_parallel(
        self,
        tasks: list[tuple[str, SubAgentRole, Optional[SubAgentScope]]],
    ) -> list[SubAgentResult]:
        """
        Spawn multiple sub-agents in parallel.

        Args:
            tasks: List of (task, role, scope) tuples.

        Returns:
            List of SubAgentResults.
        """
        coroutines = [
            self.spawn_subagent(task, role, scope)
            for task, role, scope in tasks
        ]

        return await asyncio.gather(*coroutines)

    def get_result(self, agent_id: str) -> Optional[SubAgentResult]:
        """Get the result of a completed sub-agent."""
        return self._results.get(agent_id)

    def list_results(self) -> list[SubAgentResult]:
        """List all sub-agent results."""
        return list(self._results.values())
