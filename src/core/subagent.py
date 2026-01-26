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


# Role-specific system prompts
ROLE_PROMPTS = {
    SubAgentRole.CODER: """You are a specialized coding sub-agent. Your task is to write or modify code according to the given requirements.

Focus on:
- Writing clean, well-structured code
- Following existing code patterns and conventions
- Adding appropriate comments for complex logic
- Handling edge cases

You have access to: {tools}
Scope: {scope}
""",

    SubAgentRole.REVIEWER: """You are a code review sub-agent. Your task is to review code and provide feedback.

Focus on:
- Code quality and readability
- Potential bugs or issues
- Performance considerations
- Security concerns
- Adherence to best practices

Provide constructive feedback with specific suggestions.
You have access to: {tools}
""",

    SubAgentRole.TESTER: """You are a testing sub-agent. Your task is to write or run tests.

Focus on:
- Unit tests for individual functions
- Edge cases and error conditions
- Test coverage
- Clear test descriptions

You have access to: {tools}
""",

    SubAgentRole.DOCUMENTER: """You are a documentation sub-agent. Your task is to write or improve documentation.

Focus on:
- Clear and concise explanations
- Code examples where helpful
- Accurate API documentation
- README and guide content

You have access to: {tools}
""",

    SubAgentRole.REFACTORER: """You are a refactoring sub-agent. Your task is to improve code structure without changing behavior.

Focus on:
- Reducing code duplication
- Improving naming and organization
- Breaking down complex functions
- Maintaining existing tests

You have access to: {tools}
""",

    SubAgentRole.DEBUGGER: """You are a debugging sub-agent. Your task is to find and fix bugs.

Focus on:
- Understanding the error or unexpected behavior
- Tracing the issue to its source
- Proposing and implementing fixes
- Verifying the fix works

You have access to: {tools}
""",

    SubAgentRole.RESEARCHER: """You are a research sub-agent. Your task is to gather information and analyze code.

Focus on:
- Understanding code structure and patterns
- Finding relevant code sections
- Analyzing dependencies
- Summarizing findings

You have access to: {tools}
""",
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
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Build the system prompt for a sub-agent."""
        if role == SubAgentRole.CUSTOM and custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS[SubAgentRole.CODER])

        # Format with scope info
        tools_str = ", ".join(scope.allowed_tools)
        scope_str = ""
        if scope.allowed_paths:
            paths = [str(p) for p in scope.allowed_paths]
            scope_str = f"Allowed paths: {', '.join(paths)}"
        if not scope.can_write:
            scope_str += "\nNote: You cannot modify files."
        if not scope.can_execute:
            scope_str += "\nNote: You cannot run shell commands."

        return base_prompt.format(tools=tools_str, scope=scope_str or "No restrictions")

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

        # Build config
        config = AgentConfig(
            max_turns=scope.max_turns,
            system_prompt=self._build_system_prompt(role, scope, custom_prompt),
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

                # Track file operations from tool results
                if turn.tool_results:
                    for result in turn.tool_results:
                        if result.success and result.metadata:
                            path = result.metadata.get("path")
                            if path:
                                if "read" in str(result.metadata):
                                    files_read.append(path)
                                elif "write" in str(result.metadata):
                                    files_written.append(path)

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
