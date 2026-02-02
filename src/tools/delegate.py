"""Delegation tool for multi-agent task handling."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from pathlib import Path

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory

if TYPE_CHECKING:
    from src.core.subagent import SubAgentOrchestrator, SubAgentRole, SubAgentScope


class DelegateTaskTool(Tool):
    """Tool to delegate tasks to specialized sub-agents.

    Allows the main agent to spawn sub-agents for specific tasks:
    - CODER: Write new code or implement features
    - REVIEWER: Review code for quality and issues
    - TESTER: Write or run tests
    - DOCUMENTER: Write documentation
    - REFACTORER: Improve existing code structure
    - DEBUGGER: Find and fix bugs
    - RESEARCHER: Analyze code and summarize findings
    """

    def __init__(self, orchestrator: Optional["SubAgentOrchestrator"] = None):
        """Initialize the delegation tool.

        Args:
            orchestrator: SubAgentOrchestrator instance. If None, delegation
                         will fail with an error message.
        """
        self._orchestrator = orchestrator

    def set_orchestrator(self, orchestrator: "SubAgentOrchestrator") -> None:
        """Set the orchestrator (can be set after initialization)."""
        self._orchestrator = orchestrator

    @property
    def name(self) -> str:
        return "delegate_task"

    @property
    def description(self) -> str:
        return (
            "Delegate a task to a specialized sub-agent. "
            "Available roles: coder, reviewer, tester, documenter, refactorer, debugger, researcher. "
            "Sub-agents can read files and have limited scope."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="task",
                type="string",
                description="Description of the task to delegate.",
            ),
            ToolParameter(
                name="role",
                type="string",
                description="Role of sub-agent: coder, reviewer, tester, documenter, refactorer, debugger, researcher.",
                required=False,  # Defaults to coder
            ),
            ToolParameter(
                name="scope_paths",
                type="string",
                description="Comma-separated paths the sub-agent can access.",
                required=False,
            ),
            ToolParameter(
                name="can_write",
                type="boolean",
                description="Whether the sub-agent can write files (default: false).",
                required=False,
            ),
            ToolParameter(
                name="can_execute",
                type="boolean",
                description="Whether the sub-agent can run shell commands (default: false).",
                required=False,
            ),
            ToolParameter(
                name="max_turns",
                type="integer",
                description="Maximum turns for the sub-agent (default: 10).",
                required=False,
            ),
            ToolParameter(
                name="context",
                type="string",
                description="Additional context to provide to the sub-agent.",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.AGENT

    @property
    def requires_confirmation(self) -> bool:
        # Delegation should be confirmed to prevent runaway sub-agents
        return True

    async def execute(
        self,
        task: str,
        role: str = "coder",
        scope_paths: Optional[str] = None,
        can_write: bool = False,
        can_execute: bool = False,
        max_turns: int = 10,
        context: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Delegate a task to a sub-agent."""
        if not self._orchestrator:
            return ToolResult(
                success=False,
                output="",
                error="Sub-agent orchestration not available. Delegation is disabled.",
            )

        # Import here to avoid circular imports
        from src.core.subagent import SubAgentRole, SubAgentScope

        # Parse role
        role_map = {
            "coder": SubAgentRole.CODER,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
            "documenter": SubAgentRole.DOCUMENTER,
            "refactorer": SubAgentRole.REFACTORER,
            "debugger": SubAgentRole.DEBUGGER,
            "researcher": SubAgentRole.RESEARCHER,
        }

        agent_role = role_map.get(role.lower(), SubAgentRole.CODER)

        # Parse scope paths
        allowed_paths = []
        if scope_paths:
            for p in scope_paths.split(","):
                p = p.strip()
                if p:
                    path = Path(p)
                    if path.exists():
                        allowed_paths.append(path)

        # Build allowed tools based on permissions
        allowed_tools = ["read_file", "list_dir"]
        if can_write:
            allowed_tools.append("write_file")
        if can_execute:
            allowed_tools.append("run_shell")

        # Create scope
        scope = SubAgentScope(
            allowed_paths=allowed_paths,
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            can_write=can_write,
            can_execute=can_execute,
            context=context,
            template_vars={"previous": context},
        )

        try:
            result = await self._orchestrator.spawn_subagent(
                task=task,
                role=agent_role,
                scope=scope,
            )

            # Build output
            output_parts = [
                f"Sub-agent ({result.role.value}) completed: {result.status}",
                "",
                "Output:",
                result.output[:2000] if result.output else "(no output)",
            ]

            if result.files_read:
                output_parts.append(f"\nFiles read: {', '.join(result.files_read[:10])}")
            if result.files_written:
                output_parts.append(f"Files written: {', '.join(result.files_written[:10])}")

            return ToolResult(
                success=result.status != "failed",
                output="\n".join(output_parts),
                error=result.error,
                metadata={
                    "agent_id": result.id,
                    "role": result.role.value,
                    "status": result.status,
                    "turn_count": len(result.turns),
                    "files_read": result.files_read,
                    "files_written": result.files_written,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Delegation failed: {e}",
            )


class DelegateParallelTool(Tool):
    """Tool to delegate multiple tasks to sub-agents in parallel."""

    def __init__(self, orchestrator: Optional["SubAgentOrchestrator"] = None):
        """Initialize the parallel delegation tool."""
        self._orchestrator = orchestrator

    def set_orchestrator(self, orchestrator: "SubAgentOrchestrator") -> None:
        """Set the orchestrator."""
        self._orchestrator = orchestrator

    @property
    def name(self) -> str:
        return "delegate_parallel"

    @property
    def description(self) -> str:
        return (
            "Delegate multiple tasks to sub-agents in parallel. "
            "Each task runs in its own sub-agent concurrently."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="tasks",
                type="string",
                description="JSON array of task objects: [{\"task\": \"...\", \"role\": \"coder\"}]",
            ),
            ToolParameter(
                name="scope_paths",
                type="string",
                description="Comma-separated paths all sub-agents can access.",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.AGENT

    @property
    def requires_confirmation(self) -> bool:
        return True

    async def execute(
        self,
        tasks: str,
        scope_paths: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Delegate multiple tasks in parallel."""
        import json

        if not self._orchestrator:
            return ToolResult(
                success=False,
                output="",
                error="Sub-agent orchestration not available.",
            )

        from src.core.subagent import SubAgentRole, SubAgentScope

        # Parse tasks JSON
        try:
            tasks_list = json.loads(tasks)
            if not isinstance(tasks_list, list):
                raise ValueError("Tasks must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid tasks JSON: {e}",
            )

        # Parse scope paths
        allowed_paths = []
        if scope_paths:
            for p in scope_paths.split(","):
                p = p.strip()
                if p:
                    path = Path(p)
                    if path.exists():
                        allowed_paths.append(path)

        # Build task tuples
        role_map = {
            "coder": SubAgentRole.CODER,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
            "documenter": SubAgentRole.DOCUMENTER,
            "refactorer": SubAgentRole.REFACTORER,
            "debugger": SubAgentRole.DEBUGGER,
            "researcher": SubAgentRole.RESEARCHER,
        }

        spawn_tasks = []
        for t in tasks_list:
            task_desc = t.get("task", "")
            role_str = t.get("role", "coder")
            role = role_map.get(role_str.lower(), SubAgentRole.CODER)

            # Per-task scope (merge with global)
            task_paths = allowed_paths.copy()
            if t.get("scope_paths"):
                for p in t["scope_paths"].split(","):
                    p = p.strip()
                    if p:
                        path = Path(p)
                        if path.exists() and path not in task_paths:
                            task_paths.append(path)

            scope = SubAgentScope(
                allowed_paths=task_paths,
                allowed_tools=["read_file", "list_dir"],
                max_turns=t.get("max_turns", 10),
                can_write=t.get("can_write", False),
                can_execute=t.get("can_execute", False),
            )

            spawn_tasks.append((task_desc, role, scope))

        try:
            results = await self._orchestrator.spawn_parallel(spawn_tasks)

            # Build output
            output_parts = [f"Completed {len(results)} parallel tasks:"]
            all_success = True

            for i, result in enumerate(results):
                status_icon = "✓" if result.status != "failed" else "✗"
                output_parts.append(
                    f"\n{status_icon} Task {i+1} ({result.role.value}): {result.status}"
                )
                if result.output:
                    # Truncate long outputs
                    summary = result.output[:500]
                    if len(result.output) > 500:
                        summary += "..."
                    output_parts.append(f"   {summary}")
                if result.error:
                    output_parts.append(f"   Error: {result.error}")
                    all_success = False

            return ToolResult(
                success=all_success,
                output="\n".join(output_parts),
                metadata={
                    "task_count": len(results),
                    "results": [
                        {
                            "id": r.id,
                            "role": r.role.value,
                            "status": r.status,
                        }
                        for r in results
                    ],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Parallel delegation failed: {e}",
            )


def create_delegation_tools(orchestrator: Optional["SubAgentOrchestrator"] = None) -> list[Tool]:
    """Create delegation tools.

    Args:
        orchestrator: Optional SubAgentOrchestrator. If not provided,
                     tools will be created but delegation will fail
                     until an orchestrator is set.

    Returns:
        List of delegation tools.
    """
    return [
        DelegateTaskTool(orchestrator),
        DelegateParallelTool(orchestrator),
    ]
