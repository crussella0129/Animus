"""Shell tool with command blocking and confirmation."""

from __future__ import annotations

import subprocess
import time
from typing import Any

from src.core.permission import PermissionChecker
from src.tools.base import Tool, ToolRegistry, isolated


class ExecutionBudget:
    """Track cumulative execution time across all shell commands in a session.

    Prevents runaway execution where many individual commands each stay within
    their per-call timeout but collectively consume excessive time.
    """

    def __init__(self, max_total_seconds: int = 300) -> None:
        """Initialize the execution budget.

        Args:
            max_total_seconds: Maximum cumulative execution time (default: 300s / 5 minutes)
        """
        self._max = max_total_seconds
        self._used = 0.0
        self._call_count = 0

    def consume(self, seconds: float) -> None:
        """Record time consumed by a command execution.

        Args:
            seconds: Time elapsed in seconds
        """
        self._used += seconds
        self._call_count += 1

    def check_available(self, requested_timeout: int) -> tuple[bool, str]:
        """Check if there's enough budget for a command with the requested timeout.

        Args:
            requested_timeout: Timeout being requested for the next command

        Returns:
            Tuple of (is_available, message). If not available, message explains why.
        """
        if self._used >= self._max:
            return False, f"Execution budget exhausted ({self._used:.1f}s / {self._max}s used)"

        remaining = self._max - self._used
        if requested_timeout > remaining:
            return False, (
                f"Requested timeout ({requested_timeout}s) exceeds remaining budget "
                f"({remaining:.1f}s of {self._max}s total)"
            )

        return True, ""

    @property
    def remaining(self) -> float:
        """Get remaining execution time in seconds."""
        return max(0.0, self._max - self._used)

    @property
    def stats(self) -> dict[str, Any]:
        """Get budget statistics."""
        return {
            "total_budget": self._max,
            "used": round(self._used, 2),
            "remaining": round(self.remaining, 2),
            "call_count": self._call_count,
        }


@isolated(level="ornstein")  # Shell commands recommended for isolation
class RunShellTool(Tool):
    """Execute shell commands with safety checks and execution budget tracking."""

    def __init__(self, confirm_callback: Any = None, execution_budget: ExecutionBudget | None = None) -> None:
        super().__init__()  # Initialize Tool base class
        self._confirm = confirm_callback
        self._budget = execution_budget or ExecutionBudget(max_total_seconds=300)

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Dangerous commands require confirmation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["command"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        command = args["command"]
        timeout = args.get("timeout", 30)

        # Check execution budget before running
        budget_ok, budget_msg = self._budget.check_available(timeout)
        if not budget_ok:
            stats = self._budget.stats
            return (
                f"Error: {budget_msg}\n"
                f"Budget stats: {stats['used']}s used in {stats['call_count']} commands, "
                f"{stats['remaining']}s remaining"
            )

        checker = PermissionChecker()
        blocked = checker.is_command_blocked(command)
        if blocked:
            return f"Error: Command blocked for safety: {blocked}"

        if checker.is_command_dangerous(command) and self._confirm:
            if not self._confirm(f"Allow dangerous command: {command}?"):
                return "Command cancelled by user."

        # Track actual execution time
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error executing command: {e}"


def register_shell_tools(
    registry: ToolRegistry,
    confirm_callback: Any = None,
    execution_budget: ExecutionBudget | None = None
) -> None:
    """Register shell tools with the given registry.

    Args:
        registry: The tool registry to register with
        confirm_callback: Optional callback for confirming dangerous commands
        execution_budget: Optional shared execution budget for tracking cumulative time
    """
    registry.register(RunShellTool(
        confirm_callback=confirm_callback,
        execution_budget=execution_budget
    ))
