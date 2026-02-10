"""Shell tool with command blocking and confirmation."""

from __future__ import annotations

import subprocess
from typing import Any

from src.core.permission import PermissionChecker
from src.tools.base import Tool, ToolRegistry


class RunShellTool(Tool):
    """Execute shell commands with safety checks."""

    def __init__(self, confirm_callback: Any = None) -> None:
        self._confirm = confirm_callback

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

        checker = PermissionChecker()
        blocked = checker.is_command_blocked(command)
        if blocked:
            return f"Error: Command blocked for safety: {blocked}"

        if checker.is_command_dangerous(command) and self._confirm:
            if not self._confirm(f"Allow dangerous command: {command}?"):
                return "Command cancelled by user."

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"


def register_shell_tools(registry: ToolRegistry, confirm_callback: Any = None) -> None:
    """Register shell tools with the given registry."""
    registry.register(RunShellTool(confirm_callback=confirm_callback))
