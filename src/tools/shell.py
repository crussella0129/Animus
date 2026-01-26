"""Shell execution tool with safety controls."""

from __future__ import annotations

import asyncio
import subprocess
import shlex
import sys
from typing import Any, Optional, Callable, Awaitable
from pathlib import Path

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory


# Commands that are considered destructive and require extra confirmation
DESTRUCTIVE_COMMANDS = {
    "rm", "rmdir", "del", "rd",  # Delete
    "mv", "move", "rename",       # Move/rename
    "format",                      # Format disk
    "dd",                          # Disk operations
    "mkfs",                        # Make filesystem
    "fdisk", "parted",            # Partition tools
    "chmod", "chown",             # Permission changes
    "kill", "killall", "pkill",   # Process killing
    "shutdown", "reboot", "halt", # System control
    "git push", "git reset",      # Git operations
    "docker rm", "docker rmi",    # Docker cleanup
    "kubectl delete",             # Kubernetes deletion
}

# Commands that should never be run
BLOCKED_COMMANDS = {
    ":(){ :|:& };:",  # Fork bomb
    "rm -rf /",
    "rm -rf /*",
    "dd if=/dev/zero of=/dev/sda",
}


class ShellTool(Tool):
    """
    Tool to execute shell commands.

    Includes safety controls:
    - Destructive command detection
    - Human-in-the-loop confirmation
    - Command blocking
    - Timeout protection
    """

    def __init__(
        self,
        confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        working_dir: Optional[Path] = None,
        timeout: int = 60,
    ):
        """
        Initialize the shell tool.

        Args:
            confirm_callback: Async callback for confirmation prompts.
                             Receives command string, returns True to allow.
            working_dir: Working directory for commands.
            timeout: Command timeout in seconds.
        """
        self.confirm_callback = confirm_callback
        self.working_dir = working_dir
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Use this to run system commands, "
            "build projects, run tests, etc. Destructive commands require confirmation."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The shell command to execute.",
            ),
            ToolParameter(
                name="working_dir",
                type="string",
                description="Working directory for the command. Defaults to current directory.",
                required=False,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds. Defaults to 60.",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def requires_confirmation(self) -> bool:
        return True  # All shell commands go through confirmation logic

    def _is_blocked(self, command: str) -> bool:
        """Check if command is blocked."""
        cmd_lower = command.lower().strip()
        return any(blocked in cmd_lower for blocked in BLOCKED_COMMANDS)

    def _is_destructive(self, command: str) -> bool:
        """Check if command is potentially destructive."""
        cmd_lower = command.lower().strip()

        # Check for destructive command patterns
        for pattern in DESTRUCTIVE_COMMANDS:
            if pattern in cmd_lower:
                return True

        # Check for piping to dangerous commands
        if "|" in command:
            parts = command.split("|")
            for part in parts:
                part = part.strip()
                for pattern in DESTRUCTIVE_COMMANDS:
                    if part.startswith(pattern):
                        return True

        return False

    async def _get_confirmation(self, command: str) -> bool:
        """Get confirmation for a command."""
        if self.confirm_callback:
            return await self.confirm_callback(command)

        # Default: no callback means always confirm (safe default)
        return False

    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a shell command."""
        # Check for blocked commands
        if self._is_blocked(command):
            return ToolResult(
                success=False,
                output="",
                error="This command is blocked for safety reasons.",
            )

        # Check for destructive commands
        is_destructive = self._is_destructive(command)

        # Get confirmation if needed
        if is_destructive:
            confirmed = await self._get_confirmation(command)
            if not confirmed:
                return ToolResult(
                    success=False,
                    output="",
                    error="Command requires confirmation. User declined.",
                    metadata={"requires_confirmation": True},
                )

        # Determine working directory
        cwd = Path(working_dir) if working_dir else self.working_dir
        if cwd:
            cwd = cwd.resolve()
            if not cwd.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Working directory not found: {cwd}",
                )

        # Determine timeout
        cmd_timeout = timeout or self.timeout

        try:
            # Use shell=True for command interpretation
            # This is intentional for an agent that needs to run arbitrary commands
            if sys.platform == "win32":
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=cmd_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {cmd_timeout} seconds.",
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            success = process.returncode == 0

            # Combine output
            output = stdout_str
            if stderr_str and not success:
                output = f"{stdout_str}\n\nSTDERR:\n{stderr_str}" if stdout_str else stderr_str

            return ToolResult(
                success=success,
                output=output,
                error=stderr_str if not success else None,
                metadata={
                    "return_code": process.returncode,
                    "command": command,
                    "working_dir": str(cwd) if cwd else None,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error executing command: {e}",
            )
