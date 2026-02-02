"""Git tools with safety features like diff preview before commits."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any, Optional, Callable, Awaitable
from pathlib import Path

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory


class GitTool(Tool):
    """
    Tool for git operations with safety features.

    Provides:
    - Automatic diff preview before commits
    - Status check before operations
    - Safe defaults for destructive operations
    """

    def __init__(
        self,
        confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        working_dir: Optional[Path] = None,
        timeout: int = 60,
    ):
        """
        Initialize the git tool.

        Args:
            confirm_callback: Async callback for confirmation prompts.
            working_dir: Working directory for git commands.
            timeout: Command timeout in seconds.
        """
        self.confirm_callback = confirm_callback
        self.working_dir = working_dir
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Execute git commands with safety features. "
            "Commit operations automatically show diff preview before proceeding. "
            "Supports status, diff, add, commit, push, pull, branch, and checkout."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type="string",
                description=(
                    "Git operation: status, diff, add, commit, push, pull, "
                    "branch, checkout, log, or raw (for other git commands)."
                ),
            ),
            ToolParameter(
                name="args",
                type="string",
                description="Arguments for the git operation (e.g., file paths, branch names, commit message).",
                required=False,
            ),
            ToolParameter(
                name="message",
                type="string",
                description="Commit message (for commit operation).",
                required=False,
            ),
            ToolParameter(
                name="skip_diff_preview",
                type="boolean",
                description="Skip the diff preview for commit (default: false).",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def requires_confirmation(self) -> bool:
        return True

    async def _run_git(
        self,
        args: list[str],
        cwd: Optional[Path] = None,
    ) -> tuple[bool, str, str]:
        """Run a git command and return (success, stdout, stderr)."""
        cmd = ["git"] + args
        working_dir = cwd or self.working_dir

        try:
            if sys.platform == "win32":
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(working_dir) if working_dir else None,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(working_dir) if working_dir else None,
                )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )

            return (
                process.returncode == 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            return False, "", f"Command timed out after {self.timeout} seconds"
        except Exception as e:
            return False, "", str(e)

    async def _get_diff_preview(self) -> str:
        """Get a diff preview for staged and unstaged changes."""
        # Get staged changes
        success_staged, staged_diff, _ = await self._run_git(["diff", "--cached"])
        # Get unstaged changes
        success_unstaged, unstaged_diff, _ = await self._run_git(["diff"])
        # Get untracked files
        success_status, status, _ = await self._run_git(["status", "--short"])

        preview_parts = []

        if staged_diff.strip():
            preview_parts.append("=== STAGED CHANGES ===")
            preview_parts.append(staged_diff.strip())

        if unstaged_diff.strip():
            preview_parts.append("\n=== UNSTAGED CHANGES ===")
            preview_parts.append(unstaged_diff.strip())

        # Show untracked files from status
        untracked = [
            line for line in status.split("\n")
            if line.startswith("??")
        ]
        if untracked:
            preview_parts.append("\n=== UNTRACKED FILES ===")
            for line in untracked:
                preview_parts.append(line[3:])  # Remove "?? " prefix

        if not preview_parts:
            return "No changes detected."

        return "\n".join(preview_parts)

    async def _get_confirmation(self, message: str) -> bool:
        """Get confirmation for an operation."""
        if self.confirm_callback:
            return await self.confirm_callback(message)
        return False

    async def execute(
        self,
        operation: str,
        args: Optional[str] = None,
        message: Optional[str] = None,
        skip_diff_preview: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a git operation."""
        operation = operation.lower().strip()
        arg_list = args.split() if args else []

        # Handle different operations
        if operation == "status":
            success, stdout, stderr = await self._run_git(["status"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout,
                error=stderr if not success else None,
            )

        elif operation == "diff":
            success, stdout, stderr = await self._run_git(["diff"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout if stdout else "No changes.",
                error=stderr if not success else None,
            )

        elif operation == "log":
            # Default to last 10 commits if no args
            if not arg_list:
                arg_list = ["-10", "--oneline"]
            success, stdout, stderr = await self._run_git(["log"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout,
                error=stderr if not success else None,
            )

        elif operation == "add":
            if not arg_list:
                return ToolResult(
                    success=False,
                    output="",
                    error="Please specify files to add (use '.' for all files).",
                )
            success, stdout, stderr = await self._run_git(["add"] + arg_list)
            return ToolResult(
                success=success,
                output=f"Added: {' '.join(arg_list)}" if success else "",
                error=stderr if not success else None,
            )

        elif operation == "commit":
            if not message:
                return ToolResult(
                    success=False,
                    output="",
                    error="Commit message required. Use the 'message' parameter.",
                )

            # Show diff preview unless skipped
            if not skip_diff_preview:
                diff_preview = await self._get_diff_preview()

                # Build confirmation message
                confirm_msg = f"=== COMMIT PREVIEW ===\n\nMessage: {message}\n\n{diff_preview}"

                # Get confirmation
                confirmed = await self._get_confirmation(confirm_msg)
                if not confirmed:
                    return ToolResult(
                        success=False,
                        output=diff_preview,
                        error="Commit cancelled by user.",
                        metadata={"preview_shown": True},
                    )

            # Execute commit
            success, stdout, stderr = await self._run_git(["commit", "-m", message])
            return ToolResult(
                success=success,
                output=stdout,
                error=stderr if not success else None,
            )

        elif operation == "push":
            # Always require confirmation for push
            confirm_msg = f"Push to remote: {' '.join(arg_list) if arg_list else 'origin (current branch)'}"
            confirmed = await self._get_confirmation(confirm_msg)
            if not confirmed:
                return ToolResult(
                    success=False,
                    output="",
                    error="Push cancelled by user.",
                )

            success, stdout, stderr = await self._run_git(["push"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout if stdout else "Push successful.",
                error=stderr if not success else None,
            )

        elif operation == "pull":
            success, stdout, stderr = await self._run_git(["pull"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout,
                error=stderr if not success else None,
            )

        elif operation == "branch":
            success, stdout, stderr = await self._run_git(["branch"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout,
                error=stderr if not success else None,
            )

        elif operation == "checkout":
            if not arg_list:
                return ToolResult(
                    success=False,
                    output="",
                    error="Please specify a branch or file to checkout.",
                )

            # Show warning for checkout
            confirm_msg = f"Checkout: {' '.join(arg_list)}"
            confirmed = await self._get_confirmation(confirm_msg)
            if not confirmed:
                return ToolResult(
                    success=False,
                    output="",
                    error="Checkout cancelled by user.",
                )

            success, stdout, stderr = await self._run_git(["checkout"] + arg_list)
            return ToolResult(
                success=success,
                output=stdout if stdout else f"Switched to: {arg_list[0]}",
                error=stderr if not success else None,
            )

        elif operation == "raw":
            # Run any git command (advanced users)
            if not arg_list:
                return ToolResult(
                    success=False,
                    output="",
                    error="Please specify git arguments for raw operation.",
                )
            success, stdout, stderr = await self._run_git(arg_list)
            return ToolResult(
                success=success,
                output=stdout,
                error=stderr if not success else None,
            )

        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown operation: {operation}. Use status, diff, add, commit, push, pull, branch, checkout, log, or raw.",
            )


def create_git_tool(
    confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
    working_dir: Optional[Path] = None,
) -> GitTool:
    """Create a git tool instance."""
    return GitTool(confirm_callback=confirm_callback, working_dir=working_dir)
