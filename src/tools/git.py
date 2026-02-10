"""Git tools: status, diff, log, branch, add, commit, checkout."""

from __future__ import annotations

import subprocess
from typing import Any

from src.tools.base import Tool, ToolRegistry

# Operations that are never allowed
_BLOCKED_PATTERNS = frozenset({
    "--force",
    "-f push",
    "push --force",
    "push -f",
    "reset --hard",
    "clean -f",
    "clean -fd",
    "branch -D",
})


def _run_git(args: list[str], timeout: int = 30) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
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
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH"
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {timeout}s"
    except Exception as e:
        return f"Error running git: {e}"


def _is_blocked(command_str: str) -> str | None:
    """Check if a git operation is blocked. Returns reason if blocked."""
    lower = command_str.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in lower:
            return f"Blocked operation: {pattern}"
    return None


class GitStatusTool(Tool):
    """Show working tree status."""

    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return "Show git working tree status (porcelain format)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, args: dict[str, Any]) -> str:
        return _run_git(["status", "--porcelain"])


class GitDiffTool(Tool):
    """Show changes in working tree or staging area."""

    @property
    def name(self) -> str:
        return "git_diff"

    @property
    def description(self) -> str:
        return "Show git diff. Use staged=true for staged changes."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "Show staged changes (default: false)"},
                "path": {"type": "string", "description": "Specific file path to diff (optional)"},
            },
            "required": [],
        }

    def execute(self, args: dict[str, Any]) -> str:
        cmd = ["diff"]
        if args.get("staged"):
            cmd.append("--staged")
        if args.get("path"):
            cmd.extend(["--", args["path"]])
        return _run_git(cmd)


class GitLogTool(Tool):
    """Show commit log."""

    @property
    def name(self) -> str:
        return "git_log"

    @property
    def description(self) -> str:
        return "Show recent git commits (oneline format)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of commits to show (default: 10)"},
            },
            "required": [],
        }

    def execute(self, args: dict[str, Any]) -> str:
        count = args.get("count", 10)
        return _run_git(["log", "--oneline", f"-{count}"])


class GitBranchTool(Tool):
    """List or create branches."""

    def __init__(self, confirm_callback: Any = None) -> None:
        self._confirm = confirm_callback

    @property
    def name(self) -> str:
        return "git_branch"

    @property
    def description(self) -> str:
        return "List branches, or create a new branch."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "create": {"type": "string", "description": "Name of new branch to create (omit to list)"},
            },
            "required": [],
        }

    def execute(self, args: dict[str, Any]) -> str:
        create_name = args.get("create")
        if create_name:
            if self._confirm and not self._confirm(f"Create branch '{create_name}'?"):
                return "Branch creation cancelled by user."
            return _run_git(["branch", create_name])
        return _run_git(["branch", "-a"])


class GitAddTool(Tool):
    """Stage files for commit."""

    @property
    def name(self) -> str:
        return "git_add"

    @property
    def description(self) -> str:
        return "Stage files for the next commit."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to stage",
                },
            },
            "required": ["paths"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        paths = args.get("paths", [])
        if not paths:
            return "Error: No paths specified"
        return _run_git(["add"] + paths)


class GitCommitTool(Tool):
    """Commit staged changes."""

    def __init__(self, confirm_callback: Any = None) -> None:
        self._confirm = confirm_callback

    @property
    def name(self) -> str:
        return "git_commit"

    @property
    def description(self) -> str:
        return "Commit staged changes with a message. Requires confirmation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
            },
            "required": ["message"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        message = args.get("message", "")
        if not message:
            return "Error: Commit message is required"
        if self._confirm and not self._confirm(f"Commit with message: {message}?"):
            return "Commit cancelled by user."
        return _run_git(["commit", "-m", message])


class GitCheckoutTool(Tool):
    """Switch branches or restore files."""

    def __init__(self, confirm_callback: Any = None) -> None:
        self._confirm = confirm_callback

    @property
    def name(self) -> str:
        return "git_checkout"

    @property
    def description(self) -> str:
        return "Switch to a branch. Requires confirmation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Branch name to switch to"},
            },
            "required": ["target"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        target = args.get("target", "")
        if not target:
            return "Error: Target branch is required"
        # Check for blocked operations
        blocked = _is_blocked(f"checkout {target}")
        if blocked:
            return f"Error: {blocked}"
        if self._confirm and not self._confirm(f"Checkout branch '{target}'?"):
            return "Checkout cancelled by user."
        return _run_git(["checkout", target])


def register_git_tools(registry: ToolRegistry, confirm_callback: Any = None) -> None:
    """Register all git tools with the given registry."""
    registry.register(GitStatusTool())
    registry.register(GitDiffTool())
    registry.register(GitLogTool())
    registry.register(GitBranchTool(confirm_callback=confirm_callback))
    registry.register(GitAddTool())
    registry.register(GitCommitTool(confirm_callback=confirm_callback))
    registry.register(GitCheckoutTool(confirm_callback=confirm_callback))
