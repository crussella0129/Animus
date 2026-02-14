"""Git tools: status, diff, log, branch, add, commit, checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from src.core.cwd import SessionCwd
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


def _check_git_repo(session_cwd: SessionCwd | None) -> str | None:
    """Verify git repo exists at the session CWD.

    For mutating operations (add, commit, branch create, checkout), we check
    that .git exists at or below the session CWD to prevent accidentally
    operating on a parent directory's repo.  Returns an error string if
    no repo is found, or None if OK.

    Read-only operations (status, diff, log) are allowed to inherit a
    parent .git — git does this natively and it's harmless.
    """
    if session_cwd is None:
        return None  # No session CWD tracking — use git's default behavior

    cwd = session_cwd.path
    # Walk up from session CWD looking for .git
    git_dir = _find_git_root(cwd)
    if git_dir is None:
        return f"Error: No git repository found at or above {cwd}"

    # If .git is in a parent far above the session CWD, warn the agent.
    # This catches the case where the user did `cd Downloads` but the
    # .git is at C:\Users\charl (the home dir).
    if git_dir != cwd:
        # Check how many levels up the .git is
        try:
            cwd.relative_to(git_dir)
        except ValueError:
            return f"Error: No git repository found at {cwd}"

        # If .git is more than 2 levels up and the session has changed CWD,
        # this is likely an inherited repo that the user doesn't intend to modify.
        rel_depth = len(cwd.relative_to(git_dir).parts)
        if rel_depth > 2:
            return (
                f"Error: git repository root is at {git_dir} but session CWD is "
                f"{cwd} ({rel_depth} levels deep). This is likely an unintended "
                f"repo. Use git_init or cd to the correct repository first."
            )

    return None


def _find_git_root(path: Path) -> Path | None:
    """Walk up from path to find the directory containing .git."""
    current = path.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _run_git(args: list[str], timeout: int = 30, cwd: str | None = None) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
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

    def __init__(self, session_cwd: SessionCwd | None = None) -> None:
        self._session_cwd = session_cwd

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
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        return _run_git(["status", "--porcelain"], cwd=cwd)


class GitDiffTool(Tool):
    """Show changes in working tree or staging area."""

    def __init__(self, session_cwd: SessionCwd | None = None) -> None:
        self._session_cwd = session_cwd

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
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        return _run_git(cmd, cwd=cwd)


class GitLogTool(Tool):
    """Show commit log."""

    def __init__(self, session_cwd: SessionCwd | None = None) -> None:
        self._session_cwd = session_cwd

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
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        return _run_git(["log", "--oneline", f"-{count}"], cwd=cwd)


class GitBranchTool(Tool):
    """List or create branches."""

    def __init__(self, confirm_callback: Any = None, session_cwd: SessionCwd | None = None) -> None:
        self._confirm = confirm_callback
        self._session_cwd = session_cwd

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
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        create_name = args.get("create")
        if create_name:
            repo_err = _check_git_repo(self._session_cwd)
            if repo_err:
                return repo_err
            if self._confirm and not self._confirm(f"Create branch '{create_name}'?"):
                return "Branch creation cancelled by user."
            return _run_git(["branch", create_name], cwd=cwd)
        return _run_git(["branch", "-a"], cwd=cwd)


class GitAddTool(Tool):
    """Stage files for commit."""

    def __init__(self, session_cwd: SessionCwd | None = None) -> None:
        self._session_cwd = session_cwd

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
        repo_err = _check_git_repo(self._session_cwd)
        if repo_err:
            return repo_err
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        return _run_git(["add"] + paths, cwd=cwd)


class GitCommitTool(Tool):
    """Commit staged changes."""

    def __init__(self, confirm_callback: Any = None, session_cwd: SessionCwd | None = None) -> None:
        self._confirm = confirm_callback
        self._session_cwd = session_cwd

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
        repo_err = _check_git_repo(self._session_cwd)
        if repo_err:
            return repo_err
        if self._confirm and not self._confirm(f"Commit with message: {message}?"):
            return "Commit cancelled by user."
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        return _run_git(["commit", "-m", message], cwd=cwd)


class GitCheckoutTool(Tool):
    """Switch branches or restore files."""

    def __init__(self, confirm_callback: Any = None, session_cwd: SessionCwd | None = None) -> None:
        self._confirm = confirm_callback
        self._session_cwd = session_cwd

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
        repo_err = _check_git_repo(self._session_cwd)
        if repo_err:
            return repo_err
        if self._confirm and not self._confirm(f"Checkout branch '{target}'?"):
            return "Checkout cancelled by user."
        cwd = str(self._session_cwd.path) if self._session_cwd else None
        return _run_git(["checkout", target], cwd=cwd)


def register_git_tools(
    registry: ToolRegistry,
    confirm_callback: Any = None,
    session_cwd: SessionCwd | None = None,
) -> None:
    """Register all git tools with the given registry."""
    registry.register(GitStatusTool(session_cwd=session_cwd))
    registry.register(GitDiffTool(session_cwd=session_cwd))
    registry.register(GitLogTool(session_cwd=session_cwd))
    registry.register(GitBranchTool(confirm_callback=confirm_callback, session_cwd=session_cwd))
    registry.register(GitAddTool(session_cwd=session_cwd))
    registry.register(GitCommitTool(confirm_callback=confirm_callback, session_cwd=session_cwd))
    registry.register(GitCheckoutTool(confirm_callback=confirm_callback, session_cwd=session_cwd))
