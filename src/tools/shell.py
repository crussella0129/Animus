"""Shell tool with command blocking and confirmation."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from typing import Any

from src.core.workspace import Workspace
from src.core.permission import PermissionChecker
from src.tools.base import Tool, ToolRegistry, isolated

# Match single-quoted strings that should be double-quoted on Windows.
# Captures: 'some text' but NOT contractions like don't or it's
# (contractions have a word-char immediately before the opening quote).
_SINGLE_QUOTE_RE = re.compile(r"(?<!\w)'([^']*)'")

# Commands where the trailing argument is a filesystem path.
# Used by _quote_unquoted_path_args to auto-quote paths with spaces.
_PATH_COMMANDS = ("mkdir", "rmdir", "cd", "rd", "md", "type", "del", "copy", "move", "ren")

# Matches: <command> <unquoted path containing spaces> at end of a
# simple command or before a chain operator (&& || ; &).
# Group 1 = command name, Group 2 = unquoted path with spaces.
_UNQUOTED_PATH_RE = re.compile(
    r'\b(' + '|'.join(_PATH_COMMANDS) + r')'   # command name
    r'\s+'                                      # whitespace
    r'([A-Za-z]:\\[^"&|;]+?'                    # drive letter path...
    r'|/[^"&|;]+?)'                             # ...or Unix path
    r'(?=\s*(?:&&|\|\||;|&|$))',                # lookahead for chain or EOL
    re.IGNORECASE,
)

# Shell metacharacters that indicate injection risk.
_SHELL_METACHAR_RE = re.compile(
    r'\|'       # pipe or logical or
    r'|&&'      # logical and
    r'|;'       # command separator
    r'|>'       # redirect out
    r'|<'       # redirect in
    r'|`'       # backtick substitution
    r'|\$\('    # dollar-paren substitution
    r'|&(?!&)'  # background (&) but not already matched &&
)


def _reject_shell_features(command: str) -> str | None:
    """Check command for shell metacharacters. Returns error message if found."""
    if _SHELL_METACHAR_RE.search(command):
        return (
            "Shell features (pipes, redirects, command chaining, substitution) "
            "are not supported. Use separate tool calls instead."
        )
    return None


def _normalize_quotes_for_windows(command: str) -> str:
    r"""Fix quoting issues that cause cmd.exe to mishandle paths with spaces.

    LLMs frequently generate commands with broken quoting on Windows:

    1. Single quotes:  mkdir 'test 1'  →  mkdir "test 1"
       (cmd.exe treats single quotes as literal characters)

    2. No quotes:  mkdir C:\Users\charl\Downloads\test 1  →
                   mkdir "C:\Users\charl\Downloads\test 1"
       (cmd.exe splits on spaces, creating two directories)

    Only applied on Windows.
    """
    # Pass 1: Convert single-quoted strings to double-quoted
    result = _SINGLE_QUOTE_RE.sub(r'"\1"', command)

    # Pass 2: Auto-quote unquoted path arguments that contain spaces
    # for common filesystem commands (mkdir, cd, etc.)
    def _quote_path_match(m: re.Match) -> str:
        cmd_name = m.group(1)
        path = m.group(2).rstrip()
        if ' ' in path and not path.startswith('"'):
            return f'{cmd_name} "{path}"'
        return m.group(0)

    result = _UNQUOTED_PATH_RE.sub(_quote_path_match, result)

    return result


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

    def __init__(
        self,
        confirm_callback: Any = None,
        execution_budget: ExecutionBudget | None = None,
        session_cwd: Workspace | None = None,
        allow_network: bool = False,
    ) -> None:
        super().__init__()  # Initialize Tool base class
        self._confirm = confirm_callback
        self._budget = execution_budget or ExecutionBudget(max_total_seconds=300)
        self._session_cwd = session_cwd
        self._allow_network = allow_network

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

        # On Windows, convert single-quoted args to double quotes so cmd.exe
        # treats them as proper delimiters instead of literal characters.
        if os.name == "nt":
            command = _normalize_quotes_for_windows(command)

        # Check execution budget before running
        budget_ok, budget_msg = self._budget.check_available(timeout)
        if not budget_ok:
            stats = self._budget.stats
            return (
                f"Error: {budget_msg}\n"
                f"Budget stats: {stats['used']}s used in {stats['call_count']} commands, "
                f"{stats['remaining']}s remaining"
            )

        # Reject shell metacharacters (prevents injection)
        shell_err = _reject_shell_features(command)
        if shell_err:
            return f"Error: {shell_err}"

        # Handle cd commands directly via Workspace boundary tracking (shell builtin)
        cd_match = re.match(r'^\s*cd(?:\s+(.*))?$', command)
        if cd_match:
            return self._handle_cd(cd_match.group(1))

        checker = PermissionChecker()
        blocked = checker.is_command_blocked(command)
        if blocked:
            return f"Error: Command blocked for safety: {blocked}"

        if not self._allow_network:
            net_match = checker.is_command_network(command)
            if net_match:
                return (
                    f"Error: Network command blocked: '{net_match}'. "
                    "Outbound network access is disabled by default to prevent "
                    "data exfiltration. Use allow_network=True to enable."
                )

        if checker.is_command_dangerous(command) and self._confirm:
            if not self._confirm(f"Allow dangerous command: {command}?"):
                return "Command cancelled by user."

        # Parse command into list for subprocess (no shell=True)
        try:
            if os.name == "nt":
                cmd_list = shlex.split(command, posix=False)
                # posix=False preserves surrounding quotes as literal chars;
                # strip them so subprocess receives clean tokens.
                cmd_list = [tok.strip('"').strip("'") for tok in cmd_list]
            else:
                cmd_list = shlex.split(command)
        except ValueError as e:
            return f"Error: Could not parse command: {e}"

        if not cmd_list:
            return "Error: Empty command"

        # Windows shell builtins need cmd /c prefix
        _WIN_BUILTINS = {
            "dir", "type", "mkdir", "rmdir", "del", "copy", "move",
            "ren", "rd", "md", "echo", "set", "cls", "ver", "where",
        }
        if os.name == "nt" and cmd_list[0].lower() in _WIN_BUILTINS:
            cmd_list = ["cmd", "/c"] + cmd_list

        # Resolve CWD for the subprocess
        cwd = str(self._session_cwd.path) if self._session_cwd else None

        # Track actual execution time
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
            )
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output.strip() or "(no output)"
        except FileNotFoundError:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error: Command not found: {cmd_list[0]}"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error executing command: {e}"

    def _handle_cd(self, target: str | None) -> str:
        """Handle cd command directly via Workspace."""
        if target is None or not target.strip():
            target = os.path.expanduser("~")
        else:
            target = target.strip().strip("\"'")
        if self._session_cwd is None:
            return f"Changed directory to {target} (no session tracking)"
        old_cwd = self._session_cwd.path
        self._session_cwd.set(target)
        if self._session_cwd.path == old_cwd and target != str(old_cwd):
            return f"Error: Cannot change directory to {target} (outside workspace or does not exist)"
        return f"Changed directory to {self._session_cwd.path}"


def register_shell_tools(
    registry: ToolRegistry,
    confirm_callback: Any = None,
    execution_budget: ExecutionBudget | None = None,
    session_cwd: Workspace | None = None,
    allow_network: bool = False,
) -> None:
    """Register shell tools with the given registry.

    Args:
        registry: The tool registry to register with
        confirm_callback: Optional callback for confirming dangerous commands
        execution_budget: Optional shared execution budget for tracking cumulative time
        session_cwd: Optional Workspace for persisting cd across calls with boundary enforcement
        allow_network: Whether to allow outbound network commands (default: False)
    """
    registry.register(RunShellTool(
        confirm_callback=confirm_callback,
        execution_budget=execution_budget,
        session_cwd=session_cwd,
        allow_network=allow_network,
    ))
