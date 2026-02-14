"""Shell tool with command blocking and confirmation."""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any

from src.core.cwd import SessionCwd
from src.core.permission import PermissionChecker
from src.tools.base import Tool, ToolRegistry, isolated

# Regex to detect a cd command anywhere in a chained expression
_CD_RE = re.compile(r'(?:^|&&|\|\||;)\s*cd\s')

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
        session_cwd: SessionCwd | None = None,
    ) -> None:
        super().__init__()  # Initialize Tool base class
        self._confirm = confirm_callback
        self._budget = execution_budget or ExecutionBudget(max_total_seconds=300)
        self._session_cwd = session_cwd

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

        checker = PermissionChecker()
        blocked = checker.is_command_blocked(command)
        if blocked:
            return f"Error: Command blocked for safety: {blocked}"

        if checker.is_command_dangerous(command) and self._confirm:
            if not self._confirm(f"Allow dangerous command: {command}?"):
                return "Command cancelled by user."

        # Determine if we need to capture the final CWD after the command
        has_cd = bool(_CD_RE.search(command))
        actual_command = command
        if has_cd and self._session_cwd is not None:
            if os.name == "nt":
                # On Windows, %CD% is expanded at parse time (before cd runs).
                # Instead, append a bare `cd` (prints CWD) between markers.
                actual_command = (
                    f"({command})& echo __ANIMUS_CWD_BEGIN__& cd& echo __ANIMUS_CWD_END__"
                )
            else:
                actual_command = f"({command}); echo __ANIMUS_CWD__$(pwd)__ANIMUS_CWD__"

        # Resolve CWD for the subprocess
        cwd = str(self._session_cwd.path) if self._session_cwd else None

        # Track actual execution time
        start_time = time.time()
        try:
            result = subprocess.run(
                actual_command,
                shell=True,
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

            # Extract and update session CWD from marker
            if has_cd and self._session_cwd is not None:
                output = self._extract_and_update_cwd(output)

            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error executing command: {e}"

    def _extract_and_update_cwd(self, output: str) -> str:
        """Parse CWD marker from output, update SessionCwd, and strip marker lines."""
        if os.name == "nt":
            # Windows: CWD is on a line between BEGIN and END markers
            begin_tag = "__ANIMUS_CWD_BEGIN__"
            end_tag = "__ANIMUS_CWD_END__"
            begin_idx = output.find(begin_tag)
            end_idx = output.find(end_tag)
            if begin_idx != -1 and end_idx != -1:
                between = output[begin_idx + len(begin_tag):end_idx].strip()
                if between and self._session_cwd is not None:
                    self._session_cwd.set(between)
                # Strip everything from begin marker through end marker + trailing newline
                after_end = end_idx + len(end_tag)
                if after_end < len(output) and output[after_end] == '\n':
                    after_end += 1
                output = output[:begin_idx] + output[after_end:]
        else:
            # Unix: CWD is inline between __ANIMUS_CWD__ delimiters
            marker_re = re.compile(r'__ANIMUS_CWD__(.+?)__ANIMUS_CWD__')
            match = marker_re.search(output)
            if match and self._session_cwd is not None:
                self._session_cwd.set(match.group(1).strip())
                output = marker_re.sub('', output)
        return output


def register_shell_tools(
    registry: ToolRegistry,
    confirm_callback: Any = None,
    execution_budget: ExecutionBudget | None = None,
    session_cwd: SessionCwd | None = None,
) -> None:
    """Register shell tools with the given registry.

    Args:
        registry: The tool registry to register with
        confirm_callback: Optional callback for confirming dangerous commands
        execution_budget: Optional shared execution budget for tracking cumulative time
        session_cwd: Optional session-level CWD tracker for persisting cd across calls
    """
    registry.register(RunShellTool(
        confirm_callback=confirm_callback,
        execution_budget=execution_budget,
        session_cwd=session_cwd,
    ))
