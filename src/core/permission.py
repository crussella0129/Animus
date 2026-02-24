"""Permission checks: deny lists for paths and commands."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

# Directories that should never be written to
DANGEROUS_DIRECTORIES: frozenset[str] = frozenset({
    "/etc",
    "/boot",
    "/usr",
    "/sbin",
    "/bin",
    "/lib",
    "/lib64",
    "/proc",
    "/sys",
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
})

# Files that should never be modified
DANGEROUS_FILES: frozenset[str] = frozenset({
    "/etc/passwd",
    "/etc/shadow",
    "/etc/hosts",
    "/etc/sudoers",
    ".ssh/authorized_keys",
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".animus/config.yaml",
})

# Commands that are always blocked
BLOCKED_COMMANDS: frozenset[str] = frozenset({
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    ":(){ :|:& };:",  # fork bomb
    "> /dev/sda",
    "chmod -R 777 /",
})

# Commands that make outbound network connections.
# Blocked by default in shell execution to prevent data exfiltration
# (e.g., LLM hallucinating a git push to a fabricated remote URL).
NETWORK_COMMANDS: frozenset[str] = frozenset({
    "curl",
    "wget",
    "ssh",
    "scp",
    "sftp",
    "ftp",
    "nc",
    "netcat",
    "ncat",
    "telnet",
    "rsync",
    "git push",
    "git fetch",
    "git pull",
    "git clone",
})

# Commands that require confirmation
DANGEROUS_COMMANDS: frozenset[str] = frozenset({
    "rm",
    "rmdir",
    "del",
    "format",
    "shutdown",
    "reboot",
    "kill",
    "pkill",
    "chmod",
    "chown",
    "sudo",
    "powershell",
    "cmd /c",
})


class PermissionChecker:
    """Check paths and commands against deny lists (singleton).

    This class uses the singleton pattern to avoid creating multiple
    instances that all do the same checks against the same deny lists.
    """

    _instance: PermissionChecker | None = None

    _INJECTION_RE = re.compile(
        r'\$\('       # $(command)
        r'|`'         # `command`
        r'|;'         # command separator
        r'|&&'        # logical and
        r'|\|\|'      # logical or
    )

    def __new__(cls) -> PermissionChecker:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def has_injection_pattern(self, command: str) -> bool:
        """Check if command contains shell injection patterns.

        Defense-in-depth: with shell=True removed, these patterns can't
        execute, but detecting them flags suspicious LLM output.
        """
        return bool(self._INJECTION_RE.search(command))

    def is_path_safe(self, path: Path) -> bool:
        """Check if a path is safe to access. Follows symlinks."""
        try:
            resolved = str(path.resolve(strict=False))
        except (OSError, ValueError):
            return False
        for dangerous in DANGEROUS_DIRECTORIES:
            norm_dangerous = dangerous.replace("\\", "/")
            norm_resolved = resolved.replace("\\", "/")
            if norm_resolved.startswith(norm_dangerous):
                return False
        resolved_fwd = resolved.replace("\\", "/")
        for dangerous in DANGEROUS_FILES:
            if resolved_fwd.endswith(dangerous) or dangerous in resolved_fwd:
                return False
        return True

    def is_command_blocked(self, command: str) -> str | None:
        """Check if command is in the blocked list. Returns reason if blocked."""
        cmd_lower = command.strip().lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return f"Matches blocked pattern: {blocked}"
        return None

    def is_command_dangerous(self, command: str) -> bool:
        """Check if command requires confirmation."""
        cmd_lower = command.strip().lower()
        first_word = cmd_lower.split()[0] if cmd_lower.split() else ""

        for dangerous in DANGEROUS_COMMANDS:
            # Check if first word matches exactly
            if first_word == dangerous.lower():
                return True

            # For multi-word patterns (like "cmd /c"), check if command starts with it
            if " " in dangerous and cmd_lower.startswith(dangerous.lower()):
                return True

        return False

    def is_command_network(self, command: str) -> str | None:
        """Check if command makes outbound network connections.

        Returns the matched pattern if it's a network command, None otherwise.
        """
        cmd_lower = command.strip().lower()
        first_word = cmd_lower.split()[0] if cmd_lower.split() else ""

        for net_cmd in NETWORK_COMMANDS:
            net_lower = net_cmd.lower()
            # Single-word: match first word exactly
            if " " not in net_lower and first_word == net_lower:
                return net_cmd
            # Multi-word (e.g. "git push"): match as prefix
            if " " in net_lower and cmd_lower.startswith(net_lower):
                return net_cmd
        return None
