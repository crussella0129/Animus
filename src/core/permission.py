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

    def __new__(cls) -> PermissionChecker:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_path_safe(self, path: Path) -> bool:
        """Check if a path is safe to access."""
        resolved = str(path.resolve())
        for dangerous in DANGEROUS_DIRECTORIES:
            if resolved.startswith(dangerous):
                return False
        for dangerous in DANGEROUS_FILES:
            if resolved.endswith(dangerous) or dangerous in resolved:
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
