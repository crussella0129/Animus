"""Hardcoded permission system for Animus.

This module implements security-critical permission checking using ONLY
hardcoded, deterministic logic. NO LLM inference is used for security decisions.

Design Principles:
1. Mandatory deny lists are NON-OVERRIDABLE
2. All path checks use pattern matching (fnmatch), not LLM interpretation
3. Security decisions are made by code, not by prompts
4. Default-deny for sensitive operations
"""

from __future__ import annotations

import os
import re
import shlex
import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class PermissionAction(Enum):
    """Permission actions - hardcoded enum, not strings."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionCategory(Enum):
    """Categories of operations requiring permission."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL_DIRECTORY = "external_directory"


# =============================================================================
# MANDATORY DENY LISTS - NON-OVERRIDABLE, CHECKED FIRST
# =============================================================================

# Directories that should NEVER be written to, regardless of configuration
DANGEROUS_DIRECTORIES: frozenset[str] = frozenset([
    ".git/hooks",
    ".git/hooks/",
    ".git/config",
    ".claude/",
    ".claude/commands/",
    ".cursor/",
    ".vscode/",
    ".idea/",
    ".ssh/",
    ".gnupg/",
    ".aws/",
    ".azure/",
    ".kube/",
    "__pycache__/",
])

# Files that should NEVER be written to, regardless of configuration
DANGEROUS_FILES: frozenset[str] = frozenset([
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".gitconfig",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "authorized_keys",
    "known_hosts",
    ".mcp.json",
])

# Patterns for dangerous file extensions (never write these)
DANGEROUS_PATTERNS: frozenset[str] = frozenset([
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.cer",
    "*.crt",
    "*_rsa",
    "*_ed25519",
    "*.gpg",
    "*.asc",
])

# Commands that are ALWAYS blocked, no exceptions
BLOCKED_COMMANDS: frozenset[str] = frozenset([
    # Fork bombs
    ":(){ :|:& };:",
    # Destructive file operations
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/*",
    "rm -rf .",
    "rm -rf ./*",
    # Sudo destructive (home directory, etc.)
    "sudo rm -rf /",
    "sudo rm -rf /home",
    "sudo rm -rf /etc",
    "sudo rm -rf /var",
    "sudo rm -rf /usr",
    # Disk destruction
    "dd if=/dev/zero of=/dev/sda",
    "dd if=/dev/zero of=/dev/sdb",
    "dd if=/dev/zero of=/dev/nvme",
    "dd if=/dev/random of=/dev/sda",
    "mkfs",
    "mkfs.ext4 /dev/sda",
    "format c:",
    "format c: /q",
    # Windows destructive
    "del /s /q c:\\",
    "del /s /q c:\\*",
    "rd /s /q c:\\",
    # Data exfiltration patterns
    "curl.*|.*sh",
    "wget.*|.*sh",
    "curl.*|.*bash",
    "wget.*|.*bash",
])

# Commands that require confirmation but are not blocked
DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset([
    "rm",
    "rmdir",
    "del",
    "rd",
    "mv",
    "move",
    "rename",
    "chmod",
    "chown",
    "kill",
    "killall",
    "pkill",
    "shutdown",
    "reboot",
    "halt",
    "git push",
    "git reset",
    "git checkout .",
    "git clean",
    "docker rm",
    "docker rmi",
    "docker system prune",
    "kubectl delete",
    "kubectl apply",
    "pip uninstall",
    "npm uninstall",
])

# Read-only safe commands that can auto-execute
SAFE_READ_COMMANDS: frozenset[str] = frozenset([
    "ls",
    "dir",
    "cat",
    "type",
    "head",
    "tail",
    "pwd",
    "echo",
    "which",
    "where",
    "whoami",
    "hostname",
    "uname",
    "date",
    "time",
    "env",
    "printenv",
    "git status",
    "git log",
    "git diff",
    "git branch",
    "git remote",
    "git show",
    "git blame",
    "python --version",
    "python3 --version",
    "pip list",
    "pip show",
    "pip freeze",
    "node --version",
    "npm list",
    "npm ls",
    "cargo --version",
    "rustc --version",
    "go version",
    "java --version",
    "javac --version",
])


@dataclass
class PermissionResult:
    """Result of a permission check."""
    action: PermissionAction
    reason: str
    path: Optional[str] = None
    pattern_matched: Optional[str] = None
    is_mandatory: bool = False  # True if this was a mandatory deny


@dataclass
class PermissionConfig:
    """User-configurable permission settings.

    Note: These CANNOT override DANGEROUS_* lists.
    """
    # Additional patterns to allow (read operations only for dangerous files)
    additional_allow_read: list[str] = field(default_factory=list)

    # Additional patterns to deny
    additional_deny: list[str] = field(default_factory=list)

    # Patterns that require asking
    require_ask: list[str] = field(default_factory=lambda: ["*"])

    # Profile name
    profile: str = "standard"


class PermissionChecker:
    """Hardcoded permission checking system.

    All decisions are made using pattern matching and deterministic logic.
    NO LLM inference is used.
    """

    def __init__(self, config: Optional[PermissionConfig] = None):
        """Initialize with optional configuration.

        Args:
            config: User configuration (cannot override mandatory denies).
        """
        self.config = config or PermissionConfig()

    def _normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize a path for consistent checking.

        Uses only deterministic operations:
        - Resolve to absolute path
        - Normalize separators
        - Handle home directory expansion
        """
        path_str = str(path)

        # Expand ~ to home directory
        if path_str.startswith("~"):
            path_str = os.path.expanduser(path_str)

        # Convert to Path and resolve
        resolved = Path(path_str).resolve()
        return resolved

    def _get_path_components(self, path: Path) -> list[str]:
        """Get path components for pattern matching."""
        return list(path.parts)

    def check_path_mandatory_deny(
        self,
        path: Union[str, Path],
        operation: PermissionCategory,
    ) -> Optional[PermissionResult]:
        """Check if a path hits a MANDATORY deny.

        This is checked FIRST and CANNOT be overridden.

        Args:
            path: Path to check.
            operation: The operation being performed.

        Returns:
            PermissionResult with DENY if mandatory deny hit, None otherwise.
        """
        normalized = self._normalize_path(path)
        path_str = str(normalized).replace("\\", "/")  # Normalize separators
        filename = normalized.name

        # For write/execute operations, check dangerous files and directories
        if operation in (PermissionCategory.WRITE, PermissionCategory.EXECUTE):
            # Check dangerous directories
            for dangerous_dir in DANGEROUS_DIRECTORIES:
                # Check if path is within or is the dangerous directory
                if dangerous_dir in path_str or path_str.endswith(dangerous_dir.rstrip("/")):
                    return PermissionResult(
                        action=PermissionAction.DENY,
                        reason=f"MANDATORY DENY: Path is within protected directory '{dangerous_dir}'",
                        path=path_str,
                        pattern_matched=dangerous_dir,
                        is_mandatory=True,
                    )

            # Check dangerous files
            if filename in DANGEROUS_FILES:
                return PermissionResult(
                    action=PermissionAction.DENY,
                    reason=f"MANDATORY DENY: File '{filename}' is protected",
                    path=path_str,
                    pattern_matched=filename,
                    is_mandatory=True,
                )

            # Check dangerous patterns
            for pattern in DANGEROUS_PATTERNS:
                if fnmatch.fnmatch(filename, pattern):
                    return PermissionResult(
                        action=PermissionAction.DENY,
                        reason=f"MANDATORY DENY: File matches protected pattern '{pattern}'",
                        path=path_str,
                        pattern_matched=pattern,
                        is_mandatory=True,
                    )

        return None  # No mandatory deny

    def check_command_mandatory_deny(self, command: str) -> Optional[PermissionResult]:
        """Check if a command hits a MANDATORY deny.

        This is checked FIRST and CANNOT be overridden.

        Args:
            command: Command string to check.

        Returns:
            PermissionResult with DENY if mandatory deny hit, None otherwise.
        """
        cmd_lower = command.lower().strip()

        # Check exact matches and word-bounded matches
        for blocked in BLOCKED_COMMANDS:
            blocked_lower = blocked.lower()
            # Exact match
            if cmd_lower == blocked_lower:
                return PermissionResult(
                    action=PermissionAction.DENY,
                    reason=f"MANDATORY DENY: Command matches blocked pattern '{blocked}'",
                    pattern_matched=blocked,
                    is_mandatory=True,
                )
            # Command starts with blocked pattern followed by space or nothing else
            # This prevents "rm -rf ." from matching "rm -rf ./build"
            if cmd_lower.startswith(blocked_lower + " ") or cmd_lower.startswith(blocked_lower + "\t"):
                return PermissionResult(
                    action=PermissionAction.DENY,
                    reason=f"MANDATORY DENY: Command matches blocked pattern '{blocked}'",
                    pattern_matched=blocked,
                    is_mandatory=True,
                )

        # Also check for destructive patterns with sudo prefix (HARDCODED)
        # Strip sudo and check core command against dangerous patterns
        cmd_no_sudo = cmd_lower
        if cmd_no_sudo.startswith("sudo "):
            cmd_no_sudo = cmd_no_sudo[5:].strip()
            # Re-check without sudo prefix
            for blocked in BLOCKED_COMMANDS:
                blocked_lower = blocked.lower()
                # Skip sudo-specific patterns (already checked above)
                if blocked_lower.startswith("sudo "):
                    continue
                if cmd_no_sudo == blocked_lower:
                    return PermissionResult(
                        action=PermissionAction.DENY,
                        reason=f"MANDATORY DENY: sudo command matches blocked pattern '{blocked}'",
                        pattern_matched=blocked,
                        is_mandatory=True,
                    )
                if cmd_no_sudo.startswith(blocked_lower + " ") or cmd_no_sudo.startswith(blocked_lower + "\t"):
                    return PermissionResult(
                        action=PermissionAction.DENY,
                        reason=f"MANDATORY DENY: sudo command matches blocked pattern '{blocked}'",
                        pattern_matched=blocked,
                        is_mandatory=True,
                    )

        # Check for pipe to shell patterns (data exfiltration)
        pipe_to_shell = re.compile(
            r'(curl|wget|fetch)\s+.*\|\s*(sh|bash|zsh|ksh|csh|tcsh|fish)',
            re.IGNORECASE
        )
        if pipe_to_shell.search(command):
            return PermissionResult(
                action=PermissionAction.DENY,
                reason="MANDATORY DENY: Piping remote content to shell is blocked",
                pattern_matched="pipe_to_shell",
                is_mandatory=True,
            )

        return None  # No mandatory deny

    def check_path(
        self,
        path: Union[str, Path],
        operation: PermissionCategory,
    ) -> PermissionResult:
        """Check permission for a path operation.

        Checks in order:
        1. Mandatory denies (CANNOT be overridden)
        2. User-configured additional denies
        3. Read operations on dangerous files (allowed for read, denied for write)
        4. User-configured allow patterns
        5. Default to ASK

        Args:
            path: Path to check.
            operation: The operation type.

        Returns:
            PermissionResult with the decision.
        """
        # Step 1: Check mandatory denies FIRST
        mandatory = self.check_path_mandatory_deny(path, operation)
        if mandatory:
            return mandatory

        normalized = self._normalize_path(path)
        path_str = str(normalized)
        filename = normalized.name

        # Step 2: Check user-configured additional denies
        for pattern in self.config.additional_deny:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(filename, pattern):
                return PermissionResult(
                    action=PermissionAction.DENY,
                    reason=f"Denied by user configuration: matches pattern '{pattern}'",
                    path=path_str,
                    pattern_matched=pattern,
                )

        # Step 3: For READ operations, allow most things
        if operation == PermissionCategory.READ:
            return PermissionResult(
                action=PermissionAction.ALLOW,
                reason="Read operations are allowed by default",
                path=path_str,
            )

        # Step 4: For WRITE/EXECUTE, check additional allow patterns
        for pattern in self.config.additional_allow_read:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(filename, pattern):
                return PermissionResult(
                    action=PermissionAction.ALLOW,
                    reason=f"Allowed by user configuration: matches pattern '{pattern}'",
                    path=path_str,
                    pattern_matched=pattern,
                )

        # Step 5: Default to ASK for write/execute operations
        return PermissionResult(
            action=PermissionAction.ASK,
            reason="Write/execute operations require confirmation by default",
            path=path_str,
        )

    def check_command(self, command: str) -> PermissionResult:
        """Check permission for a shell command.

        Checks in order:
        1. Mandatory denies (CANNOT be overridden)
        2. Destructive commands (require confirmation)
        3. Safe read commands (auto-allow)
        4. Default to ASK

        Args:
            command: Command string to check.

        Returns:
            PermissionResult with the decision.
        """
        # Step 1: Check mandatory denies FIRST
        mandatory = self.check_command_mandatory_deny(command)
        if mandatory:
            return mandatory

        cmd_lower = command.lower().strip()

        # Step 2: Parse command to get base command
        try:
            parts = shlex.split(command)
            base_cmd = parts[0] if parts else ""
        except ValueError:
            # Malformed command, be cautious
            base_cmd = cmd_lower.split()[0] if cmd_lower.split() else ""

        # Step 3: Check for destructive commands
        for destructive in DESTRUCTIVE_COMMANDS:
            if destructive.lower() in cmd_lower or cmd_lower.startswith(destructive.lower()):
                return PermissionResult(
                    action=PermissionAction.ASK,
                    reason=f"Command contains destructive operation '{destructive}'",
                    pattern_matched=destructive,
                )

        # Step 4: Check for safe read commands
        for safe_cmd in SAFE_READ_COMMANDS:
            if cmd_lower.startswith(safe_cmd.lower()):
                return PermissionResult(
                    action=PermissionAction.ALLOW,
                    reason="Command is safe read-only operation",
                    pattern_matched=safe_cmd,
                )

        # Step 5: Default to ASK for unknown commands
        return PermissionResult(
            action=PermissionAction.ASK,
            reason="Unknown command requires confirmation",
        )

    def is_symlink_escape(
        self,
        symlink_path: Union[str, Path],
        allowed_boundary: Union[str, Path],
    ) -> bool:
        """Check if a symlink points outside the allowed boundary.

        This is a security check to prevent symlink-based escapes.

        Args:
            symlink_path: Path to the symlink.
            allowed_boundary: The boundary directory that should contain the target.

        Returns:
            True if the symlink escapes the boundary, False if safe.
        """
        try:
            symlink = Path(symlink_path)
            boundary = Path(allowed_boundary).resolve()

            if not symlink.is_symlink():
                return False  # Not a symlink

            # Resolve the symlink target
            target = symlink.resolve()

            # Check if target is within boundary
            try:
                target.relative_to(boundary)
                return False  # Safe, within boundary
            except ValueError:
                return True  # Escapes boundary!

        except (OSError, ValueError):
            # If we can't check, assume unsafe
            return True


# Singleton for global permission checking
_default_checker: Optional[PermissionChecker] = None


def get_permission_checker(config: Optional[PermissionConfig] = None) -> PermissionChecker:
    """Get the global permission checker.

    Args:
        config: Optional configuration to apply.

    Returns:
        PermissionChecker instance.
    """
    global _default_checker
    if _default_checker is None or config is not None:
        _default_checker = PermissionChecker(config)
    return _default_checker


def check_path_permission(
    path: Union[str, Path],
    operation: PermissionCategory,
) -> PermissionResult:
    """Convenience function to check path permission.

    Args:
        path: Path to check.
        operation: Operation type.

    Returns:
        PermissionResult.
    """
    return get_permission_checker().check_path(path, operation)


def check_command_permission(command: str) -> PermissionResult:
    """Convenience function to check command permission.

    Args:
        command: Command to check.

    Returns:
        PermissionResult.
    """
    return get_permission_checker().check_command(command)


def is_mandatory_deny_path(path: Union[str, Path], operation: PermissionCategory) -> bool:
    """Quick check if a path is mandatory denied.

    Args:
        path: Path to check.
        operation: Operation type.

    Returns:
        True if mandatory denied.
    """
    result = get_permission_checker().check_path_mandatory_deny(path, operation)
    return result is not None


def is_mandatory_deny_command(command: str) -> bool:
    """Quick check if a command is mandatory denied.

    Args:
        command: Command to check.

    Returns:
        True if mandatory denied.
    """
    result = get_permission_checker().check_command_mandatory_deny(command)
    return result is not None
