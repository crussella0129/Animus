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
from functools import lru_cache
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


# =============================================================================
# DEFAULT PERMISSION PROFILES — HARDCODED CONFIGURATIONS
# =============================================================================

PERMISSION_PROFILES: dict[str, "PermissionConfig"] = {}


def _init_profiles() -> None:
    """Initialize default permission profiles (called at module load)."""
    # strict: ask for everything except reads
    PERMISSION_PROFILES["strict"] = PermissionConfig(
        additional_allow_read=[],
        additional_deny=[],
        require_ask=["*"],
        profile="strict",
    )

    # standard: allow reads, ask for writes/bash
    PERMISSION_PROFILES["standard"] = PermissionConfig(
        additional_allow_read=[],
        additional_deny=[],
        require_ask=["*"],
        profile="standard",
    )

    # trusted: allow most operations, ask only for destructive
    PERMISSION_PROFILES["trusted"] = PermissionConfig(
        additional_allow_read=["*"],
        additional_deny=[],
        require_ask=[],
        profile="trusted",
    )


_init_profiles()


def get_profile(name: str) -> PermissionConfig:
    """Get a permission profile by name.

    Args:
        name: Profile name ("strict", "standard", or "trusted").

    Returns:
        PermissionConfig for the profile.

    Raises:
        ValueError: If profile name is unknown.
    """
    if name not in PERMISSION_PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {', '.join(PERMISSION_PROFILES.keys())}"
        )
    return PERMISSION_PROFILES[name]


# =============================================================================
# PER-AGENT PERMISSION SCOPES — HARDCODED
# =============================================================================


@dataclass
class AgentPermissionScope:
    """Defines permission scope for a specific agent type.

    These are hardcoded configurations — not user-overridable for security.
    """
    name: str
    can_read: bool = True
    can_write: bool = False
    can_execute: bool = False
    allowed_shell_commands: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def check_operation(self, operation: PermissionCategory) -> PermissionAction:
        """Check if an operation is allowed for this scope."""
        if operation == PermissionCategory.READ:
            return PermissionAction.ALLOW if self.can_read else PermissionAction.DENY
        elif operation == PermissionCategory.WRITE:
            return PermissionAction.ALLOW if self.can_write else PermissionAction.DENY
        elif operation == PermissionCategory.EXECUTE:
            return PermissionAction.ALLOW if self.can_execute else PermissionAction.DENY
        return PermissionAction.ASK

    def check_command(self, command: str) -> PermissionAction:
        """Check if a shell command is allowed for this scope."""
        if not self.can_execute:
            return PermissionAction.DENY

        if not self.allowed_shell_commands:
            return PermissionAction.ALLOW  # No restrictions on which commands

        cmd_lower = command.lower().strip()

        # Check if command starts with any allowed command
        # Handles multi-word entries like "git status", "pip list"
        for allowed in self.allowed_shell_commands:
            if cmd_lower.startswith(allowed.lower()):
                return PermissionAction.ALLOW
        return PermissionAction.DENY


# Predefined agent scopes
AGENT_SCOPES: dict[str, AgentPermissionScope] = {
    "explore": AgentPermissionScope(
        name="explore",
        can_read=True,
        can_write=False,
        can_execute=False,
        description="Read-only exploration: no writes, no shell",
    ),
    "plan": AgentPermissionScope(
        name="plan",
        can_read=True,
        can_write=False,
        can_execute=True,
        allowed_shell_commands=frozenset(SAFE_READ_COMMANDS),
        description="Read-only with safe shell commands for analysis",
    ),
    "build": AgentPermissionScope(
        name="build",
        can_read=True,
        can_write=True,
        can_execute=True,
        description="Standard permissions: read, write, execute with confirmation",
    ),
}


def get_agent_scope(name: str) -> AgentPermissionScope:
    """Get a per-agent permission scope by name.

    Args:
        name: Scope name ("explore", "plan", or "build").

    Returns:
        AgentPermissionScope instance.

    Raises:
        ValueError: If scope name is unknown.
    """
    if name not in AGENT_SCOPES:
        raise ValueError(
            f"Unknown agent scope '{name}'. Available: {', '.join(AGENT_SCOPES.keys())}"
        )
    return AGENT_SCOPES[name]


class PermissionChecker:
    """Hardcoded permission checking system.

    All decisions are made using pattern matching and deterministic logic.
    NO LLM inference is used.
    """

    def __init__(
        self,
        config: Optional[PermissionConfig] = None,
        agent_scope: Optional[AgentPermissionScope] = None,
        cache_size: int = 256,
    ):
        """Initialize with optional configuration.

        Args:
            config: User configuration (cannot override mandatory denies).
            agent_scope: Per-agent permission scope (explore, plan, build).
            cache_size: Max entries in the permission cache (0 to disable).
        """
        self.config = config or PermissionConfig()
        self.agent_scope = agent_scope
        self._cache_size = cache_size
        self._path_cache: dict[tuple[str, str], PermissionResult] = {}
        self._command_cache: dict[str, PermissionResult] = {}

    def clear_cache(self) -> None:
        """Clear the permission evaluation cache."""
        self._path_cache.clear()
        self._command_cache.clear()

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "path_entries": len(self._path_cache),
            "command_entries": len(self._command_cache),
            "max_size": self._cache_size,
        }

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

    def _is_path_component_match(self, path_str: str, dangerous_pattern: str) -> bool:
        """Check if a dangerous pattern matches as a proper path component.

        This prevents false positives like ".git" matching "my.git" or
        ".ssh" matching "my.ssh_backup".

        Args:
            path_str: Normalized path string with forward slashes.
            dangerous_pattern: Pattern to check (e.g., ".git/", ".ssh/").

        Returns:
            True if the pattern matches as a proper path component.
        """
        # Normalize the pattern (remove trailing slash for comparison)
        pattern = dangerous_pattern.rstrip("/")

        # Split path into components
        path_parts = [p for p in path_str.split("/") if p]

        # Check each path component
        for i, part in enumerate(path_parts):
            # Exact match of component
            if part == pattern:
                return True

            # Check for pattern at start of component followed by / (e.g., ".git/hooks")
            # This handles cases like ".git/hooks" where we check ".git/hooks"
            if "/" in dangerous_pattern:
                # Multi-component pattern like ".git/hooks"
                pattern_parts = [p for p in dangerous_pattern.rstrip("/").split("/") if p]
                if len(pattern_parts) <= len(path_parts) - i:
                    if path_parts[i:i + len(pattern_parts)] == pattern_parts:
                        return True

        return False

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
            # Check dangerous directories using proper path component matching
            for dangerous_dir in DANGEROUS_DIRECTORIES:
                if self._is_path_component_match(path_str, dangerous_dir):
                    return PermissionResult(
                        action=PermissionAction.DENY,
                        reason=f"MANDATORY DENY: Path is within protected directory '{dangerous_dir}'",
                        path=path_str,
                        pattern_matched=dangerous_dir,
                        is_mandatory=True,
                    )

            # Check dangerous files (exact filename match)
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
        0. Cache lookup
        1. Mandatory denies (CANNOT be overridden)
        1.5. Agent scope restrictions
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
        # Step 0: Cache lookup
        cache_key = (str(path), operation.value)
        if self._cache_size > 0 and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        result = self._check_path_uncached(path, operation)

        # Store in cache (evict oldest if full)
        if self._cache_size > 0:
            if len(self._path_cache) >= self._cache_size:
                # Remove oldest entry (first key)
                oldest = next(iter(self._path_cache))
                del self._path_cache[oldest]
            self._path_cache[cache_key] = result

        return result

    def _check_path_uncached(
        self,
        path: Union[str, Path],
        operation: PermissionCategory,
    ) -> PermissionResult:
        """Check path permission without caching."""
        # Step 1: Check mandatory denies FIRST
        mandatory = self.check_path_mandatory_deny(path, operation)
        if mandatory:
            return mandatory

        normalized = self._normalize_path(path)
        path_str = str(normalized)
        filename = normalized.name

        # Step 1.5: Check agent scope restrictions
        if self.agent_scope:
            action = self.agent_scope.check_operation(operation)
            if action == PermissionAction.DENY:
                return PermissionResult(
                    action=PermissionAction.DENY,
                    reason=f"Denied by agent scope '{self.agent_scope.name}': "
                           f"{operation.value} not allowed",
                    path=path_str,
                )

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
        0. Cache lookup
        1. Mandatory denies (CANNOT be overridden)
        1.5. Agent scope restrictions
        2. Destructive commands (require confirmation)
        3. Safe read commands (auto-allow)
        4. Default to ASK

        Args:
            command: Command string to check.

        Returns:
            PermissionResult with the decision.
        """
        # Step 0: Cache lookup
        if self._cache_size > 0 and command in self._command_cache:
            return self._command_cache[command]

        result = self._check_command_uncached(command)

        # Store in cache
        if self._cache_size > 0:
            if len(self._command_cache) >= self._cache_size:
                oldest = next(iter(self._command_cache))
                del self._command_cache[oldest]
            self._command_cache[command] = result

        return result

    def _check_command_uncached(self, command: str) -> PermissionResult:
        """Check command permission without caching."""
        # Step 1: Check mandatory denies FIRST
        mandatory = self.check_command_mandatory_deny(command)
        if mandatory:
            return mandatory

        # Step 1.5: Check agent scope restrictions
        if self.agent_scope:
            action = self.agent_scope.check_command(command)
            if action == PermissionAction.DENY:
                return PermissionResult(
                    action=PermissionAction.DENY,
                    reason=f"Denied by agent scope '{self.agent_scope.name}': "
                           f"command not allowed",
                )

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
