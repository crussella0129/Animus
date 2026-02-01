"""Three-tier permission system for Animus.

Permissions can be:
- "allow": Automatically allowed without confirmation
- "deny": Automatically denied without confirmation
- "ask": Requires user confirmation before proceeding

Pattern matching supports glob-style patterns for file paths and commands.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Awaitable


class PermissionAction(Enum):
    """Permission action types."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionCategory(Enum):
    """Categories of permissions."""
    READ = "read"                      # Reading files
    EDIT = "edit"                      # Modifying files
    CREATE = "create"                  # Creating new files
    DELETE = "delete"                  # Deleting files
    BASH = "bash"                      # Shell command execution
    EXTERNAL_DIRECTORY = "external_directory"  # Access outside project
    NETWORK = "network"                # Network operations
    TOOL = "tool"                      # Tool-specific permissions


@dataclass
class PermissionRule:
    """A single permission rule with pattern matching."""
    category: PermissionCategory
    pattern: str  # Glob pattern for matching
    action: PermissionAction
    description: Optional[str] = None
    priority: int = 0  # Higher priority rules are evaluated first

    def matches(self, target: str) -> bool:
        """Check if this rule matches the target.

        Args:
            target: The target to check (file path, command, etc.)

        Returns:
            True if the pattern matches
        """
        # Handle glob patterns
        if '*' in self.pattern or '?' in self.pattern:
            return fnmatch.fnmatch(target, self.pattern)

        # Handle regex patterns (prefixed with r:)
        if self.pattern.startswith("r:"):
            regex = self.pattern[2:]
            return bool(re.match(regex, target))

        # Exact match
        return target == self.pattern


@dataclass
class PermissionRuleset:
    """A collection of permission rules."""
    rules: list[PermissionRule] = field(default_factory=list)
    default_action: PermissionAction = PermissionAction.ASK

    def add_rule(self, rule: PermissionRule) -> None:
        """Add a rule to the ruleset."""
        self.rules.append(rule)
        # Keep rules sorted by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(self, category: PermissionCategory, target: str) -> PermissionAction:
        """Evaluate permission for a target.

        Args:
            category: Permission category
            target: The target to check

        Returns:
            The permission action
        """
        for rule in self.rules:
            if rule.category == category and rule.matches(target):
                return rule.action

        return self.default_action

    def get_rules(self, category: PermissionCategory) -> list[PermissionRule]:
        """Get all rules for a category."""
        return [r for r in self.rules if r.category == category]


@dataclass
class PermissionProfile:
    """A named permission profile with rulesets."""
    name: str
    description: str
    ruleset: PermissionRuleset

    @staticmethod
    def strict() -> "PermissionProfile":
        """Create a strict profile that asks for everything except reads."""
        ruleset = PermissionRuleset(default_action=PermissionAction.ASK)

        # Allow reading any file except secrets
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/*",
            action=PermissionAction.ALLOW,
            priority=0,
        ))
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/.env*",
            action=PermissionAction.DENY,
            description="Never read environment files with secrets",
            priority=10,
        ))
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/*credentials*",
            action=PermissionAction.DENY,
            priority=10,
        ))
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/*secret*",
            action=PermissionAction.DENY,
            priority=10,
        ))

        return PermissionProfile(
            name="strict",
            description="Ask for everything except reads. Deny access to secrets.",
            ruleset=ruleset,
        )

    @staticmethod
    def standard() -> "PermissionProfile":
        """Create a standard profile with balanced permissions."""
        ruleset = PermissionRuleset(default_action=PermissionAction.ASK)

        # Allow all reads except secrets
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/*",
            action=PermissionAction.ALLOW,
            priority=0,
        ))
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/.env*",
            action=PermissionAction.DENY,
            priority=10,
        ))

        # Allow safe shell commands
        safe_commands = [
            "ls", "dir", "cat", "type", "pwd", "echo",
            "git status", "git log", "git diff", "git branch",
            "python --version", "pip list", "node --version", "npm list",
        ]
        for cmd in safe_commands:
            ruleset.add_rule(PermissionRule(
                category=PermissionCategory.BASH,
                pattern=f"{cmd}*",
                action=PermissionAction.ALLOW,
                priority=0,
            ))

        # Block dangerous commands
        dangerous_patterns = [
            "rm -rf /*", "rm -rf /", "rm -rf ~",
            "del /s /q c:\\*", "format *:",
            ":(){:|:&};:", "dd if=/dev/*",
        ]
        for pattern in dangerous_patterns:
            ruleset.add_rule(PermissionRule(
                category=PermissionCategory.BASH,
                pattern=pattern,
                action=PermissionAction.DENY,
                description="Dangerous command blocked",
                priority=100,
            ))

        return PermissionProfile(
            name="standard",
            description="Allow reads and safe commands. Ask for edits and complex commands.",
            ruleset=ruleset,
        )

    @staticmethod
    def trusted() -> "PermissionProfile":
        """Create a trusted profile that auto-allows most operations."""
        ruleset = PermissionRuleset(default_action=PermissionAction.ALLOW)

        # Still deny secrets and dangerous commands
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.READ,
            pattern="**/.env*",
            action=PermissionAction.DENY,
            priority=10,
        ))

        dangerous_patterns = [
            "rm -rf /*", "rm -rf /", "rm -rf ~",
            "del /s /q c:\\*", "format *:",
        ]
        for pattern in dangerous_patterns:
            ruleset.add_rule(PermissionRule(
                category=PermissionCategory.BASH,
                pattern=pattern,
                action=PermissionAction.DENY,
                priority=100,
            ))

        # Ask for external directory access
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.EXTERNAL_DIRECTORY,
            pattern="**/*",
            action=PermissionAction.ASK,
            priority=0,
        ))

        return PermissionProfile(
            name="trusted",
            description="Allow most operations. Still deny dangerous commands and secrets.",
            ruleset=ruleset,
        )

    @staticmethod
    def yolo() -> "PermissionProfile":
        """Create a YOLO profile that allows everything (use with caution)."""
        ruleset = PermissionRuleset(default_action=PermissionAction.ALLOW)

        # Still block the most catastrophic commands
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.BASH,
            pattern="rm -rf /",
            action=PermissionAction.DENY,
            priority=100,
        ))
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.BASH,
            pattern="rm -rf /*",
            action=PermissionAction.DENY,
            priority=100,
        ))

        return PermissionProfile(
            name="yolo",
            description="Allow everything. Only blocks catastrophic system destruction.",
            ruleset=ruleset,
        )


# Type for permission callback
PermissionCallback = Callable[[str, str, str], Awaitable[bool]]


@dataclass
class PermissionRequest:
    """A request for permission."""
    category: PermissionCategory
    target: str  # What is being accessed (file path, command, etc.)
    action_description: str  # Human-readable description
    context: Optional[str] = None  # Additional context
    patterns_to_remember: list[str] = field(default_factory=list)  # Patterns for "always allow"


class PermissionManager:
    """Manages permission evaluation and user prompts."""

    def __init__(
        self,
        profile: Optional[PermissionProfile] = None,
        project_dir: Optional[Path] = None,
    ):
        """Initialize permission manager.

        Args:
            profile: Permission profile to use. Defaults to 'standard'.
            project_dir: Project directory for relative path evaluation.
        """
        self.profile = profile or PermissionProfile.standard()
        self.project_dir = project_dir or Path.cwd()

        # Session-level overrides (from "always allow/deny" responses)
        self._session_allows: set[str] = set()  # category:pattern
        self._session_denies: set[str] = set()

        # Callback for asking user
        self._ask_callback: Optional[PermissionCallback] = None

    def set_profile(self, profile: PermissionProfile) -> None:
        """Set the permission profile."""
        self.profile = profile

    def set_ask_callback(self, callback: PermissionCallback) -> None:
        """Set the callback for asking user permission.

        Callback signature: async (category, target, description) -> bool
        """
        self._ask_callback = callback

    def is_external_path(self, path: Path) -> bool:
        """Check if a path is outside the project directory."""
        try:
            path.resolve().relative_to(self.project_dir.resolve())
            return False
        except ValueError:
            return True

    async def check(self, request: PermissionRequest) -> bool:
        """Check if an action is permitted.

        Args:
            request: Permission request

        Returns:
            True if permitted, False if denied
        """
        category = request.category
        target = request.target
        key = f"{category.value}:{target}"

        # Check session-level overrides first
        for pattern in self._session_allows:
            cat, pat = pattern.split(":", 1)
            if cat == category.value and fnmatch.fnmatch(target, pat):
                return True

        for pattern in self._session_denies:
            cat, pat = pattern.split(":", 1)
            if cat == category.value and fnmatch.fnmatch(target, pat):
                return False

        # Evaluate against profile rules
        action = self.profile.ruleset.evaluate(category, target)

        if action == PermissionAction.ALLOW:
            return True
        elif action == PermissionAction.DENY:
            return False
        else:
            # ASK - need user confirmation
            if self._ask_callback:
                result = await self._ask_callback(
                    category.value,
                    target,
                    request.action_description,
                )
                return result
            else:
                # No callback, default to deny for safety
                return False

    def add_session_allow(self, category: PermissionCategory, pattern: str) -> None:
        """Add a session-level allow pattern."""
        self._session_allows.add(f"{category.value}:{pattern}")

    def add_session_deny(self, category: PermissionCategory, pattern: str) -> None:
        """Add a session-level deny pattern."""
        self._session_denies.add(f"{category.value}:{pattern}")

    def clear_session_overrides(self) -> None:
        """Clear all session-level overrides."""
        self._session_allows.clear()
        self._session_denies.clear()

    def get_profile_by_name(self, name: str) -> PermissionProfile:
        """Get a built-in profile by name."""
        profiles = {
            "strict": PermissionProfile.strict,
            "standard": PermissionProfile.standard,
            "trusted": PermissionProfile.trusted,
            "yolo": PermissionProfile.yolo,
        }
        if name not in profiles:
            raise ValueError(f"Unknown profile: {name}. Valid: {list(profiles.keys())}")
        return profiles[name]()


# Agent-specific permission profiles
def get_explore_agent_profile() -> PermissionProfile:
    """Get permission profile for explore agent (read-only)."""
    ruleset = PermissionRuleset(default_action=PermissionAction.DENY)

    # Allow reading
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.READ,
        pattern="**/*",
        action=PermissionAction.ALLOW,
    ))

    # Deny secrets
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.READ,
        pattern="**/.env*",
        action=PermissionAction.DENY,
        priority=10,
    ))

    # Deny all writes
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.EDIT,
        pattern="**/*",
        action=PermissionAction.DENY,
    ))
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.CREATE,
        pattern="**/*",
        action=PermissionAction.DENY,
    ))
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.DELETE,
        pattern="**/*",
        action=PermissionAction.DENY,
    ))

    # Allow read-only bash commands
    read_only_commands = ["ls", "dir", "cat", "type", "pwd", "git status", "git log"]
    for cmd in read_only_commands:
        ruleset.add_rule(PermissionRule(
            category=PermissionCategory.BASH,
            pattern=f"{cmd}*",
            action=PermissionAction.ALLOW,
        ))

    return PermissionProfile(
        name="explore",
        description="Read-only profile for exploration agents",
        ruleset=ruleset,
    )


def get_plan_agent_profile() -> PermissionProfile:
    """Get permission profile for plan agent (read + ask for bash)."""
    ruleset = PermissionRuleset(default_action=PermissionAction.ASK)

    # Allow reading
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.READ,
        pattern="**/*",
        action=PermissionAction.ALLOW,
    ))

    # Deny secrets
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.READ,
        pattern="**/.env*",
        action=PermissionAction.DENY,
        priority=10,
    ))

    # Deny all writes
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.EDIT,
        pattern="**/*",
        action=PermissionAction.DENY,
    ))
    ruleset.add_rule(PermissionRule(
        category=PermissionCategory.CREATE,
        pattern="**/*",
        action=PermissionAction.DENY,
    ))

    return PermissionProfile(
        name="plan",
        description="Read-only with bash confirmation for planning agents",
        ruleset=ruleset,
    )


def get_build_agent_profile() -> PermissionProfile:
    """Get permission profile for build agent (full access with standard safeguards)."""
    return PermissionProfile.standard()
