"""Automated git workflow tools for seamless branch/commit/PR management."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any, Optional, Callable, Awaitable, TYPE_CHECKING
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory

if TYPE_CHECKING:
    from src.llm.base import ModelProvider


class ChangeRisk(Enum):
    """Risk level for code changes."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class DiffAnalysis:
    """Analysis of a git diff."""
    files_changed: int
    lines_added: int
    lines_removed: int
    risk_level: ChangeRisk
    risk_factors: list[str]
    summary: str
    suggested_message: str


class GitWorkflowTool(Tool):
    """
    High-level git workflow automation tool.

    Provides:
    - Auto-create feature branches with proper naming
    - Generate commit messages from changes
    - Create PRs via gh CLI
    - Diff analysis with risk assessment
    """

    # Patterns that indicate higher risk changes
    RISK_PATTERNS = {
        ChangeRisk.CRITICAL: [
            r"\.env",
            r"secrets?\.",
            r"credentials?",
            r"api[_-]?key",
            r"password",
            r"private[_-]?key",
            r"token",
        ],
        ChangeRisk.HIGH: [
            r"auth",
            r"security",
            r"permission",
            r"delete",
            r"drop\s+table",
            r"rm\s+-rf",
            r"force",
            r"--hard",
            r"migrations?/",
            r"schema",
        ],
        ChangeRisk.MEDIUM: [
            r"config",
            r"settings?",
            r"database",
            r"model",
            r"api",
            r"test",
        ],
    }

    # File patterns that are typically sensitive
    SENSITIVE_FILES = [
        ".env",
        ".env.local",
        ".env.production",
        "secrets.yaml",
        "secrets.json",
        "credentials.json",
        "private.key",
        "id_rsa",
        "id_ed25519",
        ".aws/credentials",
    ]

    def __init__(
        self,
        provider: Optional["ModelProvider"] = None,
        confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        working_dir: Optional[Path] = None,
        timeout: int = 120,
    ):
        """
        Initialize the git workflow tool.

        Args:
            provider: LLM provider for generating commit messages.
            confirm_callback: Async callback for confirmation prompts.
            working_dir: Working directory for git commands.
            timeout: Command timeout in seconds.
        """
        self.provider = provider
        self.confirm_callback = confirm_callback
        self.working_dir = working_dir
        self.timeout = timeout

    def set_provider(self, provider: "ModelProvider") -> None:
        """Set the LLM provider."""
        self.provider = provider

    @property
    def name(self) -> str:
        return "git_workflow"

    @property
    def description(self) -> str:
        return (
            "Automated git workflow operations: create feature branches, "
            "generate commit messages from changes, create PRs via gh CLI, "
            "and analyze diffs for risk assessment."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type="string",
                description=(
                    "Workflow operation: create_branch, analyze_changes, "
                    "generate_commit_message, commit_changes, create_pr, "
                    "full_workflow (branch + commit + pr)."
                ),
            ),
            ToolParameter(
                name="name",
                type="string",
                description="Branch name, PR title, or feature description.",
                required=False,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Detailed description for PR body or commit.",
                required=False,
            ),
            ToolParameter(
                name="base_branch",
                type="string",
                description="Base branch for PR (default: main).",
                required=False,
            ),
            ToolParameter(
                name="files",
                type="string",
                description="Comma-separated files to include (default: all staged).",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def requires_confirmation(self) -> bool:
        return True

    async def _run_cmd(
        self,
        cmd: list[str],
        cwd: Optional[Path] = None,
    ) -> tuple[bool, str, str]:
        """Run a command and return (success, stdout, stderr)."""
        working_dir = cwd or self.working_dir

        try:
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

    async def _run_git(self, args: list[str]) -> tuple[bool, str, str]:
        """Run a git command."""
        return await self._run_cmd(["git"] + args)

    async def _run_gh(self, args: list[str]) -> tuple[bool, str, str]:
        """Run a gh CLI command."""
        return await self._run_cmd(["gh"] + args)

    async def _get_confirmation(self, message: str) -> bool:
        """Get confirmation for an operation."""
        if self.confirm_callback:
            return await self.confirm_callback(message)
        return True  # Default to yes if no callback

    def _sanitize_branch_name(self, name: str) -> str:
        """Convert a description to a valid branch name."""
        # Convert to lowercase
        name = name.lower()
        # Replace spaces and special chars with hyphens
        name = re.sub(r'[^a-z0-9]+', '-', name)
        # Remove leading/trailing hyphens
        name = name.strip('-')
        # Limit length
        if len(name) > 50:
            name = name[:50].rstrip('-')
        return name

    async def _analyze_diff(self, diff_content: str, files_changed: list[str]) -> DiffAnalysis:
        """Analyze a diff for risk and generate summary."""
        # Count lines
        lines_added = len(re.findall(r'^\+[^+]', diff_content, re.MULTILINE))
        lines_removed = len(re.findall(r'^-[^-]', diff_content, re.MULTILINE))

        # Analyze risk
        risk_factors = []
        risk_level = ChangeRisk.LOW

        # Check for sensitive file patterns
        for f in files_changed:
            f_lower = f.lower()
            for sensitive in self.SENSITIVE_FILES:
                if sensitive in f_lower:
                    risk_factors.append(f"Sensitive file modified: {f}")
                    risk_level = ChangeRisk.CRITICAL
                    break

        # Check diff content for risk patterns
        diff_lower = diff_content.lower()
        for level, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, diff_lower):
                    risk_factors.append(f"Pattern '{pattern}' detected")
                    if level.value > risk_level.value:
                        risk_level = level

        # Adjust risk based on change size
        total_changes = lines_added + lines_removed
        if total_changes > 500:
            risk_factors.append(f"Large change: {total_changes} lines")
            if risk_level == ChangeRisk.LOW:
                risk_level = ChangeRisk.MEDIUM
        elif total_changes > 1000:
            risk_factors.append(f"Very large change: {total_changes} lines")
            if risk_level in (ChangeRisk.LOW, ChangeRisk.MEDIUM):
                risk_level = ChangeRisk.HIGH

        # Generate summary
        file_types = set()
        for f in files_changed:
            ext = Path(f).suffix
            if ext:
                file_types.add(ext)

        summary = f"Changed {len(files_changed)} file(s) ({', '.join(file_types) if file_types else 'no extension'}): +{lines_added}/-{lines_removed} lines"

        # Generate suggested commit message
        if len(files_changed) == 1:
            action = "Update" if lines_added > 0 and lines_removed > 0 else "Add" if lines_added > lines_removed else "Remove"
            suggested_message = f"{action} {files_changed[0]}"
        else:
            # Detect common patterns
            test_files = [f for f in files_changed if "test" in f.lower()]
            src_files = [f for f in files_changed if "src" in f.lower() or "lib" in f.lower()]

            if len(test_files) > 0 and len(test_files) == len(files_changed):
                suggested_message = f"Add/update tests ({len(test_files)} files)"
            elif len(src_files) > 0:
                suggested_message = f"Update source files ({len(files_changed)} files)"
            else:
                suggested_message = f"Update {len(files_changed)} files"

        return DiffAnalysis(
            files_changed=len(files_changed),
            lines_added=lines_added,
            lines_removed=lines_removed,
            risk_level=risk_level,
            risk_factors=risk_factors,
            summary=summary,
            suggested_message=suggested_message,
        )

    async def _generate_commit_message_llm(self, diff: str, description: str = "") -> str:
        """Generate a commit message using LLM."""
        if not self.provider:
            return ""

        from src.llm.base import Message, GenerationConfig

        prompt = f"""Generate a concise, conventional commit message for the following changes.

Rules:
- Use conventional commit format: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore
- Keep first line under 72 characters
- Be specific about what changed
- Don't include the diff in the message

{f'Context: {description}' if description else ''}

Diff:
{diff[:3000]}  # Truncate long diffs

Output only the commit message, nothing else."""

        try:
            result = await self.provider.generate(
                messages=[Message(role="user", content=prompt)],
                config=GenerationConfig(temperature=0.3, max_tokens=150),
            )
            # Extract just the commit message (first line if multiple)
            message = result.content.strip().split('\n')[0]
            # Remove any quotes if present
            message = message.strip('"\'')
            return message
        except Exception:
            return ""

    async def execute(
        self,
        operation: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        base_branch: str = "main",
        files: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a git workflow operation."""
        operation = operation.lower().strip()

        if operation == "create_branch":
            return await self._create_branch(name, base_branch)

        elif operation == "analyze_changes":
            return await self._analyze_changes(files)

        elif operation == "generate_commit_message":
            return await self._generate_commit_message(description)

        elif operation == "commit_changes":
            return await self._commit_changes(name, files)

        elif operation == "create_pr":
            return await self._create_pr(name, description, base_branch)

        elif operation == "full_workflow":
            return await self._full_workflow(name, description, base_branch, files)

        else:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unknown operation: {operation}. "
                    "Use: create_branch, analyze_changes, generate_commit_message, "
                    "commit_changes, create_pr, or full_workflow."
                ),
            )

    async def _create_branch(
        self,
        name: Optional[str],
        base_branch: str,
    ) -> ToolResult:
        """Create a new feature branch."""
        if not name:
            return ToolResult(
                success=False,
                output="",
                error="Branch name or feature description required.",
            )

        # Sanitize branch name
        branch_name = self._sanitize_branch_name(name)
        if not branch_name:
            return ToolResult(
                success=False,
                output="",
                error="Could not generate valid branch name from description.",
            )

        # Add feature/ prefix if not present
        if not branch_name.startswith(("feature/", "fix/", "docs/", "refactor/", "test/")):
            branch_name = f"feature/{branch_name}"

        # Check if branch exists
        success, stdout, _ = await self._run_git(["branch", "--list", branch_name])
        if stdout.strip():
            return ToolResult(
                success=False,
                output="",
                error=f"Branch '{branch_name}' already exists.",
            )

        # Create and checkout the branch
        success, stdout, stderr = await self._run_git(["checkout", "-b", branch_name])
        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create branch: {stderr}",
            )

        return ToolResult(
            success=True,
            output=f"Created and switched to branch: {branch_name}",
            metadata={"branch_name": branch_name},
        )

    async def _analyze_changes(
        self,
        files: Optional[str],
    ) -> ToolResult:
        """Analyze staged/unstaged changes."""
        # Get diff
        if files:
            file_list = [f.strip() for f in files.split(",")]
            success, diff, stderr = await self._run_git(["diff", "--"] + file_list)
        else:
            # Get both staged and unstaged
            success1, staged, _ = await self._run_git(["diff", "--cached"])
            success2, unstaged, _ = await self._run_git(["diff"])
            diff = staged + "\n" + unstaged
            success = success1 or success2

        if not diff.strip():
            return ToolResult(
                success=True,
                output="No changes to analyze.",
                metadata={"files_changed": 0},
            )

        # Get list of changed files
        success, files_output, _ = await self._run_git(["diff", "--name-only", "HEAD"])
        files_changed = [f for f in files_output.strip().split("\n") if f]

        # Analyze
        analysis = await self._analyze_diff(diff, files_changed)

        # Format output
        output_parts = [
            f"=== Change Analysis ===",
            f"",
            f"Summary: {analysis.summary}",
            f"Risk Level: {str(analysis.risk_level).upper()}",
        ]

        if analysis.risk_factors:
            output_parts.append(f"")
            output_parts.append(f"Risk Factors:")
            for factor in analysis.risk_factors:
                output_parts.append(f"  - {factor}")

        output_parts.append(f"")
        output_parts.append(f"Files Changed:")
        for f in files_changed[:20]:
            output_parts.append(f"  - {f}")
        if len(files_changed) > 20:
            output_parts.append(f"  ... and {len(files_changed) - 20} more")

        output_parts.append(f"")
        output_parts.append(f"Suggested Commit: {analysis.suggested_message}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={
                "files_changed": analysis.files_changed,
                "lines_added": analysis.lines_added,
                "lines_removed": analysis.lines_removed,
                "risk_level": str(analysis.risk_level),
                "risk_factors": analysis.risk_factors,
                "suggested_message": analysis.suggested_message,
            },
        )

    async def _generate_commit_message(
        self,
        description: Optional[str],
    ) -> ToolResult:
        """Generate a commit message from changes."""
        # Get diff
        success1, staged, _ = await self._run_git(["diff", "--cached"])
        success2, unstaged, _ = await self._run_git(["diff"])
        diff = staged + "\n" + unstaged

        if not diff.strip():
            return ToolResult(
                success=False,
                output="",
                error="No changes to generate message for.",
            )

        # Get changed files
        success, files_output, _ = await self._run_git(["diff", "--name-only", "HEAD"])
        files_changed = [f for f in files_output.strip().split("\n") if f]

        # Try LLM first
        llm_message = await self._generate_commit_message_llm(diff, description or "")

        # Fall back to analysis-based message
        analysis = await self._analyze_diff(diff, files_changed)

        message = llm_message or analysis.suggested_message

        return ToolResult(
            success=True,
            output=f"Suggested commit message:\n\n{message}",
            metadata={
                "message": message,
                "llm_generated": bool(llm_message),
                "files_changed": files_changed,
            },
        )

    async def _commit_changes(
        self,
        message: Optional[str],
        files: Optional[str],
    ) -> ToolResult:
        """Commit changes with generated or provided message."""
        # Stage files if specified
        if files:
            file_list = [f.strip() for f in files.split(",")]
            success, _, stderr = await self._run_git(["add"] + file_list)
            if not success:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to stage files: {stderr}",
                )

        # Check for staged changes
        success, staged, _ = await self._run_git(["diff", "--cached", "--name-only"])
        if not staged.strip():
            return ToolResult(
                success=False,
                output="",
                error="No staged changes to commit. Stage files first with 'git add'.",
            )

        # Generate message if not provided
        if not message:
            result = await self._generate_commit_message(None)
            if result.success and result.metadata:
                message = result.metadata.get("message", "")

        if not message:
            return ToolResult(
                success=False,
                output="",
                error="Could not generate commit message. Please provide one.",
            )

        # Analyze for risk
        files_changed = [f for f in staged.strip().split("\n") if f]
        success, diff, _ = await self._run_git(["diff", "--cached"])
        analysis = await self._analyze_diff(diff, files_changed)

        # Confirm if high risk
        if analysis.risk_level in (ChangeRisk.HIGH, ChangeRisk.CRITICAL):
            confirm_msg = (
                f"⚠️ HIGH RISK COMMIT\n\n"
                f"Message: {message}\n"
                f"Risk Level: {str(analysis.risk_level).upper()}\n"
                f"Risk Factors: {', '.join(analysis.risk_factors)}\n\n"
                f"Proceed with commit?"
            )
            if not await self._get_confirmation(confirm_msg):
                return ToolResult(
                    success=False,
                    output="",
                    error="Commit cancelled due to high risk.",
                    metadata={"risk_level": analysis.risk_level.value},
                )

        # Commit
        success, stdout, stderr = await self._run_git(["commit", "-m", message])
        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Commit failed: {stderr}",
            )

        return ToolResult(
            success=True,
            output=f"Committed: {message}\n\n{stdout}",
            metadata={
                "message": message,
                "files_committed": files_changed,
                "risk_level": str(analysis.risk_level),
            },
        )

    async def _create_pr(
        self,
        title: Optional[str],
        body: Optional[str],
        base_branch: str,
    ) -> ToolResult:
        """Create a PR using gh CLI."""
        # Check if gh is available
        success, _, _ = await self._run_cmd(["gh", "--version"])
        if not success:
            return ToolResult(
                success=False,
                output="",
                error="GitHub CLI (gh) not installed. Install with: https://cli.github.com/",
            )

        # Get current branch
        success, current_branch, _ = await self._run_git(["branch", "--show-current"])
        current_branch = current_branch.strip()

        if current_branch in ("main", "master"):
            return ToolResult(
                success=False,
                output="",
                error="Cannot create PR from main/master branch. Create a feature branch first.",
            )

        # Generate title if not provided
        if not title:
            # Try to extract from branch name
            title = current_branch.replace("feature/", "").replace("fix/", "").replace("-", " ").title()

        # Generate body if not provided
        if not body:
            # Get commit messages since base branch
            success, commits, _ = await self._run_git([
                "log", f"{base_branch}..HEAD", "--oneline"
            ])
            if commits.strip():
                body = f"## Changes\n\n{commits.strip()}"
            else:
                body = "## Changes\n\n(No commits yet)"

        # Push branch first
        success, _, stderr = await self._run_git(["push", "-u", "origin", current_branch])
        if not success and "already exists" not in stderr.lower():
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to push branch: {stderr}",
            )

        # Create PR
        success, stdout, stderr = await self._run_gh([
            "pr", "create",
            "--title", title,
            "--body", body,
            "--base", base_branch,
        ])

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create PR: {stderr}",
            )

        # Extract PR URL from output
        pr_url = stdout.strip().split("\n")[-1] if stdout else ""

        return ToolResult(
            success=True,
            output=f"PR created: {pr_url}\n\nTitle: {title}",
            metadata={
                "pr_url": pr_url,
                "title": title,
                "base_branch": base_branch,
                "head_branch": current_branch,
            },
        )

    async def _full_workflow(
        self,
        name: Optional[str],
        description: Optional[str],
        base_branch: str,
        files: Optional[str],
    ) -> ToolResult:
        """Execute full workflow: branch -> commit -> PR."""
        if not name:
            return ToolResult(
                success=False,
                output="",
                error="Feature name/description required for full workflow.",
            )

        output_parts = ["=== Full Git Workflow ===\n"]

        # 1. Create branch
        branch_result = await self._create_branch(name, base_branch)
        if not branch_result.success:
            return ToolResult(
                success=False,
                output="\n".join(output_parts),
                error=f"Failed at branch creation: {branch_result.error}",
            )
        output_parts.append(f"✓ Branch: {branch_result.output}")

        # 2. Stage and commit changes (if any)
        success, diff, _ = await self._run_git(["diff", "--name-only"])
        success2, staged, _ = await self._run_git(["diff", "--cached", "--name-only"])

        if diff.strip() or staged.strip():
            # Stage all if no specific files
            if not files:
                await self._run_git(["add", "."])

            commit_result = await self._commit_changes(description, files)
            if commit_result.success:
                output_parts.append(f"✓ Commit: {commit_result.metadata.get('message', 'Committed')}")
            else:
                output_parts.append(f"! Commit skipped: {commit_result.error}")

        # 3. Create PR
        pr_result = await self._create_pr(name, description, base_branch)
        if pr_result.success:
            output_parts.append(f"✓ PR: {pr_result.metadata.get('pr_url', 'Created')}")
        else:
            output_parts.append(f"! PR creation: {pr_result.error}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={
                "branch": branch_result.metadata,
                "pr": pr_result.metadata if pr_result.success else None,
            },
        )


def create_git_workflow_tool(
    provider: Optional["ModelProvider"] = None,
    confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
) -> GitWorkflowTool:
    """Create a git workflow tool instance."""
    return GitWorkflowTool(provider=provider, confirm_callback=confirm_callback)
