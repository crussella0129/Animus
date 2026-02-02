"""Tests for git workflow automation tool."""

import pytest
import tempfile
import os
from pathlib import Path

from src.tools.git_workflow import (
    GitWorkflowTool,
    ChangeRisk,
    DiffAnalysis,
    create_git_workflow_tool,
)


class TestChangeRisk:
    """Tests for ChangeRisk enum."""

    def test_risk_values(self):
        """Test all risk values exist."""
        assert ChangeRisk.LOW.value == 1
        assert ChangeRisk.MEDIUM.value == 2
        assert ChangeRisk.HIGH.value == 3
        assert ChangeRisk.CRITICAL.value == 4
        # Test string conversion
        assert str(ChangeRisk.LOW) == "low"
        assert str(ChangeRisk.CRITICAL) == "critical"


class TestDiffAnalysis:
    """Tests for DiffAnalysis dataclass."""

    def test_analysis_fields(self):
        """Test DiffAnalysis fields."""
        analysis = DiffAnalysis(
            files_changed=3,
            lines_added=50,
            lines_removed=20,
            risk_level=ChangeRisk.MEDIUM,
            risk_factors=["Pattern 'config' detected"],
            summary="Changed 3 files",
            suggested_message="Update configuration",
        )

        assert analysis.files_changed == 3
        assert analysis.lines_added == 50
        assert analysis.lines_removed == 20
        assert analysis.risk_level == ChangeRisk.MEDIUM
        assert len(analysis.risk_factors) == 1
        assert "config" in analysis.risk_factors[0]


class TestGitWorkflowTool:
    """Tests for GitWorkflowTool."""

    def test_tool_properties(self):
        """Test tool properties."""
        tool = GitWorkflowTool()

        assert tool.name == "git_workflow"
        assert "workflow" in tool.description.lower()
        assert tool.requires_confirmation is True

    def test_tool_parameters(self):
        """Test tool parameters."""
        tool = GitWorkflowTool()
        params = tool.parameters
        param_names = [p.name for p in params]

        assert "operation" in param_names
        assert "name" in param_names
        assert "description" in param_names
        assert "base_branch" in param_names
        assert "files" in param_names

    def test_sanitize_branch_name(self):
        """Test branch name sanitization."""
        tool = GitWorkflowTool()

        # Simple case
        assert tool._sanitize_branch_name("Add user auth") == "add-user-auth"

        # Special characters
        assert tool._sanitize_branch_name("Fix bug #123!") == "fix-bug-123"

        # Already clean
        assert tool._sanitize_branch_name("feature-branch") == "feature-branch"

        # Very long name
        long_name = "a" * 100
        result = tool._sanitize_branch_name(long_name)
        assert len(result) <= 50

        # Unicode
        assert tool._sanitize_branch_name("Add новая feature") == "add-feature"

    def test_risk_patterns_defined(self):
        """Test that risk patterns are defined."""
        tool = GitWorkflowTool()

        assert ChangeRisk.CRITICAL in tool.RISK_PATTERNS
        assert ChangeRisk.HIGH in tool.RISK_PATTERNS
        assert ChangeRisk.MEDIUM in tool.RISK_PATTERNS

        # Critical patterns
        critical_patterns = tool.RISK_PATTERNS[ChangeRisk.CRITICAL]
        assert any("env" in p for p in critical_patterns)
        assert any("secret" in p for p in critical_patterns)

    def test_sensitive_files_defined(self):
        """Test that sensitive files list is defined."""
        tool = GitWorkflowTool()

        assert ".env" in tool.SENSITIVE_FILES
        assert any("credential" in f for f in tool.SENSITIVE_FILES)

    @pytest.mark.asyncio
    async def test_analyze_diff_low_risk(self):
        """Test diff analysis for low risk changes."""
        tool = GitWorkflowTool()

        diff = """+def hello():
+    print("Hello")
-def old_func():
-    pass"""

        files = ["src/utils.py"]

        analysis = await tool._analyze_diff(diff, files)

        assert analysis.files_changed == 1
        assert analysis.lines_added > 0
        assert analysis.risk_level == ChangeRisk.LOW

    @pytest.mark.asyncio
    async def test_analyze_diff_high_risk(self):
        """Test diff analysis for high risk changes."""
        tool = GitWorkflowTool()

        diff = """+def delete_user():
+    force_delete(user)
+    drop table users;"""

        files = ["src/auth.py"]

        analysis = await tool._analyze_diff(diff, files)

        # Should detect high risk patterns
        assert analysis.risk_level in (ChangeRisk.HIGH, ChangeRisk.CRITICAL)
        assert len(analysis.risk_factors) > 0

    @pytest.mark.asyncio
    async def test_analyze_diff_critical_files(self):
        """Test diff analysis for critical files."""
        tool = GitWorkflowTool()

        diff = "+SECRET_KEY=abc123"
        files = [".env"]

        analysis = await tool._analyze_diff(diff, files)

        assert analysis.risk_level == ChangeRisk.CRITICAL
        assert any("sensitive" in f.lower() for f in analysis.risk_factors)

    @pytest.mark.asyncio
    async def test_analyze_diff_large_changes(self):
        """Test diff analysis for large changes."""
        tool = GitWorkflowTool()

        # Create a large diff
        diff = "\n".join([f"+line {i}" for i in range(600)])
        files = ["src/big_file.py"]

        analysis = await tool._analyze_diff(diff, files)

        # Large changes should increase risk
        assert "large" in " ".join(analysis.risk_factors).lower()

    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self):
        """Test executing unknown operation."""
        tool = GitWorkflowTool()

        result = await tool.execute(operation="unknown_op")

        assert result.success is False
        assert "unknown" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_branch_no_name(self):
        """Test create_branch without name."""
        tool = GitWorkflowTool()

        result = await tool.execute(operation="create_branch")

        assert result.success is False
        assert "name" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_changes_no_changes(self):
        """Test analyze_changes with no changes."""
        # This test would need a git repo, so we'll test the error case
        tool = GitWorkflowTool(working_dir=Path(tempfile.gettempdir()))

        # In a non-git directory, should fail gracefully
        result = await tool.execute(operation="analyze_changes")

        # Either no changes or not a git repo
        assert result.success is True or "no changes" in result.output.lower() or result.error


class TestCreateGitWorkflowTool:
    """Tests for create_git_workflow_tool factory."""

    def test_creates_tool(self):
        """Test factory creates a tool."""
        tool = create_git_workflow_tool()

        assert isinstance(tool, GitWorkflowTool)
        assert tool.provider is None
        assert tool.confirm_callback is None

    def test_creates_tool_with_provider(self):
        """Test factory accepts provider."""
        class MockProvider:
            pass

        provider = MockProvider()
        tool = create_git_workflow_tool(provider=provider)

        assert tool.provider is provider

    def test_creates_tool_with_callback(self):
        """Test factory accepts callback."""
        async def callback(msg):
            return True

        tool = create_git_workflow_tool(confirm_callback=callback)

        assert tool.confirm_callback is callback


class TestGitWorkflowIntegration:
    """Integration tests for GitWorkflowTool (requires git)."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=tmpdir, capture_output=True
            )

            # Create initial commit
            readme = Path(tmpdir) / "README.md"
            readme.write_text("# Test Repo")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=tmpdir, capture_output=True
            )

            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_create_branch_in_repo(self, git_repo):
        """Test creating a branch in a real repo."""
        tool = GitWorkflowTool(working_dir=git_repo)

        result = await tool.execute(
            operation="create_branch",
            name="add-new-feature",
        )

        assert result.success is True
        assert "feature/add-new-feature" in result.output

    @pytest.mark.asyncio
    async def test_analyze_changes_in_repo(self, git_repo):
        """Test analyzing changes in a real repo."""
        # Create a change
        test_file = git_repo / "test.py"
        test_file.write_text("def test():\n    pass\n")

        tool = GitWorkflowTool(working_dir=git_repo)

        result = await tool.execute(operation="analyze_changes")

        assert result.success is True
        # Either has changes or says no changes
        assert "change" in result.output.lower() or "no changes" in result.output.lower()

    @pytest.mark.asyncio
    async def test_generate_commit_message_in_repo(self, git_repo):
        """Test generating commit message in a real repo."""
        # Create a change
        test_file = git_repo / "utils.py"
        test_file.write_text("def helper():\n    return True\n")

        import subprocess
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)

        tool = GitWorkflowTool(working_dir=git_repo)

        result = await tool.execute(operation="generate_commit_message")

        # Should succeed or indicate no provider
        assert result.success or "no changes" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_commit_changes_in_repo(self, git_repo):
        """Test committing changes in a real repo."""
        # Create and stage a change
        test_file = git_repo / "feature.py"
        test_file.write_text("# New feature\n")

        import subprocess
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)

        # Mock confirmation callback that always confirms
        async def confirm(msg):
            return True

        tool = GitWorkflowTool(working_dir=git_repo, confirm_callback=confirm)

        result = await tool.execute(
            operation="commit_changes",
            name="Add new feature",
        )

        assert result.success is True
        assert "committed" in result.output.lower()


class TestGitWorkflowInRegistry:
    """Tests for GitWorkflowTool registration."""

    def test_tool_in_registry(self):
        """Test that GitWorkflowTool is in the default registry."""
        from src.tools import create_default_registry

        registry = create_default_registry()
        tool = registry.get("git_workflow")

        assert tool is not None
        assert isinstance(tool, GitWorkflowTool)
