"""Tests for git tools — all via subprocess mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.tools.base import ToolRegistry
from src.tools.git import (
    GitAddTool,
    GitBranchTool,
    GitCheckoutTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    register_git_tools,
)


def _mock_run(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Create a mock subprocess.run result."""
    mock = MagicMock()
    mock.stdout = stdout
    mock.stderr = stderr
    mock.returncode = returncode
    return mock


class TestGitStatusTool:
    @patch("src.tools.git.subprocess.run")
    def test_status(self, mock_run):
        mock_run.return_value = _mock_run(stdout=" M src/main.py\n?? new_file.txt")
        tool = GitStatusTool()
        result = tool.execute({})
        assert "main.py" in result
        assert "new_file.txt" in result
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["git", "status", "--porcelain"]

    @patch("src.tools.git.subprocess.run")
    def test_status_clean(self, mock_run):
        mock_run.return_value = _mock_run(stdout="")
        tool = GitStatusTool()
        result = tool.execute({})
        assert result == "(no output)"


class TestGitDiffTool:
    @patch("src.tools.git.subprocess.run")
    def test_diff_unstaged(self, mock_run):
        mock_run.return_value = _mock_run(stdout="diff --git a/file.py b/file.py\n+new line")
        tool = GitDiffTool()
        result = tool.execute({})
        assert "new line" in result
        assert mock_run.call_args[0][0] == ["git", "diff"]

    @patch("src.tools.git.subprocess.run")
    def test_diff_staged(self, mock_run):
        mock_run.return_value = _mock_run(stdout="staged changes")
        tool = GitDiffTool()
        result = tool.execute({"staged": True})
        assert "staged changes" in result
        assert "--staged" in mock_run.call_args[0][0]

    @patch("src.tools.git.subprocess.run")
    def test_diff_specific_path(self, mock_run):
        mock_run.return_value = _mock_run(stdout="path diff")
        tool = GitDiffTool()
        result = tool.execute({"path": "src/main.py"})
        assert "path diff" in result
        args = mock_run.call_args[0][0]
        assert "--" in args
        assert "src/main.py" in args


class TestGitLogTool:
    @patch("src.tools.git.subprocess.run")
    def test_log_default(self, mock_run):
        mock_run.return_value = _mock_run(stdout="abc123 Initial commit\ndef456 Second commit")
        tool = GitLogTool()
        result = tool.execute({})
        assert "Initial commit" in result
        assert "-10" in mock_run.call_args[0][0]

    @patch("src.tools.git.subprocess.run")
    def test_log_custom_count(self, mock_run):
        mock_run.return_value = _mock_run(stdout="abc123 Commit")
        tool = GitLogTool()
        tool.execute({"count": 5})
        assert "-5" in mock_run.call_args[0][0]


class TestGitBranchTool:
    @patch("src.tools.git.subprocess.run")
    def test_list_branches(self, mock_run):
        mock_run.return_value = _mock_run(stdout="* main\n  feature/test")
        tool = GitBranchTool()
        result = tool.execute({})
        assert "main" in result
        assert "feature/test" in result

    @patch("src.tools.git.subprocess.run")
    def test_create_branch(self, mock_run):
        mock_run.return_value = _mock_run(stdout="")
        tool = GitBranchTool()
        tool.execute({"create": "new-branch"})
        assert mock_run.call_args[0][0] == ["git", "branch", "new-branch"]

    @patch("src.tools.git.subprocess.run")
    def test_create_branch_with_confirm_allow(self, mock_run):
        mock_run.return_value = _mock_run(stdout="")
        tool = GitBranchTool(confirm_callback=lambda msg: True)
        tool.execute({"create": "new-branch"})
        mock_run.assert_called_once()

    def test_create_branch_with_confirm_deny(self):
        tool = GitBranchTool(confirm_callback=lambda msg: False)
        result = tool.execute({"create": "new-branch"})
        assert "cancelled" in result.lower()


class TestGitAddTool:
    @patch("src.tools.git.subprocess.run")
    def test_add_files(self, mock_run):
        mock_run.return_value = _mock_run(stdout="")
        tool = GitAddTool()
        tool.execute({"paths": ["file1.py", "file2.py"]})
        args = mock_run.call_args[0][0]
        assert "file1.py" in args
        assert "file2.py" in args

    def test_add_no_paths(self):
        tool = GitAddTool()
        result = tool.execute({"paths": []})
        assert "Error" in result


class TestGitCommitTool:
    @patch("src.tools.git.subprocess.run")
    def test_commit_with_confirm(self, mock_run):
        mock_run.return_value = _mock_run(stdout="[main abc123] Fix bug")
        tool = GitCommitTool(confirm_callback=lambda msg: True)
        result = tool.execute({"message": "Fix bug"})
        assert "Fix bug" in result
        args = mock_run.call_args[0][0]
        assert args == ["git", "commit", "-m", "Fix bug"]

    def test_commit_denied(self):
        tool = GitCommitTool(confirm_callback=lambda msg: False)
        result = tool.execute({"message": "Fix bug"})
        assert "cancelled" in result.lower()

    def test_commit_empty_message(self):
        tool = GitCommitTool()
        result = tool.execute({"message": ""})
        assert "Error" in result


class TestGitCheckoutTool:
    @patch("src.tools.git.subprocess.run")
    def test_checkout_with_confirm(self, mock_run):
        mock_run.return_value = _mock_run(stdout="Switched to branch 'main'")
        tool = GitCheckoutTool(confirm_callback=lambda msg: True)
        result = tool.execute({"target": "main"})
        assert "Switched" in result

    def test_checkout_denied(self):
        tool = GitCheckoutTool(confirm_callback=lambda msg: False)
        result = tool.execute({"target": "main"})
        assert "cancelled" in result.lower()

    def test_checkout_empty_target(self):
        tool = GitCheckoutTool()
        result = tool.execute({"target": ""})
        assert "Error" in result


class TestGitToolRegistration:
    def test_register_all_git_tools(self):
        registry = ToolRegistry()
        register_git_tools(registry)
        names = registry.names()
        assert "git_status" in names
        assert "git_diff" in names
        assert "git_log" in names
        assert "git_branch" in names
        assert "git_add" in names
        assert "git_commit" in names
        assert "git_checkout" in names

    def test_register_with_confirm_callback(self):
        registry = ToolRegistry()
        cb = lambda msg: True
        register_git_tools(registry, confirm_callback=cb)
        # Confirm tools needing confirmation got the callback
        branch_tool = registry.get("git_branch")
        commit_tool = registry.get("git_commit")
        checkout_tool = registry.get("git_checkout")
        assert branch_tool._confirm is cb
        assert commit_tool._confirm is cb
        assert checkout_tool._confirm is cb

    def test_openai_schemas(self):
        registry = ToolRegistry()
        register_git_tools(registry)
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 7
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema


class TestGitNotInstalled:
    @patch("src.tools.git.subprocess.run", side_effect=FileNotFoundError)
    def test_git_not_found(self, mock_run):
        tool = GitStatusTool()
        result = tool.execute({})
        assert "not installed" in result.lower()


class TestGitTimeout:
    @patch("src.tools.git.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
        tool = GitStatusTool()
        result = tool.execute({})
        assert "timed out" in result.lower()
