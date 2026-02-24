"""Tests for SessionCwd, shell CWD capture, and git repo safety guard."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from src.core.cwd import SessionCwd
from src.tools.git import GitAddTool, GitCommitTool, GitStatusTool, _check_git_repo
from src.tools.shell import RunShellTool, _normalize_quotes_for_windows, _UNQUOTED_PATH_RE


class TestSessionCwd:
    def test_default_init(self):
        cwd = SessionCwd()
        assert cwd.path == Path(os.getcwd()).resolve()

    def test_explicit_init(self, tmp_path: Path):
        cwd = SessionCwd(tmp_path)
        assert cwd.path == tmp_path.resolve()

    def test_set_absolute(self, tmp_path: Path):
        cwd = SessionCwd()
        cwd.set(tmp_path)
        assert cwd.path == tmp_path.resolve()

    def test_set_relative(self, tmp_path: Path):
        sub = tmp_path / "child"
        sub.mkdir()
        cwd = SessionCwd(tmp_path)
        cwd.set("child")
        assert cwd.path == sub.resolve()

    def test_set_nonexistent_ignored(self, tmp_path: Path):
        cwd = SessionCwd(tmp_path)
        original = cwd.path
        cwd.set(tmp_path / "does_not_exist")
        assert cwd.path == original

    def test_resolve_absolute(self, tmp_path: Path):
        cwd = SessionCwd(tmp_path)
        abs_path = Path("/some/absolute/path")
        resolved = cwd.resolve(abs_path)
        assert resolved == abs_path.resolve()

    def test_resolve_relative(self, tmp_path: Path):
        cwd = SessionCwd(tmp_path)
        resolved = cwd.resolve("file.txt")
        assert resolved == (tmp_path / "file.txt").resolve()


class TestShellCwdCapture:
    """Test that cd updates the session CWD and markers are stripped."""

    def test_cd_updates_session_cwd(self, tmp_path: Path):
        sub = tmp_path / "target"
        sub.mkdir()
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        tool.execute({"command": f"cd \"{sub}\""})
        assert session_cwd.path == sub.resolve()

    def test_no_cd_does_not_change_cwd(self, tmp_path: Path):
        session_cwd = SessionCwd(tmp_path)
        original = session_cwd.path
        tool = RunShellTool(session_cwd=session_cwd)
        tool.execute({"command": "echo hello"})
        assert session_cwd.path == original

    def test_cd_output_has_no_markers(self, tmp_path: Path):
        """cd is handled directly by SessionCwd; no shell markers in output."""
        sub = tmp_path / "marker_test"
        sub.mkdir()
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        result = tool.execute({"command": f'cd "{sub}"'})
        assert "__ANIMUS_CWD__" not in result
        assert "Changed directory" in result

    def test_shell_uses_session_cwd(self, tmp_path: Path):
        """Shell commands should run in the session CWD, not the process CWD."""
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        if os.name == "nt":
            # Use 'cd' via cmd /c to print CWD (not the cd handler)
            result = tool.execute({"command": "echo %CD%"})
        else:
            result = tool.execute({"command": "pwd"})
        assert str(tmp_path.resolve()).lower() in result.lower()

    def test_separate_mkdir_then_cd(self, tmp_path: Path):
        """mkdir then cd as separate commands should update session CWD."""
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        # Step 1: mkdir
        tool.execute({"command": "mkdir new_dir"})
        # Step 2: cd
        tool.execute({"command": "cd new_dir"})
        expected = (tmp_path / "new_dir").resolve()
        assert session_cwd.path == expected

    def test_without_session_cwd_no_error(self):
        """When session_cwd is None, shell still works (backward compat)."""
        tool = RunShellTool()
        result = tool.execute({"command": "echo compat"})
        assert "compat" in result


class TestQuoteNormalization:
    """Test single-to-double quote conversion for Windows cmd.exe."""

    def test_simple_single_quotes(self):
        assert _normalize_quotes_for_windows("mkdir 'test 1'") == 'mkdir "test 1"'

    def test_multiple_single_quoted_args(self):
        result = _normalize_quotes_for_windows("mkdir 'foo bar' && cd 'foo bar'")
        assert result == 'mkdir "foo bar" && cd "foo bar"'

    def test_no_quotes_unchanged(self):
        cmd = "echo hello world"
        assert _normalize_quotes_for_windows(cmd) == cmd

    def test_double_quotes_unchanged(self):
        cmd = 'mkdir "test 1"'
        assert _normalize_quotes_for_windows(cmd) == cmd

    def test_contraction_not_converted(self):
        """Words like don't should not be mangled."""
        cmd = "echo don't break this"
        assert _normalize_quotes_for_windows(cmd) == cmd

    def test_empty_single_quotes(self):
        assert _normalize_quotes_for_windows("echo ''") == 'echo ""'

    def test_path_with_spaces(self):
        cmd = "cd 'C:\\Users\\charl\\My Documents'"
        assert _normalize_quotes_for_windows(cmd) == 'cd "C:\\Users\\charl\\My Documents"'

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only integration test")
    def test_mkdir_with_spaces_via_tool(self, tmp_path: Path):
        """End-to-end: mkdir 'dir name' should create a single directory on Windows."""
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        tool.execute({"command": "mkdir 'test folder'"})
        expected = tmp_path / "test folder"
        assert expected.exists(), f"Expected '{expected}' to exist but it doesn't"
        # Make sure it didn't create 'test and folder' separately
        assert not (tmp_path / "'test").exists()
        assert not (tmp_path / "folder'").exists()

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only integration test")
    def test_mkdir_and_cd_with_spaces_via_tool(self, tmp_path: Path):
        """mkdir + cd with single-quoted spaced name should work as separate calls."""
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        tool.execute({"command": "mkdir 'my project'"})
        tool.execute({"command": "cd 'my project'"})
        expected = (tmp_path / "my project").resolve()
        assert session_cwd.path == expected


class TestUnquotedPathNormalization:
    """Test auto-quoting of unquoted paths with spaces for Windows commands."""

    def test_mkdir_unquoted_absolute_path(self):
        cmd = r"mkdir C:\Users\charl\Downloads\test 1"
        result = _normalize_quotes_for_windows(cmd)
        assert result == r'mkdir "C:\Users\charl\Downloads\test 1"'

    def test_cd_unquoted_absolute_path(self):
        cmd = r"cd C:\Users\charl\My Documents"
        result = _normalize_quotes_for_windows(cmd)
        assert result == r'cd "C:\Users\charl\My Documents"'

    def test_no_spaces_unchanged(self):
        cmd = r"mkdir C:\Users\charl\Downloads\test1"
        assert _normalize_quotes_for_windows(cmd) == cmd

    def test_already_quoted_unchanged(self):
        cmd = r'mkdir "C:\Users\charl\Downloads\test 1"'
        assert _normalize_quotes_for_windows(cmd) == cmd

    def test_chained_commands(self):
        cmd = r"mkdir C:\Users\charl\Downloads\test 1 && cd C:\Users\charl\Downloads\test 1"
        result = _normalize_quotes_for_windows(cmd)
        assert r'"C:\Users\charl\Downloads\test 1"' in result

    def test_unix_path_with_spaces(self):
        cmd = "mkdir /home/user/my folder"
        result = _normalize_quotes_for_windows(cmd)
        assert result == 'mkdir "/home/user/my folder"'

    def test_non_path_command_unchanged(self):
        """Commands not in _PATH_COMMANDS should not be affected."""
        cmd = "echo hello world"
        assert _normalize_quotes_for_windows(cmd) == cmd

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only integration test")
    def test_mkdir_unquoted_via_tool(self, tmp_path: Path):
        """End-to-end: mkdir with unquoted spaced path should create one directory."""
        session_cwd = SessionCwd(tmp_path)
        tool = RunShellTool(session_cwd=session_cwd)
        target = tmp_path / "test folder"
        tool.execute({"command": f"mkdir {target}"})
        assert target.exists(), f"Expected '{target}' to exist but it doesn't"
        # Make sure it didn't split on space
        assert not (tmp_path / "test").exists()


class TestGitRepoGuard:
    """Test the safety guard that prevents git operations on unintended repos."""

    def test_no_session_cwd_allows_all(self):
        """When session_cwd is None, no guard is applied (backward compat)."""
        assert _check_git_repo(None) is None

    def test_no_git_repo_blocks(self, tmp_path: Path):
        """Mutating ops blocked when no .git exists anywhere above session CWD."""
        session_cwd = SessionCwd(tmp_path)
        result = _check_git_repo(session_cwd)
        assert result is not None
        assert "No git repository" in result

    def test_git_repo_at_cwd_allows(self, tmp_path: Path):
        """Operations allowed when .git is directly in the session CWD."""
        (tmp_path / ".git").mkdir()
        session_cwd = SessionCwd(tmp_path)
        assert _check_git_repo(session_cwd) is None

    def test_deeply_inherited_repo_blocks(self, tmp_path: Path):
        """Block when .git is many levels above the session CWD."""
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        session_cwd = SessionCwd(deep)
        result = _check_git_repo(session_cwd)
        assert result is not None
        assert "unintended" in result.lower() or "levels deep" in result.lower()

    def test_git_add_blocks_without_repo(self, tmp_path: Path):
        """git_add should fail when session CWD has no git repo."""
        session_cwd = SessionCwd(tmp_path)
        tool = GitAddTool(session_cwd=session_cwd)
        result = tool.execute({"paths": ["file.txt"]})
        assert "Error" in result
        assert "No git repository" in result

    def test_git_commit_blocks_without_repo(self, tmp_path: Path):
        """git_commit should fail when session CWD has no git repo."""
        session_cwd = SessionCwd(tmp_path)
        tool = GitCommitTool(session_cwd=session_cwd)
        result = tool.execute({"message": "test"})
        assert "Error" in result
        assert "No git repository" in result

    def test_git_status_allowed_without_repo(self, tmp_path: Path):
        """git_status (read-only) should not be blocked by the guard."""
        session_cwd = SessionCwd(tmp_path)
        tool = GitStatusTool(session_cwd=session_cwd)
        # git_status doesn't call _check_git_repo, so it runs git directly
        # which will return its own error — the point is it doesn't get blocked
        result = tool.execute({})
        # It should NOT say "No git repository found at" from our guard
        assert "No git repository found at" not in result
