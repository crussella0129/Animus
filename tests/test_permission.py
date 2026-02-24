"""Tests for permission checking."""

from __future__ import annotations

import platform
from pathlib import Path, PurePosixPath

import pytest

from src.core.permission import PermissionChecker


class TestPermissionChecker:
    def test_safe_path(self, tmp_path: Path):
        checker = PermissionChecker()
        assert checker.is_path_safe(tmp_path / "safe_file.txt") is True

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix paths don't resolve on Windows")
    def test_dangerous_directory_unix(self):
        checker = PermissionChecker()
        assert checker.is_path_safe(Path("/etc/config")) is False
        assert checker.is_path_safe(Path("/boot/grub")) is False

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix paths don't resolve on Windows")
    def test_dangerous_file(self):
        checker = PermissionChecker()
        assert checker.is_path_safe(Path("/etc/shadow")) is False
        assert checker.is_path_safe(Path("/home/user/.ssh/id_rsa")) is False

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific path test")
    def test_dangerous_directory_windows(self):
        checker = PermissionChecker()
        assert checker.is_path_safe(Path("C:\\Windows\\System32\\config")) is False
        assert checker.is_path_safe(Path("C:\\Program Files\\test")) is False

    def test_blocked_command(self):
        checker = PermissionChecker()
        assert checker.is_command_blocked("rm -rf /") is not None
        assert checker.is_command_blocked("echo hello") is None

    def test_dangerous_command(self):
        checker = PermissionChecker()
        assert checker.is_command_dangerous("rm tempfile") is True
        assert checker.is_command_dangerous("sudo apt install") is True
        assert checker.is_command_dangerous("echo hello") is False

    def test_fork_bomb_blocked(self):
        checker = PermissionChecker()
        result = checker.is_command_blocked(":(){ :|:& };:")
        assert result is not None

    def test_network_command_curl(self):
        checker = PermissionChecker()
        assert checker.is_command_network("curl https://example.com") == "curl"

    def test_network_command_wget(self):
        checker = PermissionChecker()
        assert checker.is_command_network("wget https://example.com/file") == "wget"

    def test_network_command_git_push(self):
        checker = PermissionChecker()
        assert checker.is_command_network("git push origin main") == "git push"

    def test_network_command_git_clone(self):
        checker = PermissionChecker()
        assert checker.is_command_network("git clone https://github.com/user/repo") == "git clone"

    def test_network_command_ssh(self):
        checker = PermissionChecker()
        assert checker.is_command_network("ssh user@host") == "ssh"

    def test_non_network_command(self):
        checker = PermissionChecker()
        assert checker.is_command_network("echo hello") is None
        assert checker.is_command_network("ls -la") is None
        assert checker.is_command_network("git status") is None
        assert checker.is_command_network("git add .") is None
        assert checker.is_command_network("git commit -m 'test'") is None

    def test_animus_config_protected(self):
        checker = PermissionChecker()
        assert checker.is_path_safe(Path.home() / ".animus" / "config.yaml") is False


class TestFullCommandParsing:
    """has_injection_pattern should detect shell injection patterns."""

    def test_command_substitution_dollar_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo $(rm -rf /)")

    def test_command_substitution_backtick_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo `whoami`")

    def test_semicolon_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo safe; rm -rf /")

    def test_and_chain_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo safe && rm -rf /")

    def test_or_chain_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("false || rm -rf /")

    def test_clean_command_no_injection(self):
        checker = PermissionChecker()
        assert not checker.has_injection_pattern("echo hello world")

    def test_clean_path_no_injection(self):
        checker = PermissionChecker()
        assert not checker.has_injection_pattern("mkdir my_project")


class TestSymlinkSafePaths:
    """is_path_safe should resolve symlinks before checking."""

    def test_symlink_to_dangerous_dir_blocked(self, tmp_path):
        import os
        # Create symlink pointing to /etc (dangerous)
        link = tmp_path / "safe_looking"
        try:
            link.symlink_to("/etc")
            checker = PermissionChecker()
            # The symlink resolves to /etc which is dangerous
            assert not checker.is_path_safe(link / "passwd")
        except OSError:
            pytest.skip("Cannot create symlinks (no privileges)")

    def test_normal_path_still_works(self, tmp_path):
        checker = PermissionChecker()
        safe_file = tmp_path / "safe.txt"
        assert checker.is_path_safe(safe_file)
