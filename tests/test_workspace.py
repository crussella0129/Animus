"""Tests for Workspace boundary enforcement."""

from pathlib import Path

import pytest

from src.core.workspace import Workspace, WorkspaceBoundaryError


class TestWorkspaceInit:
    def test_root_is_resolved(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.root == tmp_path.resolve()

    def test_cwd_starts_at_root(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.cwd == ws.root

    def test_path_property_for_backward_compat(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.path == ws.cwd


class TestWorkspaceResolve:
    def test_resolve_relative_path(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        resolved = ws.resolve("subdir/file.txt")
        assert resolved == (tmp_path / "subdir" / "file.txt").resolve()

    def test_resolve_absolute_within_root(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        target = tmp_path / "inside.txt"
        resolved = ws.resolve(str(target))
        assert resolved == target.resolve()

    def test_resolve_outside_root_raises(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        with pytest.raises(WorkspaceBoundaryError):
            ws.resolve("/etc/passwd")

    def test_resolve_parent_traversal_raises(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        with pytest.raises(WorkspaceBoundaryError):
            ws.resolve("../../etc/passwd")

    def test_resolve_after_cwd_change(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set_cwd(subdir)
        resolved = ws.resolve("file.txt")
        assert resolved == (subdir / "file.txt").resolve()


class TestWorkspaceSetCwd:
    def test_set_cwd_within_root(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set_cwd(subdir)
        assert ws.cwd == subdir.resolve()

    def test_set_cwd_outside_root_raises(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        with pytest.raises(WorkspaceBoundaryError):
            ws.set_cwd(Path("/tmp") if Path("/tmp").exists() else Path("C:/Windows"))

    def test_set_cwd_relative(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set_cwd(Path("sub"))
        assert ws.cwd == subdir.resolve()

    def test_set_nonexistent_stays_unchanged(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        ws.set_cwd(tmp_path / "nonexistent")
        # Non-existent dirs don't change CWD
        assert ws.cwd == tmp_path.resolve()


class TestWorkspaceSetCompat:
    """Test .set() for backward compatibility with SessionCwd."""

    def test_set_valid_dir(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set(str(subdir))
        assert ws.cwd == subdir.resolve()

    def test_set_outside_root_silently_ignored(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        ws.set("/etc")
        assert ws.cwd == tmp_path.resolve()  # unchanged

    def test_set_nonexistent_silently_ignored(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        ws.set("/nonexistent/deep/path")
        assert ws.cwd == tmp_path.resolve()  # unchanged


class TestWorkspaceDefaultRoot:
    def test_default_root_is_cwd(self):
        import os
        ws = Workspace()
        assert ws.root == Path(os.getcwd()).resolve()
