"""Project-scoped workspace with boundary enforcement.

Replaces SessionCwd with an immutable root boundary and mutable CWD.
All file operations must resolve within the workspace root.
"""

from __future__ import annotations

import os
from pathlib import Path


class WorkspaceBoundaryError(Exception):
    """Raised when an operation attempts to escape the workspace root."""


class Workspace:
    """Project-scoped working directory with boundary enforcement.

    Enforces that all path resolutions stay within the workspace root.
    Drop-in replacement for SessionCwd with added safety.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is not None:
            self._root = Path(root).resolve()
        else:
            self._root = Path(os.getcwd()).resolve()
        self._cwd = self._root

    @property
    def root(self) -> Path:
        """Immutable project boundary."""
        return self._root

    @property
    def cwd(self) -> Path:
        """Current working directory within workspace."""
        return self._cwd

    @property
    def path(self) -> Path:
        """Alias for cwd - backward compatibility with SessionCwd."""
        return self._cwd

    def set_cwd(self, new_dir: Path | str) -> None:
        """Update CWD. Must stay within root. Raises on boundary violation."""
        candidate = Path(new_dir)
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        candidate = candidate.resolve()

        if not candidate.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                f"Cannot change directory to {new_dir}: "
                f"outside workspace root {self._root}"
            )
        if candidate.is_dir():
            self._cwd = candidate

    def set(self, new_dir: Path | str) -> None:
        """Update CWD (SessionCwd-compatible). Silently ignores violations."""
        try:
            self.set_cwd(new_dir)
        except WorkspaceBoundaryError:
            pass
        except (OSError, ValueError):
            pass

    def resolve(self, path: Path | str) -> Path:
        """Resolve path relative to CWD, enforce workspace boundary."""
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._cwd / p).resolve()

        if not resolved.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                f"Path '{path}' resolves to {resolved}: "
                f"outside workspace root {self._root}"
            )
        return resolved
