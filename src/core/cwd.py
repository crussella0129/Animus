"""Session-level working directory tracking.

Provides a mutable CWD container so that shell `cd` commands persist
across tool calls within a single agent session.
"""

from __future__ import annotations

import os
from pathlib import Path


class SessionCwd:
    """Mutable container holding the session's current working directory.

    All tool calls can share a single instance so that a ``cd`` in one
    shell command is visible to subsequent file / git operations.
    """

    def __init__(self, initial: Path | str | None = None) -> None:
        if initial is not None:
            self._cwd = Path(initial).resolve()
        else:
            self._cwd = Path(os.getcwd()).resolve()

    @property
    def path(self) -> Path:
        """Return the current session working directory."""
        return self._cwd

    def set(self, new_dir: Path | str) -> None:
        """Update the session CWD.

        Relative paths are resolved against the *current* session CWD.
        Non-existent directories are silently ignored so that a failed
        ``mkdir && cd`` does not corrupt the state.
        """
        candidate = Path(new_dir)
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        candidate = candidate.resolve()
        if candidate.is_dir():
            self._cwd = candidate

    def resolve(self, path: Path | str) -> Path:
        """Resolve *path* against the session CWD.

        Absolute paths are returned as-is (after ``.resolve()``).
        Relative paths are joined to the session CWD first.
        """
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self._cwd / p).resolve()
