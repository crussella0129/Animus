"""Ferric Layer binary discovery and availability utilities.

All Python code that needs to locate Ferric binaries should import from here.
Discovery order: bundled (src/bin/) → PATH → None.
When a binary is not found, callers must fall back to Python equivalents.
"""
from __future__ import annotations

import shutil
from pathlib import Path

_BIN_DIR = Path(__file__).parent / "bin"


def find_ferric_binary(name: str) -> str | None:
    """Locate a Ferric binary. Returns path string or None.

    Search order:
    1. src/bin/<name> (bundled in distribution wheels)
    2. Executable on PATH (development / manual install)
    """
    bundled = _BIN_DIR / name
    if bundled.exists():
        return str(bundled)
    # Windows: also check with .exe extension
    bundled_exe = _BIN_DIR / f"{name}.exe"
    if bundled_exe.exists():
        return str(bundled_exe)
    return shutil.which(name)


def is_ferric_available(name: str) -> bool:
    """Return True if the named Ferric binary is accessible."""
    return find_ferric_binary(name) is not None
