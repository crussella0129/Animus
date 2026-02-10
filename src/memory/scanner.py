"""Directory walker with .gitignore awareness."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterator

# Default patterns to always skip
_DEFAULT_IGNORE = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
}

# Binary extensions to skip
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".dll", ".so", ".dylib",
    ".gguf", ".bin", ".pt", ".pth", ".safetensors",
    ".whl", ".egg",
})


class Scanner:
    """Walk directories yielding text file paths, respecting .gitignore."""

    def __init__(self, extra_ignore: set[str] | None = None) -> None:
        self._ignore_patterns = _DEFAULT_IGNORE | (extra_ignore or set())

    def scan(self, root: Path, glob_pattern: str = "**/*") -> Iterator[Path]:
        """Yield paths of text files under root."""
        root = root.resolve()
        gitignore_patterns = self._load_gitignore(root)
        all_patterns = self._ignore_patterns | gitignore_patterns

        for path in root.glob(glob_pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() in _BINARY_EXTENSIONS:
                continue
            if self._is_ignored(path, root, all_patterns):
                continue
            yield path

    def _load_gitignore(self, root: Path) -> set[str]:
        """Load .gitignore patterns from root directory."""
        gitignore = root / ".gitignore"
        if not gitignore.exists():
            return set()
        patterns = set()
        for line in gitignore.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.add(line)
        return patterns

    def _is_ignored(self, path: Path, root: Path, patterns: set[str]) -> bool:
        """Check if a path matches any ignore pattern."""
        rel = path.relative_to(root)
        rel_str = str(rel).replace("\\", "/")
        parts = rel.parts

        for pattern in patterns:
            # Check against full relative path
            if fnmatch.fnmatch(rel_str, pattern):
                return True
            # Check against filename
            if fnmatch.fnmatch(path.name, pattern):
                return True
            # Check against directory components
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False
