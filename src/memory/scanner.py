"""Directory scanner with .gitignore support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Set
import fnmatch


class GitIgnoreParser:
    """Parser for .gitignore patterns."""

    def __init__(self):
        self.patterns: list[tuple[str, bool]] = []  # (pattern, is_negation)

    def add_pattern(self, pattern: str) -> None:
        """Add a pattern to the ignore list."""
        pattern = pattern.strip()
        if not pattern or pattern.startswith("#"):
            return

        is_negation = pattern.startswith("!")
        if is_negation:
            pattern = pattern[1:]

        # Normalize pattern
        if pattern.endswith("/"):
            pattern = pattern[:-1]

        self.patterns.append((pattern, is_negation))

    def load_file(self, gitignore_path: Path) -> None:
        """Load patterns from a .gitignore file."""
        if not gitignore_path.exists():
            return

        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    self.add_pattern(line)
        except (IOError, OSError):
            pass

    def matches(self, path: Path, is_dir: bool = False) -> bool:
        """
        Check if a path matches any ignore pattern.

        Args:
            path: Path to check (relative to repo root).
            is_dir: Whether the path is a directory.

        Returns:
            True if the path should be ignored.
        """
        path_str = str(path).replace("\\", "/")
        name = path.name

        should_ignore = False

        for pattern, is_negation in self.patterns:
            matched = False

            # Check if pattern matches
            if "/" in pattern:
                # Pattern with path separator - match against full path
                matched = fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"**/{pattern}")
            else:
                # Simple pattern - match against name
                matched = fnmatch.fnmatch(name, pattern)

            if matched:
                should_ignore = not is_negation

        return should_ignore


class DirectoryScanner:
    """
    Scans directories for files, respecting .gitignore patterns.

    Supports:
    - Recursive directory traversal
    - .gitignore pattern matching
    - File extension filtering
    - Size limits
    """

    # Default patterns to always ignore
    DEFAULT_IGNORE = [
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "Thumbs.db",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "*.egg-info",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    ]

    # Binary file extensions to skip
    BINARY_EXTENSIONS = {
        ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
        ".o", ".obj", ".bin", ".dat",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
        ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wav", ".flac",
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
        ".db", ".sqlite", ".sqlite3",
    }

    def __init__(
        self,
        root: Path,
        extensions: Optional[Set[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB default
        respect_gitignore: bool = True,
        include_hidden: bool = False,
    ):
        """
        Initialize the scanner.

        Args:
            root: Root directory to scan.
            extensions: Set of file extensions to include (e.g., {".py", ".js"}).
                       If None, includes all non-binary files.
            max_file_size: Maximum file size in bytes to include.
            respect_gitignore: Whether to respect .gitignore patterns.
            include_hidden: Whether to include hidden files/directories.
        """
        self.root = Path(root).resolve()
        self.extensions = extensions
        self.max_file_size = max_file_size
        self.respect_gitignore = respect_gitignore
        self.include_hidden = include_hidden
        self._gitignore = GitIgnoreParser()

        # Load default ignore patterns
        for pattern in self.DEFAULT_IGNORE:
            self._gitignore.add_pattern(pattern)

        # Load .gitignore from root
        if respect_gitignore:
            self._gitignore.load_file(self.root / ".gitignore")

    def _should_skip(self, path: Path, is_dir: bool = False) -> bool:
        """Check if a path should be skipped."""
        name = path.name

        # Skip hidden files/directories unless explicitly included
        if not self.include_hidden and name.startswith("."):
            return True

        # Check gitignore patterns
        try:
            rel_path = path.relative_to(self.root)
        except ValueError:
            rel_path = path

        if self._gitignore.matches(rel_path, is_dir):
            return True

        if not is_dir:
            # Check extension filter
            if self.extensions is not None:
                if path.suffix.lower() not in self.extensions:
                    return True

            # Skip binary files
            if path.suffix.lower() in self.BINARY_EXTENSIONS:
                return True

            # Check file size
            try:
                if path.stat().st_size > self.max_file_size:
                    return True
            except (OSError, IOError):
                return True

        return False

    def scan(self) -> Iterator[Path]:
        """
        Scan the directory and yield file paths.

        Yields:
            Path objects for each matching file.
        """
        if not self.root.exists():
            return

        if self.root.is_file():
            if not self._should_skip(self.root):
                yield self.root
            return

        # Stack-based traversal (avoids recursion limits)
        stack = [self.root]

        while stack:
            current = stack.pop()

            try:
                entries = list(current.iterdir())
            except (PermissionError, OSError):
                continue

            # Sort for consistent ordering
            entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

            for entry in entries:
                try:
                    if entry.is_dir():
                        if not self._should_skip(entry, is_dir=True):
                            # Load nested .gitignore
                            if self.respect_gitignore:
                                nested_gitignore = entry / ".gitignore"
                                if nested_gitignore.exists():
                                    self._gitignore.load_file(nested_gitignore)
                            stack.append(entry)
                    elif entry.is_file():
                        if not self._should_skip(entry):
                            yield entry
                except (PermissionError, OSError):
                    continue

    def count(self) -> int:
        """Count the number of files that would be scanned."""
        return sum(1 for _ in self.scan())
