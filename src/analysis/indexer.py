"""Intelligent codebase indexer using Tree-sitter for AST analysis."""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Any
import fnmatch

from src.analysis.parser import (
    CodeParser,
    ParsedCode,
    CodeSymbol,
    SymbolType,
    is_tree_sitter_available,
)


@dataclass
class IndexedFile:
    """An indexed source file."""
    path: str
    language: str
    symbols: list[CodeSymbol]
    imports: list[str]
    last_modified: float
    file_hash: str
    indexed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": self.imports,
            "last_modified": self.last_modified,
            "file_hash": self.file_hash,
            "indexed_at": self.indexed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IndexedFile":
        """Create from dictionary."""
        symbols = [
            CodeSymbol(
                name=s["name"],
                type=SymbolType(s["type"]),
                line=s["line"],
                column=s["column"],
                end_line=s["end_line"],
                end_column=s["end_column"],
                parent=s.get("parent"),
                docstring=s.get("docstring"),
                signature=s.get("signature"),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("symbols", [])
        ]
        return cls(
            path=data["path"],
            language=data["language"],
            symbols=symbols,
            imports=data.get("imports", []),
            last_modified=data["last_modified"],
            file_hash=data["file_hash"],
            indexed_at=datetime.fromisoformat(data["indexed_at"]),
        )


@dataclass
class SearchResult:
    """A search result from the index."""
    file_path: str
    symbol: Optional[CodeSymbol]
    line: int
    context: str  # Line or snippet of content
    score: float = 1.0
    match_type: str = "symbol"  # "symbol", "import", "content"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "symbol": self.symbol.to_dict() if self.symbol else None,
            "line": self.line,
            "context": self.context,
            "score": self.score,
            "match_type": self.match_type,
        }


@dataclass
class CodebaseIndex:
    """Index of a codebase."""
    root_path: str
    files: dict[str, IndexedFile] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Symbol lookup tables for fast search
    _symbol_index: dict[str, list[tuple[str, CodeSymbol]]] = field(
        default_factory=dict, repr=False
    )
    _class_index: dict[str, list[tuple[str, CodeSymbol]]] = field(
        default_factory=dict, repr=False
    )
    _function_index: dict[str, list[tuple[str, CodeSymbol]]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        """Rebuild lookup tables after init."""
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild symbol lookup indexes."""
        self._symbol_index = {}
        self._class_index = {}
        self._function_index = {}

        for file_path, indexed_file in self.files.items():
            for symbol in indexed_file.symbols:
                name_lower = symbol.name.lower()

                # Add to symbol index
                if name_lower not in self._symbol_index:
                    self._symbol_index[name_lower] = []
                self._symbol_index[name_lower].append((file_path, symbol))

                # Add to type-specific indexes
                if symbol.type == SymbolType.CLASS:
                    if name_lower not in self._class_index:
                        self._class_index[name_lower] = []
                    self._class_index[name_lower].append((file_path, symbol))
                elif symbol.type in (SymbolType.FUNCTION, SymbolType.METHOD):
                    if name_lower not in self._function_index:
                        self._function_index[name_lower] = []
                    self._function_index[name_lower].append((file_path, symbol))

    def add_file(self, indexed_file: IndexedFile) -> None:
        """Add or update a file in the index."""
        self.files[indexed_file.path] = indexed_file
        self.updated_at = datetime.now()

        # Update lookup indexes
        for symbol in indexed_file.symbols:
            name_lower = symbol.name.lower()

            if name_lower not in self._symbol_index:
                self._symbol_index[name_lower] = []
            self._symbol_index[name_lower].append((indexed_file.path, symbol))

            if symbol.type == SymbolType.CLASS:
                if name_lower not in self._class_index:
                    self._class_index[name_lower] = []
                self._class_index[name_lower].append((indexed_file.path, symbol))
            elif symbol.type in (SymbolType.FUNCTION, SymbolType.METHOD):
                if name_lower not in self._function_index:
                    self._function_index[name_lower] = []
                self._function_index[name_lower].append((indexed_file.path, symbol))

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the index."""
        if file_path in self.files:
            del self.files[file_path]
            self.updated_at = datetime.now()
            self._rebuild_indexes()

    def search_symbols(
        self,
        query: str,
        symbol_type: Optional[SymbolType] = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Search for symbols by name.

        Args:
            query: Search query (case-insensitive substring match).
            symbol_type: Optional filter by symbol type.
            limit: Maximum results to return.

        Returns:
            List of search results.
        """
        results = []
        query_lower = query.lower()

        # Choose index based on type filter
        if symbol_type == SymbolType.CLASS:
            index = self._class_index
        elif symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
            index = self._function_index
        else:
            index = self._symbol_index

        # Search through index
        for name, entries in index.items():
            if query_lower in name:
                for file_path, symbol in entries:
                    if symbol_type and symbol.type != symbol_type:
                        continue

                    # Calculate score (exact match scores higher)
                    score = 1.0 if name == query_lower else 0.8 if name.startswith(query_lower) else 0.5

                    results.append(SearchResult(
                        file_path=file_path,
                        symbol=symbol,
                        line=symbol.line,
                        context=f"{symbol.type.value}: {symbol.name}{symbol.signature or ''}",
                        score=score,
                        match_type="symbol",
                    ))

        # Sort by score (descending) and limit
        results.sort(key=lambda r: (-r.score, r.file_path, r.line))
        return results[:limit]

    def find_definition(self, symbol_name: str) -> list[SearchResult]:
        """Find the definition of a symbol.

        Args:
            symbol_name: Exact name of the symbol.

        Returns:
            List of matching definitions.
        """
        results = []
        name_lower = symbol_name.lower()

        if name_lower in self._symbol_index:
            for file_path, symbol in self._symbol_index[name_lower]:
                if symbol.name.lower() == name_lower:
                    results.append(SearchResult(
                        file_path=file_path,
                        symbol=symbol,
                        line=symbol.line,
                        context=f"{symbol.type.value}: {symbol.name}",
                        score=1.0,
                        match_type="symbol",
                    ))

        return results

    def get_file_symbols(self, file_path: str) -> list[CodeSymbol]:
        """Get all symbols in a file."""
        if file_path in self.files:
            return self.files[file_path].symbols
        return []

    def get_statistics(self) -> dict:
        """Get index statistics."""
        total_symbols = sum(len(f.symbols) for f in self.files.values())
        total_classes = sum(
            1 for f in self.files.values()
            for s in f.symbols if s.type == SymbolType.CLASS
        )
        total_functions = sum(
            1 for f in self.files.values()
            for s in f.symbols if s.type in (SymbolType.FUNCTION, SymbolType.METHOD)
        )
        languages = set(f.language for f in self.files.values())

        return {
            "root_path": self.root_path,
            "total_files": len(self.files),
            "total_symbols": total_symbols,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "languages": list(languages),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "root_path": self.root_path,
            "files": {path: f.to_dict() for path, f in self.files.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodebaseIndex":
        """Create from dictionary."""
        files = {
            path: IndexedFile.from_dict(f)
            for path, f in data.get("files", {}).items()
        }
        index = cls(
            root_path=data["root_path"],
            files=files,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
        return index


class CodebaseIndexer:
    """Indexes codebases for fast symbol search."""

    # Default patterns to ignore
    DEFAULT_IGNORE_PATTERNS = [
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".next",
        ".cache",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dll",
        "*.exe",
        "*.min.js",
        "*.min.css",
        "*.map",
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
    ]

    def __init__(
        self,
        ignore_patterns: Optional[list[str]] = None,
        max_file_size: int = 1024 * 1024,  # 1MB
    ):
        """Initialize the indexer.

        Args:
            ignore_patterns: Patterns to ignore (in addition to defaults).
            max_file_size: Maximum file size to index in bytes.
        """
        self._parser = CodeParser()
        self._ignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()
        if ignore_patterns:
            self._ignore_patterns.extend(ignore_patterns)
        self._max_file_size = max_file_size

    def _should_ignore(self, path: Path, rel_path: str) -> bool:
        """Check if a path should be ignored."""
        name = path.name

        for pattern in self._ignore_patterns:
            # Check directory name
            if path.is_dir() and fnmatch.fnmatch(name, pattern):
                return True
            # Check file name
            if fnmatch.fnmatch(name, pattern):
                return True
            # Check path pattern
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Check if pattern matches any part of path
            if any(fnmatch.fnmatch(part, pattern) for part in Path(rel_path).parts):
                return True

        return False

    def _file_hash(self, path: Path) -> str:
        """Compute hash of file contents."""
        hasher = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def _iter_files(self, root: Path) -> Iterator[Path]:
        """Iterate over indexable files."""
        for path in root.rglob("*"):
            if path.is_file():
                rel_path = str(path.relative_to(root))
                if not self._should_ignore(path, rel_path):
                    # Check file size
                    try:
                        if path.stat().st_size <= self._max_file_size:
                            # Check if language is supported
                            lang = self._parser.detect_language(str(path))
                            if lang:
                                yield path
                    except OSError:
                        pass

    def index_file(self, file_path: Path, root: Path) -> Optional[IndexedFile]:
        """Index a single file.

        Args:
            file_path: Path to the file.
            root: Root directory for relative paths.

        Returns:
            IndexedFile or None if parsing failed.
        """
        try:
            rel_path = str(file_path.relative_to(root))
            parsed = self._parser.parse_file(str(file_path))

            if parsed.errors and not parsed.symbols:
                return None

            return IndexedFile(
                path=rel_path,
                language=parsed.language,
                symbols=parsed.symbols,
                imports=parsed.imports,
                last_modified=file_path.stat().st_mtime,
                file_hash=self._file_hash(file_path),
            )
        except Exception:
            return None

    def index_directory(
        self,
        directory: str,
        progress_callback: Optional[callable] = None,
    ) -> CodebaseIndex:
        """Index a directory.

        Args:
            directory: Path to the directory to index.
            progress_callback: Optional callback(current, total, file_path).

        Returns:
            CodebaseIndex with all indexed files.
        """
        root = Path(directory).resolve()
        if not root.exists():
            raise ValueError(f"Directory not found: {directory}")

        index = CodebaseIndex(root_path=str(root))

        # Collect files first for progress reporting
        files = list(self._iter_files(root))
        total = len(files)

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, str(file_path))

            indexed = self.index_file(file_path, root)
            if indexed:
                index.add_file(indexed)

        return index

    def update_index(
        self,
        index: CodebaseIndex,
        progress_callback: Optional[callable] = None,
    ) -> tuple[int, int, int]:
        """Update an existing index with changes.

        Args:
            index: Existing index to update.
            progress_callback: Optional callback(current, total, file_path).

        Returns:
            Tuple of (added, updated, removed) file counts.
        """
        root = Path(index.root_path)
        if not root.exists():
            raise ValueError(f"Root directory not found: {index.root_path}")

        added = 0
        updated = 0
        removed = 0

        # Track which files we've seen
        seen_files = set()

        # Collect files
        files = list(self._iter_files(root))
        total = len(files)

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, str(file_path))

            rel_path = str(file_path.relative_to(root))
            seen_files.add(rel_path)

            # Check if file needs updating
            existing = index.files.get(rel_path)
            current_mtime = file_path.stat().st_mtime

            if existing:
                # Check if file changed
                if existing.last_modified < current_mtime:
                    indexed = self.index_file(file_path, root)
                    if indexed:
                        index.add_file(indexed)
                        updated += 1
            else:
                # New file
                indexed = self.index_file(file_path, root)
                if indexed:
                    index.add_file(indexed)
                    added += 1

        # Remove deleted files
        for file_path in list(index.files.keys()):
            if file_path not in seen_files:
                index.remove_file(file_path)
                removed += 1

        index.updated_at = datetime.now()
        return added, updated, removed

    def save_index(self, index: CodebaseIndex, path: str) -> None:
        """Save index to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, indent=2)

    def load_index(self, path: str) -> CodebaseIndex:
        """Load index from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CodebaseIndex.from_dict(data)
