"""Orchestrator: scan directories, parse Python files, store in graph DB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.knowledge.graph_db import GraphDB
from src.knowledge.parser import PythonParser
from src.memory.scanner import Scanner


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    files_scanned: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_nodes: int = 0
    total_edges: int = 0


def _file_hash(path: Path) -> str:
    """Compute a fast content hash for change detection."""
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


class Indexer:
    """Scan a directory tree, parse Python files, and upsert into GraphDB."""

    def __init__(self, db: GraphDB) -> None:
        self._db = db
        self._parser = PythonParser()
        self._scanner = Scanner()

    def index_directory(
        self,
        root: Path,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> IndexResult:
        """Full directory index: scan, diff against DB, parse changed files, prune stale."""
        root = root.resolve()
        result = IndexResult()

        # Scan for .py files
        py_files = [p for p in self._scanner.scan(root, "**/*.py")]
        result.files_scanned = len(py_files)

        seen_paths: set[str] = set()

        for path in py_files:
            path_str = str(path)
            seen_paths.add(path_str)

            # Check if file has changed
            mtime = path.stat().st_mtime
            file_info = self._db.get_file_info(path_str)
            if file_info is not None:
                stored_mtime, stored_hash = file_info
                if stored_mtime == mtime:
                    result.files_skipped += 1
                    continue
                # mtime changed — check hash
                current_hash = _file_hash(path)
                if current_hash == stored_hash:
                    # Content unchanged, just update mtime
                    result.files_skipped += 1
                    continue
            else:
                current_hash = _file_hash(path)

            # Parse and upsert
            if on_progress:
                on_progress(path_str)

            try:
                parse_result = self._parser.parse_file(path)
                self._db.upsert_file_results(parse_result, current_hash, mtime)
                result.files_parsed += 1
                result.total_nodes += len(parse_result.nodes)
                result.total_edges += len(parse_result.edges)
            except Exception:
                result.files_failed += 1

        # Remove stale files (tracked but no longer on disk)
        tracked = self._db.get_tracked_files()
        for tracked_path in tracked:
            if tracked_path not in seen_paths:
                self._db.remove_file(tracked_path)

        return result

    def index_file(self, path: Path) -> IndexResult:
        """Index a single file."""
        result = IndexResult()
        path = path.resolve()
        path_str = str(path)

        if not path.exists() or not path.suffix == ".py":
            return result

        result.files_scanned = 1
        current_hash = _file_hash(path)
        mtime = path.stat().st_mtime

        try:
            parse_result = self._parser.parse_file(path)
            self._db.upsert_file_results(parse_result, current_hash, mtime)
            result.files_parsed = 1
            result.total_nodes = len(parse_result.nodes)
            result.total_edges = len(parse_result.edges)
        except Exception:
            result.files_failed = 1

        return result
