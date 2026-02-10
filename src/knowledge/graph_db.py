"""SQLite-based code knowledge graph storage and queries."""

from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.knowledge.parser import FileParseResult, NodeInfo


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL UNIQUE,
    file_path TEXT,
    line_start INTEGER,
    line_end INTEGER,
    docstring TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    UNIQUE(source_id, target_id, kind)
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    last_modified REAL,
    hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_qualified_name ON nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
"""


@dataclass
class NodeRow:
    """A node as returned from the database."""

    id: int
    kind: str
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str


class GraphDB:
    """SQLite-backed code knowledge graph."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_file_results(
        self,
        result: FileParseResult,
        file_hash: str,
        mtime: float,
    ) -> None:
        """Atomic upsert: delete old nodes for file, insert new, resolve edges."""
        cur = self._conn.cursor()
        try:
            # Delete existing nodes (CASCADE deletes edges)
            cur.execute("DELETE FROM nodes WHERE file_path = ?", (result.file_path,))

            # Insert nodes
            node_ids: dict[str, int] = {}
            for node in result.nodes:
                cur.execute(
                    "INSERT OR REPLACE INTO nodes "
                    "(kind, name, qualified_name, file_path, line_start, line_end, docstring) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        node.kind,
                        node.name,
                        node.qualified_name,
                        node.file_path,
                        node.line_start,
                        node.line_end,
                        node.docstring,
                    ),
                )
                node_ids[node.qualified_name] = cur.lastrowid  # type: ignore[assignment]

            # Resolve and insert edges
            for edge in result.edges:
                source_id = node_ids.get(edge.source_qname)
                if source_id is None:
                    source_id = self._resolve_node_id(cur, edge.source_qname)
                if source_id is None:
                    continue

                target_id = node_ids.get(edge.target_name)
                if target_id is None:
                    target_id = self._resolve_node_id(cur, edge.target_name)
                if target_id is None:
                    # Try partial match: same file first, then global
                    target_id = self._resolve_by_name(
                        cur, edge.target_name, result.file_path
                    )
                if target_id is None:
                    # Create phantom "external" node
                    target_id = self._create_external_node(cur, edge.target_name)

                cur.execute(
                    "INSERT OR IGNORE INTO edges (source_id, target_id, kind) "
                    "VALUES (?, ?, ?)",
                    (source_id, target_id, edge.kind),
                )

            # Update files table
            cur.execute(
                "INSERT OR REPLACE INTO files (path, last_modified, hash) VALUES (?, ?, ?)",
                (result.file_path, mtime, file_hash),
            )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def remove_file(self, file_path: str) -> None:
        """Remove all nodes and edges for a file."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
        cur.execute("DELETE FROM files WHERE path = ?", (file_path,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Node resolution helpers
    # ------------------------------------------------------------------

    def _resolve_node_id(self, cur: sqlite3.Cursor, qname: str) -> Optional[int]:
        """Find a node by qualified name."""
        cur.execute("SELECT id FROM nodes WHERE qualified_name = ?", (qname,))
        row = cur.fetchone()
        return row[0] if row else None

    def _resolve_by_name(
        self, cur: sqlite3.Cursor, name: str, file_path: str
    ) -> Optional[int]:
        """Resolve by short name: same-file match first, then global."""
        # Same file
        cur.execute(
            "SELECT id FROM nodes WHERE name = ? AND file_path = ? LIMIT 1",
            (name, file_path),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        # Dotted suffix match (e.g. "ClassName.method" → find qualified_name ending with it)
        cur.execute(
            "SELECT id FROM nodes WHERE qualified_name LIKE ? LIMIT 1",
            (f"%.{name}",),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        # Global name match
        cur.execute("SELECT id FROM nodes WHERE name = ? LIMIT 1", (name,))
        row = cur.fetchone()
        return row[0] if row else None

    def _create_external_node(self, cur: sqlite3.Cursor, name: str) -> int:
        """Create a phantom node for an unresolved external reference."""
        cur.execute(
            "INSERT OR IGNORE INTO nodes "
            "(kind, name, qualified_name, file_path, line_start, line_end, docstring) "
            "VALUES ('external', ?, ?, '', 0, 0, '')",
            (name, f"external.{name}"),
        )
        # If it already existed (OR IGNORE), fetch its id
        cur.execute(
            "SELECT id FROM nodes WHERE qualified_name = ?", (f"external.{name}",)
        )
        row = cur.fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def search_nodes(
        self,
        pattern: str,
        kind: Optional[str] = None,
        limit: int = 20,
    ) -> list[NodeRow]:
        """Search nodes by name or qualified_name using SQL LIKE."""
        like_pattern = f"%{pattern}%"
        if kind:
            rows = self._conn.execute(
                "SELECT id, kind, name, qualified_name, file_path, "
                "line_start, line_end, docstring "
                "FROM nodes WHERE (name LIKE ? OR qualified_name LIKE ?) "
                "AND kind = ? LIMIT ?",
                (like_pattern, like_pattern, kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, kind, name, qualified_name, file_path, "
                "line_start, line_end, docstring "
                "FROM nodes WHERE (name LIKE ? OR qualified_name LIKE ?) LIMIT ?",
                (like_pattern, like_pattern, limit),
            ).fetchall()
        return [NodeRow(*r) for r in rows]

    def get_callers(self, qname: str) -> list[NodeRow]:
        """Get all nodes that call the given symbol."""
        node_id = self._get_node_id(qname)
        if node_id is None:
            return []
        rows = self._conn.execute(
            "SELECT n.id, n.kind, n.name, n.qualified_name, n.file_path, "
            "n.line_start, n.line_end, n.docstring "
            "FROM nodes n JOIN edges e ON n.id = e.source_id "
            "WHERE e.target_id = ? AND e.kind = 'CALLS'",
            (node_id,),
        ).fetchall()
        return [NodeRow(*r) for r in rows]

    def get_callees(self, qname: str) -> list[NodeRow]:
        """Get all nodes that the given symbol calls."""
        node_id = self._get_node_id(qname)
        if node_id is None:
            return []
        rows = self._conn.execute(
            "SELECT n.id, n.kind, n.name, n.qualified_name, n.file_path, "
            "n.line_start, n.line_end, n.docstring "
            "FROM nodes n JOIN edges e ON n.id = e.target_id "
            "WHERE e.source_id = ? AND e.kind = 'CALLS'",
            (node_id,),
        ).fetchall()
        return [NodeRow(*r) for r in rows]

    def get_inheritance_tree(self, qname: str) -> list[NodeRow]:
        """Get classes that inherit from the given class (direct subclasses)."""
        node_id = self._get_node_id(qname)
        if node_id is None:
            return []
        rows = self._conn.execute(
            "SELECT n.id, n.kind, n.name, n.qualified_name, n.file_path, "
            "n.line_start, n.line_end, n.docstring "
            "FROM nodes n JOIN edges e ON n.id = e.source_id "
            "WHERE e.target_id = ? AND e.kind = 'INHERITS'",
            (node_id,),
        ).fetchall()
        return [NodeRow(*r) for r in rows]

    def get_blast_radius(
        self, qname: str, max_depth: int = 5
    ) -> dict[int, list[NodeRow]]:
        """BFS along incoming CALLS/INHERITS/IMPORTS edges.

        Returns a dict mapping depth → list of affected nodes.
        """
        start_id = self._get_node_id(qname)
        if start_id is None:
            return {}

        visited: set[int] = {start_id}
        queue: deque[tuple[int, int]] = deque([(start_id, 0)])
        result: dict[int, list[NodeRow]] = {}

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Find all nodes pointing to current via CALLS, INHERITS, or IMPORTS
            rows = self._conn.execute(
                "SELECT n.id, n.kind, n.name, n.qualified_name, n.file_path, "
                "n.line_start, n.line_end, n.docstring "
                "FROM nodes n JOIN edges e ON n.id = e.source_id "
                "WHERE e.target_id = ? AND e.kind IN ('CALLS', 'INHERITS', 'IMPORTS')",
                (current_id,),
            ).fetchall()

            for row in rows:
                node = NodeRow(*row)
                if node.id not in visited:
                    visited.add(node.id)
                    next_depth = depth + 1
                    result.setdefault(next_depth, []).append(node)
                    queue.append((node.id, next_depth))

        return result

    def get_stats(self) -> dict[str, int]:
        """Return counts for display."""
        node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        file_count = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {
            "nodes": node_count,
            "edges": edge_count,
            "files": file_count,
        }

    def get_file_info(self, file_path: str) -> Optional[tuple[float, str]]:
        """Get (mtime, hash) for a tracked file, or None if not tracked."""
        row = self._conn.execute(
            "SELECT last_modified, hash FROM files WHERE path = ?", (file_path,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def get_tracked_files(self) -> set[str]:
        """Return set of all tracked file paths."""
        rows = self._conn.execute("SELECT path FROM files").fetchall()
        return {r[0] for r in rows}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_node_id(self, qname: str) -> Optional[int]:
        """Look up node ID by qualified name."""
        row = self._conn.execute(
            "SELECT id FROM nodes WHERE qualified_name = ?", (qname,)
        ).fetchone()
        return row[0] if row else None
