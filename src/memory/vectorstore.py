"""Vector stores with cosine similarity search: in-memory and SQLite-persistent."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """Simple in-memory vector store for RAG."""

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._metadata: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._texts)

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add texts with their embeddings to the store."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        meta = metadata or [{} for _ in texts]
        self._texts.extend(texts)
        self._embeddings.extend(embeddings)
        self._metadata.extend(meta)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for most similar texts by cosine similarity."""
        if not self._embeddings:
            return []

        scores = [
            (i, _cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self._embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scores[:top_k]:
            results.append(SearchResult(
                text=self._texts[i],
                score=score,
                metadata=self._metadata[i],
            ))
        return results

    def clear(self) -> None:
        """Remove all entries."""
        self._texts.clear()
        self._embeddings.clear()
        self._metadata.clear()


# ---------------------------------------------------------------------------
# Binary embedding helpers
# ---------------------------------------------------------------------------

def _pack_embedding(values: list[float]) -> bytes:
    """Pack a float list into a compact binary blob (4 bytes per float)."""
    return struct.pack(f"{len(values)}f", *values)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a float list."""
    dim = len(blob) // 4
    return list(struct.unpack(f"{dim}f", blob))


# ---------------------------------------------------------------------------
# SQLite-backed persistent vector store
# ---------------------------------------------------------------------------

_VECTOR_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    metadata TEXT DEFAULT '{}',
    file_path TEXT,
    file_hash TEXT
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    last_modified REAL,
    hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
"""


class SQLiteVectorStore:
    """SQLite-backed persistent vector store for RAG."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_VECTOR_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        file_path: str | None = None,
        file_hash: str | None = None,
    ) -> None:
        """Add texts with their embeddings to the store."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        meta = metadata or [{} for _ in texts]
        rows = [
            (
                texts[i],
                _pack_embedding(embeddings[i]),
                json.dumps(meta[i]),
                file_path,
                file_hash,
            )
            for i in range(len(texts))
        ]
        self._conn.executemany(
            "INSERT INTO chunks (text, embedding, metadata, file_path, file_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for most similar texts by cosine similarity (brute-force)."""
        rows = self._conn.execute(
            "SELECT text, embedding, metadata FROM chunks"
        ).fetchall()
        if not rows:
            return []

        scored: list[tuple[str, float, dict[str, Any]]] = []
        for text, blob, meta_json in rows:
            emb = _unpack_embedding(blob)
            score = _cosine_similarity(query_embedding, emb)
            scored.append((text, score, json.loads(meta_json)))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(text=text, score=score, metadata=meta)
            for text, score, meta in scored[:top_k]
        ]

    def clear(self) -> None:
        """Remove all chunks and file tracking records."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM files")
        self._conn.commit()

    # ------------------------------------------------------------------
    # File tracking (for incremental ingestion)
    # ------------------------------------------------------------------

    def get_file_info(self, path: str) -> tuple[float, str] | None:
        """Return (last_modified, hash) for a tracked file, or None."""
        row = self._conn.execute(
            "SELECT last_modified, hash FROM files WHERE path = ?", (path,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def upsert_file(self, path: str, last_modified: float, file_hash: str) -> None:
        """Insert or update a tracked file record."""
        self._conn.execute(
            "INSERT INTO files (path, last_modified, hash) VALUES (?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET last_modified=excluded.last_modified, hash=excluded.hash",
            (path, last_modified, file_hash),
        )
        self._conn.commit()

    def remove_file_chunks(self, path: str) -> None:
        """Remove all chunks and the file tracking record for a given file."""
        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.commit()

    def get_tracked_files(self) -> list[str]:
        """Return all tracked file paths."""
        rows = self._conn.execute("SELECT path FROM files").fetchall()
        return [r[0] for r in rows]

    def get_stats(self) -> dict[str, int]:
        """Return store statistics."""
        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"chunks": chunks, "files": files}
