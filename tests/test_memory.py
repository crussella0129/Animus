"""Tests for RAG pipeline: scanner, chunker, embedder, vectorstore."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.chunker import Chunker, _estimate_tokens
from src.memory.embedder import MockEmbedder
from src.memory.scanner import Scanner
from src.memory.vectorstore import (
    InMemoryVectorStore,
    SQLiteVectorStore,
    _cosine_similarity,
    _pack_embedding,
    _unpack_embedding,
)


class TestScanner:
    def test_scan_finds_text_files(self, sample_files: Path):
        scanner = Scanner()
        files = list(scanner.scan(sample_files))
        names = {f.name for f in files}
        assert "hello.py" in names
        assert "readme.md" in names
        assert "data.txt" in names

    def test_scan_skips_binary_extensions(self, tmp_path: Path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "text.txt").write_text("hello")
        scanner = Scanner()
        files = list(scanner.scan(tmp_path))
        names = {f.name for f in files}
        assert "text.txt" in names
        assert "image.png" not in names

    def test_scan_respects_gitignore(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "ignored.txt").write_text("should be skipped")
        (tmp_path / "kept.txt").write_text("should be kept")
        scanner = Scanner()
        files = list(scanner.scan(tmp_path))
        names = {f.name for f in files}
        assert "kept.txt" in names
        assert "ignored.txt" not in names

    def test_scan_skips_pycache(self, tmp_path: Path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.cpython-312.pyc").write_bytes(b"bytecode")
        (tmp_path / "module.py").write_text("print('hello')")
        scanner = Scanner()
        files = list(scanner.scan(tmp_path))
        names = {f.name for f in files}
        assert "module.py" in names
        assert "module.cpython-312.pyc" not in names


class TestChunker:
    def test_chunk_simple_text(self):
        chunker = Chunker(chunk_size=50, overlap=10)
        text = "Hello world. " * 100
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert "text" in c
            assert "metadata" in c

    def test_chunk_empty_text(self):
        chunker = Chunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_preserves_metadata(self):
        chunker = Chunker(chunk_size=1000)
        chunks = chunker.chunk("Some text", metadata={"source": "test.txt"})
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source"] == "test.txt"
        assert "chunk_index" in chunks[0]["metadata"]

    def test_chunk_code_aware(self):
        code = """
import os

def hello():
    return "world"

def goodbye():
    return "farewell"

class MyClass:
    def method(self):
        pass
"""
        chunker = Chunker(chunk_size=30)
        chunks = chunker.chunk(code)
        assert len(chunks) >= 1


class TestEstimateTokens:
    def test_estimate(self):
        assert _estimate_tokens("hello") >= 1
        assert _estimate_tokens("a" * 400) == 100


class TestMockEmbedder:
    def test_embed_deterministic(self):
        embedder = MockEmbedder(dimension=32)
        v1 = embedder.embed(["hello"])[0]
        v2 = embedder.embed(["hello"])[0]
        assert v1 == v2

    def test_embed_different_texts(self):
        embedder = MockEmbedder(dimension=32)
        v1 = embedder.embed(["hello"])[0]
        v2 = embedder.embed(["world"])[0]
        assert v1 != v2

    def test_embed_dimension(self):
        embedder = MockEmbedder(dimension=64)
        assert embedder.dimension == 64
        v = embedder.embed(["test"])[0]
        assert len(v) == 64

    def test_embed_unit_vector(self):
        import math

        embedder = MockEmbedder(dimension=32)
        v = embedder.embed(["test"])[0]
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6


class TestVectorStore:
    def test_add_and_search(self):
        embedder = MockEmbedder(dimension=32)
        store = InMemoryVectorStore()

        texts = ["cat", "dog", "fish"]
        embeddings = embedder.embed(texts)
        store.add(texts, embeddings)

        assert len(store) == 3

        query_emb = embedder.embed(["cat"])[0]
        results = store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert results[0].text == "cat"
        assert results[0].score > 0.99  # self-similarity

    def test_search_empty_store(self):
        store = InMemoryVectorStore()
        results = store.search([0.1] * 32, top_k=5)
        assert results == []

    def test_clear(self):
        store = InMemoryVectorStore()
        store.add(["test"], [[0.1] * 32])
        assert len(store) == 1
        store.clear()
        assert len(store) == 0

    def test_add_mismatched_lengths(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError):
            store.add(["one", "two"], [[0.1] * 32])


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0


# ---------------------------------------------------------------------------
# SQLiteVectorStore tests
# ---------------------------------------------------------------------------


class TestSQLiteVectorStore:
    def test_add_and_search(self, sqlite_vector_store):
        embedder = MockEmbedder(dimension=32)
        store = sqlite_vector_store

        texts = ["cat", "dog", "fish"]
        embeddings = embedder.embed(texts)
        store.add(texts, embeddings)

        assert len(store) == 3

        query_emb = embedder.embed(["cat"])[0]
        results = store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert results[0].text == "cat"
        assert results[0].score > 0.99

    def test_search_empty_store(self, sqlite_vector_store):
        results = sqlite_vector_store.search([0.1] * 32, top_k=5)
        assert results == []

    def test_clear(self, sqlite_vector_store):
        store = sqlite_vector_store
        store.add(["test"], [[0.1] * 32])
        assert len(store) == 1
        store.clear()
        assert len(store) == 0

    def test_add_mismatched_lengths(self, sqlite_vector_store):
        with pytest.raises(ValueError):
            sqlite_vector_store.add(["one", "two"], [[0.1] * 32])

    def test_persistence(self, tmp_path):
        """Data persists across close/reopen."""
        db_path = tmp_path / "persist.db"
        embedder = MockEmbedder(dimension=32)

        store = SQLiteVectorStore(db_path)
        store.add(["hello", "world"], embedder.embed(["hello", "world"]))
        assert len(store) == 2
        store.close()

        store2 = SQLiteVectorStore(db_path)
        assert len(store2) == 2
        results = store2.search(embedder.embed(["hello"])[0], top_k=1)
        assert results[0].text == "hello"
        store2.close()

    def test_pack_unpack_roundtrip(self):
        original = [0.1, 0.2, -0.3, 0.0, 1.0]
        blob = _pack_embedding(original)
        restored = _unpack_embedding(blob)
        assert len(restored) == len(original)
        for a, b in zip(original, restored):
            assert abs(a - b) < 1e-6

    def test_file_tracking(self, sqlite_vector_store):
        store = sqlite_vector_store
        assert store.get_file_info("/test.py") is None

        store.upsert_file("/test.py", 1000.0, "abc123")
        info = store.get_file_info("/test.py")
        assert info is not None
        assert info == (1000.0, "abc123")

        # Update
        store.upsert_file("/test.py", 2000.0, "def456")
        info = store.get_file_info("/test.py")
        assert info == (2000.0, "def456")

    def test_remove_file_chunks(self, sqlite_vector_store):
        store = sqlite_vector_store
        embedder = MockEmbedder(dimension=32)

        store.add(["chunk1", "chunk2"], embedder.embed(["chunk1", "chunk2"]),
                  file_path="/a.py", file_hash="h1")
        store.add(["chunk3"], embedder.embed(["chunk3"]),
                  file_path="/b.py", file_hash="h2")
        store.upsert_file("/a.py", 1.0, "h1")
        store.upsert_file("/b.py", 2.0, "h2")

        assert len(store) == 3
        store.remove_file_chunks("/a.py")
        assert len(store) == 1
        assert store.get_file_info("/a.py") is None
        assert store.get_file_info("/b.py") is not None

    def test_len(self, sqlite_vector_store):
        store = sqlite_vector_store
        assert len(store) == 0
        store.add(["a", "b"], [[0.1] * 8, [0.2] * 8])
        assert len(store) == 2

    def test_metadata_roundtrip(self, sqlite_vector_store):
        store = sqlite_vector_store
        meta = {"source": "test.py", "chunk_index": 0, "language": "python"}
        store.add(["code"], [[0.5] * 8], metadata=[meta])
        results = store.search([0.5] * 8, top_k=1)
        assert results[0].metadata == meta

    def test_get_stats(self, sqlite_vector_store):
        store = sqlite_vector_store
        stats = store.get_stats()
        assert stats == {"chunks": 0, "files": 0}

        store.add(["a"], [[0.1] * 8], file_path="/x.py")
        store.upsert_file("/x.py", 1.0, "hash")
        stats = store.get_stats()
        assert stats == {"chunks": 1, "files": 1}

    def test_get_tracked_files(self, sqlite_vector_store):
        store = sqlite_vector_store
        assert store.get_tracked_files() == []

        store.upsert_file("/a.py", 1.0, "h1")
        store.upsert_file("/b.py", 2.0, "h2")
        tracked = store.get_tracked_files()
        assert set(tracked) == {"/a.py", "/b.py"}

    def test_stale_pruning(self, sqlite_vector_store):
        """Simulates pruning stale files that no longer exist on disk."""
        store = sqlite_vector_store
        embedder = MockEmbedder(dimension=16)

        store.add(["old"], embedder.embed(["old"]), file_path="/old.py")
        store.upsert_file("/old.py", 1.0, "h1")
        store.add(["new"], embedder.embed(["new"]), file_path="/new.py")
        store.upsert_file("/new.py", 2.0, "h2")

        # Simulate pruning: /old.py no longer on disk
        seen = {"/new.py"}
        for tracked in store.get_tracked_files():
            if tracked not in seen:
                store.remove_file_chunks(tracked)

        assert len(store) == 1
        assert store.get_tracked_files() == ["/new.py"]

    def test_incremental_skip(self, sqlite_vector_store, tmp_path):
        """File with same mtime is skipped during incremental check."""
        store = sqlite_vector_store
        f = tmp_path / "code.py"
        f.write_text("print('hello')")
        mtime = f.stat().st_mtime

        import hashlib
        fhash = hashlib.md5(f.read_bytes()).hexdigest()
        store.upsert_file(str(f), mtime, fhash)

        # Same mtime → should be skipped
        info = store.get_file_info(str(f))
        assert info is not None
        assert info[0] == mtime


# ---------------------------------------------------------------------------
# Embedding BLOB size test
# ---------------------------------------------------------------------------


class TestEmbeddingBlob:
    def test_dimension_sizes(self):
        """64-dim = 256 bytes, 384-dim = 1536 bytes."""
        blob_64 = _pack_embedding([0.0] * 64)
        assert len(blob_64) == 256  # 64 * 4 bytes

        blob_384 = _pack_embedding([0.0] * 384)
        assert len(blob_384) == 1536  # 384 * 4 bytes


# ---------------------------------------------------------------------------
# SearchCodebaseTool tests
# ---------------------------------------------------------------------------


class TestSearchCodebaseTool:
    def test_returns_results(self, sqlite_vector_store):
        from src.tools.search import SearchCodebaseTool

        embedder = MockEmbedder(dimension=32)
        store = sqlite_vector_store
        store.add(
            ["def hello(): pass", "class Dog: pass", "import os"],
            embedder.embed(["def hello(): pass", "class Dog: pass", "import os"]),
            metadata=[{"source": "a.py"}, {"source": "b.py"}, {"source": "c.py"}],
        )

        tool = SearchCodebaseTool(store, embedder)
        result = tool.execute({"query": "hello function"})
        assert "score=" in result
        assert "a.py" in result or "b.py" in result or "c.py" in result

    def test_no_results(self, sqlite_vector_store):
        from src.tools.search import SearchCodebaseTool

        embedder = MockEmbedder(dimension=32)
        tool = SearchCodebaseTool(sqlite_vector_store, embedder)
        result = tool.execute({"query": "anything"})
        assert "No results" in result

    def test_register_search_tools(self, sqlite_vector_store):
        from src.tools.base import ToolRegistry
        from src.tools.search import register_search_tools

        embedder = MockEmbedder(dimension=32)
        registry = ToolRegistry()
        register_search_tools(registry, sqlite_vector_store, embedder)
        assert "search_codebase" in registry.names()
