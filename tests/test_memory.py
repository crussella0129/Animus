"""Tests for RAG pipeline: scanner, chunker, embedder, vectorstore."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.chunker import Chunker, _estimate_tokens
from src.memory.embedder import MockEmbedder
from src.memory.scanner import Scanner
from src.memory.vectorstore import InMemoryVectorStore, _cosine_similarity


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
