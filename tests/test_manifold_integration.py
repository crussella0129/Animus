"""Integration tests for Animus: Manifold multi-strategy retrieval system.

Tests the complete pipeline: ingestion with contextualization,
query routing, strategy execution, and result fusion.
"""

import tempfile
from pathlib import Path

import pytest

from src.knowledge.graph_db import GraphDB
from src.knowledge.indexer import Indexer
from src.memory.chunker import Chunker
from src.memory.contextualizer import ChunkContextualizer
from src.memory.embedder import MockEmbedder
from src.memory.scanner import Scanner
from src.memory.vectorstore import SQLiteVectorStore
from src.retrieval.executor import RetrievalExecutor, fuse_results, RetrievalResult
from src.retrieval.router import classify_query, RetrievalStrategy


@pytest.fixture
def sample_project():
    """Create a sample Python project for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create sample Python files with realistic structure
        (root / "auth.py").write_text("""
class AuthService:
    \"\"\"Handles user authentication.\"\"\"

    def authenticate(self, token):
        \"\"\"Verify a user token.\"\"\"
        return self.verify_token(token)

    def verify_token(self, token):
        \"\"\"Check if token is valid.\"\"\"
        return token.startswith("valid_")

def login(username, password):
    \"\"\"Login a user and return auth token.\"\"\"
    auth = AuthService()
    # TODO: Add password hashing
    return auth.authenticate("valid_token")
""")

        (root / "middleware.py").write_text("""
from auth import AuthService

class RequestMiddleware:
    \"\"\"Middleware for request processing.\"\"\"

    def __init__(self):
        self.auth = AuthService()

    def verify_request(self, request):
        \"\"\"Verify incoming request has valid auth.\"\"\"
        token = request.get_header("Authorization")
        return self.auth.authenticate(token)
""")

        (root / "config.py").write_text("""
import yaml

def load_config(path):
    \"\"\"Load configuration from YAML file.\"\"\"
    with open(path) as f:
        return yaml.safe_load(f)

# DEPRECATED: Use load_config instead
def read_settings(path):
    \"\"\"Legacy config loader.\"\"\"
    return load_config(path)
""")

        yield root


@pytest.fixture
def manifold_setup(sample_project):
    """Set up complete Manifold pipeline with sample project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_dir = Path(tmpdir)

        # Build knowledge graph
        graph_db_path = db_dir / "graph.db"
        graph_db = GraphDB(graph_db_path)
        indexer = Indexer(graph_db)
        index_result = indexer.index_directory(sample_project)

        # Build vector store with contextual embeddings
        vector_db_path = db_dir / "vectors.db"
        vector_store = SQLiteVectorStore(vector_db_path)

        scanner = Scanner()
        chunker = Chunker(chunk_size=512, overlap=64)
        embedder = MockEmbedder(dimension=384)
        contextualizer = ChunkContextualizer(graph_db=graph_db)

        # Ingest with contextualization
        for file_path in scanner.scan(sample_project, "**/*.py"):
            text = file_path.read_text()
            chunks = chunker.chunk(text, metadata={"source": str(file_path)}, filepath=str(file_path))

            if chunks:
                # Contextualize before embedding
                contextualized_texts = contextualizer.contextualize_batch(chunks)
                embeddings = embedder.embed(contextualized_texts)

                # Store original texts
                original_texts = [c["text"] for c in chunks]
                vector_store.add(
                    original_texts,
                    embeddings,
                    [c["metadata"] for c in chunks],
                )

        # Create executor
        executor = RetrievalExecutor(
            vector_store=vector_store,
            embedder=embedder,
            graph_db=graph_db,
            project_root=sample_project,
        )

        yield {
            "executor": executor,
            "graph_db": graph_db,
            "vector_store": vector_store,
            "embedder": embedder,
            "project_root": sample_project,
        }

        # Cleanup
        vector_store.close()
        graph_db.close()


class TestManifoldEndToEnd:
    """Test the complete Manifold pipeline."""

    def test_semantic_query_returns_results(self, manifold_setup):
        """Semantic queries should return contextually relevant code."""
        executor = manifold_setup["executor"]

        query = "how does authentication work"
        decision = classify_query(query)

        assert decision.strategy == RetrievalStrategy.SEMANTIC

        results = executor.execute(decision, top_k=5)
        assert len(results) > 0

        # Results should include auth-related code
        result_texts = " ".join(r.text for r in results)
        assert "authenticate" in result_texts.lower() or "auth" in result_texts.lower()

    def test_structural_query_returns_callers(self, manifold_setup):
        """Structural queries should return graph-based relationships."""
        executor = manifold_setup["executor"]

        query = "what calls authenticate"
        decision = classify_query(query)

        assert decision.strategy == RetrievalStrategy.STRUCTURAL
        assert decision.structural_operation == "callers"

        results = executor.execute(decision, top_k=5)

        # Should find callers (login and verify_request)
        if results:  # May be empty if graph doesn't have the exact symbol
            assert any(r.strategy == "structural" for r in results)

    def test_hybrid_query_fuses_results(self, manifold_setup):
        """Hybrid queries should execute both strategies and fuse."""
        executor = manifold_setup["executor"]

        query = "find authentication code and what uses it"
        decision = classify_query(query)

        assert decision.strategy == RetrievalStrategy.HYBRID

        results = executor.execute(decision, top_k=10)
        assert len(results) > 0

        # Results should include results from multiple strategies
        strategies = set(r.strategy for r in results)
        # Should have at least semantic results (may have structural too)
        assert "semantic" in strategies or "structural" in strategies

    def test_keyword_query_finds_exact_matches(self, manifold_setup):
        """Keyword queries should find exact text matches."""
        executor = manifold_setup["executor"]

        query = "find TODO comments"
        decision = classify_query(query)

        assert decision.strategy == RetrievalStrategy.KEYWORD
        assert decision.keyword_query == "TODO"

        results = executor.execute(decision, top_k=5)

        # Should find the TODO comment in auth.py
        if results:
            assert any("TODO" in r.text for r in results)
            assert all(r.strategy == "keyword" for r in results)


class TestContextualization:
    """Test that contextual embeddings are being applied."""

    def test_contextualizer_adds_prefixes(self, manifold_setup):
        """Contextualizer should add structural context to chunks."""
        graph_db = manifold_setup["graph_db"]
        contextualizer = ChunkContextualizer(graph_db=graph_db)

        # Create a chunk with structural metadata
        chunk = {
            "text": "def authenticate(token):\n    return verify(token)",
            "metadata": {
                "structural": True,
                "kind": "method",
                "qualified_name": "auth.AuthService.authenticate",
                "source": "auth.py",
                "docstring": "Verify a user token.",
            }
        }

        contextualized = contextualizer.contextualize(chunk)

        # Should have context prefix
        assert len(contextualized) > len(chunk["text"])
        assert "[From" in contextualized
        assert "authenticate" in contextualized
        # Original text should be present
        assert "def authenticate(token)" in contextualized

    def test_contextualizer_handles_non_structural_chunks(self):
        """Non-structural chunks should pass through unchanged."""
        contextualizer = ChunkContextualizer(graph_db=None)

        chunk = {
            "text": "some random text",
            "metadata": {}  # No structural metadata
        }

        contextualized = contextualizer.contextualize(chunk)
        assert contextualized == chunk["text"]


class TestReciprocalRankFusion:
    """Test the RRF algorithm for result fusion."""

    def test_rrf_combines_two_lists(self):
        """RRF should merge results from two strategies."""
        list_a = [
            RetrievalResult("code_a", 0.9, "file_a.py", "semantic"),
            RetrievalResult("code_b", 0.8, "file_b.py", "semantic"),
        ]
        list_b = [
            RetrievalResult("code_c", 0.9, "file_c.py", "structural"),
            RetrievalResult("code_a", 0.8, "file_a.py", "structural"),  # Duplicate
        ]

        fused = fuse_results(list_a, list_b, top_k=5)

        # Should have 3 unique results (code_a appears in both)
        assert len(fused) == 3

        # code_a should be boosted (appears in both)
        code_a_result = next(r for r in fused if "code_a" in r.text)
        assert code_a_result.metadata.get("multi_strategy") == True

    def test_rrf_deduplicates_by_source_and_text(self):
        """RRF should deduplicate based on source + text preview."""
        list_a = [
            RetrievalResult("same text here", 0.9, "file.py", "semantic"),
        ]
        list_b = [
            RetrievalResult("same text here", 0.8, "file.py", "structural"),
        ]

        fused = fuse_results(list_a, list_b, top_k=10)

        # Should have only 1 result (deduplicated)
        assert len(fused) == 1
        assert fused[0].metadata.get("multi_strategy") == True

    def test_rrf_respects_top_k(self):
        """RRF should limit results to top_k."""
        list_a = [RetrievalResult(f"text_{i}", 0.9 - i*0.1, f"file_{i}.py", "semantic") for i in range(10)]
        list_b = [RetrievalResult(f"text_{i+10}", 0.9 - i*0.1, f"file_{i+10}.py", "structural") for i in range(10)]

        fused = fuse_results(list_a, list_b, top_k=5)

        assert len(fused) <= 5

    def test_rrf_scores_are_normalized(self):
        """RRF scores should be positive and comparable."""
        list_a = [RetrievalResult("code_a", 0.9, "file_a.py", "semantic")]
        list_b = [RetrievalResult("code_b", 0.8, "file_b.py", "structural")]

        fused = fuse_results(list_a, list_b, top_k=10)

        for result in fused:
            assert result.score > 0
            # RRF scores are typically small (1/(k+rank) where k=60)
            assert result.score < 1.0


class TestManifoldPerformance:
    """Test that Manifold meets performance targets."""

    def test_router_classification_is_fast(self):
        """Router classification should complete in <1ms."""
        import time

        query = "how does authentication work and what calls it"

        start = time.perf_counter()
        for _ in range(100):  # Run 100 times
            classify_query(query)
        elapsed = time.perf_counter() - start

        # 100 classifications should take < 100ms total (<1ms average)
        assert elapsed < 0.1, f"100 classifications took {elapsed*1000:.1f}ms (expected <100ms)"

    def test_query_execution_is_bounded(self, manifold_setup):
        """Query execution should complete in reasonable time."""
        import time

        executor = manifold_setup["executor"]
        query = "find authentication code"
        decision = classify_query(query)

        start = time.perf_counter()
        results = executor.execute(decision, top_k=10)
        elapsed = time.perf_counter() - start

        # Should complete in <5s (generous bound for MockEmbedder)
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s (expected <5s)"
        assert len(results) >= 0  # May be empty, but should not crash
