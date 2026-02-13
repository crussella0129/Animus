"""End-to-end integration test for the ingestion pipeline.

Tests the complete flow: Scanner → Chunker → Embedder → VectorStore
"""

import tempfile
from pathlib import Path

import pytest

from src.memory.chunker import Chunker
from src.memory.embedder import NativeEmbedder
from src.memory.scanner import Scanner
from src.memory.vectorstore import create_vector_store


@pytest.fixture
def temp_project():
    """Create a temporary project directory with sample Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create sample Python files
        (root / "module_a.py").write_text("""
def hello_world():
    \"\"\"Print a greeting.\"\"\"
    print("Hello, world!")

class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def add(self, a, b):
        \"\"\"Add two numbers.\"\"\"
        return a + b

    def multiply(self, a, b):
        \"\"\"Multiply two numbers.\"\"\"
        return a * b
""")

        (root / "module_b.py").write_text("""
from module_a import Calculator

def test_calculator():
    \"\"\"Test the calculator.\"\"\"
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.multiply(4, 5) == 20

def main():
    \"\"\"Main entry point.\"\"\"
    test_calculator()
    print("All tests passed!")
""")

        # Create a subdirectory with more files
        subdir = root / "utils"
        subdir.mkdir()

        (subdir / "helpers.py").write_text("""
def format_number(n):
    \"\"\"Format a number with commas.\"\"\"
    return f"{n:,}"

def parse_config(path):
    \"\"\"Parse a configuration file.\"\"\"
    with open(path) as f:
        return f.read()
""")

        # Create a non-Python file (should be ignored)
        (root / "README.md").write_text("# Test Project\n\nThis is a test.")

        yield root


@pytest.fixture
def temp_db():
    """Create a temporary database for vector storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_vectors.db"
        yield db_path


def test_full_ingestion_pipeline(temp_project, temp_db):
    """Test the complete ingestion pipeline from scanning to storage."""
    # Step 1: Scan for Python files
    scanner = Scanner()
    files = list(scanner.scan(temp_project, "**/*.py"))

    assert len(files) == 3, "Should find 3 Python files"
    file_names = {f.name for f in files}
    assert file_names == {"module_a.py", "module_b.py", "helpers.py"}

    # Step 2: Chunk the files
    chunker = Chunker(chunk_size=512, overlap=64)
    all_chunks = []

    for file_path in files:
        text = file_path.read_text()
        chunks = chunker.chunk(text, metadata={"source": str(file_path)}, filepath=str(file_path))
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0, "Should have created chunks"

    # Verify chunks have metadata
    for chunk in all_chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        # AST chunks have 'file' instead of 'source', regex chunks have 'source'
        assert "source" in chunk["metadata"] or "file" in chunk["metadata"], \
            f"Chunk should have source or file in metadata, got: {chunk['metadata'].keys()}"
        assert "estimated_tokens" in chunk["metadata"]

    # Step 3: Generate embeddings
    embedder = NativeEmbedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == len(all_chunks), "Should have one embedding per chunk"
    assert all(len(emb) == 384 for emb in embeddings), "All embeddings should be 384-dimensional"

    # Step 4: Store in vector database
    store = create_vector_store(temp_db, dimension=384, prefer_sqlite_vec=False)

    metadata_list = [c["metadata"] for c in all_chunks]
    store.add(texts, embeddings, metadata=metadata_list)

    assert len(store) == len(all_chunks), "Store should contain all chunks"

    # Step 5: Query the vector store
    query = "how to add two numbers"
    query_embedding = embedder.embed([query])[0]
    results = store.search(query_embedding, top_k=3)

    assert len(results) > 0, "Should find results for query"

    # Verify we found relevant content (Calculator.add method)
    result_texts = [r.text for r in results]
    assert any("add" in text.lower() for text in result_texts), \
        "Should find content related to addition"

    # Step 6: Verify scores are normalized
    for result in results:
        assert 0 <= result.score <= 1, f"Score should be normalized: {result.score}"


def test_incremental_reingestion(temp_project, temp_db):
    """Test that incremental re-ingestion only processes changed files."""
    import hashlib
    import time

    # Initial ingestion
    scanner = Scanner()
    chunker = Chunker(chunk_size=512)
    embedder = NativeEmbedder()
    store = create_vector_store(temp_db, dimension=384, prefer_sqlite_vec=False)

    files = list(scanner.scan(temp_project, "**/*.py"))

    for file_path in files:
        text = file_path.read_text()
        chunks = chunker.chunk(text, metadata={"source": str(file_path)}, filepath=str(file_path))

        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed(texts)
            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
            mtime = file_path.stat().st_mtime

            store.add(
                texts,
                embeddings,
                metadata=[c["metadata"] for c in chunks],
                file_path=str(file_path),
                file_hash=file_hash,
            )
            store.upsert_file(str(file_path), mtime, file_hash)

    initial_count = len(store)
    assert initial_count > 0, "Should have chunks after initial ingestion"

    # Get tracked files
    tracked_files = store.get_tracked_files()
    assert len(tracked_files) == 3, "Should track 3 files"

    # Modify one file
    time.sleep(0.1)  # Ensure mtime changes
    modified_file = temp_project / "module_a.py"
    original_text = modified_file.read_text()
    modified_file.write_text(original_text + "\n# Modified\n")

    # Re-ingest: check which files need updating
    files_to_update = []
    for file_path in scanner.scan(temp_project, "**/*.py"):
        path_str = str(file_path)
        mtime = file_path.stat().st_mtime
        file_info = store.get_file_info(path_str)

        if file_info is None:
            files_to_update.append(file_path)
        else:
            stored_mtime, stored_hash = file_info
            if stored_mtime != mtime:
                current_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                if current_hash != stored_hash:
                    files_to_update.append(file_path)

    # Should only need to update the modified file
    assert len(files_to_update) == 1, "Only one file should need updating"
    assert files_to_update[0].name == "module_a.py"

    # Update the changed file
    for file_path in files_to_update:
        # Remove old chunks
        store.remove_file_chunks(str(file_path))

        # Re-add new chunks
        text = file_path.read_text()
        chunks = chunker.chunk(text, metadata={"source": str(file_path)}, filepath=str(file_path))

        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed(texts)
            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
            mtime = file_path.stat().st_mtime

            store.add(
                texts,
                embeddings,
                metadata=[c["metadata"] for c in chunks],
                file_path=str(file_path),
                file_hash=file_hash,
            )
            store.upsert_file(str(file_path), mtime, file_hash)

    # Verify count changed appropriately
    final_count = len(store)
    # Count may differ slightly due to re-chunking
    assert final_count > 0, "Should still have chunks after update"


def test_ast_chunking_integration(temp_project, temp_db):
    """Test that AST-based chunking produces semantic chunks."""
    scanner = Scanner()
    chunker = Chunker(chunk_size=512)

    # Get module_a.py which has classes and functions
    module_a = temp_project / "module_a.py"
    text = module_a.read_text()

    # Chunk with filepath (enables AST chunking for .py files)
    chunks = chunker.chunk(text, metadata={"source": str(module_a)}, filepath=str(module_a))

    # Verify we got semantic chunks
    assert len(chunks) > 0, "Should create chunks"

    # Check that chunks have semantic metadata (from AST parsing)
    # AST chunking should create chunks for functions and classes
    chunk_texts = [c["text"] for c in chunks]

    # Should have chunks containing class and function definitions
    has_function = any("def hello_world" in text for text in chunk_texts)
    has_class = any("class Calculator" in text for text in chunk_texts)

    # If AST chunking worked, we should have separate chunks for these
    assert has_function or has_class, "Should have chunks with functions or classes"

    # Check for AST metadata (if AST chunking succeeded)
    ast_chunks = [c for c in chunks if c["metadata"].get("chunking_method") == "ast"]
    if ast_chunks:
        # AST chunks should have additional metadata
        assert "kind" in ast_chunks[0]["metadata"], "AST chunks should have 'kind' metadata"
        assert "name" in ast_chunks[0]["metadata"], "AST chunks should have 'name' metadata"


def test_pipeline_handles_errors(temp_project, temp_db):
    """Test that the pipeline handles malformed files gracefully."""
    # Create a Python file with syntax errors
    bad_file = temp_project / "bad_syntax.py"
    bad_file.write_text("def broken(\n    # Missing closing paren and body\n")

    scanner = Scanner()
    chunker = Chunker(chunk_size=512)

    files = list(scanner.scan(temp_project, "**/*.py"))

    # Should still find all files including the broken one
    assert any(f.name == "bad_syntax.py" for f in files)

    # Chunking should handle the file (even if AST parsing fails)
    for file_path in files:
        text = file_path.read_text()
        # Should not raise an exception
        chunks = chunker.chunk(text, metadata={"source": str(file_path)}, filepath=str(file_path))
        # Should produce some chunks (fallback to regex/token-based)
        assert isinstance(chunks, list)
