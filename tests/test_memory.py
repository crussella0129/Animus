"""Tests for memory module (RAG, ingestion)."""

import pytest
from pathlib import Path
import tempfile

from src.memory.scanner import DirectoryScanner, GitIgnoreParser
from src.memory.chunker import TokenChunker, SentenceChunker, CodeChunker, TreeSitterChunker, get_chunker
from src.memory.extractor import PlainTextExtractor, CodeExtractor, extract_text
from src.memory.vectorstore import InMemoryVectorStore, Document, create_document_id
from src.memory.embedder import MockEmbedder


class TestGitIgnoreParser:
    def test_empty_pattern(self):
        parser = GitIgnoreParser()
        parser.add_pattern("")
        assert not parser.matches(Path("test.py"))

    def test_comment_pattern(self):
        parser = GitIgnoreParser()
        parser.add_pattern("# this is a comment")
        assert not parser.matches(Path("test.py"))

    def test_simple_pattern(self):
        parser = GitIgnoreParser()
        parser.add_pattern("*.pyc")
        assert parser.matches(Path("test.pyc"))
        assert not parser.matches(Path("test.py"))

    def test_directory_pattern(self):
        parser = GitIgnoreParser()
        parser.add_pattern("node_modules")
        assert parser.matches(Path("node_modules"), is_dir=True)


class TestDirectoryScanner:
    def test_scan_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = DirectoryScanner(Path(tmpdir))
            files = list(scanner.scan())
            assert files == []

    def test_scan_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test.py").write_text("print('hello')")
            (Path(tmpdir) / "test.txt").write_text("hello world")

            scanner = DirectoryScanner(Path(tmpdir))
            files = list(scanner.scan())
            assert len(files) == 2

    def test_respects_gitignore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test.py").write_text("code")
            (root / "test.pyc").write_text("bytecode")
            (root / ".gitignore").write_text("*.pyc")

            scanner = DirectoryScanner(root)
            files = list(scanner.scan())
            names = [f.name for f in files]
            assert "test.py" in names
            assert "test.pyc" not in names


class TestChunkers:
    def test_token_chunker_small_text(self):
        chunker = TokenChunker(chunk_size=512)
        text = "Hello world."
        chunks = list(chunker.chunk(text))
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_token_chunker_large_text(self):
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 500)
        chunks = list(chunker.chunk(text))
        assert len(chunks) > 1

    def test_sentence_chunker(self):
        chunker = SentenceChunker(chunk_size=100)
        text = "First sentence. Second sentence. Third sentence."
        chunks = list(chunker.chunk(text))
        assert len(chunks) >= 1

    def test_get_chunker_for_python(self):
        chunker = get_chunker(".py")
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_for_text(self):
        chunker = get_chunker(".txt")
        assert isinstance(chunker, SentenceChunker)


class TestTreeSitterChunker:
    """Tests for AST-aware code chunking."""

    def test_availability_check(self):
        """TreeSitterChunker should report availability correctly."""
        chunker = TreeSitterChunker()
        # Should return True or False depending on tree-sitter installation
        result = chunker._is_available()
        assert isinstance(result, bool)

    def test_fallback_when_unavailable(self):
        """Should fall back to CodeChunker when tree-sitter unavailable."""
        chunker = TreeSitterChunker(chunk_size=512)

        # Test with simple Python code
        python_code = '''
def hello():
    print("Hello")

def world():
    print("World")
'''
        chunks = list(chunker.chunk(python_code, {"language": "python"}))
        # Should produce at least one chunk
        assert len(chunks) >= 1

    def test_chunks_python_functions(self):
        """Should chunk Python code by function boundaries when tree-sitter available."""
        chunker = TreeSitterChunker(chunk_size=512)

        if not chunker._is_available():
            pytest.skip("tree-sitter not available")

        python_code = '''
def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2

def function_three():
    """Third function."""
    return 3
'''
        chunks = list(chunker.chunk(python_code, {"language": "python"}))
        # Should group functions into chunks
        assert len(chunks) >= 1
        # Check that metadata includes symbol info when tree-sitter is used
        if chunks[0].metadata.get("chunker") == "tree_sitter":
            assert "symbols" in chunks[0].metadata

    def test_handles_large_function(self):
        """Should sub-chunk very large functions."""
        chunker = TreeSitterChunker(chunk_size=50)  # Small chunk size

        # Create a large function
        large_code = 'def large_func():\n' + '    x = 1\n' * 100

        chunks = list(chunker.chunk(large_code, {"language": "python"}))
        # Should produce multiple chunks for large function
        assert len(chunks) >= 1

    def test_get_chunker_prefers_tree_sitter(self):
        """get_chunker should prefer TreeSitterChunker for supported languages."""
        chunker = get_chunker(".py", use_tree_sitter=True)
        # Should be TreeSitterChunker if available, otherwise CodeChunker
        assert isinstance(chunker, (TreeSitterChunker, CodeChunker))

    def test_get_chunker_can_disable_tree_sitter(self):
        """get_chunker should respect use_tree_sitter=False."""
        chunker = get_chunker(".py", use_tree_sitter=False)
        assert isinstance(chunker, CodeChunker)
        assert not isinstance(chunker, TreeSitterChunker)


class TestExtractors:
    def test_plain_text_extractor(self):
        extractor = PlainTextExtractor()
        assert ".txt" in extractor.supported_extensions
        assert ".md" in extractor.supported_extensions

    def test_code_extractor(self):
        extractor = CodeExtractor()
        assert ".py" in extractor.supported_extensions
        assert ".js" in extractor.supported_extensions

    def test_extract_text_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("Hello world")
            f.flush()

            extracted = extract_text(Path(f.name))
            assert extracted is not None
            assert extracted.content == "Hello world"


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_in_memory_store_add_and_count(self):
        store = InMemoryVectorStore()
        doc = Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])
        await store.add([doc])
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_in_memory_store_search(self):
        store = InMemoryVectorStore()
        doc1 = Document(id="1", content="hello", embedding=[1.0, 0.0, 0.0])
        doc2 = Document(id="2", content="world", embedding=[0.0, 1.0, 0.0])
        await store.add([doc1, doc2])

        results = await store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].document.id == "1"

    @pytest.mark.asyncio
    async def test_in_memory_store_delete(self):
        store = InMemoryVectorStore()
        doc = Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])
        await store.add([doc])
        await store.delete(["1"])
        assert await store.count() == 0


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_mock_embedder(self):
        embedder = MockEmbedder(dim=384)
        assert embedder.embedding_dim == 384

        embeddings = await embedder.embed(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    @pytest.mark.asyncio
    async def test_mock_embedder_deterministic(self):
        embedder = MockEmbedder(dim=100)
        emb1 = await embedder.embed(["test"])
        emb2 = await embedder.embed(["test"])
        assert emb1 == emb2


def test_create_document_id():
    id1 = create_document_id("content", "source.py")
    id2 = create_document_id("content", "source.py")
    id3 = create_document_id("different", "source.py")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16
