"""Memory module - RAG, VectorDB, and ingestion logic."""

from src.memory.scanner import DirectoryScanner, GitIgnoreParser
from src.memory.chunker import (
    TextChunk,
    ChunkingStrategy,
    TokenChunker,
    SentenceChunker,
    CodeChunker,
    TreeSitterChunker,
    get_chunker,
)
from src.memory.extractor import (
    ExtractedText,
    TextExtractor,
    PlainTextExtractor,
    CodeExtractor,
    MarkdownExtractor,
    PDFExtractor,
    extract_text,
    can_extract,
)
from src.memory.embedder import (
    Embedder,
    NativeEmbedder,
    OllamaEmbedder,
    APIEmbedder,
    MockEmbedder,
    create_embedder,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)
from src.memory.vectorstore import (
    Document,
    SearchResult,
    VectorStore,
    InMemoryVectorStore,
    ChromaVectorStore,
    create_document_id,
)
from src.memory.ingest import (
    Ingester,
    IngestionProgress,
    IngestionStats,
)
from src.memory.hybrid import (
    BM25Index,
    BM25Config,
    HybridSearch,
    HybridSearchConfig,
)

__all__ = [
    # Scanner
    "DirectoryScanner",
    "GitIgnoreParser",
    # Chunker
    "TextChunk",
    "ChunkingStrategy",
    "TokenChunker",
    "SentenceChunker",
    "CodeChunker",
    "TreeSitterChunker",
    "get_chunker",
    # Extractor
    "ExtractedText",
    "TextExtractor",
    "PlainTextExtractor",
    "CodeExtractor",
    "MarkdownExtractor",
    "PDFExtractor",
    "extract_text",
    "can_extract",
    # Embedder
    "Embedder",
    "NativeEmbedder",
    "OllamaEmbedder",
    "APIEmbedder",
    "MockEmbedder",
    "create_embedder",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    # Vector Store
    "Document",
    "SearchResult",
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "create_document_id",
    # Ingestion
    "Ingester",
    "IngestionProgress",
    "IngestionStats",
    # Hybrid Search
    "BM25Index",
    "BM25Config",
    "HybridSearch",
    "HybridSearchConfig",
]
