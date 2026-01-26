"""Document ingestion orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional, Callable
import asyncio

from src.memory.scanner import DirectoryScanner
from src.memory.extractor import extract_text, ExtractedText
from src.memory.chunker import get_chunker, TextChunk
from src.memory.embedder import Embedder, create_embedder
from src.memory.vectorstore import (
    VectorStore,
    Document,
    InMemoryVectorStore,
    ChromaVectorStore,
    create_document_id,
)


@dataclass
class IngestionProgress:
    """Progress update during ingestion."""
    stage: str  # "scanning", "extracting", "chunking", "embedding", "storing"
    current: int
    total: int
    current_file: Optional[str] = None
    message: Optional[str] = None


@dataclass
class IngestionStats:
    """Statistics from ingestion."""
    files_scanned: int
    files_processed: int
    files_skipped: int
    chunks_created: int
    embeddings_generated: int
    errors: list[tuple[str, str]]  # (file, error)


class Ingester:
    """
    Orchestrates document ingestion pipeline.

    Pipeline stages:
    1. Scan directories for files
    2. Extract text from files
    3. Chunk text into segments
    4. Generate embeddings
    5. Store in vector database
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vectorstore: Optional[VectorStore] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 32,
    ):
        """
        Initialize the ingester.

        Args:
            embedder: Embedder to use. Creates default OllamaEmbedder if None.
            vectorstore: Vector store to use. Creates ChromaVectorStore if None.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks.
            batch_size: Batch size for embedding generation.
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    async def _ensure_embedder(self) -> Embedder:
        """Ensure embedder is initialized."""
        if self.embedder is None:
            # Use "auto" to prefer native embedder, fall back to mock
            self.embedder = create_embedder("auto")
        return self.embedder

    async def _ensure_vectorstore(self, persist_dir: Optional[Path] = None) -> VectorStore:
        """Ensure vector store is initialized."""
        if self.vectorstore is None:
            try:
                self.vectorstore = ChromaVectorStore(
                    collection_name="animus",
                    persist_directory=persist_dir,
                )
            except RuntimeError:
                # ChromaDB not available, use in-memory
                self.vectorstore = InMemoryVectorStore()
        return self.vectorstore

    async def ingest(
        self,
        path: Path,
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
        persist_dir: Optional[Path] = None,
    ) -> IngestionStats:
        """
        Ingest documents from a path.

        Args:
            path: File or directory to ingest.
            progress_callback: Optional callback for progress updates.
            persist_dir: Directory to persist vector store.

        Returns:
            IngestionStats with results.
        """
        embedder = await self._ensure_embedder()
        vectorstore = await self._ensure_vectorstore(persist_dir)

        stats = IngestionStats(
            files_scanned=0,
            files_processed=0,
            files_skipped=0,
            chunks_created=0,
            embeddings_generated=0,
            errors=[],
        )

        def report(stage: str, current: int, total: int, file: str = None, msg: str = None):
            if progress_callback:
                progress_callback(IngestionProgress(
                    stage=stage,
                    current=current,
                    total=total,
                    current_file=file,
                    message=msg,
                ))

        # Stage 1: Scan
        report("scanning", 0, 0, msg="Scanning directory...")
        scanner = DirectoryScanner(path)
        files = list(scanner.scan())
        stats.files_scanned = len(files)
        report("scanning", len(files), len(files), msg=f"Found {len(files)} files")

        # Collect all chunks
        all_chunks: list[tuple[TextChunk, Path]] = []

        # Stage 2 & 3: Extract and Chunk
        for i, file_path in enumerate(files):
            report("extracting", i + 1, len(files), file=str(file_path))

            try:
                extracted = extract_text(file_path)
                if extracted is None or not extracted.content.strip():
                    stats.files_skipped += 1
                    continue

                # Get appropriate chunker
                chunker = get_chunker(
                    file_path.suffix,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )

                # Add file metadata to each chunk
                metadata = {
                    "source": str(file_path),
                    "file_type": extracted.file_type,
                    **extracted.metadata,
                }

                for chunk in chunker.chunk(extracted.content, metadata):
                    all_chunks.append((chunk, file_path))
                    stats.chunks_created += 1

                stats.files_processed += 1

            except Exception as e:
                stats.errors.append((str(file_path), str(e)))
                stats.files_skipped += 1

        report("chunking", stats.chunks_created, stats.chunks_created,
               msg=f"Created {stats.chunks_created} chunks")

        # Stage 4: Generate embeddings in batches
        documents: list[Document] = []
        total_batches = (len(all_chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(all_chunks))
            batch = all_chunks[start:end]

            report("embedding", batch_idx + 1, total_batches,
                   msg=f"Embedding batch {batch_idx + 1}/{total_batches}")

            texts = [chunk.content for chunk, _ in batch]

            try:
                embeddings = await embedder.embed(texts)
                stats.embeddings_generated += len(embeddings)

                for (chunk, file_path), embedding in zip(batch, embeddings):
                    doc_id = create_document_id(chunk.content, str(file_path))
                    documents.append(Document(
                        id=doc_id,
                        content=chunk.content,
                        embedding=embedding,
                        metadata=chunk.metadata,
                    ))

            except Exception as e:
                # Log error but continue with other batches
                for chunk, file_path in batch:
                    stats.errors.append((str(file_path), f"Embedding failed: {e}"))

        # Stage 5: Store in vector database
        if documents:
            report("storing", 0, 1, msg="Storing in vector database...")
            await vectorstore.add(documents)
            report("storing", 1, 1, msg=f"Stored {len(documents)} documents")

        return stats

    async def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
        persist_dir: Optional[Path] = None,
    ) -> list[tuple[str, float, dict]]:
        """
        Search for similar documents.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            persist_dir: Directory where vector store is persisted.

        Returns:
            List of (content, score, metadata) tuples.
        """
        embedder = await self._ensure_embedder()
        vectorstore = await self._ensure_vectorstore(persist_dir)

        # Embed query
        query_embedding = await embedder.embed_single(query)

        # Search
        results = await vectorstore.search(query_embedding, k=k, filter=filter)

        return [
            (r.document.content, r.score, r.document.metadata)
            for r in results
        ]

    async def close(self) -> None:
        """Clean up resources."""
        if self.embedder and hasattr(self.embedder, 'close'):
            await self.embedder.close()
