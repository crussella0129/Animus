"""Vector store for document embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import hashlib


@dataclass
class Document:
    """A document with embedding."""
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search result with similarity score."""
    document: Document
    score: float
    distance: Optional[float] = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(self, documents: list[Document]) -> None:
        """Add documents to the store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count documents in the store."""
        ...


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store.

    Useful for testing and small datasets.
    For production, use ChromaDB or similar.
    """

    def __init__(self):
        self._documents: dict[str, Document] = {}

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the store."""
        for doc in documents:
            self._documents[doc.id] = doc

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        results = []

        for doc in self._documents.values():
            if doc.embedding is None:
                continue

            # Apply filter
            if filter:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue

            score = self._cosine_similarity(query_embedding, doc.embedding)
            results.append(SearchResult(document=doc, score=score))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        for id in ids:
            self._documents.pop(id, None)

    async def count(self) -> int:
        """Count documents in the store."""
        return len(self._documents)

    async def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()


class ChromaVectorStore(VectorStore):
    """
    Vector store using ChromaDB.

    Requires chromadb to be installed.
    """

    def __init__(
        self,
        collection_name: str = "animus",
        persist_directory: Optional[Path] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                if self.persist_directory:
                    self._client = chromadb.PersistentClient(
                        path=str(self.persist_directory),
                        settings=Settings(anonymized_telemetry=False),
                    )
                else:
                    self._client = chromadb.Client(
                        Settings(anonymized_telemetry=False)
                    )

                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise RuntimeError(
                    "ChromaDB is not installed. "
                    "Install it with: pip install chromadb"
                )

        return self._client

    async def add(self, documents: list[Document]) -> None:
        """Add documents to ChromaDB."""
        self._get_client()

        ids = []
        embeddings = []
        contents = []
        metadatas = []

        for doc in documents:
            if doc.embedding is None:
                continue
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            contents.append(doc.content)
            metadatas.append(doc.metadata)

        if ids:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar documents."""
        self._get_client()

        where = filter if filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                doc = Document(
                    id=id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert distance to similarity (cosine distance -> similarity)
                score = 1 - distance

                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                ))

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents from ChromaDB."""
        self._get_client()
        self._collection.delete(ids=ids)

    async def count(self) -> int:
        """Count documents in ChromaDB."""
        self._get_client()
        return self._collection.count()


def create_document_id(content: str, source: str) -> str:
    """Create a deterministic document ID."""
    hash_input = f"{source}:{content[:1000]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
