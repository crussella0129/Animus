"""In-memory vector store with cosine similarity search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
