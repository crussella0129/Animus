"""Embedding providers: mock (for tests) and native (sentence-transformers)."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class MockEmbedder(Embedder):
    """Deterministic embedder for testing — produces consistent vectors from text hashes."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a deterministic unit vector from text hash."""
        h = hashlib.sha256(text.encode()).hexdigest()
        raw = []
        for i in range(self._dimension):
            # Use 2 hex chars per dimension value
            idx = (i * 2) % len(h)
            val = int(h[idx : idx + 2], 16) / 255.0
            raw.append(val - 0.5)  # center around 0
        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


class NativeEmbedder(Embedder):
    """Sentence-transformers embedder. Lazy-loads the model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim or 384

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with: pip install animus[embeddings]"
            )
        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
