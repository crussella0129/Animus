"""Text embedding generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
import httpx


# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []


class OllamaEmbedder(Embedder):
    """
    Embedder using Ollama's embedding API.

    Uses models like nomic-embed-text or all-minilm.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "localhost",
        port: int = 11434,
    ):
        self.model = model
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"
        self._client: Optional[httpx.AsyncClient] = None
        self._dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (detected on first embed)."""
        if self._dim is None:
            # Default dimensions for common models
            defaults = {
                "nomic-embed-text": 768,
                "all-minilm": 384,
                "mxbai-embed-large": 1024,
            }
            return defaults.get(self.model, 768)
        return self._dim

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama."""
        client = await self._get_client()
        embeddings = []

        for text in texts:
            response = await client.post(
                "/api/embed",
                json={"model": self.model, "input": text},
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embeddings", [[]])[0]
            if embedding and self._dim is None:
                self._dim = len(embedding)
            embeddings.append(embedding)

        return embeddings


class APIEmbedder(Embedder):
    """
    Embedder using OpenAI-compatible embedding API.

    Supports OpenAI, Azure OpenAI, and other compatible APIs.
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(self.model, 1536)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=headers,
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using API."""
        client = await self._get_client()

        response = await client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Sort by index to maintain order
        results = sorted(data["data"], key=lambda x: x["index"])
        return [r["embedding"] for r in results]


class MockEmbedder(Embedder):
    """
    Mock embedder for testing.

    Generates random-ish embeddings based on text hash.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        import hashlib

        embeddings = []
        for text in texts:
            # Generate deterministic pseudo-random embedding
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(self._dim):
                byte_idx = i % len(hash_bytes)
                val = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Range [-1, 1]
                embedding.append(val)
            embeddings.append(embedding)

        return embeddings


class NativeEmbedder(Embedder):
    """
    Native embedder using sentence-transformers.

    Runs entirely locally without requiring any external services.
    Uses models like all-MiniLM-L6-v2 or all-mpnet-base-v2.
    """

    # Common models and their dimensions
    MODEL_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize native embedder.

        Args:
            model: Model name from sentence-transformers.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model
        self._model: Any = None
        self._device = device

    def _ensure_model(self) -> Any:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self.model_name in self.MODEL_DIMS:
            return self.MODEL_DIMS[self.model_name]
        # Try to get from loaded model
        model = self._ensure_model()
        return model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers."""
        model = self._ensure_model()
        # sentence-transformers encode is synchronous, run in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings


def create_embedder(
    provider: str = "auto",
    model: Optional[str] = None,
    **kwargs,
) -> Embedder:
    """
    Create an embedder based on provider.

    Args:
        provider: "native", "ollama", "api", "mock", or "auto" (tries native first).
        model: Model name to use.
        **kwargs: Additional arguments for the embedder.

    Returns:
        Embedder instance.
    """
    if provider == "auto":
        # Try native first (fully local), then ollama, then mock
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            return NativeEmbedder(
                model=model or "all-MiniLM-L6-v2",
                **kwargs,
            )
        # Fall back to mock if nothing else available
        return MockEmbedder(**kwargs)

    if provider == "native":
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        return NativeEmbedder(
            model=model or "all-MiniLM-L6-v2",
            **kwargs,
        )
    elif provider == "ollama":
        return OllamaEmbedder(
            model=model or "nomic-embed-text",
            **kwargs,
        )
    elif provider == "api":
        return APIEmbedder(
            model=model or "text-embedding-3-small",
            **kwargs,
        )
    elif provider == "mock":
        return MockEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
