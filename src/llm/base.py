"""Model provider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelCapabilities:
    """Describes what a loaded model can do and its resource profile."""

    context_length: int = 4096
    parameter_count_b: float = 0.0  # billions, 0 = unknown
    size_tier: str = "unknown"  # small (<4B), medium (4-13B), large (13B+)
    supports_tools: bool = False
    supports_json_mode: bool = False

    @classmethod
    def from_parameter_count(cls, params_b: float, context_length: int = 4096) -> "ModelCapabilities":
        if params_b < 4:
            tier = "small"
        elif params_b < 13:
            tier = "medium"
        else:
            tier = "large"
        return cls(
            context_length=context_length,
            parameter_count_b=params_b,
            size_tier=tier,
            supports_tools=params_b >= 7,
            supports_json_mode=params_b >= 3,
        )


class ModelProvider(ABC):
    """Abstract base for all model providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model."""
        ...

    @abstractmethod
    def available(self) -> bool:
        """Check if this provider is ready to use."""
        ...

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return capabilities of the currently loaded model."""
        ...

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream response chunks from the model. Yields str chunks.

        Default implementation falls back to generate() as a single chunk.
        """
        yield self.generate(messages, tools=tools, **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings. Optional — not all providers support this."""
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")

    def pull(self, model_name: str) -> None:
        """Pull/download a model. Optional."""
        raise NotImplementedError(f"{type(self).__name__} does not support pull")
