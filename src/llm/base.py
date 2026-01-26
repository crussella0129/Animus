"""Base classes for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional, Any


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    TRTLLM = "trtllm"
    API = "api"


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: int = 40
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False


@dataclass
class GenerationResult:
    """Result from text generation."""
    content: str
    finish_reason: str  # "stop", "length", "tool_calls"
    usage: Optional[dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    tool_calls: Optional[list[dict[str, Any]]] = None


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    provider: ProviderType
    size_bytes: Optional[int] = None
    parameter_count: Optional[str] = None  # e.g., "7B", "13B"
    quantization: Optional[str] = None  # e.g., "Q4_K_M", "FP16"
    context_length: Optional[int] = None
    modified_at: Optional[str] = None


class ModelProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement these methods to provide a unified
    interface for model inference across Ollama, TensorRT-LLM, and APIs.
    """

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and can accept requests."""
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """
        List all available models from this provider.

        Returns:
            List of ModelInfo objects describing available models.
        """
        ...

    @abstractmethod
    async def model_exists(self, model_name: str) -> bool:
        """
        Check if a specific model exists/is available.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if the model is available, False otherwise.
        """
        ...

    @abstractmethod
    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """
        Pull/download a model.

        Args:
            model_name: Name of the model to pull.

        Yields:
            Progress updates as dicts with status, completed, total, etc.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response from the model.

        Args:
            messages: List of conversation messages.
            model: Name of the model to use.
            config: Generation configuration options.

        Returns:
            GenerationResult containing the model's response.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the model.

        Args:
            messages: List of conversation messages.
            model: Name of the model to use.
            config: Generation configuration options.

        Yields:
            Text chunks as they are generated.
        """
        ...

    async def health_check(self) -> bool:
        """
        Perform a health check on the provider.

        Returns:
            True if the provider is healthy, False otherwise.
        """
        return self.is_available
