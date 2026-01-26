"""Ollama provider implementation."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional, Any

import httpx

from src.llm.base import (
    ModelProvider,
    ProviderType,
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
)


class OllamaProvider(ModelProvider):
    """
    Provider for Ollama local inference.

    Connects to the Ollama server running on localhost:11434 (by default).
    Supports model listing, pulling, and chat completion.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        timeout: float = 120.0,
    ):
        """
        Initialize the Ollama provider.

        Args:
            host: Ollama server host.
            port: Ollama server port.
            timeout: Request timeout in seconds.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self._base_url = f"http://{host}:{port}"
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    @property
    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._base_url}/api/version")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> list[ModelInfo]:
        """List all models available in Ollama."""
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()

        data = response.json()
        models = []

        for model_data in data.get("models", []):
            name = model_data.get("name", "")
            details = model_data.get("details", {})

            models.append(ModelInfo(
                name=name,
                provider=ProviderType.OLLAMA,
                size_bytes=model_data.get("size"),
                parameter_count=details.get("parameter_size"),
                quantization=details.get("quantization_level"),
                modified_at=model_data.get("modified_at"),
            ))

        return models

    async def model_exists(self, model_name: str) -> bool:
        """Check if a model exists in Ollama."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/show",
                json={"name": model_name},
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """
        Pull a model from the Ollama registry.

        Yields progress updates with fields:
        - status: Current status message
        - completed: Bytes downloaded (optional)
        - total: Total bytes (optional)
        - digest: Layer digest being downloaded (optional)
        """
        client = await self._get_client()

        async with client.stream(
            "POST",
            "/api/pull",
            json={"name": model_name, "stream": True},
            timeout=None,  # No timeout for large downloads
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a chat completion."""
        config = config or GenerationConfig()
        client = await self._get_client()

        # Convert messages to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = await client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "stop": config.stop_sequences if config.stop_sequences else None,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})

        return GenerationResult(
            content=message.get("content", ""),
            finish_reason="stop" if data.get("done") else "length",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            },
        )

    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion."""
        config = config or GenerationConfig()
        client = await self._get_client()

        # Convert messages to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "stop": config.stop_sequences if config.stop_sequences else None,
                },
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        message = data.get("message", {})
                        content = message.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/api/version")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
