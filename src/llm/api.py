"""OpenAI-compatible API provider."""

from __future__ import annotations

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


class APIProvider(ModelProvider):
    """
    Provider for OpenAI-compatible API endpoints.

    Supports OpenAI, Azure OpenAI, Anthropic, Together.ai, OpenRouter,
    and any other API following the OpenAI chat completion format.
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 120.0,
        default_model: str = "gpt-4",
    ):
        """
        Initialize the API provider.

        Args:
            api_base: Base URL for the API.
            api_key: API key for authentication.
            organization: Organization ID (OpenAI-specific).
            timeout: Request timeout in seconds.
            default_model: Default model to use if none specified.
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.organization = organization
        self.timeout = timeout
        self.default_model = default_model
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.API

    @property
    def is_available(self) -> bool:
        """Check if API credentials are configured."""
        return self.api_key is not None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> list[ModelInfo]:
        """List available models from the API."""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            response.raise_for_status()

            data = response.json()
            models = []

            for model_data in data.get("data", []):
                models.append(ModelInfo(
                    name=model_data.get("id", ""),
                    provider=ProviderType.API,
                    context_length=model_data.get("context_window"),
                ))

            return models
        except httpx.HTTPError:
            # Many APIs don't support model listing
            return []

    async def model_exists(self, model_name: str) -> bool:
        """Check if a model is available."""
        try:
            models = await self.list_models()
            return any(m.name == model_name for m in models)
        except Exception:
            # Assume the model exists if we can't check
            return True

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """API models don't need to be pulled."""
        yield {
            "status": "info",
            "message": f"API models don't need to be downloaded. Model '{model_name}' is accessed remotely.",
        }

    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a chat completion via API."""
        if not self.is_available:
            raise RuntimeError("API key not configured")

        config = config or GenerationConfig()
        client = await self._get_client()

        # Convert messages to OpenAI format
        api_messages = []
        for msg in messages:
            api_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                api_msg["name"] = msg.name
            if msg.tool_call_id:
                api_msg["tool_call_id"] = msg.tool_call_id
            api_messages.append(api_msg)

        request_body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": api_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": False,
        }

        if config.stop_sequences:
            request_body["stop"] = config.stop_sequences

        response = await client.post("/chat/completions", json=request_body)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return GenerationResult(
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage"),
            tool_calls=message.get("tool_calls"),
        )

    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion via API."""
        if not self.is_available:
            raise RuntimeError("API key not configured")

        config = config or GenerationConfig()
        client = await self._get_client()

        # Convert messages to OpenAI format
        api_messages = []
        for msg in messages:
            api_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                api_msg["name"] = msg.name
            api_messages.append(api_msg)

        request_body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": api_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": True,
        }

        if config.stop_sequences:
            request_body["stop"] = config.stop_sequences

        async with client.stream(
            "POST",
            "/chat/completions",
            json=request_body,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check if the API is reachable."""
        if not self.is_available:
            return False

        try:
            client = await self._get_client()
            # Try to list models as a health check
            response = await client.get("/models")
            return response.status_code in (200, 401, 403)  # 401/403 means API is up but auth failed
        except httpx.HTTPError:
            return False
