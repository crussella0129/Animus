"""API providers for OpenAI and Anthropic-compatible endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from src.llm.base import ModelCapabilities, ModelProvider


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI-compatible API endpoints."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model_name: str = "gpt-4",
        context_length: int = 8192,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._context_length = context_length
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
        return ""

    def available(self) -> bool:
        if not self._api_key:
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._base_url}/models", headers=self._headers())
                return response.status_code == 200
        except (httpx.HTTPError, httpx.ConnectError):
            return False

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            context_length=self._context_length,
            size_tier="large",
            supports_tools=True,
            supports_json_mode=True,
        )


class AnthropicProvider(ModelProvider):
    """Provider for the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "claude-sonnet-4-5-20250929",
        context_length: int = 200_000,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = "https://api.anthropic.com/v1"
        self._api_key = api_key
        self._model_name = model_name
        self._context_length = context_length
        self._max_tokens = max_tokens
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        # Anthropic expects system prompt separate from messages
        system_text = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": filtered_messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        if system_text:
            payload["system"] = system_text
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if tools:
            # Convert OpenAI tool format to Anthropic tool format
            payload["tools"] = self._convert_tools(tools)

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/messages",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        content_blocks = data.get("content", [])
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)

    def _convert_tools(self, openai_tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-calling tool format to Anthropic tool format."""
        anthropic_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "name" in tool:
                # Already in Anthropic-like format
                anthropic_tools.append(tool)
        return anthropic_tools

    def available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            context_length=self._context_length,
            size_tier="large",
            supports_tools=True,
            supports_json_mode=True,
        )
