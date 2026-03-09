"""API providers for OpenAI and Anthropic-compatible endpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from typing import Any

import httpx

from src.llm.base import ModelCapabilities, ModelProvider

_RETRYABLE_STATUS_CODES = frozenset({429, 503, 529})
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles each attempt (1s, 2s, 4s)


def _retry_with_backoff(func):
    """Call func(), retrying on transient HTTP errors (429, 503, 529).

    Applies exponential backoff: 1s -> 2s -> 4s between retries.
    Non-retryable errors (400, 401, 404, etc.) propagate immediately.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_DELAY * (2 ** attempt))
    raise last_exc


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

        def _call():
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                return resp

        response = _retry_with_backoff(_call)
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
        return ""

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        # TODO: streaming retry is out of scope — retrying mid-stream is complex
        # because the response has already started and partial data may have been
        # yielded. A full solution would require buffering or a restart mechanism.
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError):
                        continue

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

        def _call():
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/messages",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                return resp

        response = _retry_with_backoff(_call)
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

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        # TODO: streaming retry is out of scope — retrying mid-stream is complex
        # because the response has already started and partial data may have been
        # yielded. A full solution would require buffering or a restart mechanism.
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
            "stream": True,
        }
        if system_text:
            payload["system"] = system_text
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if tools:
            payload["tools"] = self._convert_tools(tools)

        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/messages",
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue

    def available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            context_length=self._context_length,
            size_tier="large",
            supports_tools=True,
            supports_json_mode=True,
        )
