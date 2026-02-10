"""Tests for LLM providers — all mocked, no real inference."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm.base import ModelCapabilities, ModelProvider
from src.llm.factory import ProviderFactory


class TestModelCapabilities:
    def test_small_model(self):
        cap = ModelCapabilities.from_parameter_count(1.5)
        assert cap.size_tier == "small"
        assert cap.supports_tools is False

    def test_medium_model(self):
        cap = ModelCapabilities.from_parameter_count(7.0)
        assert cap.size_tier == "medium"
        assert cap.supports_tools is True

    def test_large_model(self):
        cap = ModelCapabilities.from_parameter_count(70.0)
        assert cap.size_tier == "large"
        assert cap.supports_tools is True
        assert cap.supports_json_mode is True


class TestNativeProvider:
    def test_estimate_params_from_filename(self):
        from src.llm.native import _estimate_params_from_filename

        assert _estimate_params_from_filename("llama-7B-Q4_K_M.gguf") == 7.0
        assert _estimate_params_from_filename("phi-3.5B-instruct.gguf") == 3.5
        assert _estimate_params_from_filename("model.gguf") == 0.0

    def test_available_no_llama_cpp(self):
        from src.llm.native import NativeProvider

        with patch.dict("sys.modules", {"llama_cpp": None}):
            provider = NativeProvider(model_path="nonexistent.gguf")
            # Should handle ImportError gracefully
            assert provider.available() is False

    def test_capabilities_auto_tier(self):
        from src.llm.native import NativeProvider

        provider = NativeProvider(model_path="model-7B-Q4.gguf", context_length=4096)
        caps = provider.capabilities()
        assert caps.size_tier == "medium"
        assert caps.context_length == 4096

    def test_capabilities_manual_tier(self):
        from src.llm.native import NativeProvider

        provider = NativeProvider(model_path="x.gguf", size_tier="small")
        caps = provider.capabilities()
        assert caps.size_tier == "small"


class TestOpenAIProvider:
    def test_available_no_key(self):
        from src.llm.api import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        assert provider.available() is False

    @patch("httpx.Client")
    def test_generate_mocked(self, mock_client_cls):
        from src.llm.api import OpenAIProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello back!"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        result = provider.generate([{"role": "user", "content": "Hello"}])
        assert result == "Hello back!"

    def test_capabilities(self):
        from src.llm.api import OpenAIProvider

        provider = OpenAIProvider(context_length=16384)
        caps = provider.capabilities()
        assert caps.context_length == 16384
        assert caps.supports_tools is True


class TestAnthropicProvider:
    def test_available_no_key(self):
        from src.llm.api import AnthropicProvider

        provider = AnthropicProvider(api_key="")
        assert provider.available() is False

    def test_available_with_key(self):
        from src.llm.api import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.available() is True

    @patch("httpx.Client")
    def test_generate_mocked(self, mock_client_cls):
        from src.llm.api import AnthropicProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "I am Claude."}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        result = provider.generate([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Who are you?"},
        ])
        assert result == "I am Claude."

        # Verify system was extracted from messages
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload.get("system") == "You are helpful."
        # Verify system message was not in the messages list
        for msg in payload["messages"]:
            assert msg["role"] != "system"

    def test_convert_tools(self):
        from src.llm.api import AnthropicProvider

        provider = AnthropicProvider(api_key="test")
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        result = provider._convert_tools(openai_tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert "input_schema" in result[0]

    def test_capabilities(self):
        from src.llm.api import AnthropicProvider

        provider = AnthropicProvider()
        caps = provider.capabilities()
        assert caps.context_length == 200_000
        assert caps.supports_tools is True


class TestProviderFactory:
    def test_provider_names(self):
        factory = ProviderFactory()
        names = factory.provider_names()
        assert "native" in names
        assert "openai" in names
        assert "anthropic" in names

    def test_create_native(self):
        factory = ProviderFactory()
        provider = factory.create("native", model_path="test.gguf")
        assert provider is not None

    def test_create_openai(self):
        factory = ProviderFactory()
        provider = factory.create("openai", api_key="test")
        assert provider is not None

    def test_create_anthropic(self):
        factory = ProviderFactory()
        provider = factory.create("anthropic", api_key="test")
        assert provider is not None

    def test_create_unknown(self):
        factory = ProviderFactory()
        assert factory.create("nonexistent") is None
