"""Tests for LLM providers."""

import pytest
from src.llm.base import (
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
    ProviderType,
)
from src.llm.native import NativeProvider
from src.llm.ollama import OllamaProvider
from src.llm.api import APIProvider
from src.llm.trtllm import TRTLLMProvider
from src.llm.factory import create_provider


def test_message_creation():
    """Test Message dataclass creation."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.name is None
    assert msg.tool_calls is None


def test_generation_config_defaults():
    """Test GenerationConfig has sensible defaults."""
    config = GenerationConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.top_p == 1.0
    assert config.stream is False


def test_model_info_creation():
    """Test ModelInfo dataclass creation."""
    info = ModelInfo(
        name="test-model",
        provider=ProviderType.OLLAMA,
        size_bytes=1024 * 1024 * 1024,
    )
    assert info.name == "test-model"
    assert info.provider == ProviderType.OLLAMA
    assert info.size_bytes == 1024 * 1024 * 1024


def test_ollama_provider_type():
    """Test OllamaProvider returns correct type."""
    provider = OllamaProvider()
    assert provider.provider_type == ProviderType.OLLAMA


def test_api_provider_type():
    """Test APIProvider returns correct type."""
    provider = APIProvider(api_key="test-key")
    assert provider.provider_type == ProviderType.API


def test_trtllm_provider_type():
    """Test TRTLLMProvider returns correct type."""
    provider = TRTLLMProvider()
    assert provider.provider_type == ProviderType.TRTLLM


def test_native_provider_type():
    """Test NativeProvider returns correct type."""
    provider = NativeProvider()
    assert provider.provider_type == ProviderType.NATIVE


def test_api_provider_not_available_without_key():
    """Test APIProvider reports unavailable without key."""
    provider = APIProvider()
    assert provider.is_available is False


def test_api_provider_available_with_key():
    """Test APIProvider reports available with key."""
    provider = APIProvider(api_key="test-key")
    assert provider.is_available is True


def test_create_provider_ollama():
    """Test factory creates OllamaProvider."""
    provider = create_provider("ollama")
    assert isinstance(provider, OllamaProvider)


def test_create_provider_api():
    """Test factory creates APIProvider."""
    provider = create_provider("api")
    assert isinstance(provider, APIProvider)


def test_create_provider_trtllm():
    """Test factory creates TRTLLMProvider."""
    provider = create_provider("trtllm")
    assert isinstance(provider, TRTLLMProvider)


def test_create_provider_native():
    """Test factory creates NativeProvider."""
    provider = create_provider("native")
    assert isinstance(provider, NativeProvider)


def test_create_provider_invalid():
    """Test factory raises for unknown provider."""
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("unknown")
