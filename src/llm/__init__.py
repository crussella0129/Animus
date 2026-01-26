"""LLM module - Model providers (Native, Ollama, TensorRT-LLM, API)."""

from src.llm.base import (
    ModelProvider,
    ProviderType,
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
)
from src.llm.native import NativeProvider, LLAMA_CPP_AVAILABLE, HF_HUB_AVAILABLE
from src.llm.ollama import OllamaProvider
from src.llm.trtllm import TRTLLMProvider
from src.llm.api import APIProvider
from src.llm.factory import create_provider, get_default_provider, get_available_provider

__all__ = [
    # Base classes
    "ModelProvider",
    "ProviderType",
    "Message",
    "GenerationConfig",
    "GenerationResult",
    "ModelInfo",
    # Providers
    "NativeProvider",
    "OllamaProvider",
    "TRTLLMProvider",
    "APIProvider",
    # Availability flags
    "LLAMA_CPP_AVAILABLE",
    "HF_HUB_AVAILABLE",
    # Factory
    "create_provider",
    "get_default_provider",
    "get_available_provider",
]
