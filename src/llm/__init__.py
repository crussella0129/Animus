"""LLM module - Model providers (Native, TensorRT-LLM, API)."""

from src.llm.base import (
    ModelProvider,
    ProviderType,
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
)
# Note: NativeProvider import is lazy - doesn't trigger heavy llama_cpp import
from src.llm.native import NativeProvider
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
    "TRTLLMProvider",
    "APIProvider",
    # Availability flags (lazy-loaded)
    "LLAMA_CPP_AVAILABLE",
    "HF_HUB_AVAILABLE",
    # Factory
    "create_provider",
    "get_default_provider",
    "get_available_provider",
]


# Lazy re-export of availability flags to avoid eager heavy imports
def __getattr__(name: str):
    if name == "LLAMA_CPP_AVAILABLE":
        from src.llm.native import LLAMA_CPP_AVAILABLE
        return LLAMA_CPP_AVAILABLE
    if name == "HF_HUB_AVAILABLE":
        from src.llm.native import HF_HUB_AVAILABLE
        return HF_HUB_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
