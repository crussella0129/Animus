"""LLM module - Model providers (LiteLLM, Native, TensorRT-LLM, API)."""

from src.llm.base import (
    ModelProvider,
    ProviderType,
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
)
# Lazy imports for heavy providers
from src.llm.factory import create_provider, get_default_provider, get_available_provider

__all__ = [
    # Base classes
    "ModelProvider",
    "ProviderType",
    "Message",
    "GenerationConfig",
    "GenerationResult",
    "ModelInfo",
    # Providers (lazy-loaded via factory)
    "LiteLLMProvider",
    "NativeProvider",
    "TRTLLMProvider",
    "APIProvider",
    # Availability flags (lazy-loaded)
    "LLAMA_CPP_AVAILABLE",
    "HF_HUB_AVAILABLE",
    "LITELLM_AVAILABLE",
    # Factory
    "create_provider",
    "get_default_provider",
    "get_available_provider",
]


def __getattr__(name: str):
    """Lazy-load providers and availability flags."""
    if name == "LiteLLMProvider":
        from src.llm.litellm_provider import LiteLLMProvider
        return LiteLLMProvider
    if name == "NativeProvider":
        from src.llm.native import NativeProvider
        return NativeProvider
    if name == "TRTLLMProvider":
        from src.llm.trtllm import TRTLLMProvider
        return TRTLLMProvider
    if name == "APIProvider":
        from src.llm.api import APIProvider
        return APIProvider
    if name == "LLAMA_CPP_AVAILABLE":
        from src.llm.native import LLAMA_CPP_AVAILABLE
        return LLAMA_CPP_AVAILABLE
    if name == "HF_HUB_AVAILABLE":
        from src.llm.native import HF_HUB_AVAILABLE
        return HF_HUB_AVAILABLE
    if name == "LITELLM_AVAILABLE":
        try:
            import litellm  # noqa: F401
            return True
        except ImportError:
            return False
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
