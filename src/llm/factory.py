"""Factory for creating LLM providers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.config import AnimusConfig
from src.llm.base import ModelProvider, ProviderType


def create_provider(
    provider_type: str | ProviderType,
    config: Optional[AnimusConfig] = None,
) -> ModelProvider:
    """
    Create an LLM provider based on type and configuration.

    Args:
        provider_type: Type of provider to create.
        config: Animus configuration. Uses defaults if not provided.

    Returns:
        Configured ModelProvider instance.

    Raises:
        ValueError: If provider_type is not recognized.
    """
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError:
            raise ValueError(f"Unknown provider type: {provider_type}")

    if config is None:
        from src.core.config import ConfigManager
        config = ConfigManager().config

    if provider_type == ProviderType.LITELLM:
        from src.llm.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(
            models_dir=config.native.models_dir,
            api_key=config.model.api_key,
            api_base=config.model.api_base,
            n_gpu_layers=config.native.n_gpu_layers,
            n_ctx=config.native.n_ctx,
            server_bin_dir=config.native.bin_dir,
        )

    elif provider_type == ProviderType.NATIVE:
        from src.llm.native import NativeProvider

        return NativeProvider(
            models_dir=config.native.models_dir,
            n_ctx=config.native.n_ctx,
            n_batch=config.native.n_batch,
            n_threads=config.native.n_threads,
            n_gpu_layers=config.native.n_gpu_layers,
            use_mmap=config.native.use_mmap,
            use_mlock=config.native.use_mlock,
            verbose=config.native.verbose,
        )

    elif provider_type == ProviderType.TRTLLM:
        from src.llm.trtllm import TRTLLMProvider

        engine_dir = config.data_dir / "engines"
        tokenizer_dir = config.data_dir / "tokenizers"
        return TRTLLMProvider(
            engine_dir=engine_dir,
            tokenizer_dir=tokenizer_dir,
        )

    elif provider_type == ProviderType.API:
        from src.llm.api import APIProvider

        return APIProvider(
            api_base=config.model.api_base or "https://api.openai.com/v1",
            api_key=config.model.api_key,
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def get_default_provider(config: Optional[AnimusConfig] = None) -> ModelProvider:
    """
    Get the default provider based on configuration.

    Args:
        config: Animus configuration. Uses defaults if not provided.

    Returns:
        The default ModelProvider based on config.model.provider.
    """
    if config is None:
        from src.core.config import ConfigManager
        config = ConfigManager().config

    return create_provider(config.model.provider, config)


async def get_available_provider(config: Optional[AnimusConfig] = None) -> Optional[ModelProvider]:
    """
    Get the first available provider.

    Tries providers in order of preference:
    1. LiteLLM (unified provider — local + API)
    2. Configured default provider
    3. Native (legacy llama-cpp-python)
    4. TensorRT-LLM (Jetson)
    5. API (if configured)

    Args:
        config: Animus configuration.

    Returns:
        First available ModelProvider, or None if none available.
    """
    if config is None:
        from src.core.config import ConfigManager
        config = ConfigManager().config

    # Try LiteLLM first (preferred)
    try:
        from src.llm.litellm_provider import LiteLLMProvider

        litellm_provider = LiteLLMProvider(
            models_dir=config.native.models_dir,
            api_key=config.model.api_key,
            api_base=config.model.api_base,
            n_gpu_layers=config.native.n_gpu_layers,
            n_ctx=config.native.n_ctx,
            server_bin_dir=config.native.bin_dir,
        )
        if litellm_provider.is_available:
            models = await litellm_provider.list_models()
            if models or config.model.api_key:
                return litellm_provider
    except Exception:
        pass

    # Try configured default
    try:
        default = get_default_provider(config)
        if default.is_available:
            return default
    except Exception:
        pass

    # Try Native (legacy)
    try:
        from src.llm.native import NativeProvider

        native = NativeProvider(
            models_dir=config.native.models_dir,
            n_ctx=config.native.n_ctx,
            n_batch=config.native.n_batch,
            n_threads=config.native.n_threads,
            n_gpu_layers=config.native.n_gpu_layers,
        )
        if native.is_available:
            models = await native.list_models()
            if models:
                return native
    except Exception:
        pass

    # Try TensorRT-LLM
    try:
        from src.llm.trtllm import TRTLLMProvider

        trtllm = TRTLLMProvider(engine_dir=config.data_dir / "engines")
        if trtllm.is_available:
            return trtllm
    except Exception:
        pass

    # Try API if configured
    if config.model.api_key:
        try:
            from src.llm.api import APIProvider

            api = APIProvider(
                api_base=config.model.api_base or "https://api.openai.com/v1",
                api_key=config.model.api_key,
            )
            if api.is_available:
                return api
        except Exception:
            pass

    return None
