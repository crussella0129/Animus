"""Provider factory for creating model providers by name."""

from __future__ import annotations

import os
from typing import Optional

from src.llm.base import ModelProvider


class ProviderFactory:
    """Create model providers by name with fallback chain."""

    _PROVIDERS = ("native", "openai", "anthropic")

    def provider_names(self) -> tuple[str, ...]:
        return self._PROVIDERS

    def create(self, name: str, **kwargs) -> Optional[ModelProvider]:
        """Create a provider by name. Returns None if name is unknown."""
        if name == "native":
            return self._create_native(**kwargs)
        elif name == "openai":
            return self._create_openai(**kwargs)
        elif name == "anthropic":
            return self._create_anthropic(**kwargs)
        return None

    def _create_native(self, **kwargs) -> ModelProvider:
        from src.llm.native import NativeProvider

        return NativeProvider(
            model_path=kwargs.get("model_path", ""),
            context_length=kwargs.get("context_length", 4096),
            gpu_layers=kwargs.get("gpu_layers", -1),
            size_tier=kwargs.get("size_tier", "auto"),
        )

    def _create_openai(self, **kwargs) -> ModelProvider:
        from src.llm.api import OpenAIProvider

        return OpenAIProvider(
            base_url=kwargs.get("base_url", os.environ.get("ANIMUS_OPENAI_BASE_URL", "https://api.openai.com/v1")),
            api_key=kwargs.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            model_name=kwargs.get("model_name", os.environ.get("ANIMUS_OPENAI_MODEL", "gpt-4")),
            context_length=kwargs.get("context_length", 8192),
        )

    def _create_anthropic(self, **kwargs) -> ModelProvider:
        from src.llm.api import AnthropicProvider

        return AnthropicProvider(
            api_key=kwargs.get("api_key", os.environ.get("ANTHROPIC_API_KEY", "")),
            model_name=kwargs.get("model_name", os.environ.get("ANIMUS_ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")),
            context_length=kwargs.get("context_length", 200_000),
            max_tokens=kwargs.get("max_tokens", 4096),
        )

    def create_with_fallback(self, preferred: str, **kwargs) -> Optional[ModelProvider]:
        """Try preferred provider first, then fall back through the chain."""
        provider = self.create(preferred, **kwargs)
        if provider and provider.available():
            return provider
        for name in self._PROVIDERS:
            if name == preferred:
                continue
            provider = self.create(name, **kwargs)
            if provider and provider.available():
                return provider
        return None
