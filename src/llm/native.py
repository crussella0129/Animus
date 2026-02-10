"""Native llama-cpp-python provider for local GGUF models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.llm.base import ModelCapabilities, ModelProvider


def _estimate_params_from_filename(path: str) -> float:
    """Estimate parameter count from GGUF filename conventions like 'model-7B-Q4'."""
    name = Path(path).stem.upper()
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", name)
    if match:
        return float(match.group(1))
    return 0.0


class NativeProvider(ModelProvider):
    """Local inference via llama-cpp-python. Lazy-imports the library."""

    def __init__(
        self,
        model_path: str = "",
        context_length: int = 4096,
        gpu_layers: int = -1,
        size_tier: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._context_length = context_length
        self._gpu_layers = gpu_layers
        self._size_tier = size_tier
        self._model: Any = None  # llama_cpp.Llama instance, lazy

    def _load_model(self) -> Any:
        """Lazy-load llama-cpp-python."""
        if self._model is not None:
            return self._model

        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with: pip install animus[native]"
            )

        if not self._model_path or not Path(self._model_path).exists():
            raise RuntimeError(f"Model file not found: {self._model_path}")

        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self._context_length,
            n_gpu_layers=self._gpu_layers,
            verbose=False,
        )
        return self._model

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.7),
        )
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def available(self) -> bool:
        try:
            import llama_cpp  # type: ignore[import-untyped]  # noqa: F401

            return bool(self._model_path and Path(self._model_path).exists())
        except ImportError:
            return False

    def capabilities(self) -> ModelCapabilities:
        params_b = _estimate_params_from_filename(self._model_path)
        if self._size_tier != "auto":
            cap = ModelCapabilities.from_parameter_count(params_b, self._context_length)
            cap.size_tier = self._size_tier
            return cap
        return ModelCapabilities.from_parameter_count(params_b, self._context_length)
