"""TensorRT-LLM provider for NVIDIA Jetson and GPU inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator, Optional, Any

from src.llm.base import (
    ModelProvider,
    ProviderType,
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
)


class TRTLLMProvider(ModelProvider):
    """
    Provider for TensorRT-LLM inference.

    Optimized for NVIDIA Jetson (Orin Nano) and discrete GPUs.
    Loads pre-compiled TensorRT engine files for maximum performance.

    Note: This is a placeholder implementation. Full TensorRT-LLM integration
    requires the tensorrt_llm package which is platform-specific.
    """

    def __init__(
        self,
        engine_dir: Optional[Path] = None,
        tokenizer_dir: Optional[Path] = None,
    ):
        """
        Initialize the TensorRT-LLM provider.

        Args:
            engine_dir: Directory containing TensorRT engine files.
            tokenizer_dir: Directory containing tokenizer files.
        """
        self.engine_dir = Path(engine_dir) if engine_dir else None
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self._engine = None
        self._tokenizer = None
        self._is_loaded = False

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TRTLLM

    @property
    def is_available(self) -> bool:
        """Check if TensorRT-LLM is available."""
        try:
            # Check if tensorrt_llm is installed
            import tensorrt_llm  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_jetson(self) -> bool:
        """Check if running on Jetson hardware."""
        jetson_release = Path("/etc/nv_tegra_release")
        return jetson_release.exists()

    async def list_models(self) -> list[ModelInfo]:
        """List available TensorRT engine files."""
        models = []

        if self.engine_dir and self.engine_dir.exists():
            for engine_file in self.engine_dir.glob("*.engine"):
                models.append(ModelInfo(
                    name=engine_file.stem,
                    provider=ProviderType.TRTLLM,
                    size_bytes=engine_file.stat().st_size,
                ))

        return models

    async def model_exists(self, model_name: str) -> bool:
        """Check if an engine file exists."""
        if not self.engine_dir:
            return False

        engine_path = self.engine_dir / f"{model_name}.engine"
        return engine_path.exists()

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """
        TensorRT models cannot be pulled - they must be compiled locally.

        This method provides instructions for building an engine.
        """
        yield {
            "status": "error",
            "message": (
                "TensorRT-LLM models must be compiled locally. "
                "Use 'animus build-engine' to compile a model for your hardware."
            ),
        }
        yield {
            "status": "info",
            "message": f"To build {model_name}, download the model weights and run:",
        }
        yield {
            "status": "info",
            "message": f"  python build.py --model_dir /path/to/{model_name} --output_dir {self.engine_dir}",
        }

    async def _load_engine(self, model_name: str) -> None:
        """Load a TensorRT engine."""
        if not self.is_available:
            raise RuntimeError("TensorRT-LLM is not installed")

        if not self.engine_dir:
            raise RuntimeError("Engine directory not configured")

        engine_path = self.engine_dir / f"{model_name}.engine"
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        # Placeholder for actual TensorRT-LLM engine loading
        # In production, this would use tensorrt_llm.runtime
        self._is_loaded = True

    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a response using TensorRT-LLM."""
        if not self.is_available:
            raise RuntimeError("TensorRT-LLM is not installed")

        config = config or GenerationConfig()

        # Load engine if not loaded
        if not self._is_loaded:
            await self._load_engine(model)

        # Placeholder implementation
        # In production, this would:
        # 1. Tokenize the input messages
        # 2. Run inference through the TensorRT engine
        # 3. Decode the output tokens

        raise NotImplementedError(
            "TensorRT-LLM inference requires platform-specific setup. "
            "See docs/trtllm-setup.md for instructions."
        )

    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using TensorRT-LLM."""
        if not self.is_available:
            raise RuntimeError("TensorRT-LLM is not installed")

        config = config or GenerationConfig()

        # Load engine if not loaded
        if not self._is_loaded:
            await self._load_engine(model)

        # Placeholder - would yield tokens as they're generated
        raise NotImplementedError(
            "TensorRT-LLM streaming requires platform-specific setup. "
            "See docs/trtllm-setup.md for instructions."
        )
        yield  # Make this a generator

    async def health_check(self) -> bool:
        """Check if TensorRT-LLM is properly configured."""
        if not self.is_available:
            return False

        if not self.engine_dir or not self.engine_dir.exists():
            return False

        # Check for at least one engine file
        engines = list(self.engine_dir.glob("*.engine"))
        return len(engines) > 0
