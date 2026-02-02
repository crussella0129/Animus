"""TensorRT-LLM provider for NVIDIA Jetson and GPU inference.

This provider uses NVIDIA's TensorRT-LLM library for high-performance
inference on NVIDIA GPUs, including Jetson devices.

Requirements:
    - NVIDIA GPU with compute capability 7.0+ (Volta or newer)
    - tensorrt-llm package installed
    - For Jetson: JetPack 5.0+ with TensorRT-LLM built from source

See docs/trtllm-setup.md for detailed setup instructions.
"""

from __future__ import annotations

import asyncio
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


# Global state for lazy loading
_TRTLLM_AVAILABLE: Optional[bool] = None
_LLM_CLASS = None
_SAMPLING_PARAMS_CLASS = None


def _check_trtllm() -> bool:
    """Check if TensorRT-LLM is available and cache the result."""
    global _TRTLLM_AVAILABLE, _LLM_CLASS, _SAMPLING_PARAMS_CLASS

    if _TRTLLM_AVAILABLE is not None:
        return _TRTLLM_AVAILABLE

    try:
        from tensorrt_llm import LLM, SamplingParams
        _LLM_CLASS = LLM
        _SAMPLING_PARAMS_CLASS = SamplingParams
        _TRTLLM_AVAILABLE = True
    except ImportError:
        _TRTLLM_AVAILABLE = False

    return _TRTLLM_AVAILABLE


def _get_llm_class():
    """Get the TensorRT-LLM LLM class."""
    _check_trtllm()
    return _LLM_CLASS


def _get_sampling_params_class():
    """Get the TensorRT-LLM SamplingParams class."""
    _check_trtllm()
    return _SAMPLING_PARAMS_CLASS


class TRTLLMProvider(ModelProvider):
    """
    Provider for TensorRT-LLM inference.

    Optimized for NVIDIA Jetson (Orin Nano) and discrete GPUs.
    Supports loading models from:
    - HuggingFace Hub (auto-optimizes for your hardware)
    - Local model directories
    - Pre-compiled TensorRT engine files

    Example:
        >>> provider = TRTLLMProvider()
        >>> if provider.is_available:
        ...     result = await provider.generate(messages, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        engine_dir: Optional[Path] = None,
        tokenizer_dir: Optional[Path] = None,
        tp_size: int = 1,
        pp_size: int = 1,
    ):
        """
        Initialize the TensorRT-LLM provider.

        Args:
            model_dir: Directory containing model weights (HF format).
            engine_dir: Directory containing pre-compiled TensorRT engines.
            tokenizer_dir: Directory containing tokenizer files (defaults to model_dir).
            tp_size: Tensor parallelism size (for multi-GPU).
            pp_size: Pipeline parallelism size (for multi-GPU).
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.engine_dir = Path(engine_dir) if engine_dir else None
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self.tp_size = tp_size
        self.pp_size = pp_size

        # LLM instance cache (model_name -> LLM instance)
        self._llm_instances: dict[str, Any] = {}
        self._current_model: Optional[str] = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TRTLLM

    @property
    def is_available(self) -> bool:
        """Check if TensorRT-LLM is available."""
        return _check_trtllm()

    def _check_jetson(self) -> bool:
        """Check if running on Jetson hardware."""
        jetson_release = Path("/etc/nv_tegra_release")
        return jetson_release.exists()

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to a single prompt string.

        Uses a chat template format compatible with most instruction-tuned models.
        """
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role == "user":
                prompt_parts.append(f"<|user|>\n{msg.content}</s>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg.content}</s>")

        # Add assistant prefix for generation
        prompt_parts.append("<|assistant|>\n")

        return "\n".join(prompt_parts)

    def _config_to_sampling_params(self, config: GenerationConfig) -> Any:
        """Convert GenerationConfig to TensorRT-LLM SamplingParams."""
        SamplingParams = _get_sampling_params_class()
        if SamplingParams is None:
            raise RuntimeError("TensorRT-LLM is not installed")

        params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            max_tokens=config.max_tokens,
        )

        # Add stop sequences if provided
        if config.stop_sequences:
            params.stop = config.stop_sequences

        return params

    def _get_or_create_llm(self, model: str) -> Any:
        """Get or create an LLM instance for the given model."""
        if model in self._llm_instances:
            return self._llm_instances[model]

        LLM = _get_llm_class()
        if LLM is None:
            raise RuntimeError("TensorRT-LLM is not installed")

        # Determine the model path
        model_path = model

        # Check if it's a local path
        if self.model_dir:
            local_path = self.model_dir / model
            if local_path.exists():
                model_path = str(local_path)

        # Check for pre-compiled engine
        if self.engine_dir:
            engine_path = self.engine_dir / f"{model}.engine"
            if engine_path.exists():
                model_path = str(engine_path)

        # Create LLM instance
        llm = LLM(
            model=model_path,
            tensor_parallel_size=self.tp_size,
            pipeline_parallel_size=self.pp_size,
        )

        self._llm_instances[model] = llm
        self._current_model = model

        return llm

    async def list_models(self) -> list[ModelInfo]:
        """List available TensorRT engine files and local models."""
        models = []

        # List pre-compiled engines
        if self.engine_dir and self.engine_dir.exists():
            for engine_file in self.engine_dir.glob("*.engine"):
                models.append(ModelInfo(
                    name=engine_file.stem,
                    provider=ProviderType.TRTLLM,
                    size_bytes=engine_file.stat().st_size,
                ))

        # List local model directories
        if self.model_dir and self.model_dir.exists():
            for model_dir in self.model_dir.iterdir():
                if model_dir.is_dir():
                    # Check for config.json (HF format)
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        # Calculate total size
                        total_size = sum(
                            f.stat().st_size
                            for f in model_dir.rglob("*")
                            if f.is_file()
                        )
                        models.append(ModelInfo(
                            name=model_dir.name,
                            provider=ProviderType.TRTLLM,
                            size_bytes=total_size,
                        ))

        return models

    async def model_exists(self, model_name: str) -> bool:
        """Check if an engine file or model directory exists."""
        # Check engine directory
        if self.engine_dir:
            engine_path = self.engine_dir / f"{model_name}.engine"
            if engine_path.exists():
                return True

        # Check model directory
        if self.model_dir:
            model_path = self.model_dir / model_name
            if model_path.exists() and (model_path / "config.json").exists():
                return True

        # For HuggingFace models, we assume they exist if specified
        # (TensorRT-LLM will download them automatically)
        if "/" in model_name:
            return True

        return False

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """
        Pull a model for TensorRT-LLM.

        TensorRT-LLM can automatically download from HuggingFace Hub.
        For pre-compiled engines, they must be built locally.
        """
        if not self.is_available:
            yield {
                "status": "error",
                "message": "TensorRT-LLM is not installed. See docs/trtllm-setup.md",
            }
            return

        # Check if it's a HuggingFace model
        if "/" in model_name:
            yield {
                "status": "downloading",
                "message": f"TensorRT-LLM will download {model_name} on first use",
            }
            yield {
                "status": "info",
                "message": "The model will be automatically optimized for your GPU",
            }
            yield {
                "status": "success",
                "message": f"Model {model_name} ready for use",
            }
            return

        # For local engines, provide build instructions
        yield {
            "status": "info",
            "message": (
                "TensorRT engines must be compiled locally for your specific GPU. "
                "To build an engine:"
            ),
        }
        yield {
            "status": "info",
            "message": "1. Download model weights from HuggingFace",
        }
        yield {
            "status": "info",
            "message": "2. Run: trtllm-build --model_dir <path> --output_dir <engine_dir>",
        }
        yield {
            "status": "info",
            "message": "See docs/trtllm-setup.md for detailed instructions",
        }

    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a response using TensorRT-LLM."""
        if not self.is_available:
            raise RuntimeError(
                "TensorRT-LLM is not installed. "
                "Install with: pip install tensorrt-llm "
                "See docs/trtllm-setup.md for platform-specific instructions."
            )

        config = config or GenerationConfig()

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Get sampling params
        sampling_params = self._config_to_sampling_params(config)

        # Get or create LLM instance
        llm = self._get_or_create_llm(model)

        # Run generation in executor to not block event loop
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: llm.generate([prompt], sampling_params)[0]
        )

        # Extract generated text
        generated_text = output.outputs[0].text

        # Determine finish reason
        finish_reason = "stop"
        if output.outputs[0].finish_reason:
            finish_reason = str(output.outputs[0].finish_reason).lower()

        # Build usage stats if available
        usage = None
        if hasattr(output, 'prompt_token_ids') and hasattr(output.outputs[0], 'token_ids'):
            prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
            completion_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return GenerationResult(
            content=generated_text,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using TensorRT-LLM."""
        if not self.is_available:
            raise RuntimeError(
                "TensorRT-LLM is not installed. "
                "Install with: pip install tensorrt-llm "
                "See docs/trtllm-setup.md for platform-specific instructions."
            )

        config = config or GenerationConfig()

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Get sampling params
        sampling_params = self._config_to_sampling_params(config)

        # Get or create LLM instance
        llm = self._get_or_create_llm(model)

        # Use async streaming generation
        previous_text = ""
        async for output in llm.generate_async(prompt, sampling_params, streaming=True):
            current_text = output.outputs[0].text
            # Yield only the new tokens
            if len(current_text) > len(previous_text):
                new_text = current_text[len(previous_text):]
                yield new_text
                previous_text = current_text

    async def health_check(self) -> bool:
        """Check if TensorRT-LLM is properly configured."""
        if not self.is_available:
            return False

        # Check for engine directory if configured
        if self.engine_dir and not self.engine_dir.exists():
            return False

        # Check for model directory if configured
        if self.model_dir and not self.model_dir.exists():
            return False

        return True

    def cleanup(self) -> None:
        """Clean up LLM instances to free GPU memory."""
        for llm in self._llm_instances.values():
            if hasattr(llm, '__del__'):
                try:
                    llm.__del__()
                except Exception:
                    pass
        self._llm_instances.clear()
        self._current_model = None

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()
