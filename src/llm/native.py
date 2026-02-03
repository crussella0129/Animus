"""Native model provider using llama-cpp-python for direct GGUF model loading."""

from __future__ import annotations

import json
import os
import re
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


# Lazy loading for llama-cpp-python (heavy C++ library)
_LLAMA_CPP_AVAILABLE: Optional[bool] = None
_Llama = None


def _check_llama_cpp() -> bool:
    """Lazily check if llama-cpp-python is available."""
    global _LLAMA_CPP_AVAILABLE, _Llama
    if _LLAMA_CPP_AVAILABLE is None:
        try:
            from llama_cpp import Llama
            _Llama = Llama
            _LLAMA_CPP_AVAILABLE = True
        except ImportError:
            _LLAMA_CPP_AVAILABLE = False
    return _LLAMA_CPP_AVAILABLE


def _get_llama():
    """Get the Llama class, importing if needed."""
    _check_llama_cpp()
    return _Llama


# Lazy loading for huggingface_hub
_HF_HUB_AVAILABLE: Optional[bool] = None
_hf_hub_download = None
_list_repo_files = None
_HfApi = None


def _check_hf_hub() -> bool:
    """Lazily check if huggingface_hub is available."""
    global _HF_HUB_AVAILABLE, _hf_hub_download, _list_repo_files, _HfApi
    if _HF_HUB_AVAILABLE is None:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files, HfApi
            _hf_hub_download = hf_hub_download
            _list_repo_files = list_repo_files
            _HfApi = HfApi
            _HF_HUB_AVAILABLE = True
        except ImportError:
            _HF_HUB_AVAILABLE = False
    return _HF_HUB_AVAILABLE


def _get_hf_hub():
    """Get huggingface_hub functions, importing if needed."""
    _check_hf_hub()
    return _hf_hub_download, _list_repo_files, _HfApi


# Module-level __getattr__ for backward compatibility
def __getattr__(name: str):
    if name == "LLAMA_CPP_AVAILABLE":
        return _check_llama_cpp()
    if name == "HF_HUB_AVAILABLE":
        return _check_hf_hub()
    if name == "Llama":
        return _get_llama()
    if name == "hf_hub_download":
        _check_hf_hub()
        return _hf_hub_download
    if name == "list_repo_files":
        _check_hf_hub()
        return _list_repo_files
    if name == "HfApi":
        _check_hf_hub()
        return _HfApi
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Common GGUF quantization patterns
QUANTIZATION_PATTERNS = {
    "Q2_K": "Q2_K",
    "Q3_K_S": "Q3_K_S",
    "Q3_K_M": "Q3_K_M",
    "Q3_K_L": "Q3_K_L",
    "Q4_0": "Q4_0",
    "Q4_1": "Q4_1",
    "Q4_K_S": "Q4_K_S",
    "Q4_K_M": "Q4_K_M",
    "Q5_0": "Q5_0",
    "Q5_1": "Q5_1",
    "Q5_K_S": "Q5_K_S",
    "Q5_K_M": "Q5_K_M",
    "Q6_K": "Q6_K",
    "Q8_0": "Q8_0",
    "F16": "F16",
    "F32": "F32",
}


def detect_quantization(filename: str) -> Optional[str]:
    """Detect quantization type from filename."""
    upper = filename.upper()
    for pattern, quant in QUANTIZATION_PATTERNS.items():
        if pattern in upper:
            return quant
    return None


def detect_gpu_backend() -> str:
    """Detect available GPU backend for llama-cpp-python."""
    # Check CUDA
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Metal (macOS)
    import platform
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Metal" in result.stdout:
                return "metal"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check ROCm (AMD)
    if os.path.exists("/opt/rocm"):
        return "rocm"

    return "cpu"


class NativeProvider(ModelProvider):
    """
    Provider for native model loading using llama-cpp-python.

    Loads GGUF models directly without requiring external services like Ollama.
    Supports automatic GPU detection and offloading (CUDA, Metal, ROCm).
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        use_mmap: bool = True,
        use_mlock: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the Native provider.

        Args:
            models_dir: Directory containing GGUF model files.
            n_ctx: Context window size.
            n_batch: Batch size for prompt processing.
            n_threads: Number of CPU threads (None = auto).
            n_gpu_layers: Layers to offload to GPU (-1 = all, 0 = none).
            use_mmap: Use memory mapping for model loading.
            use_mlock: Lock model in RAM.
            verbose: Enable verbose logging.
        """
        self.models_dir = models_dir or (Path.home() / ".animus" / "models")
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.verbose = verbose

        self._loaded_models: dict[str, Any] = {}  # model_name -> Llama instance
        self._gpu_backend = detect_gpu_backend()

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.NATIVE

    @property
    def is_available(self) -> bool:
        """Check if llama-cpp-python is installed."""
        return _check_llama_cpp()

    @property
    def gpu_backend(self) -> str:
        """Return detected GPU backend."""
        return self._gpu_backend

    def _get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to a model file."""
        # Check if model_name is already a path
        if Path(model_name).exists():
            return Path(model_name)

        # Check in models directory
        model_path = self.models_dir / model_name
        if model_path.exists():
            return model_path

        # Check for .gguf extension
        if not model_name.endswith(".gguf"):
            model_path = self.models_dir / f"{model_name}.gguf"
            if model_path.exists():
                return model_path

        # Search for partial matches
        for file in self.models_dir.glob("*.gguf"):
            if model_name.lower() in file.name.lower():
                return file

        return None

    def _load_model(self, model_name: str) -> Any:
        """Load a model, caching for reuse."""
        if not _check_llama_cpp():
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        model_path = self._get_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(
                f"Model not found: {model_name}. "
                f"Download with: animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
            )

        # Determine GPU layers based on backend
        n_gpu_layers = self.n_gpu_layers
        if self._gpu_backend == "cpu":
            n_gpu_layers = 0

        Llama = _get_llama()
        model = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=self.use_mmap,
            use_mlock=self.use_mlock,
            verbose=self.verbose,
        )

        self._loaded_models[model_name] = model
        return model

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]

    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._loaded_models.clear()

    async def list_models(self) -> list[ModelInfo]:
        """List all GGUF models in the models directory."""
        models = []

        if not self.models_dir.exists():
            return models

        for file in self.models_dir.glob("*.gguf"):
            stat = file.stat()
            quant = detect_quantization(file.name)

            models.append(ModelInfo(
                name=file.name,
                provider=ProviderType.NATIVE,
                size_bytes=stat.st_size,
                quantization=quant,
            ))

        return sorted(models, key=lambda m: m.name)

    async def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        return self._get_model_path(model_name) is not None

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """
        Download a model from Hugging Face.

        Supports formats:
        - repo_id/filename.gguf (specific file)
        - repo_id (auto-detect best quantization)

        Yields progress updates.
        """
        if not _check_hf_hub():
            yield {
                "status": "error",
                "error": "huggingface_hub not installed. Install with: pip install huggingface-hub",
            }
            return

        yield {"status": "starting", "model": model_name}

        try:
            # Parse model name
            if "/" in model_name:
                parts = model_name.split("/")
                if len(parts) == 2:
                    repo_id = model_name
                    filename = None
                elif len(parts) >= 3:
                    repo_id = "/".join(parts[:2])
                    filename = "/".join(parts[2:])
                else:
                    raise ValueError(f"Invalid model name format: {model_name}")
            else:
                raise ValueError(
                    f"Model name must include repo_id (e.g., 'TheBloke/model-GGUF'): {model_name}"
                )

            # If no filename specified, find best GGUF file
            if not filename:
                yield {"status": "searching", "message": f"Searching for GGUF files in {repo_id}..."}

                hf_download, list_files, _ = _get_hf_hub()
                files = list_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if not gguf_files:
                    yield {
                        "status": "error",
                        "error": f"No GGUF files found in {repo_id}",
                    }
                    return

                # Filter out split files (e.g., model-00001-of-00002.gguf)
                # These require downloading multiple parts and aren't usable alone
                split_pattern = re.compile(r'-\d{5}-of-\d{5}')
                single_files = [f for f in gguf_files if not split_pattern.search(f)]

                # Use single files if available, otherwise fall back to all files
                candidates = single_files if single_files else gguf_files

                if not single_files and gguf_files:
                    yield {
                        "status": "warning",
                        "message": f"Only split model files found. Consider downloading single-file version.",
                    }

                # Prefer Q4_K_M, then Q5_K_M, then any
                preferred = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q8_0"]
                filename = None
                for pref in preferred:
                    for f in candidates:
                        if pref in f.upper():
                            filename = f
                            break
                    if filename:
                        break

                if not filename:
                    filename = candidates[0]

                yield {"status": "selected", "filename": filename}

            # Download the file
            yield {"status": "downloading", "filename": filename}

            hf_download, _, _ = _get_hf_hub()
            local_path = hf_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
            )

            # Move to models_dir root with clean name if nested
            local_path = Path(local_path)
            target_name = filename.split("/")[-1] if "/" in filename else filename
            target_path = self.models_dir / target_name

            if local_path != target_path:
                if target_path.exists():
                    target_path.unlink()
                local_path.rename(target_path)

            yield {
                "status": "complete",
                "path": str(target_path),
                "size": target_path.stat().st_size,
            }

        except Exception as e:
            yield {"status": "error", "error": str(e)}

    def _messages_to_prompt(self, messages: list[Message], model: Any = None) -> str:
        """Convert messages to a prompt string using ChatML format."""
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

        # Add assistant prefix for generation
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a response using the loaded model."""
        config = config or GenerationConfig()
        llm = self._load_model(model)

        prompt = self._messages_to_prompt(messages)

        # Generate
        response = llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=["<|im_end|>", "<|im_start|>"] + (config.stop_sequences or []),
            echo=False,
        )

        content = response["choices"][0]["text"].strip()
        finish_reason = response["choices"][0].get("finish_reason", "stop")

        usage = response.get("usage", {})

        return GenerationResult(
            content=content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        config = config or GenerationConfig()
        llm = self._load_model(model)

        prompt = self._messages_to_prompt(messages)

        # Stream generation
        for output in llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=["<|im_end|>", "<|im_start|>"] + (config.stop_sequences or []),
            echo=False,
            stream=True,
        ):
            text = output["choices"][0]["text"]
            if text:
                yield text

    async def health_check(self) -> bool:
        """Check if the provider is available."""
        return self.is_available

    async def close(self) -> None:
        """Cleanup and unload all models."""
        self.unload_all()
