"""Native llama-cpp-python provider for local GGUF models."""

from __future__ import annotations

import re
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm.base import ModelCapabilities, ModelProvider


@dataclass
class ModelInfo:
    """Metadata for a model in the catalog."""
    repo: str
    filename: str
    params_b: float
    context_length: int = 4096
    vram_q4_gb: float = 0.0
    roles: list[str] = field(default_factory=list)
    notes: str = ""

    def __iter__(self):
        """Backward compat: repo, filename, _ = MODEL_CATALOG[name]."""
        return iter((self.repo, self.filename, self.params_b))

    def __getitem__(self, index):
        """Backward compat: MODEL_CATALOG[name][0]."""
        return (self.repo, self.filename, self.params_b)[index]


# Known model catalog: short name -> ModelInfo
# VRAM estimates use Q4_K_M rule of thumb: params_b * 0.6 + 0.3 GB + ~15% runtime overhead
MODEL_CATALOG: dict[str, ModelInfo] = {
    "llama-3.2-1b": ModelInfo(
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        params_b=1.0,
        context_length=4096,
        vram_q4_gb=1.2,
        roles=["executor"],
        notes="Fastest, minimal VRAM",
    ),
    "llama-3.2-3b": ModelInfo(
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        params_b=3.0,
        context_length=4096,
        vram_q4_gb=2.4,
        roles=["executor", "explorer"],
        notes="Good balance of speed and capability",
    ),
    "phi-4-mini": ModelInfo(
        repo="bartowski/phi-4-mini-instruct-GGUF",
        filename="phi-4-mini-instruct-Q4_K_M.gguf",
        params_b=3.8,
        context_length=8192,
        vram_q4_gb=2.9,
        roles=["executor", "planner"],
        notes="Strong reasoning for size, 8K context",
    ),
    "qwen-2.5-3b": ModelInfo(
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        params_b=3.0,
        context_length=32768,
        vram_q4_gb=2.4,
        roles=["executor", "explorer"],
        notes="32K context, good multilingual",
    ),
    "qwen-2.5-7b": ModelInfo(
        repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        params_b=7.0,
        context_length=32768,
        vram_q4_gb=4.8,
        roles=["executor", "planner", "explorer"],
        notes="Best all-round, 32K context",
    ),
    "gemma-3-4b": ModelInfo(
        repo="bartowski/google_gemma-3-4b-it-GGUF",
        filename="google_gemma-3-4b-it-Q4_K_M.gguf",
        params_b=4.0,
        context_length=8192,
        vram_q4_gb=2.9,
        roles=["executor", "explorer"],
        notes="Google, strong coding",
    ),
    "qwen-2.5-coder-7b": ModelInfo(
        repo="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        params_b=7.0,
        context_length=32768,
        vram_q4_gb=4.8,
        roles=["executor", "planner", "explorer"],
        notes="Code-specialized 7B, 32K context",
    ),
}


def get_models_for_role(role: str) -> list[str]:
    """Return model names that support the given role."""
    return [name for name, info in MODEL_CATALOG.items() if role in info.roles]


def get_models_fitting_vram(vram_gb: float) -> list[str]:
    """Return model names that fit within the given VRAM budget (GB)."""
    return [name for name, info in MODEL_CATALOG.items() if info.vram_q4_gb <= vram_gb]


def _estimate_params_from_filename(path: str) -> float:
    """Estimate parameter count from GGUF filename conventions like 'model-7B-Q4'."""
    name = Path(path).stem.upper()
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", name)
    if match:
        return float(match.group(1))
    return 0.0


def discover_local_models(models_dir: Path) -> dict[str, ModelInfo]:
    """Scan models directory for GGUF files and build catalog entries.

    Args:
        models_dir: Directory containing .gguf model files

    Returns:
        Dict mapping model name to ModelInfo for discovered models
    """
    if not models_dir.exists():
        return {}

    discovered = {}
    for path in models_dir.glob("*.gguf"):
        # Skip partial downloads
        if path.name.endswith(".part"):
            continue

        # Generate a model name from filename (lowercase, replace special chars)
        name = path.stem.lower().replace("-", "_").replace(".", "_")

        # Estimate parameters from filename
        params_b = _estimate_params_from_filename(str(path))

        # Conservative defaults
        context_length = 4096
        vram_q4_gb = params_b * 0.6 + 0.5 if params_b > 0 else 2.0

        discovered[name] = ModelInfo(
            repo="local",
            filename=path.name,
            params_b=params_b,
            context_length=context_length,
            vram_q4_gb=vram_q4_gb,
            roles=["executor"] if params_b < 4 else ["executor", "planner"],
            notes=f"Auto-discovered from {path.name}",
        )

    return discovered


def list_available_models(models_dir: Path | None = None) -> list[str]:
    """Return list of known and discovered model names.

    Args:
        models_dir: Optional directory to scan for local GGUF files

    Returns:
        Sorted list of model names (catalog + discovered)
    """
    models = set(MODEL_CATALOG.keys())

    if models_dir:
        discovered = discover_local_models(models_dir)
        models.update(discovered.keys())

    return sorted(models)


def download_gguf(
    model_name: str,
    models_dir: Path,
    on_progress: Any = None,
) -> Path:
    """Download a GGUF model from HuggingFace.

    Args:
        model_name: Short name from MODEL_CATALOG or a direct HuggingFace URL.
        models_dir: Directory to save the model file.
        on_progress: Optional callback(downloaded_bytes, total_bytes) for progress.

    Returns:
        Path to the downloaded model file.
    """
    import httpx

    if model_name in MODEL_CATALOG:
        repo, filename, _ = MODEL_CATALOG[model_name]
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    elif model_name.startswith("http"):
        url = model_name
        filename = url.split("/")[-1]
    else:
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: {', '.join(sorted(MODEL_CATALOG.keys()))}\n"
            f"Or provide a direct HuggingFace URL to a .gguf file."
        )

    dest = models_dir / filename
    if dest.exists():
        return dest

    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".gguf.part")

    # Check for existing partial download and resume if possible
    existing_size = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    mode = "wb"  # Default: overwrite

    if existing_size > 0:
        # Try to resume from existing partial download
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"  # Append mode

    with httpx.Client(timeout=None, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            # Check if server supports resume (HTTP 206) or if we need to restart (HTTP 200)
            if response.status_code == 206:
                # Partial content - resume successful
                total = existing_size + int(response.headers.get("content-length", 0))
                downloaded = existing_size
            elif response.status_code == 200:
                # Server doesn't support resume or file changed - restart download
                mode = "wb"
                existing_size = 0
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
            else:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0

            with open(tmp, mode) as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)

    tmp.rename(dest)
    return dest


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
        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        # GBNF grammar constraint: forces structurally valid JSON output
        grammar = kwargs.get("grammar")
        if grammar is not None:
            call_kwargs["grammar"] = grammar

        response = model.create_chat_completion(**call_kwargs)
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream response chunks from llama-cpp model.

        Yields individual text chunks as they're generated, enabling
        real-time streaming output.
        """
        model = self._load_model()
        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,  # Enable streaming
        }

        # Note: Grammar constraints are not supported with streaming in llama-cpp-python
        # Streaming will work without grammar enforcement

        try:
            for chunk in model.create_chat_completion(**call_kwargs):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        except Exception as e:
            # If streaming fails, fall back to non-streaming
            yield self.generate(messages, tools=tools, **kwargs)

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

    def pull(self, model_name: str) -> None:
        """Pull/download a GGUF model. Delegates to download_gguf()."""
        from src.core.config import AnimusConfig

        cfg = AnimusConfig.load()
        download_gguf(model_name, cfg.models_dir)
