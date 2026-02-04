"""
Unified LLM provider via LiteLLM.

Handles both local models (via managed llama-server) and API providers
(OpenAI, Anthropic, etc.) through a single interface with native
function calling support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from .base import (
    GenerationConfig,
    GenerationResult,
    Message,
    ModelInfo,
    ModelProvider,
    ProviderType,
)
from .server import LlamaServer

logger = logging.getLogger(__name__)


def _check_litellm() -> bool:
    """Check if litellm is importable."""
    try:
        import litellm  # noqa: F401
        return True
    except ImportError:
        return False


class LiteLLMProvider(ModelProvider):
    """
    Unified provider using LiteLLM for local and API models.

    For local models:
        - Manages a llama-server subprocess
        - LiteLLM connects to it as an OpenAI-compatible endpoint
        - Model name format: "local/model-name" (resolves to GGUF in models_dir)

    For API models:
        - Routes directly through LiteLLM
        - Model name format: "gpt-4o", "claude-sonnet-4-20250514", etc.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        server_bin_dir: Optional[Path] = None,
    ):
        self._models_dir = models_dir or (Path.home() / ".animus" / "models")
        self._api_key = api_key
        self._api_base = api_base
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx

        # Managed llama-server for local models
        self._server = LlamaServer(
            bin_dir=server_bin_dir or (Path.home() / ".animus" / "bin"),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )

        # Track which local model is currently loaded
        self._active_local_model: Optional[str] = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LITELLM

    @property
    def is_available(self) -> bool:
        return _check_litellm()

    def _is_local_model(self, model: str) -> bool:
        """Check if a model name refers to a local GGUF file."""
        if model.startswith("local/"):
            return True
        # Check if it matches a GGUF file in models_dir
        if model.endswith(".gguf"):
            return True
        # Check if a matching GGUF exists
        return self._resolve_local_model(model) is not None

    def _resolve_local_model(self, model: str) -> Optional[Path]:
        """Resolve a model name to a local GGUF file path."""
        # Strip "local/" prefix
        name = model.removeprefix("local/")

        # Direct path
        if Path(name).exists():
            return Path(name)

        # Check in models directory
        model_path = self._models_dir / name
        if model_path.exists():
            return model_path

        # Add .gguf extension
        if not name.endswith(".gguf"):
            model_path = self._models_dir / f"{name}.gguf"
            if model_path.exists():
                return model_path

        # Fuzzy match in models directory
        if self._models_dir.exists():
            for f in self._models_dir.glob("*.gguf"):
                if name.lower() in f.name.lower():
                    return f

        return None

    def _ensure_local_server(self, model: str) -> str:
        """
        Ensure llama-server is running with the right model.

        Returns the LiteLLM model string to use.
        """
        model_path = self._resolve_local_model(model)
        if model_path is None:
            raise FileNotFoundError(
                f"Local model not found: {model}. "
                f"Available models in {self._models_dir}: "
                f"{[f.name for f in self._models_dir.glob('*.gguf')]}"
            )

        # Check if we need to (re)start the server
        if not self._server.is_running or self._active_local_model != str(model_path):
            logger.info(f"Starting llama-server with {model_path.name}...")

            if not self._server.is_installed:
                logger.info("Downloading llama-server binary...")
                self._server.install()

            base_url = self._server.start(model_path)
            self._active_local_model = str(model_path)
            logger.info(f"llama-server ready at {base_url}")

        # LiteLLM talks to local server as an OpenAI-compatible endpoint
        return f"openai/{model_path.stem}"

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to dicts for LiteLLM."""
        result = []
        for msg in messages:
            d: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                d["name"] = msg.name
            if msg.tool_calls:
                # Convert internal tool call format to OpenAI format
                d["tool_calls"] = self._tool_calls_to_openai(msg.tool_calls)
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            result.append(d)
        return result

    def _tool_calls_to_openai(self, tool_calls: list[dict]) -> list[dict]:
        """Convert internal tool call format to OpenAI API format.

        Internal: {"name": "...", "arguments": {...}, "id": "..."}
        OpenAI:   {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
        """
        formatted = []
        for tc in tool_calls:
            # Already in OpenAI format (has "type" and "function" keys)
            if "type" in tc and "function" in tc:
                formatted.append(tc)
                continue

            args = tc.get("arguments", {})
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)

            formatted.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": args_str,
                },
            })
        return formatted

    def _parse_tool_calls(
        self, tool_calls: Optional[list]
    ) -> Optional[list[dict[str, Any]]]:
        """Parse tool calls from LiteLLM response into our format."""
        if not tool_calls:
            return None

        parsed = []
        for tc in tool_calls:
            func = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = func.name if hasattr(func, "name") else func.get("name", "")
            args_str = (
                func.arguments
                if hasattr(func, "arguments")
                else func.get("arguments", "{}")
            )

            try:
                arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                arguments = {}

            parsed.append({
                "id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                "name": name,
                "arguments": arguments,
            })

        return parsed if parsed else None

    async def generate(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> GenerationResult:
        """Generate a response, with optional tool calling."""
        if not _check_litellm():
            raise RuntimeError(
                "litellm is not installed. Install with: pip install litellm"
            )

        import litellm

        # Suppress litellm's verbose logging
        litellm.suppress_debug_info = True

        config = config or GenerationConfig()

        # Handle local vs API models
        kwargs: dict[str, Any] = {
            "messages": self._messages_to_dicts(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        if self._is_local_model(model):
            litellm_model = self._ensure_local_server(model)
            kwargs["model"] = litellm_model
            kwargs["api_base"] = self._server.base_url
            kwargs["api_key"] = "not-needed"
        else:
            kwargs["model"] = model
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        response = await litellm.acompletion(**kwargs)

        # Extract response content
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = self._parse_tool_calls(
            getattr(message, "tool_calls", None)
        )

        # Determine finish reason
        finish_reason = choice.finish_reason or "stop"
        if tool_calls:
            finish_reason = "tool_calls"

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return GenerationResult(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        if not _check_litellm():
            raise RuntimeError(
                "litellm is not installed. Install with: pip install litellm"
            )

        import litellm

        litellm.suppress_debug_info = True
        config = config or GenerationConfig()

        kwargs: dict[str, Any] = {
            "messages": self._messages_to_dicts(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": True,
        }

        if self._is_local_model(model):
            litellm_model = self._ensure_local_server(model)
            kwargs["model"] = litellm_model
            kwargs["api_base"] = self._server.base_url
            kwargs["api_key"] = "not-needed"
        else:
            kwargs["model"] = model
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content

    async def list_models(self) -> list[ModelInfo]:
        """List available local GGUF models."""
        models = []

        if self._models_dir.exists():
            for f in sorted(self._models_dir.glob("*.gguf")):
                stat = f.stat()
                # Detect quantization from filename
                quant = None
                for q in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_M",
                           "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32"]:
                    if q.lower() in f.name.lower():
                        quant = q
                        break

                models.append(ModelInfo(
                    name=f.name,
                    provider=ProviderType.LITELLM,
                    size_bytes=stat.st_size,
                    quantization=quant,
                ))

        return models

    async def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally or is a known API model."""
        if self._is_local_model(model_name):
            return self._resolve_local_model(model_name) is not None
        # For API models, assume they exist (will fail at generation time if not)
        return True

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        """
        Download a model from Hugging Face.

        Currently supports HuggingFace GGUF downloads.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            yield {
                "status": "error",
                "message": "huggingface_hub not installed. Install with: pip install huggingface-hub"
            }
            return

        yield {"status": "downloading", "model": model_name}

        try:
            # Parse "repo/filename" format
            if "/" in model_name:
                parts = model_name.rsplit("/", 1)
                repo_id = parts[0]
                filename = parts[1] if len(parts) > 1 else None

                self._models_dir.mkdir(parents=True, exist_ok=True)

                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(self._models_dir),
                )
                yield {"status": "complete", "path": path}
            else:
                yield {
                    "status": "error",
                    "message": f"Use format: repo_id/filename.gguf (got: {model_name})"
                }
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def stop_server(self) -> None:
        """Stop the managed llama-server if running."""
        self._server.stop()
        self._active_local_model = None

    def __del__(self):
        self.stop_server()
