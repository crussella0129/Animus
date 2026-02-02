"""OpenAI-compatible API server for Animus.

Provides endpoints compatible with OpenAI's API specification:
- POST /v1/chat/completions
- GET /v1/models
- POST /v1/embeddings

Plus Animus-specific endpoints:
- POST /v1/agent/chat (with tools)
- POST /v1/agent/ingest
- POST /v1/agent/search
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from urllib.parse import parse_qs, urlparse

from src.core.config import ConfigManager
from src.llm import get_default_provider
from src.llm.base import Message, GenerationConfig


@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[list[str]] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletionRequest":
        """Create from dictionary."""
        return cls(
            model=data.get("model", ""),
            messages=data.get("messages", []),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            tools=data.get("tools"),
            tool_choice=data.get("tool_choice"),
        )


@dataclass
class ChatCompletionChoice:
    """A choice in a chat completion response."""
    index: int
    message: dict[str, Any]
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "message": self.message,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage": self.usage,
        }


@dataclass
class ModelInfo:
    """Model information."""
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "animus"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }


class APIServer:
    """OpenAI-compatible API server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8337,
        api_key: Optional[str] = None,
    ):
        """Initialize API server.

        Args:
            host: Host to bind to
            port: Port to listen on
            api_key: Optional API key for authentication
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self._server: Optional[HTTPServer] = None

    async def handle_chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle POST /v1/chat/completions.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        config = ConfigManager().config
        provider = get_default_provider(config)

        if not provider.is_available:
            raise RuntimeError("No LLM provider available")

        # Convert messages to our format
        messages = [
            Message(role=m["role"], content=m["content"])
            for m in request.messages
        ]

        # Create generation config
        gen_config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens or config.model.max_tokens,
            stop=request.stop,
        )

        try:
            # Generate response
            result = await provider.generate(messages, gen_config)

            # Build response
            response = ChatCompletionResponse(
                model=request.model or config.model.model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message={"role": "assistant", "content": result.text},
                        finish_reason=result.finish_reason or "stop",
                    )
                ],
                usage={
                    "prompt_tokens": result.prompt_tokens or 0,
                    "completion_tokens": result.completion_tokens or 0,
                    "total_tokens": (result.prompt_tokens or 0) + (result.completion_tokens or 0),
                },
            )

            return response

        finally:
            if hasattr(provider, 'close'):
                await provider.close()

    async def handle_models(self) -> list[ModelInfo]:
        """Handle GET /v1/models.

        Returns:
            List of available models
        """
        config = ConfigManager().config
        models = []

        # Add native models
        from src.llm import NativeProvider, LLAMA_CPP_AVAILABLE
        if LLAMA_CPP_AVAILABLE:
            native = NativeProvider(models_dir=config.native.models_dir)
            native_models = await native.list_models()
            for m in native_models:
                models.append(ModelInfo(id=m.name, owned_by="native"))

        return models

    async def handle_embeddings(self, text: str | list[str], model: str = "") -> dict:
        """Handle POST /v1/embeddings.

        Args:
            text: Text or list of texts to embed
            model: Model to use (ignored, uses default embedder)

        Returns:
            Embeddings response
        """
        from src.memory import create_embedder

        embedder = await create_embedder("auto")

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings = []
        for i, t in enumerate(texts):
            embedding = await embedder.embed(t)
            embeddings.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            })

        return {
            "object": "list",
            "data": embeddings,
            "model": "animus-embedder",
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts),
            },
        }


def create_app(
    api_key: Optional[str] = None,
) -> "RequestHandler":
    """Create the HTTP request handler class.

    Args:
        api_key: Optional API key for authentication

    Returns:
        Configured request handler class
    """
    server = APIServer(api_key=api_key)

    class RequestHandler(BaseHTTPRequestHandler):
        """HTTP request handler for API server."""

        def _set_cors_headers(self):
            """Set CORS headers."""
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _check_auth(self) -> bool:
            """Check API key authentication."""
            if not api_key:
                return True

            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                return token == api_key

            return False

        def _send_json(self, data: dict, status: int = 200):
            """Send JSON response."""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def _send_error(self, message: str, status: int = 400):
            """Send error response."""
            self._send_json({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": status,
                }
            }, status)

        def do_OPTIONS(self):
            """Handle CORS preflight."""
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()

        def do_GET(self):
            """Handle GET requests."""
            if not self._check_auth():
                self._send_error("Invalid API key", 401)
                return

            parsed = urlparse(self.path)

            if parsed.path == "/v1/models":
                # List models
                try:
                    models = asyncio.run(server.handle_models())
                    self._send_json({
                        "object": "list",
                        "data": [m.to_dict() for m in models],
                    })
                except Exception as e:
                    self._send_error(str(e), 500)

            elif parsed.path == "/health":
                # Health check
                self._send_json({"status": "ok"})

            else:
                self._send_error("Not found", 404)

        def do_POST(self):
            """Handle POST requests."""
            if not self._check_auth():
                self._send_error("Invalid API key", 401)
                return

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_error("Invalid JSON")
                return

            parsed = urlparse(self.path)

            if parsed.path == "/v1/chat/completions":
                # Chat completions
                try:
                    request = ChatCompletionRequest.from_dict(data)
                    response = asyncio.run(server.handle_chat_completions(request))
                    self._send_json(response.to_dict())
                except Exception as e:
                    self._send_error(str(e), 500)

            elif parsed.path == "/v1/embeddings":
                # Embeddings
                try:
                    text = data.get("input", "")
                    model = data.get("model", "")
                    response = asyncio.run(server.handle_embeddings(text, model))
                    self._send_json(response)
                except Exception as e:
                    self._send_error(str(e), 500)

            elif parsed.path == "/v1/agent/chat":
                # Agent chat (with tools)
                try:
                    from src.core import Agent, AgentConfig

                    config = ConfigManager().config
                    provider = get_default_provider(config)

                    async def run_agent():
                        agent = Agent(
                            provider=provider,
                            config=AgentConfig(
                                model=data.get("model", config.model.model_name),
                                temperature=data.get("temperature", 0.7),
                            ),
                        )

                        messages = data.get("messages", [])
                        if messages:
                            user_message = messages[-1].get("content", "")
                            results = []
                            async for turn in agent.run(user_message):
                                results.append({
                                    "role": turn.role,
                                    "content": turn.content,
                                    "tool_calls": [
                                        {"name": tc["name"], "result": tc.get("result")}
                                        for tc in (turn.tool_calls or [])
                                    ],
                                })
                            return results
                        return []

                    results = asyncio.run(run_agent())
                    self._send_json({
                        "object": "agent.chat.completion",
                        "results": results,
                    })

                except Exception as e:
                    self._send_error(str(e), 500)

            elif parsed.path == "/v1/agent/search":
                # Search knowledge base
                try:
                    from src.memory import Ingester

                    async def do_search():
                        config = ConfigManager().config
                        ingester = Ingester()
                        try:
                            results = await ingester.search(
                                data.get("query", ""),
                                k=data.get("k", 5),
                                persist_dir=config.data_dir / "vectordb",
                            )
                            return [
                                {"content": content, "score": score, "metadata": meta}
                                for content, score, meta in results
                            ]
                        finally:
                            await ingester.close()

                    results = asyncio.run(do_search())
                    self._send_json({
                        "object": "search.results",
                        "data": results,
                    })

                except Exception as e:
                    self._send_error(str(e), 500)

            else:
                self._send_error("Not found", 404)

        def log_message(self, format, *args):
            """Suppress default logging."""
            pass

    return RequestHandler


def run_server(host: str = "localhost", port: int = 8337, api_key: Optional[str] = None):
    """Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        api_key: Optional API key for authentication
    """
    handler = create_app(api_key)
    server = HTTPServer((host, port), handler)

    print(f"Animus API Server running on http://{host}:{port}")
    print("Endpoints:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("  POST /v1/embeddings")
    print("  POST /v1/agent/chat")
    print("  POST /v1/agent/search")
    print()
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
