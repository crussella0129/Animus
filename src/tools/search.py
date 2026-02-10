"""Agent tool for searching the ingested codebase via vector similarity."""

from __future__ import annotations

from typing import Any

from src.memory.embedder import Embedder
from src.memory.vectorstore import SQLiteVectorStore
from src.tools.base import Tool, ToolRegistry


class SearchCodebaseTool(Tool):
    """Search ingested codebase chunks by semantic similarity."""

    def __init__(self, store: SQLiteVectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def name(self) -> str:
        return "search_codebase"

    @property
    def description(self) -> str:
        return "Search the ingested codebase for code chunks semantically similar to a query."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or code query to search for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default: 5)",
                },
            },
            "required": ["query"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        query = args["query"]
        top_k = args.get("top_k", 5)
        query_embedding = self._embedder.embed([query])[0]
        results = self._store.search(query_embedding, top_k=top_k)
        if not results:
            return f"No results found for '{query}'."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            lines.append(f"[{i}] (score={r.score:.3f}) {source}")
            lines.append(r.text[:200])
            lines.append("")
        return "\n".join(lines)


def register_search_tools(
    registry: ToolRegistry,
    store: SQLiteVectorStore,
    embedder: Embedder,
) -> None:
    """Register all search tools with the given registry."""
    registry.register(SearchCodebaseTool(store, embedder))
