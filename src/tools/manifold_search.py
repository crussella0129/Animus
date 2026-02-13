"""Unified search tool powered by the Animus: Manifold retrieval router.

Replaces the separate search_codebase and search_code_graph tools
with a single tool that automatically routes queries to the optimal
retrieval strategy (or combination of strategies).
"""

from __future__ import annotations

from typing import Any

from src.retrieval.executor import RetrievalExecutor
from src.retrieval.router import classify_query
from src.tools.base import Tool, ToolRegistry


class ManifoldSearchTool(Tool):
    """Unified codebase search with automatic strategy routing.

    Accepts any natural language query about the codebase and
    automatically routes it to the optimal retrieval strategy:

    - SEMANTIC: Conceptual questions ("how does auth work?")
    - STRUCTURAL: Relationship questions ("what calls authenticate()?")
    - HYBRID: Compound questions ("find auth code and its dependencies")
    - KEYWORD: Exact text matching ("find TODO comments")

    The routing is hardcoded (no LLM involvement in classification).
    Results include strategy information and confidence scores.
    """

    def __init__(self, executor: RetrievalExecutor) -> None:
        """Initialize the Manifold search tool.

        Args:
            executor: RetrievalExecutor configured with all backends
        """
        super().__init__()
        self._executor = executor

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the codebase using automatic strategy routing. "
            "Handles semantic queries ('how does authentication work?'), "
            "structural queries ('what calls authenticate()?'), "
            "hybrid queries ('find the auth code and what depends on it'), "
            "and keyword queries ('find TODO comments'). "
            "Automatically picks the best retrieval strategy and fuses results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language or code query about the codebase. "
                        "Examples: 'how does auth work?', 'what calls parse()?', "
                        "'find error handling code and its callers'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 8)",
                },
            },
            "required": ["query"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        """Execute a search using Manifold routing.

        Args:
            args: Tool arguments with 'query' and optional 'top_k'

        Returns:
            Formatted search results with strategy info
        """
        query = args["query"]
        top_k = args.get("top_k", 8)

        # Route the query using hardcoded classification
        decision = classify_query(query)

        # Execute the retrieval strategy (or strategies for HYBRID)
        results = self._executor.execute(decision, top_k=top_k)

        if not results:
            return (
                f"No results found for: '{query}'\n\n"
                f"Strategy: {decision.strategy.value}\n"
                f"Confidence: {decision.confidence:.0%}\n"
                f"Reasoning: {decision.reasoning}\n\n"
                f"Try rephrasing your query or checking that the codebase has been ingested "
                f"(run 'animus graph' and 'animus ingest' first)."
            )

        # Format results with strategy info
        lines = [
            f"╔═ Manifold Search Results ═╗",
            f"│ Strategy: {decision.strategy.value.upper()}",
            f"│ Confidence: {decision.confidence:.0%}",
            f"│ Reasoning: {decision.reasoning}",
            f"│ Results: {len(results)}",
            f"╚═{'═' * 25}╝",
            "",
        ]

        for i, r in enumerate(results, 1):
            # Strategy indicator
            strategy_tag = f"[{r.strategy}]"

            # Multi-strategy indicator (appears in both semantic and structural)
            multi = " ★" if r.metadata.get("multi_strategy") else ""

            # Source display (prefer qualified name if available)
            source_display = r.source
            qname = r.metadata.get("qualified_name", "")
            if qname:
                source_display = f"{qname} @ {r.source}"

            # Line range if available
            line_start = r.metadata.get("line_start")
            line_end = r.metadata.get("line_end")
            if line_start and line_end:
                source_display += f":{line_start}-{line_end}"

            lines.append(f"[{i}] (score={r.score:.3f}) {strategy_tag}{multi}")
            lines.append(f"    Location: {source_display}")

            # Show preview (first 200 chars)
            preview = r.text[:200].strip()
            if len(r.text) > 200:
                preview += "..."

            # Indent preview for readability
            preview_lines = preview.split("\n")
            for pline in preview_lines[:5]:  # Max 5 lines of preview
                lines.append(f"    │ {pline}")

            # Show docstring if available and not in preview
            docstring = r.metadata.get("docstring", "")
            if docstring and docstring not in preview:
                doc_preview = docstring[:100].strip()
                lines.append(f"    ╰─ {doc_preview}")

            lines.append("")

        # Footer with helpful info
        lines.append(f"Tip: ★ indicates results found by multiple strategies (high confidence)")

        return "\n".join(lines)


def register_manifold_search(
    registry: ToolRegistry,
    executor: RetrievalExecutor,
) -> None:
    """Register the Manifold unified search tool.

    This is the primary entry point for Manifold functionality.
    Replaces separate search_codebase and graph search tools with
    one intelligent, automatically-routed search tool.

    Args:
        registry: The tool registry to register with
        executor: RetrievalExecutor configured with all available backends

    Example:
        from src.retrieval.executor import RetrievalExecutor
        from src.tools.manifold_search import register_manifold_search

        executor = RetrievalExecutor(
            vector_store=store,
            embedder=embedder,
            graph_db=graph_db,
            project_root=Path.cwd(),
        )
        register_manifold_search(registry, executor)
    """
    registry.register(ManifoldSearchTool(executor))
