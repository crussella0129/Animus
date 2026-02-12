"""Agent tools for querying the code knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.knowledge.graph_db import GraphDB, NodeRow
from src.tools.base import Tool, ToolRegistry


def _format_node(node: NodeRow, include_source: bool = False, max_lines: int = 10) -> str:
    """Format a node row for display, optionally including source code snippet.

    Args:
        node: The node to format
        include_source: If True, include source code snippet (default: False)
        max_lines: Maximum lines of source code to include (default: 10)

    Returns:
        Formatted string representation of the node
    """
    loc = f"{node.file_path}:{node.line_start}" if node.file_path else "(external)"
    header = f"[{node.kind}] {node.qualified_name} @ {loc}"

    if not include_source or not node.file_path:
        return header

    # Try to read source snippet
    try:
        file_path = Path(node.file_path)
        if file_path.exists():
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            start_idx = max(0, node.line_start - 1)
            end_idx = min(len(lines), node.line_start - 1 + max_lines)
            snippet_lines = lines[start_idx:end_idx]

            # Add line numbers and format
            numbered = [
                f"  {node.line_start + i:4d} | {line}"
                for i, line in enumerate(snippet_lines)
            ]

            return f"{header}\n" + "\n".join(numbered)
    except (OSError, UnicodeDecodeError):
        pass

    # If source reading fails, return header only
    return header


class SearchCodeGraphTool(Tool):
    def __init__(self, db: GraphDB) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "search_code_graph"

    @property
    def description(self) -> str:
        return "Search the code knowledge graph for symbols matching a pattern."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (substring match on name/qualified_name)",
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by kind: module, class, function, method (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 20)",
                },
                "include_source": {
                    "type": "boolean",
                    "description": "Include source code snippets in results (default: false)",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        kind = args.get("kind")
        limit = args.get("limit", 20)
        include_source = args.get("include_source", False)
        nodes = self._db.search_nodes(pattern, kind=kind, limit=limit)
        if not nodes:
            return f"No symbols found matching '{pattern}'."
        lines = [_format_node(n, include_source=include_source) for n in nodes]
        return "\n".join(lines)


class GetCallersTool(Tool):
    def __init__(self, db: GraphDB) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "get_callers"

    @property
    def description(self) -> str:
        return "Get all callers of a symbol by its qualified name."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Fully qualified name of the symbol",
                },
                "include_source": {
                    "type": "boolean",
                    "description": "Include source code snippets in results (default: false)",
                },
            },
            "required": ["symbol"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        symbol = args["symbol"]
        include_source = args.get("include_source", False)
        callers = self._db.get_callers(symbol)
        if not callers:
            return f"No callers found for '{symbol}'."
        lines = [_format_node(n, include_source=include_source) for n in callers]
        return f"Callers of {symbol}:\n" + "\n".join(lines)


class GetBlastRadiusTool(Tool):
    def __init__(self, db: GraphDB) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "get_blast_radius"

    @property
    def description(self) -> str:
        return "Get the blast radius of a symbol: all symbols affected by changes to it."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Fully qualified name of the symbol",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum BFS depth (default: 5)",
                },
                "include_source": {
                    "type": "boolean",
                    "description": "Include source code snippets in results (default: false)",
                },
            },
            "required": ["symbol"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        symbol = args["symbol"]
        max_depth = args.get("max_depth", 5)
        include_source = args.get("include_source", False)
        radius, cycles_detected = self._db.get_blast_radius(symbol, max_depth=max_depth)
        if not radius:
            return f"No blast radius found for '{symbol}'."
        lines = [f"Blast radius for {symbol}:"]
        if cycles_detected:
            lines.append("  ⚠️  Cycles detected in call graph (recursive calls present)")
        for depth in sorted(radius.keys()):
            lines.append(f"  Depth {depth}:")
            for node in radius[depth]:
                # Indent source snippets for blast radius display
                formatted = _format_node(node, include_source=include_source)
                if include_source and "\n" in formatted:
                    # Multi-line result with source - indent all lines
                    indented = "\n".join("    " + line for line in formatted.split("\n"))
                    lines.append(indented)
                else:
                    lines.append(f"    {formatted}")
        return "\n".join(lines)


def register_graph_tools(registry: ToolRegistry, db: GraphDB) -> None:
    """Register all graph tools with the given registry."""
    registry.register(SearchCodeGraphTool(db))
    registry.register(GetCallersTool(db))
    registry.register(GetBlastRadiusTool(db))
