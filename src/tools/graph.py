"""Agent tools for querying the code knowledge graph."""

from __future__ import annotations

from typing import Any

from src.knowledge.graph_db import GraphDB, NodeRow
from src.tools.base import Tool, ToolRegistry


def _format_node(node: NodeRow) -> str:
    """Format a node row for display."""
    loc = f"{node.file_path}:{node.line_start}" if node.file_path else "(external)"
    return f"[{node.kind}] {node.qualified_name} @ {loc}"


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
            },
            "required": ["pattern"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        kind = args.get("kind")
        limit = args.get("limit", 20)
        nodes = self._db.search_nodes(pattern, kind=kind, limit=limit)
        if not nodes:
            return f"No symbols found matching '{pattern}'."
        lines = [_format_node(n) for n in nodes]
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
            },
            "required": ["symbol"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        symbol = args["symbol"]
        callers = self._db.get_callers(symbol)
        if not callers:
            return f"No callers found for '{symbol}'."
        lines = [_format_node(n) for n in callers]
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
            },
            "required": ["symbol"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        symbol = args["symbol"]
        max_depth = args.get("max_depth", 5)
        radius = self._db.get_blast_radius(symbol, max_depth=max_depth)
        if not radius:
            return f"No blast radius found for '{symbol}'."
        lines = [f"Blast radius for {symbol}:"]
        for depth in sorted(radius.keys()):
            lines.append(f"  Depth {depth}:")
            for node in radius[depth]:
                lines.append(f"    {_format_node(node)}")
        return "\n".join(lines)


def register_graph_tools(registry: ToolRegistry, db: GraphDB) -> None:
    """Register all graph tools with the given registry."""
    registry.register(SearchCodeGraphTool(db))
    registry.register(GetCallersTool(db))
    registry.register(GetBlastRadiusTool(db))
