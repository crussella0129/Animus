"""Tool ABC, registry, and schema generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool parameters."""
        ...

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> str:
        """Execute the tool and return a string result."""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI function-calling compatible schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _coerce_args(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Coerce argument types based on JSON Schema declarations.

    LLMs sometimes return integers as strings (e.g., "30" instead of 30).
    Uses the tool's parameter schema to cast values to their declared types.
    """
    properties = schema.get("properties", {})
    coerced = dict(args)
    for key, value in coerced.items():
        if key not in properties:
            continue
        expected_type = properties[key].get("type")
        if expected_type == "integer" and isinstance(value, str):
            try:
                coerced[key] = int(value)
            except (ValueError, TypeError):
                pass
        elif expected_type == "number" and isinstance(value, str):
            try:
                coerced[key] = float(value)
            except (ValueError, TypeError):
                pass
        elif expected_type == "boolean" and isinstance(value, str):
            coerced[key] = value.lower() in ("true", "1", "yes")
    return coerced


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Generate OpenAI function-calling schemas for all registered tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name. Returns error string if tool not found."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"
        try:
            coerced = _coerce_args(args, tool.parameters)
            return tool.execute(coerced)
        except Exception as e:
            return f"Error executing {name}: {e}"
