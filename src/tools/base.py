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
            return tool.execute(args)
        except Exception as e:
            return f"Error executing {name}: {e}"
