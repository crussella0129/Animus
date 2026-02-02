"""Base classes for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ToolCategory(str, Enum):
    """Categories of tools."""
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    CODE = "code"
    SEARCH = "search"
    MEMORY = "memory"
    ANALYSIS = "analysis"


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[str]] = None


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    data: Any = None  # Structured data if applicable
    metadata: dict = field(default_factory=dict)


class Tool(ABC):
    """
    Base class for agent tools.

    Tools are the actions an agent can take to interact with the environment.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used in function calls)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters."""
        ...

    @property
    def category(self) -> ToolCategory:
        """Tool category for organization."""
        return ToolCategory.CODE

    @property
    def requires_confirmation(self) -> bool:
        """Whether this tool requires human confirmation before execution."""
        return False

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters.

        Returns:
            ToolResult with output or error.
        """
        ...

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Get OpenAI schemas for all tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def by_category(self, category: ToolCategory) -> list[Tool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if t.category == category]
