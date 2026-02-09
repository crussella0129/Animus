"""Base classes for agent tools."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints
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


# ---------------------------------------------------------------------------
# Type-hint → ToolParameter mapping
# ---------------------------------------------------------------------------
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(tp: type) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    # Unwrap Optional (Union[X, None])
    origin = getattr(tp, "__origin__", None)
    if origin is type(None):
        return "string"
    # Handle Optional[X] → X
    args = getattr(tp, "__args__", None)
    if args and type(None) in args:
        # Optional[X] → take the non-None type
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            tp = non_none[0]

    return _TYPE_MAP.get(tp, "string")


def _extract_param_descriptions(func: Callable) -> dict[str, str]:
    """Extract parameter descriptions from a Google/Numpy-style docstring."""
    doc = inspect.getdoc(func) or ""
    descriptions: dict[str, str] = {}
    in_args = False

    section_headers = ("returns:", "raises:", "yields:", "note:", "notes:", "example:", "examples:")
    for line in doc.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith(("args:", "parameters:", "params:")):
            in_args = True
            continue
        if in_args:
            if lower.startswith(section_headers):
                break
            # Match "param_name: description" or "param_name (type): description"
            if ":" in stripped and not stripped.startswith("-"):
                parts = stripped.split(":", 1)
                param_name = parts[0].strip().split("(")[0].strip().split(" ")[0]
                desc = parts[1].strip()
                if param_name:
                    descriptions[param_name] = desc
            elif stripped.startswith("- "):
                # Markdown-style "- param_name: description"
                rest = stripped[2:]
                if ":" in rest:
                    parts = rest.split(":", 1)
                    param_name = parts[0].strip()
                    desc = parts[1].strip()
                    if param_name:
                        descriptions[param_name] = desc

    return descriptions


def _infer_parameters(func: Callable) -> list[ToolParameter]:
    """Infer ToolParameter list from function signature and type hints."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    param_docs = _extract_param_descriptions(func)
    params: list[ToolParameter] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls", "kwargs"):
            continue

        tp = hints.get(name, str)
        json_type = _python_type_to_json(tp)
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default
        description = param_docs.get(name, name)

        params.append(ToolParameter(
            name=name,
            type=json_type,
            description=description,
            required=required,
            default=default,
        ))

    return params


class DecoratedTool(Tool):
    """A tool created from a decorated function."""

    def __init__(
        self,
        func: Callable,
        tool_name: str,
        tool_description: str,
        tool_parameters: list[ToolParameter],
        tool_category: ToolCategory = ToolCategory.CODE,
        tool_requires_confirmation: bool = False,
    ):
        self._func = func
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters
        self._category = tool_category
        self._requires_confirmation = tool_requires_confirmation

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function."""
        try:
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)

            # If the function returns a ToolResult, use it directly
            if isinstance(result, ToolResult):
                return result

            # Otherwise wrap the return value
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def tool(
    description: Optional[str] = None,
    name: Optional[str] = None,
    category: ToolCategory = ToolCategory.CODE,
    requires_confirmation: bool = False,
) -> Callable:
    """Decorator to create a Tool from a function.

    Infers parameter names, types, and descriptions from the function's
    signature, type hints, and docstring.

    Usage:
        @tool(description="Read a file from disk")
        async def read_file(path: str) -> ToolResult:
            content = open(path).read()
            return ToolResult(success=True, output=content)

        # Register with a registry:
        registry.register(read_file)  # read_file is now a Tool instance

    Args:
        description: Tool description for the LLM. If None, uses the
            function's docstring first line.
        name: Tool name. If None, uses the function name.
        category: Tool category for organization.
        requires_confirmation: Whether the tool needs human confirmation.

    Returns:
        A decorator that replaces the function with a DecoratedTool instance.
    """
    def decorator(func: Callable) -> DecoratedTool:
        tool_name = name or func.__name__

        # Get description from decorator arg or docstring
        tool_desc = description
        if tool_desc is None:
            doc = inspect.getdoc(func)
            tool_desc = doc.split("\n")[0] if doc else tool_name

        params = _infer_parameters(func)

        return DecoratedTool(
            func=func,
            tool_name=tool_name,
            tool_description=tool_desc,
            tool_parameters=params,
            tool_category=category,
            tool_requires_confirmation=requires_confirmation,
        )

    return decorator
