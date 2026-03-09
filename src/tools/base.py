"""Tool ABC, registry, and schema generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for agent tools."""

    def __init__(self):
        """Initialize tool with default isolation level."""
        self._isolation_level: str = "none"

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

    @property
    def isolation_level(self) -> str:
        """Get recommended isolation level for this tool."""
        return getattr(self, '_isolation_level', 'none')

    @isolation_level.setter
    def isolation_level(self, level: str):
        """Set isolation level for this tool."""
        if level not in ("none", "ornstein", "smough"):
            raise ValueError(f"Invalid isolation level: {level}")
        self._isolation_level = level

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


def isolated(level: str = "ornstein"):
    """
    Decorator to mark a tool class as requiring isolation.

    Usage:
        @isolated(level="ornstein")
        class MyDangerousTool(Tool):
            def __init__(self):
                super().__init__()
            ...

    Args:
        level: Isolation level ("none", "ornstein", "smough")
    """
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Call original init
            if original_init != Tool.__init__:
                original_init(self, *args, **kwargs)
            else:
                Tool.__init__(self)

            # Set isolation level
            self._isolation_level = level

        cls.__init__ = new_init
        return cls

    return decorator


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


def _validate_args(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate tool arguments against JSON Schema declarations.

    Returns an error message if validation fails, None if OK.
    Only checks required fields and basic type constraints — not a full
    JSON Schema validator, but catches the most common LLM mistakes.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required fields
    for field in required:
        if field not in args:
            return f"Missing required argument: '{field}'"

    # Check basic types for provided fields
    for key, value in args.items():
        if key not in properties:
            continue  # Allow extra fields (LLMs often add them)
        expected_type = properties[key].get("type")
        if expected_type is None:
            continue
        if expected_type == "string" and not isinstance(value, str):
            return f"Argument '{key}' should be string, got {type(value).__name__}"
        if expected_type == "integer" and not isinstance(value, int):
            return f"Argument '{key}' should be integer, got {type(value).__name__}"
        if expected_type == "boolean" and not isinstance(value, bool):
            return f"Argument '{key}' should be boolean, got {type(value).__name__}"
        if expected_type == "array" and not isinstance(value, list):
            return f"Argument '{key}' should be array, got {type(value).__name__}"
    return None


class RespondTool(Tool):
    """Special tool the model calls to return a final natural-language response.

    Used with grammar-constrained decoding: since grammar forces JSON tool calls
    on every turn, the model calls respond() to return its final answer.
    The agent loop detects this tool and returns the message directly to the user.
    """

    @property
    def name(self) -> str:
        return "respond"

    @property
    def description(self) -> str:
        return (
            "Return a final response to the user. Call this when your task is complete "
            "or when you need to communicate results without making another tool call."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The response to return to the user.",
                }
            },
            "required": ["message"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        return args["message"]


def function_tool(description: str):
    """Decorator to create a Tool subclass from a plain annotated function.

    Auto-generates JSON Schema from type hints. Supports str, int, float, bool;
    unrecognized types default to "string". Parameters without default values
    are added to the required list.

    Returns a Tool subclass (not an instance). Instantiate with the class itself:

        @function_tool(description="Return the current time")
        def get_time(format: str = "%H:%M") -> str:
            from datetime import datetime
            return datetime.now().strftime(format)

        registry.register(get_time())  # get_time is the class; get_time() is the instance
    """
    import inspect
    from typing import get_type_hints

    _TYPE_MAP: dict = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    def decorator(func):
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            py_type = hints.get(param_name, str)
            json_type = _TYPE_MAP.get(py_type, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        _func_name = func.__name__
        _func_description = description

        class _FunctionTool(Tool):
            @property
            def name(self) -> str:
                return _func_name

            @property
            def description(self) -> str:
                return _func_description

            @property
            def parameters(self) -> dict[str, Any]:
                return schema

            def execute(self, args: dict[str, Any]) -> str:
                return str(func(**args))

        _FunctionTool.__name__ = func.__name__
        _FunctionTool.__qualname__ = func.__qualname__
        return _FunctionTool

    return decorator


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
            validation_error = _validate_args(coerced, tool.parameters)
            if validation_error:
                return f"Error: Invalid arguments for {name}: {validation_error}"
            return tool.execute(coerced)
        except Exception as e:
            return f"Error executing {name}: {e}"
