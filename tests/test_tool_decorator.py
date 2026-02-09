"""Tests for decorator-based tool registration."""

import pytest
from typing import Optional
from src.tools.base import (
    tool,
    DecoratedTool,
    Tool,
    ToolParameter,
    ToolResult,
    ToolCategory,
    ToolRegistry,
    _python_type_to_json,
    _extract_param_descriptions,
    _infer_parameters,
)


class TestPythonTypeToJson:
    """Tests for type hint → JSON Schema mapping."""

    def test_str(self):
        assert _python_type_to_json(str) == "string"

    def test_int(self):
        assert _python_type_to_json(int) == "integer"

    def test_float(self):
        assert _python_type_to_json(float) == "number"

    def test_bool(self):
        assert _python_type_to_json(bool) == "boolean"

    def test_list(self):
        assert _python_type_to_json(list) == "array"

    def test_dict(self):
        assert _python_type_to_json(dict) == "object"

    def test_unknown_defaults_to_string(self):
        assert _python_type_to_json(bytes) == "string"


class TestExtractParamDescriptions:
    """Tests for docstring parameter description extraction."""

    def test_google_style(self):
        def func():
            """Do something.

            Args:
                path: The file path to read.
                content: The content to write.
            """
        descs = _extract_param_descriptions(func)
        assert descs["path"] == "The file path to read."
        assert descs["content"] == "The content to write."

    def test_no_docstring(self):
        def func():
            pass
        assert _extract_param_descriptions(func) == {}

    def test_no_args_section(self):
        def func():
            """Just a description."""
        assert _extract_param_descriptions(func) == {}

    def test_stops_at_returns(self):
        def func():
            """Do something.

            Args:
                path: The file path.

            Returns:
                The content.
            """
        descs = _extract_param_descriptions(func)
        assert "path" in descs
        assert len(descs) == 1


class TestInferParameters:
    """Tests for parameter inference from function signatures."""

    def test_simple_params(self):
        def func(path: str, count: int):
            pass
        params = _infer_parameters(func)
        assert len(params) == 2
        assert params[0].name == "path"
        assert params[0].type == "string"
        assert params[0].required is True
        assert params[1].name == "count"
        assert params[1].type == "integer"

    def test_optional_params(self):
        def func(path: str, verbose: bool = False):
            pass
        params = _infer_parameters(func)
        assert params[0].required is True
        assert params[1].required is False
        assert params[1].default is False

    def test_skips_self_and_kwargs(self):
        def func(self, path: str, **kwargs):
            pass
        params = _infer_parameters(func)
        assert len(params) == 1
        assert params[0].name == "path"

    def test_no_type_hints(self):
        def func(path, count):
            pass
        params = _infer_parameters(func)
        assert len(params) == 2
        assert params[0].type == "string"  # Default

    def test_descriptions_from_docstring(self):
        def func(path: str):
            """Read a file.

            Args:
                path: The file to read.
            """
        params = _infer_parameters(func)
        assert params[0].description == "The file to read."


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_basic_sync_function(self):
        @tool(description="Echo back the input")
        def echo(message: str) -> str:
            return message

        assert isinstance(echo, DecoratedTool)
        assert isinstance(echo, Tool)
        assert echo.name == "echo"
        assert echo.description == "Echo back the input"

    def test_custom_name(self):
        @tool(name="my_echo", description="Echo tool")
        def echo(message: str):
            return message

        assert echo.name == "my_echo"

    def test_category_and_confirmation(self):
        @tool(
            description="Delete a file",
            category=ToolCategory.FILESYSTEM,
            requires_confirmation=True,
        )
        def delete_file(path: str):
            pass

        assert echo_tool.category == ToolCategory.FILESYSTEM if False else True
        assert delete_file.category == ToolCategory.FILESYSTEM
        assert delete_file.requires_confirmation is True

    def test_description_from_docstring(self):
        @tool()
        def my_tool(x: int):
            """Compute the square of x."""
            return x * x

        assert my_tool.description == "Compute the square of x."

    def test_parameters_inferred(self):
        @tool(description="Search for files")
        def search(query: str, limit: int = 10, recursive: bool = True):
            pass

        params = my_tool.parameters if False else search.parameters
        assert len(params) == 3
        assert params[0].name == "query"
        assert params[0].type == "string"
        assert params[0].required is True
        assert params[1].name == "limit"
        assert params[1].type == "integer"
        assert params[1].required is False
        assert params[2].name == "recursive"
        assert params[2].type == "boolean"

    def test_openai_schema(self):
        @tool(description="Read a file")
        def read(path: str):
            """Read a file.

            Args:
                path: Path to read.
            """
            pass

        schema = read.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read"
        assert schema["function"]["description"] == "Read a file"
        props = schema["function"]["parameters"]["properties"]
        assert "path" in props
        assert props["path"]["type"] == "string"
        assert "path" in schema["function"]["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_sync_execution(self):
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        result = await add.execute(a=3, b=4)
        assert result.success is True
        assert result.output == "7"

    @pytest.mark.asyncio
    async def test_async_execution(self):
        @tool(description="Async add")
        async def async_add(a: int, b: int) -> int:
            return a + b

        result = await async_add.execute(a=5, b=6)
        assert result.success is True
        assert result.output == "11"

    @pytest.mark.asyncio
    async def test_returns_tool_result(self):
        @tool(description="Custom result")
        def custom():
            return ToolResult(success=True, output="custom", data={"key": "val"})

        result = await custom.execute()
        assert result.success is True
        assert result.output == "custom"
        assert result.data == {"key": "val"}

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        @tool(description="Failing tool")
        def fail():
            raise ValueError("Something broke")

        result = await fail.execute()
        assert result.success is False
        assert "Something broke" in result.error

    def test_register_with_registry(self):
        @tool(description="Test tool")
        def my_tool(x: str):
            return x

        registry = ToolRegistry()
        registry.register(my_tool)

        assert registry.get("my_tool") is my_tool
        assert len(registry.get_schemas()) == 1

    def test_multiple_decorated_tools(self):
        @tool(description="Tool A")
        def tool_a(x: str):
            return x

        @tool(description="Tool B")
        def tool_b(y: int):
            return y

        registry = ToolRegistry()
        registry.register(tool_a)
        registry.register(tool_b)

        assert len(registry.list_tools()) == 2
        assert registry.get("tool_a").description == "Tool A"
        assert registry.get("tool_b").description == "Tool B"
