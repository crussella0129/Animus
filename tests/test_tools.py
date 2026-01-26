"""Tests for agent tools."""

import pytest
import tempfile
from pathlib import Path

from src.tools.base import Tool, ToolParameter, ToolResult, ToolRegistry, ToolCategory
from src.tools.filesystem import ReadFileTool, WriteFileTool, ListDirectoryTool
from src.tools.shell import ShellTool
from src.tools import create_default_registry


class TestToolBase:
    def test_tool_parameter(self):
        param = ToolParameter(
            name="path",
            type="string",
            description="File path",
        )
        assert param.name == "path"
        assert param.required is True

    def test_tool_result_success(self):
        result = ToolResult(success=True, output="Hello")
        assert result.success is True
        assert result.output == "Hello"
        assert result.error is None

    def test_tool_result_failure(self):
        result = ToolResult(success=False, output="", error="Not found")
        assert result.success is False
        assert result.error == "Not found"


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ReadFileTool()
        registry.register(tool)
        assert registry.get("read_file") is tool

    def test_get_unknown_tool(self):
        registry = ToolRegistry()
        assert registry.get("unknown") is None

    def test_list_tools(self):
        registry = create_default_registry()
        tools = registry.list_tools()
        assert len(tools) >= 4  # read_file, write_file, list_dir, run_shell

    def test_get_schemas(self):
        registry = create_default_registry()
        schemas = registry.get_schemas()
        assert len(schemas) >= 4
        assert all("function" in schema for schema in schemas)


class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            f.flush()

            tool = ReadFileTool()
            result = await tool.execute(path=f.name)

            assert result.success is True
            assert result.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        tool = ReadFileTool()
        result = await tool.execute(path="/nonexistent/file.txt")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_line_range(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\nline4\nline5")
            f.flush()

            tool = ReadFileTool()
            result = await tool.execute(path=f.name, start_line=2, end_line=4)

            assert result.success is True
            assert "line2" in result.output
            assert "line4" in result.output
            assert "line1" not in result.output

    def test_tool_schema(self):
        tool = ReadFileTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"


class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            tool = WriteFileTool()
            result = await tool.execute(path=str(path), content="Hello!")

            assert result.success is True
            assert path.exists()
            assert path.read_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "test.txt"
            tool = WriteFileTool()
            result = await tool.execute(path=str(path), content="Nested!")

            assert result.success is True
            assert path.exists()

    def test_requires_confirmation(self):
        tool = WriteFileTool()
        assert tool.requires_confirmation is True


class TestListDirectoryTool:
    @pytest.mark.asyncio
    async def test_list_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("content")
            (Path(tmpdir) / "file2.py").write_text("code")
            (Path(tmpdir) / "subdir").mkdir()

            tool = ListDirectoryTool()
            result = await tool.execute(path=tmpdir)

            assert result.success is True
            assert "file1.txt" in result.output
            assert "file2.py" in result.output
            assert "subdir" in result.output

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self):
        tool = ListDirectoryTool()
        result = await tool.execute(path="/nonexistent/dir")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestShellTool:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        tool = ShellTool(confirm_callback=lambda cmd: True)
        # Use a cross-platform command
        result = await tool.execute(command="echo hello")

        assert result.success is True
        assert "hello" in result.output.lower()

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        tool = ShellTool()
        result = await tool.execute(command="rm -rf /")

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_destructive_detection(self):
        tool = ShellTool()
        assert tool._is_destructive("rm -rf temp")
        assert tool._is_destructive("git push origin main")
        assert not tool._is_destructive("ls -la")
        assert not tool._is_destructive("echo hello")

    def test_requires_confirmation(self):
        tool = ShellTool()
        assert tool.requires_confirmation is True
