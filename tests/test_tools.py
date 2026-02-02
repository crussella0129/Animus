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


class TestGitTool:
    """Tests for GitTool."""

    @pytest.fixture
    def git_tool(self):
        """Create a git tool with auto-confirm callback."""
        async def auto_confirm(msg):
            return True
        from src.tools.git import GitTool
        return GitTool(confirm_callback=auto_confirm)

    @pytest.fixture
    def git_tool_deny(self):
        """Create a git tool that denies all confirmations."""
        async def deny_confirm(msg):
            return False
        from src.tools.git import GitTool
        return GitTool(confirm_callback=deny_confirm)

    def test_git_tool_name(self, git_tool):
        """Test git tool name."""
        assert git_tool.name == "git"

    def test_git_tool_requires_confirmation(self, git_tool):
        """Test that git tool requires confirmation."""
        assert git_tool.requires_confirmation is True

    @pytest.mark.asyncio
    async def test_git_status(self, git_tool):
        """Test git status operation."""
        result = await git_tool.execute(operation="status")
        # Even outside a git repo, it should return a result
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_git_unknown_operation(self, git_tool):
        """Test unknown git operation."""
        result = await git_tool.execute(operation="unknown_op")
        assert result.success is False
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_git_commit_requires_message(self, git_tool):
        """Test that commit requires a message."""
        result = await git_tool.execute(operation="commit")
        assert result.success is False
        assert "message required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_git_add_requires_files(self, git_tool):
        """Test that add requires files."""
        result = await git_tool.execute(operation="add")
        assert result.success is False
        assert "specify files" in result.error.lower()

    @pytest.mark.asyncio
    async def test_git_checkout_requires_target(self, git_tool):
        """Test that checkout requires a target."""
        result = await git_tool.execute(operation="checkout")
        assert result.success is False
        assert "specify" in result.error.lower()

    @pytest.mark.asyncio
    async def test_git_push_confirmation_denied(self, git_tool_deny):
        """Test that push can be cancelled."""
        result = await git_tool_deny.execute(operation="push")
        assert result.success is False
        assert "cancelled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_git_checkout_confirmation_denied(self, git_tool_deny):
        """Test that checkout can be cancelled."""
        result = await git_tool_deny.execute(operation="checkout", args="main")
        assert result.success is False
        assert "cancelled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_git_raw_requires_args(self, git_tool):
        """Test that raw operation requires arguments."""
        result = await git_tool.execute(operation="raw")
        assert result.success is False
        assert "specify" in result.error.lower()

    def test_git_tool_in_registry(self):
        """Test that git tool is in default registry."""
        registry = create_default_registry()
        git = registry.get("git")
        assert git is not None
        assert git.name == "git"
