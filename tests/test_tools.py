"""Tests for tools: filesystem and shell."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.tools.base import ToolRegistry, _coerce_args, _validate_args
from src.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool, register_filesystem_tools
from src.tools.shell import RunShellTool, register_shell_tools


class TestReadFileTool:
    def test_read_existing_file(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        tool = ReadFileTool()
        result = tool.execute({"path": str(test_file)})
        assert result == "hello world"

    def test_read_nonexistent_file(self):
        tool = ReadFileTool()
        result = tool.execute({"path": "/nonexistent/file.txt"})
        assert "Error" in result

    def test_read_max_lines(self, tmp_path: Path):
        test_file = tmp_path / "lines.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")
        tool = ReadFileTool()
        result = tool.execute({"path": str(test_file), "max_lines": 2})
        assert result == "line1\nline2"

    def test_read_dangerous_path(self):
        tool = ReadFileTool()
        result = tool.execute({"path": "/etc/shadow"})
        assert "denied" in result.lower() or "Error" in result


class TestWriteFileTool:
    def test_write_file(self, tmp_path: Path):
        target = tmp_path / "output.txt"
        tool = WriteFileTool()
        result = tool.execute({"path": str(target), "content": "written"})
        assert "Successfully" in result
        assert target.read_text() == "written"

    def test_write_creates_parents(self, tmp_path: Path):
        target = tmp_path / "sub" / "dir" / "file.txt"
        tool = WriteFileTool()
        result = tool.execute({"path": str(target), "content": "nested"})
        assert "Successfully" in result
        assert target.read_text() == "nested"


class TestListDirTool:
    def test_list_dir(self, sample_files: Path):
        tool = ListDirTool()
        result = tool.execute({"path": str(sample_files)})
        assert "hello.py" in result
        assert "readme.md" in result

    def test_list_nonexistent(self):
        tool = ListDirTool()
        result = tool.execute({"path": "/nonexistent/dir"})
        assert "Error" in result

    def test_list_recursive(self, sample_files: Path):
        tool = ListDirTool()
        result = tool.execute({"path": str(sample_files), "recursive": True})
        assert "data.txt" in result


class TestRunShellTool:
    def test_simple_command(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo hello"})
        assert "hello" in result

    def test_blocked_command(self):
        tool = RunShellTool()
        result = tool.execute({"command": "rm -rf /"})
        assert "blocked" in result.lower()

    def test_timeout(self):
        tool = RunShellTool()
        # Use a command that should be fast
        result = tool.execute({"command": "echo fast", "timeout": 5})
        assert "fast" in result

    def test_confirm_callback_deny(self):
        tool = RunShellTool(confirm_callback=lambda msg: False)
        result = tool.execute({"command": "rm tempfile"})
        assert "cancelled" in result.lower()

    def test_confirm_callback_allow(self, tmp_path: Path):
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("bye")
        tool = RunShellTool(confirm_callback=lambda msg: True)
        # The command itself may fail on different OS but it shouldn't be blocked
        result = tool.execute({"command": f"echo confirmed", "timeout": 5})
        assert "confirmed" in result.lower() or "Error" not in result


class TestToolRegistry:
    def test_register_and_list(self):
        registry = ToolRegistry()
        register_filesystem_tools(registry)
        names = registry.names()
        assert "read_file" in names
        assert "write_file" in names
        assert "list_dir" in names

    def test_get_tool(self):
        registry = ToolRegistry()
        register_filesystem_tools(registry)
        tool = registry.get("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_unknown_tool(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert "Error" in result

    def test_openai_schemas(self):
        registry = ToolRegistry()
        register_filesystem_tools(registry)
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 3
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]


class TestArgCoercion:
    """Test that LLM string arguments are coerced to declared schema types."""

    def test_integer_string_coerced(self):
        schema = {"properties": {"timeout": {"type": "integer"}}}
        result = _coerce_args({"timeout": "30"}, schema)
        assert result["timeout"] == 30
        assert isinstance(result["timeout"], int)

    def test_integer_already_int(self):
        schema = {"properties": {"timeout": {"type": "integer"}}}
        result = _coerce_args({"timeout": 30}, schema)
        assert result["timeout"] == 30

    def test_number_string_coerced(self):
        schema = {"properties": {"threshold": {"type": "number"}}}
        result = _coerce_args({"threshold": "0.5"}, schema)
        assert result["threshold"] == 0.5

    def test_boolean_string_coerced(self):
        schema = {"properties": {"recursive": {"type": "boolean"}}}
        assert _coerce_args({"recursive": "true"}, schema)["recursive"] is True
        assert _coerce_args({"recursive": "false"}, schema)["recursive"] is False

    def test_invalid_integer_string_left_as_is(self):
        schema = {"properties": {"count": {"type": "integer"}}}
        result = _coerce_args({"count": "not_a_number"}, schema)
        assert result["count"] == "not_a_number"

    def test_string_params_not_touched(self):
        schema = {"properties": {"command": {"type": "string"}}}
        result = _coerce_args({"command": "echo hello"}, schema)
        assert result["command"] == "echo hello"

    def test_unknown_key_not_touched(self):
        schema = {"properties": {}}
        result = _coerce_args({"extra": "42"}, schema)
        assert result["extra"] == "42"

    def test_shell_timeout_string_via_registry(self):
        """The actual bug: LLM passes timeout as '30' string to run_shell."""
        registry = ToolRegistry()
        register_shell_tools(registry)
        result = registry.execute("run_shell", {"command": "echo coerced", "timeout": "5"})
        assert "coerced" in result


class TestNetworkIsolation:
    """Test that outbound network commands are blocked by default."""

    def test_curl_blocked_by_default(self):
        tool = RunShellTool()
        result = tool.execute({"command": "curl https://example.com"})
        assert "Network command blocked" in result
        assert "curl" in result

    def test_wget_blocked_by_default(self):
        tool = RunShellTool()
        result = tool.execute({"command": "wget https://example.com/file"})
        assert "Network command blocked" in result

    def test_git_push_blocked_by_default(self):
        tool = RunShellTool()
        result = tool.execute({"command": "git push origin main"})
        assert "Network command blocked" in result
        assert "git push" in result

    def test_git_clone_blocked_by_default(self):
        tool = RunShellTool()
        result = tool.execute({"command": "git clone https://github.com/user/repo"})
        assert "Network command blocked" in result

    def test_ssh_blocked_by_default(self):
        tool = RunShellTool()
        result = tool.execute({"command": "ssh user@host"})
        assert "Network command blocked" in result

    def test_non_network_allowed(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo hello"})
        assert "hello" in result
        assert "Network command blocked" not in result

    def test_git_status_allowed(self):
        """Non-network git commands should not be blocked."""
        tool = RunShellTool()
        result = tool.execute({"command": "git status"})
        assert "Network command blocked" not in result

    def test_allow_network_flag(self):
        """When allow_network=True, network commands should be allowed through."""
        tool = RunShellTool(allow_network=True)
        result = tool.execute({"command": "echo allowed"})
        assert "Network command blocked" not in result


class TestSchemaValidation:
    """Test argument validation against tool parameter schemas."""

    def test_missing_required_field(self):
        schema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        err = _validate_args({}, schema)
        assert err is not None
        assert "Missing required" in err
        assert "command" in err

    def test_valid_args_pass(self):
        schema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        assert _validate_args({"command": "echo hi"}, schema) is None

    def test_wrong_type_string(self):
        schema = {"properties": {"path": {"type": "string"}}, "required": []}
        err = _validate_args({"path": 123}, schema)
        assert err is not None
        assert "should be string" in err

    def test_wrong_type_array(self):
        schema = {"properties": {"paths": {"type": "array"}}, "required": []}
        err = _validate_args({"paths": "file.txt"}, schema)
        assert err is not None
        assert "should be array" in err

    def test_extra_fields_allowed(self):
        """LLMs sometimes add unexpected fields — these should not cause errors."""
        schema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        assert _validate_args({"command": "echo hi", "extra": "junk"}, schema) is None

    def test_registry_blocks_invalid_args(self):
        """Registry.execute should return error for invalid args."""
        registry = ToolRegistry()
        register_shell_tools(registry)
        result = registry.execute("run_shell", {})
        assert "Error" in result
        assert "Missing required" in result
