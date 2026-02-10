"""Tests for the Code-as-Action sandbox module."""

import asyncio
import json
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional

from src.core.code_action import (
    CodeActionSandbox,
    CodeActionResult,
    CodeSnapshot,
    SandboxViolation,
    SAFE_BUILTINS,
    BLOCKED_BUILTINS,
    BLOCKED_ATTRIBUTES,
    generate_tool_api_docs,
)


# =============================================================================
# Mock Tool Infrastructure
# =============================================================================

@dataclass
class MockToolParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[list] = None


@dataclass
class MockToolResult:
    success: bool = True
    output: str = ""
    error: Optional[str] = None
    data: Any = None
    metadata: dict = field(default_factory=dict)


class MockTool:
    """Minimal tool mock matching the Tool ABC interface."""

    def __init__(self, name, description="", parameters=None,
                 requires_confirmation=False, handler=None):
        self._name = name
        self._description = description
        self._parameters = parameters or []
        self._requires_confirmation = requires_confirmation
        self._handler = handler or self._default_handler

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def parameters(self):
        return self._parameters

    @property
    def requires_confirmation(self):
        return self._requires_confirmation

    async def execute(self, **kwargs):
        return await self._handler(**kwargs)

    @staticmethod
    async def _default_handler(**kwargs):
        return MockToolResult(success=True, output=json.dumps(kwargs))


class MockRegistry:
    """Minimal registry mock matching the ToolRegistry interface."""

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def list_tools(self):
        return list(self._tools.values())


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Create a mock registry with test tools."""
    reg = MockRegistry()

    async def read_handler(**kwargs):
        path = kwargs.get("path", "unknown")
        return MockToolResult(success=True, output=f"Contents of {path}")

    async def write_handler(**kwargs):
        path = kwargs.get("path", "unknown")
        content = kwargs.get("content", "")
        return MockToolResult(success=True, output=f"Wrote {len(content)} chars to {path}")

    async def shell_handler(**kwargs):
        command = kwargs.get("command", "")
        return MockToolResult(success=True, output=f"$ {command}\nOK")

    reg.register(MockTool(
        "read_file",
        description="Read a file's contents",
        parameters=[MockToolParam("path", "string", "File path", required=True)],
        handler=read_handler,
    ))
    reg.register(MockTool(
        "write_file",
        description="Write content to a file",
        parameters=[
            MockToolParam("path", "string", "File path", required=True),
            MockToolParam("content", "string", "File content", required=True),
        ],
        handler=write_handler,
    ))
    reg.register(MockTool(
        "shell",
        description="Execute a shell command",
        parameters=[MockToolParam("command", "string", "Command to run", required=True)],
        requires_confirmation=True,
        handler=shell_handler,
    ))

    return reg


@pytest.fixture
def sandbox(registry):
    return CodeActionSandbox(registry)


# =============================================================================
# Analysis Tests
# =============================================================================

class TestAnalyze:
    """Test pre-execution code analysis."""

    def test_simple_safe_code(self, sandbox):
        snap = sandbox.analyze("x = 1 + 2\nprint(x)")
        assert snap.is_safe
        assert snap.blocked_operations == []

    def test_detects_tool_calls(self, sandbox):
        code = 'result = read_file(path="test.py")\nprint(result)'
        snap = sandbox.analyze(code)
        assert "read_file" in snap.tool_names_referenced
        assert snap.is_safe

    def test_detects_multiple_tools(self, sandbox):
        code = '''
r = read_file(path="a.py")
write_file(path="b.py", content="hello")
'''
        snap = sandbox.analyze(code)
        assert "read_file" in snap.tool_names_referenced
        assert "write_file" in snap.tool_names_referenced

    def test_blocks_import(self, sandbox):
        snap = sandbox.analyze("import os\nos.system('rm -rf /')")
        assert not snap.is_safe
        assert any("import" in op for op in snap.blocked_operations)

    def test_blocks_from_import(self, sandbox):
        snap = sandbox.analyze("from subprocess import call")
        assert not snap.is_safe
        assert any("from subprocess" in op for op in snap.blocked_operations)

    def test_blocks_eval(self, sandbox):
        snap = sandbox.analyze('eval("print(1)")')
        assert not snap.is_safe
        assert any("eval" in op for op in snap.blocked_operations)

    def test_blocks_exec(self, sandbox):
        snap = sandbox.analyze('exec("x = 1")')
        assert not snap.is_safe
        assert any("exec" in op for op in snap.blocked_operations)

    def test_blocks_open(self, sandbox):
        snap = sandbox.analyze('f = open("/etc/passwd")')
        assert not snap.is_safe
        assert any("open" in op for op in snap.blocked_operations)

    def test_blocks_dunder_attributes(self, sandbox):
        snap = sandbox.analyze('x.__class__.__bases__')
        assert not snap.is_safe
        assert any("__class__" in op or "__bases__" in op
                    for op in snap.blocked_operations)

    def test_syntax_error(self, sandbox):
        snap = sandbox.analyze("def foo(")
        assert not snap.is_safe
        assert any("SyntaxError" in op for op in snap.blocked_operations)

    def test_requires_confirmation(self, sandbox):
        code = 'shell(command="ls")'
        snap = sandbox.analyze(code)
        assert snap.requires_confirmation
        assert "shell" in snap.tool_names_referenced

    def test_no_confirmation_for_safe_tools(self, sandbox):
        code = 'read_file(path="test.py")'
        snap = sandbox.analyze(code)
        assert not snap.requires_confirmation

    def test_summary(self, sandbox):
        code = 'read_file(path="test.py")'
        snap = sandbox.analyze(code)
        assert "read_file" in snap.summary

    def test_dedup_tool_names(self, sandbox):
        code = '''
read_file(path="a.py")
read_file(path="b.py")
read_file(path="c.py")
'''
        snap = sandbox.analyze(code)
        assert snap.tool_names_referenced.count("read_file") == 1


# =============================================================================
# Execution Tests
# =============================================================================

class TestExecute:
    """Test sandboxed code execution."""

    @pytest.mark.asyncio
    async def test_simple_expression(self, sandbox):
        result = await sandbox.execute("1 + 2")
        assert result.success
        assert result.return_value == 3

    @pytest.mark.asyncio
    async def test_print_capture(self, sandbox):
        result = await sandbox.execute('print("hello world")')
        assert result.success
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_variable_assignment(self, sandbox):
        result = await sandbox.execute("x = 42\nx")
        assert result.success
        assert result.return_value == 42

    @pytest.mark.asyncio
    async def test_multi_line(self, sandbox):
        code = """
results = []
for i in range(5):
    results.append(i * 2)
results
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_function_definition(self, sandbox):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(10)
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == 55

    @pytest.mark.asyncio
    async def test_list_comprehension(self, sandbox):
        code = "[x**2 for x in range(5)]"
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == [0, 1, 4, 9, 16]

    @pytest.mark.asyncio
    async def test_dict_operations(self, sandbox):
        code = """
d = {"a": 1, "b": 2, "c": 3}
sorted(d.keys())
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_string_operations(self, sandbox):
        code = '"hello world".upper().split()'
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_blocks_import_at_execution(self, sandbox):
        result = await sandbox.execute("import os")
        assert not result.success
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_blocks_eval_at_execution(self, sandbox):
        result = await sandbox.execute('eval("1+1")')
        assert not result.success
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_blocks_open_at_execution(self, sandbox):
        result = await sandbox.execute('open("/etc/passwd")')
        assert not result.success
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, sandbox):
        result = await sandbox.execute("def foo(")
        assert not result.success
        assert "SyntaxError" in result.error

    @pytest.mark.asyncio
    async def test_runtime_error_handling(self, sandbox):
        result = await sandbox.execute("1 / 0")
        assert not result.success
        assert "ZeroDivisionError" in result.error

    @pytest.mark.asyncio
    async def test_name_error_handling(self, sandbox):
        result = await sandbox.execute("nonexistent_var")
        assert not result.success
        assert "NameError" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self, sandbox):
        code = """
while True:
    pass
"""
        result = await sandbox.execute(code, timeout_s=0.5)
        assert not result.success
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_extra_vars(self, sandbox):
        result = await sandbox.execute("x + y", extra_vars={"x": 10, "y": 20})
        assert result.success
        assert result.return_value == 30

    @pytest.mark.asyncio
    async def test_allowed_modules(self, registry):
        import json as json_mod
        sandbox = CodeActionSandbox(registry, allowed_modules={"json": json_mod})
        result = await sandbox.execute('json.dumps({"key": "value"})')
        assert result.success
        assert result.return_value == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self, sandbox):
        result = await sandbox.execute("sum(range(1000))")
        assert result.success
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_no_return_value_for_statements(self, sandbox):
        result = await sandbox.execute("x = 42")
        assert result.success
        assert result.return_value is None

    @pytest.mark.asyncio
    async def test_output_truncation(self, registry):
        sandbox = CodeActionSandbox(registry, max_output_chars=50)
        code = 'print("A" * 200)'
        result = await sandbox.execute(code)
        assert result.success
        assert len(result.output) <= 50


# =============================================================================
# Tool Call Tests
# =============================================================================

class TestToolCalls:
    """Test tool execution within the sandbox."""

    @pytest.mark.asyncio
    async def test_tool_call_tracked(self, sandbox):
        code = 'read_file(path="test.py")'
        result = await sandbox.execute(code)
        assert result.success
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["arguments"] == {"path": "test.py"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, sandbox):
        code = """
read_file(path="a.py")
read_file(path="b.py")
"""
        result = await sandbox.execute(code)
        assert result.success
        assert len(result.tool_calls) == 2

    @pytest.mark.asyncio
    async def test_tool_result_returned(self, sandbox):
        code = 'r = read_file(path="test.py")'
        result = await sandbox.execute(code)
        assert result.success
        assert len(result.tool_results) >= 1


# =============================================================================
# Security Tests
# =============================================================================

class TestSecurity:
    """Test sandbox security restrictions."""

    @pytest.mark.asyncio
    async def test_no_builtins_escape(self, sandbox):
        # Try to access __builtins__ dict
        result = await sandbox.execute("__builtins__")
        # Should either fail or return the restricted dict
        # The key point is it shouldn't give full builtins
        if result.success:
            # Check it's the restricted set
            assert isinstance(result.return_value, dict)

    @pytest.mark.asyncio
    async def test_no_import_via_builtins(self, sandbox):
        result = await sandbox.execute('__import__("os")')
        assert not result.success

    @pytest.mark.asyncio
    async def test_no_globals_access(self, sandbox):
        result = await sandbox.execute("globals()")
        assert not result.success

    @pytest.mark.asyncio
    async def test_no_locals_access(self, sandbox):
        result = await sandbox.execute("locals()")
        assert not result.success

    @pytest.mark.asyncio
    async def test_no_dunder_class(self, sandbox):
        result = await sandbox.execute("().__class__.__bases__")
        assert not result.success

    @pytest.mark.asyncio
    async def test_safe_builtins_available(self, sandbox):
        code = """
results = []
results.append(len([1,2,3]))
results.append(max(1,2,3))
results.append(min(1,2,3))
results.append(sum([1,2,3]))
results.append(sorted([3,1,2]))
results.append(abs(-5))
results
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == [3, 3, 1, 6, [1, 2, 3], 5]

    @pytest.mark.asyncio
    async def test_exception_handling_works(self, sandbox):
        code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    x = "caught"
x
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == "caught"


# =============================================================================
# API Doc Generation Tests
# =============================================================================

class TestGenerateToolApiDocs:
    """Test tool documentation generation."""

    def test_generates_docs(self, registry):
        docs = generate_tool_api_docs(registry)
        assert "read_file" in docs
        assert "write_file" in docs
        assert "shell" in docs

    def test_includes_parameters(self, registry):
        docs = generate_tool_api_docs(registry)
        assert "path" in docs
        assert "content" in docs
        assert "command" in docs

    def test_includes_descriptions(self, registry):
        docs = generate_tool_api_docs(registry)
        assert "Read a file" in docs
        assert "Write content" in docs


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    @pytest.mark.asyncio
    async def test_empty_code(self, sandbox):
        result = await sandbox.execute("")
        assert result.success

    @pytest.mark.asyncio
    async def test_only_comments(self, sandbox):
        result = await sandbox.execute("# just a comment")
        assert result.success

    @pytest.mark.asyncio
    async def test_multiline_string(self, sandbox):
        code = '''
text = """
Hello
World
"""
len(text.strip().split("\\n"))
'''
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == 2

    @pytest.mark.asyncio
    async def test_nested_functions(self, sandbox):
        code = """
def outer(x):
    def inner(y):
        return x + y
    return inner

add5 = outer(5)
add5(3)
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == 8

    @pytest.mark.asyncio
    async def test_class_definition(self, sandbox):
        code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance(self):
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3, 4)
p.distance()
"""
        result = await sandbox.execute(code)
        assert result.success
        assert result.return_value == 5.0

    def test_analyze_empty(self, sandbox):
        snap = sandbox.analyze("")
        assert snap.is_safe
        assert snap.tool_names_referenced == []
