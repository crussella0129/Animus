"""Code-as-Action Mode — sandboxed Python execution with tool access.

Allows the agent to write Python code that calls tool functions directly,
collapsing N tool-call round trips into a single code-generation step.
Executes in a restricted sandbox with timeout enforcement.

Implementation Principle: Hardcoded sandbox restrictions. No LLM
involvement in security decisions.

Usage:
    sandbox = CodeActionSandbox(tool_registry)

    # Pre-flight analysis
    snapshot = sandbox.analyze(code)
    if snapshot.requires_confirmation:
        # Human-in-the-loop approval
        ...

    # Execute
    result = await sandbox.execute(code, timeout_s=30)
    print(result.output)
    print(result.tool_results)
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# Builtins allowed in sandbox — deliberately minimal
SAFE_BUILTINS = {
    # Types
    "True", "False", "None",
    "int", "float", "str", "bool", "bytes", "bytearray",
    "list", "tuple", "dict", "set", "frozenset",
    "type", "object",
    # Functions
    "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "min", "max", "sum", "abs",
    "round", "divmod", "pow",
    "all", "any",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "repr", "str", "int", "float", "bool",
    "print",  # Captured via redirect_stdout
    "format",
    "iter", "next",
    "chr", "ord",
    "hex", "oct", "bin",
    "hash", "id",
    "callable",
    "super",
    "property", "staticmethod", "classmethod",
    "__build_class__",  # Required for class definitions
    "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "OverflowError", "ArithmeticError",
    "LookupError", "NameError", "NotImplementedError",
    "AssertionError",
    "Exception", "BaseException",
}

# Builtins explicitly blocked
BLOCKED_BUILTINS = {
    "exec", "eval", "compile",
    "__import__",
    "open", "input",
    "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit",
    "memoryview",
}

# AST node types that are forbidden
BLOCKED_AST_NODES = {
    ast.Import,
    ast.ImportFrom,
}

# Attribute access patterns that are blocked
BLOCKED_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__",
    "__globals__", "__code__", "__closure__",
    "__builtins__", "__import__",
    "__dict__",  # Prevent namespace escape
}


@dataclass
class CodeActionResult:
    """Result of executing code in the sandbox."""

    success: bool
    output: str  # Captured stdout
    errors: str  # Captured stderr
    return_value: Any = None  # Value of last expression
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list = field(default_factory=list)  # list[ToolResult]
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class CodeSnapshot:
    """Pre-execution analysis of code to be run."""

    code: str
    tool_names_referenced: list[str]  # Tool names found in code
    requires_confirmation: bool  # Any tool requires confirmation?
    blocked_operations: list[str]  # Forbidden operations found
    is_safe: bool  # True if no blocked operations found

    @property
    def summary(self) -> str:
        """Human-readable summary for approval."""
        parts = [f"Tools: {', '.join(self.tool_names_referenced) or 'none'}"]
        if self.blocked_operations:
            parts.append(f"BLOCKED: {', '.join(self.blocked_operations)}")
        if self.requires_confirmation:
            parts.append("Requires confirmation")
        return " | ".join(parts)


class SandboxViolation(Exception):
    """Raised when code attempts a forbidden operation."""
    pass


class CodeActionSandbox:
    """Sandboxed Python execution environment with tool access.

    Tools from the registry are injected as callable functions.
    Dangerous operations (imports, file I/O, eval/exec) are blocked.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        max_output_chars: int = 50_000,
        allowed_modules: Optional[dict[str, Any]] = None,
    ):
        """Initialize sandbox.

        Args:
            registry: Tool registry providing callable tools.
            max_output_chars: Max characters captured from stdout.
            allowed_modules: Optional pre-approved module objects to inject
                (e.g. {"json": json, "re": re}). No import statements allowed.
        """
        self._registry = registry
        self._max_output = max_output_chars
        self._allowed_modules = allowed_modules or {}

    def analyze(self, code: str) -> CodeSnapshot:
        """Analyze code before execution.

        Identifies which tools are referenced, checks for blocked operations,
        and determines if confirmation is needed.

        Args:
            code: Python source code to analyze.

        Returns:
            CodeSnapshot with analysis results.
        """
        blocked = []
        tool_names = []

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return CodeSnapshot(
                code=code,
                tool_names_referenced=[],
                requires_confirmation=False,
                blocked_operations=[f"SyntaxError: {e}"],
                is_safe=False,
            )

        # Walk AST for violations
        for node in ast.walk(tree):
            # Check blocked node types
            if type(node) in BLOCKED_AST_NODES:
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    blocked.append(f"import {', '.join(names)}")
                elif isinstance(node, ast.ImportFrom):
                    blocked.append(f"from {node.module} import ...")

            # Check blocked attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in BLOCKED_ATTRIBUTES:
                    blocked.append(f"attribute access: .{node.attr}")

            # Check for calls to blocked builtins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        blocked.append(f"builtin call: {node.func.id}()")

            # Track function calls that match tool names
            if isinstance(node, ast.Call):
                name = self._extract_call_name(node)
                if name and self._registry.get(name):
                    tool_names.append(name)

        # Check if any referenced tool requires confirmation
        requires_confirm = False
        for name in tool_names:
            tool = self._registry.get(name)
            if tool and tool.requires_confirmation:
                requires_confirm = True
                break

        return CodeSnapshot(
            code=code,
            tool_names_referenced=list(dict.fromkeys(tool_names)),  # dedup, preserve order
            requires_confirmation=requires_confirm,
            blocked_operations=blocked,
            is_safe=len(blocked) == 0,
        )

    async def execute(
        self,
        code: str,
        timeout_s: float = 30.0,
        extra_vars: Optional[dict[str, Any]] = None,
    ) -> CodeActionResult:
        """Execute Python code in the sandbox.

        Runs code in a daemon thread with timeout enforcement so that
        CPU-bound infinite loops can be interrupted.

        Args:
            code: Python source code to execute.
            timeout_s: Maximum execution time in seconds.
            extra_vars: Additional variables to inject into namespace.

        Returns:
            CodeActionResult with outputs and tool results.
        """
        import concurrent.futures
        import threading

        start = time.perf_counter()

        # Pre-flight check
        snapshot = self.analyze(code)
        if not snapshot.is_safe:
            return CodeActionResult(
                success=False,
                output="",
                errors="",
                error=f"Blocked operations: {', '.join(snapshot.blocked_operations)}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Build namespace
        tool_tracker = _ToolCallTracker()
        namespace = self._build_namespace(tool_tracker, extra_vars)

        # Compile
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return CodeActionResult(
                success=False,
                output="",
                errors="",
                error=f"SyntaxError: {e}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Separate last expression for return value capture
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()

        try:
            compiled_body = compile(
                ast.Module(body=tree.body, type_ignores=[]),
                "<code-action>",
                "exec",
            )
            compiled_expr = None
            if last_expr:
                compiled_expr = compile(
                    ast.Expression(body=last_expr.value),
                    "<code-action>",
                    "eval",
                )
        except SyntaxError as e:
            return CodeActionResult(
                success=False,
                output="",
                errors="",
                error=f"CompileError: {e}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Execute in a daemon thread with timeout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Get the current event loop for tool call scheduling
        loop = asyncio.get_running_loop()
        tool_tracker.loop = loop

        def _run_in_thread():
            return_value = None
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled_body, namespace)  # noqa: S102 — intentional sandbox exec
                if compiled_expr:
                    return_value = eval(compiled_expr, namespace)  # noqa: S307 — intentional sandbox eval
            return return_value

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = loop.run_in_executor(executor, _run_in_thread)
            return_value = await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return CodeActionResult(
                success=False,
                output=stdout_capture.getvalue()[:self._max_output],
                errors=stderr_capture.getvalue()[:self._max_output],
                tool_calls=tool_tracker.calls,
                tool_results=tool_tracker.results,
                error=f"Execution timed out after {timeout_s}s",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        except SandboxViolation as e:
            return CodeActionResult(
                success=False,
                output=stdout_capture.getvalue()[:self._max_output],
                errors=stderr_capture.getvalue()[:self._max_output],
                tool_calls=tool_tracker.calls,
                tool_results=tool_tracker.results,
                error=f"Sandbox violation: {e}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            tb = traceback.format_exc()
            return CodeActionResult(
                success=False,
                output=stdout_capture.getvalue()[:self._max_output],
                errors=stderr_capture.getvalue()[:self._max_output],
                tool_calls=tool_tracker.calls,
                tool_results=tool_tracker.results,
                error=f"{type(e).__name__}: {e}\n{tb}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Drain any pending tool coroutines
        await tool_tracker.drain()

        elapsed = (time.perf_counter() - start) * 1000
        output = stdout_capture.getvalue()[:self._max_output]
        errors = stderr_capture.getvalue()[:self._max_output]

        return CodeActionResult(
            success=True,
            output=output,
            errors=errors,
            return_value=return_value,
            tool_calls=tool_tracker.calls,
            tool_results=tool_tracker.results,
            execution_time_ms=elapsed,
        )

    def _build_namespace(
        self,
        tracker: "_ToolCallTracker",
        extra_vars: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build restricted execution namespace.

        Injects:
        - Safe builtins
        - Tool wrapper functions
        - Allowed modules
        - Extra variables
        """
        # Restricted builtins
        import builtins
        safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS
                if hasattr(builtins, name)}

        namespace: dict[str, Any] = {
            "__builtins__": safe,
            "__name__": "<code-action>",  # Required for class definitions
        }

        # Inject allowed modules
        for mod_name, mod_obj in self._allowed_modules.items():
            namespace[mod_name] = mod_obj

        # Inject tool wrappers
        for tool in self._registry.list_tools():
            wrapper = self._make_tool_wrapper(tool, tracker)
            namespace[tool.name] = wrapper

        # Extra variables
        if extra_vars:
            namespace.update(extra_vars)

        return namespace

    def _make_tool_wrapper(self, tool, tracker: "_ToolCallTracker"):
        """Create a synchronous wrapper for an async tool.

        Since code runs in a thread, tool calls are scheduled back
        to the main event loop via run_coroutine_threadsafe.
        """
        def _sync_wrapper(**kwargs):
            tracker.calls.append({"name": tool.name, "arguments": kwargs})

            async def _do_call():
                try:
                    return await tool.execute(**kwargs)
                except Exception as e:
                    from src.tools.base import ToolResult
                    return ToolResult(success=False, output="", error=str(e))

            # Schedule on the main event loop from the worker thread
            if tracker.loop is not None:
                future = asyncio.run_coroutine_threadsafe(_do_call(), tracker.loop)
                try:
                    result = future.result(timeout=30)
                except Exception as e:
                    from src.tools.base import ToolResult
                    result = ToolResult(success=False, output="", error=str(e))
            else:
                # Fallback: create a new event loop
                result = asyncio.run(_do_call())

            tracker.results.append(result)
            return result

        _sync_wrapper.__name__ = tool.name
        _sync_wrapper.__doc__ = tool.description
        return _sync_wrapper

    @staticmethod
    def _extract_call_name(node: ast.Call) -> Optional[str]:
        """Extract function name from an AST Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None


class _ToolCallTracker:
    """Tracks tool calls and results during code execution."""

    def __init__(self):
        self.calls: list[dict] = []
        self.results: list = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def drain(self):
        """No-op — tool calls are now synchronous via run_coroutine_threadsafe."""
        pass


def generate_tool_api_docs(registry: "ToolRegistry") -> str:
    """Generate API documentation for tools available in the sandbox.

    Useful for injecting into the system prompt so the LLM knows
    what functions are available and their signatures.

    Args:
        registry: Tool registry with registered tools.

    Returns:
        Formatted string documenting all tool functions.
    """
    lines = ["# Available Tool Functions", ""]

    for tool in registry.list_tools():
        params = tool.parameters
        param_strs = []
        for p in params:
            if p.required:
                param_strs.append(f"{p.name}: {p.type}")
            else:
                param_strs.append(f"{p.name}: {p.type} = {p.default!r}")

        sig = ", ".join(param_strs)
        lines.append(f"## {tool.name}({sig})")
        lines.append(f"{tool.description}")
        lines.append("")

        if params:
            lines.append("Parameters:")
            for p in params:
                req = " (required)" if p.required else ""
                lines.append(f"  - {p.name}: {p.description}{req}")
            lines.append("")

    return "\n".join(lines)
