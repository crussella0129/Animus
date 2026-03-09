# Feedback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the 5 priority items from the architectural feedback review.

**Architecture:** Two "this week" tasks address correctness gaps (Ornstein unenforced, GBNF only on turn 0). Three "this month" tasks address maintainability and robustness (API retries, function_tool decorator, main.py refactor).

**Tech Stack:** Python 3.11+, Typer, Rich, httpx, llama-cpp-python (optional), multiprocessing

---

## Priority Stack

| Priority | Task | Why |
|----------|------|-----|
| This week | Task 1: Enforce Ornstein for shell | Decorator is currently decorative — security gap |
| This week | Task 2: GBNF on all non-streaming turns | Multi-step tasks lose constraints after turn 0 |
| This month | Task 3: API retry/backoff | Rate limits kill multi-step tasks silently |
| This month | Task 4: @function_tool decorator | Reduce tool boilerplate by ~60% |
| This month | Task 5: Refactor main.py → cli/ | 955-line god module; rise() alone is ~300 lines |

---

## Task 1: Enforce Ornstein as Default for RunShellTool

**Problem:** `@isolated(level="ornstein")` is on `RunShellTool` but `ToolRegistry.execute()` calls
`tool.execute()` directly without checking `tool.isolation_level`. The `--cautious` flag sets config
but nothing plumbs that config into the actual execution path. The decorator is purely decorative.

**Approach:** Add `run_command()` to `OrnsteinSandbox` using a module-level subprocess wrapper
(picklable). Give `RunShellTool` an optional `sandbox` parameter. When set, use it instead of
bare `subprocess.run()`. In `main.py`, always create a default sandbox and pass it in — making
Ornstein the default for shell without requiring `--cautious`.

**Files:**
- Modify: `src/isolation/ornstein.py` — add `run_command(cmd_list, cwd, timeout)` method
- Modify: `src/tools/shell.py` — accept `sandbox` parameter in `RunShellTool.__init__()` and `register_shell_tools()`
- Modify: `src/main.py:611-614` — create default sandbox, pass to `register_shell_tools()`
- Modify: `src/tools/base.py:176-188` — log warning in `ToolRegistry.execute()` when isolated tool has no sandbox
- Test: `tests/test_tools.py` — verify sandboxed execution path
- Test: `tests/test_isolation.py` — verify `run_command()` on OrnsteinSandbox

---

### Step 1: Write failing test for OrnsteinSandbox.run_command()

In `tests/test_isolation.py`, add:

```python
def test_ornstein_sandbox_run_command_success():
    """run_command() executes a shell command in the sandbox and returns output."""
    from src.isolation.ornstein import create_sandbox
    sandbox = create_sandbox(timeout_seconds=10)
    result = sandbox.run_command(["python", "-c", "print('hello')"], cwd=None, timeout=5)
    assert result.success is True
    assert "hello" in result.output

def test_ornstein_sandbox_run_command_captures_stderr():
    """run_command() captures stderr when command fails."""
    from src.isolation.ornstein import create_sandbox
    sandbox = create_sandbox(timeout_seconds=10)
    result = sandbox.run_command(["python", "-c", "import sys; sys.exit(1)"], cwd=None, timeout=5)
    assert result.success is False
```

**Run:** `pytest tests/test_isolation.py::test_ornstein_sandbox_run_command_success -v`
**Expected:** FAIL — `OrnsteinSandbox` has no `run_command` method

---

### Step 2: Add run_command() to OrnsteinSandbox

In `src/isolation/ornstein.py`, add a module-level helper (picklable) and the method.

First, add the module-level subprocess worker near the top of the file (after imports):

```python
def _subprocess_worker(cmd_list: list[str], cwd: str | None, timeout: int) -> dict:
    """Module-level worker for OrnsteinSandbox.run_command(). Must be module-level for pickling."""
    import subprocess
    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}
```

Then add the method to `OrnsteinSandbox`:

```python
def run_command(
    self,
    cmd_list: list[str],
    cwd: str | None,
    timeout: int,
) -> "OrnsteinResult":
    """Execute a shell command in the sandbox. Returns OrnsteinResult."""
    return self.execute(_subprocess_worker, cmd_list, cwd, timeout)
```

**Run:** `pytest tests/test_isolation.py::test_ornstein_sandbox_run_command_success tests/test_isolation.py::test_ornstein_sandbox_run_command_captures_stderr -v`
**Expected:** PASS

---

### Step 3: Write failing test for sandboxed RunShellTool

In `tests/test_tools.py`, add:

```python
def test_run_shell_tool_uses_sandbox_when_provided(tmp_path):
    """When a sandbox is passed, RunShellTool routes execution through it."""
    from unittest.mock import MagicMock
    from src.tools.shell import RunShellTool
    from src.isolation.ornstein import OrnsteinResult

    mock_sandbox = MagicMock()
    mock_sandbox.run_command.return_value = OrnsteinResult(
        success=True, output="sandboxed", error="", resource_usage={}
    )
    tool = RunShellTool(
        confirm_callback=lambda _: True,
        session_cwd=None,
        sandbox=mock_sandbox,
    )
    result = tool.execute({"command": "echo hello"})
    assert mock_sandbox.run_command.called
    assert "sandboxed" in result

def test_run_shell_tool_no_sandbox_falls_back_to_subprocess():
    """When no sandbox is provided, RunShellTool uses subprocess directly."""
    from src.tools.shell import RunShellTool
    tool = RunShellTool(confirm_callback=lambda _: True, session_cwd=None, sandbox=None)
    result = tool.execute({"command": "python -c \"print('direct')\""})
    assert "direct" in result
```

**Run:** `pytest tests/test_tools.py::test_run_shell_tool_uses_sandbox_when_provided -v`
**Expected:** FAIL — `RunShellTool.__init__()` doesn't accept `sandbox` parameter

---

### Step 4: Add sandbox parameter to RunShellTool

In `src/tools/shell.py`, update `RunShellTool.__init__()`:

```python
def __init__(
    self,
    confirm_callback=None,
    session_cwd=None,
    sandbox=None,          # ← add this
) -> None:
    super().__init__()
    self._confirm_callback = confirm_callback
    self._session_cwd = session_cwd
    self._sandbox = sandbox                # ← add this
    self._budget = ExecutionBudget()
    self._permission_checker = PermissionChecker()
```

In `RunShellTool.execute()`, replace the `subprocess.run()` call with sandbox routing.
Find the block that calls `subprocess.run(cmd_list, ...)` and replace with:

```python
if self._sandbox is not None:
    cwd_str = str(self._session_cwd.path) if self._session_cwd else None
    sandbox_result = self._sandbox.run_command(cmd_list, cwd=cwd_str, timeout=timeout)
    if sandbox_result.success:
        return sandbox_result.output or "(no output)"
    return f"Error: {sandbox_result.error}"
else:
    result = subprocess.run(
        cmd_list,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(self._session_cwd.path) if self._session_cwd else None,
        stdin=subprocess.DEVNULL,
    )
```

Also update `register_shell_tools()` to accept and forward `sandbox`:

```python
def register_shell_tools(
    registry,
    confirm_callback=None,
    session_cwd=None,
    sandbox=None,           # ← add this
) -> None:
    registry.register(RunShellTool(
        confirm_callback=confirm_callback,
        session_cwd=session_cwd,
        sandbox=sandbox,    # ← add this
    ))
```

**Run:** `pytest tests/test_tools.py::test_run_shell_tool_uses_sandbox_when_provided tests/test_tools.py::test_run_shell_tool_no_sandbox_falls_back_to_subprocess -v`
**Expected:** PASS

---

### Step 5: Wire sandbox into main.py rise() by default

In `src/main.py`, find the tool registration section (~line 611). Replace:

```python
registry = ToolRegistry()
register_filesystem_tools(registry, session_cwd=session_cwd)
confirm_cb = _make_confirm_callback(cfg)
register_shell_tools(registry, confirm_callback=confirm_cb, session_cwd=session_cwd)
```

With:

```python
from src.isolation.ornstein import create_sandbox

registry = ToolRegistry()
register_filesystem_tools(registry, session_cwd=session_cwd)
confirm_cb = _make_confirm_callback(cfg)

# Ornstein is the default sandbox for shell tools.
# --cautious keeps existing behavior; default now always sandboxes.
shell_sandbox = create_sandbox(
    cpu_percent=cfg.isolation.cpu_limit_percent if hasattr(cfg.isolation, 'cpu_limit_percent') else 50.0,
    memory_mb=512,
    timeout_seconds=30,
)
register_shell_tools(
    registry,
    confirm_callback=confirm_cb,
    session_cwd=session_cwd,
    sandbox=shell_sandbox,
)
```

Also add isolation_level warning to `ToolRegistry.execute()` in `src/tools/base.py` (after the `tool is None` check):

```python
# Warn once if an isolated tool is being run without enforcement
if getattr(tool, '_isolation_level', 'none') != 'none':
    import warnings
    warnings.warn(
        f"Tool '{name}' has isolation_level='{tool.isolation_level}' but "
        "is being executed directly by ToolRegistry. Ensure sandbox is "
        "configured in the tool itself.",
        stacklevel=2,
    )
```

Wait — since sandbox is now IN the tool itself, remove this warning. The decorator is now enforced through the tool's own sandbox field. Keep `ToolRegistry.execute()` unchanged.

**Run:** `pytest tests/ -x -q --ignore=tests/test_gauntlet.py`
**Expected:** All pass

---

### Step 6: Commit

```bash
git add src/isolation/ornstein.py src/tools/shell.py src/main.py tests/test_isolation.py tests/test_tools.py
git commit -m "feat: enforce Ornstein sandbox as default for RunShellTool

- Add run_command() to OrnsteinSandbox using picklable module-level worker
- Add optional sandbox= parameter to RunShellTool and register_shell_tools()
- Wire default OrnsteinSandbox in rise() — shell always sandboxed now
- Remove --cautious requirement for basic process isolation"
```

---

## Task 2: GBNF Grammar on All Non-Streaming Turns

**Problem:** `use_grammar = (turn == 0)` in `_run_agentic_loop_stream()` means that after the
first tool result, the model generates unconstrained text. In a 5-step multi-step task, turns
1-4 get no grammar. Small models then produce free-form text that the three-strategy parser
has to rescue.

**Problem with naive fix:** Changing to `use_grammar = True` always creates a dead end — the
grammar forces `{"name": "...", "arguments": {...}}` JSON, so the model can never return a
natural-language final answer. We solve this by adding a `RespondTool` that the model calls
to give its final answer. This is the standard pattern in constrained-decoding agent loops.

**Files:**
- Modify: `src/tools/base.py` — add `RespondTool` class
- Modify: `src/core/agent.py` — auto-register RespondTool, change grammar flag, detect respond calls
- Test: `tests/test_agent.py` — verify grammar applied on turn 1+
- Test: `tests/test_tools.py` — verify RespondTool schema

---

### Step 1: Write failing test for RespondTool

In `tests/test_tools.py`, add:

```python
def test_respond_tool_exists_and_has_message_parameter():
    """RespondTool is importable and has a 'message' parameter in its schema."""
    from src.tools.base import RespondTool
    tool = RespondTool()
    assert tool.name == "respond"
    assert "message" in tool.parameters["properties"]
    assert "message" in tool.parameters.get("required", [])

def test_respond_tool_execute_returns_message():
    """RespondTool.execute() returns the message string directly."""
    from src.tools.base import RespondTool
    tool = RespondTool()
    result = tool.execute({"message": "Task complete."})
    assert result == "Task complete."
```

**Run:** `pytest tests/test_tools.py::test_respond_tool_exists_and_has_message_parameter -v`
**Expected:** FAIL — `RespondTool` not in `src/tools/base.py`

---

### Step 2: Add RespondTool to src/tools/base.py

Add after the `isolated` decorator (after line ~91):

```python
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
```

**Run:** `pytest tests/test_tools.py::test_respond_tool_exists_and_has_message_parameter tests/test_tools.py::test_respond_tool_execute_returns_message -v`
**Expected:** PASS

---

### Step 3: Write failing test for grammar on turn 1+

In `tests/test_agent.py`, add:

```python
def test_agent_applies_grammar_on_all_non_streaming_turns():
    """In non-streaming mode, _step() is called with use_grammar=True on every turn."""
    from unittest.mock import MagicMock, patch, call
    from src.core.agent import Agent
    from src.tools.base import ToolRegistry

    mock_provider = MagicMock()
    # Turn 0: returns tool call; Turn 1: calls respond tool
    mock_provider.generate.side_effect = [
        '{"name": "read_file", "arguments": {"path": "x.py"}}',
        '{"name": "respond", "arguments": {"message": "Done"}}',
    ]
    mock_provider.capabilities.return_value = MagicMock(
        context_length=4096, size_tier="small", supports_tools=False
    )

    registry = ToolRegistry()
    # Register a minimal mock tool
    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.description = "Read a file"
    mock_tool.parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    mock_tool.isolation_level = "none"
    mock_tool.execute.return_value = "file contents"
    mock_tool.to_openai_schema.return_value = {"type": "function", "function": {"name": "read_file"}}
    registry.register(mock_tool)

    agent = Agent(provider=mock_provider, tool_registry=registry, system_prompt="test")

    with patch.object(agent, '_step', wraps=agent._step) as mock_step:
        agent.run("read x.py then respond")

    # All non-streaming _step() calls should have use_grammar=True
    for call_args in mock_step.call_args_list:
        assert call_args == call(use_grammar=True), (
            f"Expected use_grammar=True on all turns, got {call_args}"
        )
```

**Run:** `pytest tests/test_agent.py::test_agent_applies_grammar_on_all_non_streaming_turns -v`
**Expected:** FAIL — currently turn 0 is True but turn 1 is False

---

### Step 4: Fix grammar flag in agent.py

In `src/core/agent.py`, find line 302:

```python
use_grammar = (turn == 0)
```

Replace with:

```python
# Apply grammar on all non-streaming turns. Streaming cannot use grammar
# (llama-cpp-python limitation). With RespondTool registered, the model
# calls respond() to return its final answer rather than free-form text.
use_grammar = (on_chunk is None)
```

---

### Step 5: Auto-register RespondTool and detect it in the loop

In `src/core/agent.py`, in `Agent.__init__()`, after tool registry is stored, auto-register RespondTool:

```python
from src.tools.base import RespondTool
if "respond" not in self._tools.names():
    self._tools.register(RespondTool())
```

In `_run_agentic_loop_stream()`, after tool calls are executed, detect respond and short-circuit:

Find the section that iterates over tool calls and builds tool results. After calling `_execute_tool()`, add:

```python
# RespondTool signals that the model is done. Return immediately.
if tool_name == "respond":
    return result  # result is the message string from RespondTool.execute()
```

This should go immediately after the result is obtained for each tool call in the loop. Inspect the actual loop structure to find the right insertion point — look for where `_execute_tool()` is called and `last_tool_result` is set.

**Run:** `pytest tests/test_agent.py::test_agent_applies_grammar_on_all_non_streaming_turns -v`
**Expected:** PASS

---

### Step 6: Run full suite

```bash
pytest tests/ -x -q --ignore=tests/test_gauntlet.py
```

**Expected:** All pass (the respond tool adds a tool to every agent, so grammar tests and schema tests may need minor adjustments — update counts as needed)

---

### Step 7: Commit

```bash
git add src/tools/base.py src/core/agent.py tests/test_tools.py tests/test_agent.py
git commit -m "feat: apply GBNF grammar on all non-streaming turns

- Add RespondTool: model calls respond() to return final answers under grammar
- Auto-register RespondTool in Agent.__init__() when not already present
- Change use_grammar from turn==0 to on_chunk is None (all non-streaming turns)
- Detect respond tool call in agentic loop and return immediately"
```

---

## Task 3: Retry/Backoff for API Providers

**Problem:** `api.py` uses `httpx.Client.post()` with no retry logic. When OpenAI 429s or
Anthropic 529s, the agent dies mid-task. Multi-step tool-use loops making 10+ sequential
calls are especially fragile.

**Files:**
- Modify: `src/llm/api.py` — add `_retry_with_backoff()` helper, apply in `generate()` and `generate_stream()`
- Test: `tests/test_llm.py` — verify retry on 429/529, verify no retry on 400/404

---

### Step 1: Write failing tests

In `tests/test_llm.py`, add:

```python
def test_openai_provider_retries_on_429(respx_mock):
    """OpenAIProvider retries up to 3 times on HTTP 429."""
    import httpx
    from src.llm.api import OpenAIProvider

    call_count = 0
    def rate_limit_then_ok(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}}]
        })

    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=rate_limit_then_ok)
    provider = OpenAIProvider(api_key="test-key", model_name="gpt-4")
    result = provider.generate([{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert call_count == 3

def test_openai_provider_does_not_retry_on_400():
    """OpenAIProvider raises immediately on 400 (bad request — not transient)."""
    import httpx
    from unittest.mock import patch, MagicMock
    from src.llm.api import OpenAIProvider

    with patch("httpx.Client") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=MagicMock(status_code=400)
        )
        mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_response
        provider = OpenAIProvider(api_key="test-key")
        try:
            provider.generate([{"role": "user", "content": "hi"}])
            assert False, "Should have raised"
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == 400
```

Note: if `respx` is not already a dev dependency, use `unittest.mock` instead. Check `pyproject.toml` first.

**Run:** `pytest tests/test_llm.py::test_openai_provider_retries_on_429 -v`
**Expected:** FAIL — no retry logic in api.py

---

### Step 2: Add _retry_with_backoff() to api.py

Add near the top of `src/llm/api.py` (after imports):

```python
import time

_RETRYABLE_STATUS_CODES = {429, 503, 529}
_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0


def _retry_with_backoff(func):
    """Call func(), retrying on transient HTTP errors (429, 503, 529).

    Uses exponential backoff: 1s, 2s, 4s between retries.
    Non-retryable errors (400, 401, 404, etc.) propagate immediately.
    """
    import httpx
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                time.sleep(delay)
    raise last_exc
```

---

### Step 3: Apply retry to OpenAIProvider.generate()

In `OpenAIProvider.generate()`, wrap the httpx call:

Replace:
```python
with httpx.Client(timeout=self._timeout) as client:
    response = client.post(
        f"{self._base_url}/chat/completions",
        headers=self._headers(),
        json=payload,
    )
    response.raise_for_status()
```

With:
```python
def _call():
    with httpx.Client(timeout=self._timeout) as client:
        response = client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        return response

response = _retry_with_backoff(_call)
```

Apply the same pattern to `AnthropicProvider.generate()`.

Note: `generate_stream()` is harder to retry since it uses a streaming context. For now, only apply retry to `generate()`. Add a TODO comment on `generate_stream()`.

---

### Step 4: Run tests

**Run:** `pytest tests/test_llm.py -v`
**Expected:** New retry tests pass; existing tests unaffected

---

### Step 5: Commit

```bash
git add src/llm/api.py tests/test_llm.py
git commit -m "feat: add exponential backoff retry to API providers

- Add _retry_with_backoff() helper with 3 retries, 1s/2s/4s delays
- Retry on HTTP 429/503/529 (transient); propagate 4xx immediately
- Apply to OpenAIProvider.generate() and AnthropicProvider.generate()
- TODO: streaming retry support"
```

---

## Task 4: @function_tool Decorator

**Problem:** Every tool requires 30+ lines of boilerplate (class, 3 abstract properties,
execute() method). For simple tools with no execution budget or sandbox needs, this is
excessive friction.

**Files:**
- Modify: `src/tools/base.py` — add `function_tool` decorator
- Test: `tests/test_tools.py` — verify schema generation, type mapping, execution

---

### Step 1: Write failing tests

In `tests/test_tools.py`, add:

```python
def test_function_tool_decorator_creates_tool_class():
    """@function_tool creates a Tool subclass from a plain function."""
    from src.tools.base import function_tool, Tool

    @function_tool(description="Add two numbers together")
    def add(a: int, b: int) -> int:
        return a + b

    instance = add()
    assert isinstance(instance, Tool)
    assert instance.name == "add"
    assert instance.description == "Add two numbers together"

def test_function_tool_generates_schema_from_type_hints():
    """@function_tool generates JSON Schema from type annotations."""
    from src.tools.base import function_tool

    @function_tool(description="Greet a user")
    def greet(name: str, loud: bool) -> str:
        return name.upper() if loud else name

    instance = greet()
    schema = instance.parameters
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["loud"]["type"] == "boolean"
    assert "name" in schema["required"]
    assert "loud" in schema["required"]

def test_function_tool_execute_calls_function():
    """@function_tool execute() calls the wrapped function with args."""
    from src.tools.base import function_tool

    @function_tool(description="Multiply")
    def multiply(x: int, y: int) -> int:
        return x * y

    instance = multiply()
    result = instance.execute({"x": 3, "y": 4})
    assert result == "12"  # execute() returns str

def test_function_tool_optional_param_not_in_required():
    """Parameters with defaults are not in required list."""
    import inspect
    from src.tools.base import function_tool

    @function_tool(description="Echo with prefix")
    def echo(message: str, prefix: str = ">>") -> str:
        return f"{prefix} {message}"

    instance = echo()
    schema = instance.parameters
    assert "message" in schema["required"]
    assert "prefix" not in schema["required"]
```

**Run:** `pytest tests/test_tools.py::test_function_tool_decorator_creates_tool_class -v`
**Expected:** FAIL — `function_tool` not in `src/tools/base.py`

---

### Step 2: Implement function_tool decorator in src/tools/base.py

Add after the `RespondTool` class:

```python
def function_tool(description: str):
    """Decorator to create a Tool subclass from a plain function.

    Auto-generates JSON Schema from type annotations. Supports str, int,
    float, bool. Parameters without defaults are added to 'required'.

    Usage:
        @function_tool(description="Return the current time")
        def get_time(format: str = "%H:%M") -> str:
            from datetime import datetime
            return datetime.now().strftime(format)

        # Register an instance:
        registry.register(get_time())
    """
    import inspect
    from typing import get_type_hints

    _TYPE_MAP: dict[type, str] = {
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

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        func_name = func.__name__
        func_description = description

        class _FunctionTool(Tool):
            @property
            def name(self) -> str:
                return func_name

            @property
            def description(self) -> str:
                return func_description

            @property
            def parameters(self) -> dict[str, Any]:
                return schema

            def execute(self, args: dict[str, Any]) -> str:
                result = func(**args)
                return str(result)

        _FunctionTool.__name__ = func.__name__
        _FunctionTool.__qualname__ = func.__qualname__
        # Return the class so users can instantiate with _FunctionTool()
        return _FunctionTool

    return decorator
```

---

### Step 3: Run tests

**Run:** `pytest tests/test_tools.py::test_function_tool_decorator_creates_tool_class tests/test_tools.py::test_function_tool_generates_schema_from_type_hints tests/test_tools.py::test_function_tool_execute_calls_function tests/test_tools.py::test_function_tool_optional_param_not_in_required -v`
**Expected:** PASS

---

### Step 4: Run full suite

```bash
pytest tests/ -x -q --ignore=tests/test_gauntlet.py
```

**Expected:** All pass

---

### Step 5: Commit

```bash
git add src/tools/base.py tests/test_tools.py
git commit -m "feat: add @function_tool decorator for lightweight tool definitions

- Auto-generate JSON Schema from type annotations (str/int/float/bool)
- Parameters without defaults are added to required list
- execute() wraps function call and coerces result to str
- Returns Tool subclass; instantiate with decorator_result()"
```

---

## Task 5: Refactor main.py → src/cli/ Package

**Problem:** `src/main.py` at 955 lines handles CLI definitions, session orchestration, tool
registration, slash command routing, TTS callbacks, audio mode, and provider setup. `rise()`
alone is ~300 lines and is untestable in isolation.

**Target structure:**
```
src/cli/
├── __init__.py          # re-export app
├── app.py               # Typer app + non-rise commands (detect, init, config, ingest, graph, sessions, status, routing_stats)
├── session_manager.py   # rise() function + _make_confirm_callback + tool registration
└── slash_commands.py    # _handle_slash_command() + _plan_mode_state
```

`src/main.py` becomes a thin shim:
```python
from src.cli import app
```

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/app.py`
- Create: `src/cli/session_manager.py`
- Create: `src/cli/slash_commands.py`
- Modify: `src/main.py` — thin shim only
- Test: `tests/test_session.py` — verify imports still work
- Test: `tests/test_design_principles.py` — verify import paths still valid

---

### Step 1: Write smoke test for new module structure

In `tests/test_session.py` (or a new `tests/test_cli.py`), add:

```python
def test_cli_app_importable():
    """The cli package exports a valid Typer app."""
    from src.cli import app
    import typer
    assert isinstance(app, typer.Typer)

def test_slash_commands_importable():
    """slash_commands module exports _handle_slash_command."""
    from src.cli.slash_commands import handle_slash_command
    assert callable(handle_slash_command)

def test_session_manager_importable():
    """session_manager module exports rise-related helpers."""
    from src.cli.session_manager import build_tool_registry
    assert callable(build_tool_registry)
```

**Run:** `pytest tests/test_session.py::test_cli_app_importable -v`
**Expected:** FAIL — `src/cli` doesn't exist yet

---

### Step 2: Create src/cli/__init__.py

```python
"""Animus CLI package. Exports the Typer app."""
from src.cli.app import app

__all__ = ["app"]
```

---

### Step 3: Create src/cli/slash_commands.py

Move `_handle_slash_command()` and `_plan_mode_state` from `src/main.py` here. Rename function to `handle_slash_command` (drop the leading underscore — it's now public API of this module):

```python
"""Slash command handling for the Animus interactive session."""
from __future__ import annotations

from src.ui import console, success, warn

_plan_mode_state: dict[str, bool] = {"active": False}


def handle_slash_command(command: str, agent, session, cfg) -> bool:
    """Handle slash commands. Returns True if command was handled."""
    # ... (paste the body of _handle_slash_command from main.py here, unchanged)
    # Update internal references from _plan_mode_state to the module-level dict above
```

---

### Step 4: Create src/cli/session_manager.py

Extract `build_tool_registry()` helper (tool registration logic from rise()) and `_make_confirm_callback()`:

```python
"""Session orchestration helpers for the rise command."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from src.core.config import AnimusConfig
from src.tools.base import ToolRegistry
from src.ui import info


def make_confirm_callback(cfg: AnimusConfig) -> Callable[[str], bool]:
    """Create a Rich-based confirmation callback for dangerous tool operations."""
    # ... (paste _make_confirm_callback body from main.py, unchanged)


def build_tool_registry(cfg: AnimusConfig, session_cwd, confirm_cb) -> ToolRegistry:
    """Register all available tools based on config and available backends.

    Tries: filesystem, shell (with sandbox), git, graph, search, manifold.
    Optional backends (graph, vector store) are skipped if DBs don't exist.
    """
    from src.tools.filesystem import register_filesystem_tools
    from src.tools.shell import register_shell_tools
    from src.isolation.ornstein import create_sandbox

    registry = ToolRegistry()
    register_filesystem_tools(registry, session_cwd=session_cwd)

    shell_sandbox = create_sandbox(memory_mb=512, timeout_seconds=30)
    register_shell_tools(registry, confirm_callback=confirm_cb, session_cwd=session_cwd, sandbox=shell_sandbox)

    # Git tools
    try:
        from src.tools.git import register_git_tools
        register_git_tools(registry, confirm_callback=confirm_cb, session_cwd=session_cwd)
    except ImportError:
        pass

    # Graph tools
    graph_db_path = cfg.graph_dir / "code_graph.db"
    graph_db = None
    if graph_db_path.exists():
        try:
            from src.knowledge.graph_db import GraphDB
            from src.tools.graph import register_graph_tools
            graph_db = GraphDB(graph_db_path)
            register_graph_tools(registry, graph_db)
        except Exception:
            pass

    # Vector search tools
    vector_db_path = cfg.vector_dir / "vectors.db"
    vector_store = None
    search_embedder = None
    if vector_db_path.exists():
        try:
            from src.memory.embedder import NativeEmbedder
            from src.memory.vectorstore import SQLiteVectorStore
            from src.tools.search import register_search_tools
            vector_store = SQLiteVectorStore(vector_db_path)
            search_embedder = NativeEmbedder()
            register_search_tools(registry, vector_store, search_embedder)
        except Exception:
            pass

    # Manifold unified search
    if vector_store or graph_db:
        try:
            from src.retrieval.executor import RetrievalExecutor
            from src.tools.manifold_search import register_manifold_search
            executor = RetrievalExecutor(
                vector_store=vector_store,
                embedder=search_embedder,
                graph_db=graph_db,
                project_root=Path.cwd(),
            )
            register_manifold_search(registry, executor)
            info("[Manifold] Unified search tool registered")
        except Exception:
            pass

    return registry
```

---

### Step 5: Create src/cli/app.py

Move all non-rise commands from `src/main.py` into this file. Include the `rise()` command
by importing from session_manager. The rise() function body should be extracted too — move
the ~300 lines of the interaction loop and all its helpers here.

```python
"""Animus Typer CLI application and all command definitions."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from src.core.config import AnimusConfig
from src.core.detection import detect_system
from src.ui import console, error, info, print_logo, success, warn
from src import audio
from src.cli.slash_commands import handle_slash_command, _plan_mode_state
from src.cli.session_manager import make_confirm_callback, build_tool_registry


def _startup_callback(ctx: typer.Context) -> None:
    print_logo()


app = typer.Typer(
    name="animus",
    help="Local-first AI agent with RAG and tool use.",
    no_args_is_help=True,
    callback=_startup_callback,
)


# --- Paste all non-rise @app.command() functions from main.py here ---
# detect(), init(), config(), ingest(), graph(), sessions(), status(), routing_stats()


@app.command()
def rise(
    resume: bool = typer.Option(False, "--resume"),
    session_id: Optional[str] = typer.Option(None, "--session"),
    cautious: bool = typer.Option(False, "--cautious"),
    paranoid: bool = typer.Option(False, "--paranoid"),
    transcript: Optional[str] = typer.Option(None, "--transcript"),
) -> None:
    """Awaken Animus. Start an interactive agent session."""
    from src.core.agent import Agent
    from src.core.workspace import Workspace
    from src.core.session import Session
    from src.llm.factory import ProviderFactory

    cfg = AnimusConfig.load()

    if paranoid:
        error("Smough layer (--paranoid) not yet implemented.")
        info("Use --cautious for Ornstein lightweight sandbox.")
        raise typer.Exit(1)

    # ... rest of rise() body, using build_tool_registry() and handle_slash_command()
```

---

### Step 6: Update src/main.py to be a thin shim

Replace the entire content of `src/main.py` with:

```python
"""Animus CLI entry point. Logic lives in src/cli/."""
from src.cli import app

if __name__ == "__main__":
    app()
```

This preserves the existing `python -m src.main` and `animus` entry points.

---

### Step 7: Run tests

```bash
pytest tests/ -x -q --ignore=tests/test_gauntlet.py
```

Fix any import errors (tests that imported from `src.main` directly need updating).

---

### Step 8: Commit

```bash
git add src/cli/ src/main.py tests/
git commit -m "refactor: extract main.py into src/cli/ package

- cli/app.py: Typer app + all command definitions
- cli/session_manager.py: build_tool_registry() + make_confirm_callback()
- cli/slash_commands.py: handle_slash_command() + _plan_mode_state
- main.py: thin shim (3 lines) that imports from cli/
- rise() is now testable in isolation via build_tool_registry()"
```

---

## Execution Order

```
Task 1 → Task 2 → run full suite → Task 3 → Task 4 → Task 5
```

Tasks 1 and 2 are both "this week" — do them first since they fix correctness gaps.
Tasks 3, 4, 5 can be done in any order within "this month".

After all tasks, run:
```bash
pytest tests/ -q --ignore=tests/test_gauntlet.py
```
and verify the total test count is >= 605 (we're adding tests, not removing them).
