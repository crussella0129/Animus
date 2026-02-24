# Phase 3: Security Hardening, Scope Enforcement & Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all documented security gaps, reduce scope bleed with hard-block enforcement, refactor planner.py into 3 files, and add project-scoped Workspace boundary enforcement.

**Architecture:** Security-first ordering. Each task is independently testable and committable. TDD throughout — write failing tests first, then implement. Workspace replaces SessionCwd as the cross-tool state container.

**Tech Stack:** Python 3.11+, pytest, shlex, threading, pathlib, pydantic

---

## Task 1: Shell Metacharacter Rejection

**Files:**
- Modify: `src/tools/shell.py:1-38` (add new validation function)
- Test: `tests/test_tools.py` (add TestShellMetacharRejection class)

**Step 1: Write the failing tests**

Add to `tests/test_tools.py`:

```python
class TestShellMetacharRejection:
    """Shell metacharacters must be rejected to prevent injection."""

    def test_pipe_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo hello | grep hello"})
        assert "Error" in result
        assert "not supported" in result.lower()

    def test_redirect_out_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo hello > /tmp/out.txt"})
        assert "Error" in result

    def test_redirect_in_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "cat < /etc/passwd"})
        assert "Error" in result

    def test_semicolon_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo safe; rm -rf /"})
        assert "Error" in result

    def test_and_chain_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "mkdir foo && cd foo"})
        assert "Error" in result

    def test_or_chain_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "false || echo fallback"})
        assert "Error" in result

    def test_command_substitution_dollar_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo $(whoami)"})
        assert "Error" in result

    def test_command_substitution_backtick_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo `whoami`"})
        assert "Error" in result

    def test_background_ampersand_rejected(self):
        tool = RunShellTool()
        result = tool.execute({"command": "sleep 100 &"})
        assert "Error" in result

    def test_simple_command_allowed(self):
        tool = RunShellTool()
        result = tool.execute({"command": "echo hello"})
        assert "hello" in result
        assert "not supported" not in result.lower()

    def test_quoted_ampersand_in_echo_rejected(self):
        """Even quoted metacharacters are rejected for safety."""
        tool = RunShellTool()
        result = tool.execute({"command": 'echo "a && b"'})
        assert "Error" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py::TestShellMetacharRejection -v`
Expected: Most tests FAIL (pipe, redirect, etc. currently execute via shell=True)

**Step 3: Implement metacharacter rejection in shell.py**

Add to `src/tools/shell.py` after the existing regex constants (around line 37):

```python
# Shell metacharacters that indicate injection risk.
# We reject these rather than trying to handle them safely.
_SHELL_METACHAR_RE = re.compile(
    r'\|'       # pipe or logical or
    r'|&&'      # logical and
    r'|;'       # command separator
    r'|>'       # redirect out
    r'|<'       # redirect in
    r'|`'       # backtick substitution
    r'|\$\('    # dollar-paren substitution
    r'|&(?!&)'  # background (& but not &&)
)


def _reject_shell_features(command: str) -> str | None:
    """Check command for shell metacharacters. Returns error message if found."""
    if _SHELL_METACHAR_RE.search(command):
        return (
            "Shell features (pipes, redirects, command chaining, substitution) "
            "are not supported. Use separate tool calls instead."
        )
    return None
```

Then in `RunShellTool.execute()`, add the check right after the budget check (after line 187):

```python
        # Reject shell metacharacters (prevents injection)
        shell_err = _reject_shell_features(command)
        if shell_err:
            return f"Error: {shell_err}"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py::TestShellMetacharRejection -v`
Expected: ALL PASS

**Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: Some existing shell tests may fail if they use pipes/chains — fix those in Task 2.

**Step 6: Commit**

```bash
git add src/tools/shell.py tests/test_tools.py
git commit -m "feat: reject shell metacharacters to prevent command injection"
```

---

## Task 2: Replace shell=True with List-Based Subprocess

**Files:**
- Modify: `src/tools/shell.py:134-283` (RunShellTool.execute + _extract_and_update_cwd)
- Test: `tests/test_tools.py` (update TestRunShellTool)
- Test: `tests/test_cwd.py` (update shell CWD tests)

**Step 1: Write failing tests for new cd handling**

Add to `tests/test_tools.py`:

```python
class TestShellCdHandling:
    """cd commands are handled directly via SessionCwd, not subprocess."""

    def test_cd_updates_session_cwd(self, tmp_path: Path):
        from src.core.cwd import SessionCwd
        cwd = SessionCwd(initial=tmp_path)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        tool = RunShellTool(session_cwd=cwd)
        result = tool.execute({"command": f"cd {subdir}"})
        assert "Changed directory" in result
        assert cwd.path == subdir

    def test_cd_with_quotes(self, tmp_path: Path):
        from src.core.cwd import SessionCwd
        cwd = SessionCwd(initial=tmp_path)
        subdir = tmp_path / "sub dir"
        subdir.mkdir()
        tool = RunShellTool(session_cwd=cwd)
        result = tool.execute({"command": f'cd "{subdir}"'})
        assert "Changed directory" in result
        assert cwd.path == subdir

    def test_cd_nonexistent_dir(self, tmp_path: Path):
        from src.core.cwd import SessionCwd
        cwd = SessionCwd(initial=tmp_path)
        tool = RunShellTool(session_cwd=cwd)
        result = tool.execute({"command": "cd /nonexistent/path"})
        assert cwd.path == tmp_path  # unchanged

    def test_cd_no_session_cwd(self):
        tool = RunShellTool()
        result = tool.execute({"command": "cd /tmp"})
        # Should handle gracefully without session_cwd
        assert "Error" not in result or "Changed" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py::TestShellCdHandling -v`
Expected: FAIL — cd is currently handled via marker injection, not direct SessionCwd

**Step 3: Rewrite RunShellTool.execute() to remove shell=True**

Replace the execute method in `src/tools/shell.py` `RunShellTool` class (lines 170-256):

```python
    def execute(self, args: dict[str, Any]) -> str:
        command = args["command"]
        timeout = args.get("timeout", 30)

        # Check execution budget before running
        budget_ok, budget_msg = self._budget.check_available(timeout)
        if not budget_ok:
            stats = self._budget.stats
            return (
                f"Error: {budget_msg}\n"
                f"Budget stats: {stats['used']}s used in {stats['call_count']} commands, "
                f"{stats['remaining']}s remaining"
            )

        # Reject shell metacharacters (prevents injection)
        shell_err = _reject_shell_features(command)
        if shell_err:
            return f"Error: {shell_err}"

        checker = PermissionChecker()
        blocked = checker.is_command_blocked(command)
        if blocked:
            return f"Error: Command blocked for safety: {blocked}"

        if not self._allow_network:
            net_match = checker.is_command_network(command)
            if net_match:
                return (
                    f"Error: Network command blocked: '{net_match}'. "
                    "Outbound network access is disabled by default to prevent "
                    "data exfiltration. Use allow_network=True to enable."
                )

        if checker.is_command_dangerous(command) and self._confirm:
            if not self._confirm(f"Allow dangerous command: {command}?"):
                return "Command cancelled by user."

        # Resolve CWD for the subprocess
        cwd = str(self._session_cwd.path) if self._session_cwd else None

        # Handle cd commands directly via SessionCwd (shell builtin, no subprocess)
        cd_match = re.match(r'^\s*cd\s+(.*)', command)
        if cd_match:
            return self._handle_cd(cd_match.group(1).strip())

        # Tokenize command into list for safe subprocess execution
        try:
            import shlex
            if os.name == "nt":
                cmd_list = shlex.split(command, posix=False)
            else:
                cmd_list = shlex.split(command)
        except ValueError as e:
            return f"Error: Could not parse command: {e}"

        if not cmd_list:
            return "Error: Empty command"

        # On Windows, shell builtins need cmd /c prefix
        if os.name == "nt":
            _WIN_BUILTINS = {
                "dir", "type", "mkdir", "rmdir", "del", "copy", "move",
                "ren", "rd", "md", "echo", "set", "cls", "ver", "where",
            }
            if cmd_list[0].lower() in _WIN_BUILTINS:
                cmd_list = ["cmd", "/c"] + cmd_list

        # Track actual execution time
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
            )
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error: Command timed out after {timeout}s"
        except FileNotFoundError:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error: Command not found: {cmd_list[0]}"
        except Exception as e:
            elapsed = time.time() - start_time
            self._budget.consume(elapsed)
            return f"Error executing command: {e}"

    def _handle_cd(self, target: str) -> str:
        """Handle cd command directly via SessionCwd."""
        # Strip quotes from target path
        target = target.strip("\"'")
        if not target:
            target = os.path.expanduser("~")

        if self._session_cwd is None:
            return f"Changed directory to {target} (no session tracking)"

        self._session_cwd.set(target)
        return f"Changed directory to {self._session_cwd.path}"
```

Also **remove** the `_extract_and_update_cwd` method (lines 258-282) — it's no longer needed.

Remove the `_CD_RE` constant at line 16 (no longer used).

Remove/simplify the `_normalize_quotes_for_windows` function and its supporting regexes (lines 20-68) — with list-based subprocess, Windows quote normalization is no longer needed. Keep the function but have it only strip wrapping quotes that would confuse shlex.

**Step 4: Run tests to verify cd handling passes**

Run: `pytest tests/test_tools.py::TestShellCdHandling -v`
Expected: PASS

**Step 5: Fix any broken existing tests**

Run: `pytest tests/test_tools.py -v --tb=short`
Expected: Some tests like `test_simple_command` should still pass. Fix any that relied on shell=True behavior (e.g., pipe tests that now correctly get rejected).

Also run: `pytest tests/test_cwd.py -v --tb=short`
Update any tests in test_cwd.py that test CWD marker extraction — these tests should now test the new cd handling instead.

**Step 6: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS (556+ tests)

**Step 7: Commit**

```bash
git add src/tools/shell.py tests/test_tools.py tests/test_cwd.py
git commit -m "fix: replace shell=True with list-based subprocess (P0 security)"
```

---

## Task 3: Fix Command Parsing in PermissionChecker

**Files:**
- Modify: `src/core/permission.py:88-157`
- Test: `tests/test_permission.py`

**Step 1: Write failing tests**

Add to `tests/test_permission.py`:

```python
class TestFullCommandParsing:
    """is_command_dangerous should check ALL tokens, not just the first word."""

    def test_dangerous_in_second_position(self):
        checker = PermissionChecker()
        # "echo" is safe but the full command contains "rm"
        # With shell=True removed, this is defense-in-depth
        assert checker.is_command_dangerous("echo foo rm bar") is False  # rm is an arg, not a command

    def test_command_substitution_dollar_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo $(rm -rf /)")

    def test_command_substitution_backtick_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo `whoami`")

    def test_semicolon_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo safe; rm -rf /")

    def test_and_chain_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("echo safe && rm -rf /")

    def test_or_chain_detected(self):
        checker = PermissionChecker()
        assert checker.has_injection_pattern("false || rm -rf /")

    def test_clean_command_no_injection(self):
        checker = PermissionChecker()
        assert not checker.has_injection_pattern("echo hello world")

    def test_clean_path_no_injection(self):
        checker = PermissionChecker()
        assert not checker.has_injection_pattern("mkdir my_project")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_permission.py::TestFullCommandParsing -v`
Expected: FAIL — `has_injection_pattern` doesn't exist yet

**Step 3: Add has_injection_pattern to PermissionChecker**

Add to `src/core/permission.py` in the `PermissionChecker` class:

```python
    _INJECTION_RE = re.compile(
        r'\$\('       # $(command)
        r'|`'         # `command`
        r'|;'         # command separator
        r'|&&'        # logical and
        r'|\|\|'      # logical or
    )

    def has_injection_pattern(self, command: str) -> bool:
        """Check if command contains shell injection patterns.

        Defense-in-depth: with shell=True removed, these patterns can't
        execute, but detecting them flags suspicious LLM output.
        """
        return bool(self._INJECTION_RE.search(command))
```

**Step 4: Also improve is_path_safe with symlink resolution (design 1C)**

Update `is_path_safe` in `src/core/permission.py`:

```python
    def is_path_safe(self, path: Path) -> bool:
        """Check if a path is safe to access. Follows symlinks."""
        try:
            resolved = str(path.resolve(strict=False))
        except (OSError, ValueError):
            return False
        for dangerous in DANGEROUS_DIRECTORIES:
            norm_dangerous = dangerous.replace("\\", "/")
            norm_resolved = resolved.replace("\\", "/")
            if norm_resolved.startswith(norm_dangerous):
                return False
        resolved_fwd = resolved.replace("\\", "/")
        for dangerous in DANGEROUS_FILES:
            if resolved_fwd.endswith(dangerous) or dangerous in resolved_fwd:
                return False
        return True
```

**Step 5: Run tests**

Run: `pytest tests/test_permission.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/core/permission.py tests/test_permission.py
git commit -m "fix: add injection pattern detection and symlink-safe path checks"
```

---

## Task 4: Thread-Safe Write Audit Log

**Files:**
- Modify: `src/tools/filesystem.py:67-141`
- Test: `tests/test_tools.py`

**Step 1: Write failing test**

Add to `tests/test_tools.py`:

```python
import threading

class TestWriteLogThreadSafety:
    """WriteFileTool._write_log must be thread-safe."""

    def test_concurrent_writes_no_data_loss(self, tmp_path: Path):
        WriteFileTool.clear_write_log()
        errors = []

        def write_file(i: int):
            try:
                tool = WriteFileTool()
                target = tmp_path / f"file_{i}.txt"
                tool.execute({"path": str(target), "content": f"content_{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_file, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        log = WriteFileTool.get_write_log()
        assert len(log) == 20
```

**Step 2: Run test (may pass by luck, but race condition exists)**

Run: `pytest tests/test_tools.py::TestWriteLogThreadSafety -v`

**Step 3: Add threading lock to WriteFileTool**

In `src/tools/filesystem.py`, modify `WriteFileTool`:

```python
import threading

class WriteFileTool(Tool):
    _write_log: list[dict[str, Any]] = []
    _write_log_lock = threading.Lock()

    # In execute(), change the append:
    def execute(self, args: dict[str, Any]) -> str:
        # ... existing path resolution and permission check ...
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args["content"]
            path.write_text(content, encoding="utf-8")

            content_hash = hashlib.md5(content.encode()).hexdigest()
            with self._write_log_lock:
                self._write_log.append({
                    "path": str(path),
                    "size": len(content),
                    "timestamp": time.time(),
                    "hash": content_hash,
                    "lines": content.count('\n') + 1,
                })

            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    @classmethod
    def clear_write_log(cls) -> None:
        with cls._write_log_lock:
            cls._write_log.clear()
```

**Step 4: Run tests**

Run: `pytest tests/test_tools.py::TestWriteLogThreadSafety -v`
Expected: PASS

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/tools/filesystem.py tests/test_tools.py
git commit -m "fix: add threading lock to WriteFileTool audit log"
```

---

## Task 5: Hard Block Scope Enforcement

**Files:**
- Modify: `src/core/planner.py:813-832` (scope enforcement section in _execute_step)
- Test: `tests/test_planner.py`

**Step 1: Write failing test**

Add to `tests/test_planner.py`:

```python
class TestHardBlockScope:
    """Out-of-scope tool calls should be blocked on first violation."""

    def test_out_of_scope_tool_blocked_immediately(self):
        """First out-of-scope tool call should be blocked (not just warned)."""
        from unittest.mock import MagicMock
        from src.core.planner import ChunkedExecutor, Step, StepType, StepStatus

        mock_provider = MagicMock()
        mock_provider.capabilities.return_value = MagicMock(
            context_length=4096, size_tier="small"
        )

        # Provider returns a tool call for run_shell (out of scope for READ step)
        mock_provider.generate.side_effect = [
            '{"name": "run_shell", "arguments": {"command": "echo hack"}}',
            '{"name": "read_file", "arguments": {"path": "test.txt"}}',
        ]

        registry = ToolRegistry()
        # Only register read tools
        from src.tools.filesystem import ReadFileTool, ListDirTool
        registry.register(ReadFileTool())
        registry.register(ListDirTool())

        executor = ChunkedExecutor(provider=mock_provider, tool_registry=registry)

        step = Step(number=1, description="Read the config file", step_type=StepType.READ)
        result = executor._execute_step(step, 1, "Read config")

        # The run_shell call should have been blocked, not executed
        # Verify by checking that no shell command was actually run
        assert result.status == StepStatus.COMPLETED
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_planner.py::TestHardBlockScope -v`
Expected: FAIL — currently scope enforcement only warns on first violation

**Step 3: Modify scope enforcement in _execute_step**

In `src/core/planner.py`, modify the scope enforcement block inside `_execute_step` (around lines 814-832):

Replace the current scope enforcement logic:

```python
                # Scope enforcement: hard block out-of-scope tool calls
                if expected_tools and call["name"] not in expected_tools:
                    out_of_scope_count += 1
                    # Block the tool call — do NOT execute it
                    blocked_msg = (
                        f"[System]: Tool '{call['name']}' is not available for this step "
                        f"(\"{step.description}\"). Available tools: {', '.join(expected_tools)}. "
                        f"Use only the listed tools."
                    )
                    messages.append({"role": "user", "content": blocked_msg})

                    if out_of_scope_count >= 2:
                        # Two blocked attempts — terminate step
                        return StepResult(
                            step=step,
                            status=StepStatus.COMPLETED,
                            output=last_tool_result or "Step terminated: repeated out-of-scope tool calls.",
                        )
                    continue  # Skip execution, give model another chance
```

Note: This replaces the existing block at lines 814-832. The key change is:
1. Block on **first** violation (not warn)
2. `continue` skips tool execution entirely
3. After 2 blocked attempts, terminate the step

Also remove the `has_successful_call` tracking since we now block before execution regardless.

**Step 4: Run tests**

Run: `pytest tests/test_planner.py::TestHardBlockScope -v`
Expected: PASS

Run: `pytest tests/test_planner.py -v --tb=short`
Expected: ALL PASS (check existing scope tests still work)

**Step 5: Commit**

```bash
git add src/core/planner.py tests/test_planner.py
git commit -m "feat: hard block out-of-scope tool calls on first violation"
```

---

## Task 6: Explicit Tool Messaging in Step Prompts

**Files:**
- Modify: `src/core/planner.py:422-445` (_EXECUTION_PROMPT and prompt building)
- Test: `tests/test_planner.py`

**Step 1: Write failing test**

Add to `tests/test_planner.py`:

```python
class TestExplicitToolMessaging:
    """Step prompts should explicitly state which tools are allowed and forbidden."""

    def test_step_prompt_includes_only_instruction(self):
        """System prompt should contain 'You may ONLY use' instruction."""
        from unittest.mock import MagicMock
        from src.core.planner import ChunkedExecutor, Step, StepType

        mock_provider = MagicMock()
        mock_provider.capabilities.return_value = MagicMock(
            context_length=4096, size_tier="small"
        )
        mock_provider.generate.return_value = "Done reading the file."

        registry = ToolRegistry()
        from src.tools.filesystem import ReadFileTool
        registry.register(ReadFileTool())

        executor = ChunkedExecutor(provider=mock_provider, tool_registry=registry)
        step = Step(number=1, description="Read config", step_type=StepType.READ)
        executor._execute_step(step, 1, "Read config")

        # Check that the system prompt passed to generate contains the restriction
        call_args = mock_provider.generate.call_args
        messages = call_args[0][0]
        system_msg = messages[0]["content"]
        assert "ONLY" in system_msg
        assert "Do not attempt" in system_msg or "do not attempt" in system_msg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_planner.py::TestExplicitToolMessaging -v`
Expected: FAIL — current prompt doesn't include "ONLY" restriction

**Step 3: Update _EXECUTION_PROMPT to include tool restriction**

In `src/core/planner.py`, update the system prompt building in `_execute_step` (after building `tool_schemas_str` around line 622):

Add after the tool_schemas_str construction:

```python
        # Add explicit tool restriction to prevent scope bleed
        tool_restriction = (
            f"\n\nIMPORTANT: You may ONLY use the tools listed above. "
            f"Do not attempt to use any other tools. If the step cannot be completed "
            f"with the available tools, return what you have."
        )
        tool_schemas_str += tool_restriction
```

**Step 4: Run tests**

Run: `pytest tests/test_planner.py::TestExplicitToolMessaging -v`
Expected: PASS

Run: `pytest tests/test_planner.py -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/planner.py tests/test_planner.py
git commit -m "feat: add explicit tool restriction messaging in step prompts"
```

---

## Task 7: Tighter Step Type Inference

**Files:**
- Modify: `src/core/planner.py:280-307` (_infer_step_type)
- Test: `tests/test_planner.py`

**Step 1: Write failing tests**

Add to `tests/test_planner.py`:

```python
class TestTighterStepTypeInference:
    """File extensions should take priority over keyword matching."""

    def test_test_py_file_is_write_not_shell(self):
        from src.core.planner import _infer_step_type, StepType
        # "test" keyword should NOT trigger SHELL when a .py file is mentioned
        assert _infer_step_type("edit test_auth.py") == StepType.WRITE

    def test_test_py_file_read(self):
        from src.core.planner import _infer_step_type, StepType
        assert _infer_step_type("read test_auth.py") == StepType.READ

    def test_run_pytest_is_shell(self):
        from src.core.planner import _infer_step_type, StepType
        # "run pytest" with no file pattern should still be SHELL
        assert _infer_step_type("run pytest") == StepType.SHELL

    def test_create_config_json_is_write(self):
        from src.core.planner import _infer_step_type, StepType
        assert _infer_step_type("create config.json") == StepType.WRITE

    def test_check_readme_md_is_read(self):
        from src.core.planner import _infer_step_type, StepType
        assert _infer_step_type("check readme.md contents") == StepType.READ
```

**Step 2: Run tests to verify some fail**

Run: `pytest tests/test_planner.py::TestTighterStepTypeInference -v`
Expected: `test_test_py_file_is_write_not_shell` likely FAILS (currently classified as SHELL due to "test" keyword)

**Step 3: Improve _infer_step_type**

The current implementation already has file pattern priority (lines 288-300). Check if the issue is that "edit" is not in write_verbs or the pattern match is failing. If `test_test_py_file_is_write_not_shell` fails, it means "edit" isn't triggering before "test" keyword. The fix is to ensure the file pattern check properly captures "edit test_auth.py":

In `src/core/planner.py`, update `_infer_step_type` to add more write verbs and make the check more robust:

```python
def _infer_step_type(description: str) -> StepType:
    lower = description.lower()

    # Priority 1: File extension patterns — most reliable signal
    has_file_pattern = re.search(r'\b[\w\-]+\.\w{1,5}\b', description)

    if has_file_pattern:
        write_verbs = [
            "edit", "modify", "update", "change", "fix", "add to", "append",
            "write", "create", "save", "implement", "build", "make",
        ]
        read_verbs = [
            "read", "view", "check", "look at", "examine", "inspect",
            "show", "cat", "display", "open", "review",
        ]

        if any(verb in lower for verb in write_verbs):
            return StepType.WRITE
        if any(verb in lower for verb in read_verbs):
            return StepType.READ

    # Priority 2: Keyword matching
    for stype, keywords in _TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return stype

    return StepType.ANALYZE
```

**Step 4: Run tests**

Run: `pytest tests/test_planner.py::TestTighterStepTypeInference -v`
Expected: ALL PASS

Run: `pytest tests/test_planner.py -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/planner.py tests/test_planner.py
git commit -m "fix: prioritize file extensions over keywords in step type inference"
```

---

## Task 8: Create Workspace Class (Replaces SessionCwd)

**Files:**
- Create: `src/core/workspace.py`
- Test: `tests/test_workspace.py`
- Modify: `src/core/errors.py` (add WorkspaceBoundaryError)

**Step 1: Write failing tests**

Create `tests/test_workspace.py`:

```python
"""Tests for Workspace boundary enforcement."""

from pathlib import Path

import pytest

from src.core.workspace import Workspace, WorkspaceBoundaryError


class TestWorkspaceInit:
    def test_root_is_resolved(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.root == tmp_path.resolve()

    def test_cwd_starts_at_root(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.cwd == ws.root

    def test_path_property_for_backward_compat(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.path == ws.cwd  # SessionCwd compatibility


class TestWorkspaceResolve:
    def test_resolve_relative_path(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        resolved = ws.resolve("subdir/file.txt")
        assert resolved == tmp_path / "subdir" / "file.txt"

    def test_resolve_absolute_within_root(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        target = tmp_path / "inside.txt"
        resolved = ws.resolve(str(target))
        assert resolved == target

    def test_resolve_outside_root_raises(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        with pytest.raises(WorkspaceBoundaryError):
            ws.resolve("/etc/passwd")

    def test_resolve_parent_traversal_raises(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        with pytest.raises(WorkspaceBoundaryError):
            ws.resolve("../../etc/passwd")

    def test_resolve_after_cwd_change(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set_cwd(subdir)
        resolved = ws.resolve("file.txt")
        assert resolved == subdir / "file.txt"


class TestWorkspaceSetCwd:
    def test_set_cwd_within_root(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set_cwd(subdir)
        assert ws.cwd == subdir

    def test_set_cwd_outside_root_raises(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        with pytest.raises(WorkspaceBoundaryError):
            ws.set_cwd(Path("/tmp"))

    def test_set_cwd_relative(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set_cwd(Path("sub"))
        assert ws.cwd == subdir

    def test_set_nonexistent_ignored(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        ws.set("/nonexistent/deep/path")
        assert ws.cwd == tmp_path  # unchanged


class TestWorkspaceSet:
    """Test the .set() method for backward compatibility with SessionCwd."""

    def test_set_valid_dir(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        ws = Workspace(root=tmp_path)
        ws.set(str(subdir))
        assert ws.cwd == subdir

    def test_set_outside_root_ignored(self, tmp_path: Path):
        """set() silently ignores out-of-root paths (SessionCwd compat)."""
        ws = Workspace(root=tmp_path)
        ws.set("/etc")
        assert ws.cwd == tmp_path  # unchanged
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workspace.py -v`
Expected: FAIL — module doesn't exist yet

**Step 3: Create src/core/workspace.py**

```python
"""Project-scoped workspace with boundary enforcement.

Replaces SessionCwd with an immutable root boundary and mutable CWD.
All file operations must resolve within the workspace root.
"""

from __future__ import annotations

import os
from pathlib import Path


class WorkspaceBoundaryError(Exception):
    """Raised when an operation attempts to escape the workspace root."""


class Workspace:
    """Project-scoped working directory with boundary enforcement.

    Enforces that all path resolutions stay within the workspace root.
    Drop-in replacement for SessionCwd with added safety.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is not None:
            self._root = Path(root).resolve()
        else:
            self._root = Path(os.getcwd()).resolve()
        self._cwd = self._root

    @property
    def root(self) -> Path:
        """Immutable project boundary."""
        return self._root

    @property
    def cwd(self) -> Path:
        """Current working directory within workspace."""
        return self._cwd

    @property
    def path(self) -> Path:
        """Alias for cwd — backward compatibility with SessionCwd."""
        return self._cwd

    def set_cwd(self, new_dir: Path | str) -> None:
        """Update CWD. Must stay within root. Raises on boundary violation."""
        candidate = Path(new_dir)
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        candidate = candidate.resolve()

        if not candidate.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                f"Cannot change directory to {new_dir}: "
                f"outside workspace root {self._root}"
            )
        if candidate.is_dir():
            self._cwd = candidate

    def set(self, new_dir: Path | str) -> None:
        """Update CWD (SessionCwd-compatible). Silently ignores violations."""
        try:
            self.set_cwd(new_dir)
        except WorkspaceBoundaryError:
            pass  # Silent ignore for backward compat
        except (OSError, ValueError):
            pass

    def resolve(self, path: Path | str) -> Path:
        """Resolve path relative to CWD, enforce workspace boundary."""
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._cwd / p).resolve()

        if not resolved.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                f"Path '{path}' resolves to {resolved}: "
                f"outside workspace root {self._root}"
            )
        return resolved
```

**Step 4: Run tests**

Run: `pytest tests/test_workspace.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/workspace.py tests/test_workspace.py
git commit -m "feat: add Workspace class with project-scoped boundary enforcement"
```

---

## Task 9: Integrate Workspace Into All Tools

**Files:**
- Modify: `src/tools/filesystem.py` (replace SessionCwd with Workspace)
- Modify: `src/tools/shell.py` (replace SessionCwd with Workspace)
- Modify: `src/tools/git.py` (replace SessionCwd with Workspace)
- Modify: `src/core/agent.py` (replace SessionCwd with Workspace)
- Modify: `src/core/planner.py` (replace SessionCwd with Workspace)
- Test: existing tests should continue to pass (Workspace is API-compatible)

**Step 1: Update imports across all files**

This is a mechanical replacement. In each file, change:
```python
from src.core.cwd import SessionCwd
```
to:
```python
from src.core.workspace import Workspace
```

And rename type annotations from `SessionCwd` to `Workspace`. The `Workspace` class has `.path`, `.set()`, and `.resolve()` methods that match `SessionCwd`'s API, so tool code doesn't change.

**Files to update:**
- `src/tools/shell.py`: `SessionCwd` → `Workspace` in imports, __init__ param, type hints
- `src/tools/filesystem.py`: Same
- `src/tools/git.py`: Same
- `src/core/agent.py`: Same
- `src/core/planner.py`: Same
- `src/tools/graph.py`: Check if it uses SessionCwd
- `src/tools/manifold_search.py`: Check if it uses SessionCwd

Also update registration functions:
- `register_shell_tools(session_cwd=...)` → keep param name for now, just change type
- `register_filesystem_tools(session_cwd=...)` → same
- `register_git_tools(session_cwd=...)` → same

**Step 2: Update filesystem tools to use workspace boundary checking**

In `src/tools/filesystem.py`, the `ReadFileTool`, `WriteFileTool`, and `ListDirTool` already use `session_cwd.resolve()`. Since `Workspace.resolve()` raises `WorkspaceBoundaryError` for out-of-root paths, add a try/except:

```python
    def execute(self, args: dict[str, Any]) -> str:
        try:
            if self._session_cwd is not None:
                path = self._session_cwd.resolve(args["path"])
            else:
                path = Path(args["path"]).resolve()
        except WorkspaceBoundaryError as e:
            return f"Error: {e}"
        # ... rest of existing logic
```

Apply the same pattern to `WriteFileTool.execute()` and `ListDirTool.execute()`.

**Step 3: Update shell tool cd handling for workspace boundary**

In `src/tools/shell.py`, the `_handle_cd` method uses `self._session_cwd.set()` which already silently ignores out-of-root paths. No change needed for backward compat, but update the message:

```python
    def _handle_cd(self, target: str) -> str:
        target = target.strip("\"'")
        if not target:
            target = os.path.expanduser("~")
        if self._session_cwd is None:
            return f"Changed directory to {target} (no session tracking)"
        old_cwd = self._session_cwd.path
        self._session_cwd.set(target)
        if self._session_cwd.path == old_cwd and target != str(old_cwd):
            return f"Error: Cannot change directory to {target} (outside workspace or does not exist)"
        return f"Changed directory to {self._session_cwd.path}"
```

**Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS — Workspace is API-compatible with SessionCwd

**Step 5: Commit**

```bash
git add src/tools/shell.py src/tools/filesystem.py src/tools/git.py src/core/agent.py src/core/planner.py src/core/workspace.py
git commit -m "refactor: replace SessionCwd with Workspace across all tools"
```

---

## Task 10: Add workspace_root to Config

**Files:**
- Modify: `src/core/config.py:41-47` (AgentConfig)
- Test: `tests/test_config.py`

**Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
class TestWorkspaceConfig:
    def test_workspace_root_default_none(self):
        config = AnimusConfig()
        assert config.agent.workspace_root is None

    def test_workspace_root_from_yaml(self, tmp_config_dir: Path):
        config_file = tmp_config_dir / "config.yaml"
        config_file.write_text("agent:\n  workspace_root: /home/user/project\n")
        config = AnimusConfig.load(config_dir=tmp_config_dir)
        assert config.agent.workspace_root == "/home/user/project"

    def test_workspace_root_roundtrip(self, tmp_config_dir: Path):
        config = AnimusConfig(config_dir=tmp_config_dir)
        config.agent.workspace_root = "/home/user/project"
        config.save()
        loaded = AnimusConfig.load(config_dir=tmp_config_dir)
        assert loaded.agent.workspace_root == "/home/user/project"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestWorkspaceConfig -v`
Expected: FAIL — workspace_root doesn't exist on AgentConfig

**Step 3: Add workspace_root to AgentConfig**

In `src/core/config.py`, update `AgentConfig`:

```python
class AgentConfig(BaseModel):
    model_config = {"extra": "ignore"}

    max_turns: int = 20
    system_prompt: str = "You are Animus, a helpful local AI assistant with tool use capabilities."
    confirm_dangerous: bool = True
    workspace_root: Optional[str] = None  # Project root directory (default: CWD at launch)
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/config.py tests/test_config.py
git commit -m "feat: add workspace_root to AgentConfig"
```

---

## Task 11: Split planner.py into Three Files

**Files:**
- Create: `src/core/planner/__init__.py`
- Create: `src/core/planner/decomposer.py`
- Create: `src/core/planner/parser.py`
- Create: `src/core/planner/executor.py`
- Delete: `src/core/planner.py`
- Modify: `tests/test_planner.py` (verify imports still work)
- Modify: `tests/test_grammar.py` (update imports if needed)

**Step 1: Create the planner package directory**

```bash
mkdir -p src/core/planner
```

**Step 2: Create parser.py with data structures and constants**

Create `src/core/planner/parser.py` containing:
- `StepType` enum
- `Step` dataclass
- `StepStatus` enum
- `StepResult` dataclass
- `PlanResult` dataclass
- `_STEP_TYPE_TOOLS` dict
- `_PLANNING_TIER_FACTORS` dict
- `_compute_planning_profile()` function
- `_PLANNING_PROFILES` dict (deprecated compat)
- `_get_planning_profile()` function
- `_TYPE_KEYWORDS` list
- `_infer_step_type()` function
- `_STEP_PATTERN` regex
- `_MAX_PLAN_STEPS` constant
- `PlanParser` class

These are lines 33-416 of the current planner.py.

**Step 3: Create decomposer.py**

Create `src/core/planner/decomposer.py` containing:
- `_PLANNING_PROMPT` template
- `TaskDecomposer` class
- `_is_simple_task()` function
- `_heuristic_decompose()` function

These are lines 310-325, 327-366, 965-1053 of the current planner.py.

Import from parser.py: `Step, StepType, _STEP_TYPE_TOOLS, _infer_step_type, _MAX_PLAN_STEPS`

**Step 4: Create executor.py**

Create `src/core/planner/executor.py` containing:
- `_EXECUTION_PROMPT` template
- `_infer_expected_tools()` function
- `_filter_tools()` function
- `ChunkedExecutor` class

These are lines 198-264, 418-837 of the current planner.py.

Import from parser.py: `Step, StepType, StepResult, StepStatus, _STEP_TYPE_TOOLS, _get_planning_profile`

**Step 5: Create __init__.py with re-exports**

Create `src/core/planner/__init__.py`:

```python
"""Plan-Then-Execute pipeline for small models.

Re-exports all public names for backward-compatible imports.
"""

from src.core.planner.parser import (
    StepType, Step, StepStatus, StepResult, PlanResult, PlanParser,
    _STEP_TYPE_TOOLS, _PLANNING_TIER_FACTORS, _compute_planning_profile,
    _PLANNING_PROFILES, _get_planning_profile, _infer_step_type,
    _MAX_PLAN_STEPS,
)
from src.core.planner.decomposer import (
    TaskDecomposer, _is_simple_task, _heuristic_decompose,
)
from src.core.planner.executor import (
    ChunkedExecutor, _infer_expected_tools, _filter_tools,
)
from src.core.planner.orchestrator import PlanExecutor, should_use_planner

__all__ = [
    "StepType", "Step", "StepStatus", "StepResult", "PlanResult",
    "PlanParser", "TaskDecomposer", "ChunkedExecutor", "PlanExecutor",
    "should_use_planner", "_infer_step_type", "_is_simple_task",
    "_heuristic_decompose", "_infer_expected_tools", "_filter_tools",
    "_compute_planning_profile", "_get_planning_profile",
    "_STEP_TYPE_TOOLS", "_PLANNING_TIER_FACTORS", "_PLANNING_PROFILES",
    "_MAX_PLAN_STEPS",
]
```

Wait — `PlanExecutor` and `should_use_planner` need a home. They tie all three together. Put them in a small `orchestrator.py`:

**Step 6: Create orchestrator.py**

Create `src/core/planner/orchestrator.py` containing:
- `PlanExecutor` class
- `should_use_planner()` function

These are lines 850-963 of the current planner.py.

**Step 7: Delete the old planner.py**

```bash
rm src/core/planner.py
```

(This was already replaced by the `src/core/planner/` package directory)

**Step 8: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS — __init__.py re-exports maintain all import paths

If any import errors: fix the specific imports in the affected test/source files.

**Step 9: Commit**

```bash
git add src/core/planner/ tests/test_planner.py tests/test_grammar.py
git rm src/core/planner.py
git commit -m "refactor: split planner.py into decomposer, parser, executor, orchestrator"
```

---

## Task 12: Delete SessionCwd (cwd.py)

**Files:**
- Delete: `src/core/cwd.py`
- Modify: Any remaining imports of SessionCwd → Workspace
- Move: `tests/test_cwd.py` relevant tests into `tests/test_workspace.py`

**Step 1: Search for remaining SessionCwd imports**

```bash
grep -r "SessionCwd\|from src.core.cwd" src/ tests/ --include="*.py"
```

Fix any remaining references.

**Step 2: Merge useful tests from test_cwd.py into test_workspace.py**

The CWD marker tests are obsolete (markers removed in Task 2). The git repo guard tests (`_check_git_repo`) should be kept in test_git.py. The quote normalization tests can be removed (no longer needed with list-based subprocess).

Keep: SessionCwd backward-compat tests (now testing Workspace.set/path).

**Step 3: Delete cwd.py**

```bash
git rm src/core/cwd.py
```

**Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove SessionCwd, Workspace is the sole CWD manager"
```

---

## Task 13: Final Integration Test and Cleanup

**Files:**
- All modified files
- Test: full suite

**Step 1: Run complete test suite**

Run: `pytest tests/ -v --tb=long`
Expected: ALL tests pass (556+ original + new tests)

**Step 2: Check for dead imports**

```bash
grep -r "from src.core.cwd" src/ tests/ --include="*.py"
```
Expected: No results

```bash
grep -r "shell=True" src/ --include="*.py"
```
Expected: No results in tool code (may appear in tests for mocking)

**Step 3: Verify no circular imports**

```bash
python -c "from src.core.planner import PlanExecutor; print('OK')"
python -c "from src.core.workspace import Workspace; print('OK')"
python -c "from src.core.agent import Agent; print('OK')"
```

**Step 4: Commit final cleanup**

```bash
git add -A
git commit -m "chore: final cleanup after Phase 3 hardening and refactor"
```

---

## Summary of Changes

| File | Action | Task |
|------|--------|------|
| `src/tools/shell.py` | Major rewrite: remove shell=True, add metachar rejection, cd handling | 1, 2 |
| `src/core/permission.py` | Add has_injection_pattern, symlink-safe is_path_safe | 3 |
| `src/tools/filesystem.py` | Add threading lock, WorkspaceBoundaryError handling | 4, 9 |
| `src/core/planner.py` | Delete (replaced by package) | 11 |
| `src/core/planner/__init__.py` | New: re-exports | 11 |
| `src/core/planner/parser.py` | New: data structures, constants, PlanParser | 11 |
| `src/core/planner/decomposer.py` | New: TaskDecomposer, heuristics | 11 |
| `src/core/planner/executor.py` | New: ChunkedExecutor with hard-block scope | 5, 6, 7, 11 |
| `src/core/planner/orchestrator.py` | New: PlanExecutor, should_use_planner | 11 |
| `src/core/workspace.py` | New: Workspace with boundary enforcement | 8 |
| `src/core/cwd.py` | Delete (replaced by workspace.py) | 12 |
| `src/core/config.py` | Add workspace_root to AgentConfig | 10 |
| `src/core/agent.py` | Update imports SessionCwd → Workspace | 9 |
| `src/tools/git.py` | Update imports SessionCwd → Workspace | 9 |
| `tests/test_workspace.py` | New: Workspace boundary tests | 8 |
| `tests/test_tools.py` | Add metachar, cd, thread safety tests | 1, 2, 4 |
| `tests/test_permission.py` | Add injection pattern tests | 3 |
| `tests/test_planner.py` | Add scope, messaging, inference tests | 5, 6, 7 |
| `tests/test_config.py` | Add workspace_root tests | 10 |

**Total: 13 tasks, ~13 commits, estimated 19 new test methods**
