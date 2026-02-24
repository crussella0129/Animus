# Animus Phase 3: Security Hardening, Scope Enforcement & Refactor

**Date:** 2026-02-24
**Status:** Approved
**Approach:** Security-First (all security fixes, then scope bleed, then refactor)

---

## Goals

1. Fix all documented security gaps (P0 shell injection, P1 command parsing, symlinks, thread safety, CWD spoofing)
2. Reduce scope bleed with hard-block enforcement after 1 violation
3. Refactor planner.py (1,053 lines) into 3 focused files
4. Add project-scoped working directory (Workspace) for boundary enforcement

---

## Section 1: Security Hardening

### 1A. Replace shell=True with list-based subprocess (P0)

**File:** `src/tools/shell.py`

- Replace `subprocess.run(shell=True, ...)` with `subprocess.run(shlex.split(cmd), ...)`
- Shell features (pipes, redirects, chaining) will no longer work implicitly
- Detect pipes/redirects in command string and reject with clear error message: "Shell pipes/redirects not supported. Use separate commands."
- CWD tracking: instead of injecting markers into shell command string, run command then separately capture CWD via `os.getcwd()` or follow-up process
- Windows quote normalization becomes unnecessary with list args (remove or simplify)
- Update all tests relying on shell features

### 1B. Fix command parsing (full structure, not first word)

**File:** `src/core/permission.py`

- `is_command_dangerous()` currently checks only the first word of the command
- New approach: use `shlex.split()` to tokenize the full command, check every token against dangerous commands list
- Detect and reject command substitution patterns: `$(...)`, backtick expressions
- Detect semicolons, `&&`, `||` within arguments as potential injection
- This becomes defense-in-depth since shell=True is removed

### 1C. Symlink-safe path checks

**File:** `src/core/permission.py`

- `is_path_safe()` currently uses `startswith()` on raw paths
- Change to `Path.resolve(strict=False)` before checking against deny lists
- Follows symlinks to real targets, preventing symlink traversal attacks

### 1D. Thread-safe write audit log

**File:** `src/tools/filesystem.py`

- `WriteFileTool._write_log` is a class-level list (shared across instances)
- Add `threading.Lock()` around append operations
- Lock acquired before append, released after

### 1E. CWD marker spoofing protection

**File:** `src/tools/shell.py`, `src/core/cwd.py`

- With shell=True removed, stdout marker injection is less relevant
- Defense-in-depth: use session-unique random token for CWD markers instead of static strings
- Generate token at session start, store in SessionCwd/Workspace

---

## Section 2: Scope Bleed Reduction

### 2A. Hard block after 1 violation

**File:** `src/core/planner.py` (later `src/core/planner/executor.py`)

- First out-of-scope tool call is blocked immediately (tool not executed)
- Inject message: "Tool '{name}' is not available for this step. Available tools: {list}"
- Step continues (model gets another chance to pick the right tool)
- After 2 blocked attempts, step terminates early
- Change from current behavior: warn on 1st, stop step on 2nd

### 2B. Explicit forbidden-tool messaging in step prompts

**File:** `src/core/planner.py` (later `src/core/planner/executor.py`)

- Per-step system prompt explicitly lists available tools
- Add instruction: "You may ONLY use the tools listed above. Do not attempt to use any other tools."
- Prompt-level complement to the hard block

### 2C. Tighter tool filtering heuristics

**File:** `src/core/planner.py` (later `src/core/planner/parser.py`)

- `_infer_step_type()` keyword overlap fix: check file extensions first (.py -> READ/WRITE), then fall back to verb-based inference
- Strict mode: if step description mentions a specific tool by name, only that tool is available

---

## Section 3: Refactor

### 3A-3D. Split planner.py into three files

**New structure:**

```
src/core/planner/
    __init__.py          # Re-exports public API
    decomposer.py        # TaskDecomposer (~100 lines)
    parser.py            # PlanParser + PlanStep + constants (~200 lines)
    executor.py          # ChunkedExecutor + execution logic (~700 lines)
```

**What moves where:**

| Content | Destination |
|---|---|
| `PlanStep` dataclass | `parser.py` |
| `_TYPE_KEYWORDS`, `_FILE_PATTERNS`, `_STEP_TYPE_TOOLS` | `parser.py` |
| `_PLANNING_TIER_FACTORS`, `_MAX_PLAN_STEPS` | `parser.py` |
| `TaskDecomposer`, `_is_simple_task()`, `_heuristic_decompose()` | `decomposer.py` |
| `PlanParser`, `_infer_step_type()` | `parser.py` |
| `ChunkedExecutor` + all execution logic | `executor.py` |
| `should_use_planner()` and public helpers | `__init__.py` |

**Backward compatibility:**
- `src/core/planner.py` deleted (no shim)
- `__init__.py` re-exports all public names
- All imports across codebase updated
- No compatibility comments or dead code

### 3E. Project Working Directory (Workspace)

**New file:** `src/core/workspace.py` (replaces `src/core/cwd.py`)

**Concept:** Animus locks to a project root directory at session start. All file operations are constrained to that directory and its children.

**Workspace class:**

```python
class WorkspaceBoundaryError(AnimusError): ...

class Workspace:
    def __init__(self, root: Path):
        self._root = root.resolve()
        self._cwd = self._root

    @property
    def root(self) -> Path:
        """Immutable project boundary."""
        return self._root

    @property
    def cwd(self) -> Path:
        """Current working directory within workspace."""
        return self._cwd

    def set_cwd(self, path: Path) -> None:
        """Update CWD. Must stay within root."""
        resolved = (self._cwd / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                f"Cannot change directory to {path}: "
                f"outside workspace root {self._root}"
            )
        self._cwd = resolved

    def resolve(self, path: str) -> Path:
        """Resolve path relative to CWD, enforce boundary."""
        resolved = (self._cwd / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                f"Path {path} resolves to {resolved}: "
                f"outside workspace root {self._root}"
            )
        return resolved
```

**Configuration:**
- `AgentConfig.workspace_root`: optional path in config.yaml
- CLI flag: `--workspace-root /path/to/project`
- Default: `os.getcwd()` at launch

**Integration points:**

| Component | Change |
|---|---|
| `PermissionChecker` | New `is_within_workspace(path, root)` method. Allowlist (workspace) + denylist (dangerous dirs) |
| `filesystem.py` tools | All paths resolved via `workspace.resolve()`. Paths outside root rejected |
| `shell.py` | Validate file path arguments within workspace. Set subprocess cwd to workspace.cwd |
| `git.py` | `_check_git_repo()` requires repo at or under workspace root |
| `Agent.__init__()` | Creates `Workspace(root=...)` and passes to all tools |
| `config.py` | Add `workspace_root: Optional[str]` to AgentConfig |

**What gets blocked:**
- `read_file("../../etc/passwd")` -> WorkspaceBoundaryError
- `write_file("/tmp/exploit.sh", ...)` -> WorkspaceBoundaryError
- `cd /` in shell -> CWD change rejected
- Symlinks resolving outside root -> blocked

**What's allowed:**
- Any path within project root and subdirectories
- Creating new subdirectories within root
- Moving CWD to any directory under root

---

## Execution Order

1. **Security hardening** (1A-1E) — fix shell injection, command parsing, symlinks, thread safety, CWD markers
2. **Scope bleed reduction** (2A-2C) — hard block, explicit prompts, tighter filtering
3. **Refactor** (3A-3E) — split planner.py, replace SessionCwd with Workspace, update all imports and tests

---

## Test Strategy

- All existing 556 tests must continue passing after each section
- New tests for:
  - Shell tool without shell=True (pipe rejection, redirect rejection)
  - Full command parsing (injection patterns caught)
  - Symlink path resolution
  - Thread-safe write log under concurrent access
  - Workspace boundary enforcement (resolve, set_cwd, blocked paths)
  - Hard block scope enforcement (1-violation block, 2-violation step termination)
  - Planner imports from new module structure

## Risk Mitigation

- Each section is committed independently so failures can be isolated
- Shell=True removal is the highest-risk change; test thoroughly before proceeding
- Planner refactor is purely structural; no logic changes during the move
