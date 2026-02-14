# Ornstein Phase 2: CLI Integration Plan

**Status:** In Progress
**Priority:** Medium
**Dependencies:** Phase 1 (Complete)

## Goals

1. Add `--cautious` flag to `animus rise` for automatic Ornstein sandboxing
2. Integrate isolation levels with the tool system
3. Add configuration UI for isolation settings
4. Create tool decorator for per-tool isolation preferences
5. Add permission prompts when changing isolation levels

## Implementation Tasks

### Task 1: Extend AnimusConfig with Isolation Settings

**File:** `src/core/config.py`

Add `IsolationConfig` class:

```python
class IsolationConfig(BaseModel):
    """Container isolation configuration."""
    model_config = {"extra": "ignore"}

    # Default isolation level for all tools
    default_level: str = "none"  # "none", "ornstein", "smough"

    # Ornstein (lightweight sandbox) settings
    ornstein_enabled: bool = False
    ornstein_timeout: int = 30
    ornstein_memory_mb: int = 512
    ornstein_allow_write: bool = False

    # Per-tool isolation overrides
    tool_isolation: dict[str, str] = {}  # {"run_shell": "ornstein", ...}

    # Auto-isolation for dangerous tools
    auto_isolate_dangerous: bool = False
```

Add to `AnimusConfig`:
```python
isolation: IsolationConfig = Field(default_factory=IsolationConfig)
```

### Task 2: Add --cautious Flag to CLI

**File:** `src/main.py`

Modify `rise()` command:

```python
@app.command()
def rise(
    resume: bool = typer.Option(False, "--resume", help="Resume the most recent session"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Resume a specific session by ID"),
    cautious: bool = typer.Option(False, "--cautious", help="Enable Ornstein sandbox for dangerous operations"),
    paranoid: bool = typer.Option(False, "--paranoid", help="Enable Smough container isolation (not yet implemented)"),
) -> None:
    """Awaken Animus. Start an interactive agent session."""
    cfg = AnimusConfig.load()

    # Apply isolation level from flags
    if paranoid:
        raise NotImplementedError("Smough layer (--paranoid) not yet implemented. Use --cautious for Ornstein sandbox.")
    elif cautious:
        cfg.isolation.default_level = "ornstein"
        cfg.isolation.ornstein_enabled = True
        info("[Isolation] Ornstein sandbox enabled (--cautious mode)")
```

### Task 3: Create Tool Isolation Decorator

**File:** `src/tools/base.py`

Add isolation metadata to Tool class:

```python
class Tool:
    """Base class for agent tools."""

    def __init__(self, isolation_level: str = "none"):
        self._isolation_level = isolation_level

    @property
    def isolation_level(self) -> str:
        """Get recommended isolation level for this tool."""
        return self._isolation_level

    @isolation_level.setter
    def isolation_level(self, level: str):
        """Set isolation level for this tool."""
        if level not in ("none", "ornstein", "smough"):
            raise ValueError(f"Invalid isolation level: {level}")
        self._isolation_level = level
```

Add decorator:

```python
def isolated(level: str = "ornstein"):
    """
    Decorator to mark a tool as requiring isolation.

    Usage:
        @isolated(level="ornstein")
        class MyDangerousTool(Tool):
            ...
    """
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._isolation_level = level

        cls.__init__ = new_init
        return cls

    return decorator
```

### Task 4: Integrate Isolation into Tool Execution

**File:** `src/core/agent.py`

Modify tool execution to respect isolation levels:

```python
def _execute_tool(self, tool_name: str, arguments: dict) -> str:
    """Execute a tool with appropriate isolation level."""
    tool = self._tools.get_tool(tool_name)

    if not tool:
        return f"Error: Unknown tool '{tool_name}'"

    # Determine isolation level
    from src.isolation import IsolationLevel, execute_with_isolation, OrnsteinConfig

    # Get config (passed to agent or loaded)
    config = getattr(self, '_config', None)

    isolation_level = IsolationLevel.NONE

    if config and config.isolation.ornstein_enabled:
        # Check per-tool override
        if tool_name in config.isolation.tool_isolation:
            level_str = config.isolation.tool_isolation[tool_name]
            isolation_level = IsolationLevel[level_str.upper()]
        # Check tool's recommended level
        elif hasattr(tool, 'isolation_level') and tool.isolation_level != "none":
            isolation_level = IsolationLevel[tool.isolation_level.upper()]
        # Use default level
        elif config.isolation.default_level != "none":
            isolation_level = IsolationLevel[config.isolation.default_level.upper()]

    # Execute with isolation
    if isolation_level == IsolationLevel.NONE:
        return tool.execute(arguments)
    else:
        ornstein_config = OrnsteinConfig(
            timeout_seconds=config.isolation.ornstein_timeout,
            memory_mb=config.isolation.ornstein_memory_mb,
            allow_write=config.isolation.ornstein_allow_write,
        )

        result = execute_with_isolation(
            tool.execute,
            args=(arguments,),
            level=isolation_level,
            config=ornstein_config,
        )

        if result.success:
            return result.output
        else:
            return f"Error (isolated): {result.error}"
```

### Task 5: Mark Dangerous Tools for Isolation

**File:** `src/tools/shell.py`

```python
from src.tools.base import isolated

@isolated(level="ornstein")  # Recommend isolation for shell commands
class RunShellTool(Tool):
    """Execute shell commands with safety checks."""
    ...
```

**File:** `src/tools/filesystem.py`

```python
# WriteFileTool could optionally be isolated
# ReadFileTool should NOT be isolated (performance)
```

### Task 6: Add Isolation Status to /tools Command

**File:** `src/main.py`

Enhance `/tools` slash command:

```python
if cmd == "/tools":
    tools = agent._tools.list_tools()
    if not tools:
        info("No tools registered.")
    else:
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Isolation", style="yellow")  # NEW

        for tool in tools:
            isolation = getattr(tool, 'isolation_level', 'none')
            isolation_display = {
                'none': '[dim]none[/]',
                'ornstein': '[yellow]ornstein[/]',
                'smough': '[red]smough[/]',
            }.get(isolation, isolation)

            table.add_row(tool.name, tool.description, isolation_display)

        console.print(table)
    return True
```

### Task 7: Configuration Management

Add to `animus config` command:

```bash
# Show isolation settings
animus config --show-isolation

# Enable Ornstein by default
animus config --set isolation.default_level=ornstein

# Override specific tool
animus config --set-tool-isolation run_shell=ornstein
```

### Task 8: Tests

**File:** `tests/test_isolation_integration.py`

```python
class TestCLIIntegration:
    def test_cautious_flag_enables_ornstein(self):
        """Test that --cautious flag enables Ornstein."""
        ...

    def test_paranoid_flag_not_implemented(self):
        """Test that --paranoid raises NotImplementedError."""
        ...

class TestToolIsolation:
    def test_isolated_decorator(self):
        """Test @isolated decorator."""
        ...

    def test_tool_execution_with_isolation(self):
        """Test that tools execute in sandbox when configured."""
        ...

    def test_per_tool_override(self):
        """Test per-tool isolation overrides."""
        ...
```

## Configuration Examples

### Example 1: Enable Ornstein by Default

```yaml
isolation:
  default_level: ornstein
  ornstein_enabled: true
  ornstein_timeout: 30
  ornstein_memory_mb: 512
  ornstein_allow_write: false
```

### Example 2: Selective Tool Isolation

```yaml
isolation:
  default_level: none
  ornstein_enabled: true
  tool_isolation:
    run_shell: ornstein      # Isolate shell commands
    web_search: ornstein     # Isolate web searches (when added)
    read_file: none          # Don't isolate read operations
```

### Example 3: Auto-Isolate Dangerous Operations

```yaml
isolation:
  default_level: none
  ornstein_enabled: true
  auto_isolate_dangerous: true  # Automatically isolate dangerous tools
```

## User Experience

### Normal Mode (default)

```bash
animus rise
# No isolation, fast execution
```

### Cautious Mode

```bash
animus rise --cautious
# Ornstein sandbox enabled for all tools
# ~100ms overhead per tool call
# Timeout enforcement, read-only filesystem
```

### Paranoid Mode (Phase 3)

```bash
animus rise --paranoid
# Full container isolation (Smough layer)
# Higher overhead but maximum security
```

## Success Criteria

- [ ] `--cautious` flag enables Ornstein sandbox
- [ ] Tool execution respects isolation levels
- [ ] `/tools` command shows isolation status
- [ ] Configuration persists isolation settings
- [ ] Per-tool isolation overrides work
- [ ] Tests cover all integration points (≥15 tests)
- [ ] Documentation updated

## Timeline

**Estimated:** 2-3 hours
- Task 1-2: Config + CLI flags (30 min)
- Task 3-4: Tool integration (60 min)
- Task 5-6: UI enhancements (30 min)
- Task 7-8: Tests (45 min)

---

**Next:** Begin implementation with Task 1 (Config extension)
