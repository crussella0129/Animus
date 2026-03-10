# Ferric Layer + Agent Reliability Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the Python agent loop with three targeted reliability fixes, then scaffold the Ferric Layer — three Rust crates (`ferric-parse`, `ferric-sandbox`, `ferric-cli`) that augment Animus with multi-language parsing, kernel-level isolation, and fast CLI startup.

**Architecture:** Python stays Python for all orchestration; Rust forms an invisible infrastructure layer called via subprocess+JSON. All Rust features degrade gracefully to Python fallbacks when binaries are absent (dev machines, pip-only installs). The three Python fixes address observed failures from reconstruction testing: write_file content corruption, planner overreach on targeted tasks, and RespondTool premature exit.

**Tech Stack:** Python 3.11+, Rust 2021 edition, tree-sitter 0.24, clap 4, serde/serde_json, rayon, nix 0.29, seccompiler 0.4, reqwest 0.12, indicatif 0.17, sysinfo 0.32

---

## PART A — Python Reliability Fixes

---

### Task 1: Fix write_file JSON Escaping Artifact

**Context:** The Qwen 14B model escapes double-quotes inside JSON string arguments — writing `\"` instead of `"`. When `write_file` receives content like `"""docstring"""`, it lands on disk as `\"\"\"docstring\"\"\"`, producing syntax errors. The fix is to unescape JSON-escaped sequences in the content argument before writing.

**Files:**
- Modify: `src/tools/filesystem.py:123-147`
- Test: `tests/test_filesystem.py`

**Step 1: Write the failing test**

In `tests/test_filesystem.py`, add a test class `TestWriteFileUnescaping`:

```python
class TestWriteFileUnescaping:
    """WriteFileTool must unescape JSON-escaped sequences in content."""

    def test_escaped_quotes_are_unescaped(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        # Model sends JSON-escaped content
        result = tool.execute({"path": str(path), "content": '\"\"\"docstring\"\"\"'})
        assert path.read_text() == '"""docstring"""'

    def test_escaped_newlines_are_unescaped(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        result = tool.execute({"path": str(path), "content": 'line1\\nline2'})
        assert path.read_text() == 'line1\nline2'

    def test_escaped_backslash_preserved(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        result = tool.execute({"path": str(path), "content": 'path = C:\\\\Users\\\\foo'})
        assert path.read_text() == 'path = C:\\Users\\foo'

    def test_normal_content_unchanged(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        result = tool.execute({"path": str(path), "content": 'def hello():\n    pass\n'})
        assert path.read_text() == 'def hello():\n    pass\n'
```

**Step 2: Run test to verify it fails**

```bash
cd C:/Users/charl/animus
pytest tests/test_filesystem.py::TestWriteFileUnescaping -v
```

Expected: FAIL — content written as-is without unescaping.

**Step 3: Implement the fix**

In `src/tools/filesystem.py`, find `execute()` in `WriteFileTool` (around line 123). After `content = args["content"]`, add:

```python
content = args["content"]
# Unescape JSON-escaped sequences that LLMs emit inside string arguments.
# Models like Qwen emit \" instead of " when writing content containing
# double-quotes, causing syntax errors in written files. We decode known
# escape sequences but preserve intentional double-backslashes (\\).
_UNESCAPE = [
    ('\\"', '"'),
    ('\\n', '\n'),
    ('\\t', '\t'),
    ('\\\\', '\\'),
]
# Only apply if the content looks like it has been JSON-escaped
# (i.e. contains \" but not actual unescaped content already)
if '\\"' in content or '\\n' in content or '\\t' in content:
    for escaped, unescaped in _UNESCAPE:
        content = content.replace(escaped, unescaped)
```

**Step 4: Run tests**

```bash
pytest tests/test_filesystem.py::TestWriteFileUnescaping -v
```

Expected: 4/4 PASS.

**Step 5: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: All existing tests pass.

**Step 6: Commit**

```bash
git add src/tools/filesystem.py tests/test_filesystem.py
git commit -m "fix: unescape JSON-escaped sequences in write_file content"
```

---

### Task 2: Add --no-plan Flag to animus rise

**Context:** `should_use_planner()` returns True for all medium-sized models (Qwen 14B Q4). The planner decomposes tasks before seeing the code, generating info-gathering plans rather than targeted fix plans. For specific, small tasks (debug this function, rewrite this class) the planner adds latency and noise. A `--no-plan` flag lets users bypass the planner and call `agent.run()` directly.

**Files:**
- Modify: `src/cli/app.py` (the `rise()` command definition)
- Modify: `src/core/agent.py` (add `run_direct()` method that always bypasses planner)
- Test: `tests/test_cli_app.py` or new `tests/test_no_plan_flag.py`

**Step 1: Write the failing test**

Create `tests/test_no_plan_flag.py`:

```python
"""Test that --no-plan flag bypasses the planner."""
from unittest.mock import patch, MagicMock
import pytest


def test_no_plan_flag_bypasses_should_use_planner():
    """When --no_plan=True, should_use_planner() is never consulted."""
    from src.core.agent import Agent

    mock_provider = MagicMock()
    mock_provider.capabilities.return_value = MagicMock(size_tier="medium", context_window=8192)
    mock_registry = MagicMock()
    mock_registry.names.return_value = []
    mock_registry.schemas.return_value = []

    agent = Agent(provider=mock_provider, tool_registry=mock_registry)

    with patch("src.core.agent.should_use_planner") as mock_sup:
        with patch.object(agent, "_run_agent_loop", return_value="result") as mock_loop:
            agent.run("fix bug in line 5", force_direct=True)
            mock_sup.assert_not_called()
            mock_loop.assert_called_once()


def test_no_plan_flag_false_consults_planner():
    """When force_direct=False, should_use_planner() is called normally."""
    from src.core.agent import Agent
    from unittest.mock import patch, MagicMock

    mock_provider = MagicMock()
    mock_provider.capabilities.return_value = MagicMock(size_tier="medium", context_window=8192)
    mock_registry = MagicMock()
    mock_registry.names.return_value = []
    mock_registry.schemas.return_value = []

    agent = Agent(provider=mock_provider, tool_registry=mock_registry)

    with patch("src.core.agent.should_use_planner", return_value=False) as mock_sup:
        with patch.object(agent, "_run_agent_loop", return_value="result"):
            agent.run("fix bug in line 5", force_direct=False)
            mock_sup.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_no_plan_flag.py -v
```

Expected: FAIL — `agent.run()` has no `force_direct` parameter.

**Step 3: Add `force_direct` parameter to `agent.run()`**

In `src/core/agent.py`, find the `run()` method definition. Add `force_direct: bool = False` parameter:

```python
def run(
    self,
    task: str,
    on_chunk: Callable[[str], None] | None = None,
    force: bool = False,
    force_direct: bool = False,  # NEW: bypass planner unconditionally
) -> str:
    """Run the agent on a task.

    Args:
        task: The task description.
        on_chunk: Optional streaming callback.
        force: Force planner usage (overrides size_tier check).
        force_direct: Skip planner entirely, use agent loop directly.
    """
    ...
    from src.core.planner import PlanExecutor, should_use_planner

    use_plan = not force_direct and (
        _plan_mode_state.get("active", False) or (not force and should_use_planner(self._provider))
        if force else True
    )
```

The actual logic change is simpler. Find the existing line:

```python
if not force and not should_use_planner(self._provider):
```

And change it to:

```python
if force_direct or (not force and not should_use_planner(self._provider)):
```

(i.e. skip planner if `force_direct=True` OR if planner is not indicated)

**Step 4: Add `--no-plan` option to `rise()` in `src/cli/app.py`**

Find the `rise()` function signature. Add:

```python
no_plan: bool = typer.Option(False, "--no-plan", help="Bypass planner — use agent loop directly for targeted tasks"),
```

Then find where `agent.run(user_input, ...)` is called and pass `force_direct=no_plan`.

**Step 5: Run tests**

```bash
pytest tests/test_no_plan_flag.py -v
pytest tests/ -x -q
```

Expected: All passing.

**Step 6: Commit**

```bash
git add src/core/agent.py src/cli/app.py tests/test_no_plan_flag.py
git commit -m "feat: add --no-plan flag to bypass planner for targeted tasks"
```

---

### Task 3: RespondTool Verification Gate

**Context:** RespondTool causes premature exit — the model calls `respond(message=...)` before verifying its work (e.g., before running tests or re-reading the file it wrote). The fix: update the RespondTool description and the system prompt to explicitly require a verification step before calling `respond`. Additionally, the agent loop should gate `respond` calls so the model has made at least one other tool call first (meaning it actually did something before claiming completion).

**Files:**
- Modify: `src/tools/base.py` (RespondTool.description, parameters)
- Modify: `src/core/agent.py` (agentic loop respond handling)
- Test: `tests/test_base_tools.py`

**Step 1: Write the failing test**

In `tests/test_base_tools.py`, add:

```python
class TestRespondToolVerificationGate:
    def test_respond_description_mentions_verification(self):
        """RespondTool description must require verification before use."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        # Description must mention verification
        assert "verif" in tool.description.lower() or "confirm" in tool.description.lower()

    def test_respond_has_verified_parameter(self):
        """RespondTool should have a 'verified' boolean parameter."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        props = tool.parameters["properties"]
        assert "verified" in props
        assert props["verified"]["type"] == "boolean"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_base_tools.py::TestRespondToolVerificationGate -v
```

Expected: FAIL — no `verified` param in RespondTool.

**Step 3: Update RespondTool**

In `src/tools/base.py`, find the `RespondTool` class. Update:

```python
class RespondTool(Tool):
    @property
    def name(self) -> str:
        return "respond"

    @property
    def description(self) -> str:
        return (
            "Return a final response to the user. IMPORTANT: Only call this after you have "
            "verified your work — e.g., re-read any file you wrote, ran the relevant test, "
            "or confirmed the output. Set verified=true once you have confirmed the result. "
            "Do NOT call this as your first action."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The final response to return to the user.",
                },
                "verified": {
                    "type": "boolean",
                    "description": "Set to true after you have verified your work (read back the file, ran tests, etc.)",
                },
            },
            "required": ["message", "verified"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        return args["message"]
```

**Step 4: Run tests**

```bash
pytest tests/test_base_tools.py::TestRespondToolVerificationGate -v
pytest tests/ -x -q
```

Expected: All pass.

**Step 5: Commit**

```bash
git add src/tools/base.py tests/test_base_tools.py
git commit -m "fix: add verification gate to RespondTool — require verified=true before exit"
```

---

## PART B — Ferric Layer: Rust Infrastructure

---

### Task 4: Rust Workspace Scaffold

**Context:** The Ferric Layer lives at `crates/` alongside `src/`. We need a workspace `Cargo.toml` at the repo root and the directory structure for all three crates. No implementation yet — just valid scaffolding that `cargo build --workspace` can compile.

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/ferric-parse/Cargo.toml`
- Create: `crates/ferric-parse/src/main.rs`
- Create: `crates/ferric-sandbox/Cargo.toml`
- Create: `crates/ferric-sandbox/src/main.rs`
- Create: `crates/ferric-cli/Cargo.toml`
- Create: `crates/ferric-cli/src/main.rs`

**Step 1: Create workspace Cargo.toml**

`Cargo.toml` at repo root:

```toml
[workspace]
members = [
    "crates/ferric-parse",
    "crates/ferric-sandbox",
    "crates/ferric-cli",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "GPL-3.0"
repository = "https://github.com/crussella0129/Animus"
```

**Step 2: Create ferric-parse scaffold**

`crates/ferric-parse/Cargo.toml`:

```toml
[package]
name = "ferric-parse"
version.workspace = true
edition.workspace = true
description = "Multi-language code parser for Animus using tree-sitter grammars"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
```

`crates/ferric-parse/src/main.rs`:

```rust
//! ferric-parse — multi-language code parser for Animus.
//! Called internally by Python via subprocess. Not for direct use.

fn main() {
    eprintln!("ferric-parse: not yet implemented");
    std::process::exit(1);
}
```

**Step 3: Create ferric-sandbox scaffold**

`crates/ferric-sandbox/Cargo.toml`:

```toml
[package]
name = "ferric-sandbox"
version.workspace = true
edition.workspace = true
description = "Kernel-level process isolation for Animus (Ornstein & Smough)"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
```

`crates/ferric-sandbox/src/main.rs`:

```rust
//! ferric-sandbox — kernel-level isolation for Animus tool execution.
//! Activated by `animus rise --ornsmo`. Not for direct use.

fn main() {
    eprintln!("ferric-sandbox: not yet implemented");
    std::process::exit(1);
}
```

**Step 4: Create ferric-cli scaffold**

`crates/ferric-cli/Cargo.toml`:

```toml
[package]
name = "ferric-cli"
version.workspace = true
edition.workspace = true
description = "Fast Rust entrypoint for Animus CLI — simple commands, delegates to Python for heavy ones"

[dependencies]
clap = { version = "4", features = ["derive", "color"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

`crates/ferric-cli/src/main.rs`:

```rust
//! ferric-cli — fast Animus entrypoint.
//! Handles detect/config/status in Rust, delegates rise/ingest/search to Python.

fn main() {
    eprintln!("ferric-cli: not yet implemented");
    std::process::exit(1);
}
```

**Step 5: Build to verify compilation**

```bash
cd C:/Users/charl/animus
cargo build --workspace
```

Expected: All three crates compile (stubs, no warnings about unimplemented).

**Step 6: Add .gitignore entry for Cargo artifacts**

Add to `.gitignore` (or create if absent):

```
/target/
```

**Step 7: Commit**

```bash
git add Cargo.toml crates/ .gitignore
git commit -m "feat: scaffold Ferric Layer Rust workspace (ferric-parse, ferric-sandbox, ferric-cli)"
```

---

### Task 5: ferric-parse — Core Implementation (Python + Rust parsers)

**Context:** `ferric-parse` reads a source file, parses it with tree-sitter, and emits a JSON `FileParseResult` matching the schema already used by `knowledge/parser.py`. Start with Python (P0) and Rust (P0) parsers. The output must be parseable by the existing Python infrastructure.

**Files:**
- Modify: `crates/ferric-parse/Cargo.toml` (add tree-sitter deps)
- Create: `crates/ferric-parse/src/output.rs`
- Create: `crates/ferric-parse/src/parsers/mod.rs`
- Create: `crates/ferric-parse/src/parsers/python.rs`
- Create: `crates/ferric-parse/src/parsers/rust.rs`
- Modify: `crates/ferric-parse/src/main.rs` (implement CLI)
- Create: `crates/ferric-parse/tests/` (integration tests)

**Step 1: Update Cargo.toml with tree-sitter dependencies**

```toml
[package]
name = "ferric-parse"
version.workspace = true
edition.workspace = true
description = "Multi-language code parser for Animus using tree-sitter grammars"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tree-sitter = "0.24"
tree-sitter-python = "0.23"
tree-sitter-rust = "0.23"
```

**Step 2: Write the output types (`src/output.rs`)**

```rust
use serde::{Deserialize, Serialize};

/// A parsed code node (class, function, method).
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct CodeNode {
    pub kind: String,           // "class" | "function" | "method"
    pub name: String,
    pub qualified_name: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub docstring: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub bases: Vec<String>,
    #[serde(default)]
    pub decorators: Vec<String>,
}

/// A call or import edge between nodes.
#[derive(Debug, Serialize, Deserialize)]
pub struct CodeEdge {
    pub source_qname: String,
    pub target_name: String,
    pub kind: String,  // "CALLS" | "IMPORTS" | "INHERITS"
}

/// Full parse result for one file — matches Python FileParseResult schema.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct FileParseResult {
    pub file_path: String,
    #[serde(default)]
    pub nodes: Vec<CodeNode>,
    #[serde(default)]
    pub edges: Vec<CodeEdge>,
}
```

**Step 3: Write the parser trait (`src/parsers/mod.rs`)**

```rust
use std::path::Path;
use crate::output::FileParseResult;

pub mod python;
pub mod rust;

pub trait LanguageParser {
    fn parse_file(&self, path: &Path, source: &str) -> FileParseResult;
}

pub fn get_parser_for_extension(ext: &str) -> Option<Box<dyn LanguageParser>> {
    match ext {
        "py" => Some(Box::new(python::PythonParser)),
        "rs" => Some(Box::new(rust::RustParser)),
        _ => None,
    }
}
```

**Step 4: Implement Python parser (`src/parsers/python.rs`)**

```rust
use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::output::{FileParseResult, CodeNode, CodeEdge};
use crate::parsers::LanguageParser;

pub struct PythonParser;

impl LanguageParser for PythonParser {
    fn parse_file(&self, path: &Path, source: &str) -> FileParseResult {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return FileParseResult { file_path: path.display().to_string(), ..Default::default() },
        };

        let mut result = FileParseResult {
            file_path: path.display().to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        let root = tree.root_node();
        let file_stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        collect_python_nodes(root, source, file_stem, &path.display().to_string(), &mut result);
        result
    }
}

fn get_node_text<'a>(node: Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn collect_python_nodes(
    node: Node,
    source: &str,
    module: &str,
    file_path: &str,
    result: &mut FileParseResult,
) {
    match node.kind() {
        "class_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = get_node_text(name_node, source).to_string();
                let qualified = format!("{}.{}", module, name);
                let docstring = extract_docstring(node, source);

                let bases = if let Some(args) = node.child_by_field_name("superclasses") {
                    collect_identifiers(args, source)
                } else {
                    Vec::new()
                };

                result.nodes.push(CodeNode {
                    kind: "class".to_string(),
                    name: name.clone(),
                    qualified_name: qualified,
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    docstring,
                    bases,
                    ..Default::default()
                });
            }
        }
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = get_node_text(name_node, source).to_string();
                let qualified = format!("{}.{}", module, name);
                let docstring = extract_docstring(node, source);

                result.nodes.push(CodeNode {
                    kind: "function".to_string(),
                    name: name.clone(),
                    qualified_name: qualified,
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    docstring,
                    ..Default::default()
                });
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_nodes(child, source, module, file_path, result);
    }
}

fn extract_docstring(node: Node, source: &str) -> Option<String> {
    // Look for a string literal as the first statement in the body
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                if let Some(string_node) = child.child(0) {
                    if string_node.kind() == "string" {
                        let text = get_node_text(string_node, source);
                        return Some(text.trim_matches('"').trim_matches('\'').to_string());
                    }
                }
            }
            break; // Only check first statement
        }
    }
    None
}

fn collect_identifiers(node: Node, source: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            ids.push(get_node_text(child, source).to_string());
        }
    }
    ids
}
```

**Step 5: Implement Rust parser (`src/parsers/rust.rs`)**

```rust
use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::output::{FileParseResult, CodeNode};
use crate::parsers::LanguageParser;

pub struct RustParser;

impl LanguageParser for RustParser {
    fn parse_file(&self, path: &Path, source: &str) -> FileParseResult {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_rust::LANGUAGE.into()).unwrap();

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return FileParseResult { file_path: path.display().to_string(), ..Default::default() },
        };

        let mut result = FileParseResult {
            file_path: path.display().to_string(),
            ..Default::default()
        };

        let module = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        collect_rust_nodes(tree.root_node(), source, module, &path.display().to_string(), &mut result);
        result
    }
}

fn get_text<'a>(node: Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn collect_rust_nodes(node: Node, source: &str, module: &str, file_path: &str, result: &mut FileParseResult) {
    match node.kind() {
        "struct_item" | "enum_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = get_text(name_node, source).to_string();
                result.nodes.push(CodeNode {
                    kind: "class".to_string(), // Map struct/enum to "class" for Python compat
                    name: name.clone(),
                    qualified_name: format!("{}::{}", module, name),
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    ..Default::default()
                });
            }
        }
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = get_text(name_node, source).to_string();
                result.nodes.push(CodeNode {
                    kind: "function".to_string(),
                    name: name.clone(),
                    qualified_name: format!("{}::{}", module, name),
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    ..Default::default()
                });
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_nodes(child, source, module, file_path, result);
    }
}
```

**Step 6: Implement `main.rs` CLI**

```rust
mod output;
mod parsers;

use clap::Parser as ClapParser;
use std::path::Path;

#[derive(ClapParser)]
#[command(name = "ferric-parse", about = "Multi-language code parser for Animus")]
struct Cli {
    /// File to parse
    file: String,
    /// Output format (only json supported)
    #[arg(long, default_value = "json")]
    format: String,
}

fn main() {
    let cli = Cli::parse();
    let path = Path::new(&cli.file);

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ferric-parse: failed to read {}: {}", cli.file, e);
            std::process::exit(1);
        }
    };

    let parser = match parsers::get_parser_for_extension(ext) {
        Some(p) => p,
        None => {
            // Unsupported extension — emit empty result, don't error
            let result = output::FileParseResult {
                file_path: cli.file.clone(),
                ..Default::default()
            };
            println!("{}", serde_json::to_string(&result).unwrap());
            return;
        }
    };

    let result = parser.parse_file(path, &source);
    println!("{}", serde_json::to_string(&result).unwrap());
}
```

**Step 7: Write integration test**

Create `crates/ferric-parse/tests/test_parse.rs`:

```rust
use std::process::Command;
use std::path::PathBuf;

fn ferric_parse_binary() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test binary name
    path.pop(); // Remove deps/
    path.push("ferric-parse");
    #[cfg(windows)]
    path.set_extension("exe");
    path
}

#[test]
fn test_parse_python_file_produces_nodes() {
    let binary = ferric_parse_binary();
    if !binary.exists() {
        eprintln!("Skipping: ferric-parse binary not found at {:?}", binary);
        return;
    }

    // Parse the ferric-parse main.rs itself as a test fixture
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sample.py");

    let output = Command::new(&binary)
        .arg(fixture.to_str().unwrap())
        .output()
        .expect("Failed to run ferric-parse");

    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .expect("Output should be valid JSON");

    assert!(json["nodes"].is_array());
    assert!(!json["nodes"].as_array().unwrap().is_empty(), "Should have found at least one node");
}
```

Create `crates/ferric-parse/tests/fixtures/sample.py`:

```python
"""Sample Python file for ferric-parse integration tests."""


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


def standalone_function(x: float) -> float:
    """A standalone function."""
    return x * 2.0
```

**Step 8: Build and test**

```bash
cd C:/Users/charl/animus
cargo build -p ferric-parse
cargo test -p ferric-parse
```

Expected: binary builds, test passes (finds Calculator class + 3 functions).

**Step 9: Commit**

```bash
git add crates/ferric-parse/
git commit -m "feat: implement ferric-parse with Python and Rust tree-sitter parsers"
```

---

### Task 6: FerricParser Python Wrapper

**Context:** The existing `knowledge/parser.py` has a `ParserRegistry` and a `LanguageParser` base class. We need a `FerricParser` that calls the `ferric-parse` binary via subprocess and falls back to the Python AST parser if the binary isn't present. Register it in the `ParserRegistry` — it auto-takes over for `.py`, `.rs`, `.ts`, `.go` files.

**Files:**
- Modify: `src/knowledge/parser.py`
- Test: `tests/test_ferric_parser.py`

**Step 1: Read the existing parser module to understand ParserRegistry**

First, read `src/knowledge/parser.py` to understand `LanguageParser`, `FileParseResult`, and `ParserRegistry` interfaces, then proceed.

**Step 2: Write failing tests**

Create `tests/test_ferric_parser.py`:

```python
"""Tests for the FerricParser Python wrapper."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestFerricParserInterface:
    def test_ferric_parser_implements_language_parser(self):
        from src.knowledge.parser import FerricParser, LanguageParser
        assert issubclass(FerricParser, LanguageParser)

    def test_ferric_parser_supported_extensions(self):
        from src.knowledge.parser import FerricParser
        parser = FerricParser()
        exts = parser.supported_extensions()
        assert ".py" in exts
        assert ".rs" in exts

    def test_ferric_parser_falls_back_when_binary_absent(self, tmp_path):
        """When ferric-parse binary is missing, parse_file returns empty FileParseResult."""
        from src.knowledge.parser import FerricParser
        parser = FerricParser(binary_path="nonexistent-binary-xyz")

        sample = tmp_path / "test.py"
        sample.write_text("def hello(): pass\n")

        result = parser.parse_file(sample)
        # Should not raise, returns empty result
        assert result.file_path == str(sample)
        assert isinstance(result.nodes, list)

    def test_ferric_parser_parses_python_file(self, tmp_path):
        """When binary is available, parse_file returns populated FileParseResult."""
        from src.knowledge.parser import FerricParser, FileParseResult

        fake_output = json.dumps({
            "file_path": "/tmp/test.py",
            "nodes": [
                {"kind": "function", "name": "hello", "qualified_name": "test.hello",
                 "file_path": "/tmp/test.py", "line_start": 1, "line_end": 1,
                 "docstring": None, "args": [], "bases": [], "decorators": []}
            ],
            "edges": []
        })

        sample = tmp_path / "test.py"
        sample.write_text("def hello(): pass\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=fake_output, stderr=""
            )
            parser = FerricParser(binary_path="ferric-parse")
            result = parser.parse_file(sample)

        assert len(result.nodes) == 1
        assert result.nodes[0].name == "hello"
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_ferric_parser.py -v
```

Expected: FAIL — `FerricParser` does not exist yet.

**Step 4: Add FerricParser to `src/knowledge/parser.py`**

After the existing parser classes, add:

```python
def _find_ferric_binary(name: str):
    """Locate a Ferric binary: bundled (src/bin/) → PATH → None."""
    import shutil
    bundled = Path(__file__).parent.parent / "bin" / name
    if bundled.exists():
        return str(bundled)
    found = shutil.which(name)
    return found  # May be None


class FerricParser(LanguageParser):
    """Multi-language parser backed by the ferric-parse Rust binary.

    Falls back gracefully to returning an empty FileParseResult if the
    binary is not installed — preserving Animus design principle #4
    (graceful absence).
    """

    _SUPPORTED_EXTENSIONS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".c", ".cpp", ".h"}

    def __init__(self, binary_path: str | None = None):
        self._binary = binary_path or _find_ferric_binary("ferric-parse")

    def supported_extensions(self) -> set[str]:
        return self._SUPPORTED_EXTENSIONS

    def is_available(self) -> bool:
        """Return True if the ferric-parse binary is accessible."""
        return self._binary is not None

    def parse_file(self, path: Path) -> "FileParseResult":
        import subprocess, json
        if self._binary is None:
            return FileParseResult(file_path=str(path))
        try:
            result = subprocess.run(
                [self._binary, str(path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return FileParseResult(file_path=str(path))
            data = json.loads(result.stdout)
            return FileParseResult(**data)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError):
            return FileParseResult(file_path=str(path))
```

Also update `ParserRegistry` to prefer `FerricParser` when available. Find where parsers are registered (likely in `__init__` or a `register_defaults()` method) and add:

```python
ferric = FerricParser()
if ferric.is_available():
    self.register(ferric)
```

**Step 5: Run tests**

```bash
pytest tests/test_ferric_parser.py -v
pytest tests/ -x -q
```

Expected: All pass.

**Step 6: Commit**

```bash
git add src/knowledge/parser.py tests/test_ferric_parser.py
git commit -m "feat: add FerricParser Python wrapper for ferric-parse binary"
```

---

### Task 7: ferric-sandbox Scaffold + --ornsmo Flag

**Context:** The kernel-level sandbox (seccomp-bpf, namespaces, cgroups v2) is Linux-only. On Windows/macOS we must detect the platform and fall back gracefully. This task: scaffold the crate with full platform detection, implement a working subprocess wrapper (no kernel primitives yet), add `--ornsmo` flag to the CLI replacing `--cautious`/`--paranoid`, and wire it into `tools/shell.py`.

**Note:** Kernel-level features (seccomp, namespaces, cgroups) are Linux-only and not implemented in this task. The stub will log a warning on Linux until kernel integration is added. On Windows (the current dev machine), the graceful fallback is used.

**Files:**
- Modify: `crates/ferric-sandbox/Cargo.toml`
- Modify: `crates/ferric-sandbox/src/main.rs` (full implementation)
- Modify: `src/cli/app.py` (replace `--cautious`/`--paranoid` with `--ornsmo`)
- Modify: `src/tools/shell.py` (handle `ornsmo` isolation level)
- Test: `tests/test_ornsmo_flag.py`

**Step 1: Update ferric-sandbox Cargo.toml**

```toml
[package]
name = "ferric-sandbox"
version.workspace = true
edition.workspace = true

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }

[target.'cfg(unix)'.dependencies]
# nix will be added for kernel-level features in a future phase
```

**Step 2: Implement ferric-sandbox main.rs (subprocess wrapper + platform detection)**

```rust
use clap::Parser as ClapParser;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::Command;
use std::time::Instant;

#[derive(ClapParser)]
#[command(name = "ferric-sandbox", about = "Process isolation for Animus tool execution")]
struct Cli {
    #[arg(long, default_value = "512")]
    memory: u64,
    #[arg(long, default_value = "30")]
    timeout: u64,
    #[arg(long)]
    no_network: bool,
    #[arg(long)]
    read_only: bool,
    #[arg(long)]
    smough: bool,
}

#[derive(Deserialize)]
struct SandboxRequest {
    command: String,
}

#[derive(Serialize)]
struct IsolationInfo {
    ornstein: bool,
    smough: bool,
    reason: String,
}

#[derive(Serialize)]
struct ResourceUsage {
    wall_time_ms: u128,
    cpu_time_ms: u64,
    peak_memory_kb: u64,
}

#[derive(Serialize)]
struct SandboxResult {
    success: bool,
    output: String,
    error: Option<String>,
    exit_code: i32,
    isolation: IsolationInfo,
    resource_usage: ResourceUsage,
}

fn main() {
    let cli = Cli::parse();

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap_or(0);

    let req: SandboxRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ferric-sandbox: invalid input JSON: {}", e);
            std::process::exit(1);
        }
    };

    let start = Instant::now();

    // Platform detection — kernel features only on Linux
    #[cfg(target_os = "linux")]
    let (ornstein_active, ornstein_reason) = (
        false, // Kernel integration added in future phase
        "Linux detected — kernel isolation not yet implemented, using process isolation".to_string(),
    );

    #[cfg(not(target_os = "linux"))]
    let (ornstein_active, ornstein_reason) = (
        false,
        format!("Platform {} does not support kernel isolation — using process isolation", std::env::consts::OS),
    );

    // Parse and run the command via subprocess
    let parts: Vec<&str> = req.command.split_whitespace().collect();
    if parts.is_empty() {
        eprintln!("ferric-sandbox: empty command");
        std::process::exit(1);
    }

    let result = Command::new(parts[0])
        .args(&parts[1..])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    let wall_time = start.elapsed().as_millis();

    let output = match result {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let exit_code = out.status.code().unwrap_or(-1);
            let success = out.status.success();
            SandboxResult {
                success,
                output: stdout,
                error: if stderr.is_empty() { None } else { Some(stderr) },
                exit_code,
                isolation: IsolationInfo {
                    ornstein: ornstein_active,
                    smough: false,
                    reason: ornstein_reason,
                },
                resource_usage: ResourceUsage {
                    wall_time_ms: wall_time,
                    cpu_time_ms: 0,
                    peak_memory_kb: 0,
                },
            }
        }
        Err(e) => SandboxResult {
            success: false,
            output: String::new(),
            error: Some(e.to_string()),
            exit_code: -1,
            isolation: IsolationInfo {
                ornstein: ornstein_active,
                smough: false,
                reason: ornstein_reason,
            },
            resource_usage: ResourceUsage { wall_time_ms: wall_time, cpu_time_ms: 0, peak_memory_kb: 0 },
        },
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
```

**Step 3: Write failing tests for --ornsmo flag**

Create `tests/test_ornsmo_flag.py`:

```python
"""Tests for --ornsmo flag replacing --cautious/--paranoid."""
import pytest


def test_ornsmo_is_valid_rise_option():
    """The rise() command must accept an --ornsmo flag."""
    from src.cli.app import app
    import typer.testing
    runner = typer.testing.CliRunner()
    # --help should list --ornsmo without error
    result = runner.invoke(app, ["rise", "--help"])
    assert result.exit_code == 0
    assert "ornsmo" in result.output.lower()


def test_cautious_flag_removed():
    """The deprecated --cautious flag should no longer be present."""
    from src.cli.app import app
    import typer.testing
    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["rise", "--help"])
    assert result.exit_code == 0
    # cautious should not appear in help text
    assert "--cautious" not in result.output
```

**Step 4: Update `src/cli/app.py` — replace --cautious/--paranoid with --ornsmo**

In the `rise()` function signature, find:
```python
cautious: bool = typer.Option(False, "--cautious", ...)
```
Replace with:
```python
ornsmo: bool = typer.Option(False, "--ornsmo",
    help="Enable Ornstein & Smough isolation (kernel sandbox on Linux, process isolation elsewhere)"),
```

Find where `cautious` is used to set `cfg.isolation.ornstein_enabled` and update:
```python
if ornsmo:
    cfg.isolation.default_level = "ornsmo"
    info("[OrnSmo] Dual-layer isolation enabled")
    info(f"[OrnSmo] Platform: {sys.platform}")
```

**Step 5: Update `src/tools/shell.py` to handle `ornsmo` isolation level**

Find the `execute()` method in `RunShellTool`. Add handling for `isolation_level == "ornsmo"`:

```python
if self._isolation_level == "ornsmo":
    import json, subprocess as _sp
    sandbox_binary = _find_ferric_binary("ferric-sandbox")
    if sandbox_binary:
        sandbox_input = json.dumps({"command": command})
        proc = _sp.run(
            [sandbox_binary, "--timeout", str(timeout)],
            input=sandbox_input, capture_output=True, text=True,
            timeout=timeout + 5
        )
        try:
            data = json.loads(proc.stdout)
            if data.get("success"):
                return data.get("output", "")
            return f"Error: {data.get('error', 'unknown error')}"
        except json.JSONDecodeError:
            pass  # Fall through to direct execution
    # Fallback: direct execution with warning
    _log.warning("[OrnSmo] ferric-sandbox binary not found, falling back to direct execution")
```

Add `_find_ferric_binary` import from `src/knowledge/parser.py` or duplicate the lookup function in `shell.py`.

**Step 6: Build ferric-sandbox**

```bash
cd C:/Users/charl/animus
cargo build -p ferric-sandbox
```

**Step 7: Run tests**

```bash
pytest tests/test_ornsmo_flag.py -v
pytest tests/ -x -q
```

**Step 8: Commit**

```bash
git add crates/ferric-sandbox/ src/cli/app.py src/tools/shell.py tests/test_ornsmo_flag.py
git commit -m "feat: scaffold ferric-sandbox and add --ornsmo flag (replaces --cautious/--paranoid)"
```

---

### Task 8: ferric-cli — Fast Rust Entrypoint

**Context:** Every `animus detect` currently pays 300–800ms of Python import overhead. `ferric-cli` provides a <10ms Rust entrypoint. It handles `detect`, `config --show`, `config --path`, `status` entirely in Rust and delegates `rise`, `ingest`, `chat`, `search`, `graph` to Python. The user still types `animus` — the binary name doesn't change.

**Files:**
- Modify: `crates/ferric-cli/Cargo.toml`
- Create: `crates/ferric-cli/src/commands/mod.rs`
- Create: `crates/ferric-cli/src/commands/detect.rs`
- Create: `crates/ferric-cli/src/commands/config_cmd.rs`
- Create: `crates/ferric-cli/src/commands/status.rs`
- Create: `crates/ferric-cli/src/delegate.rs`
- Modify: `crates/ferric-cli/src/main.rs`

**Step 1: Update ferric-cli Cargo.toml**

```toml
[package]
name = "ferric-cli"
version.workspace = true
edition.workspace = true
description = "Fast Animus CLI entrypoint — handles simple commands in Rust, delegates heavy ones to Python"

[dependencies]
clap = { version = "4", features = ["derive", "color"] }
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
sysinfo = "0.32"
colored = "2"
```

**Step 2: Implement detect command (`src/commands/detect.rs`)**

```rust
use sysinfo::System;

pub fn run() {
    let mut sys = System::new_all();
    sys.refresh_all();

    println!("System Information");
    println!("==================");
    println!("OS:          {} {}", System::name().unwrap_or_default(),
             System::os_version().unwrap_or_default());
    println!("Kernel:      {}", System::kernel_version().unwrap_or_default());
    println!("Hostname:    {}", System::host_name().unwrap_or_default());
    println!("CPU cores:   {}", sys.cpus().len());
    println!("Total RAM:   {} MB", sys.total_memory() / 1024 / 1024);
    println!("Free RAM:    {} MB", sys.available_memory() / 1024 / 1024);

    // Check for GPU (basic heuristic via process list)
    println!();
    println!("Ferric Layer:");
    println!("  ferric-parse:   {}", check_binary("ferric-parse"));
    println!("  ferric-sandbox: {}", check_binary("ferric-sandbox"));
}

fn check_binary(name: &str) -> &'static str {
    // Check if the binary is findable
    if which_exists(name) { "available" } else { "not found" }
}

fn which_exists(name: &str) -> bool {
    std::env::var("PATH")
        .unwrap_or_default()
        .split(if cfg!(windows) { ';' } else { ':' })
        .any(|dir| {
            let mut p = std::path::PathBuf::from(dir);
            p.push(name);
            if cfg!(windows) { p.set_extension("exe"); }
            p.exists()
        })
}
```

**Step 3: Implement config command (`src/commands/config_cmd.rs`)**

```rust
use std::path::PathBuf;

pub fn show_config() {
    let path = find_config_path();
    match path {
        Some(p) => {
            println!("Config: {}", p.display());
            match std::fs::read_to_string(&p) {
                Ok(contents) => println!("{}", contents),
                Err(e) => eprintln!("Error reading config: {}", e),
            }
        }
        None => println!("No config file found. Run 'animus init' to create one."),
    }
}

pub fn show_config_path() {
    match find_config_path() {
        Some(p) => println!("{}", p.display()),
        None => println!("(none)"),
    }
}

fn find_config_path() -> Option<PathBuf> {
    // Check standard locations
    let candidates = [
        PathBuf::from("config.yaml"),
        dirs_home().map(|h| h.join(".animus").join("config.yaml")).unwrap_or_default(),
    ];
    candidates.into_iter().find(|p| p.exists() && !p.as_os_str().is_empty())
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}
```

**Step 4: Implement delegate (`src/delegate.rs`)**

```rust
use std::process::{Command, ExitCode};

/// Find the Python interpreter.
fn find_python() -> String {
    for candidate in &["python3", "python"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return candidate.to_string();
        }
    }
    "python".to_string()
}

/// Delegate a command to Python's src.main module.
/// Passes all args directly through. Returns the Python process exit code.
pub fn delegate_to_python(args: &[String]) -> ExitCode {
    let python = find_python();
    let mut cmd = Command::new(&python);
    cmd.arg("-m").arg("src.main");
    cmd.args(args);

    match cmd.status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("ferric-cli: failed to launch Python runtime: {}", e);
            ExitCode::FAILURE
        }
    }
}
```

**Step 5: Implement main.rs**

```rust
mod commands {
    pub mod detect;
    pub mod config_cmd;
    pub mod status;
}
mod delegate;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "animus", about = "Animus — local-first LLM agent")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware and check system capabilities
    Detect,
    /// Show or modify configuration
    Config {
        #[arg(long)]
        show: bool,
        #[arg(long)]
        path: bool,
    },
    /// Show system status (detect + config summary)
    Status,
    /// Start an agent session (delegates to Python)
    #[command(allow_external_subcommands = true)]
    Rise,
    /// Ingest a codebase into the knowledge graph (delegates to Python)
    Ingest,
    /// Search the knowledge graph (delegates to Python)
    Search,
    /// Build knowledge graph from a path (delegates to Python)
    Graph,
    /// Download a model (delegates to Python)
    Pull,
    /// Initialize a new Animus workspace (delegates to Python)
    Init,
    /// List available models (delegates to Python)
    Models,
    /// Show routing statistics (delegates to Python)
    RoutingStats,
    /// Manage sessions (delegates to Python)
    Sessions,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Capture original args for delegation (skip the program name)
    let original_args: Vec<String> = std::env::args().skip(1).collect();

    match cli.command {
        Commands::Detect => {
            commands::detect::run();
            ExitCode::SUCCESS
        }
        Commands::Config { show, path } => {
            if path {
                commands::config_cmd::show_config_path();
            } else {
                commands::config_cmd::show_config();
            }
            ExitCode::SUCCESS
        }
        Commands::Status => {
            commands::detect::run();
            ExitCode::SUCCESS
        }
        // Everything else: delegate to Python
        _ => delegate::delegate_to_python(&original_args),
    }
}
```

Create `crates/ferric-cli/src/commands/status.rs` (re-exports detect for now):
```rust
pub use super::detect::run;
```

**Step 6: Build ferric-cli**

```bash
cd C:/Users/charl/animus
cargo build -p ferric-cli
```

**Step 7: Manual smoke test**

```bash
./target/debug/ferric-cli detect
./target/debug/ferric-cli config --show
./target/debug/ferric-cli rise --help  # Should delegate to Python
```

Expected:
- `detect` prints hardware info
- `config --show` prints config or "No config file found"
- `rise --help` spawns Python and shows the Typer help text

**Step 8: Commit**

```bash
git add crates/ferric-cli/
git commit -m "feat: implement ferric-cli with detect/config/status and Python delegation"
```

---

### Task 9: Binary Discovery + Integration Test

**Context:** Python needs to find the Ferric binaries. The lookup logic (`_find_ferric_binary`) is currently duplicated between `parser.py` and `shell.py`. Extract to a shared `src/ferric.py` utility module. Add an integration test that verifies the full chain: build the binary, call `FerricParser`, get a `FileParseResult` back with real nodes.

**Files:**
- Create: `src/ferric.py`
- Modify: `src/knowledge/parser.py` (import from ferric.py)
- Modify: `src/tools/shell.py` (import from ferric.py)
- Test: `tests/test_ferric_integration.py`

**Step 1: Create `src/ferric.py`**

```python
"""Ferric Layer binary discovery and availability utilities.

All Python code that needs to locate Ferric binaries should import from here.
The discovery order is: bundled (src/bin/) → PATH → None.
When a binary is not found, callers must fall back to Python equivalents.
"""
from __future__ import annotations

import shutil
from pathlib import Path

_BIN_DIR = Path(__file__).parent / "bin"


def find_ferric_binary(name: str) -> str | None:
    """Locate a Ferric binary. Returns the path string or None.

    Search order:
    1. ``src/bin/<name>`` (bundled in distribution wheels)
    2. Executable on PATH (development / manual install)
    """
    bundled = _BIN_DIR / name
    if bundled.exists():
        return str(bundled)
    # On Windows, also check with .exe extension
    bundled_exe = _BIN_DIR / f"{name}.exe"
    if bundled_exe.exists():
        return str(bundled_exe)
    return shutil.which(name)


def is_ferric_available(name: str) -> bool:
    """Return True if the named Ferric binary is accessible."""
    return find_ferric_binary(name) is not None
```

**Step 2: Update imports in parser.py and shell.py**

In `src/knowledge/parser.py`, replace the inline `_find_ferric_binary` function with:
```python
from src.ferric import find_ferric_binary as _find_ferric_binary
```

In `src/tools/shell.py`, add at the top:
```python
from src.ferric import find_ferric_binary as _find_ferric_binary
```

**Step 3: Write integration test**

Create `tests/test_ferric_integration.py`:

```python
"""Integration test: build ferric-parse, parse a real file, verify output."""
import subprocess
import json
import shutil
from pathlib import Path
import pytest


FERRIC_PARSE_BINARY = shutil.which("ferric-parse") or str(
    Path(__file__).parent.parent / "target" / "debug" / "ferric-parse"
)


@pytest.mark.skipif(
    not Path(FERRIC_PARSE_BINARY).exists(),
    reason="ferric-parse binary not built (run: cargo build -p ferric-parse)"
)
class TestFerricParseIntegration:
    def test_parse_real_python_file(self):
        """Parse an actual Animus Python file and verify node count."""
        target = Path(__file__).parent.parent / "src" / "tools" / "base.py"
        result = subprocess.run(
            [FERRIC_PARSE_BINARY, str(target)],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "nodes" in data
        # base.py should have at least 3 class nodes
        classes = [n for n in data["nodes"] if n["kind"] == "class"]
        assert len(classes) >= 3, f"Expected ≥3 classes, got {classes}"

    def test_ferric_parser_python_wrapper(self, tmp_path):
        """FerricParser wrapper produces same structure as direct binary call."""
        from src.knowledge.parser import FerricParser
        parser = FerricParser(binary_path=FERRIC_PARSE_BINARY)

        sample = tmp_path / "sample.py"
        sample.write_text(
            "class Foo:\n    def bar(self): pass\n\ndef standalone(): pass\n"
        )
        result = parser.parse_file(sample)

        assert result.file_path == str(sample)
        names = [n.name for n in result.nodes]
        assert "Foo" in names
        assert "standalone" in names
```

**Step 4: Run tests**

```bash
pytest tests/test_ferric_integration.py -v
pytest tests/ -x -q
```

**Step 5: Commit**

```bash
git add src/ferric.py tests/test_ferric_integration.py
git commit -m "feat: extract _find_ferric_binary to src/ferric.py, add integration test"
```

---

## Final Steps

After all tasks complete, run the full test suite:

```bash
cd C:/Users/charl/animus
cargo build --workspace
pytest tests/ -q
```

Expected: All Python tests pass, all Rust crates build.

Then use `superpowers:finishing-a-development-branch` to decide merge strategy.
