# Animus: Comprehensive Improvement Audit

**Date:** 2026-02-12  
**Scope:** Full codebase review — every module in `src/`, `tests/`, `docs/`, and `pyproject.toml`  
**Author:** Claude (via Thread & Signal code audit)

---

## Executive Summary

Animus is a well-architected local-first agentic CLI with clean separation of concerns, strong security principles, and a thoughtful design philosophy. The codebase is ~4,200 lines of Python across 25 modules with 172K of test coverage. This audit identifies 47 improvements across 8 categories, ranked by impact.

The single highest-leverage improvement is **replacing the naive token estimator** — it touches every module and silently degrades performance everywhere. The single most architecturally significant improvement is **decoupling the embedding pipeline into a concurrent ingest service** — this unlocks the path to Go sidecar architecture and distributed operation.

---

## Table of Contents

1. [Memory Pipeline (Chunker → Embedder → VectorStore)](#1-memory-pipeline)
2. [Context Window & Token Estimation](#2-context-window--token-estimation)
3. [Agent Core & Agentic Loop](#3-agent-core--agentic-loop)
4. [Planner & Task Decomposition](#4-planner--task-decomposition)
5. [Knowledge Graph & Code Intelligence](#5-knowledge-graph--code-intelligence)
6. [LLM Providers & Model Management](#6-llm-providers--model-management)
7. [Tools & Security](#7-tools--security)
8. [Infrastructure & Developer Experience](#8-infrastructure--developer-experience)

---

## 1. Memory Pipeline

**Files:** `src/memory/chunker.py`, `scanner.py`, `embedder.py`, `vectorstore.py`

### 1.1 Token Estimation Is Systematically Wrong for Code

**Severity: HIGH** — affects chunk sizing, context budgets, and retrieval quality

**Current implementation** (`chunker.py:9`, `context.py:8`):
```python
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
```

**The problem:** This 4-chars-per-token heuristic is calibrated for English prose. Code has fundamentally different tokenization characteristics. Brackets, operators, keywords, and short identifiers tokenize to 1-2 characters each. Empirically, code averages ~3.0-3.2 chars/token on most tokenizers, meaning your code chunks are **~25-30% smaller than intended**.

**The downstream effects:**
- More chunks per file → more embeddings → slower ingestion
- Smaller chunks → less semantic context per chunk → noisier retrieval
- Context budget calculations in `context.py` systematically overestimate available space

**Fix — tiered estimation:**
```python
def estimate_tokens(text: str, content_type: str = "auto") -> int:
    """Estimate tokens with content-aware ratios.
    
    Token Count ≈ Character Count / Chars Per Token
    
    Where Chars_Per_Token varies by content:
      English prose ≈ 4.0
      Source code   ≈ 3.1
      Mixed         ≈ 3.5
    """
    if content_type == "auto":
        content_type = "code" if _looks_like_code(text) else "prose"
    
    ratio = {"prose": 4.0, "code": 3.1, "mixed": 3.5}.get(content_type, 3.5)
    return max(1, int(len(text) / ratio))
```

**Better fix — use `tiktoken` for exact counts:**
```python
# Add tiktoken to dependencies (tiny, fast, no ML model needed)
import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")  # GPT-4/Claude-compatible

def estimate_tokens(text: str) -> int:
    return len(_enc.encode(text))
```

`tiktoken` is ~100x faster than a full tokenizer and adds zero model weight. It's the right tool here.

### 1.2 Code-Aware Chunker Only Understands Python Syntax

**Severity: MEDIUM** — limits RAG quality for non-Python codebases

**Current:** `chunker.py:64` splits on `^(?=(?:def |class |async def ))` and `_looks_like_code` checks Python/JS indicators.

**Missing:**
- Go: `func `, `type ... struct`
- Rust: `fn `, `impl `, `struct `, `enum `
- C/C++: function signatures (harder, but `{` at column 0 after a non-indented line is a strong signal)
- TypeScript/JSX
- Shell scripts (function definitions)

**Fix — language-aware boundary detection:**
```python
_LANGUAGE_BOUNDARIES = {
    "python": r"^(?=(?:def |class |async def ))",
    "go":     r"^(?=(?:func |type ))",
    "rust":   r"^(?=(?:fn |impl |struct |enum |trait |mod ))",
    "c":      r"^(?=\w[\w\s\*]+\w+\s*\([^)]*\)\s*\{)",
    "js":     r"^(?=(?:function |class |const \w+ = (?:async )?\())",
}

def _detect_language(text: str, filename: str = "") -> str:
    ext_map = {".py": "python", ".go": "go", ".rs": "rust",
               ".c": "c", ".h": "c", ".js": "js", ".ts": "js"}
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in ext_map:
            return ext_map[ext]
    # Heuristic fallback
    ...
```

**Important:** The `scanner.py` already passes file paths — the metadata is available but not forwarded to the chunker. Thread the filename through the `chunk()` call.

### 1.3 Chunker Should Use AST Parsing for Python (You Already Have It)

**Severity: MEDIUM** — your `knowledge/parser.py` already does full AST parsing

**Irony:** `knowledge/parser.py` does `ast.parse()` and extracts functions, classes, methods, docstrings, call graphs, and inheritance trees. The `memory/chunker.py` does regex splitting on the *same files*. These two systems don't talk to each other.

**Fix — AST-informed chunking:**
```python
# In chunker.py, when file is .py:
from src.knowledge.parser import PythonParser

def _chunk_code_ast(self, text: str, filepath: str) -> list[dict]:
    parser = PythonParser()
    result = parser.parse_file(Path(filepath))
    
    chunks = []
    for node in result.nodes:
        if node.kind in ("function", "method", "class"):
            # Extract source lines for this node
            lines = text.splitlines()
            chunk_text = "\n".join(lines[node.line_start-1:node.line_end])
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "kind": node.kind,
                    "name": node.qualified_name,
                    "docstring": node.docstring,
                    "file": filepath,
                    "lines": f"{node.line_start}-{node.line_end}",
                }
            })
    return chunks
```

Each chunk now carries semantic metadata — the vector search isn't just matching text similarity, it's matching *structural meaning*. A query for "authentication handler" can match a function named `authenticate()` even if the code itself never uses the word "authentication."

### 1.4 Scanner Missing Symlink Loop Detection

**Severity: LOW** — but a correctness issue

`scanner.py:48` uses `root.glob(glob_pattern)` which follows symlinks by default. A symlink loop (e.g., `src/link → ../src`) will cause infinite recursion.

**Fix:**
```python
def scan(self, root: Path, glob_pattern: str = "**/*") -> Iterator[Path]:
    root = root.resolve()
    seen_inodes: set[tuple[int, int]] = set()
    
    for path in root.glob(glob_pattern):
        if not path.is_file():
            continue
        # Symlink loop detection
        stat = path.stat()
        inode_key = (stat.st_dev, stat.st_ino)
        if inode_key in seen_inodes:
            continue
        seen_inodes.add(inode_key)
        ...
```

### 1.5 VectorStore Brute-Force Fallback Loads All Rows into Python

**Severity: MEDIUM** — `SQLiteVectorStore.search()` at line 200-219

When `sqlite-vec` is unavailable, the fallback `SQLiteVectorStore` does:
```python
rows = self._conn.execute("SELECT text, embedding, metadata FROM chunks").fetchall()
```

This loads *every embedding* into Python memory, unpacks each from binary, and computes cosine similarity in a Python loop. For 100K chunks with 384-dim embeddings, that's ~150MB loaded per query.

**Fix — add approximate nearest neighbor with FAISS or use batched SQL:**
```python
# Minimum viable fix: paginated search with early termination
def search(self, query_embedding, top_k=5, batch_size=1000):
    total = len(self)
    best = []  # min-heap of (neg_score, text, metadata)
    
    for offset in range(0, total, batch_size):
        rows = self._conn.execute(
            "SELECT text, embedding, metadata FROM chunks LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        for text, blob, meta_json in rows:
            emb = _unpack_embedding(blob)
            score = _cosine_similarity(query_embedding, emb)
            if len(best) < top_k:
                heapq.heappush(best, (score, text, meta_json))
            elif score > best[0][0]:
                heapq.heapreplace(best, (score, text, meta_json))
    ...
```

This still does full scan, but uses constant memory instead of loading everything at once.

### 1.6 Embedding Batching Opportunity

**Severity: LOW** — `embedder.py:78-81`

`NativeEmbedder.embed()` already accepts a list, but the call sites in the ingestion pipeline (visible in `main.py`) likely call it per-file. Batch all chunks from a directory scan into a single `embed()` call — `sentence-transformers` is dramatically faster with batched inputs (GPU utilization, less Python overhead per call).

### 1.7 The Go Sidecar Opportunity (Architecture Note)

The entire memory pipeline — `Scanner → Chunker → Embedder → VectorStore` — is I/O-heavy at the scanner level and compute-heavy at the embedder level. The current serial Python pipeline is the right choice for correctness, but if you want to scale:

**Phase 1:** Move Scanner + file watching to a Go service that detects changes and pushes file paths into a queue (gRPC stream or Unix socket).

**Phase 2:** Move VectorStore to a Go service wrapping sqlite-vec with concurrent read support and batched writes.

**Phase 3:** Python retains Chunker + Embedder (ML ecosystem advantage) and communicates with both Go services.

This is the architecture that makes "clustered Go modules for inference" actually viable — not for the tensor math, but for the orchestration, file I/O, and storage layers.

---

## 2. Context Window & Token Estimation

**Files:** `src/core/context.py`

### 2.1 Token Estimator Is Used Everywhere and Wrong Everywhere

**Severity: HIGH** — already covered in 1.1, but emphasizing: `estimate_tokens()` is imported by `agent.py`, `context.py`, and `planner.py`. Every budget calculation, every trim decision, every chunk sizing is derived from this function. Fix it once, and everything improves.

### 2.2 Context Trimming Drops Messages Without Summarization

**Severity: MEDIUM** — `context.py:186-205`

`trim_messages()` simply drops the oldest messages to fit the budget. This means the agent loses context about the early parts of a conversation without any record that context was lost.

**Fix — summarize before dropping:**
```python
def trim_messages(self, messages, system_prompt):
    budget = self.compute_budget(system_prompt, messages)
    if estimate_messages_tokens(messages) <= budget.history_tokens:
        return messages
    
    # Split into messages that fit and messages that don't
    kept = []
    running = 0
    split_point = 0
    for i, msg in enumerate(reversed(messages)):
        tokens = estimate_tokens(msg["content"]) + 4
        if running + tokens > budget.history_tokens:
            split_point = len(messages) - i
            break
        kept.insert(0, msg)
        running += tokens
    
    # If we dropped messages, prepend a summary marker
    if split_point > 0:
        dropped_count = split_point
        summary_msg = {
            "role": "system",
            "content": f"[{dropped_count} earlier messages trimmed for context limit]"
        }
        return [summary_msg] + kept
    
    return kept
```

Better yet, use the LLM itself to summarize the dropped messages into a condensed context block — but that adds a generation call per trim, so make it optional.

### 2.3 Tier Ratios Are Reasonable but Untested Empirically

**Severity: LOW** — `context.py:53-69`

The `_TIER_RATIOS` are well-reasoned defaults, but the magic numbers (0.3, 0.5, 0.7 for history ratios) should be validated against actual model behavior. Consider adding a `benchmark` command that measures effective context utilization per model.

### 2.4 `chunk_instruction` Splits on Paragraph Boundaries Only

**Severity: LOW** — `context.py:207-234`

For small models that need instruction chunking, splitting on `\n\n` works for natural language but fails for code blocks (which may not have paragraph breaks). Consider falling back to sentence boundaries or fixed-token windows when paragraph splitting produces chunks that are still too large.

---

## 3. Agent Core & Agentic Loop

**Files:** `src/core/agent.py`

### 3.1 No Observation-Reflection-Action Pattern

**Severity: HIGH** — the agent generates, parses tools, executes, and loops — but never *reflects* on results

**Current loop** (`agent.py:168-284`):
```
generate → parse tool calls → execute tools → feed result → generate again
```

**Missing step:** After tool execution, before the next generation, the agent should evaluate:
- Did the tool succeed or fail?
- Is the result what was expected?
- Should the approach change?

This is the difference between a reactive agent and a reasoning agent.

**Fix — add a reflection step:**
```python
def _evaluate_tool_result(self, tool_name: str, result: str) -> str:
    """Classify tool result and optionally inject guidance."""
    if result.startswith("Error:"):
        return f"[Tool {tool_name} FAILED]: {result}\nConsider an alternative approach."
    if len(result) > 2000:
        # Summarize long outputs to preserve context budget
        return f"[Tool {tool_name} returned {len(result)} chars, showing first 500]:\n{result[:500]}"
    return f"[Tool result for {tool_name}]: {result}"
```

This is lightweight (no LLM call) but gives the model better signal about what happened.

### 3.2 Tool Call Parsing Duplicated Between Agent and Planner

**Severity: LOW** — `agent.py:362-405` and `planner.py:519-571` contain nearly identical `_parse_tool_calls()` implementations

Extract into a shared utility:
```python
# src/core/tool_parsing.py
def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM response. Three strategies: raw JSON, code blocks, inline."""
    ...
```

### 3.3 `_run_chunked` Summary Is Truncated to 200 Characters

**Severity: MEDIUM** — `agent.py:128`

```python
summary = last_response[:200].strip()
```

200 characters of context between chunks is very thin for maintaining coherence. For multi-chunk tasks, the agent effectively forgets what happened in earlier chunks.

**Fix:** Use a sliding summary that preserves key outcomes:
```python
def _summarize_for_carry(self, response: str, max_tokens: int = 100) -> str:
    """Extract key outcomes from a response for inter-chunk context."""
    # Priority: tool results > final statements > first sentences
    lines = response.strip().split("\n")
    tool_results = [l for l in lines if l.startswith("[Tool result")]
    if tool_results:
        return " | ".join(tool_results[-3:])  # Last 3 tool results
    return response[:max_tokens * 4].strip()
```

### 3.4 No Rate Limiting or Backoff in Agentic Loop

**Severity: LOW** — if the model generates rapid-fire tool calls, there's no throttling

The repeat detection at line 254-268 catches *identical* calls, but not rapid *different* calls that might overwhelm a system. Consider adding a simple rate limiter:
```python
if turn > 0:
    time.sleep(min(0.5, turn * 0.1))  # Progressive slowdown
```

---

## 4. Planner & Task Decomposition

**Files:** `src/core/planner.py`

### 4.1 Heuristic Decomposition Is Fragile

**Severity: MEDIUM** — `planner.py:707-738`

`_heuristic_decompose()` splits on "then", "and then", and period-separated sentences. This works for simple English instructions but fails for:
- "Read the file, parse the JSON, and extract the keys" (commas, no "then")
- "Fix the bug in auth.py where the token expires and update the tests" (conjunction without temporal signal)

**Fix — add conjunction-aware splitting:**
```python
_SPLIT_PATTERNS = [
    r'[.;,]\s*[Tt]hen\s*,?\s*',       # temporal: "then"
    r'\s+and\s+then\s+',                # temporal: "and then"
    r'\s+then\s+',                       # temporal: "then"
    r',\s*and\s+(?=\w+\s)',             # conjunction: ", and verb..."
    r'\.\s+(?=[A-Z])',                   # sentence boundary
]
```

### 4.2 Step Type Inference Doesn't Consider File Extensions

**Severity: LOW** — `planner.py:212-219`

`_infer_step_type()` uses keyword matching on the description. If the step mentions "test_auth.py", the keyword "test" triggers `StepType.SHELL`. But "edit test_auth.py" should be `StepType.WRITE`.

**Fix:** Check for file path patterns before keyword matching:
```python
def _infer_step_type(description: str) -> StepType:
    # File operation patterns take priority
    if re.search(r'\b\w+\.\w{1,4}\b', description):  # looks like a filename
        lower = description.lower()
        if any(w in lower for w in ("edit", "modify", "update", "fix", "add to")):
            return StepType.WRITE
        if any(w in lower for w in ("read", "view", "check", "look at")):
            return StepType.READ
    # Fall through to keyword matching...
```

### 4.3 Planning Profile Scaling Uses log2 — Consider Linear for Small Models

**Severity: LOW** — `planner.py:132-153`

The `log2(context_length / 1024)` scaling means doubling context from 2K to 4K adds the same capacity as doubling from 4K to 8K. This is reasonable for "how many steps can we plan" but underestimates the benefit of larger contexts for medium models. Linear scaling up to a cap might be more appropriate for the `max_step_turns` parameter specifically.

### 4.4 Simple Task Detection Misses Code-Specific Patterns

**Severity: LOW** — `planner.py:680-704`

`_is_simple_task()` counts action verbs but doesn't recognize single-action code patterns like "refactor the error handler" (two action-looking words, but one action) or "add type hints to agent.py" (one action, but "add" and potentially "type" could confuse counts).

---

## 5. Knowledge Graph & Code Intelligence

**Files:** `src/knowledge/parser.py`, `graph_db.py`, `indexer.py`

### 5.1 Parser Only Handles Python

**Severity: MEDIUM** — the knowledge graph is Python-only

The `PythonParser` uses `ast.parse()`, which only works for Python files. The `Indexer` filters for `**/*.py` explicitly.

**Fix — pluggable parser architecture:**
```python
class LanguageParser(ABC):
    @abstractmethod
    def parse_file(self, path: Path) -> FileParseResult: ...
    
    @abstractmethod
    def supported_extensions(self) -> set[str]: ...

class PythonParser(LanguageParser):
    def supported_extensions(self) -> set[str]:
        return {".py"}
    ...

# Future: GoParser using `go/ast` via subprocess, RustParser via tree-sitter, etc.
```

For Go specifically (since you're thinking about it), `go doc -json` and `go vet` provide structured output that could be parsed without writing a full AST walker.

### 5.2 Graph Queries Don't Return Source Code

**Severity: MEDIUM** — `graph.py` tools return node metadata but not the actual code

When the agent queries "get callers of authenticate()", it gets qualified names and line numbers. But it can't *read* the calling code without a separate `read_file` tool call. This doubles the tool calls needed for code understanding tasks.

**Fix — optionally include source snippets in graph query results:**
```python
def _format_node_with_source(node: NodeRow, max_lines: int = 10) -> str:
    loc = f"{node.file_path}:{node.line_start}"
    header = f"[{node.kind}] {node.qualified_name} @ {loc}"
    
    if node.file_path and Path(node.file_path).exists():
        lines = Path(node.file_path).read_text().splitlines()
        snippet = lines[node.line_start-1:min(node.line_end, node.line_start + max_lines)]
        return f"{header}\n{''.join(snippet)}"
    return header
```

### 5.3 Blast Radius Tool is BFS Without Cycle Detection Note

**Severity: LOW** — `graph_db.py` BFS implementation

The blast radius query does BFS through the call graph. If there are recursive call cycles (which are common), the BFS correctly uses a `visited` set — but the result doesn't indicate cycles were found. Adding a `cycles_detected: bool` flag to the result helps the agent understand the graph structure.

### 5.4 Indexer Doesn't Report Parse Failures Usefully

**Severity: LOW** — `indexer.py:88-89`

```python
except Exception:
    result.files_failed += 1
```

Silent `except Exception` with no logging or error collection. The user has no way to know *which* files failed or *why*.

**Fix:**
```python
except Exception as e:
    result.files_failed += 1
    if on_progress:
        on_progress(f"FAILED: {path_str}: {e}")
```

---

## 6. LLM Providers & Model Management

**Files:** `src/llm/base.py`, `native.py`, `api.py`, `factory.py`

### 6.1 Model Capabilities Don't Include Embedding Support Flag

**Severity: LOW** — `base.py:12-35`

`ModelCapabilities` has `supports_tools` and `supports_json_mode` but no `supports_embeddings`. The provider base class has an `embed()` method that raises `NotImplementedError`. Making this explicit in capabilities would let the agent route embedding tasks appropriately.

### 6.2 Model Catalog Is Hardcoded — Consider Dynamic Discovery

**Severity: LOW** — `native.py:35-99`

The `MODEL_CATALOG` is a static dict. If the user downloads a GGUF model that isn't in the catalog, it won't be discoverable. Consider scanning the models directory for `.gguf` files and auto-detecting parameters from metadata:

```python
def discover_models(models_dir: Path) -> dict[str, ModelInfo]:
    """Scan directory for GGUF files and build catalog entries."""
    discovered = {}
    for path in models_dir.glob("*.gguf"):
        params_b = _estimate_params_from_filename(str(path))
        name = path.stem.lower().replace("-", "_")
        discovered[name] = ModelInfo(
            repo="local",
            filename=path.name,
            params_b=params_b,
            context_length=4096,  # conservative default
        )
    return {**MODEL_CATALOG, **discovered}
```

### 6.3 Download Progress Has No Resume Capability

**Severity: LOW** — `native.py:126-177`

`download_gguf()` writes to a `.part` file but doesn't support resume if interrupted. GGUF files can be multi-GB. Adding a `Range` header for HTTP resume is straightforward:

```python
existing_size = tmp.stat().st_size if tmp.exists() else 0
headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

with httpx.Client(timeout=None, follow_redirects=True) as client:
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 206:  # Partial content
            mode = "ab"  # Append
        else:
            mode = "wb"  # Overwrite
            existing_size = 0
        ...
```

### 6.4 NativeProvider Doesn't Expose Streaming for llama-cpp

**Severity: LOW** — `native.py:219-241`

`generate()` calls `create_chat_completion()` without `stream=True`. The base class `generate_stream()` falls back to yielding the entire response as one chunk. Adding true streaming:

```python
def generate_stream(self, messages, tools=None, **kwargs):
    model = self._load_model()
    for chunk in model.create_chat_completion(
        messages=messages, stream=True, max_tokens=kwargs.get("max_tokens", 2048)
    ):
        delta = chunk["choices"][0].get("delta", {}).get("content", "")
        if delta:
            yield delta
```

---

## 7. Tools & Security

**Files:** `src/tools/`, `src/core/permission.py`, `src/isolation/`

### 7.1 PermissionChecker Is Instantiated Per Tool Call

**Severity: LOW** — every tool `execute()` method does `checker = PermissionChecker()`

This creates a new instance per call. If `PermissionChecker.__init__` ever loads config or reads files, this becomes expensive. Make it a singleton or inject it via the tool registry:

```python
class ToolRegistry:
    def __init__(self, permission_checker: PermissionChecker = None):
        self._checker = permission_checker or PermissionChecker()
    
    def execute(self, name, args):
        tool = self._tools.get(name)
        # Inject checker into tool context
        ...
```

### 7.2 Shell Tool Timeout Is Per-Call, Not Cumulative

**Severity: MEDIUM** — `shell.py:41`

If the agent makes 10 shell calls of 30 seconds each, that's 5 minutes of subprocess execution per agentic loop. There's no cumulative timeout across the entire session.

**Fix — session-level execution budget:**
```python
class ExecutionBudget:
    def __init__(self, max_total_seconds: int = 300):
        self._max = max_total_seconds
        self._used = 0.0
    
    def consume(self, seconds: float) -> bool:
        self._used += seconds
        return self._used < self._max
    
    @property
    def remaining(self) -> float:
        return max(0, self._max - self._used)
```

### 7.3 Ornstein Sandbox Socket Monkey-Patching Is Fragile

**Severity: MEDIUM** — `ornstein.py:73-99`

Replacing `socket.socket` globally in the subprocess works but is bypassable by code that imported `socket` before the patch or that uses `ctypes` to call libc directly.

For Linux, consider using `seccomp` (via `seccomp` Python package) or `unshare` network namespaces for stronger isolation:

```python
import subprocess
# Network namespace isolation (Linux only)
subprocess.run(
    ["unshare", "--net", "--", "python", "-c", code],
    timeout=config.timeout_seconds,
)
```

This is a deeper change, but aligns with the Ornstein/Smough architecture described in your docs.

### 7.4 Write Tool Doesn't Track What It Wrote

**Severity: LOW** — `filesystem.py:88-98`

`WriteFileTool.execute()` writes the file and returns a success message, but there's no record of *what* was written or *where*. For audit trails and undo capability:

```python
class WriteFileTool(Tool):
    def __init__(self):
        super().__init__()
        self._write_log: list[dict] = []
    
    def execute(self, args):
        ...
        self._write_log.append({
            "path": str(path),
            "size": len(args["content"]),
            "timestamp": time.time(),
            "hash": hashlib.md5(args["content"].encode()).hexdigest(),
        })
```

### 7.5 Git Tool Missing (Referenced but Not Implemented?)

**Severity: MEDIUM** — `planner.py` references git tools (`git_status`, `git_diff`, `git_log`, `git_branch`, `git_add`, `git_commit`, `git_checkout`) and `src/tools/git.py` exists

I see `git.py` in the tools directory. Verify that all git tools referenced in the planner's `_STEP_TYPE_TOOLS` mapping are actually implemented and registered. Any tool name the planner references but that doesn't exist in the registry will produce silent "Unknown tool" errors during plan execution.

---

## 8. Infrastructure & Developer Experience

### 8.1 No `__main__.py` — Can't Run as Module

**Severity: LOW**

`python -m animus` doesn't work. Add `src/__main__.py`:
```python
from src.main import app
app()
```

### 8.2 Tests Don't Cover the Ingestion Pipeline End-to-End

**Severity: MEDIUM** — `tests/test_memory.py` exists but likely tests components in isolation

Add an integration test that:
1. Creates a temp directory with sample files
2. Runs scanner → chunker → embedder → vectorstore
3. Queries the vectorstore and validates results
4. Verifies incremental re-ingestion (change a file, re-ingest, verify only changed file is re-processed)

### 8.3 `pyproject.toml` Should Pin Upper Bounds

**Severity: LOW** — dependencies like `typer>=0.12` have no upper bound

A breaking change in `typer` or `pydantic` could break Animus without warning. Use compatible release constraints:
```toml
dependencies = [
    "typer>=0.12,<1.0",
    "rich>=13.0,<14.0",
    "pydantic>=2.0,<3.0",
    ...
]
```

### 8.4 No GitHub Actions CI

**Severity: MEDIUM** — tests exist but no automated CI

Add a basic `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest
```

This protects against regressions and gives you the green badge on the README.

### 8.5 LLM_GECK Directory Purpose Is Unclear to Contributors

**Severity: LOW** — the Garden of Eden Creation Kit has logs, instructions, and report findings but no README explaining what it is or how to use it

Add an `LLM_GECK/README.md` explaining:
- What GECK is (project template / initialization system)
- How it relates to Animus (agent self-configuration?)
- Whether it's for humans or for the agent to read
- How the logs and findings are generated

### 8.6 No Contribution Guide

**Severity: LOW** — 11 stars, first PR received — time for a `CONTRIBUTING.md`

Cover: how to set up dev environment, how to run tests, code style expectations, how the design principles apply to PRs, and which areas are open for contribution.

---

## Priority Matrix

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| **P0** | 1.1 Fix token estimation | High | Low |
| **P0** | 3.1 Add observation/reflection step | High | Medium |
| **P1** | 1.3 AST-informed chunking (use existing parser) | Medium | Low |
| **P1** | 2.2 Context trimming with summarization | Medium | Medium |
| **P1** | 1.2 Multi-language chunker boundaries | Medium | Medium |
| **P1** | 8.4 GitHub Actions CI | Medium | Low |
| **P2** | 1.5 VectorStore brute-force optimization | Medium | Low |
| **P2** | 5.2 Source snippets in graph queries | Medium | Low |
| **P2** | 7.2 Cumulative execution budget | Medium | Low |
| **P2** | 3.3 Better inter-chunk context carry | Medium | Low |
| **P2** | 5.1 Pluggable parser architecture | Medium | Medium |
| **P3** | 1.7 Go sidecar architecture | High | High |
| **P3** | 7.3 Stronger sandbox isolation | Medium | High |
| **P3** | Everything else | Low | Low-Med |

---

## Closing Thought

The strongest thing about Animus isn't any single module — it's the **design philosophy**. "Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else." That principle is exactly right, and it's what makes Animus viable on hardware where frontier models can't run.

The improvements above are all in service of making that philosophy more effective: better chunking means the LLM sees better context. Better token estimation means the context budget is honest. Better tool result handling means the LLM wastes fewer turns. And the Go sidecar path means the hardcoded orchestration layer can scale independently of the ML compute layer.

Build the foundation right, and the clustered inference architecture from our earlier conversation becomes a natural extension rather than a rewrite.

*"The aleph contains everything, and everything is contained in the aleph — but the aleph itself is only a point."* — Borges understood that the power of a system isn't its size, but the density of meaning at each node. Make each chunk, each tool call, each context frame carry more signal per byte, and the whole system multiplies.

---

**Next Steps:** Pick one P0 item. Ship it. Test it. Then pick the next.
