# Tasks — ANIMUS

**Last Updated:** 2026-01-26

## Legend

- `[ ]` — Not started
- `[x]` — Complete
- `[~]` — In progress
- `[BLOCKED: reason]` — Cannot proceed
- `[DECISION: topic]` — Awaiting human input

---

## Completed Phases (1-7)

### Phase 1: The Core Shell (Skeleton) ✓
- [x] Initialize project structure (src/, tests/, pyproject.toml)
- [x] Initialize a Typer app structure
- [x] Implement `animus detect` command (OS/Hardware detection)
- [x] Create Configuration Manager (~/.animus/config.yaml)
- [x] Implement `animus config` command
- [x] Implement `animus init` command
- [x] Add basic tests for detection module
- [x] All paths use pathlib for Windows compatibility

### Phase 2: The "Brain" Socket (Model Layer) ✓
- [x] Create ModelProvider abstract base class
- [x] Implement OllamaProvider (connects to localhost:11434)
- [x] Implement TRTLLMProvider (Jetson-specific placeholder)
- [x] Implement APIProvider (OpenAI-compatible HTTP requests)
- [x] Implement `animus pull <model>` command
- [x] Implement `animus models` command
- [x] Implement `animus status` command
- [x] Add provider factory for unified creation
- [x] Add LLM module tests

### Phase 3: The Librarian (RAG & Ingestion) ✓
- [x] Implement `animus ingest <path>` command
- [x] Implement `animus search <query>` command
- [x] Scanner: Walk directory respecting .gitignore
- [x] Router/Extractor: Identify and extract from file types
- [x] Chunker: Token, Sentence, and Code chunking strategies
- [x] Embedder: Ollama, API, and Mock embedders
- [x] VectorStore: InMemory and ChromaDB support
- [x] Rich progress bar during ingestion
- [x] Add memory module tests (21 new tests)

### Phase 4: The Agentic Loop (Reasoning Engine) ✓
- [x] Implement Agent class
- [x] Tool: read_file(path)
- [x] Tool: write_file(path, content)
- [x] Tool: list_dir(path)
- [x] Tool: run_shell(command) with confirmation
- [x] Tool registry and OpenAI schema generation
- [x] Human-in-the-loop confirmation for destructive operations
- [x] Blocked command detection
- [x] Implement `animus chat` command
- [x] Add tools module tests (20 new tests)

### Phase 5: The Hive (Sub-Agent Orchestration) ✓
- [x] Implement SubAgentOrchestrator
- [x] Implement SubAgentScope for restrictions
- [x] Implement SubAgentRole with specialized prompts
- [x] Implement ScopedToolRegistry for tool filtering
- [x] Sub-agent scope restriction (paths, tools)
- [x] Sub-agent reporting mechanism
- [x] Parallel sub-agent execution
- [x] Add sub-agent tests (12 new tests)

### Phase 6: Native Model Loading (Self-Contained Inference) ✓
- [x] Implement NativeProvider using llama-cpp-python
- [x] Support GGUF model format loading
- [x] Implement `animus model download <model>` command
- [x] Implement `animus model list` command
- [x] Add model storage in ~/.animus/models/
- [x] Implement GPU backend detection (CUDA, Metal, ROCm)
- [x] Implement CPU fallback for systems without GPU
- [x] Add automatic quantization format detection (Q4_K_M, Q5_K_M, Q8_0, etc.)
- [x] Implement provider fallback chain (Native → Ollama → API)
- [x] Update config to support native provider settings
- [x] Add native provider tests (28 new tests)
- [x] Implement NativeEmbedder using sentence-transformers
- [x] Update embedder to auto-detect best available (native → mock)
- [x] Animus runs without Ollama service dependency

### Phase 7: Agent Autonomy & UX Improvements ✓
- [x] Fix Windows 11 detection (build >= 22000)
- [x] Update system prompt with explicit tool call format (JSON)
- [x] Add autonomous execution policy to system prompt
- [x] Create `auto_execute_tools` configuration for read-only tools
- [x] Create `safe_shell_commands` configuration for safe commands
- [x] Update `_call_tool` to auto-execute safe operations
- [x] Improve `_parse_tool_calls` to handle multiple formats (JSON, function-style, command-style)
- [x] Add stopping cadence configuration to config.yaml (`AgentBehaviorConfig`)
- [x] Implement path change detection and confirmation
- [x] Add blocked command detection
- [ ] Integration test with actual LLM to verify tool execution

---

## Current Sprint

### Phase 8: Self-Improvement & Observability (from Hive analysis)

**Goal:** Enable agent to learn from failures and improve automatically.

**Inspiration:** Hive's decision recording and BuilderQuery patterns.

**Tasks:**
- [ ] **Decision Recording Schema** (`src/core/decision.py`)
  - [ ] Create `Decision` dataclass (intent, options, chosen, reasoning)
  - [ ] Create `Option` dataclass (id, description, pros, cons)
  - [ ] Create `Outcome` dataclass (decision_id, success, result, summary)
  - [ ] Add decision recording to Agent class
- [ ] **Run Persistence** (`src/core/run.py`)
  - [ ] Create `Run` dataclass (id, goal, decisions, metrics, status)
  - [ ] Create `RunMetrics` (tokens_used, latency, success_rate)
  - [ ] Implement JSON-based run storage (~/.animus/runs/)
  - [ ] Add run indexing by goal, status, date
- [ ] **BuilderQuery Interface** (`src/core/builder.py`)
  - [ ] Analyze runs for patterns and failures
  - [ ] Generate improvement suggestions
  - [ ] Implement `animus analyze <goal>` command
- [ ] **Triangulated Verification** (`src/core/judge.py`)
  - [ ] Implement rule-based checks (fast, deterministic)
  - [ ] Implement LLM fallback evaluation (flexible, contextual)
  - [ ] Implement human escalation protocol
  - [ ] Create `HybridJudge` class combining all three

### Phase 9: Context Resilience (from Clawdbot analysis)

**Goal:** Handle long conversations without context overflow.

**Inspiration:** Clawdbot's session compaction and error classification.

**Tasks:**
- [ ] **Session Compaction** (`src/core/compaction.py`)
  - [ ] Implement conversation summarization
  - [ ] Auto-trigger when approaching context limit
  - [ ] Preserve recent turns + summary of older turns
  - [ ] Add compaction to Agent class
- [x] **Error Classification** (`src/core/errors.py`) ✓
  - [x] Define error categories: `context_overflow`, `auth_failure`, `rate_limit`, `timeout`, `tool_failure`
  - [x] Implement error-specific recovery strategies
  - [x] Add retry logic with exponential backoff
  - [x] Create error event logging
- [ ] **Context Window Management**
  - [ ] Track token usage per turn
  - [ ] Warn before overflow (soft limit)
  - [ ] Auto-compact on overflow (hard limit)
  - [ ] Add `--max-context` CLI option

### Phase 10: Enhanced Retrieval (from both analyses)

**Goal:** Improve RAG with hybrid search combining keyword and semantic.

**Inspiration:** Clawdbot's SQLite-vec hybrid search, Hive's decision-aware retrieval.

**Tasks:**
- [ ] **Hybrid Search** (`src/memory/hybrid.py`)
  - [ ] Implement BM25 keyword search
  - [ ] Combine with existing vector search
  - [ ] Configurable weighting (keyword vs semantic)
  - [ ] Score normalization and result merging
- [ ] **SQLite-vec Backend** (`src/memory/sqlite_vec.py`)
  - [ ] Replace InMemoryVectorStore with SQLite-vec
  - [ ] Persistent storage (~/.animus/vectordb.sqlite)
  - [ ] Atomic batch writes
  - [ ] Index maintenance and compaction
- [ ] **Improved Chunking**
  - [ ] Add Tree-sitter for AST-aware code chunking
  - [ ] Markdown-aware chunking (preserve headers, lists)
  - [ ] Configurable overlap strategies

---

## Backlog (Prioritized)

### High Priority (Near-term)

- [ ] **Auth Profile Rotation** (from Clawdbot)
  - Multiple API keys with cooldown tracking
  - Automatic failover on auth failures
  - Per-profile usage metrics

- [ ] **Lane-Based Queueing** (from Clawdbot)
  - Serialize commands per session
  - Prevent interleaving of concurrent runs
  - Priority queue support

- [ ] **Media Pipeline** (from Clawdbot)
  - File download with size limits
  - MIME detection
  - TTL-based cleanup

### Medium Priority (Future sprints)

- [ ] **Skills Platform** (from Clawdbot)
  - Markdown-based skills with YAML frontmatter
  - Install specs (brew, pip, npm, etc.)
  - Eligibility checks (OS, binaries, env vars)
  - `animus skill install/list/run` commands

- [ ] **Safe Code Sandbox** (from Hive)
  - Whitelist-based `safe_eval()` and `safe_exec()`
  - Timeout and memory limits
  - No access to dangerous modules

- [ ] **MCP Server** (from Hive)
  - Expose tools via Model Context Protocol
  - Enable cross-tool communication
  - `animus mcp-server` command

### Lower Priority (Roadmap)

- [ ] **Browser Control** (from Clawdbot)
  - Playwright/CDP integration
  - Multi-profile support
  - Screenshot and ARIA tree tools

- [ ] **Multi-Channel Support** (from Clawdbot)
  - WhatsApp, Telegram, Slack plugins
  - Channel-specific message formatting
  - DM pairing and group rules

- [ ] **Goal-Driven Development** (from Hive)
  - Generate agent graphs from natural language
  - Node-based workflow architecture
  - Dynamic edge routing

- [ ] **Canvas/A2UI** (from Clawdbot)
  - Visual UI rendering
  - Real-time push/reset operations

### Existing Backlog Items

- [ ] Add ZIM archive support for Wikipedia offline ingestion
- [ ] Implement conversation memory persistence
- [ ] Add Tree-sitter parsing for code chunking (moved to Phase 10)

---

## Success Criteria

### Core Functionality (Achieved)
- [x] All commands execute without errors
- [x] Help text is accurate and complete
- [x] Exit codes are correct
- [x] Input validation works properly
- [x] Error messages are clear and actionable
- [x] LLM Agent can execute commands via the terminal
- [x] LLM Agent can create code in various languages
- [x] LLM Agent can create specialized sub-agents
- [x] Animus can load and run GGUF models directly without Ollama
- [x] Animus can download models from Hugging Face
- [x] Native inference works on CPU, CUDA, and Metal backends
- [x] Native embeddings work without Ollama (sentence-transformers)
- [x] Windows 11 correctly identified (not Windows 10)

### Autonomy (In Progress)
- [ ] Agent executes tools autonomously (doesn't ask user to run commands) — needs LLM integration test
- [x] Proper stopping cadences for file creation/modification/deletion
- [x] Path change detection and confirmation
- [x] Blocked command detection

### Self-Improvement (Phase 8)
- [ ] Decisions recorded with reasoning, not just actions
- [ ] Runs persisted with metrics for analysis
- [ ] BuilderQuery can suggest improvements
- [ ] Triangulated verification for output validation

### Resilience (Phase 9)
- [ ] Session compaction prevents context overflow
- [x] Error classification enables appropriate recovery
- [x] Retry logic with exponential backoff

### Retrieval (Phase 10)
- [ ] Hybrid search (BM25 + vector) for better RAG
- [ ] SQLite-vec persistent storage
- [ ] Tree-sitter AST-aware code chunking

---

## Completed (Recent)

- Phase 7: Agent Autonomy fixes (Windows 11 detection, auto-execute, tool parsing)
- Phase 6: Native Model Loading (GGUF support, native embeddings, Ollama-free)
- Phase 5: Sub-Agent Orchestration (roles, scopes, parallel execution)
- Phase 4: Agentic Loop (Agent class, tools, chat command)
- Phase 3: RAG & Ingestion (scanner, chunker, extractor, embedder, vectorstore)
- Phase 2: Model Layer (providers, commands, factory)
- Phase 1: Core implementation (CLI, detect, config, init)

---

## Reference: Source Repositories Analyzed

| Repository | Path | Key Patterns Borrowed |
|------------|------|----------------------|
| **Hive (Aden)** | `C:\Users\charl\hive` | Decision recording, BuilderQuery, Triangulated verification, HybridJudge |
| **Clawdbot** | `C:\Users\charl\clawdbot` | Session compaction, Hybrid search, Error classification, Auth rotation, Skills platform |
