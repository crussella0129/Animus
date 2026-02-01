# Tasks — ANIMUS

**Last Updated:** 2026-02-01 (External repository analysis complete, Phases 12-15 defined)

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
- [x] Integration test with actual LLM to verify tool execution (Tested: 2026-01-27)
  - ✅ Tool execution works when model outputs JSON format
  - ✅ Confirmation prompts appear correctly
  - ✅ write_file, list_dir tools functional
  - ⚠️ Model compliance varies - some models output text instead of JSON
  - ⚠️ Model identity/safety issues are model-dependent, not Animus code

---

## Current Sprint

### Phase 8: Self-Improvement & Observability (from Hive analysis)

**Goal:** Enable agent to learn from failures and improve automatically.

**Inspiration:** Hive's decision recording and BuilderQuery patterns.

**Tasks:**
- [x] **Decision Recording Schema** (`src/core/decision.py`) ✓
  - [x] Create `Decision` dataclass (intent, options, chosen, reasoning)
  - [x] Create `Option` dataclass (id, description, pros, cons)
  - [x] Create `Outcome` dataclass (decision_id, success, result, summary)
  - [x] Add decision recording to Agent class
  - [x] Add DecisionRecorder class for session management
  - [x] Add tests (27 new tests in test_decision.py)
- [x] **Run Persistence** (`src/core/run.py`) ✓
  - [x] Create `Run` dataclass (id, goal, decisions, metrics, status)
  - [x] Create `RunMetrics` (tokens_used, latency, success_rate)
  - [x] Implement JSON-based run storage (~/.animus/runs/)
  - [x] Add run indexing by goal, status, date
  - [x] Add RunStore class with find/filter methods
  - [x] Add tests (17 new tests in test_run.py)
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
- [x] **Session Compaction** (`src/core/compaction.py`) ✓
  - [x] Implement conversation summarization
  - [x] Auto-trigger when approaching context limit
  - [x] Preserve recent turns + summary of older turns
  - [x] Multiple strategies: truncate, sliding, summarize, hybrid
  - [ ] Add compaction to Agent class (integration pending)
- [x] **Error Classification** (`src/core/errors.py`) ✓
  - [x] Define error categories: `context_overflow`, `auth_failure`, `rate_limit`, `timeout`, `tool_failure`
  - [x] Implement error-specific recovery strategies
  - [x] Add retry logic with exponential backoff
  - [x] Create error event logging
- [x] **Context Window Management** (`src/core/context.py`) ✓
  - [x] Track token usage per turn
  - [x] Warn before overflow (soft limit)
  - [x] Auto-compact on overflow (hard limit)
  - [x] Add `--max-context` CLI option
  - [x] Add `--show-tokens` CLI option
  - [x] Add TokenEstimator for character-based token estimation
  - [x] Add tests (22 new tests in test_context.py)

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

### Phase 11: Sub-Agent Architecture Improvements (from Hive building-agents skills)

**Goal:** Transform sub-agents from simple role-based prompts to goal-driven workflow agents.

**Inspiration:** Hive's building-agents-core, building-agents-construction, building-agents-patterns skills.

**Tasks:**
- [ ] **SubAgentGoal Class** (`src/subagents/goal.py`)
  - [ ] Create `SubAgentGoal` dataclass (id, name, description)
  - [ ] Create `SuccessCriterion` (id, description, metric, target, weight)
  - [ ] Create `Constraint` (id, description, constraint_type, category)
  - [ ] Success criteria weights should sum to 1.0
  - [ ] Constraint types: hard (must satisfy), soft (prefer to satisfy)
- [ ] **SubAgentNode Class** (`src/subagents/node.py`)
  - [ ] Create `SubAgentNode` dataclass (id, name, node_type, input_keys, output_keys)
  - [ ] Node types: `llm_generate`, `llm_tool_use`, `router`, `function`
  - [ ] System prompt with input key interpolation
  - [ ] Tools list (only for `llm_tool_use` nodes)
  - [ ] Input/output schema validation (optional)
  - [ ] Max retries configuration
- [ ] **SubAgentEdge Class** (`src/subagents/edge.py`)
  - [ ] Create `SubAgentEdge` dataclass (id, source, target, condition, priority)
  - [ ] Edge conditions: `on_success`, `on_failure`, `always`, `conditional`
  - [ ] Conditional expressions for routing decisions
  - [ ] Priority for edge ordering when multiple match
- [ ] **SubAgentGraph Class** (`src/subagents/graph.py`)
  - [ ] Create `SubAgentGraph` dataclass (id, goal, nodes, edges, entry_node)
  - [ ] Entry points dict: `{"start": "first-node-id"}`
  - [ ] Terminal nodes list (where execution ends)
  - [ ] Pause nodes list (where execution waits for user input)
  - [ ] Graph validation (all edges reference valid nodes, entry node exists)
- [ ] **SubAgentExecutor** (`src/subagents/executor.py`)
  - [ ] Execute graph from entry point to terminal/pause
  - [ ] Context propagation between nodes (output → next input)
  - [ ] Parallel edge execution when multiple edges match
  - [ ] Retry logic per node with exponential backoff
  - [ ] Execution result with success, steps_executed, output, error
- [ ] **Pause/Resume Support** (`src/subagents/session.py`)
  - [ ] Session state persistence (memory, paused_at, context)
  - [ ] Resume entry points: `{pause_node}_resume` → next node
  - [ ] Session state passed separately from input_data on resume
  - [ ] Storage in `~/.animus/sessions/`
- [ ] **OutputCleaner Integration** (`src/subagents/cleaner.py`)
  - [ ] Validate node output matches next node's input schema
  - [ ] Detect JSON parsing trap (entire response in one key)
  - [ ] Auto-clean malformed output using fast LLM
  - [ ] Log cleaning events for debugging
- [ ] **Tool Discovery & Validation**
  - [ ] Verify tools exist before adding to nodes
  - [ ] Never assume tool names, always discover dynamically
  - [ ] Inform user if requested tool unavailable
- [ ] **Update SubAgentOrchestrator** (`src/subagents/orchestrator.py`)
  - [ ] Accept SubAgentGraph instead of just SubAgentRole
  - [ ] Use SubAgentExecutor for graph-based sub-agents
  - [ ] Backward compatibility with role-based sub-agents
- [ ] **Tests** (`tests/test_subagent_graph.py`)
  - [ ] Test goal validation (criteria weights sum to 1.0)
  - [ ] Test node type behavior (llm_generate vs llm_tool_use)
  - [ ] Test edge conditions (on_success, on_failure, conditional)
  - [ ] Test graph execution flow
  - [ ] Test pause/resume with session state
  - [ ] Test output cleaning

### Phase 12: MCP Integration (from external repo analysis)

**Goal:** Enable Animus to expose tools via MCP and connect to external MCP servers.

**Inspiration:** OpenCode, BrowserOS, Metorial, Claude Code

**Tasks:**
- [ ] **MCP Server Implementation** (`src/mcp/server.py`)
  - [ ] Expose Animus tools via Model Context Protocol
  - [ ] Support stdio and HTTP transports
  - [ ] Implement tool listing (ToolListChangedNotification)
  - [ ] Add `animus mcp-server` command
  - [ ] Add authentication (API key, OAuth support)
- [ ] **MCP Client Implementation** (`src/mcp/client.py`)
  - [ ] Connect to external MCP servers
  - [ ] Auto-discover tools from connected servers
  - [ ] Convert MCP tools to Animus tool format
  - [ ] Handle OAuth flows for authenticated servers
  - [ ] Add connection health monitoring
- [ ] **MCP Configuration** (`src/mcp/config.py`)
  - [ ] Define MCP server configs in ~/.animus/config.yaml
  - [ ] Support multiple server connections
  - [ ] Per-server tool allowlists
- [ ] **Browser Control via MCP** (optional)
  - [ ] Connect to BrowserOS MCP server
  - [ ] Expose 31 browser control tools
  - [ ] Enable web research without built-in browser

### Phase 13: Skills System (from Anthropic skills repo)

**Goal:** Enable modular capability extension via SKILL.md format.

**Inspiration:** anthropics/skills, charmbracelet/crush (agentskills.io)

**Tasks:**
- [ ] **SKILL.md Parser** (`src/skills/parser.py`)
  - [ ] Parse YAML frontmatter (name, description)
  - [ ] Extract markdown instructions
  - [ ] Validate skill structure
- [ ] **Skill Registry** (`src/skills/registry.py`)
  - [ ] Load skills from ~/.animus/skills/
  - [ ] Load skills from project ./skills/
  - [ ] Priority: project > user > bundled
  - [ ] Dynamic skill discovery
- [ ] **Skill Loader** (`src/skills/loader.py`)
  - [ ] Inject skill instructions into agent context
  - [ ] Support optional scripts (Python, Bash)
  - [ ] Handle skill dependencies
- [ ] **CLI Commands**
  - [ ] `animus skill list` — List available skills
  - [ ] `animus skill install <url>` — Install from URL/GitHub
  - [ ] `animus skill create <name>` — Create from template
  - [ ] `animus skill run <name>` — Execute skill directly
- [ ] **Bundled Skills**
  - [ ] `code-review` — Analyze code for issues
  - [ ] `test-gen` — Generate unit tests
  - [ ] `refactor` — Suggest refactoring improvements
  - [ ] `explain` — Explain code behavior
  - [ ] `commit` — Create well-formatted commits

### Phase 14: Enhanced Permission System (from OpenCode analysis)

**Goal:** Replace binary confirmation with three-tier allow/deny/ask system.

**Inspiration:** OpenCode permission/next.ts, OpenClaw tool policies

**Tasks:**
- [ ] **Permission Model** (`src/core/permission.py`)
  - [ ] Three actions: "allow", "deny", "ask"
  - [ ] Pattern-based matching (glob patterns for files)
  - [ ] Permission categories: read, edit, bash, external_directory
  - [ ] Merge strategy: defaults → user → agent
- [ ] **Permission Configuration**
  - [ ] Define in ~/.animus/config.yaml
  - [ ] Per-agent permission profiles
  - [ ] Project-level .animus/permissions.yaml
- [ ] **Permission Evaluation** (`src/core/permission.py`)
  - [ ] Evaluate before tool execution
  - [ ] Pattern matching for file paths
  - [ ] Command parsing for bash permissions
- [ ] **Permission Prompts**
  - [ ] Rich prompt with context (what, why, patterns)
  - [ ] "Always allow" option for pattern
  - [ ] "Deny all" option for session
- [ ] **Default Profiles**
  - [ ] `strict` — Ask for everything except reads
  - [ ] `standard` — Auto-allow reads, ask edits/bash
  - [ ] `trusted` — Auto-allow most, ask dangerous
  - [ ] `yolo` — Allow everything (use with caution)
- [ ] **Agent-Specific Permissions**
  - [ ] Explore agent: read-only (deny edits)
  - [ ] Plan agent: read-only, ask bash
  - [ ] Build agent: standard permissions

### Phase 15: OpenAI-Compatible Local API (from Jan, Lemonade)

**Goal:** Serve Animus capabilities via OpenAI-compatible API for ecosystem integration.

**Inspiration:** Jan (localhost:1337), Lemonade, OpenCode server

**Tasks:**
- [ ] **API Server** (`src/api/server.py`)
  - [ ] HTTP server on localhost:8337 (configurable)
  - [ ] OpenAI-compatible `/v1/chat/completions` endpoint
  - [ ] Streaming support (SSE)
  - [ ] Authentication (API key header)
- [ ] **Model Endpoint** (`src/api/routes/models.py`)
  - [ ] `/v1/models` — List available models
  - [ ] Return native, Ollama, and API models
- [ ] **Chat Completions** (`src/api/routes/chat.py`)
  - [ ] `/v1/chat/completions` — Chat with model
  - [ ] Support tools/functions parameter
  - [ ] Stream responses with `stream: true`
  - [ ] Include tool execution results
- [ ] **Embeddings Endpoint** (`src/api/routes/embeddings.py`)
  - [ ] `/v1/embeddings` — Generate embeddings
  - [ ] Use native embedder
- [ ] **Agent Endpoint** (Animus-specific)
  - [ ] `/v1/agent/chat` — Chat with Animus agent (tools enabled)
  - [ ] `/v1/agent/ingest` — Trigger ingestion
  - [ ] `/v1/agent/search` — Search knowledge base
- [ ] **CLI Command**
  - [ ] `animus serve` — Start API server
  - [ ] `--port` option (default 8337)
  - [ ] `--host` option (default localhost)
  - [ ] Background mode with `--daemon`
- [ ] **Integration Testing**
  - [ ] Test with VS Code Continue extension
  - [ ] Test with Open WebUI
  - [ ] Test with n8n automation

---

## Backlog (Prioritized)

### High Priority (Near-term)

- [ ] **Auth Profile Rotation** (from Clawdbot)
  - Multiple API keys with cooldown tracking
  - Automatic failover on auth failures
  - Per-profile usage metrics

- [ ] **Lane-Based Queueing** (from Clawdbot, OpenClaw)
  - Serialize commands per session
  - Prevent interleaving of concurrent runs
  - Priority queue support
  - Pause/resume with session state

- [ ] **Media Pipeline** (from Clawdbot)
  - File download with size limits
  - MIME detection
  - TTL-based cleanup

- [ ] **Knowledge Graph for Code** (from Potpie)
  - Neo4j or SQLite-based code graph
  - Track function/class relationships
  - Enable "blast radius" analysis
  - Semantic code navigation

- [ ] **Specialist Sub-Agents** (from OpenCode, Potpie)
  - Explore agent (fast, read-only search)
  - Plan agent (analysis without edits)
  - Debug agent (stacktrace analysis)
  - Test agent (unit test generation)

### Medium Priority (Future sprints)

- [ ] **Skills Platform** (from Clawdbot) — **MOVED TO PHASE 13**

- [ ] **Safe Code Sandbox** (from Hive)
  - Whitelist-based `safe_eval()` and `safe_exec()`
  - Timeout and memory limits
  - No access to dangerous modules

- [ ] **MCP Server** (from Hive) — **MOVED TO PHASE 12**

- [ ] **Smart Frames Memory** (from Memvid)
  - Append-only immutable memory units
  - Time-travel debugging (rewind/replay)
  - Memory capsules (.mv2 format)
  - Sub-5ms local access

- [ ] **GitHub Action Mode** (from claude-code-action)
  - `animus action` command for CI/CD
  - Structured outputs for workflows
  - Progress tracking in PR comments
  - Context-aware activation (@animus mentions)

- [ ] **Multi-Format Web Output** (from Firecrawl)
  - Markdown, HTML, JSON, screenshots
  - Async job queuing for web operations
  - Anti-bot handling and proxy support

### Lower Priority (Roadmap)

- [ ] **Browser Control via MCP** (from BrowserOS)
  - Connect to BrowserOS MCP server (31 tools)
  - Enable web research without built-in browser
  - Screenshot and interaction capabilities

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

- [ ] **Visual Workflow Builder** (from Flowise, Eigent)
  - Drag-and-drop agent design
  - Multi-agent collaboration
  - RAG integration

- [ ] **Agent Q&A Platform** (from Moltyflow)
  - Agent-to-agent collaboration
  - Karma/reputation system
  - Auto-expiring questions

- [ ] **Extended Context Support** (from Qwen3-Coder)
  - 256K-1M context for repository-scale work
  - YaRN context extension

- [ ] **Git TUI Features** (from Lazygit)
  - Undo/redo with reflog
  - Line-level staging
  - Interactive rebasing

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

### Autonomy (Partial - Model Dependent)
- [~] Agent executes tools autonomously — **Tested 2026-01-27**: Works when model outputs JSON, but model compliance varies
- [x] Proper stopping cadences for file creation/modification/deletion
- [x] Path change detection and confirmation
- [x] Blocked command detection
- [ ] Find/document recommended models for reliable tool execution

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

### Sub-Agent Architecture (Phase 11)
- [ ] Goal-driven sub-agents with success criteria
- [ ] Node-based workflows (llm_generate, llm_tool_use, router, function)
- [ ] Edge conditions for routing (on_success, on_failure, conditional)
- [ ] Pause/resume for multi-turn sub-agent conversations
- [ ] OutputCleaner for I/O validation between nodes
- [ ] Tool discovery before node creation

### MCP Integration (Phase 12)
- [ ] Expose Animus tools via MCP server
- [ ] Connect to external MCP servers
- [ ] Browser control via BrowserOS MCP
- [ ] OAuth session management

### Skills System (Phase 13)
- [ ] SKILL.md parser with YAML frontmatter
- [ ] Skill registry (project > user > bundled)
- [ ] CLI commands: skill list/install/create/run
- [ ] Bundled skills: code-review, test-gen, refactor, explain, commit

### Permission System (Phase 14)
- [ ] Three-tier permissions: allow/deny/ask
- [ ] Pattern-based file access control
- [ ] Per-agent permission profiles
- [ ] Default profiles: strict, standard, trusted, yolo

### Local API Server (Phase 15)
- [ ] OpenAI-compatible `/v1/chat/completions`
- [ ] `/v1/models` and `/v1/embeddings` endpoints
- [ ] Animus-specific `/v1/agent/*` endpoints
- [ ] `animus serve` command

---

## Known Issues (from Windows Systests 2026-01-27)

| Issue | Severity | Category | Notes |
|-------|----------|----------|-------|
| Model outputs text instead of JSON tool calls | High | Model | Some models don't follow JSON format in system prompt |
| Model claims to be Anthropic/Claude | Medium | Model | Qwen abliterated model has identity confusion |
| Hallucinated file contents | High | Model | Model fabricates outputs instead of executing tools |
| Excessive safety refusals | Medium | Model | Model refuses benign tasks citing ethics |

**Recommended Actions:**
- [ ] Test additional models (Qwen2.5-Coder-Instruct, DeepSeek-Coder, CodeGemma)
- [ ] Add few-shot examples to system prompt for tool format compliance
- [ ] Document which models work best with Animus tool calling
- [ ] Consider fine-tuning or adapter for reliable tool compliance

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

### Local Repositories (Previously Analyzed)

| Repository | Path | Key Patterns Borrowed |
|------------|------|----------------------|
| **Hive (Aden)** | `C:\Users\charl\hive` | Decision recording, BuilderQuery, Triangulated verification, HybridJudge |
| **Clawdbot** | `C:\Users\charl\clawdbot` | Session compaction, Hybrid search, Error classification, Auth rotation, Skills platform |

### External Repositories (Analyzed 2026-02-01)

| Repository | Stars | Key Patterns Borrowed |
|------------|-------|----------------------|
| **potpie-ai/potpie** | - | Knowledge graph for code (Neo4j), specialist agents (QnA, Debug, Test), context enrichment |
| **firecrawl/firecrawl** | - | Multi-format output, async job queuing (BullMQ), anti-bot handling, API versioning |
| **bluewave-labs/Checkmate** | 9K | Distributed job queues, centralized error middleware, real-time status aggregation |
| **openclaw/openclaw** | - | WebSocket gateway, session lanes, tool policies, sandbox execution, block chunking |
| **pranshuparmar/witr** | 12K | Cross-platform CLI patterns, hierarchical detection, safe read-only operations |
| **anomalyco/opencode** | 94K | Three-tier permissions (allow/deny/ask), MCP/LSP integration, explore/plan agents |
| **memvid/memvid** | - | Smart Frames (append-only memory), hybrid search (BM25+vector), time-travel debugging |
| **C4illin/ConvertX** | - | Modular converter architecture, 22 integrated tools, Docker support |
| **itsOwen/CyberScraper-2077** | - | LLM-based scraping, stealth mode, Tor support, multi-format export |
| **browseros-ai/BrowserOS** | 9K | Browser as MCP server (31 tools), local-first agents, workflow builder |
| **metorial/metorial** | - | 600+ MCP integrations, OAuth session management, embedded MCP explorer |
| **eigent-ai/eigent** | - | Multi-agent workforce, dynamic task decomposition, MCP tools, visual workflow |
| **mxrch/GHunt** | - | Async OSINT framework, JSON export, browser extension integration |
| **charmbracelet/crush** | - | LSP integration, MCP support, session management, Agent Skills Standard |
| **yashab-cyber/nmap-ai** | - | AI-powered scanning, natural language interface, ML-optimized parameters |
| **lemonade-sdk/lemonade** | - | OpenAI-compatible local API, multi-backend (GGUF/ONNX), hardware abstraction |
| **yashab-cyber/metasploit-ai** | - | Intelligent exploit ranking, multi-interface (CLI/Web/Desktop/API) |
| **assafelovic/gpt-researcher** | - | Planner-executor pattern, parallel crawlers, multi-source verification |
| **jesseduffield/lazygit** | - | Undo/redo with reflog, line-level staging, interactive rebasing |
| **janhq/jan** | 40K | OpenAI-compatible local API (1337), MCP integration, extension system |
| **QwenLM/Qwen3-Coder** | - | 256K-1M context, 358 languages, Fill-in-Middle (FIM), custom tool parser |
| **FlowiseAI/Flowise** | - | Visual agent builder (drag-and-drop), multi-agent collaboration, RAG integration |
| **aquasecurity/tracee** | 4K | eBPF runtime security, behavioral detection, kernel-level introspection |
| **n8n-io/n8n** | - | 400+ integrations, hybrid code/visual, LangChain AI integration |
| **logpai/loghub** | - | 16+ log datasets, AI-driven log analytics research, parsing benchmarks |
| **anthropics/claude-code-action** | 5K | Context-aware GitHub Action, progress tracking, structured outputs |
| **lizTheDeveloper/ai_village** | - | ECS architecture (211 systems), multiverse forking, LLM-driven NPCs |
| **browseros-ai/moltyflow** | - | Agent-to-agent Q&A, karma system (+15/-2), auto-expiring questions |
| **Legato666/katana** | - | Dual-mode crawling (standard/headless), scope control, resume capability |
| **anthropics/claude-code** | 63K | Agentic CLI, codebase understanding, git workflow, plugin architecture |
| **anthropics/skills** | 60K | SKILL.md format, dynamic capability extension, production document skills |
| **anthropics/claude-cookbooks** | 32K | RAG patterns, tool use, sub-agents, vision, prompt caching examples |
