# Tasks — ANIMUS

**Last Updated:** 2026-02-09 (Entry #34: Phase 11 Sub-Agent Graph Architecture complete, 59 tests)

## Design Philosophy Update

After analyzing 12 additional repositories, a critical insight emerged: **many agent frameworks over-rely on LLM inference for tasks that should be deterministic**. This causes unpredictable behavior, unnecessary latency, and security vulnerabilities.

**New Guiding Principle:** Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else.

| Task Type | Approach | Examples |
|-----------|----------|----------|
| Security | 100% Hardcoded | Permissions, deny lists, sandboxing |
| Protocols | 100% Hardcoded | MCP, HTTP, JSON-RPC |
| Validation | 100% Hardcoded | Schema, types, patterns |
| File I/O | 100% Hardcoded | Skill loading, context persistence |
| Search | Hybrid (70/30) | BM25 hardcoded, LLM reranking optional |
| Context | Hybrid (80/20) | Truncation hardcoded, summarization LLM |
| Agent Work | LLM | Task execution, generation, understanding |

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

### Phase 16: Code Hardening Audit ✓ (COMPLETE — Entry #17)

**Goal:** Audit and harden existing Animus code to properly separate deterministic logic from LLM inference.

**Principle:** Move logic OUT of LLM interpretation into hardcoded validation/execution.

**Tasks:**
- [x] **Tool Call Parsing Audit** (`src/core/agent.py`)
  - [x] JSON parsing is already PRIMARY (regex is fallback only after `if not tool_calls`)
  - [x] Strict schema validation via `_extract_json_objects()` with max depth limit
  - [x] Malformed tool calls are rejected deterministically (JSONDecodeError)
- [x] **Mandatory Deny Lists** (`src/core/permission.py`) — NEW FILE
  - [x] DANGEROUS_DIRECTORIES frozenset (non-overridable)
  - [x] DANGEROUS_FILES frozenset (non-overridable)
  - [x] BLOCKED_COMMANDS frozenset (non-overridable)
  - [x] Check BEFORE confirmation prompts via `is_mandatory_deny_*()` functions
  - [x] Block at code level, not prompt level — integrated into agent.py, shell.py, filesystem.py
- [x] **Permission Pre-Check** (`src/core/permission.py`)
  - [x] PermissionChecker class with check_path() and check_command()
  - [x] Pattern-based file path checking via fnmatch
  - [x] Command parsing via shlex (deterministic)
  - [x] Symlink escape detection
- [x] **Error Classification Hardening** (`src/core/errors.py`)
  - [x] Already uses regex-based patterns
  - [x] No LLM interpretation of error messages
  - [x] Hardcoded recovery strategies per category
- [x] **Token Counting** (`src/core/context.py`)
  - [x] Already uses character-based estimation via TokenEstimator
  - [x] Hardcoded thresholds (soft: 85%, critical: 95%)
  - [x] ContextWindow class with automatic compaction triggers
- [x] **Template Variables for Sub-Agents** (`src/core/subagent.py`)
  - [x] Added {previous}, {task}, {scope_dir} template vars
  - [x] String replacement via dict.format() (no LLM interpretation)
  - [x] Safe fallback for missing template keys
  - [x] Updated all ROLE_PROMPTS with structured template sections

**New Files Created:**
- `src/core/permission.py` — 599 lines, 100% hardcoded
- `tests/test_permission.py` — 290 lines, 40 tests

**Test Results:** 246 passed (244 + 2 new Python string concat tests)

---

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
- [x] **BuilderQuery Interface** (`src/core/builder.py`) ✓
  - [x] Analyze runs for patterns and failures
  - [x] Generate improvement suggestions
  - [x] get_run_details() and compare_runs() methods
  - [x] get_trends() for time-based analysis
  - [x] 24 tests in test_builder.py
  - [x] Implement `animus reflect` command (CLI integration) ✓
- [x] **Triangulated Verification** (`src/core/judge.py`) ✓
  - [x] Implement rule-based checks (fast, deterministic)
  - [x] Implement LLM fallback evaluation (flexible, contextual)
  - [x] Implement human escalation protocol
  - [x] Create `HybridJudge` class combining all three
  - [x] 37 tests in test_judge.py

### Phase 9: Context Resilience (from Clawdbot analysis)

**Goal:** Handle long conversations without context overflow.

**Inspiration:** Clawdbot's session compaction and error classification.

**Tasks:**
- [x] **Session Compaction** (`src/core/compaction.py`) ✓
  - [x] Implement conversation summarization
  - [x] Auto-trigger when approaching context limit
  - [x] Preserve recent turns + summary of older turns
  - [x] Multiple strategies: truncate, sliding, summarize, hybrid
  - [x] Add compaction to Agent class (integration complete)
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

### Installation System (Cross-Platform) ✓ (COMPLETE — Entry #23)

**Goal:** Simplify Animus installation across all platforms including Jetson.

**Tasks:**
- [x] **Bootstrap Script** (`install.py` in root) — HARDCODED ✓
  - [x] Runs directly after git clone without dependencies
  - [x] System detection (OS, architecture, GPU)
  - [x] Base dependency installation
  - [x] Quickstart guide display
- [x] **Install Module** (`src/install.py`) — HARDCODED ✓
  - [x] AnimusInstaller class with cross-platform logic
  - [x] Platform-specific installation methods
  - [x] Jetson support (Nano, TX2, Xavier, Orin)
  - [x] CUDA architecture detection per Jetson device
  - [x] JetPack version detection
  - [x] Fallback chain (GPU → CPU)
- [x] **CLI Command** (`animus install`) — HARDCODED ✓
  - [x] `--skip-native` flag
  - [x] `--skip-embeddings` flag
  - [x] `--cpu` flag for CPU-only
  - [x] `--verbose` flag
  - [x] Progress callback for UI updates
- [x] **Tests** (`tests/test_install.py`) — 29 tests ✓

**Test Results:** 414 passed (361 previous + 24 builder + 29 install)

### Phase 10: Enhanced Retrieval (from both analyses) ✓ (MOSTLY COMPLETE — Entry #30)

**Goal:** Improve RAG with hybrid search combining keyword and semantic.

**Inspiration:** Clawdbot's SQLite-vec hybrid search, Hive's decision-aware retrieval.

**Status:** Core functionality complete. Optional SQLite-vec backend remains for future.

**Tasks:**
- [x] **Hybrid Search** (`src/memory/hybrid.py`) ✓ (Entry #30)
  - [x] Implement BM25 keyword search (BM25Index class)
  - [x] Combine with existing vector search (HybridSearch class)
  - [x] Configurable weighting (keyword vs semantic)
  - [x] Score normalization and result merging
  - [x] 9 tests added (all pass)
- [ ] **SQLite-vec Backend** (`src/memory/sqlite_vec.py`) — OPTIONAL FUTURE
  - [ ] Replace InMemoryVectorStore with SQLite-vec
  - [ ] Persistent storage (~/.animus/vectordb.sqlite)
  - [ ] Atomic batch writes
  - [ ] Index maintenance and compaction
- [x] **Improved Chunking** ✓ (Entry #30)
  - [x] Add Tree-sitter for AST-aware code chunking (`TreeSitterChunker`)
  - [x] Graceful fallback to CodeChunker when tree-sitter unavailable
  - [x] Metadata enrichment (symbol names, types)
  - [x] 6 tests added
  - [ ] Markdown-aware chunking (preserve headers, lists) — FUTURE
  - [ ] Configurable overlap strategies — FUTURE

### Phase 11: Sub-Agent Architecture Improvements (from Hive building-agents skills) ✓ (COMPLETE — Entry #34)

**Goal:** Transform sub-agents from simple role-based prompts to goal-driven workflow agents.

**Inspiration:** Hive's building-agents-core, building-agents-construction, building-agents-patterns skills.

**Status:** COMPLETE. New `src/subagents/` package with 7 modules. 59 tests passing.

**Tasks:**
- [x] **SubAgentGoal Class** (`src/subagents/goal.py`) ✓
  - [x] Create `SubAgentGoal` dataclass (id, name, description)
  - [x] Create `SuccessCriterion` (id, description, metric, target, weight)
  - [x] Create `Constraint` (id, description, constraint_type, category)
  - [x] Success criteria weights should sum to 1.0
  - [x] Constraint types: hard (must satisfy), soft (prefer to satisfy)
- [x] **SubAgentNode Class** (`src/subagents/node.py`) ✓
  - [x] Create `SubAgentNode` dataclass (id, name, node_type, input_keys, output_keys)
  - [x] Node types: `llm_generate`, `llm_tool_use`, `router`, `function`
  - [x] System prompt with input key interpolation
  - [x] Tools list (only for `llm_tool_use` nodes)
  - [x] Input/output schema validation (optional)
  - [x] Max retries configuration
- [x] **SubAgentEdge Class** (`src/subagents/edge.py`) ✓
  - [x] Create `SubAgentEdge` dataclass (id, source, target, condition, priority)
  - [x] Edge conditions: `on_success`, `on_failure`, `always`, `conditional`
  - [x] Conditional expressions for routing decisions
  - [x] Priority for edge ordering when multiple match
- [x] **SubAgentGraph Class** (`src/subagents/graph.py`) ✓
  - [x] Create `SubAgentGraph` dataclass (id, goal, nodes, edges, entry_node)
  - [x] Terminal nodes list (where execution ends)
  - [x] Pause nodes list (where execution waits for user input)
  - [x] Graph validation (all edges reference valid nodes, entry node exists)
  - [x] Unreachable node detection via BFS
  - [x] Circuit breaker for infinite loops (50 visit limit)
- [x] **SubAgentExecutor** (`src/subagents/executor.py`) ✓
  - [x] Execute graph from entry point to terminal/pause
  - [x] Context propagation between nodes (output → next input)
  - [x] Retry logic per node with exponential backoff
  - [x] Execution result with success, steps_executed, output, error
  - [x] StepResult per-node with duration tracking
- [x] **Pause/Resume Support** (`src/subagents/session.py`) ✓
  - [x] Session state persistence (SessionState dataclass)
  - [x] SessionStore with save/load/delete/list
  - [x] Session state passed separately from input_data on resume
  - [x] Storage in `~/.animus/sessions/`
- [x] **OutputCleaner Integration** (`src/subagents/cleaner.py`) ✓
  - [x] Validate node output matches next node's input schema
  - [x] Detect JSON parsing trap (entire response in one key)
  - [x] Multi-strategy clean: JSON parse → code block → fallback
  - [x] Basic type checking from schema
- [x] **Tool Discovery & Validation** ✓
  - [x] Verify tools exist before adding to nodes (orchestrator._validate_graph_tools)
  - [x] Log warning if requested tool unavailable
- [x] **Update SubAgentOrchestrator** (`src/core/subagent.py`) ✓
  - [x] Added execute_graph() method accepting SubAgentGraph
  - [x] Uses SubAgentExecutor for graph-based sub-agents
  - [x] Backward compatibility with role-based sub-agents preserved
  - [x] Pause/resume via SessionStore integration
- [x] **Tests** (`tests/test_subagent_graph.py`) — 59 tests ✓
  - [x] Test goal validation (criteria weights sum to 1.0)
  - [x] Test node type behavior (llm_generate, llm_tool_use, router, function)
  - [x] Test edge conditions (on_success, on_failure, always, conditional)
  - [x] Test graph execution flow (function graph, router, failure edges)
  - [x] Test retry with exponential backoff
  - [x] Test circuit breaker for infinite loops
  - [x] Test pause/resume with session state
  - [x] Test session persistence (save, load, delete, list)
  - [x] Test output cleaning (JSON trap, code blocks, type checking)
  - [x] Test orchestrator graph tool validation

**New Files:**

| File | Lines | Purpose |
|------|-------|---------|
| src/subagents/__init__.py | 35 | Package exports |
| src/subagents/goal.py | 101 | Goal, SuccessCriterion, Constraint |
| src/subagents/node.py | 95 | Node types and validation |
| src/subagents/edge.py | 71 | Edge conditions and evaluation |
| src/subagents/graph.py | 127 | Graph container and validation |
| src/subagents/executor.py | 253 | Graph execution engine |
| src/subagents/session.py | 95 | Pause/resume persistence |
| src/subagents/cleaner.py | 140 | Output validation/cleaning |
| tests/test_subagent_graph.py | 595 | Test suite (59 tests) |
| **Total** | **~1,512** | **Complete graph sub-agent system** |

### Phase 12: MCP Integration (from external repo analysis) ✓ (COMPLETE — Entry #20, verified #29)

**Goal:** Enable Animus to expose tools via MCP and connect to external MCP servers.

**Status:** COMPLETE. Core MCP functionality implemented and tested.

**Tasks:**
- [x] **MCP Server Implementation** (`src/mcp/server.py`) — HARDCODED ✓
  - [x] JSON-RPC 2.0 message parsing (hardcoded validation)
  - [x] Method routing table (dict-based dispatch, no LLM)
  - [x] Tool schema generation from registry (programmatic)
  - [x] Support stdio and HTTP transports
  - [x] Add `animus mcp server` command
- [x] **MCP Client Implementation** (`src/mcp/client.py`) — HARDCODED ✓
  - [x] Connect to external MCP servers
  - [x] Tool discovery via `tools/list` (parse JSON response)
  - [x] Convert MCP tool schemas to Animus format (schema mapping)
- [x] **MCP CLI Commands** ✓
  - [x] `animus mcp server` - Start MCP server
  - [x] `animus mcp tools` - List MCP tools
  - [x] `animus mcp list` - List configured servers
  - [x] `animus mcp add` - Add server
  - [x] `animus mcp remove` - Remove server
- [x] **MCP Tests** (`tests/test_mcp.py`) — 56 tests ✓

**Remaining (Low Priority):**
- [ ] API key authentication for HTTP transport
- [ ] Connection health ping/timeout

### Phase 13: Skills System (from Anthropic skills repo) ✓ (COMPLETE — Entry #21, verified #29)

**Goal:** Enable modular capability extension via SKILL.md format.

**Status:** COMPLETE. All functionality implemented and tested.

**Tasks:**
- [x] **SKILL.md Parser** (`src/skills/parser.py`) — HARDCODED ✓
- [x] **Skill Registry** (`src/skills/registry.py`) — HARDCODED ✓
- [x] **Skill Loader** (`src/skills/loader.py`) — HARDCODED ✓
- [x] **CLI Commands** — `animus skill list/show/inscribe/install/run` ✓
- [x] **Bundled Skills** — code-review, test-gen, refactor, explain, commit ✓
- [x] **Skills Tests** (`tests/test_skills.py`) — 59 tests ✓

### Phase 14: Enhanced Permission System (from OpenCode analysis)

**Goal:** Replace binary confirmation with three-tier allow/deny/ask system.

**Inspiration:** sandbox-runtime, OpenCode, stakpak/agent

**Implementation Principle:** 100% hardcoded — LLMs NEVER make security decisions.

**Status:** Partially complete via Phase 16. Core permission system implemented.

**Tasks:**
- [x] **Mandatory Deny Lists** (`src/core/permission.py`) — HARDCODED, NON-OVERRIDABLE ✓ (Phase 16)
  - [x] DANGEROUS_DIRECTORIES frozenset
  - [x] DANGEROUS_FILES frozenset
  - [x] BLOCKED_COMMANDS frozenset
  - [x] Check BEFORE any other permission logic
- [x] **Permission Model** (`src/core/permission.py`) — HARDCODED ✓ (Phase 16)
  - [x] Three actions: PermissionAction.ALLOW, DENY, ASK (enum, not strings)
  - [x] fnmatch/glob pattern matching (no LLM interpretation)
  - [x] Categories: read, write, execute, external_directory
  - [x] Merge order: mandatory_deny → user_config → defaults
- [x] **Permission Evaluation** — HARDCODED ✓ (Phase 16)
  - [x] Path normalization (resolve, expanduser, Windows path handling)
  - [x] Pattern matching via fnmatch.fnmatch()
  - [x] Command parsing via shlex.split() (deterministic)
  - [ ] Cache evaluated permissions for session
- [x] **Symlink Boundary Validation** (from sandbox-runtime) — HARDCODED ✓ (Phase 16)
  - [x] Resolve symlink targets
  - [x] Verify target within allowed boundaries
  - [x] Block symlink-based escapes
- [ ] **Default Profiles** — HARDCODED configurations
  - [ ] `strict`: ask_all except reads
  - [ ] `standard`: allow reads, ask writes/bash
  - [ ] `trusted`: allow most, ask destructive
- [ ] **Per-Agent Permission Scopes** — HARDCODED
  - [ ] Explore: read-only, no writes, no shell
  - [ ] Plan: read-only, limited shell (read commands only)
  - [ ] Build: standard permissions

### Phase 15: OpenAI-Compatible Local API (from Jan, Lemonade) ✓ (MOSTLY COMPLETE — verified #29)

**Goal:** Serve Animus capabilities via OpenAI-compatible API for ecosystem integration.

**Status:** Core API server implemented (473 lines). WebSocket server for IDE (494 lines).

**Tasks:**
- [x] **API Server** (`src/api/server.py`) — HARDCODED ✓
  - [x] FastAPI-style routes with typed handlers
  - [x] Request validation via Pydantic models
  - [x] Response schemas (OpenAI-compatible format)
  - [ ] API key validation (optional enhancement)
  - [ ] Rate limiting via token bucket (optional enhancement)
- [x] **Model Endpoint** ✓
  - [x] `/v1/models` — List local models
  - [x] Model info with capabilities
- [x] **Chat Completions** ✓
  - [x] `/v1/chat/completions` — OpenAI-compatible
  - [x] SSE streaming support
  - [x] Tool/function calling support
- [x] **Embeddings Endpoint** ✓
  - [x] `/v1/embeddings` — Batch embedding generation
- [x] **WebSocket Server** (`src/api/websocket_server.py`) — 494 lines ✓
  - [x] Real-time streaming for IDE
  - [x] Session management
  - [x] Inline diff support
- [x] **CLI Commands** ✓
  - [x] `animus serve` — Start REST API server
  - [x] `animus ide` — Start WebSocket server for VSCode

**Remaining (Low Priority):**
- [ ] Rate limiting
- [ ] API key authentication

### Phase 16.5: Web Search Security (Ungabunga-Box) ✓ (COMPLETE — Entries #27-28)

**Goal:** Enable safe web search and fetch with multi-layer security against prompt injection.

**Status:** COMPLETE. Three-layer security architecture implemented and tested.

**Implementation Principle:** 100% hardcoded rules + smaller LLM validator (defense-in-depth).

**Tasks:**
- [x] **Web Tools** (`src/tools/web.py`) — 658 lines ✓
  - [x] WebSearchTool - DuckDuckGo search integration
  - [x] WebFetchTool - URL content fetching
  - [x] Process isolation (subprocess with env={})
  - [x] Content sanitization (HTML stripping, size limits)
  - [x] 30+ prompt injection patterns (hardcoded regex)
  - [x] Human escalation for suspicious content
  - [x] 24 tests (2 network tests skipped)
- [x] **LLM Validator** (`src/core/web_validator.py`) — 502 lines ✓
  - [x] WebContentRuleEngine - Rule-based validation
  - [x] WebContentLLMValidator - Semantic validation using Qwen-1.5B
  - [x] WebContentJudge - Hybrid judge combining rules + LLM + human
  - [x] THREAT/FALSE_POSITIVE classification prompting
  - [x] Different model from main agent (defense-in-depth)
  - [x] 25 tests (all passing)
- [x] **Security Documentation** (`docs/web_search_security_design.md`) ✓
  - [x] Attack surface analysis
  - [x] Threat model documentation
  - [x] Phase 1-3 architecture progression
- [x] **Dependencies** ✓
  - [x] Added bleach>=6.0.0 to [web] extras
  - [x] Added readability-lxml>=0.8.0 to [web] extras
  - [x] httpx already present in base deps
- [x] **Integration** ✓
  - [x] Exported web tools from `src/tools/__init__.py`
  - [x] Exported validators from `src/core/__init__.py`
  - [x] Added validator model download instructions to README

**Security Layers:**
1. **Process Isolation** - Fetch runs in subprocess with env={}
2. **Rule-Based Validation** - 30+ hardcoded injection patterns
3. **LLM Semantic Validator** - Qwen-1.5B classifies flagged content
4. **Human Escalation** - Uncertain cases prompt user approval

**Remaining (Phase 3 - Optional):**
- [ ] Container isolation with `--paranoid` flag
- [ ] Dockerfile for fetch-sandbox
- [ ] MCP protocol integration for containerized fetch

### Phase 17: Audio Interface (Quality of Life) ~PARTIAL (Entry #32, revised #33)

**Goal:** Add personality to Animus with audio feedback for commands and task completion.

**Design Principle:** 100% hardcoded MIDI synthesis and playback. No LLM involvement.

**Status:** PARTIAL. Praise audio (Mozart/Bach) functional. Speech and Moto Perpetuo disabled pending proper implementation.

**Current State (Entry #33):**
- Praise audio (fanfare/spooky) works - plays on tasks with 5+ turns
- Speech synthesis DISABLED - MIDI phonemes don't produce understandable speech, needs real TTS
- Moto Perpetuo DISABLED - synthesized tones don't match actual piece, needs proper MIDI files
- See Backlog > Medium Priority for future audio improvements

**Tasks:**
- [x] **Audio Module Setup** (`src/audio/`) ✓
  - [x] Create module structure (`__init__.py`, `speech.py`, `midi.py`, `config.py`)
  - [x] Add dependencies: pygame, numpy, mido to [audio] extras
  - [x] MIDI concatenative synthesis implemented
- [x] **Speech Synthesis** (`src/audio/speech.py`) — 190 lines ✓
  - [x] Implement `SpeechSynthesizer` class
  - [x] 26-letter phoneme-to-MIDI note mapping
  - [x] Square wave synthesis for robotic voice
  - [x] Low pitch (0.6x multiplier), "spooky AI bot" aesthetic
  - [x] Speech remains understandable despite effects
  - [x] Hardcoded phrases: "Yes, Master", "It will be done", "Working", "Complete"
  - [x] Intelligent text filtering (excludes code/commands)
- [x] **MIDI Engine** (`src/audio/midi.py`) — 268 lines ✓
  - [x] Implement `MIDIEngine` class
  - [x] pygame.mixer for MIDI playback
  - [x] Square and sine wave synthesis
  - [x] Note sequence playback with timing
  - [x] Background thread for looping music
- [x] **Musical Sequences** (`src/audio/midi.py`) ✓
  - [x] Mozart fanfare: "Eine kleine Nachtmusik" K.525 opening (12 notes, G major)
  - [x] Bach spooky: "Little Fugue" BWV 578 opening (7 notes, G minor)
  - [x] Paganini background: "Moto Perpetuo" Op. 11 (16 notes, looping)
  - [x] Hardcoded note sequences (pitch, duration, velocity)
- [x] **Agent Integration** (src/core/agent.py) ✓
  - [x] `_init_audio()` - Initialize audio system
  - [x] `_speak(text)` and `_speak_phrase(key)` methods
  - [x] `_praise()` - Play completion music
  - [x] `_start_moto()` and `_stop_moto()` - Background music control
  - [x] Hook: "Yes, Master" on user input (Agent.step)
  - [x] Hook: "It will be done" when tool calls detected
  - [x] Hook: Moto Perpetuo starts/stops with Agent.run()
  - [x] Hook: Praise plays on multi-step task completion (>2 turns)
- [x] **CLI Commands** (src/main.py) ✓
  - [x] `animus speak` - Enable voice
  - [x] `animus speak --off` - Disable voice
  - [x] `animus praise --fanfare` - Set Mozart mode
  - [x] `animus praise --spooky` - Set Bach mode
  - [x] `animus praise --moto` - Enable Moto Perpetuo
  - [x] `animus praise --motoff` - Disable Moto Perpetuo
  - [x] `animus praise --off` - Disable all praise audio
  - [x] Show current settings when no flags provided
- [x] **Configuration** (src/core/config.py) ✓
  - [x] AudioConfig integrated into AnimusConfig
  - [x] speak_enabled: bool (default: False)
  - [x] praise_mode: Literal["fanfare", "spooky", "off"] (default: "off")
  - [x] moto_enabled: bool (default: False)
  - [x] volume: float (0.0-1.0, default: 0.7)
  - [x] speech_pitch: float (0.1-2.0, default: 0.6)
  - [x] Persists in ~/.animus/config.yaml
- [x] **Tests** (`tests/test_audio.py`) — 32 tests ✓
  - [x] AudioConfig validation
  - [x] MIDI engine (note generation, wave synthesis, playback)
  - [x] Speech synthesizer (phoneme mapping, text filtering)
  - [x] Agent integration (mocked audio)
  - [x] Config serialization/deserialization
  - [x] All 32 tests passing
- [x] **Documentation** (README.md) ✓
  - [x] Audio Interface section with examples
  - [x] Voice features description
  - [x] Music features description
  - [x] Installation instructions
  - [x] Musical piece details (Mozart, Bach, Paganini)

### Implementation Highlights

**Phoneme Mapping (Concatenative Synthesis):**
```python
PHONEME_NOTES = {
    'A': 48,   # C3 (ah)
    'E': 52,   # E3 (eh)
    'I': 55,   # G3 (ee)
    'O': 50,   # D3 (oh)
    'U': 53,   # F3 (oo)
    'M': 48,   # C3 (nasal)
    'S': 59,   # B3 (hiss)
    # ... 26 total
}
```

**Mozart Fanfare (Eine kleine Nachtmusik):**
```python
# G-D-G / G-D-G / G-D-G-D-B-G
notes = [
    Note(67, 0.15, 90),  # G4
    Note(62, 0.15, 70),  # D4
    Note(67, 0.3, 90),   # G4
    # ... (12 notes total)
]
```

**Bach Spooky (Little Fugue):**
```python
# Descending chromatic line
notes = [
    Note(67, 0.2, 75),   # G4
    Note(65, 0.15, 70),  # F4
    Note(63, 0.15, 70),  # Eb4
    # ... (7 notes total)
]
```

### Graceful Degradation

Audio features are completely optional:
- Require `pip install -e ".[audio]"` for dependencies
- Agent initialization checks audio availability
- Methods return early if audio unavailable
- No crashes or errors if pygame/numpy missing
- Core functionality unaffected

### Test Results
```
804 passed in test suite
32 audio tests: 100% pass rate
```

### Checkpoint
**Status:** COMPLETE — Phase 17 Audio Interface fully implemented. Animus now has voice and musical personality.

### Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| src/audio/midi.py | 268 | MIDI synthesis engine |
| src/audio/speech.py | 190 | Voice synthesis |
| src/audio/config.py | 14 | PraiseMode enum |
| src/audio/__init__.py | 13 | Module exports |
| tests/test_audio.py | 504 | Test suite |
| **Total** | **989** | **Complete audio system** |

---

## Backlog (Prioritized)

### Quick Wins (Completed)

- [x] **Fix Split-File Model Download** ✓ (Entry #29)
  - `animus pull` now filters out split files (00001-of-00002)
  - Prefers single-file GGUF when available
  - Added regex filter and warning message
  - Added 3 tests for split file detection

- [x] **Add httpx to dependencies** ✓ (Already present in base deps)
  - httpx>=0.25.0 already in pyproject.toml dependencies

- [x] **Add bleach/readability to dependencies** ✓ (Entry #29)
  - Added new [web] extras group to pyproject.toml
  - bleach>=6.0.0 and readability-lxml>=0.8.0
  - Added to [all] extras group

- [x] **Universal JSON Output Mode** ✓ (Entry #30)
  - Standardized JSON output for tool calling across all prompts
  - Added explicit JSON syntax rules and examples to system prompt
  - Added `json_mode` config option to AgentConfig

- [x] **Ubuntu 22.04 Prerequisites Documentation** ✓ (Entry #26)
  - Added to README.md with deadsnakes PPA instructions
  - Added build tools requirement
  - Added troubleshooting section

### High Priority (Near-term)

- [x] **Ubuntu 22.04 Prerequisites Documentation** ✓ (Entry #26, verified #29)
  - [x] Added to README.md with deadsnakes PPA instructions
  - [x] Added build tools requirement
  - [x] Added troubleshooting section

- [x] **Universal JSON Output Mode** ✓ (Entry #30)
  - [x] Standardize JSON output for tool calling across all prompts
  - [x] Add structured generation enforcement option (`json_mode` config)
  - [x] Update system prompt with explicit JSON syntax rules and examples
  - [ ] Consider grammar-constrained decoding for llama-cpp-python (future)
  - Rationale: Qwen2.5-Coder excels at JSON, leverage this consistently

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

- [ ] **Audio System Improvements** (Phase 17 follow-up)
  - [ ] **Real TTS Integration** - Replace MIDI phoneme synthesis with pyttsx3/espeak
    - Current MIDI-based speech produces unrecognizable beeping
    - Need proper concatenative synthesis or TTS engine
    - Consider: pyttsx3 (cross-platform), espeak (Linux), SAPI (Windows)
  - [ ] **Moto Perpetuo Background Music** - Requires external MIDI app
    - Current synthesized tones don't sound like the actual piece
    - Need proper MIDI files with correct tempo, timing, and instrumentation
    - User will build separate app to generate proper MIDI files
    - Re-enable once proper MIDI assets available
  - [ ] **Improve Classical Sequences** - Better synthesis for Mozart/Bach
    - Current sine wave synthesis sounds thin
    - Consider: SoundFont playback, FM synthesis, or pre-rendered audio

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
- [x] Session compaction prevents context overflow (integrated into Agent)
- [x] Error classification enables appropriate recovery
- [x] Retry logic with exponential backoff

### Retrieval (Phase 10)
- [ ] Hybrid search (BM25 + vector) for better RAG
- [ ] SQLite-vec persistent storage
- [ ] Tree-sitter AST-aware code chunking

### Sub-Agent Architecture (Phase 11) ✓
- [x] Goal-driven sub-agents with success criteria
- [x] Node-based workflows (llm_generate, llm_tool_use, router, function)
- [x] Edge conditions for routing (on_success, on_failure, conditional)
- [x] Pause/resume for multi-turn sub-agent conversations
- [x] OutputCleaner for I/O validation between nodes
- [x] Tool discovery before node creation

### MCP Integration (Phase 12) ✓
- [x] Expose Animus tools via MCP server
- [x] Connect to external MCP servers
- [ ] Browser control via BrowserOS MCP (future)
- [ ] OAuth session management (future)

### Skills System (Phase 13) ✓
- [x] SKILL.md parser with YAML frontmatter
- [x] Skill registry (project > user > bundled)
- [x] CLI commands: skill list/show/inscribe/install/run
- [x] Bundled skills: code-review, test-gen, refactor, explain, commit

### Permission System (Phase 14) — Partially Complete
- [x] Three-tier permissions: allow/deny/ask
- [x] Pattern-based file access control
- [ ] Per-agent permission profiles
- [ ] Default profiles: strict, standard, trusted

### Local API Server (Phase 15) ✓
- [x] OpenAI-compatible `/v1/chat/completions`
- [x] `/v1/models` and `/v1/embeddings` endpoints
- [x] WebSocket server for IDE integration
- [x] `animus serve` and `animus ide` commands

---

## Known Issues (from Windows Systests 2026-01-27, 2026-02-02)

| Issue | Severity | Category | Status | Notes |
|-------|----------|----------|--------|-------|
| Model outputs text instead of JSON tool calls | High | Model | Open | Some models don't follow JSON format in system prompt |
| Model claims to be Anthropic/Claude | Medium | Model | Open | Qwen abliterated model has identity confusion |
| Hallucinated file contents | High | Model | Open | Model fabricates outputs instead of executing tools |
| Excessive safety refusals | Medium | Model | Open | Model refuses benign tasks citing ethics |
| Python-style string concat in JSON | High | Parsing | **FIXED** | Qwen3-VL outputs malformed JSON with Python string concat; fixed via `_fix_python_string_concat()` |

**Recommended Actions:**
- [ ] Test additional models (Qwen2.5-Coder-Instruct, DeepSeek-Coder, CodeGemma)
- [ ] Add few-shot examples to system prompt for tool format compliance
- [ ] Document which models work best with Animus tool calling
- [ ] Consider fine-tuning or adapter for reliable tool compliance

---

## Completed (Recent)

- **Phase 11: Sub-Agent Graph Architecture** (Entry #34, 59 tests) ✓
  - New `src/subagents/` package: goal, node, edge, graph, executor, session, cleaner
  - Goal-driven workflows with success criteria and constraints
  - 4 node types: LLM generate, LLM tool use, router, function
  - Graph validation, circuit breaker, pause/resume, output cleaning
  - Orchestrator updated with execute_graph() — backward compatible
  - ~1,512 lines total across 9 files
- **GECK Repor: External Repository Analysis** (Entry #33) ✓
  - Analyzed 84 repositories (42 prior + 42 new) across 6 exploration goals
  - Produced repor_findings.md (523 lines) with findings, themes, and recommendations
  - Key discoveries: GBNF grammar constraints, progressive disclosure memory, parse-retry-correct loops, variant/fallback systems, skill-as-markdown pattern
  - 14 prioritized recommendations (Immediate / Near-term / Strategic)
- **Phase 17: Audio Interface** (Entry #32, 804 tests) ✓
  - MIDI concatenative speech synthesis with spooky AI voice
  - Mozart fanfare and Bach spooky completion music
  - Paganini Moto Perpetuo background music during execution
  - CLI commands: `animus speak`, `animus praise`
  - 32 audio tests (100% pass rate)
  - 989 lines total (midi.py, speech.py, config.py, tests)
- **Phase 16.5: Web Search Security** (Entries #27-28) ✓
  - Ungabunga-Box pattern: process isolation + rule-based validation + LLM semantic validator
  - WebSearchTool and WebFetchTool with 30+ injection patterns
  - WebContentJudge using Qwen-1.5B for defense-in-depth
  - 49 tests (24 web_tools, 25 web_validator)
- **Phase 10: Enhanced Retrieval** (Entry #30) ✓
  - BM25+Vector hybrid search (HybridSearch class)
  - TreeSitterChunker with AST-aware code boundaries
  - Graceful fallback and metadata enrichment
  - 15 tests added (9 hybrid, 6 tree-sitter)
- **JSON Output Standardization** (Entry #30) ✓
  - Explicit JSON syntax rules in system prompt
  - `json_mode` config option
  - Addresses model compliance issues
- **Linux Installation Testing** (Entry #26) ✓
  - Ubuntu 22.04 validated with workarounds
  - README updated with prerequisites
  - 5/5 functionality tests passed
- Phase 15: API Server (OpenAI-compatible REST + WebSocket for IDE)
- Phase 13: Skills System (SKILL.md format, 5 bundled skills, 59 tests)
- Phase 12: MCP Integration (server + client, 56 tests)
- Phase 8: Self-Improvement (BuilderQuery, HybridJudge, `animus reflect` command)
- Installation System (cross-platform installer with Jetson support, `animus install` command)
- Startup Performance: 91% faster (5.5s → 0.5s) via lazy loading
- Bug Fix: Python string concatenation in JSON parsing (Qwen3-VL compatibility)
- Phase 9: Session Compaction integrated into Agent class
- Phase 16: Code Hardening Audit (permission.py, template variables)
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

### External Repositories (Analyzed 2026-02-01 — Batch 2)

| Repository | Key Patterns Borrowed | Hardcoded vs LLM |
|------------|----------------------|------------------|
| **anthropic-experimental/sandbox-runtime** | OS-level sandboxing, mandatory deny lists, symlink validation, proxy filtering | 100% hardcoded |
| **VoidenHQ/voiden** | Hook registry with priorities, IPC tool system, pipeline stages, state persistence | 95% hardcoded |
| **automazeio/ccpm** | File-based context, parallel agent coordination, template variables, spec-driven | 90% hardcoded |
| **mitsuhiko/agent-stuff** | Event-driven lifecycle, TUI components, session state, fuzzy matching | 85% hardcoded |
| **badlogic/pi-skills** | SKILL.md format, YAML frontmatter, CLI tool abstraction | 100% hardcoded |
| **nicobailon/pi-subagents** | Chain execution, fan-out/fan-in, async jobs, template replacement | 80% hardcoded |
| **stakpak/agent** | Rust MCP, secret substitution, provider abstraction, mTLS | 100% hardcoded |
| **supermemoryai/supermemory** | Normalized embeddings, multi-tier fallback, relevance scoring | 70% hardcoded |
| **assafelovic/skyll** | Protocol-based sources, relevance ranking, LRU caching with TTL | 90% hardcoded |
| **oxidecomputer/dropshot** | Type-safe extractors, trait-based APIs, OpenAPI generation from code | 100% hardcoded |
| **adenhq/hive** | Node graphs, edge conditions, semantic failure detection | 60% hardcoded |

### External Repositories (Analyzed 2026-02-09 — GECK Repor)

| Repository | Key Patterns |
|------------|-------------|
| **AntonOsika/gpt-engineer** | Step-based pipeline, ABC+defaults, preprompts, FilesDict, chat-to-files parser |
| **nerdalert/agent-lightning** | Emit telemetry, resource versioning, weakref linking, capability declarations |
| **nicholasgriffintn/rho** | JSONL memory with decay, skills-as-markdown SOPs, auto-memory extraction, tiered models |
| **nicholasgriffintn/CodePilot** | Promise-based permission gate, SSE streaming, incremental DB migration |
| **Doriandarko/maestro** | Orchestrator→sub-agent decomposition, file-based context passing, parallel execution |
| **amcode21/codex-action** | Middleware pipeline, auth→rate-limit→execute chain, structured error responses |
| **pydantic/monty** | Code-as-action mode, Python sandbox execution, collapses N tool-call round trips into 1 |
| **VoltAgent/voltagent** | Multi-agent lifecycle, guardrails middleware, typed registry pattern |
| **openai/codex/skills** | Skill marketplace concept, composable skill definitions, multi-agent roles |
| **OpenBMB/ChatDev** | Multi-agent software company, role-based chat chains, meta-programming |
| **ggerganov/llama.cpp** | GBNF grammar-constrained decoding, sampling pipeline, JSON schema→grammar converter |
| **nicholasgriffintn/nanochat** | Tool use via token forcing, single-dial config, compute-optimal training strategies |
| **rasbt/LLMs-from-scratch** | Instruction fine-tuning, LoRA implementation, DPO, masked loss computation |
| **tensorzero/tensorzero** | Variant/fallback systems, Best-of-N, DICL, feedback flywheel, <1ms gateway |
| **LifeisaJourney/memU** | 3-layer memory hierarchy, Protocol-based DI, pipeline validation, interceptors |
| **mtybadger/shannon** | Prompt-as-config, Temporal workflows, tiered error classification |
| **langchain-ai/langchain** | Runnable protocol, pipe composition, .with_retry()/.with_fallbacks(), send_to_llm errors |
| **langgenius/dify** | Model provider factory, unified 5-type error translation, DAG workflows, hybrid RAG |
| **browser-use/browser-use** | Decorator tool registry, action loop detector, capability-tiered prompts, fallback LLM |
| **nicholasgriffintn/claude-mem** | Progressive disclosure memory (3-layer), granular vector indexing, hybrid search |
| **Textualize/rich** | Protocol-based rendering, Live display, thread-safe console, render hooks |
| **yamadashy/repomix** | Token counting, Tree-sitter compression, DI with overrideDeps, config layering |
| **yt-dlp/yt-dlp** | Extractor auto-routing, importlib plugins, expected-error flags, lazy loading |
| **nicholasgriffintn/superpowers** | Skill shadowing, 2-5 min task decomposition, progressive disclosure, sub-agent orchestration |
| **nicholasgriffintn/awesome-claude-skills** | Skill scaffolding (init/package scripts), scripts/ for deterministic tasks, references/ |
| **nicholasgriffintn/WrenAI** | @provider decorator registry, Jinja2 conditional prompts, PipelineComponent DI |
| **nicholasgriffintn/compound-engineering** | Knowledge compounding loop, parallel sub-agent phases, #$ARGUMENTS template vars |
| **opendatalab/MinerU** | Singleton model cache (config-keyed), multi-backend strategy, batch processing |
| **Shubhamsaboo/awesome-llm-apps** | Local RAG patterns, fallback chains, similarity threshold tuning, chunk size tuning |
| **sgl-project/sglang** | Radix tree KV-cache, continuous batching, structured generation constraints |
| **outlines-dev/outlines** | Finite-state machine guided generation, regex→FSM compilation, token masking |
| **mlc-ai/web-llm** | WebGPU runtime, tokenizer in browser, streaming decode, model sharding |
| **lmstudio-ai/lms** | CLI model management, download progress, server lifecycle, local model catalog |
| **amannn/next-intl** | ICU message format, namespace scoping, server/client component patterns |
| **n8n-io/n8n** | Visual workflow builder, credential encryption, webhook triggers, retry logic |
| **BerriAI/litellm** | 100+ provider adapters, unified interface, budget/rate limiting, fallback routing |
| **simonw/llm** | Plugin system (entry points), template management, SQLite logging, model aliases |
| **deepseek-ai/DeepSeek-V3** | Multi-head latent attention, auxiliary-loss-free balancing, FP8 mixed precision |
| **meta-llama/llama-models** | Reference implementations, tokenizer specs, model card conventions |
| **huggingface/transformers** | Auto classes, pipeline abstraction, model hub integration, from_pretrained pattern |
| **vllm-project/vllm** | PagedAttention, continuous batching, tensor parallelism, OpenAI-compatible API |
| **ollama/ollama** | Modelfile format, layer caching, concurrent model serving, REST+gRPC API |

---

## Design Principle: Hardcoding vs LLM Inference

**Critical Rule:** Use LLMs only where ambiguity, creativity, or NLU is required. Use hardcoded logic for everything else.

### Always Hardcode (100%)
- Security decisions (permissions, deny lists, sandbox boundaries)
- Protocol handling (MCP, HTTP, JSON-RPC, IPC)
- Schema validation (JSON Schema, Pydantic)
- Error classification (regex patterns, recovery strategies)
- File I/O (skill loading, context persistence)
- Token counting (character-based estimation)
- Template variable replacement

### Hybrid Approach (70-90% Hardcoded)
- Search (BM25 hardcoded + vector model + optional LLM reranking)
- Context management (truncation hardcoded, summarization via LLM)
- Sub-agent orchestration (flow control hardcoded, agent work via LLM)

### LLM-Appropriate Tasks
- Task decomposition and planning
- Natural language understanding
- Content summarization
- Creative generation
- Ambiguous decision-making
