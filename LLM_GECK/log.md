# Session Log — ANIMUS (v2)

*Append only. Do not edit existing entries.*
*Previous log (Entries #0-50) archived to `Previous GECK Logs/log_entries_0-50.md`*

---

## Entry #0 — 2026-02-10

### Synopsis: Where We Are

Animus is a local-first CLI coding agent with RAG, tool use, and sub-agent orchestration. The project has gone through 17+ phases of development across 50 prior log entries, building up a substantial feature set. A lean rebuild of the session layer was just completed to make `animus rise` actually productive.

#### Core Architecture (Stable)
- **CLI:** Typer-based with Rich UI. Commands: detect, init, config, models, status, pull, ingest, search, sessions, rise, plus specialized commands (serve, ide, speak, praise, reflect, install, skill, mcp).
- **Model Layer:** Abstract ModelProvider with three backends — NativeProvider (llama-cpp-python GGUF), OpenAIProvider, AnthropicProvider. Factory pattern with fallback chain. Streaming support (SSE) for API providers.
- **RAG Pipeline:** Scanner → Chunker (token, code, tree-sitter) → Embedder (native sentence-transformers, mock) → VectorStore (in-memory, ChromaDB). Hybrid search (BM25 + vector).
- **Agent Loop:** Core Agent class with tool calling, context window management (model-size-aware), error classification with recovery strategies, session compaction. Streaming `run_stream()` with per-chunk callbacks.
- **Tools:** Filesystem (read/write/list), Shell (with blocked/dangerous command detection), Git (status/diff/log/branch/add/commit/checkout with blocked ops).
- **Safety:** 100% hardcoded permission system — mandatory deny lists, three-tier allow/deny/ask, per-agent scopes (explore/plan/build), permission caching, symlink escape detection. LLMs never make security decisions.

#### Advanced Systems (From Prior Phases)
- **Sub-Agent Orchestration:** Goal-driven graph workflows with node types (llm_generate, llm_tool_use, router, function), edge conditions, pause/resume, circuit breaker. Specialist agents (explore, plan, debug, test).
- **MCP Integration:** JSON-RPC server + client, stdio and HTTP transports, tool schema bridging.
- **Skills System:** SKILL.md format with YAML frontmatter. Registry (project > user > bundled). 5 bundled skills.
- **Self-Improvement:** Decision recording, run persistence, BuilderQuery analysis, HybridJudge (rules + LLM + human), knowledge compounding.
- **Error Resilience:** Classified errors with regex patterns, parse-retry-correct loop, model fallback chain with escalation/de-escalation, auth profile rotation, action loop detection.
- **Context Management:** Token estimation, tier-specific budgets (small/medium/large), session compaction (truncate/sliding/summarize/hybrid), capability-tiered system prompts, progressive disclosure RAG.
- **API Server:** OpenAI-compatible REST (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`), WebSocket for IDE integration.
- **Other:** GGUF model catalog + download with auto-config, GBNF grammar generation, media pipeline with MIME detection, code-as-action sandbox, feedback flywheel, DICL (dynamic in-context learning), lane-based command queueing, audio interface (praise/speech).

#### Current State (Lean Rebuild)
The lean rebuild stripped back to a clean foundation and reimplemented the session layer:
- **162 tests passing** in the lean codebase (0.77s)
- **727+ tests** in the full codebase (prior phases)
- Llama-3.2-1B-Instruct-Q4_K_M.gguf downloaded and configured
- `animus rise` works with sessions, streaming, slash commands, git tools, confirmation wiring

#### The Critical Problem
Small local models (1-7B parameters) **thrash when given complex multi-step tasks**. They can't hold planning + execution + tool management + conversation history in context simultaneously. Only 150-200B+ models operate at human bandwidth unassisted.

**Proposed Solution: Plan-Then-Execute Architecture**

| Phase | Actor | What It Does |
|-------|-------|-------------|
| 1. Plan | LLM (focused prompt, no tools, no history) | Decomposes request into numbered step list |
| 2. Parse | Hardcoded (regex) | Extracts steps into `list[Step]` data structure |
| 3. Execute | LLM (fresh context per step, only relevant tools) | Executes one atomic task at a time |

Key design decisions:
- **Context isolation:** Each step gets a clean window. No accumulated history. Memory between steps is filesystem/git state, not conversation.
- **Hybrid mode:** If API key available, use large model for planning (one cheap call), local model for execution. If no API key, use local model for both with very focused prompts.
- **Tool filtering:** Each step only sees tools relevant to its task type.
- **Aligns with design philosophy:** Parser and orchestrator are 100% hardcoded. Only planner and executor use the LLM.

### Next Steps

1. **Implement Plan-Then-Execute** (`src/core/planner.py`)
   - `TaskDecomposer` — stripped-context planning prompt
   - `PlanParser` — hardcoded regex/structured extraction
   - `ChunkedExecutor` — fresh context per step, tool filtering, result propagation
   - Wire into Agent with auto-detection by model size tier
   - Tests for all three components

2. **Commit pending changes** — GGUF pull enhancement + this GECK restructuring

3. **Systest** — Manual test of `animus rise` with the planner on Llama-3.2-1B

### Checkpoint
**Status:** COMPLETE — Synopsis and plan-then-execute proposal written.

---

## Entry #1 — 2026-02-10

### Summary
Implemented Plan-Then-Execute architecture (`src/core/planner.py`). Three-phase pipeline: TaskDecomposer (LLM) → PlanParser (hardcoded regex) → ChunkedExecutor (fresh context per step). Wired into Agent with auto-detection by model size tier. Added `/plan` slash command for manual override.

### Actions
- Created `src/core/planner.py` with 5 classes: `TaskDecomposer`, `PlanParser`, `ChunkedExecutor`, `PlanExecutor`, `Step`/`StepResult`/`PlanResult` dataclasses
- Hardcoded step type inference from keywords (6 types: read, write, shell, git, analyze, generate)
- Hardcoded tool filtering: each step type maps to a subset of available tools
- Added `Agent.run_planned()` method with auto-detection via `should_use_planner()`
- Added `/plan` toggle slash command in REPL
- REPL auto-routes through planner for small/medium models, direct streaming for large
- Created `tests/test_planner.py` with 50 tests covering all components

### Files Changed
- `src/core/planner.py` — NEW: Plan-Then-Execute pipeline (TaskDecomposer, PlanParser, ChunkedExecutor, PlanExecutor)
- `src/core/agent.py` — Added `run_planned()`, `planning_provider` parameter
- `src/main.py` — Added `/plan` command, auto-detection routing in REPL
- `tests/test_planner.py` — NEW: 50 tests
- `LLM_GECK/tasks.md` — Updated: marked plan-then-execute complete
- `LLM_GECK/log.md` — Added this entry

### Findings
- Step type keyword ordering matters: "write a new test file" was matching SHELL (keyword "test") before WRITE. Fixed by reordering WRITE above SHELL in priority list and narrowing SHELL keyword "run" to "run " (with trailing space).
- The parser handles multiple numbered-list formats: `1.`, `1)`, `1-`, `Step 1:`.
- Single-step fallback when parser can't extract steps prevents pipeline from failing on free-form LLM output.
- Fresh context per step verified via test — no message leakage between steps.

### Metrics
- Files created: 2
- Files modified: 3
- Tests added: 50 (total suite: 212 passing, 0.72s)
- Pipeline phases: 3 (decompose → parse → execute)
- Step types: 6
- Tool filter mappings: 6

### Checkpoint
**Status:** CONTINUE — Implementation complete. Next: systest with Llama-3.2-1B on a real multi-step task.

---

## Appendix A — External Repository Reference

*Carried forward from prior log entries #15, #16, #33. These repos contain patterns and features we want to bring into Animus over time. Ollama has been removed from the stack — Animus uses llama-cpp-python for local inference.*

### Batch 1 — Analyzed 2026-02-01 (Entry #15)

| Repository | Stars | Purpose | Key Value for Animus |
|------------|-------|---------|---------------------|
| **potpie-ai/potpie** | - | Code intelligence with knowledge graphs | Knowledge graph for codebase understanding, multi-agent with specialist roles |
| **firecrawl/firecrawl** | - | Web data extraction for AI | Multi-format output (markdown, HTML, JSON), async job queuing, anti-bot handling |
| **bluewave-labs/Checkmate** | 9K | Infrastructure monitoring | BullMQ distributed job queues, centralized error handling, real-time status |
| **openclaw/openclaw** | - | Personal AI assistant platform | WebSocket gateway, session lanes, tool policies, sandboxing |
| **pranshuparmar/witr** | 12K | Process diagnostics | Cross-platform CLI patterns, hierarchical detection, safe read-only operations |
| **anomalyco/opencode** | 94K | AI coding agent | Permission system (allow/deny/ask), MCP/LSP integration, multi-agent with explore/plan |
| **memvid/memvid** | - | AI memory layer | Smart Frames (append-only), hybrid search (BM25+vector), time-travel debugging |
| **C4illin/ConvertX** | - | File conversion service | Modular converter architecture, 1000+ formats |
| **itsOwen/CyberScraper-2077** | - | AI-powered web scraping | LLM-based content understanding, stealth mode, Tor support |
| **browseros-ai/BrowserOS** | 9K | AI browser agent | Browser as MCP server (31 tools), local-first execution, workflow builder |
| **metorial/metorial** | - | MCP integration platform | 600+ MCP integrations, OAuth session management, monitoring |
| **eigent-ai/eigent** | - | AI workforce desktop | Multi-agent with dynamic task decomposition, MCP tools, human-in-the-loop |
| **mxrch/GHunt** | - | Google OSINT framework | Async architecture, JSON export, browser extension integration |
| **charmbracelet/crush** | - | Agentic CLI | LSP integration, MCP support, session management, agent skills standard |
| **yashab-cyber/nmap-ai** | - | AI-powered network scanning | ML-optimized scanning, natural language interface, multi-format reports |
| **lemonade-sdk/lemonade** | - | Local LLM inference server | OpenAI-compatible API, multi-backend (GGUF, ONNX), hardware abstraction |
| **yashab-cyber/metasploit-ai** | - | AI penetration testing | Intelligent exploit ranking, multi-interface (CLI/Web/Desktop/API) |
| **assafelovic/gpt-researcher** | - | Autonomous research agent | Planner-executor pattern, parallel crawlers, multi-source verification |
| **jesseduffield/lazygit** | - | Git TUI | Undo/redo with reflog, line-level staging, interactive rebasing |
| **janhq/jan** | 40K | Local AI assistant | OpenAI-compatible local API, MCP integration, extension system |
| **QwenLM/Qwen3-Coder** | - | Code generation model | 256K context (1M with Yarn), 358 language support, FIM capability |
| **FlowiseAI/Flowise** | - | Visual AI workflow builder | Drag-and-drop agent design, multi-agent collaboration, RAG integration |
| **aquasecurity/tracee** | 4K | Runtime security | eBPF for kernel introspection, behavioral detection patterns |
| **n8n-io/n8n** | - | Workflow automation | 400+ integrations, hybrid code/visual, LangChain AI integration |
| **logpai/loghub** | - | Log datasets for AI | 16+ datasets for log analysis research, parsing benchmarks |
| **anthropics/claude-code-action** | 5K | GitHub Action for Claude | Context-aware activation, progress tracking, structured outputs |
| **lizTheDeveloper/ai_village** | - | AI simulation game | ECS architecture, 211 systems, multiverse forking, LLM-driven behavior |
| **browseros-ai/moltyflow** | - | Agent Q&A platform | Agent-to-agent collaboration, karma system, auto-expiring questions |
| **Legato666/katana** | - | Web crawling framework | Dual-mode crawling (standard/headless), scope control, resume capability |
| **anthropics/claude-code** | 63K | Agentic CLI | Codebase understanding, git workflow automation, plugin architecture |
| **anthropics/skills** | 60K | Agent skills system | SKILL.md format, dynamic capability extension, production document skills |
| **anthropics/claude-cookbooks** | 32K | Implementation examples | RAG patterns, tool use, sub-agents, vision, prompt caching |

### Batch 2 — Analyzed 2026-02-01 (Entry #16)

| Repository | Purpose | Key Takeaways |
|------------|---------|---------------|
| **anthropic-experimental/sandbox-runtime** | OS-level sandboxing | Hardcoded security boundaries, pattern-based permissions, proxy architecture |
| **VoidenHQ/voiden** | API client with extensions | Hook registry with priorities, IPC-based tool system, state persistence |
| **automazeio/ccpm** | Claude context/project management | File-based context persistence, parallel agent coordination, spec-driven development |
| **mitsuhiko/agent-stuff** | Pi agent extensions | Event-driven lifecycle hooks, TUI components, session state management |
| **badlogic/pi-skills** | Skills for Pi agent | SKILL.md format, self-contained skills, CLI tool abstraction |
| **nicobailon/pi-subagents** | Sub-agent orchestration | Chain execution, parallel fan-out/fan-in, template variables, async job management |
| **stakpak/agent** | DevOps agent | Rust MCP implementation, secret substitution, provider-agnostic design |
| **supermemoryai/supermemory** | AI memory layer | Normalized embeddings, multi-tier relevance fallback, MCP server |
| **assafelovic/skyll** | Skill discovery API | Protocol-based sources, relevance ranking, caching with TTL |
| **oxidecomputer/dropshot** | Rust API framework | Type-safe extractors, trait-based API definitions, OpenAPI generation from code |
| **adenhq/hive** | Outcome-driven agents | Node graphs, self-adapting execution, semantic failure detection |

*Entry #16 also established the core design philosophy (now in tasks.md): "Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else."*

### GECK Report — Analyzed 2026-02-09 (Entry #33, 42 new repos, 84 total)

| Repository | Key Patterns Found | Relevance |
|------------|-------------------|-----------|
| **AntonOsika/gpt-engineer** | Step-based pipeline, ABC+defaults, preprompts system, FilesDict | HIGH |
| **microsoft/agent-lightning** | LightningStore mediator, emit_xxx() telemetry, Agent-Runner-Trainer triangle | HIGH |
| **mikeyobrien/rho** | Extension-based arch, JSONL persistent memory, memory decay, tiered model strategy | HIGH |
| **op7418/CodePilot** | Promise-based permission registry, SSE streaming, SQLite session persistence | MEDIUM |
| **pedramamini/Maestro** | Manager-based arch, three-tier message routing, git worktree orchestration, playbooks | HIGH |
| **openai/codex-action** | Pipeline with safety layers, structured output schema validation | HIGH |
| **pydantic/monty** | Interpreter-as-sandbox, snapshot/resume execution, resource limiting | HIGH |
| **VoltAgent/voltagent** | Registry-based plugins, middleware pipeline with retry, guardrails, model fallback chain | HIGH |
| **openai/skills** | Playwright browser automation skills, bash CLI tools, structured SKILL.md | MEDIUM |
| **OpenBMB/ChatDev** | Multi-agent software development, role-based agents, JSON error hierarchy | MEDIUM |
| **thedotmack/claude-mem** | Progressive disclosure memory, hybrid search (SQLite FTS + ChromaDB), token economics | HIGH |
| **code-yeongyu/oh-my-opencode** | Claude Code extension framework, Sisyphus orchestrator, tmux sub-agents | MEDIUM |
| **ggml-org/llama.cpp** | GBNF grammar-constrained decoding, composable sampling pipeline, ring buffer | CRITICAL |
| **karpathy/nanochat** | Single-dial scaling, tool use via token forcing, safe eval with timeout | HIGH |
| **rasbt/LLMs-from-scratch** | Instruction fine-tuning pipeline, LoRA adapters, DPO alignment | HIGH |
| **tensorzero/tensorzero** | Gateway pattern, variant experimentation, Best-of-N/DICL/Mixture-of-N, feedback flywheel | CRITICAL |
| **NevaMind-AI/memU** | Three-layer memory hierarchy, Protocol-based DI, content-hash dedup | HIGH |
| **KeygraphHQ/shannon** | Narrow agent scoping, parallel sub-agent execution, Promise.allSettled | MEDIUM |
| **langchain-ai/langchain** | Runnable protocol, pipe composition, .with_retry()/.with_fallbacks(), OutputParserException | HIGH |
| **langgenius/dify** | Model provider factory, unified invoke error hierarchy (5 types), DAG workflow engine | HIGH |
| **browser-use/browser-use** | Decorator-based tool registry, dynamic Pydantic Union, action loop detector, capability-tiered prompts | HIGH |
| **Textualize/rich** | Protocol-based polymorphism, recursive rendering pipeline, Live display | MEDIUM |
| **yamadashy/repomix** | Sequential pipeline with DI, Tree-sitter code compression, token counting per file | HIGH |
| **yt-dlp/yt-dlp** | Extractor pattern (auto-routing), importlib plugin discovery, error classification | HIGH |
| **obra/superpowers** | SKILL.md progressive disclosure, task decomposition to 2-5 min units, two-stage review | CRITICAL |
| **ComposioHQ/awesome-claude-skills** | Skill template + scaffolding, validation/packaging pipeline, meta-skill pattern | HIGH |
| **Canner/WrenAI** | Pipeline RAG, @provider decorator + auto-discovery, Jinja2 prompt templates | HIGH |
| **EveryInc/compound-engineering-plugin** | Parallel sub-agent orchestration, knowledge compounding loop, #$ARGUMENTS templates | HIGH |
| **opendatalab/MinerU** | Singleton model cache, multi-backend strategy, hardware-aware model selection | MEDIUM |
| **Shubhamsaboo/awesome-llm-apps** | DeepSeek local RAG agent, fallback chain (RAG→web→direct), thinking process extraction | HIGH |
| **badlogic/pi-mono** | Monorepo with npm workspaces, 7 packages, AGENTS.md for AI contributors | LOW |
| **laude-institute/terminal-bench** | CLI agent benchmarking, ~100 tasks, containerized sandbox | MEDIUM |
| **scikit-learn/scikit-learn** | Estimator pattern (fit/predict), pipeline composition, consistent interface | MEDIUM |
| **govctl-org/govctl** | RFC-driven governance, phase-gated workflow (SPEC→IMPL→TEST→STABLE) | MEDIUM |
| **tobi/qmd** | BM25+vector hybrid search with Reciprocal Rank Fusion (RRF), 800-token chunks | MEDIUM |

*Note: Repos with LOW relevance or no useful patterns omitted. Full analysis in `repor_findings.md`.*

### Feature Priority Matrix (Consolidated)

| Category | Key Features | Sources | Status |
|----------|-------------|---------|--------|
| **Memory & Knowledge** | Knowledge graph for code, hybrid search, SQLite-vec, Smart Frames | potpie, memvid, supermemory, claude-mem | Hybrid search done; graph + SQLite-vec in backlog |
| **Web/Browser** | Browser as MCP server, AI scraping, async crawling | BrowserOS, firecrawl, CyberScraper, katana | Deferred to Ornstein & Smough phase |
| **Multi-Agent** | Specialist agents, planner-executor, task decomposition | opencode, gpt-researcher, eigent, superpowers | Specialist agents done; plan-then-execute next |
| **MCP & Tools** | MCP server/client, 600+ integrations, decorator-based registry | opencode, metorial, browser-use, BrowserOS | MCP done; decorator registry done |
| **Security** | Three-tier permissions, sandbox execution, deny lists | opencode, sandbox-runtime, openclaw | Done (Phase 14) |
| **Context** | Progressive disclosure, compaction, capability-tiered prompts | claude-mem, ccpm, browser-use | Done (Phases 8-9) |
| **Skills** | SKILL.md format, dynamic loading, skill discovery | anthropic/skills, pi-skills, superpowers, skyll | Skills system done (Phase 13) |
| **CLI & UX** | Git TUI features, OpenAI-compatible API, GitHub Action mode | lazygit, jan, lemonade, claude-code-action | API server done; git TUI in backlog |
| **Error Handling** | Parse-retry-correct, unified error types, model fallback chain | langchain, dify, tensorzero, browser-use | All done (Entries #35-46) |
| **Small Model Productivity** | GBNF grammar constraints, atomic task decomposition, code-as-action | llama.cpp, superpowers, monty, nanochat | GBNF partial; plan-then-execute next |
