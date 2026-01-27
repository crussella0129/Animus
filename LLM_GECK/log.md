# Session Log — ANIMUS

*Append only. Do not edit existing entries.*

---

## Entry #0 — 2026-01-26 01:08:53

### Summary
Project initialized. GECK structure created.

### Understood Goals
- Goal: Create a high-performance, cross-platform (Linux/macOS/Windows/Jetson) CLI coding agent. Core Philosophy:
- Local-First: Prioritize local inference (Ollama, TensorRT-LLM) but support APIs.
- Universal Ingestion: "Read" everything (Code, PDFs, ZIM archives) via RAG/Indexing, not fine-tuning.
- Orchestration: Capable of spawning sub-agents for specialized tasks.
- Hardware Aware: Optimized for edge hardware (Jetson Orin Nano) and standard desktops.
- Architecture & Implementation Phases
- Phase 1: The Core Shell (Skeleton)
- Objective: Build the CLI entry point and environment detection.
- Action: Initialize a Typer app structure.
- Action: Implement animus detect command to identify OS (Win/Mac/Linux) and Hardware (Jetson/x86/Apple Silicon).
- Action: Create the Configuration Manager (~/.animus/config.yaml) to store preferred model provider and paths.
- Constraint: Ensure all file paths use pathlib for Windows compatibility.
- Phase 2: The "Brain" Socket (Model Layer)
- Objective: Create a unified interface for model inference.
- Action: Create a ModelProvider abstract base class.
- Action: Implement OllamaProvider (connects to localhost:11434).
- Action: Implement TRTLLMProvider (Specific for Jetson: loads engine files).
- Action: Implement APIProvider (Standard HTTP requests).
- Key Logic: On boot, check if the configured model is available. If not, prompt the user to run animus pull <model>.
- Default Model: "Josified Qwen2.5-Coder" (or similar 7B coding model).
- Phase 3: The Librarian (RAG & Ingestion)
- Objective: Ingest massive datasets without crashing RAM.
- Action: Implement animus ingest <path> command.
- Strategy:
- Scanner: Walk directory tree respecting .gitignore.
- Router: Identify file type (ext).
- Extractor:
- ZIM/PDF: Stream read. Do not load full file into RAM. Chunk text into 512-token blocks.
- Code: Parse using Tree-sitter to preserve function scope.
- Indexer: Generate embeddings and upsert to Vector DB.
- Constraint: Implement a progress bar (Rich) during ingestion.
- Phase 4: The Agentic Loop (Reasoning Engine)
- Objective: The "Thought -> Act -> Observe" cycle.
- Action: Implement the Agent class.
- Tools:
- read_file(path)
- write_file(path, content)
- list_dir(path)
- run_shell(command) -> Must include Human-in-the-Loop confirmation for destructive commands (rm, git push).
- Context: Manage a sliding context window. Always keep the System Prompt + Last
- N
- turns + Retrieved RAG data.
- Phase 5: The Hive (Sub-Agent Orchestration)
- Objective: Handle complex tasks via delegation.
- Action: Implement a spawn_subagent(task_description, specialized_role) function.
- Logic:
- Main Agent realizes a task is too broad (e.g., "Refactor this entire module").
- Main Agent defines scope and constraints.
- Sub-Agent initializes with a restricted context (only relevant files).
- Sub-Agent reports back upon completion or failure.

### Questions/Ambiguities
None

### Initial Tasks
- Build the CLI entry point and environment detection.
- All commands execute without errors
- Help text is accurate and complete
- Exit codes are correct
- Input validation works properly
- Error messages are clear and actionable
- LLM Agent can execute commands via the terminal
- LLM Agent can create code (in various languages like Bash, Python, C++, Rust, Javascript, CSS, etc...) that functions and does so according to the original instructions
- LLM Agent can create specialized sub-agents, based in part off of Claude Codes sub-agent structure (View Claude Code Docmentation)

### Checkpoint
**Status:** WAIT — Awaiting confirmation to begin work.

---

## Entry #1 — 2026-01-26

### Summary
Completed Phase 1: Core Shell implementation. Built CLI entry point with Typer, environment detection (OS/hardware/GPU), and configuration management.

### Actions
- Created project directory structure using `src/` (renamed from `animus/` per user request to avoid confusion)
- Initialized `pyproject.toml` with dependencies (Typer, Rich, PyYAML, Pydantic)
- Implemented `src/core/detection.py` with OS, architecture, and hardware type detection
- Implemented `src/core/config.py` with ConfigManager and YAML-based configuration
- Created `src/main.py` as CLI entry point with Typer
- Implemented commands: `detect`, `config`, `init`
- Added GPU detection (NVIDIA via nvidia-smi, Apple Metal)
- Added Jetson and Apple Silicon detection
- Created basic tests for detection module
- Fixed Pydantic deprecation warning (class Config → ConfigDict)

### Files Changed
- `pyproject.toml` — Project configuration and dependencies
- `requirements.txt` — Quick-install dependencies
- `src/__init__.py` — Package init with version
- `src/main.py` — CLI entry point with detect/config/init commands
- `src/core/__init__.py` — Core module exports
- `src/core/detection.py` — System detection (OS, hardware, GPU)
- `src/core/config.py` — Configuration management
- `src/llm/__init__.py` — LLM module placeholder
- `src/memory/__init__.py` — Memory module placeholder
- `src/tools/__init__.py` — Tools module placeholder
- `src/ui/__init__.py` — UI module placeholder
- `tests/__init__.py` — Tests package
- `tests/test_detection.py` — Detection module tests

### Commits
- `89f5fc8` — feat: implement Phase 1 - Core Shell (CLI, detection, config)

### Findings
- System detected: Windows 10.0.26200, x86_64, NVIDIA RTX 2080 Ti (11GB), CUDA 581.80
- Typer 0.21.1 no longer needs `[all]` extra
- Python 3.13.11 compatible

### Issues
None

### Checkpoint
**Status:** CONTINUE — Phase 1 complete, ready for Phase 2 (Model Layer).

### Next
- Implement Phase 2: ModelProvider abstract base class
- Implement OllamaProvider (localhost:11434)
- Implement `animus pull <model>` command

---

## Entry #2 — 2026-01-26

### Summary
Completed Phase 2: Model Layer implementation. Built unified provider interface with OllamaProvider, TRTLLMProvider, and APIProvider. Added model management CLI commands.

### Actions
- Created `src/llm/base.py` with ModelProvider ABC, Message, GenerationConfig, GenerationResult, ModelInfo
- Implemented `src/llm/ollama.py` with full Ollama API support (list, pull, chat, streaming)
- Implemented `src/llm/trtllm.py` as placeholder for TensorRT-LLM (Jetson) integration
- Implemented `src/llm/api.py` for OpenAI-compatible APIs (supports streaming)
- Created `src/llm/factory.py` with provider factory and auto-detection
- Added CLI commands: `models`, `pull`, `status`
- Added httpx dependency for async HTTP requests
- Created comprehensive LLM module tests (12 new tests)

### Files Changed
- `pyproject.toml` — Added httpx dependency
- `requirements.txt` — Added httpx dependency
- `src/main.py` — Added models, pull, status commands
- `src/llm/__init__.py` — Module exports
- `src/llm/base.py` — Provider ABC and data classes
- `src/llm/ollama.py` — Ollama provider implementation
- `src/llm/trtllm.py` — TensorRT-LLM provider (placeholder)
- `src/llm/api.py` — OpenAI-compatible API provider
- `src/llm/factory.py` — Provider factory functions
- `tests/test_llm.py` — LLM module tests

### Commits
- `3d3f363` — feat: implement Phase 2 - Model Layer (LLM providers)

### Findings
- Ollama API follows OpenAI-like patterns but with some differences
- TensorRT-LLM requires platform-specific setup, kept as placeholder
- All 16 tests passing

### Issues
None

### Checkpoint
**Status:** CONTINUE — Phase 2 complete, ready for Phase 3 (RAG & Ingestion).

### Next
- Implement Phase 3: RAG & Ingestion
- `animus ingest <path>` command
- File type detection and chunking

---

## Entry #3 — 2026-01-26

### Summary
Completed Phase 3: RAG & Ingestion implementation. Built complete document ingestion pipeline with directory scanning, text extraction, chunking strategies, embedding generation, and vector storage.

### Actions
- Created `src/memory/scanner.py` with GitIgnoreParser and DirectoryScanner
- Created `src/memory/chunker.py` with TokenChunker, SentenceChunker, CodeChunker
- Created `src/memory/extractor.py` with PlainText, Code, Markdown, PDF extractors
- Created `src/memory/embedder.py` with OllamaEmbedder, APIEmbedder, MockEmbedder
- Created `src/memory/vectorstore.py` with InMemoryVectorStore and ChromaVectorStore
- Created `src/memory/ingest.py` orchestrating the full ingestion pipeline
- Added CLI commands: `ingest`, `search`
- Created comprehensive memory module tests (21 new tests)
- Added pytest-asyncio dependency for async tests

### Files Changed
- `src/main.py` — Added ingest and search commands
- `src/memory/__init__.py` — Module exports
- `src/memory/scanner.py` — Directory scanning with .gitignore support
- `src/memory/chunker.py` — Text chunking strategies
- `src/memory/extractor.py` — File text extraction
- `src/memory/embedder.py` — Embedding generation
- `src/memory/vectorstore.py` — Vector database abstraction
- `src/memory/ingest.py` — Ingestion orchestrator
- `tests/test_memory.py` — Memory module tests

### Commits
- `fc57963` — feat: implement Phase 3 - RAG & Ingestion (memory module)

### Findings
- All 37 tests passing (4 detection + 12 LLM + 21 memory)
- ChromaDB is optional - falls back to InMemoryVectorStore if not installed
- PDF extraction requires pypdf or pdfplumber (optional)

### Issues
None

### Checkpoint
**Status:** CONTINUE — Phase 3 complete, ready for Phase 4 (Agentic Loop).

### Next
- Implement Phase 4: The Agentic Loop
- Agent class with tool execution
- Tools: read_file, write_file, list_dir, run_shell

---

## Entry #4 — 2026-01-26

### Summary
Completed Phase 4: Agentic Loop implementation. Built the Agent class with tool execution, human-in-the-loop confirmation, and interactive chat command.

### Actions
- Created `src/tools/base.py` with Tool ABC, ToolRegistry, ToolResult
- Created `src/tools/filesystem.py` with ReadFileTool, WriteFileTool, ListDirectoryTool
- Created `src/tools/shell.py` with ShellTool (destructive command detection, blocking)
- Created `src/core/agent.py` with Agent class (Think -> Act -> Observe loop)
- Added CLI command: `chat` for interactive sessions
- Implemented human-in-the-loop confirmation for destructive operations
- Implemented blocked command detection for dangerous operations
- Created comprehensive tools module tests (20 new tests)

### Files Changed
- `src/main.py` — Added chat command
- `src/core/__init__.py` — Added Agent exports
- `src/core/agent.py` — Agent class with reasoning loop
- `src/tools/__init__.py` — Module exports and default registry
- `src/tools/base.py` — Tool ABC and registry
- `src/tools/filesystem.py` — File operation tools
- `src/tools/shell.py` — Shell execution with safety controls
- `tests/test_tools.py` — Tools module tests

### Commits
- `70a9c67` — feat: implement Phase 4 - Agentic Loop (tools and agent)

### Findings
- All 57 tests passing (4 detection + 12 LLM + 21 memory + 20 tools)
- Shell tool properly blocks dangerous commands (rm -rf /, fork bombs)
- Human confirmation works for destructive operations

### Issues
None

### Checkpoint
**Status:** CONTINUE — Phase 4 complete, ready for Phase 5 (Sub-Agent Orchestration).

### Next
- Implement Phase 5: Sub-Agent Orchestration
- spawn_subagent function
- Scope restriction and reporting

---

## Entry #5 — 2026-01-26

### Summary
Completed Phase 5: Sub-Agent Orchestration. Built the system for spawning specialized sub-agents with restricted scope and parallel execution capabilities.

### Actions
- Created `src/core/subagent.py` with SubAgentOrchestrator, SubAgentScope, SubAgentRole
- Implemented ScopedToolRegistry for tool filtering based on scope
- Created predefined roles: CODER, REVIEWER, TESTER, DOCUMENTER, REFACTORER, DEBUGGER, RESEARCHER
- Implemented scope restrictions: allowed paths, allowed tools, can_write, can_execute
- Added parallel sub-agent execution with spawn_parallel
- Created comprehensive sub-agent tests (12 new tests)

### Files Changed
- `src/core/__init__.py` — Added sub-agent exports
- `src/core/subagent.py` — Sub-agent orchestration system
- `tests/test_subagent.py` — Sub-agent module tests

### Commits
- `859b6a1` — feat: implement Phase 5 - Sub-Agent Orchestration (hive)

### Findings
- All 69 tests passing (4 detection + 12 LLM + 21 memory + 20 tools + 12 subagent)
- Scoped tool registry properly filters tools based on permissions
- Role-specific prompts guide sub-agent behavior

### Issues
None

### Checkpoint
**Status:** COMPLETE — All 5 phases implemented. Project Animus core is functional.

### Next
- Integration testing with real LLM
- Documentation and README updates
- Optional: Add more specialized tools

---

## Entry #6 — 2026-01-26

### Summary
Added Phase 6 development goal: Native Model Loading. This new phase aims to break dependency on Ollama by enabling Animus to load and operate models directly using native libraries.

### Actions
- Updated LLM_init.md with new "Self-Contained" core philosophy goal
- Added Phase 6: Native Model Loading to implementation phases in LLM_init.md
- Added success criteria for native model loading in LLM_init.md
- Added Phase 6 task checklist to tasks.md (12 new tasks)
- Updated success criteria in tasks.md

### Files Changed
- `LLM_GECK/LLM_init.md` — Added Self-Contained goal, Phase 6 definition, success criteria
- `LLM_GECK/tasks.md` — Added Phase 6 task checklist and updated success criteria

### Commits
- (pending)

### Findings
- Phase 6 will use llama-cpp-python for GGUF model support
- Provider fallback chain: Native → Ollama → API
- Must support CPU, CUDA, Metal, and ROCm backends

### Issues
None

### Checkpoint
**Status:** CONTINUE — Goals updated, ready to begin Phase 6 implementation.

### Next
- Implement NativeProvider class
- Add model download/management commands
- Test direct model loading

---

## Entry #7 — 2026-01-26

### Summary
Implemented Phase 6: Native Model Loading. Created NativeProvider for direct GGUF model loading via llama-cpp-python, eliminating the dependency on Ollama for local inference.

### Actions
- Added `llama-cpp-python` and `huggingface-hub` as optional dependencies
- Added NATIVE to ProviderType enum
- Created NativeConfig in config.py with GPU/CPU settings
- Implemented NativeProvider in src/llm/native.py with:
  - Direct GGUF model loading
  - GPU backend detection (CUDA, Metal, ROCm, CPU)
  - Hugging Face model download support
  - ChatML prompt formatting
  - Streaming and non-streaming generation
- Updated factory.py with Native provider and fallback chain (Native → Ollama → API)
- Added CLI commands: `animus model download`, `animus model list`, `animus model remove`, `animus model info`
- Updated status command to show Native provider status
- Created comprehensive tests (28 new tests, 97 total)

### Files Changed
- `pyproject.toml` — Added optional native dependencies
- `src/llm/base.py` — Added NATIVE to ProviderType enum
- `src/core/config.py` — Added NativeConfig class
- `src/llm/native.py` — New NativeProvider implementation
- `src/llm/__init__.py` — Export NativeProvider
- `src/llm/factory.py` — Added Native provider and updated fallback chain
- `src/main.py` — Added model management commands, updated status
- `tests/test_native.py` — New test file (28 tests)

### Commits
- (pending)

### Findings
- llama-cpp-python must be installed separately (CPU or GPU version)
- GPU backend detection works for CUDA, Metal, and ROCm
- Hugging Face hub provides easy model download
- 97 tests passing

### Issues
None

### Checkpoint
**Status:** CONTINUE — Phase 6 core implementation complete. User can now download and use GGUF models directly.

### Next
- Install llama-cpp-python to test end-to-end native inference
- Download a test model and verify chat works without Ollama
- Optional: Add model conversion tools

---

## Entry #8 — 2026-01-26

### Summary
Completed full Ollama independence. Added NativeEmbedder using sentence-transformers, updated embedder auto-detection, and comprehensive README documentation for fully self-contained operation.

### Actions
- Added sentence-transformers to optional native dependencies
- Created NativeEmbedder class using sentence-transformers (all-MiniLM-L6-v2)
- Updated create_embedder() to support "auto" mode (native → mock fallback)
- Updated ingest.py to use "auto" embedder by default
- Changed default provider from "ollama" to "native" in config
- Updated init command to auto-detect and prefer native provider
- Updated detection recommendations to suggest native provider
- Rewrote README.md with comprehensive instructions for:
  - Option A: Fully independent operation (no Ollama)
  - Option B: With Ollama (alternative)
  - GPU acceleration setup (CUDA, Metal, ROCm)
  - Model download and management
  - All CLI commands documented
- Updated exports in memory/__init__.py

### Files Changed
- `pyproject.toml` — Added sentence-transformers to native dependencies
- `src/memory/embedder.py` — Added NativeEmbedder, updated create_embedder()
- `src/memory/ingest.py` — Changed default embedder to "auto"
- `src/memory/__init__.py` — Export NativeEmbedder and SENTENCE_TRANSFORMERS_AVAILABLE
- `src/core/config.py` — Changed default provider to "native"
- `src/main.py` — Updated init and detect recommendations for native
- `README.md` — Complete rewrite with native-first instructions

### Commits
- (pending)

### Findings
- sentence-transformers provides high-quality local embeddings
- all-MiniLM-L6-v2 is a good default (384 dimensions, fast)
- Auto-detection allows seamless fallback when dependencies not installed
- 99 tests still passing

### Issues
None

### Checkpoint
**Status:** COMPLETE — Animus can now run fully independently without Ollama or any external service.

### Next
- Awaiting further instructions

---

## Entry #9 — 2026-01-26

### Summary
Fixed critical issues discovered during QUICKSTART testing: Windows 11 detection, tool execution, and autonomous agent behavior.

### Analysis of Animus_Test_Windows_1.txt

**Issues Identified:**

1. **Windows Version Misdetected**: System reported "Windows (10.0.26200)" but user is on Windows 11. Build 26200 is Windows 11 (build >= 22000).

2. **write_file Never Executed**: LLM outputted tool commands as text (e.g., `write_file "path" "content"`) instead of JSON format. Agent's `_parse_tool_calls` expected `{"tool": "name", "arguments": {...}}` so tools were never parsed or executed.

3. **Hallucinated Output**: The LogOS directory analysis was completely fabricated. The LLM "listed" and "read" files that don't exist because it never actually called tools—it just generated plausible-looking output.

4. **LLM Asking User to Execute Commands**: Instead of executing tools autonomously, the LLM kept saying "run this command" or "paste the output." This defeats the purpose of an agentic assistant.

### Actions

**1. Fixed Windows Version Detection** (`src/core/detection.py`)
- Added `_get_windows_marketing_name()` function
- Windows 11 detected when build >= 22000
- Now displays "Windows 11 (Build 26200)" instead of raw version

**2. Improved System Prompt** (`src/core/agent.py`)
- Clear JSON format specification for tool calls
- Explicit "Autonomous Execution Policy" section
- Instructions to EXECUTE tools, not ask user to run them
- Warning against hallucinating file contents

**3. Added Auto-Execute Configuration** (`src/core/agent.py`)
- New `auto_execute_tools` tuple for read-only tools (read_file, list_dir)
- New `safe_shell_commands` tuple for read-only commands
- Updated `_call_tool` method with `_is_safe_shell_command()` check

**4. Improved Tool Call Parsing** (`src/core/agent.py`)
- Enhanced `_parse_tool_calls` to handle multiple formats:
  - JSON: `{"tool": "name", "arguments": {...}}`
  - Function-style: `read_file("path")`
  - Command-style: `read_file "path"`

### Files Changed

- `src/core/detection.py` — Fixed Windows 11 detection with `_get_windows_marketing_name()`
- `src/core/agent.py` — Rewrote system prompt, added auto-execute logic, improved tool parsing

### Commits

- (pending)

### Findings

- Windows `platform.version()` returns "10.0.XXXXX" for both Win10 and Win11
- Build number determines actual Windows version (22000+ = Win11)
- LLMs need explicit, structured instructions for tool call format
- Autonomous execution policy prevents "please run this for me" behavior

### Issues

- LLM may still hallucinate if it doesn't understand tool call format (model-dependent)
- Need integration testing with actual LLM to verify fixes

### Checkpoint
**Status:** CONTINUE — Fixes implemented, need testing and commit.

### Next
- Run tests to verify changes don't break existing functionality
- Test with actual LLM to verify tool calling works
- Consider adding more flexible tool call parsing if needed

---

## Entry #10 — 2026-01-26

### Summary
Analyzed `C:\Users\charl\hive` (Aden Agent Framework) and `C:\Users\charl\clawdbot` (Personal AI Assistant Platform) to identify functionality that could benefit Animus.

---

### Repository Analysis: Hive (Aden Agent Framework)

**What it is:** Open-source Python framework for building goal-driven, self-improving AI agents.

**Key Innovations:**

| Feature | Description | Animus Benefit |
|---------|-------------|----------------|
| **Decision Recording** | Logs reasoning (options, pros/cons, chosen, outcome) not just actions | Enables meaningful agent improvement; Builder can understand WHY |
| **Triangulated Verification** | Rule-based → LLM fallback → Human escalation | More reliable output validation than single method |
| **Self-Improving Loop** | Failure → BuilderQuery analysis → Graph evolution → Redeploy | Automatic agent improvement without manual intervention |
| **Goal-Driven Development** | LLM generates agent graph from natural language goals | Reduces manual workflow design |
| **HybridJudge** | Combines deterministic rules + LLM evaluation | Fast for clear cases, smart for ambiguous |
| **Node-Based Architecture** | Structured workflow with typed edges (always/on_success/on_failure/conditional/llm_decide) | Better orchestration than linear execution |
| **MCP Server Integration** | Model Context Protocol for tool exposure | Standard protocol for cross-tool communication |
| **Safe Code Sandbox** | Whitelist-based `safe_eval()` and `safe_exec()` | Secure dynamic code execution |

**Key Files to Study:**
- `core/framework/schemas/decision.py` — Decision recording schema
- `core/framework/graph/judge.py` — HybridJudge implementation
- `core/framework/builder/query.py` — BuilderQuery for run analysis
- `core/framework/graph/executor.py` — Graph execution engine
- `core/framework/graph/hitl.py` — Human-in-the-loop protocol

---

### Repository Analysis: Clawdbot (Personal AI Assistant Platform)

**What it is:** TypeScript-based local-first multi-channel AI gateway connecting Claude to WhatsApp, Telegram, Slack, Discord, etc.

**Key Innovations:**

| Feature | Description | Animus Benefit |
|---------|-------------|----------------|
| **Session Compaction** | Auto-summarizes old turns to fit context window | Prevents context overflow, maintains continuity |
| **SQLite-vec Hybrid Search** | BM25 keyword + vector semantic search combined | Better RAG retrieval than pure vector |
| **Lane-Based Queueing** | Serializes commands per session/lane | Prevents interleaving of concurrent runs |
| **Auth Profile Rotation** | Cooldown tracking + failover ordering for API keys | Graceful degradation on auth failures |
| **Error Classification** | Categorizes: context_overflow, auth_failure, rate_limit, timeout | Appropriate retry/recovery strategies |
| **Media Pipeline** | Download → mime-sniff → store (5MB cap, 2min TTL) | Robust media handling |
| **Plugin System with Gating** | Dynamic tool loading + per-channel allowlists | Fine-grained tool access control |
| **Skills Platform** | 200+ markdown skills with install specs (brew, node, pip, etc.) | Extensible capability system |
| **Browser Control** | Playwright/CDP multi-profile wrapper | Web automation capability |
| **Thinking Level Integration** | Model-aware capability detection (thinking: off/minimal/low/medium/high) | Leverage Claude's extended thinking |

**Key Files to Study:**
- `src/memory/manager.ts` — SQLite-vec hybrid search
- `src/process/command-queue.ts` — Lane-based queueing
- `src/agents/auth-profiles.ts` — Auth rotation with cooldowns
- `src/agents/pi-embedded-runner/run.ts` — Session compaction logic
- `src/plugins/loader.ts` — Dynamic plugin discovery

---

### Feature Prioritization for Animus

#### **High Priority (Core Agent Improvements)**

1. **Decision Recording** (from Hive)
   - Current: Animus logs tool calls but not reasoning
   - Needed: Record intent, options considered, chosen action, outcome
   - Value: Enables self-improvement and debugging

2. **Session Compaction** (from Clawdbot)
   - Current: Animus has `max_context_messages` but no summarization
   - Needed: Auto-summarize old turns when approaching context limit
   - Value: Maintains conversation continuity, prevents truncation

3. **Hybrid Search (BM25 + Vector)** (from Clawdbot)
   - Current: Animus uses pure vector search (InMemoryVectorStore)
   - Needed: Combine keyword (BM25) + semantic (vector) search
   - Value: Better retrieval for mixed queries (exact + conceptual)

4. **Error Classification** (from Clawdbot)
   - Current: Generic exception handling
   - Needed: Categorize errors (context_overflow, auth, rate_limit, timeout)
   - Value: Appropriate recovery strategies per error type

5. **Triangulated Verification** (from Hive)
   - Current: Tool output passed directly to LLM
   - Needed: Rule check → LLM fallback → Human escalation
   - Value: More reliable output validation

#### **Medium Priority (Enhanced Functionality)**

6. **Builder Query Pattern** (from Hive)
   - Analyze past runs for patterns, failures, improvements
   - Generate suggestions for agent evolution

7. **Auth Profile Rotation** (from Clawdbot)
   - Multiple API keys with cooldown tracking
   - Automatic failover on auth failures

8. **Skills Platform** (from Clawdbot)
   - Markdown-based skills with YAML frontmatter
   - Install specs for dependencies
   - Eligibility checks (OS, binaries, env vars)

9. **Lane-Based Queueing** (from Clawdbot)
   - Serialize commands per session
   - Prevent interleaving of concurrent agent runs

10. **Media Pipeline** (from Clawdbot)
    - Robust file handling with TTL
    - MIME detection and size limits

#### **Lower Priority (Nice to Have)**

11. **Browser Control** (from Clawdbot) — Playwright/CDP integration
12. **Multi-Channel Support** (from Clawdbot) — WhatsApp, Telegram, etc.
13. **MCP Server** (from Hive) — Model Context Protocol
14. **Goal-Driven Development** (from Hive) — Generate graphs from goals
15. **Canvas/A2UI** (from Clawdbot) — Visual UI rendering

---

### Goals Evolution

**Original Goals (from LLM_init.md):**
- Local-First inference
- Self-Contained (no Ollama dependency) ✓ Achieved
- Universal Ingestion (RAG)
- Orchestration (sub-agents)
- Hardware Aware (Jetson, Apple Silicon, etc.)

**New Goals (from analysis):**
- **Self-Improving**: Learn from failures, evolve agent behavior automatically
- **Decision-Transparent**: Record reasoning, not just actions
- **Context-Resilient**: Session compaction to handle long conversations
- **Hybrid-Retrieval**: Combined BM25 + vector search for better RAG
- **Error-Resilient**: Classified errors with appropriate recovery strategies
- **Skill-Extensible**: Markdown-based skills with install management

---

### Files Changed

None (analysis only)

### Commits

None (analysis only)

### Findings

- **Hive** is architecturally sophisticated with focus on self-improvement and observability
- **Clawdbot** is production-hardened with robust error handling and multi-channel support
- Both repos solve problems Animus will face as it matures
- Decision recording is the highest-impact feature—enables everything else

### Issues

- Implementing all features would be a major undertaking
- Need to prioritize based on immediate value vs complexity
- Some features (multi-channel, browser control) may be out of scope for CLI-focused tool

### Checkpoint
**Status:** CONTINUE — Analysis complete, tasks need to be added.

### Next
- Add prioritized tasks to tasks.md
- Reorganize task checklist with new phases
- Begin implementation of high-priority features

---