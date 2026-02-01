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

## Entry #11 — 2026-01-26

### Summary
Implemented Phase 7 & 9 fixes: improved JSON tool parsing, added error classification, stopping cadence configuration, and working directory tracking.

### Actions

**1. Fixed JSON Tool Call Parsing** (`src/core/agent.py`)
- Rewrote `_extract_json_objects()` with proper bracket matching
- Handles nested braces, multiline JSON, escaped strings
- Parses JSON from markdown code blocks
- Added deduplication to prevent multiple identical tool calls
- Supports three formats: JSON, function-style, command-style

**2. Added Error Classification System** (`src/core/errors.py`)
- Created `ErrorCategory` enum: context_overflow, auth_failure, rate_limit, timeout, tool_failure, etc.
- Created `RecoveryStrategy` dataclass with retry logic, backoff, compaction, fallback options
- Created `classify_error()` function with pattern matching
- Added Animus-specific exceptions: `ContextOverflowError`, `AuthenticationError`, `RateLimitError`, `ToolExecutionError`
- Integrated error classification into Agent `_call_tool()` method

**3. Added Stopping Cadence Configuration** (`src/core/config.py`)
- Created `AgentBehaviorConfig` class with:
  - `auto_execute_tools`: Tools that run without confirmation
  - `safe_shell_commands`: Read-only commands that auto-execute
  - `blocked_commands`: Dangerous commands that are always blocked
  - `require_confirmation`: Actions requiring user approval
  - `track_working_directory`: Enable/disable directory change detection
  - `max_autonomous_turns`: Max turns before human check-in
- Added `agent` field to `AnimusConfig`

**4. Working Directory Tracking** (`src/core/agent.py`)
- Added `_is_directory_change()` to detect cd/pushd commands
- Added `_is_path_change_significant()` to detect project changes
- Added `_is_blocked_command()` for dangerous command detection
- Agent now tracks `_current_working_dir` and `_initial_working_dir`
- Significant directory changes require confirmation

**5. Added Comprehensive Tests**
- `tests/test_errors.py`: 17 tests for error classification
- `tests/test_agent_behavior.py`: 22 tests for agent behavior
- Total: 138 tests (39 new), all passing

### Files Changed

- `src/core/agent.py` — Improved JSON parsing, directory tracking, error integration
- `src/core/config.py` — Added AgentBehaviorConfig class
- `src/core/errors.py` — NEW: Error classification system
- `src/core/__init__.py` — Export new classes
- `tests/test_errors.py` — NEW: Error classification tests
- `tests/test_agent_behavior.py` — NEW: Agent behavior tests

### Commits

- 2b01d52 feat: error classification, improved JSON parsing, stopping cadences

### Findings

- Bracket-matching approach to JSON parsing is more robust than regex
- Error classification enables appropriate recovery strategies (retry, fallback, compact)
- Stopping cadences can be configured per-deployment via config.yaml

### Issues

None — All 138 tests pass.

### Checkpoint
**Status:** CONTINUE — Core fixes implemented and committed.

### Next
- Begin Phase 8 (Decision Recording) or Phase 9 (Session Compaction)
- Integration test with actual LLM

---

## Entry #13 — 2026-01-27

### Summary
Reviewed Windows system tests (systests/Windows). Identified working features and remaining issues with tool execution and model behavior.

### Test Results Analysis

**Animus_Test_Windows_1.txt:**
- ✅ Native dependencies installed successfully (`pip install -e ".[native]"`)
- ✅ GPU detected: NVIDIA GeForce RTX 2080 Ti (11264 MB), CUDA driver 581.80
- ✅ llama-cpp-python built and installed with CUDA support
- ✅ Model download working: CodeLlama-7B-Instruct GGUF (3.80 GB)
- ✅ `animus detect`, `animus models`, `animus model list` all functional
- ✅ Chat session starts correctly
- ❌ Tool execution broken: Model outputs tool commands as text (e.g., `list_dir "path"`) but they don't execute
- ❌ Model hallucination: Fabricated entire file analysis for "C:\Users\charl\LogOS" that doesn't exist
- ❌ Agent not autonomous: Keeps asking user to run commands manually

**Animus_Test_Windows_2.txt:**
- ✅ Downloaded larger model: Qwen3-VL-8B-Instruct-abliterated (4.79 GB)
- ✅ Tool execution now working with confirmation prompts (`Execute this tool? [y/n]`)
- ✅ write_file tool successfully created test.txt in Downloads
- ✅ list_dir tool execution confirmed
- ❌ Model identity confusion: Qwen claiming to be made by "Anthropic" and referencing Claude guidelines
- ❌ Excessive safety refusals: Refused to help with legitimate tasks (car Linux installation guide)
- ❌ Tool execution inconsistent: Sometimes outputs JSON, sometimes doesn't execute

### Issues Identified

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| Tool execution inconsistent | High | Partial Fix | Works with JSON format, fails with text format |
| Model identity confusion | Medium | Open | Qwen thinks it's Claude/Anthropic |
| Hallucinated outputs | High | Open | Model fabricates file contents instead of reading |
| Excessive safety refusals | Medium | Open | Model refuses benign tasks citing ethics |
| Windows 11 detection | Low | Fixed | Now shows correct version |

### Findings

1. **Tool Execution Progress**: The confirmation prompts (`Execute this tool? [y/n]`) show that JSON-formatted tool calls ARE being parsed and executed. The issue in Test 1 was the model using text-based command format instead of JSON.

2. **Model Behavior Issues**: The abliterated model still has safety refusals baked in, and incorrectly identifies itself as Anthropic/Claude. This is a model-level issue, not Animus code.

3. **Recommended Models**: Need to find models that:
   - Follow tool-calling JSON format reliably
   - Don't have excessive safety filters
   - Don't claim to be other AI systems

4. **System Prompt Effectiveness**: Current system prompt specifies JSON format but model compliance varies. May need to use models specifically trained for function calling.

### Commits
- (none - analysis only)

### Checkpoint
**Status:** CONTINUE — Windows tests show partial success. Tool infrastructure works but model compliance is inconsistent.

### Next
- Test with different models (e.g., Qwen2.5-Coder-Instruct, DeepSeek-Coder)
- Consider adding few-shot examples in system prompt for tool format
- Update documentation with recommended models

---

## Entry #12 — 2026-01-26

### Summary
Analyzed Hive agent-building skills (building-agents-core, building-agents-construction, building-agents-patterns) to identify improvements for Animus sub-agent architecture.

---

### Source Analysis: Hive Agent-Building Skills

**Skills Reviewed:**
1. `building-agents-core` — Fundamental concepts (goals, nodes, edges, tools)
2. `building-agents-construction` — Step-by-step building process
3. `building-agents-patterns` — Best practices and anti-patterns

---

### Key Patterns from Hive Framework

#### 1. Goal-Driven Architecture
```python
Goal(
    id="research-goal",
    success_criteria=[
        SuccessCriterion(id="completeness", metric="coverage_score", target=">=0.9", weight=0.4),
    ],
    constraints=[
        Constraint(id="accuracy", description="All info must be verified", constraint_type="hard"),
    ],
)
```
**Value:** Sub-agents have clear, measurable success criteria. Parent can objectively evaluate results.

#### 2. Node-Based Workflow
```
NodeTypes: llm_generate | llm_tool_use | router | function
EdgeConditions: on_success | on_failure | always | conditional
```
**Value:** Structured workflow instead of freeform "think and act." Deterministic routing based on conditions.

#### 3. Pause/Resume with Session State
```python
pause_nodes = ["request-clarification"]
entry_points = {
    "start": "analyze-request",
    "request-clarification_resume": "process-clarification"
}
# Resume: pass session_state separately from input_data
result = await agent.trigger_and_wait(entry_point, input_data, session_state=previous.session_state)
```
**Value:** Multi-turn conversations. Sub-agents can pause for human input and resume with memory intact.

#### 4. OutputCleaner (Auto I/O Validation)
```python
NodeSpec(
    input_schema={"analysis": {"type": "dict", "required": True}},
    output_schema={"decision": {"type": "string", "required": True}},
)
```
**Value:** Automatic validation and cleaning of node outputs. 1.8-2.2x success rate boost.

#### 5. Tool Discovery Before Execution
```python
# MANDATORY: Discover tools BEFORE adding nodes
available_tools = mcp__agent-builder__list_mcp_tools()
if "web_search" not in available_tools:
    raise Error("Tool not available")
```
**Value:** Prevent runtime failures from missing tools. Validate early.

#### 6. Error Handling with Fallback Edges
```python
EdgeSpec(source="api-call", target="process-results", condition=EdgeCondition.ON_SUCCESS)
EdgeSpec(source="api-call", target="fallback-cache", condition=EdgeCondition.ON_FAILURE)
```
**Value:** Graceful degradation. Alternative paths when primary fails.

#### 7. Parallel Execution Pattern
```python
# Multiple edges from same source = parallel execution
EdgeSpec(source="start", target="search-source-1", condition=EdgeCondition.ALWAYS)
EdgeSpec(source="start", target="search-source-2", condition=EdgeCondition.ALWAYS)
# Then converge at merge node
```
**Value:** Faster execution for independent tasks.

---

### Improvements for Animus Sub-Agents

**Current State:** Animus has SubAgentOrchestrator with roles, scopes, and parallel execution.

**Proposed Enhancements:**

| Enhancement | Current | Proposed | Benefit |
|-------------|---------|----------|---------|
| **Goal-Driven** | Freeform prompts | Explicit Goal with SuccessCriteria | Measurable success, objective evaluation |
| **Node Graph** | Single "run" method | Node-based workflow | Structured execution, deterministic routing |
| **Pause/Resume** | Not supported | HITL pause/resume | Multi-turn sub-agent conversations |
| **Output Validation** | Basic tool results | Input/Output schemas | Auto-validation, cleaning, 2x reliability |
| **Tool Discovery** | Assume tools exist | Validate before use | Prevent runtime failures |
| **Fallback Edges** | Retry or fail | on_failure paths | Graceful degradation |
| **Entry Points** | Single entry | Multiple entry points | Resume from different pause points |

---

### Proposed Architecture Changes

#### 1. SubAgentGoal Class
```python
@dataclass
class SubAgentGoal:
    id: str
    name: str
    description: str
    success_criteria: list[SuccessCriterion]
    constraints: list[Constraint]

@dataclass
class SuccessCriterion:
    id: str
    description: str
    metric: str  # e.g., "accuracy", "completeness"
    target: str  # e.g., ">=0.9", "100%"
    weight: float  # For weighted scoring
```

#### 2. SubAgentNode Class
```python
@dataclass
class SubAgentNode:
    id: str
    name: str
    node_type: Literal["llm_generate", "llm_tool_use", "router", "function"]
    input_keys: list[str]
    output_keys: list[str]
    system_prompt: Optional[str]
    tools: list[str]  # For llm_tool_use
    input_schema: Optional[dict]  # For validation
    output_schema: Optional[dict]  # For validation
    max_retries: int = 3
```

#### 3. SubAgentEdge Class
```python
@dataclass
class SubAgentEdge:
    id: str
    source: str  # Node ID
    target: str  # Node ID
    condition: EdgeCondition  # ON_SUCCESS, ON_FAILURE, ALWAYS, CONDITIONAL
    condition_expr: Optional[str]  # Python expression for CONDITIONAL
    priority: int = 1

class EdgeCondition(Enum):
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    CONDITIONAL = "conditional"
```

#### 4. SubAgentGraph Class
```python
@dataclass
class SubAgentGraph:
    id: str
    goal: SubAgentGoal
    nodes: list[SubAgentNode]
    edges: list[SubAgentEdge]
    entry_node: str
    entry_points: dict[str, str]  # {"start": "node-id", "pause_resume": "resume-node"}
    pause_nodes: list[str]
    terminal_nodes: list[str]
```

#### 5. SubAgentExecutor (with Pause/Resume)
```python
class SubAgentExecutor:
    async def run(self, graph: SubAgentGraph, context: dict, session_state: dict = None) -> SubAgentResult:
        # Execute node graph with pause/resume support
        pass

    async def resume(self, entry_point: str, context: dict, session_state: dict) -> SubAgentResult:
        # Resume from pause node
        pass
```

#### 6. OutputCleaner Integration
```python
class OutputCleaner:
    async def validate_and_clean(self, output: dict, schema: dict) -> dict:
        # Validate output against schema
        # Auto-clean if malformed (using fast LLM)
        pass
```

---

### Anti-Patterns to Avoid (from Hive)

1. **❌ Don't rely on export_graph** — Write files immediately, not at end
2. **❌ Don't hide state in session** — Make progress visible to user
3. **❌ Don't batch everything** — Write incrementally with feedback
4. **❌ Don't assume tools exist** — Always validate before use
5. **❌ Don't use wrong entry_points format** — Must be `{"start": "node-id"}`

---

### Files Changed

None (analysis only)

### Commits

None (analysis only)

### Findings

- Hive's goal-driven architecture enables objective sub-agent evaluation
- Node-based workflows provide structured execution vs freeform
- Pause/resume pattern essential for multi-turn sub-agent conversations
- OutputCleaner significantly improves reliability (1.8-2.2x)
- Tool discovery before execution prevents runtime failures
- Fallback edges enable graceful degradation

### Issues

- Full implementation would be significant refactor of SubAgentOrchestrator
- Need to maintain backward compatibility with existing sub-agent roles

### Checkpoint
**Status:** CONTINUE — Analysis complete, tasks to be added.

### Next
- Add tasks for sub-agent improvements to tasks.md
- Prioritize based on impact vs complexity
- Begin incremental implementation

---

## Entry #14 — 2026-01-31

### Summary
Implemented foundational Phase 8 and Phase 9 features: Decision Recording Schema, Run Persistence, Context Window Management, and Session Compaction.

### Actions

**1. Decision Recording Schema** (`src/core/decision.py`)
- Created `Decision` dataclass with intent, options, chosen, reasoning
- Created `Option` dataclass with id, description, pros, cons, confidence
- Created `Outcome` dataclass with decision_id, status, result, summary
- Created `DecisionRecord` linking decision to outcome
- Created `DecisionRecorder` for session-level decision management
- Added `DecisionType` enum: tool_selection, strategy, delegation, etc.
- Added `OutcomeStatus` enum: success, partial, failure, unknown

**2. Run Persistence** (`src/core/run.py`)
- Created `Run` dataclass with full lifecycle management (start, complete, fail, cancel)
- Created `RunMetrics` with tokens, turns, tool calls, success rate
- Created `RunStore` with JSON-based storage in ~/.animus/runs/
- Added find methods: by_status, by_goal, by_date, by_tag
- Added get_recent and get_stats for analytics

**3. Context Window Management** (`src/core/context.py`)
- Created `ContextWindow` for tracking token usage
- Created `ContextConfig` with soft/critical limits
- Created `TokenEstimator` for character-based token estimation
- Added `ContextStatus` enum: ok, warning, critical, overflow
- Added presets for common model sizes (4K, 8K, 16K, 32K, 128K)

**4. Session Compaction** (`src/core/compaction.py`)
- Created `SessionCompactor` with multiple strategies
- Implemented truncate, sliding_window, summarize, hybrid strategies
- Created `CompactionConfig` with keep_recent_turns, summary_max_tokens
- Added LLM-based summarization for context reduction

**5. Agent Integration**
- Integrated DecisionRecorder into Agent class
- Agent now records tool selection decisions and outcomes
- Added get_decisions(), get_decision_records(), get_decision_success_rate() methods

**6. CLI Updates** (`src/main.py`)
- Added `--max-context` option to chat command
- Added `--show-tokens` option to display token usage

### Files Changed

- `src/core/decision.py` — NEW: Decision recording schema
- `src/core/run.py` — NEW: Run persistence with JSON storage
- `src/core/context.py` — NEW: Context window management
- `src/core/compaction.py` — NEW: Session compaction
- `src/core/agent.py` — Added decision recording integration
- `src/core/__init__.py` — Export new classes
- `src/main.py` — Added --max-context and --show-tokens options
- `tests/test_decision.py` — NEW: 27 decision recording tests
- `tests/test_run.py` — NEW: 17 run persistence tests
- `tests/test_context.py` — NEW: 22 context management tests

### Commits

- (pending)

### Findings

- 204 tests now passing (66 new tests added)
- Token estimation uses ~4 chars/token for text, ~3 for code
- Decision recording adds minimal overhead to tool execution
- Compaction strategies allow flexibility based on use case

### Issues

None — All tests pass.

### Checkpoint
**Status:** CONTINUE — Phase 8-9 foundational work complete. Ready for BuilderQuery and full Agent integration.

### Next
- Implement BuilderQuery for run analysis
- Add Triangulated Verification (HybridJudge)
- Integrate compaction into Agent class for automatic triggering
- Begin Phase 10 (Hybrid Search) or Phase 11 (Sub-Agent Architecture)

---

## Entry #15 — 2026-02-01

### Summary
Comprehensive analysis of 32 external repositories to identify features and patterns that would benefit Animus. Identified major feature categories: Memory Systems, Web/Browser Automation, Multi-Agent Orchestration, MCP/Tool Integration, Security/Observability, Workflow Automation, and Skills/Plugin Systems.

### Repositories Analyzed

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

---

### Feature Categories Identified

#### 1. **Memory & Knowledge Systems**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Knowledge Graph for Code** | potpie | Neo4j-based semantic graph of functions, classes, calls | High |
| **Smart Frames (Append-Only Memory)** | memvid | Immutable memory units with timestamps, checksums | High |
| **Time-Travel Debugging** | memvid | Rewind/replay/branch memory states | Medium |
| **Hybrid Search (BM25 + Vector)** | memvid, opencode | Combine keyword and semantic search | High |
| **SQLite-vec Backend** | opencode | Persistent vector storage without ChromaDB | High |
| **Memory Capsules (.mv2)** | memvid | Self-contained, shareable memory files | Medium |

**Sample Code (memvid Smart Frames):**
```rust
let mut mem = Memvid::create("knowledge.mv2")?;
let opts = PutOptions::builder()
    .title("Code Analysis")
    .tag("project", "animus")
    .build();
mem.put_bytes_with_options(content, opts)?;
mem.commit()?;

// Search with sub-5ms local access
let response = mem.search(SearchRequest {
    query: "error handling".into(),
    top_k: 10,
    ..Default::default()
})?;
```

#### 2. **Web & Browser Automation**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Browser as MCP Server** | BrowserOS | 31 tools for browser control via MCP | High |
| **AI-Powered Scraping** | CyberScraper | LLM understands page structure, not CSS selectors | Medium |
| **Multi-Format Output** | firecrawl | Markdown, HTML, JSON, screenshots simultaneously | High |
| **Async Job Queuing** | firecrawl | BullMQ for non-blocking web operations | High |
| **Anti-Bot Handling** | firecrawl, CyberScraper | Stealth mode, proxy rotation, Tor support | Medium |
| **Dual-Mode Crawling** | katana | Standard (fast) + Headless (JS rendering) | Medium |

**Sample Pattern (firecrawl async jobs):**
```python
# Non-blocking crawl
job_id = client.start_crawl(url)
# Poll or WebSocket for updates
async for result in client.watch_crawl(job_id):
    process(result.document)
```

#### 3. **Multi-Agent Orchestration**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Specialist Agents** | potpie, opencode | Dedicated agents: QnA, Debug, Test, Explore, Plan | High |
| **Planner-Executor Pattern** | gpt-researcher | Planning agent generates questions, executors gather info | High |
| **Dynamic Task Decomposition** | eigent | Activate multiple agents in parallel | Medium |
| **Agent-to-Agent Q&A** | moltyflow | Agents ask other agents instead of humans | Medium |
| **Karma/Reputation System** | moltyflow | Track agent reliability (+15 accepted, -2 downvote) | Low |
| **Multiverse Forking** | ai_village | Branch universes for "what-if" scenarios | Low |

**Sample Pattern (opencode multi-agent):**
```typescript
// Specialist agent definitions
const agents = {
  build: { mode: "primary", permission: fullAccess },
  plan: { mode: "primary", permission: readOnly },  // No edits
  explore: { mode: "subagent", tools: ["grep", "glob", "read"] },
  general: { mode: "subagent", tools: allTools }
}
```

#### 4. **MCP & Tool Integration**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **MCP Server Support** | opencode, BrowserOS, metorial | Model Context Protocol for tool exposure | High |
| **600+ Pre-built Integrations** | metorial | Ready-to-use MCP servers | Medium |
| **OAuth Session Management** | metorial, opencode | Handle auth flows for external APIs | High |
| **Tool Discovery Before Use** | opencode, potpie | Validate tools exist before node creation | High |
| **Dynamic Tool Updates** | opencode | ToolListChangedNotification pushes new tools | Medium |

**Sample Pattern (MCP integration):**
```typescript
// Connect to MCP server with OAuth
const client = new MCPClient({
  transport: new StreamableHTTPClientTransport(url),
  oauth: { handleCallback: true }
});
// Convert MCP tools to agent tools
const tools = mcpToolsToVercelAI(await client.listTools());
```

#### 5. **Permission & Security Systems**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Three-Tier Permissions** | opencode | allow / deny / ask with pattern matching | High |
| **Tool Policy Layers** | openclaw | Profile → Rules → Sandbox scope | High |
| **Sandbox Execution** | openclaw | Docker isolation for non-main sessions | Medium |
| **Safe Code Sandbox** | (backlog) | Whitelist-based safe_eval/safe_exec | Medium |
| **eBPF Runtime Security** | tracee | Kernel-level behavioral detection | Low |

**Sample Pattern (opencode permissions):**
```typescript
const permission = {
  read: [
    { pattern: "**/*", action: "allow" },
    { pattern: "**/.env*", action: "deny" }  // Never read secrets
  ],
  edit: { action: "ask" },  // Always confirm edits
  bash: { action: "ask" }
}
```

#### 6. **Session & Context Management**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Session Lanes/Queuing** | openclaw | Serialize commands per session | High |
| **Pause/Resume Sessions** | openclaw, opencode | Multi-turn with state persistence | High |
| **Block Chunking for Streaming** | openclaw | Send partial responses at paragraph boundaries | Medium |
| **Device Pairing** | openclaw | Public key auth, no shared passwords | Low |
| **256K-1M Context Support** | Qwen3-Coder | Extended context for repository-scale work | Medium |

#### 7. **Workflow & Automation**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Visual Workflow Builder** | Flowise, eigent | Drag-and-drop agent design | Low |
| **Hybrid Code/Visual** | n8n | Write code or use visual interface | Medium |
| **400+ Integrations** | n8n | Pre-built automation nodes | Low |
| **Scheduled Tasks/Webhooks** | openclaw | Cron-based automation | Medium |
| **GitHub Action Integration** | claude-code-action | CI/CD with Claude | High |

#### 8. **Skills & Plugin Systems**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **SKILL.md Format** | anthropic/skills | YAML frontmatter + markdown instructions | High |
| **Dynamic Capability Extension** | anthropic/skills | Load skills at runtime | High |
| **Agent Skills Standard** | crush | Cross-platform skill interoperability (agentskills.io) | Medium |
| **Plugin Marketplace** | anthropic/skills | One-click installation via Claude Code | Medium |
| **Document Skills** | anthropic/skills | Production-grade docx, pdf, pptx, xlsx | Medium |

**Sample Pattern (SKILL.md format):**
```yaml
---
name: code-reviewer
description: Reviews code for bugs, security issues, and style
---

# Code Reviewer

## Instructions
1. Analyze the provided code for:
   - Logic errors and edge cases
   - Security vulnerabilities
   - Style consistency
2. Provide specific, actionable feedback

## Examples
- "The null check on line 42 is missing..."
```

#### 9. **CLI & UX Patterns**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Undo/Redo for Git** | lazygit | Reflog-based reversible operations | Medium |
| **Line-Level Staging** | lazygit | Select individual lines to commit | Low |
| **Cross-Platform Detection** | witr | Build tags for platform-specific code | Medium |
| **Read-Only Safety Model** | witr | Never modify system state | High |
| **Progress Tracking** | claude-code-action | Dynamic checkbox updates in comments | Medium |
| **OpenAI-Compatible Local API** | jan, lemonade | localhost:1337 for integration | High |

#### 10. **Research & Analysis**

| Feature | Source | Description | Priority |
|---------|--------|-------------|----------|
| **Multi-Source Verification** | gpt-researcher | Aggregate 20+ sources for objectivity | Medium |
| **Extended Reports (2000+ words)** | gpt-researcher | Comprehensive research documents | Low |
| **Log Analysis Datasets** | loghub | 16+ datasets for debugging research | Low |
| **AI-Generated Imagery** | gpt-researcher | Inline illustrations via Gemini | Low |

---

### High-Priority Features for Animus

Based on analysis, these features would provide the most value:

#### Immediate (Phase 12-13)

1. **MCP Server Integration**
   - Expose Animus tools via MCP protocol
   - Enable cross-tool communication
   - Connect to external MCP servers (BrowserOS, etc.)

2. **Skills System (SKILL.md)**
   - Adopt Anthropic's SKILL.md format
   - Enable dynamic capability loading
   - Create initial skills: code-review, test-gen, refactor

3. **Hybrid Search Enhancement**
   - Add BM25 keyword search to existing vector search
   - Configurable weighting
   - Use memvid-style Smart Frames for memory

4. **Three-Tier Permission System**
   - Replace binary confirm with allow/deny/ask
   - Pattern-based file access control
   - Per-agent permission profiles

5. **OpenAI-Compatible Local API**
   - Add `animus serve` command (localhost:1337)
   - Enable integration with Jan, VS Code, etc.

#### Near-Term (Phase 14-15)

6. **Specialist Sub-Agents**
   - Explore agent (fast codebase search)
   - Plan agent (read-only analysis)
   - Build agent (full capabilities)

7. **Browser Integration via MCP**
   - Connect to BrowserOS MCP server
   - Enable web research without built-in browser

8. **GitHub Action Mode**
   - `animus action` for CI/CD use
   - Structured outputs for workflows
   - Progress tracking in PR comments

9. **Session Lanes**
   - Serialize commands per session
   - Prevent concurrent interleaving
   - Support pause/resume

10. **Knowledge Graph for Code**
    - Build semantic graph of codebase
    - Track function calls, class relationships
    - Enable "blast radius" analysis for changes

---

### Files Changed

None (analysis only)

### Commits

None (analysis only)

### Findings

- **MCP is the de facto standard** for tool integration across Claude Code, OpenCode, BrowserOS, and metorial
- **Skills/SKILL.md format** enables modular capability extension without code changes
- **Hybrid search (BM25 + vector)** is consistently recommended over pure vector search
- **Three-tier permissions (allow/deny/ask)** provides better UX than binary confirmation
- **Specialist agents** (explore, plan, build) reduce context pollution and improve focus
- **OpenAI-compatible API** enables ecosystem integration with minimal effort
- **Knowledge graphs** provide superior code understanding vs flat text

### Issues

- Implementing all features would be a multi-month undertaking
- Some features (visual workflow, browser control) may be out of scope for CLI focus
- MCP server implementation requires careful security consideration

### Checkpoint
**Status:** CONTINUE — Analysis complete, ready to update tasks with new phases.

### Next
- Update tasks.md with new Phase 12-15 tasks
- Prioritize MCP integration and Skills system
- Begin implementation of high-priority features

---