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
**Status:** WAIT — Awaiting human approval on plan-then-execute approach before implementation.
