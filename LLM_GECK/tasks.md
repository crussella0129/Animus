# Tasks — ANIMUS

**Last Updated:** 2026-02-10 (Log v2, Entry #0: Synopsis + Plan-Then-Execute proposal)

## Design Philosophy

**Core Principle:** Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else.

| Task Type | Approach | Examples |
|-----------|----------|----------|
| Security | 100% Hardcoded | Permissions, deny lists, sandboxing |
| Protocols | 100% Hardcoded | MCP, HTTP, JSON-RPC |
| Validation | 100% Hardcoded | Schema, types, patterns |
| File I/O | 100% Hardcoded | Skill loading, context persistence |
| Orchestration | 100% Hardcoded | Plan parsing, step sequencing, tool filtering |
| Search | Hybrid (70/30) | BM25 hardcoded, LLM reranking optional |
| Context | Hybrid (80/20) | Truncation hardcoded, summarization LLM |
| Planning | LLM | Task decomposition (focused prompt) |
| Execution | LLM | Single-step task execution |

## Legend

- `[ ]` — Not started
- `[x]` — Complete
- `[~]` — In progress
- `[BLOCKED: reason]` — Cannot proceed
- `[DECISION: topic]` — Awaiting human input

---

## Completed (Prior Phases 1-17, Entries #0-50)

All prior phases archived. See `Previous GECK Logs/log_entries_0-50.md` for full history.

Summary of completed work:
- Phase 1: Core Shell (CLI, detection, config)
- Phase 2: Model Layer (providers, factory, pull)
- Phase 3: RAG Pipeline (scanner, chunker, embedder, vectorstore)
- Phase 4: Agentic Loop (agent, tools, chat)
- Phase 5: Sub-Agent Orchestration (roles, scopes, parallel)
- Phase 6: Native Model Loading (GGUF, llama-cpp-python)
- Phase 7: Agent Autonomy & UX
- Phase 8: Self-Improvement (decisions, runs, BuilderQuery, HybridJudge)
- Phase 9: Context Resilience (compaction, error classification)
- Phase 10: Enhanced Retrieval (hybrid search, tree-sitter chunking)
- Phase 11: Sub-Agent Graph Architecture (goal-driven workflows)
- Phase 12: MCP Integration (server + client)
- Phase 13: Skills System (SKILL.md format)
- Phase 14: Enhanced Permissions (3-tier, profiles, scopes)
- Phase 15: OpenAI-Compatible Local API (REST + WebSocket)
- Phase 16: Code Hardening Audit
- Phase 16.5: Web Search Security (Ungabunga-Box)
- Phase 17: Audio Interface (praise, speech, MIDI)
- Plus: Installation system, GBNF grammar, media pipeline, DICL, feedback flywheel, code-as-action sandbox, lane-based queueing, specialist agents, auth rotation, action loop detection, model fallback chain, parse-retry-correct, capability-tiered prompts, progressive disclosure RAG, knowledge compounding, decorator-based tool registry, unified error translation

**Total prior tests:** 727+

---

## Completed (Lean Rebuild — This Sprint)

### Session Layer Rebuild ✓
- [x] **Phase 5a: Session Persistence** — `src/core/session.py`, save/load/resume, `animus sessions`, auto-save on exit (14 tests)
- [x] **Phase 5b: Streaming Output** — `generate_stream()` ABC + OpenAI/Anthropic SSE + Agent `run_stream()` with on_chunk callback
- [x] **Phase 5c: Tool Confirmation Wiring** — Rich-based `[y/N]` callback, respects `confirm_dangerous` config
- [x] **Phase 5d: REPL Slash Commands** — `/save`, `/clear`, `/tools`, `/tokens`, `/help`, provider/model display on start
- [x] **Phase 6: Git Tools** — 7 tools (status, diff, log, branch, add, commit, checkout), blocked ops (force-push, reset --hard, clean -f, branch -D) (24 tests)
- [x] **Phase 7: Logging & Token Tracking** — Rotating file handler, `log_llm_call()`, `log_tool_execution()`, `TokenUsage` dataclass, `_cumulative_tokens` (17 tests)
- [x] **GGUF Pull Enhancement** — MODEL_CATALOG (6 models), `download_gguf()`, Rich progress bar, `--list` flag, auto-config after download

**Lean rebuild tests:** 212 passing (0.72s)

---

## Current Sprint

### Plan-Then-Execute Architecture [COMPLETE]

**Goal:** Enable small local models (1-7B) to handle complex multi-step tasks by decomposing them into atomic steps with isolated context.

**Solution:** Three-phase pipeline with hardcoded orchestration in `src/core/planner.py`.

**Tasks:**
- [x] **TaskDecomposer** (`src/core/planner.py`)
  - [x] Stripped planning prompt (no tools, no history, just "break this into steps")
  - [x] Support hybrid mode: API model for planning, local model for execution
  - [x] Step output format: numbered list with step type tags
- [x] **PlanParser** (`src/core/planner.py`)
  - [x] Hardcoded regex extraction of numbered steps
  - [x] Parse into `list[Step]` dataclass (description, step_type, relevant_tools)
  - [x] Step types: read, write, shell, git, analyze, generate
- [x] **ChunkedExecutor** (`src/core/planner.py`)
  - [x] Fresh context per step (no accumulated history)
  - [x] Tool filtering: only tools relevant to step type
  - [x] Result propagation: filesystem/git state IS the memory between steps
  - [x] Progress reporting (step N of M)
  - [x] Error handling: continues after failure, per-step status tracking
- [x] **Agent Integration**
  - [x] Auto-detect: use chunked execution for small/medium models, direct for large
  - [x] `agent.run_planned()` with force flag and auto-detection
  - [x] Manual override: `/plan` slash command to toggle plan mode
- [x] **Tests** (`tests/test_planner.py`) — 50 tests
  - [x] StepType inference (10 tests)
  - [x] PlanParser regex extraction (12 tests)
  - [x] TaskDecomposer with mock provider (3 tests)
  - [x] Tool filtering (5 tests)
  - [x] ChunkedExecutor with mock provider and tools (8 tests)
  - [x] PlanExecutor full pipeline integration (5 tests)
  - [x] should_use_planner auto-detection (3 tests)
  - [x] _parse_tool_calls standalone (4 tests)

---

## Backlog

### High Priority
- [ ] **Knowledge Graph for Code** (from Potpie) — Neo4j/SQLite code graph, function/class relationships, blast radius analysis
- [ ] Grammar-constrained decoding for llama-cpp-python (GBNF integration)

### Medium Priority
- [ ] Real TTS Integration (replace MIDI phoneme synthesis)
- [ ] Moto Perpetuo background music (proper MIDI files)
- [ ] SQLite-vec persistent vector store
- [ ] Container isolation with `--paranoid` flag (Ungabunga-Box Phase 3)
- [ ] API key authentication for MCP HTTP transport

### Lower Priority
- [ ] Browser Control via MCP (BrowserOS)
- [ ] Multi-Channel Support (WhatsApp, Telegram, Slack)
- [ ] Visual Workflow Builder
- [ ] Extended Context Support (256K-1M)
- [ ] Git TUI Features (undo/redo, line-level staging)
- [ ] GitHub Action Mode (`animus action`)

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Small models thrash on complex tasks | High | Mitigated by Plan-Then-Execute (auto-detected) |
| Model outputs text instead of JSON tool calls | High | Mitigated by parse-retry-correct + GBNF (partial) |
| Token estimation heuristic (~4 chars/token) inaccurate for code | Medium | Open |
| Shell tool uses `shell=True` | Medium | Mitigated by permission checker |

---

## Success Criteria

### Plan-Then-Execute (Implementation Complete — Needs Systest)
- [ ] Llama-3.2-1B can complete a multi-file task via chunked execution
- [x] Each step executes in isolated context without thrashing (verified by test_fresh_context_per_step)
- [x] Hybrid mode works: API plans, local executes (verified by test_hybrid_mode_different_providers)
- [x] Fallback to direct mode for large/API models (verified by test_large_model_skips_planner)
- [x] Tests cover decomposer, parser, executor, and integration (50 tests passing)
