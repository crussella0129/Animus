# Tasks — ANIMUS

**Last Updated:** 2026-01-26

## Legend

- `[ ]` — Not started
- `[x]` — Complete
- `[~]` — In progress
- `[BLOCKED: reason]` — Cannot proceed
- `[DECISION: topic]` — Awaiting human input

## Current Sprint

### Phase 1: The Core Shell (Skeleton)
- [x] Initialize project structure (src/, tests/, pyproject.toml)
- [x] Initialize a Typer app structure
- [x] Implement `animus detect` command (OS/Hardware detection)
- [x] Create Configuration Manager (~/.animus/config.yaml)
- [x] Implement `animus config` command
- [x] Implement `animus init` command
- [x] Add basic tests for detection module
- [x] All paths use pathlib for Windows compatibility

### Phase 2: The "Brain" Socket (Model Layer)
- [x] Create ModelProvider abstract base class
- [x] Implement OllamaProvider (connects to localhost:11434)
- [x] Implement TRTLLMProvider (Jetson-specific placeholder)
- [x] Implement APIProvider (OpenAI-compatible HTTP requests)
- [x] Implement `animus pull <model>` command
- [x] Implement `animus models` command
- [x] Implement `animus status` command
- [x] Add provider factory for unified creation
- [x] Add LLM module tests

### Phase 3: The Librarian (RAG & Ingestion)
- [x] Implement `animus ingest <path>` command
- [x] Implement `animus search <query>` command
- [x] Scanner: Walk directory respecting .gitignore
- [x] Router/Extractor: Identify and extract from file types
- [x] Chunker: Token, Sentence, and Code chunking strategies
- [x] Embedder: Ollama, API, and Mock embedders
- [x] VectorStore: InMemory and ChromaDB support
- [x] Rich progress bar during ingestion
- [x] Add memory module tests (21 new tests)

### Phase 4: The Agentic Loop (Reasoning Engine)
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

### Phase 5: The Hive (Sub-Agent Orchestration)
- [x] Implement SubAgentOrchestrator
- [x] Implement SubAgentScope for restrictions
- [x] Implement SubAgentRole with specialized prompts
- [x] Implement ScopedToolRegistry for tool filtering
- [x] Sub-agent scope restriction (paths, tools)
- [x] Sub-agent reporting mechanism
- [x] Parallel sub-agent execution
- [x] Add sub-agent tests (12 new tests)

### Phase 6: Native Model Loading (Self-Contained Inference)
- [ ] Implement NativeProvider using llama-cpp-python
- [ ] Support GGUF model format loading
- [ ] Implement `animus model download <model>` command
- [ ] Implement `animus model list-local` command
- [ ] Add model storage in ~/.animus/models/
- [ ] Implement GPU backend detection (CUDA, Metal, ROCm)
- [ ] Implement CPU fallback for systems without GPU
- [ ] Add automatic quantization format detection (Q4_K_M, Q5_K_M, Q8_0, etc.)
- [ ] Implement provider fallback chain (Native → Ollama → API)
- [ ] Update config to support native provider settings
- [ ] Add native provider tests
- [ ] Animus runs without Ollama service dependency

### Success Criteria
- [x] All commands execute without errors
- [x] Help text is accurate and complete
- [x] Exit codes are correct
- [x] Input validation works properly
- [x] Error messages are clear and actionable
- [x] LLM Agent can execute commands via the terminal
- [x] LLM Agent can create code in various languages
- [x] LLM Agent can create specialized sub-agents
- [ ] Animus can load and run GGUF models directly without Ollama
- [ ] Animus can download models from Hugging Face
- [ ] Native inference works on CPU, CUDA, and Metal backends

## Backlog

(empty)

## Completed (Recent)

- Phase 5 Sub-Agent Orchestration (roles, scopes, parallel execution)
- Phase 4 Agentic Loop (Agent class, tools, chat command)
- Phase 3 RAG & Ingestion (scanner, chunker, extractor, embedder, vectorstore)
- Phase 2 Model Layer implementation (providers, commands, factory)
- Phase 1 core implementation (CLI, detect, config, init)
