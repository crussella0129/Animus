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
- [ ] Implement Agent class
- [ ] Tool: read_file(path)
- [ ] Tool: write_file(path, content)
- [ ] Tool: list_dir(path)
- [ ] Tool: run_shell(command) with confirmation
- [ ] Sliding context window management

### Phase 5: The Hive (Sub-Agent Orchestration)
- [ ] Implement spawn_subagent(task_description, specialized_role)
- [ ] Sub-agent scope restriction
- [ ] Sub-agent reporting mechanism

### Success Criteria
- [x] All commands execute without errors
- [x] Help text is accurate and complete
- [x] Exit codes are correct
- [x] Input validation works properly
- [x] Error messages are clear and actionable
- [ ] LLM Agent can execute commands via the terminal
- [ ] LLM Agent can create code in various languages
- [ ] LLM Agent can create specialized sub-agents

## Backlog

(empty)

## Completed (Recent)

- Phase 3 RAG & Ingestion (scanner, chunker, extractor, embedder, vectorstore)
- Phase 2 Model Layer implementation (providers, commands, factory)
- Phase 1 core implementation (CLI, detect, config, init)
