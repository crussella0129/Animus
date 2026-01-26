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
- [ ] Create ModelProvider abstract base class
- [ ] Implement OllamaProvider (connects to localhost:11434)
- [ ] Implement TRTLLMProvider (Jetson-specific)
- [ ] Implement APIProvider (HTTP requests)
- [ ] Implement `animus pull <model>` command
- [ ] Model availability check on boot

### Phase 3: The Librarian (RAG & Ingestion)
- [ ] Implement `animus ingest <path>` command
- [ ] Scanner: Walk directory respecting .gitignore
- [ ] Router: Identify file type
- [ ] Extractor: ZIM/PDF stream read (512-token chunks)
- [ ] Extractor: Code parsing with Tree-sitter
- [ ] Indexer: Generate embeddings and upsert to Vector DB
- [ ] Rich progress bar during ingestion

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
- [ ] Input validation works properly
- [x] Error messages are clear and actionable
- [ ] LLM Agent can execute commands via the terminal
- [ ] LLM Agent can create code in various languages
- [ ] LLM Agent can create specialized sub-agents

## Backlog

(empty)

## Completed (Recent)

- Phase 1 core implementation (CLI, detect, config, init)
