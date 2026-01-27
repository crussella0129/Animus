# Project: ANIMUS

**Repository:** https://github.com/crussella0129/Animus/tree/main
**Local Path:** C:/Users/charl/Animus
**Created:** 2026-01-26

## Goal

Goal: Create a high-performance, cross-platform (Linux/macOS/Windows/Jetson) CLI coding agent. Core Philosophy:

Local-First: Prioritize local inference (Ollama, TensorRT-LLM) but support APIs.
Self-Contained: Break dependency on external services like Ollama; load and operate models directly using native libraries (llama.cpp, transformers, GGUF).
Universal Ingestion: "Read" everything (Code, PDFs, ZIM archives) via RAG/Indexing, not fine-tuning.
Orchestration: Capable of spawning sub-agents for specialized tasks.
Hardware Aware: Optimized for edge hardware (Jetson Orin Nano) and standard desktops.

## Success Criteria

- [x] All commands execute without errors
- [x] Help text is accurate and complete
- [x] Exit codes are correct
- [x] Input validation works properly
- [x] Error messages are clear and actionable
- [x] LLM Agent can execute commands via the terminal
- [x] LLM Agent can create code (in various languages like Bash, Python, C++, Rust, Javascript, CSS, etc...) that functions and does so according to the original instructions
- [x] LLM Agent can create specialized sub-agents, based in part off of Claude Codes sub-agent structure (View Claude Code Docmentation)
- [x] Animus can load and run GGUF models directly without Ollama installed
- [x] Animus can download models from Hugging Face or other sources
- [x] Native inference works on CPU, CUDA, and Metal backends
- [x] Windows 11 correctly identified (not misreported as Windows 10)
- [~] Agent executes tools autonomously — **Tested 2026-01-27**: Infrastructure works, model compliance varies
- [x] Proper stopping cadences implemented for file operations (confirmation prompts working)
- [ ] Recommended models documented for reliable tool execution

## Constraints

- **Languages/Frameworks:** Python 3.11+
- **Must use:** Type hints, proper exit codes
- **Must avoid:** Hardcoded paths, platform-specific assumptions
- **Target platforms:** Windows, macOS, Linux

## Context

animus/
├── animus/
│   ├── core/           # Main loop, config, detection
│   ├── llm/            # Providers (Ollama, TRT, OpenAI)
│   ├── memory/         # RAG, VectorDB, Ingestion logic
│   ├── tools/          # Filesystem and Shell tools
│   ├── ui/             # Rich TUI definitions
│   └── main.py         # Entry point
├── tests/
├── pyproject.toml
└── README.md

## Initial Task

Architecture & Implementation Phases
Phase 1: The Core Shell (Skeleton)
Objective: Build the CLI entry point and environment detection.

Action: Initialize a Typer app structure.
Action: Implement animus detect command to identify OS (Win/Mac/Linux) and Hardware (Jetson/x86/Apple Silicon).
Action: Create the Configuration Manager (~/.animus/config.yaml) to store preferred model provider and paths.
Constraint: Ensure all file paths use pathlib for Windows compatibility.
Phase 2: The "Brain" Socket (Model Layer)
Objective: Create a unified interface for model inference.

Action: Create a ModelProvider abstract base class.
Action: Implement OllamaProvider (connects to localhost:11434).
Action: Implement TRTLLMProvider (Specific for Jetson: loads engine files).
Action: Implement APIProvider (Standard HTTP requests).
Key Logic: On boot, check if the configured model is available. If not, prompt the user to run animus pull <model>.
Default Model: "Josified Qwen2.5-Coder" (or similar 7B coding model).
Phase 3: The Librarian (RAG & Ingestion)
Objective: Ingest massive datasets without crashing RAM.

Action: Implement animus ingest <path> command.
Strategy:
Scanner: Walk directory tree respecting .gitignore.
Router: Identify file type (ext).
Extractor:
ZIM/PDF: Stream read. Do not load full file into RAM. Chunk text into 512-token blocks.
Code: Parse using Tree-sitter to preserve function scope.
Indexer: Generate embeddings and upsert to Vector DB.
Constraint: Implement a progress bar (Rich) during ingestion.
Phase 4: The Agentic Loop (Reasoning Engine)
Objective: The "Thought -> Act -> Observe" cycle.

Action: Implement the Agent class.
Tools:
read_file(path)
write_file(path, content)
list_dir(path)
run_shell(command) -> Must include Human-in-the-Loop confirmation for destructive commands (rm, git push).
Context: Manage a sliding context window. Always keep the System Prompt + Last 
N
 turns + Retrieved RAG data.
Phase 5: The Hive (Sub-Agent Orchestration)
Objective: Handle complex tasks via delegation.

Action: Implement a spawn_subagent(task_description, specialized_role) function.
Logic:
Main Agent realizes a task is too broad (e.g., "Refactor this entire module").
Main Agent defines scope and constraints.
Sub-Agent initializes with a restricted context (only relevant files).
Sub-Agent reports back upon completion or failure.

Phase 6: Native Model Loading (Self-Contained Inference)
Objective: Eliminate dependency on Ollama; load and run models directly.

Action: Implement NativeProvider using llama-cpp-python for GGUF model support.
Action: Implement model download/management via `animus model download <model>`.
Action: Support automatic model format detection (GGUF, safetensors, etc.).
Action: Implement GPU acceleration detection and configuration (CUDA, Metal, ROCm).
Action: Add fallback chain: Native → Ollama → API (configurable priority).
Key Logic: On boot, check for local model files in ~/.animus/models/. If available, load directly without requiring Ollama service.
Target Models: GGUF quantized models (Q4_K_M, Q5_K_M, Q8_0) for efficient local inference.
Constraint: Must support CPU-only fallback for systems without GPU.

Phase 7: Agent Autonomy & UX Improvements
Objective: Make the agent truly autonomous—execute tools instead of asking user to run commands.

Action: Fix Windows 11 detection (build >= 22000 vs Windows 10).
Action: Update system prompt with explicit JSON tool call format.
Action: Implement autonomous execution policy for read-only operations.
Action: Define stopping cadences (when to pause for user confirmation).

Stopping Cadences (Require Confirmation):
- Authoring new documents and files
- Changing working paths (cd to different project)
- Deleting or editing existing documents
- Identified security issues
- Git push/commit operations
- Installing packages or dependencies

Auto-Execute (No Confirmation):
- Reading files and directories
- Running read-only shell commands (ls, git status, cat, etc.)
- Navigating/exploring codebase

Constraint: Agent must NEVER ask user to copy/paste commands. It executes tools autonomously.