# LLM INSTRUCTION SET: PROJECT ANIMUS

## 1. Project Manifesto
**Project Name:** Animus
**Goal:** Create a high-performance, cross-platform (Linux/macOS/Windows/Jetson) CLI coding agent.
**Core Philosophy:**
1.  **Local-First:** Prioritize local inference (Ollama, TensorRT-LLM) but support APIs.
2.  **Universal Ingestion:** "Read" everything (Code, PDFs, ZIM archives) via RAG/Indexing, not fine-tuning.
3.  **Orchestration:** Capable of spawning sub-agents for specialized tasks.
4.  **Hardware Aware:** Optimized for edge hardware (Jetson Orin Nano) and standard desktops.

---

## 2. Technology Stack & Constraints
* **Language:** Python 3.10+ (Strict requirement for Jetson JetPack/CUDA compatibility).
* **Dependency Manager:** `uv` (preferred) or `pip` within `venv`.
* **CLI Framework:** `Typer` (commands) + `Rich` (TUI/Formatting).
* **LLM Interface:** Custom abstraction layer supporting:
    * `Ollama` (Standard Local)
    * `TensorRT-LLM` (Jetson Native)
    * `OpenAI-Compatible API` (Cloud fallback)
* **Vector DB:** `ChromaDB` (Persistent, local) or `LanceDB`.
* **Ingestion:** `Tree-sitter` (Code), `PyMuPDF` (PDF), `python-libzim` (ZIM/Wiki), `BeautifulSoup` (HTML).

---

## 3. Architecture & Implementation Phases

### Phase 1: The Core Shell (Skeleton)
**Objective:** Build the CLI entry point and environment detection.
* **Action:** Initialize a Typer app structure.
* **Action:** Implement `animus detect` command to identify OS (Win/Mac/Linux) and Hardware (Jetson/x86/Apple Silicon).
* **Action:** Create the Configuration Manager (`~/.animus/config.yaml`) to store preferred model provider and paths.
* **Constraint:** Ensure all file paths use `pathlib` for Windows compatibility.

### Phase 2: The "Brain" Socket (Model Layer)
**Objective:** Create a unified interface for model inference.
* **Action:** Create a `ModelProvider` abstract base class.
* **Action:** Implement `OllamaProvider` (connects to localhost:11434).
* **Action:** Implement `TRTLLMProvider` (Specific for Jetson: loads engine files).
* **Action:** Implement `APIProvider` (Standard HTTP requests).
* **Key Logic:** On boot, check if the configured model is available. If not, prompt the user to run `animus pull <model>`.
* **Default Model:** "Josified Qwen2.5-Coder" (or similar 7B coding model).

### Phase 3: The Librarian (RAG & Ingestion)
**Objective:** Ingest massive datasets without crashing RAM.
* **Action:** Implement `animus ingest <path>` command.
* **Strategy:**
    1.  **Scanner:** Walk directory tree respecting `.gitignore`.
    2.  **Router:** Identify file type (ext).
    3.  **Extractor:**
        * *ZIM/PDF:* Stream read. **Do not load full file into RAM.** Chunk text into 512-token blocks.
        * *Code:* Parse using Tree-sitter to preserve function scope.
    4.  **Indexer:** Generate embeddings and upsert to Vector DB.
* **Constraint:** Implement a progress bar (Rich) during ingestion.

### Phase 4: The Agentic Loop (Reasoning Engine)
**Objective:** The "Thought -> Act -> Observe" cycle.
* **Action:** Implement the `Agent` class.
* **Tools:**
    * `read_file(path)`
    * `write_file(path, content)`
    * `list_dir(path)`
    * `run_shell(command)` -> **Must include Human-in-the-Loop confirmation for destructive commands (rm, git push).**
* **Context:** Manage a sliding context window. Always keep the System Prompt + Last $N$ turns + Retrieved RAG data.

### Phase 5: The Hive (Sub-Agent Orchestration)
**Objective:** Handle complex tasks via delegation.
* **Action:** Implement a `spawn_subagent(task_description, specialized_role)` function.
* **Logic:**
    1.  Main Agent realizes a task is too broad (e.g., "Refactor this entire module").
    2.  Main Agent defines scope and constraints.
    3.  Sub-Agent initializes with a restricted context (only relevant files).
    4.  Sub-Agent reports back upon completion or failure.

---

## 4. Hardware Specific Optimizations (Jetson)
If `animus detect` identifies **NVIDIA Jetson**:
1.  **Power Mode:** Suggest running `sudo nvpmodel -m 0` (MAXN) or `sudo nvpmodel -m 1` (15W) if unstable.
2.  **Memory:** Suggest mounting ZRAM if RAM < 16GB.
3.  **Quantization:** Default to 4-bit (GGUF or AWQ) models to fit in VRAM.

---

## 5. Development Directory Structure
Structure the codebase as follows:
```text
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
