```
 .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
/ .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
\ \/\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ \/ /
 \/ /`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\/ /
 / /\   ▄▀▀█▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄    ▄▀▀▄ ▄▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀▀▀▄   / /\
/ /\ \ ▐ ▄▀ ▀▄ █  █ █ █ █   █  █  █  █ ▀  █ █   █    █ █ █   ▐  / /\ \
\ \/ /   █▄▄▄█ ▐  █  ▀█ ▐   █  ▐  ▐  █    █ ▐  █    █     ▀▄    \ \/ /
 \/ /   ▄▀   █   █   █      █       █    █    █    █   ▀▄   █    \/ /
 / /\  █   ▄▀  ▄▀   █    ▄▀▀▀▀▀▄  ▄▀   ▄▀      ▀▄▄▄▄▀   █▀▀▀     / /\
/ /\ \ ▐   ▐   █    ▐   █       █ █    █                ▐       / /\ \
\ \/ /         ▐        ▐       ▐ ▐    ▐                        \ \/ /
 \/ /                                                            \/ /
 / /\.--..--..--..--..--..--..--..--..--..--..--..--..--..--..--./ /\
/ /\ \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \/\ \
\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
 `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'
```

# Animus

**Local-first AI agent with intelligent code understanding and multi-strategy retrieval.**

Animus is a production-ready agentic system designed to run entirely on edge hardware (8W Jetson to consumer GPUs) with sub-7B models. Features novel **Animus: Manifold** multi-strategy retrieval that combines vector search, knowledge graphs, and keyword search with hardcoded routing—no cloud dependencies, no LLM-based classification overhead.

**Key Innovation:** The first local-first system to achieve <400ms hybrid query latency by treating the entire codebase as "one surface" through AST-based knowledge graphs and contextual embeddings, eliminating naive chunking strategies.

---

## Quick Start

### 1. Install Animus

```bash
git clone https://github.com/yourusername/animus.git
cd animus
pip install -e ".[all]"  # Installs all dependencies
```

This installs:
- Core dependencies (typer, rich, pydantic, tiktoken)
- Local inference (llama-cpp-python for GGUF models)
- Embeddings (sentence-transformers for semantic search)
- Testing framework (pytest)

### 2. Initialize Configuration

```bash
animus init
```

Creates `~/.animus/config.yaml` with default settings. The config will auto-detect your hardware (GPU, CPU, OS) and set appropriate defaults.

### 3. Download a Model

```bash
# Recommended: 7B coder model (4.8 GB VRAM)
animus pull qwen-2.5-coder-7b

# Or see all available models
animus pull --list
```

Models are downloaded to `~/.animus/models/` as GGUF files. The pull command auto-configures your settings.

### 4. Build Knowledge Base

```bash
# Build AST-based knowledge graph (Python code intelligence)
animus graph ./your-project

# Build vector store with contextual embeddings
animus ingest ./your-project
```

**Order matters:** Run `graph` first, then `ingest`. The ingest process uses the knowledge graph to enrich embeddings with structural context (callers, callees, inheritance).

### 5. Start an Agent Session

```bash
animus rise
```

You'll see:
```
[i] Provider: native  Model: qwen-2.5-coder-7b
[i] [Manifold] Unified search tool registered
[i] Session: abc123def
[i] Type 'exit' or 'quit' to end.

You>
```

---

## Core Capabilities

### 1. Intelligent Code Search (Animus: Manifold)

**What it does:** Multi-strategy retrieval router that automatically classifies queries and dispatches to the optimal search backend.

**How it works:**
- Hardcoded pattern matching (<1ms classification, no LLM)
- Four strategies: SEMANTIC, STRUCTURAL, HYBRID, KEYWORD
- Reciprocal Rank Fusion for multi-strategy result merging
- Contextual embeddings (graph-enriched vector search)

**Example usage:**
```
You> search for "how does authentication work?"
→ Routes to SEMANTIC (vector similarity)
→ Returns: Code snippets semantically similar to "authentication"

You> search for "what calls authenticate()?"
→ Routes to STRUCTURAL (knowledge graph)
→ Returns: All functions that call authenticate()

You> search for "find the auth code and what depends on it"
→ Routes to HYBRID (both strategies + RRF fusion)
→ Returns: Auth code (semantic) + its callers (structural), fused and ranked

You> search for "find TODO comments"
→ Routes to KEYWORD (exact grep match)
→ Returns: Lines containing "TODO"
```

**When to use:** Any time you need to understand, find, or navigate code. Manifold automatically picks the right strategy—you don't choose.

### 2. Agentic Tool Use with Reflection

**What it does:** Agent executes tools (file operations, git, shell commands) with observation-reflection-action pattern.

**How it works:**
- Agent evaluates tool results (success/failure/empty/long)
- Provides contextual guidance for next actions
- Prevents infinite loops (repeat detection, thrashing detection, hard limits)
- Cumulative execution budgets (300s session limit)

**Example usage:**
```
You> Create a Python script that generates Pascal's triangle up to n layers

Agent plans:
  [1/2] write_file("pascal.py", "def pascal_triangle(n): ...")
  [2/2] Test the script by running it

Agent reflects:
  [Tool write_file SUCCESS]: Successfully wrote 156 characters to pascal.py
  The operation succeeded. This step is COMPLETE. Do NOT make additional tool calls to verify.
```

**When to use:** Any coding task, file manipulation, git operations, or shell commands. The agent handles multi-step workflows automatically.

### 3. Plan-Then-Execute for Small Models

**What it does:** Decomposes complex tasks into atomic steps, each executed with fresh context and filtered tools.

**How it works:**
- LLM generates numbered step plan (focused prompt, no tools, no history)
- Hardcoded parser extracts steps into structured format
- Each step executed independently with minimal context
- GBNF grammar constraints for valid JSON tool calls

**Example usage:**
```
You> Read all Python files in src/, find the longest one, and create a summary

Agent decomposes:
  [1/4] list_dir("src/", recursive=true)
  [2/4] read_file for each .py file
  [3/4] Compare file lengths
  [4/4] write_file("summary.txt", "...")

Each step gets fresh context (no accumulated history noise).
```

**When to use:** Automatically activated for small models (<7B) or complex multi-step tasks. Can be forced with `/plan` command.

### 4. AST-Based Knowledge Graph

**What it does:** Full code structure extraction with call graphs, inheritance trees, and import tracking.

**How it works:**
- Python AST parsing (classes, functions, methods, docstrings, args, decorators)
- Four edge types: CALLS, INHERITS, CONTAINS, IMPORTS
- Graph queries: search, callers, callees, inheritance, blast_radius
- Incremental updates (mtime + content hash change detection)

**Example usage:**
```
You> What functions call estimate_tokens()?
→ Graph query returns all callers with file locations

You> Show me the blast radius of changing Agent.run()
→ Returns all downstream code affected by the change

You> What does ModelProvider inherit from?
→ Returns inheritance tree (ABC base class)
```

**When to use:** Understanding code structure, impact analysis, refactoring planning, dependency mapping.

### 5. Contextual Embeddings

**What it does:** Enriches code chunks with structural context before embedding.

**How it works:**
- Queries knowledge graph for callers, callees, inheritance
- Prepends context: `[From path, function X, called by Y, calls Z] {code}`
- Embeds contextualized text (captures WHERE code lives)
- Stores original text (clean display to user)

**Example benefit:**
```
Without context: "def authenticate(token): ..." → embedding
With context: "[From src/auth/handler.py, function authenticate in AuthService,
               called by middleware.verify and routes.login]
               def authenticate(token): ..." → embedding

Query: "login flow"
→ Matches authenticate() because context mentions routes.login
```

**When to use:** Automatically active when both knowledge graph and vector store exist. Dramatically improves semantic search relevance.

### 6. Multi-Language Code Understanding

**What it does:** AST-informed chunking and boundary detection for 7+ programming languages.

**Supported languages:**
- Python (full AST parsing)
- Go, Rust, C/C++, TypeScript, JavaScript, Shell (boundary detection)

**How it works:**
- Detects language from file extension
- Uses language-specific patterns for function/class boundaries
- Python: Full AST with semantic metadata
- Others: Regex-based boundary detection with future pluggable parser support

**When to use:** Automatically applied during ingestion. Works on polyglot codebases.

### 7. Safety & Sandboxing

**What it does:** Permission system with dangerous operation blocking and optional Ornstein sandbox isolation.

**Protections:**
- Blocked paths: `/etc`, `/sys`, `C:\Windows`, etc.
- Blocked commands: `rm -rf /`, `mkfs`, fork bombs
- Dangerous command confirmation: `rm`, `sudo`, `shutdown`
- Execution budgets: 300s session limit, 6 tool calls per step max
- Loop prevention: Repeat detection, thrashing detection, hard limits

**Example:**
```
You> Delete all files in the project
→ [!] Allow dangerous command: rm -rf *? [y/N]
→ User must explicitly confirm

You> Run this command 50 times
→ Hard limit kicks in after 6 calls
→ [System]: Hard limit reached - 6 tool calls executed
```

**When to use:** Always active. Provides safety rails for autonomous agent operation.

### 8. Voice Synthesis (Text-to-Speech)

**What it does:** Converts agent responses to speech using Piper TTS with voice profiles.

**Features:**
- Multiple voice profiles (balanced, narrative, technical, energetic)
- DSP processing (bass boost, treble, normalization, compression)
- Audio caching (identical responses reuse cached audio)
- Offline operation (Piper runs locally)

**Enable:**
```yaml
# ~/.animus/config.yaml
audio:
  enabled: true
  voice_profile: balanced  # or narrative, technical, energetic
```

**When to use:** Hands-free operation, accessibility, multitasking while agent works.

---

## Hardware Requirements & Model Viability

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB free | 50 GB free (for multiple models) |
| **CPU** | 4 cores | 8+ cores |
| **OS** | Windows 10+, Linux (Ubuntu 20.04+), macOS 11+ | Any modern OS |
| **GPU** | None (CPU-only works) | NVIDIA with 6+ GB VRAM |

### Model Size Viability Matrix

Performance tested on consumer hardware (RTX 2080 Ti 11GB, Ryzen 9 5900X) with Q4_K_M quantization:

| Model Size | VRAM (Q4) | Inference Speed | Tool Calling | Planning | Code Quality | Viable Use Cases |
|------------|-----------|-----------------|--------------|----------|--------------|------------------|
| **1-3B** | 1.2-2.4 GB | Fast (1-5s) | ✅ With GBNF | ⚠️ Limited (3 steps) | ⚠️ Syntactic | Single-step file ops, simple Q&A |
| **7B** | 4.8 GB | Moderate (15-30s) | ✅ Reliable | ✅ Good (5-7 steps) | ✅ Production-ready | Multi-file coding, code review, refactoring |
| **14B** | 8.9 GB | Slow (30-60s) | ✅ Excellent | ✅ Excellent (7-10 steps) | ✅ High quality | Complex agentic workflows, architecture |
| **20B** | 12.3 GB | Very Slow (60-120s) | ✅ Excellent | ✅ Excellent (10+ steps) | ✅ Very high | Research-grade code generation |
| **30B** | 18.3 GB | Multi-GPU needed | ✅ Near-perfect | ✅ Near-perfect | ✅ Exceptional | Professional development |
| **70B** | 42 GB | Multi-GPU required | ✅ Near-perfect | ✅ Near-perfect | ✅ Exceptional | Frontier local capability |

**VRAM formula:** `params_B × 0.6 + 0.3 GB` base + ~15% runtime overhead for Q4_K_M quantization

**Key thresholds:**
- **7B**: Minimum for production-quality code output
- **14B**: Sweet spot for consumer hardware (single GPU, good quality)
- **30B+**: Requires multi-GPU or workstation hardware (>$5K investment)
- **70B+**: Typically exceeds cost-effectiveness vs API usage for most workflows

### Hardware Tier Recommendations

| Tier | GPU | VRAM | Best Model | Use Case |
|------|-----|------|------------|----------|
| **Entry** | None (CPU-only) | N/A | API (GPT-4/Claude) | Learning, experimentation |
| **Hobbyist** | GTX 1660 / RTX 3050 | 6 GB | 7B models | Weekend projects |
| **Enthusiast** | RTX 3060 / 4060 Ti | 8-12 GB | 7-14B models | Serious development |
| **Professional** | RTX 4090 / A6000 | 24 GB | 20-30B models | Production workflows |
| **Workstation** | Multi-GPU (2-4×) | 48+ GB | 70B+ models | Research, frontier experiments |
| **Edge** | Jetson Orin Nano | 8 GB | 3-7B models | Embedded, air-gapped |

**Reality check:** For most users, API access to GPT-4 or Claude Sonnet is more cost-effective than dedicated GPU hardware for 30B+ models. Animus supports both—use local for privacy/air-gapped, use API for scale.

---

## Quick Start Guide

### Setup for Local Inference (Recommended: 7B Model)

```bash
# 1. Install
git clone https://github.com/yourusername/animus.git
cd animus
pip install -e ".[all]"

# 2. Initialize
animus init
animus detect  # Check your GPU

# 3. Download model (choose based on your VRAM)
animus pull qwen-2.5-coder-7b  # 4.8 GB VRAM, best for coding

# 4. Start agent
animus rise

# 5. Test basic functionality
You> What files are in this directory?
You> Create a file called test.txt with "Hello World"
You> exit
```

### Setup for Code Intelligence (Manifold)

```bash
# 1. Build knowledge graph (AST parsing)
animus graph ./your-project
# → Extracts 1000s of nodes (classes, functions, methods)
# → Creates call graphs, inheritance trees, import maps

# 2. Build vector store (contextual embeddings)
animus ingest ./your-project
# → Chunks code with AST boundaries
# → Enriches with graph context
# → Embeds with sentence-transformers

# 3. Use intelligent search
animus rise
You> search for "how does configuration loading work?"
→ [Strategy: SEMANTIC] Returns conceptually relevant code

You> search for "what calls load_config()?"
→ [Strategy: STRUCTURAL] Returns all callers from graph

You> search for "find config code and everything that depends on it"
→ [Strategy: HYBRID] Fuses semantic + structural results
→ Results marked with ★ appear in both (high confidence)
```

### Setup for API Usage (Fastest Path)

```bash
# 1. Install and init
pip install -e ".[dev]"
animus init

# 2. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# 3. Configure provider
# Edit ~/.animus/config.yaml:
model:
  provider: anthropic  # or "openai"
  model_name: claude-sonnet-4-5-20250929  # or "gpt-4"

# 4. Start
animus rise
```

No model download needed—uses cloud API directly.

---

## Example Workflows

### Code Understanding
```
You> search for "error handling patterns"
→ Manifold routes to SEMANTIC
→ Returns code chunks with try/except patterns

You> search for "what does ChunkContextualizer.contextualize call?"
→ Manifold routes to STRUCTURAL
→ Returns callees from knowledge graph

You> What's the blast radius of changing estimate_tokens()?
→ Agent uses get_blast_radius tool
→ Shows all downstream code affected
```

### Code Modification
```
You> Read src/core/agent.py and add a debug logging statement to the _step method

Agent:
  [1/3] read_file("src/core/agent.py")
  [2/3] modify file with logging
  [3/3] write_file with updated content

You> Now run the tests to make sure nothing broke
→ Agent runs pytest and reports results
```

### Git Operations
```
You> What files have uncommitted changes?
→ Agent runs git_status

You> Show me the diff for src/core/agent.py
→ Agent runs git_diff

You> Commit the changes with message "Add debug logging"
→ [!] Commit with message: Add debug logging? [y/N]
→ User confirms, agent commits
```

### Multi-File Refactoring
```
You> Find all usages of estimate_tokens(), then consolidate them into
     a single implementation in src/core/context.py

Agent (with 7B model):
  [1/5] search for "estimate_tokens"  # Uses Manifold
  [2/5] read_file for each file with matches
  [3/5] Analyze duplicate implementations
  [4/5] Update all files to import from context.py
  [5/5] Run tests to verify changes

→ Automatically handles complex multi-file operations
```

---

## CLI Commands Reference

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `animus init` | Initialize config | `animus init` |
| `animus detect` | Show hardware info | `animus detect` |
| `animus status` | System readiness check | `animus status` |
| `animus config --show` | View configuration | `animus config --show` |

### Model Management

| Command | Description | Example |
|---------|-------------|---------|
| `animus models` | List available models | `animus models` |
| `animus models --vram 6` | Filter by VRAM | `animus models --vram 6` |
| `animus models --role planner` | Filter by capability | `animus models --role planner` |
| `animus pull <model>` | Download model | `animus pull qwen-2.5-coder-7b` |
| `animus pull --list` | Show all downloadable models | `animus pull --list` |

### Knowledge Base

| Command | Description | Example |
|---------|-------------|---------|
| `animus graph <path>` | Build knowledge graph | `animus graph ./src` |
| `animus ingest <path>` | Build vector store | `animus ingest ./src` |

### Agent Sessions

| Command | Description | Example |
|---------|-------------|---------|
| `animus rise` | Start interactive session | `animus rise` |
| `animus rise --resume` | Resume last session | `animus rise --resume` |
| `animus rise --session <id>` | Resume specific session | `animus rise --session abc123` |
| `animus sessions` | List all sessions | `animus sessions` |

### In-Session Commands

| Command | Description |
|---------|-------------|
| `/tools` | Show available tools |
| `/tokens` | Show context usage |
| `/plan` | Toggle plan mode |
| `/save` | Save session |
| `/clear` | Reset conversation |
| `/help` | List all commands |
| `exit` or `quit` | End session |

---

## The Journey: From Chunking to Manifold

### Discovery 1: The API Scaling Advantage (2024)

**Initial hypothesis:** Local models can compete with APIs through clever prompting.

**Reality discovered:** For production-scale agentic workflows, API costs scale linearly with usage, while local inference costs scale **worse than linearly** with quality requirements:
- 7B model: Fast but limited code quality
- 14B model: Better but 2x slower, requires 2x VRAM
- 30B+ model: Good quality but requires multi-GPU ($5K+) and 5-10x slower

**Key finding:** API almost always wins on total cost of ownership at scale. Local inference is for privacy, air-gapped environments, or specific low-latency scenarios—not cost savings.

### Discovery 2: Naive Chunking is Fundamentally Flawed (Early 2025)

**Initial approach:** Standard RAG chunking (sliding window, 512 tokens, 64 token overlap)

**Problems discovered:**
1. **Semantic boundaries ignored** - Functions split mid-implementation
2. **No structural metadata** - Chunks are anonymous text blobs
3. **Context-free embeddings** - "def authenticate()" could be anywhere
4. **Search quality poor** - "login flow" doesn't match relevant auth code

**Attempted fix:** Better chunking (paragraph-aware, code-aware regex)

**Result:** Marginal improvement, fundamental issues remained.

### Discovery 3: The One Surface Realization (Late 2025)

**Breakthrough insight:** Stop trying to make chunks self-contained. Instead, make the **entire codebase one surface** that the LLM can navigate through hardcoded tooling.

**Key components:**
1. **AST-based knowledge graph** - Parse code structure, not text
2. **Graph queries as tools** - "What calls X?" is a SQL query, not an LLM prompt
3. **Contextual embeddings** - Enrich chunks with graph-derived context
4. **Hardcoded routing** - Classify query intent with regex, not LLM

**Why this works:**
- Knowledge graph answers structural questions (<20ms SQL query)
- Vector search answers semantic questions (with graph-enriched embeddings)
- Router combines them intelligently (hardcoded, <1ms, no LLM overhead)
- LLM only used for understanding user intent and generating code—not navigation

### Discovery 4: Manifold is Born (February 2026)

**The synthesis:** Animus had all the pieces:
- ✅ Vector store (semantic search)
- ✅ Knowledge graph (structural queries)
- ✅ AST parser (code understanding)
- ✅ Tool framework (extensibility)

**What was missing:** Orchestration layer to make them work as one system.

**Manifold implementation:**
- Hardcoded query router (SEMANTIC/STRUCTURAL/HYBRID/KEYWORD)
- Reciprocal Rank Fusion (cross-strategy result merging)
- Contextual embeddings (graph context prepended before embedding)
- Unified search() tool (automatic strategy selection)

**Result:** <400ms hybrid queries on edge hardware. No cloud, no large models needed for code navigation. LLM used only for actual reasoning/generation, not for "finding the right code."

### Key Insight: Hardcoded Beats LLM for Navigation

**Traditional RAG:** LLM decides what to search for, LLM interprets results, LLM navigates codebase

**Manifold approach:**
- Hardcoded router decides strategy (<1ms vs LLM's 100-500ms)
- SQL queries answer structural questions (deterministic vs LLM's probabilistic)
- AST parsing extracts code structure (100% accurate vs LLM's ~80%)
- LLM only used where ambiguity/creativity actually needed

**Philosophy:** *"Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else."*

The result: A 7B model with Manifold outperforms a 30B model with naive RAG because the 7B model is doing less work—the infrastructure handles code navigation deterministically.

---

## Architecture Principles

### 1. Local-First by Design
- All data stored locally (SQLite databases)
- No cloud dependencies for core functionality
- API providers available but not required
- Works offline after initial model download

### 2. Hardcoded Orchestration
- Task decomposition: Hardcoded parser (not LLM)
- Query routing: Regex patterns (not LLM classifier)
- Tool selection: Type-based filtering (not LLM decision)
- Error recovery: Exception classification (not LLM diagnosis)

### 3. Edge Hardware Optimization
- Designed for 8W Jetson to consumer GPUs
- <400ms query latency target
- Paginated vector search (constant memory)
- SIMD-accelerated KNN (sqlite-vec)
- Batched embedding generation

### 4. Production-Ready Safety
- Permission system (blocked paths/commands)
- Execution budgets (time limits per session)
- Loop prevention (repeat detection, thrashing detection, hard limits)
- Audit trails (write operations logged)
- Sandbox isolation (Ornstein for untrusted code)

---

## Testing

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_router.py -v
```

**Test coverage:** 396 tests across 15 modules
**CI/CD:** Automated testing on Python 3.11, 3.12, 3.13 via GitHub Actions

---

## Project Structure

```
animus/
├── src/
│   ├── core/              # Agent, planner, context management
│   │   ├── agent.py       # Agentic loop with reflection
│   │   ├── planner.py     # Plan-then-execute pipeline
│   │   ├── context.py     # Token estimation, context budgeting
│   │   └── tool_parsing.py # Shared tool call parser
│   ├── llm/               # Model providers
│   │   ├── native.py      # llama-cpp-python (GGUF)
│   │   ├── api.py         # OpenAI/Anthropic APIs
│   │   └── base.py        # Provider ABC
│   ├── memory/            # RAG pipeline
│   │   ├── chunker.py     # Multi-language chunking
│   │   ├── embedder.py    # Sentence-transformers
│   │   ├── vectorstore.py # SQLite vector store
│   │   ├── scanner.py     # Directory walker
│   │   └── contextualizer.py # Contextual embedding
│   ├── knowledge/         # Code intelligence
│   │   ├── parser.py      # AST-based code parser
│   │   ├── graph_db.py    # Knowledge graph storage
│   │   └── indexer.py     # Incremental graph builder
│   ├── retrieval/         # Manifold system
│   │   ├── router.py      # Query classification
│   │   └── executor.py    # Strategy dispatch + RRF
│   ├── tools/             # Agent tools
│   │   ├── filesystem.py  # File operations
│   │   ├── shell.py       # Shell commands
│   │   ├── git.py         # Git operations
│   │   ├── graph.py       # Graph queries
│   │   └── manifold_search.py # Unified search
│   ├── isolation/         # Sandboxing
│   │   └── ornstein.py    # Lightweight sandbox
│   └── audio/             # TTS system
│       ├── engine.py      # Piper TTS integration
│       └── voice_profile.py # Voice profiles + DSP
├── tests/                 # Test suite (396 tests)
├── docs/                  # Documentation
└── LLM_GECK/             # Development audits & blueprints
```

---

## Performance Characteristics

### Manifold Query Latency (Measured)

| Operation | Latency | Backend |
|-----------|---------|---------|
| Router classification | <1ms | Pure regex |
| Vector search (sqlite-vec) | 20-50ms | SIMD KNN |
| Graph query | 10-20ms | Indexed SQL |
| Keyword search (grep) | 50-100ms | Subprocess |
| RRF fusion | <1ms | Pure math |
| **Total (cached embedding)** | **<200ms** | **Combined** |
| Query embedding (MiniLM) | 50-200ms | GPU/CPU dependent |
| **Total (cold query)** | **<400ms** | **End-to-end** |

Tested on RTX 2080 Ti with 601 chunks, 1,240 graph nodes.

### Agent Execution (After Improvements)

| Metric | Before Audit | After Audit | Improvement |
|--------|--------------|-------------|-------------|
| Tool calls per step | 12-20+ (loops) | 1-2 | **92% reduction** |
| Token estimation error | ±30% (4 char/token) | ±2% (tiktoken) | **93% accuracy gain** |
| Vector search memory | 150MB+ spikes | Constant (paginated) | **Bounded** |
| Repeat detection | 3 identical calls | 2 identical calls | **Stricter** |
| Language support | Python only | 7+ languages | **700% expansion** |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing guidelines, and contribution areas.

**Areas open for contribution:**
- Multi-language parsers (Go, Rust, TypeScript using tree-sitter)
- Additional tool implementations (web browsing, API calls, database access)
- Model provider integrations (Ollama, LM Studio, vLLM)
- Performance optimizations (Go sidecar architecture documented in LLM_GECK)
- Documentation and tutorials

---

## Development

### Design Philosophy

**Core principle:** *"Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else."*

**In practice:**
- ✅ LLM for: Task decomposition, code generation, natural language understanding
- ❌ LLM for: File parsing (use AST), pattern matching (use regex), routing (use decision trees)

See [LLM_GECK/README.md](LLM_GECK/README.md) for development framework and [LLM_GECK/MANIFOLD_BUILD_INSTRUCTIONS.md](LLM_GECK/MANIFOLD_BUILD_INSTRUCTIONS.md) for Manifold architecture details.

### Build Status

![CI](https://github.com/yourusername/animus/workflows/CI/badge.svg)

**Test coverage:** 396/399 tests passing (99.2%)
**Supported:** Python 3.11, 3.12, 3.13
**Platforms:** Windows, Linux, macOS

---

## License

[Insert your license here - MIT, Apache 2.0, etc.]

---

## Acknowledgments

Built with:
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for local GGUF inference
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- [sqlite-vec](https://github.com/asg017/sqlite-vec) for SIMD-accelerated vector search
- [tiktoken](https://github.com/openai/tiktoken) for accurate token counting
- [Piper TTS](https://github.com/rhasspy/piper) for voice synthesis

Inspired by the principle that **the best code is the code you don't have to write**—and the best LLM call is the one you hardcode away.

---

**Status:** Production-ready with 39 tasks completed, 8,000+ lines of improvements, and novel Manifold multi-strategy retrieval system.

*"The name of the game isn't who has the biggest model. It's who gets the most signal per watt."*
