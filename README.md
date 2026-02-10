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

Local-first AI agent with RAG and tool use. Runs on your hardware — no cloud required.

## Features

- **Local inference** via llama-cpp-python (GGUF models)
- **API providers** for OpenAI and Anthropic when you want cloud power
- **RAG pipeline** — ingest files, chunk, embed, and search
- **Tool use** — filesystem access and shell commands with permission guards
- **Model-size-aware context** — automatically adjusts context budgets for small, medium, and large models
- **GPU detection** — NVIDIA, Apple Silicon, Jetson

## Install

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[native]"       # llama-cpp-python for local GGUF models
pip install -e ".[embeddings]"   # sentence-transformers for local embeddings
```

## Quick Start

### 1. Install and initialize

```bash
pip install -e ".[dev]"
animus init
```

### 2. Set up a model

**Option A — Local model (no internet after download):**

```bash
pip install -e ".[native]"
animus pull llama-3.2-1b          # Downloads ~0.7 GB GGUF
animus pull --list                # See all available models
```

**Option B — API provider:**

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then edit `~/.animus/config.yaml`:

```yaml
model:
  provider: openai        # or "anthropic"
  model_name: gpt-4       # or "claude-sonnet-4-5-20250929"
```

### 3. Start a session

```bash
animus rise
```

### 4. Try these test tasks

Once inside the REPL, try these to verify everything works:

**Basic conversation:**
```
You> What files are in the current directory?
You> Read the README.md file
```

**Tool use (filesystem):**
```
You> Create a file called hello.txt with the text "Hello from Animus"
You> List the files in this directory to confirm it was created
```

**Git operations (run from a git repo):**
```
You> What's the git status?
You> Show me the last 5 commits
You> Show me the diff of any uncommitted changes
```

**Multi-step tasks (tests plan-then-execute on small models):**
```
You> Read all the Python files in src/core/, then tell me which one has the most lines of code
You> Check git status, then list all modified files and show the diff for each one
```

**Slash commands:**
```
/tools       Show available tools
/tokens      Show context window usage
/plan        Toggle plan-then-execute mode (auto for small models)
/save        Save session
/clear       Reset conversation
/help        List all commands
```

**Exit:**
```
You> exit
```

### 5. Resume a session

```bash
animus rise --resume              # Resume most recent session
animus rise --session <id>        # Resume a specific session
animus sessions                   # List all saved sessions
```

## Commands

| Command | Description |
|---------|-------------|
| `animus detect` | Detect OS, GPU, hardware type |
| `animus init` | Initialize configuration |
| `animus config --show` | View current settings |
| `animus models` | List providers and availability |
| `animus status` | System readiness check |
| `animus pull <model>` | Download a model |
| `animus ingest <path>` | Ingest files for RAG |
| `animus search <query>` | Search the vector store |
| `animus rise` | Start an interactive agent session |

## Providers

| Provider | Type | Config |
|----------|------|--------|
| `native` | Local GGUF via llama-cpp-python | Set `model_path` in config |
| `openai` | OpenAI-compatible API | `OPENAI_API_KEY` env var |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` env var |

## Testing

```bash
pytest tests/ --timeout=30 -v
```

All tests use mocks — no GPU, no API keys, no network required.

## A Note on Local Inference Viability

> **Disclosure — Findings from Development**
>
> Over the course of building Animus, empirical testing across multiple model families (Llama, Qwen, Phi, Mistral) at quantizations from Q4 through Q8 revealed a fundamental constraint: without an effective task-chunking strategy that decomposes complex instructions into model-digestible sub-tasks, local models below approximately 150B--200B parameters consistently fail to produce output of professional utility in agentic workflows. Smaller models exhibit compounding degradation across multi-turn tool-use chains — they lose instruction adherence, hallucinate tool arguments, and fail to synthesize results across steps.
>
> Furthermore, even assuming access to hardware capable of running a 200B-class model at interactive speeds (multi-GPU configurations north of $5,000--$10,000), the amortized total cost of ownership — electricity, GPU depreciation, thermal management, and the opportunity cost of dedicated hardware — **exceeds the amortized API cost** of running the equivalent workload against frontier cloud models over the realistic service life of consumer-grade equipment.
>
> Animus therefore ships with API provider support not as a convenience fallback, but as the economically rational default. The local inference path remains available for air-gapped environments, privacy-sensitive workloads, and — in the spirit of honest engineering — as a monument to the instructive futility of the exercise.

## Project Structure

```
src/
├── main.py              # Typer CLI (detect, init, config, rise, pull, ...)
├── core/
│   ├── agent.py         # Agent loop with tool calling + plan-then-execute
│   ├── config.py        # Pydantic config + YAML persistence
│   ├── context.py       # Model-size-aware context management
│   ├── detection.py     # OS / GPU / hardware detection
│   ├── errors.py        # Error classification + recovery
│   ├── logging.py       # Structured logging + rotating file handler
│   ├── permission.py    # Path and command deny lists
│   ├── planner.py       # Plan-then-execute pipeline (decompose → parse → execute)
│   └── session.py       # Session persistence (save/load/resume)
├── llm/
│   ├── base.py          # ModelProvider ABC + capabilities
│   ├── api.py           # OpenAI + Anthropic providers (with SSE streaming)
│   ├── factory.py       # Provider factory with fallback
│   └── native.py        # llama-cpp-python GGUF provider + model catalog
├── memory/
│   ├── scanner.py       # .gitignore-aware directory walker
│   ├── chunker.py       # Token + code-aware chunking
│   ├── embedder.py      # Mock + native embedders
│   └── vectorstore.py   # In-memory cosine similarity store
├── tools/
│   ├── base.py          # Tool ABC + registry
│   ├── filesystem.py    # read_file, write_file, list_dir
│   ├── git.py           # git status/diff/log/branch/add/commit/checkout
│   └── shell.py         # run_shell with safety checks
└── ui/
    └── __init__.py      # Rich console helpers + logo
```
