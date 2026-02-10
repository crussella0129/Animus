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
animus pull llama-3.2-1b          # Downloads ~0.7 GB GGUF (minimal)
animus pull qwen-2.5-coder-7b    # Downloads ~4.7 GB GGUF (recommended)
animus pull --list                # See all available models + VRAM/roles
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

## Wiring Up a Local Agent

Setting up a local agent end-to-end:

### Step 1: Install with native support

```bash
pip install -e ".[native]"     # includes llama-cpp-python
animus init                    # creates ~/.animus/config.yaml
```

### Step 2: Choose and pull a model

Check your GPU VRAM and pick a model that fits:

```bash
animus detect                  # shows your GPU and available VRAM
animus pull --list             # shows all models with VRAM/context/roles
animus models --vram 6         # show models fitting in 6 GB
animus models --role planner   # show models that can plan
```

Pull a model (downloads the GGUF file and auto-configures):

```bash
# Minimal (1.2 GB VRAM, fast, limited reasoning):
animus pull llama-3.2-1b

# Recommended for coding tasks (4.8 GB VRAM, strong reasoning):
animus pull qwen-2.5-coder-7b

# General purpose (4.8 GB VRAM, 32K context):
animus pull qwen-2.5-7b
```

The `pull` command downloads the GGUF file to `~/.animus/models/` and updates the config automatically.

### Step 3: Configure context length (optional)

The default `context_length` is 4096 tokens. For models with larger native context (like Qwen's 32K), you can increase it in `~/.animus/config.yaml`:

```yaml
model:
  provider: native
  model_name: qwen-2.5-coder-7b
  model_path: ~/.animus/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
  context_length: 8192    # increase for more planning steps and larger outputs
  gpu_layers: -1          # -1 = offload all layers to GPU
  size_tier: auto         # auto-detects from parameter count
```

Context length affects all budgets dynamically:

| context_length | Output budget | Chunk size | Plan steps (medium) |
|----------------|---------------|------------|---------------------|
| 4096           | 1024          | 512        | 5                   |
| 8192           | 2048          | 1024       | 7                   |
| 16384          | 4096          | 2048       | 10                  |
| 32768          | 8192          | 4096       | 10                  |

Higher context uses more VRAM. Start at 4096 and increase if your GPU has headroom.

### Step 4: Verify and run

```bash
animus status                  # should show Config=OK, GPU, Provider, Model
animus rise                    # start interactive session
```

### What to expect by model size

**1B models (small tier)** -- fast, functional with constraints:
- Valid tool calls in both direct mode (GBNF grammar) and plan-then-execute
- Cannot transition from tool-call mode to prose after receiving results
- Best for: single-step file operations (list, read, write), simple Q&A

**3-4B models (small/medium tier)** -- usable for structured tasks:
- Reliable tool calling with GBNF, occasional success without grammar
- Can handle 3-5 step plans
- Best for: file operations, simple code generation, exploration

**7B models (medium tier)** -- recommended minimum for real work:
- Better code quality (correct algorithms, error handling, docstrings)
- Shows initiative (reads back files to verify, attempts to run code)
- 5-7 step plans with GBNF-constrained execution
- Best for: multi-file coding tasks, code review, testing

**Empirical findings from testing (RTX 2080 Ti, Q4_K_M):**

| Capability | 1B (Llama 3.2) | 7B (Qwen Coder) |
|-----------|----------------|------------------|
| Tool call via GBNF | Valid JSON, correct args | Valid JSON, better argument quality |
| Direct mode tool calls | 4/5 tasks succeed (with grammar + prompt) | Reliable |
| File write quality | Generator-based flatten (works but unusual) | List-based flatten with example usage |
| Plan decomposition | 3 steps at ctx=4096 | 5 steps at ctx=4096 |
| Self-verification | None | Reads back files, attempts to run code |
| Multi-turn reasoning | Cannot switch from JSON to prose | Can summarize tool results |
| Speed (RTX 2080 Ti) | ~1-6s per step | ~30-50s per step |

## Model VRAM Requirements

All estimates use Q4_K_M quantization. Rule of thumb: `params_B * 0.6 + 0.3` GB base + ~15% runtime overhead.

| Model | Params | VRAM (Q4) | Context | Roles | Notes |
|-------|--------|-----------|---------|-------|-------|
| `llama-3.2-1b` | 1B | 1.2 GB | 4K | executor | Fastest, minimal VRAM |
| `llama-3.2-3b` | 3B | 2.4 GB | 4K | executor, explorer | Good balance of speed and capability |
| `phi-4-mini` | 3.8B | 2.9 GB | 8K | executor, planner | Strong reasoning for size, 8K context |
| `qwen-2.5-3b` | 3B | 2.4 GB | 32K | executor, explorer | 32K context, good multilingual |
| `qwen-2.5-coder-7b` | 7B | 4.8 GB | 32K | executor, planner, explorer | Code-specialized, 32K context |
| `qwen-2.5-7b` | 7B | 4.8 GB | 32K | executor, planner, explorer | Best all-round, 32K context |
| `gemma-3-4b` | 4B | 2.9 GB | 8K | executor, explorer | Google, strong coding |

**Multi-agent VRAM combos:**

| Setup | Models | Total VRAM | Use Case |
|-------|--------|------------|----------|
| Minimal | 1B executor | ~1.2 GB | Simple file ops, single-step tasks |
| Balanced | 3B executor + 3.8B planner | ~5.3 GB | Multi-step tasks with plan decomposition |
| Full | 7B executor/planner + 3B explorer | ~7.2 GB | Complex agentic workflows |

## Operating Theory: Gear Down, Don't Scale Up

The conventional approach to complex AI tasks is to throw a bigger model at the problem. Animus takes the opposite approach: **gear down the task, not up the model**.

Instead of requiring a 200B model to process raw, unstructured input, Animus pre-processes and structures input so that 3B models can handle it in steps:

1. **Task decomposition** — Break complex instructions into atomic steps (1 tool call each)
2. **Context framing** — Each step gets a fresh, minimal context window (no accumulated history noise)
3. **GBNF constraints** — Grammar-constrained output forces structurally valid JSON tool calls
4. **Tool filtering** — Each step sees only the tools relevant to its type (READ step sees `read_file`, not `git_commit`)

A 3B model with structured steps + grammar constraints accomplishes what it fails at with raw instructions. The model isn't smarter — the problem is smaller.

## Flicker Fusion: Context Per Frame

Each generation call is a "frame" in the model's processing, analogous to a frame in visual perception:

- **Too much context per frame** — Attention disperses across irrelevant tokens, instruction adherence drops, the model "forgets" what it's doing mid-generation
- **Too little context per frame** — Lost coherence between steps, the model can't synthesize information, outputs become disconnected
- **Optimal frame size** — Enough context for the current step, no more

Animus computes optimal frame size dynamically: `context_length * tier_ratio`. The formulas scale linearly with context length, while tier ratios are qualitative knobs that encode how much context each model size class can effectively attend to:

- **Small** (< 4B): 30% history, 25% output, 12.5% chunk size
- **Medium** (4-13B): 50% history, 25% output, 12.5% chunk size
- **Large** (13B+): 70% history, 25% output, no chunking

Planning complexity also scales with context via `log2(context_length / 1024)`, allowing models with larger context windows to handle more plan steps and longer execution chains.

## Model Roles

| Role | Description | Requirements | Catalog Models |
|------|-------------|--------------|----------------|
| **executor** | Executes individual tool-call steps | Must follow JSON format reliably | All models |
| **planner** | Decomposes complex tasks into step plans | Needs reasoning ability, instruction following | phi-4-mini, qwen-2.5-7b, qwen-2.5-coder-7b |
| **explorer** | Navigates codebases, reads files, searches | Needs context retention across tool rounds | llama-3.2-3b, phi-4-mini, qwen-2.5-3b, qwen-2.5-7b, qwen-2.5-coder-7b, gemma-3-4b |

Use `animus models --role planner` to filter by role, or `animus models --vram 4` to find models that fit your GPU.

## Research Findings

Empirical results from iterative development and testing on an RTX 2080 Ti (11 GB), using Llama 3.2 1B and Qwen 2.5 Coder 7B at Q4_K_M quantization.

### Phase 1: The Prose Problem

Initial observation: 1B models replied with prose explanations instead of executing tool calls.

- The `Agent._step()` path called `provider.generate()` **without** GBNF grammar constraints
- The `ChunkedExecutor` (plan-then-execute) path called `provider.generate()` **with** GBNF grammar -- and tool calls worked
- Both paths used the same model and the same tools
- The difference was entirely infrastructure, not model capability

### Phase 2: Isolating the Variables

A controlled diagnostic tested every combination of prompt style and grammar:

- **Generic prompt, no grammar** -- 1B model produced prose (explained `ls` command instead of calling `list_dir`)
- **Generic prompt + GBNF grammar** -- 1B model produced valid JSON tool calls, but used wrong argument names (copied `"key"` from the example literally)
- **Focused prompt with tool examples, no grammar** -- 1B model produced **correct** tool calls with right argument names
- **Focused prompt + GBNF grammar** -- perfect tool calls

Key finding: **GBNF grammar forces JSON structure; the prompt teaches correct content.** Both are needed together. Neither alone is sufficient for reliable tool calling.

### Phase 3: The Argument Name Problem

The system prompt example `{"name": "tool_name", "arguments": {"key": "value"}}` caused the 1B model to use `"key"` as the literal argument name in every call. The model has no concept of placeholder vs. literal text.

- Fix: generate concrete tool call examples from the actual registered tools, using real parameter names
- Example: `list_dir: {"name": "list_dir", "arguments": {"path": "example_path", "recursive": false}}`
- After fix: 1B model used `"path"` correctly on first attempt

### Phase 4: The Mode-Switching Floor

After a successful tool call, the agent feeds the tool result back for the model to summarize. The 1B model keeps producing JSON tool calls instead of switching to natural language -- it's "stuck" in JSON mode.

- GBNF grammar on first turn only (disabled after tool results) -- model still produces JSON without grammar
- The system prompt's "respond with ONLY the JSON tool call" instruction persists across turns
- The 1B model cannot reason about when to use tools vs. when to summarize results
- **Workaround**: return the last tool result directly as the response (graceful degradation)
- **Floor**: multi-turn tool-call-then-summarize requires ~3-4B models

### Phase 5: 1B vs 7B Comparison

After applying all fixes, side-by-side comparison on identical tasks:

- **1B model (Llama 3.2)**: 4/5 tool-calling tasks succeed in direct mode. Correct tool names and argument names. ~1s per generation. Cannot summarize tool results -- returns raw tool output.
- **7B model (Qwen Coder)**: All tasks succeed. Produces higher-quality code (proper error handling, docstrings, example usage). Shows initiative -- reads files back to verify writes, attempts to run generated code. 5 plan steps vs 3 at same context length. ~30-50s per generation.
- Both models benefit from GBNF grammar, but the 7B model occasionally succeeds without it
- The 7B model's plan-then-execute path produced over-decomposed plans (5 separate write_file calls that overwrote each other) -- more steps is not always better

### Summary of Floors

| Capability | Minimum Model Size | Notes |
|-----------|-------------------|-------|
| Single tool call (with grammar + prompt) | **1B** | GBNF + concrete examples in prompt |
| Single tool call (without grammar) | **3-4B** | Focused prompt alone sufficient |
| Multi-turn tool use (call, get result, summarize) | **3-4B** | 1B cannot switch from JSON to prose |
| Multi-step planning (decompose + execute) | **1B** (with planner) | Planner handles decomposition, GBNF handles execution |
| Self-verification (read back, check work) | **7B** | Requires initiative beyond instruction following |
| Code quality suitable for production use | **7B+** | Error handling, edge cases, documentation |

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
