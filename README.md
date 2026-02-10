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

```bash
animus init          # Create config at ~/.animus/config.yaml
animus detect        # Show hardware and GPU info
animus status        # Check system readiness
animus rise          # Awaken Animus
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

## Project Structure

```
src/
├── main.py              # Typer CLI (detect, init, config, rise, ...)
├── core/
│   ├── agent.py         # Agent loop with tool calling
│   ├── config.py        # Pydantic config + YAML persistence
│   ├── context.py       # Model-size-aware context management
│   ├── detection.py     # OS / GPU / hardware detection
│   ├── errors.py        # Error classification + recovery
│   └── permission.py    # Path and command deny lists
├── llm/
│   ├── base.py          # ModelProvider ABC + capabilities
│   ├── api.py           # OpenAI + Anthropic providers
│   ├── factory.py       # Provider factory with fallback
│   └── native.py        # llama-cpp-python GGUF provider
├── memory/
│   ├── scanner.py       # .gitignore-aware directory walker
│   ├── chunker.py       # Token + code-aware chunking
│   ├── embedder.py      # Mock + native embedders
│   └── vectorstore.py   # In-memory cosine similarity store
├── tools/
│   ├── base.py          # Tool ABC + registry
│   ├── filesystem.py    # read_file, write_file, list_dir
│   └── shell.py         # run_shell with safety checks
└── ui/
    └── __init__.py      # Rich console helpers + logo
```
