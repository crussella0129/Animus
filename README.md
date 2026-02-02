# Animus

A local CLI coding agent powered by GGUF models. Animus reads code, writes files, executes commands, and learns from your codebase.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/crussella0129/Animus.git
cd Animus
python install.py

# 2. Download a model
animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF

# 3. Start chatting
animus rise
```

## Requirements

- Python 3.10+
- ~8GB RAM (for 7B models)
- GPU recommended (CUDA, Metal, or ROCm)

## Installation

### Auto-Install (Recommended)

The installer auto-detects your system and installs everything:

```bash
python install.py
```

Options:
```bash
python install.py --cpu          # CPU-only (no GPU acceleration)
python install.py --skip-native  # Skip llama-cpp-python
python install.py --verbose      # Show detailed output
```

### Manual Installation

```bash
pip install -e ".[native]"

# For GPU acceleration:
# NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

# Apple Silicon (Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall

# AMD (ROCm)
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall
```

### Supported Platforms

| Platform | GPU Support |
|----------|-------------|
| Windows (x86_64) | NVIDIA CUDA |
| macOS (Intel/Apple Silicon) | Metal |
| Linux (x86_64, ARM64) | CUDA, ROCm |
| NVIDIA Jetson | CUDA (JetPack) |

## Commands

| Command | Description |
|---------|-------------|
| `animus rise` | Start an interactive chat session |
| `animus pull <model>` | Download a GGUF model from Hugging Face |
| `animus models` | List local models |
| `animus status` | Show system status and available providers |
| `animus detect` | Detect hardware (OS, GPU, etc.) |
| `animus config` | Manage configuration |
| `animus init` | Initialize Animus in current directory |
| `animus ingest <path>` | Ingest files for RAG |
| `animus search <query>` | Search ingested knowledge |
| `animus analyze` | Analyze past runs |
| `animus serve` | Start OpenAI-compatible API server |

### Model Management

```bash
animus pull <repo>           # Download from Hugging Face
animus models                # List local models
animus model list            # Same as above
animus model info <name>     # Show model details
animus model remove <name>   # Delete a model
```

### Skills

```bash
animus skill list            # List available skills
animus skill show <name>     # Show skill details
animus skill install <url>   # Install skill from URL
animus skill run <name>      # Run a skill
```

### MCP Server

```bash
animus mcp server            # Start MCP server
animus mcp tools             # List MCP tools
```

## Recommended Models

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~5 GB | General coding |
| `bartowski/c4ai-command-r7b-12-2024-abliterated-GGUF` | ~5 GB | Uncensored |
| `TheBloke/CodeLlama-7B-Instruct-GGUF` | ~4 GB | Code generation |

Download with:
```bash
animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

## Configuration

Config file: `~/.animus/config.yaml`

```yaml
model:
  provider: native           # native, trtllm, or api
  model_name: ""             # Empty = auto-detect
  temperature: 0.7

native:
  models_dir: ~/.animus/models
  n_ctx: 4096                # Context window
  n_gpu_layers: -1           # -1 = all layers on GPU
```

### Using Cloud APIs

```yaml
model:
  provider: api
  model_name: gpt-4
  api_base: https://api.openai.com/v1
  api_key: sk-your-key-here
```

## Capabilities

| Feature | Description |
|---------|-------------|
| File Reading | Read any file (no confirmation) |
| File Writing | Create/modify files (requires confirmation) |
| Shell Commands | Execute commands (safe commands auto-execute) |
| Knowledge Base | RAG over your codebase |
| Sub-Agents | Spawn specialized agents for complex tasks |

### Sub-Agent Roles

| Role | Purpose |
|------|---------|
| CODER | Write and modify code |
| REVIEWER | Review code |
| TESTER | Write and run tests |
| DOCUMENTER | Create documentation |
| DEBUGGER | Find and fix bugs |
| RESEARCHER | Analyze code |

## Architecture

```
Animus/
├── src/
│   ├── core/           # Agent, config, permissions
│   ├── llm/            # Model providers (Native, TensorRT, API)
│   ├── memory/         # RAG system
│   ├── tools/          # File, shell tools
│   ├── skills/         # Extensible skills
│   └── mcp/            # MCP server
└── tests/              # Test suite
```

## Development

```bash
# Run tests
pytest tests/ -v

# Install dev dependencies
pip install -e ".[all]"
```

## Troubleshooting

### "Model not found"

```bash
# List available models
animus models

# Download a model
animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

### "llama-cpp-python not available"

```bash
# Install with GPU support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### Slow performance

- Ensure GPU is being used: `animus status`
- Use quantized models (Q4_K_M recommended)
- Increase `n_gpu_layers` in config

## License

MIT License - see [LICENSE](LICENSE) for details.
