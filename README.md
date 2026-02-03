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

- Python 3.11+ (3.10 is NOT supported)
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

### Ubuntu 22.04 LTS (Fresh Install)

Ubuntu 22.04 ships with Python 3.10, but Animus requires Python 3.11+. Install prerequisites first:

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install -y python3-pip build-essential cmake ninja-build

# 2. Install Python 3.11 from deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 3. Clone and install Animus
git clone https://github.com/crussella0129/Animus.git
cd Animus
python3.11 install.py

# 4. If llama-cpp-python compilation fails, install manually:
python3.11 -m pip install llama-cpp-python --no-cache-dir

# 5. Download a model and start
animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
animus rise
```

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

### Main Agent Model

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~5 GB | General coding (recommended) |
| `bartowski/c4ai-command-r7b-12-2024-abliterated-GGUF` | ~5 GB | Uncensored |
| `TheBloke/CodeLlama-7B-Instruct-GGUF` | ~4 GB | Code generation |

Download with:
```bash
animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

### Web Search Validator Model (Ungabunga-Box)

For secure web search, Animus uses a smaller model to validate web content before passing it to the main agent. This provides defense-in-depth against prompt injection attacks.

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | ~1.1 GB | Fast validation |

Download with:
```bash
# Using huggingface_hub directly (recommended for single file)
python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    filename='qwen2.5-1.5b-instruct-q4_k_m.gguf',
    local_dir=os.path.expanduser('~/.animus/models')
)
print('Validator model downloaded!')
"
```

The validator model is optional but recommended for web search security.

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
