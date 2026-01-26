# Animus

A high-performance, cross-platform CLI coding agent that runs locally. Animus leverages local LLM inference or cloud APIs to provide an intelligent coding assistant with file manipulation, shell execution, RAG-based knowledge retrieval, and sub-agent orchestration.

## Features

- **Self-Contained**: Run completely locally without external services using native GGUF model loading
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Hardware Aware**: Automatic GPU detection and acceleration (CUDA, Metal, ROCm)
- **RAG/Knowledge Base**: Ingest codebases, documentation, and files for context-aware assistance
- **Tool Execution**: Read/write files, run shell commands with safety controls
- **Sub-Agent Orchestration**: Spawn specialized agents for complex tasks

## Quick Start

### Option A: Fully Independent (No External Services)

This mode runs entirely locally without Ollama or any other external service.

#### 1. Install Animus with Native Support

```bash
# Clone the repository
git clone https://github.com/crussella0129/Animus.git
cd Animus

# Install with native dependencies (includes llama-cpp-python)
pip install -e ".[native]"
```

**Note for GPU acceleration:**
- **NVIDIA (CUDA)**: Install llama-cpp-python with CUDA support:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```
- **Apple Silicon (Metal)**: Install llama-cpp-python with Metal support:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```
- **AMD (ROCm)**: Install llama-cpp-python with ROCm support:
  ```bash
  CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```

#### 2. Initialize Animus

```bash
animus init
```

#### 3. Download a Model

```bash
# Download a coding model from Hugging Face
animus model download TheBloke/CodeLlama-7B-Instruct-GGUF

# Or download Qwen2.5-Coder (recommended)
animus model download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

#### 4. Start Chatting

```bash
animus chat --model codellama-7b-instruct.Q4_K_M.gguf
```

### Option B: With Ollama (Alternative)

If you prefer using Ollama as a backend:

#### 1. Install Animus

```bash
git clone https://github.com/crussella0129/Animus.git
cd Animus
pip install -e .
```

#### 2. Install and Start Ollama

**Windows:**
```bash
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### 3. Pull a Model and Chat

```bash
ollama serve  # Start Ollama (may auto-start on Windows/macOS)
animus pull qwen2.5-coder:7b
animus chat
```

## Commands

| Command | Description |
|---------|-------------|
| `animus detect` | Detect system environment (OS, hardware, GPU) |
| `animus init` | Initialize Animus configuration |
| `animus config --show` | Show current configuration |
| `animus status` | Show provider status and available models |
| `animus models` | List available models (Ollama) |
| `animus pull <model>` | Download a model via Ollama |
| `animus model download <repo>` | Download GGUF model from Hugging Face |
| `animus model list` | List locally downloaded GGUF models |
| `animus model info <name>` | Show model details |
| `animus model remove <name>` | Delete a local model |
| `animus ingest <path>` | Ingest files into the knowledge base |
| `animus search <query>` | Search the knowledge base |
| `animus chat` | Start interactive chat with the agent |

## Usage Examples

### System Detection

```bash
$ animus detect

                     System Environment
+----------------------------------------------------------+
| Property         | Value                                 |
|------------------+---------------------------------------|
| Operating System | Windows (10.0.26200)                  |
| Architecture     | x86_64                                |
| Hardware Type    | Standard x86_64                       |
| Python Version   | 3.13.11                               |
| CPU Cores        | 24                                    |
| GPU              | NVIDIA GeForce RTX 2080 Ti (11264 MB) |
| CUDA             | Available (Driver: 581.80)            |
+----------------------------------------------------------+
```

### Provider Status

```bash
$ animus status

Animus Status

Configured Provider: native
Configured Model: codellama-7b-instruct.Q4_K_M.gguf

Native (llama-cpp-python): Available (cuda)
  Local models: 2
Ollama (http://localhost:11434): Not Running
TensorRT-LLM: Not Installed
API: Not Configured
```

### Managing Local Models

```bash
# List local GGUF models
$ animus model list

Local Models
Directory: C:\Users\you\.animus\models

+--------------------------------------------------+
| Name                            | Size   | Quant  |
|---------------------------------+--------+--------|
| codellama-7b-instruct.Q4_K_M.gguf | 4.08 GB | Q4_K_M |
| qwen2.5-coder-7b.Q5_K_M.gguf    | 5.13 GB | Q5_K_M |
+--------------------------------------------------+

# Download a new model
$ animus model download TheBloke/Mistral-7B-Instruct-v0.2-GGUF

Downloading model: TheBloke/Mistral-7B-Instruct-v0.2-GGUF
Target directory: C:\Users\you\.animus\models

Selected: mistral-7b-instruct-v0.2.Q4_K_M.gguf
Downloading: mistral-7b-instruct-v0.2.Q4_K_M.gguf

Download complete!
  Path: C:\Users\you\.animus\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf
  Size: 4.37 GB
```

### Ingesting a Codebase

```bash
$ animus ingest ./my-project

Ingesting: C:\Users\you\my-project
  Done! ---------------------------------------- 100%

Ingestion complete!
  Files scanned:  150
  Files processed: 142
  Files skipped:   8
  Chunks created:  1,234
  Embeddings:      1,234
```

### Searching the Knowledge Base

```bash
$ animus search "authentication middleware"

Results for: authentication middleware

1. (0.892) src/middleware/auth.py
   def authenticate_request(request): """Validates JWT token and extracts user...

2. (0.847) src/routes/login.py
   async def login(credentials: LoginRequest): """Authenticates user and returns...
```

### Interactive Chat

```bash
$ animus chat

Animus Chat
Type your message. Use exit or quit to end.

You: Read the main.py file and explain what it does

Animus: I'll read the file for you.

[Tool: read_file executed]

The `main.py` file is the entry point for the Animus CLI application...
```

## Configuration

Configuration is stored in `~/.animus/config.yaml`:

```yaml
model:
  provider: native          # native, ollama, trtllm, or api
  model_name: ""            # Empty for auto-detect, or specific model name
  temperature: 0.7
  max_tokens: 4096

native:
  models_dir: ~/.animus/models
  n_ctx: 4096               # Context window size
  n_batch: 512              # Batch size
  n_gpu_layers: -1          # GPU layers (-1 = all, 0 = CPU only)
  use_mmap: true            # Memory-mapped loading
  verbose: false

ollama:
  host: localhost
  port: 11434

memory:
  vector_db: chromadb
  embedding_model: all-MiniLM-L6-v2
  chunk_size: 512
```

### Using Cloud APIs

To use OpenAI or compatible APIs:

```yaml
model:
  provider: api
  model_name: gpt-4
  api_base: https://api.openai.com/v1
  api_key: sk-your-api-key-here
```

## Architecture

```
Animus/
├── src/
│   ├── core/           # Agent, config, detection, sub-agents
│   │   ├── agent.py    # Main agent with reasoning loop
│   │   ├── config.py   # Configuration management
│   │   ├── detection.py # Hardware/OS detection
│   │   └── subagent.py # Sub-agent orchestration
│   ├── llm/            # LLM providers
│   │   ├── native.py   # Native GGUF loading (llama-cpp-python)
│   │   ├── ollama.py   # Ollama provider
│   │   ├── trtllm.py   # TensorRT-LLM provider (Jetson)
│   │   └── api.py      # OpenAI-compatible API provider
│   ├── memory/         # RAG and knowledge base
│   │   ├── scanner.py  # Directory scanning
│   │   ├── chunker.py  # Text chunking strategies
│   │   ├── extractor.py # File content extraction
│   │   ├── embedder.py # Embedding generation (native + API)
│   │   ├── vectorstore.py # Vector database
│   │   └── ingest.py   # Ingestion pipeline
│   ├── tools/          # Agent tools
│   │   ├── filesystem.py # read_file, write_file, list_dir
│   │   └── shell.py    # run_shell with safety controls
│   └── main.py         # CLI entry point
├── tests/              # Test suite (99 tests)
└── LLM_GECK/           # Development documentation
```

## Agent Capabilities

### Tools Available

| Tool | Description | Requires Confirmation |
|------|-------------|----------------------|
| `read_file` | Read file contents | No |
| `write_file` | Write/create files | Yes |
| `list_dir` | List directory contents | No |
| `run_shell` | Execute shell commands | Yes (destructive) |

### Safety Features

- **Destructive Command Detection**: Commands like `rm -rf`, `git push --force` require confirmation
- **Blocked Commands**: Dangerous commands (fork bombs, `rm -rf /`) are blocked entirely
- **Human-in-the-Loop**: File writes and risky operations require user approval

### Sub-Agent Roles

Animus can spawn specialized sub-agents for complex tasks:

| Role | Purpose |
|------|---------|
| `CODER` | Write or modify code |
| `REVIEWER` | Review code and provide feedback |
| `TESTER` | Write and run tests |
| `DOCUMENTER` | Write documentation |
| `REFACTORER` | Improve code structure |
| `DEBUGGER` | Find and fix bugs |
| `RESEARCHER` | Analyze code and gather information |

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | Fully Supported | Tested on x86_64 |
| macOS | Fully Supported | Intel & Apple Silicon |
| Linux | Fully Supported | Ubuntu, Debian, etc. |
| NVIDIA Jetson | Supported | TensorRT-LLM optimization |

## Recommended Models

### For Native Provider (GGUF)

| Model | Size | Use Case |
|-------|------|----------|
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~4-5 GB | General coding (recommended) |
| `TheBloke/CodeLlama-7B-Instruct-GGUF` | ~4 GB | Code generation |
| `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | ~4 GB | General purpose |
| `TheBloke/deepseek-coder-6.7B-instruct-GGUF` | ~4 GB | Fast coding |

### For Ollama

| Model | Size | Use Case |
|-------|------|----------|
| `qwen2.5-coder:7b` | 4.4 GB | General coding (recommended) |
| `qwen2.5-coder:14b` | 8.9 GB | Complex tasks, more context |
| `codellama:7b` | 3.8 GB | Alternative coding model |
| `deepseek-coder:6.7b` | 3.8 GB | Fast, efficient coding |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Installing for Development

```bash
# Install all dependencies including dev tools
pip install -e ".[all]"
```

### Project Status

- [x] Phase 1: Core Shell (CLI, detection, config)
- [x] Phase 2: Model Layer (Ollama, TensorRT-LLM, API providers)
- [x] Phase 3: RAG & Ingestion (chunking, embeddings, vector store)
- [x] Phase 4: Agentic Loop (tools, reasoning, chat)
- [x] Phase 5: Sub-Agent Orchestration (roles, scopes, parallel execution)
- [x] Phase 6: Native Model Loading (GGUF support, no Ollama dependency)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the development documentation in `LLM_GECK/` for guidelines.
