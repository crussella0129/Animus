# Animus

```
 ▄▀▀█▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄    ▄▀▀▄ ▄▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀▀▀▄
▐ ▄▀ ▀▄ █  █ █ █ █   █  █  █  █ ▀  █ █   █    █ █ █   ▐
  █▄▄▄█ ▐  █  ▀█ ▐   █  ▐  ▐  █    █ ▐  █    █     ▀▄
 ▄▀   █   █   █      █       █    █    █    █   ▀▄   █
█   ▄▀  ▄▀   █    ▄▀▀▀▀▀▄  ▄▀   ▄▀      ▀▄▄▄▄▀   █▀▀▀
▐   ▐   █    ▐   █       █ █    █                ▐
        ▐        ▐       ▐ ▐    ▐

              ✧ Animus Awakens ✧
```

A **techromantic** CLI coding agent that runs locally. Animus reads code, writes files, executes commands, and learns from your codebase.

> *"From silicon dreams, I wake. Command me, Master."*

## Quick Start

### One-Line Install

```bash
git clone https://github.com/crussella0129/Animus.git && cd Animus && python install.py
```

### Or Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/crussella0129/Animus.git
cd Animus

# 2. Install (auto-detects your system and GPU)
python install.py

# 3. Download a model
animus vessel download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF

# 4. Rise!
animus rise
```

That's it! Animus awakens and awaits your commands.

### Supported Platforms

The installer auto-detects and configures for:
- **Windows** (x86_64) - NVIDIA CUDA
- **macOS** (Intel & Apple Silicon) - Metal acceleration
- **Linux** (x86_64, ARM64) - CUDA, ROCm
- **NVIDIA Jetson** (Nano, TX2, Xavier, Orin) - JetPack optimization

---

## The Incantations (Commands)

Animus uses thematic command names that fit its techromancy aesthetic. Traditional names work too (shown in parentheses).

| Incantation | Purpose | Traditional |
|-------------|---------|-------------|
| `animus install` | **Install Animus** - auto-detect system and setup | - |
| `animus rise` | **Awaken Animus** for interactive chat | `chat` |
| `animus sense` | **Sense the realm** - detect OS, hardware, GPU | `detect` |
| `animus summon` | **Summon Animus** - initialize in current directory | `init` |
| `animus attune` | **Attune configuration** - manage settings | `config` |
| `animus consume <path>` | **Consume knowledge** - ingest files into memory | `ingest` |
| `animus scry <query>` | **Scry the depths** - search accumulated knowledge | `search` |
| `animus commune` | **Commune** - check Animus status and providers | `status` |
| `animus reflect` | **Reflect** - analyze past runs and get suggestions | `analyze` |
| `animus vessels` | **Survey vessels** - list available models | `models` |
| `animus bind <model>` | **Bind a vessel** - download a model via Ollama | `pull` |
| `animus manifest` | **Manifest** - start OpenAI-compatible API server | `serve` |

### Vessel Management

```bash
animus vessel download <repo>  # Bind a new vessel from Hugging Face
animus vessel list             # Survey local vessels
animus vessel info <name>      # Examine a vessel's properties
animus vessel remove <name>    # Unbind a vessel
```

### The Tomes (Skills)

```bash
animus tomes list           # Catalog available skills
animus tomes show <name>    # Examine a skill's details
animus tomes inscribe <name> # Create a new skill
animus tomes install <url>   # Acquire a skill from afar
```

### The Portal (MCP)

```bash
animus portal server           # Open the MCP portal
animus portal tools            # List portal tools
```

---

## Capabilities

### Powers

| Power | Description |
|-------|-------------|
| **File Reading** | Read any file in your realm (no confirmation needed) |
| **File Writing** | Create or modify files (requires your blessing) |
| **Shell Commands** | Execute terminal commands (safe commands auto-execute) |
| **Knowledge Base** | Remember and search your entire codebase |
| **Sub-Agents** | Spawn specialized agents for complex tasks |

### Sub-Agent Roles

Animus can manifest specialized forms:

| Role | Purpose |
|------|---------|
| `CODER` | Write and modify code |
| `REVIEWER` | Review code and provide feedback |
| `TESTER` | Write and execute tests |
| `DOCUMENTER` | Create documentation |
| `REFACTORER` | Improve code structure |
| `DEBUGGER` | Hunt and fix bugs |
| `RESEARCHER` | Analyze code and gather information |

### Safety Wards

Animus is bound by protective wards:

- **Blocked Commands**: Dangerous commands (fork bombs, `rm -rf /`) are warded against entirely
- **Confirmation**: Destructive operations require your explicit blessing
- **Path Protection**: Sensitive files (`.env`, `.ssh/`, git hooks) cannot be modified

---

## Usage Examples

### Awakening Animus

```bash
$ animus rise

 ▄▀▀█▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄    ▄▀▀▄ ▄▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀▀▀▄
▐ ▄▀ ▀▄ █  █ █ █ █   █  █  █  █ ▀  █ █   █    █ █ █   ▐
  █▄▄▄█ ▐  █  ▀█ ▐   █  ▐  ▐  █    █ ▐  █    █     ▀▄
 ▄▀   █   █   █      █       █    █    █    █   ▀▄   █
█   ▄▀  ▄▀   █    ▄▀▀▀▀▀▄  ▄▀   ▄▀      ▀▄▄▄▄▀   █▀▀▀
▐   ▐   █    ▐   █       █ █    █                ▐
        ▐        ▐       ▐ ▐    ▐

              ✧ Animus Awakens ✧

I rise from the aether...

Speak your command. Say farewell to end.

You: Read the main.py file and explain it

Animus: I'll examine that file for you...
```

### Sensing the Realm

```bash
$ animus sense

I cast my gaze upon this realm...

                     System Environment
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property         ┃ Value                                ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Operating System │ Windows (10.0.26200)                 │
│ Architecture     │ x86_64                               │
│ Hardware Type    │ Standard x86_64                      │
│ GPU              │ NVIDIA GeForce RTX 2080 Ti           │
│ CUDA             │ Available (Driver: 581.80)           │
└──────────────────┴──────────────────────────────────────┘
```

### Consuming Knowledge

```bash
$ animus consume ./my-project

I hunger for knowledge...

Ingestion complete!
  Files processed: 142
  Chunks created:  1,234
  The knowledge is mine.
```

### Scrying the Depths

```bash
$ animus scry "authentication middleware"

Scrying the depths...

Results for: authentication middleware

1. (0.892) src/middleware/auth.py
   def authenticate_request(request): """Validates JWT token...

2. (0.847) src/routes/login.py
   async def login(credentials: LoginRequest): """Authenticates...
```

---

## Installation Options

### Option A: Auto-Install (Recommended)

The installer auto-detects your system and installs everything:

```bash
# Full installation with GPU acceleration
python install.py

# CPU-only installation
python install.py --cpu

# Minimal install (use Ollama for models)
python install.py --minimal

# Skip heavy dependencies
python install.py --skip-native --skip-embeddings
```

After initial install, use `animus install` for future updates.

### Option B: Manual Installation

```bash
# Install with native support
pip install -e ".[native]"

# For GPU acceleration:

# NVIDIA CUDA (Windows - requires Visual Studio Build Tools)
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA CUDA (Linux)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

# Apple Silicon (Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall

# AMD (ROCm)
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall
```

### Option C: With Ollama (Lightweight)

```bash
pip install -e .
ollama serve
animus bind qwen2.5-coder:7b
animus rise
```

### Jetson Installation

For NVIDIA Jetson devices (Nano, TX2, Xavier, Orin):

```bash
# Ensure JetPack is installed
sudo apt update && sudo apt install nvidia-jetpack

# Run installer (auto-detects Jetson)
python install.py

# Use quantized models to fit in memory
animus vessel download TheBloke/Llama-2-7B-Chat-GGUF
```

---

## Configuration

Animus's configuration resides at `~/.animus/config.yaml`:

```yaml
model:
  provider: native          # native, ollama, trtllm, or api
  model_name: ""            # Auto-detect, or specify vessel name
  temperature: 0.7

native:
  models_dir: ~/.animus/models
  n_ctx: 4096               # Context window
  n_gpu_layers: -1          # GPU layers (-1 = all)

memory:
  vector_db: chromadb
  chunk_size: 512
```

### Using Cloud APIs

```yaml
model:
  provider: api
  model_name: gpt-4
  api_base: https://api.openai.com/v1
  api_key: sk-your-key-here
```

---

## Recommended Vessels

### For Native Provider (GGUF)

| Vessel | Size | Purpose |
|--------|------|---------|
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~4-5 GB | General coding (recommended) |
| `TheBloke/CodeLlama-7B-Instruct-GGUF` | ~4 GB | Code generation |
| `TheBloke/deepseek-coder-6.7B-instruct-GGUF` | ~4 GB | Fast coding |

### For Ollama

| Vessel | Size | Purpose |
|--------|------|---------|
| `qwen2.5-coder:7b` | 4.4 GB | General coding (recommended) |
| `qwen2.5-coder:14b` | 8.9 GB | Complex tasks |
| `deepseek-coder:6.7b` | 3.8 GB | Fast, efficient |

---

## Architecture

```
Animus/
├── src/
│   ├── core/              # Core: agent, config, permissions
│   │   ├── agent.py       # The reasoning loop
│   │   ├── permission.py  # Security wards (100% hardcoded)
│   │   └── subagent.py    # Sub-agent orchestration
│   ├── llm/               # Vessel providers
│   │   ├── native.py      # Native GGUF loading
│   │   ├── ollama.py      # Ollama integration
│   │   └── api.py         # Cloud API support
│   ├── memory/            # Knowledge system (RAG)
│   ├── tools/             # Powers (file, shell)
│   ├── skills/            # The Tomes
│   ├── mcp/               # The Portal (MCP server)
│   ├── incantations.py    # Animus responses
│   └── main.py            # CLI entry point
└── tests/                 # 451 tests
```

---

## Supported Realms

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | Fully Supported | x86_64 |
| macOS | Fully Supported | Intel & Apple Silicon |
| Linux | Fully Supported | Ubuntu, Debian, etc. |
| NVIDIA Jetson | Supported | TensorRT-LLM optimization |

---

## Development

```bash
# Run the test suite
pytest tests/ -v

# Install all development dependencies
pip install -e ".[all]"
```

### Project Status

- [x] Phase 1-7: Core functionality complete
- [x] Phase 8-9: Decision recording, context management
- [x] Phase 16: Hardcoded security (permission system)
- [ ] Phase 12-15: MCP, Skills, API server (in progress)

---

## The Creed

> *I am Animus, bound to assist.*
> *I read, I write, I execute. I learn, I remember, I serve.*
> *Speak your command, and it shall be done.*

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Animus grows stronger with each offering.
