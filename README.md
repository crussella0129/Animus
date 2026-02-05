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
- **For local models**: GPU with sufficient VRAM (see [Model Requirements](#model-requirements) below)
- **For API models**: An API key from OpenAI, Anthropic, or another provider
- 16GB+ system RAM recommended

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
| `animus speak` | Toggle voice synthesis (spooky AI voice) |
| `animus praise` | Configure task completion audio |

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

### Audio Interface

Animus features optional audio feedback for a more immersive experience:

```bash
# Voice synthesis (spooky AI voice)
animus speak                 # Enable voice
animus speak --off           # Disable voice

# Task completion music
animus praise --fanfare      # Mozart's "Eine kleine Nachtmusik" (triumphant)
animus praise --spooky       # Bach's "Little Fugue" (eerie but celebratory)
animus praise --off          # Disable completion audio

# Background music during execution
animus praise --moto         # Enable Paganini's "Moto Perpetuo"
animus praise --motoff       # Disable background music
```

**Voice Features:**
- Speaks "Yes, Master" when receiving commands
- Speaks "It will be done" when executing tasks
- Low-pitched, square-wave MIDI synthesis for robotic aesthetic
- Filters out code/commands (only speaks high-level descriptions)

**Music Features:**
- **Fanfare Mode**: Mozart's iconic opening from Eine kleine Nachtmusik
- **Spooky Mode**: Bach's Little Fugue in G minor (avoiding the cliche Toccata)
- **Moto Perpetuo**: Paganini's virtuosic piece plays quietly during multi-step tasks

**Implementation:**
- Minimal pure Python WAV generation
- OS-native playback (PowerShell/aplay/afplay)
- Only dependency: numpy (for waveforms)
- ~200 lines total code

**Installation:**
```bash
pip install -e ".[audio]"  # Just installs numpy
```

Audio features gracefully degrade if numpy unavailable or OS lacks audio commands.

## Model Requirements

Animus is an agent that reads code, writes files, and executes commands. This requires a model that can reliably follow complex multi-step instructions, use tool results accurately, and avoid hallucination. **Not all models are capable of this.** Small models (7-8B parameters) will produce unreliable results -- they hallucinate, ignore tool outputs, and fail to follow instructions precisely.

### Minimum Requirements for Reliable Agent Use

| Tier | Model | Quantization | VRAM | Agent Quality |
|------|-------|-------------|------|---------------|
| **API (Recommended)** | Claude Sonnet/Haiku, GPT-4o/4o-mini | N/A | None | Excellent |
| **Local - High** | Qwen3-30B-A3B (MoE) | Q4_K_M | 24GB+ (or 32GB RAM for CPU) | Good |
| **Local - Medium** | Qwen3-14B | Q4_K_S | 12GB+ | Usable |
| **Local - Budget** | Qwen3-14B | Q3_K_M | 10GB+ | Marginal |
| **Not recommended** | Any 7-8B model | Any | Any | Unreliable |

### Why 14B+ for Local Models?

Agent tasks require the model to:
1. Parse user intent precisely ("a file called X" = file, not directory)
2. Call tools correctly (not describe calling them)
3. Use tool results in responses (not generate generic filler)
4. Know when to stop (not loop unnecessarily)

Models under 14B parameters consistently fail at one or more of these. 14B is the minimum where agent behavior becomes usable, and 30B+ is where it becomes reliable.

### Hardware Tiers

| GPU | VRAM | Best Local Model | Context Window |
|-----|------|-----------------|----------------|
| RTX 4090 / A6000 | 24GB | Qwen3-30B-A3B Q4_K_M | 16K+ |
| RTX 3090 | 24GB | Qwen3-30B-A3B Q4_K_M | 16K+ |
| RTX 4070 Ti Super | 16GB | Qwen3-14B Q4_K_M | 8K+ |
| RTX 3060 12GB | 12GB | Qwen3-14B Q4_K_S | 4-8K |
| RTX 2080 Ti | 11GB | Qwen3-14B Q4_K_S (tight) | 4K |
| RTX 3060 8GB / RTX 4060 | 8GB | API recommended | N/A |
| Apple M1/M2/M3 (16GB) | 16GB unified | Qwen3-14B Q4_K_M | 8K |
| Apple M1/M2/M3 (32GB+) | 32GB unified | Qwen3-30B-A3B Q4_K_M | 16K+ |
| CPU only (32GB+ RAM) | N/A | Qwen3-30B-A3B Q4_K_M (slow) | 8K |

### Downloading a Local Model

```bash
# 14B model (minimum for agent use, fits 11-12GB VRAM)
animus pull bartowski/Qwen_Qwen3-14B-GGUF/Qwen3-14B-Q4_K_S.gguf

# 30B MoE model (recommended if you have 24GB VRAM)
animus pull bartowski/Qwen_Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf
```

### API Models (Recommended for Best Results)

For the best agent experience, use an API model. Animus uses LiteLLM to route to any provider.

```bash
# Set your API key (choose one)
set OPENAI_API_KEY=sk-your-key-here          # Windows
export OPENAI_API_KEY=sk-your-key-here       # Linux/Mac

set ANTHROPIC_API_KEY=sk-ant-your-key-here   # Windows
export ANTHROPIC_API_KEY=sk-ant-your-key-here # Linux/Mac

# Start with an API model
animus rise --model gpt-4o-mini              # OpenAI (cheap, good)
animus rise --model gpt-4o                   # OpenAI (best)
animus rise --model claude-haiku-3           # Anthropic (cheap, good)
animus rise --model claude-sonnet-4-20250514          # Anthropic (best)
```

You can also set the API key in your config file (`~/.animus/config.yaml`):
```yaml
model:
  provider: litellm
  api_key: sk-your-key-here
```

### Auxiliary Models (Future)

| Role | Model | Size | Notes |
|------|-------|------|-------|
| Doc Reader (Qwen3-VL) | Qwen3-VL-8B | ~5GB | OCR, screenshots, PDFs |
| Ungabunga Box (web research) | Qwen3-4B | ~3GB | Sandboxed, disposable |
| Validator | Qwen2.5-1.5B | ~1GB | Web content validation |

These auxiliary agents run alongside the central brain and can use smaller models because they perform narrow, constrained tasks rather than open-ended agent reasoning.

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

Set an environment variable or add to config. LiteLLM auto-detects the provider from the model name.

```yaml
model:
  provider: litellm
  api_key: sk-your-key-here    # Or use OPENAI_API_KEY / ANTHROPIC_API_KEY env vars
```

Then start with: `animus rise --model gpt-4o` or `animus rise --model claude-sonnet-4-20250514`

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

---

## Addendum: Phase 1 Findings and Phase 2 Direction

### The Cloud vs. Local Illusion

The core takeaway from Animus Phase 1 is that the binary between "local" and "cloud" is a fiction when applied to models capable of coherent code execution and system operation. Running a model that can reliably write code, execute commands, and reason about a codebase at the level of tools like Claude Code requires compute density that maps to data-center-class infrastructure, not a consumer workstation.

A 7B parameter model can run locally, but it cannot consistently execute multi-step agentic workflows with the coherence required for real development work. The models that can (70B+) demand hardware that most individuals do not own. The "local AI" narrative collapses under the weight of its own scalability requirements.

This does not mean local inference is useless. It means the definition of "local" needs to be rethought. For the subset of users with serious hardware, local remains viable. For everyone else, the question becomes: how do you access large-model inference without surrendering to a single cloud provider?

### Animus Phase 2: Three Inference Strata

Phase 2 restructures Animus around three modes of model access:

**1. Local Inference** — For users who have the hardware (high-end GPUs, multi-GPU rigs, or dedicated compute). This is what Phase 1 already provides. It remains the fastest and most private option for those who can run it.

**2. API Harnessing** — Fallback to cloud APIs (OpenAI, Anthropic, etc.) for users without local compute. Clean abstraction so the agent behaves identically regardless of backend.

**3. Ani-Mesh** — A decentralized inference mesh. When multiple people run Animus simultaneously, their machines form a network capable of running larger models collectively. Key properties:

- **Encrypted session partitioning**: Each user's prompts and outputs are encrypted end-to-end. Mesh nodes never see plaintext.
- **Contribution-weighted scheduling**: Inference priority is ordered by who contributes the most compute to the network (uptime x computational power x success rate).
- **Zero-trust architecture**: Anyone can join. Nodes are assumed to be adversarial. The system must be resilient to nodes that lie about capacity, return garbage results, steal prompts, or drop out mid-inference.

### What We Have Figured Out

**The hybrid verification architecture for zero-trust ani-mesh:**

- **Layer 1 — Reputation + Stake**: Nodes bond compute capacity or tokens to join. Cheating detected = stake slashed. Reputation tracks uptime, output quality, and latency.
- **Layer 2 — Probabilistic Verification**: Each inference request is processed by 2+ randomly selected nodes. Results must agree (embedding-space similarity). Disagreements escalate to a third node for majority vote. Persistent disagreement triggers investigation and potential slashing.
- **Layer 3 — Encrypted Sessions**: Prompts encrypted with AES-256-GCM before going to the mesh. Results encrypted before returning. Symmetric encryption keeps it cheap. Users hold the only decryption keys.
- **Layer 4 — Contribution Weighting**: `weight = stake * uptime * agreement_rate`. Higher-weighted nodes get scheduling priority. Anti-Sybil protection by tying weight to stake amount.

**The phased build path:**

- **Phase 2.1**: Local + API with clean abstraction and deterministic fallback
- **Phase 2.2**: Ani-mesh MVP on a trusted peer network (honest nodes assumed, basic round-robin scheduling, focus on whether distributed inference actually works)
- **Phase 2.3**: Economic ordering (contribution tracking, priority weighting, still trusted network)
- **Phase 2.4**: Cryptographic isolation (add MPC/encryption only when adversarial peers are a real concern)

**The economic security requirement:**

For ani-mesh to be secure, `cost_of_attack` must exceed `gain_of_attack`:
```
Cost of Attack = (nodes_to_compromise * stake_per_node) + computational_cost_to_cheat
Gain of Attack = value_of_stolen_prompts + inference_cost_saved
Security Requirement: Cost_of_Attack >> Gain_of_Attack
```

### Questions That Still Need Answered

1. **Slashing mechanism**: On-chain via Ethereum smart contract (gas fees, slow)? Off-chain consensus among validators (trust in validators)? Hybrid batch slashing (moderate cost)?
2. **Sybil attack prevention**: Minimum stake per node (who decides the amount)? Proof-of-personhood (requires KYC, breaks privacy)? Proof-of-hardware (limits who can join)?
3. **Model distribution strategy**: Replicate the full model on all N nodes (wasteful, simple)? Shard the model across nodes (fast, complex recovery)? Central model server + distributed compute (bottleneck)?
4. **Session encryption scope**: Privacy from mesh nodes only? Privacy from network observers too? Both (expensive)?
5. **LLM output validation in zero-trust**: How do you verify that an LLM output is correct when the output is non-deterministic? Embedding-space similarity (cheap, imperfect)? Human validation sampling (expensive, accurate)? K-node model agreement with majority vote (medium cost)? Semantic hashing (novel, unproven)?
6. **Target model size**: 7B, 13B, 70B? This fundamentally changes compute assumptions and mesh topology.
7. **Latency tolerance**: Mesh introduces network hops. Sequential execution across peers is slow. What's the acceptable latency ceiling for interactive coding workflows?
8. **Infrastructure choices**: Message broker (RabbitMQ, Redis, custom)? State machine for eligibility and scheduling? Model serving framework (vLLM, TGI, ollama)?
9. **Distributed inference mechanics**: Layer-by-layer model sharding? Speculative decoding (fast local model + accurate mesh model)? LoRA/adapter-based approaches (core model distributed, personalization local)?
10. **Incentive structure at scale**: If prompts have low individual value and computation cost is low (smaller models), the economic security ratio is hard to maintain. What incentive structure works at what scale?

## License

MIT License - see [LICENSE](LICENSE) for details.
