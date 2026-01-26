# Environment — ANIMUS

**Captured:** 2026-01-26 01:08:53
**Last Updated:** 2026-01-26

## Development Machine

- **OS:** Windows 10.0.26200
- **Shell:** PowerShell
- **Architecture:** x86_64
- **CPU Cores:** 24
- **GPU:** NVIDIA GeForce RTX 2080 Ti (11264 MB)
- **CUDA:** Available (Driver: 581.80)

## Runtime Versions

| Tool | Version |
|------|---------|
| Python | 3.13.11 |
| Node.js | 24.12.0 |
| npm | 11.6.2 |
| git | 2.52.0.windows.1 |

## Package State

- See `requirements.txt` / `pyproject.toml`

## Project Structure

```
Animus/
├── src/                  # Source code (renamed from animus/ to avoid confusion)
│   ├── core/             # Main loop, config, detection
│   ├── llm/              # Providers (Ollama, TRT, OpenAI)
│   ├── memory/           # RAG, VectorDB, Ingestion logic
│   ├── tools/            # Filesystem and Shell tools
│   ├── ui/               # Rich TUI definitions
│   └── main.py           # Entry point
├── tests/                # Test suite
├── LLM_GECK/             # GECK protocol files
├── pyproject.toml        # Project configuration
└── requirements.txt      # Dependencies
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANIMUS_CONFIG_DIR` | Override default config directory (~/.animus) |

## Target Platforms

- [x] Windows
- [x] macOS
- [x] Linux
- [x] Jetson (edge hardware)
- [ ] Docker
