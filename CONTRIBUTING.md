# Contributing to Animus

Thank you for considering contributing to Animus! This guide will help you get started.

## Design Philosophy

Before contributing, understand Animus's core principle:

> **Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else.**

This means:
- ✅ LLM for: task decomposition, natural language parsing, code generation
- ❌ LLM for: file parsing, pattern matching, orchestration logic
- ✅ Hardcoded: tool routing, context management, error handling
- ❌ Hardcoded: user intent interpretation, creative problem solving

When in doubt: if it can be solved with regex, AST parsing, or state machines, don't use an LLM.

## Development Environment Setup

### Prerequisites

- Python 3.11 or later (3.12 recommended)
- Git
- (Optional) CUDA-capable GPU for local model inference

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/animus.git
cd animus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[all]"

# Verify installation
python -m animus --version
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agent.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_token_estimation"
```

### Code Quality

We use standard Python tooling (add these as you see fit):

```bash
# Type checking (optional, not enforced yet)
mypy src/

# Code formatting (run before committing)
black src/ tests/

# Linting (run before committing)
ruff check src/ tests/
```

## Project Structure

```
animus/
├── src/                    # Main codebase
│   ├── core/              # Agent, planner, context management
│   ├── llm/               # Model providers (native, API)
│   ├── memory/            # RAG pipeline (chunker, embedder, vectorstore)
│   ├── knowledge/         # Code graph (parser, indexer)
│   ├── tools/             # Agent tools (filesystem, shell, git, graph)
│   └── isolation/         # Sandbox execution (Ornstein/Smough)
├── tests/                 # Test suite
├── LLM_GECK/             # Development audits and analysis
└── docs/                  # User-facing documentation
```

## Making Changes

### 1. Choose an Issue

- Check [open issues](https://github.com/yourusername/animus/issues)
- Look for `good first issue` or `help wanted` labels
- Comment on the issue to claim it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Your Changes

#### Code Style

- Follow PEP 8
- Use type hints for public APIs
- Keep functions focused and small
- Prefer composition over inheritance
- Document complex logic with comments

#### Good Practices

- **Avoid over-engineering**: Don't add abstractions for one-time operations
- **No premature optimization**: Make it work first, measure before optimizing
- **Minimal changes**: Fix the bug or add the feature, don't refactor everything
- **Trust your tools**: Don't validate what the framework guarantees
- **Explicit is better than implicit**: Clear code > clever code

#### Testing Your Changes

- Add tests for new features
- Ensure existing tests pass
- Test edge cases and error conditions
- Integration tests for multi-component changes

### 4. Commit Your Changes

```bash
# Stage specific files (prefer over git add .)
git add src/core/agent.py tests/test_agent.py

# Write a clear commit message
git commit -m "Add reflection step to agent loop

- Evaluate tool results before feeding back to model
- Detect errors and suggest alternative approaches
- Summarize long outputs to preserve context budget

Fixes #42"
```

#### Commit Message Format

```
<type>: <short summary> (50 chars or less)

<detailed description wrapped at 72 chars>

<footer: references to issues, breaking changes>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description explaining **why** (the **what** is in the diff)
- Reference to any related issues
- Screenshots/examples if UI-related

## Areas Open for Contribution

### High-Impact Areas

1. **Multi-language parser support**
   - Add Go, Rust, C++, TypeScript parsers
   - Use tree-sitter or language-specific AST tools

2. **Model provider implementations**
   - Add OpenAI, Anthropic, local model providers
   - Implement streaming support

3. **Tool development**
   - Web browsing, API calls, database access
   - Language-specific REPLs (Python, Node)

4. **Test coverage**
   - End-to-end integration tests
   - Performance benchmarks
   - Edge case coverage

### Medium-Impact Areas

5. **Documentation**
   - User guides for specific workflows
   - Architecture deep-dives
   - Tutorial videos/screencasts

6. **Performance optimization**
   - Vector search improvements
   - Context window management
   - Embeddings caching

7. **Developer experience**
   - Better error messages
   - CLI improvements
   - Configuration management

### Exploration Areas

8. **Advanced features**
   - Multi-agent collaboration
   - Long-term memory
   - Self-improvement loops

9. **Platform support**
   - Windows-specific issues
   - macOS M1/M2 optimization
   - Docker/container support

## Review Process

### What We Look For

- ✅ Clear, focused changes
- ✅ Tests included and passing
- ✅ Documentation updated if needed
- ✅ Follows design philosophy
- ✅ No breaking changes (or clearly justified)

### What We Avoid

- ❌ Large refactors without discussion
- ❌ New dependencies without justification
- ❌ Over-engineered solutions
- ❌ Breaking changes to public APIs
- ❌ Untested code

### Timeline

- Initial review: 1-3 days
- Follow-up reviews: 1-2 days
- Merge: after approval and CI passes

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/yourusername/animus/discussions)
- **Bugs**: Open an [Issue](https://github.com/yourusername/animus/issues)
- **Chat**: Join our [Discord/Slack] (if applicable)
- **Unclear code**: Check `LLM_GECK/` for audits and design docs

## Code of Conduct

- Be respectful and constructive
- Focus on technical merit, not personal preferences
- Help newcomers learn and improve
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (check LICENSE file).

---

**Ready to contribute?** Check out [good first issues](https://github.com/yourusername/animus/labels/good%20first%20issue) or the latest audit in `LLM_GECK/Improvement_Audit_*.md`.
