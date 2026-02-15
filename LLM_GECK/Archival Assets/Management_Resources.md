# Animus: Management Resources & Learning Path

**Date:** 2026-02-15
**Scope:** Targeted learning resources for the project lead, derived from observed work patterns, delegation history, and the Efficacy & Effectiveness assessment
**Companion to:** `Efficacy_and_Effectiveness.md` (same directory)

---

## Profile Summary

Charles operates as a **systems architect who delegates implementation to an LLM agent**. Across 136+ commits, 50+ GECK log entries, and 17 development phases, a clear pattern emerges:

- **Strongest in:** Architecture and design philosophy. The "hardcode everything that isn't ambiguous" principle, Manifold's deterministic routing, fresh-context-per-step execution — these are sophisticated design decisions that required deep understanding of LLM failure modes. Charles designed them; Claude implemented them.
- **Working style:** Top-down decomposition. Charles defines what a system should do and why, then delegates the how. Evidence: every major system (`planner.py`, `permission.py`, Manifold, GBNF grammar) was specified in GECK entries before implementation.
- **Empirical instinct:** Phase 2 gauntlet testing — running three model sizes through an identical task and measuring everything — is the work of someone who trusts data over assumptions. The 7B-vs-14B finding (56% slower, no better) directly shaped architecture.
- **Gaps are infrastructure, not insight:** Charles knows that `shell=True` is dangerous (it's in MEMORY.md as a "remaining security gap"). He knows CI needs linting (the `ci.yml` comment says "later"). The gaps aren't knowledge gaps — they're execution gaps in areas where the architect's time hasn't yet been invested.

The resources below target the delta between what Charles designs and what the project needs to ship.

---

## Gap 1: CI/CD Quality Gates

**Evidence:** `ci.yml` line 55: `# Basic checks - can be expanded with black, ruff, mypy later` with `continue-on-error: true`. The lint job compiles Python files but doesn't enforce formatting, typing, or coverage. The Efficacy report scores Code Quality at 5/10 — the lowest score after Deployment.

**Why this matters for Charles specifically:** Charles delegates all implementation to Claude. Without automated quality gates, the only check on Claude's output is the test suite — which has no coverage tracking. Charles can't verify that Claude's code meets standards he hasn't defined in CI.

### Resources

**1. Ruff — The One Tool You Need**
- **What:** [astral.sh/ruff](https://docs.astral.sh/ruff/) — Replaces flake8, black, isort, pyupgrade in a single Rust-based tool. Runs in <100ms on the entire Animus codebase.
- **Why this one:** Charles's design philosophy is "use the simplest tool that works." Ruff is that tool — one config file (`ruff.toml`), one CI step, covers linting + formatting. No need to learn flake8, black, and isort separately.
- **Start here:** `pip install ruff && ruff check src/ && ruff format --check src/` — run these three commands locally, read the output, then add to CI. That's the entire learning curve.
- **Time to productive:** 30 minutes to configure, 2-4 hours to fix existing violations.

**2. Mypy — Type Checking for the Permission System**
- **What:** [mypy.readthedocs.io](https://mypy.readthedocs.io/) — Static type checker for Python.
- **Why this one:** The security-critical modules (`permission.py`, `shell.py`, `base.py`) already have type annotations. Mypy catches the class of bugs where Claude generates code that type-checks by convention but not by enforcement. Start with `--strict` on `src/core/` only.
- **Start here:** `pip install mypy && mypy --strict src/core/permission.py` — this single command will show what strict typing catches on the most security-critical file.
- **Time to productive:** 1 hour to understand output; 4-6 hours to get `src/core/` clean.

**3. pytest-cov — Coverage Visibility**
- **What:** [pytest-cov.readthedocs.io](https://pytest-cov.readthedocs.io/) — Coverage plugin for pytest.
- **Why this one:** The Efficacy report notes that 561 tests with no coverage tracking "could be misleading if tests cluster on certain modules." Charles needs to know which lines Claude's tests actually exercise.
- **Start here:** `pytest --cov=src --cov-report=term-missing` — run once, read the output. The `term-missing` flag shows exact uncovered lines.
- **CI gate:** Add `--cov-fail-under=80` to CI. This single flag prevents coverage regression.
- **Time to productive:** 15 minutes to add; ongoing value.

---

## Gap 2: Shell Security & Command Parsing

**Evidence:** `shell.py:226-228` uses `subprocess.run(command, shell=True)`. `permission.py:124-138` `is_command_dangerous()` checks only `first_word`. Charles documented both in MEMORY.md under "Remaining Security Gaps" as P1. The Efficacy report shows the injection vector: `echo $(rm -rf /)` passes first-word checks.

**Why this matters for Charles specifically:** Charles designed the deny-list approach and network isolation — the security *architecture* is sound. The gap is *implementation technique*: how to parse shell commands safely when you need shell features (pipes, cd, redirects). This is a niche topic that sits at the intersection of Unix internals and Python's subprocess module.

### Resources

**1. Python `shlex` Module — The Immediate Fix**
- **What:** [docs.python.org/3/library/shlex.html](https://docs.python.org/3/library/shlex.html) — Shell lexical analysis, included in Python's stdlib.
- **Why this one:** `shlex.split()` parses a command string into tokens the same way a POSIX shell would, but without executing anything. This is the tool the Efficacy report specifically recommends for P0.2.
- **Key concept:** `shlex.split("echo $(rm -rf /)")` → `['echo', '$(rm -rf /)']`. The second token can then be checked for command substitution patterns (`$(`, backtick).
- **Start here:** Open a Python REPL, run `import shlex; shlex.split("echo hello && rm -rf /")` — see how it decomposes chains. Then read the module docs (short, ~15 minutes).
- **Limitation to know:** `shlex` follows POSIX rules. On Windows, quoting is different. The existing `_normalize_quotes_for_windows()` in `shell.py` already handles some of this.

**2. "How Bash Processes Command Lines" — Conceptual Foundation**
- **What:** The Bash Reference Manual, Section 3.1: Shell Syntax ([gnu.org/software/bash/manual/bash.html#Shell-Syntax](https://www.gnu.org/software/bash/manual/bash.html#Shell-Syntax))
- **Why this one:** To harden command checking, Charles needs to understand *how* shells decompose commands: tokenization → expansion → command substitution → execution. The current first-word check happens at the tokenization level but misses everything in the expansion phase.
- **Key sections:** 3.5 (Shell Expansions), 3.5.4 (Command Substitution), 3.2.3 (Pipelines), 3.2.4 (Lists of Commands).
- **Time to productive:** 1-2 hours of reading. The goal isn't to memorize Bash internals — it's to understand which expansion types can smuggle dangerous commands past a first-word check.

**3. Reference Implementation: `restricted-shell-parser`**
- **What:** Search PyPI/GitHub for shell command AST parsers. The pattern Charles needs is: parse command → walk AST → check each command node against deny list.
- **Alternative approach:** Instead of parsing arbitrary shell, constrain what the LLM can generate. The GBNF grammar already does this on first turn. Extending grammar constraints to execution turns (not just planning) would eliminate most injection vectors without needing to parse shell at all.
- **Design decision for Charles:** Parse-and-check vs. constrain-at-generation. Both are valid. The former is more flexible; the latter aligns better with the "hardcode everything" philosophy.

---

## Gap 3: Containerization & Deployment

**Evidence:** Efficacy report section 2.10 — no Dockerfile, no docker-compose, no health checks, no API server (HTTP), no process manager config. The README claims edge deployment (Jetson, consumer GPUs) but the only distribution method is `pip install -e ".[all]"`. Deployment score: 3/10.

**Why this matters for Charles specifically:** Charles's project vision includes edge hardware deployment. The gap isn't "learn Docker" generically — it's specifically: how do you containerize a Python application that depends on `llama-cpp-python` with CUDA, sentence-transformers, and SQLite, targeting both x86 and ARM (Jetson)?

### Resources

**1. Docker for Python/ML Applications — Not Generic Docker**
- **What:** The official Docker Python guide is too basic. Instead:
  - NVIDIA Container Toolkit docs: [docs.nvidia.com/datacenter/cloud-native/container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — required for GPU passthrough
  - `llama-cpp-python` Docker examples: The library's GitHub repo includes Dockerfiles for CUDA builds — study these as the closest reference to Animus's needs
- **Why these:** Generic Docker tutorials assume web apps. Animus needs GPU passthrough, CUDA toolkit in the image, and multi-arch builds (x86 for dev, ARM for Jetson). These resources address that specific stack.
- **Start here:** Get `llama-cpp-python`'s own Docker image running locally with GPU access. Once that works, Animus's Dockerfile is an extension of it.
- **Time to productive:** 4-8 hours to get a working GPU-enabled container; another 4-8 hours for multi-arch and docker-compose.

**2. FastAPI — The API Server Layer**
- **What:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) — Modern Python web framework.
- **Why this one:** Animus already has an OpenAI-compatible REST API design (Phase 15 in the task list: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`). FastAPI is the standard Python framework for this pattern — it auto-generates OpenAPI docs, handles async, and has native Pydantic integration (which Animus already uses for config).
- **Start here:** FastAPI's tutorial, specifically the section on dependency injection and background tasks. Map it to Animus's existing endpoint design.
- **Time to productive:** 2-4 hours for basics; 1-2 days for production-grade API with health checks and proper error handling.

**3. Health Checks & Process Management**
- **What:** Two concepts:
  - **Liveness/readiness probes:** A `/health` endpoint that returns 200 when the service is running and the model is loaded. Kubernetes and Docker Compose both use this pattern.
  - **Process supervisor:** `supervisord` or `s6-overlay` for running multiple processes in a container (API server + model inference).
- **Why this matters:** Edge deployments need automatic recovery. If the model crashes, something needs to restart it. Health checks + a process supervisor provide this without manual intervention.
- **Start here:** Add a `/health` endpoint that checks: (1) model loaded, (2) SQLite accessible, (3) disk space available. This is 20 lines of code but enables all monitoring and orchestration tooling.

---

## Gap 4: Cross-Platform Testing

**Evidence:** `ci.yml` runs only on `ubuntu-latest`. The project claims Windows and macOS support. Two known Windows test failures exist (SQLite temp file locking in teardown — `PermissionError`). Platform-conditional tests (`@pytest.mark.skipif`) exist but only run locally on Charles's Windows machine.

**Why this matters for Charles specifically:** Charles develops on Windows 11. Claude implements on whatever CI runs (Ubuntu). The platform-specific bugs (path separators, file locking, shell quoting) are discovered late because CI doesn't catch them. The `_normalize_quotes_for_windows()` function in `shell.py` exists precisely because of this gap — it was needed but would never be tested in CI.

### Resources

**1. GitHub Actions Matrix Strategy — Multi-Platform CI**
- **What:** [docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
- **Concrete change:** Expand the existing CI matrix from:
  ```yaml
  runs-on: ubuntu-latest
  matrix:
    python-version: ["3.11", "3.12", "3.13"]
  ```
  to:
  ```yaml
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.11", "3.12", "3.13"]
  ```
  This is a 2-line change in `ci.yml`. The hard part is fixing the failures that surface.
- **Start here:** Make the 2-line change, push, and read the CI output. The failures are the learning material.
- **Time to productive:** 15 minutes for the CI change; 4-6 hours to fix platform-specific failures.

**2. Fixing the Known SQLite Locking Issue**
- **What:** The 2 pre-existing Windows test failures are `PermissionError` during teardown — SQLite holds a lock on temp files that `shutil.rmtree` or `tmp_path` cleanup can't delete.
- **Fix pattern:** Ensure every test that opens a SQLite connection has an explicit `conn.close()` in teardown (or uses a context manager). On Windows, SQLite's WAL journal mode holds file locks until the connection is closed.
- **Reference:** [sqlite.org/wal.html](https://www.sqlite.org/wal.html) — specifically the section on "Persistence of WAL mode" and file locking.
- **Start here:** Search tests for SQLite fixtures, verify each has explicit cleanup. This is a grep-and-fix task, not a learning curve.

**3. `os.name` and `platform` — Cross-Platform Python Patterns**
- **What:** Python's `os` and `platform` modules for writing platform-aware code.
- **Pattern already in use:** `permission.py` normalizes backslashes; `shell.py` has Windows-specific CWD handling. These are correct patterns. The gap is that they're only tested on one platform.
- **Reference:** [docs.python.org/3/library/os.html#os.name](https://docs.python.org/3/library/os.html#os.name) — but Charles likely already knows this. The real resource is running CI on all platforms so the existing patterns get validated.

---

## Gap 5: Retrieval Engineering (HNSW, Reranking, Sparse Search)

**Evidence:** Efficacy report sections 2.4-2.6. Vector search uses O(n) brute-force cosine similarity. No BM25 ranking (keyword search is `grep` subprocess calls). No reranking after Reciprocal Rank Fusion. Retrieval score: 7/10 — high because the architecture is novel, but the implementation uses naive algorithms.

**Why this matters for Charles specifically:** Charles designed Manifold's multi-strategy routing — the architecture is the strength. But as codebases scale beyond 10K chunks, brute-force search becomes the bottleneck. Charles needs to understand *enough* about approximate nearest neighbor search and reranking to make informed design decisions when delegating the implementation.

### Resources

**1. HNSW — The Algorithm Behind Every Vector Database**
- **What:** The original HNSW paper: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs" (2018). Available on arXiv.
- **Why the paper:** Charles makes architecture decisions based on understanding, not tutorials. The HNSW paper is readable (20 pages, clear diagrams) and explains why the algorithm achieves sub-linear search time.
- **Practical implementation options for Animus:**
  - `hnswlib` — Lightweight Python bindings, ~50 LOC to integrate, supports cosine similarity. Best fit for Animus's "minimal dependency" philosophy.
  - `usearch` — SIMD-optimized, lower memory overhead than hnswlib, good for edge hardware.
  - `sqlite-vss` — SQLite extension with HNSW. Keeps everything in SQLite (aligns with Animus's existing `SQLiteVecVectorStore`). But the extension is less maintained.
- **Start here:** `pip install hnswlib` → build a 10K-vector index → benchmark against the current brute-force. The performance delta will inform the architectural decision.
- **Time to productive:** 2-4 hours for a working prototype; 8-12 hours for full integration with incremental indexing.

**2. BM25 — Sparse Retrieval for Keyword Search**
- **What:** BM25 (Best Match 25) is the standard algorithm for keyword relevance ranking. It's what makes "search for X" return results ordered by relevance rather than by occurrence.
- **Why this matters:** Manifold's keyword strategy currently delegates to `grep`, which returns results in file order with no relevance scoring. BM25 would rank results by term frequency and document length, making keyword search actually useful for queries like "where is authentication handled?"
- **Implementation options:**
  - `rank-bm25` — Pure Python, zero dependencies, 200 LOC. Drop-in replacement for grep-based keyword search.
  - `tantivy-py` — Python bindings for Rust's Tantivy search engine. More performant but adds a compiled dependency.
- **Start here:** `pip install rank_bm25` → index the existing chunks → compare results with grep on 5-10 real queries. If the relevance improvement is visible, integrate it.
- **Time to productive:** 2-4 hours for prototype; 4-6 hours for integration.

**3. Cross-Encoder Reranking — The Highest-ROI Retrieval Improvement**
- **What:** After retrieving top-20 results via RRF fusion, pass each (query, result) pair through a cross-encoder model that scores relevance. Reorder by score.
- **Why this is the highest ROI:** Industry benchmarks consistently show 20-40% improvement in top-5 relevance from reranking. The model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M params) runs locally, takes <100ms for 20 pairs, and requires no training.
- **Reference:** Sentence-Transformers cross-encoder documentation: [sbert.net/docs/cross_encoder/usage/usage.html](https://www.sbert.net/docs/cross_encoder/usage/usage.html). This is the same library Animus already uses for embeddings (`sentence-transformers`).
- **Start here:** Since `sentence-transformers` is already a dependency, cross-encoder support is already installed. Load the model, score 20 (query, passage) pairs, sort by score. The entire integration is ~30 lines.
- **Time to productive:** 2-4 hours including benchmarking.

---

## Gap 6: Multi-Language Code Intelligence

**Evidence:** The knowledge graph parser uses Python's `ast` module — Python-only. Regex-based chunking covers 8 languages but without structural analysis (call graphs, inheritance, blast radius). The README's CONTRIBUTING.md lists Go, Rust, and JavaScript parsers as "open for contribution."

**Why this matters for Charles specifically:** Manifold's contextual embeddings (graph-enriched embedding context) are the project's most novel contribution. But they only work for Python code. Extending to other languages unlocks the full value of the architecture for polyglot codebases — which is every real-world project.

### Resources

**1. Tree-sitter — The Industry Standard for Multi-Language Parsing**
- **What:** [tree-sitter.github.io/tree-sitter](https://tree-sitter.github.io/tree-sitter/) — Incremental parsing system used by GitHub (code navigation), Neovim (syntax highlighting), Zed (everything), and most modern code intelligence tools.
- **Why this one:** Tree-sitter provides a uniform AST API across 100+ languages. The same code that extracts function definitions, call sites, and class hierarchies from Python can extract them from Go, Rust, JavaScript, and C — with different grammar files but identical traversal logic.
- **Python bindings:** `py-tree-sitter` ([github.com/tree-sitter/py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter)) — install grammars per language, parse source files, walk the CST (concrete syntax tree).
- **Key concept for Charles:** Tree-sitter produces a CST, not an AST. The CST preserves all syntax (including whitespace and comments). Animus's current `ast.parse()` produces an AST that discards surface syntax. The graph-builder would need adaptation to work with CST node types instead of AST node types — but the *pattern* (extract definitions → extract call sites → build edges) is identical.
- **Start here:** `pip install tree-sitter` → parse a Go file → walk the tree → extract function definitions. Compare with the existing Python `ast.parse()` code in the knowledge graph builder. The structural similarity will make the migration path obvious.
- **Time to productive:** 4-8 hours for a single new language; 16-24 hours for a general framework that handles 5+ languages.

**2. Language Server Protocol (LSP) — The Alternative Approach**
- **What:** [microsoft.github.io/language-server-protocol](https://microsoft.github.io/language-server-protocol/) — Protocol for IDE features (go-to-definition, find-references, diagnostics).
- **Why consider it:** LSP servers already exist for every major language. Instead of building parsers, Animus could query LSP servers for structural information (definitions, references, type hierarchies). This delegates parsing to purpose-built, well-maintained tools.
- **Tradeoff vs Tree-sitter:** LSP requires running external processes (one per language). Tree-sitter is in-process. For edge deployment on constrained hardware, Tree-sitter is more appropriate. For a development machine or container, LSP might be simpler.
- **Charles's decision:** Tree-sitter aligns better with the "no external dependencies at runtime" philosophy. LSP is worth knowing about as the alternative Charles chose *not* to use.

---

## Suggested Learning Sequence

Aligned with the Efficacy report's recommended action sequence, mapped to the resources above.

### Week 1 — Quality Foundation (Gaps 1 & 2)

| Day | Action | Resource | Deliverable |
|-----|--------|----------|-------------|
| 1 | Install ruff, run on codebase, read output | Gap 1.1 | `ruff.toml` config, understand violation categories |
| 1 | Add ruff + pytest-cov to CI (remove `continue-on-error`) | Gap 1.1, 1.3 | Updated `ci.yml` with blocking quality gates |
| 2 | Run mypy --strict on `src/core/permission.py` | Gap 1.2 | Understand type errors in security-critical code |
| 3 | Read `shlex` docs + Bash manual sections 3.1-3.5 | Gap 2.1, 2.2 | Understand command decomposition model |
| 4-5 | Design hardened command checking (shlex or grammar constraint) | Gap 2.3 | Architecture decision: parse-and-check vs constrain-at-generation |

### Week 2 — Platform & Stability (Gap 4)

| Day | Action | Resource | Deliverable |
|-----|--------|----------|-------------|
| 1 | Add `windows-latest` and `macos-latest` to CI matrix | Gap 4.1 | 2-line CI change, observe failures |
| 2-3 | Fix SQLite teardown failures, path separator issues | Gap 4.2 | Green CI on all 3 platforms |
| 4-5 | Fix any remaining platform-specific failures surfaced by new CI | Gap 4.3 | Fully cross-platform test suite |

### Week 3-4 — Retrieval Engineering (Gap 5)

| Day | Action | Resource | Deliverable |
|-----|--------|----------|-------------|
| 1-2 | Integrate cross-encoder reranking (highest ROI, smallest effort) | Gap 5.3 | Reranking after RRF fusion, benchmark results |
| 3-4 | Prototype BM25 keyword search replacement | Gap 5.2 | `rank-bm25` replacing grep, relevance comparison |
| 5-8 | Prototype HNSW indexing, benchmark against brute-force | Gap 5.1 | Performance data for architectural decision |

### Month 2+ — Strategic Extensions (Gaps 3 & 6)

| Week | Action | Resource | Deliverable |
|------|--------|----------|-------------|
| 1-2 | Containerization: GPU-enabled Docker image | Gap 3.1 | Working Dockerfile with CUDA + llama-cpp-python |
| 2-3 | API server: FastAPI + health checks | Gap 3.2, 3.3 | `/v1/chat/completions` + `/health` endpoints |
| 3-4 | Tree-sitter: First non-Python language (Go or JavaScript) | Gap 6.1 | Graph-enriched embeddings for one additional language |

---

## A Note on Delegation

Charles's most effective pattern is: **understand the architecture → specify the design → delegate implementation → validate empirically.** This document supports that pattern. The goal isn't for Charles to become an expert in ruff configuration or HNSW tuning parameters. The goal is for Charles to understand these domains well enough to:

1. **Specify correctly** when delegating to Claude ("use shlex.split to decompose commands, check each segment against the deny list, handle command substitution patterns")
2. **Validate the output** ("run mypy --strict on the result, verify the cross-encoder improves top-5 relevance on these 10 test queries")
3. **Make architectural decisions** ("use Tree-sitter over LSP because it aligns with edge deployment constraints")

The resources are ordered so that each week's learning directly enables the next week's delegation.

---

*Derived from analysis of 136 commits, 50+ GECK log entries, full codebase review, and the companion Efficacy & Effectiveness assessment. Every recommendation references specific files, decisions, or patterns observed in the Animus project.*
