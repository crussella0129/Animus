# Animus: Manifold: Multi-Strategy Retrieval Router for Animus

## Instructions for Building a Novel Local-First Code Intelligence System

**What this document is:** Step-by-step instructions for an LLM (or a human developer) to build a multi-strategy retrieval router into the Animus codebase. This is not a theoretical proposal — every file path, interface, and data structure referenced here exists in the repo today.

**Why this matters:** No one has built a local-first, multi-strategy retrieval router that runs on edge hardware (8W Jetson, consumer GPU, or CPU-only). Cloud-based versions exist (Pinecone + Neo4j + Elasticsearch with GPT-4 routing), but doing it at the edge with sub-7B models is an open problem. Animus already has the infrastructure for this — two separate retrieval systems that don't know about each other. This document describes how to connect them into something genuinely new.

**Codename:** Animus: Manifold (Multi-Strategy Retrieval Manifold)

---

## Table of Contents

1. [Situational Awareness: What Animus Already Has](#1-situational-awareness)
2. [The Problem Being Solved](#2-the-problem)
3. [Architecture Overview](#3-architecture)
4. [Phase 1: Foundation — AST-Informed Chunking](#4-phase-1)
5. [Phase 2: Contextual Embedding Enhancement](#5-phase-2)
6. [Phase 3: The Retrieval Router](#6-phase-3)
7. [Phase 4: Unified Search Tool](#7-phase-4)
8. [Phase 5: Result Fusion and Ranking](#8-phase-5)
9. [Phase 6: Router Self-Improvement Loop](#9-phase-6)
10. [Testing Strategy](#10-testing)
11. [Performance Constraints](#11-constraints)
12. [File Manifest](#12-manifest)

---

## 1. Situational Awareness

Before writing any code, read and understand the following existing modules. They are the foundation you are building on top of. Do not rewrite them — extend them.

### Retrieval System A: Vector Similarity Search
- **`src/memory/chunker.py`** — Text chunking (token-based and code-aware via regex). Splits files into chunks for embedding.
- **`src/memory/scanner.py`** — Directory walker with .gitignore awareness. Yields file paths.
- **`src/memory/embedder.py`** — Embedding providers. `NativeEmbedder` uses `all-MiniLM-L6-v2` (384-dim). `MockEmbedder` for tests.
- **`src/memory/vectorstore.py`** — Two implementations: `SQLiteVectorStore` (brute-force cosine) and `SQLiteVecVectorStore` (SIMD-accelerated via sqlite-vec extension). Both persist to SQLite with WAL mode.
- **`src/tools/search.py`** — `SearchCodebaseTool`: takes a natural language query, embeds it, searches vectorstore, returns top-k results with scores.

### Retrieval System B: Code Knowledge Graph
- **`src/knowledge/parser.py`** — `PythonParser`: full AST parsing via `ast.parse()`. Extracts `NodeInfo` (modules, classes, functions, methods with docstrings, args, decorators) and `EdgeInfo` (CALLS, INHERITS, CONTAINS, IMPORTS relationships).
- **`src/knowledge/graph_db.py`** — `GraphDB`: SQLite-backed graph storage. Supports `search_nodes()` (LIKE pattern match), `get_callers()`, `get_callees()`, `get_inheritance_tree()`, `get_blast_radius()` (BFS traversal), and phantom "external" node creation for unresolved references.
- **`src/knowledge/indexer.py`** — `Indexer`: orchestrates scan → parse → upsert with incremental change detection (mtime + content hash).
- **`src/tools/graph.py`** — Three tools: `SearchCodeGraphTool`, `GetCallersTool`, `GetBlastRadiusTool`. Each queries GraphDB and formats results.

### Agent Infrastructure
- **`src/tools/base.py`** — `Tool` ABC and `ToolRegistry`. All tools implement `name`, `description`, `parameters` (JSON Schema), and `execute(args) -> str`.
- **`src/core/agent.py`** — Agentic loop with tool call parsing. The agent currently chooses tools based on LLM output — it decides whether to use `search_codebase` or `search_code_graph` on its own, with no routing intelligence.
- **`src/core/context.py`** — `ContextWindow` with tier-based budgeting. `estimate_tokens()` at line 8 is used everywhere (NOTE: this function uses a 4-char-per-token heuristic that is inaccurate for code — fix this as part of Phase 1).
- **`src/core/planner.py`** — Plan-then-execute pipeline with step type inference via keyword matching.

### Ingestion Entry Points
- **`src/main.py` line 240-339** — `ingest` command: Scanner → Chunker → Embedder → VectorStore pipeline. Currently uses `MockEmbedder` (hash-based, not semantic).
- **`src/main.py` line 384-430** — `graph` command: Scanner → PythonParser → GraphDB pipeline. Runs independently from `ingest`.

### Critical Design Constraint
From `docs/DESIGN_PRINCIPLES.md`: **"Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else."** The router's classification logic must be primarily hardcoded. LLM involvement is only for ambiguous edge cases.

---

## 2. The Problem

Animus has two retrieval systems that solve different problems:

**Vector search** answers: "Find code that is *semantically similar* to this natural language description." Good for: "how does authentication work?", "find error handling patterns", "code that processes user input."

**Graph queries** answer: "Find code that is *structurally related* to this specific symbol." Good for: "what calls authenticate()?", "what inherits from BaseProvider?", "what's the blast radius of changing Tool.execute()?"

**Neither system** can answer: "Find the authentication code and show me everything that depends on it" — this requires vector search to find the relevant code, then graph traversal to find dependencies. Currently, the agent must make two separate tool calls and mentally compose the results. Small models (1-7B) frequently fail to do this composition.

**The missing piece** is a router that:
1. Classifies the query intent
2. Dispatches to the right retrieval strategy (or both)
3. Fuses results into a single coherent response
4. Does all of this with hardcoded classification logic that works on small models

---

## 3. Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Animus: Manifold Query Router    │  ← Hardcoded classifier (no LLM)
│                         │
│  Classifies query into: │
│  • SEMANTIC    → vector │
│  • STRUCTURAL  → graph  │
│  • HYBRID      → both   │
│  • KEYWORD     → grep   │
└────┬──────┬──────┬──────┘
     │      │      │
     ▼      ▼      ▼
  ┌─────┐┌─────┐┌─────┐
  │Vec  ││Graph││Grep │    ← Existing systems + new keyword backend
  │Store││ DB  ││     │
  └──┬──┘└──┬──┘└──┬──┘
     │      │      │
     ▼      ▼      ▼
┌─────────────────────────┐
│    Result Fusioner       │  ← Merges, deduplicates, re-ranks
│                         │
│  • Reciprocal Rank      │
│  • Source deduplication  │
│  • Context enrichment   │
└─────────────────────────┘
     │
     ▼
  Unified Results
```

---

## 4. Phase 1: Foundation — AST-Informed Chunking

**Goal:** Connect the existing AST parser to the chunker so chunks carry structural metadata. This is prerequisite infrastructure — Animus: Manifold's router uses chunk metadata for smarter retrieval.

### 4.1 Fix Token Estimation

**File:** `src/core/context.py` line 8 AND `src/memory/chunker.py` line 9

Both files have identical `estimate_tokens()` functions. Consolidate into one and make it content-aware.

**Create or modify:** `src/core/context.py`

```python
def estimate_tokens(text: str, content_type: str = "auto") -> int:
    """Content-aware token estimation.

    Token Count ≈ Character Count / Chars Per Token

    Chars_Per_Token by content type:
        English prose ≈ 4.0
        Source code   ≈ 3.1
        Mixed content ≈ 3.5

    For exact counts, use tiktoken (optional dependency).
    """
    if content_type == "auto":
        # Quick heuristic: check first 500 chars for code indicators
        sample = text[:500]
        code_signals = sum(1 for s in ["def ", "class ", "import ", "func ",
                                        "const ", "return ", "if ", "for "]
                          if s in sample)
        content_type = "code" if code_signals >= 2 else "prose"

    ratio = {"prose": 4.0, "code": 3.1, "mixed": 3.5}.get(content_type, 3.5)
    return max(1, int(len(text) / ratio))
```

Then update `src/memory/chunker.py` to import from context:
```python
from src.core.context import estimate_tokens as _estimate_tokens
```

Remove the duplicate `_estimate_tokens` function from `chunker.py`.

### 4.2 AST-Informed Chunking

**Modify:** `src/memory/chunker.py`

The key insight: `src/knowledge/parser.py` already extracts function boundaries, docstrings, and structural metadata via `ast.parse()`. The chunker should use this for Python files instead of regex splitting.

Add a new method to the `Chunker` class:

```python
def chunk_with_structure(
    self,
    text: str,
    filepath: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Chunk with AST-derived structural metadata when possible.

    For Python files, uses the AST parser to extract semantically
    complete units (functions, classes, methods) with metadata
    including docstrings, qualified names, and argument lists.

    For non-Python files, falls back to the standard chunk() method.

    The structural metadata enables the Animus: Manifold retrieval router
    to make better routing decisions and the result fusioner
    to deduplicate across retrieval strategies.
    """
    from pathlib import Path

    path = Path(filepath)

    # Only use AST chunking for Python files
    if path.suffix != ".py":
        return self.chunk(text, metadata=metadata)

    try:
        from src.knowledge.parser import PythonParser
        parser = PythonParser()
        parse_result = parser.parse_file(path)
    except Exception:
        # AST parse failed — fall back to standard chunking
        return self.chunk(text, metadata=metadata)

    if not parse_result.nodes:
        return self.chunk(text, metadata=metadata)

    lines = text.splitlines()
    chunks = []

    for node in parse_result.nodes:
        if node.kind in ("function", "method", "class"):
            # Extract the source lines for this node
            start = max(0, node.line_start - 1)
            end = node.line_end
            chunk_text = "\n".join(lines[start:end])

            if not chunk_text.strip():
                continue

            # If chunk is too large, sub-chunk it
            if _estimate_tokens(chunk_text) > self.chunk_size:
                sub_chunks = self._chunk_by_tokens(chunk_text)
                for j, sub in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub,
                        "metadata": {
                            **(metadata or {}),
                            "chunk_index": len(chunks),
                            "estimated_tokens": _estimate_tokens(sub),
                            "kind": node.kind,
                            "qualified_name": node.qualified_name,
                            "docstring": node.docstring[:200] if node.docstring else "",
                            "source": filepath,
                            "lines": f"{start+1}-{end}",
                            "sub_chunk": j,
                            "structural": True,
                        },
                    })
            else:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "estimated_tokens": _estimate_tokens(chunk_text),
                        "kind": node.kind,
                        "qualified_name": node.qualified_name,
                        "docstring": node.docstring[:200] if node.docstring else "",
                        "args": node.args,
                        "source": filepath,
                        "lines": f"{start+1}-{end}",
                        "structural": True,
                    },
                })

    # Capture any top-level code not inside functions/classes
    # (imports, module-level assignments, etc.)
    covered_lines = set()
    for node in parse_result.nodes:
        if node.kind in ("function", "method", "class"):
            for i in range(node.line_start, node.line_end + 1):
                covered_lines.add(i)

    uncovered = []
    current_block = []
    for i, line in enumerate(lines, 1):
        if i not in covered_lines:
            current_block.append(line)
        else:
            if current_block:
                block_text = "\n".join(current_block).strip()
                if block_text and _estimate_tokens(block_text) > 10:
                    uncovered.append(block_text)
                current_block = []
    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text and _estimate_tokens(block_text) > 10:
            uncovered.append(block_text)

    for block in uncovered:
        chunks.append({
            "text": block,
            "metadata": {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "estimated_tokens": _estimate_tokens(block),
                "kind": "module_level",
                "source": filepath,
                "structural": True,
            },
        })

    return chunks if chunks else self.chunk(text, metadata=metadata)
```

### 4.3 Update the Ingestion Pipeline

**Modify:** `src/main.py` in the `ingest` command (around line 300)

Change:
```python
chunks = chunker.chunk(text, metadata={"source": path_str})
```

To:
```python
chunks = chunker.chunk_with_structure(text, filepath=path_str, metadata={"source": path_str})
```

This single line change activates AST-informed chunking for all Python files while keeping backward compatibility for non-Python files.

---

## 5. Phase 2: Contextual Embedding Enhancement

**Goal:** Before embedding each chunk, prepend a short context summary that captures where the chunk lives in the codebase. This is based on Anthropic's Contextual Retrieval technique (published late 2024) adapted for local-first operation.

### 5.1 The Core Idea

Standard embedding:
```
"def authenticate(token): ..."  →  embed()  →  vector
```

Contextual embedding:
```
"[From src/auth/handler.py, function authenticate in class AuthService, 
 called by middleware.verify_request and routes.login] 
 def authenticate(token): ..."  →  embed()  →  vector
```

The context prefix means the embedding captures *where the code lives*, not just what it says. A search for "login flow" now matches `authenticate()` because the context mentions `routes.login`.

### 5.2 Implementation

**Create:** `src/memory/contextualizer.py`

```python
"""Contextual prefix generation for chunks before embedding.

Prepends structural context to each chunk so embeddings capture
not just what the code says, but where it lives in the codebase.

Based on Anthropic's Contextual Retrieval technique, adapted for
local-first operation without requiring an LLM call per chunk.
"""

from __future__ import annotations

from typing import Any, Optional

from src.knowledge.graph_db import GraphDB


class ChunkContextualizer:
    """Generate context prefixes for chunks using the knowledge graph.

    This is a HARDCODED contextualizer — it does not use an LLM.
    Context is derived from the graph database relationships:
    callers, callees, inheritance, and imports.

    For LLM-powered contextualization (higher quality but slower),
    see LLMContextualizer below.
    """

    def __init__(self, graph_db: Optional[GraphDB] = None) -> None:
        self._graph = graph_db

    def contextualize(self, chunk: dict[str, Any]) -> str:
        """Prepend structural context to a chunk's text.

        Returns the contextualized text (prefix + original text).
        The original text is NOT modified — only the returned
        string has the prefix.

        Context Format:
            [From {filepath}, {kind} {qualified_name}
             in class {parent_class},
             called by {callers},
             calls {callees}]
            {original chunk text}
        """
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")

        # If no structural metadata, return text as-is
        if not meta.get("structural"):
            return text

        parts = []

        # Source location
        source = meta.get("source", "")
        if source:
            parts.append(f"From {source}")

        # Symbol identity
        kind = meta.get("kind", "")
        qname = meta.get("qualified_name", "")
        if kind and qname:
            parts.append(f"{kind} {qname}")

        # Docstring summary (first sentence)
        docstring = meta.get("docstring", "")
        if docstring:
            first_sentence = docstring.split(".")[0].strip()
            if first_sentence and len(first_sentence) < 100:
                parts.append(f"purpose: {first_sentence}")

        # Graph-derived context (callers, callees)
        if self._graph and qname:
            graph_context = self._get_graph_context(qname)
            if graph_context:
                parts.append(graph_context)

        if not parts:
            return text

        prefix = "[" + ", ".join(parts) + "]\n"
        return prefix + text

    def _get_graph_context(self, qname: str) -> str:
        """Query the knowledge graph for structural relationships.

        Returns a compact string describing callers and callees.
        Limits to 3 each to keep the prefix short.
        """
        context_parts = []

        try:
            callers = self._graph.get_callers(qname)
            if callers:
                caller_names = [c.name for c in callers[:3]]
                suffix = f" +{len(callers)-3} more" if len(callers) > 3 else ""
                context_parts.append(f"called by {', '.join(caller_names)}{suffix}")

            callees = self._graph.get_callees(qname)
            if callees:
                callee_names = [c.name for c in callees[:3]]
                suffix = f" +{len(callees)-3} more" if len(callees) > 3 else ""
                context_parts.append(f"calls {', '.join(callee_names)}{suffix}")
        except Exception:
            pass  # Graph unavailable — degrade gracefully

        return ", ".join(context_parts)

    def contextualize_batch(self, chunks: list[dict[str, Any]]) -> list[str]:
        """Contextualize a batch of chunks. Returns list of contextualized texts."""
        return [self.contextualize(chunk) for chunk in chunks]
```

### 5.3 Wire Into Ingestion

**Modify:** `src/main.py` `ingest` command

After chunking, before embedding, add contextualization:

```python
from src.memory.contextualizer import ChunkContextualizer

# After: chunks = chunker.chunk_with_structure(...)
# Before: embeddings = embedder.embed(texts)

# If graph DB exists, use it for contextualization
graph_db_path = cfg.graph_dir / "code_graph.db"
graph_db = None
if graph_db_path.exists():
    from src.knowledge.graph_db import GraphDB
    graph_db = GraphDB(graph_db_path)

contextualizer = ChunkContextualizer(graph_db=graph_db)

# Contextualize chunks before embedding
contextualized_texts = contextualizer.contextualize_batch(chunks)

# Embed the contextualized text (richer embedding)
embeddings = embedder.embed(contextualized_texts)

# But STORE the original text (for display to user)
original_texts = [c["text"] for c in chunks]
store.add(original_texts, embeddings, ...)
```

**Critical detail:** Embed the *contextualized* text but store the *original* text. The embedding captures structural context, but when results are returned to the agent, it sees clean source code without noise.

### 5.4 Ingestion Order Matters

The contextualized embeddings are better when the graph database already exists. This means the ideal ingestion order is:

```
1. animus graph ./project     ← builds knowledge graph first
2. animus ingest ./project    ← now chunks get graph-enriched context
```

Document this in the CLI help text. Optionally, have `ingest` auto-run `graph` if the graph DB doesn't exist yet.

---

## 6. Phase 3: The Retrieval Router

**Goal:** Build a hardcoded query classifier that dispatches to the right retrieval strategy. This is the core novel component.

### 6.1 Query Intent Classification

**Create:** `src/retrieval/router.py`

```python
"""Animus: Manifold: Multi-Strategy Retrieval Manifold.

Classifies queries and routes them to the optimal retrieval strategy
(or combination of strategies) without using an LLM for classification.

The router is 100% hardcoded — no LLM involvement in routing decisions.
This follows the Animus design principle: "Use LLMs only where ambiguity,
creativity, or natural language understanding is required."

Classification Categories:
    SEMANTIC    → vector similarity search
                  "how does authentication work?"
                  "find error handling code"
                  "code that processes CSV files"

    STRUCTURAL  → knowledge graph queries
                  "what calls authenticate()?"
                  "subclasses of BaseProvider"
                  "imports in agent.py"

    HYBRID      → both, with result fusion
                  "find the auth code and what depends on it"
                  "show me the logging system and its callers"
                  "how is the config loaded and where is it used?"

    KEYWORD     → exact text match (grep-style)
                  "find TODO comments"
                  "lines containing API_KEY"
                  "files with 'deprecated'"

Routing Logic (decision tree):

    Query contains explicit symbol reference?
      YES → contains relationship word? → HYBRID
             no relationship word?      → STRUCTURAL
      NO  → contains exact-match signals? → KEYWORD
             contains conceptual language? → SEMANTIC
             ambiguous?                   → HYBRID (safe default)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalStrategy(Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


@dataclass
class RoutingDecision:
    """Result of query classification."""
    strategy: RetrievalStrategy
    confidence: float  # 0.0-1.0
    reasoning: str  # human-readable explanation
    semantic_query: str = ""  # query to send to vector search
    structural_query: str = ""  # pattern/symbol for graph search
    structural_operation: str = "search"  # search|callers|callees|blast_radius|inheritance
    keyword_query: str = ""  # exact text to grep for
    filters: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------
# Signal patterns for classification
# -----------------------------------------------------------------------

# Patterns that indicate a specific symbol is being referenced
_SYMBOL_PATTERNS = [
    r'\b\w+\(\)',                     # function_name()
    r'\b\w+\.\w+',                   # module.attribute
    r'`[^`]+`',                      # `backtick-quoted`
    r'\b(?:class|def|func)\s+\w+',   # "class Foo", "def bar"
    r'\b\w+(?:Tool|Provider|Store|Handler|Manager|Factory|Registry)\b',  # CamelCase tool names
]

# Patterns that indicate structural/relationship queries
_RELATIONSHIP_PATTERNS = [
    (r'\b(?:call[s|ed|ing]*|invoke[s|d]*|use[s|d]*)\b', "callers"),
    (r'\b(?:caller[s]?|called\s+by|who\s+calls|what\s+calls)\b', "callers"),
    (r'\b(?:callee[s]?|calls\s+to|what\s+does\s+\w+\s+call)\b', "callees"),
    (r'\b(?:inherit[s]?|subclass|extends|derived|child\s+class)\b', "inheritance"),
    (r'\b(?:parent\s+class|base\s+class|superclass)\b', "inheritance"),
    (r'\b(?:import[s|ed]*|depend[s]?|dependenc)', "imports"),
    (r'\b(?:blast\s+radius|impact|affect[s|ed]*|ripple)\b', "blast_radius"),
    (r'\b(?:contain[s]?|inside|within|member[s]?\s+of)\b', "search"),
]

# Patterns that indicate semantic/conceptual queries
_SEMANTIC_PATTERNS = [
    r'\b(?:how|why|explain|describe|what\s+is|what\s+are)\b',
    r'\b(?:find|search|look\s+for|show\s+me)\b(?!.*(?:call|inherit|import))',
    r'\b(?:similar|related|like|pattern|approach|technique)\b',
    r'\b(?:implement|handle|process|manage|validate|transform)\b',
    r'\b(?:error|bug|issue|problem|fix)\b',
    r'\b(?:example|usage|documentation)\b',
]

# Patterns that indicate exact keyword search
_KEYWORD_PATTERNS = [
    r'\b(?:TODO|FIXME|HACK|XXX|DEPRECATED|NOTE)\b',
    r'\b(?:grep|find\s+text|exact\s+match|literal|string)\b',
    r'["\'][^"\']+["\']',  # quoted strings
    r'\b(?:contain(?:s|ing)?)\s+["\']',  # "containing 'text'"
    r'\b(?:lines?\s+with|lines?\s+containing)\b',
]

# Hybrid indicators: query wants both semantic understanding AND structural context
_HYBRID_INDICATORS = [
    r'\b(?:and\s+(?:what|how|where|its?|their))\b',
    r'\b(?:then\s+show|also\s+(?:show|find|get))\b',
    r'\b(?:along\s+with|together\s+with|as\s+well\s+as)\b',
    r'\b(?:everything\s+(?:about|related|connected))\b',
    r'\b(?:full\s+picture|complete\s+view|all\s+about)\b',
    r'\b(?:and\s+(?:depend|call|import|inherit|use))',
]


def classify_query(query: str) -> RoutingDecision:
    """Classify a query into a retrieval strategy.

    Decision Procedure:
        1. Check for hybrid indicators (highest priority — catches compound queries)
        2. Check for keyword signals (exact match requests)
        3. Check for symbol references + relationship words → STRUCTURAL
        4. Check for symbol references without relationship → STRUCTURAL (search mode)
        5. Check for semantic/conceptual language → SEMANTIC
        6. Default → HYBRID (safest fallback for ambiguous queries)

    Returns a RoutingDecision with strategy, confidence, and
    pre-processed query components for each retrieval backend.
    """
    query_lower = query.lower().strip()

    # --- Step 1: Check for hybrid indicators ---
    for pattern in _HYBRID_INDICATORS:
        if re.search(pattern, query_lower):
            return _build_hybrid_decision(query, query_lower,
                                          confidence=0.8,
                                          reasoning="Query combines conceptual and structural elements")

    # --- Step 2: Check for keyword signals ---
    keyword_score = sum(1 for p in _KEYWORD_PATTERNS if re.search(p, query, re.IGNORECASE))
    if keyword_score >= 1:
        keyword_term = _extract_keyword_term(query)
        if keyword_term:
            return RoutingDecision(
                strategy=RetrievalStrategy.KEYWORD,
                confidence=0.85,
                reasoning=f"Query requests exact text match for '{keyword_term}'",
                keyword_query=keyword_term,
            )

    # --- Step 3: Check for symbol references ---
    has_symbol = any(re.search(p, query) for p in _SYMBOL_PATTERNS)

    if has_symbol:
        # Check for relationship words
        for pattern, operation in _RELATIONSHIP_PATTERNS:
            if re.search(pattern, query_lower):
                symbol = _extract_symbol(query)
                return RoutingDecision(
                    strategy=RetrievalStrategy.STRUCTURAL,
                    confidence=0.9,
                    reasoning=f"Query asks about {operation} of symbol '{symbol}'",
                    structural_query=symbol,
                    structural_operation=operation,
                )

        # Symbol reference without relationship → structural search
        symbol = _extract_symbol(query)
        return RoutingDecision(
            strategy=RetrievalStrategy.STRUCTURAL,
            confidence=0.75,
            reasoning=f"Query references specific symbol '{symbol}'",
            structural_query=symbol,
            structural_operation="search",
        )

    # --- Step 4: Check for semantic/conceptual language ---
    semantic_score = sum(1 for p in _SEMANTIC_PATTERNS if re.search(p, query_lower))
    if semantic_score >= 1:
        return RoutingDecision(
            strategy=RetrievalStrategy.SEMANTIC,
            confidence=min(0.9, 0.6 + semantic_score * 0.1),
            reasoning="Query uses conceptual/descriptive language",
            semantic_query=query,
        )

    # --- Step 5: Default to HYBRID ---
    return _build_hybrid_decision(query, query_lower,
                                  confidence=0.5,
                                  reasoning="Ambiguous query — using hybrid retrieval for safety")


def _extract_symbol(query: str) -> str:
    """Extract the most likely symbol name from a query.

    Priority:
        1. Backtick-quoted: `symbol_name`
        2. Function call: symbol_name()
        3. Dotted name: module.symbol
        4. CamelCase word: MyClassName
        5. First word that looks like an identifier
    """
    # Backtick-quoted
    match = re.search(r'`([^`]+)`', query)
    if match:
        return match.group(1)

    # Function call
    match = re.search(r'\b(\w+)\(\)', query)
    if match:
        return match.group(1)

    # Dotted name
    match = re.search(r'\b(\w+\.\w+(?:\.\w+)*)\b', query)
    if match:
        return match.group(1)

    # CamelCase
    match = re.search(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', query)
    if match:
        return match.group(1)

    # Fallback: longest word that looks like a code identifier
    words = re.findall(r'\b[a-zA-Z_]\w{2,}\b', query)
    # Filter out common English words
    _STOP_WORDS = {"the", "and", "for", "that", "this", "with", "from",
                   "what", "how", "where", "find", "show", "get", "all",
                   "are", "does", "about", "which"}
    code_words = [w for w in words if w.lower() not in _STOP_WORDS]
    if code_words:
        return max(code_words, key=len)

    return query.strip()


def _extract_keyword_term(query: str) -> str:
    """Extract the exact text to search for in keyword mode."""
    # Quoted string
    match = re.search(r'["\']([^"\']+)["\']', query)
    if match:
        return match.group(1)

    # TODO/FIXME/etc.
    match = re.search(r'\b(TODO|FIXME|HACK|XXX|DEPRECATED|NOTE)\b', query, re.IGNORECASE)
    if match:
        return match.group(1)

    return ""


def _build_hybrid_decision(query: str, query_lower: str,
                           confidence: float, reasoning: str) -> RoutingDecision:
    """Build a HYBRID routing decision with both semantic and structural components."""
    symbol = _extract_symbol(query)

    # Determine structural operation if relationship words present
    operation = "search"
    for pattern, op in _RELATIONSHIP_PATTERNS:
        if re.search(pattern, query_lower):
            operation = op
            break

    return RoutingDecision(
        strategy=RetrievalStrategy.HYBRID,
        confidence=confidence,
        reasoning=reasoning,
        semantic_query=query,
        structural_query=symbol,
        structural_operation=operation,
    )
```

---

## 7. Phase 4: Unified Search Tool

**Goal:** Replace the separate `search_codebase` and `search_code_graph` tools with a single `search` tool that uses the router internally.

### 7.1 The Retrieval Executor

**Create:** `src/retrieval/executor.py`

```python
"""Execute retrieval strategies dispatched by the Animus: Manifold router.

Each strategy is a function that takes a RoutingDecision and
returns a list of RetrievalResult. The executor dispatches
to the correct strategy and returns unified results.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.knowledge.graph_db import GraphDB, NodeRow
from src.memory.embedder import Embedder
from src.memory.vectorstore import SQLiteVectorStore, SearchResult
from src.retrieval.router import RetrievalStrategy, RoutingDecision


@dataclass
class RetrievalResult:
    """A single result from any retrieval strategy."""
    text: str
    score: float  # 0.0-1.0, normalized
    source: str  # file path
    strategy: str  # which strategy produced this result
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Key for deduplication across strategies.
        Uses source file + first 100 chars of text.
        """
        return f"{self.source}:{self.text[:100]}"


class RetrievalExecutor:
    """Execute retrieval strategies and return unified results.

    Holds references to all retrieval backends. Each strategy
    is a method that takes a RoutingDecision and returns results.
    """

    def __init__(
        self,
        vector_store: Optional[SQLiteVectorStore] = None,
        embedder: Optional[Embedder] = None,
        graph_db: Optional[GraphDB] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._graph = graph_db
        self._root = project_root

    def execute(
        self,
        decision: RoutingDecision,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Execute the retrieval strategy specified in the routing decision.

        For HYBRID strategy, executes both semantic and structural,
        then fuses results.
        """
        strategy_map = {
            RetrievalStrategy.SEMANTIC: self._execute_semantic,
            RetrievalStrategy.STRUCTURAL: self._execute_structural,
            RetrievalStrategy.KEYWORD: self._execute_keyword,
            RetrievalStrategy.HYBRID: self._execute_hybrid,
        }

        executor = strategy_map.get(decision.strategy, self._execute_hybrid)
        return executor(decision, top_k)

    def _execute_semantic(
        self, decision: RoutingDecision, top_k: int
    ) -> list[RetrievalResult]:
        """Vector similarity search."""
        if not self._store or not self._embedder:
            return []

        query_embedding = self._embedder.embed([decision.semantic_query])[0]
        results = self._store.search(query_embedding, top_k=top_k)

        return [
            RetrievalResult(
                text=r.text,
                score=r.score,
                source=r.metadata.get("source", "unknown"),
                strategy="semantic",
                metadata=r.metadata,
            )
            for r in results
        ]

    def _execute_structural(
        self, decision: RoutingDecision, top_k: int
    ) -> list[RetrievalResult]:
        """Knowledge graph query."""
        if not self._graph:
            return []

        operation = decision.structural_operation
        symbol = decision.structural_query

        nodes: list[NodeRow] = []

        if operation == "callers":
            nodes = self._graph.get_callers(symbol)
        elif operation == "callees":
            nodes = self._graph.get_callees(symbol)
        elif operation == "inheritance":
            nodes = self._graph.get_inheritance_tree(symbol)
        elif operation == "blast_radius":
            radius = self._graph.get_blast_radius(symbol, max_depth=3)
            for depth_nodes in radius.values():
                nodes.extend(depth_nodes)
        else:  # "search"
            nodes = self._graph.search_nodes(symbol, limit=top_k)

        results = []
        for i, node in enumerate(nodes[:top_k]):
            # Read source code for this node if file exists
            source_text = self._read_node_source(node)

            results.append(RetrievalResult(
                text=source_text or f"[{node.kind}] {node.qualified_name}",
                score=1.0 - (i * 0.05),  # rank-based scoring
                source=node.file_path or "unknown",
                strategy="structural",
                metadata={
                    "kind": node.kind,
                    "qualified_name": node.qualified_name,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "docstring": node.docstring,
                },
            ))

        return results

    def _execute_keyword(
        self, decision: RoutingDecision, top_k: int
    ) -> list[RetrievalResult]:
        """Exact text search via grep."""
        if not self._root or not decision.keyword_query:
            return []

        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.go",
                 "--include=*.rs", "--include=*.js", "--include=*.ts",
                 "--include=*.yaml", "--include=*.toml", "--include=*.md",
                 decision.keyword_query, str(self._root)],
                capture_output=True, text=True, timeout=10,
            )
            lines = result.stdout.strip().split("\n")[:top_k]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        results = []
        for i, line in enumerate(lines):
            if ":" not in line:
                continue
            parts = line.split(":", 2)
            if len(parts) >= 3:
                filepath, lineno, content = parts[0], parts[1], parts[2]
                results.append(RetrievalResult(
                    text=content.strip(),
                    score=1.0 - (i * 0.05),
                    source=f"{filepath}:{lineno}",
                    strategy="keyword",
                    metadata={"line_number": lineno},
                ))

        return results

    def _execute_hybrid(
        self, decision: RoutingDecision, top_k: int
    ) -> list[RetrievalResult]:
        """Execute both semantic and structural, then fuse results."""
        semantic_results = self._execute_semantic(decision, top_k)
        structural_results = self._execute_structural(decision, top_k)

        return fuse_results(semantic_results, structural_results, top_k)

    def _read_node_source(self, node: NodeRow, max_lines: int = 20) -> str:
        """Read source code lines for a graph node."""
        if not node.file_path:
            return ""
        try:
            path = Path(node.file_path)
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, node.line_start - 1)
            end = min(len(lines), node.line_start + max_lines - 1)
            return "\n".join(lines[start:end])
        except Exception:
            return ""


def fuse_results(
    list_a: list[RetrievalResult],
    list_b: list[RetrievalResult],
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Fuse results from two retrieval strategies using Reciprocal Rank Fusion.

    RRF Score = Sum over strategies of: 1 / (k + rank_in_strategy)

    Where k is a constant (typically 60) that prevents high-ranked
    results from dominating.

    Results that appear in BOTH strategies get boosted because they
    receive RRF scores from both lists. This naturally surfaces
    results that are both semantically relevant AND structurally important.
    """
    k = 60  # RRF constant

    # Build RRF scores
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(list_a):
        key = result.dedup_key
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = result

    for rank, result in enumerate(list_b):
        key = result.dedup_key
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = result
        else:
            # If present in both, note it in metadata
            result_map[key].metadata["multi_strategy"] = True

    # Sort by fused score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    fused = []
    for key in sorted_keys[:top_k]:
        result = result_map[key]
        result.score = rrf_scores[key]  # Replace with fused score
        fused.append(result)

    return fused
```

### 7.2 The Unified Search Tool

**Create:** `src/tools/manifold_search.py`

```python
"""Unified search tool powered by the Animus: Manifold retrieval router.

Replaces the separate search_codebase and search_code_graph tools
with a single tool that automatically routes queries to the optimal
retrieval strategy.
"""

from __future__ import annotations

from typing import Any, Optional

from src.retrieval.executor import RetrievalExecutor, RetrievalResult
from src.retrieval.router import classify_query
from src.tools.base import Tool, ToolRegistry


class ManifoldSearchTool(Tool):
    """Unified codebase search with automatic strategy routing.

    Accepts any natural language query about the codebase and
    automatically routes it to the optimal retrieval strategy:

    - Semantic search for conceptual questions
    - Graph queries for structural questions
    - Hybrid retrieval for compound questions
    - Keyword search for exact text matching

    The routing is hardcoded (no LLM involvement in classification).
    """

    def __init__(self, executor: RetrievalExecutor) -> None:
        super().__init__()
        self._executor = executor

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the codebase using automatic strategy routing. "
            "Handles semantic queries ('how does auth work?'), "
            "structural queries ('what calls authenticate()?'), "
            "and compound queries ('find the auth code and its dependencies'). "
            "Automatically picks the best retrieval strategy."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or code query about the codebase",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default: 8)",
                },
            },
            "required": ["query"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        query = args["query"]
        top_k = args.get("top_k", 8)

        # Route the query
        decision = classify_query(query)

        # Execute the retrieval strategy
        results = self._executor.execute(decision, top_k=top_k)

        if not results:
            return (
                f"No results found for '{query}'.\n"
                f"Strategy used: {decision.strategy.value}\n"
                f"Reason: {decision.reasoning}"
            )

        # Format results
        lines = [
            f"[Strategy: {decision.strategy.value} "
            f"(confidence: {decision.confidence:.0%})]",
            f"[Reason: {decision.reasoning}]",
            "",
        ]

        for i, r in enumerate(results, 1):
            strategy_tag = f"[{r.strategy}]" if r.strategy else ""
            multi = " ★" if r.metadata.get("multi_strategy") else ""
            source_display = r.source
            qname = r.metadata.get("qualified_name", "")
            if qname:
                source_display = f"{qname} ({r.source})"

            lines.append(f"[{i}] (score={r.score:.3f}) {strategy_tag}{multi} {source_display}")
            # Show first 300 chars of text
            preview = r.text[:300].replace("\n", "\n    ")
            lines.append(f"    {preview}")
            lines.append("")

        return "\n".join(lines)


def register_manifold_search(
    registry: ToolRegistry,
    executor: RetrievalExecutor,
) -> None:
    """Register the Animus: Manifold unified search tool.

    This replaces the separate search_codebase and graph search tools
    with a single unified tool.
    """
    registry.register(ManifoldSearchTool(executor))
```

### 7.3 Registration

**Modify:** wherever tools are registered in `src/main.py` (the `rise` command or wherever the agent is initialized). Replace:

```python
register_search_tools(registry, store, embedder)
register_graph_tools(registry, graph_db)
```

With:

```python
from src.retrieval.executor import RetrievalExecutor
from src.tools.hydra_search import register_manifold_search

executor = RetrievalExecutor(
    vector_store=store,
    embedder=embedder,
    graph_db=graph_db,
    project_root=project_path,
)
register_manifold_search(registry, executor)
```

**Keep the old tools registered too** (backward compatibility). The agent can use either `search` (routed) or the specific tools directly.

---

## 8. Phase 5: Result Fusion and Ranking

The Reciprocal Rank Fusion (RRF) algorithm in `executor.py` handles the basic fusion case. Here are additional enhancements to consider:

### 8.1 Boost Results That Appear in Multiple Strategies

Already implemented via the `multi_strategy` flag. Results marked with ★ in the output appeared in both semantic and structural results — these are the highest-confidence results because they're both semantically relevant AND structurally important.

### 8.2 Context Window Awareness

The fused results should respect the agent's context budget. Add a token-aware truncation:

```python
def truncate_results_to_budget(
    results: list[RetrievalResult],
    max_tokens: int = 2000,
) -> list[RetrievalResult]:
    """Truncate result list to fit within a token budget.

    Keeps highest-scored results until the token budget is exhausted.
    """
    from src.core.context import estimate_tokens

    truncated = []
    used = 0
    for result in results:
        tokens = estimate_tokens(result.text)
        if used + tokens > max_tokens:
            # Try truncating this result's text
            remaining = max_tokens - used
            if remaining > 50:  # worth including a truncated version
                char_limit = remaining * 3  # rough chars-per-token
                result.text = result.text[:char_limit] + "\n[truncated]"
                truncated.append(result)
            break
        truncated.append(result)
        used += tokens

    return truncated
```

---

## 9. Phase 6: Router Self-Improvement Loop

**Goal:** The router tracks which strategies produce results the agent actually *uses*, and adjusts routing over time.

### 9.1 Feedback Signal

When the agent uses a search result (references it in a tool call, includes it in a response, or modifies a file mentioned in the result), that's a positive signal. When the agent ignores all results from a strategy, that's a negative signal.

**Create:** `src/retrieval/feedback.py`

```python
"""Lightweight feedback tracking for Animus: Manifold routing decisions.

Tracks which strategies produce results the agent actually uses,
enabling future routing improvements. Stores feedback in SQLite
alongside the vector store.

This does NOT modify the router's classification logic at runtime.
It collects data for offline analysis and manual tuning of the
classification patterns in router.py.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_FEEDBACK_SCHEMA = """\
CREATE TABLE IF NOT EXISTS routing_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    query TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence REAL NOT NULL,
    result_count INTEGER NOT NULL,
    results_used INTEGER DEFAULT 0,
    user_satisfaction TEXT DEFAULT 'unknown'
);
"""


@dataclass
class RoutingFeedback:
    query: str
    strategy: str
    confidence: float
    result_count: int
    results_used: int = 0


class FeedbackStore:
    """Track routing decision outcomes for analysis."""

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_FEEDBACK_SCHEMA)
        self._conn.commit()

    def record(self, feedback: RoutingFeedback) -> None:
        self._conn.execute(
            "INSERT INTO routing_feedback "
            "(timestamp, query, strategy, confidence, result_count, results_used) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), feedback.query, feedback.strategy,
             feedback.confidence, feedback.result_count, feedback.results_used),
        )
        self._conn.commit()

    def get_strategy_stats(self) -> dict[str, dict]:
        """Get aggregated stats per strategy for tuning analysis."""
        rows = self._conn.execute("""
            SELECT strategy,
                   COUNT(*) as total,
                   AVG(confidence) as avg_confidence,
                   SUM(results_used) as total_used,
                   SUM(result_count) as total_results
            FROM routing_feedback
            GROUP BY strategy
        """).fetchall()

        stats = {}
        for strategy, total, avg_conf, used, results in rows:
            utilization = used / results if results > 0 else 0
            stats[strategy] = {
                "total_queries": total,
                "avg_confidence": round(avg_conf, 3),
                "utilization_rate": round(utilization, 3),
            }
        return stats

    def close(self) -> None:
        self._conn.close()
```

### 9.2 CLI Command for Analysis

Add a command to `main.py`:

```python
@app.command()
def routing_stats() -> None:
    """Show Animus: Manifold routing performance statistics."""
    from src.retrieval.feedback import FeedbackStore

    cfg = AnimusConfig.load()
    db_path = cfg.vector_dir / "routing_feedback.db"
    if not db_path.exists():
        info("No routing feedback data yet.")
        return

    store = FeedbackStore(db_path)
    stats = store.get_strategy_stats()
    store.close()

    table = Table(title="Animus: Manifold Routing Statistics")
    table.add_column("Strategy", style="cyan")
    table.add_column("Queries", style="green")
    table.add_column("Avg Confidence", style="yellow")
    table.add_column("Utilization", style="magenta")

    for strategy, data in stats.items():
        table.add_row(
            strategy,
            str(data["total_queries"]),
            f"{data['avg_confidence']:.1%}",
            f"{data['utilization_rate']:.1%}",
        )
    console.print(table)
```

---

## 10. Testing Strategy

### 10.1 Router Classification Tests

**Create:** `tests/test_router.py`

Test the hardcoded classifier against a matrix of query types:

```python
import pytest
from src.retrieval.router import classify_query, RetrievalStrategy

# Semantic queries
@pytest.mark.parametrize("query", [
    "how does authentication work",
    "find error handling patterns",
    "code that processes user input",
    "explain the configuration system",
    "show me the logging approach",
])
def test_semantic_classification(query):
    decision = classify_query(query)
    assert decision.strategy == RetrievalStrategy.SEMANTIC

# Structural queries
@pytest.mark.parametrize("query", [
    "what calls authenticate()",
    "callers of Agent.run",
    "subclasses of ModelProvider",
    "what does ToolRegistry.execute call",
    "imports in agent.py",
])
def test_structural_classification(query):
    decision = classify_query(query)
    assert decision.strategy == RetrievalStrategy.STRUCTURAL

# Hybrid queries
@pytest.mark.parametrize("query", [
    "find the auth code and what depends on it",
    "show me the config system and its callers",
    "everything about the permission checker",
    "how is chunking implemented and what uses it",
])
def test_hybrid_classification(query):
    decision = classify_query(query)
    assert decision.strategy == RetrievalStrategy.HYBRID

# Keyword queries
@pytest.mark.parametrize("query", [
    "find TODO comments",
    "lines containing 'API_KEY'",
    "grep for DEPRECATED",
])
def test_keyword_classification(query):
    decision = classify_query(query)
    assert decision.strategy == RetrievalStrategy.KEYWORD
```

### 10.2 Integration Test

Test the full pipeline: ingest a sample project, run queries through Animus: Manifold, verify results come from the expected strategy.

### 10.3 Fusion Test

Verify that Reciprocal Rank Fusion correctly boosts results appearing in multiple strategies and that `multi_strategy` flag is set.

---

## 11. Performance Constraints

### Edge Hardware Budget

On a Jetson Orin Nano Super (8GB, ~40 TOPS INT8):

| Operation | Budget | Notes |
|-----------|--------|-------|
| Router classification | < 1ms | Pure regex/string ops, no ML |
| Vector search (sqlite-vec) | < 50ms | SIMD-accelerated KNN |
| Graph query | < 20ms | SQLite indexed queries |
| Keyword search (grep) | < 100ms | Subprocess, capped at 10 results |
| Result fusion (RRF) | < 1ms | Simple arithmetic |
| **Total Animus: Manifold query** | **< 200ms** | **Without embedding the query** |
| Query embedding | ~50-200ms | MiniLM on GPU, depends on batch |
| **Total with embedding** | **< 400ms** | **Acceptable for interactive use** |

### Memory Budget

| Component | Memory | Notes |
|-----------|--------|-------|
| Router patterns (compiled regex) | ~100KB | Compile on startup |
| VectorStore connection | ~5MB | SQLite WAL + index |
| GraphDB connection | ~3MB | SQLite WAL + indexes |
| Embedding model (MiniLM) | ~90MB | Loaded once, shared |
| **Total Animus: Manifold overhead** | **~100MB** | **On top of existing Animus** |

---

## 12. File Manifest

New files to create:

```
src/retrieval/             ← NEW PACKAGE
src/retrieval/__init__.py
src/retrieval/router.py    ← Query classifier (Phase 3)
src/retrieval/executor.py  ← Strategy executor + RRF fusion (Phase 4-5)
src/retrieval/feedback.py  ← Routing feedback tracking (Phase 6)

src/memory/contextualizer.py  ← Context prefix generator (Phase 2)

src/tools/manifold_search.py     ← Unified search tool (Phase 4)

tests/test_router.py           ← Router classification tests
tests/test_executor.py         ← Executor + fusion tests
tests/test_contextualizer.py   ← Contextualizer tests
```

Files to modify:

```
src/core/context.py        ← Fix estimate_tokens() (Phase 1)
src/memory/chunker.py      ← Add chunk_with_structure() (Phase 1), remove duplicate estimate_tokens
src/main.py                ← Wire new ingestion pipeline + register Animus: Manifold tool + routing_stats command
```

Files to NOT modify (read-only dependencies):

```
src/knowledge/parser.py    ← Used by chunker, don't change
src/knowledge/graph_db.py  ← Used by executor, don't change
src/memory/vectorstore.py  ← Used by executor, don't change
src/memory/embedder.py     ← Used by executor, don't change
src/tools/base.py          ← Tool ABC, don't change
```

---

## Summary

This is not a research project. Every component described here can be built with the existing Animus infrastructure. The novelty is in the *combination*:

1. **AST-informed chunking** that produces structurally-aware chunks (existing parser, new wiring)
2. **Contextual embeddings** enriched by the knowledge graph (existing graph DB, new prefix generator)
3. **Hardcoded query router** that classifies intent without an LLM (new, but pure regex/heuristics)
4. **Reciprocal Rank Fusion** across retrieval strategies (new, but ~50 lines of math)
5. **All running on edge hardware** within a 400ms query budget (constraint-driven design)

Nobody has put these five things together in a local-first system. Cloud RAG systems have pieces of this, but they use GPT-4 for routing (expensive, slow, cloud-dependent). Academic papers describe each component, but don't ship them as a working system. Animus already has the hardest parts built — the knowledge graph, the vector store, the tool framework. Animus: Manifold is the orchestration layer that makes them work as one.

Build it phase by phase. Test each phase independently. Ship Phase 1-3 first (they're self-contained). Then add Phase 4-6 once the foundation is proven.

*The name of the game isn't who has the biggest model. It's who gets the most signal per watt.*
