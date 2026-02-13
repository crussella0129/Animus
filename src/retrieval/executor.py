"""Execute retrieval strategies dispatched by the Manifold router.

Each strategy is a method that takes a RoutingDecision and returns
a list of RetrievalResult. The executor dispatches to the correct
strategy and fuses results using Reciprocal Rank Fusion.
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
    """A single result from any retrieval strategy.

    Results are normalized across strategies with unified scoring,
    source attribution, and deduplication keys.
    """
    text: str
    score: float  # 0.0-1.0, normalized
    source: str  # File path or location
    strategy: str  # Which strategy produced this result
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Key for deduplication across strategies.

        Uses source file + first 100 chars of text to identify
        semantically equivalent results from different strategies.
        """
        text_preview = self.text[:100].strip()
        return f"{self.source}:{text_preview}"


class RetrievalExecutor:
    """Execute retrieval strategies and return unified results.

    Holds references to all retrieval backends (vector store, graph DB,
    embedder, project root). Each strategy is a method that dispatches
    to the appropriate backend.
    """

    def __init__(
        self,
        vector_store: Optional[SQLiteVectorStore] = None,
        embedder: Optional[Embedder] = None,
        graph_db: Optional[GraphDB] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        """Initialize the executor with available backends.

        Args:
            vector_store: Vector database for semantic search
            embedder: Embedding generator for query vectorization
            graph_db: Knowledge graph for structural queries
            project_root: Root directory for keyword (grep) search
        """
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
        then fuses results using Reciprocal Rank Fusion.

        Args:
            decision: Routing decision from classify_query()
            top_k: Maximum number of results to return

        Returns:
            List of RetrievalResult objects, sorted by relevance
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
        self,
        decision: RoutingDecision,
        top_k: int
    ) -> list[RetrievalResult]:
        """Vector similarity search.

        Args:
            decision: Routing decision with semantic_query
            top_k: Number of results to return

        Returns:
            List of results from vector store, sorted by cosine similarity
        """
        if not self._store or not self._embedder:
            return []

        # Embed the query
        query_embedding = self._embedder.embed([decision.semantic_query])[0]

        # Search vector store
        results = self._store.search(query_embedding, top_k=top_k)

        # Convert to unified format
        return [
            RetrievalResult(
                text=r.text,
                score=r.score,
                source=r.metadata.get("source") or r.metadata.get("file", "unknown"),
                strategy="semantic",
                metadata=r.metadata,
            )
            for r in results
        ]

    def _execute_structural(
        self,
        decision: RoutingDecision,
        top_k: int
    ) -> list[RetrievalResult]:
        """Knowledge graph query.

        Args:
            decision: Routing decision with structural_query and operation
            top_k: Number of results to return

        Returns:
            List of results from knowledge graph
        """
        if not self._graph:
            return []

        operation = decision.structural_operation
        symbol = decision.structural_query

        nodes: list[NodeRow] = []

        # Dispatch to appropriate graph operation
        if operation == "callers":
            nodes = self._graph.get_callers(symbol)
        elif operation == "callees":
            nodes = self._graph.get_callees(symbol)
        elif operation == "inheritance":
            nodes = self._graph.get_inheritance_tree(symbol)
        elif operation == "blast_radius":
            radius, _cycles = self._graph.get_blast_radius(symbol, max_depth=3)
            # Flatten depth map into single list
            for depth_nodes in radius.values():
                nodes.extend(depth_nodes)
        else:  # "search" or unknown
            nodes = self._graph.search_nodes(symbol, limit=top_k)

        # Convert nodes to retrieval results
        results = []
        for i, node in enumerate(nodes[:top_k]):
            # Read source code for this node
            source_text = self._read_node_source(node, max_lines=20)

            # Fallback if source reading fails
            if not source_text:
                source_text = f"[{node.kind}] {node.qualified_name}"
                if node.docstring:
                    source_text += f"\n{node.docstring[:200]}"

            results.append(RetrievalResult(
                text=source_text,
                score=1.0 - (i * 0.05),  # Rank-based scoring (0.95, 0.90, 0.85, ...)
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
        self,
        decision: RoutingDecision,
        top_k: int
    ) -> list[RetrievalResult]:
        """Exact text search via grep.

        Args:
            decision: Routing decision with keyword_query
            top_k: Number of results to return

        Returns:
            List of grep matches
        """
        if not self._root or not decision.keyword_query:
            return []

        try:
            # Use grep for fast exact text matching
            result = subprocess.run(
                [
                    "grep", "-rn",
                    "--include=*.py", "--include=*.go", "--include=*.rs",
                    "--include=*.js", "--include=*.ts", "--include=*.jsx", "--include=*.tsx",
                    "--include=*.yaml", "--include=*.toml", "--include=*.md",
                    "--include=*.json", "--include=*.txt",
                    decision.keyword_query,
                    str(self._root)
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.strip().split("\n")[:top_k]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # grep not available or timeout - return empty
            return []

        results = []
        for i, line in enumerate(lines):
            if not line or ":" not in line:
                continue

            # Parse grep output: filepath:lineno:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                filepath, lineno, content = parts[0], parts[1], parts[2]
                results.append(RetrievalResult(
                    text=content.strip(),
                    score=1.0 - (i * 0.05),  # Rank-based
                    source=f"{filepath}:{lineno}",
                    strategy="keyword",
                    metadata={"line_number": lineno, "keyword": decision.keyword_query},
                ))

        return results

    def _execute_hybrid(
        self,
        decision: RoutingDecision,
        top_k: int
    ) -> list[RetrievalResult]:
        """Execute both semantic and structural, then fuse results.

        Uses Reciprocal Rank Fusion to merge results from multiple
        strategies, boosting items that appear in both.

        Args:
            decision: Routing decision (should have both semantic and structural queries)
            top_k: Number of final fused results

        Returns:
            Fused and re-ranked results
        """
        # Execute both strategies in parallel (they're independent)
        semantic_results = self._execute_semantic(decision, top_k)
        structural_results = self._execute_structural(decision, top_k)

        # Fuse using Reciprocal Rank Fusion
        return fuse_results(semantic_results, structural_results, top_k)

    def _read_node_source(self, node: NodeRow, max_lines: int = 20) -> str:
        """Read source code lines for a graph node.

        Args:
            node: Graph node with file_path and line numbers
            max_lines: Maximum lines to read

        Returns:
            Source code text, or empty string if reading fails
        """
        if not node.file_path:
            return ""

        try:
            path = Path(node.file_path)
            if not path.exists():
                return ""

            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start_idx = max(0, node.line_start - 1)
            end_idx = min(len(lines), node.line_start - 1 + max_lines)

            return "\n".join(lines[start_idx:end_idx])
        except Exception:
            return ""


def fuse_results(
    list_a: list[RetrievalResult],
    list_b: list[RetrievalResult],
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Fuse results from two retrieval strategies using Reciprocal Rank Fusion.

    RRF Formula:
        RRF_Score(doc) = Sum over strategies of: 1 / (k + rank_in_strategy)

    Where k is a constant (typically 60) that prevents top-ranked results
    from dominating. Results that appear in BOTH strategies receive scores
    from both lists, naturally boosting them.

    Args:
        list_a: Results from first strategy
        list_b: Results from second strategy
        top_k: Number of results to return after fusion

    Returns:
        Fused results sorted by RRF score (highest first)

    Reference:
        Cormack, Clarke & Büttcher (2009): "Reciprocal Rank Fusion
        outperforms the best individual system"
    """
    k = 60  # RRF constant (standard value from literature)

    # Build RRF scores
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, RetrievalResult] = {}

    # Process first strategy
    for rank, result in enumerate(list_a):
        key = result.dedup_key
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = result

    # Process second strategy
    for rank, result in enumerate(list_b):
        key = result.dedup_key
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

        if key not in result_map:
            result_map[key] = result
        else:
            # Result appears in BOTH strategies - mark it
            result_map[key].metadata["multi_strategy"] = True
            # Combine strategy labels
            existing_strategy = result_map[key].strategy
            if existing_strategy != result.strategy:
                result_map[key].strategy = f"{existing_strategy}+{result.strategy}"

    # Sort by fused RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    # Build final result list
    fused = []
    for key in sorted_keys[:top_k]:
        result = result_map[key]
        result.score = rrf_scores[key]  # Replace with fused RRF score
        fused.append(result)

    return fused
