"""Contextual prefix generation for chunks before embedding.

Prepends structural context to each chunk so embeddings capture
not just what the code says, but where it lives in the codebase.

Based on Anthropic's Contextual Retrieval technique, adapted for
local-first operation without requiring an LLM call per chunk.

Part of Animus: Manifold multi-strategy retrieval system.
"""

from __future__ import annotations

from typing import Any, Optional

from src.knowledge.graph_db import GraphDB


class ChunkContextualizer:
    """Generate context prefixes for chunks using the knowledge graph.

    This is a HARDCODED contextualizer — it does not use an LLM.
    Context is derived from the graph database relationships:
    callers, callees, inheritance, and imports.

    Example transformation:
        Original chunk: "def authenticate(token): ..."
        Contextualized: "[From src/auth/handler.py, function authenticate
                          in class AuthService, called by middleware.verify
                          and routes.login] def authenticate(token): ..."

    The embedding now captures WHERE the code lives, not just WHAT it says.
    """

    def __init__(self, graph_db: Optional[GraphDB] = None) -> None:
        """Initialize the contextualizer.

        Args:
            graph_db: Optional knowledge graph for relationship context.
                      If None, only file-level context is added.
        """
        self._graph = graph_db

    def contextualize(self, chunk: dict[str, Any]) -> str:
        """Prepend structural context to a chunk's text.

        Args:
            chunk: Chunk dict with 'text' and 'metadata' keys

        Returns:
            Contextualized text with prefix (original text is not modified)

        Context Format:
            [From {filepath}, {kind} {qualified_name},
             purpose: {docstring_summary},
             called by {callers}, calls {callees}]
            {original chunk text}
        """
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")

        # If no structural metadata, return text as-is
        if not meta.get("structural") and not meta.get("chunking_method") == "ast":
            return text

        context_parts = []

        # Source location
        source = meta.get("source") or meta.get("file", "")
        if source:
            context_parts.append(f"From {source}")

        # Symbol identity (kind + qualified name)
        kind = meta.get("kind", "")
        qname = meta.get("qualified_name", "")
        if kind and qname:
            context_parts.append(f"{kind} {qname}")

        # Docstring summary (first sentence for conciseness)
        docstring = meta.get("docstring", "")
        if docstring:
            first_sentence = docstring.split(".")[0].strip()
            if first_sentence and len(first_sentence) < 100:
                context_parts.append(f"purpose: {first_sentence}")

        # Graph-derived context (callers, callees, inheritance)
        if self._graph and qname:
            graph_context = self._get_graph_context(qname)
            if graph_context:
                context_parts.append(graph_context)

        # If no context could be extracted, return original
        if not context_parts:
            return text

        # Build prefix and prepend to text
        prefix = "[" + ", ".join(context_parts) + "]\n"
        return prefix + text

    def _get_graph_context(self, qname: str) -> str:
        """Query the knowledge graph for structural relationships.

        Args:
            qname: Qualified name to query (e.g., "module.ClassName.method_name")

        Returns:
            Compact string describing callers, callees, and inheritance.
            Limits to 3 each to keep the prefix concise.
        """
        context_parts = []

        try:
            # Get callers (who calls this symbol)
            callers = self._graph.get_callers(qname)
            if callers:
                caller_names = [c.name for c in callers[:3]]
                suffix = f" +{len(callers)-3} more" if len(callers) > 3 else ""
                context_parts.append(f"called by {', '.join(caller_names)}{suffix}")

            # Get callees (what this symbol calls)
            callees = self._graph.get_callees(qname)
            if callees:
                callee_names = [c.name for c in callees[:3]]
                suffix = f" +{len(callees)-3} more" if len(callees) > 3 else ""
                context_parts.append(f"calls {', '.join(callee_names)}{suffix}")

            # Get inheritance info (for classes)
            # Note: get_inheritance_tree returns both parents and children
            inheritance = self._graph.get_inheritance_tree(qname)
            if inheritance:
                # Separate into parents (bases) and children (subclasses)
                bases = [n for n in inheritance if "base" in n.kind.lower() or n.line_start < 0]
                subs = [n for n in inheritance if n not in bases and n.qualified_name != qname]

                if bases:
                    base_names = [b.name for b in bases[:2]]
                    context_parts.append(f"inherits from {', '.join(base_names)}")

                if subs:
                    sub_names = [s.name for s in subs[:2]]
                    suffix = f" +{len(subs)-2} more" if len(subs) > 2 else ""
                    context_parts.append(f"inherited by {', '.join(sub_names)}{suffix}")

        except Exception:
            # Graph query failed — degrade gracefully
            pass

        return ", ".join(context_parts)

    def contextualize_batch(self, chunks: list[dict[str, Any]]) -> list[str]:
        """Contextualize a batch of chunks.

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'

        Returns:
            List of contextualized text strings (one per chunk)

        Usage:
            contextualizer = ChunkContextualizer(graph_db)
            contextualized_texts = contextualizer.contextualize_batch(chunks)
            embeddings = embedder.embed(contextualized_texts)
            # Store original texts, not contextualized versions
            store.add([c["text"] for c in chunks], embeddings, ...)
        """
        return [self.contextualize(chunk) for chunk in chunks]

    def estimate_context_overhead(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Estimate how much context is being added to chunks.

        Useful for debugging and understanding contextualization impact.

        Args:
            chunks: List of chunk dicts

        Returns:
            Stats dict with average prefix length, coverage rate, etc.
        """
        from src.core.context import estimate_tokens

        total_chunks = len(chunks)
        contextualized = 0
        total_prefix_tokens = 0

        for chunk in chunks:
            original = chunk.get("text", "")
            contextualized_text = self.contextualize(chunk)

            if len(contextualized_text) > len(original):
                contextualized += 1
                prefix = contextualized_text[:len(contextualized_text) - len(original)]
                total_prefix_tokens += estimate_tokens(prefix)

        avg_prefix_tokens = total_prefix_tokens / contextualized if contextualized > 0 else 0

        return {
            "total_chunks": total_chunks,
            "contextualized_chunks": contextualized,
            "coverage_rate": contextualized / total_chunks if total_chunks > 0 else 0,
            "avg_prefix_tokens": round(avg_prefix_tokens, 1),
            "total_prefix_tokens": total_prefix_tokens,
        }
