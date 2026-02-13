"""Animus: Manifold - Multi-Strategy Retrieval System.

Intelligent retrieval routing that combines vector search, knowledge graph
queries, and keyword search into a unified interface.
"""

from src.retrieval.router import (
    RetrievalStrategy,
    RoutingDecision,
    classify_query,
)

__all__ = [
    "RetrievalStrategy",
    "RoutingDecision",
    "classify_query",
]
