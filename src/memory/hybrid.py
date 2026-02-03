"""Hybrid search combining BM25 keyword search with vector similarity.

This module provides BM25-based keyword search that can be combined with
vector embeddings for more robust retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.memory.vectorstore import Document, SearchResult, VectorStore


@dataclass
class BM25Config:
    """Configuration for BM25 scoring."""
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Document length normalization (0 = no normalization, 1 = full)
    epsilon: float = 0.25  # Floor for IDF to handle unseen terms


class BM25Index:
    """
    BM25 keyword index for text retrieval.

    Implements the Okapi BM25 ranking function for keyword-based search.
    Can be combined with vector search for hybrid retrieval.
    """

    def __init__(self, config: Optional[BM25Config] = None):
        """Initialize BM25 index.

        Args:
            config: BM25 configuration parameters.
        """
        self.config = config or BM25Config()
        self._documents: dict[str, Document] = {}
        self._doc_freqs: Counter[str] = Counter()  # Document frequency per term
        self._doc_lengths: dict[str, int] = {}  # Token count per document
        self._avg_doc_length: float = 0.0
        self._tokenized_docs: dict[str, Counter[str]] = {}  # Term frequencies per doc

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Simple tokenization: lowercase, split on non-alphanumeric,
        filter short tokens.
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z0-9_]+\b', text.lower())
        # Filter very short tokens (likely not meaningful)
        return [t for t in tokens if len(t) > 1]

    def add(self, documents: list[Document]) -> None:
        """Add documents to the BM25 index.

        Args:
            documents: Documents to index.
        """
        for doc in documents:
            if doc.id in self._documents:
                # Remove old document stats first
                self._remove_doc_stats(doc.id)

            self._documents[doc.id] = doc

            # Tokenize and count terms
            tokens = self._tokenize(doc.content)
            term_freqs = Counter(tokens)
            self._tokenized_docs[doc.id] = term_freqs
            self._doc_lengths[doc.id] = len(tokens)

            # Update document frequencies
            for term in term_freqs:
                self._doc_freqs[term] += 1

        # Update average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0

    def _remove_doc_stats(self, doc_id: str) -> None:
        """Remove a document's contribution to the index stats."""
        if doc_id in self._tokenized_docs:
            for term in self._tokenized_docs[doc_id]:
                self._doc_freqs[term] -= 1
                if self._doc_freqs[term] <= 0:
                    del self._doc_freqs[term]
            del self._tokenized_docs[doc_id]
            del self._doc_lengths[doc_id]

    def delete(self, ids: list[str]) -> None:
        """Remove documents from the index.

        Args:
            ids: Document IDs to remove.
        """
        for doc_id in ids:
            if doc_id in self._documents:
                self._remove_doc_stats(doc_id)
                del self._documents[doc_id]

        # Recalculate average
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        n = len(self._documents)
        if n == 0:
            return 0.0

        df = self._doc_freqs.get(term, 0)

        # IDF with smoothing to handle unseen terms
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        return max(idf, self.config.epsilon)

    def _score_document(self, query_terms: list[str], doc_id: str) -> float:
        """Calculate BM25 score for a document given query terms."""
        if doc_id not in self._tokenized_docs:
            return 0.0

        term_freqs = self._tokenized_docs[doc_id]
        doc_length = self._doc_lengths[doc_id]

        score = 0.0
        k1 = self.config.k1
        b = self.config.b
        avgdl = self._avg_doc_length if self._avg_doc_length > 0 else 1.0

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 scoring formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search for documents matching the query.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Score all documents
        scores: list[tuple[str, float]] = []
        for doc_id in self._documents:
            score = self._score_document(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for doc_id, score in scores[:k]:
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=score,
            ))

        return results

    def count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    vector_weight: float = 0.5  # Weight for vector similarity (0-1)
    keyword_weight: float = 0.5  # Weight for BM25 score (0-1)
    normalize_scores: bool = True  # Whether to normalize scores before combining
    min_score: float = 0.0  # Minimum combined score to include in results


class HybridSearch:
    """
    Hybrid search combining vector similarity with BM25 keyword search.

    Provides more robust retrieval by combining:
    - Semantic similarity (via embeddings)
    - Keyword matching (via BM25)

    This helps when:
    - Query uses different words than the document (semantic helps)
    - Query uses exact terms that should match (BM25 helps)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[HybridSearchConfig] = None,
        bm25_config: Optional[BM25Config] = None,
    ):
        """Initialize hybrid search.

        Args:
            vector_store: Vector store for semantic search.
            config: Hybrid search configuration.
            bm25_config: BM25 configuration.
        """
        self.vector_store = vector_store
        self.config = config or HybridSearchConfig()
        self.bm25_index = BM25Index(bm25_config)

    async def add(self, documents: list[Document]) -> None:
        """Add documents to both indices.

        Args:
            documents: Documents to index (must have embeddings).
        """
        # Add to vector store
        await self.vector_store.add(documents)

        # Add to BM25 index
        self.bm25_index.add(documents)

    async def delete(self, ids: list[str]) -> None:
        """Delete documents from both indices.

        Args:
            ids: Document IDs to remove.
        """
        await self.vector_store.delete(ids)
        self.bm25_index.delete(ids)

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results

        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)

        if max_score == min_score:
            # All same score, set to 1.0
            for r in results:
                r.score = 1.0
            return results

        for r in results:
            r.score = (r.score - min_score) / (max_score - min_score)

        return results

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query text (for BM25).
            query_embedding: Query embedding vector (for semantic search).
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of SearchResult objects sorted by combined score.
        """
        # Get more results from each source to ensure good coverage
        fetch_k = min(k * 3, 100)

        # Vector search
        vector_results = await self.vector_store.search(
            query_embedding, k=fetch_k, filter=filter
        )

        # BM25 search
        bm25_results = self.bm25_index.search(query, k=fetch_k)

        # Normalize scores if configured
        if self.config.normalize_scores:
            vector_results = self._normalize_scores(vector_results)
            bm25_results = self._normalize_scores(bm25_results)

        # Combine results
        combined: dict[str, tuple[float, float, Document]] = {}

        for r in vector_results:
            combined[r.document.id] = (r.score, 0.0, r.document)

        for r in bm25_results:
            if r.document.id in combined:
                vec_score, _, doc = combined[r.document.id]
                combined[r.document.id] = (vec_score, r.score, doc)
            else:
                combined[r.document.id] = (0.0, r.score, r.document)

        # Calculate weighted combined scores
        results = []
        vw = self.config.vector_weight
        kw = self.config.keyword_weight

        for doc_id, (vec_score, bm25_score, doc) in combined.items():
            combined_score = (vw * vec_score) + (kw * bm25_score)

            if combined_score >= self.config.min_score:
                results.append(SearchResult(
                    document=doc,
                    score=combined_score,
                ))

        # Sort by combined score
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:k]

    async def count(self) -> int:
        """Return number of indexed documents."""
        return await self.vector_store.count()
