"""Tests for Manifold query router classification.

Validates that the hardcoded router correctly classifies queries
into SEMANTIC, STRUCTURAL, HYBRID, and KEYWORD strategies.
"""

import pytest

from src.retrieval.router import (
    RetrievalStrategy,
    classify_query,
    _extract_symbol,
    _extract_keyword_term,
)


class TestSemanticClassification:
    """Test that semantic/conceptual queries are classified correctly."""

    @pytest.mark.parametrize("query", [
        "how does authentication work",
        "find error handling patterns",
        "code that processes user input",
        "explain the configuration system",
        "show me the logging approach",
        "what is the chunking strategy",
        "describe how the agent works",
        "find examples of tool usage",
        "similar code to vector search",
        "why does the planner decompose tasks",
    ])
    def test_semantic_queries(self, query):
        """Semantic queries should route to vector search."""
        decision = classify_query(query)
        assert decision.strategy == RetrievalStrategy.SEMANTIC
        assert decision.confidence >= 0.6
        assert decision.semantic_query == query


class TestStructuralClassification:
    """Test that structural/relationship queries are classified correctly."""

    @pytest.mark.parametrize("query,expected_operation", [
        ("what calls authenticate()", "callers"),
        ("callers of Agent.run", "callers"),
        ("who calls estimate_tokens", "callers"),
        ("what does ToolRegistry.execute call", "callees"),
        ("callees of parse_tool_calls", "callees"),
        ("subclasses of ModelProvider", "inheritance"),
        ("what inherits from Tool", "inheritance"),
        ("base class of PythonParser", "inheritance"),
        ("imports in agent.py", "imports"),
        ("what does main.py import", "imports"),
        ("blast radius of estimate_tokens", "blast_radius"),
        ("impact of changing Agent.run", "blast_radius"),
        ("downstream effects of modifying chunk()", "blast_radius"),
    ])
    def test_structural_queries_with_operations(self, query, expected_operation):
        """Structural queries should route to graph with correct operation."""
        decision = classify_query(query)
        assert decision.strategy == RetrievalStrategy.STRUCTURAL
        assert decision.confidence >= 0.75
        assert decision.structural_operation == expected_operation

    @pytest.mark.parametrize("query", [
        "Agent.run",
        "find ToolRegistry",
        "ChunkContextualizer",
        "ModelProvider class",
        "`estimate_tokens`",
        "search_code_graph tool",
    ])
    def test_structural_queries_search_mode(self, query):
        """Symbol references without relationships should route to structural search."""
        decision = classify_query(query)
        assert decision.strategy == RetrievalStrategy.STRUCTURAL
        assert decision.confidence >= 0.7
        assert decision.structural_operation == "search"


class TestHybridClassification:
    """Test that compound queries requiring both strategies are detected."""

    @pytest.mark.parametrize("query", [
        "find the auth code and what depends on it",
        "show me the config system and its callers",
        "everything about the permission checker",
        "how is chunking implemented and what uses it",
        "find the parser and what calls it",
        "get the vector store code and its dependencies",
        "authentication system and everything related",
        "complete view of the agent loop",
        "all about error handling",
    ])
    def test_hybrid_queries(self, query):
        """Hybrid queries should combine semantic and structural retrieval."""
        decision = classify_query(query)
        assert decision.strategy == RetrievalStrategy.HYBRID
        assert decision.confidence >= 0.5
        # Should have both query types populated
        assert decision.semantic_query != ""


class TestKeywordClassification:
    """Test that exact-match queries are classified correctly."""

    @pytest.mark.parametrize("query,expected_keyword", [
        ("find TODO comments", "TODO"),
        ("grep for FIXME", "FIXME"),
        ("lines containing 'API_KEY'", "API_KEY"),
        ('search for "DEPRECATED"', "DEPRECATED"),
        ("find HACK markers", "HACK"),
        ("TODO in the codebase", "TODO"),
        ("lines with XXX", "XXX"),
    ])
    def test_keyword_queries(self, query, expected_keyword):
        """Keyword queries should route to grep with extracted term."""
        decision = classify_query(query)
        assert decision.strategy == RetrievalStrategy.KEYWORD
        assert decision.confidence >= 0.85
        assert decision.keyword_query == expected_keyword


class TestSymbolExtraction:
    """Test that symbol extraction correctly identifies code identifiers."""

    @pytest.mark.parametrize("query,expected_symbol", [
        ("`authenticate`", "authenticate"),  # Backtick-quoted (highest priority)
        ("authenticate()", "authenticate"),  # Function call
        ("Agent.run", "Agent.run"),  # Dotted name
        ("ModelProvider.generate", "ModelProvider.generate"),  # Dotted name
        ("ToolRegistry", "ToolRegistry"),  # CamelCase
        ("ChunkContextualizer", "ChunkContextualizer"),  # Component name
        ("what calls parse_tool_calls", "parse_tool_calls"),  # Identifier in query
    ])
    def test_symbol_extraction(self, query, expected_symbol):
        """Symbol extraction should prioritize backticks, then calls, then dotted names."""
        symbol = _extract_symbol(query)
        assert symbol == expected_symbol

    def test_symbol_extraction_filters_stopwords(self):
        """Symbol extraction should filter common English words."""
        query = "what does the function do"
        symbol = _extract_symbol(query)
        # Should extract "function" (longest code-like word after filtering stopwords)
        assert symbol == "function"

    def test_symbol_extraction_prefers_longer_words(self):
        """When multiple candidates, prefer longer identifiers."""
        query = "what does x call in the parser"
        symbol = _extract_symbol(query)
        # "parser" is longer than "x" and "call"
        assert symbol == "parser"


class TestKeywordExtraction:
    """Test that keyword term extraction works correctly."""

    @pytest.mark.parametrize("query,expected_keyword", [
        ('"exact text"', "exact text"),  # Double quotes
        ("'literal string'", "literal string"),  # Single quotes
        ("find TODO", "TODO"),  # Marker keyword
        ("containing FIXME", "FIXME"),  # Marker in context
        ('lines with "API_KEY"', "API_KEY"),  # Quoted in context
    ])
    def test_keyword_extraction(self, query, expected_keyword):
        """Keyword extraction should handle quotes and markers."""
        keyword = _extract_keyword_term(query)
        assert keyword == expected_keyword

    def test_keyword_extraction_returns_empty_when_unclear(self):
        """When no clear keyword, return empty string."""
        keyword = _extract_keyword_term("just some regular text")
        assert keyword == ""


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_defaults_to_hybrid(self):
        """Empty queries should safely default to hybrid."""
        decision = classify_query("")
        assert decision.strategy == RetrievalStrategy.HYBRID
        assert decision.confidence == 0.5

    def test_very_short_query(self):
        """Short queries should still classify reasonably."""
        decision = classify_query("auth")
        # Could be semantic or structural, should not crash
        assert decision.strategy in [
            RetrievalStrategy.SEMANTIC,
            RetrievalStrategy.STRUCTURAL,
            RetrievalStrategy.HYBRID,
        ]

    def test_query_with_multiple_symbols(self):
        """Queries mentioning multiple symbols should extract the most relevant."""
        query = "relationship between Agent and Planner"
        decision = classify_query(query)
        # Should classify based on "relationship" semantic language
        assert decision.strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID]

    def test_mixed_semantic_and_structural_indicators(self):
        """Queries with both semantic and structural hints should route appropriately."""
        query = "how does authenticate() work and what calls it"
        decision = classify_query(query)
        # "how does X work" (semantic) + "what calls it" (structural)
        assert decision.strategy == RetrievalStrategy.HYBRID

    def test_case_insensitivity(self):
        """Classification should be case-insensitive."""
        decision1 = classify_query("WHAT CALLS AUTHENTICATE()")
        decision2 = classify_query("what calls authenticate()")
        assert decision1.strategy == decision2.strategy

    def test_confidence_scores_in_valid_range(self):
        """All confidence scores should be between 0 and 1."""
        queries = [
            "how does auth work",
            "what calls foo()",
            "find auth and its callers",
            "find TODO",
        ]
        for query in queries:
            decision = classify_query(query)
            assert 0.0 <= decision.confidence <= 1.0


class TestRoutingDecisionFields:
    """Test that RoutingDecision is populated correctly for each strategy."""

    def test_semantic_decision_has_semantic_query(self):
        """SEMANTIC decisions should populate semantic_query."""
        decision = classify_query("how does authentication work")
        assert decision.strategy == RetrievalStrategy.SEMANTIC
        assert decision.semantic_query != ""
        assert decision.structural_query == ""
        assert decision.keyword_query == ""

    def test_structural_decision_has_structural_query(self):
        """STRUCTURAL decisions should populate structural_query and operation."""
        decision = classify_query("what calls authenticate()")
        assert decision.strategy == RetrievalStrategy.STRUCTURAL
        assert decision.structural_query != ""
        assert decision.structural_operation == "callers"
        assert decision.semantic_query == ""

    def test_hybrid_decision_has_both_queries(self):
        """HYBRID decisions should populate both semantic and structural."""
        decision = classify_query("find auth code and what calls it")
        assert decision.strategy == RetrievalStrategy.HYBRID
        assert decision.semantic_query != ""
        assert decision.structural_query != ""
        assert decision.structural_operation != ""

    def test_keyword_decision_has_keyword_query(self):
        """KEYWORD decisions should populate keyword_query."""
        decision = classify_query("find TODO comments")
        assert decision.strategy == RetrievalStrategy.KEYWORD
        assert decision.keyword_query == "TODO"


class TestReasoningExplanations:
    """Test that reasoning strings provide useful explanations."""

    def test_reasoning_is_not_empty(self):
        """All decisions should include reasoning."""
        queries = [
            "how does auth work",
            "what calls foo()",
            "find auth and callers",
            "find TODO",
        ]
        for query in queries:
            decision = classify_query(query)
            assert decision.reasoning != ""
            assert len(decision.reasoning) > 10  # Non-trivial explanation

    def test_reasoning_mentions_query_characteristics(self):
        """Reasoning should reference why the classification was made."""
        decision = classify_query("what calls authenticate()")
        assert "callers" in decision.reasoning.lower() or "symbol" in decision.reasoning.lower()

        decision = classify_query("how does auth work")
        assert "conceptual" in decision.reasoning.lower() or "descriptive" in decision.reasoning.lower()
