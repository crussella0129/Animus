"""Animus: Manifold - Query Router with Hardcoded Classification.

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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalStrategy(Enum):
    """Retrieval strategy classification."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


@dataclass
class RoutingDecision:
    """Result of query classification.

    Contains the chosen strategy, confidence score, reasoning,
    and pre-processed query components for each retrieval backend.
    """
    strategy: RetrievalStrategy
    confidence: float  # 0.0-1.0
    reasoning: str  # Human-readable explanation
    semantic_query: str = ""  # Query to send to vector search
    structural_query: str = ""  # Pattern/symbol for graph search
    structural_operation: str = "search"  # search|callers|callees|blast_radius|inheritance
    keyword_query: str = ""  # Exact text to grep for
    filters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal patterns for classification
# ---------------------------------------------------------------------------

# Patterns that indicate a specific symbol is being referenced
_SYMBOL_PATTERNS = [
    r'\b\w+\(\)',                     # function_name()
    r'\b\w+\.\w+',                   # module.attribute
    r'`[^`]+`',                      # `backtick-quoted`
    r'\b(?:class|def|func)\s+\w+',   # "class Foo", "def bar"
    r'\b\w+(?:Tool|Provider|Store|Handler|Manager|Factory|Registry|Parser|Executor|Router)\b',  # CamelCase component names
]

# Patterns that indicate structural/relationship queries
_RELATIONSHIP_PATTERNS = [
    (r'\b(?:call[s|ed|ing]*|invoke[s|d]*|use[s|d]*)\s+(?:to|by)?\b', "callers"),
    (r'\b(?:caller[s]?|called\s+by|who\s+calls|what\s+calls)\b', "callers"),
    (r'\b(?:callee[s]?|calls\s+to|what\s+does\s+\w+\s+call)\b', "callees"),
    (r'\b(?:inherit[s]?|subclass|extends|derived|child\s+class)\b', "inheritance"),
    (r'\b(?:parent\s+class|base\s+class|superclass)\b', "inheritance"),
    (r'\b(?:import[s|ed]*|depend[s]?|dependenc)', "imports"),
    (r'\b(?:blast\s+radius|impact|affect[s|ed]*|ripple|downstream)\b', "blast_radius"),
    (r'\b(?:contain[s]?|inside|within|member[s]?\s+of)\b', "search"),
]

# Patterns that indicate semantic/conceptual queries
_SEMANTIC_PATTERNS = [
    r'\b(?:how|why|explain|describe|what\s+is|what\s+are)\b',
    r'\b(?:find|search|look\s+for|show\s+me)\b(?!.*(?:call|inherit|import))',  # Negative lookahead
    r'\b(?:similar|related|like|pattern|approach|technique|strategy)\b',
    r'\b(?:implement[s|ed|ation]*|handle[s|d|ing]*|process[es|ed|ing]*)\b',
    r'\b(?:manage[s|d|ment]*|validate[s|d]*|transform[s|ed]*)\b',
    r'\b(?:error|bug|issue|problem|fix|debug)\b',
    r'\b(?:example|usage|documentation|tutorial)\b',
    r'\b(?:best\s+practice|idiom|convention)\b',
]

# Patterns that indicate exact keyword search
_KEYWORD_PATTERNS = [
    r'\b(?:TODO|FIXME|HACK|XXX|DEPRECATED|NOTE|WARNING|BUG)\b',
    r'\b(?:grep|find\s+text|exact\s+match|literal|string)\b',
    r'["\'][^"\']+["\']',  # Quoted strings
    r'\b(?:contain(?:s|ing)?)\s+["\']',  # "containing 'text'"
    r'\b(?:lines?\s+with|lines?\s+containing)\b',
    r'\b(?:search\s+for)\s+["\']',
]

# Hybrid indicators: query wants both semantic understanding AND structural context
_HYBRID_INDICATORS = [
    r'\b(?:and\s+(?:what|how|where|its?|their))\b',
    r'\b(?:then\s+show|also\s+(?:show|find|get))\b',
    r'\b(?:along\s+with|together\s+with|as\s+well\s+as)\b',
    r'\b(?:everything\s+(?:about|related|connected|around))\b',
    r'\b(?:full\s+picture|complete\s+view|all\s+about)\b',
    r'\b(?:and\s+(?:depend|call|import|inherit|use))',
    r'\b(?:and\s+(?:its|their|the))\s+(?:caller|callee|depend)',
]


def classify_query(query: str) -> RoutingDecision:
    """Classify a query into a retrieval strategy using hardcoded patterns.

    Decision Procedure:
        1. Check for hybrid indicators (highest priority — catches compound queries)
        2. Check for keyword signals (exact match requests)
        3. Check for symbol references + relationship words → STRUCTURAL
        4. Check for symbol references without relationship → STRUCTURAL (search mode)
        5. Check for semantic/conceptual language → SEMANTIC
        6. Default → HYBRID (safest fallback for ambiguous queries)

    Args:
        query: Natural language or code query

    Returns:
        RoutingDecision with strategy, confidence, and pre-processed query components
    """
    query_lower = query.lower().strip()

    # --- Step 1: Check for hybrid indicators ---
    for pattern in _HYBRID_INDICATORS:
        if re.search(pattern, query_lower):
            return _build_hybrid_decision(
                query, query_lower,
                confidence=0.8,
                reasoning="Query combines conceptual and structural elements"
            )

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

    # --- Step 5: Default to HYBRID (safest for ambiguous queries) ---
    return _build_hybrid_decision(
        query, query_lower,
        confidence=0.5,
        reasoning="Ambiguous query — using hybrid retrieval for comprehensive coverage"
    )


def _extract_symbol(query: str) -> str:
    """Extract the most likely symbol name from a query.

    Priority:
        1. Backtick-quoted: `symbol_name`
        2. Function call: symbol_name()
        3. Dotted name: module.symbol
        4. CamelCase word: MyClassName
        5. First word that looks like an identifier

    Args:
        query: The query string

    Returns:
        Extracted symbol name or the full query if no symbol found
    """
    # Backtick-quoted (highest confidence)
    match = re.search(r'`([^`]+)`', query)
    if match:
        return match.group(1)

    # Function call pattern
    match = re.search(r'\b(\w+)\(\)', query)
    if match:
        return match.group(1)

    # Dotted name (module.symbol or class.method)
    match = re.search(r'\b(\w+\.\w+(?:\.\w+)*)\b', query)
    if match:
        return match.group(1)

    # CamelCase identifier
    match = re.search(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', query)
    if match:
        return match.group(1)

    # Component-style names (ClassTool, BaseProvider, etc.)
    match = re.search(r'\b(\w+(?:Tool|Provider|Store|Handler|Manager|Factory|Registry|Parser|Executor|Router))\b', query)
    if match:
        return match.group(1)

    # Fallback: longest word that looks like a code identifier (not common English)
    words = re.findall(r'\b[a-zA-Z_]\w{2,}\b', query)

    # Filter out common English stop words
    _STOP_WORDS = {
        "the", "and", "for", "that", "this", "with", "from", "into",
        "what", "how", "where", "when", "why", "which", "who",
        "find", "show", "get", "all", "any", "some",
        "are", "was", "were", "been", "have", "has", "had",
        "does", "did", "will", "would", "should", "could",
        "about", "after", "before", "between", "through",
        "called", "calls", "calling",
    }

    code_words = [w for w in words if w.lower() not in _STOP_WORDS]

    if code_words:
        # Prefer longer words (more likely to be identifiers than English)
        return max(code_words, key=len)

    # Last resort: return trimmed query
    return query.strip()


def _extract_keyword_term(query: str) -> str:
    """Extract the exact text to search for in keyword mode.

    Args:
        query: The query string

    Returns:
        Exact text to grep for, or empty string if can't be determined
    """
    # Quoted string (highest confidence)
    match = re.search(r'["\']([^"\']+)["\']', query)
    if match:
        return match.group(1)

    # TODO/FIXME/etc markers
    match = re.search(r'\b(TODO|FIXME|HACK|XXX|DEPRECATED|NOTE|WARNING|BUG)\b', query, re.IGNORECASE)
    if match:
        return match.group(1)

    # "containing X" pattern
    match = re.search(r'containing\s+["\']?([^"\']+)["\']?', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ""


def _build_hybrid_decision(
    query: str,
    query_lower: str,
    confidence: float,
    reasoning: str
) -> RoutingDecision:
    """Build a HYBRID routing decision with both semantic and structural components.

    Args:
        query: Original query text
        query_lower: Lowercased query for pattern matching
        confidence: Confidence score (0.0-1.0)
        reasoning: Human-readable explanation

    Returns:
        RoutingDecision configured for hybrid retrieval
    """
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
        semantic_query=query,  # Full query for vector search
        structural_query=symbol,  # Extracted symbol for graph query
        structural_operation=operation,
    )
