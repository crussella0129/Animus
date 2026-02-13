"""Lightweight feedback tracking for Manifold routing decisions.

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
    user_satisfaction TEXT DEFAULT 'unknown',
    session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_strategy ON routing_feedback(strategy);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON routing_feedback(timestamp);
"""


@dataclass
class RoutingFeedback:
    """Feedback record for a single routing decision.

    Tracks the query, chosen strategy, confidence, and outcome
    (how many results were actually used by the agent).
    """
    query: str
    strategy: str
    confidence: float
    result_count: int
    results_used: int = 0
    user_satisfaction: str = "unknown"  # "positive", "negative", "unknown"
    session_id: Optional[str] = None


class FeedbackStore:
    """Track routing decision outcomes for offline analysis.

    Records every Manifold query with its routing decision and outcome.
    Enables data-driven tuning of classification patterns.

    Usage:
        store = FeedbackStore(Path("~/.animus/feedback.db"))
        store.record(RoutingFeedback(
            query="how does auth work",
            strategy="semantic",
            confidence=0.8,
            result_count=5,
            results_used=2,
        ))
        stats = store.get_strategy_stats()
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize feedback store.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_FEEDBACK_SCHEMA)
        self._conn.commit()

    def record(self, feedback: RoutingFeedback) -> None:
        """Record a routing decision and its outcome.

        Args:
            feedback: RoutingFeedback object with query and outcome data
        """
        self._conn.execute(
            "INSERT INTO routing_feedback "
            "(timestamp, query, strategy, confidence, result_count, results_used, "
            "user_satisfaction, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                feedback.query,
                feedback.strategy,
                feedback.confidence,
                feedback.result_count,
                feedback.results_used,
                feedback.user_satisfaction,
                feedback.session_id,
            ),
        )
        self._conn.commit()

    def get_strategy_stats(self) -> dict[str, dict]:
        """Get aggregated statistics per strategy for analysis.

        Returns:
            Dict mapping strategy name to stats dict with:
            - total_queries: Number of times this strategy was used
            - avg_confidence: Average confidence score
            - utilization_rate: Proportion of results actually used
            - total_results: Total results returned
            - total_used: Total results used by agent

        Example:
            {
                "semantic": {
                    "total_queries": 42,
                    "avg_confidence": 0.75,
                    "utilization_rate": 0.48,  # 48% of results were used
                    "total_results": 210,
                    "total_used": 100,
                },
                "structural": { ... },
                ...
            }
        """
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
            utilization = used / results if results > 0 else 0.0
            stats[strategy] = {
                "total_queries": total,
                "avg_confidence": round(avg_conf, 3),
                "utilization_rate": round(utilization, 3),
                "total_results": results,
                "total_used": used,
            }

        return stats

    def get_recent_queries(self, limit: int = 20) -> list[dict]:
        """Get recent queries for debugging.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of query dicts with all feedback fields
        """
        rows = self._conn.execute(
            """
            SELECT timestamp, query, strategy, confidence,
                   result_count, results_used, user_satisfaction, session_id
            FROM routing_feedback
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        queries = []
        for row in rows:
            queries.append({
                "timestamp": row[0],
                "query": row[1],
                "strategy": row[2],
                "confidence": row[3],
                "result_count": row[4],
                "results_used": row[5],
                "user_satisfaction": row[6],
                "session_id": row[7],
            })

        return queries

    def get_misclassified_queries(self, threshold: float = 0.3) -> list[dict]:
        """Find queries where utilization was very low (possible misclassification).

        Args:
            threshold: Utilization rate below this is considered suspicious

        Returns:
            List of queries with low utilization (may indicate routing errors)
        """
        rows = self._conn.execute(
            """
            SELECT query, strategy, confidence, result_count, results_used,
                   CAST(results_used AS REAL) / result_count as utilization
            FROM routing_feedback
            WHERE result_count > 0
              AND CAST(results_used AS REAL) / result_count < ?
            ORDER BY timestamp DESC
            LIMIT 50
            """,
            (threshold,)
        ).fetchall()

        return [
            {
                "query": row[0],
                "strategy": row[1],
                "confidence": row[2],
                "result_count": row[3],
                "results_used": row[4],
                "utilization": round(row[5], 3),
            }
            for row in rows
        ]

    def clear(self) -> None:
        """Clear all feedback records."""
        self._conn.execute("DELETE FROM routing_feedback")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
