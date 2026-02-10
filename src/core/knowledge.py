"""Knowledge compounding — index successful solutions for future reference.

Implements the Build → Solve → Document → Search → Accelerate cycle.
After each successful task, store the solution with structured metadata.
Future tasks can search past solutions for relevant approaches.

Storage: JSONL file at ~/.animus/data/solutions.jsonl
Search: keyword matching on task, approach, tags, and files.

Usage:
    store = KnowledgeStore()
    store.record(SolutionRecord(
        task="Fix NaN in evolve()",
        approach="Replace Euler with analytical phase rotation",
        files_changed=["processor.py"],
        tags=["numerical-stability", "cuda"],
    ))

    results = store.search("NaN numerical stability")
    # Returns matching SolutionRecords sorted by relevance
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default location for solution store
DEFAULT_SOLUTIONS_PATH = Path.home() / ".animus" / "data" / "solutions.jsonl"


@dataclass
class SolutionRecord:
    """A recorded solution to a task."""

    task: str  # What was the task/problem
    approach: str  # How it was solved
    outcome: str = "success"  # success, partial, failed
    files_changed: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    model_used: str = ""  # Model that produced the solution
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SolutionRecord:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def matches(self, query: str) -> float:
        """Score how well this record matches a search query.

        Returns a relevance score (0.0 = no match, higher = better).
        """
        query_lower = query.lower()
        terms = query_lower.split()
        if not terms:
            return 0.0

        # Build searchable text from all fields
        searchable = " ".join([
            self.task.lower(),
            self.approach.lower(),
            " ".join(self.tags),
            " ".join(self.files_changed),
            self.outcome.lower(),
        ])

        # Count matching terms
        hits = sum(1 for term in terms if term in searchable)
        if hits == 0:
            return 0.0

        # Score: fraction of terms matched, with bonus for task/approach matches
        base_score = hits / len(terms)

        # Bonus for matches in task description (most relevant)
        task_hits = sum(1 for term in terms if term in self.task.lower())
        task_bonus = 0.3 * (task_hits / len(terms)) if task_hits else 0.0

        # Bonus for tag matches (precise)
        tag_text = " ".join(self.tags).lower()
        tag_hits = sum(1 for term in terms if term in tag_text)
        tag_bonus = 0.2 * (tag_hits / len(terms)) if tag_hits else 0.0

        return min(base_score + task_bonus + tag_bonus, 1.0)


@dataclass
class SearchHit:
    """A search result from the knowledge store."""

    record: SolutionRecord
    score: float


class KnowledgeStore:
    """JSONL-based store for solution records.

    Stores solutions as append-only JSONL. Provides keyword-based search
    for finding relevant past solutions. Can optionally connect to a
    vector store for semantic search.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        max_results: int = 5,
    ):
        """Initialize the knowledge store.

        Args:
            path: Path to the JSONL solutions file.
            max_results: Default max results for search.
        """
        self._path = path or DEFAULT_SOLUTIONS_PATH
        self._max_results = max_results
        self._cache: Optional[list[SolutionRecord]] = None

    @property
    def path(self) -> Path:
        """Get the solutions file path."""
        return self._path

    def record(self, solution: SolutionRecord) -> None:
        """Append a solution record to the store.

        Args:
            solution: The solution to record.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(solution.to_dict()) + "\n")

        # Invalidate cache
        self._cache = None

        logger.info(
            "Knowledge recorded: %s (%d tags)",
            solution.task[:60],
            len(solution.tags),
        )

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: float = 0.1,
        outcome_filter: Optional[str] = None,
    ) -> list[SearchHit]:
        """Search for relevant solutions.

        Args:
            query: Search query (keywords).
            k: Max results to return.
            min_score: Minimum relevance score threshold.
            outcome_filter: Filter by outcome (e.g. "success").

        Returns:
            List of SearchHit sorted by relevance score (descending).
        """
        k = k or self._max_results
        records = self._load_all()

        hits = []
        for record in records:
            if outcome_filter and record.outcome != outcome_filter:
                continue

            score = record.matches(query)
            if score >= min_score:
                hits.append(SearchHit(record=record, score=score))

        # Sort by score descending
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def get_all(self) -> list[SolutionRecord]:
        """Get all solution records."""
        return list(self._load_all())

    def count(self) -> int:
        """Count total solution records."""
        return len(self._load_all())

    def format_context(
        self,
        hits: list[SearchHit],
        max_chars: int = 1000,
    ) -> str:
        """Format search hits as context for LLM injection.

        Args:
            hits: Search results to format.
            max_chars: Maximum characters in output.

        Returns:
            Formatted string for context injection.
        """
        if not hits:
            return ""

        parts = ["[Past solutions that may be relevant:]"]
        chars_used = len(parts[0])

        for i, hit in enumerate(hits, 1):
            entry = (
                f"\n{i}. Task: {hit.record.task}\n"
                f"   Approach: {hit.record.approach}\n"
                f"   Outcome: {hit.record.outcome}"
            )
            if hit.record.tags:
                entry += f"\n   Tags: {', '.join(hit.record.tags)}"

            if chars_used + len(entry) > max_chars:
                break

            parts.append(entry)
            chars_used += len(entry)

        return "".join(parts)

    def clear(self) -> None:
        """Clear all stored solutions."""
        if self._path.exists():
            self._path.unlink()
        self._cache = None

    def _load_all(self) -> list[SolutionRecord]:
        """Load all records from disk (cached)."""
        if self._cache is not None:
            return self._cache

        records = []
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        records.append(SolutionRecord.from_dict(data))
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Skipping malformed solution record")

        self._cache = records
        return records

    def stats(self) -> dict:
        """Get knowledge store statistics."""
        records = self._load_all()
        outcomes = {}
        tags = set()
        for r in records:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
            tags.update(r.tags)

        return {
            "total_records": len(records),
            "outcomes": outcomes,
            "unique_tags": len(tags),
            "path": str(self._path),
        }
