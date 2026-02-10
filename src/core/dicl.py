"""Dynamic In-Context Learning (DICL) — few-shot examples from past interactions.

Retrieves successful past tool call sequences and injects them as few-shot
examples at inference time. Based on TensorZero's DICL pattern.

Unlike KnowledgeStore which stores high-level task/approach summaries,
DICL stores complete interaction sequences:
- User request
- Tool calls made
- Tool results received
- Final response

These are injected into the conversation as concrete examples before
inference, helping the model understand expected behavior for similar tasks.

Usage:
    store = DICLStore()

    # Record a successful interaction
    store.record(ToolCallExample(
        task="List files in the src directory",
        tool_calls=[{"name": "list_dir", "arguments": {"path": "src"}}],
        tool_results=[{"files": ["main.py", "utils.py"]}],
        response="The src directory contains main.py and utils.py.",
        tags=["filesystem", "directory-listing"],
    ))

    # Retrieve relevant examples
    examples = store.search("what files are in the project folder")
    formatted = store.format_few_shot(examples)
    # Inject formatted examples into system prompt
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Default location for DICL examples
DEFAULT_DICL_PATH = Path.home() / ".animus" / "data" / "dicl_examples.jsonl"

# Maximum examples to inject at inference time (to avoid context overflow)
DEFAULT_MAX_EXAMPLES = 3


@dataclass
class ToolCallExample:
    """A recorded tool call interaction for few-shot learning.

    Captures the complete cycle: user request -> tool calls -> response.
    """

    task: str  # User's original request
    tool_calls: list[dict[str, Any]]  # Tool calls made: [{name, arguments}, ...]
    tool_results: list[Any]  # Results from each tool call
    response: str  # Agent's final response
    tags: list[str] = field(default_factory=list)  # For search/filtering
    model_used: str = ""  # Model that produced the interaction
    timestamp: float = field(default_factory=time.time)
    success: bool = True  # Whether the interaction was successful
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ToolCallExample:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def matches(self, query: str) -> float:
        """Score how well this example matches a search query.

        Returns a relevance score (0.0 = no match, higher = better).
        """
        query_lower = query.lower()
        terms = query_lower.split()
        if not terms:
            return 0.0

        # Build searchable text
        tool_names = " ".join(tc.get("name", "") for tc in self.tool_calls)
        searchable = " ".join([
            self.task.lower(),
            tool_names.lower(),
            " ".join(self.tags),
            self.response.lower()[:200],  # First 200 chars of response
        ])

        # Count matching terms
        hits = sum(1 for term in terms if term in searchable)
        if hits == 0:
            return 0.0

        # Score: fraction of terms matched
        base_score = hits / len(terms)

        # Bonus for matches in task (user intent)
        task_hits = sum(1 for term in terms if term in self.task.lower())
        task_bonus = 0.4 * (task_hits / len(terms)) if task_hits else 0.0

        # Bonus for tool name matches (action similarity)
        tool_hits = sum(1 for term in terms if term in tool_names.lower())
        tool_bonus = 0.3 * (tool_hits / len(terms)) if tool_hits else 0.0

        # Bonus for tag matches (precise)
        tag_text = " ".join(self.tags).lower()
        tag_hits = sum(1 for term in terms if term in tag_text)
        tag_bonus = 0.2 * (tag_hits / len(terms)) if tag_hits else 0.0

        return min(base_score + task_bonus + tool_bonus + tag_bonus, 1.0)


@dataclass
class DICLHit:
    """A search result from the DICL store."""

    example: ToolCallExample
    score: float


class DICLStore:
    """JSONL-based store for tool call examples (few-shot learning).

    Stores complete interaction sequences for retrieval and injection
    as few-shot examples at inference time.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        max_results: int = DEFAULT_MAX_EXAMPLES,
    ):
        """Initialize the DICL store.

        Args:
            path: Path to the JSONL examples file.
            max_results: Default max examples to return from search.
        """
        self._path = path or DEFAULT_DICL_PATH
        self._max_results = max_results
        self._cache: Optional[list[ToolCallExample]] = None

    @property
    def path(self) -> Path:
        """Get the examples file path."""
        return self._path

    def record(self, example: ToolCallExample) -> None:
        """Append an example to the store.

        Args:
            example: The tool call example to record.
        """
        if not example.success:
            logger.debug("Skipping failed example (success=False)")
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example.to_dict()) + "\n")

        # Invalidate cache
        self._cache = None

        logger.info(
            "DICL example recorded: %s (%d tool calls)",
            example.task[:50],
            len(example.tool_calls),
        )

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: float = 0.15,
        success_only: bool = True,
    ) -> list[DICLHit]:
        """Search for relevant examples.

        Args:
            query: Search query (keywords from user request).
            k: Max results to return.
            min_score: Minimum relevance score threshold.
            success_only: Only return successful examples.

        Returns:
            List of DICLHit sorted by relevance score (descending).
        """
        k = k or self._max_results
        examples = self._load_all()

        hits = []
        for example in examples:
            if success_only and not example.success:
                continue

            score = example.matches(query)
            if score >= min_score:
                hits.append(DICLHit(example=example, score=score))

        # Sort by score descending
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def get_all(self) -> list[ToolCallExample]:
        """Get all stored examples."""
        return list(self._load_all())

    def count(self) -> int:
        """Count total stored examples."""
        return len(self._load_all())

    def format_few_shot(
        self,
        hits: list[DICLHit],
        max_chars: int = 2000,
        include_results: bool = False,
    ) -> str:
        """Format examples as few-shot prompt content.

        Creates a structured format showing:
        - User request
        - Tool calls made (as JSON)
        - Optionally: tool results
        - Agent's response

        Args:
            hits: Search results to format.
            max_chars: Maximum characters in output.
            include_results: Include tool results (can be verbose).

        Returns:
            Formatted string for injection into system prompt.
        """
        if not hits:
            return ""

        parts = ["[Examples of successful tool usage:]"]
        chars_used = len(parts[0])

        for i, hit in enumerate(hits, 1):
            ex = hit.example

            # Format tool calls as compact JSON
            tools_str = json.dumps(ex.tool_calls, separators=(",", ":"))
            if len(tools_str) > 300:
                tools_str = tools_str[:297] + "..."

            entry = (
                f"\n\nExample {i}:\n"
                f"User: {ex.task}\n"
                f"Tool calls: {tools_str}\n"
            )

            if include_results and ex.tool_results:
                results_str = json.dumps(ex.tool_results, separators=(",", ":"))
                if len(results_str) > 200:
                    results_str = results_str[:197] + "..."
                entry += f"Results: {results_str}\n"

            entry += f"Response: {ex.response[:200]}"
            if len(ex.response) > 200:
                entry += "..."

            if chars_used + len(entry) > max_chars:
                break

            parts.append(entry)
            chars_used += len(entry)

        if len(parts) == 1:
            return ""  # Only header, no examples fit

        return "".join(parts)

    def format_messages(
        self,
        hits: list[DICLHit],
        max_examples: int = 2,
    ) -> list[dict[str, str]]:
        """Format examples as message history for chat context.

        Creates alternating user/assistant messages that can be prepended
        to the conversation history, providing concrete examples before
        the actual user request.

        Args:
            hits: Search results to format.
            max_examples: Maximum number of examples to include.

        Returns:
            List of message dicts: [{"role": "user", "content": ...}, ...]
        """
        messages = []

        for hit in hits[:max_examples]:
            ex = hit.example

            # User message (the original request)
            messages.append({
                "role": "user",
                "content": ex.task,
            })

            # Assistant message (tool calls + response)
            # Format as the agent would have responded
            tools_json = json.dumps(ex.tool_calls, indent=2)
            assistant_content = (
                f"I'll help you with that.\n\n"
                f"```json\n{tools_json}\n```\n\n"
                f"{ex.response}"
            )
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

        return messages

    def clear(self) -> None:
        """Clear all stored examples."""
        if self._path.exists():
            self._path.unlink()
        self._cache = None

    def _load_all(self) -> list[ToolCallExample]:
        """Load all examples from disk (cached)."""
        if self._cache is not None:
            return self._cache

        examples = []
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        examples.append(ToolCallExample.from_dict(data))
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Skipping malformed DICL example")

        self._cache = examples
        return examples

    def stats(self) -> dict:
        """Get DICL store statistics."""
        examples = self._load_all()

        # Collect tool usage stats
        tool_counts: dict[str, int] = {}
        for ex in examples:
            for tc in ex.tool_calls:
                name = tc.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1

        # Collect tag stats
        tags = set()
        for ex in examples:
            tags.update(ex.tags)

        return {
            "total_examples": len(examples),
            "successful_examples": sum(1 for ex in examples if ex.success),
            "tool_usage": tool_counts,
            "unique_tags": len(tags),
            "path": str(self._path),
        }


def create_example_from_turn(
    user_message: str,
    tool_calls: list[dict[str, Any]],
    tool_results: list[Any],
    response: str,
    tags: Optional[list[str]] = None,
    model: str = "",
) -> ToolCallExample:
    """Helper to create a ToolCallExample from agent turn data.

    Args:
        user_message: The user's original request.
        tool_calls: List of tool calls made.
        tool_results: Results from tool executions.
        response: Agent's final response.
        tags: Optional tags for categorization.
        model: Model used for generation.

    Returns:
        A ToolCallExample ready for storage.
    """
    # Auto-generate tags from tool names if not provided
    auto_tags = tags or []
    if not auto_tags:
        tool_names = set(tc.get("name", "") for tc in tool_calls)
        auto_tags = list(tool_names)

    return ToolCallExample(
        task=user_message,
        tool_calls=tool_calls,
        tool_results=tool_results,
        response=response,
        tags=auto_tags,
        model_used=model,
        success=True,
    )
