"""Run persistence for agent sessions.

This module provides data structures and storage for persisting
agent runs, enabling analysis and improvement over time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Iterator
from uuid import uuid4

from src.core.decision import Decision, DecisionRecord, DecisionRecorder


class RunStatus(Enum):
    """Status of an agent run."""
    PENDING = "pending"          # Not yet started
    RUNNING = "running"          # Currently executing
    COMPLETED = "completed"      # Finished successfully
    FAILED = "failed"            # Finished with error
    CANCELLED = "cancelled"      # User cancelled


@dataclass
class RunMetrics:
    """Metrics collected during an agent run."""
    tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0          # Total latency in milliseconds
    turn_count: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    decisions_made: int = 0
    success_rate: float = 0.0        # Tool success rate
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_used": self.tokens_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "latency_ms": self.latency_ms,
            "turn_count": self.turn_count,
            "tool_calls": self.tool_calls,
            "tool_successes": self.tool_successes,
            "tool_failures": self.tool_failures,
            "decisions_made": self.decisions_made,
            "success_rate": self.success_rate,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetrics:
        """Create from dictionary."""
        return cls(
            tokens_used=data.get("tokens_used", 0),
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            latency_ms=data.get("latency_ms", 0.0),
            turn_count=data.get("turn_count", 0),
            tool_calls=data.get("tool_calls", 0),
            tool_successes=data.get("tool_successes", 0),
            tool_failures=data.get("tool_failures", 0),
            decisions_made=data.get("decisions_made", 0),
            success_rate=data.get("success_rate", 0.0),
            errors=data.get("errors", []),
        )

    def update_success_rate(self) -> None:
        """Calculate and update success rate from tool stats."""
        if self.tool_calls > 0:
            self.success_rate = self.tool_successes / self.tool_calls
        else:
            self.success_rate = 0.0


@dataclass
class Run:
    """A complete agent run/session.

    Captures everything about a single agent execution,
    from goal to final outcome.
    """
    id: str
    goal: str                                # What the user asked for
    status: RunStatus = RunStatus.PENDING
    model: str = ""                          # Model used
    provider: str = ""                       # Provider used (native, ollama, api)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    metrics: RunMetrics = field(default_factory=RunMetrics)
    decisions: list[Decision] = field(default_factory=list)
    final_output: Optional[str] = None       # Final response to user
    error: Optional[str] = None              # Error message if failed
    tags: list[str] = field(default_factory=list)  # User-defined tags
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        goal: str,
        model: str = "",
        provider: str = "",
        tags: Optional[list[str]] = None,
        **metadata: Any,
    ) -> Run:
        """Create a new run with auto-generated ID."""
        return cls(
            id=str(uuid4()),
            goal=goal,
            status=RunStatus.PENDING,
            model=model,
            provider=provider,
            tags=tags or [],
            metadata=metadata,
        )

    def start(self) -> None:
        """Mark the run as started."""
        self.status = RunStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, final_output: str) -> None:
        """Mark the run as completed successfully."""
        self.status = RunStatus.COMPLETED
        self.ended_at = datetime.now()
        self.final_output = final_output
        self.metrics.update_success_rate()

    def fail(self, error: str) -> None:
        """Mark the run as failed."""
        self.status = RunStatus.FAILED
        self.ended_at = datetime.now()
        self.error = error
        self.metrics.errors.append(error)
        self.metrics.update_success_rate()

    def cancel(self) -> None:
        """Mark the run as cancelled."""
        self.status = RunStatus.CANCELLED
        self.ended_at = datetime.now()
        self.metrics.update_success_rate()

    def add_decision(self, decision: Decision) -> None:
        """Add a decision to this run."""
        self.decisions.append(decision)
        self.metrics.decisions_made += 1

    def record_tool_call(self, success: bool, error: Optional[str] = None) -> None:
        """Record a tool call result."""
        self.metrics.tool_calls += 1
        if success:
            self.metrics.tool_successes += 1
        else:
            self.metrics.tool_failures += 1
            if error:
                self.metrics.errors.append(error)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage."""
        self.metrics.tokens_input += input_tokens
        self.metrics.tokens_output += output_tokens
        self.metrics.tokens_used = self.metrics.tokens_input + self.metrics.tokens_output

    def increment_turn(self) -> None:
        """Increment the turn counter."""
        self.metrics.turn_count += 1

    @property
    def duration_ms(self) -> float:
        """Get run duration in milliseconds."""
        if self.started_at is None:
            return 0.0
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "model": self.model,
            "provider": self.provider,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metrics": self.metrics.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "final_output": self.final_output,
            "error": self.error,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            goal=data["goal"],
            status=RunStatus(data["status"]),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            metrics=RunMetrics.from_dict(data.get("metrics", {})),
            decisions=[Decision.from_dict(d) for d in data.get("decisions", [])],
            final_output=data.get("final_output"),
            error=data.get("error"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class RunStore:
    """Persistent storage for runs.

    Stores runs as JSON files in ~/.animus/runs/
    with indexing for fast lookup.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the run store.

        Args:
            base_path: Base directory for storage. Defaults to ~/.animus/runs/
        """
        if base_path is None:
            base_path = Path.home() / ".animus" / "runs"
        self.base_path = Path(base_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the storage directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        """Get the file path for a run."""
        return self.base_path / f"{run_id}.json"

    def save(self, run: Run) -> None:
        """Save a run to storage."""
        path = self._run_path(run.id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2)

    def load(self, run_id: str) -> Optional[Run]:
        """Load a run by ID."""
        path = self._run_path(run_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Run.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def delete(self, run_id: str) -> bool:
        """Delete a run."""
        path = self._run_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> list[str]:
        """List all run IDs."""
        return [
            p.stem for p in self.base_path.glob("*.json")
        ]

    def iter_runs(self) -> Iterator[Run]:
        """Iterate over all runs."""
        for run_id in self.list_all():
            run = self.load(run_id)
            if run:
                yield run

    def find_by_status(self, status: RunStatus) -> list[Run]:
        """Find runs by status."""
        return [run for run in self.iter_runs() if run.status == status]

    def find_by_goal(self, query: str, case_sensitive: bool = False) -> list[Run]:
        """Find runs where goal contains query string."""
        if not case_sensitive:
            query = query.lower()
        results = []
        for run in self.iter_runs():
            goal = run.goal if case_sensitive else run.goal.lower()
            if query in goal:
                results.append(run)
        return results

    def find_by_date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[Run]:
        """Find runs within a date range."""
        results = []
        for run in self.iter_runs():
            if run.started_at is None:
                continue
            if start and run.started_at < start:
                continue
            if end and run.started_at > end:
                continue
            results.append(run)
        return sorted(results, key=lambda r: r.started_at or datetime.min)

    def find_by_tag(self, tag: str) -> list[Run]:
        """Find runs with a specific tag."""
        return [run for run in self.iter_runs() if tag in run.tags]

    def get_recent(self, limit: int = 10) -> list[Run]:
        """Get most recent runs."""
        runs = list(self.iter_runs())
        runs.sort(key=lambda r: r.started_at or datetime.min, reverse=True)
        return runs[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all runs."""
        runs = list(self.iter_runs())
        if not runs:
            return {
                "total_runs": 0,
                "by_status": {},
                "avg_duration_ms": 0,
                "avg_tokens": 0,
                "avg_turns": 0,
                "overall_success_rate": 0,
            }

        by_status = {}
        total_duration = 0
        total_tokens = 0
        total_turns = 0
        total_tool_calls = 0
        total_tool_successes = 0

        for run in runs:
            status = run.status.value
            by_status[status] = by_status.get(status, 0) + 1
            total_duration += run.duration_ms
            total_tokens += run.metrics.tokens_used
            total_turns += run.metrics.turn_count
            total_tool_calls += run.metrics.tool_calls
            total_tool_successes += run.metrics.tool_successes

        return {
            "total_runs": len(runs),
            "by_status": by_status,
            "avg_duration_ms": total_duration / len(runs),
            "avg_tokens": total_tokens / len(runs),
            "avg_turns": total_turns / len(runs),
            "overall_success_rate": total_tool_successes / total_tool_calls if total_tool_calls > 0 else 0,
        }

    def cleanup_old(self, days: int = 30) -> int:
        """Remove runs older than specified days.

        Args:
            days: Remove runs older than this many days.

        Returns:
            Number of runs deleted.
        """
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)

        deleted = 0
        for run in list(self.iter_runs()):
            if run.started_at and run.started_at < cutoff:
                if self.delete(run.id):
                    deleted += 1
        return deleted
