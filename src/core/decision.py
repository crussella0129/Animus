"""Decision recording for agent self-improvement.

This module provides data structures for recording agent decisions,
enabling analysis, learning, and improvement over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class DecisionType(Enum):
    """Types of decisions an agent can make."""
    TOOL_SELECTION = "tool_selection"      # Which tool to use
    TOOL_ARGUMENTS = "tool_arguments"      # How to call the tool
    STRATEGY = "strategy"                   # High-level approach
    DELEGATION = "delegation"               # Whether to spawn sub-agent
    CONFIRMATION = "confirmation"           # Whether to proceed with action
    RECOVERY = "recovery"                   # How to handle an error
    TERMINATION = "termination"             # Whether to stop or continue


@dataclass
class Option:
    """A single option considered during decision-making."""
    id: str
    description: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        description: str,
        pros: Optional[list[str]] = None,
        cons: Optional[list[str]] = None,
        confidence: float = 0.0,
        **metadata: Any,
    ) -> Option:
        """Create an option with auto-generated ID."""
        return cls(
            id=str(uuid4())[:8],
            description=description,
            pros=pros or [],
            cons=cons or [],
            confidence=confidence,
            metadata=metadata,
        )


@dataclass
class Decision:
    """A recorded decision made by the agent.

    Captures the full context of why a decision was made,
    not just what action was taken.
    """
    id: str
    decision_type: DecisionType
    intent: str                             # What the agent was trying to accomplish
    context: str                            # Relevant context at decision time
    options: list[Option]                   # Options considered
    chosen_option_id: Optional[str]         # ID of the option chosen (None if no choice made)
    reasoning: str                          # Why this option was chosen
    timestamp: datetime = field(default_factory=datetime.now)
    turn_number: int = 0                    # Which turn in the conversation
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        decision_type: DecisionType,
        intent: str,
        context: str,
        options: list[Option],
        chosen_option_id: Optional[str],
        reasoning: str,
        turn_number: int = 0,
        **metadata: Any,
    ) -> Decision:
        """Create a decision with auto-generated ID and timestamp."""
        return cls(
            id=str(uuid4()),
            decision_type=decision_type,
            intent=intent,
            context=context,
            options=options,
            chosen_option_id=chosen_option_id,
            reasoning=reasoning,
            turn_number=turn_number,
            metadata=metadata,
        )

    @property
    def chosen_option(self) -> Optional[Option]:
        """Get the chosen option object."""
        if self.chosen_option_id is None:
            return None
        for opt in self.options:
            if opt.id == self.chosen_option_id:
                return opt
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "decision_type": self.decision_type.value,
            "intent": self.intent,
            "context": self.context,
            "options": [
                {
                    "id": opt.id,
                    "description": opt.description,
                    "pros": opt.pros,
                    "cons": opt.cons,
                    "confidence": opt.confidence,
                    "metadata": opt.metadata,
                }
                for opt in self.options
            ],
            "chosen_option_id": self.chosen_option_id,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        """Create from dictionary."""
        options = [
            Option(
                id=opt["id"],
                description=opt["description"],
                pros=opt.get("pros", []),
                cons=opt.get("cons", []),
                confidence=opt.get("confidence", 0.0),
                metadata=opt.get("metadata", {}),
            )
            for opt in data.get("options", [])
        ]
        return cls(
            id=data["id"],
            decision_type=DecisionType(data["decision_type"]),
            intent=data["intent"],
            context=data["context"],
            options=options,
            chosen_option_id=data.get("chosen_option_id"),
            reasoning=data["reasoning"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            turn_number=data.get("turn_number", 0),
            metadata=data.get("metadata", {}),
        )


class OutcomeStatus(Enum):
    """Status of a decision outcome."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class Outcome:
    """The outcome/result of a decision.

    Links back to the original decision to enable learning.
    """
    id: str
    decision_id: str                        # ID of the decision this outcome is for
    status: OutcomeStatus
    result: str                             # What actually happened
    summary: str                            # Brief summary for display
    error: Optional[str] = None             # Error message if failed
    metrics: dict[str, float] = field(default_factory=dict)  # Measurable outcomes
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        decision_id: str,
        status: OutcomeStatus,
        result: str,
        summary: str,
        error: Optional[str] = None,
        metrics: Optional[dict[str, float]] = None,
        **metadata: Any,
    ) -> Outcome:
        """Create an outcome with auto-generated ID and timestamp."""
        return cls(
            id=str(uuid4()),
            decision_id=decision_id,
            status=status,
            result=result,
            summary=summary,
            error=error,
            metrics=metrics or {},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "decision_id": self.decision_id,
            "status": self.status.value,
            "result": self.result,
            "summary": self.summary,
            "error": self.error,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Outcome:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            decision_id=data["decision_id"],
            status=OutcomeStatus(data["status"]),
            result=data["result"],
            summary=data["summary"],
            error=data.get("error"),
            metrics=data.get("metrics", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DecisionRecord:
    """A complete record linking a decision to its outcome."""
    decision: Decision
    outcome: Optional[Outcome] = None

    @property
    def was_successful(self) -> bool:
        """Check if the decision led to a successful outcome."""
        if self.outcome is None:
            return False
        return self.outcome.status == OutcomeStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.to_dict(),
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionRecord:
        """Create from dictionary."""
        return cls(
            decision=Decision.from_dict(data["decision"]),
            outcome=Outcome.from_dict(data["outcome"]) if data.get("outcome") else None,
        )


class DecisionRecorder:
    """Records and manages decisions for a session."""

    def __init__(self) -> None:
        self.decisions: list[Decision] = []
        self.outcomes: dict[str, Outcome] = {}  # decision_id -> outcome

    def record_decision(self, decision: Decision) -> None:
        """Record a new decision."""
        self.decisions.append(decision)

    def record_outcome(self, outcome: Outcome) -> None:
        """Record the outcome of a decision."""
        self.outcomes[outcome.decision_id] = outcome

    def get_records(self) -> list[DecisionRecord]:
        """Get all decision records with their outcomes."""
        return [
            DecisionRecord(
                decision=decision,
                outcome=self.outcomes.get(decision.id),
            )
            for decision in self.decisions
        ]

    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get a specific decision by ID."""
        for d in self.decisions:
            if d.id == decision_id:
                return d
        return None

    def get_outcome(self, decision_id: str) -> Optional[Outcome]:
        """Get the outcome for a decision."""
        return self.outcomes.get(decision_id)

    def get_decisions_by_type(self, decision_type: DecisionType) -> list[Decision]:
        """Get all decisions of a specific type."""
        return [d for d in self.decisions if d.decision_type == decision_type]

    def get_success_rate(self, decision_type: Optional[DecisionType] = None) -> float:
        """Calculate success rate for decisions.

        Args:
            decision_type: Filter by type, or None for all decisions.

        Returns:
            Success rate as float (0.0 to 1.0), or 0.0 if no outcomes.
        """
        records = self.get_records()
        if decision_type:
            records = [r for r in records if r.decision.decision_type == decision_type]

        with_outcomes = [r for r in records if r.outcome is not None]
        if not with_outcomes:
            return 0.0

        successful = sum(1 for r in with_outcomes if r.was_successful)
        return successful / len(with_outcomes)

    def clear(self) -> None:
        """Clear all recorded decisions and outcomes."""
        self.decisions.clear()
        self.outcomes.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionRecorder:
        """Create from dictionary."""
        recorder = cls()
        recorder.decisions = [Decision.from_dict(d) for d in data.get("decisions", [])]
        recorder.outcomes = {
            k: Outcome.from_dict(v) for k, v in data.get("outcomes", {}).items()
        }
        return recorder
