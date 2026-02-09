"""Goal definitions for graph-based sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConstraintType(str, Enum):
    """Types of constraints on sub-agent execution."""
    HARD = "hard"  # Must satisfy — failure aborts execution
    SOFT = "soft"  # Prefer to satisfy — violation logged but continues


@dataclass
class SuccessCriterion:
    """A measurable criterion for evaluating sub-agent success.

    Weights across all criteria in a goal must sum to 1.0.
    """
    id: str
    description: str
    metric: str  # e.g., "tests_passing", "files_created", "errors_fixed"
    target: float  # Target value for the metric
    weight: float  # 0.0–1.0, contribution to overall success score

    def evaluate(self, actual: float) -> float:
        """Return weighted score (0.0–weight) based on actual vs target."""
        if self.target == 0:
            return self.weight if actual == 0 else 0.0
        ratio = min(actual / self.target, 1.0)
        return ratio * self.weight


@dataclass
class Constraint:
    """A constraint on sub-agent behavior or execution."""
    id: str
    description: str
    constraint_type: ConstraintType = ConstraintType.HARD
    category: str = "general"  # e.g., "time", "scope", "resource", "security"

    def is_hard(self) -> bool:
        return self.constraint_type == ConstraintType.HARD


@dataclass
class SubAgentGoal:
    """Defines what a sub-agent graph should accomplish.

    A goal has a name, description, success criteria (weighted to sum to 1.0),
    and constraints that bound execution.
    """
    id: str
    name: str
    description: str
    criteria: list[SuccessCriterion] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate the goal definition. Returns list of error messages."""
        errors: list[str] = []

        if not self.id:
            errors.append("Goal must have an id")
        if not self.name:
            errors.append("Goal must have a name")

        if self.criteria:
            total_weight = sum(c.weight for c in self.criteria)
            if abs(total_weight - 1.0) > 1e-6:
                errors.append(
                    f"Criteria weights must sum to 1.0, got {total_weight:.4f}"
                )

            ids = [c.id for c in self.criteria]
            if len(ids) != len(set(ids)):
                errors.append("Criteria ids must be unique")

        if self.constraints:
            ids = [c.id for c in self.constraints]
            if len(ids) != len(set(ids)):
                errors.append("Constraint ids must be unique")

        return errors

    def evaluate(self, metrics: dict[str, float]) -> float:
        """Evaluate overall success score given metric values.

        Args:
            metrics: Dict mapping metric names to actual values.

        Returns:
            Score from 0.0 to 1.0.
        """
        if not self.criteria:
            return 1.0  # No criteria means always successful

        score = 0.0
        for criterion in self.criteria:
            actual = metrics.get(criterion.metric, 0.0)
            score += criterion.evaluate(actual)
        return score

    def hard_constraints(self) -> list[Constraint]:
        """Return only hard constraints."""
        return [c for c in self.constraints if c.is_hard()]

    def soft_constraints(self) -> list[Constraint]:
        """Return only soft constraints."""
        return [c for c in self.constraints if not c.is_hard()]
