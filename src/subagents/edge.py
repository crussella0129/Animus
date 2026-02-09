"""Edge definitions for sub-agent execution graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class EdgeCondition(str, Enum):
    """Conditions for edge traversal."""
    ON_SUCCESS = "on_success"    # Traverse when source node succeeds
    ON_FAILURE = "on_failure"    # Traverse when source node fails
    ALWAYS = "always"            # Always traverse after source completes
    CONDITIONAL = "conditional"  # Traverse when expression evaluates truthy


@dataclass
class SubAgentEdge:
    """A directed edge between two nodes in a sub-agent graph.

    Edges define flow control. After a node completes, the executor
    evaluates outgoing edges by priority (lower = higher priority)
    and traverses matching edges.
    """
    id: str
    source: str  # Source node id
    target: str  # Target node id
    condition: EdgeCondition = EdgeCondition.ON_SUCCESS

    # Lower priority = evaluated first. When multiple edges match,
    # only the highest-priority (lowest number) edge is followed
    # unless parallel execution is enabled.
    priority: int = 0

    # For CONDITIONAL edges: key in execution context to evaluate
    condition_key: Optional[str] = None

    # For CONDITIONAL edges: expected value (equality check)
    # If None, checks truthiness of the context value
    condition_value: Optional[Any] = None

    def evaluate(self, success: bool, context: dict[str, Any]) -> bool:
        """Evaluate whether this edge should be traversed.

        Args:
            success: Whether the source node succeeded.
            context: Current execution context.

        Returns:
            True if the edge condition is met.
        """
        if self.condition == EdgeCondition.ALWAYS:
            return True
        elif self.condition == EdgeCondition.ON_SUCCESS:
            return success
        elif self.condition == EdgeCondition.ON_FAILURE:
            return not success
        elif self.condition == EdgeCondition.CONDITIONAL:
            if self.condition_key is None:
                return False
            value = context.get(self.condition_key)
            if self.condition_value is not None:
                return value == self.condition_value
            return bool(value)
        return False

    def validate(self) -> list[str]:
        """Validate edge configuration. Returns list of error messages."""
        errors: list[str] = []

        if not self.id:
            errors.append("Edge must have an id")
        if not self.source:
            errors.append("Edge must have a source node")
        if not self.target:
            errors.append("Edge must have a target node")
        if self.condition == EdgeCondition.CONDITIONAL and not self.condition_key:
            errors.append(
                f"Edge '{self.id}' (conditional) requires a condition_key"
            )
        return errors
