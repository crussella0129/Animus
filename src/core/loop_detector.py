"""Action loop detection — detect and break repetitive agent behavior.

Monitors tool calls for repetitive patterns and applies escalating
intervention to break the agent out of loops.

Intervention levels:
  1. NUDGE:  Append a gentle reminder to try a different approach
  2. FORCE:  Append a stronger directive to change strategy
  3. BREAK:  Stop the agent loop entirely

Example:
    detector = LoopDetector(window_size=6, threshold=3)
    detector.record("read_file", {"path": "/x"})
    detector.record("read_file", {"path": "/x"})
    detector.record("read_file", {"path": "/x"})
    action = detector.check()  # → InterventionLevel.NUDGE
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InterventionLevel(str, Enum):
    """Escalating intervention levels for loop detection."""
    NONE = "none"
    NUDGE = "nudge"
    FORCE = "force"
    BREAK = "break"


@dataclass
class LoopDetectorConfig:
    """Configuration for the loop detector."""
    window_size: int = 10  # Sliding window of recent actions
    nudge_threshold: int = 3  # Repeated actions to trigger nudge
    force_threshold: int = 5  # Repeated actions to trigger force
    break_threshold: int = 7  # Repeated actions to trigger break
    include_args: bool = True  # Compare args (not just tool name)


# Intervention messages appended to conversation
INTERVENTION_MESSAGES = {
    InterventionLevel.NUDGE: (
        "I notice you've been repeating the same action. "
        "Consider trying a different approach or tool."
    ),
    InterventionLevel.FORCE: (
        "WARNING: You are stuck in a loop, repeating the same action "
        "multiple times. You MUST try a completely different approach. "
        "Do NOT repeat the same tool call again."
    ),
    InterventionLevel.BREAK: (
        "LOOP DETECTED: Stopping execution. The same action was "
        "repeated too many times without progress."
    ),
}


@dataclass
class ActionRecord:
    """A recorded action for loop detection."""
    tool_name: str
    args_hash: int  # Hash of arguments for comparison


class LoopDetector:
    """Detects repetitive agent behavior using a sliding window.

    Tracks recent tool calls and detects when the same action is
    repeated beyond configurable thresholds. Applies escalating
    intervention: nudge → force → break.
    """

    def __init__(self, config: Optional[LoopDetectorConfig] = None):
        self.config = config or LoopDetectorConfig()
        self._history: deque[ActionRecord] = deque(
            maxlen=self.config.window_size,
        )
        self._intervention_count = 0

    def record(self, tool_name: str, args: Optional[dict[str, Any]] = None) -> None:
        """Record a tool call action.

        Args:
            tool_name: Name of the tool called.
            args: Tool arguments (hashed for comparison).
        """
        if self.config.include_args and args:
            # Sort keys for consistent hashing
            args_hash = hash(frozenset(
                (k, str(v)) for k, v in sorted(args.items())
            ))
        else:
            args_hash = hash(tool_name)

        self._history.append(ActionRecord(
            tool_name=tool_name,
            args_hash=args_hash,
        ))

    def check(self) -> InterventionLevel:
        """Check for repetitive patterns in the action history.

        Returns:
            The appropriate intervention level.
        """
        if len(self._history) < self.config.nudge_threshold:
            return InterventionLevel.NONE

        # Count consecutive identical actions from the end
        consecutive = self._count_consecutive()

        if consecutive >= self.config.break_threshold:
            self._intervention_count += 1
            logger.warning(
                "Loop BREAK: %d consecutive identical actions",
                consecutive,
            )
            return InterventionLevel.BREAK

        if consecutive >= self.config.force_threshold:
            self._intervention_count += 1
            logger.warning(
                "Loop FORCE: %d consecutive identical actions",
                consecutive,
            )
            return InterventionLevel.FORCE

        if consecutive >= self.config.nudge_threshold:
            self._intervention_count += 1
            logger.info(
                "Loop NUDGE: %d consecutive identical actions",
                consecutive,
            )
            return InterventionLevel.NUDGE

        # Also check for alternating patterns (A-B-A-B)
        alternating = self._count_alternating()
        if alternating >= self.config.force_threshold:
            self._intervention_count += 1
            logger.warning(
                "Alternating loop FORCE: %d alternating actions",
                alternating,
            )
            return InterventionLevel.FORCE

        if alternating >= self.config.nudge_threshold:
            self._intervention_count += 1
            logger.info(
                "Alternating loop NUDGE: %d alternating actions",
                alternating,
            )
            return InterventionLevel.NUDGE

        return InterventionLevel.NONE

    def get_message(self, level: InterventionLevel) -> Optional[str]:
        """Get the intervention message for a given level.

        Args:
            level: The intervention level.

        Returns:
            The message string, or None for NONE level.
        """
        return INTERVENTION_MESSAGES.get(level)

    def _count_consecutive(self) -> int:
        """Count consecutive identical actions from the end of history."""
        if not self._history:
            return 0

        last = self._history[-1]
        count = 0
        for action in reversed(self._history):
            if (action.tool_name == last.tool_name
                    and action.args_hash == last.args_hash):
                count += 1
            else:
                break

        return count

    def _count_alternating(self) -> int:
        """Count alternating pattern length (A-B-A-B → 4).

        Detects when the agent alternates between two actions
        without making progress.
        """
        if len(self._history) < 4:
            return 0

        items = list(self._history)
        # Check if the last 2 items form a pattern that repeats
        if len(items) < 4:
            return 0

        a = items[-2]
        b = items[-1]
        if a.tool_name == b.tool_name and a.args_hash == b.args_hash:
            return 0  # Not alternating, just repeating

        count = 0
        for i in range(len(items) - 1, -1, -2):
            if i < 1:
                break
            curr = items[i]
            prev = items[i - 1]
            if (curr.tool_name == b.tool_name
                    and curr.args_hash == b.args_hash
                    and prev.tool_name == a.tool_name
                    and prev.args_hash == a.args_hash):
                count += 2
            else:
                break

        return count

    @property
    def intervention_count(self) -> int:
        """Total number of interventions triggered."""
        return self._intervention_count

    def reset(self) -> None:
        """Reset the detector state."""
        self._history.clear()
        self._intervention_count = 0

    def stats(self) -> dict:
        """Get detector statistics."""
        return {
            "history_length": len(self._history),
            "intervention_count": self._intervention_count,
            "current_consecutive": self._count_consecutive(),
            "current_alternating": self._count_alternating(),
        }
