"""Model fallback chain — escalate through models on consecutive failures.

Provides a reliability layer between the Agent and LLM providers.
When a model fails consecutively, the chain escalates to the next model.

Example chain: local/qwen-7b → local/qwen-32b → gpt-4o

Usage:
    chain = ModelFallbackChain([
        FallbackModel("local/qwen3-vl-7b", max_failures=3),
        FallbackModel("local/qwen3-vl-32b", max_failures=2),
        FallbackModel("gpt-4o", max_failures=1),
    ])

    model = chain.current_model  # "local/qwen3-vl-7b"
    chain.record_failure()       # failure count → 1
    chain.record_failure()       # failure count → 2
    chain.record_failure()       # failure count → 3 → escalates
    model = chain.current_model  # "local/qwen3-vl-32b"
    chain.record_success()       # resets failure count, optionally de-escalates
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FallbackModel:
    """A model in the fallback chain."""

    model: str  # Model name (e.g. "local/qwen3-vl-7b", "gpt-4o")
    max_failures: int = 3  # Consecutive failures before escalating
    cooldown_seconds: float = 60.0  # Cooldown before retrying after escalation

    # Runtime state (not part of config)
    consecutive_failures: int = field(default=0, repr=False)
    last_failure_time: float = field(default=0.0, repr=False)
    total_failures: int = field(default=0, repr=False)
    total_successes: int = field(default=0, repr=False)


@dataclass
class FallbackEvent:
    """Record of a fallback escalation/de-escalation."""

    timestamp: float
    from_model: str
    to_model: str
    reason: str  # "escalation" or "de-escalation"
    failure_count: int


class ModelFallbackChain:
    """Ordered chain of models with automatic escalation on failures.

    Models are tried in order. When a model exceeds max_failures consecutive
    failures, the chain escalates to the next model. On success, the chain
    can optionally de-escalate back to a preferred (cheaper/faster) model
    after a cooldown period.
    """

    def __init__(
        self,
        models: list[FallbackModel],
        auto_deescalate: bool = True,
    ):
        """Initialize the fallback chain.

        Args:
            models: Ordered list of fallback models (preferred first).
            auto_deescalate: If True, return to preferred model after
                cooldown period following a success.
        """
        if not models:
            raise ValueError("Fallback chain requires at least one model")

        self._models = list(models)
        self._current_index = 0
        self._auto_deescalate = auto_deescalate
        self._events: list[FallbackEvent] = []

    @property
    def current_model(self) -> str:
        """Get the current active model name."""
        return self._models[self._current_index].model

    @property
    def current_index(self) -> int:
        """Get the current position in the chain (0 = preferred)."""
        return self._current_index

    @property
    def is_escalated(self) -> bool:
        """True if we've moved past the preferred (first) model."""
        return self._current_index > 0

    @property
    def models(self) -> list[FallbackModel]:
        """Get all models in the chain."""
        return list(self._models)

    @property
    def events(self) -> list[FallbackEvent]:
        """Get the history of escalation/de-escalation events."""
        return list(self._events)

    def record_failure(self) -> bool:
        """Record a failure for the current model.

        Returns:
            True if the chain escalated to a new model.
        """
        current = self._models[self._current_index]
        current.consecutive_failures += 1
        current.total_failures += 1
        current.last_failure_time = time.monotonic()

        if current.consecutive_failures >= current.max_failures:
            return self._escalate()

        return False

    def record_success(self) -> bool:
        """Record a success for the current model.

        Returns:
            True if the chain de-escalated to a preferred model.
        """
        current = self._models[self._current_index]
        current.consecutive_failures = 0
        current.total_successes += 1

        if self._auto_deescalate and self._current_index > 0:
            return self._try_deescalate()

        return False

    def _escalate(self) -> bool:
        """Move to the next model in the chain.

        Returns:
            True if escalation happened, False if already at last model.
        """
        if self._current_index >= len(self._models) - 1:
            logger.warning(
                "Fallback chain exhausted — already at last model: %s",
                self.current_model,
            )
            return False

        old_model = self.current_model
        old_failures = self._models[self._current_index].consecutive_failures
        self._current_index += 1

        event = FallbackEvent(
            timestamp=time.monotonic(),
            from_model=old_model,
            to_model=self.current_model,
            reason="escalation",
            failure_count=old_failures,
        )
        self._events.append(event)

        logger.info(
            "Fallback escalation: %s → %s (after %d failures)",
            old_model,
            self.current_model,
            old_failures,
        )
        return True

    def _try_deescalate(self) -> bool:
        """Try to return to a preferred model after cooldown.

        Returns:
            True if de-escalation happened.
        """
        # Check if preferred model's cooldown has expired
        preferred = self._models[0]
        elapsed = time.monotonic() - preferred.last_failure_time

        if elapsed < preferred.cooldown_seconds:
            return False

        old_model = self.current_model
        self._current_index = 0
        preferred.consecutive_failures = 0  # Reset for fresh start

        event = FallbackEvent(
            timestamp=time.monotonic(),
            from_model=old_model,
            to_model=self.current_model,
            reason="de-escalation",
            failure_count=0,
        )
        self._events.append(event)

        logger.info(
            "Fallback de-escalation: %s → %s (cooldown expired)",
            old_model,
            self.current_model,
        )
        return True

    def reset(self) -> None:
        """Reset the chain to the preferred model, clearing all state."""
        self._current_index = 0
        self._events.clear()
        for m in self._models:
            m.consecutive_failures = 0
            m.last_failure_time = 0.0
            m.total_failures = 0
            m.total_successes = 0

    def stats(self) -> dict:
        """Get statistics for all models in the chain."""
        return {
            "current_model": self.current_model,
            "current_index": self._current_index,
            "is_escalated": self.is_escalated,
            "escalation_count": sum(
                1 for e in self._events if e.reason == "escalation"
            ),
            "models": [
                {
                    "model": m.model,
                    "consecutive_failures": m.consecutive_failures,
                    "total_failures": m.total_failures,
                    "total_successes": m.total_successes,
                }
                for m in self._models
            ],
        }
