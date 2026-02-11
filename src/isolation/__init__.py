"""Ornstein & Smough: Dual-layer container isolation system."""

from __future__ import annotations
from enum import Enum
from typing import Any

from src.isolation.ornstein import OrnsteinSandbox, OrnsteinConfig, OrnsteinResult


class IsolationLevel(Enum):
    """Isolation level for execution."""
    NONE = "none"           # Direct execution (no isolation)
    ORNSTEIN = "ornstein"   # Lightweight process sandbox
    SMOUGH = "smough"       # Heavy container isolation (not yet implemented)


def execute_with_isolation(
    func: callable,
    args: tuple = (),
    kwargs: dict[str, Any] = None,
    level: IsolationLevel = IsolationLevel.NONE,
    config: OrnsteinConfig = None,
) -> OrnsteinResult:
    """
    Execute function with appropriate isolation level.

    Args:
        func: The function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        level: Isolation level (NONE, ORNSTEIN, SMOUGH)
        config: Configuration for Ornstein sandbox (if level=ORNSTEIN)

    Returns:
        OrnsteinResult with output, errors, and isolation metadata
    """
    kwargs = kwargs or {}

    if level == IsolationLevel.NONE:
        # Direct execution
        try:
            result = func(*args, **kwargs)
            return OrnsteinResult(
                success=True,
                output=result,
                error=None,
                isolation_level="none",
                resource_usage={}
            )
        except Exception as e:
            return OrnsteinResult(
                success=False,
                output=None,
                error=str(e),
                isolation_level="none",
                resource_usage={}
            )

    elif level == IsolationLevel.ORNSTEIN:
        # Lightweight sandbox
        sandbox = OrnsteinSandbox(config or OrnsteinConfig())
        return sandbox.execute(func, args, kwargs)

    elif level == IsolationLevel.SMOUGH:
        # Heavy container (not yet implemented)
        raise NotImplementedError("Smough layer not yet implemented")

    else:
        raise ValueError(f"Unknown isolation level: {level}")


__all__ = [
    "IsolationLevel",
    "execute_with_isolation",
    "OrnsteinSandbox",
    "OrnsteinConfig",
    "OrnsteinResult",
]
