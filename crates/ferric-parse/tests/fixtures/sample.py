"""Sample Python file for ferric-parse integration tests."""


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


def standalone_function(x: float) -> float:
    """A standalone function."""
    return x * 2.0
