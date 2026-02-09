"""Output validation and cleaning for inter-node data flow."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OutputCleaner:
    """Validates and cleans node output for the next node's input.

    Catches common LLM output issues:
    - JSON parsing failures
    - Entire response stuffed into a single key (JSON trap)
    - Missing required output keys
    - Type mismatches for basic types
    """

    def validate(
        self,
        output: dict[str, Any],
        expected_keys: list[str],
        schema: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """Validate output against expected keys and optional schema.

        Args:
            output: The node output dict.
            expected_keys: Keys the next node expects.
            schema: Optional JSON schema-like dict for type checking.

        Returns:
            List of validation error messages (empty = valid).
        """
        errors: list[str] = []

        # Check required keys present
        for key in expected_keys:
            if key not in output:
                errors.append(f"Missing required key: '{key}'")

        # Basic type checking from schema
        if schema:
            for key, expected_type in schema.items():
                if key in output:
                    if not self._check_type(output[key], expected_type):
                        errors.append(
                            f"Key '{key}' expected type '{expected_type}', "
                            f"got '{type(output[key]).__name__}'"
                        )

        # Detect JSON trap: entire response stuffed into one key
        if self._detect_json_trap(output):
            errors.append("Possible JSON trap: entire response in single key")

        return errors

    def clean(
        self,
        raw_text: str,
        expected_keys: list[str],
    ) -> Optional[dict[str, Any]]:
        """Attempt to extract a valid output dict from raw LLM text.

        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract JSON from markdown code blocks
        3. Extract key-value pairs from text

        Args:
            raw_text: Raw LLM output text.
            expected_keys: Keys to look for.

        Returns:
            Cleaned dict or None if extraction fails.
        """
        # Strategy 1: Direct JSON parse
        result = self._try_json_parse(raw_text)
        if result and self._has_expected_keys(result, expected_keys):
            return result

        # Strategy 2: Extract from markdown code blocks
        result = self._try_code_block_parse(raw_text)
        if result and self._has_expected_keys(result, expected_keys):
            return result

        # Strategy 3: Map first expected key to full text
        if expected_keys:
            return {expected_keys[0]: raw_text}

        return None

    def _detect_json_trap(self, output: dict[str, Any]) -> bool:
        """Detect if the output is a JSON trap.

        A JSON trap occurs when the LLM wraps its entire response in a
        single JSON key like {"response": "..."} or {"result": "..."}.
        """
        if len(output) != 1:
            return False

        value = next(iter(output.values()))
        if not isinstance(value, str):
            return False

        # If the single value is very long and looks like structured content,
        # it's probably a trap
        if len(value) > 500 and any(
            marker in value for marker in ["\n\n", "```", "## ", "- ", "1. "]
        ):
            return True

        return False

    def _try_json_parse(self, text: str) -> Optional[dict[str, Any]]:
        """Try to parse text as JSON."""
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _try_code_block_parse(self, text: str) -> Optional[dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        pattern = r"```(?:json)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            result = self._try_json_parse(match)
            if result is not None:
                return result
        return None

    def _has_expected_keys(
        self, result: dict[str, Any], expected_keys: list[str]
    ) -> bool:
        """Check if result contains at least one expected key."""
        if not expected_keys:
            return True
        return any(k in result for k in expected_keys)

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches an expected type string."""
        type_map = {
            "string": str,
            "str": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }
        expected = type_map.get(expected_type.lower())
        if expected is None:
            return True  # Unknown type, skip check
        return isinstance(value, expected)
