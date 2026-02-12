"""Shared utility for parsing tool calls from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM response using multiple strategies.

    Tries three formats in order:
    1. Raw JSON (GBNF grammar output — no wrapping)
    2. JSON code blocks: ```json {"name": ..., "arguments": ...} ```
    3. Inline JSON: {"name": ..., "arguments": ...}

    Args:
        text: The LLM response text to parse

    Returns:
        List of tool call dicts with "name" and "arguments" keys
    """
    calls: list[dict[str, Any]] = []

    # Strategy 0: Raw JSON (GBNF grammar output — entire response is JSON)
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            calls.append({
                "name": data["name"],
                "arguments": data["arguments"] if isinstance(data["arguments"], dict) else {},
            })
            return calls
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 1: JSON code blocks with tool_call structure
    json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if "name" in data and "arguments" in data:
                calls.append({
                    "name": data["name"],
                    "arguments": data["arguments"] if isinstance(data["arguments"], dict) else {},
                })
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Strategy 2: Inline JSON tool calls
    inline_pattern = r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\}'
    for match in re.finditer(inline_pattern, text):
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    return calls
