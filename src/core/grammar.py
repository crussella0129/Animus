"""GBNF grammar constraints for structured tool call output from local models.

Converts tool schemas to JSON Schema, then to GBNF grammar via llama-cpp-python.
This forces small models to produce structurally valid JSON tool calls instead of
free-form text that may contain malformed JSON.

Two-level API:
- build_tool_call_schema(tools) -> dict: pure JSON Schema, no dependencies
- build_grammar(tools) -> LlamaGrammar | None: requires llama-cpp-python (graceful fallback)
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src.tools.base import Tool


def build_tool_call_schema(tools: list[Tool]) -> dict[str, Any]:
    """Build a JSON Schema constraining output to a valid tool call.

    Returns a schema for: {"name": "<tool_name>", "arguments": {...}}
    where name is constrained to an enum of available tool names.

    For a single tool, arguments are fully constrained to that tool's parameter schema.
    For multiple tools, arguments are constrained to a generic JSON object
    (oneOf with per-tool argument schemas is unreliable in GBNF conversion).
    """
    if not tools:
        return {"type": "object"}

    tool_names = [t.name for t in tools]

    if len(tools) == 1:
        tool = tools[0]
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [tool.name]},
                "arguments": tool.parameters,
            },
            "required": ["name", "arguments"],
        }

    # Multiple tools: enum for name, generic arguments object.
    # This ensures structural validity (valid JSON with correct keys)
    # while leaving argument validation to the tool registry.
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": tool_names},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }


def build_grammar(tools: list[Tool]) -> Optional[Any]:
    """Create a LlamaGrammar from tool schemas for constrained decoding.

    Returns None if llama-cpp-python is not installed or grammar conversion fails.
    Callers should treat None as "no constraint" and proceed normally.
    """
    if not tools:
        return None

    schema = build_tool_call_schema(tools)
    schema_json = json.dumps(schema)

    try:
        from llama_cpp import LlamaGrammar  # type: ignore[import-untyped]

        return LlamaGrammar.from_json_schema(schema_json)
    except ImportError:
        return None
    except Exception:
        # Grammar conversion failed — degrade gracefully to unconstrained
        return None
