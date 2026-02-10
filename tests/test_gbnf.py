"""Tests for GBNF Grammar Generation module."""

import json
import pytest
from unittest.mock import MagicMock

from src.core.gbnf import (
    schema_to_gbnf,
    tool_call_grammar,
    tool_call_array_grammar,
    _generic_tool_call_grammar,
    _primitives,
)


# =============================================================================
# Helper: Mock Tool
# =============================================================================

def _make_tool(name: str, params: dict) -> MagicMock:
    """Create a mock tool with name and OpenAI schema."""
    tool = MagicMock()
    tool.name = name
    tool.to_openai_schema.return_value = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool: {name}",
            "parameters": params,
        },
    }
    return tool


# =============================================================================
# Primitives Tests
# =============================================================================

class TestPrimitives:
    """Test GBNF primitive rules."""

    def test_primitives_count(self):
        """Primitives should include all standard JSON types."""
        prims = _primitives()
        assert len(prims) == 9  # ws, string, number, int, bool, null, value, array, object

    def test_primitives_contain_ws(self):
        """Primitives should include whitespace rule."""
        prims = _primitives()
        assert any("ws ::=" in p for p in prims)

    def test_primitives_contain_json_string(self):
        """Primitives should include json-string rule."""
        prims = _primitives()
        assert any("json-string ::=" in p for p in prims)


# =============================================================================
# Schema-to-GBNF Tests
# =============================================================================

class TestSchemaToGbnf:
    """Test schema_to_gbnf function."""

    def test_simple_string_property(self):
        """Object with string property."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        })

        assert "root" in grammar
        assert "json-string" in grammar
        assert '"path"' in grammar or "path" in grammar

    def test_multiple_required_properties(self):
        """Object with multiple required properties."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        })

        assert "root" in grammar

    def test_optional_properties(self):
        """Object with optional properties."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "required": ["path"],
        })

        assert "root" in grammar

    def test_integer_property(self):
        """Object with integer property."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
            "required": ["count"],
        })

        assert "json-int" in grammar

    def test_boolean_property(self):
        """Object with boolean property."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "recursive": {"type": "boolean"},
            },
            "required": ["recursive"],
        })

        assert "json-bool" in grammar

    def test_number_property(self):
        """Object with number property."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
            },
            "required": ["temperature"],
        })

        assert "json-number" in grammar

    def test_array_property(self):
        """Object with array property."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["files"],
        })

        assert "root" in grammar
        assert "[" in grammar  # Array brackets

    def test_enum_property(self):
        """Property with enum constraint."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "enum": ["red", "green", "blue"],
                },
            },
            "required": ["color"],
        })

        assert "red" in grammar
        assert "green" in grammar
        assert "blue" in grammar

    def test_empty_properties(self):
        """Object with no properties should use generic object rule."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {},
        })

        assert "json-object" in grammar

    def test_nested_object(self):
        """Object with nested object property."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                    },
                    "required": ["key"],
                },
            },
            "required": ["config"],
        })

        assert "root" in grammar
        assert "key" in grammar

    def test_string_type_alone(self):
        """Direct string type should reference json-string."""
        grammar = schema_to_gbnf({"type": "string"})
        assert "json-string" in grammar

    def test_all_optional_properties(self):
        """Object with all optional properties."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "opt1": {"type": "string"},
                "opt2": {"type": "integer"},
            },
            "required": [],
        })

        assert "root" in grammar


# =============================================================================
# Tool Call Grammar Tests
# =============================================================================

class TestToolCallGrammar:
    """Test tool_call_grammar function."""

    def test_single_tool(self):
        """Grammar for a single tool."""
        tool = _make_tool("read_file", {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        })

        grammar = tool_call_grammar([tool])

        assert "root" in grammar
        assert "read_file" in grammar
        assert "name" in grammar
        assert "arguments" in grammar

    def test_multiple_tools(self):
        """Grammar for multiple tools should allow any."""
        tools = [
            _make_tool("read_file", {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
            _make_tool("write_file", {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            }),
            _make_tool("list_dir", {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
        ]

        grammar = tool_call_grammar(tools)

        assert "read_file" in grammar
        assert "write_file" in grammar
        assert "list_dir" in grammar

    def test_empty_tools_gives_generic(self):
        """Empty tools list should produce generic grammar."""
        grammar = tool_call_grammar([])

        assert "root" in grammar
        assert "json-string" in grammar
        assert "json-object" in grammar

    def test_generic_grammar(self):
        """Generic grammar should accept name+arguments."""
        grammar = _generic_tool_call_grammar()

        assert "root" in grammar
        assert "name" in grammar
        assert "arguments" in grammar


# =============================================================================
# Tool Call Array Grammar Tests
# =============================================================================

class TestToolCallArrayGrammar:
    """Test tool_call_array_grammar function."""

    def test_array_wraps_single_calls(self):
        """Array grammar should wrap tool calls in brackets."""
        tool = _make_tool("test", {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        })

        grammar = tool_call_array_grammar([tool])

        assert "root" in grammar
        assert "single-call" in grammar
        # Should have array brackets
        assert "[" in grammar

    def test_array_grammar_valid_structure(self):
        """Array grammar should be well-formed GBNF."""
        tools = [
            _make_tool("a", {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }),
            _make_tool("b", {
                "type": "object",
                "properties": {"y": {"type": "integer"}},
                "required": ["y"],
            }),
        ]

        grammar = tool_call_array_grammar(tools)

        # Should have root rule pointing to array
        lines = grammar.split("\n")
        root_line = [l for l in lines if l.startswith("root ::=")]
        assert len(root_line) == 1
        assert "single-call" in root_line[0]


# =============================================================================
# Grammar Structure Validation Tests
# =============================================================================

class TestGrammarStructure:
    """Test that generated grammars are well-formed."""

    def test_grammar_has_root_rule(self):
        """Every grammar must have a root rule."""
        grammar = schema_to_gbnf({"type": "object", "properties": {}})
        assert grammar.startswith("root ::=") or "\nroot ::=" in grammar

    def test_grammar_has_primitives(self):
        """Every grammar should include primitive rules."""
        grammar = schema_to_gbnf({"type": "string"})
        assert "ws ::=" in grammar
        assert "json-string ::=" in grammar

    def test_grammar_lines_are_rules(self):
        """All non-empty lines should be valid rule definitions."""
        grammar = schema_to_gbnf({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        })

        for line in grammar.strip().split("\n"):
            line = line.strip()
            if line:
                assert "::=" in line, f"Line is not a valid rule: {line}"

    def test_no_duplicate_rule_names(self):
        """Grammar should not have duplicate rule names."""
        grammar = tool_call_grammar([
            _make_tool("t1", {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }),
            _make_tool("t2", {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            }),
        ])

        names = []
        for line in grammar.strip().split("\n"):
            if "::=" in line:
                name = line.split("::=")[0].strip()
                names.append(name)

        # Check for duplicates (primitives may appear once each)
        seen = set()
        for name in names:
            assert name not in seen, f"Duplicate rule: {name}"
            seen.add(name)
