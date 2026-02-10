"""GBNF Grammar Generation — constrained decoding for tool call schemas.

Generates GBNF (GGML BNF) grammars from tool JSON schemas, enabling
llama-cpp-python to produce structurally valid JSON tool calls through
grammar-constrained decoding.

Implementation Principle: 100% hardcoded. The grammar is derived
deterministically from the tool schema with no LLM involvement.

GBNF Format Reference (from llama.cpp):
  root   ::= <rule>
  rule   ::= <name> "::=" <alternative> ("\\n" <alternative>)*
  alternative ::= <element>+
  element ::= <literal> | <name> | <group> | <repeat>

Usage:
    from src.core.gbnf import schema_to_gbnf, tool_call_grammar

    # Generate grammar for a single tool's parameters
    grammar = schema_to_gbnf({
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path"],
    })

    # Generate grammar that accepts any tool call from a list
    grammar = tool_call_grammar([tool1, tool2, tool3])

    # Use with llama-cpp-python:
    # response = llm(prompt, grammar=LlamaGrammar.from_string(grammar))
"""

from __future__ import annotations

from typing import Any, Optional


# =============================================================================
# Primitive GBNF Rules (reusable building blocks)
# =============================================================================

# Whitespace
WS = 'ws ::= [ \\t\\n]*'

# JSON primitives
JSON_STRING = r'json-string ::= "\"" ([^"\\] | "\\" .)* "\""'
JSON_NUMBER = r'json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'
JSON_INT = r'json-int ::= "-"? [0-9]+'
JSON_BOOL = r'json-bool ::= "true" | "false"'
JSON_NULL = r'json-null ::= "null"'

# JSON value (any type)
JSON_VALUE = (
    'json-value ::= json-string | json-number | json-bool | json-null '
    '| json-array | json-object'
)

# JSON array
JSON_ARRAY = (
    'json-array ::= "[" ws "]" '
    '| "[" ws json-value (ws "," ws json-value)* ws "]"'
)

# JSON object (generic)
JSON_OBJECT = (
    'json-object ::= "{" ws "}" '
    '| "{" ws json-string ws ":" ws json-value '
    '(ws "," ws json-string ws ":" ws json-value)* ws "}"'
)


def _primitives() -> list[str]:
    """Return all primitive GBNF rules."""
    return [
        WS,
        JSON_STRING,
        JSON_NUMBER,
        JSON_INT,
        JSON_BOOL,
        JSON_NULL,
        JSON_VALUE,
        JSON_ARRAY,
        JSON_OBJECT,
    ]


# =============================================================================
# Schema-to-GBNF Conversion
# =============================================================================

def _type_rule(
    name: str,
    schema: dict[str, Any],
    rules: list[str],
    counter: list[int],
) -> str:
    """Generate a GBNF rule for a JSON schema type.

    Args:
        name: Rule name for this type.
        schema: JSON Schema fragment.
        rules: Accumulator for additional rules.
        counter: Mutable counter for unique rule names.

    Returns:
        The rule name to reference.
    """
    schema_type = schema.get("type", "string")

    # Enum constraint
    if "enum" in schema:
        alternatives = " | ".join(
            f'"\\"{v}\\""' for v in schema["enum"]
        )
        rules.append(f'{name} ::= {alternatives}')
        return name

    # Const constraint
    if "const" in schema:
        rules.append(f'{name} ::= "\\"{schema["const"]}\\""')
        return name

    if schema_type == "string":
        return "json-string"

    if schema_type == "integer":
        return "json-int"

    if schema_type == "number":
        return "json-number"

    if schema_type == "boolean":
        return "json-bool"

    if schema_type == "null":
        return "json-null"

    if schema_type == "array":
        items = schema.get("items", {})
        counter[0] += 1
        item_name = f"item-{counter[0]}"
        item_ref = _type_rule(item_name, items, rules, counter)
        rules.append(
            f'{name} ::= "[" ws "]" '
            f'| "[" ws {item_ref} (ws "," ws {item_ref})* ws "]"'
        )
        return name

    if schema_type == "object":
        return _object_rule(name, schema, rules, counter)

    # Fallback: any JSON value
    return "json-value"


def _object_rule(
    name: str,
    schema: dict[str, Any],
    rules: list[str],
    counter: list[int],
) -> str:
    """Generate a GBNF rule for a JSON object schema.

    Produces rules that enforce:
    - Required properties must be present
    - Optional properties may appear in any order
    - Property values match their declared types

    Args:
        name: Rule name for this object.
        schema: JSON Schema for the object.
        rules: Accumulator for additional rules.
        counter: Mutable counter for unique rule names.

    Returns:
        The rule name.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        # No properties defined — accept any JSON object
        return "json-object"

    # Generate a type rule for each property
    prop_refs: dict[str, str] = {}
    for prop_name, prop_schema in properties.items():
        counter[0] += 1
        prop_rule_name = f"prop-{counter[0]}"
        prop_refs[prop_name] = _type_rule(prop_rule_name, prop_schema, rules, counter)

    # Build property pair rules: "key" : value
    pair_rules = []
    for prop_name, ref in prop_refs.items():
        counter[0] += 1
        pair_name = f"pair-{counter[0]}"
        rules.append(
            f'{pair_name} ::= "\\"{prop_name}\\"" ws ":" ws {ref}'
        )
        pair_rules.append((prop_name, pair_name))

    # Strategy: If all properties are required and <= 6, generate
    # a fixed-order rule. Otherwise, generate a flexible rule.
    req_pairs = [(n, r) for n, r in pair_rules if n in required]
    opt_pairs = [(n, r) for n, r in pair_rules if n not in required]

    if not opt_pairs and len(req_pairs) <= 6:
        # Fixed order: all required, no optional
        inner = " ws \",\" ws ".join(r for _, r in req_pairs)
        rules.append(f'{name} ::= "{{" ws {inner} ws "}}"')
    elif not req_pairs:
        # All optional — accept any combination
        if len(opt_pairs) == 1:
            _, r = opt_pairs[0]
            rules.append(f'{name} ::= "{{" ws "}}" | "{{" ws {r} ws "}}"')
        else:
            any_pair = " | ".join(r for _, r in opt_pairs)
            counter[0] += 1
            any_name = f"any-pair-{counter[0]}"
            rules.append(f'{any_name} ::= {any_pair}')
            rules.append(
                f'{name} ::= "{{" ws "}}" '
                f'| "{{" ws {any_name} (ws "," ws {any_name})* ws "}}"'
            )
    else:
        # Mix of required and optional
        # Required in fixed order, optional appended
        req_inner = " ws \",\" ws ".join(r for _, r in req_pairs)

        if opt_pairs:
            opt_any = " | ".join(r for _, r in opt_pairs)
            counter[0] += 1
            opt_name = f"opt-pair-{counter[0]}"
            rules.append(f'{opt_name} ::= {opt_any}')
            rules.append(
                f'{name} ::= "{{" ws {req_inner} ws "}}" '
                f'| "{{" ws {req_inner} '
                f'(ws "," ws {opt_name})+ ws "}}"'
            )
        else:
            rules.append(f'{name} ::= "{{" ws {req_inner} ws "}}"')

    return name


def schema_to_gbnf(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema to a GBNF grammar string.

    Args:
        schema: A JSON Schema dict (typically the "parameters" from a tool).

    Returns:
        Complete GBNF grammar string.
    """
    rules: list[str] = []
    counter = [0]

    root_ref = _type_rule("root-value", schema, rules, counter)

    # Build the grammar
    lines = []
    lines.append(f'root ::= {root_ref}')
    lines.extend(rules)
    lines.extend(_primitives())

    return "\n".join(lines)


# =============================================================================
# Tool Call Grammar Generation
# =============================================================================

def tool_call_grammar(tools: list[Any]) -> str:
    """Generate a GBNF grammar that accepts a JSON tool call for any tool.

    Produces a grammar that enforces:
    ```json
    {"name": "<tool_name>", "arguments": {<valid_args>}}
    ```

    Args:
        tools: List of Tool objects with `name`, `parameters`, and
               `to_openai_schema()` method.

    Returns:
        Complete GBNF grammar string.
    """
    if not tools:
        return _generic_tool_call_grammar()

    rules: list[str] = []
    counter = [0]
    tool_alternatives = []

    for tool in tools:
        schema = tool.to_openai_schema()
        func_schema = schema.get("function", {})
        tool_name = func_schema.get("name", tool.name)
        params_schema = func_schema.get("parameters", {"type": "object"})

        counter[0] += 1
        args_name = f"args-{counter[0]}"
        _type_rule(args_name, params_schema, rules, counter)

        counter[0] += 1
        call_name = f"tool-call-{counter[0]}"
        rules.append(
            f'{call_name} ::= '
            f'"{{" ws "\\\"name\\\"" ws ":" ws "\\\"" "{tool_name}" "\\\"" '
            f'ws "," ws "\\\"arguments\\\"" ws ":" ws {args_name} ws "}}"'
        )
        tool_alternatives.append(call_name)

    # Root: any of the tool call alternatives
    root_alt = " | ".join(tool_alternatives)
    lines = [f'root ::= {root_alt}']
    lines.extend(rules)
    lines.extend(_primitives())

    return "\n".join(lines)


def _generic_tool_call_grammar() -> str:
    """Generate a generic tool call grammar (any name, any arguments).

    Used when no specific tools are provided.
    """
    lines = [
        'root ::= "{" ws "\\"name\\"" ws ":" ws json-string '
        'ws "," ws "\\"arguments\\"" ws ":" ws json-object ws "}"',
    ]
    lines.extend(_primitives())
    return "\n".join(lines)


def tool_call_array_grammar(tools: list[Any]) -> str:
    """Generate a grammar that accepts an array of tool calls.

    Produces a grammar for:
    ```json
    [{"name": "...", "arguments": {...}}, ...]
    ```

    Args:
        tools: List of Tool objects.

    Returns:
        Complete GBNF grammar string.
    """
    # Generate single tool call grammar first
    single = tool_call_grammar(tools)

    # Find the root rule and rename it
    lines = single.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("root ::="):
            # Rename root to single-call
            new_lines.append(line.replace("root ::=", "single-call ::="))
        else:
            new_lines.append(line)

    # Add array root
    new_lines.insert(0,
        'root ::= "[" ws single-call (ws "," ws single-call)* ws "]"'
    )

    return "\n".join(new_lines)
