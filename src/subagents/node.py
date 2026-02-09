"""Node definitions for sub-agent execution graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional


class NodeType(str, Enum):
    """Types of nodes in a sub-agent graph."""
    LLM_GENERATE = "llm_generate"  # LLM text generation (no tools)
    LLM_TOOL_USE = "llm_tool_use"  # LLM with tool calling
    ROUTER = "router"              # Conditional routing (deterministic)
    FUNCTION = "function"          # Pure function execution (no LLM)


@dataclass
class SubAgentNode:
    """A node in a sub-agent execution graph.

    Each node represents a single step — either an LLM call, a routing
    decision, or a deterministic function. Nodes declare their input/output
    keys so the executor can propagate context between steps.
    """
    id: str
    name: str
    node_type: NodeType

    # Keys this node reads from the execution context
    input_keys: list[str] = field(default_factory=list)

    # Keys this node writes to the execution context
    output_keys: list[str] = field(default_factory=list)

    # System prompt (for LLM_GENERATE and LLM_TOOL_USE)
    # Supports {key} interpolation from execution context
    system_prompt: str = ""

    # Tools available (only used for LLM_TOOL_USE nodes)
    tools: list[str] = field(default_factory=list)

    # Max retries on failure before propagating error
    max_retries: int = 1

    # Optional JSON schema for input validation
    input_schema: Optional[dict[str, Any]] = None

    # Optional JSON schema for output validation
    output_schema: Optional[dict[str, Any]] = None

    # For FUNCTION nodes: the callable to execute
    # Signature: async (context: dict) -> dict
    function: Optional[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = None

    # For ROUTER nodes: routing rules as list of (condition_key, expected_value, target_node_id)
    # Evaluated in order; first match wins. Use ("default", None, target) as fallback.
    routing_rules: list[tuple[str, Any, str]] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate node configuration. Returns list of error messages."""
        errors: list[str] = []

        if not self.id:
            errors.append("Node must have an id")
        if not self.name:
            errors.append("Node must have a name")

        if self.node_type in (NodeType.LLM_GENERATE, NodeType.LLM_TOOL_USE):
            if not self.system_prompt:
                errors.append(f"Node '{self.id}' ({self.node_type.value}) requires a system_prompt")

        if self.node_type == NodeType.LLM_TOOL_USE:
            if not self.tools:
                errors.append(f"Node '{self.id}' (llm_tool_use) requires at least one tool")

        if self.node_type == NodeType.FUNCTION:
            if self.function is None:
                errors.append(f"Node '{self.id}' (function) requires a callable")

        if self.node_type == NodeType.ROUTER:
            if not self.routing_rules:
                errors.append(f"Node '{self.id}' (router) requires routing_rules")

        return errors

    def interpolate_prompt(self, context: dict[str, Any]) -> str:
        """Interpolate system prompt with values from execution context.

        Uses safe replacement — missing keys are left as-is.
        """
        result = self.system_prompt
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
