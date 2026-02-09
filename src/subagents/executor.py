"""Executor for sub-agent graphs."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.llm.base import ModelProvider, Message, GenerationConfig, GenerationResult
from src.tools.base import ToolRegistry
from src.subagents.graph import SubAgentGraph
from src.subagents.node import SubAgentNode, NodeType
from src.subagents.edge import SubAgentEdge

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from executing a single node."""
    node_id: str
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


@dataclass
class ExecutionResult:
    """Result from executing an entire graph."""
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    paused_at: Optional[str] = None  # Node id if execution paused
    total_duration_ms: float = 0.0


class SubAgentExecutor:
    """Executes a sub-agent graph from entry node to terminal/pause.

    Manages context propagation between nodes, handles retries with
    exponential backoff, and supports pause/resume via session state.
    """

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        max_node_retries: int = 3,
    ):
        self.provider = provider
        self.tool_registry = tool_registry
        self.max_node_retries = max_node_retries

    async def execute(
        self,
        graph: SubAgentGraph,
        initial_context: Optional[dict[str, Any]] = None,
        resume_from: Optional[str] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a graph from entry to terminal or pause.

        Args:
            graph: The graph to execute.
            initial_context: Starting context values.
            resume_from: Node id to resume from (for pause/resume).
            session_state: Restored session state (for pause/resume).

        Returns:
            ExecutionResult with steps, output, and pause state.
        """
        start_time = time.monotonic()
        context = dict(initial_context or {})
        if session_state:
            context.update(session_state)

        steps: list[StepResult] = []
        current_node_id = resume_from or graph.entry_node

        if current_node_id not in graph.nodes:
            return ExecutionResult(
                success=False,
                error=f"Start node '{current_node_id}' not found in graph",
                total_duration_ms=_elapsed_ms(start_time),
            )

        visited_count: dict[str, int] = {}
        max_visits = 50  # Circuit breaker for cycles

        while current_node_id:
            # Circuit breaker
            visited_count[current_node_id] = visited_count.get(current_node_id, 0) + 1
            if visited_count[current_node_id] > max_visits:
                return ExecutionResult(
                    success=False,
                    steps=steps,
                    output=context,
                    error=f"Circuit breaker: node '{current_node_id}' visited {max_visits} times",
                    total_duration_ms=_elapsed_ms(start_time),
                )

            node = graph.nodes[current_node_id]

            # Check for pause
            if current_node_id in graph.pause_nodes and not resume_from:
                return ExecutionResult(
                    success=True,
                    steps=steps,
                    output=context,
                    paused_at=current_node_id,
                    total_duration_ms=_elapsed_ms(start_time),
                )
            # Clear resume_from after first iteration so pause nodes work on re-visits
            resume_from = None

            # Execute node with retries
            step_result = await self._execute_node_with_retry(node, context)
            steps.append(step_result)

            # Update context with node output
            if step_result.success:
                context.update(step_result.output)

            # Check for terminal
            if current_node_id in graph.terminal_nodes:
                return ExecutionResult(
                    success=step_result.success,
                    steps=steps,
                    output=context,
                    error=step_result.error,
                    total_duration_ms=_elapsed_ms(start_time),
                )

            # Find next node via edges
            next_node_id = self._resolve_next_node(
                graph, current_node_id, step_result.success, context
            )

            if next_node_id is None:
                # No matching edge — execution ends
                return ExecutionResult(
                    success=step_result.success,
                    steps=steps,
                    output=context,
                    error=step_result.error if not step_result.success else None,
                    total_duration_ms=_elapsed_ms(start_time),
                )

            current_node_id = next_node_id

        return ExecutionResult(
            success=True,
            steps=steps,
            output=context,
            total_duration_ms=_elapsed_ms(start_time),
        )

    async def _execute_node_with_retry(
        self,
        node: SubAgentNode,
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a node with exponential backoff retries."""
        max_retries = min(node.max_retries, self.max_node_retries)
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            start = time.monotonic()
            try:
                output = await self._execute_node(node, context)
                return StepResult(
                    node_id=node.id,
                    success=True,
                    output=output,
                    duration_ms=_elapsed_ms(start),
                    retries=attempt,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Node '%s' attempt %d/%d failed: %s",
                    node.id, attempt + 1, max_retries + 1, last_error,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        return StepResult(
            node_id=node.id,
            success=False,
            error=last_error,
            duration_ms=_elapsed_ms(start),
            retries=max_retries,
        )

    async def _execute_node(
        self,
        node: SubAgentNode,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single node and return its output dict."""
        if node.node_type == NodeType.FUNCTION:
            return await self._execute_function_node(node, context)
        elif node.node_type == NodeType.ROUTER:
            return self._execute_router_node(node, context)
        elif node.node_type in (NodeType.LLM_GENERATE, NodeType.LLM_TOOL_USE):
            return await self._execute_llm_node(node, context)
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

    async def _execute_function_node(
        self,
        node: SubAgentNode,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a FUNCTION node."""
        if node.function is None:
            raise ValueError(f"Function node '{node.id}' has no callable")

        # Build input from context using input_keys
        input_data = {k: context.get(k) for k in node.input_keys if k in context}
        result = await node.function(input_data)

        # Filter output to declared output_keys
        if node.output_keys:
            return {k: result.get(k) for k in node.output_keys if k in result}
        return result

    def _execute_router_node(
        self,
        node: SubAgentNode,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a ROUTER node — deterministic routing based on context.

        Returns a dict with '_routed_to' key indicating the chosen target.
        The actual edge traversal happens in _resolve_next_node.
        """
        for condition_key, expected_value, target_id in node.routing_rules:
            if condition_key == "default":
                return {"_routed_to": target_id}
            value = context.get(condition_key)
            if expected_value is None:
                if bool(value):
                    return {"_routed_to": target_id}
            elif value == expected_value:
                return {"_routed_to": target_id}

        return {"_routed_to": None}

    async def _execute_llm_node(
        self,
        node: SubAgentNode,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an LLM_GENERATE or LLM_TOOL_USE node."""
        prompt = node.interpolate_prompt(context)

        # Build user message from input_keys
        input_parts = []
        for key in node.input_keys:
            if key in context:
                input_parts.append(f"{key}: {context[key]}")
        user_message = "\n".join(input_parts) if input_parts else "Execute the task."

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=user_message),
        ]

        config = GenerationConfig(temperature=0.7, max_tokens=4096)

        # For tool use nodes, we'd integrate with the agent's tool loop.
        # For now, single-shot generation with tool schema in prompt.
        if node.node_type == NodeType.LLM_TOOL_USE and node.tools:
            # Add tool descriptions to system prompt
            tool_info = self._build_tool_info(node.tools)
            messages[0] = Message(
                role="system",
                content=prompt + "\n\n" + tool_info,
            )

        result: GenerationResult = await self.provider.generate(messages, config)

        # Map output to declared output_keys
        output: dict[str, Any] = {}
        if node.output_keys:
            # First output key gets the full response
            output[node.output_keys[0]] = result.content
        else:
            output["response"] = result.content

        return output

    def _build_tool_info(self, tool_names: list[str]) -> str:
        """Build tool description text for LLM prompt."""
        lines = ["## Available Tools"]
        for name in tool_names:
            tool = self.tool_registry.get(name)
            if tool:
                lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)

    def _resolve_next_node(
        self,
        graph: SubAgentGraph,
        current_node_id: str,
        success: bool,
        context: dict[str, Any],
    ) -> Optional[str]:
        """Determine the next node by evaluating outgoing edges.

        For ROUTER nodes, uses the '_routed_to' context value.
        For other nodes, evaluates edge conditions by priority.
        """
        node = graph.nodes[current_node_id]

        # Router nodes use their routing result directly
        if node.node_type == NodeType.ROUTER:
            routed_to = context.get("_routed_to")
            if routed_to and routed_to in graph.nodes:
                return routed_to
            # Fall through to edge evaluation

        # Evaluate outgoing edges by priority
        for edge in graph.get_outgoing_edges(current_node_id):
            if edge.evaluate(success, context):
                return edge.target

        return None


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return (time.monotonic() - start) * 1000
