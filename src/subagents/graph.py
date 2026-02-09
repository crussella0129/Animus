"""Graph container for sub-agent execution workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.subagents.goal import SubAgentGoal
from src.subagents.node import SubAgentNode
from src.subagents.edge import SubAgentEdge


class GraphValidationError(Exception):
    """Raised when a graph fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Graph validation failed: {'; '.join(errors)}")


@dataclass
class SubAgentGraph:
    """A directed graph defining a sub-agent workflow.

    The graph connects nodes (steps) via edges (flow control).
    Execution starts at the entry node and follows edges until
    a terminal node is reached or a pause node suspends execution.
    """
    id: str
    goal: SubAgentGoal
    nodes: dict[str, SubAgentNode] = field(default_factory=dict)
    edges: list[SubAgentEdge] = field(default_factory=list)

    # Node id where execution starts
    entry_node: str = ""

    # Node ids where execution ends successfully
    terminal_nodes: list[str] = field(default_factory=list)

    # Node ids where execution pauses for user input
    pause_nodes: list[str] = field(default_factory=list)

    def add_node(self, node: SubAgentNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: SubAgentEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[SubAgentNode]:
        """Get a node by id."""
        return self.nodes.get(node_id)

    def get_outgoing_edges(self, node_id: str) -> list[SubAgentEdge]:
        """Get all edges leaving a node, sorted by priority."""
        edges = [e for e in self.edges if e.source == node_id]
        return sorted(edges, key=lambda e: e.priority)

    def get_incoming_edges(self, node_id: str) -> list[SubAgentEdge]:
        """Get all edges entering a node."""
        return [e for e in self.edges if e.target == node_id]

    def validate(self, strict: bool = True) -> list[str]:
        """Validate the graph structure.

        Args:
            strict: If True, raises GraphValidationError on errors.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []

        # Validate goal
        errors.extend(self.goal.validate())

        # Must have nodes
        if not self.nodes:
            errors.append("Graph must have at least one node")

        # Entry node must exist
        if not self.entry_node:
            errors.append("Graph must have an entry_node")
        elif self.entry_node not in self.nodes:
            errors.append(f"Entry node '{self.entry_node}' not found in graph nodes")

        # Terminal nodes must exist
        for t in self.terminal_nodes:
            if t not in self.nodes:
                errors.append(f"Terminal node '{t}' not found in graph nodes")

        # Pause nodes must exist
        for p in self.pause_nodes:
            if p not in self.nodes:
                errors.append(f"Pause node '{p}' not found in graph nodes")

        # Validate each node
        for node in self.nodes.values():
            errors.extend(node.validate())

        # Validate each edge
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            errors.extend(edge.validate())
            if edge.source not in node_ids:
                errors.append(
                    f"Edge '{edge.id}' references unknown source node '{edge.source}'"
                )
            if edge.target not in node_ids:
                errors.append(
                    f"Edge '{edge.id}' references unknown target node '{edge.target}'"
                )

        # Every non-terminal, non-pause node should have at least one outgoing edge
        for node_id in self.nodes:
            if node_id in self.terminal_nodes or node_id in self.pause_nodes:
                continue
            if not self.get_outgoing_edges(node_id):
                errors.append(
                    f"Node '{node_id}' has no outgoing edges and is not terminal or pause"
                )

        # Check for unreachable nodes (except entry)
        reachable = self._find_reachable(self.entry_node) if self.entry_node in self.nodes else set()
        for node_id in self.nodes:
            if node_id != self.entry_node and node_id not in reachable:
                errors.append(f"Node '{node_id}' is unreachable from entry node")

        if strict and errors:
            raise GraphValidationError(errors)

        return errors

    def _find_reachable(self, start: str) -> set[str]:
        """Find all nodes reachable from a starting node via BFS."""
        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in self.get_outgoing_edges(current):
                if edge.target not in visited:
                    queue.append(edge.target)
        return visited
