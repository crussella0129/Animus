"""Tools module - Filesystem and shell tools for the agent."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.core.subagent import SubAgentOrchestrator

from src.tools.base import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolCategory,
    ToolRegistry,
)
from src.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from src.tools.shell import ShellTool
from src.tools.git import GitTool
from src.tools.delegate import (
    DelegateTaskTool,
    DelegateParallelTool,
    create_delegation_tools,
)

__all__ = [
    # Base
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolCategory",
    "ToolRegistry",
    # Filesystem
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    # Shell
    "ShellTool",
    # Git
    "GitTool",
    # Delegation
    "DelegateTaskTool",
    "DelegateParallelTool",
    "create_delegation_tools",
]


def create_default_registry(
    include_analysis: bool = True,
    include_delegation: bool = False,
    orchestrator: Optional["SubAgentOrchestrator"] = None,
) -> ToolRegistry:
    """Create a registry with default tools.

    Args:
        include_analysis: Include analysis tools if tree-sitter is available.
        include_delegation: Include delegation tools for multi-agent support.
        orchestrator: SubAgentOrchestrator for delegation tools.

    Returns:
        ToolRegistry with registered tools.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(ShellTool())
    registry.register(GitTool())

    # Register analysis tools if tree-sitter is available
    if include_analysis:
        try:
            from src.analysis.parser import is_tree_sitter_available
            if is_tree_sitter_available():
                from src.analysis.tools import create_analysis_tools
                for tool in create_analysis_tools():
                    registry.register(tool)
        except ImportError:
            pass  # Analysis module not available

    # Register delegation tools if requested
    if include_delegation:
        for tool in create_delegation_tools(orchestrator):
            registry.register(tool)

    return registry
