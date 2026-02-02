"""Tools module - Filesystem and shell tools for the agent."""

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
]


def create_default_registry(include_analysis: bool = True) -> ToolRegistry:
    """Create a registry with default tools.

    Args:
        include_analysis: Include analysis tools if tree-sitter is available.

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

    return registry
