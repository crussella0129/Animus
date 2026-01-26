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
]


def create_default_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(ShellTool())
    return registry
