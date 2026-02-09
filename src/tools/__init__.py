"""Tools module - Filesystem and shell tools for the agent."""

from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from src.core.subagent import SubAgentOrchestrator

from src.tools.base import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolCategory,
    ToolRegistry,
    DecoratedTool,
    tool,
)
from src.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from src.tools.shell import ShellTool
from src.tools.git import GitTool
from src.tools.git_workflow import (
    GitWorkflowTool,
    ChangeRisk,
    DiffAnalysis,
    create_git_workflow_tool,
)
from src.tools.delegate import (
    DelegateTaskTool,
    DelegateParallelTool,
    create_delegation_tools,
)
from src.tools.web import (
    WebSearchTool,
    WebFetchTool,
    create_web_tools,
    validate_content,
    sanitize_html,
)

__all__ = [
    # Base
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolCategory",
    "ToolRegistry",
    "DecoratedTool",
    "tool",
    # Filesystem
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    # Shell
    "ShellTool",
    # Git
    "GitTool",
    # Git Workflow
    "GitWorkflowTool",
    "ChangeRisk",
    "DiffAnalysis",
    "create_git_workflow_tool",
    # Delegation
    "DelegateTaskTool",
    "DelegateParallelTool",
    "create_delegation_tools",
    # Web
    "WebSearchTool",
    "WebFetchTool",
    "create_web_tools",
    "validate_content",
    "sanitize_html",
]


def create_default_registry(
    include_analysis: bool = True,
    include_delegation: bool = False,
    include_web: bool = True,
    orchestrator: Optional["SubAgentOrchestrator"] = None,
    confirm_callback: Optional[Callable] = None,
) -> ToolRegistry:
    """Create a registry with default tools.

    Args:
        include_analysis: Include analysis tools if tree-sitter is available.
        include_delegation: Include delegation tools for multi-agent support.
        include_web: Include web search and fetch tools.
        orchestrator: SubAgentOrchestrator for delegation tools.
        confirm_callback: Callback for web tool human confirmation.

    Returns:
        ToolRegistry with registered tools.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(ShellTool())
    registry.register(GitTool())
    registry.register(GitWorkflowTool())

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

    # Register web tools if requested
    if include_web:
        for tool in create_web_tools(confirm_callback=confirm_callback):
            registry.register(tool)

    return registry
