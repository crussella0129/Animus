"""Model Context Protocol (MCP) integration for Animus.

MCP enables:
- Exposing Animus tools to external clients
- Connecting to external MCP servers
- Standard protocol for tool communication
"""

from src.mcp.server import MCPServer, MCPTool, MCPToolResult
from src.mcp.client import MCPClient
from src.mcp.protocol import (
    MCPMessage,
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPError,
)

__all__ = [
    "MCPServer",
    "MCPTool",
    "MCPToolResult",
    "MCPClient",
    "MCPMessage",
    "MCPRequest",
    "MCPResponse",
    "MCPNotification",
    "MCPError",
]
