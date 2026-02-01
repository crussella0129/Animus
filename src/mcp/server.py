"""MCP Server implementation for Animus.

Exposes Animus tools via the Model Context Protocol.
Supports stdio and HTTP transports.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Awaitable

from src.mcp.protocol import (
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPError,
    MCPServerCapabilities,
    MCPToolInfo,
    MCPToolCallRequest,
    MCPToolCallResult,
)


@dataclass
class MCPTool:
    """A tool exposed via MCP."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Awaitable[MCPToolResult]]


@dataclass
class MCPToolResult:
    """Result from a tool execution."""
    content: str
    is_error: bool = False

    def to_mcp_result(self) -> MCPToolCallResult:
        """Convert to MCP protocol result."""
        return MCPToolCallResult(
            content=[{"type": "text", "text": self.content}],
            isError=self.is_error,
        )


class MCPServer:
    """MCP Server that exposes tools."""

    def __init__(
        self,
        name: str = "animus",
        version: str = "1.0.0",
        capabilities: Optional[MCPServerCapabilities] = None,
    ):
        """Initialize MCP server.

        Args:
            name: Server name
            version: Server version
            capabilities: Server capabilities
        """
        self.name = name
        self.version = version
        self.capabilities = capabilities or MCPServerCapabilities()
        self._tools: dict[str, MCPTool] = {}
        self._initialized = False

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the server."""
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def register_animus_tools(self) -> None:
        """Register default Animus tools."""
        from src.tools import get_default_registry

        registry = get_default_registry()

        for tool in registry.list_tools():
            schema = tool.get_schema()

            async def handler(args: dict, t=tool) -> MCPToolResult:
                try:
                    result = await t.execute(**args)
                    return MCPToolResult(
                        content=result.output if result.success else result.error,
                        is_error=not result.success,
                    )
                except Exception as e:
                    return MCPToolResult(content=str(e), is_error=True)

            self.register_tool(MCPTool(
                name=tool.name,
                description=tool.description,
                input_schema=schema.get("parameters", {}),
                handler=handler,
            ))

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request.

        Args:
            request: The MCP request

        Returns:
            MCP response
        """
        method = request.method
        params = request.params or {}

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            elif method == "ping":
                result = {}
            else:
                return MCPResponse(
                    id=request.id,
                    error=MCPError.method_not_found(f"Unknown method: {method}"),
                )

            return MCPResponse(id=request.id, result=result)

        except Exception as e:
            return MCPResponse(
                id=request.id,
                error=MCPError.internal_error(str(e)),
            )

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }

    async def _handle_tools_list(self) -> dict:
        """Handle tools/list request."""
        tools = []
        for tool in self._tools.values():
            tools.append(MCPToolInfo(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            ).to_dict())
        return {"tools": tools}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request."""
        call = MCPToolCallRequest.from_dict(params)

        if call.name not in self._tools:
            return MCPToolCallResult.error(f"Unknown tool: {call.name}").to_dict()

        tool = self._tools[call.name]
        try:
            result = await tool.handler(call.arguments)
            return result.to_mcp_result().to_dict()
        except Exception as e:
            return MCPToolCallResult.error(str(e)).to_dict()

    async def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        print(f"MCP Server '{self.name}' v{self.version} starting on stdio...", file=sys.stderr)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

        while True:
            try:
                # Read a line (JSON-RPC message)
                line = await reader.readline()
                if not line:
                    break

                line = line.decode().strip()
                if not line:
                    continue

                # Parse request
                try:
                    data = json.loads(line)
                    request = MCPRequest.from_dict(data)
                except json.JSONDecodeError as e:
                    response = MCPResponse(
                        id="",
                        error=MCPError.parse_error(str(e)),
                    )
                    writer.write((response.to_json() + "\n").encode())
                    await writer.drain()
                    continue

                # Handle request
                response = await self.handle_request(request)

                # Send response
                writer.write((response.to_json() + "\n").encode())
                await writer.drain()

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

    async def run_http(self, host: str = "localhost", port: int = 8338) -> None:
        """Run the server using HTTP transport."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        server_self = self

        class MCPHTTPHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                try:
                    data = json.loads(body)
                    request = MCPRequest.from_dict(data)
                    response = asyncio.run(server_self.handle_request(request))

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(response.to_json().encode())

                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    error_response = MCPResponse(
                        id="",
                        error=MCPError.internal_error(str(e)),
                    )
                    self.wfile.write(error_response.to_json().encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer((host, port), MCPHTTPHandler)
        print(f"MCP Server '{self.name}' v{self.version} running on http://{host}:{port}")
        server.serve_forever()


def create_animus_mcp_server() -> MCPServer:
    """Create an MCP server with Animus tools registered."""
    server = MCPServer(
        name="animus",
        version="1.0.0",
        capabilities=MCPServerCapabilities(
            tools=True,
            resources=False,
            prompts=False,
            logging=False,
        ),
    )
    server.register_animus_tools()
    return server
