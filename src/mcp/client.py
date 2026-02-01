"""MCP Client for connecting to external MCP servers."""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.mcp.protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPToolInfo,
    MCPToolCallResult,
)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    command: Optional[str] = None  # For stdio transport
    args: list[str] = field(default_factory=list)
    url: Optional[str] = None  # For HTTP transport
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ConnectedServer:
    """A connected MCP server."""
    config: MCPServerConfig
    tools: list[MCPToolInfo] = field(default_factory=list)
    protocol_version: str = ""
    server_name: str = ""
    server_version: str = ""


class MCPClient:
    """Client for connecting to MCP servers."""

    def __init__(self):
        """Initialize MCP client."""
        self._servers: dict[str, ConnectedServer] = {}
        self._processes: dict[str, subprocess.Popen] = {}

    async def connect(self, config: MCPServerConfig) -> ConnectedServer:
        """Connect to an MCP server.

        Args:
            config: Server configuration

        Returns:
            Connected server info
        """
        if config.command:
            return await self._connect_stdio(config)
        elif config.url:
            return await self._connect_http(config)
        else:
            raise ValueError("Server config must have either 'command' or 'url'")

    async def _connect_stdio(self, config: MCPServerConfig) -> ConnectedServer:
        """Connect to a server via stdio."""
        # Start the server process
        cmd = [config.command] + config.args
        env = {**dict(subprocess.os.environ), **config.env}

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._processes[config.name] = process

        # Initialize the connection
        init_request = MCPRequest(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                },
                "clientInfo": {
                    "name": "animus",
                    "version": "1.0.0",
                },
            },
        )

        response = await self._send_stdio(config.name, init_request)
        if response.error:
            raise RuntimeError(f"Initialize failed: {response.error.message}")

        result = response.result or {}

        # List tools
        tools_request = MCPRequest(method="tools/list")
        tools_response = await self._send_stdio(config.name, tools_request)

        tools = []
        if tools_response.result:
            for tool_data in tools_response.result.get("tools", []):
                tools.append(MCPToolInfo(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    inputSchema=tool_data.get("inputSchema", {}),
                ))

        connected = ConnectedServer(
            config=config,
            tools=tools,
            protocol_version=result.get("protocolVersion", ""),
            server_name=result.get("serverInfo", {}).get("name", ""),
            server_version=result.get("serverInfo", {}).get("version", ""),
        )
        self._servers[config.name] = connected

        return connected

    async def _connect_http(self, config: MCPServerConfig) -> ConnectedServer:
        """Connect to a server via HTTP."""
        import urllib.request

        # Initialize the connection
        init_request = MCPRequest(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "animus",
                    "version": "1.0.0",
                },
            },
        )

        response = await self._send_http(config.url, init_request)
        if response.error:
            raise RuntimeError(f"Initialize failed: {response.error.message}")

        result = response.result or {}

        # List tools
        tools_request = MCPRequest(method="tools/list")
        tools_response = await self._send_http(config.url, tools_request)

        tools = []
        if tools_response.result:
            for tool_data in tools_response.result.get("tools", []):
                tools.append(MCPToolInfo(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    inputSchema=tool_data.get("inputSchema", {}),
                ))

        connected = ConnectedServer(
            config=config,
            tools=tools,
            protocol_version=result.get("protocolVersion", ""),
            server_name=result.get("serverInfo", {}).get("name", ""),
            server_version=result.get("serverInfo", {}).get("version", ""),
        )
        self._servers[config.name] = connected

        return connected

    async def _send_stdio(self, server_name: str, request: MCPRequest) -> MCPResponse:
        """Send a request via stdio."""
        process = self._processes.get(server_name)
        if not process:
            raise RuntimeError(f"Server not connected: {server_name}")

        # Write request
        request_json = request.to_json() + "\n"
        process.stdin.write(request_json.encode())
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline().decode().strip()
        if not response_line:
            raise RuntimeError("No response from server")

        data = json.loads(response_line)
        return MCPResponse.from_dict(data)

    async def _send_http(self, url: str, request: MCPRequest) -> MCPResponse:
        """Send a request via HTTP."""
        import urllib.request

        req = urllib.request.Request(
            url,
            data=request.to_json().encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return MCPResponse.from_dict(data)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> MCPToolCallResult:
        """Call a tool on a connected server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        server = self._servers.get(server_name)
        if not server:
            raise RuntimeError(f"Server not connected: {server_name}")

        request = MCPRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )

        if server.config.command:
            response = await self._send_stdio(server_name, request)
        elif server.config.url:
            response = await self._send_http(server.config.url, request)
        else:
            raise RuntimeError("Invalid server config")

        if response.error:
            return MCPToolCallResult.error(response.error.message)

        result = response.result or {}
        return MCPToolCallResult(
            content=result.get("content", []),
            isError=result.get("isError", False),
        )

    def list_tools(self, server_name: Optional[str] = None) -> list[MCPToolInfo]:
        """List available tools.

        Args:
            server_name: Optional server name to filter by

        Returns:
            List of tool info
        """
        tools = []
        for name, server in self._servers.items():
            if server_name and name != server_name:
                continue
            tools.extend(server.tools)
        return tools

    def disconnect(self, server_name: str) -> bool:
        """Disconnect from a server.

        Args:
            server_name: Name of the server

        Returns:
            True if disconnected
        """
        if server_name in self._servers:
            del self._servers[server_name]

        if server_name in self._processes:
            process = self._processes[server_name]
            process.terminate()
            process.wait()
            del self._processes[server_name]

        return True

    def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._servers.keys()):
            self.disconnect(name)

    def get_server(self, name: str) -> Optional[ConnectedServer]:
        """Get a connected server by name."""
        return self._servers.get(name)

    def list_servers(self) -> list[ConnectedServer]:
        """List all connected servers."""
        return list(self._servers.values())


def convert_mcp_tools_to_animus(tools: list[MCPToolInfo], server_name: str) -> list[dict]:
    """Convert MCP tool info to Animus tool format.

    Args:
        tools: MCP tool info list
        server_name: Name of the source server

    Returns:
        List of Animus-compatible tool schemas
    """
    animus_tools = []
    for tool in tools:
        animus_tools.append({
            "type": "function",
            "function": {
                "name": f"{server_name}.{tool.name}",
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        })
    return animus_tools
