"""Tests for MCP (Model Context Protocol) implementation.

Tests cover:
- Protocol message parsing and serialization
- Server request handling
- Client connection management
- Tool registration and execution
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp.protocol import (
    MCPMessage,
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPError,
    MCPServerCapabilities,
    MCPClientCapabilities,
    MCPToolInfo,
    MCPToolCallRequest,
    MCPToolCallResult,
)
from src.mcp.server import MCPServer, MCPTool, MCPToolResult
from src.mcp.client import MCPClient, MCPServerConfig, ConnectedServer, convert_mcp_tools_to_animus


# =============================================================================
# Protocol Tests
# =============================================================================


class TestMCPRequest:
    """Tests for MCPRequest message."""

    def test_create_request(self):
        """Test creating a request."""
        request = MCPRequest(method="tools/list")
        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.id  # Should have an ID

    def test_request_to_dict(self):
        """Test serializing request to dict."""
        request = MCPRequest(id="123", method="ping", params={"foo": "bar"})
        d = request.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == "123"
        assert d["method"] == "ping"
        assert d["params"] == {"foo": "bar"}

    def test_request_to_json(self):
        """Test serializing request to JSON."""
        request = MCPRequest(id="123", method="ping")
        j = request.to_json()
        parsed = json.loads(j)
        assert parsed["id"] == "123"
        assert parsed["method"] == "ping"

    def test_request_from_dict(self):
        """Test deserializing request from dict."""
        d = {"jsonrpc": "2.0", "id": "456", "method": "tools/call", "params": {"name": "test"}}
        request = MCPRequest.from_dict(d)
        assert request.id == "456"
        assert request.method == "tools/call"
        assert request.params == {"name": "test"}

    def test_request_without_params(self):
        """Test request without params omits params key."""
        request = MCPRequest(method="ping")
        d = request.to_dict()
        assert "params" not in d


class TestMCPResponse:
    """Tests for MCPResponse message."""

    def test_success_response(self):
        """Test creating a success response."""
        response = MCPResponse(id="123", result={"tools": []})
        d = response.to_dict()
        assert d["id"] == "123"
        assert d["result"] == {"tools": []}
        assert "error" not in d

    def test_error_response(self):
        """Test creating an error response."""
        error = MCPError(code=-32601, message="Method not found")
        response = MCPResponse(id="123", error=error)
        d = response.to_dict()
        assert d["id"] == "123"
        assert "result" not in d
        assert d["error"]["code"] == -32601
        assert d["error"]["message"] == "Method not found"

    def test_response_from_dict_success(self):
        """Test deserializing success response."""
        d = {"jsonrpc": "2.0", "id": "123", "result": {"data": "test"}}
        response = MCPResponse.from_dict(d)
        assert response.id == "123"
        assert response.result == {"data": "test"}
        assert response.error is None

    def test_response_from_dict_error(self):
        """Test deserializing error response."""
        d = {"jsonrpc": "2.0", "id": "123", "error": {"code": -32600, "message": "Invalid"}}
        response = MCPResponse.from_dict(d)
        assert response.id == "123"
        assert response.result is None
        assert response.error.code == -32600


class TestMCPNotification:
    """Tests for MCPNotification message."""

    def test_notification_no_id(self):
        """Test notification has no ID."""
        notification = MCPNotification(method="notifications/tools/list_changed")
        d = notification.to_dict()
        assert "id" not in d
        assert d["method"] == "notifications/tools/list_changed"

    def test_notification_from_dict(self):
        """Test deserializing notification."""
        d = {"jsonrpc": "2.0", "method": "notify", "params": {"info": "data"}}
        notification = MCPNotification.from_dict(d)
        assert notification.method == "notify"
        assert notification.params == {"info": "data"}


class TestMCPError:
    """Tests for MCPError."""

    def test_standard_error_codes(self):
        """Test standard JSON-RPC error codes."""
        assert MCPError.PARSE_ERROR == -32700
        assert MCPError.INVALID_REQUEST == -32600
        assert MCPError.METHOD_NOT_FOUND == -32601
        assert MCPError.INVALID_PARAMS == -32602
        assert MCPError.INTERNAL_ERROR == -32603

    def test_error_factory_methods(self):
        """Test error factory methods."""
        e = MCPError.parse_error("Bad JSON")
        assert e.code == -32700
        assert e.message == "Bad JSON"

        e = MCPError.method_not_found("unknown_method")
        assert e.code == -32601

        e = MCPError.invalid_params()
        assert e.code == -32602

        e = MCPError.internal_error("Crash")
        assert e.code == -32603

    def test_error_to_dict(self):
        """Test serializing error."""
        e = MCPError(code=-32000, message="Custom error", data={"details": "info"})
        d = e.to_dict()
        assert d["code"] == -32000
        assert d["message"] == "Custom error"
        assert d["data"] == {"details": "info"}


class TestMCPMessage:
    """Tests for MCPMessage factory."""

    def test_from_dict_request(self):
        """Test parsing request from dict."""
        d = {"jsonrpc": "2.0", "id": "1", "method": "ping"}
        msg = MCPMessage.from_dict(d)
        assert isinstance(msg, MCPRequest)
        assert msg.method == "ping"

    def test_from_dict_notification(self):
        """Test parsing notification from dict."""
        d = {"jsonrpc": "2.0", "method": "notify"}
        msg = MCPMessage.from_dict(d)
        assert isinstance(msg, MCPNotification)

    def test_from_dict_response_success(self):
        """Test parsing success response from dict."""
        d = {"jsonrpc": "2.0", "id": "1", "result": {}}
        msg = MCPMessage.from_dict(d)
        assert isinstance(msg, MCPResponse)

    def test_from_dict_response_error(self):
        """Test parsing error response from dict."""
        d = {"jsonrpc": "2.0", "id": "1", "error": {"code": -1, "message": "err"}}
        msg = MCPMessage.from_dict(d)
        assert isinstance(msg, MCPResponse)
        assert msg.error is not None

    def test_from_dict_invalid(self):
        """Test parsing invalid message raises."""
        with pytest.raises(ValueError):
            MCPMessage.from_dict({"jsonrpc": "2.0"})


class TestMCPCapabilities:
    """Tests for capability objects."""

    def test_server_capabilities_default(self):
        """Test default server capabilities."""
        caps = MCPServerCapabilities()
        d = caps.to_dict()
        assert "tools" in d
        assert "resources" not in d
        assert "prompts" not in d

    def test_server_capabilities_custom(self):
        """Test custom server capabilities."""
        caps = MCPServerCapabilities(tools=True, resources=True, prompts=True, logging=True)
        d = caps.to_dict()
        assert "tools" in d
        assert "resources" in d
        assert "prompts" in d
        assert "logging" in d

    def test_client_capabilities(self):
        """Test client capabilities."""
        caps = MCPClientCapabilities(roots=True, sampling=True)
        d = caps.to_dict()
        assert "roots" in d
        assert d["roots"]["listChanged"] is True
        assert "sampling" in d


class TestMCPToolInfo:
    """Tests for MCPToolInfo."""

    def test_tool_info_to_dict(self):
        """Test serializing tool info."""
        info = MCPToolInfo(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        d = info.to_dict()
        assert d["name"] == "read_file"
        assert d["description"] == "Read a file"
        assert "properties" in d["inputSchema"]


class TestMCPToolCallResult:
    """Tests for MCPToolCallResult."""

    def test_text_result(self):
        """Test creating text result."""
        result = MCPToolCallResult.text("Hello, World!")
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello, World!"
        assert result.isError is False

    def test_error_result(self):
        """Test creating error result."""
        result = MCPToolCallResult.error("Something failed")
        assert result.content[0]["text"] == "Something failed"
        assert result.isError is True

    def test_result_to_dict(self):
        """Test serializing result."""
        result = MCPToolCallResult(
            content=[{"type": "text", "text": "data"}],
            isError=False,
        )
        d = result.to_dict()
        assert d["content"] == [{"type": "text", "text": "data"}]
        assert d["isError"] is False


# =============================================================================
# Server Tests
# =============================================================================


class TestMCPServer:
    """Tests for MCPServer."""

    @pytest.fixture
    def server(self):
        """Create a test server."""
        return MCPServer(name="test-server", version="0.1.0")

    def test_server_creation(self, server):
        """Test creating a server."""
        assert server.name == "test-server"
        assert server.version == "0.1.0"
        assert server.capabilities.tools is True

    def test_register_tool(self, server):
        """Test registering a tool."""
        async def handler(args):
            return MCPToolResult(content="done")

        tool = MCPTool(
            name="my_tool",
            description="My tool",
            input_schema={"type": "object"},
            handler=handler,
        )
        server.register_tool(tool)
        assert "my_tool" in server._tools

    def test_unregister_tool(self, server):
        """Test unregistering a tool."""
        async def handler(args):
            return MCPToolResult(content="done")

        tool = MCPTool(name="temp", description="", input_schema={}, handler=handler)
        server.register_tool(tool)
        assert server.unregister_tool("temp") is True
        assert "temp" not in server._tools
        assert server.unregister_tool("nonexistent") is False

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        """Test handling initialize request."""
        request = MCPRequest(id="1", method="initialize", params={})
        response = await server.handle_request(request)
        assert response.error is None
        assert response.result["protocolVersion"] == "2024-11-05"
        assert response.result["serverInfo"]["name"] == "test-server"
        assert server._initialized is True

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        """Test handling tools/list request."""
        async def handler(args):
            return MCPToolResult(content="done")

        server.register_tool(MCPTool(
            name="tool1",
            description="Tool 1",
            input_schema={"type": "object"},
            handler=handler,
        ))

        request = MCPRequest(id="1", method="tools/list")
        response = await server.handle_request(request)
        assert response.error is None
        assert len(response.result["tools"]) == 1
        assert response.result["tools"][0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_handle_tools_call_success(self, server):
        """Test handling successful tools/call request."""
        async def handler(args):
            return MCPToolResult(content=f"Received: {args.get('value')}")

        server.register_tool(MCPTool(
            name="echo",
            description="Echo tool",
            input_schema={"type": "object"},
            handler=handler,
        ))

        request = MCPRequest(
            id="1",
            method="tools/call",
            params={"name": "echo", "arguments": {"value": "test"}},
        )
        response = await server.handle_request(request)
        assert response.error is None
        assert response.result["content"][0]["text"] == "Received: test"
        assert response.result["isError"] is False

    @pytest.mark.asyncio
    async def test_handle_tools_call_unknown_tool(self, server):
        """Test handling tools/call for unknown tool."""
        request = MCPRequest(
            id="1",
            method="tools/call",
            params={"name": "unknown_tool", "arguments": {}},
        )
        response = await server.handle_request(request)
        assert response.result["isError"] is True
        assert "Unknown tool" in response.result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_error(self, server):
        """Test handling tools/call when tool raises."""
        async def handler(args):
            raise ValueError("Tool failed")

        server.register_tool(MCPTool(
            name="failing",
            description="Failing tool",
            input_schema={},
            handler=handler,
        ))

        request = MCPRequest(
            id="1",
            method="tools/call",
            params={"name": "failing", "arguments": {}},
        )
        response = await server.handle_request(request)
        assert response.result["isError"] is True
        assert "Tool failed" in response.result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_ping(self, server):
        """Test handling ping request."""
        request = MCPRequest(id="1", method="ping")
        response = await server.handle_request(request)
        assert response.error is None
        assert response.result == {}

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        """Test handling unknown method."""
        request = MCPRequest(id="1", method="unknown/method")
        response = await server.handle_request(request)
        assert response.error is not None
        assert response.error.code == MCPError.METHOD_NOT_FOUND


class TestMCPToolResult:
    """Tests for MCPToolResult."""

    def test_tool_result_success(self):
        """Test success tool result."""
        result = MCPToolResult(content="Success!")
        mcp_result = result.to_mcp_result()
        assert mcp_result.isError is False
        assert mcp_result.content[0]["text"] == "Success!"

    def test_tool_result_error(self):
        """Test error tool result."""
        result = MCPToolResult(content="Failed!", is_error=True)
        mcp_result = result.to_mcp_result()
        assert mcp_result.isError is True


# =============================================================================
# Client Tests
# =============================================================================


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_stdio_config(self):
        """Test stdio config."""
        config = MCPServerConfig(
            name="test",
            command="python",
            args=["-m", "mcp_server"],
        )
        assert config.command == "python"
        assert config.args == ["-m", "mcp_server"]
        assert config.url is None

    def test_http_config(self):
        """Test HTTP config."""
        config = MCPServerConfig(
            name="test",
            url="http://localhost:8338",
        )
        assert config.command is None
        assert config.url == "http://localhost:8338"

    def test_config_with_env(self):
        """Test config with environment variables."""
        config = MCPServerConfig(
            name="test",
            command="node",
            env={"API_KEY": "secret"},
        )
        assert config.env["API_KEY"] == "secret"


class TestMCPClient:
    """Tests for MCPClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return MCPClient()

    def test_client_creation(self, client):
        """Test creating a client."""
        assert len(client._servers) == 0
        assert len(client._processes) == 0

    def test_list_tools_empty(self, client):
        """Test listing tools when no servers connected."""
        tools = client.list_tools()
        assert tools == []

    def test_list_servers_empty(self, client):
        """Test listing servers when none connected."""
        servers = client.list_servers()
        assert servers == []

    def test_get_server_not_found(self, client):
        """Test getting non-existent server."""
        server = client.get_server("nonexistent")
        assert server is None


class TestConvertMCPToolsToAnimus:
    """Tests for converting MCP tools to Animus format."""

    def test_convert_single_tool(self):
        """Test converting a single tool."""
        tools = [MCPToolInfo(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        )]
        result = convert_mcp_tools_to_animus(tools, "server1")
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "server1.read_file"
        assert result[0]["function"]["description"] == "Read a file"

    def test_convert_multiple_tools(self):
        """Test converting multiple tools."""
        tools = [
            MCPToolInfo(name="tool1", description="Tool 1", inputSchema={}),
            MCPToolInfo(name="tool2", description="Tool 2", inputSchema={}),
        ]
        result = convert_mcp_tools_to_animus(tools, "srv")
        assert len(result) == 2
        assert result[0]["function"]["name"] == "srv.tool1"
        assert result[1]["function"]["name"] == "srv.tool2"

    def test_convert_empty_list(self):
        """Test converting empty tool list."""
        result = convert_mcp_tools_to_animus([], "server")
        assert result == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestMCPIntegration:
    """Integration tests for MCP server and client."""

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self):
        """Test full request-response cycle."""
        server = MCPServer(name="integration-test")

        async def add_handler(args):
            a = args.get("a", 0)
            b = args.get("b", 0)
            return MCPToolResult(content=str(a + b))

        server.register_tool(MCPTool(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
            },
            handler=add_handler,
        ))

        # Initialize
        init_req = MCPRequest(method="initialize", params={})
        init_resp = await server.handle_request(init_req)
        assert init_resp.error is None

        # List tools
        list_req = MCPRequest(method="tools/list")
        list_resp = await server.handle_request(list_req)
        assert len(list_resp.result["tools"]) == 1
        assert list_resp.result["tools"][0]["name"] == "add"

        # Call tool
        call_req = MCPRequest(
            method="tools/call",
            params={"name": "add", "arguments": {"a": 5, "b": 3}},
        )
        call_resp = await server.handle_request(call_req)
        assert call_resp.result["content"][0]["text"] == "8"

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = MCPRequest(
            id="test-id",
            method="tools/call",
            params={"name": "test", "arguments": {"key": "value"}},
        )
        json_str = original.to_json()
        parsed = json.loads(json_str)
        restored = MCPRequest.from_dict(parsed)

        assert restored.id == original.id
        assert restored.method == original.method
        assert restored.params == original.params


# =============================================================================
# Edge Cases
# =============================================================================


class TestMCPEdgeCases:
    """Tests for edge cases."""

    def test_empty_params(self):
        """Test request with empty params."""
        request = MCPRequest(method="ping", params={})
        d = request.to_dict()
        # Empty params is treated same as None - not included
        assert "params" not in d

    def test_null_params(self):
        """Test request with None params."""
        request = MCPRequest(method="ping", params=None)
        d = request.to_dict()
        # None params should not be included
        assert "params" not in d

    def test_error_with_data(self):
        """Test error with additional data."""
        error = MCPError(
            code=-32000,
            message="Custom error",
            data={"stack": "traceback...", "code": "E001"},
        )
        d = error.to_dict()
        assert d["data"]["stack"] == "traceback..."
        assert d["data"]["code"] == "E001"

    def test_error_without_data(self):
        """Test error without additional data."""
        error = MCPError(code=-32000, message="Simple error")
        d = error.to_dict()
        assert "data" not in d

    @pytest.mark.asyncio
    async def test_server_exception_handling(self):
        """Test server handles exceptions gracefully."""
        server = MCPServer()

        async def broken_handler(args):
            raise RuntimeError("Unexpected crash")

        server.register_tool(MCPTool(
            name="broken",
            description="Broken tool",
            input_schema={},
            handler=broken_handler,
        ))

        request = MCPRequest(
            method="tools/call",
            params={"name": "broken", "arguments": {}},
        )
        response = await server.handle_request(request)
        # Should return error result, not crash
        assert response.result["isError"] is True

    def test_tool_info_special_characters(self):
        """Test tool info with special characters."""
        info = MCPToolInfo(
            name="tool-with-dashes",
            description="Tool with \"quotes\" and 'apostrophes'",
            inputSchema={"type": "object"},
        )
        d = info.to_dict()
        # Should serialize without issues
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["name"] == "tool-with-dashes"
