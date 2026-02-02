"""Tests for the WebSocket server for IDE integration."""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Skip all tests if websockets not installed
pytest.importorskip("websockets")


class TestWebSocketServer:
    """Test WebSocket server functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        provider = MagicMock()
        provider.is_available = True
        return provider

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        ws = AsyncMock()
        ws.send = AsyncMock()
        return ws

    def test_websockets_available(self):
        """Test websockets package is importable."""
        from src.api.websocket_server import WEBSOCKETS_AVAILABLE
        assert WEBSOCKETS_AVAILABLE

    def test_server_init(self):
        """Test server initialization."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer(host="localhost", port=8765)
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.sessions == {}

    def test_server_init_custom_port(self):
        """Test server with custom port."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer(host="0.0.0.0", port=9999)
        assert server.host == "0.0.0.0"
        assert server.port == 9999

    def test_client_session_creation(self):
        """Test ClientSession dataclass."""
        from src.api.websocket_server import ClientSession

        ws = MagicMock()
        session = ClientSession(id="test-123", websocket=ws)

        assert session.id == "test-123"
        assert session.websocket == ws
        assert session.agent is None
        assert session.workspace_folder is None
        assert session.current_file is None

    def test_client_session_with_context(self):
        """Test ClientSession with context."""
        from src.api.websocket_server import ClientSession

        ws = MagicMock()
        session = ClientSession(
            id="test-456",
            websocket=ws,
            workspace_folder="/path/to/project",
            current_file="/path/to/project/main.py",
        )

        assert session.workspace_folder == "/path/to/project"
        assert session.current_file == "/path/to/project/main.py"

    @pytest.mark.asyncio
    async def test_send_message(self, mock_websocket):
        """Test sending a message to client."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer()
        message = {"type": "test", "data": "hello"}

        await server._send(mock_websocket, message)

        mock_websocket.send.assert_called_once_with(json.dumps(message))

    @pytest.mark.asyncio
    async def test_send_error(self, mock_websocket):
        """Test sending error message."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer()

        await server._send_error(mock_websocket, "Something went wrong")

        mock_websocket.send.assert_called_once()
        call_args = mock_websocket.send.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == "error"
        assert message["error"] == "Something went wrong"

    def test_build_prompt_simple(self):
        """Test building prompt without context."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer()
        prompt = server._build_prompt("How do I fix this?", {})

        assert "How do I fix this?" in prompt

    def test_build_prompt_with_file_context(self):
        """Test building prompt with file context."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer()
        context = {
            "filePath": "/project/src/main.py",
            "language": "python",
        }
        prompt = server._build_prompt("Explain this code", context)

        assert "[File: /project/src/main.py]" in prompt
        assert "[Language: python]" in prompt
        assert "Explain this code" in prompt

    def test_build_prompt_with_selection(self):
        """Test building prompt with code selection."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer()
        context = {
            "filePath": "/project/src/main.py",
            "selection": {
                "text": "def hello():\n    print('hi')",
                "startLine": 10,
                "endLine": 12,
            },
        }
        prompt = server._build_prompt("What does this do?", context)

        assert "Selected code (lines 11-13)" in prompt
        assert "def hello():" in prompt
        assert "What does this do?" in prompt

    def test_build_system_prompt(self):
        """Test building system prompt."""
        from src.api.websocket_server import WebSocketServer, ClientSession

        server = WebSocketServer()
        ws = MagicMock()
        session = ClientSession(id="test", websocket=ws)

        prompt = server._build_system_prompt(session)

        assert "Animus" in prompt
        assert "AI coding assistant" in prompt

    def test_build_system_prompt_with_workspace(self):
        """Test system prompt includes workspace info."""
        from src.api.websocket_server import WebSocketServer, ClientSession

        server = WebSocketServer()
        ws = MagicMock()
        session = ClientSession(
            id="test",
            websocket=ws,
            workspace_folder="/home/user/project",
        )

        prompt = server._build_system_prompt(session)

        assert "Workspace: /home/user/project" in prompt

    def test_build_system_prompt_with_current_file(self):
        """Test system prompt includes current file."""
        from src.api.websocket_server import WebSocketServer, ClientSession

        server = WebSocketServer()
        ws = MagicMock()
        session = ClientSession(
            id="test",
            websocket=ws,
            current_file="/home/user/project/main.py",
        )

        prompt = server._build_system_prompt(session)

        assert "Currently open file: /home/user/project/main.py" in prompt


class TestMessageHandling:
    """Test message handling logic."""

    @pytest.fixture
    def server(self):
        """Create server instance."""
        from src.api.websocket_server import WebSocketServer
        return WebSocketServer()

    @pytest.fixture
    def session(self):
        """Create mock session."""
        from src.api.websocket_server import ClientSession

        ws = AsyncMock()
        ws.send = AsyncMock()
        return ClientSession(id="test-session", websocket=ws)

    @pytest.mark.asyncio
    async def test_handle_ping(self, server, session):
        """Test ping message handling."""
        message = {"type": "ping"}

        await server._handle_message(session, message)

        session.websocket.send.assert_called_once()
        call_args = session.websocket.send.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_context_update(self, server, session):
        """Test context update message."""
        message = {
            "type": "context_update",
            "data": {
                "filePath": "/project/file.py",
                "workspaceFolder": "/project",
            },
        }

        await server._handle_message(session, message)

        assert session.current_file == "/project/file.py"
        assert session.workspace_folder == "/project"

    @pytest.mark.asyncio
    async def test_handle_unknown_type(self, server, session):
        """Test unknown message type sends error."""
        message = {"type": "unknown_type"}

        await server._handle_message(session, message)

        session.websocket.send.assert_called_once()
        call_args = session.websocket.send.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "error"
        assert "unknown_type" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_command_clear_history(self, server, session):
        """Test clear history command."""
        session.agent = MagicMock()
        message = {"type": "command", "content": "clear_history", "data": {}}

        await server._handle_message(session, message)

        assert session.agent is None
        session.websocket.send.assert_called_once()
        call_args = session.websocket.send.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "status"
        assert response["data"]["status"] == "history_cleared"

    @pytest.mark.asyncio
    async def test_handle_command_get_status(self, server, session):
        """Test get status command."""
        session.workspace_folder = "/project"
        session.current_file = "/project/main.py"
        message = {"type": "command", "content": "get_status", "data": {}}

        await server._handle_message(session, message)

        session.websocket.send.assert_called_once()
        call_args = session.websocket.send.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "status"
        assert response["data"]["connected"] is True
        assert response["data"]["workspaceFolder"] == "/project"
        assert response["data"]["currentFile"] == "/project/main.py"

    @pytest.mark.asyncio
    async def test_handle_command_unknown(self, server, session):
        """Test unknown command sends error."""
        message = {"type": "command", "content": "unknown_command", "data": {}}

        await server._handle_message(session, message)

        session.websocket.send.assert_called_once()
        call_args = session.websocket.send.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "error"
        assert "unknown_command" in response["error"].lower()


class TestBroadcast:
    """Test broadcast functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self):
        """Test broadcasting to all sessions."""
        from src.api.websocket_server import WebSocketServer, ClientSession

        server = WebSocketServer()

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        server.sessions = {
            "s1": ClientSession(id="s1", websocket=ws1),
            "s2": ClientSession(id="s2", websocket=ws2),
            "s3": ClientSession(id="s3", websocket=ws3),
        }

        message = {"type": "broadcast", "data": "hello all"}
        server.broadcast(message)

        # Allow tasks to be created
        await asyncio.sleep(0.01)

        # Each websocket should have a send call
        ws1.send.assert_called_once()
        ws2.send.assert_called_once()
        ws3.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_exclude(self):
        """Test broadcasting with exclusion."""
        from src.api.websocket_server import WebSocketServer, ClientSession

        server = WebSocketServer()

        ws1 = AsyncMock()
        ws2 = AsyncMock()

        server.sessions = {
            "s1": ClientSession(id="s1", websocket=ws1),
            "s2": ClientSession(id="s2", websocket=ws2),
        }

        message = {"type": "broadcast", "data": "hello"}
        server.broadcast(message, exclude="s1")

        # Allow tasks to be created
        await asyncio.sleep(0.01)

        # Only s2 should receive the message
        ws1.send.assert_not_called()
        ws2.send.assert_called_once()


class TestIntegration:
    """Integration tests for server functionality."""

    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        """Test server start and stop lifecycle."""
        from src.api.websocket_server import WebSocketServer

        server = WebSocketServer(host="localhost", port=18765)

        # Start server
        await server.start()
        assert server._running is True
        assert server._server is not None

        # Stop server
        await server.stop()
        assert server._running is False

    @pytest.mark.asyncio
    async def test_file_change_handling(self):
        """Test file change notification."""
        from src.api.websocket_server import WebSocketServer, ClientSession
        import tempfile
        import os

        server = WebSocketServer()
        ws = AsyncMock()
        ws.send = AsyncMock()
        session = ClientSession(id="test", websocket=ws)

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Original content\nprint('hello')\n")
            temp_path = f.name

        try:
            tool_call = {
                "name": "write_file",
                "arguments": {
                    "path": temp_path,
                    "content": "# New content\nprint('world')\n",
                },
            }

            await server._handle_file_change(session, tool_call)

            ws.send.assert_called_once()
            call_args = ws.send.call_args[0][0]
            message = json.loads(call_args)

            assert message["type"] == "file_change"
            assert message["data"]["filePath"] == temp_path
            assert "Original content" in message["data"]["originalContent"]
            assert "New content" in message["data"]["newContent"]

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_apply_change_command(self):
        """Test applying a file change."""
        from src.api.websocket_server import WebSocketServer, ClientSession
        import tempfile
        import os

        server = WebSocketServer()
        ws = AsyncMock()
        ws.send = AsyncMock()
        session = ClientSession(id="test", websocket=ws)

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("original")
            temp_path = f.name

        try:
            message = {
                "type": "command",
                "content": "apply_change",
                "data": {
                    "filePath": temp_path,
                    "content": "modified content",
                },
            }

            await server._handle_message(session, message)

            # Verify file was modified
            with open(temp_path, "r") as f:
                content = f.read()
            assert content == "modified content"

            # Verify status message sent
            ws.send.assert_called()
            call_args = ws.send.call_args[0][0]
            response = json.loads(call_args)
            assert response["type"] == "status"
            assert response["data"]["status"] == "change_applied"

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_reject_change_command(self):
        """Test rejecting a file change."""
        from src.api.websocket_server import WebSocketServer, ClientSession

        server = WebSocketServer()
        ws = AsyncMock()
        ws.send = AsyncMock()
        session = ClientSession(id="test", websocket=ws)

        message = {
            "type": "command",
            "content": "reject_change",
            "data": {"filePath": "/some/file.py"},
        }

        await server._handle_message(session, message)

        ws.send.assert_called_once()
        call_args = ws.send.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "status"
        assert response["data"]["status"] == "change_rejected"


class TestExports:
    """Test module exports."""

    def test_websocket_server_export(self):
        """Test WebSocketServer is exported from api module."""
        from src.api import WebSocketServer
        assert WebSocketServer is not None

    def test_run_websocket_server_export(self):
        """Test run_websocket_server is exported."""
        from src.api import run_websocket_server
        assert run_websocket_server is not None
