"""WebSocket server for IDE integration.

Provides real-time bidirectional communication for:
- Chat with streaming responses
- File change proposals
- Tool execution notifications
- Status updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import websockets
    from websockets.asyncio.server import ServerConnection
    WEBSOCKETS_AVAILABLE = True
    WebSocketConnection = ServerConnection
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketConnection = Any

from src.core import Agent, AgentConfig
from src.core.config import ConfigManager
from src.llm import get_default_provider

logger = logging.getLogger(__name__)


@dataclass
class ClientSession:
    """Represents a connected client session."""
    id: str
    websocket: WebSocketConnection
    agent: Optional[Agent] = None
    workspace_folder: Optional[str] = None
    current_file: Optional[str] = None
    created_at: float = field(default_factory=lambda: __import__('time').time())


class WebSocketServer:
    """WebSocket server for IDE integration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
    ):
        """Initialize WebSocket server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required: pip install websockets")

        self.host = host
        self.port = port
        self.sessions: dict[str, ClientSession] = {}
        self._server = None
        self._running = False

    async def start(self):
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket: WebSocketConnection, path: str = ""):
        """Handle a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            path: The connection path (unused)
        """
        session_id = str(uuid.uuid4())
        session = ClientSession(id=session_id, websocket=websocket)
        self.sessions[session_id] = session

        logger.info(f"Client connected: {session_id}")

        try:
            # Send connection confirmation
            await self._send(websocket, {
                "type": "connected",
                "data": {"sessionId": session_id}
            })

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(session, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                except Exception as e:
                    logger.exception(f"Error handling message: {e}")
                    await self._send_error(websocket, str(e))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {session_id}")
        finally:
            # Cleanup
            if session.agent:
                # Any cleanup needed for the agent
                pass
            del self.sessions[session_id]

    async def _handle_message(self, session: ClientSession, message: dict):
        """Handle an incoming message from a client.

        Args:
            session: The client session
            message: The parsed message
        """
        msg_type = message.get("type", "")
        data = message.get("data", {})
        content = message.get("content", "")

        if msg_type == "chat":
            await self._handle_chat(session, content, data)

        elif msg_type == "command":
            await self._handle_command(session, content, data)

        elif msg_type == "context_update":
            # Update session context
            session.current_file = data.get("filePath")
            session.workspace_folder = data.get("workspaceFolder")

        elif msg_type == "cancel":
            # Cancel current operation
            if session.agent:
                session.agent.cancel()

        elif msg_type == "ping":
            await self._send(session.websocket, {"type": "pong"})

        else:
            await self._send_error(session.websocket, f"Unknown message type: {msg_type}")

    async def _handle_chat(self, session: ClientSession, content: str, data: dict):
        """Handle a chat message.

        Args:
            session: The client session
            content: The chat message content
            data: Additional data (context, streaming preference, etc.)
        """
        if not content.strip():
            return

        # Initialize agent if needed
        if not session.agent:
            session.agent = await self._create_agent(session)

        streaming = data.get("streaming", True)
        context = data.get("context", {})

        # Build context-aware prompt
        prompt = self._build_prompt(content, context)

        try:
            if streaming:
                # Stream the response token by token
                full_response = ""
                async for turn in session.agent.run(prompt):
                    if turn.content:
                        # Send incremental token
                        new_content = turn.content[len(full_response):]
                        if new_content:
                            await self._send(session.websocket, {
                                "type": "token",
                                "content": new_content
                            })
                            full_response = turn.content

                    # Send tool call notifications
                    if turn.tool_calls:
                        for tool_call in turn.tool_calls:
                            await self._send(session.websocket, {
                                "type": "tool_call",
                                "data": {
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("arguments"),
                                }
                            })

                            # Check for file changes
                            if tool_call.get("name") in ["write_file", "edit_file", "patch"]:
                                await self._handle_file_change(session, tool_call)

                # Send completion
                await self._send(session.websocket, {
                    "type": "response",
                    "content": full_response
                })

            else:
                # Non-streaming response
                turns = []
                async for turn in session.agent.run(prompt):
                    turns.append(turn)

                final_turn = turns[-1] if turns else None
                await self._send(session.websocket, {
                    "type": "response",
                    "content": final_turn.content if final_turn else ""
                })

        except Exception as e:
            logger.exception(f"Error in chat: {e}")
            await self._send(session.websocket, {
                "type": "error",
                "error": str(e)
            })

    async def _handle_command(self, session: ClientSession, command: str, data: dict):
        """Handle a command message.

        Args:
            session: The client session
            command: The command name
            data: Command arguments
        """
        if command == "clear_history":
            # Reset agent conversation history
            session.agent = None
            await self._send(session.websocket, {
                "type": "status",
                "data": {"status": "history_cleared"}
            })

        elif command == "get_status":
            await self._send(session.websocket, {
                "type": "status",
                "data": {
                    "connected": True,
                    "hasAgent": session.agent is not None,
                    "workspaceFolder": session.workspace_folder,
                    "currentFile": session.current_file,
                }
            })

        elif command == "apply_change":
            # Apply a proposed file change
            file_path = data.get("filePath")
            new_content = data.get("content")
            if file_path and new_content is not None:
                try:
                    Path(file_path).write_text(new_content, encoding="utf-8")
                    await self._send(session.websocket, {
                        "type": "status",
                        "data": {"status": "change_applied", "filePath": file_path}
                    })
                except Exception as e:
                    await self._send_error(session.websocket, f"Failed to apply change: {e}")

        elif command == "reject_change":
            # Just acknowledge the rejection
            await self._send(session.websocket, {
                "type": "status",
                "data": {"status": "change_rejected", "filePath": data.get("filePath")}
            })

        else:
            await self._send_error(session.websocket, f"Unknown command: {command}")

    async def _handle_file_change(self, session: ClientSession, tool_call: dict):
        """Handle a file change from a tool call.

        Args:
            session: The client session
            tool_call: The tool call data
        """
        args = tool_call.get("arguments", {})
        file_path = args.get("path") or args.get("file_path")
        new_content = args.get("content") or args.get("new_content")

        if not file_path:
            return

        # Read original content
        original_content = ""
        path = Path(file_path)
        if path.exists():
            try:
                original_content = path.read_text(encoding="utf-8")
            except Exception:
                pass

        # Send file change proposal
        await self._send(session.websocket, {
            "type": "file_change",
            "data": {
                "filePath": str(file_path),
                "originalContent": original_content,
                "newContent": new_content or "",
                "description": f"Proposed change to {path.name}",
            }
        })

    async def _create_agent(self, session: ClientSession) -> Agent:
        """Create a new agent for the session.

        Args:
            session: The client session

        Returns:
            Configured Agent instance
        """
        config = ConfigManager().config
        provider = get_default_provider(config)

        # Build system prompt with IDE context
        system_prompt = self._build_system_prompt(session)

        agent = Agent(
            provider=provider,
            config=AgentConfig(
                model=config.model.model_name,
                system_prompt=system_prompt,
                enable_tools=True,
            ),
        )

        return agent

    def _build_system_prompt(self, session: ClientSession) -> str:
        """Build system prompt with IDE context.

        Args:
            session: The client session

        Returns:
            System prompt string
        """
        prompt = """You are Animus, an AI coding assistant integrated into the user's IDE.

You have access to tools for:
- Reading and searching files in the workspace
- Writing and editing code files
- Running terminal commands
- Searching the web for documentation

When proposing code changes:
1. Explain what you're going to change and why
2. Show the relevant code changes
3. Wait for user approval before making changes

Be concise and focused on the task at hand."""

        if session.workspace_folder:
            prompt += f"\n\nWorkspace: {session.workspace_folder}"

        if session.current_file:
            prompt += f"\nCurrently open file: {session.current_file}"

        return prompt

    def _build_prompt(self, content: str, context: dict) -> str:
        """Build prompt with context from the IDE.

        Args:
            content: The user message
            context: Context data from IDE

        Returns:
            Enhanced prompt string
        """
        prompt_parts = []

        # Add file context
        file_path = context.get("filePath")
        if file_path:
            prompt_parts.append(f"[File: {file_path}]")

        language = context.get("language")
        if language:
            prompt_parts.append(f"[Language: {language}]")

        # Add selection context
        selection = context.get("selection")
        if selection:
            text = selection.get("text", "")
            start_line = selection.get("startLine", 0)
            end_line = selection.get("endLine", 0)
            if text:
                prompt_parts.append(f"\nSelected code (lines {start_line + 1}-{end_line + 1}):\n```\n{text}\n```")

        # Add user message
        prompt_parts.append(f"\n{content}")

        return "\n".join(prompt_parts)

    async def _send(self, websocket: WebSocketConnection, message: dict):
        """Send a message to a client.

        Args:
            websocket: The WebSocket connection
            message: The message to send
        """
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def _send_error(self, websocket: WebSocketConnection, error: str):
        """Send an error message to a client.

        Args:
            websocket: The WebSocket connection
            error: The error message
        """
        await self._send(websocket, {"type": "error", "error": error})

    def broadcast(self, message: dict, exclude: Optional[str] = None):
        """Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast
            exclude: Optional session ID to exclude
        """
        for session_id, session in self.sessions.items():
            if exclude and session_id == exclude:
                continue
            asyncio.create_task(self._send(session.websocket, message))


async def run_websocket_server(
    host: str = "localhost",
    port: int = 8765,
):
    """Run the WebSocket server.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    server = WebSocketServer(host=host, port=port)
    await server.start()

    print(f"Animus WebSocket Server running on ws://{host}:{port}")
    print("Waiting for IDE connections...")
    print()
    print("Press Ctrl+C to stop")

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


def main():
    """Main entry point for WebSocket server."""
    import argparse

    parser = argparse.ArgumentParser(description="Animus WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")

    args = parser.parse_args()

    try:
        asyncio.run(run_websocket_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
