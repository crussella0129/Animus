"""MCP Protocol definitions.

Based on the Model Context Protocol specification.
https://modelcontextprotocol.io/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import uuid


class MCPMessageType(Enum):
    """Types of MCP messages."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPMessage:
    """Base MCP message."""
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "MCPMessage":
        """Create from dictionary."""
        if "result" in data or "error" in data:
            return MCPResponse.from_dict(data)
        elif "method" in data:
            if "id" in data:
                return MCPRequest.from_dict(data)
            else:
                return MCPNotification.from_dict(data)
        raise ValueError("Invalid MCP message format")


@dataclass
class MCPRequest(MCPMessage):
    """MCP request message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = ""
    params: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
        }
        if self.params:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MCPRequest":
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", str(uuid.uuid4())),
            method=data.get("method", ""),
            params=data.get("params"),
        )


@dataclass
class MCPResponse(MCPMessage):
    """MCP response message."""
    id: str = ""
    result: Optional[Any] = None
    error: Optional["MCPError"] = None

    def to_dict(self) -> dict:
        d = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MCPResponse":
        error = None
        if "error" in data:
            error = MCPError.from_dict(data["error"])
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", ""),
            result=data.get("result"),
            error=error,
        )


@dataclass
class MCPNotification(MCPMessage):
    """MCP notification message (no response expected)."""
    method: str = ""
    params: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MCPNotification":
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            params=data.get("params"),
        )


@dataclass
class MCPError:
    """MCP error object."""
    code: int
    message: str
    data: Optional[Any] = None

    # Standard error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    def to_dict(self) -> dict:
        d = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MCPError":
        return cls(
            code=data.get("code", 0),
            message=data.get("message", ""),
            data=data.get("data"),
        )

    @classmethod
    def parse_error(cls, message: str = "Parse error") -> "MCPError":
        return cls(code=cls.PARSE_ERROR, message=message)

    @classmethod
    def invalid_request(cls, message: str = "Invalid request") -> "MCPError":
        return cls(code=cls.INVALID_REQUEST, message=message)

    @classmethod
    def method_not_found(cls, message: str = "Method not found") -> "MCPError":
        return cls(code=cls.METHOD_NOT_FOUND, message=message)

    @classmethod
    def invalid_params(cls, message: str = "Invalid params") -> "MCPError":
        return cls(code=cls.INVALID_PARAMS, message=message)

    @classmethod
    def internal_error(cls, message: str = "Internal error") -> "MCPError":
        return cls(code=cls.INTERNAL_ERROR, message=message)


# MCP Capability definitions
@dataclass
class MCPServerCapabilities:
    """Server capabilities advertised to clients."""
    tools: bool = True
    resources: bool = False
    prompts: bool = False
    logging: bool = False

    def to_dict(self) -> dict:
        caps = {}
        if self.tools:
            caps["tools"] = {}
        if self.resources:
            caps["resources"] = {}
        if self.prompts:
            caps["prompts"] = {}
        if self.logging:
            caps["logging"] = {}
        return caps


@dataclass
class MCPClientCapabilities:
    """Client capabilities sent during initialization."""
    roots: bool = False
    sampling: bool = False

    def to_dict(self) -> dict:
        caps = {}
        if self.roots:
            caps["roots"] = {"listChanged": True}
        if self.sampling:
            caps["sampling"] = {}
        return caps


@dataclass
class MCPToolInfo:
    """Tool information for tools/list response."""
    name: str
    description: str
    inputSchema: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class MCPToolCallRequest:
    """Tool call request from tools/call."""
    name: str
    arguments: dict

    @classmethod
    def from_dict(cls, data: dict) -> "MCPToolCallRequest":
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
        )


@dataclass
class MCPToolCallResult:
    """Tool call result for tools/call response."""
    content: list[dict]  # Array of content blocks
    isError: bool = False

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "isError": self.isError,
        }

    @classmethod
    def text(cls, text: str) -> "MCPToolCallResult":
        """Create a text result."""
        return cls(content=[{"type": "text", "text": text}])

    @classmethod
    def error(cls, message: str) -> "MCPToolCallResult":
        """Create an error result."""
        return cls(content=[{"type": "text", "text": message}], isError=True)
