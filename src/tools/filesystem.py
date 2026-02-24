"""Filesystem tools: read_file, write_file, list_dir."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

from src.core.workspace import Workspace, WorkspaceBoundaryError
from src.core.permission import PermissionChecker
from src.tools.base import Tool, ToolRegistry


class ReadFileTool(Tool):
    """Read files (no isolation needed - read-only operation)."""

    def __init__(self, session_cwd: Workspace | None = None):
        super().__init__()
        # ReadFileTool doesn't need isolation (read-only, low risk)
        self._isolation_level = "none"
        self._session_cwd = session_cwd

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "max_lines": {"type": "integer", "description": "Maximum lines to read (default: all)"},
            },
            "required": ["path"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        try:
            if self._session_cwd is not None:
                path = self._session_cwd.resolve(args["path"])
            else:
                path = Path(args["path"]).resolve()
        except WorkspaceBoundaryError as e:
            return f"Error: {e}"
        checker = PermissionChecker()
        if not checker.is_path_safe(path):
            return f"Error: Access denied to {path}"
        if not path.exists():
            return f"Error: File not found: {path}"
        if not path.is_file():
            return f"Error: Not a file: {path}"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            max_lines = args.get("max_lines")
            if max_lines:
                lines = text.splitlines()[:max_lines]
                return "\n".join(lines)
            return text
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Write files with audit trail tracking.

    Maintains a log of all write operations for debugging and undo capability.
    """

    # Class-level audit log (shared across all instances)
    _write_log: list[dict[str, Any]] = []
    _write_log_lock = threading.Lock()

    def __init__(self, session_cwd: Workspace | None = None):
        super().__init__()
        # WriteFileTool could be isolated for untrusted content
        # But default to none for performance
        self._isolation_level = "none"
        self._session_cwd = session_cwd

    @classmethod
    def get_write_log(cls) -> list[dict[str, Any]]:
        """Get the complete write audit log.

        Returns:
            List of write operations with path, size, hash, and timestamp
        """
        with cls._write_log_lock:
            return cls._write_log.copy()

    @classmethod
    def clear_write_log(cls) -> None:
        """Clear the write audit log."""
        with cls._write_log_lock:
            cls._write_log.clear()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    def execute(self, args: dict[str, Any]) -> str:
        try:
            if self._session_cwd is not None:
                path = self._session_cwd.resolve(args["path"])
            else:
                path = Path(args["path"]).resolve()
        except WorkspaceBoundaryError as e:
            return f"Error: {e}"
        checker = PermissionChecker()
        if not checker.is_path_safe(path):
            return f"Error: Access denied to {path}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args["content"]
            path.write_text(content, encoding="utf-8")

            # Record write operation in audit log
            content_hash = hashlib.md5(content.encode()).hexdigest()
            with self._write_log_lock:
                self._write_log.append({
                    "path": str(path),
                    "size": len(content),
                    "timestamp": time.time(),
                    "hash": content_hash,
                    "lines": content.count('\n') + 1,
                })

            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class ListDirTool(Tool):
    """List directory contents (no isolation needed - read-only)."""

    def __init__(self, session_cwd: Workspace | None = None):
        super().__init__()
        # ListDirTool doesn't need isolation (read-only operation)
        self._isolation_level = "none"
        self._session_cwd = session_cwd

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files and directories at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current directory)"},
                "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
            },
            "required": [],
        }

    def execute(self, args: dict[str, Any]) -> str:
        raw_path = args.get("path", ".")
        try:
            if self._session_cwd is not None:
                path = self._session_cwd.resolve(raw_path)
            else:
                path = Path(raw_path).resolve()
        except WorkspaceBoundaryError as e:
            return f"Error: {e}"
        checker = PermissionChecker()
        if not checker.is_path_safe(path):
            return f"Error: Access denied to {path}"
        if not path.exists():
            return f"Error: Path not found: {path}"
        if not path.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            entries = []
            if args.get("recursive", False):
                for item in sorted(path.rglob("*")):
                    rel = item.relative_to(path)
                    prefix = "[DIR] " if item.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {rel}")
                    if len(entries) >= 500:
                        entries.append("... (truncated at 500 entries)")
                        break
            else:
                for item in sorted(path.iterdir()):
                    prefix = "[DIR] " if item.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {item.name}")
            return "\n".join(entries) if entries else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"


def register_filesystem_tools(registry: ToolRegistry, session_cwd: Workspace | None = None) -> None:
    """Register all filesystem tools with the given registry."""
    registry.register(ReadFileTool(session_cwd=session_cwd))
    registry.register(WriteFileTool(session_cwd=session_cwd))
    registry.register(ListDirTool(session_cwd=session_cwd))
