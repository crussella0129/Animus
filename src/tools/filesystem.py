"""Filesystem tools for the agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import os

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory


class ReadFileTool(Tool):
    """Tool to read file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to read.",
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="Starting line number (1-indexed). If not specified, reads from beginning.",
                required=False,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="Ending line number (inclusive). If not specified, reads to end.",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    async def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Read file contents."""
        try:
            file_path = Path(path).resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}",
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a file: {path}",
                )

            # Check file size (limit to 1MB)
            size = file_path.stat().st_size
            if size > 1024 * 1024:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File too large ({size} bytes). Use start_line/end_line to read portions.",
                )

            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            total_lines = len(lines)

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1  # Convert to 0-indexed
                end = end_line or total_lines
                lines = lines[start:end]
                content = "\n".join(lines)

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": str(file_path),
                    "size": size,
                    "total_lines": total_lines,
                    "lines_returned": len(lines),
                },
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error reading file: {e}",
            )


class WriteFileTool(Tool):
    """Tool to write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file if it doesn't exist, or overwrites if it does."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to write.",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file.",
            ),
            ToolParameter(
                name="create_dirs",
                type="boolean",
                description="Create parent directories if they don't exist.",
                required=False,
                default=True,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def requires_confirmation(self) -> bool:
        return True  # Writing files should be confirmed

    async def execute(
        self,
        path: str,
        content: str,
        create_dirs: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Write content to file."""
        try:
            file_path = Path(path).resolve()

            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {file_path}",
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                },
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error writing file: {e}",
            )


class ListDirectoryTool(Tool):
    """Tool to list directory contents."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the directory to list.",
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="List recursively (include subdirectories).",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_depth",
                type="integer",
                description="Maximum depth for recursive listing.",
                required=False,
                default=3,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    def _list_dir(
        self,
        path: Path,
        recursive: bool,
        max_depth: int,
        current_depth: int = 0,
        prefix: str = "",
    ) -> list[str]:
        """Recursively list directory contents."""
        items = []

        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "

                if entry.is_dir():
                    items.append(f"{prefix}{connector}{entry.name}/")
                    if recursive and current_depth < max_depth:
                        items.extend(self._list_dir(
                            entry,
                            recursive,
                            max_depth,
                            current_depth + 1,
                            prefix + extension,
                        ))
                else:
                    size = entry.stat().st_size
                    size_str = self._format_size(size)
                    items.append(f"{prefix}{connector}{entry.name} ({size_str})")

        except PermissionError:
            items.append(f"{prefix}[Permission denied]")

        return items

    def _format_size(self, size: int) -> str:
        """Format file size for display."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    async def execute(
        self,
        path: str,
        recursive: bool = False,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> ToolResult:
        """List directory contents."""
        try:
            dir_path = Path(path).resolve()

            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory not found: {path}",
                )

            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a directory: {path}",
                )

            items = self._list_dir(dir_path, recursive, max_depth)
            output = f"{dir_path}/\n" + "\n".join(items)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "path": str(dir_path),
                    "item_count": len(items),
                    "recursive": recursive,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error listing directory: {e}",
            )
