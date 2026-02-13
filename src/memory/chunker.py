"""Text chunking strategies: token-based and code-aware."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

# Import accurate token estimation from context module
from src.core.context import estimate_tokens


# Language-specific boundary patterns for code chunking
_LANGUAGE_BOUNDARIES = {
    "python": r"^(?=(?:def |class |async def ))",
    "go": r"^(?=(?:func |type \w+ struct|type \w+ interface))",
    "rust": r"^(?=(?:fn |impl |struct |enum |trait |mod |pub fn |pub struct ))",
    "c": r"^(?=\w[\w\s\*]+\w+\s*\([^)]*\)\s*\{)",  # Function signatures
    "cpp": r"^(?=(?:class |struct |namespace |\w[\w\s\*]+\w+\s*\([^)]*\)\s*\{))",
    "javascript": r"^(?=(?:function |class |const \w+ = (?:async )?\(|export ))",
    "typescript": r"^(?=(?:function |class |interface |type |const \w+ = (?:async )?\(|export ))",
    "shell": r"^(?=(?:function |\w+\(\)\s*\{))",  # Bash/sh functions
}

# File extension to language mapping
_EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}


class Chunker:
    """Split text into chunks suitable for embedding."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size  # in estimated tokens
        self.overlap = overlap  # overlap in estimated tokens

    def chunk(self, text: str, metadata: dict[str, Any] | None = None, filepath: Optional[str] = None) -> list[dict[str, Any]]:
        """Chunk text into pieces with metadata.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            filepath: Optional file path for AST-based chunking (Python files)

        Returns:
            List of chunk dicts with text and metadata
        """
        if not text.strip():
            return []

        # Try AST-informed chunking for Python files
        if filepath and filepath.endswith('.py'):
            chunks = self._chunk_python_ast(text, filepath)
            if chunks:  # AST parsing succeeded
                return chunks

        # Fallback to regex-based code chunking or token-based chunking
        if self._looks_like_code(text):
            chunk_texts = self._chunk_code(text, filepath)
        else:
            chunk_texts = self._chunk_by_tokens(text)

        result = []
        for i, chunk_text in enumerate(chunk_texts):
            entry: dict[str, Any] = {
                "text": chunk_text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "estimated_tokens": estimate_tokens(chunk_text),
                },
            }
            result.append(entry)
        return result

    def _chunk_by_tokens(self, text: str) -> list[str]:
        """Split text by estimated token boundaries (character-based)."""
        chars_per_chunk = self.chunk_size * 4
        chars_overlap = self.overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = start + chars_per_chunk
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - chars_overlap
            if start <= (end - chars_per_chunk):
                break
        return chunks or ([text.strip()] if text.strip() else [])

    def _chunk_code(self, text: str, filepath: Optional[str] = None) -> list[str]:
        """Split code by function/class boundaries using language-specific patterns.

        Args:
            text: The code to chunk
            filepath: Optional file path to detect language

        Returns:
            List of code chunks
        """
        # Detect language from filepath or heuristics
        language = self._detect_language(text, filepath)
        pattern = _LANGUAGE_BOUNDARIES.get(language)

        if not pattern:
            # Unknown language, fall back to token-based chunking
            return self._chunk_by_tokens(text)

        # Split on top-level function/class definitions
        parts = re.split(pattern, text, flags=re.MULTILINE)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            return self._chunk_by_tokens(text)

        # Merge small adjacent parts to meet chunk_size
        chunks = []
        current = ""
        for part in parts:
            if estimate_tokens(current + "\n" + part) > self.chunk_size and current:
                chunks.append(current.strip())
                current = part
            else:
                current = (current + "\n" + part).strip() if current else part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _detect_language(self, text: str, filepath: Optional[str] = None) -> str:
        """Detect programming language from file extension or content.

        Args:
            text: The source code
            filepath: Optional file path

        Returns:
            Language identifier (e.g., "python", "go", "rust")
        """
        # First try file extension
        if filepath:
            file_path = Path(filepath)
            ext = file_path.suffix.lower()
            if ext in _EXTENSION_TO_LANGUAGE:
                return _EXTENSION_TO_LANGUAGE[ext]

        # Fallback: heuristic detection from content
        # Check first 50 lines for language indicators
        lines = text.split("\n")[:50]
        sample = "\n".join(lines)

        # Python indicators
        if re.search(r"\b(def |class |import |from .* import)", sample):
            return "python"

        # Go indicators
        if re.search(r"\b(package |func |type .*struct|import \()", sample):
            return "go"

        # Rust indicators
        if re.search(r"\b(fn |impl |struct |enum |trait |use )", sample):
            return "rust"

        # C/C++ indicators
        if re.search(r"#include|std::|namespace ", sample):
            return "cpp" if "std::" in sample or "namespace" in sample else "c"

        # JavaScript/TypeScript indicators
        if re.search(r"\b(function|const |let |var |=>|export |import )", sample):
            if re.search(r"\b(interface|type |as |<.*>)", sample):
                return "typescript"
            return "javascript"

        # Shell indicators
        if re.search(r"^#!/bin/(bash|sh|zsh)|function \w+", sample):
            return "shell"

        # Default to Python pattern for unknown code
        return "python"

    def _chunk_python_ast(self, text: str, filepath: str) -> list[dict[str, Any]] | None:
        """Chunk Python code using AST parsing for semantic boundaries.

        This leverages the existing PythonParser to extract functions, classes,
        and methods with their actual boundaries, docstrings, and metadata.

        Args:
            text: Python source code
            filepath: Path to the Python file

        Returns:
            List of chunks with rich metadata, or None if parsing fails
        """
        try:
            from src.knowledge.parser import PythonParser
        except ImportError:
            return None

        parser = PythonParser()
        try:
            result = parser.parse_file(Path(filepath))
        except Exception:
            return None  # Parsing failed, fall back to regex chunking

        if not result.nodes:
            return None

        lines = text.splitlines(keepends=True)
        chunks = []

        # Skip module node (index 0), chunk classes and functions
        for node in result.nodes[1:]:  # Skip the module itself
            if node.kind in ("function", "method", "class"):
                # Extract source lines for this node
                start_idx = max(0, node.line_start - 1)
                end_idx = min(len(lines), node.line_end)
                chunk_text = "".join(lines[start_idx:end_idx]).strip()

                if not chunk_text:
                    continue

                # Check if chunk exceeds chunk_size and needs splitting
                chunk_tokens = estimate_tokens(chunk_text)
                if chunk_tokens > self.chunk_size:
                    # Large node: fall back to token-based splitting for this node
                    sub_chunks = self._chunk_by_tokens(chunk_text)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            "text": sub_chunk,
                            "metadata": {
                                "kind": node.kind,
                                "name": node.name,
                                "qualified_name": node.qualified_name,
                                "file": filepath,
                                "lines": f"{node.line_start}-{node.line_end}",
                                "docstring": node.docstring[:200] if node.docstring else "",
                                "chunk_index": i,
                                "estimated_tokens": estimate_tokens(sub_chunk),
                                "chunking_method": "ast_split",
                            }
                        })
                else:
                    # Node fits in one chunk
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "kind": node.kind,
                            "name": node.name,
                            "qualified_name": node.qualified_name,
                            "file": filepath,
                            "lines": f"{node.line_start}-{node.line_end}",
                            "docstring": node.docstring[:200] if node.docstring else "",
                            "estimated_tokens": chunk_tokens,
                            "chunking_method": "ast",
                        }
                    })

        return chunks if chunks else None

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic: does the text look like source code?"""
        code_indicators = ["def ", "class ", "import ", "function ", "const ", "var ", "let "]
        lines = text.split("\n")[:20]
        score = sum(1 for line in lines if any(ind in line for ind in code_indicators))
        return score >= 2
