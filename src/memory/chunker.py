"""Text chunking strategies: token-based and code-aware."""

from __future__ import annotations

import re
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


class Chunker:
    """Split text into chunks suitable for embedding."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size  # in estimated tokens
        self.overlap = overlap  # overlap in estimated tokens

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Chunk text into pieces with metadata."""
        if not text.strip():
            return []

        # Try code-aware chunking first
        if self._looks_like_code(text):
            chunks = self._chunk_code(text)
        else:
            chunks = self._chunk_by_tokens(text)

        result = []
        for i, chunk_text in enumerate(chunks):
            entry: dict[str, Any] = {
                "text": chunk_text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "estimated_tokens": _estimate_tokens(chunk_text),
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

    def _chunk_code(self, text: str) -> list[str]:
        """Split code by function/class boundaries, falling back to token-based."""
        # Split on top-level function/class definitions
        pattern = r"^(?=(?:def |class |async def ))"
        parts = re.split(pattern, text, flags=re.MULTILINE)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            return self._chunk_by_tokens(text)

        # Merge small adjacent parts to meet chunk_size
        chunks = []
        current = ""
        for part in parts:
            if _estimate_tokens(current + "\n" + part) > self.chunk_size and current:
                chunks.append(current.strip())
                current = part
            else:
                current = (current + "\n" + part).strip() if current else part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic: does the text look like source code?"""
        code_indicators = ["def ", "class ", "import ", "function ", "const ", "var ", "let "]
        lines = text.split("\n")[:20]
        score = sum(1 for line in lines if any(ind in line for ind in code_indicators))
        return score >= 2
