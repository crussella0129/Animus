"""Text chunking strategies for RAG ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional
import re


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    start_offset: int
    end_offset: int
    metadata: dict


class ChunkingStrategy(ABC):
    """Base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> Iterator[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Yields:
            TextChunk objects.
        """
        ...


class TokenChunker(ChunkingStrategy):
    """
    Chunk text by approximate token count.

    Uses a simple heuristic: ~4 characters per token (for English text).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chars_per_token: float = 4.0,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
            chars_per_token: Approximate characters per token.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token

    @property
    def char_chunk_size(self) -> int:
        return int(self.chunk_size * self.chars_per_token)

    @property
    def char_overlap(self) -> int:
        return int(self.chunk_overlap * self.chars_per_token)

    def chunk(self, text: str, metadata: Optional[dict] = None) -> Iterator[TextChunk]:
        """Chunk text into approximately token-sized pieces."""
        metadata = metadata or {}

        if len(text) <= self.char_chunk_size:
            yield TextChunk(
                content=text,
                start_offset=0,
                end_offset=len(text),
                metadata=metadata,
            )
            return

        step = self.char_chunk_size - self.char_overlap
        start = 0

        while start < len(text):
            end = min(start + self.char_chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look for space within last 20% of chunk
                search_start = start + int(self.char_chunk_size * 0.8)
                space_pos = text.rfind(" ", search_start, end)
                if space_pos > search_start:
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                yield TextChunk(
                    content=chunk_text,
                    start_offset=start,
                    end_offset=end,
                    metadata=metadata,
                )

            start += step


class SentenceChunker(ChunkingStrategy):
    """
    Chunk text by sentences, grouping into target size.

    Better for preserving semantic coherence.
    """

    # Sentence-ending punctuation pattern
    SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,  # Number of sentences to overlap
        chars_per_token: float = 4.0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token

    @property
    def char_chunk_size(self) -> int:
        return int(self.chunk_size * self.chars_per_token)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = self.SENTENCE_END.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> Iterator[TextChunk]:
        """Chunk text by sentences."""
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return

        current_chunk: list[str] = []
        current_size = 0
        start_offset = 0

        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)

            if current_size + sentence_size > self.char_chunk_size and current_chunk:
                # Emit current chunk
                chunk_text = " ".join(current_chunk)
                yield TextChunk(
                    content=chunk_text,
                    start_offset=start_offset,
                    end_offset=start_offset + len(chunk_text),
                    metadata=metadata,
                )

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s) for s in current_chunk)
                start_offset = text.find(current_chunk[0], start_offset) if current_chunk else start_offset + len(chunk_text)

            current_chunk.append(sentence)
            current_size += sentence_size

        # Emit final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            yield TextChunk(
                content=chunk_text,
                start_offset=start_offset,
                end_offset=start_offset + len(chunk_text),
                metadata=metadata,
            )


class CodeChunker(ChunkingStrategy):
    """
    Chunk code by logical units (functions, classes).

    Falls back to line-based chunking if parsing fails.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chars_per_token: float = 4.0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token
        self._token_chunker = TokenChunker(chunk_size, chunk_overlap, chars_per_token)

    def _find_code_blocks(self, text: str, language: str) -> list[tuple[int, int, str]]:
        """
        Find code blocks (functions, classes) in source code.

        Returns list of (start, end, block_type) tuples.
        """
        blocks = []

        # Python patterns
        if language in ("python", "py"):
            # Match function and class definitions
            pattern = re.compile(r'^(def |class |async def )', re.MULTILINE)
            matches = list(pattern.finditer(text))

            for i, match in enumerate(matches):
                start = match.start()
                # Find end: either next match or end of file
                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(text)

                block_type = "function" if "def " in match.group() else "class"
                blocks.append((start, end, block_type))

        # JavaScript/TypeScript patterns
        elif language in ("javascript", "typescript", "js", "ts"):
            pattern = re.compile(
                r'^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>)',
                re.MULTILINE
            )
            matches = list(pattern.finditer(text))

            for i, match in enumerate(matches):
                start = match.start()
                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(text)
                blocks.append((start, end, "block"))

        return blocks

    def chunk(self, text: str, metadata: Optional[dict] = None) -> Iterator[TextChunk]:
        """Chunk code into logical units."""
        metadata = metadata or {}
        language = metadata.get("language", "")

        # Try to find code blocks
        blocks = self._find_code_blocks(text, language)

        if not blocks:
            # Fall back to token chunking
            yield from self._token_chunker.chunk(text, metadata)
            return

        char_limit = int(self.chunk_size * self.chars_per_token)

        for start, end, block_type in blocks:
            block_text = text[start:end].strip()

            if len(block_text) <= char_limit:
                yield TextChunk(
                    content=block_text,
                    start_offset=start,
                    end_offset=end,
                    metadata={**metadata, "block_type": block_type},
                )
            else:
                # Block too large, sub-chunk it
                for sub_chunk in self._token_chunker.chunk(block_text, metadata):
                    sub_chunk.start_offset += start
                    sub_chunk.end_offset += start
                    sub_chunk.metadata["block_type"] = block_type
                    yield sub_chunk


def get_chunker(
    file_type: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> ChunkingStrategy:
    """
    Get appropriate chunker for a file type.

    Args:
        file_type: File extension or MIME type.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks.

    Returns:
        Appropriate ChunkingStrategy instance.
    """
    code_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".c", ".cpp", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".cs", ".fs",
    }

    if file_type.lower() in code_extensions:
        return CodeChunker(chunk_size, chunk_overlap)

    # Default to sentence chunking for text
    return SentenceChunker(chunk_size, chunk_overlap)
