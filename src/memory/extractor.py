"""Text extraction from various file formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import mimetypes
import re


@dataclass
class ExtractedText:
    """Extracted text with metadata."""
    content: str
    source: Path
    file_type: str
    encoding: str = "utf-8"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextExtractor(ABC):
    """Base class for text extractors."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return set of supported file extensions."""
        ...

    @abstractmethod
    def extract(self, path: Path) -> ExtractedText:
        """
        Extract text from a file.

        Args:
            path: Path to the file.

        Returns:
            ExtractedText object with content and metadata.
        """
        ...

    def can_extract(self, path: Path) -> bool:
        """Check if this extractor can handle the file."""
        return path.suffix.lower() in self.supported_extensions


class PlainTextExtractor(TextExtractor):
    """Extractor for plain text files."""

    EXTENSIONS = {
        ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml",
        ".xml", ".html", ".htm", ".log", ".ini", ".cfg", ".conf",
        ".toml", ".env", ".gitignore", ".dockerignore",
    }

    @property
    def supported_extensions(self) -> set[str]:
        return self.EXTENSIONS

    def extract(self, path: Path) -> ExtractedText:
        """Extract text from plain text file."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        content = ""
        used_encoding = "utf-8"

        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                used_encoding = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        return ExtractedText(
            content=content,
            source=path,
            file_type=path.suffix.lower(),
            encoding=used_encoding,
            metadata={"size": path.stat().st_size},
        )


class CodeExtractor(TextExtractor):
    """Extractor for source code files."""

    EXTENSIONS = {
        # Python
        ".py", ".pyw", ".pyi",
        # JavaScript/TypeScript
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        # Web
        ".css", ".scss", ".sass", ".less",
        # Systems
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
        ".go", ".rs", ".swift", ".kt", ".kts",
        # JVM
        ".java", ".scala", ".clj", ".cljs",
        # .NET
        ".cs", ".fs", ".vb",
        # Scripting
        ".rb", ".php", ".pl", ".pm", ".lua",
        ".sh", ".bash", ".zsh", ".fish", ".ps1",
        # Data/Config
        ".sql", ".graphql", ".proto",
        # Other
        ".r", ".R", ".jl", ".ex", ".exs", ".erl", ".hrl",
        ".hs", ".elm", ".ml", ".mli", ".v", ".sv",
    }

    # Language detection by extension
    LANGUAGE_MAP = {
        ".py": "python", ".pyw": "python", ".pyi": "python",
        ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".c": "c", ".h": "c",
        ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".cs": "csharp",
        ".sh": "bash", ".bash": "bash",
        ".sql": "sql",
    }

    @property
    def supported_extensions(self) -> set[str]:
        return self.EXTENSIONS

    def extract(self, path: Path) -> ExtractedText:
        """Extract code with language detection."""
        content = path.read_text(encoding="utf-8", errors="replace")
        ext = path.suffix.lower()

        return ExtractedText(
            content=content,
            source=path,
            file_type=ext,
            metadata={
                "language": self.LANGUAGE_MAP.get(ext, "unknown"),
                "lines": content.count("\n") + 1,
                "size": path.stat().st_size,
            },
        )


class MarkdownExtractor(TextExtractor):
    """Extractor for Markdown files with optional preprocessing."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown", ".mdx"}

    def extract(self, path: Path) -> ExtractedText:
        """Extract and preprocess Markdown."""
        content = path.read_text(encoding="utf-8", errors="replace")

        # Extract metadata
        metadata = {
            "has_code_blocks": "```" in content,
            "has_links": "](" in content or "]: " in content,
            "size": path.stat().st_size,
        }

        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        return ExtractedText(
            content=content,
            source=path,
            file_type=".md",
            metadata=metadata,
        )


class PDFExtractor(TextExtractor):
    """
    Extractor for PDF files.

    Requires pypdf or pdfplumber to be installed.
    Falls back gracefully if not available.
    """

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def _is_available(self) -> bool:
        """Check if PDF extraction is available."""
        try:
            import pypdf  # noqa: F401
            return True
        except ImportError:
            try:
                import pdfplumber  # noqa: F401
                return True
            except ImportError:
                return False

    def extract(self, path: Path) -> ExtractedText:
        """Extract text from PDF."""
        if not self._is_available():
            return ExtractedText(
                content=f"[PDF extraction not available - install pypdf or pdfplumber]",
                source=path,
                file_type=".pdf",
                metadata={"error": "missing_dependency"},
            )

        try:
            # Try pypdf first
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                content = "\n\n".join(pages)
                page_count = len(reader.pages)
            except ImportError:
                # Fall back to pdfplumber
                import pdfplumber
                pages = []
                with pdfplumber.open(path) as pdf:
                    page_count = len(pdf.pages)
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        pages.append(text)
                content = "\n\n".join(pages)

            return ExtractedText(
                content=content,
                source=path,
                file_type=".pdf",
                metadata={
                    "page_count": page_count,
                    "size": path.stat().st_size,
                },
            )
        except Exception as e:
            return ExtractedText(
                content=f"[Error extracting PDF: {e}]",
                source=path,
                file_type=".pdf",
                metadata={"error": str(e)},
            )


class ExtractorRegistry:
    """Registry of text extractors."""

    def __init__(self):
        self._extractors: list[TextExtractor] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default extractors."""
        self._extractors = [
            CodeExtractor(),
            MarkdownExtractor(),
            PDFExtractor(),
            PlainTextExtractor(),  # Should be last (most generic)
        ]

    def register(self, extractor: TextExtractor) -> None:
        """Register a custom extractor."""
        self._extractors.insert(0, extractor)

    def get_extractor(self, path: Path) -> Optional[TextExtractor]:
        """Get appropriate extractor for a file."""
        for extractor in self._extractors:
            if extractor.can_extract(path):
                return extractor
        return None

    def extract(self, path: Path) -> Optional[ExtractedText]:
        """Extract text from a file using appropriate extractor."""
        extractor = self.get_extractor(path)
        if extractor:
            return extractor.extract(path)
        return None

    def can_extract(self, path: Path) -> bool:
        """Check if any extractor can handle the file."""
        return any(e.can_extract(path) for e in self._extractors)


# Global registry instance
_registry = ExtractorRegistry()


def extract_text(path: Path) -> Optional[ExtractedText]:
    """
    Extract text from a file.

    Args:
        path: Path to the file.

    Returns:
        ExtractedText object or None if extraction failed.
    """
    return _registry.extract(path)


def can_extract(path: Path) -> bool:
    """Check if text can be extracted from a file."""
    return _registry.can_extract(path)
