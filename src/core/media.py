"""Media Pipeline — file download, MIME detection, and TTL-based cleanup.

Provides safe file downloading with size limits, automatic MIME type detection,
and time-to-live based cleanup for temporary files.

Storage: ~/.animus/media/

Implementation Principle: 100% hardcoded. No LLM involvement.

Usage:
    pipeline = MediaPipeline()

    # Download a file
    result = await pipeline.download("https://example.com/file.pdf")
    print(result.path, result.mime_type, result.size_bytes)

    # Cleanup expired files
    removed = pipeline.cleanup()

    # Detect MIME type of local file
    mime = detect_mime_type(Path("image.png"))
"""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

# Default media storage directory
DEFAULT_MEDIA_DIR = Path.home() / ".animus" / "media"

# Default size limit: 50 MB
DEFAULT_MAX_SIZE_BYTES = 50 * 1024 * 1024

# Default TTL: 24 hours
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# Magic number signatures for MIME detection (first N bytes -> MIME type)
# These are checked BEFORE extension-based detection for reliability
MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    # Images
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # Also used by WAV, AVI — refined below
    (b"BM", "image/bmp"),
    (b"\x00\x00\x01\x00", "image/x-icon"),  # ICO
    # Documents
    (b"%PDF", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),  # Also DOCX, XLSX, PPTX
    # Audio
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    # Video
    (b"\x00\x00\x00\x1cftyp", "video/mp4"),
    (b"\x00\x00\x00\x18ftyp", "video/mp4"),
    (b"\x00\x00\x00\x20ftyp", "video/mp4"),
    (b"\x1aE\xdf\xa3", "video/webm"),
    # Archives
    (b"\x1f\x8b", "application/gzip"),
    (b"BZ", "application/x-bzip2"),
    (b"\xfd7zXZ\x00", "application/x-xz"),
    (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "application/x-rar-compressed"),
    # Text (fallback — check content for text patterns)
    (b"<!DOCTYPE", "text/html"),
    (b"<html", "text/html"),
    (b"<?xml", "application/xml"),
    (b"{", "application/json"),  # Heuristic — first non-whitespace is {
]

# Allowed MIME type prefixes (block executable types by default)
ALLOWED_MIME_PREFIXES = frozenset([
    "text/",
    "image/",
    "audio/",
    "video/",
    "application/pdf",
    "application/json",
    "application/xml",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/x-bzip2",
    "application/x-xz",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/octet-stream",
    "application/yaml",
    "application/toml",
    "application/csv",
])

# Blocked extensions (never download these)
BLOCKED_EXTENSIONS = frozenset([
    ".exe", ".bat", ".cmd", ".com", ".msi", ".scr", ".pif",
    ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh",
    ".ps1", ".psm1", ".psd1",
    ".dll", ".sys", ".drv",
    ".sh", ".bash", ".csh",
])


class DownloadStatus(str, Enum):
    """Status of a download operation."""
    SUCCESS = "success"
    SIZE_EXCEEDED = "size_exceeded"
    BLOCKED_TYPE = "blocked_type"
    NETWORK_ERROR = "network_error"
    INVALID_URL = "invalid_url"


@dataclass
class DownloadResult:
    """Result of a download operation."""
    status: DownloadStatus
    path: Optional[Path] = None
    mime_type: Optional[str] = None
    size_bytes: int = 0
    filename: str = ""
    error: Optional[str] = None
    ttl_expires: float = 0.0


@dataclass
class MediaFile:
    """Metadata for a stored media file."""
    path: Path
    mime_type: str
    size_bytes: int
    downloaded_at: float
    ttl_expires: float
    source_url: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the file has exceeded its TTL."""
        return time.time() > self.ttl_expires

    @property
    def age_seconds(self) -> float:
        """Time since download in seconds."""
        return time.time() - self.downloaded_at


def detect_mime_type(path: Path) -> str:
    """Detect MIME type using magic numbers and file extension.

    Checks magic numbers first (more reliable), falls back to extension.

    Args:
        path: Path to the file.

    Returns:
        MIME type string (e.g. "image/png", "application/pdf").
    """
    # Try magic number detection first
    try:
        with open(path, "rb") as f:
            header = f.read(32)

        for signature, mime in MAGIC_SIGNATURES:
            if header.startswith(signature):
                # Refine RIFF-based detection
                if signature == b"RIFF" and len(header) >= 12:
                    if header[8:12] == b"WEBP":
                        return "image/webp"
                    elif header[8:12] == b"WAVE":
                        return "audio/wav"
                    elif header[8:12] == b"AVI ":
                        return "video/x-msvideo"
                return mime
    except (OSError, IOError):
        pass

    # Fall back to extension-based detection
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime

    return "application/octet-stream"


def is_mime_allowed(mime_type: str) -> bool:
    """Check if a MIME type is in the allowed list.

    Args:
        mime_type: MIME type string.

    Returns:
        True if allowed, False if blocked.
    """
    for prefix in ALLOWED_MIME_PREFIXES:
        if mime_type.startswith(prefix):
            return True
    return False


def is_extension_blocked(filename: str) -> bool:
    """Check if a file extension is blocked.

    Args:
        filename: The filename to check.

    Returns:
        True if the extension is blocked.
    """
    ext = Path(filename).suffix.lower()
    return ext in BLOCKED_EXTENSIONS


def _sanitize_filename(url: str) -> str:
    """Extract and sanitize a filename from a URL.

    Args:
        url: The download URL.

    Returns:
        A safe filename string.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = Path(path).name

    if not filename or filename == "/":
        filename = "download"

    # Remove unsafe characters
    safe = "".join(c for c in filename if c.isalnum() or c in ".-_")
    if not safe:
        safe = "download"

    # Limit length
    if len(safe) > 200:
        ext = Path(safe).suffix
        safe = safe[:200 - len(ext)] + ext

    return safe


class MediaPipeline:
    """Media download, detection, and cleanup pipeline.

    Handles file downloads with size limits, MIME detection, and TTL cleanup.
    All logic is hardcoded — no LLM involvement.
    """

    def __init__(
        self,
        media_dir: Optional[Path] = None,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ):
        """Initialize the media pipeline.

        Args:
            media_dir: Directory for storing downloaded files.
            max_size_bytes: Maximum file size in bytes.
            ttl_seconds: Time-to-live for downloaded files in seconds.
        """
        self._media_dir = media_dir or DEFAULT_MEDIA_DIR
        self._max_size_bytes = max_size_bytes
        self._ttl_seconds = ttl_seconds
        self._files: dict[str, MediaFile] = {}

        self._media_dir.mkdir(parents=True, exist_ok=True)

    @property
    def media_dir(self) -> Path:
        """Get the media directory path."""
        return self._media_dir

    @property
    def max_size_bytes(self) -> int:
        """Get the maximum download size."""
        return self._max_size_bytes

    async def download(
        self,
        url: str,
        filename: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> DownloadResult:
        """Download a file from a URL with safety checks.

        Steps:
        1. Validate URL
        2. Check extension against blocklist
        3. Stream download with size limit enforcement
        4. Detect MIME type
        5. Check MIME against allowlist
        6. Store with TTL metadata

        Args:
            url: URL to download from.
            filename: Optional override filename.
            ttl_seconds: Optional override TTL.

        Returns:
            DownloadResult with status and file info.
        """
        try:
            import httpx
        except ImportError:
            return DownloadResult(
                status=DownloadStatus.NETWORK_ERROR,
                error="httpx not installed. Install with: pip install httpx",
            )

        # Validate URL
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return DownloadResult(
                    status=DownloadStatus.INVALID_URL,
                    error=f"Unsupported URL scheme: {parsed.scheme}",
                )
        except Exception:
            return DownloadResult(
                status=DownloadStatus.INVALID_URL,
                error=f"Invalid URL: {url}",
            )

        # Determine filename
        target_name = filename or _sanitize_filename(url)

        # Check extension blocklist
        if is_extension_blocked(target_name):
            return DownloadResult(
                status=DownloadStatus.BLOCKED_TYPE,
                filename=target_name,
                error=f"Blocked file extension: {Path(target_name).suffix}",
            )

        # Download with size limit
        ttl = ttl_seconds or self._ttl_seconds
        target_path = self._media_dir / target_name

        # Avoid filename collisions
        if target_path.exists():
            stem = target_path.stem
            ext = target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = self._media_dir / f"{stem}_{counter}{ext}"
                counter += 1

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    # Check Content-Length header if available
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self._max_size_bytes:
                        return DownloadResult(
                            status=DownloadStatus.SIZE_EXCEEDED,
                            filename=target_name,
                            size_bytes=int(content_length),
                            error=f"File size {int(content_length)} exceeds limit {self._max_size_bytes}",
                        )

                    # Stream download with running size check
                    downloaded = 0
                    with open(target_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            downloaded += len(chunk)
                            if downloaded > self._max_size_bytes:
                                f.close()
                                target_path.unlink(missing_ok=True)
                                return DownloadResult(
                                    status=DownloadStatus.SIZE_EXCEEDED,
                                    filename=target_name,
                                    size_bytes=downloaded,
                                    error=f"Download exceeded size limit at {downloaded} bytes",
                                )
                            f.write(chunk)

        except Exception as e:
            target_path.unlink(missing_ok=True)
            return DownloadResult(
                status=DownloadStatus.NETWORK_ERROR,
                filename=target_name,
                error=str(e),
            )

        # Detect MIME type
        mime = detect_mime_type(target_path)

        # Check MIME allowlist
        if not is_mime_allowed(mime):
            target_path.unlink(missing_ok=True)
            return DownloadResult(
                status=DownloadStatus.BLOCKED_TYPE,
                filename=target_name,
                mime_type=mime,
                error=f"Blocked MIME type: {mime}",
            )

        # Record metadata
        now = time.time()
        media_file = MediaFile(
            path=target_path,
            mime_type=mime,
            size_bytes=downloaded,
            downloaded_at=now,
            ttl_expires=now + ttl,
            source_url=url,
        )
        self._files[str(target_path)] = media_file

        logger.info(
            "Downloaded %s (%s, %d bytes, TTL %.0fs)",
            target_name, mime, downloaded, ttl,
        )

        return DownloadResult(
            status=DownloadStatus.SUCCESS,
            path=target_path,
            mime_type=mime,
            size_bytes=downloaded,
            filename=target_path.name,
            ttl_expires=now + ttl,
        )

    def detect(self, path: Path) -> str:
        """Detect MIME type of a local file.

        Args:
            path: Path to the file.

        Returns:
            MIME type string.
        """
        return detect_mime_type(path)

    def cleanup(self) -> list[Path]:
        """Remove expired media files.

        Returns:
            List of paths that were removed.
        """
        removed = []

        # Check tracked files
        expired_keys = []
        for key, media_file in self._files.items():
            if media_file.is_expired:
                if media_file.path.exists():
                    media_file.path.unlink()
                    removed.append(media_file.path)
                    logger.info("Cleaned up expired: %s", media_file.path.name)
                expired_keys.append(key)

        for key in expired_keys:
            del self._files[key]

        return removed

    def list_files(self) -> list[MediaFile]:
        """List all tracked media files.

        Returns:
            List of MediaFile objects.
        """
        return list(self._files.values())

    def get_file(self, path: Path) -> Optional[MediaFile]:
        """Get metadata for a tracked file.

        Args:
            path: Path to the file.

        Returns:
            MediaFile or None if not tracked.
        """
        return self._files.get(str(path))

    def remove(self, path: Path) -> bool:
        """Remove a specific media file.

        Args:
            path: Path to the file to remove.

        Returns:
            True if the file was removed.
        """
        key = str(path)
        if key in self._files:
            if path.exists():
                path.unlink()
            del self._files[key]
            return True
        return False

    def stats(self) -> dict:
        """Get media pipeline statistics.

        Returns:
            Dict with total files, size, expired count, etc.
        """
        total_size = sum(mf.size_bytes for mf in self._files.values())
        expired = sum(1 for mf in self._files.values() if mf.is_expired)
        mime_counts: dict[str, int] = {}
        for mf in self._files.values():
            mime_counts[mf.mime_type] = mime_counts.get(mf.mime_type, 0) + 1

        return {
            "total_files": len(self._files),
            "total_size_bytes": total_size,
            "expired_files": expired,
            "mime_types": mime_counts,
            "media_dir": str(self._media_dir),
            "max_size_bytes": self._max_size_bytes,
            "ttl_seconds": self._ttl_seconds,
        }

    def clear(self) -> int:
        """Remove all tracked media files.

        Returns:
            Number of files removed.
        """
        count = 0
        for media_file in self._files.values():
            if media_file.path.exists():
                media_file.path.unlink()
                count += 1
        self._files.clear()
        return count
