"""Tests for the Media Pipeline module."""

import time
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.media import (
    MediaPipeline,
    MediaFile,
    DownloadResult,
    DownloadStatus,
    detect_mime_type,
    is_mime_allowed,
    is_extension_blocked,
    _sanitize_filename,
    BLOCKED_EXTENSIONS,
    ALLOWED_MIME_PREFIXES,
)


# =============================================================================
# MIME Detection Tests
# =============================================================================

class TestDetectMimeType:
    """Test MIME type detection."""

    def test_detect_png(self, tmp_path):
        """Detect PNG by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
        assert detect_mime_type(f) == "image/png"

    def test_detect_jpeg(self, tmp_path):
        """Detect JPEG by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 29)
        assert detect_mime_type(f) == "image/jpeg"

    def test_detect_gif87a(self, tmp_path):
        """Detect GIF87a by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"GIF87a" + b"\x00" * 26)
        assert detect_mime_type(f) == "image/gif"

    def test_detect_gif89a(self, tmp_path):
        """Detect GIF89a by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"GIF89a" + b"\x00" * 26)
        assert detect_mime_type(f) == "image/gif"

    def test_detect_pdf(self, tmp_path):
        """Detect PDF by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"%PDF-1.4" + b"\x00" * 24)
        assert detect_mime_type(f) == "application/pdf"

    def test_detect_zip(self, tmp_path):
        """Detect ZIP by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"PK\x03\x04" + b"\x00" * 28)
        assert detect_mime_type(f) == "application/zip"

    def test_detect_gzip(self, tmp_path):
        """Detect gzip by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"\x1f\x8b" + b"\x00" * 30)
        assert detect_mime_type(f) == "application/gzip"

    def test_detect_mp3_id3(self, tmp_path):
        """Detect MP3 with ID3 tag."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"ID3" + b"\x00" * 29)
        assert detect_mime_type(f) == "audio/mpeg"

    def test_detect_html(self, tmp_path):
        """Detect HTML by magic bytes."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"<!DOCTYPE html>" + b"\x00" * 17)
        assert detect_mime_type(f) == "text/html"

    def test_detect_webp(self, tmp_path):
        """Detect WebP (RIFF with WEBP marker)."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20)
        assert detect_mime_type(f) == "image/webp"

    def test_detect_wav(self, tmp_path):
        """Detect WAV (RIFF with WAVE marker)."""
        f = tmp_path / "test.dat"
        f.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20)
        assert detect_mime_type(f) == "audio/wav"

    def test_detect_by_extension(self, tmp_path):
        """Fall back to extension when magic bytes don't match."""
        f = tmp_path / "test.css"
        f.write_text("body { color: red; }")
        assert detect_mime_type(f) == "text/css"

    def test_detect_unknown_returns_octet_stream(self, tmp_path):
        """Unknown files return application/octet-stream."""
        f = tmp_path / "test.xyz123"
        f.write_bytes(b"\xde\xad\xbe\xef" + b"\x00" * 28)
        assert detect_mime_type(f) == "application/octet-stream"


# =============================================================================
# MIME Allowlist Tests
# =============================================================================

class TestMimeAllowlist:
    """Test MIME type allowlisting."""

    def test_text_allowed(self):
        assert is_mime_allowed("text/plain") is True
        assert is_mime_allowed("text/html") is True
        assert is_mime_allowed("text/css") is True

    def test_images_allowed(self):
        assert is_mime_allowed("image/png") is True
        assert is_mime_allowed("image/jpeg") is True

    def test_pdf_allowed(self):
        assert is_mime_allowed("application/pdf") is True

    def test_json_allowed(self):
        assert is_mime_allowed("application/json") is True

    def test_executable_blocked(self):
        assert is_mime_allowed("application/x-executable") is False
        assert is_mime_allowed("application/x-msdownload") is False


# =============================================================================
# Extension Blocklist Tests
# =============================================================================

class TestExtensionBlocklist:
    """Test file extension blocking."""

    def test_exe_blocked(self):
        assert is_extension_blocked("malware.exe") is True

    def test_bat_blocked(self):
        assert is_extension_blocked("script.bat") is True

    def test_ps1_blocked(self):
        assert is_extension_blocked("script.ps1") is True

    def test_sh_blocked(self):
        assert is_extension_blocked("script.sh") is True

    def test_dll_blocked(self):
        assert is_extension_blocked("library.dll") is True

    def test_pdf_not_blocked(self):
        assert is_extension_blocked("document.pdf") is False

    def test_png_not_blocked(self):
        assert is_extension_blocked("image.png") is False

    def test_txt_not_blocked(self):
        assert is_extension_blocked("readme.txt") is False

    def test_case_insensitive(self):
        assert is_extension_blocked("MALWARE.EXE") is True
        assert is_extension_blocked("Script.BAT") is True


# =============================================================================
# Filename Sanitization Tests
# =============================================================================

class TestSanitizeFilename:
    """Test URL-to-filename sanitization."""

    def test_basic_url(self):
        name = _sanitize_filename("https://example.com/file.pdf")
        assert name == "file.pdf"

    def test_url_with_path(self):
        name = _sanitize_filename("https://example.com/path/to/image.png")
        assert name == "image.png"

    def test_url_with_query(self):
        name = _sanitize_filename("https://example.com/file.pdf?v=123")
        assert name == "file.pdf"

    def test_url_encoded(self):
        name = _sanitize_filename("https://example.com/my%20file.pdf")
        assert "file" in name
        assert name.endswith(".pdf")

    def test_empty_path(self):
        name = _sanitize_filename("https://example.com/")
        assert name == "download"

    def test_unsafe_characters_removed(self):
        name = _sanitize_filename("https://example.com/../../etc/passwd")
        # Should not contain path traversal
        assert ".." not in name
        assert "/" not in name


# =============================================================================
# MediaFile Tests
# =============================================================================

class TestMediaFile:
    """Test MediaFile dataclass."""

    def test_is_expired_true(self):
        """Expired file should report as expired."""
        mf = MediaFile(
            path=Path("/tmp/test"),
            mime_type="text/plain",
            size_bytes=100,
            downloaded_at=time.time() - 200,
            ttl_expires=time.time() - 100,
        )
        assert mf.is_expired is True

    def test_is_expired_false(self):
        """Non-expired file should not report as expired."""
        mf = MediaFile(
            path=Path("/tmp/test"),
            mime_type="text/plain",
            size_bytes=100,
            downloaded_at=time.time(),
            ttl_expires=time.time() + 3600,
        )
        assert mf.is_expired is False

    def test_age_seconds(self):
        """age_seconds should reflect time since download."""
        mf = MediaFile(
            path=Path("/tmp/test"),
            mime_type="text/plain",
            size_bytes=100,
            downloaded_at=time.time() - 60,
            ttl_expires=time.time() + 3600,
        )
        assert mf.age_seconds >= 59  # Allow small timing variance


# =============================================================================
# MediaPipeline Tests
# =============================================================================

class TestMediaPipeline:
    """Test MediaPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a temporary media pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MediaPipeline(
                media_dir=Path(tmpdir),
                max_size_bytes=1024 * 1024,  # 1 MB
                ttl_seconds=60,
            )

    def test_init_creates_directory(self):
        """Pipeline should create media directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            media_dir = Path(tmpdir) / "subdir" / "media"
            pipeline = MediaPipeline(media_dir=media_dir)
            assert media_dir.exists()

    def test_stats_empty(self, pipeline):
        """Stats should work on empty pipeline."""
        stats = pipeline.stats()
        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0

    def test_detect_local_file(self, pipeline):
        """detect should identify local file MIME types."""
        f = pipeline.media_dir / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
        assert pipeline.detect(f) == "image/png"

    def test_cleanup_removes_expired(self, pipeline):
        """cleanup should remove expired files."""
        # Create a "downloaded" file with expired TTL
        f = pipeline.media_dir / "expired.txt"
        f.write_text("expired content")

        mf = MediaFile(
            path=f,
            mime_type="text/plain",
            size_bytes=15,
            downloaded_at=time.time() - 200,
            ttl_expires=time.time() - 100,  # Expired
        )
        pipeline._files[str(f)] = mf

        removed = pipeline.cleanup()

        assert len(removed) == 1
        assert not f.exists()

    def test_cleanup_keeps_valid(self, pipeline):
        """cleanup should keep non-expired files."""
        f = pipeline.media_dir / "valid.txt"
        f.write_text("valid content")

        mf = MediaFile(
            path=f,
            mime_type="text/plain",
            size_bytes=13,
            downloaded_at=time.time(),
            ttl_expires=time.time() + 3600,  # Not expired
        )
        pipeline._files[str(f)] = mf

        removed = pipeline.cleanup()

        assert len(removed) == 0
        assert f.exists()

    def test_remove_specific_file(self, pipeline):
        """remove should delete a specific tracked file."""
        f = pipeline.media_dir / "to_remove.txt"
        f.write_text("remove me")

        mf = MediaFile(
            path=f,
            mime_type="text/plain",
            size_bytes=9,
            downloaded_at=time.time(),
            ttl_expires=time.time() + 3600,
        )
        pipeline._files[str(f)] = mf

        result = pipeline.remove(f)

        assert result is True
        assert not f.exists()
        assert str(f) not in pipeline._files

    def test_remove_untracked_returns_false(self, pipeline):
        """remove should return False for untracked files."""
        result = pipeline.remove(Path("/nonexistent"))
        assert result is False

    def test_list_files(self, pipeline):
        """list_files should return tracked files."""
        f = pipeline.media_dir / "test.txt"
        f.write_text("test")

        mf = MediaFile(
            path=f,
            mime_type="text/plain",
            size_bytes=4,
            downloaded_at=time.time(),
            ttl_expires=time.time() + 3600,
        )
        pipeline._files[str(f)] = mf

        files = pipeline.list_files()
        assert len(files) == 1
        assert files[0].mime_type == "text/plain"

    def test_clear(self, pipeline):
        """clear should remove all tracked files."""
        for i in range(3):
            f = pipeline.media_dir / f"file_{i}.txt"
            f.write_text(f"content {i}")
            mf = MediaFile(
                path=f,
                mime_type="text/plain",
                size_bytes=9,
                downloaded_at=time.time(),
                ttl_expires=time.time() + 3600,
            )
            pipeline._files[str(f)] = mf

        count = pipeline.clear()

        assert count == 3
        assert len(pipeline._files) == 0

    def test_stats_populated(self, pipeline):
        """Stats should reflect tracked files."""
        for i in range(2):
            mf = MediaFile(
                path=Path(f"/tmp/file_{i}.txt"),
                mime_type="text/plain",
                size_bytes=100 * (i + 1),
                downloaded_at=time.time(),
                ttl_expires=time.time() + 3600,
            )
            pipeline._files[f"/tmp/file_{i}.txt"] = mf

        stats = pipeline.stats()
        assert stats["total_files"] == 2
        assert stats["total_size_bytes"] == 300
        assert stats["mime_types"]["text/plain"] == 2


# =============================================================================
# Download Tests (mocked network)
# =============================================================================

class TestMediaPipelineDownload:
    """Test download functionality with mocked network."""

    @pytest.fixture
    def pipeline(self):
        """Create a temporary media pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MediaPipeline(
                media_dir=Path(tmpdir),
                max_size_bytes=1024,
                ttl_seconds=60,
            )

    @pytest.mark.asyncio
    async def test_download_invalid_scheme(self, pipeline):
        """Should reject non-HTTP URLs."""
        result = await pipeline.download("ftp://example.com/file.txt")
        assert result.status == DownloadStatus.INVALID_URL

    @pytest.mark.asyncio
    async def test_download_blocked_extension(self, pipeline):
        """Should reject blocked extensions."""
        result = await pipeline.download("https://example.com/malware.exe")
        assert result.status == DownloadStatus.BLOCKED_TYPE
        assert ".exe" in result.error

    @pytest.mark.asyncio
    async def test_download_blocked_bat(self, pipeline):
        """Should reject .bat files."""
        result = await pipeline.download("https://example.com/script.bat")
        assert result.status == DownloadStatus.BLOCKED_TYPE

    @pytest.mark.asyncio
    async def test_download_blocked_ps1(self, pipeline):
        """Should reject .ps1 files."""
        result = await pipeline.download("https://example.com/script.ps1")
        assert result.status == DownloadStatus.BLOCKED_TYPE
