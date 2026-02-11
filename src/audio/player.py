"""Platform-specific audio playback."""

from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path
from threading import Thread

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Non-blocking audio playback using platform-specific tools."""

    def __init__(self):
        self._platform = sys.platform

    def play(self, wav_path: Path, blocking: bool = False) -> None:
        """Play audio file (blocking or non-blocking)."""
        if not wav_path.exists():
            logger.error(f"[Audio] File not found: {wav_path}")
            return

        if blocking:
            self._play_sync(wav_path)
        else:
            Thread(target=self._play_sync, args=(wav_path,), daemon=True).start()

    def _play_sync(self, wav_path: Path) -> None:
        """Synchronous playback (blocks until audio finishes)."""
        try:
            if self._platform == "win32":
                self._play_windows(wav_path)
            elif self._platform == "darwin":
                self._play_macos(wav_path)
            else:
                self._play_linux(wav_path)
        except Exception as e:
            logger.error(f"[Audio] Playback failed: {e}")

    def _play_windows(self, wav_path: Path) -> None:
        """Windows: use PowerShell SoundPlayer."""
        subprocess.run(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _play_macos(self, wav_path: Path) -> None:
        """macOS: use afplay."""
        subprocess.run(["afplay", str(wav_path)], check=True)

    def _play_linux(self, wav_path: Path) -> None:
        """Linux: try aplay (ALSA) or paplay (PulseAudio)."""
        try:
            subprocess.run(["aplay", str(wav_path)], check=True)
        except FileNotFoundError:
            subprocess.run(["paplay", str(wav_path)], check=True)
