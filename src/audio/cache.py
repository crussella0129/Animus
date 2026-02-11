"""Audio cache for synthesized speech."""

from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from src.audio.voice_profile import VoiceProfile


class AudioCache:
    """Hash-based cache for TTS audio files."""

    def __init__(self, cache_dir: Path, max_size_mb: int = 100):
        self._cache_dir = cache_dir
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, text: str, profile: VoiceProfile) -> Optional[Path]:
        """Get cached audio file, or None if not found."""
        cache_key = self._compute_key(text, profile)
        cache_file = self._cache_dir / f"{cache_key}.wav"

        if cache_file.exists():
            # Update access time (for LRU)
            cache_file.touch()
            return cache_file

        return None

    def put(self, text: str, profile: VoiceProfile, wav_path: Path) -> None:
        """Store audio file in cache."""
        cache_key = self._compute_key(text, profile)
        cache_file = self._cache_dir / f"{cache_key}.wav"

        # Copy to cache
        shutil.copy2(wav_path, cache_file)

        # Prune if cache too large
        self._prune_if_needed()

    def clear(self) -> None:
        """Clear all cached audio files."""
        for file in self._cache_dir.glob("*.wav"):
            file.unlink()

    def _compute_key(self, text: str, profile: VoiceProfile) -> str:
        """Compute cache key from text and profile."""
        # Hash combines text + voice settings + DSP graph
        data = {
            "text": text,
            "voice_id": profile.voice_id,
            "rate": profile.rate,
            "volume": profile.volume,
            "pitch": profile.pitch,
            "graph": profile.graph,
        }

        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def _prune_if_needed(self) -> None:
        """Remove oldest files if cache exceeds size limit."""
        total_size = sum(f.stat().st_size for f in self._cache_dir.glob("*.wav"))

        if total_size <= self._max_size_bytes:
            return

        # Sort by access time (oldest first)
        files = sorted(
            self._cache_dir.glob("*.wav"),
            key=lambda f: f.stat().st_atime
        )

        # Remove oldest files until under limit
        for file in files:
            file.unlink()
            total_size -= file.stat().st_size
            if total_size <= self._max_size_bytes:
                break
