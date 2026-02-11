"""TTS audio system for Animus."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from src.audio.cache import AudioCache
from src.audio.engine import TTSEngine
from src.audio.player import AudioPlayer
from src.audio.voice_profile import VoiceProfile

logger = logging.getLogger(__name__)

# Global state
_engine: Optional[TTSEngine] = None
_player: Optional[AudioPlayer] = None
_cache: Optional[AudioCache] = None
_profile: Optional[VoiceProfile] = None
_title_spoken = False  # Track if startup title was played


def initialize(
    tts_engine_path: Path,
    voice_profile_path: Path,
    cache_dir: Path,
) -> bool:
    """
    Initialize TTS subsystem.

    Returns True on success, False on failure.
    """
    global _engine, _player, _cache, _profile, _title_spoken

    try:
        # Load voice profile
        logger.info(f"[Audio] Loading voice profile: {voice_profile_path}")
        _profile = VoiceProfile.load(voice_profile_path)

        # Log DSP chain
        nodes = _profile.graph.get("nodes", [])
        dsp_chain = " -> ".join([n["type"] for n in nodes if n["type"] not in ("input", "output")])
        logger.info(f"[Audio] DSP chain: {dsp_chain}")

        # Initialize engine
        logger.info("[Audio] Starting TTS engine...")
        _engine = TTSEngine(tts_engine_path)
        if not _engine.start():
            logger.error("[Audio] Failed to start TTS engine")
            return False

        # Initialize player and cache
        _player = AudioPlayer()
        _cache = AudioCache(cache_dir)
        _title_spoken = False

        logger.info("[Audio] TTS system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"[Audio] Initialization failed: {e}")
        return False


def speak(
    text: str,
    blocking: bool = False,
    prepend_title: Optional[str] = None,
) -> bool:
    """
    Speak text using loaded voice profile.

    Args:
        text: Text to synthesize
        blocking: If True, wait for audio to finish
        prepend_title: Optional title to prepend ("Yes father,")

    Returns True if audio played, False on error.
    """
    global _engine, _player, _cache, _profile

    if not all([_engine, _player, _cache, _profile]):
        logger.warning("[Audio] TTS not initialized, skipping")
        return False

    # Prepend title if provided
    if prepend_title:
        text = f"{prepend_title}, {text}"

    # Check cache
    cached_audio = _cache.get(text, _profile)
    if cached_audio:
        logger.debug(f"[Audio] Cache hit: {cached_audio.name}")
        _player.play(cached_audio, blocking=blocking)
        return True

    # Synthesize
    logger.debug(f"[Audio] Synthesizing: '{text[:50]}...'")
    wav_path = _engine.synthesize_with_graph(text, _profile)

    if wav_path is None:
        logger.error("[Audio] Synthesis failed")
        return False

    # Cache and play
    _cache.put(text, _profile, wav_path)
    _player.play(wav_path, blocking=blocking)
    return True


def speak_title_greeting(title_text: str) -> bool:
    """
    Speak startup title greeting ("Yes, father").

    Only plays once per session. Subsequent calls are ignored.
    """
    global _title_spoken

    if _title_spoken:
        return False

    if not title_text:
        return False

    _title_spoken = True
    return speak(f"Yes, {title_text}", blocking=True)


def should_prepend_title(
    title_mode: str,
    title_text: str,
    is_first_response: bool,
) -> Optional[str]:
    """
    Determine if title should be prepended to this response.

    Returns title string to prepend, or None.
    """
    if not title_text:
        return None

    if title_mode == "always":
        return f"Yes {title_text}"
    elif title_mode == "first" and is_first_response:
        return f"Yes {title_text}"
    elif title_mode == "startup":
        return None  # Handled separately
    elif title_mode == "never":
        return None

    return None


def shutdown() -> None:
    """Shutdown TTS engine and cleanup resources."""
    global _engine, _player, _cache, _profile

    if _engine:
        logger.info("[Audio] Shutting down TTS engine")
        _engine.stop()

    _engine = None
    _player = None
    _cache = None
    _profile = None


__all__ = [
    "initialize",
    "speak",
    "speak_title_greeting",
    "should_prepend_title",
    "shutdown",
]
