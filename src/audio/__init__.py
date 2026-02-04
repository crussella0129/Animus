"""
Animus Audio Interface

Provides audio feedback for commands and task completion:
- Speech synthesis via pyttsx3 (cross-platform TTS)
- Task completion music via WAV playback (Mozart fanfare, Bach sophisticated)
"""

from .config import PraiseMode
from .player import (
    AudioPlayer,
    get_player,
    play_praise,
    speak,
    speak_phrase,
)

__all__ = [
    "PraiseMode",
    "AudioPlayer",
    "get_player",
    "play_praise",
    "speak",
    "speak_phrase",
]
