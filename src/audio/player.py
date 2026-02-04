"""
Cross-platform audio playback for Animus.

Uses:
- WAV file playback for praise music (Mozart/Bach)
- pyttsx3 for text-to-speech with customizable voices

Platform support:
- Windows: PowerShell Media.SoundPlayer + SAPI5
- macOS: afplay + NSSpeechSynthesizer
- Linux: aplay/paplay + espeak
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any


class AudioPlayer:
    """
    Cross-platform audio player for WAV files and TTS.

    Handles praise music playback and speech synthesis.
    """

    def __init__(self, assets_dir: Optional[Path] = None, volume: float = 0.7):
        """
        Initialize audio player.

        Args:
            assets_dir: Directory containing WAV assets (default: src/audio/assets)
            volume: Master volume 0.0-1.0 (used for TTS)
        """
        self.volume = volume

        # Find assets directory
        if assets_dir:
            self._assets_dir = assets_dir
        else:
            self._assets_dir = Path(__file__).parent / "assets"

        # Detect platform audio command
        self._playback_cmd = self._detect_audio_command()

        # Initialize TTS engine (lazy load)
        self._tts_engine = None
        self._tts_available = None
        self._tts_lock = threading.Lock()

    def _detect_audio_command(self) -> Optional[List[str]]:
        """Detect platform-specific audio playback command."""
        if sys.platform == "win32":
            return ["powershell", "-c"]
        elif sys.platform == "darwin":
            if self._command_exists("afplay"):
                return ["afplay"]
        else:
            # Linux: Try aplay (ALSA) or paplay (PulseAudio)
            if self._command_exists("aplay"):
                return ["aplay", "-q"]
            elif self._command_exists("paplay"):
                return ["paplay"]
        return None

    def _command_exists(self, cmd: str) -> bool:
        """Check if a command exists in PATH."""
        try:
            subprocess.run(
                [cmd, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def is_available(self) -> bool:
        """Check if audio playback is available."""
        return self._playback_cmd is not None

    def _init_tts(self) -> bool:
        """Initialize TTS engine. Returns True if successful."""
        if self._tts_available is not None:
            return self._tts_available

        with self._tts_lock:
            if self._tts_available is not None:
                return self._tts_available

            try:
                import pyttsx3
                self._tts_engine = pyttsx3.init()

                # Configure for "arcade boss" voice
                # David at rate 150
                self._tts_engine.setProperty('rate', 150)

                # Try to find a deeper/more robotic voice
                voices = self._tts_engine.getProperty('voices')

                # On Windows, look for "David" (deeper male voice)
                # On Mac, look for deeper voices
                # On Linux/espeak, default is already robotic
                if sys.platform == "win32" and voices:
                    for voice in voices:
                        if "david" in voice.name.lower():
                            self._tts_engine.setProperty('voice', voice.id)
                            break
                elif sys.platform == "darwin" and voices:
                    # Try to find a deeper voice
                    for voice in voices:
                        name_lower = voice.name.lower()
                        if any(x in name_lower for x in ["alex", "bruce", "fred"]):
                            self._tts_engine.setProperty('voice', voice.id)
                            break

                self._tts_available = True
            except Exception:
                self._tts_engine = None
                self._tts_available = False

        return self._tts_available

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available TTS voices."""
        if not self._init_tts():
            return []

        try:
            voices = self._tts_engine.getProperty('voices')
            return [
                {"id": v.id, "name": v.name, "languages": getattr(v, 'languages', [])}
                for v in voices
            ]
        except Exception:
            return []

    def set_voice(self, voice_id: str) -> bool:
        """Set TTS voice by ID."""
        if not self._init_tts():
            return False

        try:
            self._tts_engine.setProperty('voice', voice_id)
            return True
        except Exception:
            return False

    def set_speech_rate(self, rate: int) -> bool:
        """Set TTS speech rate (words per minute, default ~150)."""
        if not self._init_tts():
            return False

        try:
            self._tts_engine.setProperty('rate', rate)
            return True
        except Exception:
            return False

    def play_wav(self, wav_path: Path, blocking: bool = True) -> bool:
        """
        Play a WAV file.

        Args:
            wav_path: Path to WAV file
            blocking: Wait for playback to complete

        Returns:
            True if playback started successfully
        """
        if not self.is_available():
            return False

        if not wav_path.exists():
            return False

        try:
            if sys.platform == "win32":
                ps_cmd = f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"
                if blocking:
                    subprocess.run(
                        self._playback_cmd + [ps_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60
                    )
                else:
                    subprocess.Popen(
                        self._playback_cmd + [ps_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            else:
                if blocking:
                    subprocess.run(
                        self._playback_cmd + [str(wav_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60
                    )
                else:
                    subprocess.Popen(
                        self._playback_cmd + [str(wav_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            return True
        except Exception:
            return False

    def play_praise(self, mode: str) -> bool:
        """
        Play praise music.

        Args:
            mode: "fanfare" (Mozart) or "sophisticated" (Bach)

        Returns:
            True if playback started
        """
        if mode == "fanfare":
            wav_file = self._assets_dir / "mozart_eine_kleine.wav"
        elif mode == "sophisticated":
            wav_file = self._assets_dir / "bach_invention13.wav"
        else:
            return False

        if not wav_file.exists():
            # Asset not installed yet
            return False

        return self.play_wav(wav_file, blocking=True)

    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            blocking: Wait for speech to complete

        Returns:
            True if speech started
        """
        if not self._init_tts():
            return False

        try:
            if blocking:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            else:
                # Run in background thread
                thread = threading.Thread(
                    target=self._speak_thread,
                    args=(text,),
                    daemon=True
                )
                thread.start()
            return True
        except Exception:
            return False

    def _speak_thread(self, text: str) -> None:
        """Background thread for non-blocking speech."""
        try:
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception:
            pass

    def speak_phrase(self, phrase_key: str, blocking: bool = True) -> bool:
        """
        Speak a predefined phrase.

        Args:
            phrase_key: One of the predefined phrases
            blocking: Wait for speech to complete

        Returns:
            True if speech started
        """
        phrases = {
            "yes_master": "Yes, Master.",
            "it_will_be_done": "It will be done.",
            "working": "Working.",
            "complete": "Complete.",
            "acknowledged": "Acknowledged.",
            "as_you_wish": "As you wish.",
        }

        text = phrases.get(phrase_key)
        if not text:
            return False

        return self.speak(text, blocking=blocking)


# Convenience functions
_default_player: Optional[AudioPlayer] = None


def get_player() -> AudioPlayer:
    """Get or create the default audio player."""
    global _default_player
    if _default_player is None:
        _default_player = AudioPlayer()
    return _default_player


def play_praise(mode: str) -> bool:
    """Play praise music (fanfare or sophisticated)."""
    return get_player().play_praise(mode)


def speak(text: str, blocking: bool = True) -> bool:
    """Speak text using TTS."""
    return get_player().speak(text, blocking)


def speak_phrase(phrase_key: str, blocking: bool = True) -> bool:
    """Speak a predefined phrase."""
    return get_player().speak_phrase(phrase_key, blocking)
