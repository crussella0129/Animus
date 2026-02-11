"""TTS engine subprocess management."""

from __future__ import annotations
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from src.audio.voice_profile import VoiceProfile

logger = logging.getLogger(__name__)


def find_tts_engine() -> Optional[Path]:
    """Auto-detect tts_engine.py location."""
    # 1. Check common locations
    candidates = [
        Path.home() / "TTS-Soundboard" / "python" / "tts_engine.py",
        Path("C:/Users") / Path.home().name / "TTS-Soundboard" / "python" / "tts_engine.py",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


class TTSEngine:
    """Manages persistent subprocess running tts_engine.py."""

    def __init__(self, engine_path: Path):
        self._engine_path = engine_path
        self._process: Optional[subprocess.Popen] = None
        self._restart_count = 0
        self._max_restarts = 3

    def start(self) -> bool:
        """Spawn subprocess and wait for ready signal."""
        try:
            self._process = subprocess.Popen(
                ["python", str(self._engine_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Wait for ready signal
            ready_line = self._process.stdout.readline()
            ready = json.loads(ready_line)

            if ready.get("ok"):
                logger.info(f"[Audio] TTS engine started (PID {self._process.pid})")
                return True
            else:
                logger.error(f"[Audio] TTS engine failed to start: {ready.get('error')}")
                return False

        except Exception as e:
            logger.error(f"[Audio] Failed to spawn TTS engine: {e}")
            return False

    def stop(self) -> None:
        """Send quit command and wait for clean exit."""
        if not self._process:
            return

        try:
            self._send_command({"cmd": "quit"})
            self._process.wait(timeout=5)
            logger.info("[Audio] TTS engine stopped cleanly")
        except Exception:
            self._process.kill()
            logger.warning("[Audio] TTS engine killed (unclean shutdown)")
        finally:
            self._process = None

    def synthesize_with_graph(
        self,
        text: str,
        profile: VoiceProfile,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Synthesize text with DSP processing.

        Returns path to output .wav file, or None on error.
        """
        if not self._is_alive():
            if not self._attempt_restart():
                return None

        # Generate output path if not provided
        if output_path is None:
            temp_dir = Path(tempfile.gettempdir()) / "animus_audio"
            temp_dir.mkdir(exist_ok=True)
            output_path = temp_dir / f"tts_{hash(text) & 0x7FFFFFFF}.wav"

        # Build command
        cmd = {
            "cmd": "process_graph",
            "text": text,
            "voice_id": profile.voice_id,
            "rate": profile.rate,
            "volume": profile.volume,
            "graph": profile.to_graph_command(),
            "output": str(output_path),
        }

        try:
            response = self._send_command(cmd)

            if response.get("ok"):
                wav_path = Path(response["output"])
                logger.debug(f"[Audio] Synthesized: '{text[:50]}...' -> {wav_path.name}")
                return wav_path
            else:
                logger.error(f"[Audio] Synthesis failed: {response.get('error')}")
                return None

        except Exception as e:
            logger.error(f"[Audio] Synthesis error: {e}")
            return None

    def _send_command(self, cmd: dict) -> dict:
        """Send JSON command and read JSON response."""
        self._process.stdin.write(json.dumps(cmd) + "\n")
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        return json.loads(response_line)

    def _is_alive(self) -> bool:
        """Check if subprocess is still running."""
        return self._process is not None and self._process.poll() is None

    def _attempt_restart(self) -> bool:
        """Attempt to restart crashed subprocess."""
        if self._restart_count >= self._max_restarts:
            logger.error("[Audio] Max restart attempts reached, giving up")
            return False

        self._restart_count += 1
        logger.warning(f"[Audio] Subprocess crashed, restarting (attempt {self._restart_count}/{self._max_restarts})")

        return self.start()
