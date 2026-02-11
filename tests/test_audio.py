"""Tests for audio TTS system."""

import pytest
from pathlib import Path
from src.audio.voice_profile import VoiceProfile
from src.audio.cache import AudioCache
from src.audio.engine import TTSEngine, find_tts_engine


class TestVoiceProfile:
    def test_load_valid_profile(self, tmp_path):
        """Test loading a valid voice profile."""
        profile_data = {
            "name": "test",
            "systemVoiceId": "test_voice",
            "rate": 150,
            "volume": 1.0,
            "pitch": 0,
            "graph": {
                "nodes": [
                    {"id": "input", "type": "input", "params": {}},
                    {"id": "output", "type": "output", "params": {}},
                ],
                "edges": [{"source": "input", "target": "output"}]
            }
        }

        profile_file = tmp_path / "test.json"
        import json
        with open(profile_file, "w") as f:
            json.dump(profile_data, f)

        profile = VoiceProfile.load(profile_file)
        assert profile.name == "test"
        assert profile.rate == 150

    def test_load_missing_field(self, tmp_path):
        """Test that loading fails if required field missing."""
        profile_data = {"name": "incomplete"}

        profile_file = tmp_path / "bad.json"
        import json
        with open(profile_file, "w") as f:
            json.dump(profile_data, f)

        with pytest.raises(ValueError, match="Missing required field"):
            VoiceProfile.load(profile_file)

    def test_to_graph_command(self):
        """Test DSP graph command conversion."""
        profile = VoiceProfile(
            name="test",
            voice_id="test",
            rate=150,
            volume=1.0,
            pitch=0,
            graph={
                "nodes": [
                    {"id": "input", "type": "input", "params": {}},
                    {"id": "pitch", "type": "pitch_shift", "params": {"semitones": -5}},
                    {"id": "output", "type": "output", "params": {}},
                ],
                "edges": [
                    {"source": "input", "target": "pitch"},
                    {"source": "pitch", "target": "output"}
                ]
            }
        )

        cmd = profile.to_graph_command()
        # Should filter out input/output nodes
        assert len(cmd["nodes"]) == 1
        assert cmd["nodes"][0]["type"] == "pitch_shift"


class TestAudioCache:
    def test_cache_hit(self, tmp_path):
        """Test cache retrieves previously stored audio."""
        cache = AudioCache(tmp_path, max_size_mb=10)

        # Create dummy profile
        profile = VoiceProfile(
            name="test",
            voice_id="test",
            rate=150,
            volume=1.0,
            pitch=0,
            graph={"nodes": [], "edges": []}
        )

        # Create dummy wav file
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"FAKE_WAV")

        # Store in cache
        cache.put("hello world", profile, wav_file)

        # Retrieve
        cached = cache.get("hello world", profile)
        assert cached is not None
        assert cached.exists()

    def test_cache_miss(self, tmp_path):
        """Test cache returns None for uncached text."""
        cache = AudioCache(tmp_path, max_size_mb=10)

        profile = VoiceProfile(
            name="test",
            voice_id="test",
            rate=150,
            volume=1.0,
            pitch=0,
            graph={"nodes": [], "edges": []}
        )

        result = cache.get("not cached", profile)
        assert result is None

    def test_cache_key_uniqueness(self, tmp_path):
        """Test that cache keys are unique for different settings."""
        cache = AudioCache(tmp_path, max_size_mb=10)

        profile1 = VoiceProfile(
            name="test",
            voice_id="test",
            rate=150,
            volume=1.0,
            pitch=0,
            graph={"nodes": [], "edges": []}
        )

        profile2 = VoiceProfile(
            name="test",
            voice_id="test",
            rate=200,  # Different rate
            volume=1.0,
            pitch=0,
            graph={"nodes": [], "edges": []}
        )

        # Store with profile1
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"FAKE_WAV")
        cache.put("hello", profile1, wav_file)

        # Retrieve with profile2 should miss
        result = cache.get("hello", profile2)
        assert result is None


@pytest.mark.skipif(find_tts_engine() is None, reason="TTS engine not found")
class TestTTSEngine:
    def test_engine_lifecycle(self):
        """Test engine starts and stops cleanly."""
        engine_path = find_tts_engine()
        engine = TTSEngine(engine_path)

        assert engine.start()
        assert engine._is_alive()

        engine.stop()
        assert not engine._is_alive()

    def test_synthesize_basic(self, tmp_path):
        """Test basic synthesis without DSP."""
        engine_path = find_tts_engine()
        engine = TTSEngine(engine_path)
        engine.start()

        profile = VoiceProfile(
            name="test",
            voice_id="",  # Use default
            rate=150,
            volume=1.0,
            pitch=0,
            graph={"nodes": [], "edges": []}
        )

        output = tmp_path / "output.wav"
        result = engine.synthesize_with_graph("test", profile, output)

        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 1000  # Has audio data

        engine.stop()
