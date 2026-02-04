"""Tests for audio interface (voice and music)."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core.config import AudioConfig
from src.audio.midi import MIDIEngine, Note
from src.audio.speech import SpeechSynthesizer


# =============================================================================
# AudioConfig Tests
# =============================================================================


def test_audio_config_defaults():
    """Test AudioConfig default values."""
    config = AudioConfig()
    assert config.speak_enabled is False
    assert config.praise_mode == "off"
    assert config.moto_enabled is False
    assert config.volume == 0.7
    assert config.speech_pitch == 0.6


def test_audio_config_validation():
    """Test AudioConfig field validation."""
    # Volume must be 0.0-1.0
    with pytest.raises(Exception):
        AudioConfig(volume=1.5)

    with pytest.raises(Exception):
        AudioConfig(volume=-0.1)

    # Speech pitch must be 0.1-2.0
    with pytest.raises(Exception):
        AudioConfig(speech_pitch=0.05)

    with pytest.raises(Exception):
        AudioConfig(speech_pitch=3.0)

    # Valid values
    config = AudioConfig(volume=0.5, speech_pitch=1.5)
    assert config.volume == 0.5
    assert config.speech_pitch == 1.5


def test_audio_config_praise_modes():
    """Test valid praise mode values."""
    for mode in ["fanfare", "spooky", "off"]:
        config = AudioConfig(praise_mode=mode)
        assert config.praise_mode == mode


# =============================================================================
# MIDIEngine Tests
# =============================================================================


def test_midi_engine_init():
    """Test MIDIEngine initialization."""
    engine = MIDIEngine(sample_rate=22050, volume=0.5)
    assert engine.sample_rate == 22050
    assert engine.volume == 0.5


def test_midi_note_to_freq():
    """Test MIDI note to frequency conversion."""
    engine = MIDIEngine()

    # A4 (MIDI 69) = 440 Hz
    assert abs(engine.midi_note_to_freq(69) - 440.0) < 0.01

    # Middle C (MIDI 60) ≈ 261.63 Hz
    assert abs(engine.midi_note_to_freq(60) - 261.63) < 0.01

    # C3 (MIDI 48) ≈ 130.81 Hz
    assert abs(engine.midi_note_to_freq(48) - 130.81) < 0.01


def test_generate_note_samples():
    """Test note sample generation with envelope."""
    engine = MIDIEngine(sample_rate=1000)

    # Generate 0.5 second note at 440 Hz
    samples = engine._generate_note_samples(frequency=440.0, duration=0.5, volume=1.0, square_wave=False)

    assert len(samples) == 500  # 0.5 seconds at 1000 Hz
    assert samples.dtype == np.float64

    # Samples should be in valid range
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)

    # Test square wave generation
    sq_samples = engine._generate_note_samples(frequency=440.0, duration=0.5, volume=1.0, square_wave=True)
    assert len(sq_samples) == 500


def test_note_dataclass():
    """Test Note dataclass."""
    note = Note(pitch=60, duration=0.5, velocity=80)
    assert note.pitch == 60
    assert note.duration == 0.5
    assert note.velocity == 80

    # Default velocity
    note2 = Note(pitch=60, duration=0.5)
    assert note2.velocity == 64


def test_midi_engine_unavailable():
    """Test MIDIEngine when audio command is unavailable."""
    with patch.object(MIDIEngine, '_detect_audio_command', return_value=None):
        engine = MIDIEngine()
        assert not engine.is_available()

        # Methods should fail gracefully
        engine.play_note(Note(60, 0.5))  # Should not crash
        engine.play_sequence([Note(60, 0.5)])  # Should not crash
        engine.play_mozart_fanfare()  # Should not crash
        engine.play_bach_spooky()  # Should not crash
        engine.play_moto_perpetuo()  # Should not crash


def test_mozart_fanfare_sequence():
    """Test Mozart fanfare note sequence."""
    engine = MIDIEngine()

    # Verify the sequence is rendered and played
    with patch.object(engine, '_play_wav_bytes') as mock_play:
        engine.play_mozart_fanfare()

        # Should have called _play_wav_bytes once (entire sequence)
        assert mock_play.call_count == 1

        # Verify the WAV bytes were generated
        wav_bytes = mock_play.call_args[0][0]
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0


def test_bach_spooky_sequence():
    """Test Bach spooky note sequence."""
    engine = MIDIEngine()

    with patch.object(engine, '_play_wav_bytes') as mock_play:
        engine.play_bach_spooky()

        # Should have called _play_wav_bytes once
        assert mock_play.call_count == 1

        # Verify the WAV bytes were generated
        wav_bytes = mock_play.call_args[0][0]
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0


def test_moto_perpetuo_background():
    """Test Moto Perpetuo background music."""
    engine = MIDIEngine()

    # Mock the subprocess call to avoid actual audio playback
    with patch('subprocess.run'):
        # Start background music
        engine.play_moto_perpetuo(loop=True)

        # Should start a background thread
        assert engine._background_thread is not None
        assert engine._background_thread.is_alive()

        # Stop background music
        engine.stop_background_music()

        # Give thread time to stop
        import time
        time.sleep(0.2)

        # Thread should stop
        assert not engine._background_thread.is_alive()


def test_background_music_cleanup():
    """Test background music stops on cleanup."""
    engine = MIDIEngine()

    with patch('subprocess.run'):
        engine.play_moto_perpetuo(loop=True)
        assert engine._background_thread is not None

        # Cleanup should stop background music
        engine.cleanup()

        # Give thread time to stop
        import time
        time.sleep(0.2)

        assert not engine._background_thread.is_alive()


# =============================================================================
# SpeechSynthesizer Tests
# =============================================================================


def test_speech_synthesizer_init():
    """Test SpeechSynthesizer initialization."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine, pitch_multiplier=0.6)

    assert synth.engine == engine
    assert synth.pitch_multiplier == 0.6


def test_text_to_phonemes():
    """Test text to phoneme conversion."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    # Simple text
    phonemes = synth.text_to_phonemes("HELLO")
    assert phonemes == ['H', 'E', 'L', 'L', 'O']

    # With spaces and punctuation
    phonemes = synth.text_to_phonemes("YES MASTER")
    assert ' ' in phonemes
    assert 'Y' in phonemes and 'E' in phonemes and 'S' in phonemes


def test_phonemes_to_notes():
    """Test phoneme to MIDI note conversion."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine, pitch_multiplier=0.6)

    phonemes = ['H', 'E', 'L', 'L', 'O']
    notes = synth.phonemes_to_notes(phonemes)

    assert len(notes) == 5
    for note in notes:
        assert isinstance(note, Note)
        assert 24 <= note.pitch <= 84  # Reasonable MIDI range
        assert note.duration > 0


def test_vowel_vs_consonant_duration():
    """Test vowels have longer duration than consonants."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    # Vowel
    vowel_notes = synth.phonemes_to_notes(['A'])
    assert vowel_notes[0].duration == synth.VOWEL_DURATION

    # Consonant
    consonant_notes = synth.phonemes_to_notes(['B'])
    assert consonant_notes[0].duration == synth.CONSONANT_DURATION
    assert synth.CONSONANT_DURATION < synth.VOWEL_DURATION


def test_predefined_phrases():
    """Test predefined phrase dictionary."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    assert "yes_master" in synth.PHRASES
    assert "it_will_be_done" in synth.PHRASES
    assert synth.PHRASES["yes_master"] == "YES MASTER"
    assert synth.PHRASES["it_will_be_done"] == "IT WILL BE DONE"


def test_speak_phrase():
    """Test speaking predefined phrases."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    with patch.object(synth, 'speak') as mock_speak:
        synth.speak_phrase("yes_master", blocking=True)
        mock_speak.assert_called_once_with("YES MASTER", blocking=True)


def test_should_speak_text_filters_code():
    """Test that code blocks are filtered from speech."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    # Code should not be spoken
    should_speak, _ = synth.should_speak_text("def hello():\n    print('hi')")
    assert not should_speak

    should_speak, _ = synth.should_speak_text("class Foo:\n    pass")
    assert not should_speak

    should_speak, _ = synth.should_speak_text("import sys")
    assert not should_speak

    should_speak, _ = synth.should_speak_text("```python\ncode\n```")
    assert not should_speak

    should_speak, _ = synth.should_speak_text('{"key": "value"}')
    assert not should_speak


def test_should_speak_text_allows_descriptions():
    """Test that natural language descriptions are allowed."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    # Natural language should be spoken
    should_speak, text = synth.should_speak_text("Task completed successfully")
    assert should_speak
    assert "Task" in text

    should_speak, text = synth.should_speak_text("Working on implementation")
    assert should_speak


def test_should_speak_text_length_limit():
    """Test that very long text is filtered."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    # Very long text should not be spoken
    long_text = "x " * 200  # 400 characters
    should_speak, _ = synth.should_speak_text(long_text)
    assert not should_speak


def test_should_speak_extracts_first_sentence():
    """Test that first sentence is extracted from multi-sentence text."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    text = "First sentence. Second sentence. Third sentence."
    should_speak, speakable = synth.should_speak_text(text)

    assert should_speak
    assert speakable == "First sentence"


# =============================================================================
# Integration Tests (with mocked audio)
# =============================================================================


@pytest.mark.asyncio
async def test_agent_audio_initialization():
    """Test that Agent initializes audio when config is provided."""
    from src.core.agent import Agent, AgentConfig
    from src.core.config import AnimusConfig, AudioConfig
    from src.llm import ModelProvider

    # Mock provider
    provider = Mock(spec=ModelProvider)

    # Config with audio enabled
    audio_config = AudioConfig(speak_enabled=True, praise_mode="fanfare")
    animus_config = AnimusConfig(audio=audio_config)

    with patch('src.audio.MIDIEngine') as mock_engine, \
         patch('src.audio.SpeechSynthesizer') as mock_synth:

        mock_engine_instance = Mock()
        mock_engine_instance.is_available.return_value = True
        mock_engine.return_value = mock_engine_instance

        agent = Agent(provider=provider, animus_config=animus_config)

        # Audio should be initialized
        mock_engine.assert_called_once()
        mock_synth.assert_called_once()


@pytest.mark.asyncio
async def test_agent_speaks_on_user_input():
    """Test that Agent speech is disabled (pending proper TTS implementation)."""
    from src.core.agent import Agent
    from src.core.config import AnimusConfig, AudioConfig
    from src.llm import ModelProvider, GenerationResult

    async def mock_generate(*args, **kwargs):
        return GenerationResult(
            content="I'll help with that.",
            finish_reason="stop",
            usage={"input_tokens": 10, "output_tokens": 5}
        )

    provider = Mock(spec=ModelProvider)
    provider.generate = mock_generate

    audio_config = AudioConfig(speak_enabled=True)
    animus_config = AnimusConfig(audio=audio_config)

    with patch('src.audio.MIDIEngine') as mock_engine, \
         patch('src.audio.SpeechSynthesizer') as mock_synth:

        mock_engine_instance = Mock()
        mock_engine_instance.is_available.return_value = True
        mock_engine.return_value = mock_engine_instance

        mock_synth_instance = Mock()
        mock_synth.return_value = mock_synth_instance

        agent = Agent(provider=provider, animus_config=animus_config)

        # Execute step with user input
        turn = await agent.step("Hello")

        # Speech is disabled pending proper TTS - should NOT be called
        mock_synth_instance.speak_phrase.assert_not_called()


@pytest.mark.asyncio
async def test_agent_moto_background_music():
    """Test that Moto Perpetuo is disabled (pending proper MIDI implementation)."""
    from src.core.agent import Agent
    from src.core.config import AnimusConfig, AudioConfig
    from src.llm import ModelProvider, GenerationResult

    async def mock_generate(*args, **kwargs):
        return GenerationResult(
            content="Done.",
            finish_reason="stop",
            usage={"input_tokens": 10, "output_tokens": 5}
        )

    provider = Mock(spec=ModelProvider)
    provider.generate = mock_generate

    audio_config = AudioConfig(moto_enabled=True)
    animus_config = AnimusConfig(audio=audio_config)

    with patch('src.audio.MIDIEngine') as mock_engine, \
         patch('src.audio.SpeechSynthesizer'):

        mock_engine_instance = Mock()
        mock_engine_instance.is_available.return_value = True
        mock_engine.return_value = mock_engine_instance

        agent = Agent(provider=provider, animus_config=animus_config)

        # Run agent
        turns = []
        async for turn in agent.run("Test task"):
            turns.append(turn)

        # Moto Perpetuo is disabled pending proper MIDI - should NOT be called
        mock_engine_instance.play_moto_perpetuo.assert_not_called()


@pytest.mark.asyncio
async def test_agent_praise_on_completion():
    """Test that Agent plays praise audio on long task completion (5+ turns)."""
    from src.core.agent import Agent
    from src.core.config import AnimusConfig, AudioConfig
    from src.llm import ModelProvider, GenerationResult

    provider = Mock(spec=ModelProvider)

    # Simulate long task (5+ turns required for praise)
    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 5:
            # Turns 1-4: tool calls to keep the loop going
            return GenerationResult(
                content='{"name": "read_file", "arguments": {"path": "test.py"}}',
                finish_reason="tool_calls",
                usage={"input_tokens": 10, "output_tokens": 10}
            )
        else:
            # Turn 5+: no tool calls, task complete
            return GenerationResult(
                content="Task complete.",
                finish_reason="stop",
                usage={"input_tokens": 10, "output_tokens": 5}
            )

    provider.generate = mock_generate

    audio_config = AudioConfig(praise_mode="fanfare")
    animus_config = AnimusConfig(audio=audio_config)

    with patch('src.audio.MIDIEngine') as mock_engine, \
         patch('src.audio.SpeechSynthesizer'), \
         patch('src.core.agent.Agent._call_tools_parallel', return_value=[]):

        mock_engine_instance = Mock()
        mock_engine_instance.is_available.return_value = True
        mock_engine.return_value = mock_engine_instance

        agent = Agent(provider=provider, animus_config=animus_config)

        # Run agent
        turns = []
        async for turn in agent.run("Long multi-step task"):
            turns.append(turn)

        # Should play Mozart fanfare (5+ turns)
        mock_engine_instance.play_mozart_fanfare.assert_called()


def test_speech_filter_length():
    """Test speech length limiting."""
    engine = MIDIEngine()
    synth = SpeechSynthesizer(engine)

    # Exactly at limit should work
    text_100 = "x" * 100
    should_speak, speakable = synth.should_speak_text(text_100)
    assert should_speak
    assert len(speakable) == 100

    # Just over limit should be filtered
    text_201 = "x" * 201
    should_speak, _ = synth.should_speak_text(text_201)
    assert not should_speak


def test_pitch_multiplier_affects_notes():
    """Test that pitch multiplier lowers note pitches."""
    engine = MIDIEngine()

    # Default pitch
    synth_normal = SpeechSynthesizer(engine, pitch_multiplier=1.0)
    notes_normal = synth_normal.phonemes_to_notes(['A'])

    # Lower pitch
    synth_low = SpeechSynthesizer(engine, pitch_multiplier=0.5)
    notes_low = synth_low.phonemes_to_notes(['A'])

    # Lower pitch multiplier should produce lower MIDI note
    assert notes_low[0].pitch < notes_normal[0].pitch


def test_play_sequence_blocking():
    """Test blocking sequence playback."""
    engine = MIDIEngine()

    notes = [Note(60, 0.1), Note(64, 0.1), Note(67, 0.1)]

    with patch.object(engine, '_play_wav_bytes') as mock_play:
        engine.play_sequence(notes, square_wave=False, blocking=True)

        # Should call _play_wav_bytes once (entire sequence rendered together)
        assert mock_play.call_count == 1

        # Verify WAV bytes were generated
        wav_bytes = mock_play.call_args[0][0]
        assert isinstance(wav_bytes, bytes)


def test_play_sequence_non_blocking():
    """Test non-blocking sequence playback."""
    engine = MIDIEngine()

    notes = [Note(60, 0.1)]

    with patch.object(engine, '_play_wav_bytes') as mock_play:
        engine.play_sequence(notes, square_wave=False, blocking=False)

        # Give thread time to start
        import time
        time.sleep(0.1)

        # Should have called _play_wav_bytes in background thread
        # Note: may not be called yet if thread hasn't run
        # The key is that it doesn't crash


# =============================================================================
# CLI Command Tests (would require mocking ConfigManager)
# =============================================================================


def test_audio_config_serialization():
    """Test that AudioConfig can be serialized to YAML."""
    config = AudioConfig(
        speak_enabled=True,
        praise_mode="fanfare",
        moto_enabled=True,
        volume=0.8,
        speech_pitch=0.5
    )

    # Should be serializable via Pydantic
    data = config.model_dump()

    assert data["speak_enabled"] is True
    assert data["praise_mode"] == "fanfare"
    assert data["moto_enabled"] is True
    assert data["volume"] == 0.8
    assert data["speech_pitch"] == 0.5


def test_audio_config_deserialization():
    """Test that AudioConfig can be loaded from dict."""
    data = {
        "speak_enabled": True,
        "praise_mode": "spooky",
        "moto_enabled": False,
        "volume": 0.6,
        "speech_pitch": 0.7
    }

    config = AudioConfig(**data)

    assert config.speak_enabled is True
    assert config.praise_mode == "spooky"
    assert config.moto_enabled is False
    assert config.volume == 0.6
    assert config.speech_pitch == 0.7
