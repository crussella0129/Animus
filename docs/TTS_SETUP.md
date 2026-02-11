# TTS Integration Setup Guide

This guide explains how to set up and use Animus voice synthesis with DSP processing.

## Prerequisites

- **TTS-Soundboard** installed with the "Animus" voice profile configured
- Python dependencies (already included in TTS-Soundboard): `pyttsx3`, `numpy`, `scipy`, `soundfile`

## Quick Start

### 1. Export Voice Profile

Export your "Animus" voice profile from TTS-Soundboard:

```bash
cd Animus
python scripts/export_voice_profile.py ../Animus.ttsp ~/.animus/voices/animus.json Animus
```

Or manually:
1. Open TTS-Soundboard
2. Find the "Animus" voice profile
3. Export/save as `Animus.ttsp`
4. Copy to `~/.animus/voices/animus.json`

### 2. Configure Animus

Edit `~/.animus/config.yaml`:

```yaml
audio:
  enabled: true
  voice_profile: "animus"
  title_text: "father"        # or "mother", or "" for none
  title_mode: "startup"        # "startup", "first", "always", "never"
  play_mode: "responses"       # "responses", "greeting_only", "never"
  blocking: false              # false = non-blocking, true = wait for audio
```

### 3. Run Animus

```bash
animus rise
```

You should hear:
- Startup greeting: "Yes, father" (if configured)
- All responses spoken with DSP effects (lowered pitch, bitcrush)

## Configuration Options

### Title Modes

- **`startup`** (default): Say "Yes, [title]" once at launch (standalone greeting)
- **`first`**: Prepend "Yes [title]," to the first response only
- **`always`**: Prepend "Yes [title]," to every response
- **`never`**: No title acknowledgment

### Play Modes

- **`responses`** (default): Speak all agent responses
- **`greeting_only`**: Only play startup greeting, silent responses
- **`never`**: Disable TTS entirely (same as `enabled: false`)

### Blocking Mode

- **`false`** (default): Non-blocking playback - you can type while Animus speaks
- **`true`**: Blocking playback - wait for audio before returning to prompt (more formal)

## Configuration Examples

### Example 1: Full TTS (Recommended)

```yaml
audio:
  enabled: true
  voice_profile: "animus"
  title_text: "father"
  title_mode: "startup"      # Say "Yes father" once at startup
  play_mode: "responses"      # Speak all responses
  blocking: false             # Non-blocking
```

### Example 2: Greeting Only

```yaml
audio:
  enabled: true
  title_text: "mother"
  title_mode: "startup"
  play_mode: "greeting_only"  # Only startup greeting, silent responses
```

### Example 3: Formal/Ritualistic

```yaml
audio:
  enabled: true
  title_text: "father"
  title_mode: "always"        # Say "Yes father," before every response
  play_mode: "responses"
  blocking: true              # Wait for audio (more formal)
```

### Example 4: No Title

```yaml
audio:
  enabled: true
  title_text: ""              # No title
  title_mode: "never"
  play_mode: "responses"
```

## DSP Processing

The Animus voice profile includes DSP effects:

- **Pitch Shift**: Lowers voice pitch (typically -5 to -6 semitones)
- **Bitcrush**: Adds digital/robotic quality

These effects are applied automatically via the TTS-Soundboard engine.

## Troubleshooting

### "TTS engine not found"

**Solution:** Set `tts_engine_path` explicitly in config.yaml:

```yaml
audio:
  enabled: true
  tts_engine_path: "C:/Users/YourName/TTS-Soundboard/python/tts_engine.py"
```

### "Voice profile not found"

**Solution:** Ensure the voice profile is exported:

```bash
python scripts/export_voice_profile.py ../Animus.ttsp ~/.animus/voices/animus.json Animus
```

### Audio plays but no DSP effects

**Check:**
1. Verify DSP graph has nodes (not just input/output)
2. Check logs for "DSP chain: pitch_shift -> bitcrush"
3. Open the exported JSON and verify `graph.nodes` contains DSP nodes

### Subprocess crashes repeatedly

**Check:**
1. TTS-Soundboard Python environment has all dependencies
2. Try running `tts_engine.py` directly to see error messages:

```bash
cd TTS-Soundboard/python
python tts_engine.py
# Should print: {"ok": true, "message": "TTS engine ready"}
# Type: {"cmd": "quit"}
```

## Audio Cache

- Cached audio files are stored in `~/.animus/audio_cache/`
- Cache size limit: 100 MB (configurable)
- LRU eviction: oldest files removed when limit exceeded
- To clear cache: `rm -rf ~/.animus/audio_cache/*`

## Testing

Run unit tests:

```bash
cd Animus
pytest tests/test_audio.py -v
```

Run integration test:

```bash
# Test voice profile loading
python -c "from src.audio.voice_profile import VoiceProfile; from pathlib import Path; p = VoiceProfile.load(Path.home() / '.animus/voices/animus.json'); print(f'Loaded: {p.name}')"

# Test TTS engine
python -c "from src.audio.engine import find_tts_engine; print(find_tts_engine())"
```

## Architecture

```
Animus (main process)
    │
    ├─→ src/audio/
    │   ├── __init__.py          # Public API
    │   ├── engine.py            # TTSEngine subprocess manager
    │   ├── voice_profile.py     # VoiceProfile loading
    │   ├── player.py            # AudioPlayer (cross-platform)
    │   └── cache.py             # Hash-based cache
    │
    └─→ TTS Engine Subprocess
        └── tts_engine.py (from TTS-Soundboard)
            - Persistent subprocess
            - JSON command interface
            - Commands: process_graph, list_voices, quit
```

## Future Enhancements

- [ ] Multiple voice profiles (switch mid-conversation)
- [ ] Emotion tags (different voices for different contexts)
- [ ] SSML support (fine-grained prosody control)
- [ ] Real-time streaming (synthesize as text generates)
- [ ] Voice cloning integration (Coqui, Bark)
- [ ] Interrupt handling (stop audio when user types)
