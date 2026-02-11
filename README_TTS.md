# Animus TTS Integration

Voice synthesis for Animus using custom DSP-processed voices from TTS-Soundboard.

## Quick Setup

### 1. Export Voice Profile (Already Done!)

The "Animus" voice profile has been exported to `~/.animus/voices/animus.json` with DSP chain:
- **pitch_shift**: -5.76 semitones (deeper voice)
- **bitcrush**: 16-bit, 4x downsample (robotic quality)

### 2. Enable Audio in Config

Edit `~/.animus/config.yaml`:

```yaml
audio:
  enabled: true
  voice_profile: "animus"
  title_text: "father"        # or "mother", or leave "" for none
  title_mode: "startup"        # "startup", "first", "always", "never"
  play_mode: "responses"       # "responses", "greeting_only", "never"
  blocking: false              # false = non-blocking, true = blocking
```

Or copy the example config:

```bash
cp config_examples/audio_enabled.yaml ~/.animus/config.yaml
```

### 3. Test It

```bash
animus rise
```

You should hear:
1. "Yes, father" (or your configured title) at startup
2. All responses spoken with DSP effects

## Configuration Presets

### Full Experience (Recommended)
```yaml
audio:
  enabled: true
  title_text: "father"
  title_mode: "startup"      # Greeting at launch only
  play_mode: "responses"     # Speak all responses
  blocking: false            # Can type during playback
```

### Formal/Ritualistic
```yaml
audio:
  enabled: true
  title_text: "father"
  title_mode: "always"       # Say "Yes father," before every response
  play_mode: "responses"
  blocking: true             # Wait for audio (more formal)
```

### Greeting Only
```yaml
audio:
  enabled: true
  title_text: "mother"
  title_mode: "startup"
  play_mode: "greeting_only" # Only startup greeting
```

### No Title
```yaml
audio:
  enabled: true
  title_text: ""             # No title
  title_mode: "never"
  play_mode: "responses"
```

## Verification

Check your setup:

```bash
python scripts/verify_tts_setup.py
```

Expected output:
```
=== Animus TTS Setup Verification ===

[1/3] Checking TTS engine...
  [OK] Found: C:\Users\charl\TTS-Soundboard\python\tts_engine.py

[2/3] Checking voice profile...
  [OK] Found: C:\Users\charl\.animus\voices\animus.json
    Name: Animus
    Rate: 150 WPM
    Volume: 1
    DSP chain: pitch_shift -> bitcrush

[3/3] Checking configuration...
  [OK] Found: C:\Users\charl\.animus\config.yaml
    Audio: ENABLED
    Voice profile: animus
    Title: 'father' (startup)
    Play mode: responses
    Blocking: False

========================================
[SUCCESS] TTS setup is complete!

To test, run: animus rise
```

## Troubleshooting

See [docs/TTS_SETUP.md](docs/TTS_SETUP.md) for detailed troubleshooting.

### Common Issues

**"TTS engine not found"**
- Install TTS-Soundboard
- Or set `tts_engine_path` in config.yaml

**"Voice profile not found"**
- Run: `python scripts/export_voice_profile.py Animus.ttsp ~/.animus/voices/animus.json Animus`

**No DSP effects**
- Check logs for "DSP chain: pitch_shift -> bitcrush"
- Verify exported JSON has DSP nodes

## Architecture

```
Animus ──┬──> TTS Engine (subprocess)
         │      └─> pyttsx3 + DSP graph
         │
         ├──> Voice Profile Loader
         │      └─> animus.json (pitch + bitcrush)
         │
         ├──> Audio Cache (100MB, LRU)
         │      └─> ~/.animus/audio_cache/
         │
         └──> Audio Player (cross-platform)
                ├─> Windows: PowerShell SoundPlayer
                ├─> macOS: afplay
                └─> Linux: aplay/paplay
```

## File Structure

```
Animus/
├── src/audio/
│   ├── __init__.py          # Public API
│   ├── engine.py            # TTS subprocess manager
│   ├── voice_profile.py     # Profile loading
│   ├── player.py            # Cross-platform playback
│   └── cache.py             # Audio cache
│
├── scripts/
│   ├── export_voice_profile.py   # Export from .ttsp
│   └── verify_tts_setup.py       # Setup verification
│
├── config_examples/
│   └── audio_enabled.yaml        # Example config
│
├── docs/
│   └── TTS_SETUP.md              # Detailed guide
│
└── ~/.animus/
    ├── voices/
    │   └── animus.json           # Exported voice profile
    ├── audio_cache/              # Cached audio files
    └── config.yaml               # User config
```

## Testing

Run unit tests:

```bash
pytest tests/test_audio.py -v
```

## Performance

- **First synthesis**: ~2-5 seconds (includes DSP processing)
- **Cached playback**: <100ms (instant)
- **Cache size**: 100MB max (LRU eviction)
- **Memory**: ~50MB (TTS engine subprocess)

## Future Enhancements

- [ ] Multiple voice profiles (switch mid-conversation)
- [ ] Emotion-based voice selection
- [ ] SSML support (prosody control)
- [ ] Real-time streaming (synthesize as text generates)
- [ ] Voice cloning (Coqui, Bark integration)
- [ ] Interrupt handling (stop on user input)
