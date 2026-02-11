"""Verify TTS setup is correct."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.engine import find_tts_engine
from src.audio.voice_profile import VoiceProfile


def verify_setup():
    """Check TTS setup components."""
    print("=== Animus TTS Setup Verification ===\n")

    success = True

    # 1. Check TTS engine
    print("[1/3] Checking TTS engine...")
    engine_path = find_tts_engine()
    if engine_path:
        print(f"  [OK] Found: {engine_path}")
    else:
        print("  [FAIL] Not found")
        print("    Solution: Install TTS-Soundboard or set tts_engine_path in config")
        success = False

    # 2. Check voice profile
    print("\n[2/3] Checking voice profile...")
    voice_file = Path.home() / ".animus" / "voices" / "animus.json"
    if voice_file.exists():
        print(f"  [OK] Found: {voice_file}")

        try:
            profile = VoiceProfile.load(voice_file)
            print(f"    Name: {profile.name}")
            print(f"    Rate: {profile.rate} WPM")
            print(f"    Volume: {profile.volume}")

            # Check DSP chain
            nodes = profile.graph.get("nodes", [])
            dsp_nodes = [n for n in nodes if n["type"] not in ("input", "output")]

            if dsp_nodes:
                dsp_chain = " -> ".join([n["type"] for n in dsp_nodes])
                print(f"    DSP chain: {dsp_chain}")
            else:
                print("    [WARN] No DSP nodes found (voice will sound normal)")

        except Exception as e:
            print(f"  [FAIL] Failed to load: {e}")
            success = False
    else:
        print(f"  [FAIL] Not found: {voice_file}")
        print("    Solution: Run 'python scripts/export_voice_profile.py Animus.ttsp ~/.animus/voices/animus.json Animus'")
        success = False

    # 3. Check configuration
    print("\n[3/3] Checking configuration...")
    config_file = Path.home() / ".animus" / "config.yaml"
    if config_file.exists():
        print(f"  [OK] Found: {config_file}")

        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        audio_config = config.get("audio", {})
        if audio_config.get("enabled"):
            print("    Audio: ENABLED")
            print(f"    Voice profile: {audio_config.get('voice_profile', 'animus')}")
            print(f"    Title: '{audio_config.get('title_text', '')}' ({audio_config.get('title_mode', 'startup')})")
            print(f"    Play mode: {audio_config.get('play_mode', 'responses')}")
            print(f"    Blocking: {audio_config.get('blocking', False)}")
        else:
            print("    Audio: DISABLED")
            print("    Solution: Edit config.yaml and set audio.enabled: true")
    else:
        print(f"  [FAIL] Not found: {config_file}")
        print("    Solution: Run 'animus init'")
        success = False

    # Summary
    print("\n" + "="*40)
    if success:
        print("[SUCCESS] TTS setup is complete!")
        print("\nTo test, run: animus rise")
    else:
        print("[FAIL] Setup incomplete. Please fix the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(verify_setup())
