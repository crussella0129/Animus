"""Export voice profile from TTS-Soundboard project file."""

import json
import sys
from pathlib import Path


def export_profile(ttsp_path: Path, output_path: Path, profile_name: str = None):
    """Extract first voice profile from .ttsp and save as standalone JSON."""
    with open(ttsp_path) as f:
        project = json.load(f)

    profiles = project.get("voiceProfiles", [])
    if not profiles:
        print("No voice profiles found in project")
        return

    # Find by name or use first
    if profile_name:
        profile = next((p for p in profiles if p["name"] == profile_name), None)
        if not profile:
            print(f"Profile '{profile_name}' not found")
            print(f"Available profiles: {', '.join(p['name'] for p in profiles)}")
            return
    else:
        profile = profiles[0]

    # Write standalone profile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Exported '{profile['name']}' to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_voice_profile.py <input.ttsp> <output.json> [profile_name]")
        print()
        print("Example:")
        print("  python export_voice_profile.py Animus.ttsp ~/.animus/voices/animus.json Animus")
        sys.exit(1)

    export_profile(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        sys.argv[3] if len(sys.argv) > 3 else None
    )
