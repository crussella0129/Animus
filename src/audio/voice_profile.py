"""Voice profile loading and validation."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class VoiceProfile:
    """TTS voice profile with DSP graph."""
    name: str
    voice_id: str           # pyttsx3 voice ID (e.g., Windows SAPI5 HKEY path)
    rate: int               # Words per minute (80-300)
    volume: float           # Volume 0.0-1.0
    pitch: float            # Pitch offset in semitones
    graph: dict             # DSP graph: {"nodes": [...], "edges": [...]}

    @classmethod
    def load(cls, path: Path) -> VoiceProfile:
        """Load voice profile from JSON file (.ttsp or exported .json)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract from .ttsp format (has voiceProfiles array)
        if "voiceProfiles" in data and len(data["voiceProfiles"]) > 0:
            profile_data = data["voiceProfiles"][0]
        else:
            profile_data = data

        # Validate required fields
        required = ["name", "systemVoiceId", "rate", "volume", "graph"]
        for field in required:
            if field not in profile_data:
                raise ValueError(f"Missing required field: {field}")

        # Convert TTS-Soundboard format to our format
        return cls(
            name=profile_data["name"],
            voice_id=profile_data["systemVoiceId"],
            rate=profile_data["rate"],
            volume=profile_data["volume"],
            pitch=profile_data.get("pitch", 0),
            graph=profile_data["graph"]
        )

    def to_graph_command(self) -> dict:
        """Convert DSP graph to format expected by tts_engine.py."""
        # Filter out input/output nodes (added by tts_engine)
        nodes = [
            {"id": n["id"], "type": n["type"], "params": n.get("params", {})}
            for n in self.graph.get("nodes", [])
            if n["type"] not in ("input", "output")
        ]
        edges = [
            {"source": e["source"], "target": e["target"]}
            for e in self.graph.get("edges", [])
        ]
        return {"nodes": nodes, "edges": edges}
