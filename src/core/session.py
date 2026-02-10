"""Session persistence: save/load conversation history to ~/.animus/sessions/."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class Session:
    """Manages a conversation session with save/load to JSON files."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        sessions_dir: Optional[Path] = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.sessions_dir = sessions_dir or Path.home() / ".animus" / "sessions"
        self.provider = provider
        self.model = model
        self.created: float = time.time()
        self.messages: list[dict[str, str]] = []
        self.metadata: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return self.sessions_dir / f"{self.id}.json"

    def save(self) -> Path:
        """Save session to JSON file. Returns the file path."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id,
            "created": self.created,
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "metadata": self.metadata,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, session_path: Path) -> "Session":
        """Load a session from a JSON file."""
        data = json.loads(session_path.read_text(encoding="utf-8"))
        session = cls(
            session_id=data["id"],
            sessions_dir=session_path.parent,
            provider=data.get("provider", ""),
            model=data.get("model", ""),
        )
        session.created = data.get("created", time.time())
        session.messages = data.get("messages", [])
        session.metadata = data.get("metadata", {})
        return session

    @classmethod
    def load_by_id(cls, session_id: str, sessions_dir: Path) -> "Session":
        """Load a session by its ID from the sessions directory."""
        path = sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return cls.load(path)

    @classmethod
    def load_latest(cls, sessions_dir: Path) -> Optional["Session"]:
        """Load the most recently modified session."""
        if not sessions_dir.exists():
            return None
        files = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        return cls.load(files[0])

    @classmethod
    def list_sessions(cls, sessions_dir: Path) -> list[dict[str, Any]]:
        """List all saved sessions with summary info."""
        if not sessions_dir.exists():
            return []
        sessions = []
        for path in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                msg_count = len(data.get("messages", []))
                # Extract first user message as preview
                preview = ""
                for msg in data.get("messages", []):
                    if msg.get("role") == "user":
                        preview = msg.get("content", "")[:80]
                        break
                sessions.append({
                    "id": data.get("id", path.stem),
                    "created": data.get("created", 0),
                    "provider": data.get("provider", ""),
                    "model": data.get("model", ""),
                    "messages": msg_count,
                    "preview": preview,
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session history."""
        self.messages.append({"role": role, "content": content})

    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()
