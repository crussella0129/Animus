"""Session persistence for pause/resume of sub-agent graph execution."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default session storage directory
DEFAULT_SESSION_DIR = Path.home() / ".animus" / "sessions"


@dataclass
class SessionState:
    """Persisted state for a paused graph execution."""
    session_id: str
    graph_id: str
    paused_at: str  # Node id where execution paused
    context: dict[str, Any] = field(default_factory=dict)
    steps_completed: list[str] = field(default_factory=list)  # Node ids
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionStore:
    """Persists and retrieves session state for graph pause/resume.

    Sessions are stored as JSON files in ~/.animus/sessions/.
    """

    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = session_dir or DEFAULT_SESSION_DIR
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: SessionState) -> Path:
        """Save session state to disk."""
        state.updated_at = time.time()
        if state.created_at == 0.0:
            state.created_at = state.updated_at

        path = self.session_dir / f"{state.session_id}.json"
        path.write_text(json.dumps(state.to_dict(), indent=2, default=str))
        logger.debug("Saved session %s to %s", state.session_id, path)
        return path

    def load(self, session_id: str) -> Optional[SessionState]:
        """Load session state from disk."""
        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None

    def delete(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self.session_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[SessionState]:
        """List all saved sessions."""
        sessions = []
        for path in self.session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sessions.append(SessionState.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def list_for_graph(self, graph_id: str) -> list[SessionState]:
        """List sessions for a specific graph."""
        return [s for s in self.list_sessions() if s.graph_id == graph_id]
