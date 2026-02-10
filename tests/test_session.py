"""Tests for session persistence."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.session import Session


class TestSession:
    def test_create_session(self):
        session = Session(provider="openai", model="gpt-4")
        assert len(session.id) == 12
        assert session.provider == "openai"
        assert session.model == "gpt-4"
        assert session.messages == []

    def test_create_with_explicit_id(self):
        session = Session(session_id="test123")
        assert session.id == "test123"

    def test_add_message(self):
        session = Session()
        session.add_message("user", "hello")
        session.add_message("assistant", "hi there")
        assert len(session.messages) == 2
        assert session.messages[0] == {"role": "user", "content": "hello"}
        assert session.messages[1] == {"role": "assistant", "content": "hi there"}

    def test_clear_messages(self):
        session = Session()
        session.add_message("user", "hello")
        session.clear_messages()
        assert session.messages == []

    def test_save_and_load(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        session = Session(session_id="abc123", sessions_dir=sessions_dir, provider="anthropic", model="claude")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")
        session.metadata["test"] = True
        saved_path = session.save()

        assert saved_path.exists()
        data = json.loads(saved_path.read_text())
        assert data["id"] == "abc123"
        assert data["provider"] == "anthropic"
        assert len(data["messages"]) == 2

        loaded = Session.load(saved_path)
        assert loaded.id == "abc123"
        assert loaded.provider == "anthropic"
        assert loaded.model == "claude"
        assert len(loaded.messages) == 2
        assert loaded.metadata["test"] is True

    def test_load_by_id(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        session = Session(session_id="findme", sessions_dir=sessions_dir)
        session.add_message("user", "test")
        session.save()

        loaded = Session.load_by_id("findme", sessions_dir)
        assert loaded.id == "findme"
        assert len(loaded.messages) == 1

    def test_load_by_id_not_found(self, tmp_path: Path):
        import pytest
        with pytest.raises(FileNotFoundError, match="Session not found"):
            Session.load_by_id("nonexistent", tmp_path)

    def test_load_latest(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        s1 = Session(session_id="older", sessions_dir=sessions_dir)
        s1.save()
        s2 = Session(session_id="newer", sessions_dir=sessions_dir)
        s2.add_message("user", "latest")
        s2.save()

        latest = Session.load_latest(sessions_dir)
        assert latest is not None
        assert latest.id == "newer"

    def test_load_latest_empty_dir(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        assert Session.load_latest(sessions_dir) is None

    def test_load_latest_no_dir(self, tmp_path: Path):
        sessions_dir = tmp_path / "nonexistent"
        assert Session.load_latest(sessions_dir) is None

    def test_list_sessions(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        s1 = Session(session_id="s1", sessions_dir=sessions_dir, provider="openai")
        s1.add_message("user", "first session message")
        s1.save()
        s2 = Session(session_id="s2", sessions_dir=sessions_dir, provider="anthropic")
        s2.add_message("user", "second session")
        s2.add_message("assistant", "reply")
        s2.save()

        listing = Session.list_sessions(sessions_dir)
        assert len(listing) == 2
        ids = {s["id"] for s in listing}
        assert "s1" in ids
        assert "s2" in ids
        # Check preview extraction
        for item in listing:
            if item["id"] == "s1":
                assert "first session" in item["preview"]
                assert item["messages"] == 1
            if item["id"] == "s2":
                assert item["messages"] == 2

    def test_list_sessions_empty(self, tmp_path: Path):
        assert Session.list_sessions(tmp_path / "empty") == []

    def test_list_sessions_skips_bad_json(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "bad.json").write_text("not valid json")
        s1 = Session(session_id="good", sessions_dir=sessions_dir)
        s1.save()
        listing = Session.list_sessions(sessions_dir)
        assert len(listing) == 1
        assert listing[0]["id"] == "good"

    def test_session_path(self, tmp_path: Path):
        session = Session(session_id="myid", sessions_dir=tmp_path)
        assert session.path == tmp_path / "myid.json"
