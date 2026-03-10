"""Tests for Tool ABC, registry, and built-in tools in src/tools/base.py."""

import pytest


class TestRespondToolVerificationGate:
    def test_respond_description_mentions_verification(self):
        """RespondTool description must require verification before use."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        # Description must mention verification or confirmation
        assert "verif" in tool.description.lower() or "confirm" in tool.description.lower()

    def test_respond_has_verified_parameter(self):
        """RespondTool must have a 'verified' boolean parameter."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        props = tool.parameters["properties"]
        assert "verified" in props
        assert props["verified"]["type"] == "boolean"

    def test_respond_verified_is_required(self):
        """'verified' must be in the required list."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        assert "verified" in tool.parameters["required"]

    def test_respond_execute_returns_message_when_verified(self):
        """execute() returns the message when verified=True."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        result = tool.execute({"message": "hello world", "verified": True})
        assert result == "hello world"

    def test_respond_execute_returns_error_when_not_verified(self):
        """execute() returns an error string when verified=False."""
        from src.tools.base import RespondTool
        tool = RespondTool()
        result = tool.execute({"message": "hello world", "verified": False})
        assert "Error" in result
        assert "verif" in result.lower()
