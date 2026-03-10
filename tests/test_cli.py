"""Smoke tests for the src/cli/ package."""

from __future__ import annotations


class TestCLIPackage:
    def test_cli_app_importable(self):
        """The cli package exports a valid Typer app."""
        import typer
        from src.cli import app
        assert isinstance(app, typer.Typer)

    def test_slash_commands_importable(self):
        """slash_commands module exports handle_slash_command."""
        from src.cli.slash_commands import handle_slash_command
        assert callable(handle_slash_command)

    def test_session_manager_importable(self):
        """session_manager exports build_tool_registry and make_confirm_callback."""
        from src.cli.session_manager import build_tool_registry, make_confirm_callback
        assert callable(build_tool_registry)
        assert callable(make_confirm_callback)

    def test_main_shim_still_exports_app(self):
        """src.main still exports app for backward compatibility."""
        from src.main import app
        import typer
        assert isinstance(app, typer.Typer)

    def test_plan_mode_state_in_slash_commands(self):
        """_plan_mode_state is accessible from slash_commands module."""
        from src.cli.slash_commands import _plan_mode_state
        assert isinstance(_plan_mode_state, dict)
        assert "active" in _plan_mode_state

    def test_app_has_rise_command(self):
        """The app has a 'rise' command registered."""
        from src.cli import app
        # Typer uses callback.__name__ when name is not explicitly set
        callback_names = [cmd.callback.__name__ for cmd in app.registered_commands if cmd.callback]
        assert "rise" in callback_names

    def test_app_has_all_commands(self):
        """The app has all expected commands registered."""
        from src.cli import app
        # Typer uses callback.__name__ when name is not explicitly set
        callback_names = {cmd.callback.__name__ for cmd in app.registered_commands if cmd.callback}
        expected = {
            "detect", "init", "config", "models", "status",
            "pull", "ingest", "search", "graph", "sessions",
            "rise", "routing_stats",
        }
        assert expected.issubset(callback_names)
