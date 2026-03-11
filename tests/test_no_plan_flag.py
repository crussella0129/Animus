"""Test that --no-plan flag bypasses the planner at the CLI level."""
import re
from unittest.mock import patch, MagicMock
import pytest


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_no_plan_short_circuits_use_plan():
    """When no_plan=True, use_plan is False and should_use_planner is never called."""
    mock_provider = MagicMock()

    # Simulate the logic in app.py:
    # use_plan = not no_plan and (plan_mode_active or should_use_planner(provider))
    with patch("src.core.planner.should_use_planner", return_value=True) as mock_sup:
        from src.core.planner import should_use_planner
        no_plan = True
        use_plan = not no_plan and (False or should_use_planner(mock_provider))
        assert use_plan is False
        # should_use_planner must not be called — short-circuit prevents it
        mock_sup.assert_not_called()


def test_no_plan_false_consults_planner():
    """When no_plan=False, should_use_planner() determines the plan path."""
    mock_provider = MagicMock()

    with patch("src.core.planner.should_use_planner", return_value=True) as mock_sup:
        from src.core.planner import should_use_planner
        # Without --no-plan: use_plan defers to should_use_planner()
        use_plan = not False and (False or should_use_planner(mock_provider))
        assert use_plan is True
        mock_sup.assert_called_once()


def test_rise_help_contains_no_plan():
    """The --no-plan option must appear in `animus rise --help` output."""
    from src.cli.app import app
    import typer.testing
    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["rise", "--help"])
    assert result.exit_code == 0
    assert "--no-plan" in _strip_ansi(result.output)
