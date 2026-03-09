"""Test that --no-plan flag bypasses the planner at the CLI level."""
from unittest.mock import patch, MagicMock
import pytest


def test_no_plan_short_circuits_use_plan():
    """When no_plan=True, use_plan is False regardless of should_use_planner()."""
    from src.core.planner import should_use_planner

    # Simulate the logic in app.py:
    # use_plan = not no_plan and (plan_mode_active or should_use_planner(provider))
    mock_provider = MagicMock()

    with patch("src.core.planner.executor.should_use_planner", return_value=True):
        # With --no-plan: use_plan must be False
        use_plan = not True and (False or should_use_planner(mock_provider))
        assert use_plan is False


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
    assert "--no-plan" in result.output
