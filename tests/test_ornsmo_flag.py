"""Tests for --ornsmo flag replacing --cautious/--paranoid."""
import pytest


def test_ornsmo_is_valid_rise_option():
    """The rise() command must accept --ornsmo flag."""
    from src.cli.app import app
    import typer.testing
    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["rise", "--help"])
    assert result.exit_code == 0
    assert "ornsmo" in result.output.lower()


def test_cautious_flag_removed():
    """The deprecated --cautious flag should not be present."""
    from src.cli.app import app
    import typer.testing
    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["rise", "--help"])
    assert result.exit_code == 0
    assert "--cautious" not in result.output
