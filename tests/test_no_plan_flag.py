"""Test that --no-plan flag bypasses the planner."""
from unittest.mock import patch, MagicMock
import pytest


def test_no_plan_flag_bypasses_should_use_planner():
    """When force_direct=True, should_use_planner() is never consulted."""
    from src.core.agent import Agent

    mock_provider = MagicMock()
    mock_provider.capabilities.return_value = MagicMock(size_tier="medium", context_window=8192)
    mock_registry = MagicMock()
    mock_registry.names.return_value = []
    mock_registry.schemas.return_value = []

    agent = Agent(provider=mock_provider, tool_registry=mock_registry)

    with patch("src.core.agent.should_use_planner") as mock_sup:
        with patch.object(agent, "_run_agentic_loop", return_value="result") as mock_loop:
            agent.run("fix bug in line 5", force_direct=True)
            mock_sup.assert_not_called()
            mock_loop.assert_called_once()


def test_no_plan_flag_false_consults_planner():
    """When force_direct=False, should_use_planner() is called normally."""
    from src.core.agent import Agent
    from unittest.mock import patch, MagicMock

    mock_provider = MagicMock()
    mock_provider.capabilities.return_value = MagicMock(size_tier="medium", context_window=8192)
    mock_registry = MagicMock()
    mock_registry.names.return_value = []
    mock_registry.schemas.return_value = []

    agent = Agent(provider=mock_provider, tool_registry=mock_registry)

    with patch("src.core.agent.should_use_planner", return_value=False) as mock_sup:
        with patch.object(agent, "_run_agentic_loop", return_value="result"):
            agent.run("fix bug in line 5", force_direct=False)
            mock_sup.assert_called_once()
