"""Core module - Main loop, configuration, and detection."""

from src.core.config import ConfigManager, AnimusConfig
from src.core.detection import detect_environment, SystemInfo
from src.core.agent import Agent, AgentConfig, Turn
from src.core.subagent import (
    SubAgentOrchestrator,
    SubAgentRole,
    SubAgentScope,
    SubAgentResult,
)

__all__ = [
    # Config
    "ConfigManager",
    "AnimusConfig",
    # Detection
    "detect_environment",
    "SystemInfo",
    # Agent
    "Agent",
    "AgentConfig",
    "Turn",
    # Sub-agent
    "SubAgentOrchestrator",
    "SubAgentRole",
    "SubAgentScope",
    "SubAgentResult",
]
