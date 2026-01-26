"""Core module - Main loop, configuration, and detection."""

from src.core.config import ConfigManager, AnimusConfig
from src.core.detection import detect_environment, SystemInfo
from src.core.agent import Agent, AgentConfig, Turn

__all__ = [
    "ConfigManager",
    "AnimusConfig",
    "detect_environment",
    "SystemInfo",
    "Agent",
    "AgentConfig",
    "Turn",
]
