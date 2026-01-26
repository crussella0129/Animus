"""Core module - Main loop, configuration, and detection."""

from src.core.config import ConfigManager, AnimusConfig
from src.core.detection import detect_environment, SystemInfo

__all__ = ["ConfigManager", "AnimusConfig", "detect_environment", "SystemInfo"]
