"""Core module - Main loop, configuration, and detection."""

from src.core.config import ConfigManager, AnimusConfig, AgentBehaviorConfig
from src.core.detection import detect_environment, SystemInfo
from src.core.agent import Agent, AgentConfig, Turn
from src.core.subagent import (
    SubAgentOrchestrator,
    SubAgentRole,
    SubAgentScope,
    SubAgentResult,
)
from src.core.errors import (
    ErrorCategory,
    ClassifiedError,
    RecoveryStrategy,
    classify_error,
    AnimusError,
    ContextOverflowError,
    AuthenticationError,
    RateLimitError,
    ToolExecutionError,
)

__all__ = [
    # Config
    "ConfigManager",
    "AnimusConfig",
    "AgentBehaviorConfig",
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
    # Errors
    "ErrorCategory",
    "ClassifiedError",
    "RecoveryStrategy",
    "classify_error",
    "AnimusError",
    "ContextOverflowError",
    "AuthenticationError",
    "RateLimitError",
    "ToolExecutionError",
]
