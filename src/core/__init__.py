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
from src.core.decision import (
    Decision,
    DecisionType,
    Option,
    Outcome,
    OutcomeStatus,
    DecisionRecord,
    DecisionRecorder,
)
from src.core.run import (
    Run,
    RunStatus,
    RunMetrics,
    RunStore,
)
from src.core.context import (
    ContextWindow,
    ContextConfig,
    ContextStatus,
    TokenEstimator,
    TokenUsage,
    get_context_config,
    CONTEXT_PRESETS,
)
from src.core.compaction import (
    SessionCompactor,
    CompactionConfig,
    CompactionStrategy,
    CompactionResult,
    compact_conversation,
)
from src.core.permission import (
    PermissionAction,
    PermissionCategory,
    PermissionRule,
    PermissionRuleset,
    PermissionProfile,
    PermissionRequest,
    PermissionManager,
    get_explore_agent_profile,
    get_plan_agent_profile,
    get_build_agent_profile,
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
    # Decision Recording
    "Decision",
    "DecisionType",
    "Option",
    "Outcome",
    "OutcomeStatus",
    "DecisionRecord",
    "DecisionRecorder",
    # Run Persistence
    "Run",
    "RunStatus",
    "RunMetrics",
    "RunStore",
    # Context Management
    "ContextWindow",
    "ContextConfig",
    "ContextStatus",
    "TokenEstimator",
    "TokenUsage",
    "get_context_config",
    "CONTEXT_PRESETS",
    # Session Compaction
    "SessionCompactor",
    "CompactionConfig",
    "CompactionStrategy",
    "CompactionResult",
    "compact_conversation",
    # Permissions
    "PermissionAction",
    "PermissionCategory",
    "PermissionRule",
    "PermissionRuleset",
    "PermissionProfile",
    "PermissionRequest",
    "PermissionManager",
    "get_explore_agent_profile",
    "get_plan_agent_profile",
    "get_build_agent_profile",
]
