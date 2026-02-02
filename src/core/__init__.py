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
from src.core.builder import (
    BuilderQuery,
    AnalysisResult,
    Suggestion,
    SuggestionPriority,
    SuggestionCategory,
)
from src.core.judge import (
    HybridJudge,
    RuleEngine,
    RuleCheck,
    LLMEvaluator,
    HumanEscalator,
    VerificationResult,
    VerificationLevel,
    VerificationSource,
)
from src.core.permission import (
    PermissionAction,
    PermissionCategory,
    PermissionResult,
    PermissionConfig,
    PermissionChecker,
    check_path_permission,
    check_command_permission,
    is_mandatory_deny_path,
    is_mandatory_deny_command,
    get_permission_checker,
    DANGEROUS_DIRECTORIES,
    DANGEROUS_FILES,
    BLOCKED_COMMANDS,
    SAFE_READ_COMMANDS,
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
    # BuilderQuery (Self-Improvement)
    "BuilderQuery",
    "AnalysisResult",
    "Suggestion",
    "SuggestionPriority",
    "SuggestionCategory",
    # Triangulated Verification
    "HybridJudge",
    "RuleEngine",
    "RuleCheck",
    "LLMEvaluator",
    "HumanEscalator",
    "VerificationResult",
    "VerificationLevel",
    "VerificationSource",
    # Permissions (Hardcoded Security)
    "PermissionAction",
    "PermissionCategory",
    "PermissionResult",
    "PermissionConfig",
    "PermissionChecker",
    "check_path_permission",
    "check_command_permission",
    "is_mandatory_deny_path",
    "is_mandatory_deny_command",
    "get_permission_checker",
    "DANGEROUS_DIRECTORIES",
    "DANGEROUS_FILES",
    "BLOCKED_COMMANDS",
    "SAFE_READ_COMMANDS",
]
