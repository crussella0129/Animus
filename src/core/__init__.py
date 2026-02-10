"""Core module - Main loop, configuration, and detection."""

from src.core.config import ConfigManager, AnimusConfig, AgentBehaviorConfig
from src.core.detection import detect_environment, SystemInfo
from src.core.agent import Agent, AgentConfig, Turn, StreamChunk
from src.core.subagent import (
    SubAgentOrchestrator,
    SubAgentRole,
    SubAgentScope,
    SubAgentResult,
)
# Graph-based sub-agents available via `from src.subagents import ...`
# Not re-exported here to avoid circular imports (executor → llm → core).
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
    AgentPermissionScope,
    check_path_permission,
    check_command_permission,
    is_mandatory_deny_path,
    is_mandatory_deny_command,
    get_permission_checker,
    get_profile,
    get_agent_scope,
    DANGEROUS_DIRECTORIES,
    DANGEROUS_FILES,
    BLOCKED_COMMANDS,
    SAFE_READ_COMMANDS,
    PERMISSION_PROFILES,
    AGENT_SCOPES,
)
from src.core.tokenizer import (
    count_tokens,
    count_tokens_messages,
    count_tokens_cached,
    truncate_to_tokens,
    split_by_tokens,
    is_tiktoken_available,
    get_model_context_limit,
)
from src.core.planner import (
    Planner,
    ExecutionPlan,
    PlanStep,
    StepStatus,
)
from src.core.loop_detector import (
    LoopDetector,
    LoopDetectorConfig,
    InterventionLevel,
)
from src.core.fallback import (
    ModelFallbackChain,
    FallbackModel,
    FallbackEvent,
)
from src.core.auth_rotation import (
    AuthRotator,
    AuthProfile,
    RotationEvent,
)
from src.core.web_validator import (
    WebContentJudge,
    WebContentRuleEngine,
    WebContentLLMValidator,
    LLMValidatorConfig,
    create_web_content_judge,
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
    "StreamChunk",
    # Sub-agent (role-based)
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
    "AgentPermissionScope",
    "get_profile",
    "get_agent_scope",
    "PERMISSION_PROFILES",
    "AGENT_SCOPES",
    # Tokenizer
    "count_tokens",
    "count_tokens_messages",
    "count_tokens_cached",
    "truncate_to_tokens",
    "split_by_tokens",
    "is_tiktoken_available",
    "get_model_context_limit",
    # Planning
    "Planner",
    "ExecutionPlan",
    "PlanStep",
    "StepStatus",
    # Action Loop Detection
    "LoopDetector",
    "LoopDetectorConfig",
    "InterventionLevel",
    # Model Fallback Chain
    "ModelFallbackChain",
    "FallbackModel",
    "FallbackEvent",
    # Auth Profile Rotation
    "AuthRotator",
    "AuthProfile",
    "RotationEvent",
    # Web Content Validation (Ungabunga-Box)
    "WebContentJudge",
    "WebContentRuleEngine",
    "WebContentLLMValidator",
    "LLMValidatorConfig",
    "create_web_content_judge",
]
