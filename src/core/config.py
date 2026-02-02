"""Configuration management for Animus."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Model provider configuration."""
    provider: str = Field(default="native", description="Model provider: native, trtllm, api")
    model_name: str = Field(default="", description="Model name/identifier (empty = auto-detect)")
    api_base: Optional[str] = Field(default=None, description="API base URL for remote providers")
    api_key: Optional[str] = Field(default=None, description="API key for authenticated providers")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


def _default_models_dir() -> Path:
    """Get default models directory."""
    return Path.home() / ".animus" / "models"


class NativeConfig(BaseModel):
    """Native model loading configuration (llama-cpp-python)."""
    models_dir: Path = Field(
        default_factory=_default_models_dir,
        description="Directory for storing downloaded models"
    )
    n_ctx: int = Field(default=4096, description="Context window size")
    n_batch: int = Field(default=512, description="Batch size for prompt processing")
    n_threads: Optional[int] = Field(default=None, description="Number of CPU threads (None = auto)")
    n_gpu_layers: int = Field(default=-1, description="Layers to offload to GPU (-1 = all, 0 = none)")
    use_mmap: bool = Field(default=True, description="Use memory mapping for model loading")
    use_mlock: bool = Field(default=False, description="Lock model in RAM")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MemoryConfig(BaseModel):
    """RAG and memory configuration."""
    vector_db: str = Field(default="chromadb", description="Vector database: chromadb, qdrant, milvus")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    chunk_size: int = Field(default=512, description="Text chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")


class MCPServerEntry(BaseModel):
    """Configuration for a single MCP server."""
    name: str = Field(..., description="Unique identifier for this server")
    command: Optional[str] = Field(default=None, description="Command to launch stdio server")
    args: list[str] = Field(default_factory=list, description="Arguments for the command")
    url: Optional[str] = Field(default=None, description="URL for HTTP server")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    enabled: bool = Field(default=True, description="Whether this server is enabled")


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration."""
    servers: list[MCPServerEntry] = Field(
        default_factory=list,
        description="List of configured MCP servers"
    )
    auto_connect: bool = Field(
        default=False,
        description="Automatically connect to enabled servers on startup"
    )


class AgentBehaviorConfig(BaseModel):
    """Agent behavior and stopping cadence configuration."""

    # Tools that execute immediately without confirmation (read-only)
    auto_execute_tools: list[str] = Field(
        default=["read_file", "list_dir"],
        description="Tools that execute without confirmation"
    )

    # Shell commands that are safe to auto-execute (read-only operations)
    safe_shell_commands: list[str] = Field(
        default=[
            "ls", "dir", "cat", "type", "pwd", "cd", "echo",
            "git status", "git log", "git diff", "git branch", "git remote",
            "python --version", "python3 --version", "pip list", "pip show",
            "node --version", "npm list", "which", "where", "whoami",
            "date", "time", "hostname", "uname", "env", "printenv",
        ],
        description="Shell commands that auto-execute without confirmation"
    )

    # Shell command patterns that always trigger a 'STOP, Warn + Explain Risk, then Allow' (dangerous)
    blocked_commands: list[str] = Field(
        default=[
            "rm -rf /", "rm -rf /*", "rm -rf ~",
            "del /s /q c:\\", "format c:",
            ":(){:|:&};:", "dd if=/dev/zero",
            "mkfs", "fdisk", "parted",
            "> /dev/sda", "chmod -R 777 /",
        ],
        description="Shell commands that always trigger a 'STOP, Warn + Explain Risk, then Allow'"
    )

    # Stopping cadences - actions that always require confirmation
    require_confirmation: list[str] = Field(
        default=[
            "create_file",      # Creating new files
            "modify_file",      # Editing existing files
            "delete_file",      # Deleting files
            "change_directory", # Changing working directory to different project
            "git_push",         # Pushing to remote
            "git_commit",       # Creating commits
            "install_package",  # Installing dependencies
            "security_warning", # Any identified security issues
        ],
        description="Actions that always require user confirmation"
    )

    # Working directory tracking
    track_working_directory: bool = Field(
        default=True,
        description="Track and confirm working directory changes"
    )

    # Maximum turns before requiring human check-in
    max_autonomous_turns: int = Field(
        default=10,
        description="Max consecutive turns before pausing for human check-in"
    )


def _default_data_dir() -> Path:
    """Get default data directory."""
    return Path.home() / ".animus" / "data"


def _default_cache_dir() -> Path:
    """Get default cache directory."""
    return Path.home() / ".animus" / "cache"


def _default_logs_dir() -> Path:
    """Get default logs directory."""
    return Path.home() / ".animus" / "logs"


class AnimusConfig(BaseModel):
    """Main Animus configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    native: NativeConfig = Field(default_factory=NativeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentBehaviorConfig = Field(default_factory=AgentBehaviorConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # Paths - use default_factory so they're evaluated at instantiation, not import
    data_dir: Path = Field(default_factory=_default_data_dir)
    cache_dir: Path = Field(default_factory=_default_cache_dir)
    logs_dir: Path = Field(default_factory=_default_logs_dir)

    # Behavior (deprecated - use agent.* instead)
    confirm_destructive: bool = Field(default=True, description="Require confirmation for destructive operations")
    max_context_turns: int = Field(default=10, description="Maximum conversation turns to keep in context")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConfigManager:
    """Manages Animus configuration file."""

    DEFAULT_CONFIG_DIR = Path.home() / ".animus"
    CONFIG_FILENAME = "config.yaml"

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Custom configuration directory. Defaults to ~/.animus
        """
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILENAME
        self._config: Optional[AnimusConfig] = None

    @property
    def config(self) -> AnimusConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def ensure_directories(self) -> None:
        """Create configuration directory structure if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Also create data directories from config
        config = self.config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> AnimusConfig:
        """
        Load configuration from file.

        Returns:
            AnimusConfig: Loaded configuration, or defaults if file doesn't exist.
        """
        if not self.config_path.exists():
            return AnimusConfig()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Convert path strings back to Path objects
            if "data_dir" in data:
                data["data_dir"] = Path(data["data_dir"])
            if "cache_dir" in data:
                data["cache_dir"] = Path(data["cache_dir"])
            if "logs_dir" in data:
                data["logs_dir"] = Path(data["logs_dir"])
            # Handle nested Path in native config
            if "native" in data and "models_dir" in data["native"]:
                data["native"]["models_dir"] = Path(data["native"]["models_dir"])

            return AnimusConfig(**data)
        except (yaml.YAMLError, ValueError) as e:
            # Log warning and return defaults
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return AnimusConfig()

    def save(self, config: Optional[AnimusConfig] = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save. Uses current config if not provided.
        """
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = AnimusConfig()

        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        data = self._config.model_dump()
        data["data_dir"] = str(data["data_dir"])
        data["cache_dir"] = str(data["cache_dir"])
        data["logs_dir"] = str(data["logs_dir"])
        # Handle nested Path in native config
        if "native" in data and "models_dir" in data["native"]:
            data["native"]["models_dir"] = str(data["native"]["models_dir"])

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "model.provider")
            default: Default value if key not found

        Returns:
            Configuration value or default.
        """
        parts = key.split(".")
        obj: Any = self.config

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "model.provider")
            value: Value to set
        """
        parts = key.split(".")
        obj = self.config

        # Navigate to parent
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Configuration key not found: {key}")

        # Set the value
        final_key = parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        else:
            raise KeyError(f"Configuration key not found: {key}")

    def reset(self) -> AnimusConfig:
        """Reset configuration to defaults."""
        self._config = AnimusConfig()
        return self._config
