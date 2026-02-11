"""Ornstein: Lightweight process-level sandbox for web exploration."""

from __future__ import annotations
import logging
import multiprocessing
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrnsteinConfig:
    """Configuration for Ornstein lightweight sandbox."""

    # Resource limits
    cpu_percent: float = 50.0        # Max CPU usage (% of one core)
    memory_mb: int = 512             # Max memory in MB
    timeout_seconds: int = 30        # Max execution time

    # Network filtering
    allowed_domains: list[str] = field(default_factory=list)  # Empty = all allowed
    blocked_ips: list[str] = field(default_factory=lambda: [
        "127.0.0.0/8",      # Localhost
        "10.0.0.0/8",       # Private
        "172.16.0.0/12",    # Private
        "192.168.0.0/16",   # Private
        "169.254.0.0/16",   # Link-local
    ])

    # Filesystem
    allow_write: bool = False        # Allow write operations
    allowed_paths: list[str] = field(default_factory=list)  # Paths allowed to read


@dataclass
class OrnsteinResult:
    """Result from Ornstein sandbox execution."""
    success: bool
    output: Any
    error: Optional[str]
    isolation_level: str
    resource_usage: dict[str, Any]
    execution_time: float = 0.0


# Module-level helper functions (picklable for multiprocessing)

def _apply_resource_limits(config: OrnsteinConfig):
    """Apply CPU and memory limits to current process."""
    try:
        import resource

        # Memory limit (RLIMIT_AS = address space)
        memory_bytes = config.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # CPU time limit
        cpu_seconds = config.timeout_seconds
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        logger.debug(f"[Ornstein] Applied resource limits: {config.memory_mb}MB RAM, {cpu_seconds}s CPU")

    except (ImportError, AttributeError):
        # resource module not available on Windows
        logger.warning("[Ornstein] Resource limits not available on this platform")


def _apply_network_filtering(config: OrnsteinConfig):
    """Override socket to filter network connections."""
    original_socket = socket.socket

    def filtered_socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, *args, **kwargs):
        """Socket wrapper that enforces IP/domain filtering."""
        sock = original_socket(family, type, proto, *args, **kwargs)

        # Wrap connect to check IPs
        original_connect = sock.connect

        def filtered_connect(address):
            host, port = address if isinstance(address, tuple) else (address, None)

            # Check if IP is blocked
            if _is_ip_blocked(host, config):
                raise ConnectionRefusedError(f"Connection to {host} blocked by Ornstein sandbox (SSRF prevention)")

            # Check domain allowlist (if configured)
            if config.allowed_domains and not _is_domain_allowed(host, config):
                raise ConnectionRefusedError(f"Domain {host} not in allowlist")

            return original_connect(address)

        sock.connect = filtered_connect
        return sock

    # Replace socket globally
    socket.socket = filtered_socket
    logger.debug(f"[Ornstein] Applied network filtering (blocked IPs: {len(config.blocked_ips)})")


def _apply_readonly_restrictions(config: OrnsteinConfig):
    """Apply read-only filesystem restrictions."""
    import builtins

    original_open = builtins.open

    def readonly_open(file, mode='r', *args, **kwargs):
        """Open wrapper that blocks write operations."""
        # Allow read modes
        if 'r' in mode and 'w' not in mode and 'a' not in mode and '+' not in mode:
            return original_open(file, mode, *args, **kwargs)

        # Check if path is in allowed list
        file_path = str(file)
        if config.allowed_paths:
            for allowed in config.allowed_paths:
                if file_path.startswith(allowed):
                    return original_open(file, mode, *args, **kwargs)

        # Block write
        raise PermissionError(f"Write access denied by Ornstein sandbox: {file}")

    builtins.open = readonly_open
    logger.debug("[Ornstein] Applied read-only filesystem restrictions")


def _is_ip_blocked(host: str, config: OrnsteinConfig) -> bool:
    """Check if IP address is in blocklist."""
    try:
        import ipaddress

        # Resolve hostname to IP
        try:
            ip = socket.gethostbyname(host)
        except socket.gaierror:
            # Can't resolve, allow (will fail naturally)
            return False

        ip_obj = ipaddress.ip_address(ip)

        # Check against blocked ranges
        for blocked_range in config.blocked_ips:
            if "/" in blocked_range:
                network = ipaddress.ip_network(blocked_range, strict=False)
                if ip_obj in network:
                    logger.warning(f"[Ornstein] Blocked connection to {host} ({ip}) - matches {blocked_range}")
                    return True
            else:
                if str(ip_obj) == blocked_range:
                    logger.warning(f"[Ornstein] Blocked connection to {host} ({ip})")
                    return True

        return False

    except ImportError:
        # ipaddress not available, can't filter
        logger.warning("[Ornstein] ipaddress module not available, skipping IP filtering")
        return False


def _is_domain_allowed(host: str, config: OrnsteinConfig) -> bool:
    """Check if domain is in allowlist."""
    if not config.allowed_domains:
        return True  # Empty allowlist = all allowed

    # Check exact match or subdomain
    for allowed in config.allowed_domains:
        if host == allowed or host.endswith(f".{allowed}"):
            return True

    logger.warning(f"[Ornstein] Blocked connection to {host} - not in domain allowlist")
    return False


def _get_resource_usage() -> dict[str, Any]:
    """Get current resource usage statistics."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "cpu_time": usage.ru_utime + usage.ru_stime,
            "max_rss_mb": usage.ru_maxrss / 1024,  # Convert to MB
        }
    except ImportError:
        return {}


def _execute_in_sandbox(func, args, kwargs, config, result_queue):
    """Worker function that runs in isolated subprocess."""
    try:
        # Apply resource limits
        _apply_resource_limits(config)

        # Apply network filtering
        _apply_network_filtering(config)

        # Apply filesystem restrictions
        if not config.allow_write:
            _apply_readonly_restrictions(config)

        # Execute function
        result = func(*args, **kwargs)

        # Get resource usage
        resource_usage = _get_resource_usage()

        result_queue.put({
            "success": True,
            "output": result,
            "error": None,
            "resource_usage": resource_usage,
        })

    except Exception as e:
        logger.error(f"[Ornstein] Execution error: {e}")
        result_queue.put({
            "success": False,
            "output": None,
            "error": str(e),
            "resource_usage": {},
        })


class OrnsteinSandbox:
    """
    Lightweight process-level sandbox for web exploration.

    Features:
    - Process isolation via multiprocessing
    - Resource limits (CPU, memory, timeout)
    - Network filtering (IP blocklist, domain allowlist)
    - Filesystem restrictions
    - Signal-based timeout enforcement
    """

    def __init__(self, config: OrnsteinConfig):
        self.config = config

    def execute(
        self,
        func: callable,
        args: tuple = (),
        kwargs: dict[str, Any] = None,
    ) -> OrnsteinResult:
        """
        Execute function in isolated subprocess with resource limits.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            OrnsteinResult with output and resource usage
        """
        kwargs = kwargs or {}
        start_time = time.time()

        # Use multiprocessing for process isolation
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()

        # Start subprocess with module-level function (picklable)
        process = ctx.Process(
            target=_execute_in_sandbox,
            args=(func, args, kwargs, self.config, result_queue)
        )
        process.start()

        # Wait with timeout
        process.join(timeout=self.config.timeout_seconds)

        execution_time = time.time() - start_time

        # Check if timed out
        if process.is_alive():
            logger.warning(f"[Ornstein] Process timed out after {self.config.timeout_seconds}s")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()

            return OrnsteinResult(
                success=False,
                output=None,
                error=f"Execution timed out after {self.config.timeout_seconds} seconds",
                isolation_level="ornstein",
                resource_usage={},
                execution_time=execution_time,
            )

        # Get result from queue
        if not result_queue.empty():
            result_data = result_queue.get()
            return OrnsteinResult(
                success=result_data["success"],
                output=result_data["output"],
                error=result_data["error"],
                isolation_level="ornstein",
                resource_usage=result_data["resource_usage"],
                execution_time=execution_time,
            )
        else:
            return OrnsteinResult(
                success=False,
                output=None,
                error="Process terminated without returning result",
                isolation_level="ornstein",
                resource_usage={},
                execution_time=execution_time,
            )

    def _is_ip_blocked(self, host: str) -> bool:
        """Check if IP address is in blocklist."""
        return _is_ip_blocked(host, self.config)

    def _is_domain_allowed(self, host: str) -> bool:
        """Check if domain is in allowlist."""
        return _is_domain_allowed(host, self.config)


def create_sandbox(
    cpu_percent: float = 50.0,
    memory_mb: int = 512,
    timeout_seconds: int = 30,
    allow_write: bool = False,
    allowed_domains: list[str] = None,
) -> OrnsteinSandbox:
    """
    Convenience function to create an Ornstein sandbox.

    Args:
        cpu_percent: Max CPU usage (% of one core)
        memory_mb: Max memory in MB
        timeout_seconds: Max execution time
        allow_write: Allow write operations
        allowed_domains: List of allowed domains (empty = all allowed)

    Returns:
        Configured OrnsteinSandbox instance
    """
    config = OrnsteinConfig(
        cpu_percent=cpu_percent,
        memory_mb=memory_mb,
        timeout_seconds=timeout_seconds,
        allow_write=allow_write,
        allowed_domains=allowed_domains or [],
    )
    return OrnsteinSandbox(config)
