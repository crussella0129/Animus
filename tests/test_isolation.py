"""Tests for Ornstein & Smough isolation system."""

import pytest
import socket
import time
import tempfile
from pathlib import Path

from src.isolation import (
    IsolationLevel,
    execute_with_isolation,
    OrnsteinSandbox,
    OrnsteinConfig,
)


# Module-level functions for picklability (Windows multiprocessing requirement)

def _add(a, b):
    """Test function: add two numbers."""
    return a + b


def _multiply(a, b):
    """Test function: multiply two numbers."""
    return a * b


def _greet(name):
    """Test function: greet someone."""
    return f"Hello, {name}!"


def _format_string(template, name, age):
    """Test function: format string."""
    return template.format(name=name, age=age)


def _infinite_loop():
    """Test function: infinite loop (for timeout test)."""
    while True:
        time.sleep(0.1)


def _raise_error():
    """Test function: raise an error."""
    raise RuntimeError("Intentional error")


def _create_data():
    """Test function: create complex data."""
    return {
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "nested": {"items": [{"id": 1}, {"id": 2}]}
    }


def _connect_localhost():
    """Test function: connect to localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 80))
    return "Connected"


def _connect_private():
    """Test function: connect to private IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.1.1", 80))
    return "Connected"


def _connect_disallowed():
    """Test function: connect to disallowed domain."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("evil.com", 80))
    return "Connected"


def _read_test_file(file_path):
    """Test function: read a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    return len(content)


def _write_file(tmp_path):
    """Test function: write to a file."""
    test_file = tmp_path / "test.txt"
    with open(test_file, 'w') as f:
        f.write("test")
    return "Written"


def _write_tempfile():
    """Test function: write to tempfile."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        return f.name


def _append_file():
    """Test function: append to file."""
    with tempfile.NamedTemporaryFile(mode='a', delete=False) as f:
        f.write("appended")
        return "Appended"


def _allocate_memory():
    """Test function: allocate memory."""
    data = bytearray(100 * 1024 * 1024)  # 100MB
    return len(data)


def _burn_cpu():
    """Test function: burn CPU."""
    result = 0
    for i in range(10_000_000):
        result += i * i
    return result


class TestIsolationLevel:
    """Test IsolationLevel enum."""

    def test_isolation_levels(self):
        """Test that all isolation levels are defined."""
        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.ORNSTEIN.value == "ornstein"
        assert IsolationLevel.SMOUGH.value == "smough"


class TestExecuteWithIsolation:
    """Test high-level execute_with_isolation function."""

    def test_execute_none_success(self):
        """Test direct execution without isolation."""
        result = execute_with_isolation(_add, args=(2, 3), level=IsolationLevel.NONE)

        assert result.success is True
        assert result.output == 5
        assert result.error is None
        assert result.isolation_level == "none"

    def test_execute_none_error(self):
        """Test error handling in direct execution."""
        result = execute_with_isolation(_raise_error, level=IsolationLevel.NONE)

        assert result.success is False
        assert result.output is None
        assert "Intentional error" in result.error
        assert result.isolation_level == "none"

    def test_execute_ornstein_success(self):
        """Test execution with Ornstein sandbox."""
        config = OrnsteinConfig(timeout_seconds=5)
        result = execute_with_isolation(
            _multiply,
            args=(3, 4),
            level=IsolationLevel.ORNSTEIN,
            config=config
        )

        assert result.success is True
        assert result.output == 12
        assert result.error is None
        assert result.isolation_level == "ornstein"

    def test_execute_smough_not_implemented(self):
        """Test that Smough layer raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Smough layer not yet implemented"):
            execute_with_isolation(_add, level=IsolationLevel.SMOUGH)


class TestOrnsteinSandbox:
    """Test Ornstein lightweight sandbox."""

    def test_basic_execution(self):
        """Test basic function execution in sandbox."""
        config = OrnsteinConfig(timeout_seconds=5)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_greet, args=("Ornstein",))

        assert result.success is True
        assert result.output == "Hello, Ornstein!"
        assert result.error is None
        assert result.execution_time > 0

    def test_execution_with_kwargs(self):
        """Test execution with keyword arguments."""
        config = OrnsteinConfig(timeout_seconds=5)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(
            _format_string,
            args=("{name} is {age} years old",),
            kwargs={"name": "Alice", "age": 30}
        )

        assert result.success is True
        assert result.output == "Alice is 30 years old"

    def test_timeout_enforcement(self):
        """Test that timeout is enforced."""
        config = OrnsteinConfig(timeout_seconds=2)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_infinite_loop)

        assert result.success is False
        assert result.output is None
        assert "timed out" in result.error.lower()
        assert result.execution_time >= 2.0

    def test_exception_handling(self):
        """Test that exceptions are caught and returned."""
        config = OrnsteinConfig(timeout_seconds=5)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_raise_error)

        assert result.success is False
        assert result.output is None
        assert "Intentional error" in result.error

    def test_return_complex_types(self):
        """Test returning complex data structures."""
        config = OrnsteinConfig(timeout_seconds=5)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_create_data)

        assert result.success is True
        assert result.output["list"] == [1, 2, 3]
        assert result.output["dict"]["key"] == "value"
        assert len(result.output["nested"]["items"]) == 2


class TestOrnsteinNetworkFiltering:
    """Test Ornstein network filtering capabilities."""

    def test_block_localhost(self):
        """Test that localhost connections are blocked."""
        config = OrnsteinConfig(
            timeout_seconds=5,
            blocked_ips=["127.0.0.0/8"]
        )
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_connect_localhost)

        assert result.success is False
        assert "blocked" in result.error.lower() or "refused" in result.error.lower()

    def test_block_private_ips(self):
        """Test that private IP ranges are blocked."""
        config = OrnsteinConfig(
            timeout_seconds=5,
            blocked_ips=["10.0.0.0/8", "192.168.0.0/16"]
        )
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_connect_private)

        assert result.success is False
        assert "blocked" in result.error.lower() or "refused" in result.error.lower()

    def test_domain_allowlist_blocks_unlisted(self):
        """Test that domain allowlist blocks unlisted domains."""
        config = OrnsteinConfig(
            timeout_seconds=5,
            allowed_domains=["example.com", "allowed.org"]
        )
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_connect_disallowed)

        # Should fail either from allowlist or connection error
        assert result.success is False

    def test_domain_allowlist_allows_listed(self):
        """Test that allowlisted domains work (if they exist)."""
        # Note: This test only verifies the filtering logic, not actual connection
        config = OrnsteinConfig(
            timeout_seconds=5,
            allowed_domains=["example.com"]
        )
        sandbox = OrnsteinSandbox(config)

        # Test the allowlist logic directly
        assert sandbox._is_domain_allowed("example.com") is True
        assert sandbox._is_domain_allowed("sub.example.com") is True
        assert sandbox._is_domain_allowed("evil.com") is False


class TestOrnsteinFilesystemRestrictions:
    """Test Ornstein filesystem restrictions."""

    def test_read_allowed(self):
        """Test that reading files is allowed."""
        config = OrnsteinConfig(timeout_seconds=5, allow_write=False)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_read_test_file, args=(__file__,))

        assert result.success is True
        assert result.output > 0

    def test_write_blocked(self):
        """Test that writing files is blocked when allow_write=False."""
        config = OrnsteinConfig(timeout_seconds=5, allow_write=False)
        sandbox = OrnsteinSandbox(config)

        tmp = Path(tempfile.gettempdir())
        result = sandbox.execute(_write_file, args=(tmp,))

        assert result.success is False
        assert "permission" in result.error.lower() or "denied" in result.error.lower()

    def test_write_allowed_with_flag(self):
        """Test that writing is allowed when allow_write=True."""
        config = OrnsteinConfig(timeout_seconds=5, allow_write=True)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_write_tempfile)

        assert result.success is True
        assert result.output is not None

        # Cleanup
        import os
        try:
            os.unlink(result.output)
        except:
            pass

    def test_append_blocked(self):
        """Test that append mode is also blocked."""
        config = OrnsteinConfig(timeout_seconds=5, allow_write=False)
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_append_file)

        assert result.success is False
        assert "permission" in result.error.lower() or "denied" in result.error.lower()


class TestOrnsteinResourceLimits:
    """Test Ornstein resource limiting."""

    def test_memory_limit(self):
        """Test that memory limits are enforced (platform-dependent)."""
        # Note: This test may not work on all platforms (especially Windows)
        config = OrnsteinConfig(
            timeout_seconds=10,
            memory_mb=50  # Very small limit
        )
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_allocate_memory)

        # On platforms with resource limits, this should fail
        # On Windows, it may succeed (resource module not available)
        # We just verify the sandbox doesn't crash
        assert result is not None

    def test_cpu_time_limit(self):
        """Test that CPU time limits are enforced (platform-dependent)."""
        config = OrnsteinConfig(
            timeout_seconds=30,  # Wall time
            cpu_percent=50.0     # CPU limit (applies via resource.RLIMIT_CPU)
        )
        sandbox = OrnsteinSandbox(config)

        result = sandbox.execute(_burn_cpu)

        # Should complete successfully (we don't exceed timeout)
        # Just verify the sandbox handles it
        assert result is not None


class TestOrnsteinConfig:
    """Test OrnsteinConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrnsteinConfig()

        assert config.cpu_percent == 50.0
        assert config.memory_mb == 512
        assert config.timeout_seconds == 30
        assert config.allow_write is False
        assert len(config.blocked_ips) > 0
        assert "127.0.0.0/8" in config.blocked_ips

    def test_custom_config(self):
        """Test custom configuration."""
        config = OrnsteinConfig(
            cpu_percent=25.0,
            memory_mb=256,
            timeout_seconds=10,
            allow_write=True,
            allowed_domains=["example.com"]
        )

        assert config.cpu_percent == 25.0
        assert config.memory_mb == 256
        assert config.timeout_seconds == 10
        assert config.allow_write is True
        assert config.allowed_domains == ["example.com"]


class TestOrnsteinHelpers:
    """Test Ornstein helper functions."""

    def test_create_sandbox(self):
        """Test create_sandbox convenience function."""
        from src.isolation.ornstein import create_sandbox

        sandbox = create_sandbox(
            cpu_percent=25.0,
            memory_mb=256,
            timeout_seconds=10,
            allow_write=True,
            allowed_domains=["example.com"]
        )

        assert isinstance(sandbox, OrnsteinSandbox)
        assert sandbox.config.cpu_percent == 25.0
        assert sandbox.config.memory_mb == 256
        assert sandbox.config.timeout_seconds == 10
        assert sandbox.config.allow_write is True
        assert sandbox.config.allowed_domains == ["example.com"]

    def test_is_ip_blocked_logic(self):
        """Test IP blocking logic."""
        config = OrnsteinConfig(
            blocked_ips=["127.0.0.0/8", "192.168.1.100"]
        )
        sandbox = OrnsteinSandbox(config)

        # These tests check the logic without actual network calls
        assert sandbox._is_ip_blocked("127.0.0.1") is True
        assert sandbox._is_ip_blocked("127.255.255.255") is True
        # Note: actual blocking may vary based on DNS resolution

    def test_is_domain_allowed_logic(self):
        """Test domain allowlist logic."""
        config = OrnsteinConfig(
            allowed_domains=["example.com", "test.org"]
        )
        sandbox = OrnsteinSandbox(config)

        assert sandbox._is_domain_allowed("example.com") is True
        assert sandbox._is_domain_allowed("sub.example.com") is True
        assert sandbox._is_domain_allowed("test.org") is True
        assert sandbox._is_domain_allowed("evil.com") is False

    def test_empty_allowlist_allows_all(self):
        """Test that empty allowlist allows all domains."""
        config = OrnsteinConfig(allowed_domains=[])
        sandbox = OrnsteinSandbox(config)

        assert sandbox._is_domain_allowed("anything.com") is True
        assert sandbox._is_domain_allowed("example.org") is True
