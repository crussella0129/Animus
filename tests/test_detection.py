"""Tests for system detection module."""

import platform
from src.core.detection import (
    detect_environment,
    OperatingSystem,
    Architecture,
    HardwareType,
    SystemInfo,
)


def test_detect_environment_returns_system_info():
    """Test that detect_environment returns a valid SystemInfo object."""
    info = detect_environment()

    assert isinstance(info, SystemInfo)
    assert isinstance(info.os, OperatingSystem)
    assert isinstance(info.architecture, Architecture)
    assert isinstance(info.hardware_type, HardwareType)
    assert isinstance(info.python_version, str)
    assert isinstance(info.hostname, str)
    assert isinstance(info.cpu_count, int)
    assert info.cpu_count >= 1


def test_os_detection_matches_platform():
    """Test that OS detection matches Python's platform module."""
    info = detect_environment()
    system = platform.system().lower()

    if system == "windows":
        assert info.os == OperatingSystem.WINDOWS
    elif system == "darwin":
        assert info.os == OperatingSystem.MACOS
    elif system == "linux":
        assert info.os == OperatingSystem.LINUX


def test_architecture_detection():
    """Test architecture detection returns valid value."""
    info = detect_environment()

    assert info.architecture in [
        Architecture.X86_64,
        Architecture.ARM64,
        Architecture.ARMV7,
        Architecture.UNKNOWN,
    ]


def test_python_version_format():
    """Test Python version is in expected format."""
    info = detect_environment()

    parts = info.python_version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
