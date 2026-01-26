"""System detection module - OS and hardware identification."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class OperatingSystem(str, Enum):
    """Supported operating systems."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


class Architecture(str, Enum):
    """CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARMV7 = "armv7"
    UNKNOWN = "unknown"


class HardwareType(str, Enum):
    """Hardware platform types."""
    JETSON = "jetson"
    APPLE_SILICON = "apple_silicon"
    STANDARD_X86 = "standard_x86"
    STANDARD_ARM = "standard_arm"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """GPU information."""
    name: str = "Unknown"
    vendor: str = "Unknown"
    memory_mb: int = 0
    cuda_available: bool = False
    cuda_version: Optional[str] = None


@dataclass
class SystemInfo:
    """Complete system information."""
    os: OperatingSystem
    os_version: str
    architecture: Architecture
    hardware_type: HardwareType
    python_version: str
    hostname: str
    cpu_count: int
    gpu: Optional[GPUInfo] = None
    is_wsl: bool = False
    extra: dict = field(default_factory=dict)


def _detect_os() -> tuple[OperatingSystem, str]:
    """Detect the operating system and version."""
    system = platform.system().lower()
    version = platform.version()

    if system == "windows":
        return OperatingSystem.WINDOWS, version
    elif system == "darwin":
        return OperatingSystem.MACOS, platform.mac_ver()[0]
    elif system == "linux":
        # Try to get distro info
        try:
            import distro
            version = f"{distro.name()} {distro.version()}"
        except ImportError:
            version = platform.release()
        return OperatingSystem.LINUX, version
    else:
        return OperatingSystem.UNKNOWN, version


def _detect_architecture() -> Architecture:
    """Detect the CPU architecture."""
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return Architecture.X86_64
    elif machine in ("arm64", "aarch64"):
        return Architecture.ARM64
    elif machine.startswith("armv7"):
        return Architecture.ARMV7
    else:
        return Architecture.UNKNOWN


def _detect_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    if platform.system().lower() != "linux":
        return False

    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def _detect_jetson() -> bool:
    """Check if running on NVIDIA Jetson hardware."""
    jetson_release = Path("/etc/nv_tegra_release")
    jetson_model = Path("/sys/firmware/devicetree/base/model")

    if jetson_release.exists():
        return True

    if jetson_model.exists():
        try:
            model = jetson_model.read_text().lower()
            return "jetson" in model or "tegra" in model
        except (PermissionError, OSError):
            pass

    return False


def _detect_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    if platform.system().lower() != "darwin":
        return False

    return platform.machine().lower() == "arm64"


def _detect_hardware_type(os: OperatingSystem, arch: Architecture) -> HardwareType:
    """Determine the hardware platform type."""
    if _detect_jetson():
        return HardwareType.JETSON

    if _detect_apple_silicon():
        return HardwareType.APPLE_SILICON

    if arch == Architecture.X86_64:
        return HardwareType.STANDARD_X86
    elif arch in (Architecture.ARM64, Architecture.ARMV7):
        return HardwareType.STANDARD_ARM

    return HardwareType.UNKNOWN


def _detect_gpu() -> Optional[GPUInfo]:
    """Detect GPU information."""
    gpu = GPUInfo()

    # Try NVIDIA first (most common for AI workloads)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split(", ")
                gpu.name = parts[0].strip()
                gpu.vendor = "NVIDIA"
                if len(parts) > 1:
                    try:
                        gpu.memory_mb = int(float(parts[1].strip()))
                    except ValueError:
                        pass
                gpu.cuda_available = True

                # Get CUDA version
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if cuda_result.returncode == 0:
                    gpu.cuda_version = cuda_result.stdout.strip().split("\n")[0]

                return gpu
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Check for Apple Metal (macOS)
    if platform.system().lower() == "darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "Metal" in result.stdout:
                gpu.vendor = "Apple"
                gpu.name = "Apple GPU (Metal)"
                # Parse chipset name if available
                for line in result.stdout.split("\n"):
                    if "Chipset Model:" in line:
                        gpu.name = line.split(":")[-1].strip()
                        break
                return gpu
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    return None


def detect_environment() -> SystemInfo:
    """
    Detect the complete system environment.

    Returns:
        SystemInfo: Complete system information including OS, hardware, and GPU.
    """
    os_type, os_version = _detect_os()
    architecture = _detect_architecture()
    hardware_type = _detect_hardware_type(os_type, architecture)

    try:
        cpu_count = len(__import__("os").sched_getaffinity(0))
    except (AttributeError, OSError):
        import os as os_module
        cpu_count = os_module.cpu_count() or 1

    return SystemInfo(
        os=os_type,
        os_version=os_version,
        architecture=architecture,
        hardware_type=hardware_type,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        hostname=platform.node(),
        cpu_count=cpu_count,
        gpu=_detect_gpu(),
        is_wsl=_detect_wsl(),
    )
