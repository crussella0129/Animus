"""System detection module - OS and hardware identification."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# Try to import sched_getaffinity for accurate CPU count
try:
    from os import sched_getaffinity
    _HAS_SCHED_GETAFFINITY = True
except ImportError:
    _HAS_SCHED_GETAFFINITY = False


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
        # Parse Windows version to distinguish Windows 10 vs 11
        # Windows reports version as "10.0.XXXXX" for both Win10 and Win11
        # Windows 11 has build number >= 22000
        windows_version = _get_windows_marketing_name(version)
        return OperatingSystem.WINDOWS, windows_version
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


def _get_windows_marketing_name(version: str) -> str:
    """
    Convert Windows build version to marketing name.

    Windows 11: Build 22000+
    Windows 10: Build < 22000

    Args:
        version: Raw version string like "10.0.26200"

    Returns:
        Marketing name like "11 (Build 26200)" or "10 (Build 19045)"
    """
    try:
        # Parse version string "10.0.XXXXX"
        parts = version.split(".")
        if len(parts) >= 3:
            build_number = int(parts[2])
            if build_number >= 22000:
                return f"11 (Build {build_number})"
            else:
                return f"10 (Build {build_number})"
    except (ValueError, IndexError):
        pass

    # Fallback to raw version
    return version


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

    # Check for AMD ROCm (Linux)
    if Path("/opt/rocm").exists():
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu.vendor = "AMD"
                gpu.name = "AMD GPU (ROCm)"
                # Try to parse GPU name from output
                for line in result.stdout.split("\n"):
                    if "GPU" in line and ":" in line:
                        gpu.name = line.split(":")[-1].strip()
                        break
                return gpu
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # ROCm installed but rocm-smi not available or failed
            gpu.vendor = "AMD"
            gpu.name = "AMD GPU (ROCm detected)"
            return gpu

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

    # Get CPU count - prefer sched_getaffinity for accurate count in containers
    if _HAS_SCHED_GETAFFINITY:
        try:
            cpu_count = len(sched_getaffinity(0))
        except OSError:
            cpu_count = os.cpu_count() or 1
    else:
        cpu_count = os.cpu_count() or 1

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
