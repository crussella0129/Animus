"""OS, hardware, and GPU detection utilities."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str = "Unknown"
    memory_mb: int = 0
    cuda_available: bool = False
    driver_version: str = ""


@dataclass
class SystemInfo:
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    cpu_count: int = 0
    hardware_type: str = "x86"  # x86, apple_silicon, jetson
    gpu: GPUInfo = field(default_factory=GPUInfo)
    python_version: str = ""

    @property
    def is_windows(self) -> bool:
        return self.os_name == "Windows"

    @property
    def is_windows_11(self) -> bool:
        if not self.is_windows:
            return False
        try:
            build = int(self.os_version.split(".")[-1])
            return build >= 22000
        except (ValueError, IndexError):
            return False


def _detect_gpu() -> GPUInfo:
    """Detect NVIDIA GPU via nvidia-smi."""
    gpu = GPUInfo()
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return gpu

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                gpu.name = parts[0].strip()
                gpu.memory_mb = int(float(parts[1].strip()))
                gpu.driver_version = parts[2].strip()
                gpu.cuda_available = True
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return gpu


def _detect_hardware_type() -> str:
    """Detect if running on Jetson, Apple Silicon, or standard x86."""
    machine = platform.machine().lower()

    # Apple Silicon
    if platform.system() == "Darwin" and machine == "arm64":
        return "apple_silicon"

    # Jetson (aarch64 Linux with NVIDIA)
    if machine == "aarch64" and platform.system() == "Linux":
        if os.path.exists("/etc/nv_tegra_release") or os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    if "jetson" in f.read().lower():
                        return "jetson"
            except (FileNotFoundError, PermissionError):
                pass
            if os.path.exists("/etc/nv_tegra_release"):
                return "jetson"

    return "x86"


def detect_system() -> SystemInfo:
    """Detect full system information."""
    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        architecture=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        hardware_type=_detect_hardware_type(),
        gpu=_detect_gpu(),
        python_version=platform.python_version(),
    )
