#!/usr/bin/env python3
"""Animus Bootstrap Installer.

This script can be run directly after cloning to set up Animus.
It installs base dependencies first, then runs the full installer.

Usage:
    python install.py [options]

After running this script, you can use:
    animus install   - for future reinstalls/updates
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def print_banner():
    """Print Animus banner."""
    banner = """
 ▄▀▀█▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄    ▄▀▀▄ ▄▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀▀▀▄
▐ ▄▀ ▀▄ █  █ █ █ █   █  █  █  █ ▀  █ █   █    █ █ █   ▐
  █▄▄▄█ ▐  █  ▀█ ▐   █  ▐  ▐  █    █ ▐  █    █     ▀▄
 ▄▀   █   █   █      █       █    █    █    █   ▀▄   █
█   ▄▀  ▄▀   █    ▄▀▀▀▀▀▄  ▄▀   ▄▀      ▀▄▄▄▄▀   █▀▀▀
▐   ▐   █    ▐   █       █ █    █                ▐
        ▐        ▐       ▐ ▐    ▐

              ✧ Animus Installer ✧
    """
    print(banner)


def check_python_version():
    """Check Python version meets requirements."""
    if sys.version_info < (3, 11):
        print(f"Error: Python 3.11+ required, found {sys.version}")
        print("Please install Python 3.11 or later.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def detect_system():
    """Detect system information."""
    os_name = platform.system().lower()
    arch = platform.machine().lower()

    # Map architecture names
    if arch in ("x86_64", "amd64"):
        arch = "x86_64"
    elif arch in ("arm64", "aarch64"):
        arch = "arm64"

    # Check for special hardware
    hardware = "standard"

    # Jetson detection
    if os_name == "linux" and arch == "arm64":
        if Path("/etc/nv_tegra_release").exists():
            hardware = "jetson"
        elif Path("/sys/firmware/devicetree/base/model").exists():
            try:
                model = Path("/sys/firmware/devicetree/base/model").read_text().lower()
                if "jetson" in model or "tegra" in model:
                    hardware = "jetson"
            except Exception:
                pass

    # Apple Silicon detection
    if os_name == "darwin" and arch == "arm64":
        hardware = "apple_silicon"

    print(f"✓ System: {os_name} {arch} ({hardware})")
    return os_name, arch, hardware


def check_pip():
    """Ensure pip is available."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            check=True,
        )
        print("✓ pip available")
        return True
    except subprocess.CalledProcessError:
        print("✗ pip not found")
        return False


def install_base_deps():
    """Install base dependencies needed to run the full installer."""
    print("\nInstalling base dependencies...")
    deps = [
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "httpx>=0.25.0",
    ]

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + deps,
            check=True,
        )
        print("✓ Base dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install base dependencies: {e}")
        return False


def install_package():
    """Install the animus package in editable mode."""
    print("\nInstalling Animus...")
    try:
        # Get the directory containing this script
        script_dir = Path(__file__).parent.resolve()

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(script_dir), "--quiet"],
            check=True,
        )
        print("✓ Animus installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Animus: {e}")
        return False


def run_full_installer(args):
    """Run the full Animus installer."""
    print("\nRunning full installer...")
    print("-" * 40)

    # Build command
    cmd = [sys.executable, "-m", "src.main", "install"]

    if args.skip_native:
        cmd.append("--skip-native")
    if args.skip_embeddings:
        cmd.append("--skip-embeddings")
    if args.cpu:
        cmd.append("--cpu")
    if args.verbose:
        cmd.append("--verbose")

    # Run the installer
    script_dir = Path(__file__).parent.resolve()
    result = subprocess.run(cmd, cwd=script_dir)

    return result.returncode == 0


def show_quickstart():
    """Show quickstart guide."""
    print("\n" + "=" * 50)
    print("Quickstart Guide")
    print("=" * 50)
    print("""
1. Download a model:
   animus vessel download TheBloke/Llama-2-7B-Chat-GGUF

   Or use Ollama:
   ollama pull llama2

2. Start Animus:
   animus rise

3. Other useful commands:
   animus sense      - Show system info
   animus commune    - Check provider status
   animus --help     - Show all commands
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install Animus and its dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python install.py                 # Full installation with GPU support
    python install.py --cpu           # CPU-only installation
    python install.py --skip-native   # Skip llama-cpp-python (use Ollama)
    python install.py --minimal       # Minimal installation (base deps only)
        """,
    )

    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Skip llama-cpp-python installation (will use Ollama instead)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip sentence-transformers installation",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only installation (no GPU acceleration)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Minimal installation (base dependencies only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # If minimal, set skip flags
    if args.minimal:
        args.skip_native = True
        args.skip_embeddings = True

    print_banner()
    print("Checking requirements...\n")

    # Check Python version
    check_python_version()

    # Detect system
    os_name, arch, hardware = detect_system()

    # Check pip
    if not check_pip():
        print("\nPlease install pip and try again.")
        sys.exit(1)

    # Install base deps
    if not install_base_deps():
        print("\nFailed to install base dependencies.")
        sys.exit(1)

    # Install package
    if not install_package():
        print("\nFailed to install Animus package.")
        sys.exit(1)

    # Run full installer (unless minimal)
    if not args.minimal:
        success = run_full_installer(args)
        if not success:
            print("\nInstallation completed with some errors.")
            print("You may need to install some components manually.")
    else:
        print("\n✓ Minimal installation complete")

    # Show quickstart
    show_quickstart()


if __name__ == "__main__":
    main()
