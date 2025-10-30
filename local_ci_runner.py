#!/usr/bin/env python
"""
Local CI/CD Test Runner
Simulates GitHub Actions workflows locally
"""

import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_header(message):
    """Print a formatted header."""
    print(f"\n{'=' * 50}")
    print(f"🎯 {message}")
    print(f"{'=' * 50}\n")


def print_status(returncode, message):
    """Print colored status message."""
    if returncode == 0:
        print(f"{Colors.GREEN}✅ {message} passed{Colors.NC}\n")
        return True
    else:
        print(f"{Colors.RED}❌ {message} failed{Colors.NC}\n")
        return False


def run_command(cmd, check_name):
    """Run a command and check its return code."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return print_status(result.returncode, check_name)
    except Exception as e:
        print(f"{Colors.RED}Error running {check_name}: {e}{Colors.NC}\n")
        return False


def main():
    """Main CI/CD test runner."""
    print_header("Local CI/CD Test Suite")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists() and not Path("setup.py").exists():
        print(f"{Colors.RED}Error: Project configuration file not found.{Colors.NC}")
        print("Please run this from the project root directory.")
        sys.exit(1)

    all_passed = True

    # Install dependencies
    print_header("Installing Dependencies")
    if not run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Pip upgrade"
    ):
        all_passed = False

    if not run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".", "pytest", "pytest-cov", "flake8", "black"],
        "Dependency installation"
    ):
        print(f"{Colors.YELLOW}Warning: Some dependencies may not be installed{Colors.NC}")

    # 1. Linting
    print_header("Linting")

    print("1️⃣ Linting with flake8...")
    if not run_command(
        ["flake8", "ml_cli", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
        "flake8 linting (critical errors)"
    ):
        all_passed = False

    if not run_command(
        ["flake8", "ml_cli", "--count", "--exit-zero", "--max-complexity=10", "--max-line-length=127", "--statistics"],
        "flake8 linting (style checks)"
    ):
        all_passed = False

    # 2. Code Formatting
    print_header("Code Formatting Checks")

    print("2️⃣ Checking code formatting with Black...")
    if not run_command(
        ["black", "--check", "ml_cli", "tests", "--line-length=127"],
        "Black formatting"
    ):
        print(f"{Colors.YELLOW}💡 Tip: Run 'black ml_cli tests --line-length=127' to fix formatting.{Colors.NC}\n")
        all_passed = False

    # 3. Unit Tests
    print_header("Running Unit Tests")

    if not run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=ml_cli", "--cov-report=xml"],
        "Unit tests with coverage"
    ):
        all_passed = False

    # Final Summary
    print_header("Test Summary")

    if all_passed:
        print(f"{Colors.GREEN}✅ All CI/CD checks passed!{Colors.NC}\n")
        print("Next steps:")
        print("  • Commit your changes: git add . && git commit -m 'your message'")
        print("  • Push to trigger GitHub Actions: git push")
        sys.exit(0)
    else:
        print(f"{Colors.RED}❌ Some checks failed. Please fix the issues above.{Colors.NC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()