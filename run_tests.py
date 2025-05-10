#!/usr/bin/env python3
"""
Arcanum Test Runner
---------------
Runs the Arcanum test suite with proper configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_path=None, verbosity=1, failfast=False, markers=None, skip_missing=True):
    """
    Run the Arcanum test suite.

    Args:
        test_path: Specific test file or directory to run (optional)
        verbosity: Pytest verbosity level (0-3)
        failfast: Stop on first failure
        markers: Pytest markers to select specific tests
        skip_missing: Skip tests with missing dependencies (default: True)

    Returns:
        Exit code from pytest
    """
    # Ensure project root is in Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Add verbosity flag
    if verbosity > 0:
        cmd.append(f"-{'v' * verbosity}")

    # Add failfast flag
    if failfast:
        cmd.append("--exitfirst")

    # Skip tests with missing dependencies
    if skip_missing:
        cmd.append("-k")
        cmd.append("not AttributeError")  # Skip tests that fail with AttributeError

    # Add markers if specified
    if markers:
        cmd.append(f"-m '{markers}'")

    # Add test path if specified, otherwise run all tests
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    # Print command
    print(f"Running: {' '.join(cmd)}")

    # Run pytest
    return subprocess.call(cmd)


def run_integration_tests(test_path=None, verbosity=1, failfast=False, skip_missing=True):
    """Run only integration tests."""
    # If a specific test path is provided, use it, otherwise run all integration tests
    if test_path:
        path = test_path
    else:
        path = "tests/integration/"

    return run_tests(path, verbosity, failfast, skip_missing=skip_missing)


def run_unit_tests(test_path=None, verbosity=1, failfast=False, skip_missing=True):
    """Run only unit tests."""
    # If a specific test path is provided, use it, otherwise run all unit tests
    if test_path:
        path = test_path
    else:
        path = "tests/unit/"

    return run_tests(path, verbosity, failfast, skip_missing=skip_missing)


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Arcanum Test Runner")
    parser.add_argument("--path", help="Specific test file or directory to run")
    parser.add_argument("--verbosity", "-v", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Verbosity level (0-3)")
    parser.add_argument("--failfast", "-f", action="store_true",
                        help="Stop after first failure")
    parser.add_argument("--unit", action="store_true",
                        help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                        help="Run only integration tests")
    parser.add_argument("--markers", "-m", help="Pytest markers to select specific tests")
    parser.add_argument("--include-missing", action="store_true",
                        help="Include tests with missing dependencies (may cause failures)")

    args = parser.parse_args()

    # Determine if we should skip tests with missing dependencies
    skip_missing = not args.include_missing

    # Run appropriate tests
    if args.unit:
        return run_unit_tests(args.path, args.verbosity, args.failfast, skip_missing)
    elif args.integration:
        return run_integration_tests(args.path, args.verbosity, args.failfast, skip_missing)
    else:
        return run_tests(args.path, args.verbosity, args.failfast, args.markers, skip_missing)


if __name__ == "__main__":
    sys.exit(main())