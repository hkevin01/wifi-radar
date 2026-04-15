#!/usr/bin/env python3
"""
Script to automatically fix code quality issues in the WiFi-Radar codebase.

This script runs various code quality tools to automatically fix syntax,
style, and docstring issues in the Python code.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("CodeQuality")


def run_command(command: List[str], description: str) -> int:
    """
    Run a shell command and log the output.

    Args:
        command: Command to run as a list of strings
        description: Description of the command for logging

    Returns:
        Return code of the command
    """
    logger.info(f"Running {description}...")
    try:
        # Add environment variable to disable docformatter config file lookup
        env = os.environ.copy()
        if "docformatter" in command[0]:
            env["DOCFORMATTER_CONFIG_FILE"] = "none"

        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.error(f"{description} failed with return code {result.returncode}")
            logger.error(f"Output: {result.stdout}")
            logger.error(f"Error: {result.stderr}")
        else:
            logger.info(f"{description} completed successfully")
            if result.stdout:
                logger.debug(f"Output: {result.stdout}")
        return result.returncode
    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        return 1


def process_files_in_batches(
    command_prefix: List[str], files: List[str], description: str, batch_size: int
) -> int:
    """
    Process files in batches to avoid 'Argument list too long' error.

    Args:
        command_prefix: The command to run without file arguments
        files: List of files to process
        description: Description of the command for logging
        batch_size: Number of files to process in each batch

    Returns:
        Return code (0 for success, non-zero for errors)
    """
    return_code = 0
    total_files = len(files)

    if total_files == 0:
        return 0

    logger.info(
        f"Processing {total_files} files in batches of {batch_size} for {description}"
    )

    for i in range(0, total_files, batch_size):
        batch_end = min(i + batch_size, total_files)
        batch = files[i:batch_end]

        logger.info(
            f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({len(batch)} files)"
        )
        command = command_prefix + batch
        batch_result = run_command(
            command, f"{description} (batch {i//batch_size + 1})"
        )
        return_code |= batch_result

    return return_code


def fix_code(
    directory: Path,
    fix_imports: bool = True,
    fix_style: bool = True,
    fix_docstrings: bool = True,
    batch_size: int = 100,
) -> int:
    """
    Fix code quality issues in the given directory.

    Args:
        directory: Directory to process
        fix_imports: Whether to fix import sorting
        fix_style: Whether to fix code style
        fix_docstrings: Whether to fix docstrings
        batch_size: Number of files to process in each batch

    Returns:
        Return code (0 for success, non-zero for errors)
    """
    return_code = 0

    # Find Python files
    python_files = list(directory.glob("**/*.py"))
    logger.info(f"Found {len(python_files)} Python files to process")

    # Convert to strings for command-line tools
    files = [str(f) for f in python_files]

    if not files:
        logger.warning("No Python files found")
        return 0

    # Fix unused imports with autoflake
    if fix_imports:
        cmd_prefix = [
            "autoflake",
            "--in-place",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
        ]
        return_code |= process_files_in_batches(
            cmd_prefix, files, "autoflake (remove unused imports)", batch_size
        )

        # Sort imports with isort
        cmd_prefix = ["isort"]
        return_code |= process_files_in_batches(
            cmd_prefix, files, "isort (sort imports)", batch_size
        )

    # Fix style with black
    if fix_style:
        cmd_prefix = ["black"]
        return_code |= process_files_in_batches(
            cmd_prefix, files, "black (code formatting)", batch_size
        )

    # Fix docstrings with docformatter
    if fix_docstrings:
        cmd_prefix = [
            "docformatter",
            "--in-place",
            "--wrap-summaries=100",
            "--wrap-descriptions=100",
            # Skip config file to avoid setup.cfg parsing errors
            "--config=none",
        ]
        return_code |= process_files_in_batches(
            cmd_prefix, files, "docformatter (docstring formatting)", batch_size
        )

    return return_code


def check_code(directory: Path, batch_size: int = 100) -> int:
    """
    Check code quality without making changes.

    Args:
        directory: Directory to process
        batch_size: Number of files to process in each batch

    Returns:
        Return code (0 for success, non-zero for errors)
    """
    return_code = 0

    # Find Python files
    python_files = list(directory.glob("**/*.py"))
    logger.info(f"Found {len(python_files)} Python files to check")

    # Convert to strings for command-line tools
    files = [str(f) for f in python_files]

    if not files:
        logger.warning("No Python files found")
        return 0

    # Check with flake8
    cmd_prefix = ["flake8"]
    return_code |= process_files_in_batches(
        cmd_prefix, files, "flake8 (code linting)", batch_size
    )

    # Check with pydocstyle
    cmd_prefix = ["pydocstyle"]
    return_code |= process_files_in_batches(
        cmd_prefix, files, "pydocstyle (docstring checking)", batch_size
    )

    # Check with mypy
    cmd_prefix = ["mypy"]
    return_code |= process_files_in_batches(
        cmd_prefix, files, "mypy (type checking)", batch_size
    )

    return return_code


def main() -> int:
    """
    Main function to run the code quality script.

    Returns:
        Return code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Automatically fix code quality issues in the WiFi-Radar codebase"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to process (default: current directory)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check code quality without making changes",
    )
    parser.add_argument(
        "--no-imports",
        action="store_true",
        help="Skip import sorting and unused import removal",
    )
    parser.add_argument(
        "--no-style", action="store_true", help="Skip code style formatting"
    )
    parser.add_argument(
        "--no-docstrings", action="store_true", help="Skip docstring formatting"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process in each batch (default: 100)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="venv,.venv,env,build,dist",
        help="Comma-separated list of patterns to exclude from processing",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore any configuration files like setup.cfg or pyproject.toml",
    )

    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        logger.error(f"Directory {directory} does not exist")
        return 1

    # Set environment variable for exclusion patterns
    os.environ["CODE_QUALITY_EXCLUDE"] = args.exclude

    # Disable configuration file lookup to avoid errors
    if args.no_config:
        os.environ["DOCFORMATTER_CONFIG_FILE"] = "none"
        os.environ["BLACK_CONFIG_FILE"] = "none"
        os.environ["ISORT_CONFIG_FILE"] = "none"

    logger.info(f"Processing directory: {directory}")

    if args.check_only:
        return check_code(directory, batch_size=args.batch_size)
    else:
        return fix_code(
            directory,
            fix_imports=not args.no_imports,
            fix_style=not args.no_style,
            fix_docstrings=not args.no_docstrings,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    sys.exit(main())
