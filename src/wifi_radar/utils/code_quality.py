#!/usr/bin/env python3
"""
ID: WR-UTIL-QA-001
Requirement: Provide CLI-driven automation for autoflake, isort, black, docformatter,
             flake8, pydocstyle, and mypy across the entire WiFi-Radar Python codebase.
Purpose: Enforce consistent code style, import ordering, and docstring formatting
         with a single command during development and CI pipelines.
Rationale: Running all quality tools from one module avoids per-tool configuration
           drift and ensures the correct execution order (autoflake -> isort -> black
           -> docformatter) so tools do not undo each other's changes.
Assumptions: autoflake, isort, black, docformatter, flake8, pydocstyle, and mypy are
             installed in the active virtual environment.
Constraints: File batching keeps individual subprocess argument lengths within ARG_MAX.
References: autoflake, isort, black, docformatter, flake8, pydocstyle, mypy documentation.
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
    Run a shell command and return its exit code.

    ID: WR-UTIL-QA-CMD-001
    Requirement: Execute a subprocess command, capture stdout/stderr, and return
                 the exit code while logging success or failure at the correct level.
    Purpose: Provide a uniform wrapper around subprocess.run so all quality-tool
             invocations produce consistent logging output.
    Rationale: Centralising subprocess logic avoids duplicating error-handling and
               DOCFORMATTER_CONFIG_FILE environment injection across call sites.
    Inputs:
        command     — List[str]: command and arguments, e.g. ["black", "foo.py"].
        description — str: human-readable label used in INFO/ERROR log messages.
    Outputs:
        int: exit code of the subprocess (0 = success, non-zero = failure).
    Preconditions:
        command[0] must be an executable reachable via PATH or as an absolute path.
    Postconditions:
        The subprocess has completed and stdout/stderr have been fully consumed.
    Assumptions:
        The calling environment has the required quality tools installed.
    Side Effects:
        Spawns a subprocess; in-place formatters may modify files on disk.
        Logs INFO on success, ERROR on non-zero exit or exception.
        Injects DOCFORMATTER_CONFIG_FILE=none when "docformatter" is in command[0].
    Failure Modes:
        FileNotFoundError if command[0] is not found — caught, logged, returns 1.
        Any other exception from subprocess.run — caught, logged, returns 1.
    Error Handling:
        All exceptions from subprocess.run are caught; function always returns int.
    Constraints:
        stdout/stderr are buffered in memory; avoid commands that produce huge output.
    Verification:
        Unit test: pass ["echo", "hi"] and assert return code 0 and INFO logged.
    References:
        subprocess.run() Python standard library documentation.
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
    Process a list of files through a command in fixed-size batches.

    ID: WR-UTIL-QA-BATCH-001
    Requirement: Split a file list into fixed-size batches and call run_command()
                 once per batch, aggregating return codes via bitwise OR.
    Purpose: Avoid 'Argument list too long' (E2BIG) OS errors when the combined
             length of all file paths exceeds the kernel ARG_MAX limit.
    Rationale: Batching is the standard E2BIG mitigation; OR-ing return codes ensures
               any batch failure propagates to the final return value.
    Inputs:
        command_prefix — List[str]: command and fixed flags without file arguments.
        files          — List[str]: absolute or relative file paths to process.
        description    — str: label for log messages.
        batch_size     — int > 0: maximum files per batch invocation.
    Outputs:
        int: bitwise OR of all batch return codes (0 iff all batches succeeded).
    Preconditions:
        run_command() is defined in the same module.
        command_prefix[0] must be a valid executable.
    Postconditions:
        Every file in ``files`` has been passed to the command exactly once.
    Assumptions:
        Total command length per batch stays within OS ARG_MAX for batch_size <= 200.
    Side Effects:
        Calls run_command() once per batch; may modify files on disk.
        Logs batch progress at INFO level.
    Failure Modes:
        If a batch returns non-zero, processing continues for remaining batches.
    Error Handling:
        Relies on run_command() for per-batch error handling; aggregates via OR.
    Constraints:
        batch_size should be <= 200 to stay within typical ARG_MAX margins.
    Verification:
        Integration test: pass 250 temp file paths with batch_size=100;
        verify 3 separate run_command calls are made.
    References:
        POSIX ARG_MAX; run_command() in this module.
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
    Fix code quality issues in a directory using autoflake, isort, black, and docformatter.

    ID: WR-UTIL-QA-FIX-001
    Requirement: Automatically correct import ordering, unused imports, style violations,
                 and malformed docstrings in all Python files under ``directory``.
    Purpose: Enforce consistent code style across the codebase with a single call,
             reducing manual formatting effort in CI and local development.
    Rationale: Running tools in order (autoflake -> isort -> black -> docformatter)
               avoids conflicts where later tools undo earlier tool output.
    Inputs:
        directory      — Path: root directory searched recursively for *.py files.
        fix_imports    — bool: if True, run autoflake + isort.
        fix_style      — bool: if True, run black.
        fix_docstrings — bool: if True, run docformatter --in-place.
        batch_size     — int > 0: maximum files per tool invocation batch.
    Outputs:
        int: bitwise OR of all tool return codes (0 iff all tools succeeded).
    Preconditions:
        autoflake, isort, black, and docformatter must be installed.
        ``directory`` must exist and be a valid Path object.
    Postconditions:
        Python files under ``directory`` have been modified in-place by enabled tools.
    Assumptions:
        Tool binaries are available on PATH; files are writable by the current process.
    Side Effects:
        Modifies .py files in-place.
        Logs tool names, file counts, and batch progress at INFO level.
    Failure Modes:
        Tool not found: run_command() catches and returns 1 for that tool.
        Syntax errors in files: black skips malformed files and returns non-zero.
    Error Handling:
        Delegates to run_command(); individual failures do not abort subsequent tools.
    Constraints:
        Large repos may take minutes; tune batch_size vs. ARG_MAX as needed.
    Verification:
        Integration test: temp directory with a style-violating .py file;
        run fix_code; assert black and isort produce expected reformatted output.
    References:
        autoflake, isort, black, docformatter docs; process_files_in_batches().
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
    Check code quality without making changes, using flake8, pydocstyle, and mypy.

    ID: WR-UTIL-QA-CHECK-001
    Requirement: Report style violations, docstring issues, and type errors in all
                 Python files under ``directory`` without modifying any files.
    Purpose: Provide a read-only audit for CI pipelines that gate pull requests
             on passing code quality standards.
    Rationale: Check-only mode is safe in CI where write access to sources is
               undesirable; each tool's non-zero exit code surfaces violations.
    Inputs:
        directory  — Path: root directory searched recursively for *.py files.
        batch_size — int > 0: maximum files per tool invocation batch.
    Outputs:
        int: bitwise OR of all tool return codes (0 iff all checks pass).
    Preconditions:
        flake8, pydocstyle, and mypy must be installed in the active environment.
        ``directory`` must exist and be a valid Path object.
    Postconditions:
        No files are modified; tool output is logged to the logger.
    Assumptions:
        Tool binaries are reachable via PATH.
    Side Effects:
        Logs tool names, file counts, and batch progress at INFO level.
        Logs ERROR messages for each failing batch.
    Failure Modes:
        Missing tool: run_command() returns 1 for that tool; others still run.
    Error Handling:
        Delegates to run_command(); individual failures do not abort other tools.
    Constraints:
        mypy type-checking may be slow on large codebases without a type cache.
    Verification:
        Integration test: temp directory with a flake8 violation;
        run check_code; assert return code is non-zero.
    References:
        flake8, pydocstyle, mypy docs; process_files_in_batches() in this module.
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
    Entry point for the code-quality CLI script.

    ID: WR-UTIL-QA-MAIN-001
    Requirement: Parse command-line arguments and dispatch to fix_code() or
                 check_code() based on the --check-only flag.
    Purpose: Provide a standalone executable interface for quality tools during
             development and CI without importing the module directly.
    Rationale: A dedicated main() decouples CLI argument parsing from core logic
               functions, keeping them independently testable.
    Inputs:
        None — reads sys.argv via argparse.ArgumentParser.
        CLI flags: --directory, --check-only, --no-imports, --no-style,
                   --no-docstrings, --batch-size, --exclude, --no-config.
    Outputs:
        int: 0 on success, non-zero on any tool failure.
    Preconditions:
        Called as __main__ or via a script entry point; sys.argv is populated.
    Postconditions:
        fix_code() or check_code() has been called; return code reflects outcome.
    Assumptions:
        The directory argument resolves to an existing directory on disk.
    Side Effects:
        Sets os.environ['CODE_QUALITY_EXCLUDE'] and optionally formatter config vars.
        Calls fix_code() or check_code(), which may modify files on disk.
    Failure Modes:
        Non-existent directory argument: logs ERROR and returns 1.
    Error Handling:
        Directory existence is validated before dispatch; returns 1 with log on failure.
    Constraints:
        Argument list is fixed; extend argparse to add new tool support.
    Verification:
        Run python -m wifi_radar.utils.code_quality --check-only on the project root;
        assert exit code 0 on a clean codebase.
    References:
        argparse Python standard library; fix_code(), check_code() in this module.
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
