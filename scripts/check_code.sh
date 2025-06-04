#!/bin/bash

# Run code quality checks on specific directories

# Default values
DIRECTORY="."
CHECK_ONLY=false
BATCH_SIZE=100
EXCLUDE="venv,.venv,env,build,dist,__pycache__"

# Display help
function show_help {
    echo "Usage: $0 [options]"
    echo "Run code quality checks on WiFi-Radar codebase"
    echo ""
    echo "Options:"
    echo "  -d, --directory DIR   Directory to process (default: current directory)"
    echo "  -c, --check-only      Only check code quality without making changes"
    echo "  -b, --batch-size N    Number of files to process in each batch (default: 100)"
    echo "  -x, --exclude PATTERN Comma-separated list of patterns to exclude"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d wifi_radar      Check and fix code in the wifi_radar directory"
    echo "  $0 -c -d src          Check code quality in the src directory without fixing"
    echo "  $0 -b 50              Process files in batches of 50"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            DIRECTORY="$2"
            shift 2
            ;;
        -c|--check-only)
            CHECK_ONLY=true
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -x|--exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Ensure we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: No virtual environment activated"
    echo "Please activate your virtual environment first:"
    echo "  source .venv/bin/activate"
    exit 1
fi

# Build command
CMD="python src/utils/code_quality.py --directory $DIRECTORY --batch-size $BATCH_SIZE --exclude $EXCLUDE"

if $CHECK_ONLY; then
    CMD="$CMD --check-only"
fi

# Run the command
echo "Running: $CMD"
eval $CMD
