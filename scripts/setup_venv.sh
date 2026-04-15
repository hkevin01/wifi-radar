#!/bin/bash

# Set up virtual environment for WiFi-Radar
# This script creates a .venv directory, activates it, and
# configures VS Code to use it by default

echo "Setting up Python virtual environment for WiFi-Radar..."

# Create virtual environment in .venv directory
python -m venv .venv

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Create settings.json to tell VS Code to use this environment
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "\${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "terminal.integrated.defaultProfile.linux": "bash",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.mypyEnabled": true
}
EOF

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make the script executable
chmod +x scripts/code_quality.py

echo ""
echo "Virtual environment setup complete!"
echo "The VS Code Python extension will now use .venv as the default interpreter."
echo ""
echo "To activate the environment in a new terminal, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To install the pre-commit hooks, run:"
echo "  pre-commit install"
echo ""
echo "To run code quality checks, run:"
echo "  python scripts/code_quality.py"
