#!/usr/bin/env bash
set -e

echo "=== Agentic Uncertainty Setup ==="
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed successfully"
else
    echo "uv is already installed: $(uv --version)"
fi

echo

# Navigate to script directory
cd "$(dirname "$0")"

# Pin Python version and sync dependencies
echo "Setting up Python environment..."
uv python pin 3.11 2>/dev/null || true
uv sync

echo
echo "Dependencies installed successfully"

# Check for .env file
if [ ! -f .env ]; then
    echo
    echo "Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env and add your ANTHROPIC_API_KEY"
fi

# Check if modal is configured
echo
if [ ! -f ~/.modal.toml ]; then
    echo "Modal is not configured. Running modal setup..."
    echo "(This will open your browser for authentication)"
    echo
    uv run modal setup
else
    echo "Modal is already configured"
fi

echo
echo "=== Setup complete! ==="
echo
echo "Next steps:"
echo "  1. Edit .env and add your model + Modal credentials"
echo "  2. Check the unified CLI: uv run run-experiment --help"
echo
