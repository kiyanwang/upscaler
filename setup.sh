#!/bin/bash
set -e

echo "=== Upscaler Setup ==="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is required. Install from https://brew.sh"
    exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
brew install ffmpeg sox python@3.12

# Create virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating Python virtual environment..."
/opt/homebrew/bin/python3.12 -m venv "$VENV_DIR"

echo "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Setup complete! ==="
echo "Usage: $VENV_DIR/bin/python $SCRIPT_DIR/upscaler.py ~/Downloads/kaiser.mp4"
echo ""
echo "Or add an alias to your shell:"
echo "  alias upscaler='$VENV_DIR/bin/python $SCRIPT_DIR/upscaler.py'"
