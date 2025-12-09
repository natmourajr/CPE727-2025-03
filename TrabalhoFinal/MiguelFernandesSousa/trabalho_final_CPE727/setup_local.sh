#!/bin/bash
# Quick setup script for local development

set -e

echo "==================================="
echo "IARA Local Setup"
echo "==================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed"
    echo "⚠️  Please restart your shell and run this script again"
    exit 0
fi

echo "✅ uv is installed"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate and install
echo "Installing dependencies..."
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

echo ""
echo "==================================="
echo "✅ Setup complete!"
echo "==================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  python test_dataset.py --mode unittest"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
