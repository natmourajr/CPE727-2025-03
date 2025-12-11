#!/bin/bash
# Fix PyTorch import error on macOS

echo "🔧 Fixing PyTorch import issue..."
echo ""

# Remove torch cache from venv
if [ -d ".venv/lib/python3.13/site-packages/torch/__pycache__" ]; then
    echo "Removing torch __pycache__..."
    rm -rf .venv/lib/python3.13/site-packages/torch/__pycache__
    echo "✓ Done"
else
    echo "✓ Torch __pycache__ already clean"
fi

# Test import
echo ""
echo "Testing import..."
if uv run python -c "from src.data_loader import FashionMNISTLoader; print('✓ Import successful')" 2>/dev/null; then
    echo ""
    echo "🎉 PyTorch is working!"
else
    echo ""
    echo "❌ Still having issues. Try:"
    echo "   uv pip uninstall torch torchvision"
    echo "   uv pip install torch torchvision"
fi
