#!/bin/bash
# Script to build and publish ml_cli to TestPyPI

set -e  # Exit on error

echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

echo "📦 Building package..."
python -m build

echo "✅ Build complete!"
echo ""
echo "📋 Distribution files:"
ls -lh dist/

echo ""
echo "🚀 To upload to TestPyPI, run:"
echo "   python -m twine upload --repository testpypi dist/*"
echo ""
echo "Or to upload to PyPI, run:"
echo "   python -m twine upload dist/*"
echo ""
echo "Make sure you have twine installed:"
echo "   pip install twine build"
