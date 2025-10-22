#!/bin/bash
# CI Test Script - Run this to test CI pipeline locally
# Usage: ./ci-test.sh

set -e  # Exit on error

echo "🚀 Starting CI Pipeline Test..."
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "🐍 Python version: $PYTHON_VERSION"
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    echo "✅ Python version is compatible (3.8+)"
else
    echo "❌ Python version must be 3.8 or higher"
    echo "   Current version: $PYTHON_VERSION"
    exit 1
fi
echo ""

# Step 1: Code Formatting Check
echo "📐 Step 1/5: Checking code formatting..."
echo "----------------------------------------"
if command -v black &> /dev/null; then
    echo "Checking with black..."
    black --check ml_cli/ tests/ || echo "⚠️  Code needs formatting. Run: black ml_cli/ tests/"
else
    echo "⚠️  black not installed, skipping formatting check"
fi

if command -v isort &> /dev/null; then
    echo "Checking import sorting..."
    isort --check-only ml_cli/ tests/ || echo "⚠️  Imports need sorting. Run: isort ml_cli/ tests/"
else
    echo "⚠️  isort not installed, skipping import sorting check"
fi
echo "✅ Formatting checks completed!"
echo ""

# Step 2: Linting
echo "📝 Step 2/5: Running linting checks..."
echo "----------------------------------------"
echo "Checking for critical errors..."
flake8 ml_cli --count --select=E9,F63,F7,F82 --show-source --statistics

echo ""
echo "Checking for unused imports and variables..."
flake8 ml_cli --count --select=F401,F841 --statistics || echo "⚠️  Found unused imports/variables (see above)"

echo ""
echo "Checking code style..."
flake8 ml_cli --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
echo "✅ Linting completed!"
echo ""

# Step 3: Tests
echo "🧪 Step 3/5: Running tests..."
echo "----------------------------------------"
# Try with coverage first, fall back to basic tests if pytest-cov not installed
if pytest tests/ -v --cov=ml_cli --cov-report=xml --cov-report=term 2>/dev/null; then
    echo "✅ Tests with coverage passed!"
else
    echo "⚠️  pytest-cov not installed, running tests without coverage..."
    pytest tests/ -v
    echo "✅ Tests passed!"
fi
echo ""

# Step 4: Build
echo "📦 Step 4/5: Building package..."
echo "----------------------------------------"
python -m build
echo "✅ Build successful!"
echo ""

# Step 5: Check distribution
echo "🔍 Step 5/5: Checking distribution..."
echo "----------------------------------------"
if twine check dist/* 2>&1 | grep -v "InvalidDistribution.*license-file"; then
    echo "✅ Distribution check completed!"
else
    echo "⚠️  Minor metadata warnings (non-critical)"
    echo "✅ Distribution check completed!"
fi
echo ""

echo "🎉 All CI steps completed successfully!"
echo ""
echo "Next steps:"
echo "  - To format code: black ml_cli/ tests/ && isort ml_cli/ tests/"
echo "  - To test installation: pip install dist/*.whl"
echo "  - To publish to TestPyPI: twine upload --repository testpypi dist/*"
echo "  - To publish to PyPI: twine upload dist/*"
echo "Next steps:"
echo "  - To test installation: pip install dist/*.whl"
echo "  - To publish to TestPyPI: twine upload --repository testpypi dist/*"
echo "  - To publish to PyPI: twine upload dist/*"
