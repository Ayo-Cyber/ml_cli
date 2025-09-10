#!/bin/bash
# ML CLI Project Convenience Script
# This script helps you quickly navigate to your ML project and see available commands

# Get the directory of the script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo "🚀 ML CLI Project Directory"
echo "=========================="
echo "📁 Project location: $SCRIPT_DIR"
echo ""

# Change to project directory
cd "$SCRIPT_DIR"

echo "✅ Changed to project directory!"
echo ""
echo "💡 Available commands:"
echo "   ml train      - Train your model"
echo "   ml serve      - Serve your model as an API"
echo "   ml predict    - Make predictions"
echo "   ml preprocess - Preprocess your data"
echo "   ml eda        - Exploratory data analysis"
echo "   ml clean      - Clean up artifacts"
echo ""
echo "🔍 To get started, run: ml train"
echo ""

# Start a new shell in the project directory
exec $SHELL
