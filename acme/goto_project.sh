#!/bin/bash
# Simple navigation script
# Usage: source goto_project.sh

# Get the directory of the script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd "$SCRIPT_DIR"
echo "✅ Navigated to ML project directory: $SCRIPT_DIR"
