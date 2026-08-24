#!/usr/bin/env bash
# Convenience wrapper for macOS/Linux: build_solver.py with a specific python.
#
# Usage:
#   ./scripts/build.sh /path/to/python3.11
#
# If no path is given, uses whichever `python3` is first on PATH - make sure
# that's the standalone Python matching your Blender version, NOT Blender's
# own bundled interpreter.
set -e
PY="${1:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$PY" "$SCRIPT_DIR/build_solver.py"
