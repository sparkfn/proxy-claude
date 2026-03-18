#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV="$DIR/.venv"

# Ensure venv exists
if [ ! -d "$VENV" ]; then
    echo "Setting up Python environment..."
    python3 -m venv "$VENV" || { echo "Error: Python 3 is required. Install it and try again."; exit 1; }
    "$VENV/bin/pip" install -q -r "$DIR/requirements.txt" || { echo "Error: Failed to install dependencies."; exit 1; }
fi

# Ensure deps are installed (fast check: import yaml)
"$VENV/bin/python" -c "import yaml, requests" 2>/dev/null || {
    echo "Installing dependencies..."
    "$VENV/bin/pip" install -q -r "$DIR/requirements.txt" || { echo "Error: Failed to install dependencies."; exit 1; }
}

exec "$VENV/bin/python" "$DIR/cli.py" "$@"
