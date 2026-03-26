#!/bin/sh
set -e
cd "$(dirname "$0")"

# Use uv if available, otherwise fall back to a local venv.
if command -v uv >/dev/null 2>&1; then
    uv run --project .. pyinstaller \
        --onefile \
        --hidden-import="googleapiclient" \
        --hidden-import="viam" \
        --hidden-import="sam2" \
        src/main.py
else
    VENV_NAME="venv"
    PYTHON="$VENV_NAME/bin/python"
    $PYTHON -m pip install pyinstaller -Uqq
    $PYTHON -m PyInstaller \
        --onefile \
        --hidden-import="googleapiclient" \
        --hidden-import="viam" \
        --hidden-import="sam2" \
        src/main.py
fi

echo "Built dist/main"
