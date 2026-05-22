#!/bin/sh
# Wrapper script that sets AMD ROCm env vars before launching the module binary.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Required for AMD GPUs not in PyTorch's official ROCm support list.
if [ -d /opt/rocm ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi

# Support both onefile (macOS: dist/main) and onedir (Linux: dist/main/main).
if [ -f "$SCRIPT_DIR/dist/main/main" ]; then
    exec "$SCRIPT_DIR/dist/main/main" "$@"
else
    exec "$SCRIPT_DIR/dist/main" "$@"
fi
