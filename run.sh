#!/bin/sh
# Wrapper script that sets GPU env vars before launching the module binary.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Required for AMD GPUs not in PyTorch's official ROCm support list. NVIDIA needs
# nothing here: the CUDA build carries its own runtime and finds the driver itself.
if [ -d /opt/rocm ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi

# Support both onedir (GPU builds: dist/main/main) and onefile (dist/main).
if [ -f "$SCRIPT_DIR/dist/main/main" ]; then
    exec "$SCRIPT_DIR/dist/main/main" "$@"
else
    exec "$SCRIPT_DIR/dist/main" "$@"
fi
