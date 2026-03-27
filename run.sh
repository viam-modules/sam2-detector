#!/bin/sh
# Wrapper script that sets AMD ROCm env vars before launching the module binary.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[run.sh] SCRIPT_DIR=$SCRIPT_DIR" >&2
echo "[run.sh] /opt/rocm exists: $([ -d /opt/rocm ] && echo yes || echo no)" >&2
echo "[run.sh] HSA_OVERRIDE_GFX_VERSION before: ${HSA_OVERRIDE_GFX_VERSION:-NOT SET}" >&2

# Required for AMD GPUs not in PyTorch's official ROCm support list.
if [ -d /opt/rocm ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi

echo "[run.sh] HSA_OVERRIDE_GFX_VERSION after: ${HSA_OVERRIDE_GFX_VERSION:-NOT SET}" >&2
echo "[run.sh] launching: $SCRIPT_DIR/dist/main $@" >&2

exec "$SCRIPT_DIR/dist/main" "$@"
