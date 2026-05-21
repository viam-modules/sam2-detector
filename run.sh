#!/bin/sh
# Wrapper script that sets platform-specific env vars before launching the module binary.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[run.sh] SCRIPT_DIR=$SCRIPT_DIR" >&2
echo "[run.sh] /opt/rocm exists: $([ -d /opt/rocm ] && echo yes || echo no)" >&2
echo "[run.sh] /etc/nv_tegra_release exists: $([ -f /etc/nv_tegra_release ] && echo yes || echo no)" >&2
echo "[run.sh] HSA_OVERRIDE_GFX_VERSION before: ${HSA_OVERRIDE_GFX_VERSION:-NOT SET}" >&2

# Required for AMD GPUs not in PyTorch's official ROCm support list.
if [ -d /opt/rocm ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi

# On Jetson, the Jetson-specific torch wheel dlopens libcudss.so.0 which is
# bundled inside the binary's _internal directory (onedir mode). Add the
# bundled .so dirs to LD_LIBRARY_PATH so they're discoverable at load time.
if [ -f /etc/nv_tegra_release ]; then
    JETSON_LIB_DIRS="$SCRIPT_DIR/dist/main/_internal:$SCRIPT_DIR/dist/main/_internal/nvidia/cu12/lib"
    export LD_LIBRARY_PATH="$JETSON_LIB_DIRS:${LD_LIBRARY_PATH}"
    echo "[run.sh] Jetson LD_LIBRARY_PATH prefix: $JETSON_LIB_DIRS" >&2
fi

echo "[run.sh] HSA_OVERRIDE_GFX_VERSION after: ${HSA_OVERRIDE_GFX_VERSION:-NOT SET}" >&2

# Support both onefile (macOS: dist/main) and onedir (Linux: dist/main/main).
if [ -f "$SCRIPT_DIR/dist/main/main" ]; then
    echo "[run.sh] launching (onedir): $SCRIPT_DIR/dist/main/main $@" >&2
    exec "$SCRIPT_DIR/dist/main/main" "$@"
else
    echo "[run.sh] launching (onefile): $SCRIPT_DIR/dist/main $@" >&2
    exec "$SCRIPT_DIR/dist/main" "$@"
fi
