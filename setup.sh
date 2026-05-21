#!/bin/sh
set -e
cd "$(dirname "$0")"

ROCM_INDEX="https://download.pytorch.org/whl/rocm6.3"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"
JETSON_TORCH_VERSION="2.11.0"
JETSON_TORCHVISION_VERSION="0.26.0"
PYTHON_VERSION="3.11"

detect_platform() {
    # Allow explicit override via SAM2_BUILD_TARGET env var.
    if [ -n "$SAM2_BUILD_TARGET" ]; then
        echo "$SAM2_BUILD_TARGET"
        return
    fi
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        if [ -f /etc/nv_tegra_release ]; then
            echo "linux-tegra"
        elif command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
            echo "linux-rocm"
        else
            echo "linux-cpu"
        fi
    elif [ "$OS" = "Darwin" ]; then
        echo "darwin"
    else
        echo "linux-cpu"
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo "uv found: $(uv --version)"
        return
    fi
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: Failed to install uv" >&2
        exit 1
    fi
    echo "uv installed: $(uv --version)"
}

PLATFORM="$(detect_platform)"
echo "Detected platform: $PLATFORM"

# Jetson torch wheels are cp310-only; force the venv interpreter to 3.10 there.
if [ "$PLATFORM" = "linux-tegra" ]; then
    PYTHON_VERSION="3.10"
fi

ensure_uv

# Create venv if it doesn't exist.
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    uv venv --python "$PYTHON_VERSION"
fi

# Install PyTorch FIRST with the correct platform index.
if [ "$PLATFORM" = "linux-rocm" ]; then
    echo "Installing PyTorch with ROCm support..."
    uv pip install torch torchvision --index-url "$ROCM_INDEX"
elif [ "$PLATFORM" = "linux-tegra" ]; then
    echo "Installing PyTorch (Jetson JP6 / CUDA 12.6)..."
    uv pip install \
        "torch==$JETSON_TORCH_VERSION" "torchvision==$JETSON_TORCHVISION_VERSION" \
        --index-url "$JETSON_INDEX"
    # nvidia-cudss-cu12 is dlopened by Jetson torch but not declared as a
    # transitive dep. Install it with --no-deps so we don't pull in a
    # CUDA-12.9 cublas/nvrtc that conflicts with the JetPack 12.6 runtime.
    uv pip install --no-deps nvidia-cudss-cu12 --index-url "$JETSON_INDEX"
elif [ "$PLATFORM" = "linux-cpu" ]; then
    echo "Installing PyTorch (CPU only)..."
    uv pip install torch torchvision --index-url "$CPU_INDEX"
else
    echo "Installing PyTorch (standard — includes MPS on macOS)..."
    uv pip install torch torchvision
fi

# Install all other dependencies.
echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Verify torch version. On Jetson, libcudss lives in the venv's nvidia/cu12/lib
# directory and is not on the default linker search path.
VERIFY_LD_PATH=""
if [ "$PLATFORM" = "linux-tegra" ]; then
    VERIFY_LD_PATH="$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cu12/lib"
fi
TORCH_VERSION=$(LD_LIBRARY_PATH="$VERIFY_LD_PATH:$LD_LIBRARY_PATH" \
    .venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
echo "Setup complete (platform: $PLATFORM, torch: $TORCH_VERSION)"
