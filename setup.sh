#!/bin/sh
set -e
cd "$(dirname "$0")"

ROCM_INDEX="https://download.pytorch.org/whl/rocm6.3"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
PYTHON_VERSION="3.11"

detect_platform() {
    # Allow explicit override via SAM2_BUILD_TARGET env var.
    if [ -n "$SAM2_BUILD_TARGET" ]; then
        echo "$SAM2_BUILD_TARGET"
        return
    fi
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        if command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
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

# Verify torch version.
TORCH_VERSION=$(.venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
echo "Setup complete (platform: $PLATFORM, torch: $TORCH_VERSION)"
