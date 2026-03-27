#!/bin/sh
set -e
cd "$(dirname "$0")"

# Platform-specific PyTorch index.
# Linux with AMD GPU uses ROCm, everything else uses default PyPI.
ROCM_INDEX="https://download.pytorch.org/whl/rocm6.3"

detect_platform() {
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        if command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
            echo "linux-rocm"
        else
            echo "linux"
        fi
    elif [ "$OS" = "Darwin" ]; then
        echo "darwin"
    else
        echo "unknown"
    fi
}

# Install uv if not already available.
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

# Create venv and install project dependencies from pyproject.toml.
# (torch/torchvision are excluded from pyproject.toml — installed below.)
uv sync

# Install PyTorch with the correct index for the platform.
# Done after uv sync so the venv exists, and won't be overwritten since
# torch is not in pyproject.toml.
if [ "$PLATFORM" = "linux-rocm" ]; then
    echo "Installing PyTorch with ROCm support..."
    uv pip install torch torchvision --index-url "$ROCM_INDEX"
else
    echo "Installing PyTorch (standard)..."
    uv pip install torch torchvision
fi

echo "Setup complete."
