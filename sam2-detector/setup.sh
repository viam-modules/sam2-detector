#!/bin/sh
set -e
cd "$(dirname "$0")"

# Platform-specific PyTorch index.
# Linux with AMD GPU uses ROCm, everything else uses default PyPI.
ROCM_INDEX="https://download.pytorch.org/whl/rocm6.3"

detect_platform() {
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        # Check for AMD GPU (ROCm).
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

PLATFORM="$(detect_platform)"
echo "Detected platform: $PLATFORM"

if command -v uv >/dev/null 2>&1; then
    echo "Using uv for environment setup..."
    UV_PROJECT="$(pwd)/.."

    # Install PyTorch with the right index for the platform.
    if [ "$PLATFORM" = "linux-rocm" ]; then
        echo "Installing PyTorch with ROCm support..."
        uv pip install --project "$UV_PROJECT" \
            torch torchvision --index-url "$ROCM_INDEX"
    else
        echo "Installing PyTorch (standard)..."
        uv pip install --project "$UV_PROJECT" torch torchvision
    fi

    # Install remaining dependencies via the project.
    uv sync --project "$UV_PROJECT"
    echo "uv setup complete."
else
    echo "Using pip + venv for environment setup..."
    VENV_NAME="venv"
    PYTHON="$VENV_NAME/bin/python"
    ENV_ERROR="This module requires Python >=3.11, pip, and virtualenv to be installed."

    # Create venv if needed.
    if ! python3 -m venv "$VENV_NAME" >/dev/null 2>&1; then
        echo "Failed to create virtualenv."
        if command -v apt-get >/dev/null; then
            echo "Detected Debian/Ubuntu, attempting to install python3-venv."
            SUDO="sudo"
            if ! command -v $SUDO >/dev/null; then
                SUDO=""
            fi
            if ! apt info python3-venv >/dev/null 2>&1; then
                $SUDO apt -qq update >/dev/null
            fi
            $SUDO apt install -qqy python3-venv >/dev/null 2>&1
            if ! python3 -m venv "$VENV_NAME" >/dev/null 2>&1; then
                echo "$ENV_ERROR" >&2
                exit 1
            fi
        else
            echo "$ENV_ERROR" >&2
            exit 1
        fi
    fi

    # Install PyTorch with the right index for the platform.
    if [ "$PLATFORM" = "linux-rocm" ]; then
        echo "Installing PyTorch with ROCm support..."
        $PYTHON -m pip install -qq torch torchvision --index-url "$ROCM_INDEX"
    else
        echo "Installing PyTorch (standard)..."
        $PYTHON -m pip install -qq torch torchvision
    fi

    # Install remaining dependencies.
    echo "Installing requirements..."
    $PYTHON -m pip install -r requirements.txt -Uqq

    echo "pip setup complete."
fi
