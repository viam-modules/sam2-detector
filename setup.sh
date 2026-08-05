#!/bin/sh
set -e
cd "$(dirname "$0")"

ROCM_INDEX="https://download.pytorch.org/whl/rocm6.3"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
PYTHON_VERSION="3.11"

# Pinned for the CUDA build so the bundled NVIDIA library set (and therefore the
# tarball size) is reproducible. CUDA 12.8 wheels carry their own CUDA runtime,
# so target machines need only an NVIDIA driver (>= 525), not the CUDA toolkit.
CUDA_TORCH="torch==2.9.1"
CUDA_TORCHVISION="torchvision==0.24.1"

. ./detect_target.sh

install_uv() {
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: Failed to install uv" >&2
        exit 1
    fi
    echo "uv installed: $(uv --version)"
}

# uv older than 0.7 cannot resolve the +cuXXX local-version wheels from the
# PyTorch index and fails with a misleading "no matching Python implementation
# tag" error, so upgrade instead of letting the torch install fail later.
uv_is_too_old() {
    version="$(uv --version 2>/dev/null | awk '{print $2}')"
    major="${version%%.*}"
    rest="${version#*.}"
    minor="${rest%%.*}"
    case "$major$minor" in
        ''|*[!0-9]*) return 1 ;;  # unparseable: assume it is fine
    esac
    [ "$major" -eq 0 ] && [ "$minor" -lt 7 ]
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found, installing..."
        install_uv
        return
    fi
    if uv_is_too_old; then
        echo "uv $(uv --version | awk '{print $2}') is too old for PyTorch index resolution, upgrading..."
        install_uv
        return
    fi
    echo "uv found: $(uv --version)"
}

PLATFORM="$(detect_sam2_target)"
echo "Build target: $PLATFORM"

ensure_uv

# Create venv if it doesn't exist.
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    uv venv --python "$PYTHON_VERSION"
fi

# Install PyTorch FIRST with the correct platform index. The index URL decides
# whether the wheel has GPU kernels at all — a CPU wheel can never use a GPU,
# no matter what the host hardware is.
case "$PLATFORM" in
    linux-rocm)
        echo "Installing PyTorch with ROCm support (AMD GPUs)..."
        uv pip install torch torchvision --index-url "$ROCM_INDEX"
        ;;
    linux-cuda)
        echo "Installing PyTorch with CUDA 12.8 support (NVIDIA GPUs)..."
        uv pip install "$CUDA_TORCH" "$CUDA_TORCHVISION" --index-url "$CUDA_INDEX"
        ;;
    linux-cpu)
        echo "Installing PyTorch (CPU only)..."
        uv pip install torch torchvision --index-url "$CPU_INDEX"
        ;;
    *)
        echo "Installing PyTorch (standard — includes MPS on macOS)..."
        uv pip install torch torchvision
        ;;
esac

# Install all other dependencies. sam-2 only asks for torch>=2.5.1, which the
# build above already satisfies, so this will not replace the GPU wheel.
echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Report what actually landed in the venv, including the GPU flavor. torch.version.cuda
# is set for both CUDA and ROCm builds; torch.version.hip disambiguates them.
.venv/bin/python - <<'EOF' || echo "WARNING: torch did not import cleanly"
import torch
flavor = "CPU-only"
if getattr(torch.version, "hip", None):
    flavor = f"ROCm {torch.version.hip}"
elif torch.version.cuda:
    flavor = f"CUDA {torch.version.cuda}"
print(f"torch {torch.__version__} ({flavor})")
EOF

echo "Setup complete (platform: $PLATFORM)"
