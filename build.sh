#!/bin/sh
set -e
cd "$(dirname "$0")"

SAM2_MODEL="${SAM2_MODEL:-facebook/sam2.1-hiera-tiny}"

# Ensure dependencies are installed (creates venv, installs correct torch).
./setup.sh

# Use the venv python directly — never uv run/uv sync, which would
# re-resolve torch from PyPI and overwrite the ROCm version.
PYTHON=".venv/bin/python"

# On Jetson, libcudss.so.0 lives inside the venv (nvidia-cudss-cu12 wheel) and
# the Jetson torch wheel dlopens it. Make it discoverable for any python
# invocation in this script that imports torch.
if [ -f /etc/nv_tegra_release ]; then
    export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH}"
fi

# Verify torch is the right version before building.
echo "Bundling torch version: $($PYTHON -c 'import torch; print(torch.__version__)')"

# Build PyInstaller binary using spec file (includes runtime hooks for ROCm).
$PYTHON -m PyInstaller --clean main.spec

# Download the model checkpoint.
CKPT_NAME=$($PYTHON -c \
    "from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES; print(HF_MODEL_ID_TO_FILENAMES['${SAM2_MODEL}'][1])")
mkdir -p checkpoints
$PYTHON -c \
    "from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download('${SAM2_MODEL}', '${CKPT_NAME}'); shutil.copy(path, 'checkpoints/${CKPT_NAME}'); print('Downloaded checkpoints/${CKPT_NAME}')"

# Package into the tarball. dist/main is a directory (onedir/Linux) or file (onefile/macOS).
tar -czvf module.tar.gz meta.json run.sh dist/main checkpoints/

echo "Built module.tar.gz"
