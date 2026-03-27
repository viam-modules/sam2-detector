#!/bin/sh
set -e
cd "$(dirname "$0")"

SAM2_MODEL="${SAM2_MODEL:-facebook/sam2.1-hiera-tiny}"

# Ensure dependencies are installed (creates venv, installs correct torch).
./setup.sh

# Use the venv python directly — never uv run/uv sync, which would
# re-resolve torch from PyPI and overwrite the ROCm version.
PYTHON=".venv/bin/python"

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
