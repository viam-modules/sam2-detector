#!/bin/sh
set -e
cd "$(dirname "$0")"

SAM2_MODEL="${SAM2_MODEL:-facebook/sam2.1-hiera-tiny}"

# Ensure dependencies are installed (also installs uv if needed).
./setup.sh

# Use the venv python directly — NOT uv run, which re-syncs and can
# overwrite ROCm torch with the standard PyPI version.
PYTHON=".venv/bin/python"

# Build PyInstaller binary using spec file (includes runtime hooks).
$PYTHON -m PyInstaller --clean main.spec

# Download the model checkpoint.
CKPT_NAME=$($PYTHON -c \
    "from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES; print(HF_MODEL_ID_TO_FILENAMES['${SAM2_MODEL}'][1])")
mkdir -p checkpoints
$PYTHON -c \
    "from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download('${SAM2_MODEL}', '${CKPT_NAME}'); shutil.copy(path, 'checkpoints/${CKPT_NAME}'); print('Downloaded checkpoints/${CKPT_NAME}')"

# Package into the tarball that meta.json expects.
tar -czvf module.tar.gz meta.json run.sh dist/main checkpoints/

echo "Built module.tar.gz"
