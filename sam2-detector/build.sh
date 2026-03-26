#!/bin/sh
set -e
cd "$(dirname "$0")"

SAM2_MODEL="${SAM2_MODEL:-facebook/sam2.1-hiera-tiny}"

# Use uv if available, otherwise fall back to a local venv.
if command -v uv >/dev/null 2>&1; then
    UV_PROJECT="$(pwd)/.."
    uv run --project "$UV_PROJECT" pyinstaller \
        --onefile \
        --hidden-import="googleapiclient" \
        --hidden-import="viam" \
        --hidden-import="sam2" \
        src/main.py

    # Download the model checkpoint.
    CKPT_NAME=$(uv run --project "$UV_PROJECT" python -c \
        "from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES; print(HF_MODEL_ID_TO_FILENAMES['${SAM2_MODEL}'][1])")
    mkdir -p checkpoints
    uv run --project "$UV_PROJECT" python -c \
        "from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download('${SAM2_MODEL}', '${CKPT_NAME}'); shutil.copy(path, 'checkpoints/${CKPT_NAME}'); print('Downloaded checkpoints/${CKPT_NAME}')"
else
    VENV_NAME="venv"
    PYTHON="$VENV_NAME/bin/python"
    $PYTHON -m pip install pyinstaller -Uqq
    $PYTHON -m PyInstaller \
        --onefile \
        --hidden-import="googleapiclient" \
        --hidden-import="viam" \
        --hidden-import="sam2" \
        src/main.py

    # Download the model checkpoint.
    CKPT_NAME=$($PYTHON -c \
        "from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES; print(HF_MODEL_ID_TO_FILENAMES['${SAM2_MODEL}'][1])")
    mkdir -p checkpoints
    $PYTHON -c \
        "from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download('${SAM2_MODEL}', '${CKPT_NAME}'); shutil.copy(path, 'checkpoints/${CKPT_NAME}'); print('Downloaded checkpoints/${CKPT_NAME}')"
fi

# Package into the tarball that meta.json expects.
tar -czvf module.tar.gz meta.json dist/main checkpoints/

echo "Built module.tar.gz"
