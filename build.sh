#!/bin/sh
set -e
cd "$(dirname "$0")"

SAM2_MODEL="${SAM2_MODEL:-facebook/sam2.1-hiera-tiny}"

# Resolve the target once and export it, so setup.sh and main.spec cannot
# disagree about which GPU runtime is being installed versus packaged.
. ./detect_target.sh
SAM2_BUILD_TARGET="$(detect_sam2_target)"
export SAM2_BUILD_TARGET
echo "Building for target: $SAM2_BUILD_TARGET"

# Ensure dependencies are installed (creates venv, installs correct torch).
./setup.sh

# Use the venv python directly — never uv run/uv sync, which would
# re-resolve torch from PyPI and overwrite the GPU-enabled version.
PYTHON=".venv/bin/python"

# Fail early if the installed torch does not match the target, rather than
# shipping a CPU-only binary to a GPU platform.
$PYTHON - "$SAM2_BUILD_TARGET" <<'EOF'
import sys
import torch

target = sys.argv[1]
if getattr(torch.version, "hip", None):
    flavor = "linux-rocm"
elif torch.version.cuda:
    flavor = "linux-cuda"
else:
    flavor = "cpu-or-mps"
print(f"Bundling torch {torch.__version__} (GPU support: {flavor})")

if target in ("linux-cuda", "linux-rocm") and flavor != target:
    raise SystemExit(
        f"ERROR: target {target} needs a matching torch build, but the venv has "
        f"{flavor} ({torch.__version__}). Delete .venv and re-run, or check the "
        f"index URL in setup.sh."
    )
EOF

# Build PyInstaller binary using spec file (includes GPU runtime hooks).
$PYTHON -m PyInstaller --clean main.spec

# Download the model checkpoint.
CKPT_NAME=$($PYTHON -c \
    "from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES; print(HF_MODEL_ID_TO_FILENAMES['${SAM2_MODEL}'][1])")
mkdir -p checkpoints
$PYTHON -c \
    "from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download('${SAM2_MODEL}', '${CKPT_NAME}'); shutil.copy(path, 'checkpoints/${CKPT_NAME}'); print('Downloaded checkpoints/${CKPT_NAME}')"

# Package into the tarball. dist/main is a directory (onedir/GPU builds) or a
# single file (onefile/CPU and macOS builds); run.sh handles both layouts.
# Quiet tar: a GPU bundle lists thousands of files, which buries build errors.
tar -czf module.tar.gz meta.json run.sh dist/main checkpoints/

echo "Built module.tar.gz ($(du -h module.tar.gz | cut -f1) packaged, $(du -sh dist/main | cut -f1) unpacked)"
