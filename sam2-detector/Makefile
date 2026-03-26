# SAM2 Detector Module - Build
#
# Available models (set SAM2_MODEL to change):
#   facebook/sam2.1-hiera-tiny       (149MB)
#   facebook/sam2.1-hiera-small      (176MB)
#   facebook/sam2.1-hiera-base-plus  (309MB)
#   facebook/sam2.1-hiera-large      (856MB)
#
# Usage:
#   make module                          # Build PyInstaller binary
#   make module.tar.gz                   # Build tarball with bundled checkpoint
#   make module.tar.gz SAM2_MODEL=facebook/sam2.1-hiera-small  # Use a different model
#   make clean                           # Remove all build artifacts

SAM2_MODEL ?= facebook/sam2.1-hiera-large
VENV = ../venv
PYTHON = $(VENV)/bin/python
UV = uv

# Derive checkpoint filename from model ID via the SAM2 lookup table.
CKPT_NAME = $(shell $(UV) run --project .. python -c \
	"from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES; print(HF_MODEL_ID_TO_FILENAMES['$(SAM2_MODEL)'][1])")

.PHONY: clean module

clean:
	rm -rf dist/ build/ checkpoints/ *.spec
	rm -f module.tar.gz

# Download the checkpoint from HuggingFace into checkpoints/.
checkpoints/$(CKPT_NAME):
	@mkdir -p checkpoints
	$(UV) run --project .. python -c "\
		from huggingface_hub import hf_hub_download; \
		import shutil; \
		path = hf_hub_download('$(SAM2_MODEL)', '$(CKPT_NAME)'); \
		shutil.copy(path, 'checkpoints/$(CKPT_NAME)'); \
		print('Downloaded checkpoints/$(CKPT_NAME)')"

# Build the PyInstaller binary.
module: dist/main

dist/main: src/main.py src/models/sam2.py
	$(UV) run --project .. pyinstaller \
		--onefile \
		--hidden-import="googleapiclient" \
		--hidden-import="viam" \
		--hidden-import="sam2" \
		src/main.py
	@echo "Built dist/main"

# Build the distributable tarball with the module binary, metadata, and model checkpoint.
module.tar.gz: dist/main checkpoints/$(CKPT_NAME)
	tar -czvf module.tar.gz \
		meta.json \
		dist/main \
		checkpoints/$(CKPT_NAME)
	@echo ""
	@echo "Built module.tar.gz with model: $(SAM2_MODEL) ($(CKPT_NAME))"
	@ls -lh module.tar.gz
