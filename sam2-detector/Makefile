# SAM2 Detector Module - Build
#
# Available models (set SAM2_MODEL to change):
#   facebook/sam2.1-hiera-tiny       (149MB)
#   facebook/sam2.1-hiera-small      (176MB)
#   facebook/sam2.1-hiera-base-plus  (309MB)
#   facebook/sam2.1-hiera-large      (856MB)
#
# Usage:
#   make module                          # Build binary, download checkpoint, create tarball
#   make module SAM2_MODEL=facebook/sam2.1-hiera-small  # Use a different model
#   make clean                           # Remove all build artifacts

.PHONY: clean module

clean:
	rm -rf dist/ build/ checkpoints/ *.spec
	rm -f module.tar.gz

module:
	SAM2_MODEL=$(or $(SAM2_MODEL),facebook/sam2.1-hiera-tiny) ./build.sh
