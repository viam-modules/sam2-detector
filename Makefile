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
#
# The build target (GPU runtime and packaging mode) is detected by
# detect_target.sh. Override it with SAM2_BUILD_TARGET, e.g.:
#   make module SAM2_BUILD_TARGET=linux-cpu   # skip the ~5GB CUDA bundle

.PHONY: clean module

clean:
	rm -rf dist/ build/
	rm -f module.tar.gz

module:
	SAM2_MODEL=$(or $(SAM2_MODEL),facebook/sam2.1-hiera-tiny) ./build.sh
