#!/bin/sh
# Shared build-target detection, sourced by setup.sh and build.sh so that the
# PyTorch wheel installed and the PyInstaller packaging mode always agree.
#
# Targets:
#   darwin      - macOS arm64, standard wheels (MPS)
#   linux-cuda  - x86_64 Linux, CUDA 12.8 wheels for NVIDIA GPUs
#   linux-rocm  - Linux with ROCm installed, AMD GPUs
#   linux-cpu   - everything else
#
# Detection is based on the *build* platform, not on GPU presence, because Viam
# cloud builders have no GPU. Env vars set in a GitHub Actions step do not reach
# a cloud build either, so the decision has to be reproducible from the machine
# the build runs on. Override locally with SAM2_BUILD_TARGET.
detect_sam2_target() {
    if [ -n "$SAM2_BUILD_TARGET" ]; then
        echo "$SAM2_BUILD_TARGET"
        return
    fi
    case "$(uname -s)" in
        Darwin)
            echo "darwin"
            ;;
        Linux)
            # ROCm is only present when deliberately installed, so it wins.
            if command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
                echo "linux-rocm"
            elif [ "$(uname -m)" = "x86_64" ]; then
                # Published linux/amd64 artifact. CUDA wheels ship their own CUDA
                # runtime and fall back to CPU when no NVIDIA driver is present,
                # so one artifact serves both GPU and CPU machines.
                echo "linux-cuda"
            else
                echo "linux-cpu"
            fi
            ;;
        *)
            echo "linux-cpu"
            ;;
    esac
}
