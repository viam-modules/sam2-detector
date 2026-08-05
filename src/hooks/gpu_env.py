"""PyInstaller runtime hook: set GPU env vars before torch is imported.

Runs inside the frozen binary ahead of any application code, which is the only
point early enough for these variables to take effect.

There is deliberately nothing here for NVIDIA. The bundled CUDA libraries are
resolved by the loader through the RPATH entries that PyInstaller and the nvidia
wheels already provide; setting LD_LIBRARY_PATH at this point would be useless,
since glibc reads it once at process start.
"""
import os

# Required for AMD GPUs that are not on PyTorch's official ROCm support list
# (e.g. the Radeon PRO W6400, gfx1032). Harmless on machines without ROCm.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
