import os

# Must be set before torch is imported anywhere — required for AMD ROCm GPUs.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import asyncio

from viam.module.module import Module

# The model imports register each model with Viam's resource registry.
from models.sam2 import Sam2 as Sam2Model  # noqa: F401
from models.sam2_segments import Sam2Segments as Sam2SegmentsModel  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
