"""PyInstaller runtime hook: set AMD ROCm env vars before torch loads."""
import os
import sys

print(f"[rocm_env hook] /opt/rocm exists: {os.path.exists('/opt/rocm')}", file=sys.stderr)
print(f"[rocm_env hook] HSA_OVERRIDE_GFX_VERSION before: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}", file=sys.stderr)

if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

print(f"[rocm_env hook] HSA_OVERRIDE_GFX_VERSION after: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}", file=sys.stderr)
