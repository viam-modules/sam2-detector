# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import platform
sys.setrecursionlimit(5000)

# Build target: "darwin", "linux-cpu", or "linux-rocm"
# Auto-detected from platform, or override with SAM2_BUILD_TARGET env var.
def _detect_target():
    override = os.environ.get('SAM2_BUILD_TARGET')
    if override:
        return override
    if platform.system() == 'Darwin':
        return 'darwin'
    if platform.system() == 'Linux':
        if os.path.exists('/opt/rocm'):
            return 'linux-rocm'
        return 'linux-cpu'
    return 'linux-cpu'

BUILD_TARGET = _detect_target()
print(f'[main.spec] Build target: {BUILD_TARGET}')

# Large ROCm libraries not needed for SAM2 inference.
ROCM_EXCLUDE_BINARIES = [
    'librocsolver.so',    # 1.6G - linear algebra solver
    'librocsparse.so',    # 1.4G - sparse matrix ops
    'librccl.so',         # 807M - multi-GPU communication
]

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['googleapiclient', 'viam', 'sam2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['src/hooks/rocm_env.py'],
    excludes=[
        'torch.utils.tensorboard',
    ],
    noarchive=False,
    optimize=0,
)

# Filter out large unnecessary .so files for ROCm builds.
if BUILD_TARGET == 'linux-rocm':
    a.binaries = [b for b in a.binaries if b[0].split('/')[-1] not in ROCM_EXCLUDE_BINARIES]

pyz = PYZ(a.pure)

if BUILD_TARGET == 'linux-rocm':
    # ROCm: onedir to avoid 4GB single-file limit.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='main',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='main',
    )
else:
    # macOS and Linux CPU: onefile (small enough).
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='main',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
