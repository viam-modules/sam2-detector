# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

# Large ROCm/torch libraries not needed for SAM2 inference.
# This reduces the bundle from ~7GB to ~2GB.
# Large ROCm/torch libraries not needed for SAM2 inference.
# Only exclude libraries that are NOT linked at torch import time.
EXCLUDE_BINARIES = [
    'librocsolver.so',    # 1.6G - linear algebra solver (LAPACK)
    'librocsparse.so',    # 1.4G - sparse matrix ops
    'librccl.so',         # 807M - multi-GPU communication (NCCL equivalent)
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
        'torch.distributed',
        'torch.testing',
        'torch.utils.tensorboard',
    ],
    noarchive=False,
    optimize=0,
)

# Filter out the large unnecessary .so files.
a.binaries = [b for b in a.binaries if b[0].split('/')[-1] not in EXCLUDE_BINARIES]

pyz = PYZ(a.pure)

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
