# -*- mode: python ; coding: utf-8 -*-
import sys
import platform
sys.setrecursionlimit(5000)

IS_LINUX = platform.system() == 'Linux'

# Large ROCm/torch libraries not needed for SAM2 inference.
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
        'torch.utils.tensorboard',
    ],
    noarchive=False,
    optimize=0,
)

# Filter out large unnecessary .so files on Linux.
if IS_LINUX:
    a.binaries = [b for b in a.binaries if b[0].split('/')[-1] not in EXCLUDE_BINARIES]

pyz = PYZ(a.pure)

if IS_LINUX:
    # Linux/ROCm: use onedir to avoid the 4GB single-file limit.
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
    # macOS: use onefile (small enough).
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
