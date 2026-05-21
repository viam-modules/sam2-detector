# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import platform
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
sys.setrecursionlimit(5000)

# Build target: "darwin", "linux-cpu", "linux-rocm", or "linux-tegra"
# Auto-detected from platform, or override with SAM2_BUILD_TARGET env var.
def _detect_target():
    override = os.environ.get('SAM2_BUILD_TARGET')
    if override:
        return override
    if platform.system() == 'Darwin':
        return 'darwin'
    if platform.system() == 'Linux':
        if os.path.exists('/etc/nv_tegra_release'):
            return 'linux-tegra'
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

# Find the viam rust utils shared library — needed for gRPC connections.
import viam.rpc
_viam_rpc_dir = os.path.dirname(viam.rpc.__file__)
_rust_lib = None
for _ext in ('.so', '.dylib', '.dll'):
    _candidate = os.path.join(_viam_rpc_dir, f'libviam_rust_utils{_ext}')
    if os.path.exists(_candidate):
        _rust_lib = _candidate
        break
_extra_binaries = []
if _rust_lib:
    print(f'[main.spec] Including Viam rust utils: {_rust_lib}')
    _extra_binaries.append((_rust_lib, 'viam/rpc'))

# On Jetson, the Jetson-specific torch wheel dlopens libcudss.so.0 (from the
# nvidia-cudss-cu12 wheel) — PyInstaller's dependency walker won't find it
# unless we add it explicitly. Bundle every .so under nvidia/cu12/lib so that
# cudss, cublas, and friends all land in the same dir in the bundle.
if BUILD_TARGET == 'linux-tegra':
    import glob
    _nv_lib_dir = os.path.join(
        os.path.dirname(sys.executable), '..', 'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages', 'nvidia', 'cu12', 'lib',
    )
    _nv_lib_dir = os.path.abspath(_nv_lib_dir)
    if os.path.isdir(_nv_lib_dir):
        for _so in glob.glob(os.path.join(_nv_lib_dir, '*.so*')):
            print(f'[main.spec] Including Jetson nvidia lib: {_so}')
            _extra_binaries.append((_so, 'nvidia/cu12/lib'))
    else:
        print(f'[main.spec] WARNING: nvidia/cu12/lib not found at {_nv_lib_dir}')

# SAM2 loads Hydra YAML configs from inside its package at runtime; PyInstaller
# won't bundle them unless we explicitly collect them.
_sam2_datas = collect_data_files('sam2')
_sam2_hidden = collect_submodules('sam2')
# torch._dynamo.polyfills.loader iterates a hardcoded list and dynamically
# imports each polyfill submodule; PyInstaller's static analysis misses them.
_torch_dynamo_hidden = collect_submodules('torch._dynamo')

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=_extra_binaries,
    datas=_sam2_datas,
    hiddenimports=['googleapiclient', 'viam', *_sam2_hidden, *_torch_dynamo_hidden],
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

if BUILD_TARGET in ('linux-rocm', 'linux-tegra'):
    # ROCm / Jetson: onedir to avoid 4GB single-file limit.
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
