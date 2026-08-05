# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import platform
sys.setrecursionlimit(5000)

# Build target: "darwin", "linux-cpu", "linux-cuda", or "linux-rocm".
# build.sh exports SAM2_BUILD_TARGET, so this normally just reads the env var.
# The fallback mirrors detect_target.sh for direct `pyinstaller main.spec` runs.
def _detect_target():
    override = os.environ.get('SAM2_BUILD_TARGET')
    if override:
        return override
    if platform.system() == 'Darwin':
        return 'darwin'
    if platform.system() == 'Linux':
        if os.path.exists('/opt/rocm'):
            return 'linux-rocm'
        if platform.machine() == 'x86_64':
            return 'linux-cuda'
        return 'linux-cpu'
    return 'linux-cpu'

BUILD_TARGET = _detect_target()
GPU_TARGETS = ('linux-cuda', 'linux-rocm')
print(f'[main.spec] Build target: {BUILD_TARGET}')

# Large ROCm libraries not needed for SAM2 inference. On a ROCm host these also
# exist under /opt/rocm, so anything that does need them can still resolve them.
ROCM_EXCLUDE_BINARIES = [
    'librocsolver.so',    # 1.6G - linear algebra solver
    'librocsparse.so',    # 1.4G - sparse matrix ops
    'librccl.so',         # 807M - multi-GPU communication
]

# Large NVIDIA libraries not needed for SAM2 inference. Unlike the ROCm case there
# is no system CUDA to fall back on (target machines have only a driver), so this
# list is deliberately limited to libraries that PyTorch loads lazily rather than
# links against. Everything libtorch_cuda.so lists as a hard DT_NEEDED dependency
# (cublas, cudnn, cufft, curand, cusparse, cusparseLt, nccl, nvshmem, cudart,
# cufile) must stay, or `import torch` fails outright.
# If a GPU run ever fails with a missing-symbol or dlopen error, drop an entry here.
CUDA_EXCLUDE_BINARIES = [
    'libtorch_cuda_linalg.so',  # 128M - dlopened only for torch.linalg on GPU
    'libcusolver.so',           # 243M - used only by libtorch_cuda_linalg
    'libcusolverMg.so',         # 162M - multi-GPU dense solver
    'libnvrtc.alt.so',          # 105M - duplicate JIT compiler (libnvrtc.so is kept)
    'libnvperf_host.so',        #  26M - profiler backend
    'libnvperf_target.so',      #        profiler backend
]


def _basename(dest):
    """PyInstaller stores TOC destinations with forward slashes on all platforms."""
    return dest.replace('\\', '/').split('/')[-1]


def _filter_binaries(binaries, prefixes):
    """Drop bundled shared libraries whose names start with any given prefix.

    Prefix matching (not equality) is required because these libraries carry
    ABI version suffixes, e.g. librocsolver.so.0 or libcusolver.so.11.
    """
    kept, dropped = [], []
    for entry in binaries:
        name = _basename(entry[0])
        if any(name.startswith(p) for p in prefixes):
            dropped.append(name)
        else:
            kept.append(entry)
    for name in sorted(set(dropped)):
        print(f'[main.spec] Excluding bundled library: {name}')
    return kept

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

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=_extra_binaries,
    datas=[],
    hiddenimports=['googleapiclient', 'viam', 'sam2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['src/hooks/gpu_env.py'],
    excludes=[
        'torch.utils.tensorboard',
        # Triton is only used by torch.compile, which SAM2 does not enable, and
        # costs ~570M unpacked. torch degrades gracefully when it is absent.
        'triton',
    ],
    noarchive=False,
    optimize=0,
)

# Filter out large unnecessary .so files for GPU builds.
if BUILD_TARGET == 'linux-rocm':
    a.binaries = _filter_binaries(a.binaries, ROCM_EXCLUDE_BINARIES)
elif BUILD_TARGET == 'linux-cuda':
    a.binaries = _filter_binaries(a.binaries, CUDA_EXCLUDE_BINARIES)

pyz = PYZ(a.pure)

# UPX is skipped for GPU builds: compressing multi-hundred-megabyte CUDA/ROCm
# libraries costs a great deal of build time and has a history of corrupting
# shared objects that get loaded with RTLD_GLOBAL, as libtorch's are.
USE_UPX = BUILD_TARGET not in GPU_TARGETS

if BUILD_TARGET in GPU_TARGETS:
    # GPU builds: onedir. The bundled CUDA/ROCm runtimes push the payload past
    # PyInstaller's 4GB onefile ceiling, and onedir also avoids re-extracting
    # gigabytes of libraries to a temp dir on every module start.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='main',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=USE_UPX,
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
        upx=USE_UPX,
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
