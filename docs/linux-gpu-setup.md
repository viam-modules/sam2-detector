# Linux GPU Machine Setup

Reference document for running SAM2 with GPU acceleration on Linux. The
[NVIDIA](#nvidia-cuda) section covers the published `linux/amd64` build; the
[AMD ROCm](#amd-rocm) section reproduces the `vino2` machine setup.

## NVIDIA (CUDA)

The published `linux/amd64` module bundles CUDA 12.8 PyTorch along with the CUDA
runtime libraries it needs, so the machine requires **only an NVIDIA driver** —
there is no need to install the CUDA toolkit.

### Requirements

| Component | Requirement |
|---|---|
| GPU | Compute capability 5.0+ (Maxwell or newer, including Blackwell) |
| Driver | 525.60.13 or newer (any driver supporting CUDA 12.x) |
| CUDA toolkit | Not required |

### Verify the driver

```bash
nvidia-smi
```

This must list the GPU and report a CUDA version of 12.0 or higher. `nvidia-smi`
comes from the driver, so if it is missing, install the driver:

```bash
sudo apt install -y nvidia-driver-535   # or newer
sudo reboot
```

### Verify the module is using the GPU

Check the module's startup logs, which state the selected device, or ask it directly:

```bash
viam machine part run --organization <org> --location <loc> --machine <machine> \
  --data '{"name":"vision-1","command":{"command":"status"}}' \
  viam.service.vision.v1.VisionService.DoCommand
```

`device` should be `cuda` and `torch_gpu_support` should read `cuda-12.8`. If
`device` is `cpu`, the log message explains which of these is true:

- `torch_gpu_support` is `none (CPU-only build)` — the installed module is a CPU
  build. Confirm the machine resolved the `linux/amd64` artifact from a version
  that includes CUDA support.
- `no device is visible` — the driver is missing or not loaded, the module's user
  cannot open `/dev/nvidia*`, or `CUDA_VISIBLE_DEVICES` is filtering the GPU out.

To confirm from the host that CUDA libraries are actually loaded by the running
module, inspect its memory maps:

```bash
PID=$(pgrep -f 'sam2-detector.*dist/main' | head -1)
sudo grep -Eo 'lib(cuda|cudart|cublas|cudnn|torch_cuda)[^/]*' /proc/"$PID"/maps | sort -u
```

A GPU-enabled module lists `libcudart`, `libcublas`, `libcudnn` and
`libtorch_cuda`; `libcuda.so.1` additionally confirms the driver is in use.
Empty output means CPU-only inference.

### Gotchas

| Issue | Solution |
|---|---|
| `device` is `cpu` and `torch_gpu_support` is `none` | The module is a CPU-only build; the `linux/amd64` artifact must be built with `SAM2_BUILD_TARGET=linux-cuda` |
| `nvidia-smi` works but no device is visible to the module | `viam-server` runs the module as a different user; ensure it can access `/dev/nvidia*` |
| Driver older than 525 | CUDA 12.8 wheels will not initialize; upgrade the driver |
| Machine has both NVIDIA and ROCm installed | `detect_target.sh` prefers ROCm; set `SAM2_BUILD_TARGET=linux-cuda` explicitly |

## AMD ROCm

Reference setup for the `vino2` machine.

### Hardware

| Component | Spec |
|---|---|
| **CPU** | AMD Ryzen Threadripper PRO 7985WX (64 cores) |
| **GPU** | AMD Radeon PRO W6400 (Navi 24, RDNA 2, `gfx1032`) |
| **GPU PCI ID** | `03:00.0 VGA compatible controller: AMD/ATI Navi 24 [Radeon PRO W6400]` |
| **Display outputs** | 2x mini-DisplayPort on the GPU card |
| **Hostname** | `vino2` |
| **User** | `viam` |

### Software

| Component | Version |
|---|---|
| **OS** | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| **Kernel** | 6.8.0-106-generic |
| **ROCm** | 6.3.4 |
| **amdgpu-dkms** | 6.10.5-2125197.22.04 |
| **PyTorch** | 2.9.1+rocm6.3 |
| **torchvision** | 0.24.1+rocm6.3 |
| **Python** | 3.11 |
| **uv** | latest (auto-installed by setup.sh) |
| **GCC** | gcc-12 (required for DKMS kernel module build) |

### ROCm Installation

#### 1. Add ROCm repo

```bash
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.4 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3.4/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/amdgpu.list
```

#### 2. Pin ROCm repo priority

Ubuntu's universe repo ships ancient ROCm 5.0 packages that conflict. Pin the AMD repo higher:

```bash
sudo tee /etc/apt/preferences.d/rocm-pin-700 << 'EOF'
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 700
EOF
```

#### 3. Install dependencies and ROCm

```bash
sudo apt update
sudo apt install -y gcc-12 linux-headers-$(uname -r)
sudo apt install -y amdgpu-dkms rocm
```

#### 4. User permissions

```bash
sudo usermod -a -G render,video $USER
```

Log out and back in for group changes to take effect.

#### 5. Secure Boot

The `amdgpu-dkms` kernel module must be signed or Secure Boot must be disabled. On Dell machines with SafeBIOS:

- Enter BIOS (F2 at boot, or `sudo systemctl reboot --firmware-setup`)
- Go to Secure Boot Mode → set to **Audit Mode**
- Save and exit

Without this, `modprobe amdgpu` fails with "Key was rejected by service".

#### 6. Reboot and verify

```bash
sudo reboot
```

After reboot:

```bash
rocm-smi                          # Should show the GPU
rocminfo | grep "Marketing Name"  # Should show "AMD Radeon PRO W6400"
sudo modprobe amdgpu              # Should succeed without errors
```

### PyTorch ROCm Setup

#### HSA_OVERRIDE_GFX_VERSION

The W6400 (`gfx1032`, RDNA 2) requires `HSA_OVERRIDE_GFX_VERSION=10.3.0` for PyTorch ROCm compatibility. The SAM2 module sets this automatically when it detects `/opt/rocm`.

For manual testing:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
"
```

#### Install PyTorch with ROCm

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

**Do not** install torch from standard PyPI — it includes NVIDIA CUDA, not ROCm.

The SAM2 module's `setup.sh` handles this automatically by detecting ROCm and using the correct index URL.

### Gotchas

| Issue | Solution |
|---|---|
| `rocminfo` shows version 5.0.0 | Ubuntu universe repo conflict — add the ROCm pin file (step 2) |
| `gcc-12: not found` during DKMS build | `sudo apt install -y gcc-12` |
| `Key was rejected by service` on modprobe | Secure Boot is blocking unsigned modules — set Audit Mode in BIOS |
| `torch.cuda.is_available()` returns False with ROCm torch | Need `HSA_OVERRIDE_GFX_VERSION=10.3.0` for the W6400 |
| `Memory access fault by GPU` | Wrong `HSA_OVERRIDE_GFX_VERSION` — use `10.3.0` for W6400, not `11.0.0` |
| `uv run` overwrites ROCm torch with PyPI torch | Remove `torch`/`torchvision` from `pyproject.toml` — install via `setup.sh` only |
| Display not working after amdgpu install | GPU has mini-DP outputs; try both ports. BIOS/grub screens use basic display. |
| No display for MOK/BIOS screen | Need physical mini-DP connection to GPU card |

### GFX Version Reference

| GPU | Architecture | gfx target | HSA_OVERRIDE_GFX_VERSION |
|---|---|---|---|
| Radeon PRO W6400 | RDNA 2 (Navi 24) | gfx1032 | 10.3.0 |
| Radeon RX 7900 XTX | RDNA 3 (Navi 31) | gfx1100 | 11.0.0 |
| Radeon RX 9070 | RDNA 4 (Navi 48) | gfx1200 | Not yet supported in ROCm 6.3 |
