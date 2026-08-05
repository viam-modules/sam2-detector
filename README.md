# SAM2 Detector Module

A Viam vision service module powered by Meta's [SAM2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2). Provides two models:

- **`viam:sam2-detector:sam2`** — Single-object tracking across video frames using SAM2 VideoPredictor
- **`viam:sam2-detector:sam2-segments`** — 3D point cloud generation by combining upstream 2D detections with SAM2 segmentation masks and depth projection

## Models

### `viam:sam2-detector:sam2` — Object Tracking

Tracks a single object across video frames. You provide an initial point on the object, and SAM2's video predictor segments and tracks it, returning bounding box detections.

See [sam2 model documentation](viam_sam2-detector_sam2.md) for configuration details.

### `viam:sam2-detector:sam2-segments` — 3D Segmentation

Combines an upstream object detector with SAM2's precise segmentation and depth-based 3D projection. Instead of projecting all pixels in a bounding box to 3D (which includes background), this model uses SAM2's mask to project only object pixels, producing much cleaner point clouds. Point clouds are automatically transformed to the world frame using the machine's frame system.

See [sam2-segments model documentation](viam_sam2-detector_sam2-segments.md) for configuration details.

## Device selection

Both models auto-detect the best available device at startup:

| Platform | Device | Notes |
|---|---|---|
| Linux x86_64 + NVIDIA GPU | `cuda` | Published `linux/amd64` build; needs only an NVIDIA driver >= 525 |
| Linux + AMD GPU | `cuda` | ROCm build; see [Linux GPU setup](docs/linux-gpu-setup.md) |
| macOS Apple Silicon | `mps` | Metal Performance Shaders |
| Other | `cpu` | Fallback |

Whichever device is chosen is logged at startup. If the module falls back to CPU it
logs a warning explaining why, and `do_command({"command": "status"})` reports the
selected `device` alongside `torch_version` and `torch_gpu_support`, so you can tell
a driver problem apart from a build that has no GPU support at all.

A GPU can only be used if the bundled PyTorch was built for it — a CPU-only wheel
will never use a GPU no matter what hardware is present. The published
`linux/amd64` build therefore ships CUDA-enabled PyTorch.

## Build targets

`detect_target.sh` picks a target from the build machine; override it with the
`SAM2_BUILD_TARGET` environment variable.

| Target | Selected on | PyTorch wheels | Packaging |
|---|---|---|---|
| `linux-cuda` | Linux x86_64 | CUDA 12.8 (`cu128`) | onedir |
| `linux-rocm` | Linux with `/opt/rocm` | ROCm 6.3 | onedir |
| `linux-cpu` | other Linux (e.g. arm64) | CPU-only | onefile |
| `darwin` | macOS | standard (MPS) | onefile |

The CUDA target bundles NVIDIA's CUDA runtime libraries, which makes it far larger
than the CPU build (roughly 5 GB unpacked). That is what lets the module run on a
machine that has only an NVIDIA driver and no CUDA toolkit installed. To build the
small CPU-only variant instead:

```bash
SAM2_BUILD_TARGET=linux-cpu ./build.sh
```

## Testing

```bash
cd /path/to/vino/sam2
export VIAM_API_KEY="..."
export VIAM_API_KEY_ID="..."
uv run python test_camera.py --num-frames 50 --point 600,300
```
