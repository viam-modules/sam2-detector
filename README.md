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
| Linux + NVIDIA GPU | `cuda` | Standard PyTorch CUDA |
| Linux + AMD GPU | `cuda` | Via ROCm PyTorch (install `torch` from the ROCm index) |
| macOS Apple Silicon | `mps` | Metal Performance Shaders |
| Other | `cpu` | Fallback |

## Testing

```bash
cd /path/to/vino/sam2
export VIAM_API_KEY="..."
export VIAM_API_KEY_ID="..."
uv run python test_camera.py --num-frames 50 --point 600,300
```
