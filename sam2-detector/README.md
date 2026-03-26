# SAM2 Detector Module

A Viam vision service module that uses Meta's [SAM2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2) to track a single object across video frames. You provide an initial point on the object in the first frame, and the module uses SAM2's video predictor to segment and track it, returning bounding box detections.

## How it works

1. Frames arrive one at a time via `get_detections`.
2. Each frame is saved to a temporary sliding window on disk (numbered JPEGs).
3. Every `propagation_interval` frames, SAM2's VideoPredictor runs on the full window:
   - The initial point prompt identifies the object on frame 0.
   - SAM2 propagates the mask through all frames using temporal memory.
   - Masks are converted to bounding boxes and cached.
4. Cached detections are returned immediately on subsequent calls.
5. Old frames beyond `max_frames` are evicted to bound disk usage.

## Configuration

```json
{
  "initial_point_x": 600,
  "initial_point_y": 300,
  "model_name": "facebook/sam2.1-hiera-tiny",
  "label": "glass",
  "propagation_interval": 1,
  "max_frames": 300
}
```

| Attribute | Type | Required | Default | Description |
|---|---|---|---|---|
| `initial_point_x` | int | yes | - | X pixel coordinate of the object to track in the first frame |
| `initial_point_y` | int | yes | - | Y pixel coordinate of the object to track in the first frame |
| `model_name` | string | no | `facebook/sam2.1-hiera-tiny` | HuggingFace model ID. Options: `facebook/sam2.1-hiera-small`, `facebook/sam2.1-hiera-tiny` |
| `label` | string | no | `object` | Class name returned in detections |
| `propagation_interval` | int | no | `1` | Re-run SAM2 propagation every N frames. Higher values reduce compute but increase detection latency. |
| `max_frames` | int | no | `300` | Maximum frames to keep in the sliding window. Older frames are discarded to bound disk usage. |

## Supported API methods

### `get_detections(image)`

Pass a camera frame. Returns a list with a single `Detection` containing the bounding box of the tracked object, or an empty list if the object is not found.

### `get_properties()`

Returns `detections_supported=True`.

### `do_command(command)`

| Command | Parameters | Description |
|---|---|---|
| `set_point` | `x`, `y` | Set a new initial point. Clears cached detections. Call `reprocess` to re-run propagation. |
| `reprocess` | - | Re-run SAM2 propagation on all frames in the current window. |
| `reset` | - | Clear all frames and detections. Start fresh. |
| `status` | - | Return current state: frame count, window size, device, model info. |

Example:
```python
await vision_service.do_command({"command": "set_point", "x": 400, "y": 250})
await vision_service.do_command({"command": "reprocess"})
```

## Device selection

The module auto-detects the best available device at startup:

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

This pulls frames from a Viam camera, runs SAM2 tracking, and saves annotated output to `test_output/`.
