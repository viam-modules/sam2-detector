# Model viam:sam2-detector:sam2

A Viam vision service that uses Meta's [SAM2](https://github.com/facebookresearch/sam2) VideoPredictor to track a single object across video frames. You provide an initial point on the object, and the module segments and tracks it, returning bounding box detections.

Frames arrive one at a time via `get_detections`. Each frame is buffered and periodically processed by SAM2's video propagation, which uses temporal memory to maintain consistent tracking across frames.

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

### Attributes

| Name | Type | Inclusion | Description |
|---|---|---|---|
| `camera_name` | string | **Required** | Camera to use for `get_detections_from_camera`. Added as a required dependency. |
| `initial_point_x` | int | **Required** | X pixel coordinate of the object to track in the first frame |
| `initial_point_y` | int | **Required** | Y pixel coordinate of the object to track in the first frame |
| `model_name` | string | Optional | HuggingFace model ID. Default: `facebook/sam2.1-hiera-tiny`. Options: `facebook/sam2.1-hiera-tiny` (149MB), `facebook/sam2.1-hiera-small` (176MB), `facebook/sam2.1-hiera-base-plus` (309MB), `facebook/sam2.1-hiera-large` (856MB) |
| `label` | string | Optional | Class name returned in detections. Default: `object` |
| `propagation_interval` | int | Optional | Re-run SAM2 propagation every N frames. Higher values reduce compute but increase detection latency. Default: `1` |
| `max_frames` | int | Optional | Maximum frames to keep in the sliding window. Older frames are discarded to bound disk usage. Default: `300` |

### Example Configuration

```json
{
  "camera_name": "my-camera",
  "initial_point_x": 600,
  "initial_point_y": 300,
  "label": "glass"
}
```

## DoCommand

### `set_point` — Change the tracked object

```json
{
  "command": "set_point",
  "x": 400,
  "y": 250
}
```

Sets a new initial point and clears cached detections. Call `reprocess` afterward to re-run propagation with the new point.

### `reprocess` — Re-run propagation

```json
{
  "command": "reprocess"
}
```

Re-runs SAM2 video propagation on all frames in the current window. Returns `num_frames` and `detections` count.

### `reset` — Clear all state

```json
{
  "command": "reset"
}
```

Clears all buffered frames and cached detections. The module starts fresh on the next `get_detections` call.

### `status` — Get current state

```json
{
  "command": "status"
}
```

Returns: `total_frames_received`, `window_size`, `max_frames`, `detections_cached`, `device`, `model_name`, `propagation_interval`.
