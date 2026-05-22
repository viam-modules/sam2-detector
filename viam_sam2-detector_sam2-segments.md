# Model viam:sam2-detector:sam2-segments

A Viam vision service that combines upstream 2D object detections with SAM2 segmentation masks and depth-based 3D projection to produce precise object point clouds.

## How it works

1. Gets color and depth images from an RGBD camera
2. Calls an upstream detector (e.g. a TFLite object detector) to get 2D bounding boxes
3. Filters detections by label and confidence threshold
4. For each detection, uses the bounding box as a SAM2 box prompt to get a precise segmentation mask
5. Projects only the masked pixels to 3D using camera intrinsics and the depth map
6. Skips zero-depth pixels and optionally filters background by depth threshold
7. Returns each segmented object as a `PointCloudObject`

The key advantage: unlike projecting all pixels in a bounding box (which includes background), the SAM2 mask selects only object pixels, producing much cleaner point clouds.

## Configuration

```json
{
  "detector_name": "my-object-detector",
  "camera_name": "my-rgbd-camera",
  "label": "glass",
  "confidence_threshold": 0.5,
  "depth_threshold_mm": 200,
  "min_points": 50,
  "highlighting_on": true,
  "highlight_color": { "r": 0, "g": 255, "b": 0 }
}
```

### Attributes

| Name | Type | Inclusion | Description |
|---|---|---|---|
| `detector_name` | string | **Required** | Name of the upstream vision service that provides 2D detections (bounding boxes). Added as a required dependency. |
| `camera_name` | string | **Required** | Name of the RGBD camera that provides both color and depth images. Added as a required dependency. |
| `label` | string | Optional | Filter upstream detections to only this label. Default: `""` (accept all labels). |
| `confidence_threshold` | float | Optional | Minimum confidence score from the upstream detector. Default: `0.5`. |
| `depth_threshold_mm` | int | Optional | Maximum deviation from median depth (in mm) to keep a point. Removes background bleed-through. Default: `0` (disabled). |
| `min_points` | int | Optional | Minimum number of 3D points for a valid segment. Segments with fewer points are discarded. Default: `50`. |
| `highlighting_on` | bool | Optional | When true, images returned by `capture_all_from_camera` have SAM2 segmentation masks overlaid at 50% opacity. Default: `false`. |
| `highlight_color` | object | Optional | RGB color used for the mask overlay when `highlighting_on` is true. Object with `r`, `g`, `b` integer fields in the range 0–255. Default: `{ "r": 0, "g": 255, "b": 0 }` (green). |

### Example Configuration

```json
{
  "detector_name": "glass-detector",
  "camera_name": "realsense",
  "label": "glass",
  "confidence_threshold": 0.3,
  "depth_threshold_mm": 150,
  "min_points": 100
}
```

## Supported API methods

### `get_object_point_clouds(camera_name)`

Returns a list of `PointCloudObject` — one for each detected and segmented object. Each contains:
- `point_cloud`: PCD binary data (XYZRGB format, coordinates in meters)
- `geometries`: 3D bounding box geometry with label (dimensions in mm)

Point clouds are automatically transformed to the **world frame** using the machine's frame system. The module connects to the parent machine using the `VIAM_MACHINE_FQDN`, `VIAM_API_KEY_ID`, and `VIAM_API_KEY` environment variables, which are set automatically by `viam-server`. If frame transform is unavailable (e.g. no frame system configured), points are returned in the camera's reference frame.

### `get_detections(image)` / `get_detections_from_camera(camera_name)`

Returns filtered detections with **SAM2-refined bounding boxes**. The upstream detector provides initial bounding boxes, SAM2 generates a precise segmentation mask, and the returned bounding box tightly surrounds the mask — not the original detector bbox.

### `capture_all_from_camera(camera_name)`

Supports `return_image`, `return_detections`, and `return_object_point_clouds`.

### `get_properties()`

Returns `detections_supported=True, object_point_clouds_supported=True`.

## DoCommand

### `status`

```json
{
  "command": "status"
}
```

Returns current configuration: detector name, camera name, label, confidence threshold, model name, device, depth threshold, and min points.

## Camera requirements

The camera must return both **color** and **depth** images from `get_images()`. The images are matched by source name:
- Color: source name containing "color" or "rgb"
- Depth: source name containing "depth"

The camera must also provide **intrinsic parameters** via `get_properties()` (focal length, principal point).

Intel RealSense cameras work well for this purpose.

## Frame transforms

Point clouds from `get_object_point_clouds` are automatically transformed from the camera frame to the **world frame** using the machine's frame system configuration.

The module connects to the parent machine using environment variables set by `viam-server`:
- `VIAM_MACHINE_FQDN` — machine address
- `VIAM_API_KEY_ID` — API key ID
- `VIAM_API_KEY` — API key

These are set automatically — no additional configuration is needed. The module uses `transform_pose` to obtain the camera-to-world transform (rotation + translation) and applies it to all 3D points.

If the frame system is not configured or the connection fails, the module logs a warning and returns point clouds in the camera's reference frame instead.
