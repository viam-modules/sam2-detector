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
  "model_name": "facebook/sam2.1-hiera-tiny",
  "depth_threshold_mm": 200,
  "min_points": 50
}
```

### Attributes

| Name | Type | Inclusion | Description |
|---|---|---|---|
| `detector_name` | string | **Required** | Name of the upstream vision service that provides 2D detections (bounding boxes). Added as a required dependency. |
| `camera_name` | string | **Required** | Name of the RGBD camera that provides both color and depth images. Added as a required dependency. |
| `label` | string | Optional | Filter upstream detections to only this label. Default: `""` (accept all labels). |
| `confidence_threshold` | float | Optional | Minimum confidence score from the upstream detector. Default: `0.5`. |
| `model_name` | string | Optional | HuggingFace SAM2 model ID. Default: `facebook/sam2.1-hiera-tiny`. Options: `facebook/sam2.1-hiera-tiny` (149MB), `facebook/sam2.1-hiera-small` (176MB), `facebook/sam2.1-hiera-base-plus` (309MB), `facebook/sam2.1-hiera-large` (856MB). |
| `depth_threshold_mm` | int | Optional | Maximum deviation from median depth (in mm) to keep a point. Removes background bleed-through. Default: `0` (disabled). |
| `min_points` | int | Optional | Minimum number of 3D points for a valid segment. Segments with fewer points are discarded. Default: `50`. |

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
- `point_cloud`: PCD binary data (XYZRGB format)
- `geometries`: bounding box geometry with label, in the camera reference frame

### `get_detections(image)` / `get_detections_from_camera(camera_name)`

Pass-through to the upstream detector, filtered by the configured label and confidence threshold. This allows the module to also serve as a filtered 2D detector.

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
