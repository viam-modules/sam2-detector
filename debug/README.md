# Debug tools

Standalone scripts used to diagnose 3D-projection issues in `sam2-segments`. They are not run as part of the normal test suite — keep them around for the next time the per-segment point clouds look wrong.

All scripts read Viam credentials the same way `test_pointclouds.py` does: `VIAM_API_KEY` and `VIAM_API_KEY_ID` from the environment, falling back to `../../viam.env`. Default robot address is `vino2-main.kssbd6djf3.viam.cloud` (override with `VIAM_ROBOT_ADDRESS`).

Run from the repo root, e.g. `uv run python debug/debug_projection.py`.

## Scripts

### `debug_projection.py`
The main diagnostic. Pulls color + depth + intrinsics + native PCD from a raw realsense camera (`VIAM_RAW_CAMERA`, default `left-cam`), reprojects depth through the camera intrinsics using sam2's exact math, and compares against the camera's native PCD. Reports per-axis offsets, sign-flip probe, and a world-frame transform check (sam2's rust-util path vs Viam's `transform_pose`). Saves `debug_native.pcd`, `debug_reprojected.pcd`, `debug_color.png`, `debug_depth.png` for visual inspection in `pcl_viewer` / CloudCompare.

### `check_alignment.py`
Tighter version of the same idea, focused on a 2D ROI in the color image: reprojects ROI pixels via color intrinsics, crops the native PCD to the same approximate 3D region, and reports the lateral mean delta. A non-zero lateral delta at a single depth is evidence of color/depth misalignment in the upstream camera (the bug fixed by `align_color_depth: true` on `viam:camera:realsense`).

### `check_frames.py`
Pulls the camera→world pose for several candidate source frames (`left-cam`, `cam-left-cup-crop`, etc.), applies each to the cropped PCD, and reports which frame interpretation produces an upright cup (PC1 aligned with world Z). Useful when the points look right at the sensor but appear rotated in the viewer — almost always a frame-config issue.

### `check_sam2_config.py`
Quick query of `transform_pose(<frame>, "world")` for the cameras a `sam2-segments` service might depend on. Confirms whether a frame is identity (= world) or has a real transform attached.

### `save_pcds.py`
Saves two PCDs side-by-side for `pcl_viewer` comparison:
- `debug_cam_left_cup_crop.pcd` from `cam-left-cup-crop.NextPointCloud()` (override via `VIAM_CROP_CAMERA`)
- `debug_sam2_segments_left.pcd` from `sam2-segmenter-left.GetObjectPointClouds()` (override via `VIAM_SAM2_SERVICE`)

## How they helped find the alignment bug

In May 2026 the per-segment PCDs from `sam2-segments` were appearing rotated/flattened in the viewer. The chain of evidence the scripts produced:

1. `debug_projection.py` showed the reprojection vs native PCD matched within ~5 mm in camera frame (sign-flip probe came out ~equal across all four sign combos), and the world-frame transform via the rust util matched Viam's `transform_pose` to 0.00 mm — ruling out projection math and transform math as the bug.
2. `check_frames.py` showed the cup's PC1 lined up with world Z when the crop PCD was interpreted in `left-cam`'s frame, but sam2's segment cup was rotated ~87° from that — ruling out a frame-config issue.
3. `check_alignment.py` then showed a ~46-pixel vertical color-pixel mismatch at 700 mm depth, which is exactly the parallax signature of unaligned IR/RGB sensors on a D-series RealSense.

Fix landed in [`viam-modules/viam-camera-realsense#142`](https://github.com/viam-modules/viam-camera-realsense/pull/142): a new `align_color_depth: true` config attr that runs `rs2::align(RS2_STREAM_COLOR)` inside `GetImages` so depth is resampled into the color frame.
