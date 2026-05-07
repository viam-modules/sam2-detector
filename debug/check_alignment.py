"""
Quick alignment-check: pick a small bbox in the color image (where the cup
clearly is), reproject ONLY those depth pixels through sam2's color
intrinsics (the buggy path), and compare the resulting 3D bbox to the
native PCD's points cropped to roughly the same region. If color and depth
were aligned, the two should overlap; if they're parallax-shifted, the
sam2 path's bbox will be offset (and lose points where depth is missing
because the cup isn't at those pixels in the depth frame).
"""
from __future__ import annotations
import asyncio, os, sys
import numpy as np
from PIL import Image as PILImage
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient

sys.path.insert(0, os.path.dirname(__file__))
from debug_projection import _depth_image_to_numpy, _viam_image_to_numpy, parse_pcd_to_mm, reproject_full_depth

env_path = os.path.join(os.path.dirname(__file__), "..", "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "vino2-main.kssbd6djf3.viam.cloud")
RAW = os.environ.get("VIAM_RAW_CAMERA", "left-cam")


async def main() -> int:
    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id),
    )
    try:
        cam = Camera.from_robot(robot, RAW)
        images = (await cam.get_images())[0]
        props = await cam.get_properties()
        pcd_bytes, _ = await cam.get_point_cloud()

        color_img = depth_img = None
        for img in images:
            n = (getattr(img, "name", "") or "").lower()
            if "color" in n or "rgb" in n: color_img = img
            elif "depth" in n: depth_img = img
        if color_img is None: color_img = images[0]
        if depth_img is None: depth_img = images[1]

        color_np = _viam_image_to_numpy(color_img)
        depth_np = _depth_image_to_numpy(depth_img)
        intr = props.intrinsic_parameters
        fx, fy, ppx, ppy = intr.focal_x_px, intr.focal_y_px, intr.center_x_px, intr.center_y_px

        native = parse_pcd_to_mm(pcd_bytes)

        # Roughly find the cup in color: the cup-shaped cluster should be the
        # densest blue+white region. Easier: just take the central column of
        # the image which is where the cup likely lives, and a vertical strip
        # of pixels with valid depth between 500-900 mm (cup is at ~700mm).
        h, w = depth_np.shape
        mask = (depth_np > 500) & (depth_np < 900)
        # restrict to the middle 30% of the image horizontally — cuts out
        # most of the table
        cx0, cx1 = int(w * 0.35), int(w * 0.65)
        cy0, cy1 = int(h * 0.30), int(h * 0.95)
        roi = np.zeros_like(mask, dtype=bool)
        roi[cy0:cy1, cx0:cx1] = True
        mask &= roi
        print(f"  ROI mask hits {mask.sum()} pixels (color frame [{cx0}-{cx1}, {cy0}-{cy1}])")

        # Reproject just the ROI pixels through color intrinsics — sam2's path.
        d_roi = depth_np * mask  # zero outside ROI
        sam2_path = reproject_full_depth(d_roi, fx, fy, ppx, ppy)
        sam2_path = sam2_path[(sam2_path[:, 2] > 0) & (sam2_path[:, 2] < 20000)]

        # Native PCD restricted to the same approximate physical region. We
        # don't have texture coords in the saved PCD, so use a 3D bbox instead:
        # depth is 500-900mm, and at 700mm the ROI in image space corresponds
        # to roughly +/-150mm in X/Y (rough but enough).
        z_min, z_max = 500, 900
        # Rough X bounds at z=700: x_world = (u - ppx)*z/fx; for u in [cx0, cx1]:
        x_lo = (cx0 - ppx) * 700.0 / fx
        x_hi = (cx1 - ppx) * 700.0 / fx
        y_lo = (cy0 - ppy) * 700.0 / fy
        y_hi = (cy1 - ppy) * 700.0 / fy
        nat_roi_mask = (
            (native[:, 0] >= x_lo) & (native[:, 0] <= x_hi)
            & (native[:, 1] >= y_lo) & (native[:, 1] <= y_hi)
            & (native[:, 2] >= z_min) & (native[:, 2] <= z_max)
        )
        nat_roi = native[nat_roi_mask]

        def stats(name, p):
            if len(p) == 0:
                print(f"  {name}: empty")
                return
            mn, mx, mean = p.min(0), p.max(0), p.mean(0)
            print(f"  {name}: n={len(p)}  bbox=({mn[0]:.0f},{mn[1]:.0f},{mn[2]:.0f}) "
                  f"-> ({mx[0]:.0f},{mx[1]:.0f},{mx[2]:.0f})  mean=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})")

        print()
        print("Both should overlap closely if depth+color are aligned:")
        stats("sam2-path (reproj of color-ROI through color intrinsics)", sam2_path)
        stats("native    (cropped to same approx 3D bbox)", nat_roi)

        if len(sam2_path) > 0 and len(nat_roi) > 0:
            dm = sam2_path.mean(0) - nat_roi.mean(0)
            print(f"  mean delta (sam2 - native) = ({dm[0]:+.1f}, {dm[1]:+.1f}, {dm[2]:+.1f}) mm")
            print()
            if abs(dm[0]) > 10 or abs(dm[1]) > 10:
                print(f"  ⚠️  Lateral mean delta {dm[0]:+.1f}, {dm[1]:+.1f} mm at ~700mm depth is "
                      f"~{abs(dm[0])*fx/700:.0f}px / {abs(dm[1])*fy/700:.0f}px of color-pixel shift. "
                      "Consistent with depth and color being unaligned (parallax / FOV "
                      "mismatch between IR and RGB sensors).")
            else:
                print(f"  Lateral delta is small; alignment may be OK or the ROI was too coarse.")
        return 0
    finally:
        await robot.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
