"""
Save two PCDs side by side for comparison in pcl_viewer:
  - debug_cam_left_cup_crop.pcd  -> cam-left-cup-crop.get_point_cloud()
  - debug_sam2_segments_left.pcd -> sam2-segmenter-left.get_object_point_clouds()[i]
                                    (one file per returned segment if there are multiple)
"""

from __future__ import annotations

import asyncio
import os
import sys

from viam.components.camera import Camera
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

# Same env-loading pattern as the other test scripts.
env_path = os.path.join(os.path.dirname(__file__), "..", "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "vino2-main.kssbd6djf3.viam.cloud")
CROP_CAMERA = os.environ.get("VIAM_CROP_CAMERA", "cam-left-cup-crop")
SAM2_SERVICE = os.environ.get("VIAM_SAM2_SERVICE", "sam2-segmenter-left")


async def main() -> int:
    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    if not api_key or not api_key_id:
        print("Set VIAM_API_KEY and VIAM_API_KEY_ID (or put them in ../viam.env)")
        return 2

    print(f"Connecting to {ROBOT_ADDRESS}")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id),
    )
    try:
        # 1) cam-left-cup-crop -> single PCD
        cam = Camera.from_robot(robot, CROP_CAMERA)
        pcd_bytes, mime = await cam.get_point_cloud()
        out_a = "debug_cam_left_cup_crop.pcd"
        with open(out_a, "wb") as f:
            f.write(pcd_bytes)
        print(f"  wrote {out_a}: {len(pcd_bytes)} bytes  mime={mime!r}")

        # 2) sam2-segmenter-left -> one or more object PCDs
        sam2 = VisionClient.from_robot(robot, SAM2_SERVICE)
        objs = await sam2.get_object_point_clouds(camera_name="")
        if not objs:
            print(f"  {SAM2_SERVICE}: no object point clouds returned")
        for i, obj in enumerate(objs):
            label = ""
            if obj.geometries and obj.geometries.geometries:
                label = obj.geometries.geometries[0].label or ""
            suffix = f"_{i}" if len(objs) > 1 else ""
            out_b = f"debug_sam2_segments_left{suffix}.pcd"
            with open(out_b, "wb") as f:
                f.write(obj.point_cloud)
            print(f"  wrote {out_b}: {len(obj.point_cloud)} bytes  label={label!r}")
        return 0
    finally:
        await robot.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
