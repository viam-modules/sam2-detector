"""
Figure out what frame each PCD is actually in.

Hypothesis: cam-left-cup-crop returns points in CAMERA frame (Viam's RDK
default for cameras), while sam2-segmenter-left already transforms to WORLD.
Confirm by:
  1. Asking the robot for transform_pose(cam-left-cup-crop, world) and
     transform_pose(left-cam, world).
  2. Applying each to the cam-left-cup-crop PCD and seeing which result has
     PC1 aligned with world +Z (upright cup).
"""

from __future__ import annotations
import asyncio, os, sys
import numpy as np
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient

sys.path.insert(0, os.path.dirname(__file__))
from debug_projection import parse_pcd_to_mm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spatialmath import transform_points_with_pose

env_path = os.path.join(os.path.dirname(__file__), "..", "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "vino2-main.kssbd6djf3.viam.cloud")


def pc1(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(0)
    cov = (centered.T @ centered) / len(pts)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, np.argmax(vals)]


async def main() -> int:
    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id),
    )
    try:
        # Load the two PCDs we already saved.
        with open("debug_cam_left_cup_crop.pcd", "rb") as f:
            crop_pts = parse_pcd_to_mm(f.read())
        with open("debug_sam2_segments_left.pcd", "rb") as f:
            sam2_pts = parse_pcd_to_mm(f.read())

        print(f"crop  PC1 (raw)  = {pc1(crop_pts)}")
        print(f"sam2  PC1 (raw)  = {pc1(sam2_pts)}")
        print()

        # Query the camera->world transform for several plausible source frames.
        # The viewer renders cam-left-cup-crop's points in world frame using
        # whichever frame is declared in the robot config; we test both
        # candidates so we can tell which one matches the upright cup.
        for src_frame in ("cam-left-cup-crop", "left-cam"):
            try:
                origin = PoseInFrame(
                    reference_frame=src_frame,
                    pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0),
                )
                p = (await robot.transform_pose(origin, "world")).pose
                print(f"--- transform_pose({src_frame!r}, 'world') ---")
                print(f"  t=({p.x:.1f}, {p.y:.1f}, {p.z:.1f})  "
                      f"OV=({p.o_x:.4f}, {p.o_y:.4f}, {p.o_z:.4f})  theta={p.theta:.4f}")

                # Apply to the cam-left-cup-crop PCD (assumed camera frame).
                world_crop = transform_points_with_pose(
                    p.o_x, p.o_y, p.o_z, p.theta,
                    p.x, p.y, p.z, crop_pts,
                )
                v = pc1(world_crop)
                upness = abs(v[2])  # |Z| component of PC1; close to 1 = upright
                ext = world_crop.max(0) - world_crop.min(0)
                print(f"  crop -> world: PC1=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})  "
                      f"|Z|={upness:.3f}  extent=({ext[0]:.0f},{ext[1]:.0f},{ext[2]:.0f})")
                print()
            except Exception as e:
                print(f"  {src_frame}: transform failed: {e}\n")

        # And just transform_pose for the world axes themselves to confirm
        # which axis is "up" in the displayed scene.
        return 0
    finally:
        await robot.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
