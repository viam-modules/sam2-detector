"""Inspect what camera sam2-segmenter-left is actually using."""
from __future__ import annotations
import asyncio, os, sys
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions, Credentials
from viam.robot.client import RobotClient

env_path = os.path.join(os.path.dirname(__file__), "..", "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "vino2-main.kssbd6djf3.viam.cloud")


async def main() -> int:
    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")

    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id),
    )
    try:
        # List all resources to find sam2-segmenter-left and its config.
        for name in robot.resource_names:
            if "sam2-segmenter-left" in name.name or name.name == "sam2-segmenter-left":
                print(f"  {name.subtype}/{name.name}  type={name.type}")

        print()
        # Use frame system to figure out what sam2-segmenter-left's source frame is.
        # We query transform_pose for a point and see the chain.
        from viam.proto.common import Pose, PoseInFrame
        for src in ("sam2-segmenter-left", "left-cam", "cam-left-cup-crop"):
            try:
                origin = PoseInFrame(reference_frame=src,
                                     pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0))
                p = (await robot.transform_pose(origin, "world")).pose
                print(f"  transform_pose({src!r}, 'world'): "
                      f"t=({p.x:.1f},{p.y:.1f},{p.z:.1f})  "
                      f"OV=({p.o_x:.3f},{p.o_y:.3f},{p.o_z:.3f}) theta={p.theta:.2f}")
            except Exception as e:
                print(f"  transform_pose({src!r}, 'world') FAILED: {e}")
        return 0
    finally:
        await robot.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
