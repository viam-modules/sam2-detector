"""
Test script: connects to a Viam robot, pulls frames from a camera,
runs SAM2 VideoPredictor tracking, and saves annotated output.

Usage:
    export VIAM_API_KEY="your-key"
    export VIAM_API_KEY_ID="your-key-id"
    uv run python test_camera.py
"""

import asyncio
import io
import os
import tempfile
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.sam2_video_predictor import SAM2VideoPredictor
from viam.robot.client import RobotClient
from viam.components.camera import Camera

ROBOT_ADDRESS = "vino2-main.kssbd6djf3.viam.cloud"
CAMERA_NAME = "camera-image-dir"
INITIAL_POINT = (600, 300)
MODEL_NAME = "facebook/sam2.1-hiera-large"
NUM_FRAMES = 50  # How many frames to pull for the test


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


async def main():
    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    if not api_key or not api_key_id:
        print("Set VIAM_API_KEY and VIAM_API_KEY_ID environment variables")
        return

    # Connect to robot
    print(f"Connecting to {ROBOT_ADDRESS}...")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options(
            dial_options=RobotClient.Options.with_api_key(
                api_key=api_key, api_key_id=api_key_id,
            ),
        ),
    )
    print(f"Connected! Resources: {[r.name for r in robot.resource_names]}")

    camera = Camera.from_robot(robot, CAMERA_NAME)

    # Pull frames from camera and save to temp dir
    frame_dir = tempfile.mkdtemp(prefix="sam2_test_")
    print(f"Pulling {NUM_FRAMES} frames from '{CAMERA_NAME}'...")
    frames = []
    for i in range(NUM_FRAMES):
        viam_img = await camera.get_image()
        pil = Image.open(io.BytesIO(viam_img.data)).convert("RGB")
        arr = np.array(pil)
        frames.append(arr)
        pil.save(os.path.join(frame_dir, f"{i}.jpeg"), quality=90)
        if i == 0:
            print(f"  Frame size: {arr.shape}")
        if (i + 1) % 10 == 0:
            print(f"  Pulled {i + 1}/{NUM_FRAMES} frames")

    await robot.close()
    print(f"Frames saved to {frame_dir}")

    # Run SAM2 VideoPredictor
    device = select_device()
    print(f"\nLoading SAM2 ({MODEL_NAME}) on {device}...")
    predictor = SAM2VideoPredictor.from_pretrained(MODEL_NAME, device=device)
    print("Model loaded!")

    print("Initializing video state...")
    t0 = time.time()
    state = predictor.init_state(video_path=frame_dir)
    print(f"  init_state: {time.time() - t0:.1f}s for {state['num_frames']} frames")

    print(f"Adding point prompt at {INITIAL_POINT} on frame 0...")
    predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=np.array([list(INITIAL_POINT)], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    print("Propagating...")
    t0 = time.time()
    results = {}
    for frame_idx, obj_ids, masks_out in predictor.propagate_in_video(state):
        mask = (masks_out[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        bbox = mask_to_bbox(mask)
        results[frame_idx] = (mask, bbox)
    elapsed = time.time() - t0
    print(f"  Propagation: {elapsed:.1f}s ({NUM_FRAMES / elapsed:.1f} fps)")

    # Save annotated output
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    detected = 0
    for i in range(NUM_FRAMES):
        img = frames[i].copy()
        mask, bbox = results.get(i, (None, None))
        if mask is not None and bbox is not None:
            detected += 1
            img[mask] = (img[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
            draw.text((x0, y0 - 15), f"glass", fill="lime")
        else:
            pil = Image.fromarray(img)
        pil.save(os.path.join(output_dir, f"{i}.jpeg"), quality=90)

    print(f"\nResults: {detected}/{NUM_FRAMES} frames with detections")
    print(f"Annotated frames saved to {output_dir}/")

    # Sample bboxes
    for i in [0, NUM_FRAMES // 4, NUM_FRAMES // 2, 3 * NUM_FRAMES // 4, NUM_FRAMES - 1]:
        if i in results and results[i][1]:
            print(f"  Frame {i:3d}: bbox={results[i][1]}")
        else:
            print(f"  Frame {i:3d}: no detection")

    # Cleanup temp dir
    for f in os.listdir(frame_dir):
        os.remove(os.path.join(frame_dir, f))
    os.rmdir(frame_dir)


if __name__ == "__main__":
    asyncio.run(main())
