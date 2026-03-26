"""
Test script: connects to a Viam robot, pulls frames from a camera,
runs SAM2 VideoPredictor tracking, and saves annotated output + video.

Usage:
    uv run python test_camera.py [--num-frames 50] [--point 600,300]
"""

import asyncio
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.sam2_video_predictor import SAM2VideoPredictor
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.components.camera import Camera

# Load env vars from viam.env if it exists.
env_path = os.path.join(os.path.dirname(__file__), "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "computer-demo-main.496koy7yd1.viam.cloud")
CAMERA_NAME = os.environ.get("VIAM_CAMERA_NAME", "camera-image-dir")
MODEL_NAME = "facebook/sam2.1-hiera-large"
NUM_FRAMES = 50
INITIAL_POINT = (600, 300)
RAW_DIR = "test_raw_frames"
OUTPUT_DIR = "test_output"


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


async def main():
    # Parse simple args.
    num_frames = NUM_FRAMES
    point = INITIAL_POINT
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--num-frames" and i < len(sys.argv):
            num_frames = int(sys.argv[i + 1])
        if arg == "--point" and i < len(sys.argv):
            x, y = sys.argv[i + 1].split(",")
            point = (int(x), int(y))

    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    if not api_key or not api_key_id:
        print("Set VIAM_API_KEY and VIAM_API_KEY_ID (or put them in ../viam.env)")
        return

    # Connect to robot.
    print(f"Connecting to {ROBOT_ADDRESS}...")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options(
            dial_options=DialOptions.with_api_key(
                api_key=api_key, api_key_id=api_key_id,
            ),
        ),
    )
    print(f"Connected! Resources: {[r.name for r in robot.resource_names]}")

    camera = Camera.from_robot(robot, CAMERA_NAME)

    # Prepare output dirs.
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pull frames and save to temp dir for SAM2 + raw dir for inspection.
    frame_dir = tempfile.mkdtemp(prefix="sam2_test_")
    print(f"Pulling {num_frames} frames from '{CAMERA_NAME}'...")
    raw_frames = []
    pull_times = []
    prev_time = time.time()
    for i in range(num_frames):
        t0 = time.time()
        images, _ = await camera.get_images()
        viam_img = images[0]
        pull_ms = (time.time() - t0) * 1000
        gap_ms = (t0 - prev_time) * 1000 if i > 0 else 0
        prev_time = time.time()
        pull_times.append((pull_ms, gap_ms))

        pil = Image.open(io.BytesIO(viam_img.data)).convert("RGB")
        arr = np.array(pil)
        raw_frames.append(arr)
        # Save for SAM2 processing.
        pil.save(os.path.join(frame_dir, f"{i}.jpeg"), quality=90)
        # Save raw frame for comparison.
        pil.save(os.path.join(RAW_DIR, f"{i}.jpeg"), quality=90)

        if i == 0:
            print(f"  Frame size: {arr.shape}")
        if (i + 1) % 10 == 0:
            avg_pull = np.mean([t[0] for t in pull_times[-10:]])
            avg_gap = np.mean([t[1] for t in pull_times[-10:] if t[1] > 0])
            print(f"  Pulled {i + 1}/{num_frames}  "
                  f"(avg pull: {avg_pull:.0f}ms, avg gap: {avg_gap:.0f}ms)")

    await robot.close()

    # Report frame pull timing stats.
    pulls = [t[0] for t in pull_times]
    gaps = [t[1] for t in pull_times[1:]]
    print(f"\nFrame pull stats ({num_frames} frames):")
    print(f"  Pull time:  min={min(pulls):.0f}ms  avg={np.mean(pulls):.0f}ms  max={max(pulls):.0f}ms")
    if gaps:
        print(f"  Gap between: min={min(gaps):.0f}ms  avg={np.mean(gaps):.0f}ms  max={max(gaps):.0f}ms")
    print(f"  Raw frames saved to {RAW_DIR}/")

    # Run SAM2 VideoPredictor.
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading SAM2 ({MODEL_NAME}) on {device}...")
    predictor = SAM2VideoPredictor.from_pretrained(MODEL_NAME, device=device)

    print("Initializing video state...")
    t0 = time.time()
    state = predictor.init_state(video_path=frame_dir)
    print(f"  init_state: {time.time() - t0:.1f}s, {state['num_frames']} frames")

    print(f"Adding point prompt at {point} on frame 0...")
    predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=np.array([list(point)], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    print("Propagating...")
    t0 = time.time()
    results = {}
    for frame_idx, obj_ids, masks_out in predictor.propagate_in_video(state):
        mask = (masks_out[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        results[frame_idx] = (mask, mask_to_bbox(mask))
    elapsed = time.time() - t0
    detected = sum(1 for _, (_, b) in results.items() if b is not None)
    print(f"  {elapsed:.1f}s ({num_frames / elapsed:.1f} fps), {detected}/{num_frames} detections")

    # Save annotated output.
    print(f"\nSaving annotated frames to {OUTPUT_DIR}/...")
    for i in range(num_frames):
        img = raw_frames[i].copy()
        mask, bbox = results.get(i, (None, None))
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)

        # Draw frame number and timing info.
        pull_ms, gap_ms = pull_times[i]
        info = f"#{i}  pull:{pull_ms:.0f}ms"
        if i > 0:
            info += f"  gap:{gap_ms:.0f}ms"
        draw.text((10, 10), info, fill="white")

        if mask is not None and bbox is not None:
            # Draw green mask overlay.
            img_overlay = img.copy()
            img_overlay[mask] = (img_overlay[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
            pil = Image.fromarray(img_overlay)
            draw = ImageDraw.Draw(pil)
            draw.text((10, 10), info, fill="white")

            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
            draw.text((x0, y0 - 15), f"glass [{x1-x0}x{y1-y0}]", fill="lime")
        else:
            draw.text((10, 30), "NO DETECTION", fill="red")

        pil.save(os.path.join(OUTPUT_DIR, f"{i}.jpeg"), quality=90)

    # Build video from annotated frames.
    video_path = "test_tracked.mp4"
    print(f"Building video: {video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "5",
            "-i", f"{OUTPUT_DIR}/%d.jpeg",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            video_path,
        ],
        capture_output=True,
    )

    print(f"\nDone!")
    print(f"  Raw frames:      {RAW_DIR}/")
    print(f"  Annotated frames: {OUTPUT_DIR}/")
    print(f"  Tracked video:    {video_path}")

    # Sample bboxes.
    sample_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
    for i in sample_indices:
        b = results.get(i, (None, None))[1]
        print(f"  Frame {i:3d}: {b or 'no detection'}")

    # Cleanup temp dir.
    shutil.rmtree(frame_dir)


if __name__ == "__main__":
    asyncio.run(main())
