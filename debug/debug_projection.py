"""
Debug script: compare sam2-segmenter's depth -> 3D back-projection against
the underlying camera's native point cloud. The hypothesis is that
_mask_to_point_cloud (src/models/sam2_segments.py:287) is producing points
that don't match what the sensor itself reports.

This script is standalone and lightweight: it does NOT import sam2_segments
(which would pull torch + the SAM2 model), so it can run anywhere with just
viam-sdk, numpy, pillow.

Usage:
    cd /Users/bijanh/viam/vino/sam2
    uv run python debug/debug_projection.py
    # or override the camera name:
    VIAM_RAW_CAMERA=other-camera uv run python debug/debug_projection.py

Outputs (in cwd):
    debug_native.pcd        verbatim camera PCD bytes
    debug_reprojected.pcd   our reprojection of the depth map
    debug_color.png         color frame
    debug_depth.png         depth frame as 16-bit grayscale
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from viam.components.camera import Camera
from viam.media.video import ViamImage
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

# sam2's manual camera->world transform path (rust util via ctypes).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spatialmath import transform_points_with_pose  # noqa: E402

# --- env loading: same pattern as test_pointclouds.py ----------------------

env_path = os.path.join(os.path.dirname(__file__), "..", "..", "viam.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

ROBOT_ADDRESS = os.environ.get("VIAM_ROBOT_ADDRESS", "vino2-main.kssbd6djf3.viam.cloud")
RAW_CAMERA = os.environ.get("VIAM_RAW_CAMERA", "left-cam")

# --- helpers copied verbatim from src/models/sam2_segments.py / sam2.py ----
# Source of truth: src/models/sam2_segments.py:99 (_depth_image_to_numpy) and
# :116 (_encode_pcd_binary), src/models/sam2.py:126 (_viam_image_to_numpy).
# We duplicate them here to keep the debug script runnable without torch.

def _viam_image_to_numpy(image: ViamImage) -> np.ndarray:
    pil = PILImage.open(io.BytesIO(image.data)).convert("RGB")
    return np.array(pil)


def _depth_image_to_numpy(image: ViamImage) -> np.ndarray:
    mime = getattr(image, "mime_type", "")
    data = image.data
    if "viam" in str(mime) and "dep" in str(mime):
        depth_list = image.bytes_to_depth_array()
        return np.array(depth_list, dtype=np.float64)
    pil = PILImage.open(io.BytesIO(data))
    return np.array(pil, dtype=np.float64)


def _encode_pcd_binary(points_mm: np.ndarray, colors: np.ndarray) -> bytes:
    n = len(points_mm)
    if n == 0:
        return b""
    header = (
        f"VERSION .7\n"
        f"FIELDS x y z rgb\n"
        f"SIZE 4 4 4 4\n"
        f"TYPE F F F F\n"
        f"COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    ).encode("ascii")
    points_m = points_mm / 1000.0
    buf = bytearray(n * 16)
    for i in range(n):
        x, y, z = points_m[i]
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        struct.pack_into("ffff", buf, i * 16, float(x), float(y), float(z), rgb_float)
    return header + bytes(buf)


# --- PCD parser: returns (N, 3) float array in mm -------------------------

def parse_pcd_to_mm(pcd_bytes: bytes) -> np.ndarray:
    if not pcd_bytes:
        return np.empty((0, 3), dtype=np.float64)
    sep = b"DATA binary\n"
    idx = pcd_bytes.find(sep)
    is_ascii = False
    if idx < 0:
        sep = b"DATA ascii\n"
        idx = pcd_bytes.find(sep)
        is_ascii = True
    if idx < 0:
        raise ValueError("PCD: no DATA section found")
    header = pcd_bytes[:idx].decode("ascii", errors="replace")
    body = pcd_bytes[idx + len(sep):]

    fields, sizes, types, counts = [], [], [], []
    n = 0
    for line in header.splitlines():
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "FIELDS":
            fields = toks[1:]
        elif toks[0] == "SIZE":
            sizes = [int(x) for x in toks[1:]]
        elif toks[0] == "TYPE":
            types = toks[1:]
        elif toks[0] == "COUNT":
            counts = [int(x) for x in toks[1:]]
        elif toks[0] == "POINTS":
            n = int(toks[1])
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    if is_ascii:
        rows = []
        for line in body.decode("ascii", errors="replace").splitlines():
            toks = line.split()
            if len(toks) >= 3:
                rows.append([float(toks[0]), float(toks[1]), float(toks[2])])
        pts_m = np.array(rows[:n], dtype=np.float64)
    else:
        # Binary: assume the first three fields are x, y, z, each 4-byte float.
        # That matches Viam's convention and the encoder above.
        if fields[:3] != ["x", "y", "z"]:
            raise ValueError(f"PCD: unexpected field order {fields[:3]}")
        if sizes[:3] != [4, 4, 4] or types[:3] != ["F", "F", "F"]:
            raise ValueError("PCD: unexpected size/type for x,y,z")
        per_point = sum(s * c for s, c in zip(sizes, counts or [1] * len(sizes)))
        if per_point == 0:
            per_point = sum(sizes)
        arr = np.frombuffer(body[: n * per_point], dtype=np.uint8).reshape(n, per_point)
        xyz_bytes = arr[:, :12].copy()
        pts_m = xyz_bytes.view(np.float32).reshape(n, 3).astype(np.float64)
    # Viam PCDs are in meters; convert to mm to match reprojected output.
    return pts_m * 1000.0


# --- voxel-grid nearest-neighbor (no scipy) -------------------------------

class VoxelKNN:
    """Approximate nearest-neighbor over a point cloud using a fixed-size
    voxel grid. Each query inspects the query's cell + its 26 neighbors.
    Good enough for diagnostic statistics; not exact at boundaries."""

    def __init__(self, points: np.ndarray, cell_size: float = 5.0):
        self.cell_size = cell_size
        self.points = points  # (N, 3) float64
        self.cells: dict[Tuple[int, int, int], list[int]] = {}
        if len(points) == 0:
            return
        keys = np.floor(points / cell_size).astype(np.int64)
        for i, k in enumerate(keys):
            self.cells.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)

    def nearest(self, query: np.ndarray) -> np.ndarray:
        """Return distances (M,) from each query point to its nearest neighbor.
        Returns +inf where no neighbor exists in the searched cells."""
        if len(self.points) == 0 or len(query) == 0:
            return np.full(len(query), np.inf)
        out = np.full(len(query), np.inf)
        kq = np.floor(query / self.cell_size).astype(np.int64)
        for qi, q in enumerate(query):
            cx, cy, cz = int(kq[qi, 0]), int(kq[qi, 1]), int(kq[qi, 2])
            best = np.inf
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        idxs = self.cells.get((cx + dx, cy + dy, cz + dz))
                        if not idxs:
                            continue
                        diff = self.points[idxs] - q
                        d2 = np.einsum("ij,ij->i", diff, diff)
                        m = float(d2.min())
                        if m < best:
                            best = m
            out[qi] = np.sqrt(best)
        return out


# --- reprojection (mirror of _mask_to_point_cloud, no median filter) ------

def reproject_full_depth(
    depth_np: np.ndarray, fx: float, fy: float, ppx: float, ppy: float,
    sign_x: float = 1.0, sign_y: float = 1.0,
) -> np.ndarray:
    """Reproject every depth>0 pixel to 3D. Returns (N, 3) in the same units
    as depth_np (mm if the helper returned mm). sign_x/sign_y let the caller
    flip axes for the sign-flip probe."""
    h, w = depth_np.shape[:2]
    vs, us = np.where(depth_np > 0)
    depths = depth_np[vs, us].astype(np.float64)
    xs = sign_x * (us.astype(np.float64) - ppx) * depths / fx
    ys = sign_y * (vs.astype(np.float64) - ppy) * depths / fy
    zs = depths
    return np.stack([xs, ys, zs], axis=1)


# --- diagnostic blocks ----------------------------------------------------

def stats(label: str, pts: np.ndarray) -> str:
    if len(pts) == 0:
        return f"  {label}: 0 points"
    mn, mx, mean = pts.min(0), pts.max(0), pts.mean(0)
    return (
        f"  {label}: n={len(pts)}\n"
        f"    bbox min=({mn[0]:.1f}, {mn[1]:.1f}, {mn[2]:.1f}) "
        f"max=({mx[0]:.1f}, {mx[1]:.1f}, {mx[2]:.1f})\n"
        f"    mean=({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f})"
    )


def depth_diagnostic(depth_np: np.ndarray) -> str:
    nonzero = depth_np[depth_np > 0]
    pct_zero = 100.0 * (depth_np.size - len(nonzero)) / max(1, depth_np.size)
    if len(nonzero) == 0:
        return "  depth: ALL ZEROS — no usable pixels"
    in_mm_range = float(((nonzero >= 50) & (nonzero <= 10000)).mean()) * 100
    in_meter_range = float(((nonzero >= 0.05) & (nonzero <= 10.0)).mean()) * 100
    return (
        f"  depth: shape={depth_np.shape} dtype={depth_np.dtype} "
        f"min={nonzero.min():.3f} max={nonzero.max():.3f} median={np.median(nonzero):.3f}\n"
        f"    %zero={pct_zero:.1f}  %in_mm_range[50,10000]={in_mm_range:.1f}  "
        f"%in_meter_range[0.05,10]={in_meter_range:.1f}"
    )


def sign_flip_probe(
    depth_np: np.ndarray, fx: float, fy: float, ppx: float, ppy: float,
    native_mm: np.ndarray, sample: int = 5000,
) -> Tuple[Tuple[float, float], dict[Tuple[float, float], float]]:
    """For each of the four (sign_x, sign_y) combinations, reproject the depth
    map, sample ~`sample` points, and report the median nearest-neighbor
    distance to the native PCD. The combo with the smallest median wins."""
    knn = VoxelKNN(native_mm, cell_size=5.0)
    rng = np.random.default_rng(0)
    scores: dict[Tuple[float, float], float] = {}
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            pts = reproject_full_depth(depth_np, fx, fy, ppx, ppy, sx, sy)
            if len(pts) > sample:
                idx = rng.choice(len(pts), size=sample, replace=False)
                pts = pts[idx]
            d = knn.nearest(pts)
            d = d[np.isfinite(d)]
            scores[(sx, sy)] = float(np.median(d)) if len(d) else float("inf")
    best = min(scores.items(), key=lambda kv: kv[1])[0]
    return best, scores


def per_axis_offsets(
    reproj_mm: np.ndarray, native_mm: np.ndarray, sample: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean signed delta and abs delta per axis between each query point and
    its nearest native neighbor. Reveals constant offsets and frame flips."""
    if len(reproj_mm) == 0 or len(native_mm) == 0:
        return np.zeros(3), np.zeros(3)
    rng = np.random.default_rng(1)
    if len(reproj_mm) > sample:
        idx = rng.choice(len(reproj_mm), size=sample, replace=False)
        q = reproj_mm[idx]
    else:
        q = reproj_mm
    knn = VoxelKNN(native_mm, cell_size=5.0)
    # We need the actual nearest point, not just the distance. Re-do the
    # search in a small loop here so we have per-axis deltas.
    deltas = np.zeros((len(q), 3))
    for i, qp in enumerate(q):
        cx, cy, cz = (np.floor(qp / knn.cell_size).astype(int)).tolist()
        best_d2 = np.inf
        best_p = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    idxs = knn.cells.get((cx + dx, cy + dy, cz + dz))
                    if not idxs:
                        continue
                    diff = native_mm[idxs] - qp
                    d2 = np.einsum("ij,ij->i", diff, diff)
                    j = int(d2.argmin())
                    if d2[j] < best_d2:
                        best_d2 = float(d2[j])
                        best_p = native_mm[idxs[j]]
        if best_p is not None:
            deltas[i] = qp - best_p
        else:
            deltas[i] = np.nan
    deltas = deltas[~np.isnan(deltas).any(axis=1)]
    if len(deltas) == 0:
        return np.zeros(3), np.zeros(3)
    return deltas.mean(axis=0), np.abs(deltas).mean(axis=0)


# --- main -----------------------------------------------------------------

async def main() -> int:
    api_key = os.environ.get("VIAM_API_KEY", "")
    api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
    if not api_key or not api_key_id:
        print("Set VIAM_API_KEY and VIAM_API_KEY_ID (or put them in ../viam.env)")
        return 2

    print(f"Connecting to {ROBOT_ADDRESS}, raw camera = {RAW_CAMERA!r}")
    robot = await RobotClient.at_address(
        ROBOT_ADDRESS,
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id),
    )
    try:
        cam = Camera.from_robot(robot, RAW_CAMERA)

        t0 = time.time()
        images_resp = await cam.get_images()
        images = images_resp[0] if isinstance(images_resp, tuple) else images_resp
        props = await cam.get_properties()
        pcd_bytes, pcd_mime = await cam.get_point_cloud()
        print(f"  fetched everything in {time.time() - t0:.2f}s")

        # --- pick color + depth from images list (mirrors _get_color_and_depth)
        color_img: Optional[ViamImage] = None
        depth_img: Optional[ViamImage] = None
        for img in images:
            name = (getattr(img, "name", "") or getattr(img, "source_name", "") or "").lower()
            if "color" in name or "rgb" in name:
                color_img = img
            elif "depth" in name:
                depth_img = img
        if color_img is None and len(images) >= 1:
            color_img = images[0]
        if depth_img is None and len(images) >= 2:
            depth_img = images[1]
        if color_img is None or depth_img is None:
            print(f"  ERROR: camera returned {len(images)} images, need both color and depth")
            return 3

        color_np = _viam_image_to_numpy(color_img)
        depth_np = _depth_image_to_numpy(depth_img)

        intr = props.intrinsic_parameters
        if intr is None:
            print("  ERROR: camera reports no intrinsic_parameters; sam2 cannot project")
            return 4
        fx, fy = intr.focal_x_px, intr.focal_y_px
        ppx, ppy = intr.center_x_px, intr.center_y_px
        intr_w = getattr(intr, "width_px", 0)
        intr_h = getattr(intr, "height_px", 0)

        ch, cw = color_np.shape[:2]
        dh, dw = depth_np.shape[:2]

        print()
        print("=" * 70)
        print("INPUTS")
        print("=" * 70)
        print(f"  color: shape=({ch}, {cw}, 3) mime={getattr(color_img, 'mime_type', '?')}")
        print(depth_diagnostic(depth_np))
        print(f"  intrinsics: fx={fx:.2f} fy={fy:.2f} ppx={ppx:.2f} ppy={ppy:.2f} "
              f"reported_size=({intr_w},{intr_h})")
        sane_w = abs(2 * ppx - cw) < 0.2 * cw
        sane_h = abs(2 * ppy - ch) < 0.2 * ch
        print(f"    sanity: 2*ppx ≈ color_w? {sane_w}   2*ppy ≈ color_h? {sane_h}")
        print(f"  native PCD: {len(pcd_bytes)} bytes, mime={pcd_mime!r}")

        if (ch, cw) != (dh, dw):
            print(f"  ⚠️  color {cw}x{ch} ≠ depth {dw}x{dh} — mask indices into "
                  "depth_np will MISALIGN. Likely root cause.")

        # --- parse native PCD
        try:
            native_mm = parse_pcd_to_mm(pcd_bytes)
        except Exception as e:
            print(f"  ERROR parsing PCD: {e}")
            return 5

        # --- reproject (units = whatever depth is in; we annotate later)
        reproj = reproject_full_depth(depth_np, fx, fy, ppx, ppy)

        print()
        print("=" * 70)
        print("POINT CLOUDS")
        print("=" * 70)
        print(stats("native (mm)", native_mm))
        print(stats("reprojected (depth-units)", reproj))

        # --- unit guess
        if len(native_mm) > 0 and len(reproj) > 0:
            ratio = float(np.linalg.norm(reproj.mean(0))) / max(1.0, float(np.linalg.norm(native_mm.mean(0))))
            print(f"  ||mean(reproj)|| / ||mean(native_mm)|| = {ratio:.3f}")
            if 0.0008 < ratio < 0.0015:
                print("    → reproj is ~1000x smaller: depth is in METERS, not mm. "
                      "Fix: depth_np already in meters, multiply by 1000 OR keep mm and "
                      "skip the /1000 in _encode_pcd_binary. Check _depth_image_to_numpy.")
            elif 0.8 < ratio < 1.25:
                print("    → magnitudes match: depth units appear to be mm, OK.")
            elif ratio > 800:
                print("    → reproj is ~1000x bigger: depth is in 1/1000 mm or similar.")

        # --- sign-flip probe (assumes both clouds are in same units)
        if len(native_mm) > 100 and len(reproj) > 100:
            print()
            print("=" * 70)
            print("SIGN-FLIP PROBE (smaller median NN distance = better match)")
            print("=" * 70)
            best, scores = sign_flip_probe(depth_np, fx, fy, ppx, ppy, native_mm)
            for combo, s in scores.items():
                marker = "  <-- best" if combo == best else ""
                print(f"  sign_x={combo[0]:+.0f} sign_y={combo[1]:+.0f}: median NN = {s:.2f} mm{marker}")
            if best != (1.0, 1.0):
                print(f"  ⚠️  Identity (no flip) is NOT best. Apply sign_x={best[0]}, "
                      f"sign_y={best[1]} in _mask_to_point_cloud.")

            # --- per-axis offsets at the BEST sign combo
            best_reproj = reproject_full_depth(depth_np, fx, fy, ppx, ppy, best[0], best[1])
            mean_d, abs_d = per_axis_offsets(best_reproj, native_mm)
            print(f"  per-axis signed mean delta (mm): "
                  f"X={mean_d[0]:+.2f} Y={mean_d[1]:+.2f} Z={mean_d[2]:+.2f}")
            print(f"  per-axis  abs   mean delta (mm): "
                  f"X={abs_d[0]:.2f} Y={abs_d[1]:.2f} Z={abs_d[2]:.2f}")

        # --- world-frame transform check ---------------------------------
        # The viewer screenshots show sam2's cup lying flat while
        # cam-left-cup-crop's stands upright — the camera->world rotation
        # sam2 applies in Python disagrees with what Viam applies on the
        # server. Cross-check sam2's path (rust util) against Viam's
        # authoritative client.transform_pose for the same camera-frame
        # points.
        print()
        print("=" * 70)
        print("WORLD-FRAME TRANSFORM CHECK")
        print("=" * 70)
        try:
            origin = PoseInFrame(
                reference_frame=RAW_CAMERA,
                pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0),
            )
            cam_in_world = await robot.transform_pose(origin, "world")
            cp = cam_in_world.pose
            print(f"  camera->world pose:")
            print(f"    translation (mm): ({cp.x:.1f}, {cp.y:.1f}, {cp.z:.1f})")
            print(f"    OV: ({cp.o_x:.4f}, {cp.o_y:.4f}, {cp.o_z:.4f})  theta_deg: {cp.theta:.4f}")

            # Test points (camera frame, mm). These exercise each axis.
            test_pts = np.array([
                [0.0, 0.0, 0.0],        # camera origin
                [0.0, 0.0, 1000.0],     # 1m forward (CV: along optical axis)
                [1000.0, 0.0, 0.0],     # 1m to the right of optical axis
                [0.0, 1000.0, 0.0],     # 1m down (CV Y points down)
            ], dtype=np.float64)
            labels = ["origin", "+Z (forward)", "+X (right)", "+Y (down)"]

            sam2_world = transform_points_with_pose(
                cp.o_x, cp.o_y, cp.o_z, cp.theta,
                cp.x, cp.y, cp.z, test_pts,
            )
            print()
            print("  SAM2 path (rust util) — points in world frame (mm):")
            for lbl, p, w in zip(labels, test_pts, sam2_world):
                print(f"    cam {lbl:>14}=({p[0]:6.1f},{p[1]:6.1f},{p[2]:6.1f})  -> "
                      f"world=({w[0]:8.1f},{w[1]:8.1f},{w[2]:8.1f})")

            print()
            print("  Viam path (server transform_pose) — same points:")
            viam_world = np.zeros_like(sam2_world)
            for i, p in enumerate(test_pts):
                pose_in_cam = PoseInFrame(
                    reference_frame=RAW_CAMERA,
                    pose=Pose(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                              o_x=0, o_y=0, o_z=1, theta=0),
                )
                pin = await robot.transform_pose(pose_in_cam, "world")
                viam_world[i] = [pin.pose.x, pin.pose.y, pin.pose.z]
                lbl = labels[i]
                w = viam_world[i]
                print(f"    cam {lbl:>14}=({p[0]:6.1f},{p[1]:6.1f},{p[2]:6.1f})  -> "
                      f"world=({w[0]:8.1f},{w[1]:8.1f},{w[2]:8.1f})")

            print()
            print("  Per-test deltas (sam2_path - viam_path), mm:")
            deltas = sam2_world - viam_world
            for lbl, d in zip(labels, deltas):
                norm = float(np.linalg.norm(d))
                marker = "  ⚠️ DIFFERS" if norm > 1.0 else ""
                print(f"    {lbl:>14}: ({d[0]:+8.2f},{d[1]:+8.2f},{d[2]:+8.2f})  |d|={norm:.2f}{marker}")

            max_norm = float(np.linalg.norm(deltas, axis=1).max())
            if max_norm > 1.0:
                print()
                print(f"  ⚠️  sam2's manual transform diverges from Viam's by up to "
                      f"{max_norm:.1f}mm. The bug is in src/spatialmath.py "
                      f"(transform_points_with_pose) or its rust-util call. "
                      f"Likely culprit: theta units (deg vs rad — line 121 does "
                      f"math.radians(theta_deg); the rust util may already expect "
                      f"degrees, causing a double-conversion).")
            else:
                print(f"  ✓ sam2's transform agrees with Viam's within {max_norm:.2f}mm "
                      "— transform is not the bug; look elsewhere (Z-flip in viewer? "
                      "different camera frame for cam-left-cup-crop?).")

            # Save the native PCD transformed to world via BOTH paths so the
            # user can overlay them in CloudCompare against cam-left-cup-crop.
            if len(native_mm) > 0:
                native_world_sam2 = transform_points_with_pose(
                    cp.o_x, cp.o_y, cp.o_z, cp.theta,
                    cp.x, cp.y, cp.z, native_mm,
                )
                colors_gray = np.tile(np.array([[200, 200, 200]], dtype=np.uint8),
                                       (len(native_world_sam2), 1))
                with open("debug_native_world_sam2path.pcd", "wb") as f:
                    f.write(_encode_pcd_binary(native_world_sam2, colors_gray))
                print("  Wrote debug_native_world_sam2path.pcd (native PCD via sam2's transform).")
        except Exception as e:
            print(f"  ERROR running world-transform check: {e}")

        # --- artifacts
        with open("debug_native.pcd", "wb") as f:
            f.write(pcd_bytes)
        # Reuse the shipping encoder so the saved PCD format matches what
        # sam2 produces in production.
        gray = np.zeros_like(reproj[:, :1], dtype=np.uint8) if len(reproj) else np.empty((0, 1), dtype=np.uint8)
        colors = np.tile(np.array([[200, 200, 200]], dtype=np.uint8), (len(reproj), 1))
        with open("debug_reprojected.pcd", "wb") as f:
            f.write(_encode_pcd_binary(reproj, colors))
        PILImage.fromarray(color_np).save("debug_color.png")
        # Save depth as 16-bit grayscale; clip to uint16 max to be safe.
        d16 = np.clip(depth_np, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        PILImage.fromarray(d16, mode="I;16").save("debug_depth.png")
        print()
        print("Wrote: debug_native.pcd, debug_reprojected.pcd, debug_color.png, debug_depth.png")
        print("Open both .pcd files in CloudCompare to visually compare.")
        return 0
    finally:
        await robot.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
