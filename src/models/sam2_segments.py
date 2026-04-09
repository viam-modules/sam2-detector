"""
SAM2 Segments: combines upstream 2D detections with SAM2 segmentation masks
and depth-based 3D projection to produce precise object point clouds.

Pipeline: detector bbox → SAM2 mask → depth projection → PointCloudObject
"""

import io
import struct
import threading
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    Pose,
    PoseInFrame,
    PointCloudObject,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes

# Reuse shared utilities from the sam2 module.
from models.sam2 import _select_device, _find_bundled_checkpoint, _viam_image_to_numpy

LOGGER = getLogger(__name__)


def _load_image_predictor(model_name: str, device: str) -> SAM2ImagePredictor:
    """Load SAM2 ImagePredictor, preferring a bundled checkpoint."""
    bundled = _find_bundled_checkpoint(model_name)
    if bundled is not None:
        config_name, ckpt_path = bundled
        LOGGER.info(f"Loading ImagePredictor from bundled checkpoint: {ckpt_path}")
        from sam2.build_sam import build_sam2
        model = build_sam2(config_name, ckpt_path, device=device)
        return SAM2ImagePredictor(model)
    LOGGER.info(f"Downloading ImagePredictor from HuggingFace: {model_name}")
    return SAM2ImagePredictor.from_pretrained(model_name, device=device)


def _depth_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a Viam depth image to a numpy array (H, W) of depth in mm."""
    mime = getattr(image, "mime_type", "")
    if "viam" in str(mime) and "dep" in str(mime):
        # Viam raw depth format: 24-byte header + uint16 LE pixels.
        data = image.data
        # Header: 8 bytes magic, 8 bytes width, 8 bytes height (all little-endian uint64).
        width = int.from_bytes(data[8:16], "little")
        height = int.from_bytes(data[16:24], "little")
        pixels = np.frombuffer(data[24:], dtype=np.uint16).reshape((height, width))
        return pixels.astype(np.float64)
    else:
        # PNG or other format: decode as 16-bit grayscale.
        pil = PILImage.open(io.BytesIO(image.data))
        return np.array(pil, dtype=np.float64)


def _encode_pcd_binary(points: np.ndarray, colors: np.ndarray) -> bytes:
    """Encode points (N,3) float64 and colors (N,3) uint8 as binary PCD."""
    n = len(points)
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

    # Pack points + RGB into binary.
    buf = bytearray(n * 16)  # 4 floats * 4 bytes each
    for i in range(n):
        x, y, z = points[i]
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        struct.pack_into("ffff", buf, i * 16, float(x), float(y), float(z), rgb_float)

    return header + bytes(buf)


class Sam2Segments(Vision, EasyResource):
    """Vision service that combines upstream detections + SAM2 masks + depth → 3D point clouds."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam2-detector"), "sam2-segments")

    _predictor: Optional[SAM2ImagePredictor] = None
    _device: str = "cpu"
    _detector: Optional[ResourceBase] = None
    _camera: Optional[ResourceBase] = None
    _detector_name: str = ""
    _camera_name: str = ""
    _label: str = ""
    _confidence_threshold: float = 0.5
    _model_name: str = "facebook/sam2.1-hiera-tiny"
    _depth_threshold_mm: int = 0
    _min_points: int = 50
    _lock: threading.Lock

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        instance._lock = threading.Lock()
        attrs = config.attributes.fields

        instance._detector_name = attrs["detector_name"].string_value
        instance._camera_name = attrs["camera_name"].string_value

        # Resolve dependencies.
        instance._detector = dependencies[Vision.get_resource_name(instance._detector_name)]
        instance._camera = dependencies[Camera.get_resource_name(instance._camera_name)]
        LOGGER.info(f"Using detector: {instance._detector_name}, camera: {instance._camera_name}")

        if "label" in attrs:
            instance._label = attrs["label"].string_value
        if "confidence_threshold" in attrs:
            instance._confidence_threshold = attrs["confidence_threshold"].number_value
        if "model_name" in attrs:
            instance._model_name = attrs["model_name"].string_value
        if "depth_threshold_mm" in attrs:
            instance._depth_threshold_mm = int(attrs["depth_threshold_mm"].number_value)
        if "min_points" in attrs:
            instance._min_points = int(attrs["min_points"].number_value)

        instance._device = _select_device()
        LOGGER.info(f"Loading SAM2 ImagePredictor ({instance._model_name}) on {instance._device}")
        instance._predictor = _load_image_predictor(instance._model_name, instance._device)
        LOGGER.info("SAM2 ImagePredictor loaded")

        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = config.attributes.fields
        if "detector_name" not in attrs or not attrs["detector_name"].string_value:
            raise ValueError("detector_name is required")
        if "camera_name" not in attrs or not attrs["camera_name"].string_value:
            raise ValueError("camera_name is required")
        return [attrs["detector_name"].string_value, attrs["camera_name"].string_value], []

    # ---- Core pipeline helpers ----

    async def _get_color_and_depth(self) -> Tuple[ViamImage, np.ndarray, np.ndarray]:
        """Get color and depth images from camera. Returns (color_viam, color_np, depth_np)."""
        images, _ = await self._camera.get_images()
        color_img = None
        depth_img = None
        for img in images:
            name = getattr(img, "name", "") or getattr(img, "source_name", "") or ""
            if "color" in name.lower() or "rgb" in name.lower():
                color_img = img
            elif "depth" in name.lower():
                depth_img = img
        # Fallback: first is color, second is depth.
        if color_img is None and len(images) >= 1:
            color_img = images[0]
        if depth_img is None and len(images) >= 2:
            depth_img = images[1]
        if color_img is None or depth_img is None:
            raise ValueError("Camera must return both color and depth images")

        color_np = _viam_image_to_numpy(color_img)
        depth_np = _depth_image_to_numpy(depth_img)
        return color_img, color_np, depth_np

    async def _get_filtered_detections(self, image: ViamImage) -> List[Detection]:
        """Call upstream detector and filter by label/confidence."""
        detections = await self._detector.get_detections(image)
        filtered = []
        for det in detections:
            if det.confidence < self._confidence_threshold:
                continue
            if self._label and det.class_name != self._label:
                continue
            filtered.append(det)
        return filtered

    def _sam2_refine(self, color_np: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Use SAM2 to get a precise mask from a bounding box prompt."""
        with self._lock:
            self._predictor.set_image(color_np)
            box = np.array(list(bbox), dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=box, multimask_output=False
            )
            mask = masks[0].astype(bool)
            if not mask.any():
                return None
            return mask

    def _mask_to_point_cloud(
        self,
        mask: np.ndarray,
        color_np: np.ndarray,
        depth_np: np.ndarray,
        fx: float, fy: float, ppx: float, ppy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project masked pixels to 3D using pinhole camera model. Returns (points, colors)."""
        vs, us = np.where(mask)
        depths = depth_np[vs, us].astype(np.float64)

        # Filter zero-depth pixels.
        valid = depths > 0
        vs, us, depths = vs[valid], us[valid], depths[valid]
        if len(depths) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

        # Filter by depth threshold (median ± threshold).
        if self._depth_threshold_mm > 0:
            median = np.median(depths)
            within = np.abs(depths - median) <= self._depth_threshold_mm
            vs, us, depths = vs[within], us[within], depths[within]
            if len(depths) == 0:
                return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

        # Pinhole projection: pixel (u, v, Z) → 3D (X, Y, Z).
        xs = (us.astype(np.float64) - ppx) * depths / fx
        ys = (vs.astype(np.float64) - ppy) * depths / fy
        zs = depths

        points = np.stack([xs, ys, zs], axis=1)
        colors = color_np[vs, us]  # (N, 3) uint8

        return points, colors

    def _build_point_cloud_object(
        self, points: np.ndarray, colors: np.ndarray, label: str
    ) -> PointCloudObject:
        """Create a PointCloudObject from 3D points and colors."""
        pcd_bytes = _encode_pcd_binary(points, colors)

        # Compute bounding box geometry.
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        dims = maxs - mins

        geometry = Geometry(
            center=Pose(
                x=float(center[0]), y=float(center[1]), z=float(center[2]),
                o_x=0, o_y=0, o_z=1, theta=0,
            ),
            box=RectangularPrism(dims_mm=Vector3(
                x=float(dims[0]), y=float(dims[1]), z=float(dims[2]),
            )),
            label=label,
        )

        return PointCloudObject(
            point_cloud=pcd_bytes,
            geometries=GeometriesInFrame(
                reference_frame=self._camera_name,
                geometries=[geometry],
            ),
        )

    async def _transform_points_to_world(
        self, points: np.ndarray, robot_client
    ) -> np.ndarray:
        """Transform points from camera frame to world frame using the frame system."""
        try:
            # Get the camera-to-world transform via transform_pose.
            origin_in_camera = PoseInFrame(
                reference_frame=self._camera_name,
                pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0),
            )
            world_pose = await robot_client.transform_pose(origin_in_camera, "world")
            p = world_pose.pose

            # Extract translation.
            translation = np.array([p.x, p.y, p.z])

            # Extract rotation from orientation vector + theta.
            # For simplicity, if the pose is identity-ish, just apply translation.
            # Full quaternion rotation would require more complex math.
            # TODO: implement full rotation if needed.
            return points + translation

        except Exception as e:
            LOGGER.warn(f"Could not transform to world frame, returning camera frame: {e}")
            return points

    # ---- Vision service API ----

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        # Get camera intrinsics.
        props = await self._camera.get_properties()
        intrinsics = props.intrinsic_parameters
        if intrinsics is None:
            raise ValueError("Camera must provide intrinsic parameters for 3D projection")
        fx = intrinsics.focal_x_px
        fy = intrinsics.focal_y_px
        ppx = intrinsics.center_x_px
        ppy = intrinsics.center_y_px
        LOGGER.debug(f"Intrinsics: fx={fx}, fy={fy}, ppx={ppx}, ppy={ppy}")

        # Get color + depth images.
        color_viam, color_np, depth_np = await self._get_color_and_depth()

        # Get upstream detections.
        detections = await self._get_filtered_detections(color_viam)
        LOGGER.debug(f"Got {len(detections)} filtered detections")

        results = []
        for det in detections:
            bbox = (det.x_min, det.y_min, det.x_max, det.y_max)

            # SAM2 refine: bbox → precise mask.
            mask = self._sam2_refine(color_np, bbox)
            if mask is None:
                LOGGER.debug(f"SAM2 returned empty mask for bbox {bbox}")
                continue

            # Project masked pixels to 3D.
            points, colors = self._mask_to_point_cloud(
                mask, color_np, depth_np, fx, fy, ppx, ppy
            )
            if len(points) < self._min_points:
                LOGGER.debug(f"Skipping segment with {len(points)} points (min={self._min_points})")
                continue

            label = det.class_name or self._label or "object"
            pco = self._build_point_cloud_object(points, colors, label)
            results.append(pco)

        LOGGER.info(f"Returning {len(results)} point cloud objects")
        return results

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Pass-through: return filtered detections from upstream detector."""
        return await self._get_filtered_detections(image)

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Get detections using the configured camera."""
        images, _ = await self._camera.get_images()
        if not images:
            return []
        return await self._get_filtered_detections(images[0])

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        if camera_name not in (self._camera_name, ""):
            raise ValueError(
                f"Camera name '{camera_name}' does not match configured camera '{self._camera_name}'."
            )

        result = CaptureAllResult()
        images, _ = await self._camera.get_images()
        if not images:
            return result

        if return_image:
            result.image = images[0]

        if return_detections:
            result.detections = await self._get_filtered_detections(images[0])

        if return_object_point_clouds:
            result.objects = await self.get_object_point_clouds(
                camera_name, extra=extra, timeout=timeout
            )

        return result

    async def get_classifications_from_camera(
        self, camera_name: str, count: int, *, extra=None, timeout=None
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_classifications(
        self, image: ViamImage, count: int, *, extra=None, timeout=None
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_properties(self, *, extra=None, timeout=None) -> Vision.Properties:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
        )

    async def close(self):
        LOGGER.info("Sam2Segments shutting down")

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command", "")
        if cmd == "status":
            return {
                "detector_name": self._detector_name,
                "camera_name": self._camera_name,
                "label": self._label,
                "confidence_threshold": self._confidence_threshold,
                "model_name": self._model_name,
                "device": self._device,
                "depth_threshold_mm": float(self._depth_threshold_mm),
                "min_points": float(self._min_points),
            }
        return {"error": f"unknown command: {cmd}"}
