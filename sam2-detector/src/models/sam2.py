import io
import threading
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing_extensions import Self
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask (H, W) to a bounding box (x_min, y_min, x_max, y_max)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _viam_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a ViamImage to a numpy RGB array (H, W, 3)."""
    pil = PILImage.open(io.BytesIO(image.data)).convert("RGB")
    return np.array(pil)


class Sam2(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam2-detector"), "sam2")

    _predictor: Optional[SAM2ImagePredictor] = None
    _device: Optional[torch.device] = None
    _initial_point: Optional[Tuple[int, int]] = None
    _label: str = "object"
    _model_name: str = "facebook/sam2.1-hiera-small"

    # Tracking state: the bounding box from the previous frame.
    _prev_bbox: Optional[Tuple[int, int, int, int]] = None
    _lock: threading.Lock

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        instance._lock = threading.Lock()
        attrs = config.attributes.fields

        if "initial_point_x" in attrs and "initial_point_y" in attrs:
            instance._initial_point = (
                int(attrs["initial_point_x"].number_value),
                int(attrs["initial_point_y"].number_value),
            )
        else:
            instance._initial_point = None

        if "label" in attrs:
            instance._label = attrs["label"].string_value
        if "model_name" in attrs:
            instance._model_name = attrs["model_name"].string_value

        instance._prev_bbox = None
        instance._device = _select_device()
        instance.logger.info(f"Loading SAM2 model {instance._model_name} on {instance._device}")
        instance._predictor = SAM2ImagePredictor.from_pretrained(
            instance._model_name, device=instance._device
        )
        instance.logger.info("SAM2 model loaded")
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = config.attributes.fields
        has_x = "initial_point_x" in attrs
        has_y = "initial_point_y" in attrs
        if has_x != has_y:
            raise ValueError("Must provide both initial_point_x and initial_point_y, or neither")
        return [], []

    def _segment(self, image_np: np.ndarray) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """Run SAM2 on a single frame. Returns (mask, score, bbox) or None."""
        with self._lock:
            self._predictor.set_image(image_np)

            if self._prev_bbox is not None:
                # Track using previous frame's bounding box as a box prompt.
                box = np.array(list(self._prev_bbox), dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=box,
                    multimask_output=False,
                )
            elif self._initial_point is not None:
                # First frame: use the configured point prompt.
                point_coords = np.array([list(self._initial_point)], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                masks, scores, _ = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
            else:
                return None

            mask = masks[0]
            score = float(scores[0])
            bbox = _mask_to_bbox(mask)
            if bbox is None:
                self._prev_bbox = None
                return None

            self._prev_bbox = bbox
            return mask, score, bbox

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        image_np = _viam_image_to_numpy(image)
        result = self._segment(image_np)
        if result is None:
            return []
        _, score, (x_min, y_min, x_max, y_max) = result
        return [
            Detection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=self._label,
            )
        ]

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        raise NotImplementedError(
            "get_detections_from_camera not implemented; use get_detections with an image"
        )

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
        raise NotImplementedError(
            "capture_all_from_camera not implemented; use get_detections with an image"
        )

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("classifications not supported")

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        raise NotImplementedError("object point clouds not supported")

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command", "")
        if cmd == "reset_tracking":
            with self._lock:
                self._prev_bbox = None
            return {"status": "tracking reset"}
        if cmd == "set_point":
            x = int(command["x"])
            y = int(command["y"])
            with self._lock:
                self._initial_point = (x, y)
                self._prev_bbox = None
            return {"status": f"initial point set to ({x}, {y})"}
        return {"error": f"unknown command: {cmd}"}
