import io
import threading
from collections import OrderedDict
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from sam2.sam2_video_predictor import SAM2VideoPredictor
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

OBJ_ID = 1
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMG_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask (H, W) to (x_min, y_min, x_max, y_max)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _viam_image_to_numpy(image: ViamImage) -> np.ndarray:
    """Convert a ViamImage to a numpy RGB array (H, W, 3)."""
    pil = PILImage.open(io.BytesIO(image.data)).convert("RGB")
    return np.array(pil)


def _numpy_to_tensor(img_np: np.ndarray, image_size: int) -> Tuple[torch.Tensor, int, int]:
    """Convert an RGB numpy array to the normalized tensor SAM2 expects."""
    h, w = img_np.shape[:2]
    pil = PILImage.fromarray(img_np).resize((image_size, image_size))
    arr = np.array(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor, h, w


def _build_inference_state(
    predictor: SAM2VideoPredictor,
    frame_tensors: List[torch.Tensor],
    video_height: int,
    video_width: int,
    device: str,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
) -> dict:
    """Build a SAM2 inference_state dict from in-memory frame tensors."""
    compute_device = torch.device(device)
    images = torch.stack(frame_tensors)  # (N, 3, H, W)

    img_mean = IMG_MEAN.clone()
    img_std = IMG_STD.clone()
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    images -= img_mean
    images /= img_std

    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(frame_tensors)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    # Warm up visual backbone on frame 0.
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


class Sam2(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam2-detector"), "sam2")

    _predictor: Optional[SAM2VideoPredictor] = None
    _device: str = "cpu"
    _initial_point: Optional[Tuple[int, int]] = None
    _label: str = "object"
    _model_name: str = "facebook/sam2.1-hiera-large"

    # In-memory frame buffer: list of preprocessed tensors.
    _frame_tensors: List[torch.Tensor] = []
    _video_height: int = 0
    _video_width: int = 0
    _frame_count: int = 0
    _frames_since_propagation: int = 0
    _propagation_interval: int = 1

    # Cached detections from the most recent propagation, keyed by frame index.
    _detections: Dict[int, Detection] = {}
    _last_detection: Optional[Detection] = None
    _lock: threading.Lock

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        instance._lock = threading.Lock()
        instance._detections = {}
        instance._last_detection = None
        instance._frame_tensors = []
        instance._frame_count = 0
        instance._frames_since_propagation = 0
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
        if "propagation_interval" in attrs:
            instance._propagation_interval = int(attrs["propagation_interval"].number_value)

        instance._device = _select_device()
        instance.logger.info(f"Loading SAM2 model {instance._model_name} on {instance._device}")
        instance._predictor = SAM2VideoPredictor.from_pretrained(
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

    def _add_frame(self, image_np: np.ndarray) -> int:
        """Preprocess and buffer a frame in memory. Returns the frame index."""
        tensor, h, w = _numpy_to_tensor(image_np, self._predictor.image_size)
        if self._frame_count == 0:
            self._video_height = h
            self._video_width = w
        self._frame_tensors.append(tensor)
        idx = self._frame_count
        self._frame_count += 1
        self._frames_since_propagation += 1
        return idx

    def _run_propagation(self):
        """Run SAM2 video propagation on all buffered frames."""
        if self._frame_count == 0 or self._initial_point is None:
            return

        state = _build_inference_state(
            self._predictor,
            self._frame_tensors,
            self._video_height,
            self._video_width,
            self._device,
        )

        self._predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=OBJ_ID,
            points=np.array([list(self._initial_point)], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        self._detections = {}
        for frame_idx, obj_ids, masks_out in self._predictor.propagate_in_video(state):
            mask = (masks_out[0] > 0.0).cpu().numpy().squeeze().astype(bool)
            bbox = _mask_to_bbox(mask)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                self._detections[frame_idx] = Detection(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=1.0,
                    class_name=self._label,
                )

        self._frames_since_propagation = 0
        if self._detections:
            max_idx = max(self._detections.keys())
            self._last_detection = self._detections[max_idx]

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        image_np = _viam_image_to_numpy(image)

        with self._lock:
            frame_idx = self._add_frame(image_np)

            if self._frames_since_propagation >= self._propagation_interval:
                self._run_propagation()

            det = self._detections.get(frame_idx)
            if det is not None:
                return [det]
            if self._last_detection is not None:
                return [self._last_detection]
            return []

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        raise NotImplementedError(
            "get_detections_from_camera not implemented; use get_detections"
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
            "capture_all_from_camera not implemented; use get_detections"
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

        if cmd == "set_point":
            x = int(command["x"])
            y = int(command["y"])
            with self._lock:
                self._initial_point = (x, y)
                self._detections = {}
                self._last_detection = None
            return {"status": f"initial point set to ({x}, {y})"}

        if cmd == "reprocess":
            with self._lock:
                self._run_propagation()
            return {
                "status": "ok",
                "num_frames": float(self._frame_count),
                "detections": float(len(self._detections)),
            }

        if cmd == "reset":
            with self._lock:
                self._frame_tensors = []
                self._frame_count = 0
                self._frames_since_propagation = 0
                self._detections = {}
                self._last_detection = None
            return {"status": "reset complete"}

        if cmd == "status":
            return {
                "num_frames": float(self._frame_count),
                "detections_cached": float(len(self._detections)),
                "device": self._device,
                "model_name": self._model_name,
                "propagation_interval": float(self._propagation_interval),
            }

        return {"error": f"unknown command: {cmd}"}
