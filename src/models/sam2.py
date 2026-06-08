import io
import os
import shutil
import sys
import tempfile
import threading
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

# Set HSA_OVERRIDE_GFX_VERSION before torch is imported — required for AMD GPUs
# not yet in PyTorch's official ROCm support list. Must happen before any torch import.
if os.path.exists("/opt/rocm") and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Disable tqdm progress bars before any other imports — SAM2 uses tqdm internally
# and the output goes to stderr, which Viam logs as errors.
import tqdm as _tqdm_module
import tqdm.auto as _tqdm_auto_module


class _SilentTqdm:
    """A no-op tqdm replacement that acts as an identity iterator."""
    def __init__(self, iterable=None, *args, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable) if self._iterable is not None else iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc=None, refresh=True):
        pass


_tqdm_module.tqdm = _SilentTqdm
_tqdm_auto_module.tqdm = _SilentTqdm

import warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

import numpy as np
import torch
from PIL import Image as PILImage
from sam2.build_sam import build_sam2_video_predictor, HF_MODEL_ID_TO_FILENAMES
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing_extensions import Self
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.logging import getLogger
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes

LOGGER = getLogger(__name__)

OBJ_ID = 1
# Max frames to keep in the sliding window. Older frames are discarded.
DEFAULT_MAX_FRAMES = 300

# Bundled model. The build script downloads this checkpoint into
# checkpoints/ and packages it inside the module tarball. There is no
# config knob to change it: callers always get this exact model.
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"


def _select_device() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        LOGGER.debug(f"Using CUDA GPU: {device_name}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        LOGGER.debug("Using Apple MPS (Metal Performance Shaders)")
        return "mps"
    LOGGER.debug("No GPU detected, using CPU")
    return "cpu"


def _find_bundled_checkpoint() -> Tuple[str, str]:
    """Locate the bundled SAM2 checkpoint shipped with the module. Raises if missing."""
    config_name, ckpt_filename = HF_MODEL_ID_TO_FILENAMES[SAM2_MODEL_ID]
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),  # src/models/
        os.path.dirname(os.path.abspath(__file__)) + "/../..",  # sam2-detector/
        os.getcwd(),
    ]
    if hasattr(sys, "_MEIPASS"):
        search_dirs.insert(0, sys._MEIPASS)
    for d in search_dirs:
        path = os.path.join(d, "checkpoints", ckpt_filename)
        if os.path.isfile(path):
            return config_name, path
    raise FileNotFoundError(
        f"Bundled SAM2 checkpoint {ckpt_filename} not found in any of: "
        f"{[os.path.join(d, 'checkpoints') for d in search_dirs]}"
    )


def _load_predictor(device: str) -> SAM2VideoPredictor:
    """Load SAM2 VideoPredictor from the bundled checkpoint."""
    config_name, ckpt_path = _find_bundled_checkpoint()
    LOGGER.debug(f"Loading SAM2 VideoPredictor from bundled checkpoint: {ckpt_path}")
    return build_sam2_video_predictor(config_name, ckpt_path, device=device)


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


class Sam2(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "sam2-detector"), "sam2")

    _predictor: SAM2VideoPredictor
    _device: str = "cpu"
    _initial_point: Optional[Tuple[int, int]] = None
    _label: str = "object"
    _max_frames: int = DEFAULT_MAX_FRAMES
    _camera_name: str = ""
    _camera: ResourceBase

    # Sliding window of frames stored as numbered JPEGs in a temp dir.
    # Only the most recent _max_frames are kept.
    _frame_dir: Optional[str] = None
    _frame_count: int = 0  # total frames received (monotonic)
    _window_start: int = 0  # first frame index in the current window
    _frames_since_propagation: int = 0
    _propagation_interval: int = 1

    # Cached detections from the most recent propagation.
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
        instance._frame_count = 0
        instance._window_start = 0
        instance._frames_since_propagation = 0
        attrs = config.attributes.fields

        instance._camera_name = attrs["camera_name"].string_value
        from viam.components.camera import Camera
        instance._camera = dependencies[Camera.get_resource_name(instance._camera_name)]
        LOGGER.debug(f"Using camera: {instance._camera_name}")

        if "initial_point_x" in attrs and "initial_point_y" in attrs:
            instance._initial_point = (
                int(attrs["initial_point_x"].number_value),
                int(attrs["initial_point_y"].number_value),
            )
            LOGGER.debug(f"Initial point: {instance._initial_point}")
        else:
            instance._initial_point = None
            LOGGER.warn("No initial point configured; tracking will not start until set_point is called")

        if "label" in attrs:
            instance._label = attrs["label"].string_value
        if "propagation_interval" in attrs:
            instance._propagation_interval = int(attrs["propagation_interval"].number_value)
        if "max_frames" in attrs:
            instance._max_frames = int(attrs["max_frames"].number_value)
        else:
            instance._max_frames = DEFAULT_MAX_FRAMES

        # Create temp directory for frame JPEG storage.
        instance._frame_dir = tempfile.mkdtemp(prefix="sam2_frames_")

        instance._device = _select_device()
        LOGGER.debug(f"Loading SAM2 model {SAM2_MODEL_ID} on {instance._device}")
        instance._predictor = _load_predictor(instance._device)
        LOGGER.info("SAM2 model loaded")
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = config.attributes.fields
        if "camera_name" not in attrs or not attrs["camera_name"].string_value:
            raise ValueError("camera_name is required")
        has_x = "initial_point_x" in attrs
        has_y = "initial_point_y" in attrs
        if has_x != has_y:
            raise ValueError("Must provide both initial_point_x and initial_point_y, or neither")
        return [attrs["camera_name"].string_value], []

    async def _get_camera_image(self) -> np.ndarray:
        """Fetch an image from the configured camera dependency."""
        images, _ = await self._camera.get_images()
        viam_img = images[0]
        return _viam_image_to_numpy(viam_img)

    def _save_frame(self, image_np: np.ndarray) -> int:
        """Save a frame as a numbered JPEG and maintain the sliding window."""
        # Window-local index for the JPEG filename.
        local_idx = self._frame_count - self._window_start
        path = os.path.join(self._frame_dir, f"{local_idx}.jpeg")
        PILImage.fromarray(image_np).save(path, quality=90)
        frame_idx = self._frame_count
        self._frame_count += 1
        self._frames_since_propagation += 1

        # Evict old frames if we exceed the window size.
        window_size = self._frame_count - self._window_start
        if window_size > self._max_frames:
            self._compact_window()

        return frame_idx

    def _compact_window(self):
        """Remove old frames, keeping only the most recent max_frames."""
        window_size = self._frame_count - self._window_start
        if window_size <= self._max_frames:
            return

        keep_count = self._max_frames
        drop_count = window_size - keep_count
        new_start = self._window_start + drop_count

        # Rebuild the temp dir with renumbered files.
        new_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        for new_idx in range(keep_count):
            old_idx = drop_count + new_idx
            old_path = os.path.join(self._frame_dir, f"{old_idx}.jpeg")
            new_path = os.path.join(new_dir, f"{new_idx}.jpeg")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        shutil.rmtree(self._frame_dir, ignore_errors=True)
        self._frame_dir = new_dir
        self._window_start = new_start
        LOGGER.debug(f"Compacted window: dropped {drop_count} frames, keeping {keep_count}")

    def _run_propagation(self):
        """Run SAM2 video propagation on frames in the current window."""
        window_size = self._frame_count - self._window_start
        if window_size == 0 or self._initial_point is None:
            return

        LOGGER.debug(f"Running propagation on {window_size} frames")
        state = self._predictor.init_state(video_path=self._frame_dir)

        # Add initial point prompt on the first frame in the window.
        self._predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=OBJ_ID,
            points=np.array([list(self._initial_point)], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        # Propagate and cache detections (keyed by global frame index).
        self._detections = {}
        for local_idx, obj_ids, masks_out in self._predictor.propagate_in_video(state):
            mask = (masks_out[0] > 0.0).cpu().numpy().squeeze().astype(bool)
            bbox = _mask_to_bbox(mask)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                global_idx = self._window_start + local_idx
                self._detections[global_idx] = Detection(
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
        LOGGER.debug(f"Propagation complete: {len(self._detections)}/{window_size} detections")

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        image_np = _viam_image_to_numpy(image)

        with self._lock:
            frame_idx = self._save_frame(image_np)

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
        image_np = await self._get_camera_image()

        with self._lock:
            frame_idx = self._save_frame(image_np)

            if self._frames_since_propagation >= self._propagation_interval:
                self._run_propagation()

            det = self._detections.get(frame_idx)
            if det is not None:
                return [det]
            if self._last_detection is not None:
                return [self._last_detection]
            return []

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
                f"Camera name '{camera_name}' does not match "
                f"configured camera '{self._camera_name}'."
            )

        result = CaptureAllResult()

        images, _ = await self._camera.get_images()
        if images is None or len(images) == 0:
            raise ValueError("No images returned by get_images")

        if return_image:
            result.image = images[0]

        if return_detections:
            result.detections = await self.get_detections(
                images[0], extra=extra, timeout=timeout
            )

        return result

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

    async def close(self):
        """Clean up temp files on shutdown."""
        if self._frame_dir and os.path.isdir(self._frame_dir):
            shutil.rmtree(self._frame_dir, ignore_errors=True)
            LOGGER.debug(f"Cleaned up frame directory: {self._frame_dir}")
        self._frame_dir = None

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
                "num_frames": float(self._frame_count - self._window_start),
                "detections": float(len(self._detections)),
            }

        if cmd == "reset":
            with self._lock:
                if self._frame_dir:
                    shutil.rmtree(self._frame_dir, ignore_errors=True)
                    self._frame_dir = tempfile.mkdtemp(prefix="sam2_frames_")
                self._frame_count = 0
                self._window_start = 0
                self._frames_since_propagation = 0
                self._detections = {}
                self._last_detection = None
            return {"status": "reset complete"}

        if cmd == "status":
            return {
                "total_frames_received": float(self._frame_count),
                "window_size": float(self._frame_count - self._window_start),
                "max_frames": float(self._max_frames),
                "detections_cached": float(len(self._detections)),
                "device": self._device,
                "model_name": SAM2_MODEL_ID,
                "propagation_interval": float(self._propagation_interval),
            }

        return {"error": f"unknown command: {cmd}"}
