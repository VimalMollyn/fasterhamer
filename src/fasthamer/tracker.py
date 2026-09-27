"""The main fasthamer API: HandMesh."""
import sys
import time
from typing import Optional, Sequence

import numpy as np

from .detection import make_detector
from .preprocess import cam_crop_to_full, preprocess_hands, scaled_focal_length
from .rendering import MESH_COLOR, MeshRenderer
from .stabilize import HandednessStabilizer
from .types import Hand, HandMeshResult


def default_backend() -> str:
    """The inference backend `backend="auto"` resolves to: CoreML on macOS (Apple Neural
    Engine), PyTorch everywhere else (CUDA if available)."""
    return "coreml" if sys.platform == "darwin" else "torch"


def _make_engine(backend: str, model_dir: Optional[str], compute_units: str, device: str,
                 dtype: str, input_size):
    if backend == "auto":
        backend = default_backend()
    if backend == "coreml":
        from .assets import resolve_model_dir
        from .inference import CoreMLHamer
        return CoreMLHamer(resolve_model_dir(model_dir), compute_units=compute_units), backend
    if backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("the torch backend needs PyTorch: pip install torch "
                              "(see https://pytorch.org for the CUDA build), or "
                              "pip install 'fasthamer[torch]'") from e
        from .assets import resolve_torch_model
        from .inference_torch import TorchHamer
        return TorchHamer(resolve_torch_model(model_dir), device=device, dtype=dtype,
                          input_size=input_size), backend
    raise ValueError(f"unknown backend '{backend}' (use 'auto', 'coreml' or 'torch')")


class HandMesh:
    """Realtime 3D hand mesh recovery (HaMeR on the Apple Neural Engine, or on
    PyTorch/CUDA elsewhere).

    Give it an RGB image; get MANO parameters, 3D joints/vertices, camera
    params, and 2D joints per hand — MediaPipe-Hands style.

        import fasthamer
        hands = fasthamer.load(mode="video")
        result = hands(rgb_frame)
        for hand in result:
            hand.keypoints_2d       # (21, 2) pixels
            hand.keypoints_3d       # (21, 3) meters, hand-centered
            hand.hand_pose          # (15, 3, 3) MANO pose rotation matrices
            hand.betas              # (10,) MANO shape
        overlay = hands.render(rgb_frame, result)

    Args:
        mode: "image" for independent stills, "video" for tracked sequential
            frames (faster + temporally smoother detection).
        max_hands: maximum number of hands to detect.
        backend: inference engine — "auto" (CoreML on macOS, PyTorch elsewhere),
            "coreml" (Apple Neural Engine) or "torch" (PyTorch: CUDA, CPU, MPS;
            needs `pip install torch` and a torch bundle, see
            `fasthamer-convert-torch`).
        detector: which detection stack — "auto" (fasthands on macOS, mediapipe
            elsewhere), "fasthands" (CoreML/ANE) or "mediapipe" (Google's
            MediaPipe Tasks API; requires the fasthamer[mediapipe] extra).
        fasthands_detector: which detector model *inside* fasthands (needs
            fasthands >= 0.4.0): "whim" — the WHIM-fine-tuned full-hand-box
            detector, steadier than the palm detector; "mediapipe" — the
            original palm detector. None uses fasthands' own default. Only
            valid with detector="fasthands". Note the two "mediapipe" values
            mean different things: `detector="mediapipe"` is the separate
            TFLite Tasks stack, `fasthands_detector="mediapipe"` is the palm
            detector running on the ANE inside fasthands.
        model_dir: directory holding the model bundle; defaults to the
            fasthamer cache (CoreML: auto-downloaded on first use; torch: built
            with `fasthamer-convert-torch`). For the torch backend this may
            also be a hamer_torch.pt file or the official hamer.ckpt.
        rescale_factor: hand-box padding before cropping (HaMeR default 2.0).
        swap_handedness: flip left/right labels (use for mirrored inputs
            where handedness looks inverted).
        stabilize_handedness: video mode only — lock each hand's Right/Left
            label across frames via IoU tracking, so a spatially continuous
            hand keeps the label it first appeared with instead of flickering.
            The locked label drives the crop mirroring and geometry
            un-mirroring, not just the reported `is_right`. No-op in image mode.
        handedness_iou: min IoU for a detection to be considered the same hand
            as an existing track (stabilizer).
        handedness_ttl: drop a track after this many consecutive unseen frames
            (stabilizer), so a genuinely new hand can re-acquire.
        handedness_switch_frames: a tracked hand only switches its locked label
            once the detector disagrees for this many *consecutive* frames, so
            isolated flicker is rejected but a sustained correction still wins.
            0 never switches (a hard lock for the life of the track).
        force_handedness: "right" or "left" — override every detection's
            handedness with a constant. For rigs where handedness is known and
            fixed (e.g. single-hand egocentric/wrist cameras). Takes precedence
            over the stabilizer.
        compute_units: CoreML compute units, e.g. "CPU_AND_NE" (default),
            "ALL", "CPU_ONLY". CoreML backend only.
        device: torch backend only — "auto" (CUDA if available), "cuda",
            "cuda:1", "cpu", "mps".
        dtype: torch backend only — network precision: "auto" (float16 on
            CUDA GPUs with tensor cores, float32 elsewhere), "float16",
            "bfloat16" or "float32".
        input_size: torch backend only — (H, W) backbone input; default
            (256, 192) is the reference model, smaller runs faster.
    """

    def __init__(self, mode: str = "image", max_hands: int = 2,
                 backend: str = "auto",
                 detector: str = "auto",
                 fasthands_detector: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 rescale_factor: float = 2.0, swap_handedness: bool = False,
                 stabilize_handedness: bool = False,
                 handedness_iou: float = 0.3, handedness_ttl: int = 10,
                 handedness_switch_frames: int = 5,
                 force_handedness: Optional[str] = None,
                 compute_units: str = "CPU_AND_NE",
                 device: str = "auto", dtype: str = "auto",
                 input_size: Optional[tuple] = None,
                 **detector_kwargs):
        if mode not in ("image", "video"):
            raise ValueError(f"mode must be 'image' or 'video', got '{mode}'")
        if force_handedness not in (None, "right", "left"):
            raise ValueError("force_handedness must be 'right', 'left', or None, "
                             f"got '{force_handedness}'")
        self.mode = mode
        self.rescale_factor = float(rescale_factor)
        self.swap_handedness = bool(swap_handedness)
        self.force_handedness = force_handedness
        self.engine, self.backend = _make_engine(backend, model_dir, compute_units,
                                                 device, dtype, input_size)
        self.detector = make_detector(detector, max_hands, video=(mode == "video"),
                                      fasthands_detector=fasthands_detector,
                                      compute_units=compute_units, **detector_kwargs)
        self._stabilizer = (
            HandednessStabilizer(handedness_iou, handedness_ttl,
                                 handedness_switch_frames)
            if (mode == "video" and stabilize_handedness) else None)
        self._t0 = time.monotonic()
        self._last_ts = -1
        self._renderer: Optional[MeshRenderer] = None
        self._renderer_opts = {}

    @property
    def faces(self) -> np.ndarray:
        """(1538, 3) MANO triangle faces (right hand; flip winding for left)."""
        return self.engine.faces

    def reset(self) -> None:
        """Reset temporal state for a new video sequence (clears the
        handedness-stabilizer tracks)."""
        if self._stabilizer is not None:
            self._stabilizer.reset()

    def _new_result(self, image_rgb: np.ndarray) -> HandMeshResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got {image_rgb.shape}")
        H, W = image_rgb.shape[:2]
        img_size = np.array([W, H], dtype=np.float64)
        return HandMeshResult(hands=[], focal=scaled_focal_length(img_size),
                              cx=W / 2.0, cy=H / 2.0, image_size=(W, H))

    def _predict(self, crops):
        if hasattr(self.engine, "predict_batch"):
            return self.engine.predict_batch(crops)
        return [self.engine.predict(c) for c in crops]

    def reconstruct(self, image_rgb: np.ndarray, boxes: Sequence[np.ndarray],
                    is_right: Optional[Sequence[int]] = None) -> HandMeshResult:
        """Reconstruct hands from your own detections, skipping the detector.

        boxes: (N, 4) xyxy pixel boxes (e.g. the min/max of MediaPipe landmarks).
        is_right: N handedness flags (1 right, 0 left); defaults to
            `force_handedness` if set, else right.
        """
        result = self._new_result(image_rgb)
        boxes = [np.asarray(b, dtype=np.float32).reshape(4) for b in boxes]
        if not boxes:
            return result
        if self.force_handedness is not None:
            is_right = [int(self.force_handedness == "right")] * len(boxes)
        elif is_right is None:
            is_right = [1] * len(boxes)
        else:
            is_right = [int(r) for r in is_right]
        crops = preprocess_hands(image_rgb, np.stack(boxes), np.array(is_right),
                                 rescale_factor=self.rescale_factor)
        for box, crop, pred in zip(boxes, crops, self._predict(crops)):
            result.hands.append(self._make_hand(result, box, crop, pred))
        return result

    @staticmethod
    def _make_hand(result: HandMeshResult, box, crop, pred) -> Hand:
        s = crop.right  # 1.0 right, 0.0 left (left crops were mirrored)
        sign = 2.0 * s - 1.0
        cam = np.array(pred["cam"], dtype=np.float64)
        cam[1] = sign * cam[1]
        cam_t = cam_crop_to_full(cam, crop.box_center, crop.box_size,
                                 crop.img_size, result.focal)
        verts = np.array(pred["vertices"], dtype=np.float64)
        verts[:, 0] = sign * verts[:, 0]
        kp3d = np.array(pred["keypoints3d"], dtype=np.float64)
        kp3d[:, 0] = sign * kp3d[:, 0]
        go, hp = pred.get("global_orient"), pred.get("hand_pose")
        return Hand(
            is_right=bool(s),
            bbox=np.asarray(box, dtype=np.float32),
            vertices=verts,
            keypoints_3d=kp3d,
            cam_t=cam_t,
            keypoints_2d=result.project(kp3d + cam_t),
            global_orient=None if go is None else np.asarray(go).reshape(3, 3),
            hand_pose=None if hp is None else np.asarray(hp).reshape(15, 3, 3),
            betas=None if pred.get("betas") is None else np.asarray(pred["betas"]).reshape(-1),
        )

    def reconstruct_many(self, images_rgb: Sequence[np.ndarray], boxes_per_image: Sequence[Sequence[np.ndarray]],
                         is_right_per_image: Optional[Sequence[Sequence[int]]] = None,
                         batch_size: int = 64) -> list:
        """Offline batch version of `reconstruct`: all hands of all images go through the network
        in batches of `batch_size` (10-20x faster than per-image calls on a GPU). Returns one
        HandMeshResult per image. Images may differ in size."""
        crops, owners, results = [], [], []
        for i, (img, boxes) in enumerate(zip(images_rgb, boxes_per_image)):
            res = self._new_result(img)
            results.append(res)
            boxes = [np.asarray(b, dtype=np.float32).reshape(4) for b in boxes]
            if not boxes:
                continue
            if self.force_handedness is not None:
                right = [int(self.force_handedness == "right")] * len(boxes)
            elif is_right_per_image is None:
                right = [1] * len(boxes)
            else:
                right = [int(r) for r in is_right_per_image[i]]
            for box, crop in zip(boxes, preprocess_hands(img, np.stack(boxes), np.array(right), rescale_factor=self.rescale_factor)):
                crops.append(crop)
                owners.append((i, box))
        if not crops:
            return results
        if hasattr(self.engine, "predict_batch"):
            preds = self.engine.predict_batch(crops, batch_size=batch_size)
        else:
            preds = [self.engine.predict(c) for c in crops]
        for (i, box), crop, pred in zip(owners, crops, preds):
            results[i].hands.append(self._make_hand(results[i], box, crop, pred))
        return results

    def process(self, image_rgb: np.ndarray,
                timestamp_ms: Optional[int] = None,
                boxes: Optional[Sequence[np.ndarray]] = None,
                is_right: Optional[Sequence[int]] = None) -> HandMeshResult:
        """Detect hands in an RGB uint8 image and reconstruct each in 3D.

        In video mode, pass a monotonically increasing `timestamp_ms` if you
        have real frame timestamps; otherwise wall-clock time is used.

        Pass `boxes` (and optionally `is_right`) to use your own hand
        detections instead of the built-in detector (see `reconstruct`).
        """
        if boxes is not None:
            return self.reconstruct(image_rgb, boxes, is_right)
        result = self._new_result(image_rgb)

        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        boxes, is_right = self.detector.detect(image_rgb, timestamp_ms,
                                               swap_handedness=self.swap_handedness)
        if not boxes:
            return result

        # Handedness overrides run BEFORE preprocessing: the label drives the
        # crop mirroring and geometry un-mirroring, so fixing it later would
        # leave the mesh mirror-wrong on relabeled frames.
        if self.force_handedness is not None:
            is_right = [int(self.force_handedness == "right")] * len(boxes)
        elif self._stabilizer is not None:
            is_right = self._stabilizer(boxes, is_right)
        # force_handedness is re-applied inside reconstruct (idempotent)
        return self.reconstruct(image_rgb, boxes, is_right)

    __call__ = process

    def render(self, image_rgb: np.ndarray, result: HandMeshResult,
               color=MESH_COLOR, ambient: float = 0.5, alpha: float = 1.0,
               antialias: int = 2) -> np.ndarray:
        """Overlay the reconstructed meshes on an RGB uint8 image (returns a copy)."""
        opts = {"color": tuple(color), "ambient": ambient, "alpha": alpha,
                "antialias": antialias}
        if self._renderer is None or opts != self._renderer_opts:
            self._renderer = MeshRenderer(self.engine.faces, **opts)
            self._renderer_opts = opts
        return self._renderer.render(image_rgb, result)


def load(**kwargs) -> HandMesh:
    """Create a HandMesh tracker (see HandMesh for arguments)."""
    return HandMesh(**kwargs)
