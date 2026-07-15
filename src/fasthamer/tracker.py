"""The main fasthamer API: HandMesh."""
import time
from typing import Optional

import numpy as np

from .assets import resolve_model_dir
from .detection import make_detector
from .inference import CoreMLHamer
from .preprocess import cam_crop_to_full, preprocess_hands, scaled_focal_length
from .rendering import MESH_COLOR, MeshRenderer
from .types import Hand, HandMeshResult


class HandMesh:
    """Realtime 3D hand mesh recovery (HaMeR on the Apple Neural Engine).

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
        detector: "fasthands" (CoreML/ANE, default) or "mediapipe"
            (requires the fasthamer[mediapipe] extra).
        model_dir: directory holding the model bundle; defaults to the
            fasthamer cache (auto-downloaded on first use).
        rescale_factor: hand-box padding before cropping (HaMeR default 2.0).
        swap_handedness: flip left/right labels (use for mirrored inputs
            where handedness looks inverted).
        compute_units: CoreML compute units, e.g. "CPU_AND_NE" (default),
            "ALL", "CPU_ONLY".
    """

    def __init__(self, mode: str = "image", max_hands: int = 2,
                 detector: str = "fasthands", model_dir: Optional[str] = None,
                 rescale_factor: float = 2.0, swap_handedness: bool = False,
                 compute_units: str = "CPU_AND_NE", **detector_kwargs):
        if mode not in ("image", "video"):
            raise ValueError(f"mode must be 'image' or 'video', got '{mode}'")
        self.mode = mode
        self.rescale_factor = float(rescale_factor)
        self.swap_handedness = bool(swap_handedness)
        bundle = resolve_model_dir(model_dir)
        self.engine = CoreMLHamer(bundle, compute_units=compute_units)
        self.detector = make_detector(detector, max_hands, video=(mode == "video"),
                                      compute_units=compute_units, **detector_kwargs)
        self._t0 = time.monotonic()
        self._last_ts = -1
        self._renderer: Optional[MeshRenderer] = None
        self._renderer_opts = {}

    @property
    def faces(self) -> np.ndarray:
        """(1538, 3) MANO triangle faces (right hand; flip winding for left)."""
        return self.engine.faces

    def process(self, image_rgb: np.ndarray,
                timestamp_ms: Optional[int] = None) -> HandMeshResult:
        """Detect hands in an RGB uint8 image and reconstruct each in 3D.

        In video mode, pass a monotonically increasing `timestamp_ms` if you
        have real frame timestamps; otherwise wall-clock time is used.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got {image_rgb.shape}")
        H, W = image_rgb.shape[:2]
        img_size = np.array([W, H], dtype=np.float64)
        focal = scaled_focal_length(img_size)
        result = HandMeshResult(hands=[], focal=focal, cx=W / 2.0, cy=H / 2.0,
                                image_size=(W, H))

        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        boxes, is_right = self.detector.detect(image_rgb, timestamp_ms,
                                               swap_handedness=self.swap_handedness)
        if not boxes:
            return result

        crops = preprocess_hands(image_rgb, np.stack(boxes), np.array(is_right),
                                 rescale_factor=self.rescale_factor)
        for box, crop in zip(boxes, crops):
            pred = self.engine.predict(crop)
            s = crop.right  # 1.0 right, 0.0 left (left crops were mirrored)
            sign = 2.0 * s - 1.0
            cam = pred["cam"]
            cam[1] = sign * cam[1]
            cam_t = cam_crop_to_full(cam, crop.box_center, crop.box_size,
                                     crop.img_size, focal)
            verts = pred["vertices"]
            verts[:, 0] = sign * verts[:, 0]
            kp3d = pred["keypoints3d"]
            kp3d[:, 0] = sign * kp3d[:, 0]
            go, hp = pred.get("global_orient"), pred.get("hand_pose")
            hand = Hand(
                is_right=bool(s),
                bbox=np.asarray(box, dtype=np.float32),
                vertices=verts,
                keypoints_3d=kp3d,
                cam_t=cam_t,
                keypoints_2d=result.project(kp3d + cam_t),
                global_orient=None if go is None else go.reshape(3, 3),
                hand_pose=None if hp is None else hp.reshape(15, 3, 3),
                betas=pred.get("betas"),
            )
            result.hands.append(hand)
        return result

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
