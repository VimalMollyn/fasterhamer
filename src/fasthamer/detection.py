"""Hand detection front-ends. Both return (boxes, is_right) for an RGB frame:
boxes as a list of (4,) xyxy float32 arrays, is_right as a list of 0/1 ints.

- "fasthands": MediaPipe Hands ported to CoreML on the Apple Neural Engine
  (~1 ms/frame, numpy I/O, no torch). Default.
- "mediapipe": Google's MediaPipe Tasks HandLandmarker (install the
  `fasthamer[mediapipe]` extra). CPU/GPU, cross-check fallback.
"""
import os
import urllib.request
from typing import List, Tuple

import numpy as np

from .assets import cache_dir

_HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")


class FastHandsDetector:
    def __init__(self, max_hands: int = 2, video: bool = True,
                 compute_units: str = "CPU_AND_NE"):
        import fasthands
        self.tracker = fasthands.load(num_hands=max_hands, compute_units=compute_units)
        self.video = video

    def detect(self, rgb: np.ndarray, timestamp_ms: int = 0,
               swap_handedness: bool = False) -> Tuple[List[np.ndarray], List[int]]:
        hands = self.tracker.detect_video(rgb) if self.video else self.tracker(rgb)
        H, W = rgb.shape[:2]
        boxes, is_right = [], []
        for h in hands:
            lm = h["landmarks"]  # (21, 3), x/y normalized to [0, 1]
            xs, ys = lm[:, 0] * W, lm[:, 1] * H
            boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32))
            right = (h["handedness"] == "Right") ^ swap_handedness
            is_right.append(int(right))
        return boxes, is_right


class MediaPipeDetector:
    def __init__(self, max_hands: int = 2, video: bool = True,
                 det_conf: float = 0.5, track_conf: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError("the mediapipe detector needs the optional extra: "
                              "pip install 'fasthamer[mediapipe]'") from e
        self._mp = mp
        task_path = os.path.join(cache_dir(), "hand_landmarker.task")
        if not os.path.exists(task_path):
            os.makedirs(os.path.dirname(task_path), exist_ok=True)
            urllib.request.urlretrieve(_HAND_TASK_URL, task_path)
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=task_path),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.video = video

    def detect(self, rgb: np.ndarray, timestamp_ms: int = 0,
               swap_handedness: bool = False) -> Tuple[List[np.ndarray], List[int]]:
        mp = self._mp
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        if self.video:
            res = self.landmarker.detect_for_video(mpimg, timestamp_ms)
        else:
            res = self.landmarker.detect(mpimg)
        H, W = rgb.shape[:2]
        boxes, is_right = [], []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            xs = np.array([lm.x for lm in lms]) * W
            ys = np.array([lm.y for lm in lms]) * H
            boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32))
            right = (hd[0].category_name == "Right") ^ swap_handedness
            is_right.append(int(right))
        return boxes, is_right


def make_detector(kind: str, max_hands: int, video: bool, **kwargs):
    if kind == "fasthands":
        return FastHandsDetector(max_hands, video,
                                 compute_units=kwargs.get("compute_units", "CPU_AND_NE"))
    if kind == "mediapipe":
        return MediaPipeDetector(max_hands, video,
                                 det_conf=kwargs.get("det_conf", 0.5),
                                 track_conf=kwargs.get("track_conf", 0.5))
    raise ValueError(f"unknown detector '{kind}' (use 'fasthands' or 'mediapipe')")
