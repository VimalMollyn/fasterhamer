"""Result types returned by fasthamer.HandMesh."""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# MANO/OpenPose-style 21-keypoint hand skeleton (HaMeR joint order):
# 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _rotmats_to_axis_angle(rotmats: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle via cv2.Rodrigues."""
    flat = rotmats.reshape(-1, 3, 3)
    out = np.stack([cv2.Rodrigues(R.astype(np.float64))[0].ravel() for R in flat])
    return out.reshape(rotmats.shape[:-2] + (3,))


@dataclass
class Hand:
    """A single reconstructed hand.

    3D quantities live in a right-handed camera-style frame (x right, y down,
    z forward) — `vertices`/`keypoints_3d` are hand-centered; add `cam_t` (or use
    the `*_camera` properties) to place them in the camera frame of the full image.

    MANO parameters always describe a *right* MANO hand: left hands are mirrored
    through the crop pipeline, so to pose a left hand mesh apply the params to a
    right-hand MANO model and negate x (exactly what `vertices` already has done).
    """
    is_right: bool
    bbox: np.ndarray                       # (4,) detector box, xyxy pixels
    vertices: np.ndarray                   # (778, 3) hand-centered mesh, meters
    keypoints_3d: np.ndarray               # (21, 3) hand-centered joints, meters
    cam_t: np.ndarray                      # (3,) translation hand -> camera frame
    keypoints_2d: np.ndarray               # (21, 2) pixels in the input image
    global_orient: Optional[np.ndarray] = None  # (3, 3) rotation matrix
    hand_pose: Optional[np.ndarray] = None      # (15, 3, 3) rotation matrices
    betas: Optional[np.ndarray] = None          # (10,) MANO shape

    @property
    def vertices_camera(self) -> np.ndarray:
        """(778, 3) mesh vertices in the camera frame."""
        return self.vertices + self.cam_t

    @property
    def keypoints_camera(self) -> np.ndarray:
        """(21, 3) joints in the camera frame."""
        return self.keypoints_3d + self.cam_t

    @property
    def global_orient_aa(self) -> Optional[np.ndarray]:
        """(3,) global orientation as axis-angle."""
        return None if self.global_orient is None else _rotmats_to_axis_angle(self.global_orient)

    @property
    def hand_pose_aa(self) -> Optional[np.ndarray]:
        """(15, 3) MANO hand pose as axis-angle (smplx `hand_pose` convention)."""
        return None if self.hand_pose is None else _rotmats_to_axis_angle(self.hand_pose)


@dataclass
class HandMeshResult:
    """All hands found in one image, plus the pinhole camera that maps the 3D
    outputs onto the image: u = fx * X/Z + cx, v = fy * Y/Z + cy."""
    hands: List[Hand] = field(default_factory=list)
    focal: float = 0.0            # pixels (fx == fy)
    cx: float = 0.0               # principal point, pixels
    cy: float = 0.0
    image_size: tuple = (0, 0)    # (width, height)

    def __len__(self) -> int:
        return len(self.hands)

    def __iter__(self):
        return iter(self.hands)

    @property
    def multi_hand_world_landmarks(self) -> List[np.ndarray]:
        """MediaPipe-style alias: per-hand (21, 3) camera-frame joints."""
        return [h.keypoints_camera for h in self.hands]

    @property
    def multi_hand_landmarks(self) -> List[np.ndarray]:
        """MediaPipe-style alias: per-hand (21, 2) pixel joints."""
        return [h.keypoints_2d for h in self.hands]

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project (N, 3) camera-frame points to (N, 2) pixels."""
        z = np.clip(points_3d[:, 2], 1e-5, None)
        u = self.focal * points_3d[:, 0] / z + self.cx
        v = self.focal * points_3d[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)
