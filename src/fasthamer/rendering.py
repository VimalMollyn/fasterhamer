"""Mesh overlay + skeleton drawing.

The mesh renderer is a tiny C rasterizer (fast_render.c, ~10x faster than
pyrender for this workload, CPU-only so it never competes with the ANE). It is
compiled once on first use with the system C compiler into the fasthamer cache
directory, then loaded via ctypes.
"""
import ctypes
import hashlib
import os
import subprocess
from typing import Optional

import cv2
import numpy as np

from .assets import cache_dir
from .types import HAND_EDGES, Hand, HandMeshResult

_C_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_render.c")

# Extra faces that close the MANO wrist opening (same as hamer.utils.renderer).
_FACES_NEW = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239],
                       [239, 122, 279], [122, 118, 279], [279, 118, 215],
                       [118, 117, 215], [215, 117, 214], [117, 119, 214],
                       [214, 119, 121], [119, 120, 121], [121, 120, 78],
                       [120, 108, 78], [78, 108, 79]], dtype=np.int32)

# Azure / steel-blue default (matches the SMPL/MANO viewer look).
MESH_COLOR = (101 / 255.0, 168 / 255.0, 197 / 255.0)

_FINGER_COLORS_RGB = [
    (255, 255, 255),  # wrist
    (255, 0, 0),      # thumb
    (0, 255, 0),      # index
    (0, 0, 255),      # middle
    (255, 255, 0),    # ring
    (255, 0, 255),    # pinky
]

_lib = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with open(_C_SRC, "rb") as f:
        tag = hashlib.sha256(f.read()).hexdigest()[:16]
    so_path = os.path.join(cache_dir(), f"fast_render_{tag}.so")
    if not os.path.exists(so_path):
        os.makedirs(os.path.dirname(so_path), exist_ok=True)
        cmd = ["cc", "-O3", "-ffast-math", "-shared", "-fPIC", _C_SRC, "-o", so_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(
                "fasthamer could not compile its C renderer. Install the Xcode "
                "command line tools (`xcode-select --install`) and retry. "
                f"Command: {' '.join(cmd)}"
            ) from e
    lib = ctypes.CDLL(so_path)
    lib.rasterize_mesh.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int,
    ]
    lib.rasterize_mesh.restype = None
    _lib = lib
    return lib


class MeshRenderer:
    """Renders MANO meshes over an image with a pinhole camera (u = f*x/z + cx)."""

    def __init__(self, faces: np.ndarray, color=MESH_COLOR, ambient: float = 0.5,
                 alpha: float = 1.0, antialias: int = 2):
        faces = np.asarray(faces, dtype=np.int32)
        self.faces = np.ascontiguousarray(np.concatenate([faces, _FACES_NEW]), dtype=np.int32)
        self.faces_left = np.ascontiguousarray(self.faces[:, [0, 2, 1]], dtype=np.int32)
        self.color = tuple(float(c) for c in color)
        self.ambient = float(ambient)
        self.alpha = float(alpha)
        self.antialias = int(antialias)
        self._lib = _load_lib()

    def render(self, image_rgb: np.ndarray, result: HandMeshResult) -> np.ndarray:
        """Composite all hand meshes onto a copy of `image_rgb` (uint8, H×W×3)."""
        img = np.ascontiguousarray(image_rgb, dtype=np.uint8).copy()
        if not result.hands:
            return img
        H, W = img.shape[:2]
        Vs, Fs, off = [], [], 0
        for h in result.hands:
            Vs.append(h.vertices_camera.astype(np.float32))
            Fs.append((self.faces if h.is_right else self.faces_left) + off)
            off += len(Vs[-1])
        V = np.ascontiguousarray(np.concatenate(Vs), dtype=np.float32)
        F = np.ascontiguousarray(np.concatenate(Fs), dtype=np.int32)
        self._lib.rasterize_mesh(
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(V),
            F.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(F),
            result.focal, result.focal, result.cx, result.cy,
            W, H,
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self.color[0], self.color[1], self.color[2],
            self.ambient, self.alpha, self.antialias)
        return img


def draw_landmarks(image: np.ndarray, result: HandMeshResult,
                   bgr: bool = False) -> np.ndarray:
    """Draw the 21-joint skeleton for every hand on a copy of `image` (uint8).
    Set bgr=True if the image is BGR (e.g. straight from cv2) so finger colors
    stay correct."""
    img = np.ascontiguousarray(image, dtype=np.uint8).copy()
    for hand in result.hands:
        kp2d = hand.keypoints_2d
        for a, b in HAND_EDGES:
            col = _color_for_joint(a, bgr)
            cv2.line(img, tuple(np.round(kp2d[a]).astype(int)),
                     tuple(np.round(kp2d[b]).astype(int)), col, 2, cv2.LINE_AA)
        for j in range(kp2d.shape[0]):
            cv2.circle(img, tuple(np.round(kp2d[j]).astype(int)), 3,
                       _color_for_joint(j, bgr), -1, cv2.LINE_AA)
    return img


def _color_for_joint(j: int, bgr: bool):
    col = _FINGER_COLORS_RGB[(j - 1) // 4 + 1] if j > 0 else _FINGER_COLORS_RGB[0]
    return col[::-1] if bgr else col
