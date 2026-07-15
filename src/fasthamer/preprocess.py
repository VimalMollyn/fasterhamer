"""Hand-crop preprocessing for HaMeR, vendored from hamer.datasets (numpy/cv2 only,
no torch). Produces the normalized (3, 256, 256) crop plus the box metadata needed
to lift the crop-camera prediction back to the full image.
"""
from typing import List, Tuple

import cv2
import numpy as np

# HaMeR model constants (from the released model_config.yaml — fixed for the
# public checkpoint, so no yacs/config machinery is needed at runtime).
IMAGE_SIZE = 256
BBOX_SHAPE = (192, 256)  # (w, h) aspect target for the ViT backbone
FOCAL_LENGTH = 5000.0    # crop-space focal length used in training
IMAGE_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
IMAGE_STD = 255.0 * np.array([0.229, 0.224, 0.225])


def expand_to_aspect_ratio(input_shape, target_aspect_ratio) -> np.ndarray:
    """Grow (w, h) minimally so it matches the target aspect ratio."""
    w, h = input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)])
    return np.array([max(h * w_t / h_t, w), h])


def _rotate_2d(pt: np.ndarray, rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height,
                            dst_width, dst_height, scale, rot) -> np.ndarray:
    """Affine transform mapping the (rotated, scaled) source box to the output patch."""
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = _rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.stack([src_center, src_center + src_downdir, src_center + src_rightdir])
    dst = np.stack([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def generate_image_patch_cv2(img, c_x, c_y, bb_width, bb_height,
                             patch_width, patch_height, do_flip) -> np.ndarray:
    img_width = img.shape[1]
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height,
                                    patch_width, patch_height, 1.0, 0.0)
    return cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _antialias_blur(img: np.ndarray, bbox_size: float, patch_width: float) -> np.ndarray:
    """Gaussian pre-blur before heavy downsampling, matching hamer's
    skimage.filters.gaussian(..., mode='nearest', truncate=4) via cv2."""
    downsampling_factor = (bbox_size / patch_width) / 2.0
    if downsampling_factor <= 1.1:
        return img
    sigma = (downsampling_factor - 1) / 2
    img = img.astype(np.float32)
    radius = int(4.0 * sigma + 0.5)
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REPLICATE)


class HandCrop:
    """One preprocessed hand: the model input crop + box metadata."""

    __slots__ = ("img", "box_center", "box_size", "img_size", "right")

    def __init__(self, img, box_center, box_size, img_size, right):
        self.img = img                # (3, 256, 256) float32, normalized, RGB
        self.box_center = box_center  # (2,) pixels
        self.box_size = box_size      # float, pixels (square box side)
        self.img_size = img_size      # (2,) [W, H] of the full image
        self.right = right            # 1.0 right hand, 0.0 left (crop is mirrored)


def preprocess_hands(img_rgb: np.ndarray, boxes: np.ndarray, right: np.ndarray,
                     rescale_factor: float = 2.0) -> List[HandCrop]:
    """Crop each detected hand box out of an RGB image, HaMeR-style.

    Left-hand crops are mirrored so the model always sees a right hand
    (predictions are un-mirrored downstream).
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    scales = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    img_size = 1.0 * np.array([img_rgb.shape[1], img_rgb.shape[0]])

    crops = []
    for center, scale, r in zip(centers, scales, right):
        bbox_size = float(expand_to_aspect_ratio(scale * 200, BBOX_SHAPE).max())
        cvimg = _antialias_blur(img_rgb, bbox_size, IMAGE_SIZE)
        patch = generate_image_patch_cv2(cvimg, center[0], center[1],
                                         bbox_size, bbox_size,
                                         IMAGE_SIZE, IMAGE_SIZE, do_flip=(r == 0))
        chw = np.transpose(patch, (2, 0, 1)).astype(np.float32)
        # float64 math, float32 storage — matches hamer's in-place normalization
        chw = ((chw - IMAGE_MEAN[:, None, None]) / IMAGE_STD[:, None, None]).astype(np.float32)
        crops.append(HandCrop(chw, center.copy(), bbox_size, img_size, float(r)))
    return crops


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size,
                     focal_length) -> np.ndarray:
    """Convert the crop-space weak-perspective camera (s, tx, ty) to a full-image
    camera translation (numpy port of hamer.utils.renderer.cam_crop_to_full)."""
    img_w, img_h = float(img_size[0]), float(img_size[1])
    cx, cy = float(box_center[0]), float(box_center[1])
    b = float(box_size)
    bs = b * cam_bbox[0] + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - img_w / 2.0) / bs) + cam_bbox[1]
    ty = (2.0 * (cy - img_h / 2.0) / bs) + cam_bbox[2]
    return np.array([tx, ty, tz], dtype=np.float64)


def scaled_focal_length(img_size) -> float:
    """Full-image focal length (pixels) HaMeR assumes for a given image size."""
    return FOCAL_LENGTH / IMAGE_SIZE * float(np.max(img_size))
