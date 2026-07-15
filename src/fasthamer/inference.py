"""CoreML inference for the full HaMeR model (ViT-H backbone + MANO head + MANO
mesh in one mlpackage) on the Apple Neural Engine. Pure numpy I/O — no torch."""
import os
from typing import Dict, List

import cv2
import numpy as np

from .assets import FACES_NAME, MODEL_NAME
from .preprocess import HandCrop, IMAGE_SIZE


class CoreMLHamer:
    def __init__(self, model_dir: str, compute_units: str = "CPU_AND_NE"):
        import coremltools as ct
        units = getattr(ct.ComputeUnit, compute_units)
        self.model = ct.models.MLModel(os.path.join(model_dir, MODEL_NAME),
                                       compute_units=units)
        self.faces = np.load(os.path.join(model_dir, FACES_NAME))
        spec = self.model.get_spec().description
        shape = spec.input[0].type.multiArrayType.shape
        self.in_h, self.in_w = int(shape[-2]), int(shape[-1])
        self.full_input = (self.in_h == IMAGE_SIZE and self.in_w == IMAGE_SIZE)
        self.output_names = {o.name for o in spec.output}
        self.has_mano_params = {"global_orient", "hand_pose", "betas"} <= self.output_names

    def _prep(self, img_chw: np.ndarray) -> np.ndarray:
        """(3, 256, 256) normalized crop -> model input (1, 3, in_h, in_w)."""
        if self.full_input:
            return img_chw[None].astype(np.float32)
        # Low-res variant: slice the 256x192 center the backbone would see,
        # then resize to the model's input resolution.
        sl = img_chw[:, :, 32:-32]
        hwc = np.transpose(sl, (1, 2, 0))
        res = cv2.resize(hwc, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        return np.transpose(res, (2, 0, 1))[None].astype(np.float32)

    def predict(self, crop: HandCrop) -> Dict[str, np.ndarray]:
        out = self.model.predict({"image": self._prep(crop.img)})
        pred = {
            "vertices": np.asarray(out["vertices"][0], dtype=np.float64),
            "keypoints3d": np.asarray(out["keypoints3d"][0], dtype=np.float64),
            "cam": np.asarray(out["cam"][0], dtype=np.float64),
        }
        if self.has_mano_params:
            pred["global_orient"] = np.asarray(out["global_orient"][0], dtype=np.float64)
            pred["hand_pose"] = np.asarray(out["hand_pose"][0], dtype=np.float64)
            pred["betas"] = np.asarray(out["betas"][0], dtype=np.float64)
        return pred
