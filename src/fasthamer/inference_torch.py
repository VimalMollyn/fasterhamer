"""PyTorch inference engine for the full HaMeR model (Linux/Windows/macOS, CUDA or CPU).

Same interface as CoreMLHamer (`predict(crop)`, `faces`) plus `predict_batch(crops)`, which
runs all crops of a frame (or of many frames) through the network in one forward pass —
that is where a GPU pays off.
"""
from typing import Dict, List, Optional

import cv2
import numpy as np

from .preprocess import HandCrop, IMAGE_SIZE
from .torch_model import BACKBONE_INPUT, build_model


class TorchHamer:
    """HaMeR on PyTorch.

    Args:
        model_path: a fasthamer torch bundle directory (hamer_torch.pt + mano_faces.npy),
            the hamer_torch.pt file itself, or the official hamer.ckpt.
        device: "auto" (CUDA if available), "cuda", "cuda:1", "cpu", "mps", ...
        dtype: compute precision of the network, "auto" (float16 on CUDA GPUs with tensor
            cores, float32 elsewhere), "float16", "bfloat16" or "float32". MANO skinning
            always runs in float32.
        input_size: (H, W) fed to the backbone. Default (256, 192) = the reference model; a
            smaller size (e.g. (192, 144), what the CoreML bundle uses) runs faster with the
            position embeddings resampled, at a small accuracy cost.
    """

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "auto",
                 input_size: Optional[tuple] = None):
        import torch
        self.torch = torch
        self.model = build_model(model_path, device=device, dtype=dtype)
        self.device = next(self.model.parameters()).device
        self.dtype = self.model.compute_dtype
        self.in_h, self.in_w = tuple(input_size) if input_size else BACKBONE_INPUT
        self.faces = self.model.faces
        # keep_feats: also keep the backbone's token map of every crop of the last predict_batch() in
        # last_feats (list of (1280, 16, 12) float32, crop order), for heads that reuse the backbone
        self.keep_feats = False
        self.last_feats = None
        self.has_mano_params = True

    def _prep(self, img_chw: np.ndarray) -> np.ndarray:
        """(3, 256, 256) normalized crop -> (3, in_h, in_w): the backbone sees the central
        256x192 strip; resized only if a non-default input size was requested."""
        sl = img_chw[:, :, (IMAGE_SIZE - BACKBONE_INPUT[1]) // 2: -(IMAGE_SIZE - BACKBONE_INPUT[1]) // 2]
        if (self.in_h, self.in_w) == BACKBONE_INPUT:
            return np.ascontiguousarray(sl, dtype=np.float32)
        hwc = np.transpose(sl, (1, 2, 0))
        res = cv2.resize(hwc, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(np.transpose(res, (2, 0, 1)), dtype=np.float32)

    def predict_batch(self, crops: List[HandCrop], batch_size: int = 64) -> List[Dict[str, np.ndarray]]:
        torch = self.torch
        preds = []
        for s in range(0, len(crops), batch_size):
            x = np.stack([self._prep(c.img) for c in crops[s:s + batch_size]])
            x = torch.from_numpy(x).to(self.device, self.dtype)
            with torch.inference_mode():
                out = self.model(x, return_feats=self.keep_feats)
            if self.keep_feats:
                feats = out.pop("feats").float().cpu().numpy()
                self.last_feats = (self.last_feats or []) + list(feats) if s else list(feats)
            out = {k: v.float().cpu().numpy().astype(np.float64) for k, v in out.items()}
            for i in range(x.shape[0]):
                preds.append({"vertices": out["vertices"][i], "keypoints3d": out["keypoints3d"][i], "cam": out["cam"][i],
                              "global_orient": out["global_orient"][i], "hand_pose": out["hand_pose"][i], "betas": out["betas"][i]})
        return preds

    def predict(self, crop: HandCrop) -> Dict[str, np.ndarray]:
        return self.predict_batch([crop])[0]
