"""CoreML inference for the full HaMeR model (ViT-H backbone + MANO head + MANO
mesh in one mlpackage) on the Apple Neural Engine. Pure numpy I/O — no torch."""
import hashlib
import os
import shutil
import sys
from typing import Dict, List

import cv2
import numpy as np

from .assets import FACES_NAME, MODEL_NAME, cache_dir
from .preprocess import HandCrop, IMAGE_SIZE


def _compiled_cache_path(mlpackage_path: str) -> str:
    """Stable, per-mlpackage location for the compiled .mlmodelc."""
    key = hashlib.sha1(os.path.abspath(mlpackage_path).encode()).hexdigest()[:12]
    return os.path.join(cache_dir(), "compiled", f"hamer_mano_{key}.mlmodelc")


def _load_prediction_model(mlpackage_path: str, units):
    """Load the CoreML model for prediction, compiling the mlpackage to a
    persistent .mlmodelc on first use and reusing it thereafter.

    coremltools' `MLModel(mlpackage)` recompiles on every load — it compiles to
    a fresh temp directory each time, and Core ML keys its on-disk compile cache
    by path, so the cache always misses (~15 s per load). We instead compile
    once to a stable path and load a `CompiledMLModel` from there; the OS also
    caches the Neural Engine compilation against that path, so every later
    process loads in ~0.1 s.
    """
    import coremltools as ct
    compiled = _compiled_cache_path(mlpackage_path)
    if os.path.isdir(compiled):
        try:
            return ct.models.CompiledMLModel(compiled, compute_units=units)
        except Exception:
            shutil.rmtree(compiled, ignore_errors=True)  # stale/corrupt; rebuild

    sys.stderr.write("[fasthamer] compiling the model for your device "
                     "(one-time, ~30 s; cached for next time)...\n")
    sys.stderr.flush()
    # Compile the mlpackage to a .mlmodelc once. CPU_ONLY keeps this step light
    # and avoids a wasted Neural Engine warm-up on the throwaway temp path — the
    # .mlmodelc is compute-unit-agnostic, so it still runs on the ANE below.
    src_model = ct.models.MLModel(mlpackage_path,
                                  compute_units=ct.ComputeUnit.CPU_ONLY)
    try:
        src = src_model.get_compiled_model_path()  # valid while src_model alive
        os.makedirs(os.path.dirname(compiled), exist_ok=True)
        tmp = compiled + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        os.replace(tmp, compiled)
    except Exception:
        # Caching failed (e.g. a read-only cache dir) — fall back to a normal
        # in-place load with the requested compute units (slow, but works).
        return ct.models.MLModel(mlpackage_path, compute_units=units)
    # Load from the persistent path so the OS caches the ANE compilation there.
    return ct.models.CompiledMLModel(compiled, compute_units=units)


class CoreMLHamer:
    def __init__(self, model_dir: str, compute_units: str = "CPU_AND_NE"):
        import coremltools as ct
        units = getattr(ct.ComputeUnit, compute_units)
        mlpackage = os.path.join(model_dir, MODEL_NAME)
        # Read the spec cheaply (no compile) for I/O metadata, then load the
        # prediction model via the compile-once cache.
        spec = ct.models.MLModel(mlpackage, skip_model_load=True).get_spec().description
        self.model = _load_prediction_model(mlpackage, units)
        self.faces = np.load(os.path.join(model_dir, FACES_NAME))
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
