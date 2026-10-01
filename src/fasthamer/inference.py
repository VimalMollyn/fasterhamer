"""CoreML inference for HaMeR (ViT-H backbone + MANO head in one mlpackage) on
the Apple Neural Engine, followed by the MANO mesh in numpy from the user's own
MANO model. Pure numpy I/O — no torch.

Legacy bundles that have the MANO mesh baked into the mlpackage (outputs
`vertices` / `keypoints3d`) are still supported and need no MANO file."""
import hashlib
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .assets import FACES_NAME, bundle_model_path, cache_dir
from .mano import ManoRight, load_mano_npz
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
                     "(one-time, typically 10-30 s; cached for next time)...\n")
    sys.stderr.flush()
    # Compile the mlpackage straight to a .mlmodelc with `compile_model` (a pure
    # spec -> mlmodelc step, ~0.1 s). Do NOT do this by instantiating
    # `MLModel(mlpackage, compute_units=CPU_ONLY)` and copying its compiled
    # path: constructing an MLModel also *loads* the program for that compute
    # unit, and a CPU_ONLY load makes Core ML lower the entire ViT-H graph to
    # the CPU (BNNS) backend -- many minutes on macOS 15+ -- for a model that
    # never runs there. The .mlmodelc is compute-unit-agnostic, so the ANE
    # compilation happens (once, OS-cached) in the CompiledMLModel load below.
    stem = compiled[:-len(".mlmodelc")]
    tmp = stem + "_tmp.mlmodelc"  # compile_model requires a .mlmodelc suffix
    try:
        os.makedirs(os.path.dirname(compiled), exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        ct.utils.compile_model(mlpackage_path, destination_path=tmp)
        os.replace(tmp, compiled)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isdir(compiled):  # another process compiled it meanwhile
            try:
                return ct.models.CompiledMLModel(compiled, compute_units=units)
            except Exception:
                pass
        # Caching failed (e.g. a read-only cache dir) -- fall back to a normal
        # in-place load with the requested compute units (slow, but works).
        return ct.models.MLModel(mlpackage_path, compute_units=units)
    # Load from the persistent path so the OS caches the ANE compilation there.
    # This load is where the (one-time) Neural Engine compilation happens.
    t0 = time.time()
    model = ct.models.CompiledMLModel(compiled, compute_units=units)
    sys.stderr.write(f"[fasthamer] model ready ({time.time() - t0:.0f} s); "
                     "later loads will be fast.\n")
    sys.stderr.flush()
    return model


class CoreMLHamer:
    """image crop -> MANO params (CoreML) -> vertices + 21 joints (numpy MANO).

    `mano_npz` is the user's MANO model as cached by `fasthamer.assets`
    (required unless the bundle is a legacy one with the mesh baked in)."""

    def __init__(self, model_dir: str, compute_units: str = "CPU_AND_NE",
                 mano_npz: Optional[str] = None):
        import coremltools as ct
        units = getattr(ct.ComputeUnit, compute_units)
        mlpackage, legacy = bundle_model_path(model_dir)
        # Read the spec cheaply (no compile) for I/O metadata, then load the
        # prediction model via the compile-once cache.
        spec = ct.models.MLModel(mlpackage, skip_model_load=True).get_spec().description
        self.model = _load_prediction_model(mlpackage, units)
        shape = spec.input[0].type.multiArrayType.shape
        self.in_h, self.in_w = int(shape[-2]), int(shape[-1])
        self.full_input = (self.in_h == IMAGE_SIZE and self.in_w == IMAGE_SIZE)
        self.output_names = {o.name for o in spec.output}
        self.has_mano_params = {"global_orient", "hand_pose", "betas"} <= self.output_names
        self.has_mesh = {"vertices", "keypoints3d"} <= self.output_names

        self.mano: Optional[ManoRight] = None
        if self.has_mesh:  # legacy fused bundle: mesh comes out of CoreML
            self.faces = np.load(os.path.join(model_dir, FACES_NAME)) if legacy \
                else ManoRight(load_mano_npz(mano_npz)).faces
        else:
            if not self.has_mano_params:
                raise RuntimeError(f"{mlpackage} outputs neither a mesh nor MANO "
                                   f"parameters (outputs: {sorted(self.output_names)})")
            if mano_npz is None:
                raise ValueError("this model bundle has no MANO mesh layer; a MANO "
                                 "model is required (fasthamer.assets.resolve_mano)")
            self.mano = ManoRight(load_mano_npz(mano_npz))
            self.faces = self.mano.faces

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
        pred = {"cam": np.asarray(out["cam"][0], dtype=np.float64)}
        if self.has_mano_params:
            pred["global_orient"] = np.asarray(out["global_orient"][0], dtype=np.float64)
            pred["hand_pose"] = np.asarray(out["hand_pose"][0], dtype=np.float64)
            pred["betas"] = np.asarray(out["betas"][0], dtype=np.float64)
        if self.mano is not None:
            pred["vertices"], pred["keypoints3d"] = self.mano(
                pred["global_orient"], pred["hand_pose"], pred["betas"])
        else:
            pred["vertices"] = np.asarray(out["vertices"][0], dtype=np.float64)
            pred["keypoints3d"] = np.asarray(out["keypoints3d"][0], dtype=np.float64)
        return pred
