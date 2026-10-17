"""Optional SOMA-X hand overlay.

Turns HaMeR's MANO parameters into a SOMA-X hand mesh with the mano2soma-x
regressor (https://github.com/VimalMollyn/mano2soma-x) so the two meshes can
be compared on the live image. Two engines:

  * `.npz` asset file (mano2soma-x `scripts/export_numpy_assets.py`): the
    torch-free numpy port in `fasthamer.soma_np`, optionally with the
    regressor MLP as a CoreML model on the Neural Engine
    (`regressor=` path to `mano2somax_regressor_ane.mlpackage`, "auto" picks
    one next to the npz). ~1 ms per hand, no extra dependencies.
  * `.pt` checkpoint: the reference torch + py-soma-x + Warp implementation
    (all imported lazily; ~3 ms per hand).

    soma = SomaOverlay("export/mano2somax_resmlp512x4_numpy.npz")
    soma_result, fit_mm = soma.convert(result)          # same frame as result
    img = MeshRenderer(soma.faces, color=SOMA_COLOR, close_wrist=False).render(img, soma_result)

`fit_mm` is the mean per-vertex distance (mm) between the regressed SOMA hand
and HaMeR's MANO mesh transferred onto SOMA topology with SOMA-X's own
barycentric correspondence -- the metric mano2soma-x was trained and
evaluated with. The npz engine carries that correspondence; the torch engine
builds it from the user's MANO_RIGHT.pkl (default: the fasthamer cache copy).
Without it `convert` still returns the meshes, with None errors.
"""
import dataclasses
import os
from typing import List, Optional, Tuple

import numpy as np

from .assets import mano_cache_dir
from .mano import PKL_NAME
from .types import HandMeshResult

SOMA_COLOR = (0.96, 0.56, 0.16)  # orange, against fasthamer's steel-blue MANO
# CoreML regressor files looked up next to the npz for regressor="auto", in order of
# preference: the fp32 one is exact (vertices within 0.05 mm of the reference, ~0.4 ms
# on the CPU); the fp16 Neural Engine one is a little faster (~0.3 ms) but fp16 in
# the rotation head costs up to ~1.5 mm at the fingertips.
AUTO_REGRESSOR_NAMES = ("mano2somax_regressor_fp32.mlpackage", "mano2somax_regressor_ane.mlpackage")


class SomaOverlay:
    def __init__(self, checkpoint: str, lod: str = "mid", device: str = "cpu",
                 mano_pkl: Optional[str] = None, regressor: str = "auto",
                 compute_units: str = "CPU_AND_NE"):
        self.engine = "numpy" if checkpoint.lower().endswith(".npz") else "torch"
        if self.engine == "numpy":
            self._init_numpy(checkpoint, regressor, compute_units)
            return
        try:
            import torch
            from soma.hand import SOMAHandLayer
            from mano_to_somax.infer import ManoToSomaX
        except ImportError as e:
            raise ImportError(
                "the torch SOMA engine needs torch, py-soma-x, warp-lang and the "
                "mano2soma-x package (pip install -e <mano2soma-x checkout>); or pass "
                "the exported .npz asset file to use the torch-free engine") from e
        self.torch = torch
        self.device = torch.device(device)
        self.lod = lod
        self.net = ManoToSomaX.from_checkpoint(checkpoint, device=device)
        self.layer = SOMAHandLayer(hand_type="right", device=device, identity_model_type="soma",
                                   lod=lod, correctives_model_path=None)
        self.faces = self.layer.faces.detach().cpu().numpy().astype(np.int64)
        self.num_verts = int(self.faces.max()) + 1

        # Exact fit metric: MANO mesh -> SOMA topology via SOMA-X's MANO identity
        # backend (needs the user's MANO pkl). 38 wrist-boundary verts have no
        # MANO correspondence and are excluded, as in mano2soma-x's training.
        self._interp = None
        self._valid = None
        if mano_pkl is None:
            cand = os.path.join(mano_cache_dir(), PKL_NAME)
            mano_pkl = cand if os.path.isfile(cand) else None
        if mano_pkl:
            ml = SOMAHandLayer(hand_type="right", device=device, identity_model_type="mano",
                               identity_model_kwargs={"model_path": mano_pkl}, lod=lod,
                               correctives_model_path=None)
            self._interp = ml.identity_model._to_soma_interp
            valid = np.ones(self.num_verts, dtype=bool)
            excl = ml.excluded_vert_ids
            if excl is not None and len(excl):
                valid[np.asarray(excl, dtype=np.int64)] = False
            self._valid = torch.from_numpy(valid).to(self.device)

        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy conversion so Warp's kernels are loaded before the first frame."""
        torch = self.torch
        eye = torch.eye(3, device=self.device)
        with torch.no_grad():
            p = self.net(eye.repeat(1, 15, 1, 1), torch.zeros(1, 10, device=self.device),
                         global_orient=eye[None])
            self.layer.prepare_identity(p["identity"], scale_params=p["scale_params"])
            self.layer.pose(p["pose"], pose2rot=False, absolute_pose=True,
                            global_translation=p["root_translation"])
            if self._interp is not None:
                self._interp(torch.zeros(1, 778, 3, device=self.device))

    def _init_numpy(self, npz: str, regressor: str, compute_units: str) -> None:
        from .soma_np import SomaStageNumpy
        mlpackage = None
        if regressor == "auto":
            for name in AUTO_REGRESSOR_NAMES:
                cand = os.path.join(os.path.dirname(os.path.abspath(npz)), name)
                if os.path.isdir(cand):
                    mlpackage = cand
                    break
        elif regressor not in ("numpy", "none", ""):
            mlpackage = regressor
        # the fp32 regressor graph is CPU-only; CPU_AND_NE would just add a device hop
        if mlpackage and "fp32" in os.path.basename(mlpackage):
            compute_units = "CPU_ONLY"
        self.stage = SomaStageNumpy(npz, regressor_mlpackage=mlpackage, compute_units=compute_units)
        self.regressor_backend = "coreml" if mlpackage else "numpy"
        self.faces = self.stage.faces
        self.lod = self.stage.lod
        self._interp = None
        self._valid = None

    @property
    def can_measure(self) -> bool:
        if self.engine == "numpy":
            return self.stage.bary is not None
        return self._interp is not None

    @staticmethod
    def _right_frame(x: np.ndarray, is_right: bool) -> np.ndarray:
        """Undo fasthamer's left-hand x-flip: coordinates in the right-hand MANO frame."""
        x = np.array(x, dtype=np.float64, copy=True)
        if not is_right:
            x[..., 0] = -x[..., 0]
        return x

    def convert(self, result: HandMeshResult) -> Tuple[HandMeshResult, List[Optional[float]]]:
        """SOMA-X meshes for every hand in `result` (same hand-centred frame, so
        `cam_t` / `vertices_camera` keep working) plus per-hand fit error in mm
        (None when no MANO pkl is available)."""
        hands = [h for h in result.hands
                 if h.hand_pose is not None and h.betas is not None and h.global_orient is not None]
        out_result = HandMeshResult(hands=[], focal=result.focal, cx=result.cx, cy=result.cy,
                                    image_size=result.image_size)
        if not hands:
            return out_result, []
        B = len(hands)
        # SOMA output lives in the right-hand MANO frame with the MANO wrist joint
        # at the origin; fasthamer's meshes keep MANO's own origin.
        wrist = np.stack([self._right_frame(h.keypoints_3d[0], h.is_right) for h in hands])  # (B, 3)
        if self.engine == "numpy":
            soma_np, _ = self.stage.forward(np.stack([h.hand_pose for h in hands]),
                                            np.stack([h.betas for h in hands]),
                                            np.stack([h.global_orient for h in hands]))
            errs: List[Optional[float]] = [None] * B
            if self.stage.bary is not None:
                mano = np.stack([self._right_frame(h.vertices, h.is_right) for h in hands])
                errs = [float(e) for e in self.stage.fit_error_mm(soma_np, mano - wrist[:, None, :])]
            for i, h in enumerate(hands):
                v = soma_np[i] + wrist[i]
                if not h.is_right:
                    v[:, 0] = -v[:, 0]
                out_result.hands.append(dataclasses.replace(h, vertices=v))
            return out_result, errs
        torch = self.torch
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)
        hp = f32(np.stack([h.hand_pose for h in hands])).reshape(B, 15, 3, 3)
        go = f32(np.stack([h.global_orient for h in hands])).reshape(B, 3, 3)
        betas = f32(np.stack([h.betas for h in hands])).reshape(B, 10)

        with torch.no_grad():
            p = self.net(hp, betas, global_orient=go)
            self.layer.prepare_identity(p["identity"], scale_params=p["scale_params"])
            soma = self.layer.pose(p["pose"], pose2rot=False, absolute_pose=True,
                                   global_translation=p["root_translation"])["vertices"]  # (B, V, 3)
            errs: List[Optional[float]] = [None] * B
            if self._interp is not None:
                mano = np.stack([self._right_frame(h.vertices, h.is_right) for h in hands])
                target = self._interp(f32(mano - wrist[:, None, :]))
                d = (soma - target).norm(dim=-1)[:, self._valid].mean(dim=1) * 1000.0
                errs = [float(x) for x in d.cpu()]

        soma_np = soma.cpu().numpy().astype(np.float64)
        for i, h in enumerate(hands):
            v = soma_np[i] + wrist[i]
            if not h.is_right:
                v[:, 0] = -v[:, 0]
            out_result.hands.append(dataclasses.replace(h, vertices=v))
        return out_result, errs
